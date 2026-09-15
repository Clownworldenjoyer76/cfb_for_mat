#!/usr/bin/env python3
# docs/win/football/cfb/scripts/00_intake/league_master.py

"""Build CFB conference membership and long-format standings masters."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit,
)
from urllib.request import Request, urlopen

import yaml


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


CURRENT_WEEK_CONFIG_PATH = (
    CFB_ROOT
    / "config"
    / "current_week.yaml"
)

TEAM_MASTER_PATH = (
    CFB_ROOT
    / "data"
    / "master"
    / "team_master.csv"
)

LEAGUE_MASTER_PATH = (
    CFB_ROOT
    / "data"
    / "master"
    / "league_master.csv"
)

LEAGUE_STANDINGS_PATH = (
    CFB_ROOT
    / "data"
    / "master"
    / "league_standings.csv"
)

REPORT_ROOT = CFB_ROOT / "errors"

SCRIPT_VERSION = (
    "cfb-league-master-v3-retries-2026-09-15"
)

ESPN_BASE = (
    "https://sports.core.api.espn.com/v2/"
    "sports/football/leagues/college-football"
)

TEAM_ID_PATTERN = re.compile(
    r"/teams/(\d+)(?:[/?]|$)"
)

GROUP_ID_PATTERN = re.compile(
    r"/groups/([^/?]+)(?:[/?]|$)"
)

MASTER_COLUMNS = [
    "team_id",
    "team_abbr",
    "conference",
    "conference_abbr",
    "division",
    "division_abbr",
    "season",
    "season_type",
]

STANDINGS_COLUMNS = [
    "team_id",
    "team_abbr",
    "conference",
    "conference_abbr",
    "division",
    "division_abbr",
    "standings_type",
    "record_name",
    "record_type",
    "record_abbreviation",
    "stat_name",
    "stat_value",
    "season",
    "season_type",
]


_GROUP_CACHE: dict[str, dict] = {}
_COLLECTION_CACHE: dict[str, dict] = {}
_TEAM_CACHE: dict[str, dict] = {}

TRANSIENT_HTTP_STATUSES = frozenset(
    {
        429,
        500,
        502,
        503,
        504,
    }
)

MAX_REQUEST_ATTEMPTS = 4

RETRY_BACKOFF_SECONDS = (
    1.0,
    2.0,
    4.0,
)


_REQUEST_COUNT = 0
_REQUEST_ATTEMPT_COUNT = 0
_REQUEST_RETRY_COUNT = 0
_RECOVERED_TRANSIENT_REQUESTS = 0
_EXHAUSTED_TRANSIENT_FAILURES = 0

_REQUEST_FAILURES: list[dict[str, str]] = []
_RETRY_DETAILS: list[dict[str, str]] = []


def reset_runtime_state() -> None:
    global _REQUEST_COUNT
    global _REQUEST_ATTEMPT_COUNT
    global _REQUEST_RETRY_COUNT
    global _RECOVERED_TRANSIENT_REQUESTS
    global _EXHAUSTED_TRANSIENT_FAILURES

    _GROUP_CACHE.clear()
    _COLLECTION_CACHE.clear()
    _TEAM_CACHE.clear()

    _REQUEST_FAILURES.clear()
    _RETRY_DETAILS.clear()

    _REQUEST_COUNT = 0
    _REQUEST_ATTEMPT_COUNT = 0
    _REQUEST_RETRY_COUNT = 0
    _RECOVERED_TRANSIENT_REQUESTS = 0
    _EXHAUSTED_TRANSIENT_FAILURES = 0


def load_current_week() -> tuple[int, int, int]:
    if not CURRENT_WEEK_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Missing current-week config: "
            f"{CURRENT_WEEK_CONFIG_PATH}"
        )

    with CURRENT_WEEK_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            "Current-week config must contain a YAML mapping"
        )

    values: dict[str, int] = {}

    for key in (
        "season",
        "season_type",
        "week",
    ):
        if key not in payload:
            raise ValueError(
                "Current-week config missing "
                f"required key: {key}"
            )

        raw = payload.get(key)

        if isinstance(raw, bool):
            raise ValueError(
                f"Current-week config {key} must be an integer"
            )

        try:
            values[key] = int(
                str(raw).strip()
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Current-week config {key} must be an integer"
            ) from exc

    if values["season"] < 2000:
        raise ValueError(
            f"Invalid configured season: {values['season']}"
        )

    if values["season_type"] < 1:
        raise ValueError(
            f"Invalid configured season_type: {values['season_type']}"
        )

    if values["week"] < 1:
        raise ValueError(
            f"Invalid configured week: {values['week']}"
        )

    return (
        values["season"],
        values["season_type"],
        values["week"],
    )


def groups_url(
    season: int,
    season_type: int,
) -> str:
    return (
        f"{ESPN_BASE}/seasons/{season}/"
        f"types/{season_type}/groups?limit=1000"
    )


def normalize_ref_url(
    value: object,
) -> str:
    url = str(
        value or ""
    ).strip()

    if url.startswith(
        "http://sports.core.api.espn.com/"
    ):
        url = (
            "https://sports.core.api.espn.com/"
            + url[len("http://sports.core.api.espn.com/"):]
        )

    return url


def fetch_json(
    url: str,
    *,
    label: str,
    timeout: int = 20,
) -> dict:
    global _REQUEST_COUNT
    global _REQUEST_ATTEMPT_COUNT
    global _REQUEST_RETRY_COUNT
    global _RECOVERED_TRANSIENT_REQUESTS
    global _EXHAUSTED_TRANSIENT_FAILURES

    url = normalize_ref_url(
        url
    )

    if not url:
        raise ValueError(
            f"{label} URL is blank"
        )

    # Count the requested resource once regardless
    # of how many network attempts are required.
    _REQUEST_COUNT += 1

    for attempt_number in range(
        1,
        MAX_REQUEST_ATTEMPTS + 1,
    ):
        _REQUEST_ATTEMPT_COUNT += 1

        request = Request(
            url,
            headers={
                "User-Agent": (
                    "cfb-league-master/3.0"
                ),
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(
                request,
                timeout=timeout,
            ) as response:
                status = int(
                    response.status
                )

                body = (
                    response.read()
                    .decode(
                        "utf-8"
                    )
                )

        except HTTPError as exc:
            error_body = ""

            try:
                error_body = (
                    exc.read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )
            except Exception:
                pass

            is_transient = (
                exc.code
                in TRANSIENT_HTTP_STATUSES
            )

            if (
                is_transient
                and attempt_number
                < MAX_REQUEST_ATTEMPTS
            ):
                delay_seconds = (
                    RETRY_BACKOFF_SECONDS[
                        attempt_number - 1
                    ]
                )

                _REQUEST_RETRY_COUNT += 1

                _RETRY_DETAILS.append(
                    {
                        "label": label,
                        "url": url,
                        "attempt": str(
                            attempt_number
                        ),
                        "status": str(
                            exc.code
                        ),
                        "error": (
                            error_body[:1000]
                            or str(exc)
                        ),
                        "delay_seconds": str(
                            delay_seconds
                        ),
                    }
                )

                time.sleep(
                    delay_seconds
                )

                continue

            if is_transient:
                _EXHAUSTED_TRANSIENT_FAILURES += 1

            failure = {
                "label": label,
                "url": url,
                "status": str(
                    exc.code
                ),
                "attempt": str(
                    attempt_number
                ),
                "transient": str(
                    is_transient
                ).lower(),
                "error": (
                    error_body[:2000]
                    or str(exc)
                ),
            }

            _REQUEST_FAILURES.append(
                failure
            )

            raise RuntimeError(
                f"{label} request failed: "
                f"status={exc.code}, "
                f"attempt={attempt_number}/"
                f"{MAX_REQUEST_ATTEMPTS}, "
                f"url={url}, "
                f"error={failure['error']}"
            ) from exc

        except (
            URLError,
            TimeoutError,
        ) as exc:
            if (
                attempt_number
                < MAX_REQUEST_ATTEMPTS
            ):
                delay_seconds = (
                    RETRY_BACKOFF_SECONDS[
                        attempt_number - 1
                    ]
                )

                _REQUEST_RETRY_COUNT += 1

                _RETRY_DETAILS.append(
                    {
                        "label": label,
                        "url": url,
                        "attempt": str(
                            attempt_number
                        ),
                        "status": "",
                        "error": str(
                            exc
                        )[:1000],
                        "delay_seconds": str(
                            delay_seconds
                        ),
                    }
                )

                time.sleep(
                    delay_seconds
                )

                continue

            _EXHAUSTED_TRANSIENT_FAILURES += 1

            failure = {
                "label": label,
                "url": url,
                "status": "",
                "attempt": str(
                    attempt_number
                ),
                "transient": "true",
                "error": str(
                    exc
                )[:2000],
            }

            _REQUEST_FAILURES.append(
                failure
            )

            raise RuntimeError(
                f"{label} request failed after "
                f"{attempt_number} attempts: "
                f"url={url}, error={exc}"
            ) from exc

        except Exception as exc:
            failure = {
                "label": label,
                "url": url,
                "status": "",
                "attempt": str(
                    attempt_number
                ),
                "transient": "false",
                "error": str(
                    exc
                )[:2000],
            }

            _REQUEST_FAILURES.append(
                failure
            )

            raise RuntimeError(
                f"{label} request failed: "
                f"url={url}, error={exc}"
            ) from exc

        if (
            status < 200
            or status >= 300
        ):
            is_transient = (
                status
                in TRANSIENT_HTTP_STATUSES
            )

            if (
                is_transient
                and attempt_number
                < MAX_REQUEST_ATTEMPTS
            ):
                delay_seconds = (
                    RETRY_BACKOFF_SECONDS[
                        attempt_number - 1
                    ]
                )

                _REQUEST_RETRY_COUNT += 1

                _RETRY_DETAILS.append(
                    {
                        "label": label,
                        "url": url,
                        "attempt": str(
                            attempt_number
                        ),
                        "status": str(
                            status
                        ),
                        "error": body[:1000],
                        "delay_seconds": str(
                            delay_seconds
                        ),
                    }
                )

                time.sleep(
                    delay_seconds
                )

                continue

            if is_transient:
                _EXHAUSTED_TRANSIENT_FAILURES += 1

            failure = {
                "label": label,
                "url": url,
                "status": str(
                    status
                ),
                "attempt": str(
                    attempt_number
                ),
                "transient": str(
                    is_transient
                ).lower(),
                "error": body[:2000],
            }

            _REQUEST_FAILURES.append(
                failure
            )

            raise RuntimeError(
                f"{label} request failed: "
                f"status={status}, "
                f"attempt={attempt_number}/"
                f"{MAX_REQUEST_ATTEMPTS}, "
                f"url={url}"
            )

        try:
            payload = json.loads(
                body
            )
        except Exception as exc:
            failure = {
                "label": label,
                "url": url,
                "status": str(
                    status
                ),
                "attempt": str(
                    attempt_number
                ),
                "transient": "false",
                "error": (
                    "JSON parse failed: "
                    f"{exc}"
                ),
            }

            _REQUEST_FAILURES.append(
                failure
            )

            raise RuntimeError(
                f"{label} returned malformed JSON: "
                f"url={url}"
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            failure = {
                "label": label,
                "url": url,
                "status": str(
                    status
                ),
                "attempt": str(
                    attempt_number
                ),
                "transient": "false",
                "error": (
                    "response JSON is not an object"
                ),
            }

            _REQUEST_FAILURES.append(
                failure
            )

            raise RuntimeError(
                f"{label} returned non-object JSON: "
                f"url={url}"
            )

        if attempt_number > 1:
            _RECOVERED_TRANSIENT_REQUESTS += 1

        return payload

    raise RuntimeError(
        f"{label} exhausted request loop unexpectedly: "
        f"url={url}"
    )


def fetch_cached(
    url: str,
    *,
    label: str,
) -> dict:
    url = normalize_ref_url(url)

    if url not in _COLLECTION_CACHE:
        _COLLECTION_CACHE[url] = fetch_json(
            url,
            label=label,
        )

    return _COLLECTION_CACHE[url]


def with_limit(
    url: str,
    limit: int = 1000,
) -> str:
    url = normalize_ref_url(url)
    parts = urlsplit(url)

    query = dict(
        parse_qsl(
            parts.query,
            keep_blank_values=True,
        )
    )

    query["limit"] = str(limit)

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query),
            parts.fragment,
        )
    )


def extract_team_id(
    ref_url: object,
) -> str:
    match = TEAM_ID_PATTERN.search(
        str(ref_url or "")
    )

    return match.group(1) if match else ""


def extract_group_id(
    ref_url: object,
) -> str:
    match = GROUP_ID_PATTERN.search(
        str(ref_url or "")
    )

    return match.group(1) if match else ""


def is_conference_group(
    group: dict,
) -> bool:
    value = group.get("isConference")

    return (
        value is True
        or str(value).strip().casefold() == "true"
    )


def group_identity(
    group: dict,
    ref_url: str = "",
) -> str:
    return (
        str(group.get("id", "")).strip()
        or extract_group_id(ref_url)
        or normalize_ref_url(ref_url)
    )


def read_team_master() -> dict[
    str,
    dict[str, str],
]:
    if not TEAM_MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing team master: {TEAM_MASTER_PATH}"
        )

    with TEAM_MASTER_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        required = [
            "team_id",
            "team_abbr",
        ]

        missing = [
            column
            for column in required
            if column not in fieldnames
        ]

        if missing:
            raise ValueError(
                "team_master.csv missing "
                f"required columns: {missing}"
            )

        teams: dict[
            str,
            dict[str, str],
        ] = {}

        for index, row in enumerate(reader):
            team_id = str(
                row.get(
                    "team_id",
                    "",
                )
            ).strip()

            if not team_id:
                raise ValueError(
                    "team_master.csv contains "
                    f"blank team_id at row {index}"
                )

            if (
                not team_id.isdigit()
                or int(team_id) <= 0
            ):
                raise ValueError(
                    "team_master.csv contains "
                    f"invalid team_id={team_id!r}"
                )

            team_abbr = str(
                row.get(
                    "team_abbr",
                    "",
                )
            ).strip()

            canonical_team = str(
                row.get(
                    "canonical_team",
                    "",
                )
            ).strip()

            existing = teams.get(team_id)

            if existing is None:
                teams[team_id] = {
                    "team_id": team_id,
                    "team_abbr": team_abbr,
                    "canonical_team": canonical_team,
                }
                continue

            existing_abbr = str(
                existing.get(
                    "team_abbr",
                    "",
                )
            ).strip()

            if (
                team_abbr
                and existing_abbr
                and team_abbr != existing_abbr
            ):
                raise ValueError(
                    "team_master.csv contains "
                    "conflicting abbreviations "
                    f"for team_id={team_id}: "
                    f"{existing_abbr!r} vs {team_abbr!r}"
                )

            if team_abbr and not existing_abbr:
                existing["team_abbr"] = team_abbr

            existing_name = str(
                existing.get(
                    "canonical_team",
                    "",
                )
            ).strip()

            if (
                canonical_team
                and existing_name
                and canonical_team != existing_name
            ):
                raise ValueError(
                    "team_master.csv contains "
                    "conflicting canonical_team values "
                    f"for team_id={team_id}: "
                    f"{existing_name!r} vs "
                    f"{canonical_team!r}"
                )

            if canonical_team and not existing_name:
                existing[
                    "canonical_team"
                ] = canonical_team

    if not teams:
        raise ValueError(
            "team_master.csv contains no teams"
        )

    return teams


def resolve_group(
    ref_url: str,
) -> dict:
    ref_url = normalize_ref_url(ref_url)

    if not ref_url:
        raise ValueError(
            "Group reference URL is blank"
        )

    group_id = extract_group_id(ref_url)
    cache_key = group_id or ref_url

    if cache_key not in _GROUP_CACHE:
        _GROUP_CACHE[
            cache_key
        ] = fetch_json(
            ref_url,
            label=f"group {cache_key}",
        )

    return _GROUP_CACHE[cache_key]


def resolve_team(
    ref_url: str,
) -> dict:
    ref_url = normalize_ref_url(ref_url)

    if not ref_url:
        raise ValueError(
            "Team reference URL is blank"
        )

    team_id = extract_team_id(ref_url)
    cache_key = team_id or ref_url

    if cache_key not in _TEAM_CACHE:
        _TEAM_CACHE[
            cache_key
        ] = fetch_json(
            ref_url,
            label=f"team {cache_key}",
        )

    return _TEAM_CACHE[cache_key]


def collection_items(
    ref_obj: object,
    *,
    label: str,
) -> list[dict]:
    if not isinstance(ref_obj, dict):
        return []

    ref_url = normalize_ref_url(
        ref_obj.get(
            "$ref",
            "",
        )
    )

    if not ref_url:
        return []

    payload = fetch_cached(
        with_limit(ref_url),
        label=label,
    )

    items = payload.get(
        "items",
        [],
    )

    if not isinstance(items, list):
        raise RuntimeError(
            f"{label} response items field is not a list"
        )

    normalized: list[dict] = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"{label} contains non-object "
                f"item at index {index}"
            )

        normalized.append(item)

    return normalized


def get_child_groups(
    group: dict,
) -> list[
    tuple[str, dict]
]:
    group_id = group_identity(group)

    children: list[
        tuple[str, dict]
    ] = []

    for item in collection_items(
        group.get(
            "children",
            {},
        ),
        label=f"group {group_id} children",
    ):
        child_ref = normalize_ref_url(
            item.get(
                "$ref",
                "",
            )
        )

        if not child_ref:
            raise RuntimeError(
                "Group children collection "
                "contains item without $ref "
                f"for group={group_id}"
            )

        child = resolve_group(child_ref)

        children.append(
            (
                child_ref,
                child,
            )
        )

    return children


def get_group_team_refs(
    group: dict,
) -> list[
    tuple[
        str,
        str,
        dict,
    ]
]:
    group_id = group_identity(group)

    teams: list[
        tuple[
            str,
            str,
            dict,
        ]
    ] = []

    seen: set[str] = set()

    for item in collection_items(
        group.get(
            "teams",
            {},
        ),
        label=f"group {group_id} teams",
    ):
        team_ref = normalize_ref_url(
            item.get(
                "$ref",
                "",
            )
        )

        team_id = (
            extract_team_id(team_ref)
            or str(
                item.get(
                    "id",
                    "",
                )
            ).strip()
        )

        if not team_id:
            raise RuntimeError(
                "Group teams collection "
                "contains item without team "
                f"identity for group={group_id}"
            )

        if team_id in seen:
            continue

        seen.add(team_id)

        teams.append(
            (
                team_id,
                team_ref,
                item,
            )
        )

    return teams


def resolve_team_abbreviation(
    team_id: str,
    team_ref: str,
    inline_item: dict,
) -> str:
    inline_abbr = str(
        inline_item.get(
            "abbreviation",
            "",
        )
    ).strip()

    if inline_abbr:
        return inline_abbr

    if not team_ref:
        raise RuntimeError(
            "Cannot resolve abbreviation "
            f"for team_id={team_id}: "
            "no ESPN team reference"
        )

    payload = resolve_team(team_ref)

    abbreviation = str(
        payload.get(
            "abbreviation",
            "",
        )
    ).strip()

    if not abbreviation:
        raise RuntimeError(
            "ESPN team metadata has blank "
            "abbreviation for "
            f"team_id={team_id}"
        )

    payload_id = str(
        payload.get(
            "id",
            "",
        )
    ).strip()

    if payload_id and payload_id != team_id:
        raise RuntimeError(
            "ESPN team metadata identity "
            "mismatch: expected "
            f"team_id={team_id}, "
            f"received={payload_id}"
        )

    return abbreviation


def parent_ref(
    group: dict,
) -> str:
    parent = group.get(
        "parent",
        {},
    )

    if not isinstance(parent, dict):
        return ""

    return normalize_ref_url(
        parent.get(
            "$ref",
            "",
        )
    )


def nearest_conference(
    group: dict,
    ref_url: str = "",
) -> tuple[
    str,
    dict | None,
]:
    current = group
    current_ref = normalize_ref_url(
        ref_url
    )

    seen: set[str] = set()

    while isinstance(current, dict):
        identity = group_identity(
            current,
            current_ref,
        )

        if not identity:
            raise RuntimeError(
                "Encountered ESPN group without identity"
            )

        if identity in seen:
            raise RuntimeError(
                "Detected cycle in ESPN "
                "group hierarchy at "
                f"group={identity}"
            )

        seen.add(identity)

        if is_conference_group(current):
            return (
                current_ref,
                current,
            )

        next_ref = parent_ref(current)

        if not next_ref:
            break

        current = resolve_group(
            next_ref
        )

        current_ref = next_ref

    return (
        "",
        None,
    )


def hierarchy_labels(
    group: dict,
    ref_url: str = "",
) -> tuple[
    str,
    str,
    str,
    str,
    bool,
]:
    (
        conf_ref,
        conference,
    ) = nearest_conference(
        group,
        ref_url,
    )

    if conference:
        conf_name = str(
            conference.get(
                "name",
                "",
            )
        ).strip()

        conf_abbr = str(
            conference.get(
                "abbreviation",
                "",
            )
        ).strip()

        if not conf_name:
            raise RuntimeError(
                "Conference group has "
                "blank name: "
                f"{group_identity(conference, conf_ref)}"
            )

        current_id = group_identity(
            group,
            ref_url,
        )

        conference_id = (
            group_identity(
                conference,
                conf_ref,
            )
        )

        if current_id == conference_id:
            return (
                conf_name,
                conf_abbr,
                "",
                "",
                True,
            )

        return (
            conf_name,
            conf_abbr,
            str(
                group.get(
                    "name",
                    "",
                )
            ).strip(),
            str(
                group.get(
                    "abbreviation",
                    "",
                )
            ).strip(),
            True,
        )

    return (
        "",
        "",
        "",
        "",
        False,
    )


def standings_payloads(
    group: dict,
) -> list[dict]:
    standings = group.get(
        "standings",
        {},
    )

    if not isinstance(
        standings,
        dict,
    ):
        return []

    standings_ref = normalize_ref_url(
        standings.get(
            "$ref",
            "",
        )
    )

    if not standings_ref:
        return []

    group_id = group_identity(group)

    root = fetch_cached(
        standings_ref,
        label=f"group {group_id} standings",
    )

    payloads: list[dict] = []

    if isinstance(
        root.get(
            "standings"
        ),
        list,
    ):
        payloads.append(root)

    items = root.get(
        "items",
        [],
    )

    if (
        items is not None
        and not isinstance(
            items,
            list,
        )
    ):
        raise RuntimeError(
            "Standings collection items "
            "field is not a list for "
            f"group={group_id}"
        )

    for index, item in enumerate(
        items or []
    ):
        if not isinstance(item, dict):
            raise RuntimeError(
                "Standings collection contains "
                "non-object at "
                f"index={index}, "
                f"group={group_id}"
            )

        if isinstance(
            item.get(
                "standings"
            ),
            list,
        ):
            payloads.append(item)
            continue

        type_ref = normalize_ref_url(
            item.get(
                "$ref",
                "",
            )
        )

        if not type_ref:
            continue

        payload = fetch_cached(
            type_ref,
            label=(
                "standings type "
                f"{group_id}:{index}"
            ),
        )

        payloads.append(payload)

    return payloads


def record_scope(
    record: dict,
) -> tuple[
    str,
    str,
    str,
]:
    record_name = str(
        record.get(
            "name",
            "",
        )
        or record.get(
            "displayName",
            "",
        )
        or ""
    ).strip()

    record_type = str(
        record.get(
            "type",
            "",
        )
        or ""
    ).strip()

    record_abbreviation = str(
        record.get(
            "abbreviation",
            "",
        )
        or ""
    ).strip()

    if not (
        record_name
        or record_type
        or record_abbreviation
    ):
        raise RuntimeError(
            "ESPN standings record "
            "has no record-scope identity"
        )

    return (
        record_name,
        record_type,
        record_abbreviation,
    )


def get_standings_rows(
    group: dict,
    ref_url: str,
    team_abbr_lookup: dict[
        str,
        str,
    ],
    accepted_team_ids: set[str],
    season: int,
    season_type: int,
) -> list[dict[str, object]]:
    (
        conf_name,
        conf_abbr,
        div_name,
        div_abbr,
        has_conference,
    ) = hierarchy_labels(
        group,
        ref_url,
    )

    if not has_conference:
        return []

    rows: list[
        dict[str, object]
    ] = []

    for standings_payload in standings_payloads(
        group
    ):
        type_name = str(
            standings_payload.get("name")
            or standings_payload.get(
                "displayName"
            )
            or standings_payload.get("type")
            or ""
        ).strip()

        if not type_name:
            raise RuntimeError(
                "ESPN standings payload has "
                "blank standings type for "
                f"conference={conf_name}"
            )

        team_standings = (
            standings_payload.get(
                "standings",
                [],
            )
        )

        if not isinstance(
            team_standings,
            list,
        ):
            raise RuntimeError(
                "ESPN standings payload "
                "standings field is not "
                f"a list for type={type_name}"
            )

        for team_standing in team_standings:
            if not isinstance(
                team_standing,
                dict,
            ):
                raise RuntimeError(
                    "ESPN standings contains "
                    "non-object team entry "
                    f"for type={type_name}"
                )

            team_obj = (
                team_standing.get(
                    "team",
                    {},
                )
            )

            if not isinstance(
                team_obj,
                dict,
            ):
                raise RuntimeError(
                    "ESPN standings team "
                    "reference is not an object"
                )

            team_ref = normalize_ref_url(
                team_obj.get(
                    "$ref",
                    "",
                )
            )

            team_id = (
                extract_team_id(team_ref)
                or str(
                    team_obj.get(
                        "id",
                        "",
                    )
                ).strip()
            )

            if (
                not team_id
                or team_id
                not in accepted_team_ids
            ):
                continue

            team_abbr = str(
                team_abbr_lookup.get(
                    team_id,
                    "",
                )
            ).strip()

            if not team_abbr:
                raise RuntimeError(
                    "Missing resolved abbreviation "
                    "for standings "
                    f"team_id={team_id}"
                )

            records = team_standing.get(
                "records",
                [],
            )

            if not isinstance(
                records,
                list,
            ):
                raise RuntimeError(
                    "ESPN standings records "
                    "field is not a list for "
                    f"team_id={team_id}"
                )

            for record in records:
                if not isinstance(
                    record,
                    dict,
                ):
                    raise RuntimeError(
                        "ESPN standings records "
                        "contains non-object for "
                        f"team_id={team_id}"
                    )

                (
                    record_name,
                    record_type,
                    record_abbreviation,
                ) = record_scope(record)

                stats = record.get(
                    "stats",
                    [],
                )

                if not isinstance(
                    stats,
                    list,
                ):
                    raise RuntimeError(
                        "ESPN standings stats "
                        "field is not a list for "
                        f"team_id={team_id}"
                    )

                for stat in stats:
                    if not isinstance(
                        stat,
                        dict,
                    ):
                        raise RuntimeError(
                            "ESPN standings stats "
                            "contains non-object "
                            f"for team_id={team_id}"
                        )

                    stat_name = str(
                        stat.get(
                            "name",
                            "",
                        )
                    ).strip()

                    if not stat_name:
                        raise RuntimeError(
                            "ESPN standings stat "
                            "has blank name for "
                            f"team_id={team_id}"
                        )

                    rows.append(
                        {
                            "team_id": team_id,
                            "team_abbr": team_abbr,
                            "conference": conf_name,
                            "conference_abbr": conf_abbr,
                            "division": div_name,
                            "division_abbr": div_abbr,
                            "standings_type": type_name,
                            "record_name": record_name,
                            "record_type": record_type,
                            "record_abbreviation": (
                                record_abbreviation
                            ),
                            "stat_name": stat_name,
                            "stat_value": stat.get(
                                "value",
                                "",
                            ),
                            "season": season,
                            "season_type": (
                                season_type
                            ),
                        }
                    )

    return rows


def discover_groups(
    top_groups: dict,
) -> list[
    tuple[str, dict]
]:
    items = top_groups.get(
        "items",
        [],
    )

    if not isinstance(items, list):
        raise RuntimeError(
            "Top-level ESPN groups "
            "items field is not a list"
        )

    discovered: dict[
        str,
        tuple[str, dict],
    ] = {}

    def visit(
        ref_url: str,
    ) -> None:
        ref_url = normalize_ref_url(
            ref_url
        )

        group = resolve_group(
            ref_url
        )

        identity = group_identity(
            group,
            ref_url,
        )

        if not identity:
            raise RuntimeError(
                "Resolved ESPN group "
                "has no identity"
            )

        if identity in discovered:
            return

        discovered[
            identity
        ] = (
            ref_url,
            group,
        )

        upstream_ref = parent_ref(
            group
        )

        if upstream_ref:
            visit(
                upstream_ref
            )

        for (
            child_ref,
            _child,
        ) in get_child_groups(group):
            visit(
                child_ref
            )

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise RuntimeError(
                "Top-level ESPN groups "
                "contains non-object at "
                f"index={index}"
            )

        ref_url = normalize_ref_url(
            item.get(
                "$ref",
                "",
            )
        )

        if not ref_url:
            raise RuntimeError(
                "Top-level ESPN groups "
                "contains item without $ref "
                f"at index={index}"
            )

        visit(ref_url)

    if not discovered:
        raise RuntimeError(
            "ESPN returned no discoverable CFB groups"
        )

    return list(
        discovered.values()
    )


def membership_signature(
    row: dict[str, object],
) -> tuple[str, ...]:
    return (
        str(
            row.get(
                "conference",
                "",
            )
        ),
        str(
            row.get(
                "conference_abbr",
                "",
            )
        ),
        str(
            row.get(
                "division",
                "",
            )
        ),
        str(
            row.get(
                "division_abbr",
                "",
            )
        ),
    )


def build_memberships(
    groups: list[
        tuple[str, dict]
    ],
    team_index: dict[
        str,
        dict[str, str],
    ],
    season: int,
    season_type: int,
) -> tuple[
    dict[
        str,
        tuple[
            int,
            dict[str, object],
        ],
    ],
    dict[str, str],
    int,
]:
    accepted_ids = set(
        team_index
    )

    team_abbr_lookup = {
        team_id: str(
            info.get(
                "team_abbr",
                "",
            )
        ).strip()
        for team_id, info
        in team_index.items()
    }

    memberships: dict[
        str,
        tuple[
            int,
            dict[str, object],
        ],
    ] = {}

    abbreviations_resolved = 0

    for ref_url, group in groups:
        group_name = str(
            group.get(
                "name",
                "",
            )
        ).strip()

        child_groups = get_child_groups(
            group
        )

        team_refs = get_group_team_refs(
            group
        )

        (
            conf_name,
            conf_abbr,
            div_name,
            div_abbr,
            has_conference,
        ) = hierarchy_labels(
            group,
            ref_url,
        )

        use_fallback = (
            bool(team_refs)
            and not has_conference
            and not child_groups
        )

        if not (
            team_refs
            and (
                has_conference
                or use_fallback
            )
        ):
            continue

        if use_fallback:
            if not group_name:
                raise RuntimeError(
                    "Fallback leaf group "
                    "has blank name"
                )

            conf_name = group_name

            conf_abbr = str(
                group.get(
                    "abbreviation",
                    "",
                )
            ).strip()

            div_name = ""
            div_abbr = ""

        if not conf_name:
            raise RuntimeError(
                "Membership candidate "
                "has blank conference"
            )

        if has_conference:
            membership_priority = (
                3
                if div_name
                else 2
            )
        else:
            membership_priority = 1

        for (
            team_id,
            team_ref,
            inline_item,
        ) in team_refs:
            if team_id not in accepted_ids:
                continue

            current_abbr = str(
                team_abbr_lookup.get(
                    team_id,
                    "",
                )
            ).strip()

            if not current_abbr:
                current_abbr = (
                    resolve_team_abbreviation(
                        team_id,
                        team_ref,
                        inline_item,
                    )
                )

                team_abbr_lookup[
                    team_id
                ] = current_abbr

                abbreviations_resolved += 1

            row = {
                "team_id": team_id,
                "team_abbr": current_abbr,
                "conference": conf_name,
                "conference_abbr": conf_abbr,
                "division": div_name,
                "division_abbr": div_abbr,
                "season": season,
                "season_type": season_type,
            }

            existing = memberships.get(
                team_id
            )

            if existing is None:
                memberships[
                    team_id
                ] = (
                    membership_priority,
                    row,
                )
                continue

            (
                existing_priority,
                existing_row,
            ) = existing

            if (
                existing_priority
                > membership_priority
            ):
                continue

            if (
                existing_priority
                == membership_priority
            ):
                if (
                    membership_signature(
                        existing_row
                    )
                    != membership_signature(
                        row
                    )
                ):
                    raise RuntimeError(
                        "Conflicting equal-priority "
                        "conference memberships for "
                        f"team_id={team_id}: "
                        f"{membership_signature(existing_row)} "
                        "vs "
                        f"{membership_signature(row)}"
                    )

                continue

            memberships[
                team_id
            ] = (
                membership_priority,
                row,
            )

    return (
        memberships,
        team_abbr_lookup,
        abbreviations_resolved,
    )


def standings_key(
    row: dict[str, object],
) -> tuple[str, ...]:
    return (
        str(
            row.get(
                "team_id",
                "",
            )
        ),
        str(
            row.get(
                "conference",
                "",
            )
        ),
        str(
            row.get(
                "conference_abbr",
                "",
            )
        ),
        str(
            row.get(
                "division",
                "",
            )
        ),
        str(
            row.get(
                "division_abbr",
                "",
            )
        ),
        str(
            row.get(
                "standings_type",
                "",
            )
        ),
        str(
            row.get(
                "record_name",
                "",
            )
        ),
        str(
            row.get(
                "record_type",
                "",
            )
        ),
        str(
            row.get(
                "record_abbreviation",
                "",
            )
        ),
        str(
            row.get(
                "stat_name",
                "",
            )
        ),
        str(
            row.get(
                "season",
                "",
            )
        ),
        str(
            row.get(
                "season_type",
                "",
            )
        ),
    )


def build_standings(
    groups: list[
        tuple[str, dict]
    ],
    team_abbr_lookup: dict[
        str,
        str,
    ],
    accepted_team_ids: set[str],
    season: int,
    season_type: int,
) -> tuple[
    list[dict[str, object]],
    int,
]:
    keyed: dict[
        tuple[str, ...],
        dict[str, object],
    ] = {}

    exact_duplicates = 0

    for ref_url, group in groups:
        standings_obj = group.get(
            "standings",
            {},
        )

        standings_ref = (
            normalize_ref_url(
                standings_obj.get(
                    "$ref",
                    "",
                )
            )
            if isinstance(
                standings_obj,
                dict,
            )
            else ""
        )

        if not standings_ref:
            continue

        for row in get_standings_rows(
            group,
            ref_url,
            team_abbr_lookup,
            accepted_team_ids,
            season,
            season_type,
        ):
            key = standings_key(
                row
            )

            existing = keyed.get(
                key
            )

            if existing is None:
                keyed[key] = row
                continue

            existing_value = str(
                existing.get(
                    "stat_value",
                    "",
                )
            )

            new_value = str(
                row.get(
                    "stat_value",
                    "",
                )
            )

            if existing_value != new_value:
                raise RuntimeError(
                    "Conflicting standings "
                    "values at true grain: "
                    f"key={key}, "
                    f"existing={existing_value!r}, "
                    f"new={new_value!r}"
                )

            exact_duplicates += 1

    rows = list(
        keyed.values()
    )

    rows.sort(
        key=lambda row: (
            str(
                row.get(
                    "conference",
                    "",
                )
            ),
            str(
                row.get(
                    "division",
                    "",
                )
            ),
            str(
                row.get(
                    "team_abbr",
                    "",
                )
            ),
            str(
                row.get(
                    "standings_type",
                    "",
                )
            ),
            str(
                row.get(
                    "record_type",
                    "",
                )
            ),
            str(
                row.get(
                    "record_name",
                    "",
                )
            ),
            str(
                row.get(
                    "stat_name",
                    "",
                )
            ),
        )
    )

    return (
        rows,
        exact_duplicates,
    )


def validate_master_rows(
    rows: list[
        dict[str, object]
    ],
    team_index: dict[
        str,
        dict[str, str],
    ],
    season: int,
    season_type: int,
) -> None:
    if not rows:
        raise ValueError(
            "League master output is empty"
        )

    accepted_ids = set(
        team_index
    )

    observed_ids: set[str] = set()

    for index, row in enumerate(rows):
        team_id = str(
            row.get(
                "team_id",
                "",
            )
        ).strip()

        if not team_id:
            raise ValueError(
                "League master row "
                f"{index} has blank team_id"
            )

        if team_id in observed_ids:
            raise ValueError(
                "League master contains "
                "duplicate team_id="
                f"{team_id}"
            )

        observed_ids.add(
            team_id
        )

        if team_id not in accepted_ids:
            raise ValueError(
                "League master contains "
                "foreign team_id="
                f"{team_id}"
            )

        if not str(
            row.get(
                "team_abbr",
                "",
            )
        ).strip():
            raise ValueError(
                "League master contains "
                "blank team_abbr for "
                f"team_id={team_id}"
            )

        if not str(
            row.get(
                "conference",
                "",
            )
        ).strip():
            raise ValueError(
                "League master contains "
                "blank conference for "
                f"team_id={team_id}"
            )

        if str(
            row.get(
                "season",
                "",
            )
        ) != str(season):
            raise ValueError(
                "League master season mismatch "
                f"for team_id={team_id}"
            )

        if str(
            row.get(
                "season_type",
                "",
            )
        ) != str(season_type):
            raise ValueError(
                "League master season_type "
                "mismatch for "
                f"team_id={team_id}"
            )

    if observed_ids != accepted_ids:
        missing = sorted(
            accepted_ids
            - observed_ids
        )

        extra = sorted(
            observed_ids
            - accepted_ids
        )

        raise ValueError(
            "League master coverage "
            "does not exactly match "
            "accepted team_master teams. "
            f"missing={missing[:30]}, "
            f"extra={extra[:30]}"
        )


def validate_standings_rows(
    rows: list[
        dict[str, object]
    ],
    master_rows: list[
        dict[str, object]
    ],
    season: int,
    season_type: int,
) -> None:
    if not rows:
        raise ValueError(
            "League standings output is empty"
        )

    master_by_id = {
        str(
            row[
                "team_id"
            ]
        ): row
        for row in master_rows
    }

    seen: set[
        tuple[str, ...]
    ] = set()

    for index, row in enumerate(rows):
        team_id = str(
            row.get(
                "team_id",
                "",
            )
        ).strip()

        if (
            not team_id
            or team_id not in master_by_id
        ):
            raise ValueError(
                "League standings row "
                f"{index} references invalid "
                f"team_id={team_id!r}"
            )

        master = master_by_id[
            team_id
        ]

        if str(
            row.get(
                "team_abbr",
                "",
            )
        ).strip() != str(
            master.get(
                "team_abbr",
                "",
            )
        ).strip():
            raise ValueError(
                "League standings "
                "team_abbr mismatch for "
                f"team_id={team_id}"
            )

        if str(
            row.get(
                "conference",
                "",
            )
        ).strip() != str(
            master.get(
                "conference",
                "",
            )
        ).strip():
            raise ValueError(
                "League standings "
                "conference mismatch for "
                f"team_id={team_id}"
            )

        if not str(
            row.get(
                "standings_type",
                "",
            )
        ).strip():
            raise ValueError(
                "League standings row "
                f"{index} has blank "
                "standings_type"
            )

        if not (
            str(
                row.get(
                    "record_name",
                    "",
                )
            ).strip()
            or str(
                row.get(
                    "record_type",
                    "",
                )
            ).strip()
            or str(
                row.get(
                    "record_abbreviation",
                    "",
                )
            ).strip()
        ):
            raise ValueError(
                "League standings row "
                f"{index} has blank record scope"
            )

        if not str(
            row.get(
                "stat_name",
                "",
            )
        ).strip():
            raise ValueError(
                "League standings row "
                f"{index} has blank stat_name"
            )

        if str(
            row.get(
                "season",
                "",
            )
        ) != str(season):
            raise ValueError(
                "League standings season "
                "mismatch for "
                f"team_id={team_id}"
            )

        if str(
            row.get(
                "season_type",
                "",
            )
        ) != str(season_type):
            raise ValueError(
                "League standings season_type "
                "mismatch for "
                f"team_id={team_id}"
            )

        key = standings_key(
            row
        )

        if key in seen:
            raise ValueError(
                "League standings contains "
                "duplicate true-grain key: "
                f"{key}"
            )

        seen.add(key)


def write_csv_file(
    path: Path,
    rows: list[
        dict[str, object]
    ],
    columns: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: row.get(
                        column,
                        "",
                    )
                    for column in columns
                }
            )

        handle.flush()
        os.fsync(
            handle.fileno()
        )


def temporary_path(
    final_path: Path,
) -> Path:
    return final_path.with_name(
        f".{final_path.name}."
        f"{uuid.uuid4().hex}.tmp"
    )


def backup_path(
    final_path: Path,
) -> Path:
    return final_path.with_name(
        f".{final_path.name}."
        f"{uuid.uuid4().hex}.bak"
    )


def publish_bundle(
    master_rows: list[
        dict[str, object]
    ],
    standings_rows: list[
        dict[str, object]
    ],
) -> None:
    LEAGUE_MASTER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LEAGUE_STANDINGS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    master_temp = temporary_path(
        LEAGUE_MASTER_PATH
    )

    standings_temp = temporary_path(
        LEAGUE_STANDINGS_PATH
    )

    master_backup = backup_path(
        LEAGUE_MASTER_PATH
    )

    standings_backup = backup_path(
        LEAGUE_STANDINGS_PATH
    )

    master_existed = (
        LEAGUE_MASTER_PATH.exists()
    )

    standings_existed = (
        LEAGUE_STANDINGS_PATH.exists()
    )

    master_published = False
    standings_published = False
    success = False

    try:
        write_csv_file(
            master_temp,
            master_rows,
            MASTER_COLUMNS,
        )

        write_csv_file(
            standings_temp,
            standings_rows,
            STANDINGS_COLUMNS,
        )

        if master_existed:
            os.replace(
                LEAGUE_MASTER_PATH,
                master_backup,
            )

        if standings_existed:
            os.replace(
                LEAGUE_STANDINGS_PATH,
                standings_backup,
            )

        os.replace(
            master_temp,
            LEAGUE_MASTER_PATH,
        )

        master_published = True

        os.replace(
            standings_temp,
            LEAGUE_STANDINGS_PATH,
        )

        standings_published = True
        success = True

    except Exception:
        if master_published:
            try:
                LEAGUE_MASTER_PATH.unlink(
                    missing_ok=True
                )
            except Exception:
                pass

        if standings_published:
            try:
                LEAGUE_STANDINGS_PATH.unlink(
                    missing_ok=True
                )
            except Exception:
                pass

        if master_backup.exists():
            try:
                os.replace(
                    master_backup,
                    LEAGUE_MASTER_PATH,
                )
            except Exception:
                pass

        elif not master_existed:
            try:
                LEAGUE_MASTER_PATH.unlink(
                    missing_ok=True
                )
            except Exception:
                pass

        if standings_backup.exists():
            try:
                os.replace(
                    standings_backup,
                    LEAGUE_STANDINGS_PATH,
                )
            except Exception:
                pass

        elif not standings_existed:
            try:
                LEAGUE_STANDINGS_PATH.unlink(
                    missing_ok=True
                )
            except Exception:
                pass

        raise

    finally:
        for path in (
            master_temp,
            standings_temp,
        ):
            try:
                path.unlink(
                    missing_ok=True
                )
            except Exception:
                pass

        if success:
            for path in (
                master_backup,
                standings_backup,
            ):
                try:
                    path.unlink(
                        missing_ok=True
                    )
                except Exception:
                    pass


def run(
    report: PipelineReporter,
) -> int:
    (
        season,
        season_type,
        week,
    ) = load_current_week()

    report.season = season
    report.week = week

    report.set_detail(
        "season_type",
        season_type,
    )

    team_index = read_team_master()

    initial_abbr_count = sum(
        1
        for info in team_index.values()
        if str(
            info.get(
                "team_abbr",
                "",
            )
        ).strip()
    )

    report.set_rows(
        rows_in=len(
            team_index
        ),
    )

    report.update_details(
        {
            "team_master_unique_teams": (
                len(
                    team_index
                )
            ),
            "team_master_nonblank_abbreviations": (
                initial_abbr_count
            ),
        }
    )

    top_groups = fetch_json(
        groups_url(
            season,
            season_type,
        ),
        label="top-level CFB groups",
    )

    groups = discover_groups(
        top_groups
    )

    report.set_detail(
        "groups_discovered",
        len(groups),
    )

    (
        memberships,
        team_abbr_lookup,
        abbreviations_resolved,
    ) = build_memberships(
        groups,
        team_index,
        season,
        season_type,
    )

    missing_membership = sorted(
        set(team_index)
        - set(memberships)
    )

    missing_abbreviations = sorted(
        team_id
        for team_id in team_index
        if not str(
            team_abbr_lookup.get(
                team_id,
                "",
            )
        ).strip()
    )

    report.update_details(
        {
            "memberships_resolved": (
                len(
                    memberships
                )
            ),
            "missing_membership_count": (
                len(
                    missing_membership
                )
            ),
            "missing_membership_ids": (
                missing_membership
            ),
            "abbreviations_resolved_from_espn": (
                abbreviations_resolved
            ),
            "missing_abbreviation_count": (
                len(
                    missing_abbreviations
                )
            ),
            "missing_abbreviation_ids": (
                missing_abbreviations
            ),
        }
    )

    if missing_membership:
        raise RuntimeError(
            "Not every accepted team_master "
            "team received a league membership. "
            f"missing={missing_membership[:30]}"
        )

    if missing_abbreviations:
        raise RuntimeError(
            "Not every accepted team_master "
            "team has a resolved abbreviation. "
            f"missing={missing_abbreviations[:30]}"
        )

    master_rows = [
        entry[1]
        for entry
        in memberships.values()
    ]

    master_rows.sort(
        key=lambda row: (
            str(
                row.get(
                    "conference",
                    "",
                )
            ),
            str(
                row.get(
                    "division",
                    "",
                )
            ),
            str(
                row.get(
                    "team_abbr",
                    "",
                )
            ),
            str(
                row.get(
                    "team_id",
                    "",
                )
            ),
        )
    )

    validate_master_rows(
        master_rows,
        team_index,
        season,
        season_type,
    )

    (
        standings_rows,
        exact_standings_duplicates,
    ) = build_standings(
        groups,
        team_abbr_lookup,
        set(team_index),
        season,
        season_type,
    )

    validate_standings_rows(
        standings_rows,
        master_rows,
        season,
        season_type,
    )

    record_scopes = {
        (
            str(
                row.get(
                    "standings_type",
                    "",
                )
            ),
            str(
                row.get(
                    "record_name",
                    "",
                )
            ),
            str(
                row.get(
                    "record_type",
                    "",
                )
            ),
            str(
                row.get(
                    "record_abbreviation",
                    "",
                )
            ),
        )
        for row in standings_rows
    }

    report.update_details(
        {
            "master_rows": (
                len(
                    master_rows
                )
            ),
            "standings_rows": (
                len(
                    standings_rows
                )
            ),
            "standings_record_scopes": (
                len(
                    record_scopes
                )
            ),
            "exact_standings_duplicates_removed": (
                exact_standings_duplicates
            ),
            "standings_conflicts": 0,
            "output_modified": False,
        }
    )

    publish_bundle(
        master_rows,
        standings_rows,
    )

    report.set_rows(
        rows_out=len(
            master_rows
        ),
    )

    report.update_details(
        {
            "output_modified": True,
            "league_master_path": str(
                LEAGUE_MASTER_PATH
            ),
            "league_standings_path": str(
                LEAGUE_STANDINGS_PATH
            ),
        }
    )

    print(
        "league_master.py completed"
    )

    print(
        f"season={season} "
        f"season_type={season_type} "
        f"week={week}"
    )

    print(
        "team_master_unique_teams="
        f"{len(team_index)}"
    )

    print(
        f"groups_discovered="
        f"{len(groups)}"
    )

    print(
        f"master_rows="
        f"{len(master_rows)}"
    )

    print(
        f"standings_rows="
        f"{len(standings_rows)}"
    )

    print(
        "abbreviations_resolved_from_espn="
        f"{abbreviations_resolved}"
    )

    print(
        f"league_master="
        f"{LEAGUE_MASTER_PATH}"
    )

    print(
        f"league_standings="
        f"{LEAGUE_STANDINGS_PATH}"
    )

    return 0


def main() -> int:
    reset_runtime_state()

    with PipelineReporter(
        script=__file__,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        extra_context={
            "script_version": (
                SCRIPT_VERSION
            ),
            "source": "ESPN Core API",
        },
    ) as report:
        report.add_input(
            CURRENT_WEEK_CONFIG_PATH
        )

        report.add_input(
            TEAM_MASTER_PATH
        )

        report.add_output(
            LEAGUE_MASTER_PATH
        )

        report.add_output(
            LEAGUE_STANDINGS_PATH
        )

        report.set_detail(
            "output_modified",
            False,
        )

        try:
            return run(report)

        finally:
            report.update_details(
                {
                    "espn_request_count": (
                        _REQUEST_COUNT
                    ),
                    "espn_request_attempt_count": (
                        _REQUEST_ATTEMPT_COUNT
                    ),
                    "espn_request_retry_count": (
                        _REQUEST_RETRY_COUNT
                    ),
                    "espn_recovered_transient_requests": (
                        _RECOVERED_TRANSIENT_REQUESTS
                    ),
                    "espn_exhausted_transient_failures": (
                        _EXHAUSTED_TRANSIENT_FAILURES
                    ),
                    "espn_retry_details": (
                        list(
                            _RETRY_DETAILS
                        )
                    ),
                    "espn_request_failures": (
                        len(
                            _REQUEST_FAILURES
                        )
                    ),
                    "espn_request_failure_details": (
                        list(
                            _REQUEST_FAILURES
                        )
                    ),
                }
            )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )

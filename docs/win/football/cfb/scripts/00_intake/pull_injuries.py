#!/usr/bin/env python3
"""
pull_injuries.py

Pull the configured CFB injury report from ESPN's site API, validate provider
season/team/player identities, and atomically publish the season injury file.

Inputs:
    docs/win/football/cfb/config/current_week.yaml
    docs/win/football/cfb/data/master/league_master.csv
    docs/win/football/cfb/config/mapping/team_map.csv

Output:
    docs/win/football/cfb/00_intake/injuries/{season}_injuries.csv

This script is an automated CFB pipeline intake step.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import urllib.parse
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


CURRENT_WEEK_CONFIG_PATH = CFB_ROOT / "config" / "current_week.yaml"
LEAGUE_MASTER_PATH = CFB_ROOT / "data" / "master" / "league_master.csv"
TEAM_MAP_PATH = CFB_ROOT / "config" / "mapping" / "team_map.csv"
OUTPUT_DIR = CFB_ROOT / "00_intake" / "injuries"
REPORT_ROOT = CFB_ROOT / "errors"

SCRIPT_VERSION = "cfb-pull-injuries-v4-stale-zero-current-2026-09-15"

# Keep intake freshness aligned with the projection's default
# injury-report freshness window. ESPN's payload timestamp is used
# as the reference clock so runner clock drift cannot change results.
MAX_REPORT_AGE_DAYS = 60.0

INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/"
    "college-football/injuries"
)

ESPN_SITE_HOST = "site.api.espn.com"
ESPN_WEB_HOSTS = {
    "espn.com",
    "www.espn.com",
}

PLAYER_ID_PATTERN = re.compile(
    r"/id/(\d+)(?:/|$)"
)

OUTPUT_HEADERS = [
    "season",
    "team",
    "player_id",
    "player_name",
    "position",
    "game_status",
    "report_date",
]

_REQUEST_COUNT = 0
_REQUEST_FAILURES: list[dict[str, str]] = []

_PROVIDER_STATUS = ""
_PROVIDER_TIMESTAMP = ""
_PROVIDER_TIMESTAMP_UTC: datetime | None = None
_PROVIDER_SEASON: int | None = None
_PROVIDER_SEASON_TYPE: int | None = None

_PROVIDER_TEAM_GROUP_COUNT = 0
_PROVIDER_TEAM_GROUPS_WITH_INJURIES = 0
_PROVIDER_TEAM_GROUPS_WITHOUT_INJURIES = 0

_RAW_INJURY_COUNT = 0
_FRESH_INJURY_COUNT = 0
_STALE_INJURY_COUNT = 0
_DUPLICATE_INJURY_ID_COUNT = 0
_DUPLICATE_OUTPUT_IDENTITY_COUNT = 0
_MISSING_PLAYER_ID_COUNT = 0
_REPORT_YEAR_MISMATCH_COUNT = 0

_FOREIGN_TEAM_IDS: set[str] = set()
_PROVIDER_TEAM_IDS: set[str] = set()
_REPORT_DATES_UTC: list[datetime] = []


class InjuryValidationError(RuntimeError):
    pass


def reset_runtime_state() -> None:
    global _REQUEST_COUNT
    global _PROVIDER_STATUS
    global _PROVIDER_TIMESTAMP
    global _PROVIDER_TIMESTAMP_UTC
    global _PROVIDER_SEASON
    global _PROVIDER_SEASON_TYPE
    global _PROVIDER_TEAM_GROUP_COUNT
    global _PROVIDER_TEAM_GROUPS_WITH_INJURIES
    global _PROVIDER_TEAM_GROUPS_WITHOUT_INJURIES
    global _RAW_INJURY_COUNT
    global _FRESH_INJURY_COUNT
    global _STALE_INJURY_COUNT
    global _DUPLICATE_INJURY_ID_COUNT
    global _DUPLICATE_OUTPUT_IDENTITY_COUNT
    global _MISSING_PLAYER_ID_COUNT
    global _REPORT_YEAR_MISMATCH_COUNT

    _REQUEST_COUNT = 0
    _PROVIDER_STATUS = ""
    _PROVIDER_TIMESTAMP = ""
    _PROVIDER_TIMESTAMP_UTC = None
    _PROVIDER_SEASON = None
    _PROVIDER_SEASON_TYPE = None

    _PROVIDER_TEAM_GROUP_COUNT = 0
    _PROVIDER_TEAM_GROUPS_WITH_INJURIES = 0
    _PROVIDER_TEAM_GROUPS_WITHOUT_INJURIES = 0

    _RAW_INJURY_COUNT = 0
    _FRESH_INJURY_COUNT = 0
    _STALE_INJURY_COUNT = 0
    _DUPLICATE_INJURY_ID_COUNT = 0
    _DUPLICATE_OUTPUT_IDENTITY_COUNT = 0
    _MISSING_PLAYER_ID_COUNT = 0
    _REPORT_YEAR_MISMATCH_COUNT = 0

    _REQUEST_FAILURES.clear()
    _FOREIGN_TEAM_IDS.clear()
    _PROVIDER_TEAM_IDS.clear()
    _REPORT_DATES_UTC.clear()


def parse_positive_int_text(
    value: object,
    *,
    label: str,
) -> str:
    if isinstance(value, bool):
        raise InjuryValidationError(
            f"{label} must be a positive integer, not boolean"
        )

    text = str(value or "").strip()

    if not re.fullmatch(r"\d+", text):
        raise InjuryValidationError(
            f"{label} must be a positive integer: {value!r}"
        )

    parsed = int(text)

    if parsed <= 0:
        raise InjuryValidationError(
            f"{label} must be positive: {parsed}"
        )

    return str(parsed)


def parse_positive_int(
    value: object,
    *,
    label: str,
) -> int:
    return int(
        parse_positive_int_text(
            value,
            label=label,
        )
    )


def load_current_week() -> tuple[int, int, int]:
    if not CURRENT_WEEK_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing current-week config: {CURRENT_WEEK_CONFIG_PATH}"
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
                f"Current-week config missing required key: {key}"
            )

        values[key] = parse_positive_int(
            payload.get(key),
            label=f"current_week.{key}",
        )

    if values["season"] < 2000:
        raise ValueError(
            f"Invalid configured season: {values['season']}"
        )

    return (
        values["season"],
        values["season_type"],
        values["week"],
    )


def load_authoritative_team_ids(
    *,
    season: int,
    season_type: int,
) -> list[str]:
    if not LEAGUE_MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing league master: {LEAGUE_MASTER_PATH}"
        )

    with LEAGUE_MASTER_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        required = {
            "team_id",
            "season",
            "season_type",
        }

        missing = sorted(
            required - set(fieldnames)
        )

        if missing:
            raise ValueError(
                "league_master.csv missing required columns: "
                f"{missing}"
            )

        team_ids: set[str] = set()

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            if None in row:
                raise ValueError(
                    "league_master.csv contains malformed row at "
                    f"CSV line {row_number}"
                )

            team_id = parse_positive_int_text(
                row.get("team_id"),
                label=(
                    "league_master.csv team_id at "
                    f"CSV line {row_number}"
                ),
            )

            row_season = parse_positive_int(
                row.get("season"),
                label=(
                    "league_master.csv season at "
                    f"CSV line {row_number}"
                ),
            )

            row_season_type = parse_positive_int(
                row.get("season_type"),
                label=(
                    "league_master.csv season_type at "
                    f"CSV line {row_number}"
                ),
            )

            if row_season != season:
                raise ValueError(
                    "league_master.csv season mismatch at "
                    f"CSV line {row_number}: "
                    f"expected={season}, actual={row_season}"
                )

            if row_season_type != season_type:
                raise ValueError(
                    "league_master.csv season_type mismatch at "
                    f"CSV line {row_number}: "
                    f"expected={season_type}, "
                    f"actual={row_season_type}"
                )

            if team_id in team_ids:
                raise ValueError(
                    "league_master.csv contains duplicate "
                    f"team_id={team_id}"
                )

            team_ids.add(team_id)

    if not team_ids:
        raise ValueError(
            "league_master.csv contains no authoritative teams"
        )

    return sorted(
        team_ids,
        key=lambda value: int(value),
    )


def load_canonical_team_names(
    authoritative_team_ids: list[str],
) -> dict[str, str]:
    if not TEAM_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing team map: {TEAM_MAP_PATH}"
        )

    authoritative_set = set(
        authoritative_team_ids
    )

    canonical_by_id: dict[str, str] = {}

    with TEAM_MAP_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        required = {
            "team_id",
            "canonical_team",
        }

        missing = sorted(
            required - set(fieldnames)
        )

        if missing:
            raise ValueError(
                "team_map.csv missing required columns: "
                f"{missing}"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            if None in row:
                raise ValueError(
                    "team_map.csv contains malformed row at "
                    f"CSV line {row_number}"
                )

            raw_team_id = str(
                row.get("team_id") or ""
            ).strip()

            if not raw_team_id:
                continue

            team_id = parse_positive_int_text(
                raw_team_id,
                label=(
                    "team_map.csv team_id at "
                    f"CSV line {row_number}"
                ),
            )

            if team_id not in authoritative_set:
                continue

            canonical = str(
                row.get("canonical_team") or ""
            ).strip()

            if not canonical:
                raise ValueError(
                    "team_map.csv has blank canonical_team for "
                    f"authoritative team_id={team_id} at "
                    f"CSV line {row_number}"
                )

            prior = canonical_by_id.get(
                team_id
            )

            if (
                prior is not None
                and prior != canonical
            ):
                raise ValueError(
                    "team_map.csv has conflicting canonical names for "
                    f"team_id={team_id}: "
                    f"{prior!r} vs {canonical!r}"
                )

            canonical_by_id[
                team_id
            ] = canonical

    missing_ids = sorted(
        authoritative_set
        - set(canonical_by_id),
        key=lambda value: int(value),
    )

    if missing_ids:
        raise ValueError(
            "team_map.csv is missing canonical mappings for "
            f"authoritative team IDs: {missing_ids[:50]}"
        )

    inverse: dict[str, str] = {}

    for team_id, canonical in (
        canonical_by_id.items()
    ):
        prior_id = inverse.get(
            canonical
        )

        if (
            prior_id is not None
            and prior_id != team_id
        ):
            raise ValueError(
                "team_map.csv maps one canonical team name to "
                "multiple authoritative IDs: "
                f"canonical={canonical!r}, "
                f"team_ids={prior_id},{team_id}"
            )

        inverse[
            canonical
        ] = team_id

    return canonical_by_id


def output_path_for_season(
    season: int,
) -> Path:
    return (
        OUTPUT_DIR
        / f"{season}_injuries.csv"
    )


def injuries_url(
    *,
    season: int,
    season_type: int,
) -> str:
    query = urllib.parse.urlencode(
        {
            "season": season,
            "seasontype": season_type,
        }
    )

    return (
        f"{INJURIES_URL}?{query}"
    )


def validate_injuries_url(
    url: str,
) -> str:
    text = str(
        url or ""
    ).strip()

    parsed = urllib.parse.urlparse(
        text
    )

    if parsed.scheme != "https":
        raise InjuryValidationError(
            "Injuries URL must use HTTPS: "
            f"{text!r}"
        )

    if parsed.hostname != ESPN_SITE_HOST:
        raise InjuryValidationError(
            "Unexpected injuries URL host: "
            f"{text!r}"
        )

    return text


def fetch_json(
    url: str,
    *,
    timeout: int = 30,
) -> dict:
    global _REQUEST_COUNT

    url = validate_injuries_url(
        url
    )

    _REQUEST_COUNT += 1

    request = Request(
        url,
        headers={
            "User-Agent": "cfb-pull-injuries/2.0",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(
            request,
            timeout=timeout,
        ) as response:
            status = response.status
            body = (
                response.read()
                .decode("utf-8")
            )

    except HTTPError as exc:
        error_body = ""

        try:
            error_body = (
                exc.read()
                .decode("utf-8")
            )
        except Exception:
            pass

        failure = {
            "url": url,
            "http_status": str(
                exc.code
            ),
            "error": (
                error_body
                or str(exc)
            ),
        }

        _REQUEST_FAILURES.append(
            failure
        )

        raise RuntimeError(
            "ESPN injuries request failed: "
            f"status={exc.code}, "
            f"error={failure['error']}"
        ) from exc

    except URLError as exc:
        failure = {
            "url": url,
            "http_status": "",
            "error": str(exc),
        }

        _REQUEST_FAILURES.append(
            failure
        )

        raise RuntimeError(
            f"ESPN injuries request failed: {exc}"
        ) from exc

    except Exception as exc:
        failure = {
            "url": url,
            "http_status": "",
            "error": str(exc),
        }

        _REQUEST_FAILURES.append(
            failure
        )

        raise RuntimeError(
            f"ESPN injuries request failed: {exc}"
        ) from exc

    if (
        status < 200
        or status >= 300
    ):
        failure = {
            "url": url,
            "http_status": str(
                status
            ),
            "error": body,
        }

        _REQUEST_FAILURES.append(
            failure
        )

        raise RuntimeError(
            "ESPN injuries request returned "
            f"unexpected status: {status}"
        )

    try:
        payload = json.loads(
            body
        )

    except Exception as exc:
        failure = {
            "url": url,
            "http_status": str(
                status
            ),
            "error": (
                f"JSON parse failed: {exc}"
            ),
        }

        _REQUEST_FAILURES.append(
            failure
        )

        raise RuntimeError(
            "ESPN injuries request returned malformed JSON"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        failure = {
            "url": url,
            "http_status": str(
                status
            ),
            "error": (
                "JSON root is not an object"
            ),
        }

        _REQUEST_FAILURES.append(
            failure
        )

        raise RuntimeError(
            "ESPN injuries request returned non-object JSON"
        )

    return payload


def parse_timestamp(
    value: object,
    *,
    label: str,
) -> tuple[str, datetime]:
    text = str(
        value or ""
    ).strip()

    if not text:
        raise InjuryValidationError(
            f"{label} is blank"
        )

    normalized = text

    if normalized.endswith("Z"):
        normalized = (
            normalized[:-1]
            + "+00:00"
        )

    try:
        parsed = datetime.fromisoformat(
            normalized
        )

    except ValueError as exc:
        raise InjuryValidationError(
            f"{label} is not a valid ISO timestamp: {text!r}"
        ) from exc

    if parsed.tzinfo is None:
        raise InjuryValidationError(
            f"{label} lacks timezone information: {text!r}"
        )

    return (
        text,
        parsed.astimezone(
            timezone.utc
        ),
    )


def report_age_days(
    report_date_utc: datetime,
) -> float:
    if _PROVIDER_TIMESTAMP_UTC is None:
        raise InjuryValidationError(
            "Provider timestamp is unavailable for "
            "injury freshness validation"
        )

    age_seconds = (
        _PROVIDER_TIMESTAMP_UTC
        - report_date_utc
    ).total_seconds()

    # Small provider clock skew must not make a report stale.
    if age_seconds < 0:
        return 0.0

    return (
        age_seconds
        / 86400.0
    )


def validate_provider_envelope(
    data: dict,
    *,
    season: int,
    season_type: int,
) -> list[dict]:
    global _PROVIDER_STATUS
    global _PROVIDER_TIMESTAMP
    global _PROVIDER_TIMESTAMP_UTC
    global _PROVIDER_SEASON
    global _PROVIDER_SEASON_TYPE

    status = str(
        data.get("status") or ""
    ).strip()

    _PROVIDER_STATUS = status

    if (
        status
        and status.casefold() != "success"
    ):
        raise InjuryValidationError(
            "ESPN injuries payload status "
            f"is not success: {status!r}"
        )

    provider_timestamp = str(
        data.get("timestamp") or ""
    ).strip()

    if not provider_timestamp:
        raise InjuryValidationError(
            "ESPN injuries payload timestamp is required "
            "for freshness validation"
        )

    (
        _,
        provider_timestamp_utc,
    ) = parse_timestamp(
        provider_timestamp,
        label=(
            "ESPN injuries payload timestamp"
        ),
    )

    _PROVIDER_TIMESTAMP = (
        provider_timestamp
    )
    _PROVIDER_TIMESTAMP_UTC = (
        provider_timestamp_utc
    )

    season_obj = data.get(
        "season"
    )

    if not isinstance(
        season_obj,
        dict,
    ):
        raise InjuryValidationError(
            "ESPN injuries payload season "
            "field is not an object"
        )

    provider_season = (
        parse_positive_int(
            season_obj.get("year"),
            label=(
                "ESPN injuries payload season.year"
            ),
        )
    )

    _PROVIDER_SEASON = provider_season

    if provider_season != season:
        raise InjuryValidationError(
            "ESPN injuries payload season mismatch: "
            f"expected={season}, "
            f"actual={provider_season}"
        )

    raw_type = season_obj.get(
        "type"
    )

    if raw_type not in (
        None,
        "",
    ):
        provider_type = (
            parse_positive_int(
                raw_type,
                label=(
                    "ESPN injuries payload season.type"
                ),
            )
        )

        _PROVIDER_SEASON_TYPE = (
            provider_type
        )

        if provider_type != season_type:
            raise InjuryValidationError(
                "ESPN injuries payload season_type mismatch: "
                f"expected={season_type}, "
                f"actual={provider_type}"
            )

    injuries = data.get(
        "injuries"
    )

    if not isinstance(
        injuries,
        list,
    ):
        raise InjuryValidationError(
            "ESPN injuries payload injuries "
            "field is not a list"
        )

    return injuries


def extract_playercard_ids(
    athlete: dict,
    *,
    context: str,
) -> set[str]:
    links = athlete.get(
        "links"
    )

    if links is None:
        return set()

    if not isinstance(
        links,
        list,
    ):
        raise InjuryValidationError(
            "Athlete links field is not a list "
            f"for {context}"
        )

    playercard_ids: set[str] = set()

    for link_index, link in enumerate(
        links
    ):
        if not isinstance(
            link,
            dict,
        ):
            raise InjuryValidationError(
                "Athlete links contains non-object "
                f"entry for {context}, "
                f"link_index={link_index}"
            )

        rel = link.get(
            "rel"
        )

        if rel is None:
            continue

        if not isinstance(
            rel,
            list,
        ):
            raise InjuryValidationError(
                "Athlete link rel is not a list "
                f"for {context}, "
                f"link_index={link_index}"
            )

        if "playercard" not in rel:
            continue

        href = str(
            link.get("href") or ""
        ).strip()

        parsed = urllib.parse.urlparse(
            href
        )

        if parsed.scheme not in {
            "http",
            "https",
        }:
            raise InjuryValidationError(
                "Playercard link has unsupported scheme "
                f"for {context}: {href!r}"
            )

        if parsed.hostname not in (
            ESPN_WEB_HOSTS
        ):
            raise InjuryValidationError(
                "Playercard link has unexpected host "
                f"for {context}: {href!r}"
            )

        match = PLAYER_ID_PATTERN.search(
            parsed.path
        )

        if not match:
            raise InjuryValidationError(
                "Playercard link does not contain athlete id "
                f"for {context}: {href!r}"
            )

        playercard_ids.add(
            parse_positive_int_text(
                match.group(1),
                label=(
                    f"playercard athlete id for {context}"
                ),
            )
        )

    return playercard_ids


def extract_player_id(
    athlete: dict,
    *,
    context: str,
) -> str:
    global _MISSING_PLAYER_ID_COUNT

    direct_id = str(
        athlete.get("id") or ""
    ).strip()

    parsed_direct = ""

    if direct_id:
        parsed_direct = (
            parse_positive_int_text(
                direct_id,
                label=f"athlete.id for {context}",
            )
        )

    playercard_ids = (
        extract_playercard_ids(
            athlete,
            context=context,
        )
    )

    if len(playercard_ids) > 1:
        raise InjuryValidationError(
            "Athlete has conflicting playercard IDs for "
            f"{context}: {sorted(playercard_ids)}"
        )

    playercard_id = (
        next(iter(playercard_ids))
        if playercard_ids
        else ""
    )

    if (
        parsed_direct
        and playercard_id
        and parsed_direct != playercard_id
    ):
        raise InjuryValidationError(
            "Athlete direct ID and playercard ID disagree for "
            f"{context}: direct={parsed_direct}, "
            f"playercard={playercard_id}"
        )

    player_id = (
        parsed_direct
        or playercard_id
    )

    if not player_id:
        _MISSING_PLAYER_ID_COUNT += 1

        raise InjuryValidationError(
            f"Athlete has no usable player ID for {context}"
        )

    return player_id


def validate_athlete_team(
    athlete: dict,
    *,
    team_id: str,
    context: str,
) -> None:
    if "team" not in athlete:
        return

    team_obj = athlete.get(
        "team"
    )

    if team_obj is None:
        return

    if not isinstance(
        team_obj,
        dict,
    ):
        raise InjuryValidationError(
            f"Athlete team field is not an object for {context}"
        )

    raw_athlete_team_id = str(
        team_obj.get("id") or ""
    ).strip()

    if not raw_athlete_team_id:
        return

    athlete_team_id = (
        parse_positive_int_text(
            raw_athlete_team_id,
            label=(
                f"athlete team.id for {context}"
            ),
        )
    )

    if athlete_team_id != team_id:
        raise InjuryValidationError(
            "Athlete team identity mismatch for "
            f"{context}: "
            f"group_team_id={team_id}, "
            f"athlete_team_id={athlete_team_id}"
        )


def position_abbreviation(
    athlete: dict,
    *,
    context: str,
) -> str:
    if "position" not in athlete:
        return ""

    position = athlete.get(
        "position"
    )

    if position is None:
        return ""

    if not isinstance(
        position,
        dict,
    ):
        raise InjuryValidationError(
            "Athlete position field is not an object "
            f"for {context}"
        )

    return str(
        position.get("abbreviation") or ""
    ).strip()


def build_rows(
    team_entries: list[dict],
    *,
    season: int,
    authoritative_team_ids: list[str],
    canonical_by_id: dict[str, str],
) -> list[dict[str, str]]:
    global _PROVIDER_TEAM_GROUP_COUNT
    global _PROVIDER_TEAM_GROUPS_WITH_INJURIES
    global _PROVIDER_TEAM_GROUPS_WITHOUT_INJURIES
    global _RAW_INJURY_COUNT
    global _FRESH_INJURY_COUNT
    global _STALE_INJURY_COUNT
    global _DUPLICATE_INJURY_ID_COUNT
    global _DUPLICATE_OUTPUT_IDENTITY_COUNT
    global _REPORT_YEAR_MISMATCH_COUNT

    authoritative_set = set(
        authoritative_team_ids
    )

    rows: list[dict[str, str]] = []

    injury_id_records: dict[
        str,
        dict[str, str],
    ] = {}

    output_identity_rows: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ] = {}

    _PROVIDER_TEAM_GROUP_COUNT = len(
        team_entries
    )

    for team_index, team_entry in enumerate(
        team_entries
    ):
        if not isinstance(
            team_entry,
            dict,
        ):
            raise InjuryValidationError(
                "ESPN injuries collection contains "
                "non-object team entry at "
                f"team_index={team_index}"
            )

        team_id = parse_positive_int_text(
            team_entry.get("id"),
            label=(
                "ESPN injuries team.id at "
                f"team_index={team_index}"
            ),
        )

        _PROVIDER_TEAM_IDS.add(
            team_id
        )

        if team_id not in authoritative_set:
            _FOREIGN_TEAM_IDS.add(
                team_id
            )

            raise InjuryValidationError(
                "ESPN injuries payload contains "
                f"foreign team_id={team_id}"
            )

        injuries = team_entry.get(
            "injuries"
        )

        if not isinstance(
            injuries,
            list,
        ):
            raise InjuryValidationError(
                "ESPN injuries team entry injuries "
                "field is not a list for "
                f"team_id={team_id}"
            )

        _RAW_INJURY_COUNT += len(
            injuries
        )

        if injuries:
            _PROVIDER_TEAM_GROUPS_WITH_INJURIES += 1
        else:
            _PROVIDER_TEAM_GROUPS_WITHOUT_INJURIES += 1

        canonical_team = (
            canonical_by_id[
                team_id
            ]
        )

        for injury_index, injury in enumerate(
            injuries
        ):
            if not isinstance(
                injury,
                dict,
            ):
                raise InjuryValidationError(
                    "ESPN injuries team group contains "
                    "non-object injury for "
                    f"team_id={team_id}, "
                    f"injury_index={injury_index}"
                )

            context = (
                f"team_id={team_id}, "
                f"injury_index={injury_index}"
            )

            injury_id = str(
                injury.get("id") or ""
            ).strip()

            if injury_id:
                injury_id = (
                    parse_positive_int_text(
                        injury_id,
                        label=f"injury.id for {context}",
                    )
                )

            athlete = injury.get(
                "athlete"
            )

            if not isinstance(
                athlete,
                dict,
            ):
                raise InjuryValidationError(
                    "Injury athlete field is not "
                    f"an object for {context}"
                )

            validate_athlete_team(
                athlete,
                team_id=team_id,
                context=context,
            )

            player_id = extract_player_id(
                athlete,
                context=context,
            )

            player_name = str(
                athlete.get("displayName") or ""
            ).strip()

            if not player_name:
                raise InjuryValidationError(
                    "Athlete displayName is blank "
                    f"for {context}"
                )

            position = position_abbreviation(
                athlete,
                context=context,
            )

            game_status = str(
                injury.get("status") or ""
            ).strip()

            if not game_status:
                raise InjuryValidationError(
                    "Injury status is blank "
                    f"for {context}"
                )

            (
                report_date,
                report_date_utc,
            ) = parse_timestamp(
                injury.get("date"),
                label=(
                    f"injury date for {context}"
                ),
            )

            _REPORT_DATES_UTC.append(
                report_date_utc
            )

            if (
                report_date_utc.year
                != season
            ):
                _REPORT_YEAR_MISMATCH_COUNT += 1

            age_days = report_age_days(
                report_date_utc
            )

            if age_days > MAX_REPORT_AGE_DAYS:
                _STALE_INJURY_COUNT += 1
                continue

            _FRESH_INJURY_COUNT += 1

            provider_identity = (
                team_id,
                player_id,
                game_status,
                report_date,
            )

            row = {
                "season": str(season),
                "team": canonical_team,
                "player_id": player_id,
                "player_name": player_name,
                "position": position,
                "game_status": game_status,
                "report_date": report_date,
            }

            if injury_id:
                prior_provider_row = (
                    injury_id_records.get(
                        injury_id
                    )
                )

                if (
                    prior_provider_row
                    is not None
                ):
                    _DUPLICATE_INJURY_ID_COUNT += 1

                    if (
                        prior_provider_row
                        != row
                    ):
                        raise InjuryValidationError(
                            "Conflicting duplicate provider "
                            "injury ID: "
                            f"injury_id={injury_id}, "
                            f"first={prior_provider_row}, "
                            f"second={row}"
                        )

                    continue

                injury_id_records[
                    injury_id
                ] = row.copy()

            output_identity = (
                provider_identity
            )

            prior_row = (
                output_identity_rows.get(
                    output_identity
                )
            )

            if prior_row is not None:
                _DUPLICATE_OUTPUT_IDENTITY_COUNT += 1

                if prior_row != row:
                    raise InjuryValidationError(
                        "Conflicting duplicate injury "
                        "output identity: "
                        f"identity={output_identity}, "
                        f"first={prior_row}, "
                        f"second={row}"
                    )

                continue

            output_identity_rows[
                output_identity
            ] = row

            rows.append(
                row
            )

    rows.sort(
        key=lambda row: (
            row["team"].casefold(),
            row["player_name"].casefold(),
            int(row["player_id"]),
            row["report_date"],
            row["game_status"].casefold(),
        )
    )

    return rows


def validate_output_rows(
    rows: list[dict[str, str]],
    *,
    season: int,
    canonical_by_id: dict[str, str],
) -> None:
    canonical_to_id = {
        canonical: team_id
        for team_id, canonical
        in canonical_by_id.items()
    }

    seen_identity: set[
        tuple[str, str, str, str]
    ] = set()

    for row_index, row in enumerate(
        rows
    ):
        if list(row.keys()) != OUTPUT_HEADERS:
            raise InjuryValidationError(
                "Injury output row schema mismatch at "
                f"row_index={row_index}"
            )

        row_season = parse_positive_int(
            row.get("season"),
            label=(
                "injury output season at "
                f"row_index={row_index}"
            ),
        )

        if row_season != season:
            raise InjuryValidationError(
                "Injury output season mismatch at "
                f"row_index={row_index}: "
                f"expected={season}, "
                f"actual={row_season}"
            )

        team = str(
            row.get("team") or ""
        ).strip()

        if team not in canonical_to_id:
            raise InjuryValidationError(
                "Injury output contains "
                "non-authoritative team at "
                f"row_index={row_index}: "
                f"{team!r}"
            )

        team_id = canonical_to_id[
            team
        ]

        player_id = (
            parse_positive_int_text(
                row.get("player_id"),
                label=(
                    "injury output player_id at "
                    f"row_index={row_index}"
                ),
            )
        )

        player_name = str(
            row.get("player_name") or ""
        ).strip()

        game_status = str(
            row.get("game_status") or ""
        ).strip()

        if not player_name:
            raise InjuryValidationError(
                "Injury output contains blank "
                "player_name at "
                f"row_index={row_index}"
            )

        if not game_status:
            raise InjuryValidationError(
                "Injury output contains blank "
                "game_status at "
                f"row_index={row_index}"
            )

        (
            report_date,
            report_date_utc,
        ) = parse_timestamp(
            row.get("report_date"),
            label=(
                "injury output report_date at "
                f"row_index={row_index}"
            ),
        )

        age_days = report_age_days(
            report_date_utc
        )

        if age_days > MAX_REPORT_AGE_DAYS:
            raise InjuryValidationError(
                "Injury output contains stale report at "
                f"row_index={row_index}: "
                f"report_date={report_date!r}, "
                f"age_days={age_days:.3f}, "
                f"max_age_days={MAX_REPORT_AGE_DAYS:.3f}"
            )

        identity = (
            team_id,
            player_id,
            game_status,
            report_date,
        )

        if identity in seen_identity:
            raise InjuryValidationError(
                "Injury output contains duplicate "
                f"identity: {identity}"
            )

        seen_identity.add(
            identity
        )


def temporary_path(
    final_path: Path,
) -> Path:
    return final_path.with_name(
        f".{final_path.name}."
        f"{uuid.uuid4().hex}.tmp"
    )


def write_staged_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_HEADERS,
        )

        writer.writeheader()
        writer.writerows(rows)

        handle.flush()
        os.fsync(
            handle.fileno()
        )


def validate_staged_csv(
    path: Path,
    *,
    expected_rows: list[dict[str, str]],
    season: int,
    canonical_by_id: dict[str, str],
) -> None:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        fieldnames = (
            reader.fieldnames
            or []
        )

        if fieldnames != OUTPUT_HEADERS:
            raise InjuryValidationError(
                "Serialized injuries CSV "
                "header mismatch"
            )

        rows = list(
            reader
        )

    if rows != expected_rows:
        raise InjuryValidationError(
            "Serialized injuries CSV does not "
            "exactly match validated rows"
        )

    validate_output_rows(
        rows,
        season=season,
        canonical_by_id=canonical_by_id,
    )


def publish_atomic(
    output_path: Path,
    *,
    rows: list[dict[str, str]],
    season: int,
    canonical_by_id: dict[str, str],
) -> bool:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = temporary_path(
        output_path
    )

    try:
        write_staged_csv(
            temp_path,
            rows,
        )

        validate_staged_csv(
            temp_path,
            expected_rows=rows,
            season=season,
            canonical_by_id=canonical_by_id,
        )

        if output_path.exists():
            if (
                output_path.read_bytes()
                == temp_path.read_bytes()
            ):
                return False

        os.replace(
            temp_path,
            output_path,
        )

        return True

    finally:
        try:
            temp_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass


def report_date_range() -> tuple[str, str]:
    if not _REPORT_DATES_UTC:
        return "", ""

    minimum = min(
        _REPORT_DATES_UTC
    )

    maximum = max(
        _REPORT_DATES_UTC
    )

    return (
        minimum.isoformat(),
        maximum.isoformat(),
    )


def update_report_details(
    report: PipelineReporter,
    *,
    authoritative_team_ids: list[str],
    canonical_by_id: dict[str, str],
    rows: list[dict[str, str]],
    output_path: Path | None,
    output_modified: bool | None,
) -> None:
    status_counts = Counter(
        str(
            row.get("game_status") or ""
        ).strip()
        for row in rows
        if str(
            row.get("game_status") or ""
        ).strip()
    )

    represented_names = {
        str(
            row.get("team") or ""
        ).strip()
        for row in rows
        if str(
            row.get("team") or ""
        ).strip()
    }

    canonical_to_id = {
        canonical: team_id
        for team_id, canonical
        in canonical_by_id.items()
    }

    represented_ids = sorted(
        {
            canonical_to_id[name]
            for name in represented_names
            if name in canonical_to_id
        },
        key=lambda value: int(value),
    )

    unique_players = {
        str(
            row.get("player_id") or ""
        ).strip()
        for row in rows
        if str(
            row.get("player_id") or ""
        ).strip()
    }

    (
        report_date_min,
        report_date_max,
    ) = report_date_range()

    details: dict[str, object] = {
        "provider_status": _PROVIDER_STATUS,
        "provider_timestamp": _PROVIDER_TIMESTAMP,
        "freshness_reference": "espn_payload_timestamp",
        "freshness_reference_utc": (
            _PROVIDER_TIMESTAMP_UTC.isoformat()
            if _PROVIDER_TIMESTAMP_UTC is not None
            else ""
        ),
        "max_report_age_days": MAX_REPORT_AGE_DAYS,
        "provider_season": _PROVIDER_SEASON,
        "provider_season_type": _PROVIDER_SEASON_TYPE,
        "espn_request_count": _REQUEST_COUNT,
        "espn_request_failure_count": len(
            _REQUEST_FAILURES
        ),
        "espn_request_failure_details": (
            _REQUEST_FAILURES
        ),
        "authoritative_team_count": len(
            authoritative_team_ids
        ),
        "canonical_team_mapping_count": len(
            canonical_by_id
        ),
        "provider_team_group_count": (
            _PROVIDER_TEAM_GROUP_COUNT
        ),
        "provider_unique_team_count": len(
            _PROVIDER_TEAM_IDS
        ),
        "provider_team_groups_with_injuries": (
            _PROVIDER_TEAM_GROUPS_WITH_INJURIES
        ),
        "provider_team_groups_without_injuries": (
            _PROVIDER_TEAM_GROUPS_WITHOUT_INJURIES
        ),
        "represented_authoritative_team_count": len(
            represented_ids
        ),
        "represented_authoritative_team_ids": (
            represented_ids
        ),
        "foreign_team_count": len(
            _FOREIGN_TEAM_IDS
        ),
        "foreign_team_ids": sorted(
            _FOREIGN_TEAM_IDS,
            key=lambda value: int(value),
        ),
        "raw_injury_count": _RAW_INJURY_COUNT,
        "fresh_injury_count": _FRESH_INJURY_COUNT,
        "stale_injury_count": _STALE_INJURY_COUNT,
        "stale_injury_disposition": "excluded_from_output",
        "all_provider_injuries_stale": (
            _RAW_INJURY_COUNT > 0
            and _FRESH_INJURY_COUNT == 0
        ),
        "freshness_outcome": (
            "all_stale_zero_current"
            if (
                _RAW_INJURY_COUNT > 0
                and _FRESH_INJURY_COUNT == 0
            )
            else (
                "mixed_fresh_and_stale"
                if (
                    _FRESH_INJURY_COUNT > 0
                    and _STALE_INJURY_COUNT > 0
                )
                else (
                    "fresh_only"
                    if _FRESH_INJURY_COUNT > 0
                    else "provider_zero_injuries"
                )
            )
        ),
        "published_injury_count": len(
            rows
        ),
        "unique_player_count": len(
            unique_players
        ),
        "duplicate_provider_injury_id_count": (
            _DUPLICATE_INJURY_ID_COUNT
        ),
        "duplicate_output_identity_count": (
            _DUPLICATE_OUTPUT_IDENTITY_COUNT
        ),
        "missing_player_id_count": (
            _MISSING_PLAYER_ID_COUNT
        ),
        "status_counts": dict(
            sorted(
                status_counts.items()
            )
        ),
        "provider_report_date_min": (
            report_date_min
        ),
        "provider_report_date_max": (
            report_date_max
        ),
        "report_year_mismatch_count": (
            _REPORT_YEAR_MISMATCH_COUNT
        ),
        "output_columns": OUTPUT_HEADERS,
    }

    if output_path is not None:
        details[
            "output_path"
        ] = str(
            output_path
        )

    if output_modified is not None:
        details[
            "output_modified"
        ] = output_modified

    report.update_details(
        details
    )


def run(
    report: PipelineReporter,
) -> int:
    reset_runtime_state()

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

    output_path = (
        output_path_for_season(
            season
        )
    )

    report.add_output(
        output_path
    )

    authoritative_team_ids: list[
        str
    ] = []

    canonical_by_id: dict[
        str,
        str,
    ] = {}

    rows: list[
        dict[str, str]
    ] = []

    output_modified: bool | None = None

    try:
        authoritative_team_ids = (
            load_authoritative_team_ids(
                season=season,
                season_type=season_type,
            )
        )

        canonical_by_id = (
            load_canonical_team_names(
                authoritative_team_ids
            )
        )

        url = injuries_url(
            season=season,
            season_type=season_type,
        )

        data = fetch_json(
            url
        )

        team_entries = (
            validate_provider_envelope(
                data,
                season=season,
                season_type=season_type,
            )
        )

        rows = build_rows(
            team_entries,
            season=season,
            authoritative_team_ids=(
                authoritative_team_ids
            ),
            canonical_by_id=canonical_by_id,
        )

        if (
            _RAW_INJURY_COUNT > 0
            and _FRESH_INJURY_COUNT == 0
        ):
            report.warning(
                "ESPN returned only stale injury records; "
                "all stale records were excluded and zero "
                "current injuries will be published: "
                f"raw={_RAW_INJURY_COUNT}, "
                f"stale={_STALE_INJURY_COUNT}, "
                f"fresh={_FRESH_INJURY_COUNT}, "
                f"max_age_days={MAX_REPORT_AGE_DAYS:.1f}"
            )

        elif _STALE_INJURY_COUNT:
            report.warning(
                "Excluded stale ESPN injury records: "
                f"stale={_STALE_INJURY_COUNT}, "
                f"fresh={_FRESH_INJURY_COUNT}, "
                f"max_age_days={MAX_REPORT_AGE_DAYS:.1f}"
            )

        validate_output_rows(
            rows,
            season=season,
            canonical_by_id=canonical_by_id,
        )

        output_modified = publish_atomic(
            output_path,
            rows=rows,
            season=season,
            canonical_by_id=canonical_by_id,
        )

        report.set_rows(
            rows_in=_RAW_INJURY_COUNT,
            rows_out=len(rows),
        )

        update_report_details(
            report,
            authoritative_team_ids=(
                authoritative_team_ids
            ),
            canonical_by_id=canonical_by_id,
            rows=rows,
            output_path=output_path,
            output_modified=output_modified,
        )

        return 0

    except Exception:
        report.set_rows(
            rows_in=_RAW_INJURY_COUNT,
            rows_out=len(rows),
        )

        update_report_details(
            report,
            authoritative_team_ids=(
                authoritative_team_ids
            ),
            canonical_by_id=canonical_by_id,
            rows=rows,
            output_path=output_path,
            output_modified=output_modified,
        )

        raise


def main() -> int:
    with PipelineReporter(
        script=SCRIPT_PATH,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        extra_context={
            "script_version": SCRIPT_VERSION,
            "source": "ESPN Site API",
        },
    ) as report:
        report.set_detail(
            "output_modified",
            False,
        )

        report.add_input(
            CURRENT_WEEK_CONFIG_PATH
        )
        report.add_input(
            LEAGUE_MASTER_PATH
        )
        report.add_input(
            TEAM_MAP_PATH
        )

        return run(
            report
        )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )

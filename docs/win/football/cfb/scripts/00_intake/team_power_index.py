#!/usr/bin/env python3
"""
team_power_index.py

Pull configured-season ESPN Football Power Index data, validate it against
the authoritative CFB team universe, and atomically publish the season file.

Inputs:
    docs/win/football/cfb/config/current_week.yaml
    docs/win/football/cfb/data/master/league_master.csv

Output:
    docs/win/football/cfb/data/team_power_index/
        team_power_index_{season}.csv
"""

from __future__ import annotations

from http.client import HTTPException

import csv
import json
import math
import os
import re
import sys
import urllib.parse
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request



SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from http_security import open_https
from pipeline_reporter import PipelineReporter
from pipeline_shared import (
    load_current_week_config,
    require_csv_fieldnames,
)
from type_support import ScalarValue


CURRENT_WEEK_CONFIG_PATH = CFB_ROOT / "config" / "current_week.yaml"
LEAGUE_MASTER_PATH = CFB_ROOT / "data" / "master" / "league_master.csv"
OUTPUT_DIR = CFB_ROOT / "data" / "team_power_index"
REPORT_ROOT = CFB_ROOT / "errors"

SCRIPT_VERSION = "cfb-team-power-index-v2-2026-09-15"

POWERINDEX_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/football/"
    "leagues/college-football/seasons/{season}/powerindex"
)

ESPN_CORE_HOST = "sports.core.api.espn.com"

SEASON_REF_PATTERN = re.compile(
    r"/seasons/(\d+)(?:/|$)"
)

TEAM_REF_PATTERN = re.compile(
    r"/seasons/(\d+)(?:/types/\d+)?/teams/(\d+)(?:/|$)"
)

BASE_FIELDNAMES = [
    "season",
    "team_id",
    "lastUpdated",
    "fpi",
]

@dataclass
class RuntimeState:
    request_count: int = 0
    request_failures: list[dict[str, str]] = field(
        default_factory=list
    )
    page_diagnostics: list[dict[str, object]] = field(
        default_factory=list
    )
    provider_page_count: int | None = None
    provider_count: int | None = None
    raw_item_count: int = 0
    duplicate_team_record_count: int = 0
    duplicate_predictive_name_count: int = 0
    malformed_item_count: int = 0
    parsed_team_ids: set[str] = field(
        default_factory=set
    )
    predictive_fieldnames: set[str] = field(
        default_factory=set
    )
    last_updated_values: list[datetime] = field(
        default_factory=list
    )


class PowerIndexValidationError(RuntimeError):
    pass


def parse_integer(
    value: ScalarValue,
    *,
    label: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise PowerIndexValidationError(
            f"{label} must be an integer, not boolean"
        )

    if isinstance(value, int):
        result = value

    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise PowerIndexValidationError(
                f"{label} must be an integer: {value!r}"
            )
        result = int(value)

    else:
        text = str(value or "").strip()

        if not re.fullmatch(r"[+-]?\d+", text):
            raise PowerIndexValidationError(
                f"{label} must be an integer: {value!r}"
            )

        result = int(text)

    if minimum is not None and result < minimum:
        raise PowerIndexValidationError(
            f"{label} must be >= {minimum}: {result}"
        )

    return result


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

        require_csv_fieldnames(
            fieldnames,
            {
                "team_id",
                "season",
                "season_type",
            },
            "league_master.csv",
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

            team_id = str(
                row.get("team_id") or ""
            ).strip()

            if not team_id:
                raise ValueError(
                    "league_master.csv contains blank team_id at "
                    f"CSV line {row_number}"
                )

            parsed_team_id = parse_integer(
                team_id,
                label=(
                    "league_master.csv team_id at "
                    f"CSV line {row_number}"
                ),
                minimum=1,
            )

            team_id = str(parsed_team_id)

            row_season = parse_integer(
                row.get("season"),
                label=(
                    "league_master.csv season at "
                    f"CSV line {row_number}"
                ),
                minimum=1,
            )

            row_season_type = parse_integer(
                row.get("season_type"),
                label=(
                    "league_master.csv season_type at "
                    f"CSV line {row_number}"
                ),
                minimum=1,
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
        key=int,
    )

def output_path_for_season(
    season: int,
) -> Path:
    return (
        OUTPUT_DIR
        / f"team_power_index_{season}.csv"
    )


def validate_espn_core_url(
    url: str,
    *,
    label: str,
) -> str:
    text = str(url or "").strip()

    if not text:
        raise PowerIndexValidationError(
            f"{label} is blank"
        )

    parsed = urllib.parse.urlparse(text)

    if parsed.scheme not in {"http", "https"}:
        raise PowerIndexValidationError(
            f"{label} has unsupported URL scheme: {text!r}"
        )

    if parsed.hostname != ESPN_CORE_HOST:
        raise PowerIndexValidationError(
            f"{label} has unexpected host: {text!r}"
        )

    if parsed.scheme == "http":
        return urllib.parse.urlunparse(
            ("https", parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
        )

    return text


def fetch_json(
    url: str,
    *,
    label: str,
    state: RuntimeState,
    timeout: int = 30,
) -> dict:

    url = validate_espn_core_url(
        url,
        label=f"{label} URL",
    )

    state.request_count += 1

    request = Request(
        url,
        headers={
            "User-Agent": "cfb-team-power-index/2.0",
            "Accept": "application/json",
        },
    )

    try:
        with open_https(
            request,
            allowed_hosts={ESPN_CORE_HOST},
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
        except (HTTPException, OSError, UnicodeError, ValueError):
            pass

        failure = {
            "label": label,
            "url": url,
            "http_status": str(exc.code),
            "error": error_body or str(exc),
        }

        state.request_failures.append(failure)

        raise RuntimeError(
            f"{label} request failed: "
            f"status={exc.code}, "
            f"error={failure['error']}"
        ) from exc

    except URLError as exc:
        failure = {
            "label": label,
            "url": url,
            "http_status": "",
            "error": str(exc),
        }

        state.request_failures.append(failure)

        raise RuntimeError(
            f"{label} request failed: {exc}"
        ) from exc

    except Exception as exc:
        failure = {
            "label": label,
            "url": url,
            "http_status": "",
            "error": str(exc),
        }

        state.request_failures.append(failure)

        raise RuntimeError(
            f"{label} request failed: {exc}"
        ) from exc

    if status < 200 or status >= 300:
        failure = {
            "label": label,
            "url": url,
            "http_status": str(status),
            "error": body,
        }

        state.request_failures.append(failure)

        raise RuntimeError(
            f"{label} request failed: status={status}"
        )

    try:
        payload = json.loads(body)

    except Exception as exc:
        failure = {
            "label": label,
            "url": url,
            "http_status": str(status),
            "error": f"JSON parse failed: {exc}",
        }

        state.request_failures.append(failure)

        raise RuntimeError(
            f"{label} returned malformed JSON"
        ) from exc

    if not isinstance(payload, dict):
        failure = {
            "label": label,
            "url": url,
            "http_status": str(status),
            "error": "JSON root is not an object",
        }

        state.request_failures.append(failure)

        raise RuntimeError(
            f"{label} returned non-object JSON"
        )

    return payload


def build_page_url(
    base_url: str,
    page: int,
) -> str:
    parsed = urllib.parse.urlparse(base_url)
    query = urllib.parse.parse_qs(
        parsed.query
    )
    query["page"] = [str(page)]

    return urllib.parse.urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            urllib.parse.urlencode(query, doseq=True),
            parsed.fragment,
        )
    )


def optional_nonnegative_integer(
    value: ScalarValue,
    *,
    label: str,
) -> int | None:
    if value is None:
        return None

    return parse_integer(
        value,
        label=label,
        minimum=0,
    )


def validate_collection_page(
    payload: dict,
    *,
    expected_page: int,
    expected_page_count: int | None,
    expected_count: int | None,
    state: RuntimeState,
) -> tuple[
    list[dict],
    int,
    int | None,
]:
    if "pageCount" not in payload:
        raise PowerIndexValidationError(
            "Power-index collection missing pageCount "
            f"on page {expected_page}"
        )

    page_count = parse_integer(
        payload.get("pageCount"),
        label=(
            "power-index pageCount on "
            f"page {expected_page}"
        ),
        minimum=1,
    )

    if page_count < expected_page:
        raise PowerIndexValidationError(
            "Power-index pageCount is smaller than "
            f"requested page: page={expected_page}, "
            f"pageCount={page_count}"
        )

    if (
        expected_page_count is not None
        and page_count != expected_page_count
    ):
        raise PowerIndexValidationError(
            "Power-index pageCount changed during pagination: "
            f"expected={expected_page_count}, "
            f"actual={page_count}, "
            f"page={expected_page}"
        )

    page_index = payload.get("pageIndex")

    if page_index is not None:
        parsed_page_index = parse_integer(
            page_index,
            label=(
                "power-index pageIndex on "
                f"page {expected_page}"
            ),
            minimum=1,
        )

        if parsed_page_index != expected_page:
            raise PowerIndexValidationError(
                "Power-index page identity mismatch: "
                f"requested={expected_page}, "
                f"returned={parsed_page_index}"
            )

    provider_count = optional_nonnegative_integer(
        payload.get("count"),
        label=(
            "power-index count on "
            f"page {expected_page}"
        ),
    )

    if (
        expected_count is not None
        and provider_count != expected_count
    ):
        raise PowerIndexValidationError(
            "Power-index count changed or disappeared "
            "during pagination: "
            f"expected={expected_count}, "
            f"actual={provider_count}, "
            f"page={expected_page}"
        )

    page_size = optional_nonnegative_integer(
        payload.get("pageSize"),
        label=(
            "power-index pageSize on "
            f"page {expected_page}"
        ),
    )

    if page_size == 0:
        raise PowerIndexValidationError(
            f"Power-index pageSize is zero on page {expected_page}"
        )

    items = payload.get("items")

    if not isinstance(items, list):
        raise PowerIndexValidationError(
            "Power-index items field is not a list "
            f"on page {expected_page}"
        )

    if (
        page_size is not None
        and len(items) > page_size
    ):
        raise PowerIndexValidationError(
            "Power-index page contains more items than "
            f"pageSize on page {expected_page}: "
            f"rows={len(items)}, pageSize={page_size}"
        )

    state.page_diagnostics.append(
        {
            "page": expected_page,
            "rows": len(items),
            "page_index": page_index,
            "page_count": page_count,
            "page_size": page_size,
            "provider_count": provider_count,
        }
    )

    return items, page_count, provider_count


def fetch_all_items(
    *,
    season: int,
    state: RuntimeState,
) -> list[dict]:

    base_url = POWERINDEX_URL_TEMPLATE.format(
        season=season
    )

    first_url = build_page_url(
        base_url,
        1,
    )

    first_payload = fetch_json(
        first_url,
        label="team power index page 1",
        state=state,
    )

    (
        first_items,
        page_count,
        provider_count,
    ) = validate_collection_page(
        first_payload,
        expected_page=1,
        expected_page_count=None,
        expected_count=None,
        state=state,
    )

    state.provider_page_count = page_count
    state.provider_count = provider_count

    all_items = list(first_items)

    for page in range(
        2,
        page_count + 1,
    ):
        page_url = build_page_url(
            base_url,
            page,
        )

        page_payload = fetch_json(
            page_url,
            label=f"team power index page {page}",
            state=state,
        )

        (
            page_items,
            returned_page_count,
            returned_count,
        ) = validate_collection_page(
            page_payload,
            expected_page=page,
            expected_page_count=page_count,
            expected_count=provider_count,
            state=state,
        )

        if returned_page_count != page_count:
            raise PowerIndexValidationError(
                "Unexpected pageCount mismatch after validation"
            )

        if (
            provider_count is not None
            and returned_count != provider_count
        ):
            raise PowerIndexValidationError(
                "Unexpected provider count mismatch after validation"
            )

        all_items.extend(page_items)

    state.raw_item_count = len(all_items)

    if not all_items:
        raise PowerIndexValidationError(
            "No team power index data was returned"
        )

    if (
        provider_count is not None
        and len(all_items) != provider_count
    ):
        raise PowerIndexValidationError(
            "Fetched power-index item count does not match "
            "provider count: "
            f"fetched={len(all_items)}, "
            f"provider_count={provider_count}"
        )

    return all_items


def extract_season_from_ref(
    ref_url: str,
    *,
    label: str,
) -> int:
    text = validate_espn_core_url(
        ref_url,
        label=label,
    )

    parsed = urllib.parse.urlparse(text)
    match = SEASON_REF_PATTERN.search(
        parsed.path
    )

    if not match:
        raise PowerIndexValidationError(
            f"{label} does not contain season identity: {text!r}"
        )

    return parse_integer(
        match.group(1),
        label=f"{label} season",
        minimum=1,
    )


def item_season_year(
    item: dict,
) -> int:
    if "season" not in item:
        raise PowerIndexValidationError(
            "Power-index item missing explicit season metadata"
        )

    season_obj = item.get("season")

    if isinstance(season_obj, dict):
        for key in (
            "year",
            "season",
        ):
            if season_obj.get(key) is not None:
                return parse_integer(
                    season_obj.get(key),
                    label=(
                        "power-index item season."
                        f"{key}"
                    ),
                    minimum=1,
                )

        season_ref = str(
            season_obj.get("$ref") or ""
        ).strip()

        if season_ref:
            return extract_season_from_ref(
                season_ref,
                label="power-index item season $ref",
            )

        raise PowerIndexValidationError(
            "Power-index item season object has no usable "
            "year or $ref"
        )

    if isinstance(season_obj, str):
        text = season_obj.strip()

        if "/seasons/" in text:
            return extract_season_from_ref(
                text,
                label="power-index item season",
            )

    return parse_integer(
        season_obj,
        label="power-index item season",
        minimum=1,
    )


def extract_team_identity(
    item: dict,
) -> tuple[int, str]:
    team_obj = item.get("team")

    if not isinstance(team_obj, dict):
        raise PowerIndexValidationError(
            "Power-index item missing team object"
        )

    team_ref = validate_espn_core_url(
        str(team_obj.get("$ref") or ""),
        label="power-index team $ref",
    )

    parsed = urllib.parse.urlparse(
        team_ref
    )

    match = TEAM_REF_PATTERN.search(
        parsed.path
    )

    if not match:
        raise PowerIndexValidationError(
            "Power-index team $ref does not contain "
            f"season/team identity: {team_ref!r}"
        )

    ref_season = parse_integer(
        match.group(1),
        label="power-index team $ref season",
        minimum=1,
    )

    team_id = str(
        parse_integer(
            match.group(2),
            label="power-index team $ref team_id",
            minimum=1,
        )
    )

    return ref_season, team_id


def scalar_to_text(
    value: ScalarValue,
    *,
    label: str,
) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        (
            dict,
            list,
            tuple,
            set,
        ),
    ):
        raise PowerIndexValidationError(
            f"{label} must be scalar: {type(value).__name__}"
        )

    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value).strip()


def parse_finite_number(
    value: ScalarValue,
    *,
    label: str,
) -> float:
    text = scalar_to_text(
        value,
        label=label,
    )

    if not text:
        raise PowerIndexValidationError(
            f"{label} is blank"
        )

    try:
        number = float(text)
    except ValueError as exc:
        raise PowerIndexValidationError(
            f"{label} is not numeric: {text!r}"
        ) from exc

    if not math.isfinite(number):
        raise PowerIndexValidationError(
            f"{label} is not finite: {text!r}"
        )

    return number


def parse_provider_timestamp(
    value: ScalarValue,
    *,
    label: str,
) -> tuple[str, datetime]:
    text = scalar_to_text(
        value,
        label=label,
    )

    if not text:
        raise PowerIndexValidationError(
            f"{label} is blank"
        )

    iso_text = text

    if iso_text.endswith("Z"):
        iso_text = (
            iso_text[:-1]
            + "+00:00"
        )

    try:
        parsed = datetime.fromisoformat(
            iso_text
        )
    except ValueError as exc:
        raise PowerIndexValidationError(
            f"{label} is not a valid ISO timestamp: {text!r}"
        ) from exc

    if parsed.tzinfo is None:
        raise PowerIndexValidationError(
            f"{label} lacks timezone information: {text!r}"
        )

    return (
        text,
        parsed.astimezone(timezone.utc),
    )



def _parse_powerindex_predictives(
    predictives: list,
    *,
    item_index: int,
    team_id: str,
    row: dict[str, str],
    item_fieldnames: list[str],
    seen_predictives: dict[
        str,
        tuple[str, str],
    ],
    state: RuntimeState,
) -> int:
    fpi_occurrences = 0

    for predictive_index, stat in enumerate(
        predictives
    ):
        if not isinstance(stat, dict):
            raise PowerIndexValidationError(
                "Power-index predictive entry is not an object at "
                f"item_index={item_index}, "
                f"team_id={team_id}, "
                f"predictive_index={predictive_index}"
            )

        raw_name = str(
            stat.get("name") or ""
        ).strip()

        if not raw_name:
            raise PowerIndexValidationError(
                "Power-index predictive entry has blank name at "
                f"item_index={item_index}, "
                f"team_id={team_id}, "
                f"predictive_index={predictive_index}"
            )

        name_key = raw_name.casefold()

        canonical_name = (
            "fpi"
            if name_key == "fpi"
            else raw_name
        )

        if name_key in {
            "season",
            "team_id",
            "lastupdated",
        }:
            raise PowerIndexValidationError(
                "Power-index predictive name collides with "
                f"base output column: {raw_name!r}"
            )

        value_text = scalar_to_text(
            stat.get("value"),
            label=(
                "power-index predictive value "
                f"{raw_name!r} for team_id={team_id}"
            ),
        )

        if name_key in seen_predictives:
            state.duplicate_predictive_name_count += 1

            prior_name, prior_value = (
                seen_predictives[name_key]
            )

            if name_key == "fpi":
                raise PowerIndexValidationError(
                    "Power-index item contains duplicate fpi "
                    f"statistics for team_id={team_id}"
                )

            if prior_value != value_text:
                raise PowerIndexValidationError(
                    "Power-index item contains conflicting "
                    "duplicate predictive statistic for "
                    f"team_id={team_id}: "
                    f"name={raw_name!r}, "
                    f"first_name={prior_name!r}, "
                    f"first_value={prior_value!r}, "
                    f"second_value={value_text!r}"
                )

            continue

        seen_predictives[
            name_key
        ] = (
            canonical_name,
            value_text,
        )

        if name_key == "fpi":
            fpi_occurrences += 1

            parse_finite_number(
                value_text,
                label=(
                    "FPI value for "
                    f"team_id={team_id}"
                ),
            )

        row[canonical_name] = value_text

        if canonical_name not in BASE_FIELDNAMES:
            item_fieldnames.append(
                canonical_name
            )

    return fpi_occurrences


def parse_powerindex_item(
    item: dict,
    *,
    item_index: int,
    season: int,
    state: RuntimeState,
) -> tuple[
    dict[str, str],
    list[str],
]:
    explicit_season = item_season_year(
        item
    )

    if explicit_season != season:
        raise PowerIndexValidationError(
            "Power-index item season mismatch at "
            f"item_index={item_index}: "
            f"expected={season}, actual={explicit_season}"
        )

    ref_season, team_id = (
        extract_team_identity(item)
    )

    if ref_season != season:
        raise PowerIndexValidationError(
            "Power-index team reference season mismatch at "
            f"item_index={item_index}, team_id={team_id}: "
            f"expected={season}, actual={ref_season}"
        )

    (
        last_updated,
        last_updated_dt,
    ) = parse_provider_timestamp(
        item.get("lastUpdated"),
        label=(
            "power-index lastUpdated at "
            f"item_index={item_index}, team_id={team_id}"
        ),
    )

    predictives = item.get(
        "predictives"
    )

    if not isinstance(
        predictives,
        list,
    ):
        raise PowerIndexValidationError(
            "Power-index predictives field is not a list at "
            f"item_index={item_index}, team_id={team_id}"
        )

    row: dict[str, str] = {
        "season": str(explicit_season),
        "team_id": team_id,
        "lastUpdated": last_updated,
    }

    item_fieldnames: list[str] = []
    seen_predictives: dict[
        str,
        tuple[str, str],
    ] = {}

    fpi_occurrences = (
        _parse_powerindex_predictives(
            predictives,
            item_index=item_index,
            team_id=team_id,
            row=row,
            item_fieldnames=item_fieldnames,
            seen_predictives=seen_predictives,
            state=state,
        )
    )

    if fpi_occurrences != 1:
        raise PowerIndexValidationError(
            "Power-index item must contain exactly one "
            f"usable fpi statistic for team_id={team_id}; "
            f"found={fpi_occurrences}"
        )

    if "fpi" not in row:
        raise PowerIndexValidationError(
            f"Power-index item has no fpi value for team_id={team_id}"
        )

    state.last_updated_values.append(
        last_updated_dt
    )

    return row, item_fieldnames


def build_rows(
    items: list[dict],
    *,
    season: int,
    state: RuntimeState,
) -> tuple[
    list[dict[str, str]],
    list[str],
]:

    rows_by_team_id: dict[
        str,
        dict[str, str],
    ] = {}

    ordered_fieldnames = list(
        BASE_FIELDNAMES
    )

    seen_fieldnames = set(
        BASE_FIELDNAMES
    )

    for item_index, item in enumerate(
        items
    ):
        if not isinstance(item, dict):
            state.malformed_item_count += 1

            raise PowerIndexValidationError(
                "Power-index collection contains non-object "
                f"item at item_index={item_index}"
            )

        try:
            (
                row,
                item_fieldnames,
            ) = parse_powerindex_item(
                item,
                item_index=item_index,
                season=season,
                state=state,
            )

        except Exception:
            state.malformed_item_count += 1
            raise

        team_id = row["team_id"]

        prior_row = rows_by_team_id.get(
            team_id
        )

        if prior_row is not None:
            state.duplicate_team_record_count += 1

            if prior_row != row:
                raise PowerIndexValidationError(
                    "Power-index collection contains conflicting "
                    "records for "
                    f"team_id={team_id}"
                )

            continue

        rows_by_team_id[
            team_id
        ] = row

        state.parsed_team_ids.add(
            team_id
        )

        for fieldname in item_fieldnames:
            if fieldname not in seen_fieldnames:
                ordered_fieldnames.append(
                    fieldname
                )
                seen_fieldnames.add(
                    fieldname
                )
                state.predictive_fieldnames.add(
                    fieldname
                )

        state.predictive_fieldnames.add(
            "fpi"
        )

    rows = list(
        rows_by_team_id.values()
    )

    rows.sort(
        key=lambda sort_row: int(
            sort_row["team_id"]
        )
    )

    normalized_rows = [
        {
            column: str(
                row.get(
                    column,
                    "",
                )
            )
            for column in ordered_fieldnames
        }
        for row in rows
    ]

    return (
        normalized_rows,
        ordered_fieldnames,
    )


def validate_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    *,
    authoritative_team_ids: list[str],
    season: int,
) -> tuple[
    list[str],
    list[str],
    int,
]:
    if not rows:
        raise PowerIndexValidationError(
            "No usable team power index rows were built"
        )

    if fieldnames[:4] != BASE_FIELDNAMES:
        raise PowerIndexValidationError(
            "Power-index output base-column order mismatch"
        )

    if len(fieldnames) != len(set(fieldnames)):
        raise PowerIndexValidationError(
            "Power-index output contains duplicate column names"
        )

    if "fpi" not in fieldnames:
        raise PowerIndexValidationError(
            "Power-index output is missing required fpi column"
        )

    authoritative_set = set(
        authoritative_team_ids
    )

    represented: set[str] = set()
    fpi_coverage = 0

    for row_index, row in enumerate(
        rows
    ):
        if list(row.keys()) != fieldnames:
            raise PowerIndexValidationError(
                "Power-index output row schema mismatch at "
                f"row_index={row_index}"
            )

        row_season = parse_integer(
            row.get("season"),
            label=(
                "power-index output season at "
                f"row_index={row_index}"
            ),
            minimum=1,
        )

        if row_season != season:
            raise PowerIndexValidationError(
                "Power-index output season mismatch at "
                f"row_index={row_index}: "
                f"expected={season}, actual={row_season}"
            )

        team_id = str(
            row.get("team_id") or ""
        ).strip()

        parsed_team_id = str(
            parse_integer(
                team_id,
                label=(
                    "power-index output team_id at "
                    f"row_index={row_index}"
                ),
                minimum=1,
            )
        )

        if parsed_team_id != team_id:
            raise PowerIndexValidationError(
                "Power-index output team_id is not canonical "
                f"at row_index={row_index}: {team_id!r}"
            )

        if team_id in represented:
            raise PowerIndexValidationError(
                "Power-index output contains duplicate "
                f"team_id={team_id}"
            )

        represented.add(
            team_id
        )

        parse_provider_timestamp(
            row.get("lastUpdated"),
            label=(
                "power-index output lastUpdated for "
                f"team_id={team_id}"
            ),
        )

        parse_finite_number(
            row.get("fpi"),
            label=(
                "power-index output fpi for "
                f"team_id={team_id}"
            ),
        )

        fpi_coverage += 1

    missing = sorted(
        authoritative_set - represented,
        key=int,
    )

    foreign = sorted(
        represented - authoritative_set,
        key=int,
    )

    if missing or foreign:
        raise PowerIndexValidationError(
            "Power-index team coverage mismatch. "
            f"missing={missing[:50]}, "
            f"foreign={foreign[:50]}"
        )

    if len(rows) != len(
        authoritative_team_ids
    ):
        raise PowerIndexValidationError(
            "Power-index row count does not match "
            "authoritative team count: "
            f"rows={len(rows)}, "
            f"authoritative={len(authoritative_team_ids)}"
        )

    if fpi_coverage != len(
        authoritative_team_ids
    ):
        raise PowerIndexValidationError(
            "Power-index FPI coverage is incomplete: "
            f"fpi_rows={fpi_coverage}, "
            f"authoritative={len(authoritative_team_ids)}"
        )

    return (
        missing,
        foreign,
        fpi_coverage,
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
    *,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
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
    expected_fieldnames: list[str],
    authoritative_team_ids: list[str],
    season: int,
) -> None:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        actual_fieldnames = (
            reader.fieldnames
            or []
        )

        if (
            actual_fieldnames
            != expected_fieldnames
        ):
            raise PowerIndexValidationError(
                "Serialized power-index header mismatch"
            )

        rows = list(
            reader
        )

    if rows != expected_rows:
        raise PowerIndexValidationError(
            "Serialized power-index rows do not exactly "
            "match validated rows"
        )

    validate_rows(
        rows,
        expected_fieldnames,
        authoritative_team_ids=(
            authoritative_team_ids
        ),
        season=season,
    )


def publish_atomic(
    output_path: Path,
    *,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    authoritative_team_ids: list[str],
    season: int,
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
            rows=rows,
            fieldnames=fieldnames,
        )

        validate_staged_csv(
            temp_path,
            expected_rows=rows,
            expected_fieldnames=fieldnames,
            authoritative_team_ids=(
                authoritative_team_ids
            ),
            season=season,
        )

        if (
            output_path.exists()
            and output_path.read_bytes()
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
        except OSError:
            pass


def provider_last_updated_range(
    state: RuntimeState,
) -> tuple[
    str,
    str,
]:
    if not state.last_updated_values:
        return "", ""

    earliest = min(
        state.last_updated_values
    ).astimezone(
        timezone.utc
    )

    latest = max(
        state.last_updated_values
    ).astimezone(
        timezone.utc
    )

    return (
        earliest.isoformat(),
        latest.isoformat(),
    )


def update_report_details(
    report: PipelineReporter,
    state: RuntimeState,
    *,
    authoritative_team_ids: list[str],
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    output_modified: bool | None,
) -> None:
    authoritative_set = set(
        authoritative_team_ids
    )

    represented = {
        str(
            row.get(
                "team_id",
                "",
            )
        ).strip()
        for row in rows
        if str(
            row.get(
                "team_id",
                "",
            )
        ).strip()
    }

    missing = sorted(
        authoritative_set - represented,
        key=int,
    )

    foreign = sorted(
        represented - authoritative_set,
        key=int,
    )

    fpi_coverage = 0

    for row in rows:
        try:
            parse_finite_number(
                row.get("fpi"),
                label="report fpi",
            )
            fpi_coverage += 1
        except PowerIndexValidationError:
            pass

    (
        last_updated_min,
        last_updated_max,
    ) = provider_last_updated_range(
        state
    )

    details: dict[str, object] = {
        "authoritative_team_count": len(
            authoritative_team_ids
        ),
        "espn_page_count": state.provider_page_count,
        "pages_validated": len(
            state.page_diagnostics
        ),
        "page_diagnostics": state.page_diagnostics,
        "provider_reported_count": state.provider_count,
        "espn_request_count": state.request_count,
        "espn_request_failure_count": len(
            state.request_failures
        ),
        "espn_request_failure_details": (
            state.request_failures
        ),
        "raw_item_count": state.raw_item_count,
        "duplicate_team_record_count": (
            state.duplicate_team_record_count
        ),
        "duplicate_predictive_name_count": (
            state.duplicate_predictive_name_count
        ),
        "malformed_item_count": (
            state.malformed_item_count
        ),
        "skipped_item_count": 0,
        "represented_team_count": len(
            represented
        ),
        "missing_team_count": len(
            missing
        ),
        "missing_team_ids": missing,
        "foreign_team_count": len(
            foreign
        ),
        "foreign_team_ids": foreign,
        "fpi_coverage_count": fpi_coverage,
        "predictive_stat_column_count": len(
            state.predictive_fieldnames
        ),
        "output_column_count": len(
            fieldnames
        ),
        "output_columns": fieldnames,
        "provider_last_updated_min": (
            last_updated_min
        ),
        "provider_last_updated_max": (
            last_updated_max
        ),
        "output_path": str(
            output_path
        ),
        "final_row_count": len(
            rows
        ),
    }

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
    state = RuntimeState()

    (
        season,
        season_type,
        week,
    ) = load_current_week_config(CURRENT_WEEK_CONFIG_PATH)

    report.season = season
    report.week = week
    report.set_detail(
        "season_type",
        season_type,
    )

    output_path = output_path_for_season(
        season
    )

    report.add_output(
        output_path
    )

    authoritative_team_ids = (
        load_authoritative_team_ids(
            season=season,
            season_type=season_type,
        )
    )

    rows: list[
        dict[str, str]
    ] = []

    fieldnames = list(
        BASE_FIELDNAMES
    )

    output_modified: bool | None = None

    try:
        items = fetch_all_items(
            season=season,
            state=state,
        )

        (
            rows,
            fieldnames,
        ) = build_rows(
            items,
            season=season,
            state=state,
        )

        validate_rows(
            rows,
            fieldnames,
            authoritative_team_ids=(
                authoritative_team_ids
            ),
            season=season,
        )

        output_modified = publish_atomic(
            output_path,
            rows=rows,
            fieldnames=fieldnames,
            authoritative_team_ids=(
                authoritative_team_ids
            ),
            season=season,
        )

        report.set_rows(
            rows_in=state.raw_item_count,
            rows_out=len(rows),
        )

        update_report_details(
            report,
            state,
            authoritative_team_ids=(
                authoritative_team_ids
            ),
            output_path=output_path,
            fieldnames=fieldnames,
            rows=rows,
            output_modified=output_modified,
        )

        return 0

    except Exception:
        report.set_rows(
            rows_in=state.raw_item_count,
            rows_out=len(rows),
        )

        update_report_details(
            report,
            state,
            authoritative_team_ids=(
                authoritative_team_ids
            ),
            output_path=output_path,
            fieldnames=fieldnames,
            rows=rows,
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
            "source": "ESPN Core API",
        },
    ) as report:
        report.add_input(
            CURRENT_WEEK_CONFIG_PATH
        )
        report.add_input(
            LEAGUE_MASTER_PATH
        )

        return run(
            report
        )

    raise RuntimeError("context manager unexpectedly suppressed an exception")

if __name__ == "__main__":
    raise SystemExit(
        main()
    )

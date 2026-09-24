#!/usr/bin/env python3
"""
Build travel features for the configured CFB target week.

The configured weekly schedule is authoritative for game identity and venue
name. Every scheduled venue must resolve to stadium_map.csv. Missing team home
origins are allowed as a degraded condition and leave only that team's travel
features blank.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
import unicodedata
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter
from pipeline_shared import write_csv_rows_durable
from type_support import ScalarValue

CONFIG_PATH = CFB_ROOT / "config" / "current_week.yaml"
SCHEDULE_DIR = CFB_ROOT / "00_intake" / "schedule" / "weekly"
STADIUM_MAP_PATH = CFB_ROOT / "config" / "mapping" / "stadium_map.csv"
OUTPUT_DIR = CFB_ROOT / "data" / "travel"
REPORT_ROOT = CFB_ROOT / "errors"

SCRIPT_VERSION = "cfb-build-travel-v2-2026-09-16"

OUTPUT_HEADERS = [
    "game_id",
    "away_team",
    "home_team",
    "stadium",
    "neutral_site_flag",
    "venue_resolution_status",
    "venue_lat",
    "venue_lon",
    "venue_timezone",
    "venue_country",
    "away_home_lat",
    "away_home_lon",
    "away_home_timezone",
    "away_miles_traveled",
    "away_time_zone_change_hours",
    "away_time_zones_crossed",
    "away_east_to_west",
    "away_west_to_east",
    "home_home_lat",
    "home_home_lon",
    "home_home_timezone",
    "home_miles_traveled",
    "home_time_zone_change_hours",
    "home_time_zones_crossed",
    "home_east_to_west",
    "home_west_to_east",
    "international_flag",
]

SCHEDULE_REQUIRED_COLUMNS = {
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "away_team",
    "home_team",
    "stadium",
    "neutral_site",
    "game_timezone",
}

STADIUM_REQUIRED_COLUMNS = {
    "team",
    "stadium",
    "venue_full_name",
    "latitude",
    "longitude",
    "timezone",
    "venue_country",
}

RESOLVED_VENUE_STATUSES = {
    "resolved_schedule_stadium",
    "resolved_schedule_stadium_timezone",
    "resolved_schedule_stadium_home_match",
}

TEAM_OUTPUT_FIELDS = {
    "away": (
        "away_home_lat",
        "away_home_lon",
        "away_home_timezone",
        "away_miles_traveled",
        "away_time_zone_change_hours",
        "away_time_zones_crossed",
        "away_east_to_west",
        "away_west_to_east",
    ),
    "home": (
        "home_home_lat",
        "home_home_lon",
        "home_home_timezone",
        "home_miles_traveled",
        "home_time_zone_change_hours",
        "home_time_zones_crossed",
        "home_east_to_west",
        "home_west_to_east",
    ),
}


class TravelValidationError(RuntimeError):
    pass


def clean(value: ScalarValue) -> str:
    return "" if value is None else str(value).strip()


def normalize_key(value: ScalarValue) -> str:
    value_text = clean(value)

    if not value_text:
        return ""

    value_text = unicodedata.normalize(
        "NFKD",
        value_text,
    )

    value_text = "".join(
        character
        for character in value_text
        if not unicodedata.combining(character)
    )

    value_text = value_text.casefold()

    return re.sub(
        r"[^a-z0-9]+",
        "",
        value_text,
    )


def strip_parenthetical(value: ScalarValue) -> str:
    value_text = clean(value)

    if not value_text:
        return ""

    return re.sub(
        r"\s*\([^)]*\)\s*$",
        "",
        value_text,
    ).strip()


def positive_int(value: ScalarValue, *, label: str) -> int:
    value_text = clean(value)

    if not re.fullmatch(r"\d+", value_text):
        raise TravelValidationError(
            f"{label} must be a positive integer: {value!r}"
        )

    parsed = int(value_text)

    if parsed <= 0:
        raise TravelValidationError(
            f"{label} must be positive: {parsed}"
        )

    return parsed


def parse_binary_flag(value: ScalarValue, *, label: str) -> int:
    value_text = clean(value).casefold()

    if value_text in {"1", "true", "yes", "y"}:
        return 1

    if value_text in {"0", "false", "no", "n"}:
        return 0

    raise TravelValidationError(
        f"{label} must be a recognized boolean/0/1 value: {value!r}"
    )


def finite_float(value: ScalarValue, *, label: str) -> float:
    value_text = clean(value)

    if not value_text:
        raise TravelValidationError(
            f"{label} is blank"
        )

    try:
        number = float(value_text)
    except (TypeError, ValueError) as exc:
        raise TravelValidationError(
            f"{label} is not numeric: {value_text!r}"
        ) from exc

    if not math.isfinite(number):
        raise TravelValidationError(
            f"{label} is not finite: {value_text!r}"
        )

    return number


def coordinate_pair(
    row: dict[str, str],
    *,
    label: str,
) -> tuple[float, float]:
    latitude = finite_float(
        row.get("latitude"),
        label=f"{label} latitude",
    )

    longitude = finite_float(
        row.get("longitude"),
        label=f"{label} longitude",
    )

    if not -90.0 <= latitude <= 90.0:
        raise TravelValidationError(
            f"{label} latitude outside [-90, 90]: {latitude}"
        )

    if not -180.0 <= longitude <= 180.0:
        raise TravelValidationError(
            f"{label} longitude outside [-180, 180]: {longitude}"
        )

    return latitude, longitude


def validate_timezone(
    value: ScalarValue,
    *,
    label: str,
) -> str:
    timezone_name = clean(value)

    if not timezone_name:
        raise TravelValidationError(
            f"{label} is blank"
        )

    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise TravelValidationError(
            f"{label} is invalid: {timezone_name!r}"
        ) from exc

    return timezone_name


def validate_date(
    value: ScalarValue,
    *,
    label: str,
) -> str:
    value_text = clean(value)

    if not value_text:
        raise TravelValidationError(
            f"{label} is blank"
        )

    try:
        datetime.strptime(
            value_text,
            "%Y-%m-%d",
        )
    except ValueError as exc:
        raise TravelValidationError(
            f"{label} must use YYYY-MM-DD: {value_text!r}"
        ) from exc

    return value_text


def load_csv(
    path: Path,
    *,
    label: str,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(
                handle,
                strict=True,
            )

            fieldnames = list(
                reader.fieldnames or []
            )

            rows: list[dict[str, str]] = []

            for line_number, row in enumerate(
                reader,
                start=2,
            ):
                if (
                    None in row
                    or any(
                        value is None
                        for value in row.values()
                    )
                ):
                    raise TravelValidationError(
                        f"{label} malformed row at CSV line {line_number}"
                    )

                rows.append({
                    key: clean(value)
                    for key, value in row.items()
                })

    except csv.Error as exc:
        raise TravelValidationError(
            f"Malformed CSV in {label}: {path}: {exc}"
        ) from exc

    if not fieldnames:
        raise TravelValidationError(
            f"{label} has no CSV header: {path}"
        )

    return fieldnames, rows


def require_columns(
    fieldnames: list[str],
    required: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(
        required - set(fieldnames)
    )

    if missing:
        raise TravelValidationError(
            f"{label} missing required columns: {missing}"
        )


def load_config() -> tuple[int, int, int]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Missing current-week config: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise TravelValidationError(
            "current_week.yaml must contain a mapping"
        )

    values: dict[str, int] = {}

    for key in (
        "season",
        "season_type",
        "week",
    ):
        if key not in payload:
            raise TravelValidationError(
                f"current_week.yaml missing required key: {key}"
            )

        values[key] = positive_int(
            payload.get(key),
            label=f"current_week.{key}",
        )

    if values["season"] < 2000:
        raise TravelValidationError(
            f"Invalid configured season: {values['season']}"
        )

    return (
        values["season"],
        values["season_type"],
        values["week"],
    )


def target_paths(
    season: int,
    week: int,
) -> tuple[Path, Path]:
    schedule_path = (
        SCHEDULE_DIR
        / f"week_{week}_CFB_weekly_schedule.csv"
    )

    output_path = (
        OUTPUT_DIR
        / f"{season}_week_{week}_travel.csv"
    )

    return schedule_path, output_path


def load_schedule(
    path: Path,
    *,
    season: int,
    season_type: int,
    week: int,
) -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, str]],
]:
    fieldnames, rows = load_csv(
        path,
        label="target weekly schedule",
    )

    require_columns(
        fieldnames,
        SCHEDULE_REQUIRED_COLUMNS,
        label="target weekly schedule",
    )

    if not rows:
        raise TravelValidationError(
            "Target weekly schedule contains no data rows"
        )

    lookup: dict[str, dict[str, str]] = {}

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        row_target = (
            positive_int(
                row.get("season"),
                label=f"schedule season line {line_number}",
            ),
            positive_int(
                row.get("season_type"),
                label=f"schedule season_type line {line_number}",
            ),
            positive_int(
                row.get("week"),
                label=f"schedule week line {line_number}",
            ),
        )

        if row_target != (
            season,
            season_type,
            week,
        ):
            raise TravelValidationError(
                "Weekly schedule target mismatch at "
                f"CSV line {line_number}: "
                f"expected={season}/{season_type}/{week}, "
                f"actual={row_target[0]}/{row_target[1]}/{row_target[2]}"
            )

        game_id = str(
            positive_int(
                row.get("game_id"),
                label=f"schedule game_id line {line_number}",
            )
        )

        if game_id in lookup:
            raise TravelValidationError(
                f"Duplicate target game_id in weekly schedule: {game_id}"
            )

        away_team = clean(
            row.get("away_team")
        )

        home_team = clean(
            row.get("home_team")
        )

        stadium = clean(
            row.get("stadium")
        )

        if not away_team or not home_team:
            raise TravelValidationError(
                f"Blank team identity for game_id={game_id}"
            )

        if normalize_key(away_team) == normalize_key(home_team):
            raise TravelValidationError(
                f"Identical home/away team for game_id={game_id}"
            )

        if not stadium:
            raise TravelValidationError(
                f"Blank scheduled stadium for game_id={game_id}"
            )

        validate_date(
            row.get("game_date"),
            label=f"game_date for game_id={game_id}",
        )

        parse_binary_flag(
            row.get("neutral_site"),
            label=f"neutral_site for game_id={game_id}",
        )

        schedule_timezone = clean(
            row.get("game_timezone")
        )

        if schedule_timezone:
            validate_timezone(
                schedule_timezone,
                label=f"game_timezone for game_id={game_id}",
            )

        normalized = dict(row)
        normalized["game_id"] = game_id

        lookup[game_id] = normalized

    return rows, lookup


def origin_signature(
    row: dict[str, str],
) -> tuple[str, str, str]:
    return (
        clean(row.get("latitude")),
        clean(row.get("longitude")),
        clean(row.get("timezone")),
    )


def load_stadium_maps() -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, list[dict[str, str]]],
]:
    fieldnames, rows = load_csv(
        STADIUM_MAP_PATH,
        label="stadium map",
    )

    require_columns(
        fieldnames,
        STADIUM_REQUIRED_COLUMNS,
        label="stadium map",
    )

    if not rows:
        raise TravelValidationError(
            "stadium_map.csv contains no data rows"
        )

    team_lookup: dict[
        str,
        dict[str, str],
    ] = {}

    venue_lookup: dict[
        str,
        list[dict[str, str]],
    ] = {}

    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        team_name = clean(
            row.get("team")
        )

        team_key = normalize_key(
            team_name
        )

        if team_key:
            prior = team_lookup.get(
                team_key
            )

            if (
                prior is not None
                and origin_signature(prior)
                != origin_signature(row)
            ):
                raise TravelValidationError(
                    "Conflicting normalized team mapping "
                    f"for team={team_name!r} at "
                    f"stadium_map.csv line {line_number}"
                )

            team_lookup.setdefault(
                team_key,
                row,
            )

        venue_names = {
            clean(
                row.get("stadium")
            ),
            clean(
                row.get("venue_full_name")
            ),
            strip_parenthetical(
                row.get("stadium")
            ),
            strip_parenthetical(
                row.get("venue_full_name")
            ),
        }

        for venue_name in venue_names:
            venue_key = normalize_key(
                venue_name
            )

            if not venue_key:
                continue

            venue_lookup.setdefault(
                venue_key,
                [],
            ).append(
                row
            )

    return (
        rows,
        team_lookup,
        venue_lookup,
    )


def validate_origin_row(
    row: dict[str, str],
    *,
    label: str,
) -> tuple[float, float, str]:
    latitude, longitude = coordinate_pair(
        row,
        label=label,
    )

    timezone_name = validate_timezone(
        row.get("timezone"),
        label=f"{label} timezone",
    )

    return (
        latitude,
        longitude,
        timezone_name,
    )


def validate_venue_row(
    row: dict[str, str],
    *,
    label: str,
) -> tuple[float, float, str]:
    latitude, longitude = coordinate_pair(
        row,
        label=label,
    )

    timezone_name = validate_timezone(
        row.get("timezone"),
        label=f"{label} timezone",
    )

    return (
        latitude,
        longitude,
        timezone_name,
    )


def same_coordinates(
    row_a: dict[str, str],
    row_b: dict[str, str],
    *,
    label: str,
) -> bool:
    lat_a, lon_a = coordinate_pair(
        row_a,
        label=f"{label} candidate A",
    )

    lat_b, lon_b = coordinate_pair(
        row_b,
        label=f"{label} candidate B",
    )

    return (
        abs(lat_a - lat_b) < 1e-7
        and abs(lon_a - lon_b) < 1e-7
    )


def dedupe_venue_rows(
    rows: list[dict[str, str]],
    *,
    game_id: str,
) -> list[dict[str, str]]:
    unique: list[
        dict[str, str]
    ] = []

    for row in rows:
        validate_venue_row(
            row,
            label=(
                "stadium_map venue candidate "
                f"for game_id={game_id}"
            ),
        )

        duplicate = False

        for existing in unique:
            if same_coordinates(
                row,
                existing,
                label=f"game_id={game_id}",
            ):
                duplicate = True
                break

        if not duplicate:
            unique.append(
                row
            )

    return unique


def resolve_venue(
    game: dict[str, str],
    *,
    home_team_row: dict[str, str] | None,
    venue_lookup: dict[
        str,
        list[dict[str, str]],
    ],
) -> tuple[dict[str, str], str]:
    game_id = clean(
        game.get("game_id")
    )

    scheduled_stadium = clean(
        game.get("stadium")
    )

    game_timezone = clean(
        game.get("game_timezone")
    )

    candidates: list[
        dict[str, str]
    ] = []

    for venue_name in {
        scheduled_stadium,
        strip_parenthetical(
            scheduled_stadium
        ),
    }:
        venue_key = normalize_key(
            venue_name
        )

        if venue_key:
            candidates.extend(
                venue_lookup.get(
                    venue_key,
                    [],
                )
            )

    candidates = dedupe_venue_rows(
        candidates,
        game_id=game_id,
    )

    if len(candidates) == 1:
        return (
            candidates[0],
            "resolved_schedule_stadium",
        )

    if (
        len(candidates) > 1
        and game_timezone
    ):
        timezone_matches = [
            row
            for row in candidates
            if clean(
                row.get("timezone")
            ) == game_timezone
        ]

        timezone_matches = dedupe_venue_rows(
            timezone_matches,
            game_id=game_id,
        )

        if len(timezone_matches) == 1:
            return (
                timezone_matches[0],
                "resolved_schedule_stadium_timezone",
            )

    if (
        len(candidates) > 1
        and home_team_row is not None
    ):
        validate_origin_row(
            home_team_row,
            label=(
                "home-team stadium mapping "
                f"for game_id={game_id}"
            ),
        )

        home_matches = [
            row
            for row in candidates
            if same_coordinates(
                row,
                home_team_row,
                label=f"game_id={game_id}",
            )
        ]

        home_matches = dedupe_venue_rows(
            home_matches,
            game_id=game_id,
        )

        if len(home_matches) == 1:
            return (
                home_matches[0],
                "resolved_schedule_stadium_home_match",
            )

    if len(candidates) > 1:
        raise TravelValidationError(
            "Ambiguous scheduled venue for "
            f"game_id={game_id}: "
            f"stadium={scheduled_stadium!r}, "
            f"candidate_count={len(candidates)}"
        )

    raise TravelValidationError(
        "Scheduled venue could not be resolved from "
        "stadium_map.csv for "
        f"game_id={game_id}: "
        f"stadium={scheduled_stadium!r}"
    )


def haversine_miles(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius_miles = 3958.8

    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

    dphi = math.radians(
        lat2 - lat1
    )

    dlambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(
            dphi / 2
        )
        ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(
            dlambda / 2
        )
        ** 2
    )

    result = (
        radius_miles
        * 2
        * math.asin(
            math.sqrt(a)
        )
    )

    if (
        not math.isfinite(result)
        or result < 0
    ):
        raise TravelValidationError(
            f"Invalid haversine result: {result}"
        )

    return result


def utc_offset_hours(
    timezone_name: str,
    game_date: str,
) -> float:
    timezone_name = validate_timezone(
        timezone_name,
        label="timezone",
    )

    validate_date(
        game_date,
        label="game_date",
    )

    dt = datetime.strptime(
        f"{game_date} 12:00",
        "%Y-%m-%d %H:%M",
    ).replace(
        tzinfo=ZoneInfo(
            timezone_name
        )
    )

    offset = dt.utcoffset()

    if offset is None:
        raise TravelValidationError(
            f"Unable to determine UTC offset for {timezone_name}"
        )

    result = (
        offset.total_seconds()
        / 3600
    )

    if not math.isfinite(result):
        raise TravelValidationError(
            f"Invalid UTC offset for {timezone_name}: {result}"
        )

    return result


def travel_direction(
    origin_longitude: float,
    destination_longitude: float,
) -> tuple[int, int]:
    if destination_longitude > origin_longitude:
        return 0, 1

    if destination_longitude < origin_longitude:
        return 1, 0

    return 0, 0


def blank_team_travel() -> dict[str, ScalarValue]:
    return {
        "home_lat": "",
        "home_lon": "",
        "home_timezone": "",
        "miles_traveled": "",
        "time_zone_change_hours": "",
        "time_zones_crossed": "",
        "east_to_west": "",
        "west_to_east": "",
    }


def team_travel_values(
    team_row: dict[str, str] | None,
    *,
    venue_latitude: float,
    venue_longitude: float,
    venue_timezone: str,
    game_date: str,
    game_id: str,
    team_label: str,
) -> dict[str, ScalarValue]:
    if team_row is None:
        return blank_team_travel()

    (
        team_latitude,
        team_longitude,
        team_timezone,
    ) = validate_origin_row(
        team_row,
        label=(
            f"{team_label} team mapping "
            f"for game_id={game_id}"
        ),
    )

    miles = round(
        haversine_miles(
            team_latitude,
            team_longitude,
            venue_latitude,
            venue_longitude,
        ),
        1,
    )

    team_offset = utc_offset_hours(
        team_timezone,
        game_date,
    )

    venue_offset = utc_offset_hours(
        venue_timezone,
        game_date,
    )

    timezone_change = round(
        venue_offset - team_offset,
        1,
    )

    timezones_crossed = round(
        abs(timezone_change),
        1,
    )

    (
        east_to_west,
        west_to_east,
    ) = travel_direction(
        team_longitude,
        venue_longitude,
    )

    return {
        "home_lat": clean(
            team_row.get("latitude")
        ),
        "home_lon": clean(
            team_row.get("longitude")
        ),
        "home_timezone": team_timezone,
        "miles_traveled": miles,
        "time_zone_change_hours": (
            timezone_change
        ),
        "time_zones_crossed": (
            timezones_crossed
        ),
        "east_to_west": east_to_west,
        "west_to_east": west_to_east,
    }


def build_output_rows(
    schedule_rows: list[dict[str, str]],
    *,
    team_lookup: dict[
        str,
        dict[str, str],
    ],
    venue_lookup: dict[
        str,
        list[dict[str, str]],
    ],
) -> tuple[
    list[dict[str, ScalarValue]],
    dict[str, Any],
]:
    output_rows: list[
        dict[str, ScalarValue]
    ] = []

    venue_status_counts: Counter[str] = Counter()
    missing_origins: list[
        dict[str, str]
    ] = []

    venue_timezone_fallback_count = 0
    venue_country_missing_count = 0

    neutral_game_count = 0
    international_game_count = 0
    away_origin_mapped_count = 0
    home_origin_mapped_count = 0

    for game in schedule_rows:
        game_id = clean(
            game.get("game_id")
        )

        away_team = clean(
            game.get("away_team")
        )

        home_team = clean(
            game.get("home_team")
        )

        stadium = clean(
            game.get("stadium")
        )

        game_date = clean(
            game.get("game_date")
        )

        neutral_site_flag = (
            parse_binary_flag(
                game.get("neutral_site"),
                label=(
                    f"neutral_site for game_id={game_id}"
                ),
            )
        )

        neutral_game_count += (
            neutral_site_flag
        )

        away_team_row = team_lookup.get(
            normalize_key(
                away_team
            )
        )

        home_team_row = team_lookup.get(
            normalize_key(
                home_team
            )
        )

        if away_team_row is None:
            missing_origins.append({
                "game_id": game_id,
                "side": "away",
                "team": away_team,
            })
        else:
            away_origin_mapped_count += 1

        if home_team_row is None:
            missing_origins.append({
                "game_id": game_id,
                "side": "home",
                "team": home_team,
            })
        else:
            home_origin_mapped_count += 1

        (
            venue_row,
            venue_status,
        ) = resolve_venue(
            game,
            home_team_row=home_team_row,
            venue_lookup=venue_lookup,
        )

        (
            venue_latitude,
            venue_longitude,
            mapped_venue_timezone,
        ) = validate_venue_row(
            venue_row,
            label=(
                f"resolved venue for game_id={game_id}"
            ),
        )

        schedule_timezone = clean(
            game.get("game_timezone")
        )

        if schedule_timezone:
            venue_timezone = (
                validate_timezone(
                    schedule_timezone,
                    label=(
                        "schedule game_timezone for "
                        f"game_id={game_id}"
                    ),
                )
            )
        else:
            venue_timezone = (
                mapped_venue_timezone
            )
            venue_timezone_fallback_count += 1

        venue_country = clean(
            venue_row.get(
                "venue_country"
            )
        )

        if venue_country:
            international_flag = (
                0
                if venue_country.upper()
                in {
                    "USA",
                    "US",
                    "UNITED STATES",
                    "UNITED STATES OF AMERICA",
                }
                else 1
            )
        else:
            international_flag = ""
            venue_country_missing_count += 1

        if international_flag == 1:
            international_game_count += 1

        away_values = team_travel_values(
            away_team_row,
            venue_latitude=venue_latitude,
            venue_longitude=venue_longitude,
            venue_timezone=venue_timezone,
            game_date=game_date,
            game_id=game_id,
            team_label="away",
        )

        home_values = team_travel_values(
            home_team_row,
            venue_latitude=venue_latitude,
            venue_longitude=venue_longitude,
            venue_timezone=venue_timezone,
            game_date=game_date,
            game_id=game_id,
            team_label="home",
        )

        venue_status_counts[
            venue_status
        ] += 1

        output_rows.append({
            "game_id": game_id,
            "away_team": away_team,
            "home_team": home_team,
            "stadium": stadium,
            "neutral_site_flag": (
                neutral_site_flag
            ),
            "venue_resolution_status": (
                venue_status
            ),
            "venue_lat": clean(
                venue_row.get("latitude")
            ),
            "venue_lon": clean(
                venue_row.get("longitude")
            ),
            "venue_timezone": (
                venue_timezone
            ),
            "venue_country": (
                venue_country
            ),
            "away_home_lat": (
                away_values["home_lat"]
            ),
            "away_home_lon": (
                away_values["home_lon"]
            ),
            "away_home_timezone": (
                away_values["home_timezone"]
            ),
            "away_miles_traveled": (
                away_values["miles_traveled"]
            ),
            "away_time_zone_change_hours": (
                away_values[
                    "time_zone_change_hours"
                ]
            ),
            "away_time_zones_crossed": (
                away_values[
                    "time_zones_crossed"
                ]
            ),
            "away_east_to_west": (
                away_values[
                    "east_to_west"
                ]
            ),
            "away_west_to_east": (
                away_values[
                    "west_to_east"
                ]
            ),
            "home_home_lat": (
                home_values["home_lat"]
            ),
            "home_home_lon": (
                home_values["home_lon"]
            ),
            "home_home_timezone": (
                home_values["home_timezone"]
            ),
            "home_miles_traveled": (
                home_values["miles_traveled"]
            ),
            "home_time_zone_change_hours": (
                home_values[
                    "time_zone_change_hours"
                ]
            ),
            "home_time_zones_crossed": (
                home_values[
                    "time_zones_crossed"
                ]
            ),
            "home_east_to_west": (
                home_values[
                    "east_to_west"
                ]
            ),
            "home_west_to_east": (
                home_values[
                    "west_to_east"
                ]
            ),
            "international_flag": (
                international_flag
            ),
        })

    away_complete = sum(
        bool(
            clean(
                row[
                    "away_miles_traveled"
                ]
            )
        )
        for row in output_rows
    )

    home_complete = sum(
        bool(
            clean(
                row[
                    "home_miles_traveled"
                ]
            )
        )
        for row in output_rows
    )

    away_timezone_complete = sum(
        bool(
            clean(
                row[
                    "away_time_zone_change_hours"
                ]
            )
        )
        for row in output_rows
    )

    home_timezone_complete = sum(
        bool(
            clean(
                row[
                    "home_time_zone_change_hours"
                ]
            )
        )
        for row in output_rows
    )

    metrics: dict[str, Any] = {
        "neutral_game_count": (
            neutral_game_count
        ),
        "international_game_count": (
            international_game_count
        ),
        "venue_resolution_status_counts": dict(
            sorted(
                venue_status_counts.items()
            )
        ),
        "unresolved_venue_count": 0,
        "venue_timezone_fallback_count": (
            venue_timezone_fallback_count
        ),
        "venue_country_missing_count": (
            venue_country_missing_count
        ),
        "away_origin_mapped_count": (
            away_origin_mapped_count
        ),
        "home_origin_mapped_count": (
            home_origin_mapped_count
        ),
        "away_origin_missing_count": sum(
            item["side"] == "away"
            for item in missing_origins
        ),
        "home_origin_missing_count": sum(
            item["side"] == "home"
            for item in missing_origins
        ),
        "missing_origin_count": (
            len(missing_origins)
        ),
        "missing_origin_details": (
            missing_origins
        ),
        "games_with_complete_away_travel": (
            away_complete
        ),
        "games_with_complete_home_travel": (
            home_complete
        ),
        "away_timezone_change_coverage": (
            away_timezone_complete
        ),
        "home_timezone_change_coverage": (
            home_timezone_complete
        ),
    }

    return output_rows, metrics


def validate_team_output_block(
    row: dict[str, str],
    *,
    side: str,
    mapped: bool,
    game_id: str,
) -> None:
    fields = TEAM_OUTPUT_FIELDS[
        side
    ]

    values = [
        clean(
            row.get(field)
        )
        for field in fields
    ]

    if not mapped:
        if any(values):
            raise TravelValidationError(
                f"Unmapped {side} team has populated "
                f"travel fields for game_id={game_id}"
            )

        return

    if any(
        not value
        for value in values
    ):
        raise TravelValidationError(
            f"Mapped {side} team has incomplete "
            f"travel fields for game_id={game_id}"
        )

    prefix = (
        "away"
        if side == "away"
        else "home"
    )

    latitude = finite_float(
        row.get(
            f"{prefix}_home_lat"
        ),
        label=(
            f"{side} home latitude for "
            f"game_id={game_id}"
        ),
    )

    longitude = finite_float(
        row.get(
            f"{prefix}_home_lon"
        ),
        label=(
            f"{side} home longitude for "
            f"game_id={game_id}"
        ),
    )

    if not -90 <= latitude <= 90:
        raise TravelValidationError(
            f"Invalid {side} latitude for game_id={game_id}"
        )

    if not -180 <= longitude <= 180:
        raise TravelValidationError(
            f"Invalid {side} longitude for game_id={game_id}"
        )

    validate_timezone(
        row.get(
            f"{prefix}_home_timezone"
        ),
        label=(
            f"{side} home timezone for "
            f"game_id={game_id}"
        ),
    )

    miles = finite_float(
        row.get(
            f"{prefix}_miles_traveled"
        ),
        label=(
            f"{side} miles for game_id={game_id}"
        ),
    )

    if miles < 0:
        raise TravelValidationError(
            f"Negative {side} travel miles for game_id={game_id}"
        )

    timezone_change = finite_float(
        row.get(
            f"{prefix}_time_zone_change_hours"
        ),
        label=(
            f"{side} timezone change for game_id={game_id}"
        ),
    )

    timezones_crossed = finite_float(
        row.get(
            f"{prefix}_time_zones_crossed"
        ),
        label=(
            f"{side} timezones crossed for game_id={game_id}"
        ),
    )

    if timezones_crossed < 0:
        raise TravelValidationError(
            f"Negative {side} timezones crossed for game_id={game_id}"
        )

    if (
        abs(
            abs(timezone_change)
            - timezones_crossed
        )
        > 1e-9
    ):
        raise TravelValidationError(
            f"{side} timezone-change mismatch for game_id={game_id}"
        )

    east_to_west = clean(
        row.get(
            f"{prefix}_east_to_west"
        )
    )

    west_to_east = clean(
        row.get(
            f"{prefix}_west_to_east"
        )
    )

    if east_to_west not in {
        "0",
        "1",
    }:
        raise TravelValidationError(
            f"Invalid {side}_east_to_west for game_id={game_id}"
        )

    if west_to_east not in {
        "0",
        "1",
    }:
        raise TravelValidationError(
            f"Invalid {side}_west_to_east for game_id={game_id}"
        )

    if (
        int(east_to_west)
        + int(west_to_east)
        > 1
    ):
        raise TravelValidationError(
            f"Conflicting {side} direction flags for game_id={game_id}"
        )


def _validate_travel_output_rows(
    rows: list[dict[str, str]],
    *,
    schedule_lookup: dict[str, dict[str, str]],
    team_lookup: dict[str, dict[str, str]],
    seen: set[str],
) -> None:
    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        if list(row.keys()) != OUTPUT_HEADERS:
            raise TravelValidationError(
                f"Travel schema mismatch at row {line_number}"
            )

        game_id = str(
            positive_int(
                row.get("game_id"),
                label=f"travel game_id row {line_number}",
            )
        )

        if game_id in seen:
            raise TravelValidationError(
                f"Duplicate travel game_id={game_id}"
            )

        game = schedule_lookup.get(
            game_id
        )

        if game is None:
            raise TravelValidationError(
                f"Foreign travel game_id={game_id}"
            )

        seen.add(
            game_id
        )

        for field in (
            "away_team",
            "home_team",
            "stadium",
        ):
            if clean(
                row.get(field)
            ) != clean(
                game.get(field)
            ):
                raise TravelValidationError(
                    f"Travel {field} mismatch for game_id={game_id}"
                )

        expected_neutral = str(
            parse_binary_flag(
                game.get("neutral_site"),
                label=(
                    "schedule neutral_site "
                    f"for game_id={game_id}"
                ),
            )
        )

        if clean(
            row.get(
                "neutral_site_flag"
            )
        ) != expected_neutral:
            raise TravelValidationError(
                f"Travel neutral flag mismatch for game_id={game_id}"
            )

        status = clean(
            row.get(
                "venue_resolution_status"
            )
        )

        if status not in RESOLVED_VENUE_STATUSES:
            raise TravelValidationError(
                "Travel contains unresolved venue "
                f"for game_id={game_id}: status={status!r}"
            )

        venue_latitude = finite_float(
            row.get("venue_lat"),
            label=(
                f"venue_lat for game_id={game_id}"
            ),
        )

        venue_longitude = finite_float(
            row.get("venue_lon"),
            label=(
                f"venue_lon for game_id={game_id}"
            ),
        )

        if not -90 <= venue_latitude <= 90:
            raise TravelValidationError(
                f"Invalid venue latitude for game_id={game_id}"
            )

        if not -180 <= venue_longitude <= 180:
            raise TravelValidationError(
                f"Invalid venue longitude for game_id={game_id}"
            )

        validate_timezone(
            row.get("venue_timezone"),
            label=(
                f"venue timezone for game_id={game_id}"
            ),
        )

        country = clean(
            row.get("venue_country")
        )

        international = clean(
            row.get(
                "international_flag"
            )
        )

        if country:
            expected_international = (
                "0"
                if country.upper()
                in {
                    "USA",
                    "US",
                    "UNITED STATES",
                    "UNITED STATES OF AMERICA",
                }
                else "1"
            )

            if international != expected_international:
                raise TravelValidationError(
                    "International flag mismatch "
                    f"for game_id={game_id}"
                )
        elif international:
            raise TravelValidationError(
                "International flag populated without "
                f"venue country for game_id={game_id}"
            )

        away_mapped = (
            normalize_key(
                game.get("away_team")
            )
            in team_lookup
        )

        home_mapped = (
            normalize_key(
                game.get("home_team")
            )
            in team_lookup
        )

        validate_team_output_block(
            row,
            side="away",
            mapped=away_mapped,
            game_id=game_id,
        )

        validate_team_output_block(
            row,
            side="home",
            mapped=home_mapped,
            game_id=game_id,
        )


def validate_output_rows(
    rows: list[dict[str, str]],
    *,
    schedule_lookup: dict[
        str,
        dict[str, str],
    ],
    team_lookup: dict[
        str,
        dict[str, str],
    ],
) -> None:
    if len(rows) != len(
        schedule_lookup
    ):
        raise TravelValidationError(
            "Travel row-count mismatch: "
            f"expected={len(schedule_lookup)}, "
            f"actual={len(rows)}"
        )

    seen: set[str] = set()

    _validate_travel_output_rows(
        rows,
        schedule_lookup=schedule_lookup,
        team_lookup=team_lookup,
        seen=seen,
    )

    if seen != set(
        schedule_lookup
    ):
        raise TravelValidationError(
            "Travel output game coverage does not "
            "match target schedule"
        )


def read_staged_rows(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle,
            strict=True,
        )

        if reader.fieldnames != OUTPUT_HEADERS:
            raise TravelValidationError(
                "Staged travel header mismatch: "
                f"{reader.fieldnames}"
            )

        rows: list[
            dict[str, str]
        ] = []

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            if (
                None in row
                or any(
                    value is None
                    for value in row.values()
                )
            ):
                raise TravelValidationError(
                    "Malformed staged travel row "
                    f"at CSV line {line_number}"
                )

            rows.append({
                key: clean(value)
                for key, value in row.items()
            })

    return rows


def publish_atomic(
    rows: list[dict[str, ScalarValue]],
    path: Path,
    *,
    schedule_lookup: dict[
        str,
        dict[str, str],
    ],
    team_lookup: dict[
        str,
        dict[str, str],
    ],
) -> tuple[bool, int]:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        write_csv_rows_durable(
            temp_path,
            rows,
            OUTPUT_HEADERS,
        )

        staged_rows = read_staged_rows(
            temp_path
        )

        validate_output_rows(
            staged_rows,
            schedule_lookup=schedule_lookup,
            team_lookup=team_lookup,
        )

        output_modified = (
            not path.exists()
            or path.read_bytes()
            != temp_path.read_bytes()
        )

        if output_modified:
            os.replace(
                temp_path,
                path,
            )
        else:
            temp_path.unlink()

        return (
            output_modified,
            len(staged_rows),
        )

    except Exception:
        try:
            temp_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

        raise


def main() -> None:
    with PipelineReporter(
        script=__file__,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        extra_context={
            "script_version": SCRIPT_VERSION,
            "source": (
                "configured weekly schedule + stadium map"
            ),
        },
    ) as report:
        report.add_input(
            CONFIG_PATH
        )

        (
            season,
            season_type,
            week,
        ) = load_config()

        report.season = season
        report.week = week

        (
            schedule_path,
            output_path,
        ) = target_paths(
            season,
            week,
        )

        report.add_input(
            schedule_path
        )
        report.add_input(
            STADIUM_MAP_PATH
        )
        report.add_output(
            output_path
        )

        report.update_details({
            "season_type": season_type,
            "target_schedule_path": (
                schedule_path
            ),
            "stadium_map_path": (
                STADIUM_MAP_PATH
            ),
            "output_path": output_path,
            "output_columns": (
                OUTPUT_HEADERS
            ),
            "output_modified": False,
        })

        (
            schedule_rows,
            schedule_lookup,
        ) = load_schedule(
            schedule_path,
            season=season,
            season_type=season_type,
            week=week,
        )

        (
            stadium_rows,
            team_lookup,
            venue_lookup,
        ) = load_stadium_maps()

        report.set_rows(
            rows_in=len(
                schedule_rows
            )
        )

        (
            output_rows,
            metrics,
        ) = build_output_rows(
            schedule_rows,
            team_lookup=team_lookup,
            venue_lookup=venue_lookup,
        )

        validate_output_rows(
            [
                {
                    key: clean(value)
                    for key, value in row.items()
                }
                for row in output_rows
            ],
            schedule_lookup=schedule_lookup,
            team_lookup=team_lookup,
        )

        (
            output_modified,
            staged_rows,
        ) = publish_atomic(
            output_rows,
            output_path,
            schedule_lookup=schedule_lookup,
            team_lookup=team_lookup,
        )

        report.set_rows(
            rows_out=staged_rows
        )

        report.update_details({
            "target_schedule_rows": (
                len(schedule_rows)
            ),
            "stadium_map_rows": (
                len(stadium_rows)
            ),
            "team_mapping_count": (
                len(team_lookup)
            ),
            "venue_lookup_key_count": (
                len(venue_lookup)
            ),
            "output_rows": (
                staged_rows
            ),
            "output_modified": (
                output_modified
            ),
            **metrics,
        })

        missing_origins = metrics[
            "missing_origin_details"
        ]

        if missing_origins:
            report.warning(
                "One or more teams do not have "
                "home-stadium origin mappings; "
                "team-specific travel fields were left blank",
                count=len(
                    missing_origins
                ),
                details=(
                    missing_origins
                ),
            )

        if metrics[
            "venue_country_missing_count"
        ]:
            report.warning(
                "One or more resolved venues are "
                "missing venue_country; "
                "international_flag was left blank",
                count=metrics[
                    "venue_country_missing_count"
                ],
            )

        print(
            "build_travel.py "
            f"version={SCRIPT_VERSION} "
            f"target={season}/{season_type}/{week} "
            f"rows={staged_rows} "
            "venues_resolved="
            f"{staged_rows} "
            "missing_origins="
            f"{metrics['missing_origin_count']} "
            f"output_modified={output_modified}"
        )


if __name__ == "__main__":
    main()

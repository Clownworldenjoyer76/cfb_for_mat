#!/usr/bin/env python3
"""
fetch_weather.py

Fetches CFB kickoff weather from api.met.no.

Reads:
    docs/win/football/cfb/00_intake/schedule/weekly/
        week_{week}_CFB_weekly_schedule.csv
    docs/win/football/cfb/data/travel/
        {season}_week_{week}_travel.csv
    docs/win/football/cfb/config/mapping/stadium_map.csv

Writes:
    docs/win/football/cfb/data/weather/
        week_{week}_CFB_weekly_weather.csv

Important:
    build_travel.py is the single source of truth for the actual resolved venue.
    This script uses travel's venue latitude/longitude/timezone/status by game_id.
    stadium_map.csv is used only to recover roof metadata from those coordinates.
"""

import csv
import glob
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

METNO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/complete"
METNO_USER_AGENT = os.environ.get(
    "METNO_USER_AGENT",
    "MatsPicksWeather/1.0 local-dev",
)

REQUEST_TIMEOUT = 20
REQUEST_SLEEP_SECONDS = 1.25

BASE_DIR = "docs/win/football/cfb"

SCHEDULE_DIR = os.path.join(
    BASE_DIR,
    "00_intake/schedule/weekly",
)

TRAVEL_DIR = os.path.join(
    BASE_DIR,
    "data/travel",
)

STADIUM_MAP_PATH = os.path.join(
    BASE_DIR,
    "config/mapping/stadium_map.csv",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data/weather",
)

ERROR_LOG_DIR = os.path.join(
    BASE_DIR,
    "errors/00_intake",
)

ERROR_LOG_PATH = os.path.join(
    ERROR_LOG_DIR,
    "fetch_weather.txt",
)

OUTPUT_HEADERS = [
    "game_id",
    "stadium",
    "venue_resolution_status",
    "latitude",
    "longitude",
    "game_time",
    "game_timezone",
    "kickoff_utc",
    "weather_timestep_utc",
    "temperature",
    "wind_speed",
    "wind_gust",
    "precip_probability",
    "rain_flag",
    "snow_flag",
    "humidity",
    "roof",
    "roof_type",
    "dome_flag",
    "retractable_roof_flag",
    "open_air_flag",
    "weather_fetched_at",
]

WEATHER_COLUMNS = [
    "weather_timestep_utc",
    "temperature",
    "wind_speed",
    "wind_gust",
    "precip_probability",
    "rain_flag",
    "snow_flag",
    "humidity",
]

TRAVEL_REQUIRED_COLUMNS = [
    "game_id",
    "stadium",
    "venue_resolution_status",
    "venue_lat",
    "venue_lon",
    "venue_timezone",
]

STADIUM_ROOF_COLUMNS = [
    "roof_type",
    "dome_flag",
    "retractable_roof_flag",
    "open_air_flag",
]


def clean(value):
    return "" if value is None else str(value).strip()


def normalize_game_id(value):
    return re.sub(
        r"\.0$",
        "",
        clean(value),
    )


def load_csv(path):
    with open(
        path,
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def require_columns(
    rows,
    required,
    label,
    path,
):
    if not rows:
        raise RuntimeError(
            f"{label} contains no data rows: {path}"
        )

    available = set(
        rows[0].keys()
    )

    missing = [
        column
        for column in required
        if column not in available
    ]

    if missing:
        raise RuntimeError(
            f"{label} missing required columns "
            f"{missing}: {path}"
        )


def coordinate_key(
    lat,
    lon,
):
    try:
        return (
            round(
                float(lat),
                6,
            ),
            round(
                float(lon),
                6,
            ),
        )

    except Exception:
        return None


def load_stadium_coordinate_lookup():
    rows = load_csv(
        STADIUM_MAP_PATH
    )

    require_columns(
        rows,
        [
            "latitude",
            "longitude",
            *STADIUM_ROOF_COLUMNS,
        ],
        "stadium map",
        STADIUM_MAP_PATH,
    )

    lookup = {}

    for row in rows:
        key = coordinate_key(
            row.get(
                "latitude",
                "",
            ),
            row.get(
                "longitude",
                "",
            ),
        )

        if key is None:
            continue

        existing = lookup.get(
            key
        )

        if existing is None:
            lookup[
                key
            ] = row
            continue

        existing_score = sum(
            bool(
                clean(
                    existing.get(
                        column,
                        "",
                    )
                )
            )
            for column
            in STADIUM_ROOF_COLUMNS
        )

        row_score = sum(
            bool(
                clean(
                    row.get(
                        column,
                        "",
                    )
                )
            )
            for column
            in STADIUM_ROOF_COLUMNS
        )

        if row_score > existing_score:
            lookup[
                key
            ] = row

    return lookup


def get_schedule_season(
    schedule_rows,
    schedule_path,
):
    seasons = {
        clean(
            row.get(
                "season",
                "",
            )
        )
        for row
        in schedule_rows
        if clean(
            row.get(
                "season",
                "",
            )
        )
    }

    if len(
        seasons
    ) != 1:
        raise RuntimeError(
            "Weekly schedule must contain "
            "exactly one season: "
            f"path={schedule_path} "
            f"seasons={sorted(seasons)}"
        )

    return next(
        iter(
            seasons
        )
    )


def load_travel_lookup(
    season,
    week,
    schedule_rows,
):
    travel_path = os.path.join(
        TRAVEL_DIR,
        f"{season}_week_{week}_travel.csv",
    )

    if not os.path.exists(
        travel_path
    ):
        raise FileNotFoundError(
            "Missing travel file required "
            "by fetch_weather.py: "
            f"{travel_path}. "
            "Run build_travel.py first."
        )

    travel_rows = load_csv(
        travel_path
    )

    require_columns(
        travel_rows,
        TRAVEL_REQUIRED_COLUMNS,
        "weekly travel",
        travel_path,
    )

    lookup = {}

    for row in travel_rows:
        game_id = normalize_game_id(
            row.get(
                "game_id",
                "",
            )
        )

        if not game_id:
            raise RuntimeError(
                "weekly travel contains "
                f"blank game_id: {travel_path}"
            )

        if game_id in lookup:
            raise RuntimeError(
                "weekly travel contains "
                f"duplicate game_id={game_id}: "
                f"{travel_path}"
            )

        lookup[
            game_id
        ] = row

    missing = []

    for game in schedule_rows:
        game_id = normalize_game_id(
            game.get(
                "game_id",
                "",
            )
        )

        if (
            game_id
            and game_id not in lookup
        ):
            missing.append(
                game_id
            )

    if missing:
        raise RuntimeError(
            "weekly travel is missing "
            "scheduled game_ids; "
            f"count={len(missing)} "
            f"examples={missing[:10]} "
            f"path={travel_path}"
        )

    return (
        lookup,
        travel_path,
    )


def venue_from_travel(
    game,
    travel_lookup,
    stadium_coordinate_lookup,
    log_lines,
):
    game_id = normalize_game_id(
        game.get(
            "game_id",
            "",
        )
    )

    travel_row = travel_lookup.get(
        game_id
    )

    if travel_row is None:
        log_lines.append(
            f"ERROR: game_id={game_id} "
            "missing from weekly travel output"
        )

        return (
            None,
            "missing_travel_game",
        )

    venue_status = (
        clean(
            travel_row.get(
                "venue_resolution_status",
                "",
            )
        )
        or "missing_travel_venue_status"
    )

    lat = clean(
        travel_row.get(
            "venue_lat",
            "",
        )
    )

    lon = clean(
        travel_row.get(
            "venue_lon",
            "",
        )
    )

    venue_timezone = clean(
        travel_row.get(
            "venue_timezone",
            "",
        )
    )

    if (
        not lat
        or not lon
    ):
        log_lines.append(
            f"ERROR: game_id={game_id} "
            "travel output has no resolved "
            "venue coordinates "
            f"status='{venue_status}'"
        )

        return (
            None,
            venue_status,
        )

    venue_row = {
        "latitude":
            lat,
        "longitude":
            lon,
        "timezone":
            venue_timezone,
        "roof_type":
            "",
        "dome_flag":
            "",
        "retractable_roof_flag":
            "",
        "open_air_flag":
            "",
    }

    key = coordinate_key(
        lat,
        lon,
    )

    stadium_row = (
        stadium_coordinate_lookup.get(
            key
        )
        if key is not None
        else None
    )

    if stadium_row is not None:
        for column in STADIUM_ROOF_COLUMNS:
            venue_row[
                column
            ] = clean(
                stadium_row.get(
                    column,
                    "",
                )
            )

        if not venue_row[
            "timezone"
        ]:
            venue_row[
                "timezone"
            ] = clean(
                stadium_row.get(
                    "timezone",
                    "",
                )
            )

    else:
        log_lines.append(
            f"WARNING: game_id={game_id} "
            "resolved travel coordinates "
            "were not found in stadium_map.csv "
            f"lat={lat} lon={lon}; "
            "roof metadata left blank"
        )

    return (
        venue_row,
        venue_status,
    )


def parse_iso_utc(
    value,
):
    text = clean(
        value
    )

    if not text:
        return None

    try:
        if text.endswith(
            "Z"
        ):
            text = (
                text[:-1]
                + "+00:00"
            )

        dt = datetime.fromisoformat(
            text
        )

        if dt.tzinfo is None:
            return None

        return dt.astimezone(
            timezone.utc
        )

    except Exception:
        return None


def resolve_kickoff_utc(
    game,
    venue_row,
    log_lines,
):
    game_id = normalize_game_id(
        game.get(
            "game_id",
            "",
        )
    )

    kickoff = parse_iso_utc(
        game.get(
            "commence_time",
            "",
        )
    )

    if kickoff is not None:
        return kickoff

    game_date = clean(
        game.get(
            "game_date",
            "",
        )
    )

    game_time = clean(
        game.get(
            "game_time",
            "",
        )
    )

    game_timezone = clean(
        game.get(
            "game_timezone",
            "",
        )
    )

    if (
        not game_timezone
        and venue_row is not None
    ):
        game_timezone = clean(
            venue_row.get(
                "timezone",
                "",
            )
        )

    if not (
        game_date
        and game_time
        and game_timezone
    ):
        log_lines.append(
            f"WARNING: game_id={game_id} "
            "missing kickoff "
            "date/time/timezone"
        )

        return None

    try:
        naive = datetime.strptime(
            f"{game_date} {game_time}",
            "%Y-%m-%d %H:%M",
        )

        local_dt = naive.replace(
            tzinfo=ZoneInfo(
                game_timezone
            )
        )

        return local_dt.astimezone(
            timezone.utc
        )

    except Exception as exc:
        log_lines.append(
            f"ERROR: game_id={game_id} "
            f"failed parsing kickoff: {exc}"
        )

        return None


def fetch_weather(
    lat,
    lon,
    log_lines,
    game_id,
):
    url = (
        f"{METNO_URL}"
        f"?lat={lat}"
        f"&lon={lon}"
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                METNO_USER_AGENT,
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            return json.loads(
                response.read().decode()
            )

    except urllib.error.HTTPError as exc:
        log_lines.append(
            f"ERROR: game_id={game_id} "
            "met.no request failed "
            f"HTTP={exc.code} "
            f"lat={lat} lon={lon}"
        )

    except Exception as exc:
        log_lines.append(
            f"ERROR: game_id={game_id} "
            "met.no request failed "
            f"lat={lat} lon={lon}: {exc}"
        )

    return None


def find_closest_timestep(
    weather_json,
    kickoff_utc,
):
    if not weather_json:
        return (
            None,
            None,
        )

    timeseries = (
        weather_json
        .get(
            "properties",
            {},
        )
        .get(
            "timeseries",
            [],
        )
    )

    best_entry = None
    best_time = None
    best_diff = None

    for entry in timeseries:
        entry_time = parse_iso_utc(
            entry.get(
                "time",
                "",
            )
        )

        if entry_time is None:
            continue

        difference = abs(
            (
                entry_time
                - kickoff_utc
            ).total_seconds()
        )

        if (
            best_diff is None
            or difference < best_diff
        ):
            best_entry = entry
            best_time = entry_time
            best_diff = difference

    if (
        best_entry is None
        or best_diff is None
        or best_diff
        > 12 * 3600
    ):
        return (
            None,
            None,
        )

    return (
        best_entry,
        best_time,
    )


def extract_precip_probability(
    entry,
):
    data = entry.get(
        "data",
        {},
    )

    for period_key in (
        "next_1_hours",
        "next_6_hours",
        "next_12_hours",
    ):
        probability = (
            data
            .get(
                period_key,
                {},
            )
            .get(
                "details",
                {},
            )
            .get(
                "probability_of_precipitation"
            )
        )

        if probability is not None:
            return probability

    return None


def extract_symbol_code(
    entry,
):
    data = entry.get(
        "data",
        {},
    )

    for period_key in (
        "next_1_hours",
        "next_6_hours",
        "next_12_hours",
    ):
        symbol = (
            data
            .get(
                period_key,
                {},
            )
            .get(
                "summary",
                {},
            )
            .get(
                "symbol_code"
            )
        )

        if symbol:
            return symbol

    return ""


def derive_rain_snow_flags(
    symbol_code,
):
    code = clean(
        symbol_code
    ).casefold()

    return (
        int(
            "rain" in code
            or "sleet" in code
        ),
        int(
            "snow" in code
        ),
    )


def blank_weather():
    return {
        "weather_timestep_utc":
            "",
        "temperature":
            "",
        "wind_speed":
            "",
        "wind_gust":
            "",
        "precip_probability":
            "",
        "rain_flag":
            "",
        "snow_flag":
            "",
        "humidity":
            "",
    }


def extract_weather_values(
    weather_json,
    kickoff_utc,
    game_id,
    log_lines,
):
    (
        entry,
        timestep,
    ) = find_closest_timestep(
        weather_json,
        kickoff_utc,
    )

    if entry is None:
        log_lines.append(
            f"INFO: game_id={game_id} "
            "no kickoff weather available "
            "(outside forecast range "
            "or fetch failed)"
        )

        return blank_weather()

    instant = (
        entry
        .get(
            "data",
            {},
        )
        .get(
            "instant",
            {},
        )
        .get(
            "details",
            {},
        )
    )

    precip_probability = (
        extract_precip_probability(
            entry
        )
    )

    (
        rain_flag,
        snow_flag,
    ) = derive_rain_snow_flags(
        extract_symbol_code(
            entry
        )
    )

    return {
        "weather_timestep_utc": (
            timestep.isoformat()
            if timestep
            else ""
        ),
        "temperature":
            instant.get(
                "air_temperature",
                "",
            ),
        "wind_speed":
            instant.get(
                "wind_speed",
                "",
            ),
        "wind_gust":
            instant.get(
                "wind_speed_of_gust",
                "",
            ),
        "precip_probability": (
            precip_probability
            if precip_probability
            is not None
            else ""
        ),
        "rain_flag":
            rain_flag,
        "snow_flag":
            snow_flag,
        "humidity":
            instant.get(
                "relative_humidity",
                "",
            ),
    }


def build_base_row(
    game,
    venue_row,
    venue_status,
    kickoff_utc,
    weather_fetched_at,
):
    game_timezone = clean(
        game.get(
            "game_timezone",
            "",
        )
    )

    if (
        venue_row is not None
        and not game_timezone
    ):
        game_timezone = clean(
            venue_row.get(
                "timezone",
                "",
            )
        )

    return {
        "game_id":
            normalize_game_id(
                game.get(
                    "game_id",
                    "",
                )
            ),
        "stadium":
            clean(
                game.get(
                    "stadium",
                    "",
                )
            ),
        "venue_resolution_status":
            venue_status,
        "latitude": (
            clean(
                venue_row.get(
                    "latitude",
                    "",
                )
            )
            if venue_row
            else ""
        ),
        "longitude": (
            clean(
                venue_row.get(
                    "longitude",
                    "",
                )
            )
            if venue_row
            else ""
        ),
        "game_time":
            clean(
                game.get(
                    "game_time",
                    "",
                )
            ),
        "game_timezone":
            game_timezone,
        "kickoff_utc": (
            kickoff_utc.isoformat()
            if kickoff_utc
            else ""
        ),
        **blank_weather(),
        "roof":
            clean(
                game.get(
                    "roof",
                    "",
                )
            ),
        "roof_type": (
            clean(
                venue_row.get(
                    "roof_type",
                    "",
                )
            )
            if venue_row
            else ""
        ),
        "dome_flag": (
            clean(
                venue_row.get(
                    "dome_flag",
                    "",
                )
            )
            if venue_row
            else ""
        ),
        "retractable_roof_flag": (
            clean(
                venue_row.get(
                    "retractable_roof_flag",
                    "",
                )
            )
            if venue_row
            else ""
        ),
        "open_air_flag": (
            clean(
                venue_row.get(
                    "open_air_flag",
                    "",
                )
            )
            if venue_row
            else ""
        ),
        "weather_fetched_at":
            weather_fetched_at,
    }


def process_week(
    week,
    schedule_path,
    stadium_coordinate_lookup,
    weather_fetched_at,
    log_lines,
):
    output_path = os.path.join(
        OUTPUT_DIR,
        f"week_{week}_CFB_weekly_weather.csv",
    )

    schedule_rows = load_csv(
        schedule_path
    )

    if not schedule_rows:
        log_lines.append(
            "WARNING: skipped empty "
            f"weekly schedule: {schedule_path}"
        )
        return

    season = get_schedule_season(
        schedule_rows,
        schedule_path,
    )

    (
        travel_lookup,
        travel_path,
    ) = load_travel_lookup(
        season,
        week,
        schedule_rows,
    )

    existing_rows = {}

    if os.path.exists(
        output_path
    ):
        for row in load_csv(
            output_path
        ):
            game_id = normalize_game_id(
                row.get(
                    "game_id",
                    "",
                )
            )

            if game_id:
                existing_rows[
                    game_id
                ] = row

    output_rows = []
    weather_cache = {}

    for game in schedule_rows:
        game_id = normalize_game_id(
            game.get(
                "game_id",
                "",
            )
        )

        (
            venue_row,
            venue_status,
        ) = venue_from_travel(
            game,
            travel_lookup,
            stadium_coordinate_lookup,
            log_lines,
        )

        kickoff_utc = resolve_kickoff_utc(
            game,
            venue_row,
            log_lines,
        )

        row = build_base_row(
            game,
            venue_row,
            venue_status,
            kickoff_utc,
            weather_fetched_at,
        )

        if kickoff_utc is None:
            output_rows.append(
                row
            )
            continue

        future = (
            kickoff_utc
            > datetime.now(
                timezone.utc
            )
        )

        if (
            not future
            and game_id
            in existing_rows
        ):
            old_row = existing_rows[
                game_id
            ]

            for column in WEATHER_COLUMNS:
                row[
                    column
                ] = clean(
                    old_row.get(
                        column,
                        "",
                    )
                )

            row[
                "weather_fetched_at"
            ] = clean(
                old_row.get(
                    "weather_fetched_at",
                    "",
                )
            )

            output_rows.append(
                row
            )
            continue

        if not future:
            log_lines.append(
                f"INFO: game_id={game_id} "
                "game already completed "
                "and no stored weather exists"
            )

            output_rows.append(
                row
            )
            continue

        if venue_row is None:
            output_rows.append(
                row
            )
            continue

        lat = clean(
            venue_row.get(
                "latitude",
                "",
            )
        )

        lon = clean(
            venue_row.get(
                "longitude",
                "",
            )
        )

        if not (
            lat
            and lon
        ):
            log_lines.append(
                f"ERROR: game_id={game_id} "
                "resolved travel venue has "
                "no latitude/longitude"
            )

            output_rows.append(
                row
            )
            continue

        cache_key = (
            lat,
            lon,
        )

        if cache_key not in weather_cache:
            weather_cache[
                cache_key
            ] = fetch_weather(
                lat,
                lon,
                log_lines,
                game_id,
            )

            time.sleep(
                REQUEST_SLEEP_SECONDS
            )

        row.update(
            extract_weather_values(
                weather_cache[
                    cache_key
                ],
                kickoff_utc,
                game_id,
                log_lines,
            )
        )

        output_rows.append(
            row
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_HEADERS,
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    resolved = sum(
        1
        for row in output_rows
        if clean(
            row.get(
                "venue_resolution_status",
                "",
            )
        ).startswith(
            "resolved_"
        )
    )

    with_weather = sum(
        1
        for row in output_rows
        if clean(
            row.get(
                "wind_speed",
                "",
            )
        )
    )

    print(
        f"Wrote {len(output_rows)} rows "
        f"to {output_path} | "
        f"travel_source={travel_path} | "
        f"venues_resolved={resolved} | "
        f"weather_available={with_weather}"
    )


def main():
    log_lines = []

    weather_fetched_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    stadium_coordinate_lookup = (
        load_stadium_coordinate_lookup()
    )

    schedule_files = sorted(
        glob.glob(
            os.path.join(
                SCHEDULE_DIR,
                "week_*_CFB_weekly_schedule.csv",
            )
        )
    )

    if not schedule_files:
        print(
            "WARNING: no weekly schedule "
            f"files found in {SCHEDULE_DIR}"
        )
        return

    for schedule_path in schedule_files:
        filename = os.path.basename(
            schedule_path
        )

        match = re.fullmatch(
            r"week_(\d+)_CFB_weekly_schedule\.csv",
            filename,
        )

        if not match:
            log_lines.append(
                "WARNING: skipped "
                f"unrecognized file: {filename}"
            )
            continue

        process_week(
            int(
                match.group(
                    1
                )
            ),
            schedule_path,
            stadium_coordinate_lookup,
            weather_fetched_at,
            log_lines,
        )

    os.makedirs(
        ERROR_LOG_DIR,
        exist_ok=True,
    )

    with open(
        ERROR_LOG_PATH,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"\n--- Run at "
            f"{weather_fetched_at} ---\n"
        )

        if log_lines:
            for line in log_lines:
                f.write(
                    line
                    + "\n"
                )

        else:
            f.write(
                "No issues.\n"
            )

    print(
        f"Log written to "
        f"{ERROR_LOG_PATH}"
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
fetch_weather.py

Fetches CFB kickoff weather from api.met.no.

Reads:
    docs/win/football/cfb/00_intake/schedule/weekly/
        week_{week}_CFB_weekly_schedule.csv

    docs/win/football/cfb/config/mapping/stadium_map.csv

Writes:
    docs/win/football/cfb/data/weather/
        week_{week}_CFB_weekly_weather.csv

Behavior:
    - Resolves the ACTUAL scheduled stadium by stadium name.
    - Uses the actual venue latitude/longitude.
    - Uses the scheduled kickoff timestamp.
    - Handles neutral-site games without assuming the listed home team's stadium.
    - Preserves temperature, wind, gusts, precipitation, rain/snow, humidity,
      and roof information.
    - Existing weather for completed games is preserved.
"""

import csv
import glob
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


METNO_URL = (
    "https://api.met.no/weatherapi/locationforecast/2.0/complete"
)

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


def clean(value):
    if value is None:
        return ""

    return str(value).strip()


def normalize_key(value):
    text = clean(value)

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        ch
        for ch in text
        if not unicodedata.combining(ch)
    )

    text = text.casefold()

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text,
    )


def strip_parenthetical(value):
    return re.sub(
        r"\s*\([^)]*\)\s*$",
        "",
        clean(value),
    ).strip()


def load_csv(path):
    with open(
        path,
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def same_coordinates(row_a, row_b):
    try:
        return (
            abs(
                float(row_a["latitude"])
                - float(row_b["latitude"])
            )
            < 1e-7
            and
            abs(
                float(row_a["longitude"])
                - float(row_b["longitude"])
            )
            < 1e-7
        )

    except Exception:
        return False


def dedupe_venue_rows(rows):
    unique = []

    for row in rows:
        if not any(
            same_coordinates(
                row,
                existing,
            )
            for existing in unique
        ):
            unique.append(row)

    return unique


def load_stadium_map():
    rows = load_csv(
        STADIUM_MAP_PATH
    )

    venue_lookup = {}

    for row in rows:
        names = {
            clean(
                row.get(
                    "stadium",
                    "",
                )
            ),
            clean(
                row.get(
                    "venue_full_name",
                    "",
                )
            ),
            strip_parenthetical(
                row.get(
                    "stadium",
                    "",
                )
            ),
            strip_parenthetical(
                row.get(
                    "venue_full_name",
                    "",
                )
            ),
        }

        for name in names:
            key = normalize_key(
                name
            )

            if not key:
                continue

            venue_lookup.setdefault(
                key,
                [],
            ).append(
                row
            )

    return venue_lookup


def resolve_venue(
    game,
    venue_lookup,
    log_lines,
):
    game_id = clean(
        game.get(
            "game_id",
            "",
        )
    )

    stadium = clean(
        game.get(
            "stadium",
            "",
        )
    )

    game_timezone = clean(
        game.get(
            "game_timezone",
            "",
        )
    )

    if not stadium:
        log_lines.append(
            f"ERROR: game_id={game_id} "
            "schedule has no stadium"
        )

        return (
            None,
            "missing_schedule_stadium",
        )

    candidates = []

    for name in {
        stadium,
        strip_parenthetical(
            stadium
        ),
    }:
        key = normalize_key(
            name
        )

        candidates.extend(
            venue_lookup.get(
                key,
                [],
            )
        )

    candidates = dedupe_venue_rows(
        candidates
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
                row.get(
                    "timezone",
                    "",
                )
            )
            == game_timezone
        ]

        timezone_matches = (
            dedupe_venue_rows(
                timezone_matches
            )
        )

        if len(
            timezone_matches
        ) == 1:
            return (
                timezone_matches[0],
                "resolved_schedule_stadium_timezone",
            )

    if len(candidates) > 1:
        log_lines.append(
            f"ERROR: game_id={game_id} "
            f"ambiguous stadium='{stadium}' "
            f"matches={len(candidates)}"
        )

        return (
            None,
            "ambiguous_schedule_stadium",
        )

    log_lines.append(
        f"ERROR: game_id={game_id} "
        f"actual venue not found in stadium_map.csv: "
        f"stadium='{stadium}'"
    )

    return (
        None,
        "unresolved_schedule_stadium",
    )


def parse_iso_utc(value):
    text = clean(value)

    if not text:
        return None

    try:
        if text.endswith("Z"):
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
    log_lines,
):
    game_id = clean(
        game.get(
            "game_id",
            "",
        )
    )

    commence_time = clean(
        game.get(
            "commence_time",
            "",
        )
    )

    kickoff = parse_iso_utc(
        commence_time
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

    if not (
        game_date
        and game_time
        and game_timezone
    ):
        log_lines.append(
            f"WARNING: game_id={game_id} "
            "missing kickoff date/time/timezone"
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
            "User-Agent": (
                METNO_USER_AGENT
            ),
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


def parse_weather_time(value):
    return parse_iso_utc(
        value
    )


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
        entry_time = (
            parse_weather_time(
                entry.get(
                    "time",
                    "",
                )
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
    ):
        return (
            None,
            None,
        )

    # Reject forecasts that are nowhere near kickoff.
    if best_diff > 12 * 3600:
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

    rain_flag = int(
        "rain" in code
        or "sleet" in code
    )

    snow_flag = int(
        "snow" in code
    )

    return (
        rain_flag,
        snow_flag,
    )


def blank_weather():
    return {
        "weather_timestep_utc": "",
        "temperature": "",
        "wind_speed": "",
        "wind_gust": "",
        "precip_probability": "",
        "rain_flag": "",
        "snow_flag": "",
        "humidity": "",
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
            "(outside forecast range or fetch failed)"
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

    symbol_code = (
        extract_symbol_code(
            entry
        )
    )

    (
        rain_flag,
        snow_flag,
    ) = derive_rain_snow_flags(
        symbol_code
    )

    return {
        "weather_timestep_utc": (
            timestep.isoformat()
            if timestep is not None
            else ""
        ),
        "temperature": instant.get(
            "air_temperature",
            "",
        ),
        "wind_speed": instant.get(
            "wind_speed",
            "",
        ),
        "wind_gust": instant.get(
            "wind_speed_of_gust",
            "",
        ),
        "precip_probability": (
            precip_probability
            if precip_probability
            is not None
            else ""
        ),
        "rain_flag": rain_flag,
        "snow_flag": snow_flag,
        "humidity": instant.get(
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
    stadium = clean(
        game.get(
            "stadium",
            "",
        )
    )

    roof = clean(
        game.get(
            "roof",
            "",
        )
    )

    game_timezone = clean(
        game.get(
            "game_timezone",
            "",
        )
    )

    if venue_row is None:
        return {
            "game_id": clean(
                game.get(
                    "game_id",
                    "",
                )
            ),
            "stadium": stadium,
            "venue_resolution_status": (
                venue_status
            ),
            "latitude": "",
            "longitude": "",
            "game_time": clean(
                game.get(
                    "game_time",
                    "",
                )
            ),
            "game_timezone": (
                game_timezone
            ),
            "kickoff_utc": (
                kickoff_utc.isoformat()
                if kickoff_utc
                is not None
                else ""
            ),
            **blank_weather(),
            "roof": roof,
            "roof_type": "",
            "dome_flag": "",
            "retractable_roof_flag": "",
            "open_air_flag": "",
            "weather_fetched_at": (
                weather_fetched_at
            ),
        }

    return {
        "game_id": clean(
            game.get(
                "game_id",
                "",
            )
        ),
        "stadium": stadium,
        "venue_resolution_status": (
            venue_status
        ),
        "latitude": clean(
            venue_row.get(
                "latitude",
                "",
            )
        ),
        "longitude": clean(
            venue_row.get(
                "longitude",
                "",
            )
        ),
        "game_time": clean(
            game.get(
                "game_time",
                "",
            )
        ),
        "game_timezone": (
            game_timezone
            or clean(
                venue_row.get(
                    "timezone",
                    "",
                )
            )
        ),
        "kickoff_utc": (
            kickoff_utc.isoformat()
            if kickoff_utc
            is not None
            else ""
        ),
        **blank_weather(),
        "roof": roof,
        "roof_type": clean(
            venue_row.get(
                "roof_type",
                "",
            )
        ),
        "dome_flag": clean(
            venue_row.get(
                "dome_flag",
                "",
            )
        ),
        "retractable_roof_flag": clean(
            venue_row.get(
                "retractable_roof_flag",
                "",
            )
        ),
        "open_air_flag": clean(
            venue_row.get(
                "open_air_flag",
                "",
            )
        ),
        "weather_fetched_at": (
            weather_fetched_at
        ),
    }


def process_week(
    week,
    schedule_path,
    venue_lookup,
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

    existing_rows = {}

    if os.path.exists(
        output_path
    ):
        for row in load_csv(
            output_path
        ):
            existing_rows[
                clean(
                    row.get(
                        "game_id",
                        "",
                    )
                )
            ] = row

    output_rows = []

    # Avoid repeated met.no calls when multiple games use the same venue.
    weather_cache = {}

    for game in schedule_rows:
        game_id = clean(
            game.get(
                "game_id",
                "",
            )
        )

        (
            venue_row,
            venue_status,
        ) = resolve_venue(
            game,
            venue_lookup,
            log_lines,
        )

        kickoff_utc = (
            resolve_kickoff_utc(
                game,
                log_lines,
            )
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

            # Preserve historical weather, but refresh venue/roof metadata.
            for column in WEATHER_COLUMNS:
                row[column] = clean(
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
                "game already completed and no "
                "stored weather exists"
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

        if not lat or not lon:
            log_lines.append(
                f"ERROR: game_id={game_id} "
                "resolved venue has no "
                "latitude/longitude"
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

        weather_values = (
            extract_weather_values(
                weather_cache[
                    cache_key
                ],
                kickoff_utc,
                game_id,
                log_lines,
            )
        )

        row.update(
            weather_values
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
        if row[
            "venue_resolution_status"
        ].startswith(
            "resolved_"
        )
    )

    print(
        f"Wrote {len(output_rows)} rows "
        f"to {output_path} | "
        f"venues_resolved={resolved}"
    )


def main():
    log_lines = []

    weather_fetched_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    venue_lookup = (
        load_stadium_map()
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

        week = int(
            match.group(1)
        )

        process_week(
            week,
            schedule_path,
            venue_lookup,
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
            "\n--- Run at "
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
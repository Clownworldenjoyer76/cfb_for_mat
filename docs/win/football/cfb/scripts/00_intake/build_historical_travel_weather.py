#!/usr/bin/env python3
"""
build_historical_travel_weather.py

One-off historical CFB travel/weather feature builder for 2021-2025.

Reads:
    docs/win/football/cfb/00_intake/schedule/{season}_schedule.csv
    docs/win/football/cfb/config/mapping/stadium_map.csv

Writes:
    docs/win/football/cfb/data/historical_features/
        {season}_travel_weather.csv

Historical weather source:
    Open-Meteo Historical Weather API (ERA5)
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


SEASONS = range(2021, 2026)

BASE_DIR = Path("docs/win/football/cfb")
SCHEDULE_DIR = BASE_DIR / "00_intake" / "schedule"
STADIUM_MAP_PATH = BASE_DIR / "config" / "mapping" / "stadium_map.csv"
OUTPUT_DIR = BASE_DIR / "data" / "historical_features"
ERROR_DIR = BASE_DIR / "errors" / "00_intake"
ERROR_LOG_PATH = ERROR_DIR / "build_historical_travel_weather.txt"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

OPEN_METEO_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
]

REQUEST_TIMEOUT = 45
REQUEST_SLEEP_SECONDS = 0.25
MAX_REQUEST_ATTEMPTS = 4


OUTPUT_HEADERS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "kickoff_utc",
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
    "weather_timestep_utc",
    "temperature",
    "wind_speed",
    "wind_gust",
    "precip_probability",
    "precipitation",
    "rain",
    "snowfall",
    "rain_flag",
    "snow_flag",
    "humidity",
    "weather_code",
    "roof",
    "roof_type",
    "dome_flag",
    "retractable_roof_flag",
    "open_air_flag",
    "weather_source",
    "weather_status",
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


def parse_flag(value):
    return int(
        clean(value).casefold()
        in {
            "1",
            "true",
            "yes",
            "y",
        }
    )


def parse_float(value):
    text = clean(value)

    if not text:
        return None

    try:
        number = float(text)

    except (TypeError, ValueError):
        return None

    return (
        number
        if math.isfinite(number)
        else None
    )


def load_csv(path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def write_csv(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_HEADERS,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(
        temp_path,
        path,
    )


def same_coordinates(
    row_a,
    row_b,
):
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


def load_stadium_maps():
    rows = load_csv(
        STADIUM_MAP_PATH
    )

    team_lookup = {}
    venue_lookup = defaultdict(list)

    for row in rows:

        team_key = normalize_key(
            row.get(
                "team",
                "",
            )
        )

        if team_key:
            team_lookup[
                team_key
            ] = row

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

            key = normalize_key(name)

            if key:
                venue_lookup[
                    key
                ].append(row)

    return (
        team_lookup,
        venue_lookup,
    )


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

    neutral_site_flag = parse_flag(
        game.get(
            "neutral_site",
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

        key = normalize_key(name)

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

        timezone_matches = (
            dedupe_venue_rows(
                [
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
            "ambiguous venue "
            f"stadium='{stadium}' "
            f"matches={len(candidates)}"
        )

        return (
            None,
            "ambiguous_schedule_stadium",
        )

    status = (
        "unresolved_neutral_venue"
        if neutral_site_flag == 1
        else "unresolved_schedule_stadium"
    )

    log_lines.append(
        f"ERROR: game_id={game_id} "
        "actual venue not found in stadium_map.csv: "
        f"stadium='{stadium}' "
        f"neutral_site={neutral_site_flag}"
    )

    return (
        None,
        status,
    )


def haversine_miles(
    lat1,
    lon1,
    lat2,
    lon2,
):
    radius_miles = 3958.8

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

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
        +
        math.cos(phi1)
        * math.cos(phi2)
        * math.sin(
            dlambda / 2
        )
        ** 2
    )

    return (
        radius_miles
        * (
            2
            * math.asin(
                math.sqrt(a)
            )
        )
    )


def utc_offset_hours(
    tz_name,
    game_date,
):
    tz_name = clean(tz_name)
    game_date = clean(game_date)

    if (
        not tz_name
        or not game_date
    ):
        return None

    try:
        dt = datetime.strptime(
            f"{game_date} 12:00",
            "%Y-%m-%d %H:%M",
        ).replace(
            tzinfo=ZoneInfo(
                tz_name
            )
        )

        offset = dt.utcoffset()

        if offset is None:
            return None

        return (
            offset.total_seconds()
            / 3600
        )

    except Exception:
        return None


def travel_direction(
    origin_lon,
    destination_lon,
):
    origin = parse_float(
        origin_lon
    )

    destination = parse_float(
        destination_lon
    )

    if (
        origin is None
        or destination is None
    ):
        return (
            "",
            "",
        )

    if destination < origin:
        return (
            1,
            0,
        )

    if destination > origin:
        return (
            0,
            1,
        )

    return (
        0,
        0,
    )


def team_travel_values(
    team_row,
    venue_row,
    venue_timezone,
    game_date,
    game_id,
    label,
    log_lines,
):
    values = {
        "home_lat": "",
        "home_lon": "",
        "home_timezone": "",
        "miles_traveled": "",
        "time_zone_change_hours": "",
        "time_zones_crossed": "",
        "east_to_west": "",
        "west_to_east": "",
    }

    if team_row is None:
        return values

    team_lat = clean(
        team_row.get(
            "latitude",
            "",
        )
    )

    team_lon = clean(
        team_row.get(
            "longitude",
            "",
        )
    )

    team_timezone = clean(
        team_row.get(
            "timezone",
            "",
        )
    )

    values.update(
        {
            "home_lat": team_lat,
            "home_lon": team_lon,
            "home_timezone": team_timezone,
        }
    )

    if venue_row is None:
        return values

    venue_lat = clean(
        venue_row.get(
            "latitude",
            "",
        )
    )

    venue_lon = clean(
        venue_row.get(
            "longitude",
            "",
        )
    )

    try:
        values[
            "miles_traveled"
        ] = round(
            haversine_miles(
                float(team_lat),
                float(team_lon),
                float(venue_lat),
                float(venue_lon),
            ),
            1,
        )

    except Exception as exc:

        log_lines.append(
            f"ERROR: game_id={game_id} "
            f"failed computing "
            f"{label}_miles_traveled: "
            f"{exc}"
        )

    team_offset = utc_offset_hours(
        team_timezone,
        game_date,
    )

    venue_offset = utc_offset_hours(
        venue_timezone,
        game_date,
    )

    if (
        team_offset is not None
        and venue_offset is not None
    ):

        signed_change = (
            venue_offset
            - team_offset
        )

        values[
            "time_zone_change_hours"
        ] = round(
            signed_change,
            1,
        )

        values[
            "time_zones_crossed"
        ] = round(
            abs(
                signed_change
            ),
            1,
        )

    (
        east_to_west,
        west_to_east,
    ) = travel_direction(
        team_lon,
        venue_lon,
    )

    values[
        "east_to_west"
    ] = east_to_west

    values[
        "west_to_east"
    ] = west_to_east

    return values


def resolve_kickoff_utc(
    game,
    venue_timezone,
    log_lines,
):
    game_id = clean(
        game.get(
            "game_id",
            "",
        )
    )

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

    game_timezone = (
        clean(
            game.get(
                "game_timezone",
                "",
            )
        )
        or clean(
            venue_timezone
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
        local_dt = datetime.strptime(
            f"{game_date} {game_time}",
            "%Y-%m-%d %H:%M",
        ).replace(
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


def build_base_row(
    game,
    team_lookup,
    venue_lookup,
    log_lines,
):
    game_id = clean(
        game.get(
            "game_id",
            "",
        )
    )

    away_team = clean(
        game.get(
            "away_team",
            "",
        )
    )

    home_team = clean(
        game.get(
            "home_team",
            "",
        )
    )

    game_date = clean(
        game.get(
            "game_date",
            "",
        )
    )

    away_row = team_lookup.get(
        normalize_key(
            away_team
        )
    )

    home_row = team_lookup.get(
        normalize_key(
            home_team
        )
    )

    if away_row is None:

        log_lines.append(
            f"ERROR: game_id={game_id} "
            "no stadium_map team match "
            f"for away_team='{away_team}'"
        )

    if home_row is None:

        log_lines.append(
            f"ERROR: game_id={game_id} "
            "no stadium_map team match "
            f"for home_team='{home_team}'"
        )

    (
        venue_row,
        venue_status,
    ) = resolve_venue(
        game,
        venue_lookup,
        log_lines,
    )

    venue_timezone = clean(
        game.get(
            "game_timezone",
            "",
        )
    )

    if (
        not venue_timezone
        and venue_row is not None
    ):
        venue_timezone = clean(
            venue_row.get(
                "timezone",
                "",
            )
        )

    kickoff_utc = (
        resolve_kickoff_utc(
            game,
            venue_timezone,
            log_lines,
        )
    )

    away_values = (
        team_travel_values(
            away_row,
            venue_row,
            venue_timezone,
            game_date,
            game_id,
            "away",
            log_lines,
        )
    )

    home_values = (
        team_travel_values(
            home_row,
            venue_row,
            venue_timezone,
            game_date,
            game_id,
            "home",
            log_lines,
        )
    )

    venue_country = (
        clean(
            venue_row.get(
                "venue_country",
                "",
            )
        )
        if venue_row is not None
        else ""
    )

    international_flag = (
        ""
        if not venue_country
        else int(
            venue_country.upper()
            != "USA"
        )
    )

    return {
        "season": clean(
            game.get(
                "season",
                "",
            )
        ),
        "season_type": clean(
            game.get(
                "season_type",
                "",
            )
        ),
        "week": clean(
            game.get(
                "week",
                "",
            )
        ),
        "game_id": game_id,
        "game_date": game_date,
        "game_time": clean(
            game.get(
                "game_time",
                "",
            )
        ),
        "kickoff_utc": (
            kickoff_utc.isoformat()
            if kickoff_utc
            else ""
        ),
        "away_team": away_team,
        "home_team": home_team,
        "stadium": clean(
            game.get(
                "stadium",
                "",
            )
        ),
        "neutral_site_flag": (
            parse_flag(
                game.get(
                    "neutral_site",
                    "",
                )
            )
        ),
        "venue_resolution_status": (
            venue_status
        ),
        "venue_lat": (
            clean(
                venue_row.get(
                    "latitude",
                    "",
                )
            )
            if venue_row
            else ""
        ),
        "venue_lon": (
            clean(
                venue_row.get(
                    "longitude",
                    "",
                )
            )
            if venue_row
            else ""
        ),
        "venue_timezone": venue_timezone,
        "venue_country": venue_country,
        "away_home_lat": (
            away_values[
                "home_lat"
            ]
        ),
        "away_home_lon": (
            away_values[
                "home_lon"
            ]
        ),
        "away_home_timezone": (
            away_values[
                "home_timezone"
            ]
        ),
        "away_miles_traveled": (
            away_values[
                "miles_traveled"
            ]
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
            home_values[
                "home_lat"
            ]
        ),
        "home_home_lon": (
            home_values[
                "home_lon"
            ]
        ),
        "home_home_timezone": (
            home_values[
                "home_timezone"
            ]
        ),
        "home_miles_traveled": (
            home_values[
                "miles_traveled"
            ]
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
        "weather_timestep_utc": "",
        "temperature": "",
        "wind_speed": "",
        "wind_gust": "",
        "precip_probability": "",
        "precipitation": "",
        "rain": "",
        "snowfall": "",
        "rain_flag": "",
        "snow_flag": "",
        "humidity": "",
        "weather_code": "",
        "roof": clean(
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
        "weather_source": "",
        "weather_status": "not_requested",
    }


def fetch_json(
    url,
    log_lines,
    label,
):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "cfb-historical-weather/1.0"
            ),
            "Accept": "application/json",
        },
    )

    for attempt in range(
        1,
        MAX_REQUEST_ATTEMPTS + 1,
    ):

        try:
            with urllib.request.urlopen(
                request,
                timeout=REQUEST_TIMEOUT,
            ) as response:

                return json.loads(
                    response
                    .read()
                    .decode(
                        "utf-8"
                    )
                )

        except urllib.error.HTTPError as exc:

            retryable = (
                exc.code == 429
                or 500 <= exc.code < 600
            )

            log_lines.append(
                f"WARNING: {label} "
                f"HTTP {exc.code} "
                f"attempt={attempt}"
            )

            if (
                not retryable
                or attempt
                == MAX_REQUEST_ATTEMPTS
            ):
                return None

        except Exception as exc:

            log_lines.append(
                f"WARNING: {label} "
                "request failed "
                f"attempt={attempt}: "
                f"{exc}"
            )

            if (
                attempt
                == MAX_REQUEST_ATTEMPTS
            ):
                return None

        time.sleep(
            min(
                2 ** (
                    attempt - 1
                ),
                8,
            )
        )

    return None


def fetch_venue_weather(
    lat,
    lon,
    start_date,
    end_date,
    log_lines,
    label,
):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(
            OPEN_METEO_HOURLY
        ),
        "timezone": "GMT",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "models": "era5",
    }

    url = (
        OPEN_METEO_ARCHIVE_URL
        + "?"
        + urllib.parse.urlencode(
            params
        )
    )

    payload = fetch_json(
        url,
        log_lines,
        label,
    )

    time.sleep(
        REQUEST_SLEEP_SECONDS
    )

    return payload


def parse_archive_time(value):
    text = clean(value)

    if not text:
        return None

    try:
        return datetime.strptime(
            text,
            "%Y-%m-%dT%H:%M",
        ).replace(
            tzinfo=timezone.utc
        )

    except Exception:
        return None


def hourly_lookup(payload):
    if not isinstance(
        payload,
        dict,
    ):
        return {}

    hourly = payload.get(
        "hourly"
    )

    if not isinstance(
        hourly,
        dict,
    ):
        return {}

    times = (
        hourly.get(
            "time"
        )
        or []
    )

    lookup = {}

    for index, raw_time in enumerate(
        times
    ):
        dt = parse_archive_time(
            raw_time
        )

        if dt is None:
            continue

        record = {
            "time": dt
        }

        for variable in OPEN_METEO_HOURLY:

            values = (
                hourly.get(
                    variable
                )
                or []
            )

            record[
                variable
            ] = (
                values[index]
                if index < len(values)
                else None
            )

        lookup[
            dt
        ] = record

    return lookup


def closest_hour_record(
    lookup,
    kickoff_utc,
):
    if (
        not lookup
        or kickoff_utc is None
    ):
        return None

    hour = kickoff_utc.replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    candidates = [
        hour,
        hour
        + timedelta(
            hours=1
        ),
        hour
        - timedelta(
            hours=1
        ),
    ]

    available = [
        lookup[candidate]
        for candidate in candidates
        if candidate in lookup
    ]

    if not available:
        return None

    best = min(
        available,
        key=lambda record: abs(
            (
                record["time"]
                - kickoff_utc
            ).total_seconds()
        ),
    )

    difference = abs(
        (
            best["time"]
            - kickoff_utc
        ).total_seconds()
    )

    if difference > 3600:
        return None

    return best


def apply_weather(
    row,
    weather_record,
):
    if weather_record is None:

        row[
            "weather_status"
        ] = "weather_unavailable"

        return

    precipitation = (
        weather_record.get(
            "precipitation"
        )
    )

    rain = weather_record.get(
        "rain"
    )

    snowfall = (
        weather_record.get(
            "snowfall"
        )
    )

    rain_value = parse_float(
        rain
    )

    snowfall_value = (
        parse_float(
            snowfall
        )
    )

    row.update(
        {
            "weather_timestep_utc": (
                weather_record[
                    "time"
                ].isoformat()
            ),
            "temperature": (
                weather_record.get(
                    "temperature_2m",
                    "",
                )
            ),
            "wind_speed": (
                weather_record.get(
                    "wind_speed_10m",
                    "",
                )
            ),
            "wind_gust": (
                weather_record.get(
                    "wind_gusts_10m",
                    "",
                )
            ),
            "precip_probability": "",
            "precipitation": (
                precipitation
                if precipitation
                is not None
                else ""
            ),
            "rain": (
                rain
                if rain is not None
                else ""
            ),
            "snowfall": (
                snowfall
                if snowfall is not None
                else ""
            ),
            "rain_flag": (
                ""
                if rain_value is None
                else int(
                    rain_value > 0
                )
            ),
            "snow_flag": (
                ""
                if snowfall_value is None
                else int(
                    snowfall_value > 0
                )
            ),
            "humidity": (
                weather_record.get(
                    "relative_humidity_2m",
                    "",
                )
            ),
            "weather_code": (
                weather_record.get(
                    "weather_code",
                    "",
                )
            ),
            "weather_source": (
                "open-meteo-era5"
            ),
            "weather_status": "ok",
        }
    )


def add_historical_weather(
    rows,
    season,
    log_lines,
):
    venue_groups = defaultdict(
        list
    )

    for row in rows:

        lat = parse_float(
            row.get(
                "venue_lat"
            )
        )

        lon = parse_float(
            row.get(
                "venue_lon"
            )
        )

        kickoff_text = clean(
            row.get(
                "kickoff_utc"
            )
        )

        if (
            lat is None
            or lon is None
        ):
            row[
                "weather_status"
            ] = "venue_unresolved"

            continue

        if not kickoff_text:

            row[
                "weather_status"
            ] = "kickoff_unresolved"

            continue

        try:
            kickoff_utc = (
                datetime.fromisoformat(
                    kickoff_text
                )
            )

        except Exception:

            row[
                "weather_status"
            ] = "kickoff_unresolved"

            continue

        venue_key = (
            round(
                lat,
                5,
            ),
            round(
                lon,
                5,
            ),
        )

        venue_groups[
            venue_key
        ].append(
            (
                row,
                kickoff_utc,
            )
        )

    total_venues = len(
        venue_groups
    )

    for venue_number, (
        (
            lat,
            lon,
        ),
        games,
    ) in enumerate(
        sorted(
            venue_groups.items()
        ),
        start=1,
    ):

        kickoff_times = [
            kickoff
            for _, kickoff in games
        ]

        start_date = (
            min(
                kickoff_times
            )
            - timedelta(
                days=1
            )
        ).date().isoformat()

        end_date = (
            max(
                kickoff_times
            )
            + timedelta(
                days=1
            )
        ).date().isoformat()

        label = (
            f"season={season} "
            f"venue={venue_number}/"
            f"{total_venues} "
            f"lat={lat} "
            f"lon={lon}"
        )

        print(
            f"Weather {label}"
        )

        payload = (
            fetch_venue_weather(
                lat,
                lon,
                start_date,
                end_date,
                log_lines,
                label,
            )
        )

        lookup = hourly_lookup(
            payload
        )

        if not lookup:

            for row, _ in games:
                row[
                    "weather_status"
                ] = "api_request_failed"

            continue

        for row, kickoff_utc in games:

            record = (
                closest_hour_record(
                    lookup,
                    kickoff_utc,
                )
            )

            apply_weather(
                row,
                record,
            )


def validate_schedule(
    rows,
    season,
    path,
):
    if not rows:

        raise RuntimeError(
            f"Empty historical schedule: "
            f"{path}"
        )

    required = {
        "season",
        "season_type",
        "week",
        "game_id",
        "game_date",
        "game_time",
        "away_team",
        "home_team",
        "neutral_site",
        "stadium",
    }

    missing = (
        required
        - set(
            rows[0].keys()
        )
    )

    if missing:

        raise RuntimeError(
            f"{path} missing required "
            f"columns: "
            f"{sorted(missing)}"
        )

    wrong_seasons = sorted(
        {
            clean(
                row.get(
                    "season"
                )
            )
            for row in rows
            if clean(
                row.get(
                    "season"
                )
            )
            != str(season)
        }
    )

    if wrong_seasons:

        raise RuntimeError(
            f"{path} contains "
            "unexpected seasons: "
            f"{wrong_seasons[:10]}"
        )


def process_season(
    season,
    team_lookup,
    venue_lookup,
    log_lines,
):
    schedule_path = (
        SCHEDULE_DIR
        / f"{season}_schedule.csv"
    )

    output_path = (
        OUTPUT_DIR
        / f"{season}_travel_weather.csv"
    )

    if not schedule_path.exists():

        raise FileNotFoundError(
            "Missing historical schedule: "
            f"{schedule_path}"
        )

    schedule_rows = load_csv(
        schedule_path
    )

    validate_schedule(
        schedule_rows,
        season,
        schedule_path,
    )

    rows = [
        build_base_row(
            game,
            team_lookup,
            venue_lookup,
            log_lines,
        )
        for game in schedule_rows
    ]

    add_historical_weather(
        rows,
        season,
        log_lines,
    )

    write_csv(
        output_path,
        rows,
    )

    resolved_venues = sum(
        1
        for row in rows
        if (
            clean(
                row.get(
                    "venue_lat"
                )
            )
            and clean(
                row.get(
                    "venue_lon"
                )
            )
        )
    )

    weather_ok = sum(
        1
        for row in rows
        if row.get(
            "weather_status"
        )
        == "ok"
    )

    print(
        f"WROTE {output_path} | "
        f"games={len(rows)} | "
        f"venues_resolved="
        f"{resolved_venues} | "
        f"weather_ok={weather_ok}"
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ERROR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_lines = []

    (
        team_lookup,
        venue_lookup,
    ) = load_stadium_maps()

    for season in SEASONS:

        print(
            f"Processing {season}..."
        )

        process_season(
            season,
            team_lookup,
            venue_lookup,
            log_lines,
        )

    with ERROR_LOG_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

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

    print(
        "status=success"
    )


if __name__ == "__main__":
    main()
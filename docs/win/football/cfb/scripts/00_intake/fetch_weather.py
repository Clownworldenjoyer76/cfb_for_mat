#!/usr/bin/env python3
"""
fetch_weather.py

Fetches weather forecast data from api.met.no for NFL games in a given
week's schedule file, and writes/updates:
    docs/win/football/nfl/data/weather/week_{week}_NFL_weekly_weather.csv

Input:
    docs/win/football/nfl/00_intake/schedule/weekly/week_{week}_NFL_weekly_schedule.csv

Stadium/location data joined from:
    docs/win/football/nfl/config/mapping/stadium_map_nfl.csv
    (joined on stadium, home_team -> team, and game_date)

Weather source:
    https://api.met.no/weatherapi/locationforecast/2.0/complete

Behavior:
    - Only fetches/updates rows for games that have not yet happened
      (game date/time in the future, using game_timezone).
    - If the output file already exists, past-game rows already present
      are left untouched.
    - If weather data is not available for a game (e.g. game is more
      than 9 days out, outside met.no forecast range), weather columns
      are left blank, but all other attainable columns are still
      written.
    - Logs any issues or failed matches to:
        docs/win/football/nfl/errors/00_intake/fetch_weather.txt


"""

import csv
import glob
import json
import os
import re
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
METNO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/complete"
METNO_USER_AGENT = os.environ.get(
    "METNO_USER_AGENT",
    "MatsPicksWeather/1.0 local-dev",
)
REQUEST_TIMEOUT = 20
REQUEST_SLEEP_SECONDS = 1.25

BASE_DIR = "docs/win/football/nfl"
SCHEDULE_DIR = os.path.join(BASE_DIR, "00_intake/schedule/weekly")
STADIUM_MAP_PATH = os.path.join(BASE_DIR, "config/mapping/stadium_map_nfl.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/weather")
ERROR_LOG_DIR = os.path.join(BASE_DIR, "errors/00_intake")
ERROR_LOG_PATH = os.path.join(ERROR_LOG_DIR, "fetch_weather.txt")

OUTPUT_HEADERS = [
    "game_id",
    "stadium",
    "latitude",
    "longitude",
    "game_time",
    "game_timezone",
    "temperature",
    "wind_speed",
    "wind_gust",
    "precip_probability",
    "rain_flag",
    "snow_flag",
    "humidity",
    "roof_type",
    "dome_flag",
    "retractable_roof_flag",
    "open_air_flag",
    "weather_fetched_at",
]


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_stadium_map():
    """
    Keyed by (stadium, team) since join is on stadium + home_team + game_date;
    game_date isn't part of the stadium map itself, so it's matched separately.
    """
    rows = load_csv(STADIUM_MAP_PATH)
    lookup = {}
    for row in rows:
        key = (row["stadium"].strip(), row["team"].strip())
        lookup[key] = row
    return lookup


def is_future_game(game_date, game_time, game_timezone, log_lines):
    """
    Returns True if the game's date/time (in game_timezone) is in the future
    relative to now. Returns False if it's in the past. Returns None if the
    timezone/date/time could not be parsed (treated as unknown/skip).
    """
    if not game_date or not game_time or not game_timezone:
        return None
    try:
        naive = datetime.strptime(f"{game_date} {game_time}", "%Y-%m-%d %H:%M")
        tz = ZoneInfo(game_timezone)
        game_dt = naive.replace(tzinfo=tz)
        return game_dt > datetime.now(timezone.utc)
    except Exception as e:
        log_lines.append(f"WARNING: could not parse game date/time/timezone "
                          f"({game_date} {game_time} {game_timezone}): {e}")
        return None


def fetch_weather(lat, lon, log_lines, game_id):
    url = f"{METNO_URL}?lat={lat}&lon={lon}"
    req = urllib.request.Request(url, headers={"User-Agent": METNO_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        log_lines.append(f"ERROR: game_id={game_id} met.no request failed "
                          f"(HTTP {e.code}) for lat={lat}, lon={lon}")
        return None
    except Exception as e:
        log_lines.append(f"ERROR: game_id={game_id} met.no request failed "
                          f"for lat={lat}, lon={lon}: {e}")
        return None


def find_closest_timestep(weather_json, target_dt_utc):
    """
    Returns the timeseries entry whose 'time' is closest to target_dt_utc,
    or None if no timeseries data exists (e.g. lookup failed) or the target
    is out of the ~9 day forecast range covered by the response.
    """
    if not weather_json:
        return None
    timeseries = weather_json.get("properties", {}).get("timeseries", [])
    if not timeseries:
        return None

    best = None
    best_diff = None
    for entry in timeseries:
        try:
            entry_time = datetime.strptime(entry["time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        diff = abs((entry_time - target_dt_utc).total_seconds())
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = entry

    # If closest available timestep is more than ~12 hours off, treat as
    # out of forecast range (met.no only forecasts ~9 days out).
    if best is not None and best_diff is not None and best_diff > 12 * 3600:
        return None

    return best


def extract_precip_probability(entry):
    details = entry.get("data", {})
    for period_key in ("next_1_hours", "next_6_hours", "next_12_hours"):
        period = details.get(period_key, {})
        prob = period.get("details", {}).get("probability_of_precipitation")
        if prob is not None:
            return prob
    return None


def extract_symbol_code(entry):
    details = entry.get("data", {})
    for period_key in ("next_1_hours", "next_6_hours", "next_12_hours"):
        period = details.get(period_key, {})
        code = period.get("summary", {}).get("symbol_code")
        if code:
            return code
    return ""


def derive_rain_snow_flags(symbol_code):
    code = symbol_code.lower()
    rain_flag = 1 if "rain" in code or "sleet" in code else 0
    snow_flag = 1 if "snow" in code else 0
    return rain_flag, snow_flag


def get_weather_row_values(lat, lon, target_dt_utc, log_lines, game_id):
    weather_json = fetch_weather(lat, lon, log_lines, game_id)
    entry = find_closest_timestep(weather_json, target_dt_utc)

    if entry is None:
        log_lines.append(f"INFO: game_id={game_id} no weather data available "
                          f"(outside forecast range or fetch failed) — "
                          f"weather columns left blank.")
        return {
            "temperature": "",
            "wind_speed": "",
            "wind_gust": "",
            "precip_probability": "",
            "rain_flag": "",
            "snow_flag": "",
            "humidity": "",
        }

    instant = entry.get("data", {}).get("instant", {}).get("details", {})
    precip_prob = extract_precip_probability(entry)
    symbol_code = extract_symbol_code(entry)
    rain_flag, snow_flag = derive_rain_snow_flags(symbol_code)

    return {
        "temperature": instant.get("air_temperature", ""),
        "wind_speed": instant.get("wind_speed", ""),
        "wind_gust": instant.get("wind_speed_of_gust", ""),
        "precip_probability": precip_prob if precip_prob is not None else "",
        "rain_flag": rain_flag,
        "snow_flag": snow_flag,
        "humidity": instant.get("relative_humidity", ""),
    }


def build_row(game, stadium_row, log_lines, weather_fetched_at):
    game_id = game["game_id"]

    if stadium_row is None:
        log_lines.append(f"ERROR: game_id={game_id} no stadium_map match for "
                          f"stadium='{game.get('stadium')}' "
                          f"home_team='{game.get('home_team')}'")
        return {
            "game_id": game_id,
            "stadium": game.get("stadium", ""),
            "latitude": "",
            "longitude": "",
            "game_time": game.get("game_time", ""),
            "game_timezone": game.get("game_timezone", ""),
            "temperature": "",
            "wind_speed": "",
            "wind_gust": "",
            "precip_probability": "",
            "rain_flag": "",
            "snow_flag": "",
            "humidity": "",
            "roof_type": "",
            "dome_flag": "",
            "retractable_roof_flag": "",
            "open_air_flag": "",
            "weather_fetched_at": weather_fetched_at,
        }

    lat = stadium_row["latitude"]
    lon = stadium_row["longitude"]

    game_date = game.get("game_date", "")
    game_time_str = game.get("game_time", "")
    game_timezone = game.get("game_timezone", "")

    weather_values = {
        "temperature": "",
        "wind_speed": "",
        "wind_gust": "",
        "precip_probability": "",
        "rain_flag": "",
        "snow_flag": "",
        "humidity": "",
    }

    if lat and lon and game_date and game_time_str and game_timezone:
        try:
            naive = datetime.strptime(f"{game_date} {game_time_str}", "%Y-%m-%d %H:%M")
            tz = ZoneInfo(game_timezone)
            target_dt_utc = naive.replace(tzinfo=tz).astimezone(timezone.utc)
            weather_values = get_weather_row_values(lat, lon, target_dt_utc, log_lines, game_id)
            time.sleep(REQUEST_SLEEP_SECONDS)
        except Exception as e:
            log_lines.append(f"ERROR: game_id={game_id} failed computing target "
                              f"datetime for weather lookup: {e}")
    else:
        log_lines.append(f"WARNING: game_id={game_id} missing lat/lon/date/time/"
                          f"timezone — weather columns left blank.")

    return {
        "game_id": game_id,
        "stadium": game.get("stadium", ""),
        "latitude": lat,
        "longitude": lon,
        "game_time": game.get("game_time", ""),
        "game_timezone": game.get("game_timezone", ""),
        **weather_values,
        "roof_type": stadium_row.get("roof_type", ""),
        "dome_flag": stadium_row.get("dome_flag", ""),
        "retractable_roof_flag": stadium_row.get("retractable_roof_flag", ""),
        "open_air_flag": stadium_row.get("open_air_flag", ""),
        "weather_fetched_at": weather_fetched_at,
    }


def process_week(week, schedule_path, log_lines, weather_fetched_at, stadium_lookup):
    output_path = os.path.join(OUTPUT_DIR, f"week_{week}_NFL_weekly_weather.csv")

    schedule_rows = load_csv(schedule_path)
    existing_rows_by_id = {}
    if os.path.exists(output_path):
        for row in load_csv(output_path):
            existing_rows_by_id[row["game_id"]] = row

    output_rows = []

    for game in schedule_rows:
        game_id = game["game_id"]
        game_date = game.get("game_date", "")
        game_time_str = game.get("game_time", "")
        game_timezone = game.get("game_timezone", "")

        future = is_future_game(game_date, game_time_str, game_timezone, log_lines)

        if future is False and game_id in existing_rows_by_id:
            # Past game already present in output — leave untouched.
            output_rows.append(existing_rows_by_id[game_id])
            continue

        if future is False and game_id not in existing_rows_by_id:
            # Past game not yet in output — still record attainable data,
            # but do not attempt a weather fetch (game already happened).
            log_lines.append(f"INFO: game_id={game_id} game is in the past and "
                              f"not in existing output — recording non-weather "
                              f"fields only.")
            stadium_key = (game.get("stadium", "").strip(), game.get("home_team", "").strip())
            stadium_row = stadium_lookup.get(stadium_key)
            row = build_row(game, stadium_row, log_lines, weather_fetched_at)
            for col in ["temperature", "wind_speed", "wind_gust", "precip_probability",
                        "rain_flag", "snow_flag", "humidity"]:
                row[col] = ""
            output_rows.append(row)
            continue

        # Future game (or unknown/unparseable date-time) — fetch/update weather.
        stadium_key = (game.get("stadium", "").strip(), game.get("home_team", "").strip())
        stadium_row = stadium_lookup.get(stadium_key)
        row = build_row(game, stadium_row, log_lines, weather_fetched_at)
        output_rows.append(row)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADERS)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print(f"Wrote {len(output_rows)} rows to {output_path}")


def main():
    log_lines = []
    weather_fetched_at = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")
    stadium_lookup = load_stadium_map()

    schedule_files = sorted(glob.glob(os.path.join(SCHEDULE_DIR, "week_*_NFL_weekly_schedule.csv")))

    if not schedule_files:
        print(f"WARNING: no weekly schedule files found in {SCHEDULE_DIR}")

    for schedule_path in schedule_files:
        filename = os.path.basename(schedule_path)
        match = re.match(r"week_(\d+)_NFL_weekly_schedule\.csv", filename)
        if not match:
            log_lines.append(f"WARNING: skipped unrecognized file name: {filename}")
            continue
        week = int(match.group(1))
        process_week(week, schedule_path, log_lines, weather_fetched_at, stadium_lookup)

    os.makedirs(ERROR_LOG_DIR, exist_ok=True)
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n--- Run at {weather_fetched_at} ---\n")
        if log_lines:
            for line in log_lines:
                f.write(line + "\n")
        else:
            f.write("No issues.\n")

    print(f"Log written to {ERROR_LOG_PATH}")


if __name__ == "__main__":
    main()

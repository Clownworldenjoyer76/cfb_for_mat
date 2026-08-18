#!/usr/bin/env python3
"""
build_travel.py

Builds docs/win/football/nfl/data/travel/{season}_week_{week}_travel.csv

Reads all weekly schedule files found in:
    docs/win/football/nfl/00_intake/schedule/weekly/week_{week}_NFL_weekly_schedule.csv

Joins stadium/location data from:
    docs/win/football/nfl/config/mapping/stadium_map_nfl.csv
    (joined on team name)

Output columns:
    game_id, away_team, home_team, away_lat, away_lon, home_lat, home_lon,
    miles_traveled, time_zones_crossed, east_to_west, west_to_east,
    international_flag, neutral_site_flag

Notes:
    - away_lat/away_lon/home_lat/home_lon are each team's home stadium
      coordinates (from stadium_map_nfl.csv), regardless of neutral_site.
    - miles_traveled is the great-circle (haversine) distance between the
      away team's home stadium and the home team's stadium.
    - time_zones_crossed is the difference in UTC offset (in hours) between
      the away team's home timezone and the home team's timezone, computed
      using the game_date to account for DST.
    - east_to_west / west_to_east are binary flags describing the direction
      of travel for the away team based on longitude change.
    - international_flag is 1 if the home team's stadium is outside the USA
      (venue_country != "USA"), else 0.
    - neutral_site_flag is taken directly from the schedule file's
      neutral_site column.

Manual run only.
"""

import csv
import glob
import math
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

BASE_DIR = "docs/win/football/nfl"
SCHEDULE_DIR = os.path.join(BASE_DIR, "00_intake/schedule/weekly")
STADIUM_MAP_PATH = os.path.join(BASE_DIR, "config/mapping/stadium_map_nfl.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/travel")

OUTPUT_HEADERS = [
    "game_id",
    "away_team",
    "home_team",
    "away_lat",
    "away_lon",
    "home_lat",
    "home_lon",
    "miles_traveled",
    "time_zones_crossed",
    "east_to_west",
    "west_to_east",
    "international_flag",
    "neutral_site_flag",
]


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_stadium_map():
    rows = load_csv(STADIUM_MAP_PATH)
    return {row["team"].strip(): row for row in rows}


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8  # earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def utc_offset_hours(tz_name, game_date):
    try:
        dt = datetime.strptime(game_date, "%Y-%m-%d").replace(tzinfo=ZoneInfo(tz_name))
        return dt.utcoffset().total_seconds() / 3600
    except Exception:
        return None


def build_row(game, stadium_lookup, log_lines):
    game_id = game.get("game_id", "")
    away_team = game.get("away_team", "").strip()
    home_team = game.get("home_team", "").strip()
    game_date = game.get("game_date", "")
    neutral_site = game.get("neutral_site", "")

    away_row = stadium_lookup.get(away_team)
    home_row = stadium_lookup.get(home_team)

    if away_row is None:
        log_lines.append(f"ERROR: game_id={game_id} no stadium_map match for away_team='{away_team}'")
    if home_row is None:
        log_lines.append(f"ERROR: game_id={game_id} no stadium_map match for home_team='{home_team}'")

    away_lat = away_row["latitude"] if away_row else ""
    away_lon = away_row["longitude"] if away_row else ""
    home_lat = home_row["latitude"] if home_row else ""
    home_lon = home_row["longitude"] if home_row else ""

    miles_traveled = ""
    time_zones_crossed = ""
    east_to_west = ""
    west_to_east = ""
    international_flag = ""

    if away_row and home_row:
        try:
            miles_traveled = round(
                haversine_miles(float(away_lat), float(away_lon), float(home_lat), float(home_lon)), 1
            )
        except Exception as e:
            log_lines.append(f"ERROR: game_id={game_id} failed computing miles_traveled: {e}")

        away_offset = utc_offset_hours(away_row.get("timezone", ""), game_date)
        home_offset = utc_offset_hours(home_row.get("timezone", ""), game_date)
        if away_offset is not None and home_offset is not None:
            time_zones_crossed = abs(home_offset - away_offset)
        else:
            log_lines.append(f"WARNING: game_id={game_id} could not compute time_zones_crossed "
                              f"(timezone/date parse issue)")

        try:
            away_lon_f = float(away_lon)
            home_lon_f = float(home_lon)
            if home_lon_f > away_lon_f:
                west_to_east = 1
                east_to_west = 0
            elif home_lon_f < away_lon_f:
                west_to_east = 0
                east_to_west = 1
            else:
                west_to_east = 0
                east_to_west = 0
        except Exception as e:
            log_lines.append(f"ERROR: game_id={game_id} failed computing travel direction: {e}")

        home_country = home_row.get("venue_country", "").strip()
        international_flag = 0 if home_country == "USA" else 1

    return {
        "game_id": game_id,
        "away_team": away_team,
        "home_team": home_team,
        "away_lat": away_lat,
        "away_lon": away_lon,
        "home_lat": home_lat,
        "home_lon": home_lon,
        "miles_traveled": miles_traveled,
        "time_zones_crossed": time_zones_crossed,
        "east_to_west": east_to_west,
        "west_to_east": west_to_east,
        "international_flag": international_flag,
        "neutral_site_flag": neutral_site,
    }


def process_week(season, week, schedule_path, stadium_lookup, log_lines):
    output_path = os.path.join(OUTPUT_DIR, f"{season}_week_{week}_travel.csv")

    schedule_rows = load_csv(schedule_path)
    output_rows = [build_row(game, stadium_lookup, log_lines) for game in schedule_rows]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADERS)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    print(f"Wrote {len(output_rows)} rows to {output_path}")


def main():
    log_lines = []
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

        rows = load_csv(schedule_path)
        season = rows[0]["season"] if rows else ""

        process_week(season, week, schedule_path, stadium_lookup, log_lines)

    if log_lines:
        print("Issues encountered:")
        for line in log_lines:
            print(line)


if __name__ == "__main__":
    main()

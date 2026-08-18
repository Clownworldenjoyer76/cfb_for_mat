#!/usr/bin/env python3
# docs/win/football/nfl/scripts/00_intake/build_weekly_schedule.py

import csv
import json
import re
import sys
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


BASE_DIR = Path("docs/win/football/nfl")

SCHEDULE_DIR = BASE_DIR / "00_intake" / "schedule"
WEEKLY_DIR = SCHEDULE_DIR / "weekly"

ODDS_DIR = BASE_DIR / "00_intake" / "odds"
RAW_ODDS_DIR = ODDS_DIR / "raw"

TEAM_MAP_PATH = BASE_DIR / "config" / "mapping" / "team_map_nfl.csv"

ERROR_DIR = BASE_DIR / "errors" / "00_intake"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = ERROR_DIR / "build_weekly_schedule.txt"

WEEKLY_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "odds_provider_game_id",
    "game_date",
    "game_time",
    "commence_time",
    "away_team",
    "home_team",
    "odds_away_team",
    "odds_home_team",
    "neutral_site",
    "stadium",
    "roof",
    "surface",
    "home_timezone",
    "away_timezone",
    "game_timezone",
    "bookmaker",
    "home_moneyline_american",
    "away_moneyline_american",
    "home_spread",
    "away_spread",
    "home_spread_american",
    "away_spread_american",
    "total",
    "over_american",
    "under_american",
    "odds_last_update",
    "odds_available",
    "odds_missing_reason",
]

SCHEDULE_REQUIRED_COLUMNS = [
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
    "roof",
    "surface",
    "home_timezone",
    "away_timezone",
    "game_timezone",
]

ODDS_REQUIRED_COLUMNS = [
    "game_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker",
    "market_type",
    "bet_side",
    "line",
    "odds_american",
    "odds_decimal",
    "last_update",
    "home_moneyline_american",
    "away_moneyline_american",
    "home_spread",
    "away_spread",
    "home_spread_american",
    "away_spread_american",
    "total",
    "over_american",
    "under_american",
]

OBSERVED_ODDS_TEAM_MAP = {
    "Arizona": "Arizona Cardinals",
    "Atlanta": "Atlanta Falcons",
    "Baltimore": "Baltimore Ravens",
    "Buffalo": "Buffalo Bills",
    "Carolina": "Carolina Panthers",
    "Chicago": "Chicago Bears",
    "Cincinnati": "Cincinnati Bengals",
    "Cleveland": "Cleveland Browns",
    "Dallas": "Dallas Cowboys",
    "Denver": "Denver Broncos",
    "Detroit": "Detroit Lions",
    "Green Bay": "Green Bay Packers",
    "Houston": "Houston Texans",
    "Indianapolis": "Indianapolis Colts",
    "Jacksonville": "Jacksonville Jaguars",
    "Kansas City": "Kansas City Chiefs",
    "LA Chargers": "Los Angeles Chargers",
    "LA Rams": "Los Angeles Rams",
    "Las Vegas": "Las Vegas Raiders",
    "Miami": "Miami Dolphins",
    "Minnesota": "Minnesota Vikings",
    "New England": "New England Patriots",
    "New Orleans": "New Orleans Saints",
    "NY Giants": "New York Giants",
    "NY Jets": "New York Jets",
    "Philadelphia": "Philadelphia Eagles",
    "Pittsburgh": "Pittsburgh Steelers",
    "San Francisco": "San Francisco 49ers",
    "Seattle": "Seattle Seahawks",
    "Tampa Bay": "Tampa Bay Buccaneers",
    "Tennessee": "Tennessee Titans",
    "Washington": "Washington Commanders",
}


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now_iso()}] {message}\n")


def fail(message):
    log(f"ERROR: {message}")
    raise RuntimeError(message)


def read_csv(path, required_columns, label):
    if not path.exists():
        fail(f"Missing {label}: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            fail(f"{label} missing columns: {missing}")

        return list(reader)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def latest_file(directory, pattern, label):
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if not files:
        fail(f"No {label} found in {directory} matching {pattern}")

    return files[0]


def parse_date(value):
    text = str(value or "").strip()

    if not text:
        return None

    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except Exception:
        pass

    try:
        return datetime.strptime(text, "%Y_%m_%d").date()
    except Exception:
        pass

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(text).date()
    except Exception:
        return None


def normalize_key(value):
    text = str(value or "").strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_team_map():
    mapping = {}

    for source_name, canonical_team in OBSERVED_ODDS_TEAM_MAP.items():
        mapping[normalize_key(source_name)] = canonical_team
        mapping[normalize_key(canonical_team)] = canonical_team

    if TEAM_MAP_PATH.exists():
        with TEAM_MAP_PATH.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames and "source_name" in reader.fieldnames and "canonical_team" in reader.fieldnames:
                for row in reader:
                    source_name = str(row.get("source_name", "")).strip()
                    canonical_team = str(row.get("canonical_team", "")).strip()

                    if source_name and canonical_team:
                        mapping[normalize_key(source_name)] = canonical_team
                        mapping[normalize_key(canonical_team)] = canonical_team

    return mapping


def canonical_team(value, team_map):
    raw = str(value or "").strip()

    if not raw:
        return ""

    return team_map.get(normalize_key(raw), raw)


def load_raw_odds_events(path):
    if not path.exists():
        fail(f"Missing raw odds JSON: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    events = payload.get("events", [])
    odds = payload.get("odds", [])

    if not isinstance(events, list):
        fail("Raw odds JSON field 'events' is not a list")

    if not isinstance(odds, list):
        fail("Raw odds JSON field 'odds' is not a list")

    return events, odds


def latest_last_update(rows):
    values = [str(row.get("last_update", "")).strip() for row in rows if str(row.get("last_update", "")).strip()]
    return max(values) if values else ""


def build_odds_summary(odds_rows):
    grouped = {}

    for row in odds_rows:
        odds_game_id = str(row.get("game_id", "")).strip()

        if not odds_game_id:
            continue

        grouped.setdefault(odds_game_id, []).append(row)

    summaries = {}

    for odds_game_id, rows in grouped.items():
        first = rows[0]

        summaries[odds_game_id] = {
            "bookmaker": first.get("bookmaker", ""),
            "home_moneyline_american": first.get("home_moneyline_american", ""),
            "away_moneyline_american": first.get("away_moneyline_american", ""),
            "home_spread": first.get("home_spread", ""),
            "away_spread": first.get("away_spread", ""),
            "home_spread_american": first.get("home_spread_american", ""),
            "away_spread_american": first.get("away_spread_american", ""),
            "total": first.get("total", ""),
            "over_american": first.get("over_american", ""),
            "under_american": first.get("under_american", ""),
            "odds_last_update": latest_last_update(rows),
            "odds_available": "1",
            "odds_missing_reason": "",
        }

    return summaries


def build_schedule_index(schedule_rows, team_map):
    index = {}

    for row in schedule_rows:
        home = canonical_team(row.get("home_team", ""), team_map)
        away = canonical_team(row.get("away_team", ""), team_map)
        game_date = parse_date(row.get("game_date", ""))

        if not home or not away or game_date is None:
            continue

        key = (game_date.isoformat(), normalize_key(home), normalize_key(away))
        index.setdefault(key, []).append(row)

    return index


def schedule_candidate_keys(raw_event, team_map):
    odds_home = canonical_team(raw_event.get("home", ""), team_map)
    odds_away = canonical_team(raw_event.get("away", ""), team_map)
    odds_date = parse_date(raw_event.get("date", ""))

    if not odds_home or not odds_away or odds_date is None:
        return []

    keys = []

    for candidate_date in [odds_date, odds_date - timedelta(days=1)]:
        keys.append(
            (
                candidate_date.isoformat(),
                normalize_key(odds_home),
                normalize_key(odds_away),
            )
        )

    return keys


def match_raw_events_to_schedule(raw_events, schedule_index, team_map):
    matches = {}
    unmatched_events = []

    for event in raw_events:
        odds_provider_game_id = str(event.get("id", "")).strip()
        matched_schedule = None

        for key in schedule_candidate_keys(event, team_map):
            candidates = schedule_index.get(key, [])

            if candidates:
                matched_schedule = candidates[0]
                break

        if matched_schedule:
            schedule_game_id = str(matched_schedule.get("game_id", "")).strip()
            matches[schedule_game_id] = {
                "odds_provider_game_id": odds_provider_game_id,
                "commence_time": str(event.get("date", "")).strip(),
                "odds_home_team": str(event.get("home", "")).strip(),
                "odds_away_team": str(event.get("away", "")).strip(),
            }
        else:
            unmatched_events.append(event)

    return matches, unmatched_events


def choose_target_week(schedule_rows, schedule_matches):
    week_counter = Counter()

    matched_schedule_ids = set(schedule_matches.keys())

    for row in schedule_rows:
        schedule_game_id = str(row.get("game_id", "")).strip()

        if schedule_game_id in matched_schedule_ids:
            key = (
                str(row.get("season", "")).strip(),
                str(row.get("season_type", "")).strip(),
                str(row.get("week", "")).strip(),
            )
            week_counter[key] += 1

    if not week_counter:
        fail("No schedule week could be identified from odds/schedule matches")

    return week_counter.most_common(1)[0][0]


def build_output_rows(schedule_rows, target_week, schedule_matches, odds_summary):
    output_rows = []

    target_season, target_season_type, target_week_number = target_week

    for schedule_row in schedule_rows:
        if str(schedule_row.get("season", "")).strip() != target_season:
            continue

        if str(schedule_row.get("season_type", "")).strip() != target_season_type:
            continue

        if str(schedule_row.get("week", "")).strip() != target_week_number:
            continue

        schedule_game_id = str(schedule_row.get("game_id", "")).strip()
        match = schedule_matches.get(schedule_game_id, {})
        odds_provider_game_id = str(match.get("odds_provider_game_id", "")).strip()
        odds = odds_summary.get(odds_provider_game_id, {})

        row = {
            "season": schedule_row.get("season", ""),
            "season_type": schedule_row.get("season_type", ""),
            "week": schedule_row.get("week", ""),
            "game_id": schedule_game_id,
            "odds_provider_game_id": odds_provider_game_id,
            "game_date": schedule_row.get("game_date", ""),
            "game_time": schedule_row.get("game_time", ""),
            "commence_time": match.get("commence_time", ""),
            "away_team": schedule_row.get("away_team", ""),
            "home_team": schedule_row.get("home_team", ""),
            "odds_away_team": match.get("odds_away_team", ""),
            "odds_home_team": match.get("odds_home_team", ""),
            "neutral_site": schedule_row.get("neutral_site", ""),
            "stadium": schedule_row.get("stadium", ""),
            "roof": schedule_row.get("roof", ""),
            "surface": schedule_row.get("surface", ""),
            "home_timezone": schedule_row.get("home_timezone", ""),
            "away_timezone": schedule_row.get("away_timezone", ""),
            "game_timezone": schedule_row.get("game_timezone", ""),
            "bookmaker": odds.get("bookmaker", ""),
            "home_moneyline_american": odds.get("home_moneyline_american", ""),
            "away_moneyline_american": odds.get("away_moneyline_american", ""),
            "home_spread": odds.get("home_spread", ""),
            "away_spread": odds.get("away_spread", ""),
            "home_spread_american": odds.get("home_spread_american", ""),
            "away_spread_american": odds.get("away_spread_american", ""),
            "total": odds.get("total", ""),
            "over_american": odds.get("over_american", ""),
            "under_american": odds.get("under_american", ""),
            "odds_last_update": odds.get("odds_last_update", ""),
            "odds_available": "",
            "odds_missing_reason": "",
        }

        if odds_provider_game_id and odds:
            row["odds_available"] = "1"
            row["odds_missing_reason"] = ""
        elif odds_provider_game_id and not odds:
            row["odds_available"] = "0"
            row["odds_missing_reason"] = "no_odds_returned"
        else:
            row["odds_available"] = "0"
            row["odds_missing_reason"] = "no_odds_event_match"

        output_rows.append(row)

    output_rows.sort(
        key=lambda row: (
            row.get("game_date", ""),
            row.get("game_time", ""),
            row.get("away_team", ""),
            row.get("home_team", ""),
        )
    )

    return output_rows


def main():
    LOG_FILE.write_text("", encoding="utf-8")

    schedule_path = latest_file(SCHEDULE_DIR, "*_schedule.csv", "schedule CSV")
    odds_csv_path = latest_file(ODDS_DIR, "*_NFL_odds.csv", "NFL odds CSV")
    raw_odds_path = latest_file(RAW_ODDS_DIR, "*_nfl_odds.json", "raw NFL odds JSON")

    log(f"Schedule input: {schedule_path}")
    log(f"Odds CSV input: {odds_csv_path}")
    log(f"Raw odds input: {raw_odds_path}")

    team_map = load_team_map()

    schedule_rows = read_csv(schedule_path, SCHEDULE_REQUIRED_COLUMNS, "schedule CSV")
    odds_rows = read_csv(odds_csv_path, ODDS_REQUIRED_COLUMNS, "odds CSV")
    raw_events, raw_odds = load_raw_odds_events(raw_odds_path)

    schedule_index = build_schedule_index(schedule_rows, team_map)
    schedule_matches, unmatched_events = match_raw_events_to_schedule(raw_events, schedule_index, team_map)
    odds_summary = build_odds_summary(odds_rows)

    target_week = choose_target_week(schedule_rows, schedule_matches)
    target_week_number = str(target_week[2]).strip()

    output_path = WEEKLY_DIR / f"week_{target_week_number}_NFL_weekly_schedule.csv"

    output_rows = build_output_rows(schedule_rows, target_week, schedule_matches, odds_summary)

    write_csv(output_path, output_rows)

    matched_with_odds = sum(1 for row in output_rows if row.get("odds_available") == "1")
    matched_without_odds = sum(1 for row in output_rows if row.get("odds_missing_reason") == "no_odds_returned")
    no_event_match = sum(1 for row in output_rows if row.get("odds_missing_reason") == "no_odds_event_match")

    log(f"Schedule rows loaded: {len(schedule_rows)}")
    log(f"Raw odds events loaded: {len(raw_events)}")
    log(f"Raw odds objects loaded: {len(raw_odds)}")
    log(f"Odds CSV rows loaded: {len(odds_rows)}")
    log(f"Schedule matches from raw odds events: {len(schedule_matches)}")
    log(f"Unmatched raw odds events: {len(unmatched_events)}")
    log(f"Target week: season={target_week[0]}, season_type={target_week[1]}, week={target_week[2]}")
    log(f"Weekly schedule rows written: {len(output_rows)}")
    log(f"Rows with odds: {matched_with_odds}")
    log(f"Rows with event but no odds: {matched_without_odds}")
    log(f"Rows with no odds event match: {no_event_match}")
    log(f"Output written: {output_path}")

    if unmatched_events:
        for event in unmatched_events:
            log(
                "UNMATCHED_RAW_EVENT "
                f"id={event.get('id', '')} "
                f"date={event.get('date', '')} "
                f"away={event.get('away', '')} "
                f"home={event.get('home', '')}"
            )

    print(f"Weekly schedule written: {output_path}")
    print(f"Rows written: {len(output_rows)}")
    print(f"Rows with odds: {matched_with_odds}")
    print(f"Rows with event but no odds: {matched_without_odds}")
    print(f"Rows with no odds event match: {no_event_match}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        print(f"ERROR: see {LOG_FILE}", file=sys.stderr)
        raise

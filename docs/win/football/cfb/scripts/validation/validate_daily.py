#!/usr/bin/env python3

import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

NFL_ROOT = Path("docs/win/football/nfl")


def fail(message):
    print(f"VALIDATION FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def passed(message):
    print(f"PASS: {message}")


def read_csv(path, required_columns, *, allow_empty=False, unique_by=None):
    if not path.is_file():
        fail(f"Missing file: {path}")

    if path.stat().st_size == 0:
        fail(f"Zero-byte file: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)

    missing = [column for column in required_columns if column not in columns]
    if missing:
        fail(f"{path} is missing columns: {missing}")

    if not rows and not allow_empty:
        fail(f"{path} has a header but no data rows")

    if unique_by and rows:
        seen = set()
        duplicates = []
        for line_number, row in enumerate(rows, start=2):
            key = tuple(str(row.get(column, "")).strip() for column in unique_by)
            if not all(key):
                fail(
                    f"{path} line {line_number} has a blank unique-key value "
                    f"for columns {unique_by}: {key}"
                )
            if key in seen:
                duplicates.append(key)
            seen.add(key)

        if duplicates:
            fail(
                f"{path} contains duplicate keys for {unique_by}. "
                f"Examples: {duplicates[:5]}"
            )

    passed(f"{path} | rows={len(rows)}")
    return rows


def current_schedule():
    candidates = []
    for path in (NFL_ROOT / "00_intake/schedule").glob("*_schedule.csv"):
        match = re.fullmatch(r"(\d{4})_schedule\.csv", path.name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        fail("No season schedule file found")

    season, path = max(candidates)
    rows = read_csv(
        path,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "game_date",
            "game_time",
            "away_team",
            "home_team",
        ],
        unique_by=["game_id"],
    )
    return season, rows


def validate_json(path):
    if not path.is_file():
        fail(f"Missing file: {path}")
    if path.stat().st_size == 0:
        fail(f"Zero-byte file: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"Invalid JSON in {path}: {exc}")

    events = payload.get("events")
    odds = payload.get("odds")

    if not isinstance(events, list) or not events:
        fail(f"{path} has no events list data")
    if not isinstance(odds, list) or not odds:
        fail(f"{path} has no odds list data")

    passed(f"{path} | events={len(events)} odds_events={len(odds)}")


def main():
    print("Validating NFL daily workflow outputs")

    season, full_schedule_rows = current_schedule()
    full_schedule_game_ids = {
        str(row["game_id"]).strip() for row in full_schedule_rows
    }

    run_stamp = date.today().strftime("%Y_%m_%d")
    odds_csv = NFL_ROOT / f"00_intake/odds/{run_stamp}_NFL_odds.csv"
    odds_rows = read_csv(
        odds_csv,
        [
            "game_id",
            "commence_time",
            "home_team",
            "away_team",
            "bookmaker",
            "market_type",
            "bet_side",
            "odds_american",
        ],
        unique_by=["game_id", "market_type", "bet_side", "bookmaker"],
    )

    odds_game_ids = {str(row["game_id"]).strip() for row in odds_rows}
    if not odds_game_ids:
        fail(f"{odds_csv} contains no game IDs")

    validate_json(
        NFL_ROOT / f"00_intake/odds/raw/{run_stamp}_nfl_odds.json"
    )

    weekly_files = sorted(
        (NFL_ROOT / "00_intake/schedule/weekly").glob(
            "week_*_NFL_weekly_schedule.csv"
        )
    )
    if not weekly_files:
        fail("No weekly schedule files found")

    weekly_game_ids = set()
    weekly_by_week = {}

    for path in weekly_files:
        match = re.fullmatch(r"week_(\d+)_NFL_weekly_schedule\.csv", path.name)
        if not match:
            continue

        rows = read_csv(
            path,
            [
                "season",
                "season_type",
                "week",
                "game_id",
                "odds_provider_game_id",
                "game_date",
                "game_time",
                "away_team",
                "home_team",
                "bookmaker",
                "odds_available",
            ],
            unique_by=["game_id"],
        )

        file_seasons = {str(row["season"]).strip() for row in rows}
        if file_seasons != {str(season)}:
            continue

        week = int(match.group(1))
        game_ids = {str(row["game_id"]).strip() for row in rows}

        unknown = game_ids - full_schedule_game_ids
        if unknown:
            fail(f"{path} contains game IDs absent from {season}_schedule.csv: {sorted(unknown)[:5]}")

        weekly_game_ids.update(game_ids)
        weekly_by_week[week] = game_ids

    if not weekly_by_week:
        fail(f"No weekly schedule files contain season {season}")

    openers_path = NFL_ROOT / f"00_intake/odds/openers/{season}_NFL_openers.csv"
    opener_rows = read_csv(
        openers_path,
        [
            "game_id",
            "odds_provider_game_id",
            "market_type",
            "bet_side",
            "bookmaker",
            "opener_status",
        ],
        unique_by=["game_id", "market_type", "bet_side", "bookmaker"],
    )

    opener_game_ids = {str(row["game_id"]).strip() for row in opener_rows}
    unknown_openers = opener_game_ids - weekly_game_ids
    if unknown_openers:
        fail(
            f"{openers_path} contains game IDs absent from weekly schedules: "
            f"{sorted(unknown_openers)[:5]}"
        )

    roster_path = NFL_ROOT / "data/master/roster_master.csv"
    read_csv(
        roster_path,
        [
            "id",
            "displayName",
            "position.id",
            "position.abbreviation",
            "team_id",
        ],
    )

    depth_files = sorted(
        (NFL_ROOT / "data/master/depth_charts").glob("*/*_depth.csv")
    )
    if len(depth_files) < 32:
        fail(
            "Expected at least 32 team depth-chart files, "
            f"found {len(depth_files)}"
        )

    for path in depth_files:
        read_csv(
            path,
            [
                "player_id",
                "name",
                "team",
                "position_abb",
                "depth_chart_rank",
                "starter_flag",
                "backup_flag",
                "team_id",
                "season",
            ],
            unique_by=[
                "team",
                "player_id",
                "position_abb",
                "depth_chart_rank",
            ],
        )

    read_csv(
        NFL_ROOT / "config/mapping/qb_map_nfl.csv",
        [
            "player_id",
            "qb_name",
            "team_abbr",
            "depth_chart_rank",
            "starter_flag",
            "position.id",
            "team_id",
        ],
        unique_by=["player_id"],
    )

    read_csv(
        NFL_ROOT / f"00_intake/injuries/{season}_injuries.csv",
        [
            "season",
            "team",
            "player_id",
            "player_name",
            "position",
            "game_status",
            "report_date",
        ],
    )

    for week, schedule_ids in weekly_by_week.items():
        weather_path = (
            NFL_ROOT / f"data/weather/week_{week}_NFL_weekly_weather.csv"
        )
        weather_rows = read_csv(
            weather_path,
            [
                "game_id",
                "stadium",
                "latitude",
                "longitude",
                "game_time",
                "game_timezone",
                "temperature",
                "wind_speed",
                "roof_type",
                "weather_fetched_at",
            ],
            unique_by=["game_id"],
        )
        weather_ids = {
            str(row["game_id"]).strip() for row in weather_rows
        }
        if weather_ids != schedule_ids:
            fail(
                f"{weather_path} game IDs do not exactly match "
                f"week {week} schedule game IDs"
            )

    prediction_files = sorted(
        (
            NFL_ROOT / "00_intake/predictions/e_predictions"
        ).glob(f"{season}_*_e_predictions.csv")
    )
    if not prediction_files:
        fail(f"No {season} ESPN prediction files found")

    prediction_rows_total = 0
    for path in prediction_files:
        rows = read_csv(
            path,
            [
                "season",
                "season_type",
                "week",
                "game_id",
                "home_away",
                "team_id",
            ],
            unique_by=["game_id", "home_away"],
        )
        prediction_rows_total += len(rows)

        unknown = {
            str(row["game_id"]).strip() for row in rows
        } - full_schedule_game_ids
        if unknown:
            fail(
                f"{path} contains game IDs absent from "
                f"{season}_schedule.csv: {sorted(unknown)[:5]}"
            )

    if prediction_rows_total == 0:
        fail("ESPN prediction files contain no data rows")

    print("DAILY VALIDATION PASSED")


if __name__ == "__main__":
    main()

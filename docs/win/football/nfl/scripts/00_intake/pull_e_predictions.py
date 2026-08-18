"""
pull_espn_predictions.py

Reads game_ids from the existing schedule output and pulls ESPN's
predictor data (pre-game win probability, matchup quality, predicted
point differential, etc.) for every game, one file per week.

Input:
    docs/win/football/nfl/00_intake/schedule/{season}_schedule.csv

Source:
    https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{game_id}/competitions/{game_id}/predictor

Output:
    docs/win/football/nfl/00_intake/predictions/e_predictions/{season}_{season_type}_{week}_e_predictions.csv

Error/run log:
    docs/win/football/nfl/errors/00_intake/pull_espn_predictions.txt
"""

import csv
import json
import os
import urllib.request
from datetime import datetime, timezone
from collections import defaultdict

SEASON = 2026

SCHEDULE_PATH = f"docs/win/football/nfl/00_intake/schedule/{SEASON}_schedule.csv"
PREDICTOR_URL_TEMPLATE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{game_id}/competitions/{game_id}/predictor"

OUTPUT_DIR = "docs/win/football/nfl/00_intake/predictions/e_predictions"
ERROR_LOG_PATH = "docs/win/football/nfl/errors/00_intake/pull_e_predictions.txt"

OUTPUT_HEADER = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_name",
    "home_away",
    "team_id",
    "gameProjection",
    "matchupQuality",
    "oppSeasonStrengthFbsRank",
    "oppSeasonStrengthRating",
    "teamChanceLoss",
    "teamChanceTie",
    "teamPredPtDiff",
]


def log(lines):
    os.makedirs(os.path.dirname(ERROR_LOG_PATH), exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"--- run {timestamp} ---\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n")


def fetch_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def extract_team_id(ref_url):
    if not ref_url:
        return ""
    parts = ref_url.split("/teams/")
    if len(parts) < 2:
        return ""
    return parts[1].split("?")[0]


def get_predictor_rows(game_id, season, season_type, week):
    url = PREDICTOR_URL_TEMPLATE.format(game_id=game_id)

    try:
        predictor = fetch_json(url)
    except Exception as e:
        log([f"game_id={game_id} predictor unavailable: {e}"])
        return []

    game_name = predictor.get("name", "")
    rows = []

    for side in ["homeTeam", "awayTeam"]:
        side_data = predictor.get(side, {})
        if not side_data:
            continue

        team_ref = side_data.get("team", {}).get("$ref", "")
        team_id = extract_team_id(team_ref)

        stats = {stat.get("name", ""): stat.get("value", "") for stat in side_data.get("statistics", [])}

        rows.append({
            "season": season,
            "season_type": season_type,
            "week": week,
            "game_id": game_id,
            "game_name": game_name,
            "home_away": side,
            "team_id": team_id,
            "gameProjection": stats.get("gameProjection", ""),
            "matchupQuality": stats.get("matchupQuality", ""),
            "oppSeasonStrengthFbsRank": stats.get("oppSeasonStrengthFbsRank", ""),
            "oppSeasonStrengthRating": stats.get("oppSeasonStrengthRating", ""),
            "teamChanceLoss": stats.get("teamChanceLoss", ""),
            "teamChanceTie": stats.get("teamChanceTie", ""),
            "teamPredPtDiff": stats.get("teamPredPtDiff", ""),
        })

    return rows


def main():
    log_lines = [f"season={SEASON}"]

    if not os.path.exists(SCHEDULE_PATH):
        msg = f"ERROR: schedule file not found: {SCHEDULE_PATH}"
        print(msg)
        log([msg])
        return

    with open(SCHEDULE_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        schedule_rows = list(reader)

    rows_by_week = defaultdict(list)
    resolved_count = 0
    failed_count = 0

    for row in schedule_rows:
        game_id = row.get("game_id", "").strip()
        if not game_id:
            continue

        season = row.get("season", "")
        season_type = row.get("season_type", "")
        week = row.get("week", "")

        rows = get_predictor_rows(game_id, season, season_type, week)

        if rows:
            resolved_count += 1
            rows_by_week[(season, season_type, week)].extend(rows)
        else:
            failed_count += 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files_written = 0
    for (season, season_type, week), rows in rows_by_week.items():
        if not season or not season_type or not week:
            log_lines.append(f"SKIPPED group missing season/season_type/week: {season}/{season_type}/{week}")
            continue

        filename = f"{season}_{season_type}_{week}_e_predictions.csv"
        out_path = os.path.join(OUTPUT_DIR, filename)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADER)
            writer.writeheader()
            writer.writerows(rows)

        files_written += 1
        log_lines.append(f"wrote {len(rows)} rows to {out_path}")

    summary = f"games_processed={len(schedule_rows)} resolved={resolved_count} failed={failed_count} files_written={files_written}"
    print(summary)
    log_lines.append(summary)
    log(log_lines)


if __name__ == "__main__":
    main()

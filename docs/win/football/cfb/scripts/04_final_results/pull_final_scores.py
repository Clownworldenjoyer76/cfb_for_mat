"""
pull_final_scores.py

Reads game_ids from the existing schedule output and resolves final
score + status for each game from ESPN's core API. Writes one CSV per
season/season_type/week.

Input:
    docs/win/football/nfl/00_intake/schedule/{season}_schedule.csv

Source:
    https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{game_id}/competitions/{game_id}
    (status + per-competitor score resolved automatically)

Output:
    docs/win/football/nfl/04_final_results/results/{season}_{season_type}_{week}.csv

Error/run log:
    docs/win/football/nfl/errors/04_final_results/pull_final_scores.txt
"""

import csv
import json
import os
import urllib.request
from datetime import datetime, timezone
from collections import defaultdict

SEASON = 2026

SCHEDULE_PATH = f"docs/win/football/nfl/00_intake/schedule/{SEASON}_schedule.csv"
COMPETITION_URL_TEMPLATE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{game_id}/competitions/{game_id}"

RESULTS_DIR = "docs/win/football/nfl/04_final_results/results"
ERROR_LOG_PATH = "docs/win/football/nfl/errors/04_final_results/pull_final_scores.txt"

OUTPUT_HEADER = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "status",
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


def get_score_and_status(game_id):
    """
    Returns (away_score, home_score, is_final, status_text, error) for a
    given game_id. Empty strings for scores if not yet available.
    """
    url = COMPETITION_URL_TEMPLATE.format(game_id=game_id)

    try:
        comp = fetch_json(url)
    except Exception as e:
        return "", "", False, "", f"failed to fetch competition: {e}"

    status_ref = comp.get("status", {}).get("$ref")
    is_final = False
    status_text = ""

    if status_ref:
        try:
            status = fetch_json(status_ref)
            status_type = status.get("type", {})
            is_final = status_type.get("completed", False)
            status_text = status_type.get("description", "")
        except Exception as e:
            log([f"game_id={game_id} failed to resolve status: {e}"])

    away_score = ""
    home_score = ""

    for competitor in comp.get("competitors", []):
        home_away = competitor.get("homeAway", "")
        score_ref = competitor.get("score", {}).get("$ref")

        if not score_ref:
            continue

        try:
            score_data = fetch_json(score_ref)
        except Exception as e:
            log([f"game_id={game_id} failed to resolve score for {home_away}: {e}"])
            continue

        value = score_data.get("displayValue", score_data.get("value", ""))

        if home_away == "away":
            away_score = value
        elif home_away == "home":
            home_score = value

    return away_score, home_score, is_final, status_text, ""


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
    completed_count = 0
    not_final_count = 0
    failed_count = 0

    for row in schedule_rows:
        game_id = row.get("game_id", "").strip()
        if not game_id:
            continue

        away_score, home_score, is_final, status_text, error = get_score_and_status(game_id)

        if error:
            failed_count += 1
            log_lines.append(f"game_id={game_id} error={error}")
        elif is_final:
            completed_count += 1
        else:
            not_final_count += 1

        season_type = row.get("season_type", "")
        week = row.get("week", "")

        out_row = {
            "season": row.get("season", ""),
            "season_type": season_type,
            "week": week,
            "game_id": game_id,
            "game_date": row.get("game_date", ""),
            "game_time": row.get("game_time", ""),
            "away_team": row.get("away_team", ""),
            "home_team": row.get("home_team", ""),
            "away_score": away_score,
            "home_score": home_score,
            "status": status_text,
        }

        key = (out_row["season"], season_type, week)
        rows_by_week[key].append(out_row)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    files_written = 0
    for (season, season_type, week), rows in rows_by_week.items():
        if not season or not season_type or not week:
            log_lines.append(
                f"SKIPPED group missing season/season_type/week: "
                f"{season}/{season_type}/{week}"
            )
            continue

        filename = f"{season}_{season_type}_{week}.csv"
        out_path = os.path.join(RESULTS_DIR, filename)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADER)
            writer.writeheader()
            writer.writerows(rows)

        files_written += 1
        log_lines.append(f"wrote {len(rows)} rows to {out_path}")

    summary = (
        f"games_processed={len(schedule_rows)} completed={completed_count} "
        f"not_final={not_final_count} failed={failed_count} files_written={files_written}"
    )
    print(summary)
    log_lines.append(summary)
    log(log_lines)


if __name__ == "__main__":
    main()
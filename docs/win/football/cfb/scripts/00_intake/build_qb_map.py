#!/usr/bin/env python3
"""
build_qb_map.py

Builds docs/win/football/nfl/config/mapping/qb_map_nfl.csv

Order of operations:
1. Filter roster_master.csv to rows where position.id == 8 (QB).
2. Match team_id to team_master.csv to get team_abbr.
3. Use team_abbr to load docs/win/football/nfl/data/master/depth_charts/{team_abbr}/{team_abbr}_depth.csv
   and fill in remaining values by matching player_id (depth file) == id (roster_master.csv).
   If no match / file missing, leave those fields blank.

Manual run only.
"""

import csv
import os

BASE_DIR = "docs/win/football/nfl"
ROSTER_MASTER = os.path.join(BASE_DIR, "data/master/roster_master.csv")
TEAM_MASTER = os.path.join(BASE_DIR, "data/master/team_master.csv")
DEPTH_CHART_DIR = os.path.join(BASE_DIR, "data/master/depth_charts")
OUTPUT_FILE = os.path.join(BASE_DIR, "config/mapping/qb_map_nfl.csv")

OUTPUT_HEADERS = [
    "sport",
    "league",
    "player_id",
    "qb_name",
    "team_abbr",
    "depth_chart_rank",
    "starter_flag",
    "backup_flag",
    "injury",
    "position_abb",
    "position.id",
    "team_id",
]


def load_team_master(path):
    """Return dict keyed by team_id -> team_abbr."""
    team_map = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            team_map[row["team_id"]] = row["team_abbr"]
    return team_map


def load_depth_chart(team_abbr):
    """Return dict keyed by player_id -> depth chart row for a given team_abbr.
    Returns empty dict if file does not exist."""
    path = os.path.join(DEPTH_CHART_DIR, team_abbr, f"{team_abbr}_depth.csv")
    depth_map = {}
    if not os.path.isfile(path):
        return depth_map
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            depth_map[row["player_id"]] = row
    return depth_map


def main():
    team_map = load_team_master(TEAM_MASTER)

    depth_chart_cache = {}

    rows_out = []

    with open(ROSTER_MASTER, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("position.id") != "8":
                continue

            player_id = row.get("id", "")
            team_id = row.get("team_id", "")
            team_abbr = team_map.get(team_id, "")

            depth_row = {}
            if team_abbr:
                if team_abbr not in depth_chart_cache:
                    depth_chart_cache[team_abbr] = load_depth_chart(team_abbr)
                depth_row = depth_chart_cache[team_abbr].get(player_id, {})

            out_row = {
                "sport": "football",
                "league": "nfl",
                "player_id": player_id,
                "qb_name": row.get("displayName", ""),
                "team_abbr": team_abbr,
                "depth_chart_rank": depth_row.get("depth_chart_rank", ""),
                "starter_flag": depth_row.get("starter_flag", ""),
                "backup_flag": depth_row.get("backup_flag", ""),
                "injury": depth_row.get("injury", ""),
                "position_abb": depth_row.get("position_abb", ""),
                "position.id": "8",
                "team_id": team_id,
            }
            rows_out.append(out_row)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADERS)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

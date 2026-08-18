"""
depth_cleanup.py

Reads raw_depth.csv (one row per team, fully flattened depth chart JSON)
and writes one cleaned depth chart CSV per team.

Input:
    docs/win/football/nfl/data/raw/raw_depth.csv

Output:
    docs/win/football/nfl/data/master/depth_charts/{team.abbreviation}/{team.abbreviation}_depth.csv
"""

import csv
import os
import re

INPUT_PATH = "docs/win/football/nfl/data/raw/raw_depth.csv"
OUTPUT_ROOT = "docs/win/football/nfl/data/master/depth_charts"

OUT_HEADER = [
    "sport",
    "league",
    "player_id",
    "name",
    "team",
    "position_abb",
    "position",
    "injury",
    "depth_chart_rank",
    "starter_flag",
    "backup_flag",
    "team_id",
    "season",
    "guid",
    "uid",
]

# Matches: depthchart.<n>.positions.<pos>.athletes.<idx>.id
ATHLETE_ID_PATTERN = re.compile(
    r"^depthchart\.(\d+)\.positions\.([a-z0-9]+)\.athletes\.(\d+)\.id$"
)


def main():
    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames

    athlete_id_cols = [c for c in header if ATHLETE_ID_PATTERN.match(c)]

    output_rows_by_team = {}

    for row in rows:
        team_abbr = row.get("team.abbreviation", "")
        team_id = row.get("team_id", "")
        season = row.get("season.year", "")

        team_out_rows = output_rows_by_team.setdefault(team_abbr, [])

        for id_col in athlete_id_cols:
            player_id = row[id_col]
            if not player_id:
                continue

            match = ATHLETE_ID_PATTERN.match(id_col)
            depth_num, position_key, athlete_idx = match.groups()
            prefix = f"depthchart.{depth_num}.positions.{position_key}"

            name = row.get(f"{prefix}.athletes.{athlete_idx}.displayName", "")
            guid = row.get(f"{prefix}.athletes.{athlete_idx}.guid", "")
            uid = row.get(f"{prefix}.athletes.{athlete_idx}.uid", "")
            position_abb = row.get(f"{prefix}.position.abbreviation", "")
            position_display = row.get(f"{prefix}.position.displayName", "")

            injury = ""
            injury_idx = 0
            while True:
                injury_col = f"{prefix}.athletes.{athlete_idx}.injuries.{injury_idx}.status"
                if injury_col not in row:
                    break
                if row[injury_col]:
                    injury = row[injury_col]
                    break
                injury_idx += 1
            if not injury:
                injury = "healthy"

            rank = int(athlete_idx) + 1
            starter_flag = 1 if rank == 1 else 0
            backup_flag = 1 if rank > 1 else 0

            team_out_rows.append({
                "sport": "football",
                "league": "nfl",
                "player_id": player_id,
                "name": name,
                "team": team_abbr,
                "position_abb": position_abb,
                "position": position_display,
                "injury": injury,
                "depth_chart_rank": rank,
                "starter_flag": starter_flag,
                "backup_flag": backup_flag,
                "team_id": team_id,
                "season": season,
                "guid": guid,
                "uid": uid,
            })

    total_rows_written = 0
    for team_abbr, out_rows in output_rows_by_team.items():
        if not team_abbr:
            continue
        team_folder = os.path.join(OUTPUT_ROOT, team_abbr)
        os.makedirs(team_folder, exist_ok=True)
        out_path = os.path.join(team_folder, f"{team_abbr}_depth.csv")

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUT_HEADER)
            writer.writeheader()
            writer.writerows(out_rows)

        total_rows_written += len(out_rows)
        print(f"team={team_abbr} rows={len(out_rows)} output={out_path}")

    print(f"total_rows_written={total_rows_written}")


if __name__ == "__main__":
    main()

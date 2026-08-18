#!/usr/bin/env python3
"""
pull_qb_stats.py

Builds docs/win/football/nfl/00_intake/qb/{season}_qb_stats.csv

Aggregates play-by-play data into per season/week/team/QB stats.

Order of operations:
1. Read PBP file.
2. Build dropback rows (qb_dropback == 1) and pass-attempt rows (pass_attempt == 1).
3. Group both by season, week, team (posteam), player_id (passer_player_id),
   qb_name (passer_player_name).
4. Compute stats:
   - dropbacks = count of dropback rows
   - epa_per_play = mean(qb_epa) over dropback rows
   - cpoe = mean(cpoe) over pass-attempt rows
   - air_yards = mean(air_yards) over pass-attempt rows
   - sack_rate = sum(sack) / dropbacks
   - interception_rate = sum(interception) / dropbacks
   - fumble_rate = count of dropback rows where fumbled_1_player_id == passer_player_id,
     divided by dropbacks
5. Leave starts, adjusted_completion_pct, pressure_to_sack_rate,
   turnover_worthy_play_rate blank.

Manual run only.
"""

import glob
import os
import pandas as pd

BASE_DIR = "docs/win/football/nfl"
INPUT_DIR = os.path.join(BASE_DIR, "00_intake/pbp")
OUTPUT_DIR = os.path.join(BASE_DIR, "00_intake/qb")

GROUP_COLS = ["season", "week", "posteam", "passer_player_id", "passer_player_name"]

OUTPUT_HEADERS = [
    "sport",
    "league",
    "season",
    "week",
    "team",
    "player_id",
    "qb_name",
    "starts",
    "dropbacks",
    "epa_per_play",
    "cpoe",
    "air_yards",
    "adjusted_completion_pct",
    "sack_rate",
    "pressure_to_sack_rate",
    "turnover_worthy_play_rate",
    "interception_rate",
    "fumble_rate",
]

def main():
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_pbp.csv.gz")))

    df_list = []
    for f in input_files:
        if os.path.getsize(f) == 0:
            print(f"WARNING: {f} is empty (0 bytes) — skipping.")
            continue
        try:
            df_list.append(pd.read_csv(f, compression="gzip", low_memory=False))
        except pd.errors.EmptyDataError:
            print(f"WARNING: {f} has no data rows — skipping.")
            continue

    if not df_list:
        print("WARNING: No valid PBP files found. Exiting.")
        return

    df = pd.concat(df_list, ignore_index=True)

    dropback_df = df[df["qb_dropback"] == 1].copy()
    pass_df = df[df["pass_attempt"] == 1].copy()

    # dropbacks, epa_per_play, sacks, interceptions
    dropback_agg = dropback_df.groupby(GROUP_COLS).agg(
        dropbacks=("qb_dropback", "size"),
        epa_per_play=("qb_epa", "mean"),
        sacks=("sack", "sum"),
        interceptions=("interception", "sum"),
    ).reset_index()

    # fumbles where the fumbler is the passer
    fumble_rows = dropback_df[
        dropback_df["fumbled_1_player_id"] == dropback_df["passer_player_id"]
    ]
    fumble_agg = fumble_rows.groupby(GROUP_COLS).size().reset_index(name="fumbles")

    # cpoe, air_yards from pass-attempt rows
    pass_agg = pass_df.groupby(GROUP_COLS).agg(
        cpoe=("cpoe", "mean"),
        air_yards=("air_yards", "mean"),
    ).reset_index()

    merged = dropback_agg.merge(fumble_agg, on=GROUP_COLS, how="left")
    merged = merged.merge(pass_agg, on=GROUP_COLS, how="left")

    merged["fumbles"] = merged["fumbles"].fillna(0)

    merged["sack_rate"] = merged["sacks"] / merged["dropbacks"]
    merged["interception_rate"] = merged["interceptions"] / merged["dropbacks"]
    merged["fumble_rate"] = merged["fumbles"] / merged["dropbacks"]

    merged["sport"] = "football"
    merged["league"] = "nfl"
    merged["starts"] = ""
    merged["adjusted_completion_pct"] = ""
    merged["pressure_to_sack_rate"] = ""
    merged["turnover_worthy_play_rate"] = ""

    merged = merged.rename(columns={
        "posteam": "team",
        "passer_player_id": "player_id",
        "passer_player_name": "qb_name",
    })

    merged = merged[OUTPUT_HEADERS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for season, season_df in merged.groupby("season"):
        output_file = os.path.join(OUTPUT_DIR, f"{season}_qb_stats.csv")
        season_df.to_csv(output_file, index=False)
        print(f"Wrote {len(season_df)} rows to {output_file}")


if __name__ == "__main__":
    main()

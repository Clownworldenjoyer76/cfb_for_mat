#!/usr/bin/env python3
"""
Clean CFBD scores to ensure one row per game ID, for safe merging in regression.

Input  : data/game_scores.csv
         (columns from CFBD /games: id,season,week,home_team,away_team,home_points,away_points)
Output : data/game_scores_clean.csv
         - only finished games (numeric scores)
         - deduplicated by id (keep the first complete record)
         - stable, deterministic ordering
"""

from pathlib import Path
import pandas as pd

IN_CSV  = Path("data/game_scores.csv")
OUT_CSV = Path("data/game_scores_clean.csv")

def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"{IN_CSV} not found. Run the fetch step first.")

    df = pd.read_csv(IN_CSV)

    # Keep only rows with numeric scores (finished games)
    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")
    df = df.dropna(subset=["home_points", "away_points"])

    # Basic sanity trims
    keep_cols = ["id","season","week","home_team","away_team","home_points","away_points"]
    df = df[keep_cols].copy()

    # Deduplicate by game id (CFBD ids should be unique; if duplicates exist, keep the first)
    before = len(df)
    df = df.sort_values(["season","week","id"]).drop_duplicates(subset=["id"], keep="first")
    after = len(df)

    # Write out
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"Cleaned scores: input_rows={before}, unique_game_ids={after}, wrote={OUT_CSV}")

if __name__ == "__main__":
    main()

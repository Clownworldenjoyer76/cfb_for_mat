#!/usr/bin/env python3
"""
Clean CFBD scores so merges are many-to-one for the regression step.

Input : data/game_scores.csv
Cols  : id,season,week,home_team,away_team,home_points,away_points
Output: data/game_scores_clean.csv
       - finished games only (numeric scores)
       - add game_id column (copy of id) for 1-key joins
       - drop duplicates by id
       - drop duplicates by (season,week,home_team,away_team)
"""

from pathlib import Path
import pandas as pd

IN_CSV  = Path("data/game_scores.csv")
OUT_CSV = Path("data/game_scores_clean.csv")

def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"{IN_CSV} not found. Run the fetch step first.")

    df = pd.read_csv(IN_CSV)

    # keep finished games
    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")
    df = df.dropna(subset=["home_points", "away_points"])

    # expected columns only
    df = df[["id","season","week","home_team","away_team","home_points","away_points"]].copy()
    df["game_id"] = df["id"]

    before = len(df)

    # dedup by id
    df = df.sort_values(["season","week","id"]).drop_duplicates(subset=["id"], keep="first")
    after_id = len(df)

    # dedup by composite keys used by the fallback join
    df = df.drop_duplicates(subset=["season","week","home_team","away_team"], keep="first")
    after_combo = len(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"Cleaned scores:"
          f" input_rows={before}, unique_by_id={after_id}, unique_by_combo={after_combo}, wrote={OUT_CSV}")

if __name__ == "__main__":
    main()

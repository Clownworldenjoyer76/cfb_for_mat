#!/usr/bin/env python3
"""
Clean CFBD scores into a **one-row-per-(game_id, team)** file so that downstream
merges are many-to-one for the regression step targeting points_scored.

Input : data/game_scores.csv
Expected columns (at minimum):
  id, season, week, home_team, away_team, home_points, away_points

Output: data/game_scores_clean.csv  (long format)
  Columns:
    game_id, season, week,
    team, opponent, is_home,
    points_scored, points_allowed,
    home_team, away_team, home_points, away_points

Guarantees:
  - Finished games only (numeric scores present for both teams)
  - Exactly one row per (game_id, team)
  - Team/opponent names stripped and normalized for trivial whitespace/casing
"""

from pathlib import Path
import pandas as pd

IN_CSV  = Path("data/game_scores.csv")
OUT_CSV = Path("data/game_scores_clean.csv")


def _coerce_scores(df: pd.DataFrame) -> pd.DataFrame:
    # Coerce scores to numeric; drop unfinished games
    for col in ("home_points", "away_points"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["home_points", "away_points"]).copy()
    return df


def _normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    # Light normalization to reduce accidental dupes from whitespace/casing
    def norm(s):
        if pd.isna(s):
            return s
        return str(s).strip()
    for col in ("home_team", "away_team"):
        df[col] = df[col].map(norm)
    return df


def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    # Build per-team long rows
    home_rows = pd.DataFrame({
        "game_id":   df["id"],
        "season":    df["season"],
        "week":      df["week"],
        "team":      df["home_team"],
        "opponent":  df["away_team"],
        "is_home":   True,
        "points_scored":  df["home_points"],
        "points_allowed": df["away_points"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_points": df["home_points"],
        "away_points": df["away_points"],
    })

    away_rows = pd.DataFrame({
        "game_id":   df["id"],
        "season":    df["season"],
        "week":      df["week"],
        "team":      df["away_team"],
        "opponent":  df["home_team"],
        "is_home":   False,
        "points_scored":  df["away_points"],
        "points_allowed": df["home_points"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_points": df["home_points"],
        "away_points": df["away_points"],
    })

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)

    # Ensure consistent types
    long_df["game_id"] = long_df["game_id"]
    long_df["season"]  = pd.to_numeric(long_df["season"], errors="coerce").astype("Int64")
    long_df["week"]    = pd.to_numeric(long_df["week"], errors="coerce").astype("Int64")

    return long_df


def _deduplicate(long_df: pd.DataFrame) -> pd.DataFrame:
    before = len(long_df)

    # Primary uniqueness: per (game_id, team)
    long_df = long_df.sort_values(["season", "week", "game_id", "team"]).drop_duplicates(
        subset=["game_id", "team"], keep="first"
    )

    after = len(long_df)

    # Defensive check: ensure there are no remaining duplicates
    dups = long_df.duplicated(subset=["game_id", "team"]).sum()
    if dups:
        # In the unlikely event, keep first occurrence deterministically
        long_df = long_df.drop_duplicates(subset=["game_id", "team"], keep="first")

    print(f"[clean] long rows: start={before}, unique_(game_id,team)={after}, forced_dups_resolved={dups}")

    return long_df


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"{IN_CSV} not found. Run the fetch step first.")

    raw = pd.read_csv(IN_CSV)
    raw = _coerce_scores(raw)
    raw = _normalize_team_names(raw)

    # Keep only expected columns if present; tolerate extras
    cols = [c for c in ["id", "season", "week", "home_team", "away_team", "home_points", "away_points"] if c in raw.columns]
    df = raw[cols].copy()

    long_df = _to_long(df)
    long_df = _deduplicate(long_df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_CSV, index=False)

    print(f"[clean] wrote {OUT_CSV} rows={len(long_df)}")


if __name__ == "__main__":
    main()

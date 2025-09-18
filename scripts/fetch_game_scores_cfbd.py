#!/usr/bin/env python3
"""
Fetch per-game scores for all seasons present in data/modeling_dataset.csv
and write a normalized CSV at data/game_scores.csv with columns:
  id,season,week,home_team,away_team,home_points,away_points

Source: CollegeFootballData API /games.
Requires env CFBD_API_KEY (GitHub Actions secret).
Exit codes:
  0  success
  2  missing CFBD_API_KEY
  3  modeling_dataset.csv missing/invalid
  4  API error or empty results
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests

INPUT_DATASET = Path("data/modeling_dataset.csv")
OUT_CSV       = Path("data/game_scores.csv")
API_BASE      = "https://api.collegefootballdata.com/games"

def die(code: int, msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def seasons_from_modeling_dataset() -> List[int]:
    if not INPUT_DATASET.exists():
        die(3, f"{INPUT_DATASET} not found")
    try:
        df = pd.read_csv(INPUT_DATASET, usecols=["season"])
    except Exception as e:
        die(3, f"Failed reading {INPUT_DATASET}: {e}")
    if "season" not in df.columns or df.empty:
        die(3, f"{INPUT_DATASET} missing 'season' column or no rows")
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    if not seasons:
        die(3, "No seasons found in modeling_dataset.csv")
    return seasons

def fetch_games_for_season(session: requests.Session, season: int):
    all_rows = []
    # Regular + Postseason to be safe
    for season_type in ("regular", "postseason"):
        r = session.get(API_BASE, params={"year": season, "seasonType": season_type}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"/games {season} {season_type} -> {r.status_code}: {r.text[:300]}")
        rows = r.json() or []
        all_rows.extend(rows)
        time.sleep(0.25)  # gentle rate limiting
    return all_rows

def normalize_rows(rows: List[Dict]) -> pd.DataFrame:
    # Keep only the columns you approved
    out = []
    for g in rows:
        out.append({
            "id": g.get("id"),
            "season": g.get("season"),
            "week": g.get("week"),
            "home_team": g.get("home_team"),
            "away_team": g.get("away_team"),
            "home_points": g.get("home_points"),
            "away_points": g.get("away_points"),
        })
    df = pd.DataFrame(out)
    # Only finished games with numeric scores
    df = df.dropna(subset=["home_points", "away_points"])
    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")
    df = df.dropna(subset=["home_points", "away_points"])
    # Final column order
    return df[["id","season","week","home_team","away_team","home_points","away_points"]].reset_index(drop=True)

def main():
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        die(2, "CFBD_API_KEY env var is required (add it as a GitHub Actions secret).")

    seasons = seasons_from_modeling_dataset()
    print(f"Fetching scores for seasons: {seasons}")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}"})

    all_frames = []
    for yr in seasons:
        try:
            rows = fetch_games_for_season(session, yr)
            df = normalize_rows(rows)
        except Exception as e:
            die(4, f"API error for season {yr}: {e}")
        if df.empty:
            print(f"WARNING: no finished games found for season {yr}")
        all_frames.append(df)

    if not any(len(df) for df in all_frames):
        die(4, "No game scores retrieved for any season.")

    out = pd.concat(all_frames, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()

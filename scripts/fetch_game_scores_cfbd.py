#!/usr/bin/env python3
"""
Fetch per-game scores for all seasons present in data/modeling_dataset.csv
and write data/game_scores.csv with columns:
  id,season,week,home_team,away_team,home_points,away_points

Source: CollegeFootballData /games endpoint.
Requires env CFBD_API_KEY (GitHub Actions secret).

Improvements:
- Per-week fetching (regular + postseason) so finished games are captured during an active season.
- Force division=fbs to match typical modeling datasets.
- Robust retries with backoff and longer timeouts.
- Verbose counts per request and per season.

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

def fetch_with_retries(session: requests.Session, params: dict, max_retries: int = 3, timeout: int = 90):
    """Try API call multiple times with backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(API_BASE, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json() or []
            else:
                print(f"WARNING: API {params} -> {r.status_code}")
        except requests.RequestException as e:
            print(f"WARNING: API {params} failed (attempt {attempt}/{max_retries}): {e}")
        if attempt < max_retries:
            sleep_time = attempt * 5
            print(f"Retrying after {sleep_time}s...")
            time.sleep(sleep_time)
    raise RuntimeError(f"API request failed after {max_retries} retries for params={params}")

def fetch_games_for_season(session: requests.Session, season: int) -> List[Dict]:
    rows: List[Dict] = []

    # Regular season: iterate weeks 1..20 (some years use 0/Week 0; include it)
    regular_weeks = list(range(0, 21))
    for w in regular_weeks:
        params = {"year": season, "seasonType": "regular", "week": w, "division": "fbs"}
        chunk = fetch_with_retries(session, params)
        if chunk:
            rows.extend(chunk)
            # Print a short count to the CI logs
            finished = sum(1 for g in chunk if g.get("home_points") is not None and g.get("away_points") is not None)
            print(f"season={season} regular week={w}: got {len(chunk)} (finished={finished})")
        time.sleep(0.15)  # gentle rate limiting

    # Postseason (no week range guarantees, but 1..6 is safe enough)
    for w in range(1, 7):
        params = {"year": season, "seasonType": "postseason", "week": w, "division": "fbs"}
        chunk = fetch_with_retries(session, params)
        if chunk:
            rows.extend(chunk)
            finished = sum(1 for g in chunk if g.get("home_points") is not None and g.get("away_points") is not None)
            print(f"season={season} postseason week={w}: got {len(chunk)} (finished={finished})")
        time.sleep(0.15)

    return rows

def normalize_rows(rows: List[Dict]) -> pd.DataFrame:
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
    # Keep only finished games with numeric scores
    df = df.dropna(subset=["home_points", "away_points"])
    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")
    df = df.dropna(subset=["home_points", "away_points"])
    return df[["id","season","week","home_team","away_team","home_points","away_points"]].reset_index(drop=True)

def main():
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        die(2, "CFBD_API_KEY env var is required (add it as a GitHub Actions secret).")

    seasons = seasons_from_modeling_dataset()
    print(f"Fetching scores for seasons: {seasons}")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}"})

    season_frames = []
    for yr in seasons:
        try:
            rows = fetch_games_for_season(session, yr)
            df = normalize_rows(rows)
            print(f"season={yr}: finished_games={len(df)}")
        except Exception as e:
            die(4, f"API error for season {yr}: {e}")
        if df.empty:
            print(f"WARNING: no finished FBS games found for season {yr}")
        season_frames.append(df)

    all_len = sum(len(f) for f in season_frames)
    if all_len == 0:
        die(4, "No game scores retrieved for any season.")

    out = pd.concat(season_frames, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()

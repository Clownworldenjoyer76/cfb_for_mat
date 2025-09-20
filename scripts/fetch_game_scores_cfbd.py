#!/usr/bin/env python3
"""
Fetch per-game scores for all seasons present in data/modeling_dataset.csv
and write data/game_scores.csv with columns:
  id,season,week,home_team,away_team,home_points,away_points

Source: CollegeFootballData /games endpoint.
Requires env CFBD_API_KEY (GitHub Actions secret).

Behavior:
- Per-week fetching (regular + postseason) with division=fbs.
- Retries with backoff and longer timeouts.
- Verbose counts per request and per season.
- If no finished games are found, dump a RAW SAMPLE of the API response for
  the first such week to:
    data/debug_cfbd_week_sample.json
  and print the first few items to the CI log so you can see the actual fields.

Exit codes:
  0  success
  2  missing CFBD_API_KEY
  3  modeling_dataset.csv missing/invalid
  4  API error or empty results
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import requests

INPUT_DATASET = Path("data/modeling_dataset.csv")
OUT_CSV       = Path("data/game_scores.csv")
DEBUG_JSON    = Path("data/debug_cfbd_week_sample.json")
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
                print(f"WARNING: API {params} -> {r.status_code} body={r.text[:200]}")
        except requests.RequestException as e:
            print(f"WARNING: API {params} failed (attempt {attempt}/{max_retries}): {e}")
        if attempt < max_retries:
            sleep_time = attempt * 5
            print(f"Retrying after {sleep_time}s...")
            time.sleep(sleep_time)
    raise RuntimeError(f"API request failed after {max_retries} retries for params={params}")

def fetch_games_for_week(session: requests.Session, season: int, season_type: str, week: int) -> List[Dict]:
    params = {"year": season, "seasonType": season_type, "week": week, "division": "fbs"}
    return fetch_with_retries(session, params)

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

def maybe_dump_raw_sample(rows: List[Dict], season: int, season_type: str, week: int, already_dumped: bool) -> bool:
    """
    If we haven't dumped a sample yet, write the raw payload for this week to DEBUG_JSON
    and print a small preview to the log. Returns True if we dumped.
    """
    if already_dumped:
        return True
    try:
        DEBUG_JSON.parent.mkdir(parents=True, exist_ok=True)
        sample_obj = {
            "season": season,
            "season_type": season_type,
            "week": week,
            "count": len(rows),
            "sample_first_5": rows[:5],
        }
        with DEBUG_JSON.open("w", encoding="utf-8") as f:
            json.dump(sample_obj, f, ensure_ascii=False, indent=2)
        print("---- CFBD RAW SAMPLE (first 5 rows) ----")
        print(json.dumps(sample_obj, ensure_ascii=False, indent=2)[:4000])  # cap output
        print(f"Raw sample saved to {DEBUG_JSON}")
        return True
    except Exception as e:
        print(f"WARNING: failed to write {DEBUG_JSON}: {e}")
        return True  # don't keep trying
    # not reached

def main():
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        die(2, "CFBD_API_KEY env var is required (add it as a GitHub Actions secret).")

    seasons = seasons_from_modeling_dataset()
    print(f"Fetching scores for seasons: {seasons}")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}"})

    season_frames: List[pd.DataFrame] = []
    dumped = False  # whether we dumped a raw sample yet

    for yr in seasons:
        rows_all: List[Dict] = []

        # Regular season weeks 0..20
        for w in range(0, 21):
            chunk = fetch_games_for_week(session, yr, "regular", w)
            rows_all.extend(chunk)
            finished = sum(1 for g in chunk if g.get("home_points") is not None and g.get("away_points") is not None)
            print(f"season={yr} regular week={w}: got {len(chunk)} (finished={finished})")
            if finished == 0 and len(chunk) > 0 and not dumped:
                dumped = maybe_dump_raw_sample(chunk, yr, "regular", w, dumped)
            time.sleep(0.15)

        # Postseason weeks 1..6
        for w in range(1, 7):
            chunk = fetch_games_for_week(session, yr, "postseason", w)
            rows_all.extend(chunk)
            finished = sum(1 for g in chunk if g.get("home_points") is not None and g.get("away_points") is not None)
            print(f"season={yr} postseason week={w}: got {len(chunk)} (finished={finished})")
            if finished == 0 and len(chunk) > 0 and not dumped:
                dumped = maybe_dump_raw_sample(chunk, yr, "postseason", w, dumped)
            time.sleep(0.15)

        df = normalize_rows(rows_all)
        print(f"season={yr}: finished_games={len(df)}")
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

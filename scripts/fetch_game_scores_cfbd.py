#!/usr/bin/env python3
"""
Fetch CFBD game scores for ALL (season, week) pairs present in data/modeling_dataset.csv,
so that every (game_id, team) used by the modeling dataset can be matched.

Output: data/game_scores.csv with columns:
  id, season, week, home_team, away_team, home_points, away_points

Requires env:
  CFBD_API_KEY
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import requests
import pandas as pd

MODEL_CSV  = Path("data/modeling_dataset.csv")
OUT_CSV    = Path("data/game_scores.csv")

CFBD_API_KEY = os.environ.get("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com/games"

# polite rate-limit for CFBD (keep it light)
SLEEP_SEC = 0.25

def die(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def cfbd_get(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"} if CFBD_API_KEY else {}
    r = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()  # list of games

def normalize_games_to_rows(games: List[Dict[str, Any]], season: int, week: int) -> List[Dict[str, Any]]:
    rows = []
    for g in games:
        # Ids & names vary by endpoint fields; handle common keys safely
        gid = g.get("id", g.get("game_id", None))
        ht  = g.get("home_team") or g.get("homeTeam")
        at  = g.get("away_team") or g.get("awayTeam")
        hp  = g.get("home_points", g.get("homePoints"))
        ap  = g.get("away_points", g.get("awayPoints"))

        rows.append({
            "id": gid,
            "season": season,
            "week": week,
            "home_team": ht,
            "away_team": at,
            "home_points": hp,
            "away_points": ap,
        })
    return rows

def main():
    if not MODEL_CSV.exists():
        die(f"{MODEL_CSV} not found. Run upstream steps that produce modeling_dataset.csv first.")

    # Derive the full set of (season, week) present in the modeling data
    model = pd.read_csv(MODEL_CSV, usecols=lambda c: c in {"season","week"})
    if not {"season","week"}.issubset(model.columns):
        die("modeling_dataset.csv must include 'season' and 'week' columns.")

    model["season"] = pd.to_numeric(model["season"], errors="coerce").astype("Int64")
    model["week"]   = pd.to_numeric(model["week"], errors="coerce").astype("Int64")
    sw = (
        model.dropna(subset=["season","week"])
             .drop_duplicates(subset=["season","week"])
             .sort_values(["season","week"])
    )

    if sw.empty:
        die("No (season, week) pairs found in modeling_dataset.csv.")

    if not CFBD_API_KEY:
        print("WARNING: CFBD_API_KEY not set — requests may be rate-limited or rejected.", file=sys.stderr)

    all_rows: List[Dict[str, Any]] = []
    for _, row in sw.iterrows():
        season = int(row["season"])
        week   = int(row["week"])

        # query regular + postseason defensively (some datasets encode bowls/champ games)
        for season_type in ("regular", "postseason"):
            params = {"year": season, "week": week, "seasonType": season_type}
            try:
                games = cfbd_get(params)
            except requests.HTTPError as e:
                # continue but log; we still want whatever we can fetch
                print(f"[warn] CFBD {season} wk{week} type={season_type}: {e}", file=sys.stderr)
                games = []
            rows = normalize_games_to_rows(games, season, week)
            all_rows.extend(rows)
            time.sleep(SLEEP_SEC)

    # Write CSV
    df = pd.DataFrame(all_rows)

    # Keep only expected columns and ensure types
    cols = ["id","season","week","home_team","away_team","home_points","away_points"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    df = df[cols].copy()

    # numeric coercion for finished games (cleaner will drop NAs later)
    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"[fetch] wrote {OUT_CSV} rows={len(df)} from {len(sw)} (season,week) pairs")

if __name__ == "__main__":
    main()

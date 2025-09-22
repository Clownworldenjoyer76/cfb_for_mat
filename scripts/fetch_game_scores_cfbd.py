#!/usr/bin/env python3
"""
Fetch CFBD game scores comprehensively:

1) Pull all (season, week) pairs present in data/modeling_dataset.csv
   - For both seasonType in {regular, postseason}
   - For both division in {fbs, fcs}
2) If available, read data/diagnostics/unmatched_missing_game_id.csv
   and backfill those game_ids directly via /games?id=<gid> (and fall back to gameId=<gid>)
3) Write a single unified CSV at data/game_scores.csv

Output columns:
  id, season, week, home_team, away_team, home_points, away_points

Requires:
  CFBD_API_KEY  (recommended)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import requests

MODEL_CSV   = Path("data/modeling_dataset.csv")
OUT_CSV     = Path("data/game_scores.csv")
DIAG_MISS   = Path("data/diagnostics/unmatched_missing_game_id.csv")

CFBD_API_KEY = os.environ.get("CFBD_API_KEY")
BASE_URL     = "https://api.collegefootballdata.com/games"

SLEEP_SEC = 0.25  # polite rate limit

def die(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def cfbd_get(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"} if CFBD_API_KEY else {}
    r = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        # Some proxies might return dict with 'games' key; normalize to list
        data = data.get("games", [])
    return data

def normalize_games_to_rows(games: List[Dict[str, Any]], season: int | None, week: int | None) -> List[Dict[str, Any]]:
    rows = []
    for g in games:
        gid = g.get("id", g.get("game_id", None))
        ht  = g.get("home_team") or g.get("homeTeam")
        at  = g.get("away_team") or g.get("awayTeam")
        hp  = g.get("home_points", g.get("homePoints"))
        ap  = g.get("away_points", g.get("awayPoints"))
        # If the API returns season/week per-game, prefer those; else fall back to our request values
        s   = g.get("season", season)
        w   = g.get("week", week)
        rows.append({
            "id": gid,
            "season": s,
            "week": w,
            "home_team": ht,
            "away_team": at,
            "home_points": hp,
            "away_points": ap,
        })
    return rows

def fetch_by_season_week(sw: pd.DataFrame) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for _, row in sw.iterrows():
        season = int(row["season"])
        week   = int(row["week"])
        for season_type in ("regular", "postseason"):
            for division in ("fbs", "fcs"):
                params = {"year": season, "week": week, "seasonType": season_type, "division": division}
                try:
                    games = cfbd_get(params)
                except requests.HTTPError as e:
                    print(f"[warn] CFBD year={season} week={week} type={season_type} div={division}: {e}", file=sys.stderr)
                    games = []
                all_rows.extend(normalize_games_to_rows(games, season, week))
                time.sleep(SLEEP_SEC)
    return all_rows

def fetch_by_game_ids(game_ids: List[int | str]) -> List[Dict[str, Any]]:
    """Backfill exact game ids. Try id=<gid>, else gameId=<gid> if first attempt returns empty."""
    rows: List[Dict[str, Any]] = []
    for gid in game_ids:
        # try id
        params = {"id": str(gid)}
        try:
            games = cfbd_get(params)
        except requests.HTTPError as e:
            print(f"[warn] CFBD id={gid} (param 'id') HTTPError: {e}", file=sys.stderr)
            games = []
        if not games:
            # try gameId (some clients refer to it this way)
            params2 = {"gameId": str(gid)}
            try:
                games = cfbd_get(params2)
            except requests.HTTPError as e:
                print(f"[warn] CFBD id={gid} (param 'gameId') HTTPError: {e}", file=sys.stderr)
                games = []
        if games:
            # When fetching by id, season/week may be in payload; pass None to avoid overwriting with wrong week
            rows.extend(normalize_games_to_rows(games, season=None, week=None))
        else:
            print(f"[info] CFBD returned no record for game_id={gid}", file=sys.stderr)
        time.sleep(SLEEP_SEC)
    return rows

def main():
    if not MODEL_CSV.exists():
        die(f"{MODEL_CSV} not found. Upstream modeling dataset required.")

    # Build (season, week) list from modeling dataset
    model = pd.read_csv(MODEL_CSV, usecols=lambda c: c in {"season","week"})
    if not {"season","week"}.issubset(model.columns):
        die("modeling_dataset.csv must include 'season' and 'week' columns.")
    model["season"] = pd.to_numeric(model["season"], errors="coerce").astype("Int64")
    model["week"]   = pd.to_numeric(model["week"], errors="coerce").astype("Int64")
    sw = (model.dropna(subset=["season","week"])
                .drop_duplicates(subset=["season","week"])
                .sort_values(["season","week"]))

    # 1) Fetch by (season, week) across season types + divisions
    base_rows = fetch_by_season_week(sw)

    # 2) Backfill by missing game_ids if diagnostics file exists
    backfill_rows: List[Dict[str, Any]] = []
    if DIAG_MISS.exists():
        miss = pd.read_csv(DIAG_MISS)
        # Some CSVs may store game_id as float; force to str then int-like
        gids = (miss["game_id"].dropna().astype(str).str.replace(r"\.0$", "", regex=True).unique().tolist())
        # De-dup and only backfill those not already in base_rows
        have_ids = {r["id"] for r in base_rows if r.get("id") is not None}
        backfill_ids = [g for g in gids if g not in have_ids]
        print(f"[info] Backfilling {len(backfill_ids)} game_ids directly via CFBD…")
        backfill_rows = fetch_by_game_ids(backfill_ids)

    # Combine + normalize types
    all_rows = base_rows + backfill_rows
    df = pd.DataFrame(all_rows)

    # Ensure columns exist
    cols = ["id","season","week","home_team","away_team","home_points","away_points"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    df = df[cols].copy()

    # Coerce points to numeric
    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[fetch] wrote {OUT_CSV} rows={len(df)} (base={len(base_rows)} + backfill={len(backfill_rows)})")

if __name__ == "__main__":
    if not CFBD_API_KEY:
        print("WARNING: CFBD_API_KEY not set — requests may be rate-limited or incomplete.", file=sys.stderr)
    main()

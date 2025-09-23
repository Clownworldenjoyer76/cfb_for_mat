#!/usr/bin/env python3
"""
Fetch CFBD game scores comprehensively with BATCHED backfill.

What it does
------------
1) Builds the set of (season, week) pairs present in data/modeling_dataset.csv.
   - For each pair, fetches /games for BOTH seasonType in {regular, postseason}
     and BOTH division in {fbs, fcs}.
2) If diagnostics exist, reads data/diagnostics/unmatched_missing_game_id.csv
   and BATCHES the "backfill" by (season, week) for those game_ids as well.
   - If some rows have season but no week, it batches by (season) only.
   - If neither season nor week are known, it falls back to sparse per-ID fetch
     with retry/backoff (only for the stragglers).
3) Writes a unified CSV at data/game_scores.csv with:
     id, season, week, home_team, away_team, home_points, away_points

Environment
-----------
CFBD_API_KEY  (recommended to avoid harsh rate limits)

Output
------
data/game_scores.csv
and progress logs like:
  [fetch] wrote data/game_scores.csv rows=XXXXX (base=YYYY + batched_backfill=ZZZZ + per_id_backfill=KKK)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Iterable, Tuple

import pandas as pd
import requests

# ---- Paths ----
MODEL_CSV    = Path("data/modeling_dataset.csv")
OUT_CSV      = Path("data/game_scores.csv")
DIAG_MISS    = Path("data/diagnostics/unmatched_missing_game_id.csv")

# ---- CFBD API ----
CFBD_API_KEY = os.environ.get("CFBD_API_KEY", "")
BASE_URL     = "https://api.collegefootballdata.com/games"

# polite defaults
REQ_TIMEOUT_S = 30
SLEEP_BETWEEN_CALLS_S = 0.15
MAX_RETRIES = 5
BACKOFF_BASE_S = 0.8

# ---- Helpers ----
def log(msg: str) -> None:
    print(msg, flush=True)

def die(msg: str, code: int = 2) -> None:
    log(f"ERROR: {msg}")
    sys.exit(code)

def headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {CFBD_API_KEY}"} if CFBD_API_KEY else {}

def http_get(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """GET with retry/backoff for 429/5xx. Returns list of games."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(BASE_URL, headers=headers(), params=params, timeout=REQ_TIMEOUT_S)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.reason}")
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                data = data.get("games", [])
            return data if isinstance(data, list) else []
        except requests.HTTPError as e:
            last_err = e
            sleep_s = BACKOFF_BASE_S * (2 ** (attempt - 1))
            log(f"[warn] {e} for params={params} (attempt {attempt}/{MAX_RETRIES}); sleeping {sleep_s:.1f}s…")
            time.sleep(sleep_s)
        except Exception as e:
            last_err = e
            log(f"[warn] non-HTTP error for params={params}: {e} (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(0.5)
    log(f"[warn] giving up after {MAX_RETRIES} attempts; params={params} last_err={last_err}")
    return []

def norm_rows(games: List[Dict[str, Any]], season: int | None, week: int | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for g in games:
        gid = g.get("id", g.get("game_id"))
        ht  = g.get("home_team") or g.get("homeTeam")
        at  = g.get("away_team") or g.get("awayTeam")
        hp  = g.get("home_points", g.get("homePoints"))
        ap  = g.get("away_points", g.get("awayPoints"))
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

def fetch_batch_by_season_week(pairs: Iterable[Tuple[int,int]]) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for season, week in pairs:
        for season_type in ("regular", "postseason"):
            for division in ("fbs", "fcs"):
                params = {"year": int(season), "week": int(week),
                          "seasonType": season_type, "division": division}
                games = http_get(params)
                all_rows.extend(norm_rows(games, season, week))
                time.sleep(SLEEP_BETWEEN_CALLS_S)
    return all_rows

def fetch_batch_by_season(seasons: Iterable[int]) -> List[Dict[str, Any]]:
    """For rows where week is unknown, fetch whole seasons once per season_type/division."""
    all_rows: List[Dict[str, Any]] = []
    for season in seasons:
        for season_type in ("regular", "postseason"):
            for division in ("fbs", "fcs"):
                params = {"year": int(season), "seasonType": season_type, "division": division}
                games = http_get(params)
                all_rows.extend(norm_rows(games, season, None))
                time.sleep(SLEEP_BETWEEN_CALLS_S)
    return all_rows

def fetch_sparse_by_ids(game_ids: Iterable[str | int]) -> List[Dict[str, Any]]:
    """Only used for stubborn stragglers after batching."""
    rows: List[Dict[str, Any]] = []
    for gid in game_ids:
        gid_str = str(gid)
        params1 = {"id": gid_str}
        games = http_get(params1)
        if not games:
            params2 = {"gameId": gid_str}
            games = http_get(params2)
        if games:
            rows.extend(norm_rows(games, season=None, week=None))
        else:
            log(f"[info] no record found for game_id={gid_str} after retries")
        time.sleep(SLEEP_BETWEEN_CALLS_S)
    return rows

# ---- Main ----
def main() -> None:
    if not MODEL_CSV.exists():
        die(f"{MODEL_CSV} not found. Upstream modeling dataset required.")

    # Model: we want (season, week) for base pulls; also try to get game_id->(season,week)
    model = pd.read_csv(MODEL_CSV, usecols=lambda c: c in {"season","week","game_id"})
    if "season" not in model.columns or "week" not in model.columns:
        die("modeling_dataset.csv must include 'season' and 'week' columns.")

    model["season"] = pd.to_numeric(model["season"], errors="coerce").astype("Int64")
    model["week"]   = pd.to_numeric(model["week"], errors="coerce").astype("Int64")

    # Base (season, week) pairs for comprehensive pulls
    sw = (model.dropna(subset=["season","week"])
                .drop_duplicates(subset=["season","week"])
                .sort_values(["season","week"]))
    base_pairs = [(int(r.season), int(r.week)) for r in sw.itertuples(index=False)]

    # 1) Base pull: all season/week pairs across seasonType & division
    base_rows = fetch_batch_by_season_week(base_pairs)

    # 2) Batched backfill from diagnostics (if present)
    batched_backfill_rows: List[Dict[str, Any]] = []
    per_id_rows: List[Dict[str, Any]] = []

    if DIAG_MISS.exists():
        miss = pd.read_csv(DIAG_MISS)

        # Normalize id types to string (handle float-y CSVs)
        if "game_id" not in miss.columns:
            log("[info] unmatched_missing_game_id.csv lacked 'game_id' column; skipping backfill block.")
            miss = pd.DataFrame(columns=["game_id"])
        miss["game_id"] = miss["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)

        # Attach season/week to missing ids by merging with model (if model has game_id)
        if "game_id" in model.columns:
            # ensure comparable type
            model_gid_sw = (model.assign(game_id=model["game_id"].astype(str))
                                  [["game_id","season","week"]]
                                  .drop_duplicates())
            miss_sw = miss.merge(model_gid_sw, on="game_id", how="left")
        else:
            # No game_id in model; create empty season/week so we fall back gracefully
            miss_sw = miss.copy()
            if "season" not in miss_sw.columns: miss_sw["season"] = pd.Series(dtype="Int64")
            if "week"   not in miss_sw.columns: miss_sw["week"]   = pd.Series(dtype="Int64")

        # Prepare buckets safely even if cols are missing
        has_sw_cols = {"season","week"}.issubset(miss_sw.columns)

        if has_sw_cols:
            sw_known = miss_sw.dropna(subset=["season","week"])
            seasons_only = miss_sw[miss_sw["season"].notna() & miss_sw["week"].isna()]
        else:
            sw_known = pd.DataFrame(columns=["game_id","season","week"])
            seasons_only = pd.DataFrame(columns=["game_id","season","week"])

        # Unique groups
        batched_pairs = sorted({(int(s), int(w))
                                for s, w in sw_known[["season","week"]].dropna().itertuples(index=False, name=None)})
        batched_seasons = sorted({int(s) for s in seasons_only["season"].dropna().unique().tolist()})

        if batched_pairs:
            log(f"[info] Backfill batching by (season,week): {len(batched_pairs)} groups")
            batched_backfill_rows.extend(fetch_batch_by_season_week(batched_pairs))
        if batched_seasons:
            log(f"[info] Backfill batching by (season): {len(batched_seasons)} seasons")
            batched_backfill_rows.extend(fetch_batch_by_season(batched_seasons))

        # Determine which ids still missing after batched pulls
        have_ids = {r["id"] for r in (base_rows + batched_backfill_rows) if r.get("id") is not None}
        still_missing_ids = [gid for gid in miss_sw["game_id"].dropna().astype(str).unique().tolist()
                             if gid not in have_ids]

        # 3) Sparse per-ID fetch for stragglers only (with backoff)
        if still_missing_ids:
            log(f"[info] Sparse per-ID backfill for {len(still_missing_ids)} remaining ids (with retry/backoff)…")
            per_id_rows = fetch_sparse_by_ids(still_missing_ids)

    # Combine all rows and normalize types
    all_rows = base_rows + batched_backfill_rows + per_id_rows
    df = pd.DataFrame(all_rows)

    # Ensure columns exist
    cols = ["id","season","week","home_team","away_team","home_points","away_points"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    df = df[cols].copy()

    # Coerce numeric points
    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")

    # Write
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    log(f"[fetch] wrote {OUT_CSV} rows={len(df)} "
        f"(base={len(base_rows)} + batched_backfill={len(batched_backfill_rows)} + per_id_backfill={len(per_id_rows)})")

if __name__ == "__main__":
    if not CFBD_API_KEY:
        log("WARNING: CFBD_API_KEY not set — requests may be limited.")
    main()

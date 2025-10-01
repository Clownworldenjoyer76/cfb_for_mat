# scripts/fetch_game_scores_cfbd.py
#!/usr/bin/env python3
"""
CFBD score fetcher with batching + bounded per-ID backfill (resume/time-budgeted).

Env (optional):
  MAX_PER_ID_BACKFILL=100
  MAX_BACKFILL_MINUTES=20
  ENABLE_PER_ID_BACKFILL=1
  CFBD_API_KEY=...   # recommended

Outputs:
  data/game_scores.csv   (id, season, week, home_team, away_team, home_points, away_points)
"""

from __future__ import annotations

import os, sys, time
from pathlib import Path
from typing import Dict, Any, List, Iterable, Tuple
import pandas as pd
import requests

# Paths
MODEL_CSV = Path("data/modeling_dataset.csv")
OUT_CSV   = Path("data/game_scores.csv")
DIAG_MISS = Path("data/diagnostics/unmatched_missing_game_id.csv")

# API
CFBD_API_KEY = os.environ.get("CFBD_API_KEY", "")
BASE_URL = "https://api.collegefootballdata.com/games"

# Tunables
REQ_TIMEOUT_S = 25
SLEEP_BETWEEN_CALLS_S = 0.12
MAX_RETRIES = 3
BACKOFF_BASE_S = 0.7

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

MAX_PER_ID_BACKFILL  = _int_env("MAX_PER_ID_BACKFILL", 100)
MAX_BACKFILL_MINUTES = _int_env("MAX_BACKFILL_MINUTES", 20)
ENABLE_PER_ID        = _int_env("ENABLE_PER_ID_BACKFILL", 1) == 1

def log(msg: str): print(msg, flush=True)
def die(msg: str, code: int = 2): log(f"ERROR: {msg}"); sys.exit(code)
def headers() -> Dict[str,str]:
    return {"Authorization": f"Bearer {CFBD_API_KEY}"} if CFBD_API_KEY else {}

def http_get(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """GET with short retry/backoff; returns list of games."""
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
            log(f"[warn] {e} params={params} attempt {attempt}/{MAX_RETRIES}; sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
        except Exception as e:
            last_err = e
            log(f"[warn] non-HTTP error {e} params={params} attempt {attempt}/{MAX_RETRIES}")
            time.sleep(0.5)
    log(f"[warn] giving up after {MAX_RETRIES} attempts; params={params} last_err={last_err}")
    return []

def norm_rows(games: List[Dict[str, Any]], season: int | None, week: int | None) -> List[Dict[str, Any]]:
    rows = []
    for g in games:
        gid = g.get("id", g.get("game_id"))
        rows.append({
            "id": gid,
            "season": g.get("season", season),
            "week": g.get("week", week),
            "home_team": g.get("home_team") or g.get("homeTeam"),
            "away_team": g.get("away_team") or g.get("awayTeam"),
            "home_points": g.get("home_points", g.get("homePoints")),
            "away_points": g.get("away_points", g.get("awayPoints")),
        })
    return rows

# -------------------------
# IMPORTANT CHANGE: no 'division' filter
# -------------------------
def fetch_batch_by_season_week(pairs: Iterable[Tuple[int,int]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for season, week in pairs:
        for season_type in ("regular", "postseason"):
            params = {"year": int(season), "week": int(week), "seasonType": season_type}
            out.extend(norm_rows(http_get(params), season, week))
            time.sleep(SLEEP_BETWEEN_CALLS_S)
    return out

def fetch_batch_by_season(seasons: Iterable[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for season in seasons:
        for season_type in ("regular", "postseason"):
            params = {"year": int(season), "seasonType": season_type}
            out.extend(norm_rows(http_get(params), season, None))
            time.sleep(SLEEP_BETWEEN_CALLS_S)
    return out

def fetch_sparse_by_ids(game_ids: List[str], time_budget_s: int) -> List[Dict[str, Any]]:
    """Resume-aware, time-budgeted per-ID fetch with hard cap."""
    start = time.time()
    rows: List[Dict[str, Any]] = []
    count = 0
    for gid in game_ids:
        if count >= MAX_PER_ID_BACKFILL:
            log(f"[info] Hit per-ID cap (MAX_PER_ID_BACKFILL={MAX_PER_ID_BACKFILL}).")
            break
        if time.time() - start > time_budget_s:
            log(f"[info] Hit time budget for per-ID ({time_budget_s}s).")
            break

        for key in ("id", "gameId"):
            params = {key: gid}
            games = http_get(params)
            if games:
                rows.extend(norm_rows(games, season=None, week=None))
                break
        if not games:
            log(f"[info] no record for game_id={gid}")
        count += 1
        time.sleep(SLEEP_BETWEEN_CALLS_S)
    return rows

def existing_ids_from_scores(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["id"])
        return set(df["id"].dropna().astype(str))
    except Exception:
        return set()

def main():
    if not MODEL_CSV.exists():
        die(f"{MODEL_CSV} not found.")

    model = pd.read_csv(MODEL_CSV, usecols=lambda c: c in {"season","week","game_id"})
    if not {"season","week"}.issubset(model.columns):
        die("modeling_dataset.csv must include 'season' and 'week'.")

    model["season"] = pd.to_numeric(model["season"], errors="coerce").astype("Int64")
    model["week"]   = pd.to_numeric(model["week"], errors="coerce").astype("Int64")

    # Base season/week pairs
    sw = (model.dropna(subset=["season","week"])
                .drop_duplicates(subset=["season","week"])
                .sort_values(["season","week"]))
    base_pairs = [(int(r.season), int(r.week)) for r in sw.itertuples(index=False)]

    # 1) Base fetch
    base_rows = fetch_batch_by_season_week(base_pairs)

    # 2) Batched backfill (by season)
    batched_rows: List[Dict[str, Any]] = []
    per_id_rows: List[Dict[str, Any]] = []

    # Resume cache: known IDs from current output (if any) and base batch
    known_ids = existing_ids_from_scores(OUT_CSV) | {r["id"] for r in base_rows if r.get("id") is not None}

    if DIAG_MISS.exists():
        miss = pd.read_csv(DIAG_MISS)
        if "game_id" not in miss.columns:
            log("[info] unmatched_missing_game_id.csv lacks 'game_id'; skipping backfill.")
            miss = pd.DataFrame(columns=["game_id"])
        miss["game_id"] = miss["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)

        # Join season/week from model (if available)
        if "game_id" in model.columns:
            model_gid_sw = (model.assign(game_id=model["game_id"].astype(str))
                                  [["game_id","season","week"]]
                                  .drop_duplicates())
            miss_sw = miss.merge(model_gid_sw, on="game_id", how="left")
        else:
            miss_sw = miss.copy()
            if "season" not in miss_sw.columns: miss_sw["season"] = pd.Series(dtype="Int64")
            if "week"   not in miss_sw.columns: miss_sw["week"]   = pd.Series(dtype="Int64")

        # Buckets
        batched_pairs = sorted({(int(s), int(w))
                                for s, w in miss_sw.dropna(subset=["season","week"])[["season","week"]]
                                                     .itertuples(index=False, name=None)})
        batched_seasons = sorted({int(s) for s in miss_sw["season"].dropna().unique().tolist()})

        if batched_pairs:
            log(f"[info] Backfill (batch) by (season,week): {len(batched_pairs)} groups")
            batched_rows.extend(fetch_batch_by_season_week(batched_pairs))
        if batched_seasons:
            log(f"[info] Backfill (batch) by (season): {len(batched_seasons)} seasons")
            batched_rows.extend(fetch_batch_by_season(batched_seasons))

        # Which IDs still missing after base + batched + resume cache?
        have_ids = known_ids | {r["id"] for r in batched_rows if r.get("id") is not None}
        remaining_ids = [gid for gid in miss_sw["game_id"].dropna().astype(str).unique().tolist()
                         if gid not in have_ids]

        # 3) Optional sparse per-ID
        if ENABLE_PER_ID and remaining_ids:
            time_budget_s = max(60, int(MAX_BACKFILL_MINUTES) * 60)
            take_n = min(len(remaining_ids), MAX_PER_ID_BACKFILL)
            subset = remaining_ids[:take_n]
            log(f"[info] Sparse per-ID backfill for {take_n} ids "
                f"(of {len(remaining_ids)} remaining; cap={MAX_PER_ID_BACKFILL}; budget={time_budget_s}s)…")
            per_id_rows = fetch_sparse_by_ids(subset, time_budget_s)
        else:
            log("[info] Skipping per-ID backfill (disabled or nothing remaining).")

    # Combine and finalize
    all_rows = base_rows + batched_rows + per_id_rows
    df = pd.DataFrame(all_rows)

    # Ensure columns
    cols = ["id","season","week","home_team","away_team","home_points","away_points"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    df = df[cols].copy()

    # Coerce numerics
    df["home_points"] = pd.to_numeric(df["home_points"], errors="coerce")
    df["away_points"] = pd.to_numeric(df["away_points"], errors="coerce")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    log(f"[fetch] wrote {OUT_CSV} rows={len(df)} "
        f"(base={len(base_rows)} + batched_backfill={len(batched_rows)} + per_id_backfill={len(per_id_rows)}; "
        f"per_id_cap={MAX_PER_ID_BACKFILL}, time_budget_min={MAX_BACKFILL_MINUTES}, resume_known_ids={len(known_ids)})")

if __name__ == "__main__":
    if not CFBD_API_KEY:
        log("WARNING: CFBD_API_KEY not set — you may be rate-limited.")
    main()

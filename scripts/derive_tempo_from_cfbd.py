#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Derive tempo metrics (seconds/play, run-pass ratio) per team-game using CFBD drives & plays.

import os
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import requests

GAMES = Path("data/raw/games.csv")
OUT = Path("data/processed/tempo_metrics.csv")
LOG = Path("summaries/tempo_metrics_summary.txt")

CFBD_BASE = "https://api.collegefootballdata.com"
TIMEOUT = 60
PERIOD_LEN = 15 * 60  # seconds

# ---------- helpers ----------

def cfbd_get(path, params, api_key):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.get(CFBD_BASE + path, params=params, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def mmss_to_seconds(s):
    if not s or pd.isna(s):
        return None
    try:
        mm, ss = str(s).split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return None

def abs_time_since_start(period, clock_mmss):
    """
    Convert (period, clock) to absolute seconds since game start.
    CFBD clocks are time remaining in the period.
    """
    if period is None or clock_mmss is None:
        return None
    try:
        p = int(period)
    except Exception:
        return None
    secs_left = mmss_to_seconds(clock_mmss)
    if secs_left is None:
        return None
    # Seconds elapsed in this period:
    elapsed_in_period = PERIOD_LEN - secs_left
    if elapsed_in_period < 0:
        return None
    return (p - 1) * PERIOD_LEN + elapsed_in_period

def drive_duration_seconds(d):
    """
    Compute drive duration from start/end period+clock.
    Falls back to 0 if insufficient data.
    """
    s_abs = abs_time_since_start(d.get("startPeriod"), d.get("startClock"))
    e_abs = abs_time_since_start(d.get("endPeriod"), d.get("endClock"))
    if s_abs is None or e_abs is None:
        # Some CFBD responses include "duration" as MM:SS; try that as a last resort.
        dur = mmss_to_seconds(d.get("duration"))
        return int(dur) if dur is not None else 0
    dur = int(e_abs - s_abs)
    return max(dur, 0)

def classify_play(play):
    """
    Returns 'rush' | 'pass' | 'other' using robust fields.
    Treat sacks as passes.
    """
    t_raw = play.get("playTypeAbbreviation") or play.get("playType") or ""
    t = str(t_raw).lower()

    # direct buckets
    if any(k in t for k in ("pass", "sack", "sck")):
        return "pass"
    if any(k in t for k in ("rush", "run", "rushing")):
        return "rush"

    # some providers leave abbreviation blank; try textual hint
    text = (play.get("text") or "").lower()
    if "pass" in text or "sack" in text:
        return "pass"
    if "rush" in text or "run" in text:
        return "rush"

    return "other"

def norm_team(s):
    return ("" if s is None else str(s)).strip().lower()

# ---------- main ----------

def main():
    if not GAMES.exists():
        print(f"[tempo] ERROR: missing {GAMES}", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("CFBD_API_KEY", "")
    if not api_key:
        print("[tempo] WARNING: CFBD_API_KEY not set; public rate limits may apply.", file=sys.stderr)

    games = pd.read_csv(GAMES)
    games.columns = games.columns.str.lower().str.strip()

    req = ["season", "week", "game_id", "home_team", "away_team"]
    miss = [c for c in req if c not in games.columns]
    if miss:
        print(f"[tempo] ERROR: games.csv missing columns: {miss}", file=sys.stderr)
        sys.exit(1)

    # maps of (game_id, team_lower) -> aggregates
    pos_time = defaultdict(int)   # seconds of possession
    plays_total = defaultdict(int)
    plays_rush = defaultdict(int)
    plays_pass = defaultdict(int)

    game_ids = games["game_id"].dropna().astype(int).unique().tolist()

    for gid in game_ids:
        # For name matching robustness, know the two official team strings for this game
        subset = games.loc[games["game_id"] == gid].head(1)
        if subset.empty:
            continue
        h_name = norm_team(subset.iloc[0]["home_team"])
        a_name = norm_team(subset.iloc[0]["away_team"])
        valid_names = {h_name, a_name}

        # Drives -> possession time
        try:
            drives = cfbd_get("/drives", {"gameId": gid}, api_key)
            for d in drives or []:
                off = norm_team(d.get("offense"))
                # Some feeds use school names; try to coerce to one of the two teams
                if off not in valid_names:
                    # simple contains heuristic
                    if h_name and h_name in off:
                        off = h_name
                    elif a_name and a_name in off:
                        off = a_name
                if off not in valid_names:
                    continue
                pos_time[(gid, off)] += drive_duration_seconds(d)
        except Exception as e:
            print(f"[tempo] WARN: drives fetch failed for game {gid}: {e}", file=sys.stderr)

        # Plays -> counts
        try:
            plays = cfbd_get("/plays", {"gameId": gid}, api_key)
            for p in plays or []:
                off = norm_team(p.get("offense"))
                if off not in valid_names:
                    if h_name and h_name in off:
                        off = h_name
                    elif a_name and a_name in off:
                        off = a_name
                if off not in valid_names:
                    continue

                cat = classify_play(p)
                if cat == "rush":
                    plays_rush[(gid, off)] += 1
                    plays_total[(gid, off)] += 1
                elif cat == "pass":
                    plays_pass[(gid, off)] += 1
                    plays_total[(gid, off)] += 1
                else:
                    plays_total[(gid, off)] += 1
        except Exception as e:
            print(f"[tempo] WARN: plays fetch failed for game {gid}: {e}", file=sys.stderr)

    # Build per-team-game rows
    rows = []
    for _, r in games.iterrows():
        season = int(r["season"])
        week = int(r["week"])
        gid = int(r["game_id"])
        for team_col, opp_col, is_home in [("home_team", "away_team", 1), ("away_team", "home_team", 0)]:
            team = str(r[team_col]).strip()
            team_l = norm_team(team)
            opp = str(r[opp_col]).strip()

            total = plays_total.get((gid, team_l), 0)
            rush = plays_rush.get((gid, team_l), 0)
            pss  = plays_pass.get((gid, team_l), 0)
            pos_sec = pos_time.get((gid, team_l), 0)

            sec_per_play = round(pos_sec / total, 3) if total > 0 and pos_sec > 0 else None
            rpr = round(rush / pss, 3) if pss > 0 else (round(rush / total, 3) if total > 0 else None)

            rows.append({
                "season": season,
                "week": week,
                "game_id": gid,
                "team": team,
                "opponent": opp,
                "is_home": is_home,
                "off_plays": total if total > 0 else None,
                "off_rush_att": rush if total > 0 else None,
                "off_pass_att": pss if total > 0 else None,
                "off_possession_sec": pos_sec if pos_sec > 0 else None,
                "sec_per_play": sec_per_play,
                "run_pass_ratio": rpr
            })

    out = pd.DataFrame(rows).sort_values(["season", "week", "game_id", "team"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    # Summary
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w") as f:
        f.write(f"rows: {len(out)}\n")
        f.write(f"unique_games: {out['game_id'].nunique()}\n")
        f.write(f"teams: {out['team'].nunique()}\n")
        mean_spp = out["sec_per_play"].dropna()
        mean_rpr = out["run_pass_ratio"].dropna()
        f.write(f"mean_sec_per_play: {mean_spp.mean():.2f}\n" if not mean_spp.empty else "mean_sec_per_play: nan\n")
        f.write(f"median_sec_per_play: {mean_spp.median():.2f}\n" if not mean_spp.empty else "median_sec_per_play: nan\n")
        f.write(f"mean_run_pass_ratio: {mean_rpr.mean():.2f}\n" if not mean_rpr.empty else "mean_run_pass_ratio: nan\n")
        f.write(f"nonzero_possessions: {(out['off_possession_sec'].notna()).sum()}\n")

    print(f"[tempo] wrote {OUT}")
    print(f"[tempo] summary -> {LOG}")

if __name__ == "__main__":
    main()

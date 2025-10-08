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

# ---- helpers ----

def cfbd_get(path, params, api_key):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.get(CFBD_BASE + path, params=params, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def mmss_to_seconds(s):
    # Examples: "2:34", "10:00", may be None
    if not s or pd.isna(s):
        return 0
    try:
        mm, ss = str(s).split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return 0

def drive_duration_seconds(d):
    # CFBD drives often include "duration" as "MM:SS"
    dur = d.get("duration")
    if dur:
        return mmss_to_seconds(dur)
    # fallback: if start/end clocks exist per period, skip (complex). Return 0.
    return 0

def classify_play(play):
    """
    Returns ('rush' | 'pass' | 'other')
    CFBD playType categories vary; we handle common names safely.
    """
    t = (play.get("playType") or "").lower()
    # Heuristics
    if "rush" in t or "run" in t or "rushing" in t:
        return "rush"
    if "pass" in t or "sack" in t:
        return "pass"
    return "other"

# ---- main ----

def main():
    if not GAMES.exists():
        print(f"[tempo] ERROR: missing {GAMES}", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("CFBD_API_KEY", "")
    if not api_key:
        print("[tempo] WARNING: CFBD_API_KEY not set; public rate limits may apply.", file=sys.stderr)

    games = pd.read_csv(GAMES)
    games.columns = games.columns.str.lower().str.strip()

    required = ["season", "week", "game_id", "home_team", "away_team"]
    miss = [c for c in required if c not in games.columns]
    if miss:
        print(f"[tempo] ERROR: games.csv missing columns: {miss}", file=sys.stderr)
        sys.exit(1)

    game_ids = games["game_id"].dropna().astype(int).unique().tolist()

    # Aggregation dicts
    # Drives: offense -> sum duration seconds
    pos_time = defaultdict(int)
    # Plays: offense -> counts
    plays_total = defaultdict(int)
    plays_rush = defaultdict(int)
    plays_pass = defaultdict(int)

    # Pull per game
    for gid in game_ids:
        # Drives
        try:
            drives = cfbd_get("/drives", {"gameId": gid}, api_key)
            for d in drives or []:
                off = (d.get("offense") or "").strip()
                if not off:
                    continue
                pos_time[(gid, off)] += drive_duration_seconds(d)
        except Exception as e:
            print(f"[tempo] WARN: drives fetch failed for game {gid}: {e}", file=sys.stderr)

        # Plays
        try:
            plays = cfbd_get("/plays", {"gameId": gid}, api_key)
            for p in plays or []:
                off = (p.get("offense") or "").strip()
                if not off:
                    continue
                cat = classify_play(p)
                if cat == "rush":
                    plays_rush[(gid, off)] += 1
                    plays_total[(gid, off)] += 1
                elif cat == "pass":
                    plays_pass[(gid, off)] += 1
                    plays_total[(gid, off)] += 1
                else:
                    # Count toward total plays
                    plays_total[(gid, off)] += 1
        except Exception as e:
            print(f"[tempo] WARN: plays fetch failed for game {gid}: {e}", file=sys.stderr)

    # Build per-team-game rows for both home and away teams in games.csv
    rows = []
    for _, r in games.iterrows():
        season = int(r["season"])
        week = int(r["week"])
        gid = int(r["game_id"])
        for team_col, opp_col, is_home in [("home_team", "away_team", 1), ("away_team", "home_team", 0)]:
            team = str(r[team_col]).strip()
            opp = str(r[opp_col]).strip()
            key = (gid, team)

            total = plays_total.get(key, 0)
            rush = plays_rush.get(key, 0)
            pss = plays_pass.get(key, 0)
            pos_sec = pos_time.get(key, 0)

            sec_per_play = round(pos_sec / total, 3) if total > 0 else None
            rpr = round(rush / pss, 3) if pss > 0 else None

            rows.append({
                "season": season,
                "week": week,
                "game_id": gid,
                "team": team,
                "opponent": opp,
                "is_home": is_home,
                "off_plays": total,
                "off_rush_att": rush,
                "off_pass_att": pss,
                "off_possession_sec": pos_sec if pos_sec else None,
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
        f.write(f"mean_sec_per_play: {out['sec_per_play'].dropna().mean():.2f}\n")
        f.write(f"median_sec_per_play: {out['sec_per_play'].dropna().median():.2f}\n")
        f.write(f"mean_run_pass_ratio: {out['run_pass_ratio'].dropna().mean():.2f}\n")
        f.write(f"nonzero_possessions: {(out['off_possession_sec'].fillna(0)>0).sum()}\n")

    print(f"[tempo] wrote {OUT}")
    print(f"[tempo] summary -> {LOG}")

if __name__ == "__main__":
    main()

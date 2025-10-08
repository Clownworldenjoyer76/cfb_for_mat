#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Map your games to CFBD official game IDs and write back to data/raw/games.csv (adds cfbd_game_id).

import os
import sys
from pathlib import Path
import pandas as pd
import requests

GAMES = Path("data/raw/games.csv")
BACKUP = Path("data/raw/games_backup_before_cfbd_id.csv")

BASE = "https://api.collegefootballdata.com"
TIMEOUT = 45

def cfbd_get(path, params, api_key):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.get(BASE + path, params=params, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def norm(s):
    return ("" if s is None else str(s)).strip().lower()

def main():
    if not GAMES.exists():
        print("[map] missing data/raw/games.csv", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("CFBD_API_KEY", "")
    if not api_key:
        print("[map] WARNING: CFBD_API_KEY not set; you may be rate-limited.", file=sys.stderr)

    df = pd.read_csv(GAMES)
    df.columns = df.columns.str.lower().str.strip()

    required = ["season","week","game_id","home_team","away_team"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        print(f"[map] games.csv missing columns: {miss}", file=sys.stderr)
        sys.exit(1)

    if "cfbd_game_id" not in df.columns:
        df["cfbd_game_id"] = pd.NA

    mapped = 0
    for i, row in df.iterrows():
        if pd.notna(row.get("cfbd_game_id")):
            continue
        season = int(row["season"])
        week = int(row["week"])
        home = norm(row["home_team"])
        away = norm(row["away_team"])

        try:
            # Primary: search by season/week and filter by teams
            cand = cfbd_get("/games", {"year": season, "week": week}, api_key)
        except Exception as e:
            print(f"[map] WARN fetch games season={season} week={week}: {e}", file=sys.stderr)
            continue

        got = None
        for g in cand or []:
            h = norm(g.get("home_team") or g.get("home"))
            a = norm(g.get("away_team") or g.get("away"))
            if {h,a} == {home, away}:
                got = g.get("id")
                break

        # Fallback: try filtering by each team directly if needed
        if got is None:
            try:
                c2 = cfbd_get("/games", {"year": season, "week": week, "team": row["home_team"]}, api_key)
                for g in c2 or []:
                    h = norm(g.get("home_team") or g.get("home"))
                    a = norm(g.get("away_team") or g.get("away"))
                    if {h,a} == {home, away}:
                        got = g.get("id"); break
            except Exception:
                pass

        if got is not None:
            df.at[i, "cfbd_game_id"] = int(got)
            mapped += 1

    # Backup and write
    if not BACKUP.exists():
        df_backup = pd.read_csv(GAMES)
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        df_backup.to_csv(BACKUP, index=False)

    df.to_csv(GAMES, index=False)
    print(f"[map] mapped {mapped} rows to cfbd_game_id")

if __name__ == "__main__":
    main()

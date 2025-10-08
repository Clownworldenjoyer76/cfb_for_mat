#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Using cfbd_game_id, fill venue fields in data/raw/games.csv from CFBD.

import os
import sys
from pathlib import Path
import pandas as pd
import requests

GAMES = Path("data/raw/games.csv")
OUT = GAMES  # in-place
LOG = Path("summaries/backfill_games_venue_summary.txt")

BASE = "https://api.collegefootballdata.com"
TIMEOUT = 45

def cfbd_get(path, params, api_key):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.get(BASE + path, params=params, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def main():
    if not GAMES.exists():
        print(f"[venue] ERROR: missing {GAMES}", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("CFBD_API_KEY", "")
    if not api_key:
        print("[venue] WARNING: CFBD_API_KEY not set; you may be rate-limited.", file=sys.stderr)

    df = pd.read_csv(GAMES)
    df.columns = df.columns.str.lower().str.strip()

    if "cfbd_game_id" not in df.columns:
        print("[venue] ERROR: cfbd_game_id column not found. Run mapping first.", file=sys.stderr)
        sys.exit(1)

    # Ensure venue columns exist
    for c in ["venue", "venue_lat", "venue_lon", "venue_timezone", "venue_altitude_m"]:
        if c not in df.columns:
            df[c] = pd.NA

    ids = df["cfbd_game_id"].dropna().astype(int).unique().tolist()
    filled = 0

    # Pull CFBD game objects by season/week batches (more efficient) or per ID
    # Not all CFBD endpoints support filtering by ID; fetch by season and filter.
    seasons = sorted(df["season"].dropna().astype(int).unique().tolist())
    by_season = {}
    for yr in seasons:
        try:
            g = cfbd_get("/games", {"year": yr, "seasonType": "both"}, api_key)
            gd = pd.DataFrame(g or [])
            if gd.empty:
                by_season[yr] = pd.DataFrame()
            else:
                # normalize columns
                if "home_team" not in gd.columns and "home" in gd.columns:
                    gd["home_team"] = gd["home"]
                if "away_team" not in gd.columns and "away" in gd.columns:
                    gd["away_team"] = gd["away"]
                for c in ["id","venue","venue_id","home_team","away_team","start_date","season","week"]:
                    if c not in gd.columns:
                        gd[c] = pd.NA
                by_season[yr] = gd
        except Exception as e:
            print(f"[venue] WARN: CFBD /games fetch failed for {yr}: {e}", file=sys.stderr)
            by_season[yr] = pd.DataFrame()

    # Build venue detail cache by venue_id
    venue_cache = {}
    def get_venue_details(venue_id: int):
        if venue_id in venue_cache:
            return venue_cache[venue_id]
        try:
            vjs = cfbd_get("/venues", {"id": venue_id}, api_key)
            if isinstance(vjs, list) and vjs:
                v = vjs[0]
                venue_cache[venue_id] = v
                return v
        except Exception:
            pass
        venue_cache[venue_id] = None
        return None

    for i, row in df.iterrows():
        gid = row.get("cfbd_game_id")
        if pd.isna(gid):
            continue
        gid = int(gid)
        yr = int(row["season"])
        allg = by_season.get(yr, pd.DataFrame())
        if allg.empty:
            continue
        match = allg.loc[allg["id"] == gid]
        if match.empty:
            continue

        venue_name = match.iloc[0].get("venue")
        venue_id = match.iloc[0].get("venue_id")
        lat = lon = tz = alt = None

        if pd.notna(venue_id):
            vd = get_venue_details(int(venue_id))
            if vd:
                lat = vd.get("latitude")
                lon = vd.get("longitude")
                tz  = vd.get("timezone")
                alt = vd.get("elevation")

        # Write into df if missing
        changed = False
        if pd.isna(row["venue"]) and pd.notna(venue_name):
            df.at[i, "venue"] = venue_name; changed = True
        if pd.isna(row["venue_lat"]) and pd.notna(lat):
            df.at[i, "venue_lat"] = float(lat); changed = True
        if pd.isna(row["venue_lon"]) and pd.notna(lon):
            df.at[i, "venue_lon"] = float(lon); changed = True
        if pd.isna(row["venue_timezone"]) and pd.notna(tz):
            df.at[i, "venue_timezone"] = str(tz); changed = True
        if pd.isna(row["venue_altitude_m"]) and pd.notna(alt):
            try:
                df.at[i, "venue_altitude_m"] = round(float(alt) * 0.3048, 1)  # feet->meters if CFBD returns feet
            except Exception:
                df.at[i, "venue_altitude_m"] = float(alt)
            changed = True

        if changed:
            filled += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w") as f:
        f.write(f"rows_total: {len(df)}\n")
        f.write(f"with_cfbd_game_id: {df['cfbd_game_id'].notna().sum()}\n")
        f.write(f"rows_updated_with_venue_info: {filled}\n")

    print(f"[venue] wrote {OUT} and {LOG} (updated {filled} rows)")

if __name__ == "__main__":
    main()

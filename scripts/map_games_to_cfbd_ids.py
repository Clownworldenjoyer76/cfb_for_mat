#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Add `cfbd_game_id` to data/raw/games.csv using robust alias + fuzzy matching.

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from rapidfuzz import process, fuzz

GAMES = Path("data/raw/games.csv")
BACKUP = Path("data/raw/games_backup_before_cfbd_id.csv")

CFBD_BASE = "https://api.collegefootballdata.com"
TIMEOUT = 45

# Lightweight alias map for common short names -> CFBD canonical
ALIAS: Dict[str, str] = {
    "app state": "Appalachian State",
    "texas a&m": "Texas A&M",
    "texas a and m": "Texas A&M",
    "utsa": "UTSA",
    "ucf": "UCF",
    "usc": "USC",
    "ucla": "UCLA",
    "umass": "UMass",
    "uconn": "Connecticut",
    "byu": "BYU",
    "hawaii": "Hawai'i",
    "ole miss": "Ole Miss",
    "pitt": "Pittsburgh",
    "smu": "SMU",
    "miami (oh)": "Miami (OH)",
    "louisiana": "Louisiana",
    "louisiana-lafayette": "Louisiana",
    "louisiana monroe": "UL Monroe",
    "ul monroe": "UL Monroe",
    "utsa": "UTSA",
    "texas-san antonio": "UTSA",
    "washington st": "Washington State",
    "boise st": "Boise State",
    "san jose st": "San Jose State",
    "arizona st": "Arizona State",
    "colorado st": "Colorado State",
    "fiu": "FIU",
    "fau": "Florida Atlantic",
    "unlv": "UNLV",
    "niu": "Northern Illinois",
    "sjsu": "San Jose State",
    "temple": "Temple",
    "tcu": "TCU",
}

def cfbd_get(path, params, api_key):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.get(CFBD_BASE + path, params=params, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def norm(s: Optional[str]) -> str:
    return ("" if s is None else str(s)).strip().lower()

def pick_best(name: str, candidates: pd.Series, cutoff: int = 90) -> Optional[str]:
    """Fuzzy-pick a canonical team name from CFBD candidate list."""
    if not name:
        return None
    labels = candidates.tolist()
    best = process.extractOne(name, labels, scorer=fuzz.WRatio)
    if not best:
        return None
    label, score, _ = best
    return label if score >= cutoff else None

def main():
    if not GAMES.exists():
        print("[map] ERROR: missing data/raw/games.csv", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("CFBD_API_KEY", "")
    if not api_key:
        print("[map] WARNING: CFBD_API_KEY not set; you may be rate-limited.", file=sys.stderr)

    df = pd.read_csv(GAMES)
    df.columns = df.columns.str.lower().str.strip()

    required = ["season", "week", "game_id", "home_team", "away_team"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        print(f"[map] ERROR: games.csv missing columns: {miss}", file=sys.stderr)
        sys.exit(1)

    if "cfbd_game_id" not in df.columns:
        df["cfbd_game_id"] = pd.NA

    # fetch season/week games from CFBD and build a lookup per (season,week)
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    # cache: {(season, week): DataFrame}
    cache = {}

    mapped = 0
    for idx, row in df.iterrows():
        if pd.notna(row.get("cfbd_game_id")):
            continue

        season = int(row["season"])
        week = int(row["week"])
        home_in = str(row["home_team"]).strip()
        away_in = str(row["away_team"]).strip()

        key = (season, week)
        if key not in cache:
            try:
                games = cfbd_get("/games", {"year": season, "week": week}, api_key)
                # Normalize into DataFrame
                gdf = pd.DataFrame(games or [])
                if gdf.empty:
                    cache[key] = pd.DataFrame(columns=["id","home_team","away_team"])
                else:
                    # Ensure canonical columns and normalize text
                    if "home_team" not in gdf.columns and "home" in gdf.columns:
                        gdf["home_team"] = gdf["home"]
                    if "away_team" not in gdf.columns and "away" in gdf.columns:
                        gdf["away_team"] = gdf["away"]
                    need = ["id", "home_team", "away_team"]
                    for c in need:
                        if c not in gdf.columns:
                            gdf[c] = pd.NA
                    gdf["home_team_n"] = gdf["home_team"].astype(str).str.strip()
                    gdf["away_team_n"] = gdf["away_team"].astype(str).str.strip()
                    cache[key] = gdf[["id","home_team_n","away_team_n"]].copy()
            except Exception as e:
                print(f"[map] WARN: CFBD /games fetch failed for {season} wk {week}: {e}", file=sys.stderr)
                cache[key] = pd.DataFrame(columns=["id","home_team_n","away_team_n"])

        g = cache[key]
        if g.empty:
            continue

        # Build a canonical name list for fuzzy
        canon = pd.unique(pd.concat([g["home_team_n"], g["away_team_n"]], ignore_index=True))

        def canon_name(x: str) -> str:
            x_norm = norm(x)
            return ALIAS.get(x_norm, x.strip())

        home_try = canon_name(home_in)
        away_try = canon_name(away_in)

        # exact first
        exact = g[(g["home_team_n"].str.casefold() == home_try.casefold()) &
                  (g["away_team_n"].str.casefold() == away_try.casefold())]
        if exact.empty:
            # fuzzy
            h_pick = pick_best(home_try, pd.Series(canon), cutoff=88)
            a_pick = pick_best(away_try, pd.Series(canon), cutoff=88)
            if h_pick and a_pick:
                fuzzy = g[(g["home_team_n"].str.casefold() == h_pick.casefold()) &
                          (g["away_team_n"].str.casefold() == a_pick.casefold())]
                match_df = fuzzy
            else:
                match_df = pd.DataFrame()
        else:
            match_df = exact

        if not match_df.empty and pd.notna(match_df.iloc[0]["id"]):
            df.at[idx, "cfbd_game_id"] = int(match_df.iloc[0]["id"])
            mapped += 1

    # backup and write
    if not BACKUP.exists():
        orig = pd.read_csv(GAMES)
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        orig.to_csv(BACKUP, index=False)

    df.to_csv(GAMES, index=False)
    print(f"[map] mapped {mapped} of {len(df)} rows to cfbd_game_id")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Add `cfbd_game_id` to data/raw/games.csv using robust alias + fuzzy + date proximity matching.
#
# Outputs:
#   - data/raw/games.csv (updated in place; adds/updates cfbd_game_id)
#   - summaries/map_cfbd_ids_summary.txt
#   - summaries/unmapped_games.csv  (diagnostics for any rows we couldn't map)

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd
import requests
from rapidfuzz import process, fuzz

GAMES = Path("data/raw/games.csv")
BACKUP = Path("data/raw/games_backup_before_cfbd_id.csv")
SUM_LOG = Path("summaries/map_cfbd_ids_summary.txt")
UNMAPPED_CSV = Path("summaries/unmapped_games.csv")

CFBD_BASE = "https://api.collegefootballdata.com"
TIMEOUT = 45

# --------- Team alias map (expandable) ----------
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
    "miami oh": "Miami (OH)",
    "louisiana-lafayette": "Louisiana",
    "louisiana lafayette": "Louisiana",
    "louisiana monroe": "UL Monroe",
    "ul monroe": "UL Monroe",
    "washington st": "Washington State",
    "wash st": "Washington State",
    "boise st": "Boise State",
    "san jose st": "San Jose State",
    "arizona st": "Arizona State",
    "colorado st": "Colorado State",
    "fiu": "FIU",
    "fau": "Florida Atlantic",
    "unlv": "UNLV",
    "niu": "Northern Illinois",
    "sjsu": "San Jose State",
    "tcu": "TCU",
    "olemiss": "Ole Miss",
    "texasam": "Texas A&M",
    "la tech": "Louisiana Tech",
    "southern miss": "Southern Miss",
    "texas san antonio": "UTSA",
    "central florida": "UCF",
    "southern california": "USC",
    "california-los angeles": "UCLA",
    "massachusetts": "UMass",
}

# --------- HTTP helper ----------
def cfbd_get(path: str, params: dict, api_key: str):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.get(CFBD_BASE + path, params=params, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

# --------- text helpers ----------
def norm(s: Optional[str]) -> str:
    return ("" if s is None else str(s)).strip().lower()

def alias_or_self(name: str) -> str:
    n = norm(name)
    return ALIAS.get(n, name.strip())

def fuzzy_pick(name: str, choices: List[str], cutoff: int = 90) -> Optional[str]:
    if not name or not choices:
        return None
    best = process.extractOne(name, choices, scorer=fuzz.WRatio)
    if not best:
        return None
    label, score, _ = best
    return label if score >= cutoff else None

# --------- matching strategies ----------
def match_in_df(gdf: pd.DataFrame, home_try: str, away_try: str) -> Optional[int]:
    """Exact casefold match inside a normalized CFBD games dataframe."""
    exact = gdf[
        (gdf["home_team_n"].str.casefold() == home_try.casefold()) &
        (gdf["away_team_n"].str.casefold() == away_try.casefold())
    ]
    if not exact.empty and pd.notna(exact.iloc[0]["id"]):
        return int(exact.iloc[0]["id"])
    return None

def match_fuzzy_in_df(gdf: pd.DataFrame, home_try: str, away_try: str, cutoff: int = 88) -> Optional[int]:
    choices = pd.unique(pd.concat([gdf["home_team_n"], gdf["away_team_n"]], ignore_index=True)).tolist()
    h_pick = fuzzy_pick(home_try, choices, cutoff=cutoff)
    a_pick = fuzzy_pick(away_try, choices, cutoff=cutoff)
    if h_pick and a_pick:
        fuzzy = gdf[
            (gdf["home_team_n"].str.casefold() == h_pick.casefold()) &
            (gdf["away_team_n"].str.casefold() == a_pick.casefold())
        ]
        if not fuzzy.empty and pd.notna(fuzzy.iloc[0]["id"]):
            return int(fuzzy.iloc[0]["id"])
    return None

def date_distance_days(iso_a: str, iso_b: str) -> Optional[int]:
    try:
        a = pd.to_datetime(iso_a, utc=True)
        b = pd.to_datetime(iso_b, utc=True)
        return abs((a - b).days)
    except Exception:
        return None

# --------- main ----------
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

    # Ensure types
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    if "kickoff_utc" in df.columns:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)

    # Cache by (season, week) AND by season-only
    cache_sw = {}
    cache_season = {}

    def fetch_games_by_sw(season: int, week: int) -> pd.DataFrame:
        key = (season, week)
        if key in cache_sw:
            return cache_sw[key]
        try:
            js = cfbd_get("/games", {"year": season, "week": week, "seasonType": "both"}, api_key)
        except Exception as e:
            print(f"[map] WARN: /games year={season} week={week}: {e}", file=sys.stderr)
            js = []
        gdf = pd.DataFrame(js or [])
        gdf = normalize_cfbd_games_df(gdf)
        cache_sw[key] = gdf
        return gdf

    def fetch_games_by_season(season: int) -> pd.DataFrame:
        if season in cache_season:
            return cache_season[season]
        try:
            js = cfbd_get("/games", {"year": season, "seasonType": "both"}, api_key)
        except Exception as e:
            print(f"[map] WARN: /games year={season}: {e}", file=sys.stderr)
            js = []
        gdf = pd.DataFrame(js or [])
        gdf = normalize_cfbd_games_df(gdf)
        cache_season[season] = gdf
        return gdf

    def normalize_cfbd_games_df(gdf: pd.DataFrame) -> pd.DataFrame:
        if gdf.empty:
            return pd.DataFrame(columns=["id", "home_team_n", "away_team_n", "start_date", "week", "season"])
        # promote columns
        if "home_team" not in gdf.columns and "home" in gdf.columns:
            gdf["home_team"] = gdf["home"]
        if "away_team" not in gdf.columns and "away" in gdf.columns:
            gdf["away_team"] = gdf["away"]
        for c in ["id", "home_team", "away_team", "start_date", "week", "season"]:
            if c not in gdf.columns:
                gdf[c] = pd.NA
        out = pd.DataFrame({
            "id": gdf["id"],
            "home_team_n": gdf["home_team"].astype(str).str.strip(),
            "away_team_n": gdf["away_team"].astype(str).str.strip(),
            "start_date": pd.to_datetime(gdf["start_date"], errors="coerce", utc=True),
            "week": pd.to_numeric(gdf["week"], errors="coerce"),
            "season": pd.to_numeric(gdf["season"], errors="coerce"),
        })
        return out

    mapped = 0
    unmapped_rows = []

    for i, row in df.iterrows():
        if pd.notna(row.get("cfbd_game_id")):
            continue
        if pd.isna(row["season"]) or pd.isna(row["week"]):
            unmapped_rows.append({"index": i, "reason": "missing season/week", **row.to_dict()})
            continue

        season = int(row["season"])
        week = int(row["week"])
        home_raw = str(row["home_team"]).strip()
        away_raw = str(row["away_team"]).strip()
        home_try = alias_or_self(home_raw)
        away_try = alias_or_self(away_raw)

        # 1) Try exact in (season, week)
        g_sw = fetch_games_by_sw(season, week)
        gid = match_in_df(g_sw, home_try, away_try)

        # 2) Try fuzzy in (season, week)
        if gid is None and not g_sw.empty:
            gid = match_fuzzy_in_df(g_sw, home_try, away_try, cutoff=88)

        # 3) Try season-wide fuzzy + date proximity (±3 days of kickoff_utc if available)
        if gid is None:
            g_season = fetch_games_by_season(season)
            if not g_season.empty:
                gid = match_fuzzy_in_df(g_season, home_try, away_try, cutoff=88)
                if gid is None and pd.notna(row.get("kickoff_utc")):
                    # Pick best by date distance among all games involving either team
                    t_home = g_season[(g_season["home_team_n"].str.casefold() == home_try.casefold()) |
                                      (g_season["away_team_n"].str.casefold() == home_try.casefold())].copy()
                    t_away = g_season[(g_season["home_team_n"].str.casefold() == away_try.casefold()) |
                                      (g_season["away_team_n"].str.casefold() == away_try.casefold())].copy()
                    pool = pd.concat([t_home, t_away]).drop_duplicates(subset=["id"])
                    if not pool.empty:
                        pool["date_gap"] = (pool["start_date"] - row["kickoff_utc"]).abs().dt.days
                        pool = pool.sort_values(["date_gap"])
                        # require <= 3-day gap to select
                        pool = pool[pool["date_gap"] <= 3]
                        if not pool.empty:
                            gid = int(pool.iloc[0]["id"])

        if gid is not None:
            df.at[i, "cfbd_game_id"] = int(gid)
            mapped += 1
        else:
            unmapped_rows.append({
                "index": i,
                "season": season,
                "week": week,
                "home_in": home_raw,
                "away_in": away_raw,
                "home_try": home_try,
                "away_try": away_try,
                "kickoff_utc": row.get("kickoff_utc")
            })

    # backup once
    if not BACKUP.exists():
        orig = pd.read_csv(GAMES)
        BACKUP.parent.mkdir(parents=True, exist_ok=True)
        orig.to_csv(BACKUP, index=False)

    # write updated games
    df.to_csv(GAMES, index=False)

    # write summary + unmapped
    SUM_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SUM_LOG, "w", encoding="utf-8") as f:
        total = len(df)
        have_ids = int(df["cfbd_game_id"].notna().sum())
        f.write(f"rows_total: {total}\n")
        f.write(f"mapped_rows: {have_ids}\n")
        f.write(f"unmapped_rows: {len(unmapped_rows)}\n")

    if unmapped_rows:
        UNMAPPED_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(unmapped_rows).to_csv(UNMAPPED_CSV, index=False)

    print(f"[map] mapped {mapped} of {len(df)} rows to cfbd_game_id")
    if unmapped_rows:
        print(f"[map] wrote diagnostics -> {UNMAPPED_CSV}")

if __name__ == "__main__":
    main()

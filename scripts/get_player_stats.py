#!/usr/bin/env python3
# Minimal CFBD player data pull (season/game stats + injuries)
# Writes three CSVs in repo root. No external config required.

import os
import sys
import datetime as dt
import time
import requests
import pandas as pd

API_BASE = "https://api.collegefootballdata.com"

def env(name, default=None):
    v = os.getenv(name)
    return v if v not in (None, "", "None") else default

def now_utc_iso():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

def fetch_df(endpoint, params, api_key, retries=3, backoff=1.5):
    url = f"{API_BASE}{endpoint}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=60)
            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    return pd.DataFrame()
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    return pd.json_normalize(data)
                else:
                    return pd.DataFrame()
            # retry on transient errors
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff ** attempt)
                continue
            # non-retryable error → empty
            return pd.DataFrame()
        except requests.RequestException:
            time.sleep(backoff ** attempt)
    return pd.DataFrame()

def main():
    api_key = env("CFBD_API_KEY", "")
    year = env("CFB_YEAR", str(dt.datetime.now(dt.timezone.utc).year))
    season_type = env("CFB_SEASON_TYPE", "regular").lower()
    if season_type not in ("regular", "postseason"):
        season_type = "regular"

    season_params = {"year": year, "seasonType": season_type}
    game_params = {"year": year, "seasonType": season_type}
    injuries_params = {"year": year}

    df_season = fetch_df("/stats/player/season", season_params, api_key)
    df_game = fetch_df("/stats/player/game", game_params, api_key)
    df_inj = fetch_df("/injuries", injuries_params, api_key)

    df_season.to_csv("player_stats_season_raw.csv", index=False)
    df_game.to_csv("player_stats_game_raw.csv", index=False)
    df_inj.to_csv("player_injuries_raw.csv", index=False)

    # Simple run log to help CI output
    with open("logs_player_stats_run.txt", "w", encoding="utf-8") as f:
        f.write(f"[{now_utc_iso()}] year={year} season_type={season_type}\n")
        f.write(f"season_rows={len(df_season)}\n")
        f.write(f"game_rows={len(df_game)}\n")
        f.write(f"injury_rows={len(df_inj)}\n")

if __name__ == "__main__":
    # Ensure log file can be written even if no logs/ directory exists
    try:
        main()
    except Exception as e:
        # Fail soft: write a minimal error file so the workflow can commit it for debugging
        with open("logs_player_stats_error.txt", "w", encoding="utf-8") as f:
            f.write(f"[{now_utc_iso()}] ERROR: {repr(e)}\n")
        sys.exit(0)

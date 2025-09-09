import os
import time
import datetime as dt
import requests
import pandas as pd

API_BASE = "https://api.collegefootballdata.com"

def env(name, default=None):
    v = os.getenv(name)
    return v if v not in (None, "", "None") else default

def fetch_df(endpoint, params, api_key, retries=3, backoff=1.5):
    url = API_BASE + endpoint
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=60)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return pd.DataFrame(data)
                if isinstance(data, dict):
                    return pd.json_normalize(data)
                return pd.DataFrame()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff ** (i + 1))
                continue
            return pd.DataFrame()
        except requests.RequestException:
            time.sleep(backoff ** (i + 1))
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

    with open("logs_player_stats_run.txt", "w", encoding="utf-8") as f:
        f.write("year=" + year + "\n")
        f.write("season_type=" + season_type + "\n")
        f.write("season_rows=" + str(len(df_season)) + "\n")
        f.write("game_rows=" + str(len(df_game)) + "\n")
        f.write("injury_rows=" + str(len(df_inj)) + "\n")

if __name__ == "__main__":
    main()

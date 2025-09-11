import os
import time
import datetime as dt
from typing import Any, Dict, Optional, List

import requests
import pandas as pd

API_BASE = "https://api.collegefootballdata.com"

# -------------------------
# Helpers
# -------------------------

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "", "None") else default

def _get(endpoint: str, params: Dict[str, Any], api_key: str, retries: int = 3, backoff: float = 1.6) -> Any:
    url = API_BASE + endpoint
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=60)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff ** (i + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(backoff ** (i + 1))
    return None

def _to_df(obj: Any) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        return pd.json_normalize(obj)
    return pd.DataFrame()

def _first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------------
# Build
# -------------------------

def build_weather(year: str, season_type: str, api_key: str) -> pd.DataFrame:
    # Pull all games for year/season_type
    games = _to_df(_get("/games", {"year": year, "seasonType": season_type}, api_key))

    # Normalize basics
    if "id" in games.columns and "game_id" not in games.columns:
        games = games.rename(columns={"id": "game_id"})
    if "startDate" in games.columns and "start_date" not in games.columns:
        games = games.rename(columns={"startDate": "start_date"})

    # Home/Away team name columns vary; pick best-available
    home_col = _first(games, ["home_team", "homeTeam", "homeTeam.school", "homeTeam.name"])
    away_col = _first(games, ["away_team", "awayTeam", "awayTeam.school", "awayTeam.name"])

    # Weather fields vary; select common candidates
    weather_text = _first(games, ["weather", "game.weather", "conditions"])
    temp_col = _first(games, ["temperature", "game.temperature", "temp"])
    wind_spd = _first(games, ["windSpeed", "game.windSpeed", "wind_speed"])
    wind_dir = _first(games, ["windDirection", "game.windDirection", "wind_direction"])
    precip   = _first(games, ["precipitation", "game.precipitation", "precip"])

    # Assemble output columns safely
    out = pd.DataFrame()
    out["game_id"] = games["game_id"] if "game_id" in games.columns else pd.NA
    out["season"] = games["season"] if "season" in games.columns else pd.NA
    out["week"] = games["week"] if "week" in games.columns else pd.NA

    if "start_date" in games.columns:
        out["start_date"] = pd.to_datetime(games["start_date"], errors="coerce", utc=True)
    else:
        out["start_date"] = pd.NaT

    out["home_team"] = games[home_col] if home_col else pd.NA
    out["away_team"] = games[away_col] if away_col else pd.NA

    out["weather"] = games[weather_text] if weather_text else pd.NA
    out["temperature"] = pd.to_numeric(games[temp_col], errors="coerce") if temp_col else pd.NA
    out["wind_speed"] = pd.to_numeric(games[wind_spd], errors="coerce") if wind_spd else pd.NA
    out["wind_direction"] = pd.to_numeric(games[wind_dir], errors="coerce") if wind_dir else pd.NA
    out["precipitation"] = pd.to_numeric(games[precip], errors="coerce") if precip else pd.NA

    # Deduplicate per game_id if needed
    if "game_id" in out.columns:
        out = out.sort_values(["season","week","game_id"]).drop_duplicates(subset=["game_id"], keep="first")

    return out

# -------------------------
# Main
# -------------------------

def main():
    api_key = _env("CFBD_API_KEY", "")
    year = _env("CFB_YEAR", str(dt.datetime.now(dt.timezone.utc).year))
    season_type = _env("CFB_SEASON_TYPE", "regular").lower()
    if season_type not in ("regular", "postseason"):
        season_type = "regular"

    df = build_weather(year, season_type, api_key)

    # Write outputs
    df.to_csv("weather_raw.csv", index=False)

    # Simple latest = same as raw (one row per game)
    df.to_csv("weather_latest.csv", index=False)

    # Log
    with open("logs_weather_run.txt", "w", encoding="utf-8") as f:
        f.write(f"year={year}\n")
        f.write(f"season_type={season_type}\n")
        f.write(f"rows={len(df)}\n")
        # Missingness summary
        for col in ["weather","temperature","wind_speed","wind_direction","precipitation"]:
            if col in df.columns:
                pct = float(df[col].isna().mean()) if len(df) else 1.0
                f.write(f"{col}_pct_missing={pct:.6f}\n")

if __name__ == "__main__":
    main()

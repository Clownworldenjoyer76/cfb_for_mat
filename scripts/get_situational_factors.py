import os
import time
import math
import datetime as dt
from typing import Dict, Any, Optional

import requests
import pandas as pd

API_BASE = "https://api.collegefootballdata.com"

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

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    try:
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return float("nan")
        R = 6371.0088
        phi1 = math.radians(float(lat1))
        phi2 = math.radians(float(lat2))
        dphi = phi2 - phi1
        dl = math.radians(float(lon2) - float(lon1))
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    except Exception:
        return float("nan")

def build_situational(year: str, season_type: str, api_key: str) -> pd.DataFrame:
    games_json = _get("/games", {"year": year, "seasonType": season_type}, api_key)
    df_games = _to_df(games_json)

    if "id" in df_games.columns and "game_id" not in df_games.columns:
        df_games = df_games.rename(columns={"id": "game_id"})
    if "startDate" in df_games.columns and "start_date" not in df_games.columns:
        df_games = df_games.rename(columns={"startDate": "start_date"})

    if "home_team" not in df_games.columns:
        for c in ["homeTeam", "homeTeam.school", "homeTeam.name"]:
            if c in df_games.columns:
                df_games["home_team"] = df_games[c]
                break
    if "away_team" not in df_games.columns:
        for c in ["awayTeam", "awayTeam.school", "awayTeam.name"]:
            if c in df_games.columns:
                df_games["away_team"] = df_games[c]
                break

    keep_cols = [c for c in ["game_id","season","week","start_date","home_team","away_team"] if c in df_games.columns]
    base = df_games[keep_cols].copy()
    if "start_date" in base.columns:
        base["start_date"] = pd.to_datetime(base["start_date"], errors="coerce", utc=True)

    home_rows = base.copy()
    home_rows["team"] = home_rows["home_team"]
    home_rows["opponent"] = home_rows["away_team"]
    home_rows["is_home"] = 1

    away_rows = base.copy()
    away_rows["team"] = away_rows["away_team"]
    away_rows["opponent"] = away_rows["home_team"]
    away_rows["is_home"] = 0

    team_games = pd.concat([home_rows, away_rows], ignore_index=True)

    if "venue.latitude" in df_games.columns and "venue.longitude" in df_games.columns:
        vg = df_games[["game_id","venue.latitude","venue.longitude"]].drop_duplicates("game_id")
        vg = vg.rename(columns={"venue.latitude":"venue_lat","venue.longitude":"venue_lon"})
        team_games = team_games.merge(vg, on="game_id", how="left")
    else:
        team_games["venue_lat"] = float("nan")
        team_games["venue_lon"] = float("nan")

    teams_json = _get("/teams/fbs", {"year": year}, api_key)
    df_teams = _to_df(teams_json)
    if "school" in df_teams.columns and "team" not in df_teams.columns:
        df_teams = df_teams.rename(columns={"school":"team"})
    if "location.latitude" in df_teams.columns and "team_lat" not in df_teams.columns:
        df_teams = df_teams.rename(columns={"location.latitude":"team_lat"})
    if "location.longitude" in df_teams.columns and "team_lon" not in df_teams.columns:
        df_teams = df_teams.rename(columns={"location.longitude":"team_lon"})
    team_loc = df_teams[["team","team_lat","team_lon"]].dropna().drop_duplicates() if {"team","team_lat","team_lon"}.issubset(df_teams.columns) else pd.DataFrame()

    if not team_loc.empty:
        team_games = team_games.merge(team_loc, on="team", how="left")
    else:
        team_games["team_lat"] = float("nan")
        team_games["team_lon"] = float("nan")

    try:
        static_path = os.path.join("data","fbs_fcs_stadium_coordinates.csv")
        if os.path.exists(static_path):
            df_static = pd.read_csv(static_path)
            df_static = df_static.rename(columns={"team":"team_name","latitude":"static_lat","longitude":"static_lon"})
            df_static["team_name"] = df_static["team_name"].astype(str).str.strip()
            team_games["team_norm"] = team_games["team"].astype(str).str.strip()
            team_games = team_games.merge(df_static[["team_name","static_lat","static_lon"]],
                                          left_on="team_norm", right_on="team_name", how="left")
            team_games["team_lat"] = team_games["team_lat"].fillna(team_games["static_lat"])
            team_games["team_lon"] = team_games["team_lon"].fillna(team_games["static_lon"])
            team_games = team_games.drop(columns=["team_name","team_norm","static_lat","static_lon"])
    except Exception:
        pass

    home_coords = team_games.loc[team_games["is_home"]==1, ["game_id","team_lat","team_lon"]].drop_duplicates("game_id")
    home_coords = home_coords.rename(columns={"team_lat":"home_lat","team_lon":"home_lon"})
    team_games = team_games.merge(home_coords, on="game_id", how="left")
    team_games["venue_lat"] = team_games["venue_lat"].fillna(team_games["home_lat"])
    team_games["venue_lon"] = team_games["venue_lon"].fillna(team_games["home_lon"])

    team_games["travel_distance_km"] = team_games.apply(
        lambda r: 0.0 if r.get("is_home", 0)==1 else _haversine_km(r.get("team_lat"), r.get("team_lon"), r.get("venue_lat"), r.get("venue_lon")),
        axis=1
    )

    if "start_date" in team_games.columns:
        team_games = team_games.sort_values(["team","start_date","game_id"])
        team_games["prev_date"] = team_games.groupby("team")["start_date"].shift(1)
        team_games["rest_days"] = (team_games["start_date"] - team_games["prev_date"]).dt.total_seconds()/86400.0
    else:
        team_games["rest_days"] = float("nan")

    return team_games[["game_id","season","week","start_date","team","opponent","is_home","travel_distance_km","rest_days"]]

def main():
    api_key = _env("CFBD_API_KEY", "")
    year = _env("CFB_YEAR", str(dt.datetime.now(dt.timezone.utc).year))
    season_type = _env("CFB_SEASON_TYPE", "regular").lower()
    if season_type not in ("regular","postseason"):
        season_type = "regular"

    df = build_situational(year, season_type, api_key)
    df.to_csv("situational_factors.csv", index=False)

    with open("logs_situational_factors_run.txt","w",encoding="utf-8") as f:
        f.write("year="+str(year)+"\n")
        f.write("season_type="+season_type+"\n")
        f.write("rows="+str(len(df))+"\n")
        f.write("pct_missing_distance="+str(round(float(df["travel_distance_km"].isna().mean()),6))+"\n")
        f.write("pct_missing_rest="+str(round(float(df["rest_days"].isna().mean()),6))+"\n")

if __name__=="__main__":
    main()

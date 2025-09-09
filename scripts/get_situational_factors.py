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

def _first_present(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def build_situational(year: str, season_type: str, api_key: str) -> pd.DataFrame:
    games_json = _get("/games", {"year": year, "seasonType": season_type}, api_key)
    df_games = _to_df(games_json)

    # Normalize common field names
    rename_pairs = {
        "id": "game_id",
        "startDate": "start_date",
        "seasonType": "season_type",
        "homeTeam": "homeTeam",   # may be object; keep as-is for dot columns
        "awayTeam": "awayTeam",
    }
    for src, dst in rename_pairs.items():
        if src in df_games.columns and dst not in df_games.columns:
            df_games = df_games.rename(columns={src: dst})

    # Derive home_team / away_team text from possible locations
    if "home_team" not in df_games.columns:
        home_name_col = _first_present(df_games, ["home_team", "homeTeam", "homeTeam.school", "homeTeam.name", "homeTeam.team"])
        if home_name_col is not None:
            df_games["home_team"] = df_games[home_name_col]
    if "away_team" not in df_games.columns:
        away_name_col = _first_present(df_games, ["away_team", "awayTeam", "awayTeam.school", "awayTeam.name", "awayTeam.team"])
        if away_name_col is not None:
            df_games["away_team"] = df_games[away_name_col]

    # Ensure game_id exists
    if "game_id" not in df_games.columns and "id" in df_games.columns:
        df_games = df_games.rename(columns={"id": "game_id"})

    # Venues and teams for coordinates
    venues_json = _get("/venues", {}, api_key)
    df_venues = _to_df(venues_json)
    if "id" in df_venues.columns and "venue_id" not in df_venues.columns:
        df_venues = df_venues.rename(columns={"id": "venue_id"})
    if "latitude" in df_venues.columns and "venue_lat" not in df_venues.columns:
        df_venues = df_venues.rename(columns={"latitude": "venue_lat"})
    if "longitude" in df_venues.columns and "venue_lon" not in df_venues.columns:
        df_venues = df_venues.rename(columns={"longitude": "venue_lon"})

    teams_json = _get("/teams/fbs", {"year": year}, api_key)
    df_teams = _to_df(teams_json)
    if "school" in df_teams.columns and "team" not in df_teams.columns:
        df_teams = df_teams.rename(columns={"school": "team"})
    if "location.latitude" in df_teams.columns and "team_lat" not in df_teams.columns:
        df_teams = df_teams.rename(columns={"location.latitude": "team_lat"})
    if "location.longitude" in df_teams.columns and "team_lon" not in df_teams.columns:
        df_teams = df_teams.rename(columns={"location.longitude": "team_lon"})
    if "latitude" in df_teams.columns and "team_lat" not in df_teams.columns:
        df_teams = df_teams.rename(columns={"latitude": "team_lat"})
    if "longitude" in df_teams.columns and "team_lon" not in df_teams.columns:
        df_teams = df_teams.rename(columns={"longitude": "team_lon"})
    team_loc = df_teams[["team","team_lat","team_lon"]].dropna(subset=["team_lat","team_lon"]).drop_duplicates() if {"team","team_lat","team_lon"}.issubset(df_teams.columns) else pd.DataFrame(columns=["team","team_lat","team_lon"])

    # Pick venue id/coords from games
    venue_id_col = _first_present(df_games, ["venue_id","venue.id","venueId"])
    venue_lat_col = _first_present(df_games, ["venue.latitude","venue_lat","latitude"])
    venue_lon_col = _first_present(df_games, ["venue.longitude","venue_lon","longitude"])

    # Base columns
    keep_cols = [c for c in ["game_id","season","week","start_date","home_team","away_team"] if c in df_games.columns]
    base = df_games[keep_cols].copy()

    if "start_date" in base.columns:
        base["start_date"] = pd.to_datetime(base["start_date"], errors="coerce", utc=True)

    home_rows = base.copy()
    home_rows["team"] = home_rows["home_team"] if "home_team" in home_rows.columns else None
    home_rows["opponent"] = home_rows["away_team"] if "away_team" in home_rows.columns else None
    home_rows["is_home"] = 1

    away_rows = base.copy()
    away_rows["team"] = away_rows["away_team"] if "away_team" in away_rows.columns else None
    away_rows["opponent"] = away_rows["home_team"] if "home_team" in away_rows.columns else None
    away_rows["is_home"] = 0

    team_games = pd.concat([home_rows, away_rows], ignore_index=True)

    # Venue coordinates
    if venue_lat_col and venue_lon_col:
        vg = df_games[["game_id", venue_lat_col, venue_lon_col]].drop_duplicates("game_id")
        vg = vg.rename(columns={venue_lat_col: "venue_lat", venue_lon_col: "venue_lon"})
        team_games = team_games.merge(vg, on="game_id", how="left")
    elif venue_id_col and not df_venues.empty:
        vid = df_games[["game_id", venue_id_col]].drop_duplicates("game_id").rename(columns={venue_id_col: "venue_id"})
        team_games = team_games.merge(vid, on="game_id", how="left")
        team_games = team_games.merge(df_venues[["venue_id","venue_lat","venue_lon"]].drop_duplicates("venue_id"), on="venue_id", how="left")
    else:
        team_games["venue_lat"] = float("nan")
        team_games["venue_lon"] = float("nan")

    # Team home coordinates
    if not team_loc.empty:
        team_games = team_games.merge(team_loc, on="team", how="left")
    else:
        team_games["team_lat"] = float("nan")
        team_games["team_lon"] = float("nan")

    # Travel distance
    team_games["travel_distance_km"] = team_games.apply(
        lambda r: 0.0 if r.get("is_home", 0) == 1 else _haversine_km(r.get("team_lat"), r.get("team_lon"), r.get("venue_lat"), r.get("venue_lon")),
        axis=1
    )

    # Rest days
    if "start_date" in team_games.columns:
        team_games = team_games.sort_values(["team","start_date","game_id"])
        team_games["prev_date"] = team_games.groupby("team")["start_date"].shift(1)
        team_games["rest_days"] = (team_games["start_date"] - team_games["prev_date"]).dt.total_seconds() / 86400.0
    else:
        team_games["rest_days"] = float("nan")

    out_cols = [c for c in ["game_id","season","week","start_date","team","opponent","is_home","travel_distance_km","rest_days"] if c in team_games.columns]
    return team_games[out_cols].copy()

def main():
    api_key = _env("CFBD_API_KEY", "")
    year = _env("CFB_YEAR", str(dt.datetime.now(dt.timezone.utc).year))
    season_type = _env("CFB_SEASON_TYPE", "regular").lower()
    if season_type not in ("regular","postseason"):
        season_type = "regular"

    df = build_situational(year, season_type, api_key)
    df.to_csv("situational_factors.csv", index=False)

    with open("logs_situational_factors_run.txt", "w", encoding="utf-8") as f:
        f.write("year=" + str(year) + "\n")
        f.write("season_type=" + season_type + "\n")
        f.write("rows=" + str(len(df)) + "\n")
        try:
            f.write("pct_missing_distance=" + str(round(float(df["travel_distance_km"].isna().mean()), 6)) + "\n")
            f.write("pct_missing_rest=" + str(round(float(df["rest_days"].isna().mean()), 6)) + "\n")
        except Exception:
            pass

if __name__ == "__main__":
    main()

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
        # flatten dict-of-lists if needed
        if all(isinstance(v, list) for v in obj.values()):
            rows = []
            for k, vs in obj.items():
                for rec in vs:
                    if isinstance(rec, dict):
                        rec["_group_key"] = k
                        rows.append(rec)
            return pd.DataFrame(rows)
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
    # 1) Pull games (schedule)
    games_json = _get("/games", {"year": year, "seasonType": season_type}, api_key)
    df_games = _to_df(games_json)

    # Expected columns from CFBD games: id, season, week, start_date, home_team, away_team, venue_id (or venue.id), venue.latitude, venue.longitude
    # Normalize common variants
    if "id" in df_games.columns and "game_id" not in df_games.columns:
        df_games = df_games.rename(columns={"id": "game_id"})
    if "start_date" not in df_games.columns and "startDate" in df_games.columns:
        df_games = df_games.rename(columns={"startDate": "start_date"})
    # venue fields: try nested flatten or simple columns
    # If nested like venue.id / venue.latitude / venue.longitude, pandas json_normalize already flattened
    venue_id_col = None
    for cand in ["venue_id", "venue.id", "venueId"]:
        if cand in df_games.columns:
            venue_id_col = cand
            break

    # 2) Pull venues for coordinates (fallback if games don’t include lat/lon)
    venues_json = _get("/venues", {}, api_key)
    df_venues = _to_df(venues_json)
    # Normalize venue id and coords
    if "id" in df_venues.columns and "venue_id" not in df_venues.columns:
        df_venues = df_venues.rename(columns={"id": "venue_id"})
    for old,new in [("latitude","venue_lat"),("longitude","venue_lon")]:
        if old in df_venues.columns and new not in df_venues.columns:
            df_venues = df_venues.rename(columns={old:new})

    # 3) Pull team locations (home lat/lon for distance calc)
    # CFBD provides team locations via /teams/fbs (by year) or /teams?year=...
    teams_json = _get("/teams/fbs", {"year": year}, api_key)
    df_teams = _to_df(teams_json)
    # Common location fields: location.latitude, location.longitude or latitude/longitude directly
    # Normalize to team_lat, team_lon and team name field "school" or "team"
    if "school" in df_teams.columns:
        df_teams = df_teams.rename(columns={"school": "team"})
    if "location.latitude" in df_teams.columns:
        df_teams = df_teams.rename(columns={"location.latitude": "team_lat"})
    if "location.longitude" in df_teams.columns:
        df_teams = df_teams.rename(columns={"location.longitude": "team_lon"})
    if "latitude" in df_teams.columns and "team_lat" not in df_teams.columns:
        df_teams = df_teams.rename(columns={"latitude": "team_lat"})
    if "longitude" in df_teams.columns and "team_lon" not in df_teams.columns:
        df_teams = df_teams.rename(columns={"longitude": "team_lon"})
    team_loc = df_teams[["team","team_lat","team_lon"]].dropna(subset=["team_lat","team_lon"]).drop_duplicates() if {"team","team_lat","team_lon"}.issubset(df_teams.columns) else pd.DataFrame(columns=["team","team_lat","team_lon"])

    # 4) Derive a team-game table (one row per team per game)
    keep_cols = ["game_id","season","week","start_date","home_team","away_team"]
    keep_cols = [c for c in keep_cols if c in df_games.columns]
    base = df_games[keep_cols].copy()

    # Ensure date parsing
    if "start_date" in base.columns:
        base["start_date"] = pd.to_datetime(base["start_date"], errors="coerce", utc=True)

    home_rows = base.copy()
    home_rows["team"] = home_rows["home_team"]
    home_rows["opponent"] = home_rows["away_team"] if "away_team" in home_rows.columns else None
    home_rows["is_home"] = 1

    away_rows = base.copy()
    away_rows["team"] = away_rows["away_team"]
    away_rows["opponent"] = away_rows["home_team"] if "home_team" in away_rows.columns else None
    away_rows["is_home"] = 0

    team_games = pd.concat([home_rows, away_rows], ignore_index=True)

    # 5) Attach venue coordinates for travel distance
    # Try to get lat/lon from games directly (venue.latitude, venue.longitude)
    if "venue.latitude" in df_games.columns and "venue.longitude" in df_games.columns:
        # Prefer direct columns if available
        team_games = team_games.merge(
            df_games[["game_id","venue.latitude","venue.longitude"]]\
                   .rename(columns={"venue.latitude":"venue_lat","venue.longitude":"venue_lon"})\
                   .drop_duplicates("game_id"),
            on="game_id", how="left"
        )
    elif venue_id_col and not df_venues.empty:
        # Join via venue id
        team_games = team_games.merge(
            df_games[["game_id", venue_id_col]].drop_duplicates("game_id").rename(columns={venue_id_col: "venue_id"}),
            on="game_id", how="left"
        )
        team_games = team_games.merge(
            df_venues[["venue_id","venue_lat","venue_lon"]].drop_duplicates("venue_id"),
            on="venue_id", how="left"
        )
    else:
        # No venue info available
        team_games["venue_lat"] = float("nan")
        team_games["venue_lon"] = float("nan")

    # 6) Attach team home coordinates
    if not team_loc.empty:
        team_games = team_games.merge(team_loc, on="team", how="left")
    else:
        team_games["team_lat"] = float("nan")
        team_games["team_lon"] = float("nan")

    # 7) Compute travel distance (away team only; home = 0)
    team_games["travel_distance_km"] = team_games.apply(
        lambda r: 0.0 if r.get("is_home", 0) == 1 else _haversine_km(r.get("team_lat"), r.get("team_lon"), r.get("venue_lat"), r.get("venue_lon")),
        axis=1
    )

    # 8) Compute rest days per team
    # Sort by team, then start_date and lag
    if "start_date" in team_games.columns:
        team_games = team_games.sort_values(["team","start_date","game_id"])
        team_games["prev_date"] = team_games.groupby("team")["start_date"].shift(1)
        team_games["rest_days"] = (team_games["start_date"] - team_games["prev_date"]).dt.total_seconds() / 86400.0
    else:
        team_games["rest_days"] = float("nan")

    # 9) Final output columns
    out_cols = []
    for c in ["game_id","season","week","start_date","team","opponent","is_home","travel_distance_km","rest_days"]:
        if c in team_games.columns:
            out_cols.append(c)
    df_out = team_games[out_cols].copy()

    return df_out

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
        # quick aggregates
        try:
            f.write("pct_missing_distance=" + str(round(float(df["travel_distance_km"].isna().mean()), 6)) + "\n")
            f.write("pct_missing_rest=" + str(round(float(df["rest_days"].isna().mean()), 6)) + "\n")
        except Exception:
            pass

if __name__ == "__main__":
    main()

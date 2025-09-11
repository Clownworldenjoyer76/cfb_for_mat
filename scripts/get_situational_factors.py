import os
import time
import math
import re
import datetime as dt
from typing import Dict, Any, Optional, List

import requests
import pandas as pd

API_BASE = "https://api.collegefootballdata.com"

# -------------------------
# ENV
# -------------------------
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "", "None") else default

# -------------------------
# HTTP
# -------------------------
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

# -------------------------
# NAME NORMALIZATION / ALIASES
# -------------------------
_norm_re = re.compile(r"[\W_]+", re.UNICODE)

def _norm_name(x: Any) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.lower()
    s = re.sub(r"\([^)]*\)", "", s)          # drop parenthetical (e.g., (FL))
    s = s.replace("&", "and")
    s = s.replace("university of ", "").replace("univ. of ", "").replace("univ of ", "")
    s = s.replace("st.", "state").replace(" st ", " state ")
    s = s.replace(" - ", " ").replace("-", " ")
    s = s.replace("'", "").replace("’", "")
    s = _norm_re.sub(" ", s)
    s = " ".join(s.split())
    return s

def _load_alias_map() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for p in ("mappings/team_aliases.csv", "team_aliases.csv"):
        if os.path.exists(p):
            df = pd.read_csv(p)
            if {"cfbd_name", "alias"}.issubset(df.columns):
                for _, r in df.iterrows():
                    k = _norm_name(r["cfbd_name"])
                    v = _norm_name(r["alias"])
                    if k:
                        m[k] = v if v else k
    return m

def _apply_alias_series(s: pd.Series, alias_map: Dict[str, str]) -> pd.Series:
    return s.astype(str).map(lambda x: alias_map.get(_norm_name(x), _norm_name(x)))

# -------------------------
# GEO
# -------------------------
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

# -------------------------
# BUILD
# -------------------------
def build_situational(year: str, season_type: str, api_key: str) -> pd.DataFrame:
    alias_map = _load_alias_map()

    # Games
    games_json = _get("/games", {"year": year, "seasonType": season_type}, api_key)
    df_games = _to_df(games_json)

    if "id" in df_games.columns and "game_id" not in df_games.columns:
        df_games = df_games.rename(columns={"id": "game_id"})
    if "startDate" in df_games.columns and "start_date" not in df_games.columns:
        df_games = df_games.rename(columns={"startDate": "start_date"})

    # Home/Away fields
    if "home_team" not in df_games.columns:
        for c in ["homeTeam", "homeTeam.school", "homeTeam.name", "homeTeam.team", "home_team"]:
            if c in df_games.columns:
                df_games["home_team"] = df_games[c]
                break
    if "away_team" not in df_games.columns:
        for c in ["awayTeam", "awayTeam.school", "awayTeam.name", "awayTeam.team", "away_team"]:
            if c in df_games.columns:
                df_games["away_team"] = df_games[c]
                break

    # Venue coords direct in /games
    if "venue.latitude" in df_games.columns and "venue.longitude" in df_games.columns:
        df_games = df_games.rename(columns={"venue.latitude": "venue_lat", "venue.longitude": "venue_lon"})
    else:
        df_games["venue_lat"] = float("nan")
        df_games["venue_lon"] = float("nan")

    keep_cols = [c for c in ["game_id","season","week","start_date","home_team","away_team","venue_lat","venue_lon"] if c in df_games.columns]
    base = df_games[keep_cols].copy()
    if "start_date" in base.columns:
        base["start_date"] = pd.to_datetime(base["start_date"], errors="coerce", utc=True)

    # Normalize team labels
    for col in ["home_team", "away_team"]:
        if col in base.columns:
            base[col] = _apply_alias_series(base[col], alias_map)

    # Try venue lookup from /venues if missing
    missing_venue = base["venue_lat"].isna() | base["venue_lon"].isna()
    if missing_venue.any():
        venues_json = _get("/venues", {}, api_key)
        df_venues = _to_df(venues_json)
        name_col = None
        for c in ["name", "venue", "stadium", "full_name"]:
            if c in df_venues.columns:
                name_col = c
                break
        game_vname_col = None
        for c in ["venue", "venue.name", "venueName", "venue_name"]:
            if c in df_games.columns:
                game_vname_col = c
                break
        if name_col and game_vname_col and {"latitude","longitude"}.issubset(df_venues.columns):
            v = df_venues[[name_col,"latitude","longitude"]].dropna().copy()
            v["__venue_key"] = v[name_col].astype(str).map(_norm_name)
            g = df_games[["game_id", game_vname_col]].drop_duplicates("game_id").copy()
            g["__venue_key"] = g[game_vname_col].astype(str).map(_norm_name)
            v = v.rename(columns={"latitude":"_v_lat","longitude":"_v_lon"})
            base = base.merge(g[["game_id","__venue_key"]], on="game_id", how="left")
            base = base.merge(v[["__venue_key","_v_lat","_v_lon"]].drop_duplicates("__venue_key"), on="__venue_key", how="left")
            base["venue_lat"] = base["venue_lat"].fillna(base["_v_lat"])
            base["venue_lon"] = base["venue_lon"].fillna(base["_v_lon"])
            base = base.drop(columns=["__venue_key","_v_lat","_v_lon"], errors="ignore")

    # Team home coords from CFBD
    teams_json = _get("/teams/fbs", {"year": year}, api_key)
    df_teams = _to_df(teams_json)
    if "school" in df_teams.columns and "team" not in df_teams.columns:
        df_teams = df_teams.rename(columns={"school":"team"})
    if "location.latitude" in df_teams.columns and "team_lat" not in df_teams.columns:
        df_teams = df_teams.rename(columns={"location.latitude":"team_lat"})
    if "location.longitude" in df_teams.columns and "team_lon" not in df_teams.columns:
        df_teams = df_teams.rename(columns={"location.longitude":"team_lon"})
    if "team" in df_teams.columns:
        df_teams["team"] = _apply_alias_series(df_teams["team"], alias_map)
    team_loc = df_teams[["team","team_lat","team_lon"]].dropna().drop_duplicates() if {"team","team_lat","team_lon"}.issubset(df_teams.columns) else pd.DataFrame()

    # Build per-team rows
    home_rows = base.copy()
    home_rows["team"] = home_rows["home_team"]
    home_rows["opponent"] = home_rows["away_team"]
    home_rows["is_home"] = 1

    away_rows = base.copy()
    away_rows["team"] = away_rows["away_team"]
    away_rows["opponent"] = away_rows["home_team"]
    away_rows["is_home"] = 0

    team_games = pd.concat([home_rows, away_rows], ignore_index=True)

    # Merge team CFBD coords
    if not team_loc.empty:
        team_games = team_games.merge(team_loc, on="team", how="left")
    else:
        team_games["team_lat"] = float("nan")
        team_games["team_lon"] = float("nan")

    # Static stadium fallback
    try:
        static_path = os.path.join("data", "fbs_fcs_stadium_coordinates.csv")
        if os.path.exists(static_path):
            df_static = pd.read_csv(static_path)
            if {"team","latitude","longitude"}.issubset(df_static.columns):
                df_static = df_static.rename(columns={"latitude":"static_lat","longitude":"static_lon"})
                df_static["_team_key"] = df_static["team"].astype(str).map(_norm_name)
                team_games["_team_key"] = team_games["team"].astype(str).map(_norm_name)
                team_games = team_games.merge(
                    df_static[["_team_key","static_lat","static_lon"]].drop_duplicates("_team_key"),
                    on="_team_key", how="left"
                )
                # fill missing team coords
                team_games["team_lat"] = team_games["team_lat"].fillna(team_games["static_lat"])
                team_games["team_lon"] = team_games["team_lon"].fillna(team_games["static_lon"])
                # fill missing VENUE coords for home team from static
                home_fill = team_games["is_home"] == 1
                team_games.loc[home_fill, "venue_lat"] = team_games.loc[home_fill, "venue_lat"].fillna(team_games.loc[home_fill, "static_lat"])
                team_games.loc[home_fill, "venue_lon"] = team_games.loc[home_fill, "venue_lon"].fillna(team_games.loc[home_fill, "static_lon"])
                team_games = team_games.drop(columns=["_team_key","static_lat","static_lon"], errors="ignore")
    except Exception:
        pass

    # As last resort: set venue coords = home team coords
    home_coords = team_games.loc[team_games["is_home"]==1, ["game_id","team_lat","team_lon"]].drop_duplicates("game_id")
    home_coords = home_coords.rename(columns={"team_lat":"home_lat","team_lon":"home_lon"})
    team_games = team_games.merge(home_coords, on="game_id", how="left")
    team_games["venue_lat"] = team_games["venue_lat"].fillna(team_games["home_lat"])
    team_games["venue_lon"] = team_games["venue_lon"].fillna(team_games["home_lon"])
    team_games = team_games.drop(columns=["home_lat","home_lon"], errors="ignore")

    # Distance
    team_games["travel_distance_km"] = team_games.apply(
        lambda r: 0.0 if r.get("is_home",0)==1 else _haversine_km(r.get("team_lat"), r.get("team_lon"), r.get("venue_lat"), r.get("venue_lon")),
        axis=1
    )

    # Rest days
    if "start_date" in team_games.columns:
        team_games = team_games.sort_values(["team","start_date","game_id"])
        team_games["prev_date"] = team_games.groupby("team")["start_date"].shift(1)
        team_games["rest_days"] = (team_games["start_date"] - team_games["prev_date"]).dt.total_seconds()/86400.0
        team_games = team_games.drop(columns=["prev_date"])
    else:
        team_games["rest_days"] = float("nan")

    # Output columns (ensure lat/lon are present)
    out_cols: List[str] = [
        "game_id","season","week","start_date","team","opponent","is_home",
        "venue_lat","venue_lon","team_lat","team_lon",
        "travel_distance_km","rest_days"
    ]
    for c in out_cols:
        if c not in team_games.columns:
            team_games[c] = pd.NA

    return team_games[out_cols]

# -------------------------
# MAIN
# -------------------------
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
        f.write("pct_missing_venue_latlon="+str(round(float(df["venue_lat"].isna().mean() if len(df) else 1.0),6))+"\n")
        f.write("pct_missing_team_latlon="+str(round(float(df["team_lat"].isna().mean() if len(df) else 1.0),6))+"\n")
        f.write("pct_missing_distance="+str(round(float(df["travel_distance_km"].isna().mean() if len(df) else 1.0),6))+"\n")
        f.write("pct_missing_rest="+str(round(float(df["rest_days"].isna().mean() if len(df) else 1.0),6))+"\n")

if __name__=="__main__":
    main()

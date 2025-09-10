import os
import time
import math
import re
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

_norm_re = re.compile(r"[\W_]+", re.UNICODE)
def _norm_name(x: Any) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.lower()
    s = re.sub(r"\([^)]*\)", "", s)          # drop parenthetical (e.g., (fl))
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
            if {"cfbd_name","alias"}.issubset(df.columns):
                for _, r in df.iterrows():
                    k = _norm_name(r["cfbd_name"])
                    v = _norm_name(r["alias"])
                    if k:
                        m[k] = v if v else k
    return m

def _apply_alias_series(s: pd.Series, alias_map: Dict[str, str]) -> pd.Series:
    return s.astype(str).map(lambda x: alias_map.get(_norm_name(x), _norm_name(x)))

def build_situational(year: str, season_type: str, api_key: str) -> pd.DataFrame:
    alias_map = _load_alias_map()

    # games
    games_json = _get("/games", {"year": year, "seasonType": season_type}, api_key)
    df_games = _to_df(games_json)

    if "id" in df_games.columns and "game_id" not in df_games.columns:
        df_games = df_games.rename(columns={"id": "game_id"})
    if "startDate" in df_games.columns and "start_date" not in df_games.columns:
        df_games = df_games.rename(columns={"startDate": "start_date"})

    # candidate home/away fields
    if "home_team" not in df_games.columns:
        for c in ["homeTeam", "homeTeam.school", "homeTeam.name", "homeTeam.team"]:
            if c in df_games.columns:
                df_games["home_team"] = df_games[c]
                break
    if "away_team" not in df_games.columns:
        for c in ["awayTeam", "awayTeam.school", "awayTeam.name", "awayTeam.team"]:
            if c in df_games.columns:
                df_games["away_team"] = df_games[c]
                break

    # candidate venue name/coords (for venue->lat/lon join later)
    venue_name_col = None
    for c in ["venue", "venue.name", "venueName", "venue_name"]:
        if c in df_games.columns:
            venue_name_col = c
            break

    keep_cols = [c for c in ["game_id","season","week","start_date","home_team","away_team"] if c in df_games.columns]
    base = df_games[keep_cols].copy()
    if "start_date" in base.columns:
        base["start_date"] = pd.to_datetime(base["start_date"], errors="coerce", utc=True)

    # normalize and alias team labels
    for col in ["home_team", "away_team"]:
        if col in base.columns:
            base[col] = _apply_alias_series(base[col], alias_map)

    home_rows = base.copy()
    home_rows["team"] = home_rows["home_team"]
    home_rows["opponent"] = home_rows["away_team"]
    home_rows["is_home"] = 1

    away_rows = base.copy()
    away_rows["team"] = away_rows["away_team"]
    away_rows["opponent"] = away_rows["home_team"]
    away_rows["is_home"] = 0

    team_games = pd.concat([home_rows, away_rows], ignore_index=True)

    # venue coords via games payload, else NaN
    if "venue.latitude" in df_games.columns and "venue.longitude" in df_games.columns:
        vg = df_games[["game_id","venue.latitude","venue.longitude"]].drop_duplicates("game_id")
        vg = vg.rename(columns={"venue.latitude":"venue_lat","venue.longitude":"venue_lon"})
        team_games = team_games.merge(vg, on="game_id", how="left")
    else:
        team_games["venue_lat"] = float("nan")
        team_games["venue_lon"] = float("nan")

    # fallback: match venue by name to /venues
    venues_json = _get("/venues", {}, api_key)
    df_venues = _to_df(venues_json)
    if not df_venues.empty:
        # normalize venue name and coords
        name_col = None
        for c in ["name", "venue", "stadium", "full_name"]:
            if c in df_venues.columns:
                name_col = c
                break
        if name_col:
            df_venues["__venue_key"] = df_venues[name_col].astype(str).map(_norm_name)
            if "latitude" in df_venues.columns and "longitude" in df_venues.columns:
                df_venues = df_venues.rename(columns={"latitude":"_v_lat","longitude":"_v_lon"})
                if venue_name_col:
                    vmap = df_games[["game_id", venue_name_col]].drop_duplicates("game_id").copy()
                    vmap["__venue_key"] = vmap[venue_name_col].astype(str).map(_norm_name)
                    team_games = team_games.merge(vmap[["game_id","__venue_key"]], on="game_id", how="left")
                    team_games = team_games.merge(
                        df_venues[["__venue_key","_v_lat","_v_lon"]].drop_duplicates("__venue_key"),
                        on="__venue_key", how="left"
                    )
                    # fill venue from name-joined coords if missing
                    team_games["venue_lat"] = team_games["venue_lat"].fillna(team_games["_v_lat"])
                    team_games["venue_lon"] = team_games["venue_lon"].fillna(team_games["_v_lon"])
                    team_games = team_games.drop(columns=["__venue_key","_v_lat","_v_lon"], errors="ignore")

    # team home coords from CFBD
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

    if not team_loc.empty:
        team_games = team_games.merge(team_loc, on="team", how="left")
    else:
        team_games["team_lat"] = float("nan")
        team_games["team_lon"] = float("nan")

    # static stadium fallback: normalize and join on team
    try:
        static_path = os.path.join("data","fbs_fcs_stadium_coordinates.csv")
        if os.path.exists(static_path):
            df_static = pd.read_csv(static_path)
            if "team" in df_static.columns:
                df_static["_team_key"] = df_static["team"].astype(str).map(_norm_name)
                df_static = df_static.rename(columns={"latitude":"static_lat","longitude":"static_lon"})
                team_games["_team_key"] = team_games["team"].astype(str).map(_norm_name)
                team_games = team_games.merge(
                    df_static[["_team_key","static_lat","static_lon"]].drop_duplicates("_team_key"),
                    on="_team_key", how="left"
                )
                team_games["team_lat"] = team_games["team_lat"].fillna(team_games["static_lat"])
                team_games["team_lon"] = team_games["team_lon"].fillna(team_games["static_lon"])
                team_games = team_games.drop(columns=["_team_key","static_lat","static_lon"])
    except Exception:
        pass

    # home venue fallback = home team coords
    home_coords = team_games.loc[team_games["is_home"]==1, ["game_id","team_lat","team_lon"]].drop_duplicates("game_id")
    home_coords = home_coords.rename(columns={"team_lat":"home_lat","team_lon":"home_lon"})
    team_games = team_games.merge(home_coords, on="game_id", how="left")
    team_games["venue_lat"] = team_games["venue_lat"].fillna(team_games["home_lat"])
    team_games["venue_lon"] = team_games["venue_lon"].fillna(team_games["home_lon"])

    # distance
    team_games["travel_distance_km"] = team_games.apply(
        lambda r: 0.0 if r.get("is_home", 0)==1 else _haversine_km(r.get("team_lat"), r.get("team_lon"), r.get("venue_lat"), r.get("venue_lon")),
        axis=1
    )

    # rest days
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

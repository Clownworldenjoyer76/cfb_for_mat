#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Derive situational factors (rest, travel, tz diff, altitude diff, home/away)
# from data/raw/games.csv and data/reference/stadiums.csv

import math
from datetime import datetime, date, time
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

# ---------- helper functions ----------

def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return float("nan")
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def tz_offset_hours(tz_str, on_date):
    if not tz_str or pd.isna(tz_str):
        return float("nan")
    try:
        tz = ZoneInfo(tz_str)
        dt = datetime.combine(on_date, time(12, 0)).replace(tzinfo=tz)
        return dt.utcoffset().total_seconds() / 3600.0
    except Exception:
        return float("nan")

def parse_date(x):
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None

# ---------- paths ----------

stad_path = Path("data/reference/stadiums.csv")
games_path = Path("data/raw/games.csv")
out_path = Path("data/processed/situational_factors.csv")
log_path = Path("summaries/situational_factors_summary.txt")

# ---------- load ----------

if not stad_path.exists() or not games_path.exists():
    raise SystemExit("[derive] Missing required input files.")

stad = pd.read_csv(stad_path)
games = pd.read_csv(games_path)

stad.columns = stad.columns.str.lower().str.strip()
games.columns = games.columns.str.lower().str.strip()

stad["lat"] = pd.to_numeric(stad["lat"], errors="coerce")
stad["lon"] = pd.to_numeric(stad["lon"], errors="coerce")
stad["altitude_m"] = pd.to_numeric(stad["altitude_m"], errors="coerce")

# ---------- prep games ----------

games["game_date"] = games["date"].apply(parse_date)
games["neutral_site"] = games["neutral_site"].fillna(0).astype(int)

# ---------- join stadiums for home/away teams ----------

home = stad.rename(columns={
    "team": "home_team",
    "lat": "home_lat",
    "lon": "home_lon",
    "timezone": "home_tz",
    "altitude_m": "home_alt_m"
})
away = stad.rename(columns={
    "team": "away_team",
    "lat": "away_lat",
    "lon": "away_lon",
    "timezone": "away_tz",
    "altitude_m": "away_alt_m"
})

df = (
    games
    .merge(home, on="home_team", how="left")
    .merge(away, on="away_team", how="left", suffixes=("", "_away"))
)

# ---------- compute per-team rows ----------

# explode into team perspective rows
home_rows = df.copy()
home_rows["team"] = home_rows["home_team"]
home_rows["opponent"] = home_rows["away_team"]
home_rows["is_home"] = 1
home_rows["is_away"] = 0
home_rows["is_neutral"] = home_rows["neutral_site"]

away_rows = df.copy()
away_rows["team"] = away_rows["away_team"]
away_rows["opponent"] = away_rows["home_team"]
away_rows["is_home"] = 0
away_rows["is_away"] = 1
away_rows["is_neutral"] = home_rows["neutral_site"]

tg = pd.concat([home_rows, away_rows], ignore_index=True)

# ---------- compute travel_km ----------

tg["travel_km"] = tg.apply(lambda r: (
    0.0 if r["is_home"] == 1 and r["is_neutral"] == 0
    else haversine_km(r.get("home_lat", r.get("away_lat")), r.get("home_lon", r.get("away_lon")),
                      r.get("away_lat", r.get("home_lat")), r.get("away_lon", r.get("home_lon")))
), axis=1)

# ---------- compute rest days ----------

tg = tg.sort_values(["team", "game_date"])
tg["prev_date"] = tg.groupby("team")["game_date"].shift(1)
tg["rest_days"] = (tg["game_date"] - tg["prev_date"]).dt.days
tg["bye_week"] = (tg["rest_days"] >= 13).astype(int)

# ---------- compute timezone and altitude differences ----------

tg["tz_home_offset"] = tg.apply(lambda r: tz_offset_hours(r.get("home_tz"), r["game_date"]), axis=1)
tg["tz_away_offset"] = tg.apply(lambda r: tz_offset_hours(r.get("away_tz"), r["game_date"]), axis=1)
tg["tz_diff_from_home"] = tg["tz_away_offset"] - tg["tz_home_offset"]

tg["altitude_diff_m"] = tg["away_alt_m"] - tg["home_alt_m"]

# ---------- select outputs ----------

out_cols = [
    "season", "week", "game_id", "game_date", "team", "opponent",
    "is_home", "is_away", "is_neutral",
    "rest_days", "bye_week", "travel_km",
    "tz_diff_from_home", "altitude_diff_m"
]

out = tg[out_cols].copy()

# ---------- write ----------

out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)

# ---------- summary ----------

summary = {
    "total_rows": len(out),
    "teams": out["team"].nunique(),
    "games": out["game_id"].nunique(),
    "avg_travel_km": round(out["travel_km"].mean(), 2),
    "avg_rest_days": round(out["rest_days"].dropna().mean(), 2),
    "bye_weeks": int(out["bye_week"].sum())
}

log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, "w") as f:
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")

print("[derive] situational_factors.csv written.")
for k, v in summary.items():
    print(f"  {k}: {v}") in

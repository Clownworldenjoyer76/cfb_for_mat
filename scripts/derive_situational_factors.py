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
    """Return UTC offset (hours) for tz on the given calendar date."""
    if not tz_str or pd.isna(tz_str) or on_date is None or pd.isna(on_date):
        return float("nan")
    try:
        if hasattr(on_date, "date"):  # pandas Timestamp
            on_date = on_date.date()
        tz = ZoneInfo(tz_str)
        dt = datetime.combine(on_date, time(12, 0)).replace(tzinfo=tz)
        return dt.utcoffset().total_seconds() / 3600.0
    except Exception:
        return float("nan")

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

# Ensure true datetime dtype for later .dt operations
games["game_date"] = pd.to_datetime(games["date"], errors="coerce")
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
away_rows["is_neutral"] = df["neutral_site"]

tg = pd.concat([home_rows, away_rows], ignore_index=True)

# ---------- compute travel_km ----------
# For campus games (neutral_site==0), venue is home team's stadium.
# Distance = great-circle(team_home -> venue). Home rows -> 0.

def travel_row(r):
    if r["is_neutral"] == 0:
        # venue = home team's stadium
        venue_lat, venue_lon = r["home_lat"], r["home_lon"]
    else:
        # If you later add neutral venues, plug them here.
        venue_lat, venue_lon = r["home_lat"], r["home_lon"]  # fallback

    team_lat = r["home_lat"] if r["is_home"] == 1 else r["away_lat"]
    team_lon = r["home_lon"] if r["is_home"] == 1 else r["away_lon"]

    if r["is_home"] == 1 and r["is_neutral"] == 0:
        return 0.0
    return haversine_km(team_lat, team_lon, venue_lat, venue_lon)

tg["travel_km"] = tg.apply(travel_row, axis=1)

# ---------- compute rest days ----------

tg = tg.sort_values(["team", "game_date"])
tg["prev_date"] = tg.groupby("team")["game_date"].shift(1)

# Ensure datetime dtype before using .dt
tg["game_date"] = pd.to_datetime(tg["game_date"], errors="coerce")
tg["prev_date"] = pd.to_datetime(tg["prev_date"], errors="coerce")

tg["rest_days"] = (tg["game_date"] - tg["prev_date"]).dt.days
tg["bye_week"] = (tg["rest_days"] >= 13).fillna(False).astype(int)

# ---------- compute timezone and altitude differences ----------
# Venue tz = home_tz (campus games assumption). Team tz = home_tz for home rows, away_tz for away rows.

tg["venue_offset"] = tg.apply(lambda r: tz_offset_hours(r.get("home_tz"), r["game_date"]), axis=1)
tg["team_home_offset"] = tg.apply(
    lambda r: tz_offset_hours(r.get("home_tz") if r["is_home"] == 1 else r.get("away_tz"), r["game_date"]),
    axis=1
)
tg["tz_diff_from_home"] = tg["venue_offset"] - tg["team_home_offset"]

# Altitude: venue altitude - team home altitude
team_alt = tg["home_alt_m"].where(tg["is_home"] == 1, tg["away_alt_m"])
venue_alt = tg["home_alt_m"]  # campus venue assumption
tg["altitude_diff_m"] = venue_alt - team_alt

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
    "avg_travel_km": round(out["travel_km"].mean(skipna=True), 2),
    "avg_rest_days": round(out["rest_days"].dropna().mean() if out["rest_days"].notna().any() else 0, 2),
    "bye_weeks": int(out["bye_week"].sum())
}

log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, "w") as f:
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")

print("[derive] situational_factors.csv written.")
for k, v in summary.items():
    print(f"  {k}: {v}")

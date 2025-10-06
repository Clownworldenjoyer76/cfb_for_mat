#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Build per-team, per-game situational factors using finalized stadiums and a games table.
#
# Inputs (required):
#   - data/reference/stadiums.csv
#       columns (lowercase): team, venue, city, state, country, lat, lon, timezone, altitude_m
#   - data/raw/games.csv
#       expected columns (lowercase):
#         season, week, game_id, date (YYYY-MM-DD),
#         home_team, away_team, neutral_site (0/1)
#       optional (if present, used to override venue resolution):
#         venue, venue_lat, venue_lon, venue_timezone, venue_altitude_m,
#         kickoff_utc (ISO 8601, e.g. 2024-09-07T19:30:00Z)
#
# Inputs (optional):
#   - data/reference/venues_override.csv
#       columns: venue, city, state, country, lat, lon, timezone, altitude_m
#
# Outputs:
#   - data/processed/situational_factors.csv  (one row per team-game)
#   - summaries/logs_situational_factors_run.txt

import math
from pathlib import Path
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

# ---------- helpers ----------

def to_num(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def haversine_km(lat1, lon1, lat2, lon2):
    # All args in decimal degrees
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return float("nan")
    r = 6371.0088  # mean Earth radius (km)
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c

def utc_offset_hours(tz_str: str, on_date: date) -> float:
    """Return UTC offset (hours) for a tz on the given calendar date (local noon)."""
    if not tz_str or str(tz_str).strip() == "":
        return float("nan")
    try:
        tz = ZoneInfo(tz_str)
        local_dt = datetime.combine(on_date, time(12, 0)).replace(tzinfo=tz)
        return local_dt.utcoffset().total_seconds() / 3600.0
    except Exception:
        return float("nan")

def coerce_bool01(x):
    try:
        i = int(x)
        return 1 if i == 1 else 0
    except Exception:
        s = str(x).strip().lower()
        if s in ("true", "yes", "y"):
            return 1
        if s in ("false", "no", "n"):
            return 0
        return 0

def parse_date(s):
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        # try ISO date component
        try:
            return datetime.fromisoformat(str(s)[:10]).date()
        except Exception:
            return None

# ---------- load inputs ----------

stad_path = Path("data/reference/stadiums.csv")
games_path = Path("data/raw/games.csv")
venues_override_path = Path("data/reference/venues_override.csv")

if not stad_path.exists():
    raise SystemExit(f"[situational] ERROR: missing {stad_path}")
if not games_path.exists():
    raise SystemExit(f"[situational] ERROR: missing {games_path}")

stad = pd.read_csv(stad_path)
games = pd.read_csv(games_path)
stad.columns = stad.columns.str.lower().str.strip()
games.columns = games.columns.str.lower().str.strip()

# Optional overrides
if venues_override_path.exists():
    venues_override = pd.read_csv(venues_override_path)
    venues_override.columns = venues_override.columns.str.lower().str.strip()
else:
    venues_override = pd.DataFrame(columns=["venue", "city", "state", "country", "lat", "lon", "timezone", "altitude_m"])

# Ensure required stadium columns
for c in ["team", "lat", "lon", "timezone", "altitude_m", "venue"]:
    if c not in stad.columns:
        stad[c] = pd.NA

stad["lat"] = pd.to_numeric(stad["lat"], errors="coerce")
stad["lon"] = pd.to_numeric(stad["lon"], errors="coerce")
stad["altitude_m"] = pd.to_numeric(stad["altitude_m"], errors="coerce")

# Ensure required game columns
required_game_cols = ["season", "week", "game_id", "date", "home_team", "away_team", "neutral_site"]
missing_req = [c for c in required_game_cols if c not in games.columns]
if missing_req:
    raise SystemExit(f"[situational] ERROR: games.csv missing required columns: {missing_req}")

# Normalize / parse
games["neutral_site"] = games["neutral_site"].apply(coerce_bool01)
games["game_date"] = games["date"].apply(parse_date)

# ---------- build "home base" per team ----------

home_cols = ["team", "lat", "lon", "timezone", "altitude_m", "venue"]
team_home = stad[home_cols].copy().rename(
    columns={
        "lat": "home_lat",
        "lon": "home_lon",
        "timezone": "home_tz",
        "altitude_m": "home_altitude_m",
        "venue": "home_venue"
    }
)

# ---------- resolve venue per game ----------

# Start with provided venue columns if any
for col in ["venue", "venue_lat", "venue_lon", "venue_timezone", "venue_altitude_m", "kickoff_utc"]:
    if col not in games.columns:
        games[col] = pd.NA

# If venue lat/lon missing:
# - if neutral_site == 0 => use home_team's stadium
# - else try venues_override by exact venue name
# - else (still missing) leave NaN and flag imputed
games = games.merge(
    team_home.add_prefix("home_team_"),
    how="left",
    left_on="home_team",
    right_on="home_team_team"
)
games = games.drop(columns=["home_team_team"], errors="ignore")

# Prefer explicit venue_* if present
games["venue_name"] = games["venue"]
games["venue_lat_res"] = pd.to_numeric(games["venue_lat"], errors="coerce")
games["venue_lon_res"] = pd.to_numeric(games["venue_lon"], errors="coerce")
games["venue_alt_res"] = pd.to_numeric(games["venue_altitude_m"], errors="coerce")
games["venue_tz_res"] = games["venue_timezone"]

# Fill from home stadium when not neutral
mask_need_coords = games["venue_lat_res"].isna() | games["venue_lon_res"].isna()
mask_home = (games["neutral_site"] == 0) & mask_need_coords
games.loc[mask_home, "venue_name"] = games.loc[mask_home, "home_team_home_venue"]
games.loc[mask_home, "venue_lat_res"] = games.loc[mask_home, "home_team_home_lat"]
games.loc[mask_home, "venue_lon_res"] = games.loc[mask_home, "home_team_home_lon"]
games.loc[mask_home, "venue_alt_res"] = games.loc[mask_home, "home_team_home_altitude_m"]
games.loc[mask_home, "venue_tz_res"] = games.loc[mask_home, "home_team_home_tz"]

# For remaining missing coords, try override by venue name
still_need = (games["venue_lat_res"].isna() | games["venue_lon_res"].isna()) & games["venue_name"].notna()
if not venues_override.empty and "venue" in venues_override.columns:
    vo = venues_override.copy()
    vo["venue"] = vo["venue"].astype(str).str.strip().str.lower()
    vo["lat"] = pd.to_numeric(vo.get("lat"), errors="coerce")
    vo["lon"] = pd.to_numeric(vo.get("lon"), errors="coerce")
    vo["altitude_m"] = pd.to_numeric(vo.get("altitude_m"), errors="coerce")
    vo = vo.rename(columns={"timezone": "tz"})
    # exact case-insensitive match
    games["_vn_key"] = games["venue_name"].astype(str).str.strip().str.lower()
    merged_vo = games[still_need].merge(vo, how="left", left_on="_vn_key", right_on="venue", suffixes=("", "_ov"))
    # Apply where found
    if not merged_vo.empty:
        idx = merged_vo.index
        games.loc[idx, "venue_lat_res"] = merged_vo["lat"].values
        games.loc[idx, "venue_lon_res"] = merged_vo["lon"].values
        games.loc[idx, "venue_alt_res"] = merged_vo["altitude_m"].values
        games.loc[idx, "venue_tz_res"] = merged_vo["tz"].values

games["venue_imputed"] = ((games["venue_lat_res"].isna()) | (games["venue_lon_res"].isna())).astype(int)

# ---------- explode games into two team-rows ----------

home_rows = games.copy()
home_rows["team"] = home_rows["home_team"]
home_rows["opponent"] = home_rows["away_team"]
home_rows["is_home"] = 1
home_rows["is_away"] = 0
home_rows["is_neutral"] = home_rows["neutral_site"]

away_rows = games.copy()
away_rows["team"] = away_rows["away_team"]
away_rows["opponent"] = away_rows["home_team"]
away_rows["is_home"] = 0
away_rows["is_away"] = 1
away_rows["is_neutral"] = away_rows["neutral_site"]

tg = pd.concat([home_rows, away_rows], ignore_index=True)

# Bring team home base
tg = tg.merge(team_home, how="left", on="team")

# Final venue columns for computation
tg["venue_lat"] = pd.to_numeric(tg["venue_lat_res"], errors="coerce")
tg["venue_lon"] = pd.to_numeric(tg["venue_lon_res"], errors="coerce")
tg["venue_altitude_m"] = pd.to_numeric(tg["venue_alt_res"], errors="coerce")
tg["venue_tz"] = tg["venue_tz_res"]

# ---------- compute rest_days per team ----------

tg = tg.sort_values(["team", "game_date", "game_id"], na_position="last")
tg["prev_date"] = tg.groupby("team")["game_date"].shift(1)
tg["rest_days"] = (tg["game_date"] - tg["prev_date"]).dt.days

# Define bye as 13+ days since last game
tg["bye_week"] = (tg["rest_days"] >= 13).astype(int)

# ---------- travel_km ----------

tg["home_lat"] = pd.to_numeric(tg["home_lat"], errors="coerce")
tg["home_lon"] = pd.to_numeric(tg["home_lon"], errors="coerce")
tg["home_altitude_m"] = pd.to_numeric(tg["home_altitude_m"], errors="coerce")

tg["travel_km"] = tg.apply(
    lambda r: 0.0 if (r["is_home"] == 1 and r["is_neutral"] == 0)
    else haversine_km(r["home_lat"], r["home_lon"], r["venue_lat"], r["venue_lon"]),
    axis=1
)

# ---------- timezone difference (venue - home) ----------

# Prefer kickoff_utc date if provided; else use game_date
def game_calendar_date(row):
    if isinstance(row.get("kickoff_utc"), str) and row["kickoff_utc"]:
        try:
            # Expecting 'Z' or offset
            dt_utc = datetime.fromisoformat(row["kickoff_utc"].replace("Z", "+00:00"))
            return dt_utc.date()
        except Exception:
            pass
    return row.get("game_date")

tg["tz_home_offset_hours"] = tg.apply(lambda r: utc_offset_hours(r["home_tz"], game_calendar_date(r)) if pd.notna(r["home_tz"]) and pd.notna(game_calendar_date(r)) else float("nan"), axis=1)
tg["tz_venue_offset_hours"] = tg.apply(lambda r: utc_offset_hours(r["venue_tz"], game_calendar_date(r)) if pd.notna(r["venue_tz"]) and pd.notna(game_calendar_date(r)) else float("nan"), axis=1)
tg["tz_diff_from_home"] = tg["tz_venue_offset_hours"] - tg["tz_home_offset_hours"]

# ---------- altitude diffs ----------

tg["altitude_diff_from_home_m"] = tg["venue_altitude_m"] - tg["home_altitude_m"]

# ---------- select / order columns ----------

out_cols = [
    "season", "week", "game_id", "game_date",
    "team", "opponent",
    "is_home", "is_away", "is_neutral",
    "venue_name", "venue_lat", "venue_lon", "venue_tz", "venue_altitude_m",
    "home_venue", "home_lat", "home_lon", "home_tz", "home_altitude_m",
    "rest_days", "bye_week", "travel_km", "tz_diff_from_home", "altitude_diff_from_home_m",
    "venue_imputed"
]
# Ensure existence
for c in out_cols:
    if c not in tg.columns:
        tg[c] = pd.NA

out = tg[out_cols].copy()

# ---------- write outputs and summary ----------

out_dir = Path("data/processed")
sum_dir = Path("summaries")
out_dir.mkdir(parents=True, exist_ok=True)
sum_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "situational_factors.csv"
log_path = sum_dir / "logs_situational_factors_run.txt"

out.to_csv(out_path, index=False)

missing_coords_rows = int(out["venue_lat"].isna().sum() + out["venue_lon"].isna().sum())
imputed_count = int(out["venue_imputed"].fillna(0).astype(int).sum())
na_tz = int(out["venue_tz"].isna().sum() + out["home_tz"].isna().sum())

with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"[situational] rows: {len(out)}\n")
    f.write(f"[situational] unique games: {out['game_id'].nunique()}\n")
    f.write(f"[situational] venue_imputed rows: {imputed_count}\n")
    f.write(f"[situational] rows missing venue coords (lat/lon NaN): {missing_coords_rows}\n")
    f.write(f"[situational] rows missing any tz: {na_tz}\n")
    f.write(f"[situational] mean travel_km: {pd.to_numeric(out['travel_km'], errors='coerce').mean():.2f}\n")
    f.write(f"[situational] mean rest_days: {pd.to_numeric(out['rest_days'], errors='coerce').mean():.2f}\n")

print(f"[situational] wrote {out_path}")
print(f"[situational] summary -> {log_path}")

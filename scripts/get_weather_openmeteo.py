import os
import time
import math
import json
import datetime as dt
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import requests

# -----------------------------
# Config
# -----------------------------
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "precipitation_probability"
]

# -----------------------------
# Helpers
# -----------------------------
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "", "None") else default

def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _to_utc(ts: Any) -> Optional[dt.datetime]:
    try:
        return pd.to_datetime(ts, utc=True).to_pydatetime()
    except Exception:
        return None

def _nearest_hour_index(times: List[str], target_utc: dt.datetime) -> Optional[int]:
    # times are ISO8601 strings, assumed UTC from Open-Meteo
    best_i = None
    best_diff = None
    for i, t in enumerate(times):
        try:
            tt = pd.to_datetime(t, utc=True).to_pydatetime()
        except Exception:
            continue
        diff = abs((tt - target_utc).total_seconds())
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_i = i
    return best_i

def _req(url: str, params: Dict[str, Any], retries: int = 3, backoff: float = 1.7) -> Tuple[int, Optional[Dict[str, Any]]]:
    last = 0
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            last = r.status_code
            if r.status_code == 200:
                try:
                    return r.status_code, r.json()
                except Exception:
                    return r.status_code, None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff ** (i + 1))
                continue
            return r.status_code, None
        except requests.RequestException:
            time.sleep(backoff ** (i + 1))
    return last, None

def _select_endpoint(kickoff_utc: dt.datetime) -> str:
    # Archive for past dates; forecast for future (>= today + 1h buffer)
    now = _now_utc()
    if kickoff_utc <= now + dt.timedelta(hours=1):
        return ARCHIVE_URL
    return FORECAST_URL

def _build_params(lat: float, lon: float, kickoff_utc: dt.datetime) -> Dict[str, Any]:
    d = kickoff_utc.date().isoformat()
    params = {
        "latitude": round(float(lat), 6),
        "longitude": round(float(lon), 6),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "start_date": d,
        "end_date": d
    }
    return params

def _extract_hour(row: Dict[str, Any], idx: Optional[int]) -> Dict[str, Optional[float]]:
    if not row or "hourly" not in row or idx is None:
        return {
            "temperature_c": None,
            "wind_speed_mps": None,
            "wind_direction_deg": None,
            "precip_mm": None,
            "precip_prob_pct": None
        }
    h = row["hourly"]
    def _get(name: str) -> Optional[float]:
        arr = h.get(name)
        if not isinstance(arr, list):
            return None
        if idx < 0 or idx >= len(arr):
            return None
        try:
            return float(arr[idx]) if arr[idx] is not None else None
        except Exception:
            return None

    return {
        "temperature_c": _get("temperature_2m"),
        "wind_speed_mps": _get("wind_speed_10m"),
        "wind_direction_deg": _get("wind_direction_10m"),
        "precip_mm": _get("precipitation"),
        "precip_prob_pct": _get("precipitation_probability")
    }

def _coalesce_latlon(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    # Prefer venue lat/lon; fallback to home team lat/lon if present
    lat = row.get("venue_lat")
    lon = row.get("venue_lon")
    if pd.isna(lat) or pd.isna(lon):
        lat = row.get("team_lat")
        lon = row.get("team_lon")
    try:
        return (float(lat), float(lon)) if not (pd.isna(lat) or pd.isna(lon)) else (None, None)
    except Exception:
        return (None, None)

# -----------------------------
# Load inputs
# -----------------------------
def _load_games() -> pd.DataFrame:
    # Source of game schedule + coordinates
    # Required: /situational_factors.csv (already produced in your pipeline)
    path = "situational_factors.csv"
    if not os.path.exists(path):
        # If absent, create empty schema to fail gracefully
        cols = ["game_id","season","week","start_date","team","opponent","is_home","travel_distance_km","rest_days","venue_lat","venue_lon","team_lat","team_lon"]
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path)
    # Ensure start_date in UTC
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
    # Keep one row per game (home side)
    if "is_home" in df.columns:
        df = df.loc[df["is_home"] == 1].copy()
    # Try to ensure venue coords exist; earlier pipeline should have filled these.
    # If missing, attempt to merge team coords from static CSV.
    if ("venue_lat" not in df.columns) or ("venue_lon" not in df.columns):
        df["venue_lat"] = pd.NA
        df["venue_lon"] = pd.NA

    # Add team_lat/lon if absent (for fallback and logging)
    if "team_lat" not in df.columns:
        df["team_lat"] = pd.NA
    if "team_lon" not in df.columns:
        df["team_lon"] = pd.NA

    # Fallback with static stadium coords if necessary
    static_path = os.path.join("data","fbs_fcs_stadium_coordinates.csv")
    if os.path.exists(static_path):
        st = pd.read_csv(static_path)
        if "team" in st.columns and {"latitude","longitude"}.issubset(st.columns):
            st = st.rename(columns={"latitude":"static_lat","longitude":"static_lon"})
            # Join on home 'team'
            df = df.merge(st[["team","static_lat","static_lon"]].drop_duplicates("team"),
                          on="team", how="left")
            # Fill missing venue/team coords from static
            if "venue_lat" in df.columns and "venue_lon" in df.columns:
                df["venue_lat"] = df["venue_lat"].fillna(df["static_lat"])
                df["venue_lon"] = df["venue_lon"].fillna(df["static_lon"])
            df["team_lat"] = df["team_lat"].fillna(df["static_lat"])
            df["team_lon"] = df["team_lon"].fillna(df["static_lon"])
            df = df.drop(columns=["static_lat","static_lon"])
    return df

# -----------------------------
# Main build
# -----------------------------
def build(year: str, season_type: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    games = _load_games()

    # Filter by season if available
    if "season" in games.columns:
        games = games.loc[games["season"].astype(str) == str(year)].copy()

    # Ensure essentials
    need_cols = ["game_id","season","week","start_date","team","opponent","venue_lat","venue_lon","team_lat","team_lon"]
    for c in need_cols:
        if c not in games.columns:
            games[c] = pd.NA

    # Drop rows without kickoff datetime or coordinates
    games["kickoff_ok"] = ~games["start_date"].isna()
    games["latlon_ok"] = ~(games["venue_lat"].isna() | games["venue_lon"].isna())
    work = games.loc[games["kickoff_ok"] & games["latlon_ok"]].copy()

    # Prepare output
    out_rows = []
    dbg = {
        "total_games": int(len(games)),
        "workable_games": int(len(work)),
        "skipped_no_kickoff": int(len(games) - games["kickoff_ok"].sum()),
        "skipped_no_latlon": int((~games["latlon_ok"]).sum()),
        "calls_archive": 0,
        "calls_forecast": 0,
        "errors": 0
    }

    for _, r in work.iterrows():
        gid = r.get("game_id")
        season = r.get("season")
        week = r.get("week")
        start_date = r.get("start_date")
        lat, lon = _coalesce_latlon(r)
        if lat is None or lon is None or start_date is None:
            dbg["errors"] += 1
            continue

        kickoff_utc = _to_utc(start_date)
        if kickoff_utc is None:
            dbg["errors"] += 1
            continue

        endpoint = _select_endpoint(kickoff_utc)
        if endpoint == ARCHIVE_URL:
            dbg["calls_archive"] += 1
        else:
            dbg["calls_forecast"] += 1

        params = _build_params(lat, lon, kickoff_utc)
        status, payload = _req(endpoint, params)
        times = []
        idx = None
        if payload and "hourly" in payload and "time" in payload["hourly"]:
            times = payload["hourly"]["time"]
            idx = _nearest_hour_index(times, kickoff_utc)

        met = _extract_hour(payload, idx)
        out_rows.append({
            "game_id": gid,
            "season": season,
            "week": week,
            "start_date": kickoff_utc.isoformat(),
            "home_team": r.get("team"),
            "away_team": r.get("opponent"),
            "latitude": lat,
            "longitude": lon,
            "temperature_c": met["temperature_c"],
            "wind_speed_mps": met["wind_speed_mps"],
            "wind_direction_deg": met["wind_direction_deg"],
            "precip_mm": met["precip_mm"],
            "precip_prob_pct": met["precip_prob_pct"],
            "source_endpoint": "archive" if endpoint == ARCHIVE_URL else "forecast",
            "http_status": status,
            "hour_index": idx
        })

        # modest delay to be polite to API
        time.sleep(0.05)

    df_out = pd.DataFrame(out_rows)

    # Summary stats
    def _pct_missing(series_name: str) -> float:
        return float(df_out[series_name].isna().mean()) if series_name in df_out.columns and len(df_out) else 1.0

    summary = {
        "rows": int(len(df_out)),
        "pct_missing_temperature_c": _pct_missing("temperature_c"),
        "pct_missing_wind_speed_mps": _pct_missing("wind_speed_mps"),
        "pct_missing_wind_direction_deg": _pct_missing("wind_direction_deg"),
        "pct_missing_precip_mm": _pct_missing("precip_mm"),
        "pct_missing_precip_prob_pct": _pct_missing("precip_prob_pct"),
    }
    summary.update(dbg)
    return df_out, summary

# -----------------------------
# Entry
# -----------------------------
def main():
    year = _env("CFB_YEAR", str(_now_utc().year))
    season_type = _env("CFB_SEASON_TYPE", "regular").lower()
    if season_type not in ("regular", "postseason"):
        season_type = "regular"

    df, info = build(year, season_type)

    # Outputs
    df.to_csv("weather_enriched.csv", index=False)

    with open("logs_weather_openmeteo.txt", "w", encoding="utf-8") as f:
        f.write("year=" + str(year) + "\n")
        f.write("season_type=" + season_type + "\n")
        for k, v in info.items():
            f.write(f"{k}={v}\n")

if __name__ == "__main__":
    main()

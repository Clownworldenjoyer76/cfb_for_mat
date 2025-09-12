import os
import time
import datetime as dt
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "precipitation_probability"
]

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "", "None") else default

def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _iso_now() -> str:
    return _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")

def _to_utc(ts: Any) -> Optional[dt.datetime]:
    try:
        return pd.to_datetime(ts, errors="coerce", utc=True).to_pydatetime()
    except Exception:
        return None

def _nearest_hour_index(times: List[str], target_utc: dt.datetime) -> Optional[int]:
    best_i = None
    best_diff = None
    for i, t in enumerate(times or []):
        try:
            tt = pd.to_datetime(t, utc=True).to_pydatetime()
        except Exception:
            continue
        diff = abs((tt - target_utc).total_seconds())
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_i = i
    return best_i

def _req(url: str, params: Dict[str, Any], retries: int = 2, timeout: int = 15, backoff: float = 1.6) -> Tuple[int, Optional[Dict[str, Any]]]:
    last = 0
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
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
    now = _now_utc()
    return ARCHIVE_URL if kickoff_utc <= now + dt.timedelta(hours=1) else FORECAST_URL

def _build_params(lat: float, lon: float, kickoff_utc: dt.datetime) -> Dict[str, Any]:
    d = kickoff_utc.date().isoformat()
    return {
        "latitude": round(float(lat), 6),
        "longitude": round(float(lon), 6),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "start_date": d,
        "end_date": d
    }

def _extract_hour(payload: Dict[str, Any], idx: Optional[int]) -> Dict[str, Optional[float]]:
    if not payload or "hourly" not in payload or idx is None:
        return {
            "temperature_c": None,
            "wind_speed_mps": None,
            "wind_direction_deg": None,
            "precip_mm": None,
            "precip_prob_pct": None
        }
    h = payload["hourly"]
    def _get(name: str) -> Optional[float]:
        arr = h.get(name)
        if not isinstance(arr, list) or idx < 0 or idx >= len(arr):
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
    lat = row.get("venue_lat")
    lon = row.get("venue_lon")
    if pd.isna(lat) or pd.isna(lon):
        lat = row.get("team_lat")
        lon = row.get("team_lon")
    try:
        return (float(lat), float(lon)) if not (pd.isna(lat) or pd.isna(lon)) else (None, None)
    except Exception:
        return (None, None)

def _load_games() -> pd.DataFrame:
    p = "situational_factors.csv"
    if not os.path.exists(p):
        cols = ["game_id","season","week","start_date","team","opponent","is_home","venue_lat","venue_lon","team_lat","team_lon"]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(p)
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
    if "is_home" in df.columns:
        df = df.loc[df["is_home"] == 1].copy()
    for c in ["venue_lat","venue_lon","team_lat","team_lon"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def build(year: str, season_type: str, window_days: int, max_calls: int, include_past_hours: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    games = _load_games()
    if "season" in games.columns:
        games = games.loc[games["season"].astype(str) == str(year)].copy()

    now = _now_utc()
    start_window = now - dt.timedelta(hours=include_past_hours)
    end_window = now + dt.timedelta(days=window_days)

    games["kickoff_ok"] = ~games["start_date"].isna()
    games["in_window"] = games["kickoff_ok"] & (games["start_date"] >= start_window) & (games["start_date"] <= end_window)
    games["latlon_ok"] = ~(games["venue_lat"].isna() | games["venue_lon"].isna())

    work = games.loc[games["in_window"] & games["latlon_ok"]].copy()

    out_rows = []
    dbg = {
        "snapshot_utc": _iso_now(),
        "total_games": int(len(games)),
        "workable_games": int(len(work)),
        "skipped_no_kickoff": int(len(games) - games["kickoff_ok"].sum()),
        "skipped_no_latlon": int((~games["latlon_ok"]).sum()),
        "skipped_out_of_window": int((~games["in_window"]).sum()),
        "calls_archive": 0,
        "calls_forecast": 0,
        "errors": 0
    }

    calls = 0
    for _, r in work.iterrows():
        if calls >= max_calls:
            break

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
        status, payload = _req(endpoint, params, retries=2, timeout=15)

        idx = None
        if payload and "hourly" in payload and "time" in payload["hourly"]:
            idx = _nearest_hour_index(payload["hourly"]["time"], kickoff_utc)

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
        calls += 1
        time.sleep(0.03)

    df_out = pd.DataFrame(out_rows)

    def _pct_missing(name: str) -> float:
        return float(df_out[name].isna().mean()) if name in df_out.columns and len(df_out) else 1.0

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

def main():
    year = _env("CFB_YEAR", str(_now_utc().year))
    season_type = _env("CFB_SEASON_TYPE", "regular").lower()
    if season_type not in ("regular", "postseason"):
        season_type = "regular"

    window_days = int(_env("CFB_WEATHER_WINDOW_DAYS", "14"))
    max_calls = int(_env("CFB_WEATHER_MAX_CALLS", "150"))
    include_past_hours = int(_env("CFB_WEATHER_PAST_HOURS", "2"))

    df, info = build(year, season_type, window_days, max_calls, include_past_hours)

    df.to_csv("weather_enriched.csv", index=False)
    with open("logs_weather_openmeteo.txt", "w", encoding="utf-8") as f:
        f.write("snapshot_utc=" + info.get("snapshot_utc","") + "\n")
        f.write("year=" + str(year) + "\n")
        f.write("season_type=" + season_type + "\n")
        f.write("window_days=" + str(window_days) + "\n")
        f.write("max_calls=" + str(max_calls) + "\n")
        f.write("include_past_hours=" + str(include_past_hours) + "\n")
        for k, v in info.items():
            if k == "snapshot_utc":
                continue
            f.write(f"{k}={v}\n")

if __name__ == "__main__":
    main()

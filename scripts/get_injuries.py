import os
import time
import datetime as dt
from typing import Any, Dict, Optional, List

import requests
import pandas as pd

API_BASE = "https://api.collegefootballdata.com"

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "", "None") else default

def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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

def _first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _normalize_injuries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "season","team","conference","player","position","status",
            "body_part","injury_desc","start_date","return_date",
            "source","last_updated"
        ])

    # Common/possible fields from CFBD
    rename_map = {}
    for old, new in [
        ("season","season"),
        ("team","team"),
        ("school","team"),
        ("conference","conference"),
        ("athlete.name","player"),
        ("player","player"),
        ("athlete.position","position"),
        ("position","position"),
        ("status","status"),
        ("bodyPart","body_part"),
        ("body_part","body_part"),
        ("details","injury_desc"),
        ("description","injury_desc"),
        ("startDate","start_date"),
        ("start_date","start_date"),
        ("returnDate","return_date"),
        ("return_date","return_date"),
        ("source","source"),
        ("updated","last_updated"),
        ("lastUpdated","last_updated"),
        ("last_updated","last_updated"),
    ]:
        if old in df.columns and new not in df.columns:
            rename_map[old] = new
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure required columns exist
    for col in ["season","team","conference","player","position","status","body_part",
                "injury_desc","start_date","return_date","source","last_updated"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Trim strings
    for col in ["team","conference","player","position","status","body_part","injury_desc","source"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Parse dates
    for col in ["start_date","return_date","last_updated"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Minimal selection
    out = df[[
        "season","team","conference","player","position","status",
        "body_part","injury_desc","start_date","return_date",
        "source","last_updated"
    ]].copy()

    # Coerce season to int where possible
    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")

    return out

def _append_daily_snapshot(df_norm: pd.DataFrame, daily_path: str, snapshot_utc: str) -> int:
    if df_norm.empty:
        # still append a header-only file if not exists
        if not os.path.exists(daily_path):
            df_norm.assign(snapshot_utc=pd.Series(dtype="string")).to_csv(daily_path, index=False)
        return 0

    snap = df_norm.copy()
    snap["snapshot_utc"] = pd.to_datetime(snapshot_utc, utc=True)

    if os.path.exists(daily_path):
        try:
            prev = pd.read_csv(daily_path, parse_dates=["snapshot_utc","start_date","return_date","last_updated"], infer_datetime_format=True)
        except Exception:
            prev = pd.DataFrame()
        df_out = pd.concat([prev, snap], ignore_index=True, sort=False)
    else:
        df_out = snap

    df_out.to_csv(daily_path, index=False)
    return len(snap)

def _latest_per_player(daily_path: str, latest_path: str) -> int:
    if not os.path.exists(daily_path):
        pd.DataFrame(columns=[
            "season","team","conference","player","position","status",
            "body_part","injury_desc","start_date","return_date","source",
            "last_updated","snapshot_utc"
        ]).to_csv(latest_path, index=False)
        return 0

    df = pd.read_csv(daily_path, parse_dates=["snapshot_utc","start_date","return_date","last_updated"], infer_datetime_format=True)

    # Normalize keys
    for col in ["team","player"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Determine recency
    # Use last_updated if present else start_date else snapshot_utc
    recency = pd.Series(pd.to_datetime("1970-01-01", utc=True), index=df.index)
    if "last_updated" in df.columns:
        recency = df["last_updated"].fillna(recency)
    if "start_date" in df.columns:
        recency = recency.fillna(df["start_date"])
    if "snapshot_utc" in df.columns:
        recency = recency.fillna(df["snapshot_utc"])
    df["_recency"] = recency

    # Latest per (team, player)
    df = df.sort_values(["team","player","_recency"]).groupby(["team","player"], as_index=False).tail(1)
    df = df.drop(columns=["_recency"])
    df.to_csv(latest_path, index=False)
    return len(df)

def main():
    api_key = _env("CFBD_API_KEY", "")
    year = _env("CFB_YEAR", str(dt.datetime.now(dt.timezone.utc).year))

    # Pull injuries for the year
    raw_json = _get("/injuries", {"year": year}, api_key)
    df_raw = _to_df(raw_json)

    # Normalize
    df_norm = _normalize_injuries(df_raw)

    # Write raw
    df_norm.to_csv("injuries_raw.csv", index=False)

    # Append daily snapshot
    snapshot_iso = _now_utc_iso()
    appended = _append_daily_snapshot(df_norm, "injuries_daily.csv", snapshot_iso)

    # Build latest
    latest_rows = _latest_per_player("injuries_daily.csv", "injuries_latest.csv")

    # Log
    with open("logs_injuries_run.txt", "w", encoding="utf-8") as f:
        f.write(f"year={year}\n")
        f.write(f"snapshot_utc={snapshot_iso}\n")
        f.write(f"raw_rows={len(df_norm)}\n")
        f.write(f"daily_appended_rows={appended}\n")
        f.write(f"latest_rows={latest_rows}\n")
        f.write(f"distinct_teams={df_norm['team'].nunique(dropna=True) if 'team' in df_norm.columns else 0}\n")
        f.write(f"distinct_players={df_norm['player'].nunique(dropna=True) if 'player' in df_norm.columns else 0}\n")

if __name__ == "__main__":
    main()

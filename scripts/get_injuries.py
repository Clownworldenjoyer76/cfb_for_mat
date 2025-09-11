import os
import time
import datetime as dt
from typing import Any, Dict, Optional, List

import requests
import pandas as pd

API_BASE = "https://api.collegefootballdata.com"

# -------------------------
# Helpers
# -------------------------

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "", "None") else default

def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _req(endpoint: str, params: Dict[str, Any], api_key: str, retries: int = 3, backoff: float = 1.6):
    """Return (status_code, json_or_none). Never raises."""
    url = API_BASE + endpoint
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_status = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=60)
            last_status = r.status_code
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
    return last_status or 0, None

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

# -------------------------
# Normalization
# -------------------------

def _normalize_injuries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "season","team","conference","player","position","status",
            "body_part","injury_desc","start_date","return_date",
            "source","last_updated"
        ])

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

    for col in ["season","team","conference","player","position","status","body_part",
                "injury_desc","start_date","return_date","source","last_updated"]:
        if col not in df.columns:
            df[col] = pd.NA

    for col in ["team","conference","player","position","status","body_part","injury_desc","source"]:
        df[col] = df[col].astype(str).str.strip()

    for col in ["start_date","return_date","last_updated"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    out = df[[
        "season","team","conference","player","position","status",
        "body_part","injury_desc","start_date","return_date",
        "source","last_updated"
    ]].copy()

    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    return out

# -------------------------
# Aggregation with diagnostics
# -------------------------

def _season_weeks(year: int, api_key: str, season_type: str, dbg: List[str]) -> List[int]:
    sc, j = _req("/games", {"year": year, "seasonType": season_type}, api_key)
    df = _to_df(j)
    dbg.append(f"/games year={year} seasonType={season_type} status={sc} rows={len(df)}")
    if "week" in df.columns:
        w = sorted(pd.to_numeric(df["week"], errors="coerce").dropna().astype(int).unique().tolist())
        return w or list(range(1, 21))
    return list(range(1, 21))

def _season_teams(year: int, api_key: str, dbg: List[str]) -> List[str]:
    sc, j = _req("/teams/fbs", {"year": year}, api_key)
    df = _to_df(j)
    dbg.append(f"/teams/fbs year={year} status={sc} rows={len(df)}")
    name_col = _first(df, ["school","team","name"])
    if name_col:
        vals = df[name_col].dropna().astype(str).str.strip().unique().tolist()
        return sorted(vals)
    return []

def _pull_injuries_exhaustive(year: int, api_key: str, season_type: str, dbg: List[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    # pass 1: year only
    sc, j = _req("/injuries", {"year": year}, api_key)
    d = _to_df(j)
    dbg.append(f"/injuries year={year} status={sc} rows={len(d)}")
    if not d.empty:
        frames.append(d)

    weeks = _season_weeks(year, api_key, season_type, dbg)
    teams = _season_teams(year, api_key, dbg)

    # pass 2: year+week
    for wk in weeks:
        sc, j = _req("/injuries", {"year": year, "week": wk}, api_key)
        d = _to_df(j)
        dbg.append(f"/injuries year={year} week={wk} status={sc} rows={len(d)}")
        if not d.empty:
            frames.append(d)

    # pass 3: year+team (FBS only)
    for t in teams:
        sc, j = _req("/injuries", {"year": year, "team": t}, api_key)
        d = _to_df(j)
        dbg.append(f"/injuries year={year} team={t} status={sc} rows={len(d)}")
        if not d.empty:
            frames.append(d)

    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)

    # pass 4: cap week×team subset
    cap = teams[:40]
    for wk in weeks:
        for t in cap:
            sc, j = _req("/injuries", {"year": year, "week": wk, "team": t}, api_key)
            d = _to_df(j)
            dbg.append(f"/injuries year={year} week={wk} team={t} status={sc} rows={len(d)}")
            if not d.empty:
                frames.append(d)
        if frames:
            break

    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

# -------------------------
# Daily snapshot + latest
# -------------------------

def _append_daily_snapshot(df_norm: pd.DataFrame, daily_path: str, snapshot_utc: str) -> int:
    if df_norm.empty:
        if not os.path.exists(daily_path):
            df_norm.assign(snapshot_utc=pd.Series(dtype="string")).to_csv(daily_path, index=False)
        return 0
    snap = df_norm.copy()
    snap["snapshot_utc"] = pd.to_datetime(snapshot_utc, utc=True)
    if os.path.exists(daily_path):
        try:
            prev = pd.read_csv(
                daily_path,
                parse_dates=["snapshot_utc","start_date","return_date","last_updated"],
                infer_datetime_format=True
            )
        except Exception:
            prev = pd.DataFrame()
        out = pd.concat([prev, snap], ignore_index=True, sort=False)
    else:
        out = snap
    out.to_csv(daily_path, index=False)
    return len(snap)

def _latest_per_player(daily_path: str, latest_path: str) -> int:
    if not os.path.exists(daily_path):
        pd.DataFrame(columns=[
            "season","team","conference","player","position","status",
            "body_part","injury_desc","start_date","return_date","source",
            "last_updated","snapshot_utc"
        ]).to_csv(latest_path, index=False)
        return 0

    df = pd.read_csv(
        daily_path,
        parse_dates=["snapshot_utc","start_date","return_date","last_updated"],
        infer_datetime_format=True
    )

    for col in ["team","player"]:
        df[col] = df[col].astype(str).str.strip()

    recency = pd.Series(pd.to_datetime("1970-01-01", utc=True), index=df.index)
    if "last_updated" in df.columns:
        recency = df["last_updated"].fillna(recency)
    if "start_date" in df.columns:
        recency = recency.fillna(df["start_date"])
    if "snapshot_utc" in df.columns:
        recency = recency.fillna(df["snapshot_utc"])
    df["_recency"] = recency

    df = df.sort_values(["team","player","_recency"]).groupby(["team","player"], as_index=False).tail(1)
    df = df.drop(columns=["_recency"])
    df.to_csv(latest_path, index=False)
    return len(df)

# -------------------------
# Main
# -------------------------

def main():
    api_key = _env("CFBD_API_KEY", "")
    year = int(_env("CFB_YEAR", str(dt.datetime.now(dt.timezone.utc).year)))
    season_type = _env("CFB_SEASON_TYPE", "regular").lower()
    if season_type not in ("regular","postseason"):
        season_type = "regular"

    debug_lines: List[str] = []
    raw_all = _pull_injuries_exhaustive(year, api_key, season_type, debug_lines)
    df_norm = _normalize_injuries(raw_all)

    df_norm.to_csv("injuries_raw.csv", index=False)

    snapshot_iso = _now_utc_iso()
    appended = _append_daily_snapshot(df_norm, "injuries_daily.csv", snapshot_iso)
    latest_rows = _latest_per_player("injuries_daily.csv", "injuries_latest.csv")

    with open("logs_injuries_run.txt", "w", encoding="utf-8") as f:
        f.write(f"year={year}\n")
        f.write(f"season_type={season_type}\n")
        f.write(f"snapshot_utc={snapshot_iso}\n")
        f.write(f"raw_rows={len(df_norm)}\n")
        f.write(f"daily_appended_rows={appended}\n")
        f.write(f"latest_rows={latest_rows}\n")
        f.write(f"distinct_teams={df_norm['team'].nunique(dropna=True) if 'team' in df_norm.columns else 0}\n")
        f.write(f"distinct_players={df_norm['player'].nunique(dropna=True) if 'player' in df_norm.columns else 0}\n")

    # request-level diagnostics
    with open("logs_injuries_debug.txt", "w", encoding="utf-8") as fdbg:
        for line in debug_lines:
            fdbg.write(line + "\n")

if __name__ == "__main__":
    main()

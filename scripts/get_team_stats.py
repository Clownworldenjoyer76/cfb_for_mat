#!/usr/bin/env python3
"""
get_team_stats.py

Purpose:
  - Fetch season-level college football team stats from CFBD HTTP API.
  - Write:
      1) team_stats_raw.csv  -> long format (team, stat_name, value, source)
      2) team_stats.csv      -> wide format with target columns populated ONLY by exact label matches:
           * Points Per Game                -> scoring_offense_ppg
           * Opponent Points Per Game       -> scoring_defense_ppg
           * Yards Per Play                 -> yards_per_play
           * Seconds Per Play               -> seconds_per_play

Env (blank-safe):
  CFBD_API_KEY    -> REQUIRED (CollegeFootballData.com API key)
  CFB_YEAR        -> OPTIONAL (defaults to current UTC year if unset/blank/invalid)
  CFB_SEASON_TYPE -> OPTIONAL ('regular' default, or 'postseason')
  CFB_OUTPUT_CSV  -> OPTIONAL (wide output path; default 'team_stats.csv')
  CFB_OUTPUT_RAW  -> OPTIONAL (raw output path;  default 'team_stats_raw.csv')

Dependencies: requests, pandas
"""

import os
import sys
import json
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd

BASE = "https://api.collegefootballdata.com"

def die(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)

# ------------------ Config (robust env parsing) ------------------
API_KEY = os.getenv("CFBD_API_KEY")
if not API_KEY:
    die("CFBD_API_KEY is not set (Actions Secret).")

def _get_year() -> int:
    s = os.getenv("CFB_YEAR")
    if s and s.strip():
        try:
            return int(s.strip())
        except Exception:
            pass
    return dt.datetime.utcnow().year

YEAR = _get_year()
SEASON_TYPE = (os.getenv("CFB_SEASON_TYPE", "regular") or "regular").strip().lower()
OUTPUT_CSV = (os.getenv("CFB_OUTPUT_CSV", "team_stats.csv") or "team_stats.csv").strip()
OUTPUT_RAW = (os.getenv("CFB_OUTPUT_RAW", "team_stats_raw.csv") or "team_stats_raw.csv").strip()

# ------------------ HTTP helper ------------------
def http_get(path: str, params: Dict[str, Any]) -> Any:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(f"{BASE}{path}", headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

# ------------------ Flatten helpers ------------------
VALUE_KEYS = ("value", "statValue", "stat", "avg", "average")

def _lc(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _extract_value(stat_obj: Dict[str, Any]) -> Optional[float]:
    for k in VALUE_KEYS:
        if k in stat_obj and stat_obj[k] is not None:
            try:
                return float(stat_obj[k])
            except Exception:
                pass
    for v in stat_obj.values():
        if isinstance(v, (int, float)):
            try:
                return float(v)
            except Exception:
                pass
    return None

def _extract_name(stat_obj: Dict[str, Any]) -> str:
    for k in ("name", "statName", "metric", "title", "displayName"):
        if k in stat_obj and stat_obj[k]:
            return str(stat_obj[k]).strip()
    # last resort: compact JSON slice
    return json.dumps(stat_obj, separators=(",", ":"))[:64]

def _flatten_stats_container(team_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    if isinstance(team_obj.get("stats"), list):
        stats.extend([x for x in team_obj["stats"] if isinstance(x, dict)])
    if isinstance(team_obj.get("categories"), list):
        for cat in team_obj["categories"]:
            if isinstance(cat, dict) and isinstance(cat.get("stats"), list):
                stats.extend([x for x in cat["stats"] if isinstance(x, dict)])
    return stats

def _index_by_team(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in items or []:
        team = it.get("team") or it.get("school") or it.get("team_name")
        if team:
            out[str(team)] = it
    return out

# ------------------ Builders ------------------
def fetch_season_basic(year: int, season_type: str) -> List[Dict[str, Any]]:
    payload = http_get("/stats/season", {"year": year, "seasonType": season_type})
    return payload if isinstance(payload, list) else []

def fetch_season_advanced(year: int, season_type: str) -> List[Dict[str, Any]]:
    payload = http_get("/stats/season/advanced", {"year": year, "seasonType": season_type})
    return payload if isinstance(payload, list) else []

TARGET_LABELS = {
    "Points Per Game": "scoring_offense_ppg",
    "Opponent Points Per Game": "scoring_defense_ppg",
    "Yards Per Play": "yards_per_play",
    "Seconds Per Play": "seconds_per_play",
}

def build_and_write(year: int, season_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    basic = fetch_season_basic(year, season_type)
    adv   = fetch_season_advanced(year, season_type)

    ibasic = _index_by_team(basic)
    iadv   = _index_by_team(adv)

    # -------- RAW LONG FORMAT --------
    raw_rows: List[Dict[str, Any]] = []
    for source_name, index_map in (("basic", ibasic), ("advanced", iadv)):
        for team, obj in index_map.items():
            conf = obj.get("conference") or ""
            for st in _flatten_stats_container(obj):
                nm = _extract_name(st)        # keep original case for audit
                val = _extract_value(st)
                if val is None:
                    continue
                raw_rows.append({
                    "year": year,
                    "season_type": season_type,
                    "team": team,
                    "conference": conf,
                    "stat_name": nm,
                    "value": val,
                    "source": source_name
                })
    df_raw = pd.DataFrame(raw_rows, columns=[
        "year","season_type","team","conference","stat_name","value","source"
    ])
    df_raw.to_csv(OUTPUT_RAW, index=False)

    # -------- WIDE TARGET FORMAT (exact label matches only) --------
    # Start with unique team rows
    teams = sorted(set(list(ibasic.keys()) + list(iadv.keys())))
    wide_rows: List[Dict[str, Any]] = []

    # Create a lookup: (team -> {label -> value}) using EXACT label keys
    def collect_exact(team_obj: Dict[str, Any]) -> Dict[str, float]:
        found: Dict[str, float] = {}
        for st in _flatten_stats_container(team_obj):
            nm = _extract_name(st)       # original label
            val = _extract_value(st)
            if val is None:
                continue
            if nm in TARGET_LABELS and TARGET_LABELS[nm] not in found:
                found[TARGET_LABELS[nm]] = val
        return found

    for team in teams:
        b = ibasic.get(team, {})
        a = iadv.get(team, {})
        conf = b.get("conference") or a.get("conference") or ""

        found_b = collect_exact(b)
        found_a = collect_exact(a)

        # prefer basic; fill gaps from advanced
        vals: Dict[str, Optional[float]] = {
            "scoring_offense_ppg": None,
            "scoring_defense_ppg": None,
            "yards_per_play": None,
            "seconds_per_play": None,
        }
        for k in list(vals.keys()):
            vals[k] = found_b.get(k, None)
            if vals[k] is None:
                vals[k] = found_a.get(k, None)

        wide_rows.append({
            "year": year,
            "season_type": season_type,
            "team": team,
            "conference": conf,
            "scoring_offense_ppg": vals["scoring_offense_ppg"],
            "scoring_defense_ppg": vals["scoring_defense_ppg"],
            "yards_per_play": vals["yards_per_play"],
            "seconds_per_play": vals["seconds_per_play"],
        })

    df_wide = pd.DataFrame(wide_rows, columns=[
        "year","season_type","team","conference",
        "scoring_offense_ppg","scoring_defense_ppg","yards_per_play","seconds_per_play"
    ])
    df_wide = df_wide.drop_duplicates(subset=["year","season_type","team"]).reset_index(drop=True)
    df_wide.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(df_wide)} rows to {OUTPUT_CSV} and {len(df_raw)} raw rows to {OUTPUT_RAW} (year={year}, season_type={season_type})")
    return df_wide, df_raw

def main() -> None:
    build_and_write(YEAR, SEASON_TYPE)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
get_team_stats.py

Purpose:
  1) Pull season-level CFB team stats from CFBD HTTP API (basic + advanced).
  2) Write a RAW audit CSV with every (team, stat_name, value_text, value_num, source).
  3) Write a WIDE CSV with target columns populated ONLY when exact labels match and a numeric value exists:
       - Points Per Game              -> scoring_offense_ppg
       - Opponent Points Per Game     -> scoring_defense_ppg
       - Yards Per Play               -> yards_per_play
       - Seconds Per Play             -> seconds_per_play

Env (blank-safe):
  CFBD_API_KEY    -> REQUIRED
  CFB_YEAR        -> OPTIONAL (defaults to current UTC year)
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

# ------------------ Config ------------------
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

# ------------------ HTTP ------------------
def http_get(path: str, params: Dict[str, Any]) -> Any:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(f"{BASE}{path}", headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

# ------------------ Helpers ------------------
VALUE_KEYS = ("value", "statValue", "stat", "avg", "average")

TARGET_LABELS = {
    "Points Per Game": "scoring_offense_ppg",
    "Opponent Points Per Game": "scoring_defense_ppg",
    "Yards Per Play": "yards_per_play",
    "Seconds Per Play": "seconds_per_play",
}

def _extract_name(stat_obj: Dict[str, Any]) -> str:
    for k in ("name", "statName", "metric", "title", "displayName"):
        if k in stat_obj and stat_obj[k]:
            return str(stat_obj[k]).strip()
    return json.dumps(stat_obj, separators=(",", ":"))[:128]

def _extract_value_text(stat_obj: Dict[str, Any]) -> Optional[str]:
    # Try common value keys first
    for k in VALUE_KEYS:
        if k in stat_obj and stat_obj[k] is not None:
            return str(stat_obj[k]).strip()
    # Else find first primitive as text
    for v in stat_obj.values():
        if isinstance(v, (str, int, float, bool)):
            return str(v).strip()
    return None

def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except Exception:
        # tolerate values like "35.1%" or "35,1"
        t = s.replace("%", "").replace(",", ".")
        try:
            return float(t)
        except Exception:
            return None

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

def build_and_write(year: int, season_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    basic = fetch_season_basic(year, season_type)
    adv   = fetch_season_advanced(year, season_type)

    ibasic = _index_by_team(basic)
    iadv   = _index_by_team(adv)

    # -------- RAW (always write rows) --------
    raw_rows: List[Dict[str, Any]] = []
    for source_name, index_map in (("basic", ibasic), ("advanced", iadv)):
        for team, obj in index_map.items():
            conf = obj.get("conference") or ""
            for st in _flatten_stats_container(obj):
                nm = _extract_name(st)
                vt = _extract_value_text(st)
                vn = _to_float(vt)
                # Always write a row if we have at least a name
                if nm:
                    raw_rows.append({
                        "year": year,
                        "season_type": season_type,
                        "team": team,
                        "conference": conf,
                        "stat_name": nm,
                        "value_text": vt if vt is not None else "",
                        "value_num": vn if vn is not None else "",
                        "source": source_name
                    })

    df_raw = pd.DataFrame(raw_rows, columns=[
        "year","season_type","team","conference","stat_name","value_text","value_num","source"
    ])
    df_raw.to_csv(OUTPUT_RAW, index=False)

    # -------- WIDE (exact-label + numeric only) --------
    teams = sorted(set(list(ibasic.keys()) + list(iadv.keys())))
    # Build quick lookups from RAW for exact labels
    # Prefer basic over advanced if both exist with numeric values
    def pick(team: str, label: str) -> Optional[float]:
        # basic first
        sub = df_raw[(df_raw.team == team) & (df_raw.stat_name == label) & (df_raw.source == "basic")]
        sub = sub[sub.value_num != ""]
        if not sub.empty:
            return float(sub.iloc[0]["value_num"])
        # advanced fallback
        sub = df_raw[(df_raw.team == team) & (df_raw.stat_name == label) & (df_raw.source == "advanced")]
        sub = sub[sub.value_num != ""]
        if not sub.empty:
            return float(sub.iloc[0]["value_num"])
        return None

    wide_rows: List[Dict[str, Any]] = []
    for team in teams:
        b = ibasic.get(team, {})
        a = iadv.get(team, {})
        conf = b.get("conference") or a.get("conference") or ""

        row = {
            "year": year,
            "season_type": season_type,
            "team": team,
            "conference": conf,
            "scoring_offense_ppg": pick(team, "Points Per Game"),
            "scoring_defense_ppg": pick(team, "Opponent Points Per Game"),
            "yards_per_play":       pick(team, "Yards Per Play"),
            "seconds_per_play":     pick(team, "Seconds Per Play"),
        }
        wide_rows.append(row)

    df_wide = pd.DataFrame(wide_rows, columns=[
        "year","season_type","team","conference",
        "scoring_offense_ppg","scoring_defense_ppg","yards_per_play","seconds_per_play"
    ])
    df_wide = df_wide.drop_duplicates(subset=["year","season_type","team"]).reset_index(drop=True)
    df_wide.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(df_wide)} rows to {OUTPUT_CSV} and {len(df_raw)} rows to {OUTPUT_RAW} (year={year}, season_type={season_type})")
    return df_wide, df_raw

def main() -> None:
    build_and_write(YEAR, SEASON_TYPE)

if __name__ == "__main__":
    main()

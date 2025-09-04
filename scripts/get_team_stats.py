#!/usr/bin/env python3
"""
get_team_stats.py

Purpose:
  1) Pull season-level CFB team stats from CFBD HTTP API (basic + advanced).
  2) Save unmodified payloads for audit:
       - cfbd_stats_season.json
       - cfbd_stats_advanced.json
  3) Write RAW audit CSV with every (team, stat_name, value_text, value_num, source).
  4) Write WIDE CSV with target columns populated ONLY when exact label matches (case-insensitive)
     and numeric value exists:
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
from typing import Any, Dict, List, Optional, Tuple, Union

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

# ------------------ Targets (strict, exact label match; case-insensitive) ------------------
TARGET_LABELS = {
    "points per game": "scoring_offense_ppg",
    "opponent points per game": "scoring_defense_ppg",
    "yards per play": "yards_per_play",
    "seconds per play": "seconds_per_play",
}

VALUE_KEYS = ("value", "statValue", "stat", "avg", "average")

def _lc(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _extract_name(stat_obj: Dict[str, Any]) -> Optional[str]:
    for k in ("name", "statName", "metric", "title", "displayName", "label"):
        if k in stat_obj and stat_obj[k]:
            return str(stat_obj[k]).strip()
    return None

def _extract_value_text(stat_obj: Dict[str, Any]) -> Optional[str]:
    for k in VALUE_KEYS:
        if k in stat_obj and stat_obj[k] is not None:
            return str(stat_obj[k]).strip()
    # else first primitive as text
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
        t = s.replace("%", "").replace(",", ".")
        try:
            return float(t)
        except Exception:
            return None

def _flatten_any_stats(container: Union[Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
    """
    Recursively traverse dict/list and collect dict nodes that look like stat entries:
      - have a name-like key (name/statName/metric/title/displayName/label)
      - and carry any primitive value in known value keys or elsewhere
    Returns a list of { "name": <label>, "value_text": <text or ''>, "value_num": <float or None> }.
    """
    out: List[Dict[str, Any]] = []

    def walk(node: Any):
        if isinstance(node, dict):
            nm = _extract_name(node)
            vt = _extract_value_text(node)
            if nm:
                out.append({
                    "name": nm,
                    "value_text": vt if vt is not None else "",
                    "value_num": _to_float(vt) if vt is not None else None
                })
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(container)
    return out

def _index_by_team(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in items or []:
        team = it.get("team") or it.get("school") or it.get("team_name")
        if team:
            out[str(team)] = it
    return out

# ------------------ Builders ------------------
def fetch_and_dump(year: int, season_type: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    basic = http_get("/stats/season", {"year": year, "seasonType": season_type})
    adv   = http_get("/stats/season/advanced", {"year": year, "seasonType": season_type})

    # Save unmodified payloads for audit
    with open("cfbd_stats_season.json", "w") as f:
        json.dump(basic, f, indent=2)
    with open("cfbd_stats_advanced.json", "w") as f:
        json.dump(adv, f, indent=2)

    return (basic if isinstance(basic, list) else []), (adv if isinstance(adv, list) else [])

def build_and_write(year: int, season_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    basic, adv = fetch_and_dump(year, season_type)
    ibasic = _index_by_team(basic)
    iadv   = _index_by_team(adv)

    # -------- RAW (always write rows) --------
    raw_rows: List[Dict[str, Any]] = []
    for source_name, idx in (("basic", ibasic), ("advanced", iadv)):
        for team, obj in idx.items():
            conf = obj.get("conference") or ""
            stats = _flatten_any_stats(obj)
            for st in stats:
                raw_rows.append({
                    "year": year,
                    "season_type": season_type,
                    "team": team,
                    "conference": conf,
                    "stat_name": st["name"],
                    "value_text": st["value_text"],
                    "value_num": st["value_num"] if st["value_num"] is not None else "",
                    "source": source_name
                })

    df_raw = pd.DataFrame(raw_rows, columns=[
        "year","season_type","team","conference","stat_name","value_text","value_num","source"
    ])
    df_raw = df_raw.drop_duplicates()
    df_raw.to_csv(OUTPUT_RAW, index=False)

    # -------- WIDE (exact label match, case-insensitive; vectorized masks) --------
    teams = sorted(set(list(ibasic.keys()) + list(iadv.keys())))

    # Precompute normalized stat_name and numeric mask to avoid Series truth errors
    if not df_raw.empty:
        stat_ci = df_raw["stat_name"].astype(str).str.strip().str.lower()
        has_num = df_raw["value_num"].astype(str) != ""
    else:
        stat_ci = pd.Series([], dtype=str)
        has_num = pd.Series([], dtype=bool)

    def pick(team: str, label_ci: str) -> Optional[float]:
        # prefer basic
        sub = df_raw[(df_raw.team == team) & (df_raw.source == "basic")]
        if not sub.empty:
            mask = (sub["stat_name"].astype(str).str.strip().str.lower() == label_ci) & (sub["value_num"].astype(str) != "")
            sub2 = sub[mask]
            if not sub2.empty:
                return float(sub2.iloc[0]["value_num"])
        # advanced fallback
        sub = df_raw[(df_raw.team == team) & (df_raw.source == "advanced")]
        if not sub.empty:
            mask = (sub["stat_name"].astype(str).str.strip().str.lower() == label_ci) & (sub["value_num"].astype(str) != "")
            sub2 = sub[mask]
            if not sub2.empty:
                return float(sub2.iloc[0]["value_num"])
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
            "scoring_offense_ppg": pick(team, "points per game"),
            "scoring_defense_ppg": pick(team, "opponent points per game"),
            "yards_per_play":       pick(team, "yards per play"),
            "seconds_per_play":     pick(team, "seconds per play"),
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

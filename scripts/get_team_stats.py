#!/usr/bin/env python3
"""
get_team_stats.py

Purpose:
  - Fetch season-level college football team stats from the CFBD HTTP API.
  - Extract core metrics: scoring offense (PPG), scoring defense (PPG allowed),
    yards per play (YPP), and pace (seconds per play).
  - Output a normalized CSV for downstream use.

Env (safe handling of blanks):
  CFBD_API_KEY    -> REQUIRED: CollegeFootballData.com API key (Actions secret)
  CFB_YEAR        -> OPTIONAL: season year; defaults to current UTC year if unset/blank/invalid
  CFB_SEASON_TYPE -> OPTIONAL: 'regular' (default) or 'postseason'
  CFB_OUTPUT_CSV  -> OPTIONAL: output path; defaults to 'team_stats.csv' in repo root

Dependencies: requests, pandas
"""

import os
import sys
import json
import datetime as dt
from typing import Any, Dict, List, Optional

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

# ------------------ HTTP helper ------------------
def http_get(path: str, params: Dict[str, Any]) -> Any:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(f"{BASE}{path}", headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

# ------------------ Matching config ------------------
# Exact labels observed in CFBD plus tolerant aliases (all compared lowercase/contains)
EXACT_LABELS = {
    "scoring_offense_ppg": [
        "points per game",            # CFBD display label
        "offense points per game",
        "offensive points per game",
        "ppg",                        # alias
        "scoring offense"
    ],
    "scoring_defense_ppg": [
        "opponent points per game",   # CFBD display label
        "opp points per game",
        "points allowed per game",
        "scoring defense",
        "defensive points per game"
    ],
    "yards_per_play": [
        "yards per play",             # CFBD display label (offense)
        "net yards per play",
        "offense yards per play",
        "ypp"
    ],
    "seconds_per_play": [
        "seconds per play",           # CFBD display label (pace)
        "sec/play",
        "pace"
    ],
}

VALUE_KEYS = ("value", "statValue", "stat", "avg", "average")  # try in order

def _lc(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _extract_value(stat_obj: Dict[str, Any]) -> Optional[float]:
    # Try common numeric keys first
    for k in VALUE_KEYS:
        if k in stat_obj and stat_obj[k] is not None:
            try:
                return float(stat_obj[k])
            except Exception:
                pass
    # Fallback: first numeric in values
    for v in stat_obj.values():
        if isinstance(v, (int, float)):
            try:
                return float(v)
            except Exception:
                pass
    return None

def _extract_name(stat_obj: Dict[str, Any]) -> str:
    # Common name keys in CFBD payloads
    for k in ("name", "statName", "metric", "title", "displayName"):
        if k in stat_obj and stat_obj[k]:
            return _lc(str(stat_obj[k]))
    # As a last resort, stringify object
    return _lc(json.dumps(stat_obj, separators=(",", ":")))

def _match_field(stat_name_lc: str) -> Optional[str]:
    for field, labels in EXACT_LABELS.items():
        for label in labels:
            if label in stat_name_lc:
                return field
    return None

def _flatten_stats_container(team_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    CFBD /stats/season can return either:
      - {"team": "...", "stats": [ { "name": "Points Per Game", "value": ... }, ... ]}
      - {"team": "...", "categories": [ { "name": "...", "stats": [ ... ] }, ... ]}
    Return a flat list of stat dicts.
    """
    stats = []
    # Primary flat list
    if isinstance(team_obj.get("stats"), list):
        stats.extend([x for x in team_obj["stats"] if isinstance(x, dict)])
    # Nested categories
    if isinstance(team_obj.get("categories"), list):
        for cat in team_obj["categories"]:
            if isinstance(cat, dict) and isinstance(cat.get("stats"), list):
                stats.extend([x for x in cat["stats"] if isinstance(x, dict)])
    return stats

def _pick_core_metrics(stats_list: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "scoring_offense_ppg": None,
        "scoring_defense_ppg": None,
        "yards_per_play": None,
        "seconds_per_play": None,
    }
    trace: Dict[str, str] = {}

    for st in stats_list:
        name_lc = _extract_name(st)
        val = _extract_value(st)
        if val is None:
            continue
        field = _match_field(name_lc)
        if field and out[field] is None:
            out[field] = val
            trace[field] = name_lc

    out["_trace"] = trace
    return out

# ------------------ Build ------------------
def fetch_season_stats(year: int, season_type: str) -> List[Dict[str, Any]]:
    payload = http_get("/stats/season", {"year": year, "seasonType": season_type})
    return payload if isinstance(payload, list) else []

def build_dataframe(year: int, season_type: str) -> pd.DataFrame:
    teams = fetch_season_stats(year, season_type)
    rows: List[Dict[str, Any]] = []

    for item in teams:
        team = item.get("team") or item.get("school") or item.get("team_name")
        conf = item.get("conference") or item.get("conf") or ""
        stats_list = _flatten_stats_container(item)

        core = _pick_core_metrics(stats_list)
        row = {
            "year": year,
            "season_type": season_type,
            "team": team,
            "conference": conf,
            "scoring_offense_ppg": core.get("scoring_offense_ppg"),
            "scoring_defense_ppg": core.get("scoring_defense_ppg"),
            "yards_per_play": core.get("yards_per_play"),
            "seconds_per_play": core.get("seconds_per_play"),
            "source_stat_map": json.dumps(core.get("_trace", {}), separators=(",", ":")),
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=[
        "year", "season_type", "team", "conference",
        "scoring_offense_ppg", "scoring_defense_ppg",
        "yards_per_play", "seconds_per_play",
        "source_stat_map"
    ])
    df = df.drop_duplicates(subset=["year", "season_type", "team"]).reset_index(drop=True)
    return df

def main() -> None:
    df = build_dataframe(YEAR, SEASON_TYPE)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV} (year={YEAR}, season_type={SEASON_TYPE})")

if __name__ == "__main__":
    main()

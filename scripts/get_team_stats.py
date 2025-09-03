#!/usr/bin/env python3
"""
get_team_stats.py
Purpose:
  - Fetch season-level college football team stats from the CFBD HTTP API.
  - Extract core metrics for modeling: scoring offense (PPG), scoring defense (PPG allowed),
    yards per play (YPP), and pace (seconds per play).
  - Output a normalized CSV for downstream use.

Env (safe handling of blanks):
  CFBD_API_KEY    -> REQUIRED: CollegeFootballData.com API key (Actions secret)
  CFB_YEAR        -> OPTIONAL: season year; defaults to current UTC year if unset/blank/invalid
  CFB_SEASON_TYPE -> OPTIONAL: 'regular' (default) or 'postseason'
  CFB_OUTPUT_CSV  -> OPTIONAL: output path; defaults to 'team_stats.csv' in repo root

Dependencies: requests, pandas
  (Ensure 'requests' and 'pandas' are in requirements.txt)
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

SEASON_TYPE = os.getenv("CFB_SEASON_TYPE", "regular").strip().lower() or "regular"
OUTPUT_CSV = os.getenv("CFB_OUTPUT_CSV", "team_stats.csv").strip() or "team_stats.csv"

# ------------------ HTTP helper ------------------
def http_get(path: str, params: Dict[str, Any]) -> Any:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(f"{BASE}{path}", headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

# ------------------ Normalization logic ------------------
# We match by common stat names in CFBD payloads. Matching is case-insensitive and tolerant.
MATCH_MAP = {
    "scoring_offense_ppg": [
        "points per game", "ppg", "scoring offense", "offense points per game",
        "offensive points per game"
    ],
    "scoring_defense_ppg": [
        "opponent points per game", "points allowed per game", "opp ppg",
        "scoring defense", "defense points per game", "defensive points per game"
    ],
    "yards_per_play": [
        "yards per play", "ypp", "net yards per play", "offense yards per play"
    ],
    "seconds_per_play": [
        "seconds per play", "sec/play", "pace"
    ],
}

def _lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _extract_value(stat_obj: Dict[str, Any]) -> Optional[float]:
    """
    CFBD season stats often use one of:
      {"name": "Points Per Game", "value": "35.1"} OR
      {"statName": "Points Per Game", "statValue": "35.1"} OR
      {"stat": "35.1", "name": "Points Per Game"}
    We attempt several keys and return float if possible.
    """
    # try common value keys
    for k in ("value", "statValue", "stat"):
        if k in stat_obj and stat_obj[k] is not None:
            try:
                return float(stat_obj[k])
            except Exception:
                pass
    # sometimes numeric already under unexpected key
    for v in stat_obj.values():
        if isinstance(v, (int, float)):
            try:
                return float(v)
            except Exception:
                pass
    return None

def _extract_name(stat_obj: Dict[str, Any]) -> str:
    for k in ("name", "statName", "metric", "title"):
        if k in stat_obj and stat_obj[k]:
            return _lower(str(stat_obj[k]))
    # fallback: stringified keys
    return _lower(json.dumps(stat_obj, separators=(",", ":"))[:64])

def _pick_core_metrics(stats_list: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Iterate through the team's stat objects and map them into our core fields
    using fuzzy name matching from MATCH_MAP.
    """
    out: Dict[str, Optional[float]] = {
        "scoring_offense_ppg": None,
        "scoring_defense_ppg": None,
        "yards_per_play": None,
        "seconds_per_play": None,
    }
    # track which original names mapped to which fields (for transparency)
    trace: Dict[str, str] = {}

    for st in stats_list or []:
        name = _extract_name(st)  # lowercased
        val = _extract_value(st)
        if val is None:
            continue
        for field, candidates in MATCH_MAP.items():
            for c in candidates:
                if c in name:
                    # first match wins; avoid overwriting
                    if out[field] is None:
                        out[field] = val
                        trace[field] = name
                    break

    out["_trace"] = trace
    return out

def fetch_season_stats(year: int, season_type: str) -> List[Dict[str, Any]]:
    """
    Calls CFBD /stats/season which returns a list of teams with nested 'stats'.
    """
    payload = http_get("/stats/season", {"year": year, "seasonType": season_type})
    # payload commonly looks like:
    # [{"team": "Alabama", "conference": "SEC", "stats": [ {...}, {...} ]}, ...]
    if not isinstance(payload, list):
        return []
    return payload

def build_dataframe(year: int, season_type: str) -> pd.DataFrame:
    teams = fetch_season_stats(year, season_type)

    rows: List[Dict[str, Any]] = []
    for item in teams:
        team = item.get("team") or item.get("school") or item.get("team_name")
        conf = item.get("conference") or item.get("conf") or ""
        stats_list = item.get("stats") or item.get("statistics") or []

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
    # Deduplicate by (year, season_type, team)
    df = df.drop_duplicates(subset=["year", "season_type", "team"]).reset_index(drop=True)
    return df

def main() -> None:
    df = build_dataframe(YEAR, SEASON_TYPE)
    # Write CSV in repo root (or overridden by env)
    out_path = OUTPUT_CSV
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path} (year={YEAR}, season_type={SEASON_TYPE})")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
get_team_efficiency.py

Purpose:
  - Produce nflfastR-style team-level efficiency metrics for CFB.
  - Prefer SportsDataverse (sportsdataverse-py) if available.
  - Fallback to CFBD HTTP endpoint /ppa/teams (same data family).
  - Extract offense/defense PPA and Success Rate for overall, rushing, and passing.
  - Write a normalized CSV: team_efficiency.csv

Env:
  CFBD_API_KEY     -> REQUIRED (GitHub Actions secret)
  CFB_YEAR         -> OPTIONAL (defaults to current UTC year if unset/blank/invalid)
  CFB_SEASON_TYPE  -> OPTIONAL ('regular' default; forwarded to HTTP fallback)
  CFB_OUTPUT_EFF   -> OPTIONAL (output path; default 'team_efficiency.csv')

Dependencies:
  - requests
  - pandas
  - (optional) sportsdataverse >= 0.6 (for sdv.cfbd.get_ppa_teams if present)

Notes:
  - No assumptions: if a field is not present, leave it blank.
  - Robust parsing: tolerate nested shapes and missing keys.
"""

import os
import sys
import json
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

BASE = "https://api.collegefootballdata.com"

# ------------------ Config ------------------
def _die_ok(msg: str) -> None:
    print(msg, file=sys.stderr)

API_KEY = os.getenv("CFBD_API_KEY")
if not API_KEY:
    _die_ok("CFBD_API_KEY is not set; writing empty team_efficiency.csv.")
    pd.DataFrame(columns=[
        "year","season_type","team","conference",
        "off_ppa","off_sr","off_rush_ppa","off_rush_sr","off_pass_ppa","off_pass_sr",
        "def_ppa","def_sr","def_rush_ppa","def_rush_sr","def_pass_ppa","def_pass_sr",
        "source_meta"
    ]).to_csv(os.getenv("CFB_OUTPUT_EFF", "team_efficiency.csv"), index=False)
    sys.exit(0)

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
OUTPUT_EFF = (os.getenv("CFB_OUTPUT_EFF", "team_efficiency.csv") or "team_efficiency.csv").strip()

# ------------------ IO helpers ------------------
def http_get(path: str, params: Dict[str, Any]) -> Any:
    import requests
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(f"{BASE}{path}", headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

# ------------------ Fetchers ------------------
def fetch_with_sdv(year: int) -> List[Dict[str, Any]]:
    """
    Try SportsDataverse if present (schema may vary by version).
    Expected target: PPA-like per-team efficiency (offense/defense, overall/rush/pass).
    """
    try:
        from sportsdataverse import cfbd as sdv_cfbd  # optional dependency
    except Exception as e:
        raise RuntimeError(f"SportsDataverse not available: {e}")

    # Best-effort: look for a PPA teams accessor if present
    # If unavailable, raise to trigger HTTP fallback.
    getter = None
    for name in ("get_ppa_teams", "get_ppa_team", "ppa_teams"):
        if hasattr(sdv_cfbd, name):
            getter = getattr(sdv_cfbd, name)
            break
    if getter is None:
        raise RuntimeError("SportsDataverse cfbd.*get_ppa_teams* not found; use HTTP fallback.")

    df = getter(year=year, season_type=SEASON_TYPE, authorization=API_KEY)
    if hasattr(df, "to_dict"):
        return df.to_dict("records")
    if isinstance(df, list):
        return df
    raise RuntimeError("Unexpected SportsDataverse return type.")

def fetch_with_http(year: int) -> List[Dict[str, Any]]:
    """
    CFBD HTTP fallback.
    Endpoint: /ppa/teams?year=YYYY&seasonType=regular|postseason
    """
    params = {"year": year, "seasonType": SEASON_TYPE}
    data = http_get("/ppa/teams", params)
    return data if isinstance(data, list) else []

# ------------------ Parsing ------------------
def _num(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        try:
            s = str(x).replace("%", "").replace(",", ".")
            return float(s)
        except Exception:
            return None

def _get(d: Dict[str, Any], *keys: str) -> Optional[float]:
    """
    Safe nested getter for numeric fields, e.g. _get(d, "offense", "overall", "ppa")
    """
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return _num(cur)

def _first_str(*vals: Any) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def normalize_records(items: List[Dict[str, Any]], year: int, season_type: str) -> List[Dict[str, Any]]:
    """
    Normalize a list of PPA/efficiency records into a flat structure.
    Leave blanks if a field is missing (no assumptions).
    """
    rows: List[Dict[str, Any]] = []
    for obj in items or []:
        team = _first_str(obj.get("team"), obj.get("school"), obj.get("team_name"))
        conf = _first_str(obj.get("conference"), obj.get("conf"))
        row = {
            "year": year,
            "season_type": season_type,
            "team": team,
            "conference": conf,
            # offense (overall/rush/pass)
            "off_ppa": _get(obj, "offense", "overall", "ppa"),
            "off_sr": _get(obj, "offense", "overall", "successRate"),
            "off_rush_ppa": _get(obj, "offense", "rushing", "ppa"),
            "off_rush_sr": _get(obj, "offense", "rushing", "successRate"),
            "off_pass_ppa": _get(obj, "offense", "passing", "ppa"),
            "off_pass_sr": _get(obj, "offense", "passing", "successRate"),
            # defense (overall/rush/pass)
            "def_ppa": _get(obj, "defense", "overall", "ppa"),
            "def_sr": _get(obj, "defense", "overall", "successRate"),
            "def_rush_ppa": _get(obj, "defense", "rushing", "ppa"),
            "def_rush_sr": _get(obj, "defense", "rushing", "successRate"),
            "def_pass_ppa": _get(obj, "defense", "passing", "ppa"),
            "def_pass_sr": _get(obj, "defense", "passing", "successRate"),
            # source meta (for audit / debugging)
            "source_meta": json.dumps({
                "has_offense": isinstance(obj.get("offense"), dict),
                "has_defense": isinstance(obj.get("defense"), dict),
            }, separators=(",", ":"))
        }
        rows.append(row)
    return rows

# ------------------ Main ------------------
def main() -> None:
    # Try SportsDataverse first; if it fails for any reason, fallback to HTTP
    using = "sdv"
    try:
        items = fetch_with_sdv(YEAR)
    except Exception as e:
        _die_ok(f"[warn] SportsDataverse failed or unavailable: {e}")
        items = fetch_with_http(YEAR)
        using = "http"

    rows = normalize_records(items, YEAR, SEASON_TYPE)
    df = pd.DataFrame(rows, columns=[
        "year","season_type","team","conference",
        "off_ppa","off_sr","off_rush_ppa","off_rush_sr","off_pass_ppa","off_pass_sr",
        "def_ppa","def_sr","def_rush_ppa","def_rush_sr","def_pass_ppa","def_pass_sr",
        "source_meta"
    ])
    df = df.drop_duplicates(subset=["year","season_type","team"]).reset_index(drop=True)
    df.to_csv(OUTPUT_EFF, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_EFF} via {using}")

if __name__ == "__main__":
    main()

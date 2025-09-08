#!/usr/bin/env python3
"""
get_team_efficiency.py

Purpose:
  - Pull team-level PPA efficiency for CFB.
  - Prefer SportsDataverse (sportsdataverse-py) if available.
  - Fallback to CFBD HTTP endpoint /ppa/teams.
  - Map EXACT keys seen in your payload:
      offense.overall, offense.passing, offense.rushing,
      offense.firstDown, offense.secondDown, offense.thirdDown,
      offense.cumulative.total, offense.cumulative.passing, offense.cumulative.rushing
    and same structure for defense.*
  - Write:
      - team_efficiency.csv
      - cfbd_ppa_teams.json  (raw audit)

Env:
  CFBD_API_KEY     -> REQUIRED
  CFB_YEAR         -> OPTIONAL (defaults to current UTC year)
  CFB_SEASON_TYPE  -> OPTIONAL ('regular' default; forwarded to HTTP if used)
  CFB_OUTPUT_EFF   -> OPTIONAL (output CSV path; default 'team_efficiency.csv')

Dependencies:
  pandas, requests
  (optional) sportsdataverse >= 0.6

Behavior:
  - No assumptions: if a field is missing, leave it blank.
"""

import os
import sys
import json
import datetime as dt
from typing import Any, Dict, List, Optional

import pandas as pd

BASE = "https://api.collegefootballdata.com"

# ------------------ Config ------------------
def _warn(msg: str) -> None:
    print(msg, file=sys.stderr)

API_KEY = os.getenv("CFBD_API_KEY")
if not API_KEY:
    _warn("CFBD_API_KEY is not set; writing empty CSV and empty raw JSON.")
    pd.DataFrame(columns=[
        "year","season_type","team","conference",
        "off_overall_ppa","off_passing_ppa","off_rushing_ppa",
        "off_first_down_ppa","off_second_down_ppa","off_third_down_ppa",
        "off_cum_total_ppa","off_cum_passing_ppa","off_cum_rushing_ppa",
        "def_overall_ppa","def_passing_ppa","def_rushing_ppa",
        "def_first_down_ppa","def_second_down_ppa","def_third_down_ppa",
        "def_cum_total_ppa","def_cum_passing_ppa","def_cum_rushing_ppa",
        "source_meta"
    ]).to_csv(os.getenv("CFB_OUTPUT_EFF", "team_efficiency.csv"), index=False)
    with open("cfbd_ppa_teams.json", "w") as f:
        json.dump({"error":"missing CFBD_API_KEY"}, f, indent=2)
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
OUTPUT_RAW_JSON = "cfbd_ppa_teams.json"

# ------------------ IO helpers ------------------
def http_get(path: str, params: Dict[str, Any]) -> Any:
    import requests
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(f"{BASE}{path}", headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

# ------------------ Fetchers ------------------
def fetch_with_sdv(year: int) -> List[Dict[str, Any]]:
    try:
        from sportsdataverse import cfbd as sdv_cfbd
    except Exception as e:
        raise RuntimeError(f"SportsDataverse not available: {e}")

    getter = None
    for name in ("get_ppa_teams", "get_ppa_team", "ppa_teams"):
        if hasattr(sdv_cfbd, name):
            getter = getattr(sdv_cfbd, name)
            break
    if getter is None:
        raise RuntimeError("SportsDataverse cfbd get_ppa_teams-like function not found.")

    df = getter(year=year, season_type=SEASON_TYPE, authorization=API_KEY)
    if hasattr(df, "to_dict"):
        return df.to_dict("records")
    if isinstance(df, list):
        return df
    raise RuntimeError("Unexpected SportsDataverse return type.")

def fetch_with_http(year: int) -> List[Dict[str, Any]]:
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
    rows: List[Dict[str, Any]] = []
    for obj in items or []:
        team = _first_str(obj.get("team"), obj.get("school"), obj.get("team_name"))
        conf = _first_str(obj.get("conference"), obj.get("conf"))
        off = obj.get("offense") if isinstance(obj.get("offense"), dict) else {}
        deff = obj.get("defense") if isinstance(obj.get("defense"), dict) else {}
        off_cum = off.get("cumulative") if isinstance(off.get("cumulative"), dict) else {}
        def_cum = deff.get("cumulative") if isinstance(deff.get("cumulative"), dict) else {}

        row = {
            "year": year,
            "season_type": season_type,
            "team": team,
            "conference": conf,
            # offense PPA (exact keys from payload)
            "off_overall_ppa": _get(off, "overall"),
            "off_passing_ppa": _get(off, "passing"),
            "off_rushing_ppa": _get(off, "rushing"),
            "off_first_down_ppa": _get(off, "firstDown"),
            "off_second_down_ppa": _get(off, "secondDown"),
            "off_third_down_ppa": _get(off, "thirdDown"),
            "off_cum_total_ppa": _get(off_cum, "total"),
            "off_cum_passing_ppa": _get(off_cum, "passing"),
            "off_cum_rushing_ppa": _get(off_cum, "rushing"),
            # defense PPA (exact keys from payload)
            "def_overall_ppa": _get(deff, "overall"),
            "def_passing_ppa": _get(deff, "passing"),
            "def_rushing_ppa": _get(deff, "rushing"),
            "def_first_down_ppa": _get(deff, "firstDown"),
            "def_second_down_ppa": _get(deff, "secondDown"),
            "def_third_down_ppa": _get(deff, "thirdDown"),
            "def_cum_total_ppa": _get(def_cum, "total"),
            "def_cum_passing_ppa": _get(def_cum, "passing"),
            "def_cum_rushing_ppa": _get(def_cum, "rushing"),
            # meta audit
            "source_meta": json.dumps({
                "has_offense": isinstance(off, dict),
                "has_defense": isinstance(deff, dict),
                "has_off_cumulative": isinstance(off_cum, dict),
                "has_def_cumulative": isinstance(def_cum, dict),
            }, separators=(",", ":"))
        }
        rows.append(row)
    return rows

# ------------------ Main ------------------
def main() -> None:
    using = "sdv"
    try:
        items = fetch_with_sdv(YEAR)
    except Exception as e:
        _warn(f"[warn] SportsDataverse failed or unavailable; using HTTP fallback: {e}")
        items = fetch_with_http(YEAR)
        using = "http"

    # Raw audit
    try:
        with open(OUTPUT_RAW_JSON, "w") as f:
            json.dump(
                {"source": using, "year": YEAR, "season_type": SEASON_TYPE, "count": len(items), "data": items},
                f,
                indent=2
            )
    except Exception as e:
        _warn(f"[warn] failed to write {OUTPUT_RAW_JSON}: {e}")

    # Normalize -> CSV
    rows = normalize_records(items, YEAR, SEASON_TYPE)
    df = pd.DataFrame(rows, columns=[
        "year","season_type","team","conference",
        "off_overall_ppa","off_passing_ppa","off_rushing_ppa",
        "off_first_down_ppa","off_second_down_ppa","off_third_down_ppa",
        "off_cum_total_ppa","off_cum_passing_ppa","off_cum_rushing_ppa",
        "def_overall_ppa","def_passing_ppa","def_rushing_ppa",
        "def_first_down_ppa","def_second_down_ppa","def_third_down_ppa",
        "def_cum_total_ppa","def_cum_passing_ppa","def_cum_rushing_ppa",
        "source_meta"
    ])
    df = df.drop_duplicates(subset=["year","season_type","team"]).reset_index(drop=True)
    df.to_csv(OUTPUT_EFF, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_EFF} (source={using}); raw saved to {OUTPUT_RAW_JSON}")

if __name__ == "__main__":
    main()

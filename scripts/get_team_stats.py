#!/usr/bin/env python3
"""
get_team_stats.py

Function:
  - Pull CFBD season stats (basic + advanced).
  - Save raw payload JSON for audit.
  - Write RAW CSV of all (team, stat_name, value_text, value_num, source).
  - Validate presence of required labels:
       * pointsPerGame / Points Per Game
       * oppPointsPerGame / Opponent Points Per Game
       * yardsPerPlay / Yards Per Play
       * secondsPerPlay / Seconds Per Play
  - If any required label is missing, write:
       * team_stats_names_unique.csv  (all distinct stat_names discovered with counts)
       * team_stats.csv               (empty wide by design)
    then exit with non-zero code explaining the exact missing labels.
  - If labels exist, write populated team_stats.csv.

Env:
  CFBD_API_KEY     REQUIRED
  CFB_YEAR         OPTIONAL (defaults to current UTC year)
  CFB_SEASON_TYPE  OPTIONAL ('regular' default)
  CFB_OUTPUT_CSV   OPTIONAL (default 'team_stats.csv')
  CFB_OUTPUT_RAW   OPTIONAL (default 'team_stats_raw.csv')

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

# ---------- Config ----------
def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)

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
OUTPUT_NAMES = "team_stats_names_unique.csv"
PAYLOAD_BASIC = "cfbd_stats_season.json"
PAYLOAD_ADV = "cfbd_stats_advanced.json"

# ---------- HTTP ----------
def http_get(path: str, params: Dict[str, Any]) -> Any:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(f"{BASE}{path}", headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

# ---------- Normalization ----------
def norm(s: str) -> str:
    return "".join(ch.lower() for ch in (s or "") if ch.isalnum())

VALUE_KEYS = ("value", "statValue", "stat", "avg", "average")

# Accept both human labels and camelCase keys after normalization
REQUIRED_KEYS = {
    "pointspergame": "scoring_offense_ppg",
    "opponentpointspergame": "scoring_defense_ppg",
    "opppointspergame": "scoring_defense_ppg",  # alt
    "yardsperplay": "yards_per_play",
    "secondsperplay": "seconds_per_play",
}

# ---------- Extractors ----------
def _extract_name(stat_obj: Dict[str, Any]) -> Optional[str]:
    for k in ("name", "statName", "metric", "title", "displayName", "label", "stat"):
        if k in stat_obj and isinstance(stat_obj[k], str) and stat_obj[k].strip():
            return stat_obj[k].strip()
    return None

def _extract_value_text(stat_obj: Dict[str, Any]) -> Optional[str]:
    for k in VALUE_KEYS:
        if k in stat_obj and stat_obj[k] is not None:
            return str(stat_obj[k]).strip()
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

# ---------- Build ----------
def fetch_and_dump(year: int, season_type: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    basic = http_get("/stats/season", {"year": year, "seasonType": season_type})
    adv   = http_get("/stats/season/advanced", {"year": year, "seasonType": season_type})
    with open(PAYLOAD_BASIC, "w") as f:
        json.dump(basic, f, indent=2)
    with open(PAYLOAD_ADV, "w") as f:
        json.dump(adv, f, indent=2)
    return (basic if isinstance(basic, list) else []), (adv if isinstance(adv, list) else [])

def build_raw_df(basic: List[Dict[str, Any]], adv: List[Dict[str, Any]]) -> pd.DataFrame:
    ibasic = _index_by_team(basic)
    iadv = _index_by_team(adv)
    rows: List[Dict[str, Any]] = []
    for source_name, idx in (("basic", ibasic), ("advanced", iadv)):
        for team, obj in idx.items():
            conf = obj.get("conference") or ""
            for st in _flatten_any_stats(obj):
                rows.append({
                    "year": YEAR,
                    "season_type": SEASON_TYPE,
                    "team": team,
                    "conference": conf,
                    "stat_name": st["name"],
                    "value_text": st["value_text"],
                    "value_num": st["value_num"] if st["value_num"] is not None else "",
                    "source": source_name
                })
    df = pd.DataFrame(rows, columns=[
        "year","season_type","team","conference","stat_name","value_text","value_num","source"
    ])
    df = df.drop_duplicates()
    return df

def write_names_audit(df_raw: pd.DataFrame) -> None:
    if df_raw.empty:
        pd.DataFrame(columns=["stat_name","count"]).to_csv(OUTPUT_NAMES, index=False)
        return
    tmp = df_raw.groupby("stat_name", as_index=False).size().rename(columns={"size":"count"})
    tmp.sort_values(["count","stat_name"], ascending=[False, True]).to_csv(OUTPUT_NAMES, index=False)

def make_wide(df_raw: pd.DataFrame, ibasic: Dict[str, Dict[str, Any]], iadv: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    teams = sorted(set(list(ibasic.keys()) + list(iadv.keys())))
    def pick_norm(team: str, key_norm: str) -> Optional[float]:
        sub = df_raw[(df_raw.team == team) & (df_raw.source == "basic")]
        if not sub.empty:
            mask = sub["stat_name"].astype(str).apply(norm) == key_norm
            sub2 = sub[mask & (sub["value_num"].astype(str) != "")]
            if not sub2.empty:
                return float(sub2.iloc[0]["value_num"])
        sub = df_raw[(df_raw.team == team) & (df_raw.source == "advanced")]
        if not sub.empty:
            mask = sub["stat_name"].astype(str).apply(norm) == key_norm
            sub2 = sub[mask & (sub["value_num"].astype(str) != "")]
            if not sub2.empty:
                return float(sub2.iloc[0]["value_num"])
        return None

    rows: List[Dict[str, Any]] = []
    for team in teams:
        b = ibasic.get(team, {})
        a = iadv.get(team, {})
        conf = b.get("conference") or a.get("conference") or ""
        rows.append({
            "year": YEAR,
            "season_type": SEASON_TYPE,
            "team": team,
            "conference": conf,
            "scoring_offense_ppg": pick_norm(team, "pointspergame"),
            "scoring_defense_ppg": (pick_norm(team, "opponentpointspergame") or pick_norm(team, "opppointspergame")),
            "yards_per_play": pick_norm(team, "yardsperplay"),
            "seconds_per_play": pick_norm(team, "secondsperplay"),
        })
    wide = pd.DataFrame(rows, columns=[
        "year","season_type","team","conference",
        "scoring_offense_ppg","scoring_defense_ppg","yards_per_play","seconds_per_play"
    ])
    wide = wide.drop_duplicates(subset=["year","season_type","team"]).reset_index(drop=True)
    return wide

def main() -> None:
    basic, adv = fetch_and_dump(YEAR, SEASON_TYPE)
    df_raw = build_raw_df(basic, adv)
    df_raw.to_csv(OUTPUT_RAW, index=False)
    write_names_audit(df_raw)

    ibasic = _index_by_team(basic)
    iadv = _index_by_team(adv)
    wide = make_wide(df_raw, ibasic, iadv)

    # Validate required labels exist somewhere in RAW
    all_norms = set(df_raw["stat_name"].astype(str).apply(norm).unique())
    missing = [label for label in REQUIRED_KEYS.keys() if label not in all_norms]
    if missing:
        # Write empty wide CSV (by design) and fail with explicit reason
        wide.to_csv(OUTPUT_CSV, index=False)
        die(
            "Missing required CFBD stat labels in payload. "
            f"Not found (normalized): {', '.join(sorted(missing))}. "
            f"See {OUTPUT_NAMES} for all discovered stat_name values."
        )

    # If labels exist, write wide CSV (populated where available)
    wide.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df_raw)} RAW rows -> {OUTPUT_RAW}")
    print(f"Wrote unique names audit -> {OUTPUT_NAMES}")
    print(f"Wrote {len(wide)} WIDE rows -> {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

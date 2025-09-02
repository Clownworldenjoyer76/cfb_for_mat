#!/usr/bin/env python3
"""
generate_scores.py
- Creates score predictions from CFBD market totals and home spreads.
- Writes scores.json at repo root for the static site.

Requirements:
  Env:
    CFBD_API_KEY   -> CollegeFootballData.com API key (Actions secret)
    CFB_YEAR       -> optional (defaults to current UTC year if unset or blank)
    CFB_WEEK       -> optional (if absent or blank, autodetects)
    CFB_SCAN_WEEKS -> optional scan range if initial week has no usable lines (default "1-20")

  Dependencies: requests (already added to requirements.txt)
"""

import os
import json
import datetime as dt
from typing import Any, Dict, List, Optional

BASE = "https://api.collegefootballdata.com"

def die(msg: str) -> None:
    raise SystemExit(msg)

# ---- Config ----
API_KEY = os.getenv("CFBD_API_KEY")
if not API_KEY:
    die("CFBD_API_KEY is not set (Actions Secret).")

year_str = os.getenv("CFB_YEAR")
if year_str and year_str.strip():
    try:
        YEAR = int(year_str.strip())
    except Exception:
        YEAR = dt.datetime.utcnow().year
else:
    YEAR = dt.datetime.utcnow().year

week_str = os.getenv("CFB_WEEK")
WEEK_ENV = week_str.strip() if week_str else None
SCAN_ENV = os.getenv("CFB_SCAN_WEEKS", "1-20").strip()

# ---- HTTP ----
import requests

def http_get(path: str, params: Dict[str, Any]) -> Any:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(f"{BASE}{path}", headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

# ---- Helpers ----
def parse_scan_range(spec: str) -> List[int]:
    spec = spec.replace(" ", "")
    if "-" in spec:
        a, b = spec.split("-", 1)
        try:
            lo, hi = int(a), int(b)
            if lo <= hi:
                return list(range(lo, hi + 1))
        except Exception:
            return list(range(1, 21))
    try:
        single = int(spec)
        return [single]
    except Exception:
        return list(range(1, 21))

def autodetect_week(year: int) -> int:
    games = http_get("/games", {"year": year, "seasonType": "regular"})
    now = dt.datetime.utcnow()
    future_weeks = []
    all_weeks = set()
    for g in games or []:
        wk = g.get("week")
        if isinstance(wk, int):
            all_weeks.add(wk)
        start_iso = g.get("startDate") or g.get("start_date")
        if isinstance(start_iso, str):
            try:
                gd = dt.datetime.fromisoformat(start_iso[:19])
            except Exception:
                gd = None
            if gd is not None and gd >= now and isinstance(wk, int):
                future_weeks.append(wk)
    if future_weeks:
        return min(future_weeks)
    if all_weeks:
        return max(all_weeks)
    return 1

def extract_total_and_home_spread(lines_payload: List[Dict[str, Any]]) -> Dict[int, Dict[str, Optional[float]]]:
    """
    Normalize {game_id: {total, home_spread}}.
    CFBD /lines format: per game, 'lines' is a list of books; each may have:
      moneyline: { homePrice, awayPrice }
      spread:    { spread }          # home spread (negative => home favored)
      overUnder  OR total: number
    We take the first book that provides both total and spread.
    """
    out: Dict[int, Dict[str, Optional[float]]] = {}
    for obj in lines_payload or []:
        gid = obj.get("id")
        if gid is None:
            continue
        total = None
        home_spread = None
        books = obj.get("lines", []) or []
        for book in books:
            t = book.get("overUnder", None)
            if t is None:
                t = book.get("total", None)
            sp = book.get("spread", None)
            if isinstance(sp, dict):
                sp = sp.get("spread", None)
            if t is not None and sp is not None:
                total = float(t)
                home_spread = float(sp)
                break
        out[int(gid)] = {"total": total, "home_spread": home_spread}
    return out

def build_scores(year: int, week: int) -> List[Dict[str, Any]]:
    params = {"year": year, "week": week, "seasonType": "regular"}
    games = http_get("/games", params)
    lines = http_get("/lines", params)
    lm = extract_total_and_home_spread(lines)

    rows: List[Dict[str, Any]] = []
    for g in games or []:
        if g.get("week") != week:
            continue
        gid = g.get("id")
        if gid is None:
            continue
        gid = int(gid)

        home = g.get("homeTeam")
        away = g.get("awayTeam")
        start_iso = g.get("startDate") or g.get("start_date") or ""

        totals = lm.get(gid, {})
        total = totals.get("total")
        home_spread = totals.get("home_spread")

        if total is None or home_spread is None:
            continue

        m = -float(home_spread)
        T = float(total)

        home_pred = (T + m) / 2.0
        away_pred = (T - m) / 2.0

        rows.append({
            "game_id": str(gid),
            "kickoff_utc": start_iso,
            "home": home,
            "away": away,
            "total": round(T, 1),
            "home_spread": round(home_spread, 1),
            "home_pred": round(home_pred, 1),
            "away_pred": round(away_pred, 1)
        })

    rows.sort(key=lambda r: r["kickoff_utc"] or "")
    return rows

def main() -> None:
    if WEEK_ENV:
        try:
            week = int(WEEK_ENV)
        except Exception:
            die("CFB_WEEK is not an integer.")
    else:
        week = autodetect_week(YEAR)

    rows = build_scores(YEAR, week)

    if not rows:
        for wk in parse_scan_range(SCAN_ENV):
            if wk == week:
                continue
            try:
                alt = build_scores(YEAR, wk)
            except Exception:
                alt = []
            if alt:
                rows = alt
                week = wk
                break

    with open("scores.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {len(rows)} games to scores.json (year={YEAR}, week={week})")

if __name__ == "__main__":
    main()

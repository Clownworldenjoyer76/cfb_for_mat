#!/usr/bin/env python3
"""
generate_scores.py
- Creates score predictions from CFBD market totals and home spreads.
- Optional realism layers (off by default):
    * Score snapping to plausible football scores
    * Margin shrink toward mean
    * Total clamp band
- Writes scores.json at repo root for the static site.

Env:
  CFBD_API_KEY       -> CollegeFootballData.com API key (Actions secret)
  CFB_YEAR           -> optional (defaults to current UTC year if unset or blank)
  CFB_WEEK           -> optional (if absent or blank, autodetects)
  CFB_SCAN_WEEKS     -> optional scan range if initial week has no usable lines (default "1-20")

  # Optional realism toggles (all OFF unless explicitly set)
  CFB_SNAP_SCORES    -> "1" to snap to football-like scores (2/3/6/7/8 pt combos)
  CFB_SHRINK         -> e.g., "0.15" to shrink margin by 15% (0.00–0.50)
  CFB_TOTAL_MIN      -> e.g., "34"  (clamp total lower bound)
  CFB_TOTAL_MAX      -> e.g., "78"  (clamp total upper bound)

Output:
  scores.json -> array with raw and (if enabled) snapped scores
"""

import os
import json
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

BASE = "https://api.collegefootballdata.com"

def die(msg: str) -> None:
    raise SystemExit(msg)

# ---- Config (safe handling of blank env) ----
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

# Optional toggles (defaults: OFF / no change)
SNAP_ON = (os.getenv("CFB_SNAP_SCORES", "").strip() == "1")
def _parse_float(env_key: str, default: Optional[float]) -> Optional[float]:
    v = os.getenv(env_key)
    if v is None or not v.strip():
        return default
    try:
        return float(v.strip())
    except Exception:
        return default
SHRINK = _parse_float("CFB_SHRINK", 0.0)  # 0.0 = no shrink
TOTAL_MIN = _parse_float("CFB_TOTAL_MIN", None)
TOTAL_MAX = _parse_float("CFB_TOTAL_MAX", None)

# ---- HTTP ----
import requests

def http_get(path: str, params: Dict[str, Any]) -> Any:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(f"{BASE}{path}", headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

# ---- Helpers: math + realism layers ----
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
    Normalize {game_id: {total, home_spread}} from CFBD /lines.
    We take the first book that provides both total and spread.
    """
    out: Dict[int, Dict[str, Optional[float]]] = {}
    for obj in lines_payload or []:
        gid = obj.get("id")
        if gid is None:
            continue
        total = None
        home_spread = None
        for book in (obj.get("lines", []) or []):
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

def apply_total_clamp(total: float) -> float:
    if TOTAL_MIN is not None and total < TOTAL_MIN:
        return TOTAL_MIN
    if TOTAL_MAX is not None and total > TOTAL_MAX:
        return TOTAL_MAX
    return total

def apply_shrink(margin: float) -> float:
    # SHRINK in [0.0, 0.5] gently pulls extreme margins toward zero.
    if SHRINK and SHRINK > 0.0:
        return margin * (1.0 - max(0.0, min(SHRINK, 0.5)))
    return margin

# Precompute plausible football scores (0..80) via combinations of 2/3/6/7/8
def _plausible_scores(max_points: int = 80) -> List[int]:
    scores = set([0])
    increments = [2, 3, 6, 7, 8]
    # bounded knapsack
    frontier = {0}
    while frontier:
        new = set()
        for s in frontier:
            for inc in increments:
                v = s + inc
                if v <= max_points and v not in scores:
                    scores.add(v)
                    new.add(v)
        frontier = new
    return sorted(scores)

_PLAUSIBLE = _plausible_scores(80)

def _nearest_plausible(x: float) -> int:
    # nearest integer within plausible set
    xi = round(x)
    best = None
    best_err = 1e9
    for s in _PLAUSIBLE:
        err = abs(s - xi)
        if err < best_err or (err == best_err and s > (best or -999)):
            best = s
            best_err = err
    return int(best if best is not None else xi)

def snap_scores(home_raw: float, away_raw: float) -> Tuple[int, int]:
    # Preserve total/margin approximately while snapping to plausible scores.
    # Try small neighborhood around raw values and pick pair closest in L1 distance.
    candidates_home = {_nearest_plausible(home_raw)}
    candidates_away = {_nearest_plausible(away_raw)}
    # add +/-1 nearest integers snapped
    for d in (-1, 1):
        candidates_home.add(_nearest_plausible(home_raw + d))
        candidates_away.add(_nearest_plausible(away_raw + d))
    best_pair = (int(round(home_raw)), int(round(away_raw)))
    best_err = abs(best_pair[0] - home_raw) + abs(best_pair[1] - away_raw)
    for h in candidates_home:
        for a in candidates_away:
            err = abs(h - home_raw) + abs(a - away_raw)
            if err < best_err or (err == best_err and (h + a) > sum(best_pair)):
                best_pair = (h, a)
                best_err = err
    return best_pair

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

        # Optional total clamp
        T = apply_total_clamp(float(total))
        # Convert home spread (home spread negative => home favored)
        m_raw = -float(home_spread)
        # Optional margin shrink
        m = apply_shrink(m_raw)

        home_raw = (T + m) / 2.0
        away_raw = (T - m) / 2.0

        if SNAP_ON:
            home_snap, away_snap = snap_scores(home_raw, away_raw)
        else:
            home_snap, away_snap = None, None

        row = {
            "game_id": str(gid),
            "kickoff_utc": start_iso,
            "home": home,
            "away": away,
            "total": round(T, 1),
            "home_spread": round(float(home_spread), 1),
            "home_pred": round(home_raw, 1),
            "away_pred": round(away_raw, 1)
        }
        if SNAP_ON:
            row["home_pred_snapped"] = int(home_snap)
            row["away_pred_snapped"] = int(away_snap)

        rows.append(row)

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

    print(f"Wrote {len(rows)} games to scores.json (year={YEAR}, week={week}, snap={SNAP_ON}, shrink={SHRINK})")

if __name__ == "__main__":
    main()

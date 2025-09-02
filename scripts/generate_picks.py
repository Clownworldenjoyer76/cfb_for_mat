#!/usr/bin/env python3
"""
generate_picks.py
- Pulls this week's CFB games & betting lines using CollegeFootballData HTTP API.
- Chooses a conservative pick per game (favorite ML if available, else favorite spread).
- Writes picks.json for your static HTML site.

Env:
  CFBD_API_KEY   -> your CollegeFootballData.com API key (GitHub Actions secret)
  CFB_YEAR       -> optional override (defaults to current UTC year)
  CFB_WEEK       -> optional override (if absent, script auto-detects the nearest valid week)

Output:
  picks.json     -> array of picks with fields your site expects
"""

import os
import json
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

BASE = "https://api.collegefootballdata.com"

def die(msg: str) -> None:
    raise SystemExit(msg)

# ---------- Config ----------
API_KEY = os.getenv("CFBD_API_KEY")
if not API_KEY:
    die("CFBD_API_KEY is not set (Actions Secret).")

YEAR = int(os.getenv("CFB_YEAR", dt.datetime.utcnow().year))
WEEK_ENV = os.getenv("CFB_WEEK")

# ---------- HTTP ----------
import requests  # requires 'requests' in requirements.txt

def http_get(path: str, params: Dict[str, Any]) -> Any:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(f"{BASE}{path}", headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

# ---------- Helpers ----------
def implied_prob_from_moneyline(ml: Optional[float]) -> Optional[float]:
    if ml is None:
        return None
    try:
        ml = float(ml)
    except Exception:
        return None
    if ml >= 0:
        return 100.0 / (ml + 100.0)
    return (-ml) / ((-ml) + 100.0)

def pick_from_lines(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Given a normalized line row:
      {home_team, away_team, home_ml, away_ml, spread}
    Return a conservative pick dict.
    """
    home_ml = row.get("home_ml")
    away_ml = row.get("away_ml")
    spread = row.get("spread")

    hp = implied_prob_from_moneyline(home_ml) if home_ml is not None else None
    ap = implied_prob_from_moneyline(away_ml) if away_ml is not None else None

    # Prefer ML favorite if available
    if hp is not None or ap is not None:
        if hp is not None and (ap is None or hp >= ap):
            return {
                "market": "ML",
                "selection": row["home_team"],
                "line": 0.0,
                "odds": float(home_ml),
                "model_prob": round(max(min(hp, 0.99), 0.50), 4),
            }
        if ap is not None:
            return {
                "market": "ML",
                "selection": row["away_team"],
                "line": 0.0,
                "odds": float(away_ml),
                "model_prob": round(max(min(ap, 0.99), 0.50), 4),
            }

    # Fallback to favorite spread if no ML
    if spread is not None:
        try:
            sp = float(spread)
        except Exception:
            sp = None
        if sp is not None:
            # CFBD spreads are typically home spread (negative = home favored)
            if sp < 0:
                selection = row["home_team"]
                fav_line = sp
            else:
                selection = row["away_team"]
                fav_line = -sp
            model_prob = max(min(0.50 + (abs(fav_line) / 50.0), 0.90), 0.55)
            return {
                "market": "SPREAD",
                "selection": selection,
                "line": fav_line,
                "odds": -110,
                "model_prob": round(model_prob, 4),
            }

    return None

def autodetect_week(year: int) -> int:
    """
    Auto-detect the nearest valid week using the year's games.
    Strategy:
      - Fetch all regular-season games for the year.
      - If any games are today or in the future (UTC), pick the smallest week among them.
      - Otherwise pick the largest week present (season completed / late in season).
    """
    games = http_get("/games", {"year": year, "seasonType": "regular"})
    now = dt.datetime.utcnow()
    future_weeks = []
    all_weeks = set()

    for g in games or []:
        wk = g.get("week")
        if isinstance(wk, int):
            all_weeks.add(wk)
        # Start dates can be 'startDate' ISO strings; guard defensively.
        start_iso = g.get("startDate") or g.get("start_date")
        try:
            if isinstance(start_iso, str):
                # CFBD uses ISO8601; parse minimal (YYYY-MM-DD or full).
                # Only compare date portion; treat same-day as future/ongoing.
                dt_part = start_iso[:19]
                try:
                    gd = dt.datetime.fromisoformat(dt_part)
                except Exception:
                    # Fallback: just compare strings lexicographically for YYYY-MM-DD
                    gd = None
                if gd is not None and gd >= now:
                    if isinstance(wk, int):
                        future_weeks.append(wk)
        except Exception:
            pass

    if future_weeks:
        return min(future_weeks)
    if all_weeks:
        return max(all_weeks)
    # Fallback if no week info at all
    return 1

def normalize_http_lines(lines_payload: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    CFBD /lines: each item is a game with 'lines' (books).
    Normalize to {game_id: {home_team, away_team, home_ml, away_ml, spread}}.
    """
    result: Dict[int, Dict[str, Any]] = {}
    for obj in lines_payload or []:
        gid = obj.get("id")
        if gid is None:
            continue
        home = obj.get("homeTeam")
        away = obj.get("awayTeam")

        home_ml = None
        away_ml = None
        spread = None

        for book in obj.get("lines", []) or []:
            ml = book.get("moneyline")
            if isinstance(ml, dict):
                # keys commonly: homePrice, awayPrice
                home_ml = ml.get("homePrice", home_ml)
                away_ml = ml.get("awayPrice", away_ml)
            sp = book.get("spread")
            if isinstance(sp, dict):
                # key commonly: spread (home spread; negative if home favored)
                spread = sp.get("spread", spread)

            if home_ml is not None or away_ml is not None or spread is not None:
                break

        result[int(gid)] = {
            "home_team": home,
            "away_team": away,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "spread": spread,
        }
    return result

def main() -> None:
    # Resolve target week
    if WEEK_ENV:
        try:
            week = int(WEEK_ENV)
        except Exception:
            die("CFB_WEEK is not an integer.")
    else:
        week = autodetect_week(YEAR)

    # Fetch games and lines for the chosen week
    params = {"year": YEAR, "week": week, "seasonType": "regular"}
    games = http_get("/games", params)
    lines = http_get("/lines", params)

    line_map = normalize_http_lines(lines)

    rows: List[Dict[str, Any]] = []
    for g in games or []:
        # Only include games from the chosen week (guard if API returns more)
        if g.get("week") != week:
            continue

        gid = g.get("id")
        if gid is None:
            continue
        gid = int(gid)

        home = g.get("homeTeam")
        away = g.get("awayTeam")
        start_iso = g.get("startDate") or g.get("start_date") or ""

        lm = line_map.get(gid, {
            "home_team": home,
            "away_team": away,
            "home_ml": None,
            "away_ml": None,
            "spread": None
        })

        pick = pick_from_lines(lm)
        if not pick:
            continue

        # implied prob for edge
        if pick["market"] == "ML":
            imp = implied_prob_from_moneyline(pick["odds"]) or 0.5238
        else:
            imp = 0.5238  # -110 baseline

        edge = round(max(min(pick["model_prob"] - imp, 0.49), -0.49), 4)

        rows.append({
            "game_id": str(gid),
            "kickoff_utc": start_iso,
            "home": home,
            "away": away,
            "market": pick["market"],
            "selection": pick["selection"],
            "line": pick["line"],
            "odds": pick["odds"],
            "book": "CFBD",
            "model_prob": pick["model_prob"],
            "edge": edge
        })

    # Sort by kickoff and write file
    rows.sort(key=lambda r: r["kickoff_utc"] or "")
    with open("picks.json", "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {len(rows)} picks to picks.json (year={YEAR}, week={week})")

if __name__ == "__main__":
    main()

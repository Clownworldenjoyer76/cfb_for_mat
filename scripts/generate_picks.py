#!/usr/bin/env python3
"""
generate_picks.py
- Pulls this week's CFB games & betting lines using sportsdataverse-py (CFBD).
- Chooses a conservative pick per game (favorite ML if available, else favorite spread).
- Writes picks.json for your static HTML site.

Env:
  CFBD_API_KEY   -> your CollegeFootballData.com API key (GitHub Actions secret)
  CFB_YEAR       -> optional override (defaults to current year)
  CFB_WEEK       -> optional override (defaults to 1 if autodetect not available)

Output:
  picks.json     -> array of picks with fields your site expects
"""

import os
import json
import datetime as dt
from typing import Any, Dict, List, Optional

# ---------- Config ----------
API_KEY = os.getenv("CFBD_API_KEY")
if not API_KEY:
    raise SystemExit("CFBD_API_KEY is not set (Actions Secret).")

YEAR = int(os.getenv("CFB_YEAR", dt.datetime.utcnow().year))
# Week autodetection via sportsdataverse is inconsistent; default to 1 unless you override:
WEEK = int(os.getenv("CFB_WEEK", "1"))

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

    # Prefer ML favorite if available
    hp = implied_prob_from_moneyline(home_ml) if home_ml is not None else None
    ap = implied_prob_from_moneyline(away_ml) if away_ml is not None else None

    if hp is not None or ap is not None:
        # choose higher implied probability
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
            # conservative prob mapping from spread magnitude
            model_prob = max(min(0.50 + (abs(fav_line) / 50.0), 0.90), 0.55)
            return {
                "market": "SPREAD",
                "selection": selection,
                "line": fav_line,
                "odds": -110,
                "model_prob": round(model_prob, 4),
            }

    return None

def normalize_sv_lines(lines: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    sportsdataverse-py returns a list of books per game.
    Normalize to {game_id: {home_team, away_team, home_ml, away_ml, spread}}
    by taking the first book that has usable data.
    """
    result: Dict[int, Dict[str, Any]] = {}
    for obj in lines or []:
        # Expected keys (varies by SDV version). We defensively access.
        game_id = obj.get("id") or obj.get("game_id")
        if game_id is None:
            continue
        home = obj.get("home_team")
        away = obj.get("away_team")

        home_ml = None
        away_ml = None
        spread = None

        for book in obj.get("lines", []) or []:
            # moneyline
            ml = book.get("moneyline")
            if isinstance(ml, dict):
                home_ml = ml.get("home_price", home_ml)
                away_ml = ml.get("away_price", away_ml)
            # spread
            sp = book.get("spread")
            if isinstance(sp, dict):
                # sp['spread'] is home spread (negative if home favored)
                spread = sp.get("spread", spread)

            if home_ml is not None or away_ml is not None or spread is not None:
                break

        result[int(game_id)] = {
            "home_team": home,
            "away_team": away,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "spread": spread,
        }
    return result

# ---------- Data fetch (sportsdataverse-py first, HTTP fallback) ----------
def fetch_with_sdv(year: int, week: int):
    from sportsdataverse import cfbd as sdv_cfbd
    games = sdv_cfbd.get_games(year=year, week=week, season_type="regular", authorization=API_KEY)
    lines = sdv_cfbd.get_lines(year=year, week=week, season_type="regular", authorization=API_KEY)
    # Ensure lists of dicts
    if hasattr(games, "to_dict"):
        games = games.to_dict("records")
    if hasattr(lines, "to_dict"):
        lines = lines.to_dict("records")
    return games, lines

def fetch_with_http(year: int, week: int):
    import requests
    base = "https://api.collegefootballdata.com"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"year": year, "week": week, "seasonType": "regular"}

    g = requests.get(f"{base}/games", headers=headers, params=params, timeout=60)
    g.raise_for_status()
    games = g.json()

    l = requests.get(f"{base}/lines", headers=headers, params=params, timeout=60)
    l.raise_for_status()
    lines = l.json()
    return games, lines

def main():
    # Try SDV first; if anything goes wrong, fall back to HTTP
    try:
        games, lines = fetch_with_sdv(YEAR, WEEK)
        using = "sportsdataverse"
    except Exception as e:
        print(f"[warn] sportsdataverse failed, falling back to HTTP: {e}")
        games, lines = fetch_with_http(YEAR, WEEK)
        using = "http"

    # Normalize line objects
    line_map = normalize_sv_lines(lines) if using == "sportsdataverse" else {}
    if using == "http":
        # HTTP shape: each item is a game with a list of books; normalize similarly
        tmp = []
        for obj in lines or []:
            # http payload uses 'id' consistently
            gid = obj.get("id")
            if gid is None:
                continue
            home = obj.get("homeTeam")
            away = obj.get("awayTeam")
            home_ml = away_ml = spread = None
            for book in obj.get("lines", []) or []:
                ml = book.get("moneyline")
                if isinstance(ml, dict):
                    home_ml = ml.get("homePrice", home_ml)
                    away_ml = ml.get("awayPrice", away_ml)
                sp = book.get("spread")
                if isinstance(sp, dict):
                    spread = sp.get("spread", spread)
                if home_ml is not None or away_ml is not None or spread is not None:
                    break
            tmp.append({
                "id": gid,
                "home_team": home,
                "away_team": away,
                "home_ml": home_ml,
                "away_ml": away_ml,
                "spread": spread
            })
        line_map = {int(x["id"]): x for x in tmp}

    # Build picks
    rows: List[Dict[str, Any]] = []
    for g in games or []:
        gid = g.get("id") or g.get("game_id")
        if gid is None:
            continue
        gid = int(gid)
        home = g.get("home_team") or g.get("homeTeam")
        away = g.get("away_team") or g.get("awayTeam")
        start_iso = g.get("start_date") or g.get("startDate") or g.get("start_time_tbd")

        kickoff_utc = None
        if isinstance(start_iso, str):
            kickoff_utc = start_iso
        else:
            kickoff_utc = ""

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
            "kickoff_utc": kickoff_utc,
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
    print(f"Wrote {len(rows)} picks to picks.json (year={YEAR}, week={WEEK})")

if __name__ == "__main__":
    main()

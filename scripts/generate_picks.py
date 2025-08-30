import os, json, math, datetime as dt
from dateutil import tz
import pandas as pd

# CFBD official Python client
from cfbd import Configuration, ApiClient
from cfbd.api.games_api import GamesApi
from cfbd.api.betting_api import BettingApi
from cfbd.api.calendar_api import CalendarApi

# ---------- Config ----------
API_KEY = os.getenv("CFBD_API_KEY")
if not API_KEY:
    raise SystemExit("CFBD_API_KEY is not set (Actions Secret).")

# Use America/New_York for your timeline preference
TZ = tz.gettz("America/New_York")
today_local = dt.datetime.now(TZ).date()

# ---------- CFBD client setup ----------
cfg = Configuration()
cfg.api_key["Authorization"] = API_KEY
cfg.api_key_prefix["Authorization"] = "Bearer"

def implied_prob_from_moneyline(ml: float) -> float:
    # ML to implied probability
    if ml is None:
        return None
    try:
        ml = float(ml)
    except:
        return None
    if ml >= 0:
        return 100.0 / (ml + 100.0)
    else:
        return (-ml) / ((-ml) + 100.0)

def pick_from_lines(lines_row):
    """
    Choose a conservative pick from a betting line row.
    Priority:
      1) Favorite moneyline if available (safer).
      2) Favorite spread if ML not available.
    Returns dict with market, selection, line, odds, model_prob.
    """
    # moneyline fields vary; cfbd returns list of books and prices.
    # We’ll scan for any ML and spread from the first available book.
    home_ml = lines_row.get("home_moneyline")
    away_ml = lines_row.get("away_moneyline")
    spread = lines_row.get("spread")
    # favorite detection
    pick = None

    # Prefer ML favorite
    home_ml_prob = implied_prob_from_moneyline(home_ml) if home_ml is not None else None
    away_ml_prob = implied_prob_from_moneyline(away_ml) if away_ml is not None else None

    if home_ml_prob is not None or away_ml_prob is not None:
        if home_ml_prob is not None and (away_ml_prob is None or home_ml_prob > away_ml_prob):
            pick = {
                "market": "ML",
                "selection": lines_row["home_team"],
                "line": 0.0,
                "odds": float(home_ml),
                "model_prob": round(min(max(home_ml_prob, 0.50), 0.99), 4)
            }
        elif away_ml_prob is not None:
            pick = {
                "market": "ML",
                "selection": lines_row["away_team"],
                "line": 0.0,
                "odds": float(away_ml),
                "model_prob": round(min(max(away_ml_prob, 0.50), 0.99), 4)
            }

    # Fallback to spread (use favorite minus points)
    if pick is None and spread is not None:
        try:
            sp = float(spread)
        except:
            sp = None
        if sp is not None:
            # cfbd spread is typically home spread (negative if home favorite)
            if sp < 0:
                selection = f"{lines_row['home_team']}"
                fav_line = sp
                # crude probability map from spread (conservative)
                model_prob = min(max(0.50 + (abs(sp)/50.0), 0.55), 0.90)
            else:
                selection = f"{lines_row['away_team']}"
                fav_line = -sp
                model_prob = min(max(0.50 + (abs(sp)/50.0), 0.55), 0.90)
            pick = {
                "market": "SPREAD",
                "selection": selection,
                "line": fav_line,
                "odds": -110,
                "model_prob": round(model_prob, 4)
            }

    return pick

def flatten_line_obj(obj):
    """
    cfbd Betting API returns a list of books per game.
    We normalize into a single representative line (first book that has ML or spread).
    """
    base = {
        "game_id": obj.id,
        "home_team": obj.home_team,
        "away_team": obj.away_team,
        "start_date": obj.start_date
    }
    home_ml = away_ml = spread = None

    # iterate books to find first with ML or spread
    for book in (obj.lines or []):
        # moneylines
        if hasattr(book, "moneyline"):
            ml = book.moneyline
            # ml.home_price / ml.away_price
            try:
                if getattr(ml, "home_price", None) is not None:
                    home_ml = ml.home_price
                if getattr(ml, "away_price", None) is not None:
                    away_ml = ml.away_price
            except:
                pass
        # spreads
        if hasattr(book, "spread"):
            sp = book.spread
            # sp.spread is home spread (negative if home favored)
            try:
                if getattr(sp, "spread", None) is not None:
                    spread = sp.spread
            except:
                pass
        # if we have either ML or spread, stop scanning
        if home_ml is not None or away_ml is not None or spread is not None:
            break

    base.update({
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
        "spread": spread
    })
    return base

def determine_current_week(calendar):
    # Find the calendar entry where today falls between start/end dates
    for c in calendar:
        # c.first_game_start, c.last_game_start are ISO strings
        try:
            start = dt.datetime.fromisoformat(c.first_game_start.replace("Z","+00:00")).date()
            end = dt.datetime.fromisoformat(c.last_game_start.replace("Z","+00:00")).date()
        except:
            continue
        if start <= today_local <= end and c.season_type == "regular":
            return c.week
    # fallback to the nearest upcoming regular-season week
    for c in calendar:
        if c.season_type == "regular":
            return c.week
    return None

with ApiClient(cfg) as api:
    cal_api = CalendarApi(api)
    year = today_local.year
    cal = cal_api.get_calendar(year=year)
    week = determine_current_week(cal) or 1

    games_api = GamesApi(api)
    bet_api = BettingApi(api)

    # Get games & lines for the computed week
    games = games_api.get_games(year=year, week=week, season_type="regular")
    lines = bet_api.get_lines(year=year, week=week, season_type="regular")

# Index lines by game id for quick lookup
line_map = {}
for obj in lines or []:
    try:
        flat = flatten_line_obj(obj)
        line_map[flat["game_id"]] = flat
    except Exception:
        continue

# Build picks JSON
rows = []
for g in games or []:
    gid = g.id
    start_iso = g.start_date
    # normalize kickoff
    kickoff_utc = None
    try:
        kickoff_utc = dt.datetime.fromisoformat(start_iso.replace("Z","+00:00")).astimezone(dt.timezone.utc).isoformat()
    except:
        kickoff_utc = start_iso

    lm = line_map.get(gid, {
        "home_team": g.home_team,
        "away_team": g.away_team,
        "home_moneyline": None,
        "away_moneyline": None,
        "spread": None,
    })

    pick = pick_from_lines(lm)
    if not pick:
        # skip if we have nothing usable
        continue

    # implied from odds for edge (if ML), else from -110 ≈ 52.38%
    if pick["market"] == "ML":
        imp = implied_prob_from_moneyline(pick["odds"]) or 0.52
    else:
        imp = 0.5238

    edge = max(round(pick["model_prob"] - imp, 4), -0.49)

    rows.append({
        "game_id": str(gid),
        "kickoff_utc": kickoff_utc,
        "home": g.home_team,
        "away": g.away_team,
        "market": pick["market"],
        "selection": pick["selection"],
        "line": pick["line"],
        "odds": pick["odds"],
        "book": "CFBD",
        "model_prob": pick["model_prob"],
        "edge": edge
    })

# Sort by kickoff
rows.sort(key=lambda r: r["kickoff_utc"] or "")

# Write picks.json
out_path = "picks.json"
with open(out_path, "w") as f:
    json.dump(rows, f, indent=2)

print(f"Wrote {len(rows)} picks to {out_path}")

#!/usr/bin/env python3
"""
scripts/parse_cfbd_lines_to_csv.py

Reads cfbd_lines.json (written by scripts/get_market_lines.py) and
writes/updates market_lines.csv in the repo root.

- Accepts either a top-level list of games OR {"data": [...]} shape.
- DraftKings-first selection with graceful fallback to other books.
- Tracks which book supplied each metric (book_spread, book_total, book_ml).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import csv
import sys
import datetime as dt


# ---- Config ----

REPO_ROOT = Path(".")
IN_JSON = REPO_ROOT / "cfbd_lines.json"
OUT_CSV = REPO_ROOT / "market_lines.csv"

# DraftKings first, then common books (extend as needed).
BOOK_ORDER = [
    "DraftKings",
    "ESPN Bet",
    "FanDuel",
    "Caesars",
    "BetMGM",
    "PointsBet",
    "BetRivers",
    "Pinnacle",
    "Circa",
    "SuperBook",
    "Barstool",
    "Bovada",
    "William Hill",
    "WynnBET",
]

CSV_FIELDS = [
    "year",
    "season_type",
    "week",
    "game_id",
    "kickoff_utc",
    "home_team",
    "away_team",
    "spread",
    "total",
    "home_ml",
    "away_ml",
    "book_spread",
    "book_total",
    "book_ml",
]


# ---- Helpers ----

def _load_games(any_obj: Any) -> List[Dict[str, Any]]:
    """
    Accept either:
      - a list of game objects, or
      - {"data": [ ... ]} (common API wrapper)
    """
    if isinstance(any_obj, list):
        games = any_obj
    elif isinstance(any_obj, dict) and "data" in any_obj and isinstance(any_obj["data"], list):
        games = any_obj["data"]
    else:
        raise ValueError("cfbd_lines.json must contain a list of games or a {'data': [...]} object")

    return games


def _first_present(provider_map: Dict[str, Dict[str, Any]], field: str) -> (Optional[Any], Optional[str]):
    """Pick the first non-None value for `field` using BOOK_ORDER."""
    for book in BOOK_ORDER:
        info = provider_map.get(book)
        if not info:
            continue
        val = info.get(field)
        if val is not None:
            return val, book
    return None, None


def normalize_game(game: Dict[str, Any]) -> Dict[str, Any]:
    # Core fields
    year = game.get("season") or game.get("year")
    season_type = (game.get("seasonType") or game.get("season_type") or "").lower() or "regular"
    week = game.get("week")
    game_id = game.get("id") or game.get("game_id")
    kickoff = game.get("startDate") or game.get("kickoff_utc")
    home = game.get("homeTeam") or game.get("home_team")
    away = game.get("awayTeam") or game.get("away_team")

    # Compile provider lines into a dict keyed by provider name
    provider_map: Dict[str, Dict[str, Any]] = {}
    for ln in game.get("lines", []) or []:
        prov = ln.get("provider")
        if not prov:
            continue
        provider_map[prov] = {
            "spread": ln.get("spread"),
            "total": ln.get("overUnder"),
            "home_ml": ln.get("homeMoneyline"),
            "away_ml": ln.get("awayMoneyline"),
        }

    # Select with DK-first fallback
    spread, book_spread = _first_present(provider_map, "spread")
    total, book_total = _first_present(provider_map, "total")

    # Moneylines: try to take both from the same provider when possible,
    # but still allow independent fallback if one side is missing.
    home_ml, ml_book_h = _first_present(provider_map, "home_ml")
    away_ml, ml_book_a = _first_present(provider_map, "away_ml")
    book_ml = ml_book_h if ml_book_h == ml_book_a and ml_book_h is not None else (ml_book_h or ml_book_a)

    return {
        "year": year,
        "season_type": season_type,
        "week": week,
        "game_id": game_id,
        "kickoff_utc": kickoff,
        "home_team": home,
        "away_team": away,
        "spread": spread,
        "total": total,
        "home_ml": home_ml,
        "away_ml": away_ml,
        "book_spread": book_spread,
        "book_total": book_total,
        "book_ml": book_ml,
    }


def main() -> None:
    if not IN_JSON.exists():
        raise FileNotFoundError(f"Missing {IN_JSON} – upstream step should save it.")

    with IN_JSON.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    games = _load_games(raw)

    if not isinstance(games, list) or not games:
        raise ValueError("cfbd_lines.json contains no games to parse")

    rows: List[Dict[str, Any]] = []
    for g in games:
        try:
            rows.append(normalize_game(g))
        except Exception as e:
            # Keep going but surface context on stderr
            print(f"[warn] skipped game {g.get('id') or g.get('game_id')}: {e}", file=sys.stderr)

    # Sort for stability
    rows.sort(key=lambda r: (r.get("year") or 0, r.get("week") or 0, r.get("game_id") or 0))

    # Write CSV (overwrite)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) if r.get(k) is not None else "" for k in CSV_FIELDS})

    print(f"Wrote {OUT_CSV} with {len(rows)} rows.")


if __name__ == "__main__":
    main()

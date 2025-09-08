#!/usr/bin/env python3
# /mnt/data/cfb_for_mat-main/cfb_for_mat-main/scripts/parse_cfbd_lines_to_csv.py
"""
parse_cfbd_lines_to_csv.py

Purpose:
  Convert raw CFBD audit JSON (cfbd_lines.json or cfbd_lines.json.txt) into
  a normalized CSV (market_lines.csv) using preferred book order with fallbacks.

Behavior:
  - Preferred book order: ["ESPN Bet", "Bovada"] then any others as final fallback.
  - Each field (spread, home_ml, away_ml, total) is filled independently:
      try preferred book #1, else preferred #2, else first other book with a number.
  - Spread is written as HOME spread (positive = home underdog), consistent with CFBD.

Input (repo root):
  - cfbd_lines.json        (preferred)
  - cfbd_lines.json.txt    (fallback)

Output (repo root):
  - market_lines.csv with columns:
      year,season_type,week,game_id,kickoff_utc,
      home_team,away_team,
      spread,total,home_ml,away_ml,book_spread,book_total,book_ml

Notes:
  - Supports JSON shaped as { "data": [ ... ] } or plain [ ... ].
  - Ignores clearly invalid ML sentinels (e.g., -100000).
"""

import os
import sys
import json
import csv
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREFERRED_JSON = os.path.join(REPO_ROOT, "cfbd_lines.json")
FALLBACK_JSON = os.path.join(REPO_ROOT, "cfbd_lines.json.txt")
OUTPUT_CSV = os.path.join(REPO_ROOT, "market_lines.csv")

PREFERRED_BOOKS = ["ESPN Bet", "Bovada"]  # ordered preference


def _num(x: Any) -> Optional[float]:
    """Parse numeric safely; return None if not parseable."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            return float(x)
        except Exception:
            return None
    try:
        s = str(x).strip()
        if not s:
            return None
        s = s.replace("%", "")
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None


def _valid_ml(v: Optional[float]) -> bool:
    """Filter out absurd/sentinel moneylines seen in feeds."""
    if v is None:
        return False
    # Typical ML range [-10000, +10000]; reject extreme sentinel values
    return -10000.0 <= v <= 10000.0


def _load_games() -> List[Dict[str, Any]]:
    """Load raw JSON and return the list of game objects."""
    path = PREFERRED_JSON if os.path.exists(PREFERRED_JSON) else FALLBACK_JSON
    if not os.path.exists(path):
        print("[error] Missing cfbd_lines.json and cfbd_lines.json.txt in repo root.", file=sys.stderr)
        return []
    with open(path, "r", encoding="utf-8") as f:
        root = json.load(f)

    if isinstance(root, dict):
        if isinstance(root.get("data"), list):
            return root["data"]
        # conservative unwraps if present
        for key in ("items", "results", "payload"):
            if isinstance(root.get(key), list):
                return root[key]
    if isinstance(root, list):
        return root

    print("[warn] Unrecognized JSON shape; returning empty list.", file=sys.stderr)
    return []


def _collect_books(game: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a clean list of book dicts from game['lines']."""
    lines = game.get("lines") or []
    return [b for b in lines if isinstance(b, dict)]


def _extract_fields_from_book(book: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Return (spread_home, total, home_ml, away_ml) from a single book entry.
    - spread_home is CFBD home spread number (positive = home underdog).
    """
    spread_home = None
    total = None
    home_ml = None
    away_ml = None

    sp = book.get("spread")
    if isinstance(sp, dict):
        spread_home = _num(sp.get("spread"))

    # totals sometimes appear as 'overUnder' or 'total'
    total = _num(book.get("overUnder"))
    if total is None:
        total = _num(book.get("total"))

    ml = book.get("moneyline")
    if isinstance(ml, dict):
        home_ml = _num(ml.get("homePrice"))
        away_ml = _num(ml.get("awayPrice"))
        if not _valid_ml(home_ml):
            home_ml = None
        if not _valid_ml(away_ml):
            away_ml = None

    return spread_home, total, home_ml, away_ml


def _choose_by_preference(books: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fill each field by preferred order, falling back as needed.
    Returns dict: {spread, total, home_ml, away_ml, book_spread, book_total, book_ml}
    """
    # Prepare lookup by provider
    by_provider: Dict[str, Dict[str, Any]] = {}
    for b in books:
        provider = (b.get("provider") or b.get("book") or "").strip()
        by_provider.setdefault(provider, b)

    # Build search order: preferred then all others (stable)
    others = [p for p in by_provider.keys() if p not in PREFERRED_BOOKS]
    search_order = PREFERRED_BOOKS + others

    result = {
        "spread": None, "total": None, "home_ml": None, "away_ml": None,
        "book_spread": "", "book_total": "", "book_ml": ""
    }

    # spread
    for prov in search_order:
        b = by_provider.get(prov)
        if not b:
            continue
        spread_home, _, _, _ = _extract_fields_from_book(b)
        if spread_home is not None:
            result["spread"] = spread_home
            result["book_spread"] = prov
            break

    # total
    for prov in search_order:
        b = by_provider.get(prov)
        if not b:
            continue
        _, total, _, _ = _extract_fields_from_book(b)
        if total is not None:
            result["total"] = total
            result["book_total"] = prov
            break

    # moneyline (prefer to take both from same provider if available)
    for prov in search_order:
        b = by_provider.get(prov)
        if not b:
            continue
        _, _, hm, am = _extract_fields_from_book(b)
        if hm is not None or am is not None:
            result["home_ml"] = hm
            result["away_ml"] = am
            result["book_ml"] = prov
            break

    return result


def _row_from_game(g: Dict[str, Any]) -> Dict[str, Any]:
    gid = g.get("id") or g.get("game_id") or g.get("gameId")
    week = g.get("week")
    season = g.get("season")
    season_type = g.get("seasonType") or g.get("season_type")
    start_iso = g.get("startDate") or g.get("start_time_tbd") or g.get("start_date") or ""
    home = g.get("homeTeam") or g.get("home_team")
    away = g.get("awayTeam") or g.get("away_team")

    books = _collect_books(g)
    values = _choose_by_preference(books)

    return {
        "year": season,
        "season_type": season_type,
        "week": week,
        "game_id": str(gid) if gid is not None else "",
        "kickoff_utc": start_iso if isinstance(start_iso, str) else "",
        "home_team": home,
        "away_team": away,
        "spread": values["spread"],
        "total": values["total"],
        "home_ml": values["home_ml"],
        "away_ml": values["away_ml"],
        "book_spread": values["book_spread"],
        "book_total": values["book_total"],
        "book_ml": values["book_ml"],
    }


def main() -> None:
    games = _load_games()
    rows: List[Dict[str, Any]] = []
    for g in games:
        if isinstance(g, dict):
            rows.append(_row_from_game(g))

    # Sort by kickoff time then game_id; keep first per game_id
    rows.sort(key=lambda r: (r.get("kickoff_utc") or "", r.get("game_id") or ""))
    seen = set()
    final: List[Dict[str, Any]] = []
    for r in rows:
        gid = r.get("game_id")
        if gid and gid in seen:
            continue
        if gid:
            seen.add(gid)
        final.append(r)

    # Write CSV
    fieldnames = [
        "year","season_type","week","game_id","kickoff_utc",
        "home_team","away_team",
        "spread","total","home_ml","away_ml",
        "book_spread","book_total","book_ml"
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in final:
            w.writerow(r)

    print(f"Wrote {len(final)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

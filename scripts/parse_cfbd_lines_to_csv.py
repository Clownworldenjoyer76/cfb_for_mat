#!/usr/bin/env python3
# /mnt/data/cfb_for_mat-main/cfb_for_mat-main/scripts/parse_cfbd_lines_to_csv.py
"""
parse_cfbd_lines_to_csv.py

Purpose:
  Convert the raw CFBD audit JSON (cfbd_lines.json or cfbd_lines.json.txt)
  into a normalized CSV: market_lines.csv

Input (repo root):
  - cfbd_lines.json        (preferred)
  - cfbd_lines.json.txt    (fallback)

Output (repo root):
  - market_lines.csv with columns:
      year,season_type,week,game_id,kickoff_utc,
      home_team,away_team,
      spread,total,home_ml,away_ml,book

Notes:
  - Picks the first provider ("book") that supplies any usable number.
  - Handles both shapes:
      { "data": [ ... game objects ... ] } or plain [ ... game objects ... ]
  - Numbers are parsed safely; blanks remain empty in CSV.
"""

import os
import sys
import json
import csv
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREFERRED_JSON = os.path.join(REPO_ROOT, "cfbd_lines.json")
FALLBACK_JSON = os.path.join(REPO_ROOT, "cfbd_lines.json.txt")
OUTPUT_CSV = os.path.join(REPO_ROOT, "market_lines.csv")


def _num(x: Any) -> Optional[float]:
    """Parse numeric with common cleanup; return None if not parseable."""
    if x is None:
        return None
    try:
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return None
        s = s.replace("%", "")
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None


def _load_raw() -> List[Dict[str, Any]]:
    """Load the raw JSON and return the list of game objects."""
    path = PREFERRED_JSON if os.path.exists(PREFERRED_JSON) else FALLBACK_JSON
    if not os.path.exists(path):
        print(f"[error] Missing cfbd_lines.json and cfbd_lines.json.txt in repo root.", file=sys.stderr)
        return []
    with open(path, "r", encoding="utf-8") as f:
        root = json.load(f)

    # Accept either {"data":[...]} or plain list
    if isinstance(root, dict) and "data" in root and isinstance(root["data"], list):
        return root["data"]
    if isinstance(root, list):
        return root

    # Some audit files may nest deeper; try a conservative unwrap
    for key in ("items", "results", "payload"):
        if isinstance(root, dict) and key in root and isinstance(root[key], list):
            return root[key]

    print(f"[warn] Unrecognized JSON shape; proceeding with empty list.", file=sys.stderr)
    return []


def _first_book_values(game: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    From game['lines'] list, capture the FIRST provider that gives any usable:
      spread, total (overUnder|total), moneyline home/away.
    Return dict with spread,total,home_ml,away_ml,book (name string).
    """
    spread = total = home_ml = away_ml = None
    book = ""

    lines = game.get("lines") or []
    if not isinstance(lines, list):
        return {"spread": None, "total": None, "home_ml": None, "away_ml": None, "book": ""}

    for b in lines:
        if not isinstance(b, dict):
            continue
        provider = b.get("provider") or b.get("book") or ""
        # spread
        sp = b.get("spread")
        if isinstance(sp, dict):
            if spread is None:
                spread = _num(sp.get("spread"))
                if spread is not None and not book:
                    book = provider
        # moneyline
        ml = b.get("moneyline")
        if isinstance(ml, dict):
            if home_ml is None or away_ml is None:
                hm = _num(ml.get("homePrice"))
                am = _num(ml.get("awayPrice"))
                # only set if at least one is numeric
                if hm is not None or am is not None:
                    home_ml = hm
                    away_ml = am
                    if not book:
                        book = provider
        # total / overUnder (field name may vary)
        ou = b.get("overUnder")
        if ou is None:
            ou = b.get("total")
        if total is None:
            val = _num(ou)
            if val is not None:
                total = val
                if not book:
                    book = provider

        if spread is not None or total is not None or home_ml is not None or away_ml is not None:
            # We still continue to prefer the first provider encountered; break now.
            break

    return {"spread": spread, "total": total, "home_ml": home_ml, "away_ml": away_ml, "book": book}


def _row_from_game(g: Dict[str, Any]) -> Dict[str, Any]:
    """Map a single game object to the CSV row schema."""
    gid = g.get("id") or g.get("game_id") or g.get("gameId")
    week = g.get("week")
    # Prefer API-provided season fields if present
    season = g.get("season")
    season_type = g.get("seasonType") or g.get("season_type")

    start_iso = g.get("startDate") or g.get("start_time_tbd") or g.get("start_date") or ""
    home = g.get("homeTeam") or g.get("home_team")
    away = g.get("awayTeam") or g.get("away_team")

    v = _first_book_values(g)
    return {
        "year": season,  # may be None if not present in the raw
        "season_type": season_type,
        "week": week,
        "game_id": str(gid) if gid is not None else "",
        "kickoff_utc": start_iso if isinstance(start_iso, str) else "",
        "home_team": home,
        "away_team": away,
        "spread": v["spread"],
        "total": v["total"],
        "home_ml": v["home_ml"],
        "away_ml": v["away_ml"],
        "book": v["book"],
    }


def main() -> None:
    games = _load_raw()
    rows: List[Dict[str, Any]] = []
    for g in games:
        if isinstance(g, dict):
            rows.append(_row_from_game(g))

    # Sort by kickoff then game_id; drop duplicate game_id (keep first)
    rows.sort(key=lambda r: (r.get("kickoff_utc") or "", r.get("game_id") or ""))

    seen = set()
    deduped: List[Dict[str, Any]] = []
    for r in rows:
        gid = r.get("game_id")
        if gid and gid in seen:
            continue
        if gid:
            seen.add(gid)
        deduped.append(r)

    # Write CSV
    fieldnames = [
        "year","season_type","week","game_id","kickoff_utc",
        "home_team","away_team",
        "spread","total","home_ml","away_ml","book"
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in deduped:
            w.writerow(r)

    print(f"Wrote {len(deduped)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

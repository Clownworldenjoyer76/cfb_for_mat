#!/usr/bin/env python3
"""
get_market_lines.py

Purpose:
  - Pull College Football market lines (spreads, totals, moneylines) from CFBD.
  - Prefer sportsdataverse-py if available; fallback to CFBD HTTP /lines.
  - Normalize into a single flat CSV for your baseline benchmarking.
  - ALWAYS save the raw payload to cfbd_lines.json for audit.

Env:
  CFBD_API_KEY     -> REQUIRED (GitHub Actions secret)
  CFB_YEAR         -> OPTIONAL (defaults to current UTC year)
  CFB_WEEK         -> OPTIONAL (omit to fetch all weeks returned by endpoint)
  CFB_SEASON_TYPE  -> OPTIONAL ('regular' default)
  CFB_OUTPUT_LINES -> OPTIONAL (output CSV path; default: market_lines.csv)

Output files (repo root):
  - market_lines.csv
  - cfbd_lines.json
"""

import os
import sys
import json
import datetime as dt
from typing import Any, Dict, List, Optional

import pandas as pd

BASE = "https://api.collegefootballdata.com"


# ------------------ logging ------------------
def _warn(msg: str) -> None:
    print(msg, file=sys.stderr)


# ------------------ config ------------------
API_KEY = os.getenv("CFBD_API_KEY")
if not API_KEY:
    _warn("CFBD_API_KEY is not set; writing empty artifacts.")
    pd.DataFrame(columns=[
        "year","season_type","week","game_id","kickoff_utc",
        "home_team","away_team",
        "spread","total","home_ml","away_ml","book"
    ]).to_csv(os.getenv("CFB_OUTPUT_LINES", "market_lines.csv"), index=False)
    with open("cfbd_lines.json", "w") as f:
        json.dump({"error": "missing CFBD_API_KEY"}, f, indent=2)
    sys.exit(0)

def _get_year() -> int:
    s = os.getenv("CFB_YEAR")
    if s and s.strip():
        try:
            return int(s.strip())
        except Exception:
            pass
    return dt.datetime.utcnow().year

def _get_week() -> Optional[int]:
    s = os.getenv("CFB_WEEK")
    if s and s.strip():
        try:
            return int(s.strip())
        except Exception:
            return None
    return None  # omit to allow full-season pull if desired

YEAR = _get_year()
WEEK = _get_week()
SEASON_TYPE = (os.getenv("CFB_SEASON_TYPE", "regular") or "regular").strip().lower()
OUTPUT_CSV = (os.getenv("CFB_OUTPUT_LINES", "market_lines.csv") or "market_lines.csv").strip()
OUTPUT_RAW = "cfbd_lines.json"


# ------------------ HTTP helper ------------------
def http_get(path: str, params: Dict[str, Any]) -> Any:
    import requests
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(f"{BASE}{path}", headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


# ------------------ fetchers ------------------
def fetch_with_sdv(year: int, week: Optional[int], season_type: str) -> List[Dict[str, Any]]:
    """
    Use sportsdataverse cfbd get_lines if available.
    """
    try:
        from sportsdataverse import cfbd as sdv_cfbd
    except Exception as e:
        raise RuntimeError(f"SportsDataverse not available: {e}")

    kwargs = {"year": year, "season_type": season_type, "authorization": API_KEY}
    if week is not None:
        kwargs["week"] = week

    df = sdv_cfbd.get_lines(**kwargs)
    if hasattr(df, "to_dict"):
        return df.to_dict("records")
    if isinstance(df, list):
        return df
    raise RuntimeError("Unexpected sportsdataverse return type for get_lines().")


def fetch_with_http(year: int, week: Optional[int], season_type: str) -> List[Dict[str, Any]]:
    """
    CFBD HTTP fallback: /lines (returns list of game objects, each with list of books).
    """
    params: Dict[str, Any] = {"year": year, "seasonType": season_type}
    if week is not None:
        params["week"] = week
    data = http_get("/lines", params)
    return data if isinstance(data, list) else []


# ------------------ normalization ------------------
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

def normalize_http_shape(items: List[Dict[str, Any]], year: int, season_type: str) -> List[Dict[str, Any]]:
    """
    HTTP shape is [{ id, season, seasonType, week, startDate, homeTeam, awayTeam, lines: [ { provider, spread:{spread}, moneyline:{homePrice,awayPrice}, overUnder } ... ] }, ...]
    Take the first book that has any usable data; record book name.
    """
    rows: List[Dict[str, Any]] = []

    for g in items or []:
        gid = g.get("id")
        if gid is None:
            continue
        game_week = g.get("week")
        home = g.get("homeTeam")
        away = g.get("awayTeam")
        start_iso = g.get("startDate") or g.get("start_time_tbd") or ""

        spread = total = home_ml = away_ml = None
        book = ""

        for b in g.get("lines", []) or []:
            provider = b.get("provider") or b.get("book") or ""
            sp = b.get("spread") if isinstance(b.get("spread"), dict) else None
            ml = b.get("moneyline") if isinstance(b.get("moneyline"), dict) else None
            ou = b.get("overUnder") if "overUnder" in b else b.get("total")

            # pick first usable set
            if spread is None and sp is not None and ("spread" in sp):
                spread = _num(sp.get("spread"))
                book = provider or book
            if home_ml is None and ml is not None:
                home_ml = _num(ml.get("homePrice"))
                away_ml = _num(ml.get("awayPrice"))
                book = provider or book
            if total is None and ou is not None:
                total = _num(ou)
                book = provider or book

            if spread is not None or total is not None or home_ml is not None or away_ml is not None:
                # good enough; we recorded a book already if present
                pass

        rows.append({
            "year": year,
            "season_type": season_type,
            "week": game_week,
            "game_id": str(gid),
            "kickoff_utc": start_iso if isinstance(start_iso, str) else "",
            "home_team": home,
            "away_team": away,
            "spread": spread,
            "total": total,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "book": book,
        })

    return rows


def normalize_sdv_shape(items: List[Dict[str, Any]], year: int, season_type: str) -> List[Dict[str, Any]]:
    """
    sportsdataverse get_lines() often returns a row per book per game.
    Attempt to coalesce similarly (first usable book).
    Expected keys vary: id/game_id, home_team, away_team, start_date, lines: [...]
    """
    # Some SDV versions flatten already; if so, pass to HTTP normalizer by re-wrapping.
    # Detect by presence of "lines" list; if not present, synthesize a similar shape.
    shaped: List[Dict[str, Any]] = []
    has_lines_list = any(isinstance(obj.get("lines"), list) for obj in items if isinstance(obj, dict))

    if has_lines_list:
        return normalize_http_shape(items, year, season_type)

    # Flattened SDV: we may have moneyline_home/away, spread (home spread), total, provider
    # Group by game id and take first usable.
    by_gid: Dict[str, Dict[str, Any]] = {}
    for obj in items or []:
        gid = obj.get("id") or obj.get("game_id")
        if gid is None:
            continue
        gid = str(gid)
        rec = by_gid.get(gid, {
            "year": year,
            "season_type": season_type,
            "week": obj.get("week"),
            "game_id": gid,
            "kickoff_utc": obj.get("start_date") or obj.get("startDate") or "",
            "home_team": obj.get("home_team") or obj.get("homeTeam"),
            "away_team": obj.get("away_team") or obj.get("awayTeam"),
            "spread": None,
            "total": None,
            "home_ml": None,
            "away_ml": None,
            "book": obj.get("provider") or obj.get("book") or "",
        })

        # possible keys
        if rec["spread"] is None:
            rec["spread"] = _num(obj.get("spread") or obj.get("home_spread"))
        if rec["total"] is None:
            rec["total"] = _num(obj.get("total") or obj.get("over_under") or obj.get("overUnder"))
        if rec["home_ml"] is None:
            rec["home_ml"] = _num(obj.get("home_price") or obj.get("home_ml") or obj.get("moneyline_home"))
        if rec["away_ml"] is None:
            rec["away_ml"] = _num(obj.get("away_price") or obj.get("away_ml") or obj.get("moneyline_away"))

        if not rec["book"]:
            rec["book"] = obj.get("provider") or obj.get("book") or rec["book"]

        by_gid[gid] = rec

    return list(by_gid.values())


# ------------------ main ------------------
def main() -> None:
    using = "sdv"
    try:
        items = fetch_with_sdv(YEAR, WEEK, SEASON_TYPE)
        normalized = normalize_sdv_shape(items, YEAR, SEASON_TYPE)
    except Exception as e:
        _warn(f"[warn] sportsdataverse failed or unavailable; using HTTP fallback: {e}")
        items = fetch_with_http(YEAR, WEEK, SEASON_TYPE)
        normalized = normalize_http_shape(items, YEAR, SEASON_TYPE)
        using = "http"

    # write raw audit
    try:
        with open(OUTPUT_RAW, "w") as f:
            json.dump(
                {"source": using, "year": YEAR, "season_type": SEASON_TYPE, "week": WEEK, "count": len(items), "data": items},
                f,
                indent=2
            )
    except Exception as e:
        _warn(f"[warn] failed to write {OUTPUT_RAW}: {e}")

    # to CSV
    df = pd.DataFrame(normalized, columns=[
        "year","season_type","week","game_id","kickoff_utc",
        "home_team","away_team",
        "spread","total","home_ml","away_ml","book"
    ])
    # remove dupes by game_id, keep first (book choice policy = first usable)
    if not df.empty:
        df = df.sort_values(["kickoff_utc","game_id"], na_position="last").drop_duplicates(subset=["game_id"]).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}; raw saved to {OUTPUT_RAW} (source={using})")


if __name__ == "__main__":
    main()

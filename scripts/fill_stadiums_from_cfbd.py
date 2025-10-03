#!/usr/bin/env python3
"""
Fill missing lat/lon/timezone/altitude_m in `data/reference/stadiums.csv`
using the CollegeFootballData (CFBD) Venues API.

- Preserves existing non-null values.
- Tries exact & normalized venue-name matches first; falls back to city/state;
  final fallback uses fuzzy matching (rapidfuzz).
- Converts CFBD elevation (feet) -> meters when present.

Usage:
  CFBD_API_KEY=xxxxxxxx python scripts/fill_stadiums_from_cfbd.py \
    --stadiums data/reference/stadiums.csv

Optional:
  --min-fuzzy 92   # minimum ratio for fuzzy match (default 90)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import unicodedata
import re
import math
import pandas as pd
import requests
from rapidfuzz import fuzz, process

VENUES_URL = "https://api.collegefootballdata.com/venues"

REQ_COLS = ["team","venue","city","state","country","lat","lon","timezone","altitude_m","is_neutral_site","notes"]

def norm(s: str) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^\w\s-]", " ", s)   # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("–", "-").replace("—", "-")
    return s

def feet_to_meters(x):
    if x is None:
        return None
    try:
        val = float(x)
    except:
        return None
    # CFBD `elevation` is in feet
    return round(val * 0.3048, 1)

def pull_cfbd(api_key: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.get(VENUES_URL, headers=headers, timeout=60)
    r.raise_for_status()
    js = r.json()
    # Common fields in CFBD venues
    # name, city, state, latitude, longitude, elevation, timezone, country, id
    df = pd.DataFrame(js)
    # Normalize columns we care about
    for col in ["name","city","state","country","timezone"]:
        if col not in df.columns:
            df[col] = None
        df[col+"_n"] = df[col].map(norm)
    for col in ["latitude","longitude","elevation"]:
        if col not in df.columns:
            df[col] = None
    return df

def choose_best(row, candidates, min_fuzzy):
    """Return index of best fuzzy match on venue name against candidates, or None."""
    if not candidates:
        return None
    target = norm(row.get("venue"))
    if not target:
        return None
    names = {i: candidates.loc[i, "name_n"] for i in candidates.index}
    # rapidfuzz process
    best = process.extractOne(target, names, scorer=fuzz.WRatio)
    if not best:
        return None
    label, score, idx = best
    return idx if score >= min_fuzzy else None

def fill_row(row, venue):
    """Fill only missing fields in row from venue record."""
    def maybe_set(key_row, key_venue, transform=None):
        if pd.isna(row.get(key_row)) or row.get(key_row) == "" or (isinstance(row.get(key_row), float) and math.isnan(row.get(key_row))):
            val = venue.get(key_venue)
            if transform:
                val = transform(val)
            row[key_row] = val
    maybe_set("lat", "latitude")
    maybe_set("lon", "longitude")
    maybe_set("timezone", "timezone")
    maybe_set("altitude_m", "elevation", feet_to_meters)
    # Try to backfill city/state if missing
    maybe_set("city", "city")
    maybe_set("state", "state")
    # country if present
    maybe_set("country", "country")
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stadiums", type=Path, required=True, help="Path to data/reference/stadiums.csv")
    ap.add_argument("--min-fuzzy", type=int, default=90, help="Minimum score for fuzzy venue-name match")
    args = ap.parse_args()

    api_key = os.getenv("CFBD_API_KEY", "")
    if not api_key:
        print("[fill] WARNING: CFBD_API_KEY not set; request may be rate-limited or rejected.", file=sys.stderr)

    stad = pd.read_csv(args.stadiums)
    # Ensure required columns exist
    for c in REQ_COLS:
        if c not in stad.columns:
            stad[c] = pd.NA

    cfbd = pull_cfbd(api_key)

    # Pre-split complete vs needs-fill
    needs = stad[
        stad[["lat","lon","timezone","altitude_m"]].isna().any(axis=1)
    ].copy()

    if needs.empty:
        print("[fill] Nothing to fill; all good.")
        return

    # Exact venue name (normalized) join
    stad["venue_n"] = stad["venue"].map(norm)
    cfbd["name_n"] = cfbd["name_n"].fillna("")
    exact = needs.merge(cfbd, left_on="venue_n", right_on="name_n", how="left", suffixes=("", "_cfbd"))

    # For those still unmatched, try city+state
    still = exact[exact["latitude"].isna()].copy()
    if not still.empty:
        still["city_n"] = still["city"].map(norm)
        still["state_n"] = still["state"].map(norm)
        cfbd["city_n"] = cfbd["city_n"].fillna("")
        cfbd["state_n"] = cfbd["state_n"].fillna("")
        by_loc = still.merge(
            cfbd,
            on=["city_n","state_n"],
            how="left",
            suffixes=("", "_cfbd2"),
        )
        # fill blanks from by_loc
        mask = exact["latitude"].isna()
        exact.loc[mask, ["latitude","longitude","elevation","timezone","city","state","country"]] = \
            by_loc[["latitude","longitude","elevation","timezone","city","state","country"]].values

    # Fuzzy fallback for any still missing on latitude
    remain_idx = exact[exact["latitude"].isna()].index
    if len(remain_idx) > 0:
        for i in remain_idx:
            row = exact.loc[i]
            idx = choose_best(row, cfbd, args.min_fuzzy)
            if idx is not None:
                v = cfbd.loc[idx]
                for k in ["latitude","longitude","elevation","timezone","city","state","country"]:
                    exact.at[i, k] = v.get(k)

    # Write values back into stad, filling only missing entries
    updated = stad.copy()
    exact = exact.set_index("team")
    updated = updated.set_index("team")
    common = updated.index.intersection(exact.index)
    for t in common:
        row = updated.loc[t].to_dict()
        venue = exact.loc[t].to_dict()
        row = fill_row(row, venue)
        updated.loc[t] = pd.Series(row)
    updated = updated.reset_index()

    # Normalize types
    for c in ["lat","lon","altitude_m"]:
        updated[c] = pd.to_numeric(updated[c], errors="coerce")
    if "is_neutral_site" in updated.columns:
        updated["is_neutral_site"] = pd.to_numeric(updated["is_neutral_site"], errors="coerce").fillna(0).astype(int)

    # Save
    updated.to_csv(args.stadiums, index=False)
    # Report any still missing
    still_missing = updated[updated[["lat","lon","timezone","altitude_m"]].isna().any(axis=1)][["team","venue","city","state","lat","lon","timezone","altitude_m"]]
    if len(still_missing) > 0:
        print("[fill] WARNING: Some rows still incomplete:")
        print(still_missing.to_string(index=False))
    else:
        print("[fill] All stadium rows now have lat/lon/timezone/altitude_m.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[fill] ERROR: {e}", file=sys.stderr)
        raise

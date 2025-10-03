#!/usr/bin/env python3
"""
Fill missing stadium metadata (lat, lon, timezone, altitude_m) in data/reference/stadiums.csv
using the CollegeFootballData (CFBD) API.

Requirements:
- Python: requests, pandas
- Env var: CFBD_API_KEY  (Settings > Secrets and variables > Actions > New repository secret)

Behavior:
- Reads data/reference/stadiums.csv
- Downloads CFBD venues catalog
- Matches rows by venue name (primary) and by (city,state) (fallback)
- Fills missing: lat, lon, timezone, altitude_m
- Leaves existing values intact
- Writes updates back to data/reference/stadiums.csv

Usage (local or CI):
  CFBD_API_KEY=xxxx python scripts/fill_stadiums_from_cfbd.py
"""

import os
import sys
import time
import math
import json
import re
from typing import Dict, Any, Tuple, Optional

import requests
import pandas as pd

CFBD_API_KEY = os.environ.get("CFBD_API_KEY")
VENUES_URL = "https://api.collegefootballdata.com/venues"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STADIUMS_PATH = os.path.join(REPO_ROOT, "data", "reference", "stadiums.csv")

# -----------------------------------------
# Helpers
# -----------------------------------------
def norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[\u2010-\u2015]", "-", s)  # normalize dashes
    s = re.sub(r"[^a-z0-9\s\-&/]", "", s)   # drop punctuation other than a few safe chars
    s = re.sub(r"\s+", " ", s)
    return s

def feet_to_meters(x):
    try:
        if pd.isna(x):
            return None
        return float(x) * 0.3048
    except Exception:
        return None

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def http_get(url: str, headers: Dict[str, str], params: Dict[str, Any] = None, retries: int = 3, backoff: float = 1.5):
    last = None
    for i in range(retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            last = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except Exception as e:
            last = str(e)
        time.sleep(backoff * (i + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts; last={last}")

def load_cfbd_venues() -> pd.DataFrame:
    if not CFBD_API_KEY:
        raise RuntimeError("CFBD_API_KEY not set. Add repository secret CFBD_API_KEY.")
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    data = http_get(VENUES_URL, headers=headers, params={})
    # Normalize into dataframe
    rows = []
    for v in data:
        rows.append({
            "venue_name": v.get("name"),
            "city": v.get("city"),
            "state": v.get("state"),
            "country": v.get("country"),
            "lat": v.get("latitude"),
            "lon": v.get("longitude"),
            "timezone": v.get("timezone"),
            # CFBD returns elevation (feet) for many venues; key may be 'elevation'
            "elevation_ft": v.get("elevation"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("CFBD venues response empty.")
    # Normalized keys for matching
    df["__k_venue"] = df["venue_name"].map(norm)
    df["__k_city_state"] = (df["city"].fillna("").astype(str) + "," + df["state"].fillna("").astype(str)).map(norm)
    # Convert elevation to meters
    df["altitude_m"] = df["elevation_ft"].map(feet_to_meters)
    # Ensure types
    df["lat"] = df["lat"].map(safe_float)
    df["lon"] = df["lon"].map(safe_float)
    return df

def ensure_stadiums_schema(df: pd.DataFrame) -> pd.DataFrame:
    need = ["team","venue","city","state","country","lat","lon","timezone","altitude_m","is_neutral_site","notes"]
    for c in need:
        if c not in df.columns:
            df[c] = pd.NA
    return df[need]

def best_match(row: pd.Series, venues: pd.DataFrame) -> Optional[pd.Series]:
    """
    Match strategy:
      1) Exact normalized venue name
      2) If city+state present, exact normalized pair
      3) If multiple matches, prefer row with non-null lat/lon/timezone
    """
    k_venue = norm(row.get("venue"))
    k_city_state = norm(f"{row.get('city','')},{row.get('state','')}")
    cand = pd.DataFrame()
    if k_venue:
        cand = venues[venues["__k_venue"] == k_venue]
    if cand.empty and (row.get("city") or row.get("state")):
        cand = venues[venues["__k_city_state"] == k_city_state]

    if cand.empty:
        return None
    # prefer rows with lat/lon/tz present
    cand = cand.copy()
    cand["score"] = (
        cand["lat"].notna().astype(int) +
        cand["lon"].notna().astype(int) +
        cand["timezone"].notna().astype(int) +
        cand["altitude_m"].notna().astype(int)
    )
    cand = cand.sort_values(["score"], ascending=False)
    return cand.iloc[0]

def fill_row(row: pd.Series, m: pd.Series) -> pd.Series:
    # Only fill if missing
    if pd.isna(row.get("lat")) or row.get("lat") == "":
        row["lat"] = m.get("lat")
    if pd.isna(row.get("lon")) or row.get("lon") == "":
        row["lon"] = m.get("lon")
    if pd.isna(row.get("timezone")) or row.get("timezone") == "":
        row["timezone"] = m.get("timezone")
    if pd.isna(row.get("altitude_m")) or row.get("altitude_m") == "":
        row["altitude_m"] = m.get("altitude_m")
    # Ensure country defaults if absent
    if pd.isna(row.get("country")) or str(row.get("country")).strip() == "":
        row["country"] = "USA"
    return row

# -----------------------------------------
# Main
# -----------------------------------------
def main():
    # Load stadiums.csv
    if not os.path.exists(STADIUMS_PATH):
        raise FileNotFoundError(f"Not found: {STADIUMS_PATH}")
    sdf = pd.read_csv(STADIUMS_PATH)
    sdf = ensure_stadiums_schema(sdf)

    # Prepare keys
    sdf["__needs_fill"] = (
        sdf["lat"].isna() | (sdf["lat"] == "") |
        sdf["lon"].isna() | (sdf["lon"] == "") |
        sdf["timezone"].isna() | (sdf["timezone"] == "") |
        sdf["altitude_m"].isna() | (sdf["altitude_m"] == "")
    )

    # Load venues index once
    venues = load_cfbd_venues()

    # Fill rows needing data
    filled = 0
    for idx, row in sdf.iterrows():
        if not bool(row["__needs_fill"]):
            continue
        match = best_match(row, venues)
        if match is None:
            continue
        new_row = fill_row(row.copy(), match)
        # Count improvements
        improved = int(pd.isna(row.get("lat")) or row.get("lat") == "") + \
                   int(pd.isna(row.get("lon")) or row.get("lon") == "") + \
                   int(pd.isna(row.get("timezone")) or row.get("timezone") == "") + \
                   int(pd.isna(row.get("altitude_m")) or row.get("altitude_m") == "")
        if improved > 0:
            filled += 1
        # Write back
        for c in ["lat","lon","timezone","altitude_m","country"]:
            sdf.at[idx, c] = new_row.get(c)

    # Cleanup types
    sdf["lat"] = pd.to_numeric(sdf["lat"], errors="coerce")
    sdf["lon"] = pd.to_numeric(sdf["lon"], errors="coerce")
    sdf["altitude_m"] = pd.to_numeric(sdf["altitude_m"], errors="coerce")

    # Drop helper
    sdf = sdf.drop(columns=["__needs_fill"], errors="ignore")

    # Save in place
    sdf.to_csv(STADIUMS_PATH, index=False)

    # Minimal stdout (useful in CI logs)
    print(json.dumps({
        "updated_rows": int(filled),
        "total_rows": int(len(sdf)),
        "output": STADIUMS_PATH
    }))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail with clear message for CI
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

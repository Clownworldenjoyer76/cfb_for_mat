#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Fill missing venue/lat/lon (and city/state if blank) in data/reference/stadiums.csv
# using data/reference/stadiums-geocoded.csv from GitHub.
#
# Robust to column name clashes: we pre-rename geocode columns to *_geo to avoid city_x/city_y bugs.

import argparse
from pathlib import Path
import sys
import re
import unicodedata
import pandas as pd


def norm_team(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def main():
    ap = argparse.ArgumentParser(description="Backfill venue/lat/lon from geocoded CSV (GitHub).")
    ap.add_argument("--stadiums", type=Path, default=Path("data/reference/stadiums.csv"),
                    help="Path to main stadiums CSV (default: data/reference/stadiums.csv)")
    ap.add_argument("--geocodes", type=Path, default=Path("data/reference/stadiums-geocoded.csv"),
                    help="Path to GitHub geocoded CSV (default: data/reference/stadiums-geocoded.csv)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional output path. If omitted, overwrites --stadiums in place.")
    args = ap.parse_args()

    stad_path = args.stadiums
    geo_path = args.geocodes
    out_path = args.out or stad_path

    if not stad_path.exists():
        print(f"[geofill] ERROR: stadiums file not found: {stad_path}", file=sys.stderr)
        sys.exit(1)
    if not geo_path.exists():
        print(f"[geofill] ERROR: geocoded file not found: {geo_path}", file=sys.stderr)
        sys.exit(1)

    # Load data
    df = pd.read_csv(stad_path)
    geo = pd.read_csv(geo_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    geo.columns = geo.columns.str.strip().str.lower()

    # Ensure required columns exist in main file
    for c in ["team", "venue", "city", "state", "lat", "lon"]:
        if c not in df.columns:
            df[c] = pd.NA

    # Prepare geocoded dataset columns -> rename to *_geo to avoid _x/_y suffix issues
    # Expected geocode columns present per your file: stadium, city, state, team, latitude, longitude
    # If any are missing, create as NA
    for c in ["team", "stadium", "city", "state", "latitude", "longitude"]:
        if c not in geo.columns:
            geo[c] = pd.NA

    geo["_team_n"] = geo["team"].astype(str).map(norm_team)
    df["_team_n"] = df["team"].astype(str).map(norm_team)

    # Prefer rows with coordinates when deduplicating geocode file
    geo["lat_num"] = pd.to_numeric(geo["latitude"], errors="coerce")
    geo["lon_num"] = pd.to_numeric(geo["longitude"], errors="coerce")
    geo["has_coords"] = geo["lat_num"].notna() & geo["lon_num"].notna()
    geo_sorted = geo.sort_values(by=["_team_n", "has_coords"], ascending=[True, False])
    geo_dedup = geo_sorted.drop_duplicates(subset=["_team_n"], keep="first").copy()

    geo_keep = geo_dedup[["_team_n", "stadium", "city", "state", "lat_num", "lon_num"]].rename(
        columns={
            "stadium": "venue_geo",
            "city": "city_geo",
            "state": "state_geo",
            "lat_num": "lat_geo",
            "lon_num": "lon_geo",
        }
    )

    # Left-merge onto main using normalized team key
    merged = df.merge(geo_keep, how="left", on="_team_n", copy=False)

    # Convert types
    merged["lat"] = pd.to_numeric(merged["lat"], errors="coerce")
    merged["lon"] = pd.to_numeric(merged["lon"], errors="coerce")

    fills = 0

    # Venue: fill only if blank/NaN
    need_venue = merged["venue"].isna() | (merged["venue"].astype(str).str.strip() == "")
    to_fill = need_venue & merged["venue_geo"].notna()
    merged.loc[to_fill, "venue"] = merged.loc[to_fill, "venue_geo"]
    fills += int(to_fill.sum())

    # City
    need_city = merged["city"].isna() | (merged["city"].astype(str).str.strip() == "")
    to_fill = need_city & merged["city_geo"].notna()
    merged.loc[to_fill, "city"] = merged.loc[to_fill, "city_geo"]
    fills += int(to_fill.sum())

    # State
    need_state = merged["state"].isna() | (merged["state"].astype(str).str.strip() == "")
    to_fill = need_state & merged["state_geo"].notna()
    merged.loc[to_fill, "state"] = merged.loc[to_fill, "state_geo"]
    fills += int(to_fill.sum())

    # lat / lon: fill only if missing
    need_lat = merged["lat"].isna()
    to_fill = need_lat & merged["lat_geo"].notna()
    merged.loc[to_fill, "lat"] = merged.loc[to_fill, "lat_geo"]
    fills += int(to_fill.sum())

    need_lon = merged["lon"].isna()
    to_fill = need_lon & merged["lon_geo"].notna()
    merged.loc[to_fill, "lon"] = merged.loc[to_fill, "lon_geo"]
    fills += int(to_fill.sum())

    # Drop helper/geo columns
    merged = merged.drop(columns=[c for c in ["_team_n", "venue_geo", "city_geo", "state_geo", "lat_geo", "lon_geo"] if c in merged.columns])

    # Write out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    # Minimal summary
    remaining = {
        "venue": int((merged["venue"].astype(str).str.strip() == "").sum() + merged["venue"].isna().sum()),
        "lat": int(merged["lat"].isna().sum()),
        "lon": int(merged["lon"].isna().sum()),
    }
    print(f"[geofill] Done. Fields filled: {fills}. Remaining blanks -> {remaining}. Wrote: {out_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Fill missing venue/lat/lon in data/reference/stadiums.csv from
# data/reference/stadiums-geocoded.csv (GitHub dataset).
#
# Matching is by team (case-insensitive, normalized). Only fills blanks.
# Does NOT overwrite non-empty values.

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


def coerce_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Backfill venue/lat/lon from geocoded CSV.")
    ap.add_argument("--stadiums", type=Path, default=Path("data/reference/stadiums.csv"),
                    help="Path to your main stadiums CSV (default: data/reference/stadiums.csv)")
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

    # Map geocoded fields to our schema
    # Expected geocoded columns: team, stadium, city, state, latitude, longitude (plus extras)
    if "stadium" not in geo.columns:
        # Some forks may use 'venue' already—fallback
        if "venue" in geo.columns:
            geo["stadium"] = geo["venue"]
        else:
            geo["stadium"] = pd.NA
    if "latitude" not in geo.columns:
        geo["latitude"] = pd.NA
    if "longitude" not in geo.columns:
        geo["longitude"] = pd.NA

    # Build normalized keys for joining
    df["_team_n"] = df["team"].astype(str).map(norm_team)
    geo["_team_n"] = geo["team"].astype(str).map(norm_team) if "team" in geo.columns else ""

    # Drop rows in geocodes without a team key
    geo = geo[geo["_team_n"] != ""].copy()

    # Deduplicate geocodes on team key: prefer rows with lat/lon present
    geo["lat_num"] = pd.to_numeric(geo["latitude"], errors="coerce")
    geo["lon_num"] = pd.to_numeric(geo["longitude"], errors="coerce")
    geo["has_coords"] = geo["lat_num"].notna() & geo["lon_num"].notna()
    # Keep first with coords if available; otherwise first
    geo_sorted = geo.sort_values(by=["_team_n", "has_coords"], ascending=[True, False])
    geo_dedup = geo_sorted.drop_duplicates(subset=["_team_n"], keep="first").copy()

    # Keep only columns we need for fill
    geo_keep = geo_dedup[["_team_n", "stadium", "city", "state", "lat_num", "lon_num"]].rename(
        columns={"stadium": "venue_geo", "lat_num": "lat_geo", "lon_num": "lon_geo"}
    )

    # Merge onto main by normalized team
    merged = df.merge(geo_keep, how="left", left_on="_team_n", right_on="_team_n", copy=False)

    # Fill blanks only
    fills = 0

    # Venue
    need_venue = merged["venue"].isna() | (merged["venue"].astype(str).str.strip() == "")
    merged.loc[need_venue & merged["venue_geo"].notna(), "venue"] = merged.loc[need_venue, "venue_geo"]
    fills += int((need_venue & merged["venue_geo"].notna()).sum())

    # City
    need_city = merged["city"].isna() | (merged["city"].astype(str).str.strip() == "")
    merged.loc[need_city & merged["city_y"].notna(), "city"] = merged.loc[need_city, "city_y"]
    fills += int((need_city & merged["city_y"].notna()).sum())

    # State
    need_state = merged["state"].isna() | (merged["state"].astype(str).str.strip() == "")
    merged.loc[need_state & merged["state_y"].notna(), "state"] = merged.loc[need_state, "state_y"]
    fills += int((need_state & merged["state_y"].notna()).sum())

    # lat / lon (only if blank)
    merged["lat"] = pd.to_numeric(merged["lat"], errors="coerce")
    merged["lon"] = pd.to_numeric(merged["lon"], errors="coerce")
    need_lat = merged["lat"].isna()
    need_lon = merged["lon"].isna()

    merged.loc[need_lat & merged["lat_geo"].notna(), "lat"] = merged.loc[need_lat, "lat_geo"]
    merged.loc[need_lon & merged["lon_geo"].notna(), "lon"] = merged.loc[need_lon, "lon_geo"]
    fills += int((need_lat & merged["lat_geo"].notna()).sum())
    fills += int((need_lon & merged["lon_geo"].notna()).sum())

    # Clean columns (drop merge artifacts)
    drop_cols = [c for c in ["venue_geo", "city_y", "state_y", "lat_geo", "lon_geo"] if c in merged.columns]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    # Rename potential duplicates back to canonical if needed
    if "city_x" in merged.columns:
        merged = merged.rename(columns={"city_x": "city"})
    if "state_x" in merged.columns:
        merged = merged.rename(columns={"state_x": "state"})

    # Drop helper
    merged = merged.drop(columns=["_team_n"], errors="ignore")

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    # Minimal stdout summary
    remaining_lat = merged["lat"].isna().sum()
    remaining_lon = merged["lon"].isna().sum()
    remaining_venue = (merged["venue"].astype(str).str.strip() == "").sum() + merged["venue"].isna().sum()
    print(f"[geofill] Done. Filled fields: {fills}. Remaining blanks -> venue:{remaining_venue} lat:{remaining_lat} lon:{remaining_lon}. Wrote: {out_path}")

if __name__ == "__main__":
    main()

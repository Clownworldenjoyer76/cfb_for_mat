#!/usr/bin/env python3
# Fill empty timezone cells using IANA tz from lat/lon in a stadiums CSV.

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from timezonefinder import TimezoneFinder

def is_missing_tz(x):
    if x is None:
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "none" or pd.isna(x)

def coerce_float(s):
    try:
        return float(s)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Backfill timezone from lat/lon with IANA strings.")
    ap.add_argument("--stadiums", type=Path, required=True, help="Path to data/reference/stadiums.csv")
    ap.add_argument("--out", type=Path, default=None, help="Optional output path (defaults to overwrite input)")
    args = ap.parse_args()

    src = args.stadiums
    dst = args.out or src

    if not src.exists():
        print("[tz] ERROR: file not found: %s" % src, file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(src)

    # Ensure required columns exist
    for c in ["lat", "lon", "timezone"]:
        if c not in df.columns:
            df[c] = pd.NA

    # Normalize types
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Rows eligible for backfill: tz missing and lat/lon present
    mask = df.apply(lambda r: is_missing_tz(r.get("timezone")) and pd.notna(r.get("lat")) and pd.notna(r.get("lon")), axis=1)

    if mask.any():
        tf = TimezoneFinder()
        for i in df.index[mask]:
            lat = coerce_float(df.at[i, "lat"])
            lon = coerce_float(df.at[i, "lon"])
            if lat is None or lon is None:
                continue
            tz = None
            try:
                tz = tf.timezone_at(lat=lat, lng=lon)
                if tz is None:
                    tz = tf.closest_timezone_at(lat=lat, lng=lon)
            except Exception:
                tz = None
            if tz and isinstance(tz, str) and tz.strip():
                df.at[i, "timezone"] = tz

    # Write file
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)

    # Basic output
    remaining = df["timezone"].isna().sum() + sum(str(x).strip().lower() == "none" for x in df["timezone"].fillna(""))
    print("[tz] Done. Rows updated: %d, remaining blank/None timezones: %d" % (mask.sum(), remaining))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Recompute timezone for EVERY row using IANA tz from lat/lon in stadiums.csv (row-by-row).

import argparse
import sys
from pathlib import Path

import pandas as pd
from timezonefinder import TimezoneFinder

def coerce_float(x):
    try:
        return float(x)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Recompute timezone for every row from lat/lon (IANA strings).")
    ap.add_argument("--stadiums", type=Path, required=True, help="Path to data/reference/stadiums.csv")
    ap.add_argument("--out", type=Path, default=None, help="Optional output path (defaults to overwrite input)")
    args = ap.parse_args()

    src = args.stadiums
    dst = args.out or src

    if not src.exists():
        print("[tz] ERROR: file not found: %s" % src, file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(src)

    # Ensure columns exist
    for c in ["lat", "lon", "timezone"]:
        if c not in df.columns:
            df[c] = pd.NA

    # Normalize numeric types
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    tf = TimezoneFinder()

    updated = 0
    for i in df.index:
        lat = coerce_float(df.at[i, "lat"])
        lon = coerce_float(df.at[i, "lon"])
        if lat is None or lon is None:
            # leave timezone as-is if coordinates are missing/invalid
            continue

        tz = None
        try:
            tz = tf.timezone_at(lat=lat, lng=lon)
            if tz is None:
                tz = tf.closest_timezone_at(lat=lat, lng=lon)
        except Exception:
            tz = None

        if tz and isinstance(tz, str) and tz.strip():
            if df.at[i, "timezone"] != tz:
                df.at[i, "timezone"] = tz
                updated += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print("[tz] Done. Rows recalculated: %d" % updated)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Remove any rows from data/reference/stadiums.csv that lack lat or lon.
# Keeps a summary log of how many rows were dropped.

import pandas as pd
from pathlib import Path

def main():
    in_path = Path("data/reference/stadiums.csv")
    out_path = in_path
    log_path = Path("summaries/clean_stadiums_summary.txt")

    if not in_path.exists():
        print(f"[clean] ERROR: file not found: {in_path}")
        return

    df = pd.read_csv(in_path)
    df.columns = df.columns.str.lower().str.strip()

    # Identify missing coordinates
    lat_missing = pd.to_numeric(df["lat"], errors="coerce").isna()
    lon_missing = pd.to_numeric(df["lon"], errors="coerce").isna()
    mask_drop = lat_missing | lon_missing
    n_drop = int(mask_drop.sum())

    # Keep only rows with both coords present
    cleaned = df.loc[~mask_drop].copy()
    cleaned.to_csv(out_path, index=False)

    # Write summary
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"Total rows before: {len(df)}\n")
        f.write(f"Rows removed (missing lat/lon): {n_drop}\n")
        f.write(f"Rows remaining: {len(cleaned)}\n")
    print(f"[clean] Removed {n_drop} rows missing coordinates. Wrote cleaned file and summary log.")

if __name__ == "__main__":
    main()

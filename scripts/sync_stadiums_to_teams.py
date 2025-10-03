#!/usr/bin/env python3
"""
Sync stadiums.csv rows to exactly match team names used in modeling_dataset.csv.

- Preserves ALL existing rows/values in data/reference/stadiums.csv.
- Adds rows for ANY team present in data/modeling_dataset.csv that is missing in stadiums.csv.
- Does NOT delete or change existing rows; only appends missing teams with blank venue metadata placeholders.
- Writes the result back to data/reference/stadiums.csv (sorted by team).

Required files in repo:
- data/modeling_dataset.csv  (must have a 'team' column)
- data/reference/stadiums.csv

Columns guaranteed in output:
  team,venue,city,state,country,lat,lon,timezone,altitude_m,is_neutral_site,notes
"""

import os
import sys
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "data", "modeling_dataset.csv")
STAD_PATH  = os.path.join(REPO_ROOT, "data", "reference", "stadiums.csv")

REQUIRED_COLS = ["team","venue","city","state","country","lat","lon","timezone","altitude_m","is_neutral_site","notes"]

def norm(s): 
    return str(s).strip()

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[REQUIRED_COLS]

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Missing {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(STAD_PATH):
        print(f"ERROR: Missing {STAD_PATH}", file=sys.stderr)
        sys.exit(1)

    mdf = pd.read_csv(MODEL_PATH, usecols=["team"])
    teams = (
        mdf["team"]
        .dropna()
        .astype(str)
        .map(norm)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    sdf = pd.read_csv(STAD_PATH)
    sdf = ensure_schema(sdf)
    sdf["__k"] = sdf["team"].astype(str).map(norm)

    present = set(sdf["__k"].tolist())

    # Append missing teams with placeholders
    new_rows = []
    for t in teams:
        if t not in present:
            new_rows.append({
                "team": t,
                "venue": "",
                "city": "",
                "state": "",
                "country": "USA",
                "lat": "",
                "lon": "",
                "timezone": "",
                "altitude_m": "",
                "is_neutral_site": 0,
                "notes": ""
            })

    if new_rows:
        add_df = pd.DataFrame(new_rows)[REQUIRED_COLS]
        sdf = pd.concat([sdf.drop(columns=["__k"], errors="ignore"), add_df], ignore_index=True)
    else:
        sdf = sdf.drop(columns=["__k"], errors="ignore")

    # Finalize & write
    sdf = sdf[REQUIRED_COLS].sort_values("team").reset_index(drop=True)
    os.makedirs(os.path.dirname(STAD_PATH), exist_ok=True)
    sdf.to_csv(STAD_PATH, index=False)

    print(f"Synced stadiums: wrote {len(sdf)} rows to {STAD_PATH}")
    print(f"New rows added: {len(new_rows)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Minimal sync for data/reference/stadiums.csv:
- Ensures required columns exist.
- Deduplicates by team (keeps first values).
- Optionally adds teams from --teams or --schedules.
- If neither is given, uses existing teams in stadiums.csv.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

REQUIRED_COLS = [
    "team", "venue", "city", "state", "country",
    "lat", "lon", "timezone", "altitude_m",
    "is_neutral_site", "notes",
]

DEFAULTS = {"country": "USA", "is_neutral_site": 0}

def canonicalize(series):
    s = series.dropna().astype(str).str.strip()
    s = s.replace("", pd.NA).dropna()
    return s.drop_duplicates().sort_values(key=lambda x: x.str.lower())

def find_team_col(df):
    for c in ["team","Team","school","School","name","Name"]:
        if c in df.columns:
            return c
    return None

def read_team_list(teams_csv, schedules_csv, stad_df):
    if teams_csv:
        df = pd.read_csv(teams_csv)
        col = find_team_col(df)
        if not col:
            raise SystemExit("[sync] No team column in %s" % teams_csv)
        return canonicalize(df[col])
    if schedules_csv:
        df = pd.read_csv(schedules_csv)
        cols = [c for c in ["team","home_team","away_team","Home Team","Away Team","Home","Away"] if c in df.columns]
        if not cols:
            raise SystemExit("[sync] No team columns in %s" % schedules_csv)
        return canonicalize(pd.concat([df[c] for c in cols], ignore_index=True))
    if "team" not in stad_df.columns:
        raise SystemExit("[sync] 'team' missing in stadiums.csv and no inputs provided")
    return canonicalize(stad_df["team"])

def ensure_columns(df):
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    extras = [c for c in df.columns if c not in REQUIRED_COLS]
    df = df[REQUIRED_COLS + extras]
    for k,v in DEFAULTS.items():
        if k in df.columns:
            df[k] = df[k].fillna(v)
    return df

def add_missing_rows(df, teams):
    have = df["team"].astype(str).str.strip().fillna("")
    missing = sorted(set(teams) - set(have))
    if not missing:
        return df, []
    new_rows = []
    for t in missing:
        row = {c: pd.NA for c in df.columns}
        row.update(DEFAULTS)
        row["team"] = t
        new_rows.append(row)
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True), missing

def normalize(df):
    for c in ["lat","lon","altitude_m"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "is_neutral_site" in df.columns:
        df["is_neutral_site"] = pd.to_numeric(df["is_neutral_site"], errors="coerce").fillna(0).astype(int)
    return df

def dedupe(df):
    df["_k"] = df["team"].astype(str).str.lower()
    df.sort_values("_k", kind="mergesort", inplace=True)
    df.drop(columns="_k", inplace=True)
    df = df.groupby("team", as_index=False).first()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stadiums", type=Path, required=True)
    ap.add_argument("--teams", type=Path)
    ap.add_argument("--schedules", type=Path)
    args = ap.parse_args()

    if not args.stadiums.exists():
        raise SystemExit("[sync] Stadiums file not found: %s" % args.stadiums)

    df = pd.read_csv(args.stadiums)
    df = ensure_columns(df)

    team_list = read_team_list(args.teams, args.schedules, df)

    before = len(df)
    df, added = add_missing_rows(df, team_list)
    df = dedupe(df)
    df = normalize(df)

    args.stadiums.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.stadiums, index=False)

    print("[sync] OK: %s" % args.stadiums)
    print("[sync] Rows before: %d, after: %d" % (before, len(df)))
    print("[sync] New teams added: %d" % len(added))
    if added:
        print("[sync] Added preview: %s" % ", ".join(list(added)[:50]))

if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        print("[sync] ERROR: %s" % e)
        raise

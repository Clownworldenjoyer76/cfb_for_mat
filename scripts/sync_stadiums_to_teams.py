#!/usr/bin/env python3
"""
Sync `data/reference/stadiums.csv` to ensure:
- It contains one row per team found in your source list (teams file or schedules),
- It has all required columns with a consistent order,
- Existing non-null values are preserved.

Usage examples:
  python scripts/sync_stadiums_to_teams.py \
    --stadiums data/reference/stadiums.csv \
    --teams data/reference/teams.csv

  python scripts/sync_stadiums_to_teams.py \
    --stadiums data/reference/stadiums.csv \
    --schedules data/games.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

REQUIRED_COLS = [
    "team", "venue", "city", "state", "country",
    "lat", "lon", "timezone", "altitude_m",
    "is_neutral_site", "notes",
]

DEFAULTS = {
    "country": "USA",
    "is_neutral_site": 0,
}

def read_team_list(teams_csv: Path | None, schedules_csv: Path | None) -> pd.Series:
    if teams_csv and teams_csv.exists():
        df = pd.read_csv(teams_csv)
        # Accept common column names
        for col in ["team", "school", "name", "Team", "School"]:
            if col in df.columns:
                return df[col].dropna().astype(str).str.strip().drop_duplicates().sort_values()
        raise SystemExit(f"[sync] Could not find a team-name column in {teams_csv}")
    if schedules_csv and schedules_csv.exists():
        df = pd.read_csv(schedules_csv)
        candidates = [c for c in ["team", "home_team", "away_team", "Home Team", "Away Team"] if c in df.columns]
        if not candidates:
            raise SystemExit(f"[sync] Could not find team columns in {schedules_csv}")
        teams = pd.concat([df[c] for c in candidates], ignore_index=True)
        return teams.dropna().astype(str).str.strip().drop_duplicates().sort_values()
    raise SystemExit("[sync] Provide --teams or --schedules")

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Create any missing cols
    for c in REQUIRED_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    # Drop unexpected columns only if they fully duplicate required ones? No: keep them.
    # Reorder to required-first, then the rest
    rest = [c for c in out.columns if c not in REQUIRED_COLS]
    out = out[REQUIRED_COLS + rest]
    # Apply defaults where appropriate
    for k, v in DEFAULTS.items():
        if k in out.columns:
            out[k] = out[k].fillna(v)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stadiums", type=Path, required=True, help="Path to data/reference/stadiums.csv")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--teams", type=Path, help="CSV with a 'team' column (or School/Name)")
    g.add_argument("--schedules", type=Path, help="CSV with team columns (home_team/away_team)")
    args = p.parse_args()

    stad_path = args.stadiums
    if not stad_path.exists():
        # Start a new file if missing
        df = pd.DataFrame(columns=REQUIRED_COLS)
    else:
        df = pd.read_csv(stad_path)

    df = ensure_columns(df)

    wanted = read_team_list(args.teams, args.schedules)

    have = df["team"].fillna("").astype(str).str.strip()
    missing = sorted(set(wanted) - set(have))

    if missing:
        new_rows = []
        for t in missing:
            row = {c: pd.NA for c in df.columns}
            row.update(DEFAULTS)
            row["team"] = t
            new_rows.append(row)
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # Deduplicate by team (keep first non-null values)
    df.sort_values(by=["team"], key=lambda s: s.str.lower(), inplace=True)
    df = df.groupby("team", as_index=False).first()

    # Normalize types
    for num in ["lat", "lon", "altitude_m", "is_neutral_site"]:
        if num in df.columns:
            df[num] = pd.to_numeric(df[num], errors="coerce")

    # Persist
    stad_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stad_path, index=False)
    print(f"[sync] Stadiums synced: {stad_path} (rows={len(df)})")

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        raise
    except Exception as e:
        print(f"[sync] ERROR: {e}", file=sys.stderr)
        raise

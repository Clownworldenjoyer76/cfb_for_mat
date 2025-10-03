#!/usr/bin/env python3
"""
Sync data/reference/stadiums.csv so that:
- Required columns exist (created if missing).
- One row per team (dedup by team, keeping first seen values).
- Existing non-null values are preserved.
- Optional --teams or --schedules can add teams; if neither is provided,
  the script infers the list from the stadiums file itself.

Usage:
  python scripts/sync_stadiums_to_teams.py --stadiums data/reference/stadiums.csv
  optional: --teams data/reference/teams.csv
  optional: --schedules data/games.csv
"""

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


def canonicalize_team_series(series):
    s = series.dropna().astype(str).str.strip()
    s = s.replace("", pd.NA).dropna()
    # stable, case-insensitive sort
    return s.drop_duplicates().sort_values(key=lambda x: x.str.lower())


def find_team_column(df):
    for c in ["team", "Team", "school", "School", "name", "Name"]:
        if c in df.columns:
            return c
    return None


def read_team_list(teams_csv, schedules_csv, stad_df):
    if teams_csv:
        df = pd.read_csv(teams_csv)
        col = find_team_column(df)
        if not col:
            raise SystemExit(
                "[sync] Could not find a team column in %s. Expected one of: team, Team, school, School, name, Name"
                % teams_csv
            )
        return canonicalize_team_series(df[col])

    if schedules_csv:
        df = pd.read_csv(schedules_csv)
        cols = [c for c in ["team", "home_team", "away_team", "Home Team", "Away Team", "Home", "Away"] if c in df.columns]
        if not cols:
            raise SystemExit(
                "[sync] Could not find team columns in %s. Looked for: team, home_team, away_team, Home Team, Away Team, Home, Away"
                % schedules_csv
            )
        teams = pd.concat([df[c] for c in cols], ignore_index=True)
        return canonicalize_team_series(teams)

    if "team" not in stad_df.columns:
        raise SystemExit(
            "[sync] 'team' column missing in stadiums.csv and no --teams or --schedules provided."
        )
    return canonicalize_team_series(stad_df["team"])


def ensure_columns(df):
    out = df.copy()
    for c in REQUIRED_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    extras = [c for c in out.columns if c not in REQUIRED_COLS]
    out = out[REQUIRED_COLS + extras]
    for k, v in DEFAULTS.items():
        if k in out.columns:
            out[k] = out[k].fillna(v)
    return out


def add_missing_rows(df, desired_teams):
    have = df["team"].astype(str).str.strip().fillna("")
    missing = sorted(set(desired_teams) - set(have))
    if not missing:
        return df, []

    new_rows = []
    for t in missing:
        row = {c: pd.NA for c in df.columns}
        row.update(DEFAULTS)
        row["team"] = t
        new_rows.append(row)

    out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return out, missing


def normalize_types(df):
    out = df.copy()
    for num in ["lat", "lon", "altitude_m"]:
        if num in out.columns:
            out[num] = pd.to_numeric(out[num], errors="coerce")
    if "is_neutral_site" in out.columns:
        out["is_neutral_site"] = pd.to_numeric(out["is_neutral_site"], errors="coerce").fillna(0).astype(int)
    return out


def dedupe_by_team_preserve_first(df):
    out = df.copy()
    # stable sort by lowercase team to make groupby-first deterministic
    out["_team_sort"] = out["team"].astype(str).str.lower()
    out.sort_values(by="_team_sort", kind="mergesort", inplace=True)
    out.drop(columns="_team_sort", inplace=True)
    out = out.groupby("team", as_index=False).first()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stadiums", type=Path, required=True, help="Path to data/reference/stadiums.csv")
    p.add_argument("--teams", type=Path, help="CSV with a team-like column")
    p.add_argument("--schedules", type=Path, help="CSV with home/away team columns")
    args = p.parse_args()

    stad_path = args.stadiums
    if not stad_path.exists():
        raise SystemExit("[sync] Stadiums file not found: %s" % stad_path)

    stad = pd.read_csv(stad_path)
    stad = ensure_columns(stad)

    desired = read_team_list(args.teams, args.schedules, stad)

    before_rows = len(stad)
    stad, added = add_missing_rows(stad, desired)

    stad = dedupe_by_team_preserve_first(stad)
    stad = normalize_types(stad)

    stad_path.parent.mkdir(parents=True, exist_ok=True)
    stad.to_csv(stad_path, index=False)

    print("[sync] Stadiums synced: %s" % stad_path)
    print("[sync] Rows before: %d, after: %d" % (before_rows, len(stad)))
    print("[sync] New teams added: %d" % len(added))
    if added:
        preview = ", ".join(list(added)[:50])
        if len(added) > 50:
            preview += " (more omitted)"
        print("[sync] Added: %s" % preview)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        print("[sync] ERROR: %s" % e)
        raise

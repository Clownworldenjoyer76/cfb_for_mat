#!/usr/bin/env python3
"""
Sync data/reference/stadiums.csv so that:
- Required columns exist.
- There is one row per team.
- Existing non-null values are preserved.
- --teams and --schedules are optional. If neither is provided, the team list
  is inferred from the stadiums file itself.

Usage:
  python scripts/sync_stadiums_to_teams.py --stadiums data/reference/stadiums.csv
  (optional) --teams data/reference/teams.csv
  (optional) --schedules data/games.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

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


def canonicalize_team_series(series: pd.Series) -> pd.Series:
    s = (
        series.dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )
    # sort case-insensitively but stably
    return s.drop_duplicates().sort_values(key=lambda x: x.str.lower())


def find_team_column(df: pd.DataFrame) -> str | None:
    for c in ["team", "Team", "school", "School", "name", "Name"]:
        if c in df.columns:
            return c
    return None


def read_team_list(teams_csv: Path | None, schedules_csv: Path | None, stad_df: pd.DataFrame) -> pd.Series:
    if teams_csv:
        df = pd.read_csv(teams_csv)
        col = find_team_column(df)
        if not col:
            raise SystemExit(
                f"[sync] Could not find a team column in {teams_csv}. "
                "Expected one of: team, Team, school, School, name, Name"
            )
        return canonicalize_team_series(df[col])

    if schedules_csv:
        df = pd.read_csv(schedules_csv)
        cols = [c for c in ["team", "home_team", "away_team", "Home Team", "Away Team", "Home", "Away"] if c in df.columns]
        if not cols:
            raise SystemExit(
                f"[sync] Could not find team columns in {schedules_csv}. "
                "Looked for: team, home_team, away_team, Home Team, Away Team, Home, Away"
            )
        teams = pd.concat([df[c] for c in cols], ignore_index=True)
        return canonicalize_team_series(teams)

    if "team" not in stad_df.columns:
        raise SystemExit(
            "[sync] 'team' column missing in stadiums.csv and no --teams or --schedules provided."
        )
    return canonicalize_team_series(stad_df["team"])


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def add_missing_rows(df: pd.DataFrame, desired_teams: Iterable[str]) -> tuple[pd.DataFrame, List[str]]:
    have = df["team"].astype(str).str.strip().fillna("")
    missing = sorted(set(desired_teams) - set(have))
    if not missing:
        return df, []

    new_rows: List[dict] = []
    for t in missing:
        row = {c: pd.NA for c in df.columns}
        row.update(DEFAULTS)
        row["team"] = t
        new_rows.append(row)

    out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return out, missing


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for num in ["lat", "lon", "altitude_m"]:
        if num in out.columns:
            out[num] = pd.to_numeric(out[num], errors="coerce")
    if "is_neutral_site" in out.columns:
        out["is_neutral_site"] = pd.to_numeric(out["is_neutral_site"], errors="coerce").fillna(0).astype(int)
    return out


def dedupe_by_team_preserve_first(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["__team_sort__"] = out["team"].astype(str).str.lower()
    out.sort_values(by="__team_sort__", kind="mergesort", inplace=True)
    out.drop(columns="__team_sort__", inplace=True)
    out = out.groupby("team", as_index=False).first()
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stadiums", type=Path, required=True, help="Path to data/reference/stadiums.csv")
    p.add_argument("--teams", type=Path, help="CSV with a team-like column")
    p.add_argument("--schedules", type=Path, help="CSV with home/away team columns")
    args = p.parse_args()

    stad_path = args.stadiums
    if not stad_path.exists():
        raise SystemExit(f"[sync] Stadiums file not found: {stad_path}")

    stad = pd.read_csv(stad_path)
    stad = ensure_columns(stad)

    desired = read_team_list(args.teams, args.schedules, stad)

    before_rows = len(stad)
    stad, added = add_missing_rows(stad, desired)

    stad = dedupe_by_team_preserve_first(stad)
    stad = normalize_types(stad)

    stad_path.parent.mkdir(parents=True, exist_ok=True)
    stad.to_csv(stad_path, index=False)

    print(f"[sync] Stadiums synced: {stad_path}")
    print(f"[sync] Rows before: {before_rows}, after: {len(stad)}")
    print(f"[sync] New teams added: {len(added)}")
    if added:
        preview = ", ".join(list(added)[:50])
        if len(added) > 50:
            preview += " (more omitted)"
        print(f"[sync] Added: {preview}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        print(f"[sync] ERROR: {e}", file=sys.stderr)
        raise

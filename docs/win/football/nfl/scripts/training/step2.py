#!/usr/bin/env python3
"""
Step 2: append historical odds/weather fields to the Step 1 training table.

READS ONLY:
  docs/win/football/nfl/training/historical_core_2021_2025.csv
  docs/win/football/nfl/data/historic_data/odds/nfl_odds_{season}.csv
  docs/win/football/nfl/config/mapping/team_map.csv

WRITES:
  docs/win/football/nfl/training/historical_core_2021_2025.csv

The raw historical source files are never edited.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


NFL_ROOT = Path("docs/win/football/nfl")

TRAINING_PATH = NFL_ROOT / "training/historical_core_2021_2025.csv"
ODDS_DIR = NFL_ROOT / "data/historic_data/odds"
TEAM_MAP_PATH = NFL_ROOT / "config/mapping/team_map.csv"

SEASONS = range(2021, 2026)

JOIN_COLUMNS = ["season", "week", "home_team", "away_team"]

ODDS_REQUIRED_COLUMNS = [
    "season",
    "week",
    "home_team",
    "away_team",
    "odds_total",
    "home_spread",
    "away_spread",
    "surface",
    "weather_icon",
    "temperature",
    "precip_probability",
    "precip_type",
    "wind_speed",
    "wind_bearing",
]

ADD_COLUMN_MAP = {
    "odds_total": "hist_odds_total",
    "home_spread": "hist_home_spread",
    "away_spread": "hist_away_spread",
    "surface": "hist_surface",
    "weather_icon": "hist_weather_icon",
    "temperature": "hist_temperature",
    "precip_probability": "hist_precip_probability",
    "precip_type": "hist_precip_type",
    "wind_speed": "hist_wind_speed",
    "wind_bearing": "hist_wind_bearing",
}

FINAL_ADDED_COLUMNS = list(ADD_COLUMN_MAP.values())


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        low_memory=False,
    )


def require_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def clean_int_text(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64").astype("string")


def build_team_lookup(team_map: pd.DataFrame) -> dict[str, str]:
    require_columns(
        team_map,
        [
            "canonical_team",
            "team_abbr",
            "alias",
            "location",
            "team_name",
            "nickname",
            "shortDisplayName",
        ],
        TEAM_MAP_PATH,
    )

    lookup: dict[str, str] = {}

    for _, row in team_map.iterrows():
        team_abbr = str(row.get("team_abbr", "")).strip()
        if not team_abbr:
            continue

        candidates = [
            row.get("canonical_team"),
            row.get("team_abbr"),
            row.get("alias"),
            row.get("team_name"),
            row.get("nickname"),
            row.get("shortDisplayName"),
        ]

        location = str(row.get("location", "")).strip()
        team_name = str(row.get("team_name", "")).strip()
        if location and team_name:
            candidates.append(f"{location} {team_name}")

        for value in candidates:
            if pd.isna(value):
                continue
            key = " ".join(str(value).strip().casefold().split())
            if key:
                lookup[key] = team_abbr

    return lookup


def normalize_team(series: pd.Series, lookup: dict[str, str]) -> pd.Series:
    def convert(value):
        if pd.isna(value):
            return pd.NA

        text = " ".join(str(value).strip().split())
        if not text:
            return pd.NA

        key = text.casefold()
        return lookup.get(key, text)

    return series.map(convert).astype("string")


def require_unique_join_keys(df: pd.DataFrame, label: str) -> None:
    duplicate_mask = df.duplicated(
        subset=["_join_season", "_join_week", "_join_home_team", "_join_away_team"],
        keep=False,
    )

    if duplicate_mask.any():
        sample = (
            df.loc[
                duplicate_mask,
                ["_join_season", "_join_week", "_join_home_team", "_join_away_team"],
            ]
            .drop_duplicates()
            .head(10)
        )
        raise ValueError(
            f"{label}: duplicate join keys found:\n"
            + sample.to_string(index=False)
        )


def main() -> int:
    core = read_csv(TRAINING_PATH)
    require_columns(core, JOIN_COLUMNS, TRAINING_PATH)

    # Prevent accidental duplicate Step 2 columns on a rerun.
    existing_step2 = [column for column in FINAL_ADDED_COLUMNS if column in core.columns]
    if existing_step2:
        core = core.drop(columns=existing_step2)

    team_map = read_csv(TEAM_MAP_PATH)
    team_lookup = build_team_lookup(team_map)

    # Temporary normalized join keys. Original Step 1 columns remain unchanged.
    core["_join_season"] = clean_int_text(core["season"])
    core["_join_week"] = clean_int_text(core["week"])
    core["_join_home_team"] = normalize_team(core["home_team"], team_lookup)
    core["_join_away_team"] = normalize_team(core["away_team"], team_lookup)

    odds_frames: list[pd.DataFrame] = []

    for season in SEASONS:
        odds_path = ODDS_DIR / f"nfl_odds_{season}.csv"
        odds = read_csv(odds_path)
        require_columns(odds, ODDS_REQUIRED_COLUMNS, odds_path)

        odds = odds[ODDS_REQUIRED_COLUMNS].copy()

        odds["_join_season"] = clean_int_text(odds["season"])
        odds["_join_week"] = clean_int_text(odds["week"])
        odds["_join_home_team"] = normalize_team(odds["home_team"], team_lookup)
        odds["_join_away_team"] = normalize_team(odds["away_team"], team_lookup)

        odds = odds[
            [
                "_join_season",
                "_join_week",
                "_join_home_team",
                "_join_away_team",
                *ADD_COLUMN_MAP.keys(),
            ]
        ].copy()

        odds = odds.rename(columns=ADD_COLUMN_MAP)
        odds_frames.append(odds)

    historical_odds = pd.concat(odds_frames, ignore_index=True)

    require_unique_join_keys(historical_odds, "historical odds")

    merged = core.merge(
        historical_odds,
        on=[
            "_join_season",
            "_join_week",
            "_join_home_team",
            "_join_away_team",
        ],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    matched = int((merged["_merge"] == "both").sum())
    unmatched = int((merged["_merge"] == "left_only").sum())

    merged = merged.drop(
        columns=[
            "_join_season",
            "_join_week",
            "_join_home_team",
            "_join_away_team",
            "_merge",
        ]
    )

    # Preserve the Step 1 row count and row order.
    if len(merged) != len(core):
        raise RuntimeError(
            f"Row count changed during Step 2: before={len(core)} after={len(merged)}"
        )

    # Atomic replacement of the generated training artifact only.
    temp_path = TRAINING_PATH.with_suffix(".step2.tmp.csv")
    merged.to_csv(temp_path, index=False, encoding="utf-8")
    temp_path.replace(TRAINING_PATH)

    print(f"Rows before Step 2: {len(core)}")
    print(f"Historical odds/weather rows matched: {matched}")
    print(f"Historical odds/weather rows unmatched: {unmatched}")
    print(f"Added columns: {', '.join(FINAL_ADDED_COLUMNS)}")
    print(f"Wrote: {TRAINING_PATH}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

#!/usr/bin/env python3
"""
Step 5: append lagged team statistics to the historical NFL training table.

For a game in Week N, each team receives its Week N-1 team-stat row.

READS:
  docs/win/football/nfl/training/historical_core_2021_2025.csv
  docs/win/football/nfl/00_intake/team_stats/{season}_team_stats.csv

WRITES:
  docs/win/football/nfl/training/historical_core_2021_2025.csv

Appends:
  home_<metric>
  away_<metric>
  <metric>_diff = home_<metric> - away_<metric>

Week 1 remains unmatched here. Previous-season Week 1 fallback is handled
separately in Step 6.

No raw input/source files are edited.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


NFL_ROOT = Path("docs/win/football/nfl")

TRAINING_PATH = (
    NFL_ROOT / "training/historical_core_2021_2025.csv"
)

TEAM_STATS_DIR = (
    NFL_ROOT / "00_intake/team_stats"
)

KEY_COLUMNS = [
    "season",
    "week",
    "team",
]

TEAM_METRICS = [
    "off_epa_per_play",
    "def_epa_per_play",
    "off_success_rate",
    "def_success_rate",
    "yards_per_play",
    "yards_per_play_allowed",
    "points_per_drive",
    "points_per_drive_allowed",
    "red_zone_td_rate",
    "red_zone_td_rate_allowed",
    "early_down_epa",
    "third_down_conversion_rate",
]

HOME_COLUMNS = [
    f"home_{metric}"
    for metric in TEAM_METRICS
]

AWAY_COLUMNS = [
    f"away_{metric}"
    for metric in TEAM_METRICS
]

DIFF_COLUMNS = [
    f"{metric}_diff"
    for metric in TEAM_METRICS
]

STEP5_COLUMNS = [
    *HOME_COLUMNS,
    *AWAY_COLUMNS,
    *DIFF_COLUMNS,
]

REQUIRED_TRAINING_COLUMNS = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
]

REQUIRED_TEAM_STATS_COLUMNS = [
    *KEY_COLUMNS,
    *TEAM_METRICS,
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}"
        )

    return pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        low_memory=False,
    )


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label}: missing required columns: {missing}"
        )


def normalize_integer_key(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    bad = (
        numeric.isna()
        & series.notna()
        & series.astype(str).str.strip().ne("")
    )

    if bad.any():
        values = (
            series.loc[bad]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{column_name}: invalid numeric key values: "
            + ", ".join(values)
        )

    non_integer = (
        numeric.notna()
        & ((numeric % 1).abs() > 1e-9)
    )

    if non_integer.any():
        values = (
            series.loc[non_integer]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{column_name}: non-integer key values: "
            + ", ".join(values)
        )

    return numeric.astype("Int64")


def normalize_team_key(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )


def numeric_metric(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    converted = pd.to_numeric(
        series,
        errors="coerce",
    )

    bad = (
        converted.isna()
        & series.notna()
        & series.astype(str).str.strip().ne("")
    )

    if bad.any():
        values = (
            series.loc[bad]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{column_name}: non-numeric metric values: "
            + ", ".join(values)
        )

    return converted


def load_team_stats(
    seasons: list[int],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for season in seasons:
        path = (
            TEAM_STATS_DIR
            / f"{season}_team_stats.csv"
        )

        df = read_csv(path)

        require_columns(
            df,
            REQUIRED_TEAM_STATS_COLUMNS,
            f"team stats {season}",
        )

        df = df[
            REQUIRED_TEAM_STATS_COLUMNS
        ].copy()

        df["_join_season"] = (
            normalize_integer_key(
                df["season"],
                f"{path}: season",
            )
        )

        df["_join_week"] = (
            normalize_integer_key(
                df["week"],
                f"{path}: week",
            )
        )

        df["_join_team"] = (
            normalize_team_key(
                df["team"]
            )
        )

        for metric in TEAM_METRICS:
            df[metric] = numeric_metric(
                df[metric],
                f"{path}: {metric}",
            )

        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No historical team-stat files loaded."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    duplicate_mask = combined.duplicated(
        subset=[
            "_join_season",
            "_join_week",
            "_join_team",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = (
            combined.loc[
                duplicate_mask,
                [
                    "season",
                    "week",
                    "team",
                ],
            ]
            .drop_duplicates()
            .head(20)
        )

        sample = " | ".join(
            (
                f"season={row.season}, "
                f"week={row.week}, "
                f"team={row.team}"
            )
            for row in duplicate_rows.itertuples(
                index=False
            )
        )

        raise ValueError(
            "Historical team stats contain duplicate "
            f"season/week/team keys: {sample}"
        )

    return combined


def build_side_lookup(
    team_stats: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    if side not in ("home", "away"):
        raise ValueError(
            f"Unsupported side: {side}"
        )

    rename_map = {
        metric: f"{side}_{metric}"
        for metric in TEAM_METRICS
    }

    lookup = team_stats[
        [
            "_join_season",
            "_join_week",
            "_join_team",
            *TEAM_METRICS,
        ]
    ].copy()

    lookup = lookup.rename(
        columns=rename_map
    )

    return lookup


def main() -> int:
    training = read_csv(
        TRAINING_PATH
    )

    require_columns(
        training,
        REQUIRED_TRAINING_COLUMNS,
        "historical training table",
    )

    original_row_count = len(
        training
    )

    # Make reruns idempotent by replacing only
    # Step 5-generated columns.
    existing_step5_columns = [
        column
        for column in STEP5_COLUMNS
        if column in training.columns
    ]

    if existing_step5_columns:
        training = training.drop(
            columns=existing_step5_columns
        )

    training["_join_season"] = (
        normalize_integer_key(
            training["season"],
            "training season",
        )
    )

    training["_game_week"] = (
        normalize_integer_key(
            training["week"],
            "training week",
        )
    )

    training["_lag_week"] = (
        training["_game_week"] - 1
    )

    training["_home_team_key"] = (
        normalize_team_key(
            training["home_team"]
        )
    )

    training["_away_team_key"] = (
        normalize_team_key(
            training["away_team"]
        )
    )

    seasons = sorted(
        int(value)
        for value in (
            training["_join_season"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    team_stats = load_team_stats(
        seasons
    )

    home_lookup = build_side_lookup(
        team_stats,
        "home",
    ).rename(
        columns={
            "_join_week": "_lag_week",
            "_join_team": "_home_team_key",
        }
    )

    away_lookup = build_side_lookup(
        team_stats,
        "away",
    ).rename(
        columns={
            "_join_week": "_lag_week",
            "_join_team": "_away_team_key",
        }
    )

    training = training.merge(
        home_lookup,
        on=[
            "_join_season",
            "_lag_week",
            "_home_team_key",
        ],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    training = training.merge(
        away_lookup,
        on=[
            "_join_season",
            "_lag_week",
            "_away_team_key",
        ],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    if len(training) != original_row_count:
        raise RuntimeError(
            "Row count changed during Step 5: "
            f"before={original_row_count} "
            f"after={len(training)}"
        )

    for metric in TEAM_METRICS:
        home_column = f"home_{metric}"
        away_column = f"away_{metric}"
        diff_column = f"{metric}_diff"

        training[diff_column] = (
            training[home_column]
            - training[away_column]
        )

    # Week 1 must remain unfilled in this step.
    week_one_mask = (
        training["_game_week"] == 1
    )

    for column in STEP5_COLUMNS:
        if training.loc[
            week_one_mask,
            column,
        ].notna().any():
            raise RuntimeError(
                f"Week 1 unexpectedly populated in "
                f"Step 5 column: {column}"
            )

    home_match_count = int(
        training["home_off_epa_per_play"]
        .notna()
        .sum()
    )

    away_match_count = int(
        training["away_off_epa_per_play"]
        .notna()
        .sum()
    )

    both_match_count = int(
        (
            training["home_off_epa_per_play"]
            .notna()
            & training["away_off_epa_per_play"]
            .notna()
        )
        .sum()
    )

    week_one_count = int(
        week_one_mask.sum()
    )

    training = training.drop(
        columns=[
            "_join_season",
            "_game_week",
            "_lag_week",
            "_home_team_key",
            "_away_team_key",
        ]
    )

    missing_step5_columns = [
        column
        for column in STEP5_COLUMNS
        if column not in training.columns
    ]

    if missing_step5_columns:
        raise RuntimeError(
            "Step 5 output missing generated columns: "
            f"{missing_step5_columns}"
        )

    temp_path = (
        TRAINING_PATH
        .with_suffix(".step5.tmp.csv")
    )

    training.to_csv(
        temp_path,
        index=False,
        encoding="utf-8",
    )

    temp_path.replace(
        TRAINING_PATH
    )

    print(
        f"Rows processed: {len(training)}"
    )

    print(
        f"Team metrics: {len(TEAM_METRICS)}"
    )

    print(
        f"Step 5 columns added: "
        f"{len(STEP5_COLUMNS)}"
    )

    print(
        f"Home lag matches: "
        f"{home_match_count}/{len(training)}"
    )

    print(
        f"Away lag matches: "
        f"{away_match_count}/{len(training)}"
    )

    print(
        f"Both-team lag matches: "
        f"{both_match_count}/{len(training)}"
    )

    print(
        f"Week 1 rows left for Step 6 fallback: "
        f"{week_one_count}"
    )

    print(
        f"Wrote: {TRAINING_PATH}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise

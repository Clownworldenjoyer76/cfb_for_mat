#!/usr/bin/env python3
"""
Step 6: append lagged QB statistics to the split historical NFL training tables.

For a game in Week N, use each historical starting QB's latest available
weekly QB-stat row from the same season with source week < N.

Historical starting QB identity is already carried in the training table from:
  docs/win/football/nfl/data/historic_data/games/games_2010_2025.csv

via:
  home_qb_id
  away_qb_id

READS:
  docs/win/football/nfl/training/historical_core_2021.csv
  docs/win/football/nfl/training/historical_core_2022.csv
  docs/win/football/nfl/training/historical_core_2023.csv
  docs/win/football/nfl/training/historical_core_2024.csv
  docs/win/football/nfl/training/historical_core_2025.csv

  docs/win/football/nfl/00_intake/qb/{season}_qb_stats.csv

WRITES:
  docs/win/football/nfl/training/historical_core_2021.csv
  docs/win/football/nfl/training/historical_core_2022.csv
  docs/win/football/nfl/training/historical_core_2023.csv
  docs/win/football/nfl/training/historical_core_2024.csv
  docs/win/football/nfl/training/historical_core_2025.csv

Appends:
  home_qb_<metric>
  away_qb_<metric>
  qb_<metric>_diff = home_qb_<metric> - away_qb_<metric>

Metrics:
  epa_per_play
  cpoe
  air_yards
  sack_rate
  interception_rate
  fumble_rate

The source field dropbacks is loaded and used for deterministic duplicate
resolution if the same player has more than one row in the same season/week.

Week 1 remains unmatched here. Previous-season Week 1 fallback is handled
separately in the next step.

No raw input/source files are edited.
"""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
import math
import sys

import pandas as pd


NFL_ROOT = Path("docs/win/football/nfl")

TRAINING_DIR = (
    NFL_ROOT / "training"
)

TRAINING_SEASONS = [
    2021,
    2022,
    2023,
    2024,
    2025,
]

TRAINING_PATHS = {
    season: (
        TRAINING_DIR
        / f"historical_core_{season}.csv"
    )
    for season in TRAINING_SEASONS
}

QB_STATS_DIR = (
    NFL_ROOT / "00_intake/qb"
)

QB_METRICS = [
    "epa_per_play",
    "cpoe",
    "air_yards",
    "sack_rate",
    "interception_rate",
    "fumble_rate",
]

SOURCE_REQUIRED_COLUMNS = [
    "season",
    "week",
    "team",
    "player_id",
    "qb_name",
    "dropbacks",
    *QB_METRICS,
]

REQUIRED_TRAINING_COLUMNS = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_qb_id",
    "away_qb_id",
]

HOME_COLUMNS = [
    f"home_qb_{metric}"
    for metric in QB_METRICS
]

AWAY_COLUMNS = [
    f"away_qb_{metric}"
    for metric in QB_METRICS
]

DIFF_COLUMNS = [
    f"qb_{metric}_diff"
    for metric in QB_METRICS
]

STEP6_COLUMNS = [
    *HOME_COLUMNS,
    *AWAY_COLUMNS,
    *DIFF_COLUMNS,
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
            f"{column_name}: invalid numeric values: "
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
            f"{column_name}: non-integer values: "
            + ", ".join(values)
        )

    return numeric.astype("Int64")


def normalize_player_id(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
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
            f"{column_name}: non-numeric values: "
            + ", ".join(values)
        )

    return converted


def load_qb_stats(
    seasons: list[int],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for season in seasons:
        path = (
            QB_STATS_DIR
            / f"{season}_qb_stats.csv"
        )

        df = read_csv(path)

        require_columns(
            df,
            SOURCE_REQUIRED_COLUMNS,
            f"QB stats {season}",
        )

        df = df[
            SOURCE_REQUIRED_COLUMNS
        ].copy()

        df["_season_key"] = (
            normalize_integer_key(
                df["season"],
                f"{path}: season",
            )
        )

        df["_week_key"] = (
            normalize_integer_key(
                df["week"],
                f"{path}: week",
            )
        )

        df["_player_key"] = (
            normalize_player_id(
                df["player_id"]
            )
        )

        df["dropbacks"] = numeric_metric(
            df["dropbacks"],
            f"{path}: dropbacks",
        )

        for metric in QB_METRICS:
            df[metric] = numeric_metric(
                df[metric],
                f"{path}: {metric}",
            )

        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No historical QB-stat files loaded."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined[
        combined["_season_key"].notna()
        & combined["_week_key"].notna()
        & combined["_player_key"].notna()
        & combined["_player_key"].ne("")
    ].copy()

    # The generator groups by team as well as player. If a player ever has
    # multiple rows in the same season/week, keep the row with the most
    # dropbacks so the player/week lookup remains deterministic.
    combined["_dropbacks_sort"] = (
        combined["dropbacks"]
        .fillna(-1)
    )

    combined = combined.sort_values(
        by=[
            "_season_key",
            "_player_key",
            "_week_key",
            "_dropbacks_sort",
        ],
        kind="stable",
    )

    combined = combined.drop_duplicates(
        subset=[
            "_season_key",
            "_player_key",
            "_week_key",
        ],
        keep="last",
    )

    combined = combined.drop(
        columns=["_dropbacks_sort"]
    )

    return combined


def build_history_index(
    qb_stats: pd.DataFrame,
) -> dict[
    tuple[int, str],
    tuple[list[int], list[dict[str, float | None]]],
]:
    history: dict[
        tuple[int, str],
        tuple[list[int], list[dict[str, float | None]]],
    ] = {}

    grouped = qb_stats.groupby(
        ["_season_key", "_player_key"],
        sort=False,
        dropna=False,
    )

    for (season_value, player_id), group in grouped:
        season = int(season_value)
        player = str(player_id)

        group = group.sort_values(
            "_week_key",
            kind="stable",
        )

        weeks: list[int] = []
        values: list[dict[str, float | None]] = []

        for _, row in group.iterrows():
            week = int(
                row["_week_key"]
            )

            metric_values: dict[
                str,
                float | None,
            ] = {}

            for metric in QB_METRICS:
                value = row[metric]

                if pd.isna(value):
                    metric_values[metric] = None
                else:
                    numeric_value = float(value)

                    if math.isfinite(numeric_value):
                        metric_values[metric] = numeric_value
                    else:
                        metric_values[metric] = None

            weeks.append(week)
            values.append(metric_values)

        history[(season, player)] = (
            weeks,
            values,
        )

    return history


def latest_prior_values(
    history: dict[
        tuple[int, str],
        tuple[list[int], list[dict[str, float | None]]],
    ],
    season: int,
    game_week: int,
    player_id: str,
) -> dict[str, float | None] | None:
    entry = history.get(
        (season, player_id)
    )

    if entry is None:
        return None

    weeks, values = entry

    # First index whose source week is >= game week.
    # The row immediately before it is therefore the latest week < game week.
    position = bisect_left(
        weeks,
        game_week,
    )

    if position == 0:
        return None

    return values[position - 1]


def process_training_file(
    season: int,
    training_path: Path,
) -> dict[str, int]:
    training = read_csv(
        training_path
    )

    require_columns(
        training,
        REQUIRED_TRAINING_COLUMNS,
        f"historical training table {season}",
    )

    original_row_count = len(
        training
    )

    existing_step6_columns = [
        column
        for column in STEP6_COLUMNS
        if column in training.columns
    ]

    if existing_step6_columns:
        training = training.drop(
            columns=existing_step6_columns
        )

    training["_season_key"] = (
        normalize_integer_key(
            training["season"],
            f"{training_path}: season",
        )
    )

    training["_week_key"] = (
        normalize_integer_key(
            training["week"],
            f"{training_path}: week",
        )
    )

    training["_home_qb_key"] = (
        normalize_player_id(
            training["home_qb_id"]
        )
    )

    training["_away_qb_key"] = (
        normalize_player_id(
            training["away_qb_id"]
        )
    )

    seasons_in_file = sorted(
        int(value)
        for value in (
            training["_season_key"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    if seasons_in_file != [season]:
        raise ValueError(
            f"{training_path}: expected only season "
            f"{season}, found {seasons_in_file}"
        )

    qb_stats = load_qb_stats(
        [season]
    )

    history = build_history_index(
        qb_stats
    )

    for column in HOME_COLUMNS + AWAY_COLUMNS:
        training[column] = pd.NA

    home_match_count = 0
    away_match_count = 0
    both_match_count = 0

    for index, row in training.iterrows():
        if (
            pd.isna(row["_season_key"])
            or pd.isna(row["_week_key"])
        ):
            continue

        row_season = int(
            row["_season_key"]
        )

        game_week = int(
            row["_week_key"]
        )

        home_qb_id = str(
            row["_home_qb_key"]
        ).strip()

        away_qb_id = str(
            row["_away_qb_key"]
        ).strip()

        home_values = None
        away_values = None

        if (
            home_qb_id
            and home_qb_id.lower() != "<na>"
        ):
            home_values = latest_prior_values(
                history,
                row_season,
                game_week,
                home_qb_id,
            )

        if (
            away_qb_id
            and away_qb_id.lower() != "<na>"
        ):
            away_values = latest_prior_values(
                history,
                row_season,
                game_week,
                away_qb_id,
            )

        if home_values is not None:
            home_match_count += 1

            for metric in QB_METRICS:
                training.at[
                    index,
                    f"home_qb_{metric}",
                ] = home_values[metric]

        if away_values is not None:
            away_match_count += 1

            for metric in QB_METRICS:
                training.at[
                    index,
                    f"away_qb_{metric}",
                ] = away_values[metric]

        if (
            home_values is not None
            and away_values is not None
        ):
            both_match_count += 1

    for column in HOME_COLUMNS + AWAY_COLUMNS:
        training[column] = pd.to_numeric(
            training[column],
            errors="coerce",
        )

    for metric in QB_METRICS:
        home_column = (
            f"home_qb_{metric}"
        )

        away_column = (
            f"away_qb_{metric}"
        )

        diff_column = (
            f"qb_{metric}_diff"
        )

        training[diff_column] = (
            training[home_column]
            - training[away_column]
        )

    if len(training) != original_row_count:
        raise RuntimeError(
            f"Row count changed during Step 6 for "
            f"{season}: before={original_row_count} "
            f"after={len(training)}"
        )

    # Same-season prior-week logic means Week 1 must be blank here.
    week_one_mask = (
        training["_week_key"] == 1
    )

    for column in STEP6_COLUMNS:
        if training.loc[
            week_one_mask,
            column,
        ].notna().any():
            raise RuntimeError(
                f"{season}: Week 1 unexpectedly populated "
                f"in Step 6 column: {column}"
            )

    missing_generated_columns = [
        column
        for column in STEP6_COLUMNS
        if column not in training.columns
    ]

    if missing_generated_columns:
        raise RuntimeError(
            f"{season}: Step 6 output missing generated "
            f"columns: {missing_generated_columns}"
        )

    week_one_count = int(
        week_one_mask.sum()
    )

    training = training.drop(
        columns=[
            "_season_key",
            "_week_key",
            "_home_qb_key",
            "_away_qb_key",
        ]
    )

    temp_path = (
        training_path
        .with_suffix(".step6.tmp.csv")
    )

    training.to_csv(
        temp_path,
        index=False,
        encoding="utf-8",
    )

    temp_path.replace(
        training_path
    )

    print(
        f"Season {season}"
    )

    print(
        f"Rows processed: {len(training)}"
    )

    print(
        f"QB metrics added: {len(QB_METRICS)}"
    )

    print(
        f"Step 6 columns added: "
        f"{len(STEP6_COLUMNS)}"
    )

    print(
        f"Home QB prior-row matches: "
        f"{home_match_count}/{len(training)}"
    )

    print(
        f"Away QB prior-row matches: "
        f"{away_match_count}/{len(training)}"
    )

    print(
        f"Both-QB prior-row matches: "
        f"{both_match_count}/{len(training)}"
    )

    print(
        f"Week 1 rows left for fallback: "
        f"{week_one_count}"
    )

    print(
        f"Wrote: {training_path}"
    )

    print()

    return {
        "rows": len(training),
        "home_matches": home_match_count,
        "away_matches": away_match_count,
        "both_matches": both_match_count,
        "week_one_rows": week_one_count,
    }


def main() -> int:
    total_rows = 0
    total_home_matches = 0
    total_away_matches = 0
    total_both_matches = 0
    total_week_one_rows = 0

    for season in TRAINING_SEASONS:
        training_path = TRAINING_PATHS[
            season
        ]

        result = process_training_file(
            season,
            training_path,
        )

        total_rows += result[
            "rows"
        ]

        total_home_matches += result[
            "home_matches"
        ]

        total_away_matches += result[
            "away_matches"
        ]

        total_both_matches += result[
            "both_matches"
        ]

        total_week_one_rows += result[
            "week_one_rows"
        ]

    print(
        "Step 6 complete."
    )

    print(
        f"Season files processed: "
        f"{len(TRAINING_SEASONS)}"
    )

    print(
        f"Total rows processed: "
        f"{total_rows}"
    )

    print(
        f"Total home QB prior-row matches: "
        f"{total_home_matches}/{total_rows}"
    )

    print(
        f"Total away QB prior-row matches: "
        f"{total_away_matches}/{total_rows}"
    )

    print(
        f"Total both-QB prior-row matches: "
        f"{total_both_matches}/{total_rows}"
    )

    print(
        f"Total Week 1 rows left for fallback: "
        f"{total_week_one_rows}"
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

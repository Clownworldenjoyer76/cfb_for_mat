#!/usr/bin/env python3
"""
Step 7: fill Week 1 team and QB lagged features from the prior season.

For each Week 1 game in the 2022-2025 historical training files:

TEAM FEATURES
  Use the final available prior-season row for each team from:
    docs/win/football/nfl/00_intake/team_stats/{season-1}_team_stats.csv

QB FEATURES
  Find the Week 1 game in:
    docs/win/football/nfl/data/historic_data/games/games_2010_2025.csv

  Match that game by:
    season
    week
    home_team
    away_team

  Take:
    home_qb_id
    away_qb_id

  Then use each QB's final available prior-season row from:
    docs/win/football/nfl/00_intake/qb/{season-1}_qb_stats.csv

READS/WRITES IN PLACE:
  docs/win/football/nfl/training/historical_core_2022.csv
  docs/win/football/nfl/training/historical_core_2023.csv
  docs/win/football/nfl/training/historical_core_2024.csv
  docs/win/football/nfl/training/historical_core_2025.csv

2021 is intentionally left unchanged because the required 2020 team/QB
source files do not exist in the intake directories.

Only Week 1 Step 5 / Step 6 feature columns are changed.
Rows from Week 2 onward are not changed.
No raw source files are edited.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import pandas as pd


NFL_ROOT = Path("docs/win/football/nfl")

TRAINING_DIR = NFL_ROOT / "training"
TEAM_STATS_DIR = NFL_ROOT / "00_intake/team_stats"
QB_STATS_DIR = NFL_ROOT / "00_intake/qb"
GAMES_PATH = NFL_ROOT / "data/historic_data/games/games_2010_2025.csv"

TRAINING_SEASONS = [2022, 2023, 2024, 2025]

TRAINING_PATHS = {
    season: TRAINING_DIR / f"historical_core_{season}.csv"
    for season in TRAINING_SEASONS
}

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

QB_METRICS = [
    "epa_per_play",
    "cpoe",
    "air_yards",
    "sack_rate",
    "interception_rate",
    "fumble_rate",
]

TRAINING_REQUIRED_COLUMNS = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_qb_id",
    "away_qb_id",
]

GAMES_REQUIRED_COLUMNS = [
    "season",
    "week",
    "home_team",
    "away_team",
    "home_qb_id",
    "away_qb_id",
]

TEAM_SOURCE_REQUIRED_COLUMNS = [
    "season",
    "week",
    "team",
    *TEAM_METRICS,
]

QB_SOURCE_REQUIRED_COLUMNS = [
    "season",
    "week",
    "player_id",
    "dropbacks",
    *QB_METRICS,
]

HOME_TEAM_COLUMNS = [
    f"home_{metric}"
    for metric in TEAM_METRICS
]

AWAY_TEAM_COLUMNS = [
    f"away_{metric}"
    for metric in TEAM_METRICS
]

TEAM_DIFF_COLUMNS = [
    f"{metric}_diff"
    for metric in TEAM_METRICS
]

HOME_QB_COLUMNS = [
    f"home_qb_{metric}"
    for metric in QB_METRICS
]

AWAY_QB_COLUMNS = [
    f"away_qb_{metric}"
    for metric in QB_METRICS
]

QB_DIFF_COLUMNS = [
    f"qb_{metric}_diff"
    for metric in QB_METRICS
]

STEP7_FEATURE_COLUMNS = [
    *HOME_TEAM_COLUMNS,
    *AWAY_TEAM_COLUMNS,
    *TEAM_DIFF_COLUMNS,
    *HOME_QB_COLUMNS,
    *AWAY_QB_COLUMNS,
    *QB_DIFF_COLUMNS,
]

BLANK_STRINGS = {
    "",
    "nan",
    "none",
    "<na>",
    "null",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}"
        )

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
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


def clean_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in BLANK_STRINGS:
        return ""

    return text


def normalize_team(value: object) -> str:
    return clean_text(value).upper()


def normalize_player_id(value: object) -> str:
    return clean_text(value)


def parse_int(
    value: object,
    label: str,
) -> int:
    text = clean_text(value)

    if text == "":
        raise ValueError(
            f"{label}: blank integer value"
        )

    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(
            f"{label}: invalid integer value {text!r}"
        ) from exc

    if not math.isfinite(numeric):
        raise ValueError(
            f"{label}: non-finite integer value {text!r}"
        )

    rounded = round(numeric)

    if abs(numeric - rounded) > 1e-9:
        raise ValueError(
            f"{label}: non-integer value {text!r}"
        )

    return int(rounded)


def parse_optional_float(
    value: object,
    label: str,
) -> float | None:
    text = clean_text(value)

    if text == "":
        return None

    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(
            f"{label}: invalid numeric value {text!r}"
        ) from exc

    if not math.isfinite(numeric):
        return None

    return numeric


def normalize_numeric_string(
    value: object,
    label: str,
) -> str:
    text = clean_text(value)

    if text == "":
        return ""

    numeric = parse_optional_float(
        text,
        label,
    )

    if numeric is None:
        return ""

    return text


def diff_string(
    home_value: str,
    away_value: str,
    label: str,
) -> str:
    home_numeric = parse_optional_float(
        home_value,
        f"{label}: home",
    )

    away_numeric = parse_optional_float(
        away_value,
        f"{label}: away",
    )

    if (
        home_numeric is None
        or away_numeric is None
    ):
        return ""

    return str(
        home_numeric - away_numeric
    )


def build_games_qb_index() -> dict[
    tuple[int, int, str, str],
    tuple[str, str],
]:
    games = read_csv(
        GAMES_PATH
    )

    require_columns(
        games,
        GAMES_REQUIRED_COLUMNS,
        "historical games",
    )

    index: dict[
        tuple[int, int, str, str],
        tuple[str, str],
    ] = {}

    for row_number, row in games.iterrows():
        season_text = clean_text(
            row["season"]
        )
        week_text = clean_text(
            row["week"]
        )

        if (
            season_text == ""
            or week_text == ""
        ):
            continue

        season = parse_int(
            season_text,
            f"{GAMES_PATH} row {row_number + 2}: season",
        )

        week = parse_int(
            week_text,
            f"{GAMES_PATH} row {row_number + 2}: week",
        )

        if (
            season not in TRAINING_SEASONS
            or week != 1
        ):
            continue

        home_team = normalize_team(
            row["home_team"]
        )

        away_team = normalize_team(
            row["away_team"]
        )

        if (
            home_team == ""
            or away_team == ""
        ):
            raise ValueError(
                f"{GAMES_PATH} row {row_number + 2}: "
                "blank home/away team for a required Week 1 game"
            )

        key = (
            season,
            week,
            home_team,
            away_team,
        )

        if key in index:
            raise ValueError(
                f"{GAMES_PATH}: duplicate Week 1 game key: {key}"
            )

        index[key] = (
            normalize_player_id(
                row["home_qb_id"]
            ),
            normalize_player_id(
                row["away_qb_id"]
            ),
        )

    return index


def load_final_team_rows(
    prior_season: int,
) -> dict[
    str,
    dict[str, str],
]:
    path = (
        TEAM_STATS_DIR
        / f"{prior_season}_team_stats.csv"
    )

    team_stats = read_csv(
        path
    )

    require_columns(
        team_stats,
        TEAM_SOURCE_REQUIRED_COLUMNS,
        f"team stats {prior_season}",
    )

    candidates: dict[
        str,
        tuple[int, dict[str, str]],
    ] = {}

    seen_keys: set[
        tuple[int, int, str]
    ] = set()

    for row_number, row in team_stats.iterrows():
        season = parse_int(
            row["season"],
            f"{path} row {row_number + 2}: season",
        )

        if season != prior_season:
            raise ValueError(
                f"{path} row {row_number + 2}: "
                f"expected season {prior_season}, found {season}"
            )

        week = parse_int(
            row["week"],
            f"{path} row {row_number + 2}: week",
        )

        team = normalize_team(
            row["team"]
        )

        if team == "":
            raise ValueError(
                f"{path} row {row_number + 2}: blank team"
            )

        source_key = (
            season,
            week,
            team,
        )

        if source_key in seen_keys:
            raise ValueError(
                f"{path}: duplicate season/week/team row: "
                f"{source_key}"
            )

        seen_keys.add(
            source_key
        )

        values: dict[
            str,
            str,
        ] = {}

        for metric in TEAM_METRICS:
            values[metric] = (
                normalize_numeric_string(
                    row[metric],
                    f"{path} row {row_number + 2}: {metric}",
                )
            )

        current = candidates.get(
            team
        )

        if (
            current is None
            or week > current[0]
        ):
            candidates[team] = (
                week,
                values,
            )

    return {
        team: values
        for team, (_, values) in candidates.items()
    }


def load_final_qb_rows(
    prior_season: int,
) -> dict[
    str,
    dict[str, str],
]:
    path = (
        QB_STATS_DIR
        / f"{prior_season}_qb_stats.csv"
    )

    qb_stats = read_csv(
        path
    )

    require_columns(
        qb_stats,
        QB_SOURCE_REQUIRED_COLUMNS,
        f"QB stats {prior_season}",
    )

    per_week: dict[
        tuple[str, int],
        tuple[float, int, dict[str, str]],
    ] = {}

    for row_number, row in qb_stats.iterrows():
        season = parse_int(
            row["season"],
            f"{path} row {row_number + 2}: season",
        )

        if season != prior_season:
            raise ValueError(
                f"{path} row {row_number + 2}: "
                f"expected season {prior_season}, found {season}"
            )

        week = parse_int(
            row["week"],
            f"{path} row {row_number + 2}: week",
        )

        player_id = normalize_player_id(
            row["player_id"]
        )

        if player_id == "":
            continue

        dropbacks_numeric = parse_optional_float(
            row["dropbacks"],
            f"{path} row {row_number + 2}: dropbacks",
        )

        dropbacks = (
            -1.0
            if dropbacks_numeric is None
            else dropbacks_numeric
        )

        values: dict[
            str,
            str,
        ] = {}

        for metric in QB_METRICS:
            values[metric] = (
                normalize_numeric_string(
                    row[metric],
                    f"{path} row {row_number + 2}: {metric}",
                )
            )

        key = (
            player_id,
            week,
        )

        current = per_week.get(
            key
        )

        if (
            current is None
            or dropbacks > current[0]
            or (
                dropbacks == current[0]
                and row_number > current[1]
            )
        ):
            per_week[key] = (
                dropbacks,
                row_number,
                values,
            )

    final_rows: dict[
        str,
        tuple[int, dict[str, str]],
    ] = {}

    for (
        player_id,
        week,
    ), (
        _dropbacks,
        _row_number,
        values,
    ) in per_week.items():
        current = final_rows.get(
            player_id
        )

        if (
            current is None
            or week > current[0]
        ):
            final_rows[player_id] = (
                week,
                values,
            )

    return {
        player_id: values
        for player_id, (_, values) in final_rows.items()
    }


def validate_training_file(
    training: pd.DataFrame,
    season: int,
    path: Path,
) -> list[int]:
    require_columns(
        training,
        [
            *TRAINING_REQUIRED_COLUMNS,
            *STEP7_FEATURE_COLUMNS,
        ],
        f"historical training table {season}",
    )

    if len(
        training.columns
    ) != len(
        set(training.columns)
    ):
        raise ValueError(
            f"{path}: duplicate column names"
        )

    seasons_found: set[
        int
    ] = set()

    week_one_indexes: list[
        int
    ] = []

    for index, row in training.iterrows():
        row_season = parse_int(
            row["season"],
            f"{path} row {index + 2}: season",
        )

        row_week = parse_int(
            row["week"],
            f"{path} row {index + 2}: week",
        )

        seasons_found.add(
            row_season
        )

        if row_week == 1:
            week_one_indexes.append(
                index
            )

    if seasons_found != {
        season
    }:
        raise ValueError(
            f"{path}: expected only season {season}, "
            f"found {sorted(seasons_found)}"
        )

    if not week_one_indexes:
        raise RuntimeError(
            f"{path}: no Week 1 rows found"
        )

    return week_one_indexes


def process_season(
    season: int,
    games_qb_index: dict[
        tuple[int, int, str, str],
        tuple[str, str],
    ],
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    training_path = (
        TRAINING_PATHS[
            season
        ]
    )

    training = read_csv(
        training_path
    )

    original_columns = (
        training.columns.tolist()
    )

    original_row_count = len(
        training
    )

    week_one_indexes = (
        validate_training_file(
            training,
            season,
            training_path,
        )
    )

    non_week_one_mask = pd.Series(
        True,
        index=training.index,
    )

    non_week_one_mask.loc[
        week_one_indexes
    ] = False

    non_week_one_before = (
        training.loc[
            non_week_one_mask,
            :,
        ]
        .copy(deep=True)
    )

    prior_season = (
        season - 1
    )

    final_team_rows = (
        load_final_team_rows(
            prior_season
        )
    )

    final_qb_rows = (
        load_final_qb_rows(
            prior_season
        )
    )

    team_side_matches = 0
    game_matches = 0
    qb_id_available = 0
    qb_prior_source_matches = 0
    training_qb_id_mismatches = 0

    for index in week_one_indexes:
        home_team = normalize_team(
            training.at[
                index,
                "home_team",
            ]
        )

        away_team = normalize_team(
            training.at[
                index,
                "away_team",
            ]
        )

        home_team_values = (
            final_team_rows.get(
                home_team
            )
        )

        away_team_values = (
            final_team_rows.get(
                away_team
            )
        )

        if home_team_values is None:
            raise RuntimeError(
                f"{season} Week 1: no prior-season team row "
                f"found for home team {home_team}"
            )

        if away_team_values is None:
            raise RuntimeError(
                f"{season} Week 1: no prior-season team row "
                f"found for away team {away_team}"
            )

        team_side_matches += 2

        for metric in TEAM_METRICS:
            home_value = (
                home_team_values[
                    metric
                ]
            )

            away_value = (
                away_team_values[
                    metric
                ]
            )

            training.at[
                index,
                f"home_{metric}",
            ] = home_value

            training.at[
                index,
                f"away_{metric}",
            ] = away_value

            training.at[
                index,
                f"{metric}_diff",
            ] = diff_string(
                home_value,
                away_value,
                (
                    f"{season} Week 1 "
                    f"{away_team} at {home_team} "
                    f"{metric}"
                ),
            )

        game_key = (
            season,
            1,
            home_team,
            away_team,
        )

        qb_ids = (
            games_qb_index.get(
                game_key
            )
        )

        if qb_ids is None:
            raise RuntimeError(
                f"{season} Week 1: historical game not found "
                f"for {away_team} at {home_team}"
            )

        game_matches += 1

        (
            home_qb_id,
            away_qb_id,
        ) = qb_ids

        training_home_qb_id = (
            normalize_player_id(
                training.at[
                    index,
                    "home_qb_id",
                ]
            )
        )

        training_away_qb_id = (
            normalize_player_id(
                training.at[
                    index,
                    "away_qb_id",
                ]
            )
        )

        if (
            home_qb_id
            and training_home_qb_id
            and home_qb_id
            != training_home_qb_id
        ):
            training_qb_id_mismatches += 1

        if (
            away_qb_id
            and training_away_qb_id
            and away_qb_id
            != training_away_qb_id
        ):
            training_qb_id_mismatches += 1

        home_qb_values = (
            final_qb_rows.get(
                home_qb_id
            )
            if home_qb_id
            else None
        )

        away_qb_values = (
            final_qb_rows.get(
                away_qb_id
            )
            if away_qb_id
            else None
        )

        if home_qb_id:
            qb_id_available += 1

        if away_qb_id:
            qb_id_available += 1

        if home_qb_values is not None:
            qb_prior_source_matches += 1

        if away_qb_values is not None:
            qb_prior_source_matches += 1

        for metric in QB_METRICS:
            home_value = (
                ""
                if home_qb_values is None
                else home_qb_values[
                    metric
                ]
            )

            away_value = (
                ""
                if away_qb_values is None
                else away_qb_values[
                    metric
                ]
            )

            training.at[
                index,
                f"home_qb_{metric}",
            ] = home_value

            training.at[
                index,
                f"away_qb_{metric}",
            ] = away_value

            training.at[
                index,
                f"qb_{metric}_diff",
            ] = diff_string(
                home_value,
                away_value,
                (
                    f"{season} Week 1 "
                    f"{away_team} at {home_team} "
                    f"QB {metric}"
                ),
            )

    if len(
        training
    ) != original_row_count:
        raise RuntimeError(
            f"{season}: row count changed during Step 7"
        )

    if (
        training.columns.tolist()
        != original_columns
    ):
        raise RuntimeError(
            f"{season}: column order changed during Step 7"
        )

    non_week_one_after = (
        training.loc[
            non_week_one_mask,
            :,
        ]
    )

    if not non_week_one_after.equals(
        non_week_one_before
    ):
        raise RuntimeError(
            f"{season}: Step 7 modified Week 2+ rows"
        )

    for index in week_one_indexes:
        for metric in TEAM_METRICS:
            expected = diff_string(
                training.at[
                    index,
                    f"home_{metric}",
                ],
                training.at[
                    index,
                    f"away_{metric}",
                ],
                (
                    f"{season} validation "
                    f"{metric}"
                ),
            )

            actual = clean_text(
                training.at[
                    index,
                    f"{metric}_diff",
                ]
            )

            if actual != expected:
                raise RuntimeError(
                    f"{season}: invalid Week 1 "
                    f"{metric}_diff at row {index + 2}"
                )

        for metric in QB_METRICS:
            expected = diff_string(
                training.at[
                    index,
                    f"home_qb_{metric}",
                ],
                training.at[
                    index,
                    f"away_qb_{metric}",
                ],
                (
                    f"{season} validation "
                    f"QB {metric}"
                ),
            )

            actual = clean_text(
                training.at[
                    index,
                    f"qb_{metric}_diff",
                ]
            )

            if actual != expected:
                raise RuntimeError(
                    f"{season}: invalid Week 1 "
                    f"qb_{metric}_diff at row {index + 2}"
                )

    stats = {
        "week_one_games": len(
            week_one_indexes
        ),
        "team_side_matches": (
            team_side_matches
        ),
        "historical_game_matches": (
            game_matches
        ),
        "qb_ids_available": (
            qb_id_available
        ),
        "qb_prior_source_matches": (
            qb_prior_source_matches
        ),
        "training_qb_id_mismatches": (
            training_qb_id_mismatches
        ),
    }

    return (
        training,
        stats,
    )


def write_outputs(
    outputs: dict[
        int,
        pd.DataFrame,
    ],
) -> None:
    temp_paths: dict[
        int,
        Path,
    ] = {}

    try:
        for season in TRAINING_SEASONS:
            output_path = (
                TRAINING_PATHS[
                    season
                ]
            )

            temp_path = (
                output_path
                .with_suffix(
                    ".step7.tmp.csv"
                )
            )

            temp_paths[
                season
            ] = temp_path

            outputs[
                season
            ].to_csv(
                temp_path,
                index=False,
                encoding="utf-8",
                lineterminator="\n",
            )

        for season in TRAINING_SEASONS:
            temp_paths[
                season
            ].replace(
                TRAINING_PATHS[
                    season
                ]
            )

    except Exception:
        for temp_path in (
            temp_paths.values()
        ):
            if temp_path.exists():
                temp_path.unlink()

        raise


def main() -> int:
    games_qb_index = (
        build_games_qb_index()
    )

    outputs: dict[
        int,
        pd.DataFrame,
    ] = {}

    results: dict[
        int,
        dict[str, int],
    ] = {}

    for season in TRAINING_SEASONS:
        (
            training,
            stats,
        ) = process_season(
            season,
            games_qb_index,
        )

        outputs[
            season
        ] = training

        results[
            season
        ] = stats

    write_outputs(
        outputs
    )

    print(
        "Step 7 complete."
    )

    for season in TRAINING_SEASONS:
        stats = results[
            season
        ]

        total_qb_sides = (
            stats[
                "week_one_games"
            ]
            * 2
        )

        print(
            f"{season}: "
            f"Week 1 games="
            f"{stats['week_one_games']}, "
            f"team sides matched="
            f"{stats['team_side_matches']}/"
            f"{total_qb_sides}, "
            f"historical games matched="
            f"{stats['historical_game_matches']}/"
            f"{stats['week_one_games']}, "
            f"QB IDs available="
            f"{stats['qb_ids_available']}/"
            f"{total_qb_sides}, "
            f"QBs with prior-season stats="
            f"{stats['qb_prior_source_matches']}/"
            f"{total_qb_sides}, "
            f"training/source QB-ID mismatches="
            f"{stats['training_qb_id_mismatches']}"
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

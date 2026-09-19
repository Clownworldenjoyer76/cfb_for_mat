#!/usr/bin/env python3
"""
CFB Week 2+ in-season projection.

This replaces the old NFL Week 2+ projection wrapper.

It reuses the proven CFB projection logic from projection_week1.py, but changes
the team-strength source from prior-season stats to completed current-season
team stats.

Inputs
------
1. docs/win/football/cfb/00_intake/schedule/weekly/
   week_{week}_CFB_weekly_schedule.csv

2. docs/win/football/cfb/00_intake/team_stats/
   {season}_team_stats.csv

3. docs/win/football/cfb/data/team_power_index/
   team_power_index_{season}.csv

4. docs/win/football/cfb/00_intake/predictions/final/
   {season}_*_{week}_clean_predictions.csv

5. docs/win/football/cfb/00_intake/injuries/
   {season}_injuries.csv

6. docs/win/football/cfb/config/mapping/team_map.csv

7. docs/win/football/cfb/config/mapping/stadium_map.csv

8. docs/win/football/cfb/data/travel/
   {season}_week_{week}_travel.csv

9. docs/win/football/cfb/data/weather/
   week_{week}_CFB_weekly_weather.csv

10. docs/win/football/cfb/config/
    travel_weather_coefficients.csv

Output
------
docs/win/football/cfb/01_merge/week_{week}_CFB_enriched.csv

Target week
-----------
If --season or --week is omitted, the missing target value is read from
config/current_week.yaml. Explicit CLI values take precedence.

Safety checks
-------------
- Week 1 is rejected. Use projection_week1.py for Week 1.
- Only current-season team-stat rows with source week < target week are used.
- The latest completed team-stat week must equal target_week - 1.
- Prior completed-game IDs must exactly match the stored PBP game IDs.
- PBP-derived prior team/week coverage must exactly match team-stat coverage.
- Betting probabilities are validated.
- Output season/week/game_id integrity is validated.

Model
-----
The shared projection logic uses:
- current market spread
- ESPN FPI
- finalized ESPN predictions
- completed current-season team-strength statistics
- current injuries
- fitted travel adjustment
- current market total
- fitted outdoor-weather adjustment
- current-season points-per-drive information

For Week 2+, fields containing "prior" in the shared projection output mean
information available prior to the target game. Their underlying team-strength
source is the current season rather than the previous season.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import yaml

import projection_week1 as base


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]
REPORT_ROOT = CFB_ROOT / "errors"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPTS_DIR),
    )

from pipeline_reporter import PipelineReporter


SCRIPT_VERSION = "cfb-inseason-v6-history-coverage-2026-09-16"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build CFB Week 2+ in-season projections "
            "and betting probabilities."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Target season. Defaults to config/current_week.yaml.",
    )

    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help=(
            "Target week. Defaults to config/current_week.yaml."
        ),
    )

    parser.add_argument(
        "--min-current-team-weeks",
        type=int,
        default=1,
        help=(
            "Minimum completed current-season team-stat weeks "
            "required before using that team's team-strength "
            "margin component. Default: 1."
        ),
    )

    parser.add_argument(
        "--home-field",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--drives-per-team",
        type=float,
        default=11.5,
    )

    parser.add_argument(
        "--market-margin-weight",
        type=float,
        default=0.36,
    )

    parser.add_argument(
        "--fpi-margin-weight",
        type=float,
        default=0.28,
    )

    parser.add_argument(
        "--espn-margin-weight",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--prior-margin-weight",
        type=float,
        default=0.16,
        help=(
            "Weight for the completed current-season "
            "team-strength margin component. The argument name "
            "is retained for compatibility with projection_week1.py."
        ),
    )

    parser.add_argument(
        "--market-total-weight",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--fresh-injury-days",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--margin-sd",
        type=float,
        default=base.DEFAULT_MARGIN_SD,
        help=(
            "Margin forecast error SD used for "
            "win/cover probabilities."
        ),
    )

    parser.add_argument(
        "--total-sd",
        type=float,
        default=base.DEFAULT_TOTAL_SD,
        help=(
            "Total forecast error SD used for "
            "over/under probabilities."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build and validate projections "
            "but do not write output."
        ),
    )

    return parser.parse_args()


def load_current_week_config(
    cfb_root: Path,
) -> tuple[int, int]:
    config_path = (
        cfb_root
        / "config"
        / "current_week.yaml"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            "Missing current-week config: "
            f"{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = yaml.safe_load(
            handle
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "current_week.yaml must contain a mapping"
        )

    values: dict[
        str,
        int,
    ] = {}

    for key in (
        "season",
        "week",
    ):
        value = payload.get(
            key
        )

        if isinstance(
            value,
            bool,
        ):
            raise ValueError(
                f"current_week.{key} must be an integer"
            )

        try:
            number = float(
                str(
                    value
                ).strip()
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"current_week.{key} must be an integer; "
                f"found {value!r}"
            ) from exc

        if (
            not math.isfinite(
                number
            )
            or not number.is_integer()
            or number <= 0
        ):
            raise ValueError(
                f"current_week.{key} must be a positive integer; "
                f"found {value!r}"
            )

        values[
            key
        ] = int(
            number
        )

    if values[
        "season"
    ] < 2000:
        raise ValueError(
            "current_week.season is outside the supported range: "
            f"{values['season']}"
        )

    return (
        values[
            "season"
        ],
        values[
            "week"
        ],
    )


def resolve_target(
    cli_season: int | None,
    cli_week: int | None,
    cfb_root: Path,
) -> tuple[int, int]:
    config_season: int | None = None
    config_week: int | None = None

    if (
        cli_season is None
        or cli_week is None
    ):
        (
            config_season,
            config_week,
        ) = load_current_week_config(
            cfb_root
        )

    season = int(
        cli_season
        if cli_season is not None
        else config_season
    )

    week = int(
        cli_week
        if cli_week is not None
        else config_week
    )

    if season < 2000:
        raise ValueError(
            f"Invalid target season: {season}"
        )

    if week <= 0:
        raise ValueError(
            f"Invalid target week: {week}"
        )

    return (
        season,
        week,
    )



def _validate_projection_weight_args(
    args: argparse.Namespace,
    margin_weights: list[float],
) -> None:
    if any(
        not math.isfinite(
            float(
                weight
            )
        )
        or weight < 0
        for weight in margin_weights
    ):
        raise ValueError(
            "Margin component weights must be "
            "finite and non-negative"
        )

    if sum(
        margin_weights
    ) <= 0:
        raise ValueError(
            "At least one margin component "
            "weight must be positive"
        )

    if (
        not math.isfinite(
            float(
                args.market_total_weight
            )
        )
        or not (
            0.0
            <= args.market_total_weight
            <= 1.0
        )
    ):
        raise ValueError(
            "--market-total-weight must be "
            "between 0 and 1"
        )

    if (
        not math.isfinite(
            float(
                args.margin_sd
            )
        )
        or args.margin_sd <= 0
    ):
        raise ValueError(
            "--margin-sd must be a finite "
            "value greater than 0"
        )

    if (
        not math.isfinite(
            float(
                args.total_sd
            )
        )
        or args.total_sd <= 0
    ):
        raise ValueError(
            "--total-sd must be a finite "
            "value greater than 0"
        )

    if args.fresh_injury_days < 0:
        raise ValueError(
            "--fresh-injury-days must be non-negative"
        )


def validate_args(
    args: argparse.Namespace,
) -> None:
    if (
        args.week is not None
        and args.week <= 1
    ):
        raise ValueError(
            "projection.py is for Week 2+. "
            "Use projection_week1.py for Week 1."
        )

    if args.min_current_team_weeks < 1:
        raise ValueError(
            "--min-current-team-weeks must be at least 1"
        )

    if (
        not math.isfinite(
            float(
                args.home_field
            )
        )
    ):
        raise ValueError(
            "--home-field must be finite"
        )

    if (
        not math.isfinite(
            float(
                args.drives_per_team
            )
        )
        or args.drives_per_team <= 0
    ):
        raise ValueError(
            "--drives-per-team must be a finite "
            "value greater than 0"
        )

    margin_weights = [
        args.market_margin_weight,
        args.fpi_margin_weight,
        args.espn_margin_weight,
        args.prior_margin_weight,
    ]

    _validate_projection_weight_args(
        args,
        margin_weights,
    )


def normalized_game_ids(
    frame: pd.DataFrame,
    label: str,
) -> pd.Series:
    if "game_id" not in frame.columns:
        raise ValueError(
            f"{label} missing required column: game_id"
        )

    game_ids = frame[
        "game_id"
    ].map(
        base.normalize_game_id
    )

    if game_ids.eq(
        ""
    ).any():
        raise ValueError(
            f"{label} contains blank game_id values"
        )

    duplicate_mask = game_ids.duplicated(
        keep=False
    )

    if duplicate_mask.any():
        duplicates = (
            game_ids[
                duplicate_mask
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"{label} contains duplicate game_id values: "
            f"{duplicates[:10]}"
        )

    return game_ids


def validate_exact_game_coverage(
    schedule: pd.DataFrame,
    frame: pd.DataFrame,
    label: str,
) -> None:
    expected_ids = normalized_game_ids(
        schedule,
        "target schedule",
    )

    actual_ids = normalized_game_ids(
        frame,
        label,
    )

    expected_set = set(
        expected_ids
    )

    actual_set = set(
        actual_ids
    )

    missing = sorted(
        expected_set
        - actual_set
    )

    unexpected = sorted(
        actual_set
        - expected_set
    )

    if (
        missing
        or unexpected
    ):
        raise RuntimeError(
            f"{label} game_id coverage does not match "
            "target schedule; "
            f"missing_count={len(missing)} "
            f"unexpected_count={len(unexpected)} "
            f"missing_examples={missing[:10]} "
            f"unexpected_examples={unexpected[:10]}"
        )


def validate_espn_identity(
    schedule: pd.DataFrame,
    predictions: pd.DataFrame,
    resolver: base.TeamResolver,
) -> None:
    required_columns = [
        "game_id",
        "home_team",
        "away_team",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            "Finalized ESPN predictions missing identity columns: "
            f"{missing_columns}"
        )

    schedule_work = schedule.copy()

    schedule_work[
        "_normalized_game_id"
    ] = normalized_game_ids(
        schedule_work,
        "target schedule",
    )

    predictions_work = predictions.copy()

    predictions_work[
        "_normalized_game_id"
    ] = normalized_game_ids(
        predictions_work,
        "finalized ESPN predictions",
    )

    schedule_lookup = schedule_work.set_index(
        "_normalized_game_id",
        drop=False,
    )

    prediction_lookup = predictions_work.set_index(
        "_normalized_game_id",
        drop=False,
    )

    mismatches: list[
        dict[
            str,
            str,
        ]
    ] = []

    for game_id in schedule_lookup.index:
        schedule_row = schedule_lookup.loc[
            game_id
        ]

        prediction_row = prediction_lookup.loc[
            game_id
        ]

        expected_home = resolver.resolve(
            schedule_row.get(
                "home_team"
            )
        )

        expected_away = resolver.resolve(
            schedule_row.get(
                "away_team"
            )
        )

        actual_home = resolver.resolve(
            prediction_row.get(
                "home_team"
            )
        )

        actual_away = resolver.resolve(
            prediction_row.get(
                "away_team"
            )
        )

        if (
            actual_home != expected_home
            or actual_away != expected_away
        ):
            mismatches.append(
                {
                    "game_id":
                        game_id,
                    "expected_home":
                        expected_home,
                    "actual_home":
                        actual_home,
                    "expected_away":
                        expected_away,
                    "actual_away":
                        actual_away,
                }
            )

    if mismatches:
        raise RuntimeError(
            "Finalized ESPN prediction team identity "
            "does not match target schedule; "
            f"count={len(mismatches)} "
            f"examples={mismatches[:10]}"
        )


def validate_serialized_output(
    serialized: pd.DataFrame,
    projected: pd.DataFrame,
    schedule: pd.DataFrame,
    season: int,
    week: int,
) -> None:
    if list(
        serialized.columns
    ) != list(
        projected.columns
    ):
        raise RuntimeError(
            "Serialized projection schema changed during write"
        )

    required_output_columns = set(
        base.OUTPUT_BASE_COLUMNS
        + base.REQUIRED_PREDICTION_COLUMNS
    )

    missing_columns = sorted(
        required_output_columns
        - set(
            serialized.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Serialized projection is missing required columns: "
            f"{missing_columns}"
        )

    validate_output(
        serialized,
        schedule,
        season,
        week,
    )

    required_numeric = [
        "predicted_margin",
        "predicted_total",
        "predicted_home_score",
        "predicted_away_score",
    ]

    for column in required_numeric:
        values = pd.to_numeric(
            serialized[
                column
            ],
            errors="coerce",
        )

        if values.isna().any():
            raise RuntimeError(
                "Serialized projection contains blank "
                f"or non-numeric {column}"
            )

        finite = values.map(
            lambda value:
                math.isfinite(
                    float(
                        value
                    )
                )
        )

        if not finite.all():
            raise RuntimeError(
                "Serialized projection contains non-finite "
                f"{column}"
            )


def publish_output_atomic(
    projected: pd.DataFrame,
    schedule: pd.DataFrame,
    output_path: Path,
    season: int,
    week: int,
) -> bool:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f".{output_path.name}."
        f"{uuid.uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            projected.to_csv(
                handle,
                index=False,
                lineterminator="\n",
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        serialized = pd.read_csv(
            temporary_path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
            low_memory=False,
        )

        validate_serialized_output(
            serialized,
            projected,
            schedule,
            season,
            week,
        )

        new_bytes = (
            temporary_path.read_bytes()
        )

        if (
            output_path.is_file()
            and output_path.read_bytes()
            == new_bytes
        ):
            return False

        os.replace(
            temporary_path,
            output_path,
        )

        return True

    finally:
        temporary_path.unlink(
            missing_ok=True
        )



def load_target_schedule(
    path: Path,
    season: int,
    week: int,
) -> pd.DataFrame:
    schedule = base.read_csv(
        path,
        base.OUTPUT_BASE_COLUMNS,
        "weekly schedule",
    )

    season_num = pd.to_numeric(
        schedule[
            "season"
        ],
        errors="coerce",
    )

    week_num = pd.to_numeric(
        schedule[
            "week"
        ],
        errors="coerce",
    )

    schedule = schedule[
        season_num.eq(
            season
        )
        & week_num.eq(
            week
        )
    ].copy()

    if schedule.empty:
        raise ValueError(
            "No schedule rows for "
            f"season={season}, "
            f"week={week} "
            f"in {path}"
        )

    game_ids = schedule[
        "game_id"
    ].map(
        base.clean
    )

    if game_ids.eq(
        ""
    ).any():
        raise ValueError(
            f"{path}: blank game_id found"
        )

    duplicates = (
        game_ids[
            game_ids.duplicated(
                keep=False
            )
        ]
        .drop_duplicates()
        .tolist()
    )

    if duplicates:
        raise ValueError(
            f"{path}: duplicate game_id values: "
            f"{duplicates[:10]}"
        )

    return schedule


def load_current_team_stats(
    path: Path,
    season: int,
    target_week: int,
) -> tuple[
    pd.DataFrame,
    int,
]:
    team_stats = base.read_csv(
        path,
        [
            "season",
            "week",
            "team",
            *base.TEAM_METRICS,
        ],
        "current-season team stats",
    )

    season_num = pd.to_numeric(
        team_stats[
            "season"
        ],
        errors="coerce",
    )

    week_num = pd.to_numeric(
        team_stats[
            "week"
        ],
        errors="coerce",
    )

    usable = team_stats[
        season_num.eq(
            season
        )
        & week_num.notna()
        & week_num.lt(
            target_week
        )
    ].copy()

    if usable.empty:
        raise ValueError(
            "No completed current-season team-stat "
            "rows are available before target "
            f"week {target_week} in {path}"
        )

    usable_week_num = pd.to_numeric(
        usable[
            "week"
        ],
        errors="coerce",
    )

    latest_completed_week = int(
        usable_week_num.max()
    )

    expected_latest_week = (
        target_week
        - 1
    )

    if (
        latest_completed_week
        != expected_latest_week
    ):
        raise RuntimeError(
            "Current-season team stats are not "
            "caught up to the projection target. "
            f"target_week={target_week}, "
            "expected_latest_completed_week="
            f"{expected_latest_week}, "
            "latest_team_stats_week="
            f"{latest_completed_week}, "
            f"path={path}"
        )

    return (
        usable,
        latest_completed_week,
    )


def target_season_type(
    schedule: pd.DataFrame,
    season: int,
    week: int,
) -> int:
    if "season_type" not in schedule.columns:
        raise ValueError(
            "Target weekly schedule is missing season_type"
        )

    values = pd.to_numeric(
        schedule["season_type"],
        errors="coerce",
    )

    invalid = (
        values.isna()
        | values.mod(1).ne(0)
        | values.le(0)
    )

    if invalid.any():
        examples = (
            schedule.loc[
                invalid,
                "season_type",
            ]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Target weekly schedule contains invalid "
            f"season_type values: {examples}"
        )

    observed = sorted(
        {
            int(value)
            for value
            in values.tolist()
        }
    )

    if len(observed) != 1:
        raise RuntimeError(
            "Target weekly schedule contains multiple "
            f"season_type values for season={season}, "
            f"week={week}: {observed}"
        )

    return observed[0]


def historical_result_paths(
    results_dir: Path,
    season: int,
    season_type: int,
    target_week: int,
) -> list[Path]:
    return [
        (
            results_dir
            / f"{season}_{season_type}_{prior_week}.csv"
        )
        for prior_week
        in range(
            1,
            target_week,
        )
    ]


def _history_integer_column(
    frame: pd.DataFrame,
    column: str,
    label: str,
) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(
            f"{label} missing required column: {column}"
        )

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    )

    invalid = (
        values.isna()
        | values.mod(1).ne(0)
    )

    if invalid.any():
        examples = (
            frame.loc[
                invalid,
                column,
            ]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{label} contains invalid integer "
            f"{column} values: {examples}"
        )

    return values.astype(int)


def _history_completed_flag(
    value: object,
    label: str,
) -> bool:
    text = base.clean(
        value
    ).casefold()

    if text in {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
    }:
        return True

    if text in {
        "0",
        "0.0",
        "false",
        "no",
        "n",
    }:
        return False

    raise ValueError(
        f"{label} must be a valid completed 0/1 flag; "
        f"found {value!r}"
    )


def _stage1_projection_completed_results(
    result_paths: list[Path],
    *,
    season: int,
    season_type: int,
) -> dict[str, int]:
    expected_completed: dict[str, int] = {}
    for expected_week, result_path in enumerate(result_paths, start=1):
        results = base.read_csv(
            result_path,
            ["season", "season_type", "week", "game_id", "completed"],
            f"final-score results for week {expected_week}",
        )
        result_season = _history_integer_column(results, "season", str(result_path))
        result_type = _history_integer_column(results, "season_type", str(result_path))
        result_week = _history_integer_column(results, "week", str(result_path))
        if not result_season.eq(season).all():
            raise RuntimeError(f"{result_path}: wrong season")
        if not result_type.eq(season_type).all():
            raise RuntimeError(f"{result_path}: wrong season_type")
        if not result_week.eq(expected_week).all():
            raise RuntimeError(f"{result_path}: wrong week")

        result_ids = results["game_id"].map(base.normalize_game_id)
        if result_ids.eq("").any():
            raise RuntimeError(f"{result_path}: blank game_id")
        if result_ids.duplicated().any():
            duplicates = (
                result_ids[result_ids.duplicated(False)]
                .drop_duplicates().head(10).tolist()
            )
            raise RuntimeError(
                f"{result_path}: duplicate game_id values: {duplicates}"
            )

        for position, (game_id, completed_raw) in enumerate(
            zip(result_ids, results["completed"], strict=True),
            start=2,
        ):
            completed = _history_completed_flag(
                completed_raw,
                f"{result_path} line {position}: completed",
            )
            if not completed:
                continue
            if game_id in expected_completed:
                raise RuntimeError(
                    "Completed game_id appears in multiple prior result weeks: "
                    f"{game_id}"
                )
            expected_completed[game_id] = expected_week
    return expected_completed


def _stage1_projection_pbp_coverage(
    pbp_path: Path,
    *,
    season: int,
    target_week: int,
) -> tuple[set[str], set[tuple[int, str]]]:
    if not pbp_path.is_file():
        raise FileNotFoundError(f"Missing current-season PBP: {pbp_path}")
    try:
        pbp = pd.read_parquet(
            pbp_path,
            columns=["season", "week", "game_id", "homeTeamId", "awayTeamId"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read PBP coverage input {pbp_path}: {exc}"
        ) from exc
    if pbp.empty:
        raise RuntimeError(f"PBP coverage input is empty: {pbp_path}")

    pbp_season = _history_integer_column(pbp, "season", str(pbp_path))
    pbp_week = _history_integer_column(pbp, "week", str(pbp_path))
    if not pbp_season.eq(season).all():
        observed = sorted(pbp_season.unique().tolist())
        raise RuntimeError(f"{pbp_path}: PBP season mismatch: {observed}")
    if pbp_week.le(0).any():
        raise RuntimeError(f"{pbp_path}: PBP contains non-positive week")

    prior_pbp = pbp.loc[pbp_week.lt(target_week)].copy()
    prior_pbp["_history_week"] = pbp_week.loc[prior_pbp.index].astype(int)
    prior_pbp["_history_game_id"] = prior_pbp["game_id"].map(base.normalize_game_id)
    if prior_pbp["_history_game_id"].eq("").any():
        raise RuntimeError(f"{pbp_path}: prior PBP contains blank game_id")

    pbp_game_ids: set[str] = set()
    expected_pairs: set[tuple[int, str]] = set()
    for game_id, group in prior_pbp.groupby("_history_game_id", sort=False):
        weeks = {int(value) for value in group["_history_week"].tolist()}
        if len(weeks) != 1:
            raise RuntimeError(
                "PBP game appears in multiple weeks: "
                f"game_id={game_id}, weeks={sorted(weeks)}"
            )
        home_ids = {
            base.normalize_game_id(value)
            for value in group["homeTeamId"]
            if base.normalize_game_id(value)
        }
        away_ids = {
            base.normalize_game_id(value)
            for value in group["awayTeamId"]
            if base.normalize_game_id(value)
        }
        if len(home_ids) != 1 or len(away_ids) != 1:
            raise RuntimeError(
                "PBP does not contain one stable home/away team identity for "
                f"game_id={game_id}; home_ids={sorted(home_ids)}, "
                f"away_ids={sorted(away_ids)}"
            )
        week_value = next(iter(weeks))
        home_id = next(iter(home_ids))
        away_id = next(iter(away_ids))
        if home_id == away_id:
            raise RuntimeError(
                "PBP home/away team IDs are identical "
                f"for game_id={game_id}: {home_id}"
            )
        pbp_game_ids.add(game_id)
        expected_pairs.add((week_value, home_id))
        expected_pairs.add((week_value, away_id))
    return pbp_game_ids, expected_pairs


def _stage1_projection_validate_team_stats(
    team_stats: pd.DataFrame,
    expected_pairs: set[tuple[int, str]],
) -> set[tuple[int, str]]:
    stat_weeks = _history_integer_column(
        team_stats, "week", "current-season team stats"
    )
    stat_team_ids = team_stats["team"].map(base.normalize_game_id)
    if stat_team_ids.eq("").any():
        raise RuntimeError("Current-season team stats contain blank team IDs")
    actual_pair_rows = list(
        zip(stat_weeks.tolist(), stat_team_ids.tolist(), strict=True)
    )
    if len(actual_pair_rows) != len(set(actual_pair_rows)):
        duplicates = (
            pd.DataFrame(actual_pair_rows, columns=["week", "team"])
            .loc[lambda frame: frame.duplicated(["week", "team"], keep=False)]
            .drop_duplicates().head(10).to_dict(orient="records")
        )
        raise RuntimeError(
            "Current-season team stats contain duplicate (week, team) rows: "
            f"{duplicates}"
        )
    actual_pairs = set(actual_pair_rows)
    missing = sorted(expected_pairs - actual_pairs)
    unexpected = sorted(actual_pairs - expected_pairs)
    if missing or unexpected:
        raise RuntimeError(
            "Current-season team-stat coverage does not exactly match "
            "completed PBP team/week coverage. "
            f"expected_rows={len(expected_pairs)} actual_rows={len(actual_pairs)} "
            f"missing_count={len(missing)} unexpected_count={len(unexpected)} "
            f"missing_examples={missing[:10]} unexpected_examples={unexpected[:10]}"
        )
    return actual_pairs


def validate_completed_history_coverage(
    *,
    pbp_path: Path,
    result_paths: list[Path],
    team_stats: pd.DataFrame,
    season: int,
    season_type: int,
    target_week: int,
) -> dict[str, object]:
    expected_completed = _stage1_projection_completed_results(
        result_paths,
        season=season,
        season_type=season_type,
    )
    if not expected_completed:
        raise RuntimeError(
            "No completed games were found in prior-week final-score results"
        )

    pbp_game_ids, expected_pairs = _stage1_projection_pbp_coverage(
        pbp_path,
        season=season,
        target_week=target_week,
    )
    expected_game_ids = set(expected_completed)
    missing_pbp = sorted(expected_game_ids - pbp_game_ids)
    unexpected_pbp = sorted(pbp_game_ids - expected_game_ids)
    if missing_pbp or unexpected_pbp:
        raise RuntimeError(
            "Completed-game PBP coverage mismatch. "
            f"expected_completed_games={len(expected_game_ids)} "
            f"pbp_games={len(pbp_game_ids)} "
            f"missing_count={len(missing_pbp)} "
            f"unexpected_count={len(unexpected_pbp)} "
            f"missing_examples={missing_pbp[:10]} "
            f"unexpected_examples={unexpected_pbp[:10]}"
        )

    actual_pairs = _stage1_projection_validate_team_stats(
        team_stats,
        expected_pairs,
    )
    return {
        "completed_result_games": len(expected_game_ids),
        "pbp_prior_games": len(pbp_game_ids),
        "expected_team_week_rows": len(expected_pairs),
        "team_stats_team_week_rows": len(actual_pairs),
        "result_files": [str(path) for path in result_paths],
        "pbp_path": str(pbp_path),
    }



def validate_output(
    projected: pd.DataFrame,
    schedule: pd.DataFrame,
    season: int,
    week: int,
) -> None:
    if len(
        projected
    ) != len(
        schedule
    ):
        raise RuntimeError(
            "Projection row count does not match "
            "target schedule row count"
        )

    projected_season = pd.to_numeric(
        projected[
            "season"
        ],
        errors="coerce",
    )

    projected_week = pd.to_numeric(
        projected[
            "week"
        ],
        errors="coerce",
    )

    if not projected_season.eq(
        season
    ).all():
        raise RuntimeError(
            "Projection output contains "
            "an unexpected season"
        )

    if not projected_week.eq(
        week
    ).all():
        raise RuntimeError(
            "Projection output contains "
            "an unexpected week"
        )

    expected_ids = schedule[
        "game_id"
    ].map(
        base.clean
    ).tolist()

    actual_ids = projected[
        "game_id"
    ].map(
        base.clean
    ).tolist()

    if (
        actual_ids
        != expected_ids
    ):
        raise RuntimeError(
            "Projection output game_id "
            "order changed"
        )

    if projected[
        "game_id"
    ].map(
        base.clean
    ).duplicated().any():
        raise RuntimeError(
            "Projection output contains "
            "duplicate game_id values"
        )

    base.validate_probability_output(
        projected
    )


def run(
    report: PipelineReporter,
    args: argparse.Namespace,
    season: int,
    week: int,
) -> int:
    cfb_root = base.repo_cfb_root()

    schedule_path = (
        cfb_root
        / "00_intake"
        / "schedule"
        / "weekly"
        / (
            f"week_{week}_"
            "CFB_weekly_schedule.csv"
        )
    )

    if week <= 1:
        raise ValueError(
            "projection.py is for Week 2+. "
            "Use projection_week1.py for Week 1."
        )

    output_path = (
        cfb_root
        / "01_merge"
        / f"week_{week}_CFB_enriched.csv"
    )

    team_stats_path = (
        cfb_root
        / "00_intake"
        / "team_stats"
        / f"{season}_team_stats.csv"
    )

    pbp_path = (
        cfb_root
        / "00_intake"
        / "pbp"
        / f"{season}_pbp.parquet"
    )

    results_dir = (
        cfb_root
        / "04_final_results"
        / "results"
    )

    team_map_path = (
        cfb_root
        / "config"
        / "mapping"
        / "team_map.csv"
    )

    stadium_map_path = (
        cfb_root
        / "config"
        / "mapping"
        / "stadium_map.csv"
    )

    fpi_path = (
        cfb_root
        / "data"
        / "team_power_index"
        / (
            f"team_power_index_"
            f"{season}.csv"
        )
    )

    predictions_dir = (
        cfb_root
        / "00_intake"
        / "predictions"
        / "final"
    )

    injuries_path = (
        cfb_root
        / "00_intake"
        / "injuries"
        / f"{season}_injuries.csv"
    )

    travel_path = (
        cfb_root
        / "data"
        / "travel"
        / f"{season}_week_{week}_travel.csv"
    )

    weather_path = (
        cfb_root
        / "data"
        / "weather"
        / f"week_{week}_CFB_weekly_weather.csv"
    )

    travel_weather_coefficients_path = (
        cfb_root
        / "config"
        / "travel_weather_coefficients.csv"
    )

    report.season = season
    report.week = week

    for input_path in (
        schedule_path,
        pbp_path,
        team_stats_path,
        team_map_path,
        stadium_map_path,
        fpi_path,
        predictions_dir,
        injuries_path,
        travel_path,
        weather_path,
        travel_weather_coefficients_path,
    ):
        report.add_input(
            input_path
        )

    report.add_output(
        output_path
    )

    report.update_details(
        {
            "dry_run": bool(
                args.dry_run
            ),
            "fresh_injury_days": int(
                args.fresh_injury_days
            ),
            "min_current_team_weeks": int(
                args.min_current_team_weeks
            ),
            "output_modified": False,
        }
    )

    schedule = load_target_schedule(
        schedule_path,
        season,
        week,
    )

    report.set_rows(
        rows_in=len(
            schedule
        ),
    )

    report.set_detail(
        "schedule_game_count",
        len(
            schedule
        ),
    )

    season_type = target_season_type(
        schedule,
        season,
        week,
    )

    result_paths = historical_result_paths(
        results_dir,
        season,
        season_type,
        week,
    )

    for result_path in result_paths:
        report.add_input(
            result_path
        )

    report.set_detail(
        "season_type",
        season_type,
    )

    team_map = base.read_csv(
        team_map_path,
        [
            "team_id",
            "canonical_team",
        ],
        "team map",
    )

    resolver = base.TeamResolver(
        team_map
    )

    (
        current_team_stats,
        latest_completed_week,
    ) = load_current_team_stats(
        team_stats_path,
        season,
        week,
    )

    history_coverage = (
        validate_completed_history_coverage(
            pbp_path=pbp_path,
            result_paths=result_paths,
            team_stats=current_team_stats,
            season=season,
            season_type=season_type,
            target_week=week,
        )
    )

    report.set_detail(
        "completed_history_coverage",
        history_coverage,
    )

    current_prior = base.build_prior_table(
        current_team_stats,
        resolver,
    )

    fpi = base.load_fpi(
        fpi_path,
        resolver,
    )

    current_prior = base.scale_prior_to_fpi(
        current_prior,
        fpi,
    )

    espn_predictions = (
        base.load_espn_predictions(
            predictions_dir,
            season,
            week,
            resolver,
        )
    )

    home_stadium_lookup = (
        base.build_home_stadium_lookup(
            stadium_map_path,
            resolver,
        )
    )

    injury_lookup = (
        base.build_injury_lookup(
            injuries_path,
            resolver,
        )
    )

    travel = base.load_game_feature_file(
        travel_path,
        base.TRAVEL_REQUIRED_COLUMNS,
        "weekly travel",
    )

    weather = base.load_game_feature_file(
        weather_path,
        base.WEATHER_REQUIRED_COLUMNS,
        "weekly weather",
    )

    validate_exact_game_coverage(
        schedule,
        espn_predictions,
        "finalized ESPN predictions",
    )

    validate_espn_identity(
        schedule,
        espn_predictions,
        resolver,
    )

    validate_exact_game_coverage(
        schedule,
        travel,
        "weekly travel",
    )

    validate_exact_game_coverage(
        schedule,
        weather,
        "weekly weather",
    )

    travel_weather_coefficients = (
        base.load_travel_weather_coefficients(
            travel_weather_coefficients_path
        )
    )

    selected_margin_features = sorted(
        travel_weather_coefficients.get(
            "margin",
            {},
        )
    )

    selected_total_features = sorted(
        travel_weather_coefficients.get(
            "total",
            {},
        )
    )

    selected_coefficient_count = (
        len(
            selected_margin_features
        )
        + len(
            selected_total_features
        )
    )

    # projection_week1.py uses these module globals when determining
    # whether the team-strength component is reliable and when writing
    # the projection-version audit field.
    base.MIN_PRIOR_TEAM_WEEKS = int(
        args.min_current_team_weeks
    )

    base.SCRIPT_VERSION = (
        SCRIPT_VERSION
    )

    projected = base.build_projection(
        schedule,
        current_prior,
        fpi,
        espn_predictions,
        resolver,
        home_stadium_lookup,
        injury_lookup,
        travel,
        weather,
        travel_weather_coefficients,
        args,
    )

    (
        projected,
        locked_games_preserved,
    ) = base.preserve_locked_rows(
        projected,
        schedule,
        output_path,
        "Week 2+ projection",
    )

    validate_output(
        projected,
        schedule,
        season,
        week,
    )

    current_week_counts = pd.to_numeric(
        current_prior[
            "prior_team_weeks"
        ],
        errors="coerce",
    )

    market_spread_count = int(
        pd.to_numeric(
            projected[
                "market_home_margin"
            ],
            errors="coerce",
        ).notna().sum()
    )

    fpi_component_count = int(
        pd.to_numeric(
            projected[
                "fpi_home_margin"
            ],
            errors="coerce",
        ).notna().sum()
    )

    espn_component_count = int(
        pd.to_numeric(
            projected[
                "espn_home_margin"
            ],
            errors="coerce",
        ).notna().sum()
    )

    team_stats_component_count = int(
        pd.to_numeric(
            projected[
                "prior_home_margin"
            ],
            errors="coerce",
        ).notna().sum()
    )

    market_total_count = int(
        pd.to_numeric(
            projected[
                "market_total"
            ],
            errors="coerce",
        ).notna().sum()
    )

    injury_adjustment_count = int(
        pd.to_numeric(
            projected[
                "injury_margin_adjustment"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .abs()
        .gt(
            0
        )
        .sum()
    )

    travel_adjustment_count = int(
        pd.to_numeric(
            projected[
                "travel_margin_adjustment"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .abs()
        .gt(
            0
        )
        .sum()
    )

    weather_adjustment_count = int(
        pd.to_numeric(
            projected[
                "weather_total_adjustment"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .abs()
        .gt(
            0
        )
        .sum()
    )

    injury_input_rows = sum(
        len(
            group
        )
        for group
        in injury_lookup.values()
    )

    report.set_rows(
        rows_out=len(
            projected
        ),
    )

    report.update_details(
        {
            "projection_game_count": len(
                projected
            ),
            "locked_games_preserved": (
                locked_games_preserved
            ),
            "team_stats_source_rows": len(
                current_team_stats
            ),
            "latest_completed_team_stats_week": (
                latest_completed_week
            ),
            "teams_with_current_stats": len(
                current_prior
            ),
            "fpi_team_count": len(
                fpi
            ),
            "espn_prediction_game_count": len(
                espn_predictions
            ),
            "injury_lookup_team_count": len(
                injury_lookup
            ),
            "injury_input_rows": (
                injury_input_rows
            ),
            "travel_input_rows": len(
                travel
            ),
            "weather_input_rows": len(
                weather
            ),
            "travel_weather_coefficient_count": (
                selected_coefficient_count
            ),
            "travel_margin_features_selected": (
                selected_margin_features
            ),
            "weather_total_features_selected": (
                selected_total_features
            ),
            "games_with_market_spread": (
                market_spread_count
            ),
            "games_with_fpi_component": (
                fpi_component_count
            ),
            "games_with_espn_component": (
                espn_component_count
            ),
            "games_with_current_team_stats_component": (
                team_stats_component_count
            ),
            "games_with_market_total": (
                market_total_count
            ),
            "games_with_fresh_injury_adjustment": (
                injury_adjustment_count
            ),
            "games_with_travel_adjustment": (
                travel_adjustment_count
            ),
            "games_with_weather_adjustment": (
                weather_adjustment_count
            ),
            "home_team_stats_fallbacks": int(
                pd.to_numeric(
                    projected[
                        "home_prior_fallback"
                    ],
                    errors="coerce",
                )
                .fillna(
                    0
                )
                .sum()
            ),
            "away_team_stats_fallbacks": int(
                pd.to_numeric(
                    projected[
                        "away_prior_fallback"
                    ],
                    errors="coerce",
                )
                .fillna(
                    0
                )
                .sum()
            ),
            "team_stats_margin_disabled": int(
                pd.to_numeric(
                    projected[
                        "prior_home_margin"
                    ],
                    errors="coerce",
                )
                .isna()
                .sum()
            ),
            "probability_margin_sd": float(
                args.margin_sd
            ),
            "probability_total_sd": float(
                args.total_sd
            ),
            "output_path": str(
                output_path
            ),
        }
    )

    print(
        f"projection.py version="
        f"{SCRIPT_VERSION}"
    )

    print(
        f"season={season}"
    )

    print(
        f"week={week}"
    )

    print(
        f"schedule={schedule_path}"
    )

    print(
        f"games={len(projected)}"
    )

    print(
        "locked_games_preserved="
        f"{locked_games_preserved}"
    )

    print(
        "team_stats_source="
        f"{team_stats_path}"
    )

    print(
        "latest_completed_team_stats_week="
        f"{latest_completed_week}"
    )

    print(
        "min_current_team_weeks="
        f"{args.min_current_team_weeks}"
    )

    print(
        "teams_with_current_stats="
        f"{len(current_prior)}"
    )

    if not current_week_counts.empty:
        print(
            "current_team_weeks_min="
            f"{int(current_week_counts.min())}"
        )

        print(
            "current_team_weeks_max="
            f"{int(current_week_counts.max())}"
        )

    print(
        "home_team_stats_fallbacks="
        f"{int(pd.to_numeric(projected['home_prior_fallback'], errors='coerce').fillna(0).sum())}"
    )

    print(
        "away_team_stats_fallbacks="
        f"{int(pd.to_numeric(projected['away_prior_fallback'], errors='coerce').fillna(0).sum())}"
    )

    print(
        "team_stats_margin_disabled="
        f"{int(pd.to_numeric(projected['prior_home_margin'], errors='coerce').isna().sum())}"
    )

    print(
        "with_market_spread="
        f"{int(pd.to_numeric(projected['market_home_margin'], errors='coerce').notna().sum())}"
    )

    print(
        "with_fpi="
        f"{int(pd.to_numeric(projected['fpi_home_margin'], errors='coerce').notna().sum())}"
    )

    print(
        "with_espn="
        f"{int(pd.to_numeric(projected['espn_home_margin'], errors='coerce').notna().sum())}"
    )

    print(
        "with_current_team_stats_margin="
        f"{int(pd.to_numeric(projected['prior_home_margin'], errors='coerce').notna().sum())}"
    )

    print(
        "with_market_total="
        f"{int(pd.to_numeric(projected['market_total'], errors='coerce').notna().sum())}"
    )

    print(
        "fresh_injury_adjustments="
        f"{int(pd.to_numeric(projected['injury_margin_adjustment'], errors='coerce').fillna(0).abs().gt(0).sum())}"
    )

    print(
        "travel_adjustments="
        f"{int(pd.to_numeric(projected['travel_margin_adjustment'], errors='coerce').fillna(0).abs().gt(0).sum())}"
    )

    print(
        "weather_adjustments="
        f"{int(pd.to_numeric(projected['weather_total_adjustment'], errors='coerce').fillna(0).abs().gt(0).sum())}"
    )

    print(
        "probability_margin_sd="
        f"{float(args.margin_sd):g}"
    )

    print(
        "probability_total_sd="
        f"{float(args.total_sd):g}"
    )

    if args.dry_run:
        print(
            "output_modified=no"
        )

        print(
            "status=dry_run_success"
        )

        return 0

    output_modified = publish_output_atomic(
        projected,
        schedule,
        output_path,
        season,
        week,
    )

    report.set_detail(
        "output_modified",
        output_modified,
    )

    print(
        f"output={output_path}"
    )

    print(
        "output_modified="
        f"{'yes' if output_modified else 'no'}"
    )

    print(
        "status=success"
    )

    return 0


def main() -> int:
    args = parse_args()

    with PipelineReporter(
        script=__file__,
        stage="01_merge",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        season=(
            args.season
            if args.season is not None
            else None
        ),
        extra_context={
            "script_version": (
                SCRIPT_VERSION
            ),
            "projection_scope": (
                "week_2_plus"
            ),
        },
    ) as report:
        validate_args(
            args
        )

        (
            season,
            week,
        ) = resolve_target(
            args.season,
            args.week,
            CFB_ROOT,
        )

        report.season = season
        report.week = week

        return run(
            report,
            args,
            season,
            week,
        )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )

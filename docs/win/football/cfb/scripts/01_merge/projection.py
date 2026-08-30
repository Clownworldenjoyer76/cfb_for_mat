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
If --week is omitted, this script uses the most recently written
week_*_CFB_weekly_schedule.csv. That matches the target week selected by the
current odds/schedule intake pipeline.

Safety checks
-------------
- Week 1 is rejected. Use projection_week1.py for Week 1.
- Only current-season team-stat rows with source week < target week are used.
- The latest completed team-stat week must equal target_week - 1.
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
import re
import sys
from pathlib import Path

import pandas as pd

import projection_week1 as base


SCRIPT_VERSION = "cfb-inseason-v3-game-lock-2026-08-30"

WEEKLY_FILE_RE = re.compile(
    r"^week_(\d+)_CFB_weekly_schedule\.csv$"
)


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
        help="Target season. Defaults to CFB_SEASON, then 2026.",
    )

    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help=(
            "Target week. If omitted, use the most recently "
            "written week_*_CFB_weekly_schedule.csv."
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


def resolve_season(
    cli_season: int | None,
) -> int:
    if cli_season is not None:
        return int(
            cli_season
        )

    env_season = os.getenv(
        "CFB_SEASON",
        "",
    ).strip()

    if env_season:
        try:
            return int(
                env_season
            )

        except ValueError as exc:
            raise ValueError(
                "CFB_SEASON must be an integer; "
                f"found {env_season!r}"
            ) from exc

    return 2026


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


def infer_target_week(
    cfb_root: Path,
    season: int,
) -> tuple[int, Path]:
    weekly_dir = (
        cfb_root
        / "00_intake"
        / "schedule"
        / "weekly"
    )

    if not weekly_dir.is_dir():
        raise FileNotFoundError(
            "Weekly schedule directory not found: "
            f"{weekly_dir}"
        )

    candidates: list[
        tuple[
            float,
            int,
            Path,
        ]
    ] = []

    for path in weekly_dir.glob(
        "week_*_CFB_weekly_schedule.csv"
    ):
        match = WEEKLY_FILE_RE.fullmatch(
            path.name
        )

        if match is None:
            continue

        filename_week = int(
            match.group(
                1
            )
        )

        try:
            frame = pd.read_csv(
                path,
                dtype=str,
                encoding="utf-8-sig",
                low_memory=False,
                usecols=[
                    "season",
                    "week",
                ],
            )

        except ValueError:
            continue

        if frame.empty:
            continue

        season_num = pd.to_numeric(
            frame[
                "season"
            ],
            errors="coerce",
        )

        week_num = pd.to_numeric(
            frame[
                "week"
            ],
            errors="coerce",
        )

        matching = frame[
            season_num.eq(
                season
            )
            & week_num.eq(
                filename_week
            )
        ]

        if matching.empty:
            continue

        candidates.append(
            (
                path.stat().st_mtime,
                filename_week,
                path,
            )
        )

    if not candidates:
        raise FileNotFoundError(
            "No usable CFB weekly schedule found "
            f"for season={season} in {weekly_dir}"
        )

    (
        _,
        week,
        path,
    ) = max(
        candidates,
        key=lambda item: (
            item[
                0
            ],
            item[
                1
            ],
        ),
    )

    return (
        week,
        path,
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


def main() -> int:
    args = parse_args()

    validate_args(
        args
    )

    season = resolve_season(
        args.season
    )

    cfb_root = base.repo_cfb_root()

    if args.week is None:
        (
            week,
            schedule_path,
        ) = infer_target_week(
            cfb_root,
            season,
        )

    else:
        week = int(
            args.week
        )

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

    schedule = load_target_schedule(
        schedule_path,
        season,
        week,
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
            args.fresh_injury_days,
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

    travel_weather_coefficients = (
        base.load_travel_weather_coefficients(
            travel_weather_coefficients_path
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
        f"{int(projected['home_prior_fallback'].sum())}"
    )

    print(
        "away_team_stats_fallbacks="
        f"{int(projected['away_prior_fallback'].sum())}"
    )

    print(
        "team_stats_margin_disabled="
        f"{int(projected['prior_home_margin'].isna().sum())}"
    )

    print(
        "with_market_spread="
        f"{int(projected['market_home_margin'].notna().sum())}"
    )

    print(
        "with_fpi="
        f"{int(projected['fpi_home_margin'].notna().sum())}"
    )

    print(
        "with_espn="
        f"{int(projected['espn_home_margin'].notna().sum())}"
    )

    print(
        "with_current_team_stats_margin="
        f"{int(projected['prior_home_margin'].notna().sum())}"
    )

    print(
        "with_market_total="
        f"{int(projected['market_total'].notna().sum())}"
    )

    print(
        "fresh_injury_adjustments="
        f"{int(projected['injury_margin_adjustment'].abs().gt(0).sum())}"
    )

    print(
        "travel_adjustments="
        f"{int(projected['travel_margin_adjustment'].abs().gt(0).sum())}"
    )

    print(
        "weather_adjustments="
        f"{int(projected['weather_total_adjustment'].abs().gt(0).sum())}"
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix
        + ".tmp"
    )

    projected.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        output_path,
    )

    print(
        f"output={output_path}"
    )

    print(
        "status=success"
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
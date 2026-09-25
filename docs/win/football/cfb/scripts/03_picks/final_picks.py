#!/usr/bin/env python3
"""
Build final CFB selected-picks output.

READS:
  docs/win/football/cfb/config/current_week.yaml
  docs/win/football/cfb/03_picks/week_{week}_CFB_picks.csv
  docs/win/football/cfb/00_intake/schedule/weekly/
      week_{week}_CFB_weekly_schedule.csv

WRITES:
  docs/win/football/cfb/03_picks/selected/
      week_{week}_CFB_select_picks.csv

  docs/win/football/cfb/03_picks/selected/locked/
      week_{week}_CFB_select_picks.csv

A row is included when at least one validated selection flag is true.
A header-only output is valid when the week has zero selected wagers.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Never

import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
CFB_ROOT = SCRIPT_DIR.parents[1]
REPORT_ROOT = CFB_ROOT / "errors"

INPUT_DIR = CFB_ROOT / "03_picks"
OUTPUT_DIR = INPUT_DIR / "selected"
LOCKED_DIR = OUTPUT_DIR / "locked"

CURRENT_WEEK_CONFIG = (
    CFB_ROOT
    / "config"
    / "current_week.yaml"
)

SCRIPT_VERSION = (
    "cfb-final-picks-v2-hardened-2026-09-16"
)

PICKS_FILE_RE = re.compile(
    r"^week_(\d+)_CFB_picks\.csv$",
    flags=re.IGNORECASE,
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPTS_DIR),
    )

from pipeline_reporter import PipelineReporter
from pipeline_shared import (
    clean_text as clean,
    stage_dataframe_csv,
    validate_game_ids as validate_ids,
    weekly_schedule_path,
)


OUTPUT_COLUMNS = [
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "ml_selection",
    "ml_odds_american",
    "ml_model_probability",
    "spread_selection",
    "spread_line",
    "spread_odds_american",
    "total_selection",
    "total_line",
    "total_odds_american",
    "total_model_probability",
    "season",
    "season_type",
    "ml_selected",
    "spread_selected",
    "total_selected",
]

REQUIRED_INPUT_COLUMNS = [
    *OUTPUT_COLUMNS,
    "spread_model_probability",
]

SCHEDULE_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "away_team",
    "home_team",
]


def fail(
    message: str,
) -> Never:
    raise RuntimeError(
        message
    )


def normalize_game_id(
    value: Any,
) -> str:
    text = clean(
        value
    )

    if re.fullmatch(
        r"\d+\.0",
        text,
    ):
        return text[:-2]

    return text


def parse_number(
    value: Any,
    label: str,
) -> float:
    text = clean(
        value
    )

    if not text:
        fail(
            f"{label} is required"
        )

    try:
        result = float(
            text
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            f"{label} must be numeric; "
            f"found {value!r}"
        ) from exc

    if not math.isfinite(
        result
    ):
        fail(
            f"{label} must be finite; "
            f"found {value!r}"
        )

    return result


def parse_integer(
    value: Any,
    label: str,
) -> int:
    result = parse_number(
        value,
        label,
    )

    if not result.is_integer():
        fail(
            f"{label} must be an integer; "
            f"found {value!r}"
        )

    return int(
        result
    )


def selection_flag(
    value: Any,
    label: str,
) -> bool:
    text = clean(
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

    fail(
        f"{label} must be a valid "
        f"boolean/0/1 flag; found {value!r}"
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
        fail(
            f"{label}: missing required columns: "
            f"{missing}"
        )


def load_config(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        fail(
            f"Missing current-week config: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(
            handle
        )

    if not isinstance(
        data,
        dict,
    ):
        fail(
            "current_week.yaml must contain "
            "a YAML mapping"
        )

    return data


def load_csv(
    path: Path,
    label: str,
) -> pd.DataFrame:
    if not path.is_file():
        fail(
            f"Missing {label}: {path}"
        )

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if df.empty:
        fail(
            f"{label} contains no data rows: "
            f"{path}"
        )

    return df


def resolve_target(
    config: dict[str, Any],
    week_override: int | None,
) -> tuple[
    int,
    int,
    int,
]:
    season = parse_integer(
        config.get(
            "season"
        ),
        "current_week.season",
    )

    season_type = parse_integer(
        config.get(
            "season_type"
        ),
        "current_week.season_type",
    )

    configured_week = parse_integer(
        config.get(
            "week"
        ),
        "current_week.week",
    )

    week = (
        int(
            week_override
        )
        if week_override is not None
        else configured_week
    )

    if season < 1900:
        fail(
            f"Invalid season: {season}"
        )

    if season_type <= 0:
        fail(
            f"Invalid season_type: {season_type}"
        )

    if week <= 0:
        fail(
            f"Invalid week: {week}"
        )

    return (
        season,
        season_type,
        week,
    )


def validate_target_values(
    df: pd.DataFrame,
    label: str,
    season: int,
    season_type: int,
    week: int,
) -> None:
    seasons = {
        parse_integer(
            value,
            f"{label}: season",
        )
        for value in df[
            "season"
        ]
    }

    season_types = {
        parse_integer(
            value,
            f"{label}: season_type",
        )
        for value in df[
            "season_type"
        ]
    }

    weeks = {
        parse_integer(
            value,
            f"{label}: week",
        )
        for value in df[
            "week"
        ]
    }

    if seasons != {
        season
    }:
        fail(
            f"{label}: expected season={season}; "
            f"found {sorted(seasons)}"
        )

    if season_types != {
        season_type
    }:
        fail(
            f"{label}: expected season_type="
            f"{season_type}; "
            f"found {sorted(season_types)}"
        )

    if weeks != {
        week
    }:
        fail(
            f"{label}: expected week={week}; "
            f"found {sorted(weeks)}"
        )


def validate_selected_wagers(
    df: pd.DataFrame,
) -> dict[str, int]:
    specs = (
        (
            "ml",
            "ml_selected",
            "ml_selection",
            {
                "HOME",
                "AWAY",
            },
            "ml_odds_american",
            "ml_model_probability",
            None,
        ),
        (
            "spread",
            "spread_selected",
            "spread_selection",
            {
                "HOME",
                "AWAY",
            },
            "spread_odds_american",
            "spread_model_probability",
            "spread_line",
        ),
        (
            "total",
            "total_selected",
            "total_selection",
            {
                "OVER",
                "UNDER",
            },
            "total_odds_american",
            "total_model_probability",
            "total_line",
        ),
    )

    counts = {
        "ml_selected_bets": 0,
        "spread_selected_bets": 0,
        "total_selected_bets": 0,
    }

    for _, row in df.iterrows():
        game_id = normalize_game_id(
            row[
                "game_id"
            ]
        )

        for (
            market,
            selected_column,
            selection_column,
            allowed,
            odds_column,
            probability_column,
            line_column,
        ) in specs:
            selected = selection_flag(
                row[
                    selected_column
                ],
                (
                    f"game_id={game_id}: "
                    f"{selected_column}"
                ),
            )

            if not selected:
                continue

            side = clean(
                row[
                    selection_column
                ]
            ).upper()

            if side not in allowed:
                fail(
                    f"game_id={game_id}: "
                    f"selected {market} side must "
                    f"be one of {sorted(allowed)}; "
                    f"found {side!r}"
                )

            odds = parse_number(
                row[
                    odds_column
                ],
                (
                    f"game_id={game_id}: "
                    f"{odds_column}"
                ),
            )

            if odds == 0:
                fail(
                    f"game_id={game_id}: "
                    f"selected {market} odds "
                    "cannot be zero"
                )

            probability = parse_number(
                row[
                    probability_column
                ],
                (
                    f"game_id={game_id}: "
                    f"{probability_column}"
                ),
            )

            if not (
                0.0
                <= probability
                <= 1.0
            ):
                fail(
                    f"game_id={game_id}: "
                    f"selected {market} probability "
                    "must be in [0,1]"
                )

            if line_column is not None:
                parse_number(
                    row[
                        line_column
                    ],
                    (
                        f"game_id={game_id}: "
                        f"{line_column}"
                    ),
                )

            counts[
                f"{market}_selected_bets"
            ] += 1

    return counts


def validate_source(
    source: pd.DataFrame,
    path: Path,
    season: int,
    season_type: int,
    week: int,
) -> dict[str, int]:
    require_columns(
        source,
        REQUIRED_INPUT_COLUMNS,
        str(
            path
        ),
    )

    match = PICKS_FILE_RE.fullmatch(
        path.name
    )

    if match is None:
        fail(
            f"Unexpected picks filename: "
            f"{path.name}"
        )

    if int(
        match.group(
            1
        )
    ) != week:
        fail(
            f"{path}: filename week does "
            f"not match target week={week}"
        )

    validate_ids(
        source,
        str(
            path
        ),
    )

    validate_target_values(
        source,
        str(
            path
        ),
        season,
        season_type,
        week,
    )

    for column in (
        "away_team",
        "home_team",
    ):
        blank = source[
            column
        ].map(
            clean
        ).eq(
            ""
        )

        if blank.any():
            examples = (
                source.loc[
                    blank,
                    [
                        "game_id",
                        column,
                    ],
                ]
                .head(
                    10
                )
                .to_dict(
                    "records"
                )
            )

            fail(
                f"{path}: blank {column} values; "
                f"examples={examples}"
            )

    return validate_selected_wagers(
        source
    )


def validate_schedule_alignment(
    source: pd.DataFrame,
    schedule: pd.DataFrame,
    path: Path,
    season: int,
    season_type: int,
    week: int,
) -> None:
    require_columns(
        schedule,
        SCHEDULE_COLUMNS,
        str(
            path
        ),
    )

    validate_ids(
        schedule,
        str(
            path
        ),
    )

    validate_target_values(
        schedule,
        str(
            path
        ),
        season,
        season_type,
        week,
    )

    source_ids = set(
        source[
            "game_id"
        ]
    )

    schedule_ids = set(
        schedule[
            "game_id"
        ]
    )

    missing = sorted(
        schedule_ids
        - source_ids
    )

    unexpected = sorted(
        source_ids
        - schedule_ids
    )

    if (
        missing
        or unexpected
    ):
        fail(
            "Picks input game coverage does "
            "not match the weekly schedule; "
            f"missing_count={len(missing)} "
            f"unexpected_count={len(unexpected)} "
            f"missing_examples={missing[:10]} "
            f"unexpected_examples={unexpected[:10]}"
        )

    lookup = schedule.set_index(
        "game_id",
        drop=False,
    )

    mismatches: list[
        dict[
            str,
            str,
        ]
    ] = []

    for _, row in source.iterrows():
        game_id = row[
            "game_id"
        ]

        scheduled = lookup.loc[
            game_id
        ]

        source_away = clean(
            row[
                "away_team"
            ]
        )

        source_home = clean(
            row[
                "home_team"
            ]
        )

        schedule_away = clean(
            scheduled[
                "away_team"
            ]
        )

        schedule_home = clean(
            scheduled[
                "home_team"
            ]
        )

        if (
            source_away != schedule_away
            or source_home != schedule_home
        ):
            mismatches.append(
                {
                    "game_id":
                        game_id,
                    "source_away":
                        source_away,
                    "schedule_away":
                        schedule_away,
                    "source_home":
                        source_home,
                    "schedule_home":
                        schedule_home,
                }
            )

    if mismatches:
        fail(
            "Picks team identity does not "
            "match the weekly schedule; "
            f"count={len(mismatches)} "
            f"examples={mismatches[:10]}"
        )


def build_selected_mask(
    source: pd.DataFrame,
) -> pd.Series:
    masks: list[
        pd.Series
    ] = []

    for column in (
        "ml_selected",
        "spread_selected",
        "total_selected",
    ):
        values = [
            selection_flag(
                value,
                (
                    f"game_id={game_id}: "
                    f"{column}"
                ),
            )
            for (
                value,
                game_id,
            ) in zip(
                source[
                    column
                ],
                source[
                    "game_id"
                ],
                strict=True,
            )
        ]

        masks.append(
            pd.Series(
                values,
                index=source.index,
            )
        )

    return (
        masks[
            0
        ]
        | masks[
            1
        ]
        | masks[
            2
        ]
    )


def build_output(
    source: pd.DataFrame,
) -> pd.DataFrame:
    mask = build_selected_mask(
        source
    )

    return (
        source.loc[
            mask,
            OUTPUT_COLUMNS,
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


def normalized_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy().reset_index(
        drop=True
    )

    for column in result.columns:
        result[
            column
        ] = result[
            column
        ].map(
            clean
        )

    return result


def validate_output(
    output: pd.DataFrame,
    source: pd.DataFrame,
    season: int,
    season_type: int,
    week: int,
) -> None:
    if list(
        output.columns
    ) != OUTPUT_COLUMNS:
        fail(
            "Output columns changed"
        )

    expected = build_output(
        source
    )

    if not normalized_frame(
        output
    ).equals(
        normalized_frame(
            expected
        )
    ):
        fail(
            "Output does not match "
            "the selected source rows"
        )

    if output.empty:
        return

    validate_ids(
        output,
        "final selected output",
    )

    validate_target_values(
        output,
        "final selected output",
        season,
        season_type,
        week,
    )


def stage_csv(
    output: pd.DataFrame,
    source: pd.DataFrame,
    path: Path,
    season: int,
    season_type: int,
    week: int,
) -> tuple[
    Path,
    bool,
]:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        f".{path.name}."
        f"{uuid.uuid4().hex}.tmp"
    )

    try:
        serialized = stage_dataframe_csv(temporary, output)

        validate_output(
            serialized,
            source,
            season,
            season_type,
            week,
        )

        modified = (
            not path.is_file()
            or path.read_bytes()
            != temporary.read_bytes()
        )

        return (
            temporary,
            modified,
        )

    except Exception:
        temporary.unlink(
            missing_ok=True
        )

        raise


def publish_outputs(
    output: pd.DataFrame,
    source: pd.DataFrame,
    output_path: Path,
    locked_path: Path,
    season: int,
    season_type: int,
    week: int,
) -> tuple[
    bool,
    bool,
]:
    staged: list[
        tuple[
            Path,
            Path,
            bool,
        ]
    ] = []

    backups: dict[
        Path,
        Path | None,
    ] = {}

    replaced: list[
        Path
    ] = []

    try:
        for path in (
            output_path,
            locked_path,
        ):
            temporary, modified = stage_csv(
                output,
                source,
                path,
                season,
                season_type,
                week,
            )

            staged.append(
                (
                    temporary,
                    path,
                    modified,
                )
            )

        if (
            staged[
                0
            ][
                0
            ].read_bytes()
            != staged[
                1
            ][
                0
            ].read_bytes()
        ):
            fail(
                "Staged selected output files "
                "do not match"
            )

        for (
            _,
            path,
            modified,
        ) in staged:
            if (
                modified
                and path.is_file()
            ):
                backup = path.with_name(
                    f".{path.name}."
                    f"{uuid.uuid4().hex}.bak"
                )

                shutil.copy2(
                    path,
                    backup,
                )

                backups[
                    path
                ] = backup

            else:
                backups[
                    path
                ] = None

        try:
            for (
                temporary,
                path,
                modified,
            ) in staged:
                if modified:
                    os.replace(
                        temporary,
                        path,
                    )

                    replaced.append(
                        path
                    )

        except Exception:
            for path in reversed(
                replaced
            ):
                backup = backups[
                    path
                ]

                if backup is None:
                    path.unlink(
                        missing_ok=True
                    )

                else:
                    os.replace(
                        backup,
                        path,
                    )

            raise

        return (
            staged[
                0
            ][
                2
            ],
            staged[
                1
            ][
                2
            ],
        )

    finally:
        for (
            temporary,
            _,
            _,
        ) in staged:
            temporary.unlink(
                missing_ok=True
            )

        for backup in backups.values():
            if backup is not None:
                backup.unlink(
                    missing_ok=True
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build final CFB selected picks CSV."
        )
    )

    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help=(
            "Target week. Defaults to "
            "config/current_week.yaml."
        ),
    )

    return parser.parse_args()


def run(
    report: PipelineReporter,
    args: argparse.Namespace,
) -> int:
    config = load_config(
        CURRENT_WEEK_CONFIG
    )

    (
        season,
        season_type,
        week,
    ) = resolve_target(
        config,
        args.week,
    )

    report.season = season
    report.week = week

    input_path = (
        INPUT_DIR
        / f"week_{week}_CFB_picks.csv"
    )

    schedule_path = weekly_schedule_path(
        week
    )

    output_path = (
        OUTPUT_DIR
        / f"week_{week}_CFB_select_picks.csv"
    )

    locked_path = (
        LOCKED_DIR
        / f"week_{week}_CFB_select_picks.csv"
    )

    report.add_input(
        CURRENT_WEEK_CONFIG
    )

    report.add_input(
        input_path
    )

    report.add_input(
        schedule_path
    )

    report.add_output(
        output_path
    )

    report.add_output(
        locked_path
    )

    report.update_details(
        {
            "script_version":
                SCRIPT_VERSION,
            "season_type":
                season_type,
            "input_path":
                str(
                    input_path
                ),
            "schedule_path":
                str(
                    schedule_path
                ),
            "output_path":
                str(
                    output_path
                ),
            "locked_output_path":
                str(
                    locked_path
                ),
            "output_modified":
                False,
            "locked_output_modified":
                False,
        }
    )

    source = load_csv(
        input_path,
        "picks input",
    )

    schedule = load_csv(
        schedule_path,
        "weekly schedule",
    )

    counts = validate_source(
        source,
        input_path,
        season,
        season_type,
        week,
    )

    validate_schedule_alignment(
        source,
        schedule,
        schedule_path,
        season,
        season_type,
        week,
    )

    output = build_output(
        source
    )

    validate_output(
        output,
        source,
        season,
        season_type,
        week,
    )

    (
        output_modified,
        locked_output_modified,
    ) = publish_outputs(
        output,
        source,
        output_path,
        locked_path,
        season,
        season_type,
        week,
    )

    selected_games = len(
        output
    )

    selected_bets = sum(
        counts.values()
    )

    report.set_rows(
        rows_in=len(
            source
        ),
        rows_out=selected_games,
    )

    report.update_details(
        {
            "source_games":
                len(
                    source
                ),
            "schedule_games":
                len(
                    schedule
                ),
            "selected_games":
                selected_games,
            "selected_bets":
                selected_bets,
            **counts,
            "output_modified":
                output_modified,
            "locked_output_modified":
                locked_output_modified,
        }
    )

    print(
        "final_picks.py "
        f"version={SCRIPT_VERSION}"
    )

    print(
        f"WROTE {output_path} "
        f"| selected_games={selected_games} "
        f"| selected_bets={selected_bets} "
        "| output_modified="
        f"{'yes' if output_modified else 'no'}"
    )

    print(
        f"WROTE {locked_path} "
        f"| selected_games={selected_games} "
        f"| selected_bets={selected_bets} "
        "| output_modified="
        f"{'yes' if locked_output_modified else 'no'}"
    )

    return 0


def main() -> int:
    args = parse_args()

    with PipelineReporter(
        script=__file__,
        stage="03_picks",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        season=None,
        week=args.week,
        extra_context={
            "script_version":
                SCRIPT_VERSION,
            "output_scope":
                "final_selected_picks",
        },
    ) as report:
        return run(
            report,
            args,
        )

    raise RuntimeError("context manager unexpectedly suppressed an exception")

if __name__ == "__main__":
    raise SystemExit(
        main()
    )

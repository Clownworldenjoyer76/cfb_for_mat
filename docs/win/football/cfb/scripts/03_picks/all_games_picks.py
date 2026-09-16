#!/usr/bin/env python3
"""
Build compact all-game CFB projection output.

READS:
  docs/win/football/cfb/config/current_week.yaml
  docs/win/football/cfb/02_select/week_{week}_CFB_selected.csv
  docs/win/football/cfb/00_intake/schedule/weekly/
      week_{week}_CFB_weekly_schedule.csv

WRITES:
  docs/win/football/cfb/03_picks/all_games/
      all_week_{week}_CFB_picks.csv

OUTPUT COLUMNS:
  season
  week
  game_id
  away_team
  home_team
  predicted_away_score
  predicted_home_score
  predicted_total
  predicted_home_spread
  predicted_away_spread

Display contract:
- Predicted away score, home score, and total are independently rounded
  from the original model values to exactly one decimal place.
- Displayed spreads are calculated from the displayed one-decimal scores.

Spread definitions:
  predicted_home_spread =
      predicted_away_score - predicted_home_score

  predicted_away_spread =
      predicted_home_score - predicted_away_score
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
CFB_ROOT = SCRIPT_DIR.parents[1]
REPORT_ROOT = CFB_ROOT / "errors"

DEFAULT_INPUT_DIR = CFB_ROOT / "02_select"
DEFAULT_OUTPUT_DIR = (
    CFB_ROOT
    / "03_picks"
    / "all_games"
)

CURRENT_WEEK_CONFIG_PATH = (
    CFB_ROOT
    / "config"
    / "current_week.yaml"
)

SCRIPT_VERSION = (
    "cfb-all-games-picks-v2-hardened-2026-09-16"
)

SELECTED_FILE_RE = re.compile(
    r"^week_(\d+)_CFB_selected\.csv$",
    flags=re.IGNORECASE,
)

PROJECTION_CONSISTENCY_TOLERANCE = 0.021
SERIALIZED_NUMERIC_TOLERANCE = 1e-9


if str(
    SCRIPTS_DIR
) not in sys.path:
    sys.path.insert(
        0,
        str(
            SCRIPTS_DIR
        ),
    )

from pipeline_reporter import PipelineReporter


OUTPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
    "predicted_away_score",
    "predicted_home_score",
    "predicted_total",
    "predicted_home_spread",
    "predicted_away_spread",
]


REQUIRED_INPUT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "away_team",
    "home_team",
    "predicted_away_score",
    "predicted_home_score",
    "predicted_total",
    "predicted_margin",
]


def fail(
    message: str,
) -> None:
    raise RuntimeError(
        message
    )


def clean(
    value: Any,
) -> str:
    if value is None:
        return ""

    text = str(
        value
    ).strip()

    if text.casefold() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }:
        return ""

    return text


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
        return text[
            :-2
        ]

    return text


def parse_float(
    value: Any,
    label: str,
) -> float:
    text = clean(
        value
    )

    if not text:
        fail(
            f"{label} is blank"
        )

    try:
        number = float(
            text
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            f"{label} is not numeric: "
            f"{value!r}"
        ) from exc

    if not math.isfinite(
        number
    ):
        fail(
            f"{label} is non-finite: "
            f"{value!r}"
        )

    return number


def integer_value(
    value: Any,
    label: str,
) -> int:
    number = parse_float(
        value,
        label,
    )

    if not number.is_integer():
        fail(
            f"{label} must be an integer; "
            f"found {value!r}"
        )

    return int(
        number
    )


def read_yaml(
    path: Path,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        fail(
            f"Missing {label}: {path}"
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
            f"{label} must contain a YAML mapping: "
            f"{path}"
        )

    return data


def read_csv(
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


def resolve_target(
    current_week: dict[str, Any],
    season_override: int | None,
    week_override: int | None,
) -> tuple[
    int,
    int,
]:
    configured_season = integer_value(
        current_week.get(
            "season"
        ),
        "current_week.season",
    )

    configured_week = integer_value(
        current_week.get(
            "week"
        ),
        "current_week.week",
    )

    season = (
        int(
            season_override
        )
        if season_override is not None
        else configured_season
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
            f"Invalid target season: {season}"
        )

    if week <= 0:
        fail(
            f"Invalid target week: {week}"
        )

    return (
        season,
        week,
    )


def selected_file_week(
    path: Path,
) -> int:
    match = SELECTED_FILE_RE.fullmatch(
        path.name
    )

    if match is None:
        fail(
            "Unexpected selected filename: "
            f"{path.name}"
        )

    return int(
        match.group(
            1
        )
    )


def weekly_schedule_path(
    week: int,
) -> Path:
    return (
        CFB_ROOT
        / "00_intake"
        / "schedule"
        / "weekly"
        / f"week_{week}_CFB_weekly_schedule.csv"
    )


def validate_game_ids(
    df: pd.DataFrame,
    label: str,
) -> None:
    game_ids = df[
        "game_id"
    ].map(
        normalize_game_id
    )

    if game_ids.eq(
        ""
    ).any():
        fail(
            f"{label}: blank game_id found"
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
        fail(
            f"{label}: duplicate game_id values: "
            f"{duplicates[:10]}"
        )

    df[
        "game_id"
    ] = game_ids


def validate_source_target(
    source: pd.DataFrame,
    input_path: Path,
    season: int,
    week: int,
) -> None:
    require_columns(
        source,
        REQUIRED_INPUT_COLUMNS,
        str(
            input_path
        ),
    )

    filename_week = selected_file_week(
        input_path
    )

    if filename_week != week:
        fail(
            f"{input_path}: filename week="
            f"{filename_week} does not match "
            f"target week={week}"
        )

    validate_game_ids(
        source,
        str(
            input_path
        ),
    )

    seasons = {
        integer_value(
            value,
            f"{input_path}: season",
        )
        for value in source[
            "season"
        ]
    }

    weeks = {
        integer_value(
            value,
            f"{input_path}: week",
        )
        for value in source[
            "week"
        ]
    }

    if seasons != {
        season
    }:
        fail(
            f"{input_path}: expected only "
            f"season={season}; "
            f"found {sorted(seasons)}"
        )

    if weeks != {
        week
    }:
        fail(
            f"{input_path}: expected only "
            f"week={week}; "
            f"found {sorted(weeks)}"
        )

    for column in [
        "away_team",
        "home_team",
    ]:
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
                f"{input_path}: blank {column} "
                f"values; examples={examples}"
            )


def validate_schedule_alignment(
    source: pd.DataFrame,
    schedule: pd.DataFrame,
    schedule_path: Path,
    season: int,
    week: int,
) -> None:
    require_columns(
        schedule,
        [
            "season",
            "week",
            "game_id",
            "away_team",
            "home_team",
        ],
        str(
            schedule_path
        ),
    )

    validate_game_ids(
        schedule,
        str(
            schedule_path
        ),
    )

    schedule_seasons = {
        integer_value(
            value,
            f"{schedule_path}: season",
        )
        for value in schedule[
            "season"
        ]
    }

    schedule_weeks = {
        integer_value(
            value,
            f"{schedule_path}: week",
        )
        for value in schedule[
            "week"
        ]
    }

    if schedule_seasons != {
        season
    }:
        fail(
            f"{schedule_path}: expected only "
            f"season={season}; "
            f"found {sorted(schedule_seasons)}"
        )

    if schedule_weeks != {
        week
    }:
        fail(
            f"{schedule_path}: expected only "
            f"week={week}; "
            f"found {sorted(schedule_weeks)}"
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
            "Selected input game coverage does not "
            "match the target weekly schedule; "
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

        schedule_row = lookup.loc[
            game_id
        ]

        source_away = clean(
            row.get(
                "away_team"
            )
        )

        source_home = clean(
            row.get(
                "home_team"
            )
        )

        schedule_away = clean(
            schedule_row.get(
                "away_team"
            )
        )

        schedule_home = clean(
            schedule_row.get(
                "home_team"
            )
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
            "Selected input team identity does not "
            "match the target weekly schedule; "
            f"count={len(mismatches)} "
            f"examples={mismatches[:10]}"
        )


def validate_projection_consistency(
    source: pd.DataFrame,
) -> None:
    for position, row in enumerate(
        source.itertuples(
            index=False
        ),
        start=2,
    ):
        values = row._asdict()

        game_id = normalize_game_id(
            values.get(
                "game_id"
            )
        )

        away_score = parse_float(
            values.get(
                "predicted_away_score"
            ),
            (
                f"game_id={game_id} "
                "predicted_away_score"
            ),
        )

        home_score = parse_float(
            values.get(
                "predicted_home_score"
            ),
            (
                f"game_id={game_id} "
                "predicted_home_score"
            ),
        )

        predicted_total = parse_float(
            values.get(
                "predicted_total"
            ),
            (
                f"game_id={game_id} "
                "predicted_total"
            ),
        )

        predicted_margin = parse_float(
            values.get(
                "predicted_margin"
            ),
            (
                f"game_id={game_id} "
                "predicted_margin"
            ),
        )

        score_total = (
            away_score
            + home_score
        )

        score_margin = (
            home_score
            - away_score
        )

        if not math.isclose(
            score_total,
            predicted_total,
            rel_tol=0.0,
            abs_tol=PROJECTION_CONSISTENCY_TOLERANCE,
        ):
            fail(
                f"CSV row {position}, game_id={game_id}: "
                "predicted scores do not reconcile "
                "with predicted_total within "
                f"{PROJECTION_CONSISTENCY_TOLERANCE}; "
                f"away={away_score} "
                f"home={home_score} "
                f"score_total={score_total} "
                f"predicted_total={predicted_total}"
            )

        if not math.isclose(
            score_margin,
            predicted_margin,
            rel_tol=0.0,
            abs_tol=PROJECTION_CONSISTENCY_TOLERANCE,
        ):
            fail(
                f"CSV row {position}, game_id={game_id}: "
                "predicted scores do not reconcile "
                "with predicted_margin within "
                f"{PROJECTION_CONSISTENCY_TOLERANCE}; "
                f"away={away_score} "
                f"home={home_score} "
                f"score_margin={score_margin} "
                f"predicted_margin={predicted_margin}"
            )


def round_one_decimal(
    value: float,
) -> float:
    return round(
        value,
        1,
    )


def format_one_decimal(
    value: float,
) -> str:
    return f"{value:.1f}"


def build_output(
    source: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for position, row in enumerate(
        source.itertuples(
            index=False
        ),
        start=2,
    ):
        values = row._asdict()

        game_id = normalize_game_id(
            values.get(
                "game_id"
            )
        )

        away_score = parse_float(
            values.get(
                "predicted_away_score"
            ),
            (
                f"CSV row {position}, "
                f"game_id={game_id}: "
                "predicted_away_score"
            ),
        )

        home_score = parse_float(
            values.get(
                "predicted_home_score"
            ),
            (
                f"CSV row {position}, "
                f"game_id={game_id}: "
                "predicted_home_score"
            ),
        )

        predicted_total = parse_float(
            values.get(
                "predicted_total"
            ),
            (
                f"CSV row {position}, "
                f"game_id={game_id}: "
                "predicted_total"
            ),
        )

        away_score_display = round_one_decimal(
            away_score
        )

        home_score_display = round_one_decimal(
            home_score
        )

        total_display = round_one_decimal(
            predicted_total
        )

        predicted_home_spread = round_one_decimal(
            away_score_display
            - home_score_display
        )

        predicted_away_spread = round_one_decimal(
            home_score_display
            - away_score_display
        )

        rows.append(
            {
                "season":
                    clean(
                        values.get(
                            "season"
                        )
                    ),
                "week":
                    clean(
                        values.get(
                            "week"
                        )
                    ),
                "game_id":
                    game_id,
                "away_team":
                    clean(
                        values.get(
                            "away_team"
                        )
                    ),
                "home_team":
                    clean(
                        values.get(
                            "home_team"
                        )
                    ),
                "predicted_away_score":
                    format_one_decimal(
                        away_score_display
                    ),
                "predicted_home_score":
                    format_one_decimal(
                        home_score_display
                    ),
                "predicted_total":
                    format_one_decimal(
                        total_display
                    ),
                "predicted_home_spread":
                    format_one_decimal(
                        predicted_home_spread
                    ),
                "predicted_away_spread":
                    format_one_decimal(
                        predicted_away_spread
                    ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )


def validate_one_decimal_string(
    value: Any,
    label: str,
) -> float:
    text = clean(
        value
    )

    if not re.fullmatch(
        r"-?\d+\.\d",
        text,
    ):
        fail(
            f"{label} is not formatted to "
            f"exactly one decimal: {value!r}"
        )

    return parse_float(
        text,
        label,
    )


def validate_output_frame(
    output: pd.DataFrame,
    source: pd.DataFrame,
    season: int,
    week: int,
) -> None:
    if list(
        output.columns
    ) != OUTPUT_COLUMNS:
        fail(
            "Output column integrity check failed"
        )

    if len(
        output
    ) != len(
        source
    ):
        fail(
            "Output row count does not match "
            "input row count"
        )

    validate_game_ids(
        output,
        "all-games output",
    )

    source_ids = source[
        "game_id"
    ].map(
        normalize_game_id
    ).tolist()

    output_ids = output[
        "game_id"
    ].map(
        normalize_game_id
    ).tolist()

    if output_ids != source_ids:
        fail(
            "game_id order changed during processing"
        )

    for position in range(
        len(
            output
        )
    ):
        source_row = source.iloc[
            position
        ]

        output_row = output.iloc[
            position
        ]

        game_id = normalize_game_id(
            source_row[
                "game_id"
            ]
        )

        if integer_value(
            output_row[
                "season"
            ],
            (
                f"game_id={game_id}: "
                "output season"
            ),
        ) != season:
            fail(
                f"game_id={game_id}: "
                "output season changed"
            )

        if integer_value(
            output_row[
                "week"
            ],
            (
                f"game_id={game_id}: "
                "output week"
            ),
        ) != week:
            fail(
                f"game_id={game_id}: "
                "output week changed"
            )

        if clean(
            output_row[
                "away_team"
            ]
        ) != clean(
            source_row[
                "away_team"
            ]
        ):
            fail(
                f"game_id={game_id}: "
                "away_team changed"
            )

        if clean(
            output_row[
                "home_team"
            ]
        ) != clean(
            source_row[
                "home_team"
            ]
        ):
            fail(
                f"game_id={game_id}: "
                "home_team changed"
            )

        away_display = validate_one_decimal_string(
            output_row[
                "predicted_away_score"
            ],
            (
                f"game_id={game_id}: "
                "predicted_away_score"
            ),
        )

        home_display = validate_one_decimal_string(
            output_row[
                "predicted_home_score"
            ],
            (
                f"game_id={game_id}: "
                "predicted_home_score"
            ),
        )

        total_display = validate_one_decimal_string(
            output_row[
                "predicted_total"
            ],
            (
                f"game_id={game_id}: "
                "predicted_total"
            ),
        )

        home_spread = validate_one_decimal_string(
            output_row[
                "predicted_home_spread"
            ],
            (
                f"game_id={game_id}: "
                "predicted_home_spread"
            ),
        )

        away_spread = validate_one_decimal_string(
            output_row[
                "predicted_away_spread"
            ],
            (
                f"game_id={game_id}: "
                "predicted_away_spread"
            ),
        )

        expected_away_display = round_one_decimal(
            parse_float(
                source_row[
                    "predicted_away_score"
                ],
                (
                    f"game_id={game_id}: "
                    "source predicted_away_score"
                ),
            )
        )

        expected_home_display = round_one_decimal(
            parse_float(
                source_row[
                    "predicted_home_score"
                ],
                (
                    f"game_id={game_id}: "
                    "source predicted_home_score"
                ),
            )
        )

        expected_total_display = round_one_decimal(
            parse_float(
                source_row[
                    "predicted_total"
                ],
                (
                    f"game_id={game_id}: "
                    "source predicted_total"
                ),
            )
        )

        if not math.isclose(
            away_display,
            expected_away_display,
            rel_tol=0.0,
            abs_tol=SERIALIZED_NUMERIC_TOLERANCE,
        ):
            fail(
                f"game_id={game_id}: displayed "
                "away score changed"
            )

        if not math.isclose(
            home_display,
            expected_home_display,
            rel_tol=0.0,
            abs_tol=SERIALIZED_NUMERIC_TOLERANCE,
        ):
            fail(
                f"game_id={game_id}: displayed "
                "home score changed"
            )

        if not math.isclose(
            total_display,
            expected_total_display,
            rel_tol=0.0,
            abs_tol=SERIALIZED_NUMERIC_TOLERANCE,
        ):
            fail(
                f"game_id={game_id}: displayed "
                "predicted total changed"
            )

        expected_home_spread = round_one_decimal(
            away_display
            - home_display
        )

        expected_away_spread = round_one_decimal(
            home_display
            - away_display
        )

        if not math.isclose(
            home_spread,
            expected_home_spread,
            rel_tol=0.0,
            abs_tol=SERIALIZED_NUMERIC_TOLERANCE,
        ):
            fail(
                f"game_id={game_id}: displayed "
                "home spread is inconsistent "
                "with displayed scores"
            )

        if not math.isclose(
            away_spread,
            expected_away_spread,
            rel_tol=0.0,
            abs_tol=SERIALIZED_NUMERIC_TOLERANCE,
        ):
            fail(
                f"game_id={game_id}: displayed "
                "away spread is inconsistent "
                "with displayed scores"
            )

        if not math.isclose(
            home_spread
            + away_spread,
            0.0,
            rel_tol=0.0,
            abs_tol=SERIALIZED_NUMERIC_TOLERANCE,
        ):
            fail(
                f"game_id={game_id}: displayed "
                "spreads are not exact opposites"
            )


def validate_serialized_output(
    serialized: pd.DataFrame,
    expected: pd.DataFrame,
    source: pd.DataFrame,
    season: int,
    week: int,
) -> None:
    validate_output_frame(
        serialized,
        source,
        season,
        week,
    )

    if len(
        serialized
    ) != len(
        expected
    ):
        fail(
            "Serialized output row count changed"
        )

    if list(
        serialized.columns
    ) != list(
        expected.columns
    ):
        fail(
            "Serialized output columns changed"
        )

    left = serialized.reset_index(
        drop=True
    ).copy()

    right = expected.reset_index(
        drop=True
    ).copy()

    for column in OUTPUT_COLUMNS:
        left[
            column
        ] = left[
            column
        ].map(
            clean
        )

        right[
            column
        ] = right[
            column
        ].map(
            clean
        )

    if not left.equals(
        right
    ):
        fail(
            "Serialized output does not match "
            "validated in-memory output"
        )


def publish_atomic_csv(
    output: pd.DataFrame,
    source: pd.DataFrame,
    path: Path,
    season: int,
    week: int,
) -> bool:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        f".{path.name}."
        f"{uuid.uuid4().hex}.tmp"
    )

    try:
        with temporary.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            output.to_csv(
                handle,
                index=False,
                lineterminator="\n",
            )

            handle.flush()

            os.fsync(
                handle.fileno()
            )

        serialized = pd.read_csv(
            temporary,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
            low_memory=False,
        )

        validate_serialized_output(
            serialized,
            output,
            source,
            season,
            week,
        )

        new_bytes = temporary.read_bytes()

        if (
            path.is_file()
            and path.read_bytes()
            == new_bytes
        ):
            return False

        os.replace(
            temporary,
            path,
        )

        return True

    finally:
        temporary.unlink(
            missing_ok=True
        )


def summarize_output(
    output: pd.DataFrame,
) -> dict[str, Any]:
    totals = [
        parse_float(
            value,
            "output predicted_total",
        )
        for value in output[
            "predicted_total"
        ]
    ]

    home_spreads = [
        abs(
            parse_float(
                value,
                "output predicted_home_spread",
            )
        )
        for value in output[
            "predicted_home_spread"
        ]
    ]

    return {
        "games":
            len(
                output
            ),
        "finite_prediction_count":
            len(
                output
            ),
        "min_predicted_total":
            min(
                totals
            ),
        "max_predicted_total":
            max(
                totals
            ),
        "min_abs_displayed_spread":
            min(
                home_spreads
            ),
        "max_abs_displayed_spread":
            max(
                home_spreads
            ),
    }


def process_file(
    input_path: Path,
    output_dir: Path,
    season: int,
    week: int,
) -> tuple[
    pd.DataFrame,
    Path,
    Path,
    bool,
    dict[str, Any],
]:
    source = read_csv(
        input_path,
        "selected input",
    )

    validate_source_target(
        source,
        input_path,
        season,
        week,
    )

    schedule_path = weekly_schedule_path(
        week
    )

    schedule = read_csv(
        schedule_path,
        "weekly schedule",
    )

    validate_schedule_alignment(
        source,
        schedule,
        schedule_path,
        season,
        week,
    )

    validate_projection_consistency(
        source
    )

    output = build_output(
        source
    )

    validate_output_frame(
        output,
        source,
        season,
        week,
    )

    output_path = (
        output_dir
        / f"all_week_{week}_CFB_picks.csv"
    )

    output_modified = publish_atomic_csv(
        output,
        source,
        output_path,
        season,
        week,
    )

    stats = summarize_output(
        output
    )

    return (
        output,
        output_path,
        schedule_path,
        output_modified,
        stats,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build compact all-game "
            "CFB prediction output."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help=(
            "Target season. Defaults to "
            "config/current_week.yaml."
        ),
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
    current_week = read_yaml(
        CURRENT_WEEK_CONFIG_PATH,
        "current-week config",
    )

    (
        season,
        week,
    ) = resolve_target(
        current_week,
        args.season,
        args.week,
    )

    report.season = season
    report.week = week

    input_path = (
        DEFAULT_INPUT_DIR
        / f"week_{week}_CFB_selected.csv"
    )

    schedule_path = weekly_schedule_path(
        week
    )

    output_path = (
        DEFAULT_OUTPUT_DIR
        / f"all_week_{week}_CFB_picks.csv"
    )

    report.add_input(
        CURRENT_WEEK_CONFIG_PATH
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

    report.update_details(
        {
            "script_version":
                SCRIPT_VERSION,
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
            "projection_consistency_tolerance":
                PROJECTION_CONSISTENCY_TOLERANCE,
            "display_decimal_places":
                1,
            "output_modified":
                False,
        }
    )

    (
        output,
        actual_output_path,
        actual_schedule_path,
        output_modified,
        stats,
    ) = process_file(
        input_path,
        DEFAULT_OUTPUT_DIR,
        season,
        week,
    )

    if actual_output_path != output_path:
        fail(
            "Unexpected all-games output path"
        )

    if actual_schedule_path != schedule_path:
        fail(
            "Unexpected weekly schedule path"
        )

    report.set_rows(
        rows_in=len(
            output
        ),
        rows_out=len(
            output
        ),
    )

    report.update_details(
        {
            **stats,
            "output_modified":
                output_modified,
        }
    )

    print(
        "all_games_picks.py "
        f"version={SCRIPT_VERSION}"
    )

    print(
        f"WROTE {output_path} "
        f"| games={stats['games']} "
        "| finite_predictions="
        f"{stats['finite_prediction_count']} "
        "| min_predicted_total="
        f"{stats['min_predicted_total']:.1f} "
        "| max_predicted_total="
        f"{stats['max_predicted_total']:.1f} "
        "| max_abs_displayed_spread="
        f"{stats['max_abs_displayed_spread']:.1f} "
        "| output_modified="
        f"{'yes' if output_modified else 'no'}"
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
        season=(
            args.season
            if args.season is not None
            else None
        ),
        week=(
            args.week
            if args.week is not None
            else None
        ),
        extra_context={
            "script_version":
                SCRIPT_VERSION,
            "output_scope":
                "compact_all_games_projection",
        },
    ) as report:
        return run(
            report,
            args,
        )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
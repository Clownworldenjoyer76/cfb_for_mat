#!/usr/bin/env python3
"""
Create a human-readable weekly CFB picks file.

READS:
  docs/win/football/cfb/config/current_week.yaml
  docs/win/football/cfb/03_picks/week_{week}_CFB_picks.csv
  docs/win/football/cfb/00_intake/schedule/weekly/
      week_{week}_CFB_weekly_schedule.csv

WRITES:
  docs/win/football/cfb/03_picks/cleaned/
      week_{week}_CFB_clean_picks.csv

The PICKS column contains every actual selected wager for the game.

Probability columns contain the model probability for the actual selected
wager in that market. If no wager was selected, the probability is blank.

The displayed market spread and total come from the current candidate-line
fields produced by selections.py, not the older projection-time market fields.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Never

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
CFB_ROOT = SCRIPT_DIR.parents[1]
REPORT_ROOT = CFB_ROOT / "errors"

INPUT_DIR = CFB_ROOT / "03_picks"
OUTPUT_DIR = INPUT_DIR / "cleaned"

CURRENT_WEEK_CONFIG_PATH = (
    CFB_ROOT
    / "config"
    / "current_week.yaml"
)

SCRIPT_VERSION = (
    "cfb-clean-weekly-picks-v2-hardened-2026-09-16"
)

WEEKLY_FILE_RE = re.compile(
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
    clean_text,
    normalized_frame_pair,
    prepare_schedule_coverage,
    register_report_paths,
    require_columns,
    resolve_weekly_report_target,
    stage_dataframe_csv,
    validate_target_columns,
)


REQUIRED_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",

    "predicted_away_score",
    "predicted_home_score",
    "predicted_total",
    "predicted_margin",

    "ml_home_odds_american",
    "ml_away_odds_american",

    # Current candidate market lines.
    "spread_home_line",
    "total_over_line",

    "ml_selected",
    "ml_selection",
    "ml_odds_american",
    "ml_model_probability",

    "spread_selected",
    "spread_selection",
    "spread_line",
    "spread_odds_american",
    "spread_model_probability",

    "total_selected",
    "total_selection",
    "total_line",
    "total_odds_american",
    "total_model_probability",
]


OUTPUT_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",

    "PICKS",

    "ml_probability",
    "spread_probability",
    "total_probability",

    "predicted_away_score",
    "predicted_home_score",
    "predicted_total",
    "predicted_margin",

    "away_moneyline",
    "home_moneyline",
    "home_spread",
    "market_total",
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
    text = clean_text(
        value
    )

    if re.fullmatch(
        r"\d+\.0",
        text,
    ):
        return text[:-2]

    return text


def parse_optional_float(
    value: Any,
    label: str,
) -> float | None:
    text = clean_text(
        value
    )

    if not text:
        return None

    try:
        number = float(
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
        number
    ):
        fail(
            f"{label} must be finite; "
            f"found {value!r}"
        )

    return number


def required_float(
    value: Any,
    label: str,
) -> float:
    number = parse_optional_float(
        value,
        label,
    )

    if number is None:
        fail(
            f"{label} is required"
        )

    return number


def integer_value(
    value: Any,
    label: str,
) -> int:
    number = required_float(
        value,
        label,
    )

    if not float(
        number
    ).is_integer():
        fail(
            f"{label} must be an integer; "
            f"found {value!r}"
        )

    return int(
        number
    )


def selection_flag(
    value: Any,
    label: str,
) -> bool:
    text = clean_text(
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
        f"{label} must be a valid boolean/0/1 flag; "
        f"found {value!r}"
    )


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
            f"{label} contains no data rows: {path}"
        )

    return df


def week_from_filename(
    path: Path,
) -> int:
    match = WEEKLY_FILE_RE.fullmatch(
        path.name
    )

    if match is None:
        fail(
            f"Invalid weekly filename: "
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


def validate_input_target(
    df: pd.DataFrame,
    path: Path,
    season: int,
    week: int,
) -> None:
    require_columns(
        df,
        REQUIRED_COLUMNS,
        str(
            path
        ),
    )

    filename_week = week_from_filename(
        path
    )

    if filename_week != week:
        fail(
            f"{path}: filename week={filename_week} "
            f"does not match target week={week}"
        )

    ids = df[
        "game_id"
    ].map(
        normalize_game_id
    )

    if ids.eq(
        ""
    ).any():
        fail(
            f"{path} contains blank game_id values"
        )

    if ids.duplicated().any():
        examples = ids[
            ids.duplicated(
                False
            )
        ].head(
            10
        ).tolist()

        fail(
            f"{path} contains duplicate game_id "
            f"values: {examples}"
        )

    df[
        "game_id"
    ] = ids

    validate_target_columns(
        df,
        str(path),
        season,
        week,
        integer_value,
    )

def validate_schedule_alignment(
    picks: pd.DataFrame,
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

    ids = schedule[
        "game_id"
    ].map(
        normalize_game_id
    )

    if ids.eq(
        ""
    ).any():
        fail(
            f"{schedule_path} contains "
            "blank game_id values"
        )

    if ids.duplicated().any():
        examples = ids[
            ids.duplicated(
                False
            )
        ].head(
            10
        ).tolist()

        fail(
            f"{schedule_path} contains duplicate "
            f"game_id values: {examples}"
        )

    (
        schedule,
        missing,
        unexpected,
    ) = prepare_schedule_coverage(
        schedule,
        ids,
        picks,
        str(schedule_path),
        season,
        week,
        integer_value,
    )

    if (
        missing
        or unexpected
    ):
        fail(
            "Picks input game coverage does not "
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

    for _, row in picks.iterrows():
        game_id = row[
            "game_id"
        ]

        schedule_row = lookup.loc[
            game_id
        ]

        pick_away = clean_text(
            row.get(
                "away_team"
            )
        )

        pick_home = clean_text(
            row.get(
                "home_team"
            )
        )

        schedule_away = clean_text(
            schedule_row.get(
                "away_team"
            )
        )

        schedule_home = clean_text(
            schedule_row.get(
                "home_team"
            )
        )

        if (
            pick_away != schedule_away
            or pick_home != schedule_home
        ):
            mismatches.append(
                {
                    "game_id":
                        game_id,
                    "picks_away":
                        pick_away,
                    "schedule_away":
                        schedule_away,
                    "picks_home":
                        pick_home,
                    "schedule_home":
                        schedule_home,
                }
            )

    if mismatches:
        fail(
            "Picks input team identity does not "
            "match the target weekly schedule; "
            f"count={len(mismatches)} "
            f"examples={mismatches[:10]}"
        )

def validate_selected_wagers(
    df: pd.DataFrame,
) -> None:
    specs = [
        {
            "market":
                "moneyline",
            "selected":
                "ml_selected",
            "selection":
                "ml_selection",
            "allowed":
                {
                    "HOME",
                    "AWAY",
                },
            "odds":
                "ml_odds_american",
            "probability":
                "ml_model_probability",
            "line":
                None,
        },
        {
            "market":
                "spread",
            "selected":
                "spread_selected",
            "selection":
                "spread_selection",
            "allowed":
                {
                    "HOME",
                    "AWAY",
                },
            "odds":
                "spread_odds_american",
            "probability":
                "spread_model_probability",
            "line":
                "spread_line",
        },
        {
            "market":
                "total",
            "selected":
                "total_selected",
            "selection":
                "total_selection",
            "allowed":
                {
                    "OVER",
                    "UNDER",
                },
            "odds":
                "total_odds_american",
            "probability":
                "total_model_probability",
            "line":
                "total_line",
        },
    ]

    for _, row in df.iterrows():
        game_id = clean_text(
            row.get(
                "game_id"
            )
        )

        for spec in specs:
            market = str(
                spec[
                    "market"
                ]
            )

            is_selected = selection_flag(
                row.get(
                    spec[
                        "selected"
                    ]
                ),
                (
                    f"game_id={game_id}: "
                    f"{spec['selected']}"
                ),
            )

            if not is_selected:
                continue

            side = clean_text(
                row.get(
                    spec[
                        "selection"
                    ]
                )
            ).upper()

            if side not in spec[
                "allowed"
            ]:
                fail(
                    f"game_id={game_id}: selected "
                    f"{market} side must be one of "
                    f"{sorted(spec['allowed'])}; "
                    f"found {side!r}"
                )

            odds = required_float(
                row.get(
                    spec[
                        "odds"
                    ]
                ),
                (
                    f"game_id={game_id}: "
                    f"{spec['odds']}"
                ),
            )

            if odds == 0:
                fail(
                    f"game_id={game_id}: "
                    f"selected {market} odds "
                    "cannot be zero"
                )

            probability = required_float(
                row.get(
                    spec[
                        "probability"
                    ]
                ),
                (
                    f"game_id={game_id}: "
                    f"{spec['probability']}"
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

            line_column = spec[
                "line"
            ]

            if line_column is not None:
                required_float(
                    row.get(
                        line_column
                    ),
                    (
                        f"game_id={game_id}: "
                        f"{line_column}"
                    ),
                )


def validate_chronology(
    df: pd.DataFrame,
) -> tuple[
    pd.Series,
    pd.Series,
]:
    dates = pd.to_datetime(
        df[
            "game_date"
        ],
        format="%Y-%m-%d",
        errors="coerce",
    )

    times = pd.to_datetime(
        df[
            "game_time"
        ],
        format="%H:%M",
        errors="coerce",
    )

    bad_dates = dates.isna()

    if bad_dates.any():
        examples = (
            df.loc[
                bad_dates,
                [
                    "game_id",
                    "game_date",
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
            "Invalid game_date values; "
            f"examples={examples}"
        )

    bad_times = times.isna()

    if bad_times.any():
        examples = (
            df.loc[
                bad_times,
                [
                    "game_id",
                    "game_time",
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
            "Invalid game_time values; "
            f"examples={examples}"
        )

    return (
        dates,
        times,
    )


def format_number(
    value: Any,
    decimals: int = 2,
) -> str:
    number = parse_optional_float(
        value,
        "formatted numeric value",
    )

    if number is None:
        return ""

    text = f"{number:.{decimals}f}"

    return text.rstrip(
        "0"
    ).rstrip(
        "."
    )


def format_probability(
    value: Any,
) -> str:
    number = required_float(
        value,
        "selected probability",
    )

    if not (
        0.0
        <= number
        <= 1.0
    ):
        fail(
            "Selected probability must "
            "be in [0,1]"
        )

    return f"{number:.4f}"


def format_signed(
    value: Any,
    decimals: int = 1,
) -> str:
    number = parse_optional_float(
        value,
        "formatted signed numeric value",
    )

    if number is None:
        return ""

    text = f"{number:+.{decimals}f}"

    return text.rstrip(
        "0"
    ).rstrip(
        "."
    )


def format_odds(
    value: Any,
) -> str:
    number = parse_optional_float(
        value,
        "American odds",
    )

    if number is None:
        return ""

    if number == 0:
        fail(
            "American odds cannot be zero"
        )

    odds = int(
        round(
            number
        )
    )

    if odds > 0:
        return f"+{odds}"

    return str(
        odds
    )


def team_for_side(
    row: pd.Series,
    side: str,
) -> str:
    normalized = clean_text(
        side
    ).upper()

    team = ""

    if normalized == "HOME":
        team = clean_text(
            row[
                "home_team"
            ]
        )

    elif normalized == "AWAY":
        team = clean_text(
            row[
                "away_team"
            ]
        )

    else:
        fail(
            f"game_id={row['game_id']}: "
            f"invalid team side {side!r}"
        )

    if not team:
        fail(
            f"game_id={row['game_id']}: "
            "selected team name is blank"
        )

    return team


def build_picks(
    row: pd.Series,
) -> str:
    picks: list[
        str
    ] = []

    if selection_flag(
        row[
            "ml_selected"
        ],
        (
            f"game_id={row['game_id']}: "
            "ml_selected"
        ),
    ):
        side = clean_text(
            row[
                "ml_selection"
            ]
        ).upper()

        team = team_for_side(
            row,
            side,
        )

        odds = format_odds(
            row[
                "ml_odds_american"
            ]
        )

        picks.append(
            f"ML {team} {odds}"
        )

    if selection_flag(
        row[
            "spread_selected"
        ],
        (
            f"game_id={row['game_id']}: "
            "spread_selected"
        ),
    ):
        side = clean_text(
            row[
                "spread_selection"
            ]
        ).upper()

        team = team_for_side(
            row,
            side,
        )

        line = format_signed(
            row[
                "spread_line"
            ],
            1,
        )

        odds = format_odds(
            row[
                "spread_odds_american"
            ]
        )

        picks.append(
            f"SPREAD {team} "
            f"{line} ({odds})"
        )

    if selection_flag(
        row[
            "total_selected"
        ],
        (
            f"game_id={row['game_id']}: "
            "total_selected"
        ),
    ):
        side = clean_text(
            row[
                "total_selection"
            ]
        ).upper()

        if side not in {
            "OVER",
            "UNDER",
        }:
            fail(
                f"game_id={row['game_id']}: "
                f"invalid total side {side!r}"
            )

        line = format_number(
            row[
                "total_line"
            ],
            1,
        )

        odds = format_odds(
            row[
                "total_odds_american"
            ]
        )

        picks.append(
            f"TOTAL {side} "
            f"{line} ({odds})"
        )

    if not picks:
        return "NO PICK"

    return " | ".join(
        picks
    )


def selected_probability(
    row: pd.Series,
    selected_column: str,
    probability_column: str,
) -> str:
    if not selection_flag(
        row[
            selected_column
        ],
        (
            f"game_id={row['game_id']}: "
            f"{selected_column}"
        ),
    ):
        return ""

    return format_probability(
        row[
            probability_column
        ]
    )


def build_output(
    df: pd.DataFrame,
) -> pd.DataFrame:
    output = pd.DataFrame(
        index=df.index
    )

    output[
        "season"
    ] = df[
        "season"
    ]

    output[
        "season_type"
    ] = df[
        "season_type"
    ]

    output[
        "week"
    ] = df[
        "week"
    ]

    output[
        "game_id"
    ] = df[
        "game_id"
    ]

    output[
        "game_date"
    ] = df[
        "game_date"
    ]

    output[
        "game_time"
    ] = df[
        "game_time"
    ]

    output[
        "away_team"
    ] = df[
        "away_team"
    ]

    output[
        "home_team"
    ] = df[
        "home_team"
    ]

    output[
        "PICKS"
    ] = df.apply(
        build_picks,
        axis=1,
    )

    output[
        "ml_probability"
    ] = df.apply(
        lambda row:
            selected_probability(
                row,
                "ml_selected",
                "ml_model_probability",
            ),
        axis=1,
    )

    output[
        "spread_probability"
    ] = df.apply(
        lambda row:
            selected_probability(
                row,
                "spread_selected",
                "spread_model_probability",
            ),
        axis=1,
    )

    output[
        "total_probability"
    ] = df.apply(
        lambda row:
            selected_probability(
                row,
                "total_selected",
                "total_model_probability",
            ),
        axis=1,
    )

    output[
        "predicted_away_score"
    ] = df[
        "predicted_away_score"
    ].map(
        lambda value:
            format_number(
                value,
                2,
            )
    )

    output[
        "predicted_home_score"
    ] = df[
        "predicted_home_score"
    ].map(
        lambda value:
            format_number(
                value,
                2,
            )
    )

    output[
        "predicted_total"
    ] = df[
        "predicted_total"
    ].map(
        lambda value:
            format_number(
                value,
                2,
            )
    )

    output[
        "predicted_margin"
    ] = df[
        "predicted_margin"
    ].map(
        lambda value:
            format_number(
                value,
                2,
            )
    )

    output[
        "away_moneyline"
    ] = df[
        "ml_away_odds_american"
    ].map(
        format_odds
    )

    output[
        "home_moneyline"
    ] = df[
        "ml_home_odds_american"
    ].map(
        format_odds
    )

    # These are the current candidate lines, not the older
    # projection-time home_spread / total fields.
    output[
        "home_spread"
    ] = df[
        "spread_home_line"
    ].map(
        lambda value:
            format_signed(
                value,
                1,
            )
    )

    output[
        "market_total"
    ] = df[
        "total_over_line"
    ].map(
        lambda value:
            format_number(
                value,
                1,
            )
    )

    return output[
        OUTPUT_COLUMNS
    ].copy()


def sort_output(
    output: pd.DataFrame,
) -> pd.DataFrame:
    dates, times = validate_chronology(
        output
    )

    result = output.copy()

    result[
        "_date"
    ] = dates

    result[
        "_time"
    ] = times

    result = result.sort_values(
        [
            "_date",
            "_time",
        ],
        kind="stable",
    )

    result = result.drop(
        columns=[
            "_date",
            "_time",
        ]
    )

    return result.reset_index(
        drop=True
    )


def validate_clean_output(
    output: pd.DataFrame,
    source: pd.DataFrame,
) -> None:
    if list(
        output.columns
    ) != OUTPUT_COLUMNS:
        fail(
            "Clean output column order/integrity "
            "check failed"
        )

    if len(
        output
    ) != len(
        source
    ):
        fail(
            "Clean output row count does not "
            "match picks input"
        )

    ids = output[
        "game_id"
    ].map(
        normalize_game_id
    )

    if ids.eq(
        ""
    ).any():
        fail(
            "Clean output contains blank game_id values"
        )

    if ids.duplicated().any():
        fail(
            "Clean output contains duplicate "
            "game_id values"
        )

    source_ids = set(
        source[
            "game_id"
        ].map(
            normalize_game_id
        )
    )

    output_ids = set(
        ids
    )

    if output_ids != source_ids:
        fail(
            "Clean output game_id coverage changed"
        )

    if (
        output[
            "PICKS"
        ].map(
            clean_text
        ).eq(
            ""
        ).any()
    ):
        fail(
            "Clean output contains blank PICKS values"
        )


def validate_serialized_output(
    serialized: pd.DataFrame,
    expected: pd.DataFrame,
    source: pd.DataFrame,
) -> None:
    validate_clean_output(
        serialized,
        source,
    )

    if list(
        serialized.columns
    ) != list(
        expected.columns
    ):
        fail(
            "Serialized clean output columns changed"
        )

    if len(
        serialized
    ) != len(
        expected
    ):
        fail(
            "Serialized clean output row count changed"
        )

    left, right = normalized_frame_pair(
        serialized,
        expected,
        OUTPUT_COLUMNS,
        clean_text,
    )

    if not left.equals(
        right
    ):
        fail(
            "Serialized clean output does not "
            "match validated in-memory output"
        )


def publish_atomic_csv(
    output: pd.DataFrame,
    source: pd.DataFrame,
    output_path: Path,
) -> bool:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output_path.with_name(
        f".{output_path.name}."
        f"{uuid.uuid4().hex}.tmp"
    )

    try:
        serialized = stage_dataframe_csv(temporary, output)

        validate_serialized_output(
            serialized,
            output,
            source,
        )

        new_bytes = temporary.read_bytes()

        if (
            output_path.is_file()
            and output_path.read_bytes()
            == new_bytes
        ):
            return False

        os.replace(
            temporary,
            output_path,
        )

        return True

    finally:
        temporary.unlink(
            missing_ok=True
        )


def summarize_output(
    output: pd.DataFrame,
) -> dict[
    str,
    int,
]:
    games_with_picks = int(
        output[
            "PICKS"
        ].ne(
            "NO PICK"
        ).sum()
    )

    ml_picks = int(
        output[
            "ml_probability"
        ].ne(
            ""
        ).sum()
    )

    spread_picks = int(
        output[
            "spread_probability"
        ].ne(
            ""
        ).sum()
    )

    total_picks = int(
        output[
            "total_probability"
        ].ne(
            ""
        ).sum()
    )

    total_wagers = (
        ml_picks
        + spread_picks
        + total_picks
    )

    return {
        "games":
            len(
                output
            ),
        "games_with_picks":
            games_with_picks,
        "no_pick_games":
            len(
                output
            )
            - games_with_picks,
        "total_wagers":
            total_wagers,
        "ml_picks":
            ml_picks,
        "spread_picks":
            spread_picks,
        "total_picks":
            total_picks,
        "away_moneyline_available":
            int(
                output[
                    "away_moneyline"
                ].ne(
                    ""
                ).sum()
            ),
        "home_moneyline_available":
            int(
                output[
                    "home_moneyline"
                ].ne(
                    ""
                ).sum()
            ),
        "home_spread_available":
            int(
                output[
                    "home_spread"
                ].ne(
                    ""
                ).sum()
            ),
        "market_total_available":
            int(
                output[
                    "market_total"
                ].ne(
                    ""
                ).sum()
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
    bool,
    dict[
        str,
        int,
    ],
    Path,
]:
    source = read_csv(
        input_path,
        "weekly picks input",
    )

    validate_input_target(
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

    validate_selected_wagers(
        source
    )

    validate_chronology(
        source
    )

    output = build_output(
        source
    )

    output = sort_output(
        output
    )

    validate_clean_output(
        output,
        source,
    )

    output_path = (
        output_dir
        / f"week_{week}_CFB_clean_picks.csv"
    )

    output_modified = publish_atomic_csv(
        output,
        source,
        output_path,
    )

    stats = summarize_output(
        output
    )

    return (
        output,
        output_path,
        output_modified,
        stats,
        schedule_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create human-readable weekly CFB picks."
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
    season, week = resolve_weekly_report_target(
        report,
        CURRENT_WEEK_CONFIG_PATH,
        args.season,
        args.week,
    )

    input_path = (
        INPUT_DIR
        / f"week_{week}_CFB_picks.csv"
    )

    expected_schedule_path = weekly_schedule_path(
        week
    )

    expected_output_path = (
        OUTPUT_DIR
        / f"week_{week}_CFB_clean_picks.csv"
    )

    register_report_paths(
        report,
        inputs=(
            CURRENT_WEEK_CONFIG_PATH,
            input_path,
            expected_schedule_path,
        ),
        output=expected_output_path,
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
                    expected_schedule_path
                ),
            "output_path":
                str(
                    expected_output_path
                ),
            "market_spread_source":
                "spread_home_line",
            "market_total_source":
                "total_over_line",
            "output_modified":
                False,
        }
    )

    (
        output,
        output_path,
        output_modified,
        stats,
        schedule_path,
    ) = process_file(
        input_path,
        OUTPUT_DIR,
        season,
        week,
    )

    if output_path != expected_output_path:
        fail(
            "Unexpected clean output path"
        )

    if schedule_path != expected_schedule_path:
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
        "clean_weekly_picks.py "
        f"version={SCRIPT_VERSION}"
    )

    print(
        f"Wrote {len(output)} games to "
        f"{output_path} "
        f"| games_with_picks={stats['games_with_picks']} "
        f"| no_pick_games={stats['no_pick_games']} "
        f"| total_wagers={stats['total_wagers']} "
        f"| ml_picks={stats['ml_picks']} "
        f"| spread_picks={stats['spread_picks']} "
        f"| total_market_picks={stats['total_picks']} "
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
                "human_readable_weekly_picks",
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
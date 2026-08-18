#!/usr/bin/env python3
"""
Build compact all-game NFL projection picks output.

READS:
  docs/win/football/nfl/02_select/week_{week}_NFL_selected.csv

WRITES:
  docs/win/football/nfl/03_picks/all_games/all_week_{week}_NFL_picks.csv

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

The projected away score, home score, and total use the original model
projection values and are displayed to exactly 1 decimal place.

The projected spreads are calculated from the displayed 1-decimal projected
scores.

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
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_INPUT_DIR = NFL_ROOT / "02_select"
DEFAULT_OUTPUT_DIR = NFL_ROOT / "03_picks" / "all_games"

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
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

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


def parse_float(
    value: Any,
    *,
    column: str,
    row_number: int,
) -> float:
    text = clean(value)

    if not text:
        fail(
            f"Row {row_number}: "
            f"{column} is blank"
        )

    try:
        number = float(text)
    except (TypeError, ValueError):
        fail(
            f"Row {row_number}: "
            f"{column} is not numeric: "
            f"{value!r}"
        )

    if not math.isfinite(number):
        fail(
            f"Row {row_number}: "
            f"{column} is non-finite: "
            f"{value!r}"
        )

    return number


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


def validate_game_ids(
    df: pd.DataFrame,
    label: str,
) -> None:
    game_ids = df["game_id"].map(clean)

    if (game_ids == "").any():
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


def round_one_decimal(
    value: float,
) -> float:
    return round(value, 1)


def format_one_decimal(
    value: float,
) -> str:
    return f"{value:.1f}"


def build_output(
    source: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for index, row in source.iterrows():
        row_number = index + 2

        away_score = parse_float(
            row["predicted_away_score"],
            column="predicted_away_score",
            row_number=row_number,
        )

        home_score = parse_float(
            row["predicted_home_score"],
            column="predicted_home_score",
            row_number=row_number,
        )

        predicted_total = parse_float(
            row["predicted_total"],
            column="predicted_total",
            row_number=row_number,
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
                "season": clean(
                    row["season"]
                ),
                "week": clean(
                    row["week"]
                ),
                "game_id": clean(
                    row["game_id"]
                ),
                "away_team": clean(
                    row["away_team"]
                ),
                "home_team": clean(
                    row["home_team"]
                ),
                "predicted_away_score": (
                    format_one_decimal(
                        away_score_display
                    )
                ),
                "predicted_home_score": (
                    format_one_decimal(
                        home_score_display
                    )
                ),
                "predicted_total": (
                    format_one_decimal(
                        total_display
                    )
                ),
                "predicted_home_spread": (
                    format_one_decimal(
                        predicted_home_spread
                    )
                ),
                "predicted_away_spread": (
                    format_one_decimal(
                        predicted_away_spread
                    )
                ),
            }
        )

    output = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    return output


def write_atomic_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="NFL week number",
    )

    args = parser.parse_args()

    if args.week <= 0:
        fail(
            "--week must be greater than 0"
        )

    input_path = (
        DEFAULT_INPUT_DIR
        / f"week_{args.week}_NFL_selected.csv"
    )

    output_path = (
        DEFAULT_OUTPUT_DIR
        / f"all_week_{args.week}_NFL_picks.csv"
    )

    if not input_path.is_file():
        fail(
            f"Input file not found: "
            f"{input_path}"
        )

    source = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
    )

    require_columns(
        source,
        REQUIRED_INPUT_COLUMNS,
        str(input_path),
    )

    validate_game_ids(
        source,
        str(input_path),
    )

    output = build_output(
        source
    )

    if len(output) != len(source):
        fail(
            "Output row count does not match "
            "input row count"
        )

    if list(output.columns) != OUTPUT_COLUMNS:
        fail(
            "Output column integrity check failed"
        )

    if (
        output["game_id"].tolist()
        != source["game_id"].map(clean).tolist()
    ):
        fail(
            "game_id order changed during processing"
        )

    write_atomic_csv(
        output,
        output_path,
    )

    print(
        f"WROTE {output_path} | "
        f"games={len(output)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

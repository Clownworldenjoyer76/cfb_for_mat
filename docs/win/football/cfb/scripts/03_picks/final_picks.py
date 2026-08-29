#!/usr/bin/env python3
"""
Build final CFB selected-picks output.

READS:
  docs/win/football/cfb/03_picks/week_{week}_CFB_picks.csv

WRITES:
  docs/win/football/cfb/03_picks/selected/
      week_{week}_CFB_select_picks.csv

A row is included when any of these equal 1:
  ml_selected
  spread_selected
  total_selected
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CFB_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_INPUT_DIR = CFB_ROOT / "03_picks"
DEFAULT_OUTPUT_DIR = CFB_ROOT / "03_picks" / "selected"


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


REQUIRED_INPUT_COLUMNS = OUTPUT_COLUMNS.copy()


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


def selection_flag(value: Any) -> bool:
    text = clean(value)

    if not text:
        return False

    try:
        return float(text) == 1.0
    except (TypeError, ValueError):
        return False


def build_output(
    source: pd.DataFrame,
) -> pd.DataFrame:
    selected_mask = (
        source["ml_selected"].map(selection_flag)
        | source["spread_selected"].map(selection_flag)
        | source["total_selected"].map(selection_flag)
    )

    selected = source.loc[
        selected_mask
    ].copy()

    return selected[
        OUTPUT_COLUMNS
    ].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final CFB selected picks CSV."
    )

    parser.add_argument(
        "--week",
        required=True,
        type=int,
        help="CFB week number",
    )

    args = parser.parse_args()

    week = args.week

    input_path = (
        DEFAULT_INPUT_DIR
        / f"week_{week}_CFB_picks.csv"
    )

    output_path = (
        DEFAULT_OUTPUT_DIR
        / f"week_{week}_CFB_select_picks.csv"
    )

    if not input_path.exists():
        fail(
            f"Input file not found: {input_path}"
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

    output = build_output(source)

    DEFAULT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
        lineterminator="\n",
    )

    print(
        f"WROTE {output_path} "
        f"| rows={len(output)}"
    )


if __name__ == "__main__":
    main()

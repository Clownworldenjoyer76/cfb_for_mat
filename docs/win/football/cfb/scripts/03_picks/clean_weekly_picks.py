#!/usr/bin/env python3
"""
clean_weekly_picks.py

Creates a simple, human-readable weekly CFB picks file.

READS:
    docs/win/football/cfb/03_picks/week_*_CFB_picks.csv

WRITES:
    docs/win/football/cfb/03_picks/cleaned/week_*_CFB_clean_picks.csv

The PICKS column contains every actual selected wager for the game.

Example:
    ML North Carolina +250 |
    SPREAD North Carolina +7.5 (-105) |
    TOTAL UNDER 47.5 (-108)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CFB_ROOT = SCRIPT_DIR.parents[1]

INPUT_DIR = CFB_ROOT / "03_picks"
OUTPUT_DIR = INPUT_DIR / "cleaned"

INPUT_PATTERN = "week_*_CFB_picks.csv"


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

    "home_spread",
    "total",

    "ml_home_odds_american",
    "ml_away_odds_american",

    "ml_selected",
    "ml_selection",
    "ml_odds_american",

    "spread_selected",
    "spread_selection",
    "spread_line",
    "spread_odds_american",

    "total_selected",
    "total_selection",
    "total_line",
    "total_odds_american",
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
    "predicted_away_score",
    "predicted_home_score",
    "predicted_total",
    "predicted_margin",
    "away_moneyline",
    "home_moneyline",
    "home_spread",
    "market_total",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def clean_text(value) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.casefold() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
    }:
        return ""

    return text


def to_float(value):
    text = clean_text(value)

    if not text:
        return None

    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def selected(value) -> bool:
    text = clean_text(value).casefold()

    return text in {
        "1",
        "true",
        "yes",
        "y",
    }


def format_number(
    value,
    decimals: int = 2,
) -> str:
    number = to_float(value)

    if number is None:
        return ""

    text = f"{number:.{decimals}f}"

    return text.rstrip("0").rstrip(".")


def format_signed(
    value,
    decimals: int = 1,
) -> str:
    number = to_float(value)

    if number is None:
        return ""

    text = f"{number:+.{decimals}f}"

    return text.rstrip("0").rstrip(".")


def format_odds(value) -> str:
    number = to_float(value)

    if number is None:
        return ""

    odds = int(round(number))

    if odds > 0:
        return f"+{odds}"

    return str(odds)


def team_for_side(
    row,
    side: str,
) -> str:
    side = clean_text(side).upper()

    if side == "HOME":
        return clean_text(
            row["home_team"]
        )

    if side == "AWAY":
        return clean_text(
            row["away_team"]
        )

    return side


def build_picks(row) -> str:
    picks = []

    # MONEYLINE
    if selected(row["ml_selected"]):
        side = clean_text(
            row["ml_selection"]
        ).upper()

        team = team_for_side(
            row,
            side,
        )

        odds = format_odds(
            row["ml_odds_american"]
        )

        text = f"ML {team}"

        if odds:
            text += f" {odds}"

        picks.append(text)

    # SPREAD
    if selected(row["spread_selected"]):
        side = clean_text(
            row["spread_selection"]
        ).upper()

        team = team_for_side(
            row,
            side,
        )

        line = format_signed(
            row["spread_line"],
            1,
        )

        odds = format_odds(
            row["spread_odds_american"]
        )

        text = f"SPREAD {team}"

        if line:
            text += f" {line}"

        if odds:
            text += f" ({odds})"

        picks.append(text)

    # TOTAL
    if selected(row["total_selected"]):
        side = clean_text(
            row["total_selection"]
        ).upper()

        line = format_number(
            row["total_line"],
            1,
        )

        odds = format_odds(
            row["total_odds_american"]
        )

        text = f"TOTAL {side}"

        if line:
            text += f" {line}"

        if odds:
            text += f" ({odds})"

        picks.append(text)

    if not picks:
        return "NO PICK"

    return " | ".join(picks)


def load_csv(path: Path) -> pd.DataFrame:
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
            f"Input contains no rows: {path}"
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{path} missing required columns: "
            f"{missing}"
        )

    return df


def build_output(
    df: pd.DataFrame,
) -> pd.DataFrame:

    output = pd.DataFrame()

    output["season"] = df["season"]
    output["season_type"] = df["season_type"]
    output["week"] = df["week"]
    output["game_id"] = df["game_id"]
    output["game_date"] = df["game_date"]
    output["game_time"] = df["game_time"]
    output["away_team"] = df["away_team"]
    output["home_team"] = df["home_team"]

    # MOST IMPORTANT COLUMN
    output["PICKS"] = df.apply(
        build_picks,
        axis=1,
    )

    output["predicted_away_score"] = (
        df["predicted_away_score"].map(
            lambda x: format_number(x, 2)
        )
    )

    output["predicted_home_score"] = (
        df["predicted_home_score"].map(
            lambda x: format_number(x, 2)
        )
    )

    output["predicted_total"] = (
        df["predicted_total"].map(
            lambda x: format_number(x, 2)
        )
    )

    output["predicted_margin"] = (
        df["predicted_margin"].map(
            lambda x: format_number(x, 2)
        )
    )

    output["away_moneyline"] = (
        df["ml_away_odds_american"].map(
            format_odds
        )
    )

    output["home_moneyline"] = (
        df["ml_home_odds_american"].map(
            format_odds
        )
    )

    output["home_spread"] = (
        df["home_spread"].map(
            lambda x: format_signed(x, 1)
        )
    )

    output["market_total"] = (
        df["total"].map(
            lambda x: format_number(x, 1)
        )
    )

    return output[OUTPUT_COLUMNS]


def week_from_filename(
    path: Path,
) -> int:
    match = re.fullmatch(
        r"week_(\d+)_CFB_picks\.csv",
        path.name,
        flags=re.IGNORECASE,
    )

    if not match:
        fail(
            f"Invalid weekly filename: "
            f"{path.name}"
        )

    return int(match.group(1))


def process_file(
    path: Path,
) -> None:

    df = load_csv(path)

    duplicates = df[
        df["game_id"].duplicated(
            keep=False
        )
    ]

    if not duplicates.empty:
        fail(
            f"{path} contains duplicate game_ids"
        )

    output = build_output(df)

    output["_date"] = pd.to_datetime(
        output["game_date"],
        errors="coerce",
    )

    output["_time"] = pd.to_datetime(
        output["game_time"],
        format="%H:%M",
        errors="coerce",
    )

    output = output.sort_values(
        [
            "_date",
            "_time",
        ],
        kind="stable",
        na_position="last",
    )

    output = output.drop(
        columns=[
            "_date",
            "_time",
        ]
    )

    week = week_from_filename(path)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"week_{week}_CFB_clean_picks.csv"
    )

    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    games_with_picks = (
        output["PICKS"]
        .ne("NO PICK")
        .sum()
    )

    total_picks = (
        output["PICKS"]
        .str.count(r"\|")
        .add(
            output["PICKS"].ne(
                "NO PICK"
            ).astype(int)
        )
        .sum()
    )

    print(
        f"Wrote {len(output)} games to "
        f"{output_path} "
        f"| games_with_picks={games_with_picks} "
        f"| total_picks={total_picks}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--week",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.week is not None:
        files = [
            INPUT_DIR
            / f"week_{args.week}_CFB_picks.csv"
        ]
    else:
        files = sorted(
            INPUT_DIR.glob(
                INPUT_PATTERN
            ),
            key=week_from_filename,
        )

    if not files:
        fail(
            "No weekly picks files found."
        )

    for path in files:
        if not path.is_file():
            fail(
                f"Missing input file: {path}"
            )

        process_file(path)


if __name__ == "__main__":
    main()
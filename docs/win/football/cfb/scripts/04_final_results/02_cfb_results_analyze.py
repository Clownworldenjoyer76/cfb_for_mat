#!/usr/bin/env python3
"""
CFB graded-results analysis layer.

Reads the existing per-game graded CFB files produced by grade_picks.py and
converts selected wagers into a one-row-per-bet ledger suitable for reporting.

READS:
    docs/win/football/cfb/04_final_results/graded/
        week_*_CFB_graded.csv

WRITES:
    docs/win/football/cfb/04_final_results/intermediate/
        work_cfb.csv

The existing CFB grader remains the source of truth for bet result and units.
This script does not re-grade wagers or recalculate P&L.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CFB_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_GRADED_DIR = CFB_ROOT / "04_final_results" / "graded"
DEFAULT_OUTPUT_DIR = CFB_ROOT / "04_final_results" / "intermediate"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "work_cfb.csv"

GRADED_FILE_RE = re.compile(r"^week_(\d+)_CFB_graded\.csv$")

MARKETS = {
    "moneyline": {
        "prefix": "ml",
        "line_column": None,
    },
    "spread": {
        "prefix": "spread",
        "line_column": "spread_line",
    },
    "total": {
        "prefix": "total",
        "line_column": "total_line",
    },
}

BASE_REQUIRED_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "away_team",
    "home_team",
]

MARKET_REQUIRED_SUFFIXES = [
    "selected",
    "selection",
    "odds_american",
    "model_probability",
    "implied_probability",
    "edge",
    "ev",
    "full_kelly",
    "kelly",
    "grade",
    "profit_units",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def to_float(value: Any) -> float | None:
    text = clean(value)
    if not text:
        return None

    try:
        result = float(text)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def selected_flag(value: Any) -> bool:
    return clean(value).casefold() in {"1", "1.0", "true", "yes", "y"}


def normalize_grade(value: Any) -> str:
    raw = clean(value).upper()
    mapping = {
        "WIN": "Win",
        "LOSS": "Loss",
        "PUSH": "Push",
        "VOID": "Void",
        "PENDING": "Pending",
        "NO_BET": "No Bet",
        "INVALID_SELECTION": "Invalid Selection",
        "INVALID_LINE": "Invalid Line",
    }
    return mapping.get(raw, raw.title() if raw else "")


def build_side_group(market_type: str, bet_side: str) -> str:
    side = clean(bet_side).upper()

    if market_type in {"moneyline", "spread"} and side in {"HOME", "AWAY"}:
        return side

    if market_type == "total" and side in {"OVER", "UNDER"}:
        return side

    return ""


def build_day_night(value: Any) -> str:
    raw = clean(value)
    if not raw:
        return ""

    formats = (
        "%I:%M %p",
        "%H:%M",
        "%I:%M%p",
        "%H:%M:%S",
    )

    for fmt in formats:
        try:
            parsed = datetime.strptime(raw, fmt)
            return "Day" if parsed.hour < 17 else "Night"
        except ValueError:
            continue

    return ""


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, path)


# ---------------------------------------------------------------------------
# Buckets -- intentionally modeled on the MLB reporting layer
# ---------------------------------------------------------------------------

def ev_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None:
        return "UNBUCKETED"
    if value < 0:
        return "<0"
    if value < 0.01:
        return "0.00_to_0.0099"
    if value < 0.02:
        return "0.01_to_0.0199"
    if value < 0.03:
        return "0.02_to_0.0299"
    if value < 0.04:
        return "0.03_to_0.0399"
    if value < 0.05:
        return "0.04_to_0.0499"
    if value < 0.075:
        return "0.05_to_0.0749"
    if value < 0.10:
        return "0.075_to_0.0999"
    return "0.10_plus"


def odds_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None:
        return "UNBUCKETED"
    if value <= -200:
        return "minus_200_or_lower"
    if value <= -150:
        return "minus_199_to_minus_150"
    if value <= -125:
        return "minus_149_to_minus_125"
    if value <= -110:
        return "minus_124_to_minus_110"
    if value <= -101:
        return "minus_109_to_minus_101"
    if value <= 100:
        return "minus_100_to_plus_100"
    if value <= 125:
        return "plus_101_to_plus_125"
    if value <= 150:
        return "plus_126_to_plus_150"
    if value <= 200:
        return "plus_151_to_plus_200"
    return "plus_201_or_higher"


def kelly_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None:
        return "UNBUCKETED"
    if value <= 0:
        return "zero_or_below"
    if value < 0.01:
        return "0.001_to_0.0099"
    if value < 0.02:
        return "0.01_to_0.0199"
    if value < 0.03:
        return "0.02_to_0.0299"
    if value < 0.05:
        return "0.03_to_0.0499"
    if value < 0.10:
        return "0.05_to_0.0999"
    if value < 0.15:
        return "0.10_to_0.1499"
    if value < 0.20:
        return "0.15_to_0.1999"
    return "0.20_plus"


def model_prob_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None:
        return "UNBUCKETED"

    pct = value * 100.0 if value <= 1.0 else value

    if pct < 50:
        return "<50"
    if pct < 55:
        return "50_to_54.9"
    if pct < 60:
        return "55_to_59.9"
    if pct < 65:
        return "60_to_64.9"
    if pct < 70:
        return "65_to_69.9"
    if pct < 75:
        return "70_to_74.9"
    if pct < 80:
        return "75_to_79.9"
    return "80_plus"


def spread_line_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None:
        return "UNBUCKETED"

    absolute = abs(value)
    if absolute < 3:
        return "0.0_to_2.9"
    if absolute < 7:
        return "3.0_to_6.9"
    if absolute < 10:
        return "7.0_to_9.9"
    if absolute < 14:
        return "10.0_to_13.9"
    if absolute < 21:
        return "14.0_to_20.9"
    if absolute < 28:
        return "21.0_to_27.9"
    return "28.0_plus"


def spread_role(value: Any) -> str:
    value = to_float(value)
    if value is None:
        return "UNBUCKETED"
    if value < 0:
        return "FAVORITE"
    if value > 0:
        return "UNDERDOG"
    return "PICKEM"


def total_bucket(value: Any) -> str:
    value = to_float(value)
    if value is None:
        return "UNBUCKETED"

    start = int(math.floor(value / 5.0) * 5)
    return f"{start}_to_{start + 4.9:.1f}"


# ---------------------------------------------------------------------------
# Input discovery / conversion
# ---------------------------------------------------------------------------

def required_columns_for_file() -> list[str]:
    required = list(BASE_REQUIRED_COLUMNS)

    optional_base = [
        "game_date",
        "game_time",
        "final_status",
        "final_completed",
        "final_away_score",
        "final_home_score",
        "final_total",
        "final_home_margin",
    ]

    for column in optional_base:
        if column not in required:
            required.append(column)

    for market_name, spec in MARKETS.items():
        prefix = spec["prefix"]
        for suffix in MARKET_REQUIRED_SUFFIXES:
            required.append(f"{prefix}_{suffix}")

        if spec["line_column"]:
            required.append(spec["line_column"])

    return required


def discover_files(graded_dir: Path) -> list[tuple[int, Path]]:
    discovered: list[tuple[int, Path]] = []

    for path in graded_dir.glob("week_*_CFB_graded.csv"):
        match = GRADED_FILE_RE.match(path.name)
        if match:
            discovered.append((int(match.group(1)), path))

    return sorted(discovered, key=lambda item: (item[0], item[1].name))


def read_graded_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if df.empty:
        return df

    require_columns(df, required_columns_for_file(), str(path))

    game_ids = df["game_id"].map(clean)
    if game_ids.eq("").any():
        raise ValueError(f"{path} contains blank game_id values")

    if game_ids.duplicated().any():
        examples = game_ids[game_ids.duplicated(False)].head(10).tolist()
        raise ValueError(f"{path} contains duplicate game_id values: {examples}")

    return df


def bet_row(row: pd.Series, market_type: str) -> dict[str, Any] | None:
    spec = MARKETS[market_type]
    prefix = spec["prefix"]

    if not selected_flag(row.get(f"{prefix}_selected", "")):
        return None

    selection = clean(row.get(f"{prefix}_selection", ""))
    line = (
        to_float(row.get(spec["line_column"], ""))
        if spec["line_column"]
        else None
    )

    result = normalize_grade(row.get(f"{prefix}_grade", ""))

    output = {
        "season": to_int(row.get("season")),
        "season_type": to_int(row.get("season_type")),
        "week": to_int(row.get("week")),
        "game_id": clean(row.get("game_id")),
        "game_date": clean(row.get("game_date")),
        "game_time": clean(row.get("game_time")),
        "away_team": clean(row.get("away_team")),
        "home_team": clean(row.get("home_team")),
        "market_type": market_type,
        "bet_side": selection.lower(),
        "side_group": build_side_group(market_type, selection),
        "line": line,
        "odds_american": to_float(row.get(f"{prefix}_odds_american", "")),
        "model_prob": to_float(row.get(f"{prefix}_model_probability", "")),
        "implied_prob": to_float(row.get(f"{prefix}_implied_probability", "")),
        "edge": to_float(row.get(f"{prefix}_edge", "")),
        "ev": to_float(row.get(f"{prefix}_ev", "")),
        "full_kelly": to_float(row.get(f"{prefix}_full_kelly", "")),
        "kelly": to_float(row.get(f"{prefix}_kelly", "")),
        "selection_reason": clean(row.get(f"{prefix}_selection_reason", "")),
        "bet_result": result,
        "bet_units": to_float(row.get(f"{prefix}_profit_units", "")),
        "final_status": clean(row.get("final_status", "")),
        "final_completed": to_int(row.get("final_completed")),
        "final_away_score": to_float(row.get("final_away_score")),
        "final_home_score": to_float(row.get("final_home_score")),
        "final_total": to_float(row.get("final_total")),
        "final_home_margin": to_float(row.get("final_home_margin")),
    }

    output["week_label"] = (
        f"Week {output['week']}" if output["week"] is not None else ""
    )
    output["day_night"] = build_day_night(output["game_time"])
    output["ev_bucket"] = ev_bucket(output["ev"])
    output["odds_bucket"] = odds_bucket(output["odds_american"])
    output["kelly_bucket"] = kelly_bucket(output["kelly"])
    output["model_prob_bucket"] = model_prob_bucket(output["model_prob"])
    output["win_prob_bucket"] = output["model_prob_bucket"]
    output["spread_line_bucket"] = (
        spread_line_bucket(line) if market_type == "spread" else "UNBUCKETED"
    )
    output["spread_role"] = (
        spread_role(line) if market_type == "spread" else "UNBUCKETED"
    )
    output["total_bucket"] = (
        total_bucket(line) if market_type == "total" else "UNBUCKETED"
    )

    return output


def build_work_file(
    graded_dir: Path,
    output_file: Path,
    season: int | None,
) -> pd.DataFrame:
    files = discover_files(graded_dir)
    if not files:
        raise FileNotFoundError(
            f"No week_*_CFB_graded.csv files found in {graded_dir}"
        )

    rows: list[dict[str, Any]] = []
    source_files = 0
    source_games = 0

    for filename_week, path in files:
        df = read_graded_file(path)
        if df.empty:
            continue

        if season is not None:
            season_values = pd.to_numeric(df["season"], errors="coerce")
            df = df.loc[season_values.eq(season)].copy()
            if df.empty:
                continue

        file_weeks = pd.to_numeric(df["week"], errors="coerce").dropna().unique()
        if len(file_weeks) != 1 or int(file_weeks[0]) != filename_week:
            raise ValueError(
                f"{path}: filename week={filename_week} does not match row week values "
                f"{file_weeks.tolist()}"
            )

        source_files += 1
        source_games += len(df)

        for _, game in df.iterrows():
            for market_type in MARKETS:
                item = bet_row(game, market_type)
                if item is not None:
                    rows.append(item)

    columns = [
        "season",
        "season_type",
        "week",
        "week_label",
        "game_id",
        "game_date",
        "game_time",
        "day_night",
        "away_team",
        "home_team",
        "market_type",
        "bet_side",
        "side_group",
        "line",
        "odds_american",
        "model_prob",
        "implied_prob",
        "edge",
        "ev",
        "full_kelly",
        "kelly",
        "selection_reason",
        "bet_result",
        "bet_units",
        "final_status",
        "final_completed",
        "final_away_score",
        "final_home_score",
        "final_total",
        "final_home_margin",
        "ev_bucket",
        "odds_bucket",
        "kelly_bucket",
        "model_prob_bucket",
        "win_prob_bucket",
        "spread_line_bucket",
        "spread_role",
        "total_bucket",
    ]

    work = pd.DataFrame(rows, columns=columns)

    if not work.empty:
        duplicate_key = ["season", "week", "game_id", "market_type"]
        if work.duplicated(duplicate_key).any():
            examples = work.loc[
                work.duplicated(duplicate_key, keep=False), duplicate_key
            ].head(10)
            raise ValueError(
                "Duplicate selected market rows detected:\n"
                + examples.to_string(index=False)
            )

        work = work.sort_values(
            ["season", "week", "game_date", "game_time", "game_id", "market_type"],
            kind="stable",
        ).reset_index(drop=True)

    atomic_write_csv(work, output_file)

    counts = (
        work["bet_result"].value_counts(dropna=False).to_dict()
        if not work.empty
        else {}
    )

    print(
        "CFB analyze complete: "
        f"files={source_files} games={source_games} selected_bets={len(work)} "
        f"results={counts} output={output_file}"
    )

    return work


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert graded CFB game rows into a one-row-per-bet reporting ledger."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Optional season filter. If omitted, all seasons in graded files are included.",
    )
    parser.add_argument(
        "--graded-dir",
        type=Path,
        default=DEFAULT_GRADED_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_work_file(
        graded_dir=args.graded_dir.resolve(),
        output_file=args.output.resolve(),
        season=args.season,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

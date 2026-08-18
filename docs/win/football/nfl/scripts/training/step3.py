#!/usr/bin/env python3
"""
Step 3: append final-game target fields to the historical training table.

READS:
  docs/win/football/nfl/training/historical_core_2021_2025.csv

WRITES:
  docs/win/football/nfl/training/historical_core_2021_2025.csv

No raw input/source files are edited.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


NFL_ROOT = Path("docs/win/football/nfl")
TRAINING_PATH = NFL_ROOT / "training/historical_core_2021_2025.csv"

REQUIRED_COLUMNS = [
    "away_score",
    "home_score",
    "spread_line",
    "total_line",
]

TARGET_COLUMNS = [
    "margin",
    "total_points",
    "home_win",
    "home_ats_margin",
    "home_ats_result",
    "total_result",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )


def require_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")


def numeric(series: pd.Series, column_name: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")

    bad_mask = (
        converted.isna()
        & series.notna()
        & series.astype(str).str.strip().ne("")
    )

    if bad_mask.any():
        bad_values = series.loc[bad_mask].astype(str).unique()[:10]
        raise ValueError(
            f"{column_name}: non-numeric values found: {', '.join(bad_values)}"
        )

    return converted


def main() -> int:
    df = read_csv(TRAINING_PATH)
    require_columns(df, REQUIRED_COLUMNS, TRAINING_PATH)

    original_row_count = len(df)

    # Make reruns idempotent.
    existing_targets = [
        column for column in TARGET_COLUMNS if column in df.columns
    ]
    if existing_targets:
        df = df.drop(columns=existing_targets)

    away_score = numeric(df["away_score"], "away_score")
    home_score = numeric(df["home_score"], "home_score")
    spread_line = numeric(df["spread_line"], "spread_line")
    total_line = numeric(df["total_line"], "total_line")

    # Final score targets.
    df["margin"] = home_score - away_score
    df["total_points"] = home_score + away_score
    df["home_win"] = (home_score > away_score).astype("int64")

    # ATS target.
    # spread_line convention in this dataset:
    # positive = home favorite
    # negative = home underdog
    df["home_ats_margin"] = df["margin"] - spread_line

    df["home_ats_result"] = "PUSH"
    df.loc[df["home_ats_margin"] > 0, "home_ats_result"] = "WIN"
    df.loc[df["home_ats_margin"] < 0, "home_ats_result"] = "LOSS"

    # Total target.
    df["total_result"] = "PUSH"
    df.loc[df["total_points"] > total_line, "total_result"] = "OVER"
    df.loc[df["total_points"] < total_line, "total_result"] = "UNDER"

    if len(df) != original_row_count:
        raise RuntimeError(
            f"Row count changed during Step 3: "
            f"before={original_row_count} after={len(df)}"
        )

    temp_path = TRAINING_PATH.with_suffix(".step3.tmp.csv")
    df.to_csv(temp_path, index=False, encoding="utf-8")
    temp_path.replace(TRAINING_PATH)

    print(f"Rows processed: {len(df)}")
    print(f"Added columns: {', '.join(TARGET_COLUMNS)}")
    print(f"Wrote: {TRAINING_PATH}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

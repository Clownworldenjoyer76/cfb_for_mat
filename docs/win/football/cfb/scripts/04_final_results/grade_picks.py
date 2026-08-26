#!/usr/bin/env python3
"""
grade_picks.py

Grade final CFB betting selections against ESPN final scores.

Reads:
    docs/win/football/cfb/03_picks/
        week_{week}_CFB_picks.csv

    docs/win/football/cfb/04_final_results/results/
        {season}_{season_type}_{week}.csv

Writes:
    docs/win/football/cfb/04_final_results/graded/
        week_{week}_CFB_graded.csv

    docs/win/football/cfb/04_final_results/graded/
        season_{season}_summary.csv

Grading
-------
Moneyline:
    HOME wins if home_score > away_score
    AWAY wins if away_score > home_score

Spread:
    HOME compares home_score + spread_line vs away_score
    AWAY compares away_score + spread_line vs home_score

Total:
    OVER compares final_total > total_line
    UNDER compares final_total < total_line

Push:
    Exact equality for spread/total.

Canceled/no-contest:
    Selected bets are VOID with 0 units profit.

P&L:
    Flat 1-unit risk per selected bet.

    Win at +200 -> +2.00 units
    Win at -200 -> +0.50 units
    Loss        -> -1.00 units
    Push/Void   ->  0.00 units

Only rows with completed=1 and valid final scores are graded as final.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "cfb-grade-picks-v1-2026-08-26"

SCRIPT_DIR = Path(__file__).resolve().parent
CFB_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_PICKS_DIR = (
    CFB_ROOT
    / "03_picks"
)

DEFAULT_RESULTS_DIR = (
    CFB_ROOT
    / "04_final_results"
    / "results"
)

DEFAULT_OUTPUT_DIR = (
    CFB_ROOT
    / "04_final_results"
    / "graded"
)

PICKS_FILE_RE = re.compile(
    r"^week_(\d+)_CFB_picks\.csv$"
)

GRADED_FILE_RE = re.compile(
    r"^week_(\d+)_CFB_graded\.csv$"
)

REQUIRED_PICK_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "away_team",
    "home_team",
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

REQUIRED_RESULT_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "away_score",
    "home_score",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grade CFB picks against final scores."
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season. Defaults to CFB_SEASON, then 2026.",
    )

    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Optional specific week to grade.",
    )

    parser.add_argument(
        "--picks-dir",
        type=Path,
        default=DEFAULT_PICKS_DIR,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def get_season(
    cli_season: int | None,
) -> int:
    if cli_season is not None:
        return int(
            cli_season
        )

    env = os.getenv(
        "CFB_SEASON",
        "",
    ).strip()

    if env:
        return int(
            env
        )

    return 2026


def clean(
    value: Any,
) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(
            value
        ):
            return ""

    except Exception:
        pass

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
        return text[:-2]

    return text


def parse_float(
    value: Any,
) -> float | None:
    text = clean(
        value
    )

    if not text:
        return None

    try:
        result = float(
            text
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        result
    ):
        return None

    return result


def selected_flag(
    value: Any,
) -> bool:
    text = clean(
        value
    ).casefold()

    return text in {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
    }


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column
        in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label} missing required "
            f"columns: {missing}"
        )


def american_win_profit(
    odds: float,
) -> float:
    if odds == 0:
        raise ValueError(
            "American odds cannot be 0"
        )

    if odds > 0:
        return (
            odds
            / 100.0
        )

    return (
        100.0
        / abs(
            odds
        )
    )


def profit_for_grade(
    grade: str,
    odds: float | None,
) -> float | None:
    if grade == "WIN":
        if odds is None:
            return None

        return american_win_profit(
            odds
        )

    if grade == "LOSS":
        return -1.0

    if grade in {
        "PUSH",
        "VOID",
        "NO_BET",
    }:
        return 0.0

    return None


def is_void_status(
    status: Any,
) -> bool:
    text = clean(
        status
    ).casefold()

    void_tokens = [
        "canceled",
        "cancelled",
        "no contest",
    ]

    return any(
        token in text
        for token
        in void_tokens
    )


def result_completed(
    result_row: pd.Series,
) -> bool:
    if "completed" in result_row.index:
        value = clean(
            result_row.get(
                "completed",
                "",
            )
        ).casefold()

        if value in {
            "1",
            "1.0",
            "true",
            "yes",
        }:
            return True

        if value in {
            "0",
            "0.0",
            "false",
            "no",
        }:
            return False

    status = clean(
        result_row.get(
            "status",
            "",
        )
    ).casefold()

    return (
        status == "final"
        or status.startswith(
            "final/"
        )
        or status.startswith(
            "final "
        )
    )


def grade_moneyline(
    selected: bool,
    selection: str,
    home_score: float | None,
    away_score: float | None,
    completed: bool,
    voided: bool,
) -> str:
    if not selected:
        return "NO_BET"

    if voided:
        return "VOID"

    if (
        not completed
        or home_score is None
        or away_score is None
    ):
        return "PENDING"

    side = clean(
        selection
    ).upper()

    if side == "HOME":
        if home_score > away_score:
            return "WIN"

        if home_score < away_score:
            return "LOSS"

        return "PUSH"

    if side == "AWAY":
        if away_score > home_score:
            return "WIN"

        if away_score < home_score:
            return "LOSS"

        return "PUSH"

    return "INVALID_SELECTION"


def grade_spread(
    selected: bool,
    selection: str,
    line: float | None,
    home_score: float | None,
    away_score: float | None,
    completed: bool,
    voided: bool,
) -> str:
    if not selected:
        return "NO_BET"

    if voided:
        return "VOID"

    if (
        not completed
        or home_score is None
        or away_score is None
    ):
        return "PENDING"

    if line is None:
        return "INVALID_LINE"

    side = clean(
        selection
    ).upper()

    if side == "HOME":
        adjusted = (
            home_score
            + line
            - away_score
        )

    elif side == "AWAY":
        adjusted = (
            away_score
            + line
            - home_score
        )

    else:
        return "INVALID_SELECTION"

    if adjusted > 1e-9:
        return "WIN"

    if adjusted < -1e-9:
        return "LOSS"

    return "PUSH"


def grade_total(
    selected: bool,
    selection: str,
    line: float | None,
    home_score: float | None,
    away_score: float | None,
    completed: bool,
    voided: bool,
) -> str:
    if not selected:
        return "NO_BET"

    if voided:
        return "VOID"

    if (
        not completed
        or home_score is None
        or away_score is None
    ):
        return "PENDING"

    if line is None:
        return "INVALID_LINE"

    final_total = (
        home_score
        + away_score
    )

    side = clean(
        selection
    ).upper()

    if side == "OVER":
        difference = (
            final_total
            - line
        )

    elif side == "UNDER":
        difference = (
            line
            - final_total
        )

    else:
        return "INVALID_SELECTION"

    if difference > 1e-9:
        return "WIN"

    if difference < -1e-9:
        return "LOSS"

    return "PUSH"


def load_results_for_week(
    results_dir: Path,
    season: int,
    week: int,
) -> pd.DataFrame:
    paths = sorted(
        results_dir.glob(
            f"{season}_*_{week}.csv"
        )
    )

    if not paths:
        return pd.DataFrame()

    frames: list[
        pd.DataFrame
    ] = []

    for path in paths:
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
            low_memory=False,
        )

        require_columns(
            frame,
            REQUIRED_RESULT_COLUMNS,
            str(path),
        )

        frame[
            "game_id"
        ] = frame[
            "game_id"
        ].map(
            normalize_game_id
        )

        frames.append(
            frame
        )

    results = pd.concat(
        frames,
        ignore_index=True,
    )

    if results[
        "game_id"
    ].duplicated().any():
        results = results.drop_duplicates(
            "game_id",
            keep="last",
        )

    return results


def grade_week(
    picks_path: Path,
    results_dir: Path,
    output_dir: Path,
    season: int,
    week: int,
) -> Path:
    picks = pd.read_csv(
        picks_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )

    require_columns(
        picks,
        REQUIRED_PICK_COLUMNS,
        str(
            picks_path
        ),
    )

    picks[
        "game_id"
    ] = picks[
        "game_id"
    ].map(
        normalize_game_id
    )

    if picks[
        "game_id"
    ].duplicated().any():
        duplicates = (
            picks.loc[
                picks[
                    "game_id"
                ].duplicated(
                    keep=False
                ),
                "game_id",
            ]
            .head(
                10
            )
            .tolist()
        )

        raise ValueError(
            f"{picks_path}: duplicate "
            f"game_id values: {duplicates}"
        )

    results = load_results_for_week(
        results_dir,
        season,
        week,
    )

    result_lookup = (
        results.set_index(
            "game_id",
            drop=False,
        )
        if not results.empty
        else None
    )

    output = picks.copy()

    final_status: list[str] = []
    final_completed: list[int] = []
    final_away_scores: list[
        float | None
    ] = []
    final_home_scores: list[
        float | None
    ] = []
    final_totals: list[
        float | None
    ] = []
    final_home_margins: list[
        float | None
    ] = []

    ml_grades: list[str] = []
    ml_profits: list[
        float | None
    ] = []

    spread_grades: list[str] = []
    spread_profits: list[
        float | None
    ] = []

    total_grades: list[str] = []
    total_profits: list[
        float | None
    ] = []

    selected_bets_col: list[int] = []
    graded_bets_col: list[int] = []
    wins_col: list[int] = []
    losses_col: list[int] = []
    pushes_col: list[int] = []
    voids_col: list[int] = []
    pending_col: list[int] = []
    net_units_col: list[float] = []

    for _, pick in picks.iterrows():
        game_id = pick[
            "game_id"
        ]

        result_row: pd.Series | None = None

        if (
            result_lookup is not None
            and game_id
            in result_lookup.index
        ):
            located = result_lookup.loc[
                game_id
            ]

            if isinstance(
                located,
                pd.DataFrame,
            ):
                result_row = (
                    located.iloc[-1]
                )

            else:
                result_row = located

        if result_row is None:
            status = ""
            completed = False
            voided = False
            away_score = None
            home_score = None

        else:
            status = clean(
                result_row.get(
                    "status",
                    "",
                )
            )

            completed = (
                result_completed(
                    result_row
                )
            )

            voided = (
                is_void_status(
                    status
                )
            )

            away_score = parse_float(
                result_row.get(
                    "away_score",
                    "",
                )
            )

            home_score = parse_float(
                result_row.get(
                    "home_score",
                    "",
                )
            )

        final_status.append(
            status
        )

        final_completed.append(
            int(
                completed
            )
        )

        final_away_scores.append(
            away_score
        )

        final_home_scores.append(
            home_score
        )

        if (
            away_score is not None
            and home_score is not None
        ):
            final_totals.append(
                away_score
                + home_score
            )

            final_home_margins.append(
                home_score
                - away_score
            )

        else:
            final_totals.append(
                None
            )

            final_home_margins.append(
                None
            )

        ml_selected = selected_flag(
            pick.get(
                "ml_selected",
                "",
            )
        )

        spread_selected = selected_flag(
            pick.get(
                "spread_selected",
                "",
            )
        )

        total_selected = selected_flag(
            pick.get(
                "total_selected",
                "",
            )
        )

        ml_grade = grade_moneyline(
            selected=ml_selected,
            selection=pick.get(
                "ml_selection",
                "",
            ),
            home_score=home_score,
            away_score=away_score,
            completed=completed,
            voided=voided,
        )

        spread_grade = grade_spread(
            selected=spread_selected,
            selection=pick.get(
                "spread_selection",
                "",
            ),
            line=parse_float(
                pick.get(
                    "spread_line",
                    "",
                )
            ),
            home_score=home_score,
            away_score=away_score,
            completed=completed,
            voided=voided,
        )

        total_grade = grade_total(
            selected=total_selected,
            selection=pick.get(
                "total_selection",
                "",
            ),
            line=parse_float(
                pick.get(
                    "total_line",
                    "",
                )
            ),
            home_score=home_score,
            away_score=away_score,
            completed=completed,
            voided=voided,
        )

        ml_profit = profit_for_grade(
            ml_grade,
            parse_float(
                pick.get(
                    "ml_odds_american",
                    "",
                )
            ),
        )

        spread_profit = profit_for_grade(
            spread_grade,
            parse_float(
                pick.get(
                    "spread_odds_american",
                    "",
                )
            ),
        )

        total_profit = profit_for_grade(
            total_grade,
            parse_float(
                pick.get(
                    "total_odds_american",
                    "",
                )
            ),
        )

        ml_grades.append(
            ml_grade
        )

        ml_profits.append(
            ml_profit
        )

        spread_grades.append(
            spread_grade
        )

        spread_profits.append(
            spread_profit
        )

        total_grades.append(
            total_grade
        )

        total_profits.append(
            total_profit
        )

        grades = [
            ml_grade,
            spread_grade,
            total_grade,
        ]

        profits = [
            ml_profit,
            spread_profit,
            total_profit,
        ]

        selected_bets = sum(
            grade != "NO_BET"
            for grade
            in grades
        )

        wins = grades.count(
            "WIN"
        )

        losses = grades.count(
            "LOSS"
        )

        pushes = grades.count(
            "PUSH"
        )

        voids = grades.count(
            "VOID"
        )

        pending = sum(
            grade in {
                "PENDING",
                "INVALID_SELECTION",
                "INVALID_LINE",
            }
            for grade
            in grades
        )

        graded_bets = (
            wins
            + losses
            + pushes
            + voids
        )

        net_units = sum(
            float(
                profit
            )
            for profit
            in profits
            if profit is not None
        )

        selected_bets_col.append(
            selected_bets
        )

        graded_bets_col.append(
            graded_bets
        )

        wins_col.append(
            wins
        )

        losses_col.append(
            losses
        )

        pushes_col.append(
            pushes
        )

        voids_col.append(
            voids
        )

        pending_col.append(
            pending
        )

        net_units_col.append(
            round(
                net_units,
                6,
            )
        )

    output[
        "final_status"
    ] = final_status

    output[
        "final_completed"
    ] = final_completed

    output[
        "final_away_score"
    ] = final_away_scores

    output[
        "final_home_score"
    ] = final_home_scores

    output[
        "final_total"
    ] = final_totals

    output[
        "final_home_margin"
    ] = final_home_margins

    output[
        "ml_grade"
    ] = ml_grades

    output[
        "ml_profit_units"
    ] = ml_profits

    output[
        "spread_grade"
    ] = spread_grades

    output[
        "spread_profit_units"
    ] = spread_profits

    output[
        "total_grade"
    ] = total_grades

    output[
        "total_profit_units"
    ] = total_profits

    output[
        "selected_bets"
    ] = selected_bets_col

    output[
        "graded_bets"
    ] = graded_bets_col

    output[
        "wins"
    ] = wins_col

    output[
        "losses"
    ] = losses_col

    output[
        "pushes"
    ] = pushes_col

    output[
        "voids"
    ] = voids_col

    output[
        "pending_bets"
    ] = pending_col

    output[
        "net_units"
    ] = net_units_col

    output[
        "grading_version"
    ] = SCRIPT_VERSION

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            f"week_{week}_"
            "CFB_graded.csv"
        )
    )

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    output.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        output_path,
    )

    selected_bets = int(
        pd.to_numeric(
            output[
                "selected_bets"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .sum()
    )

    graded_bets = int(
        pd.to_numeric(
            output[
                "graded_bets"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .sum()
    )

    wins = int(
        pd.to_numeric(
            output[
                "wins"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .sum()
    )

    losses = int(
        pd.to_numeric(
            output[
                "losses"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .sum()
    )

    pushes = int(
        pd.to_numeric(
            output[
                "pushes"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .sum()
    )

    voids = int(
        pd.to_numeric(
            output[
                "voids"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .sum()
    )

    pending = int(
        pd.to_numeric(
            output[
                "pending_bets"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .sum()
    )

    units = float(
        pd.to_numeric(
            output[
                "net_units"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .sum()
    )

    print(
        f"WROTE {output_path} "
        f"| games={len(output)} "
        f"| selected={selected_bets} "
        f"| graded={graded_bets} "
        f"| W={wins} "
        f"L={losses} "
        f"P={pushes} "
        f"V={voids} "
        f"pending={pending} "
        f"| units={units:.4f}"
    )

    return output_path


def build_season_summary(
    output_dir: Path,
    season: int,
) -> Path:
    summary_rows: list[
        dict[str, Any]
    ] = []

    for path in sorted(
        output_dir.glob(
            "week_*_CFB_graded.csv"
        )
    ):
        match = GRADED_FILE_RE.fullmatch(
            path.name
        )

        if match is None:
            continue

        week = int(
            match.group(
                1
            )
        )

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        if df.empty:
            continue

        season_values = pd.to_numeric(
            df.get(
                "season"
            ),
            errors="coerce",
        )

        if not season_values.eq(
            season
        ).any():
            continue

        selected = int(
            pd.to_numeric(
                df[
                    "selected_bets"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .sum()
        )

        graded = int(
            pd.to_numeric(
                df[
                    "graded_bets"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .sum()
        )

        wins = int(
            pd.to_numeric(
                df[
                    "wins"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .sum()
        )

        losses = int(
            pd.to_numeric(
                df[
                    "losses"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .sum()
        )

        pushes = int(
            pd.to_numeric(
                df[
                    "pushes"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .sum()
        )

        voids = int(
            pd.to_numeric(
                df[
                    "voids"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .sum()
        )

        pending = int(
            pd.to_numeric(
                df[
                    "pending_bets"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .sum()
        )

        net_units = float(
            pd.to_numeric(
                df[
                    "net_units"
                ],
                errors="coerce",
            )
            .fillna(
                0.0
            )
            .sum()
        )

        roi = (
            net_units
            / graded
            if graded > 0
            else np.nan
        )

        summary_rows.append(
            {
                "season": season,
                "week": week,
                "games": len(
                    df
                ),
                "selected_bets": (
                    selected
                ),
                "graded_bets": (
                    graded
                ),
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "voids": voids,
                "pending_bets": (
                    pending
                ),
                "net_units": round(
                    net_units,
                    6,
                ),
                "roi_per_graded_bet": (
                    round(
                        roi,
                        6,
                    )
                    if math.isfinite(
                        roi
                    )
                    else ""
                ),
            }
        )

    if summary_rows:
        total_selected = sum(
            row[
                "selected_bets"
            ]
            for row
            in summary_rows
        )

        total_graded = sum(
            row[
                "graded_bets"
            ]
            for row
            in summary_rows
        )

        total_wins = sum(
            row[
                "wins"
            ]
            for row
            in summary_rows
        )

        total_losses = sum(
            row[
                "losses"
            ]
            for row
            in summary_rows
        )

        total_pushes = sum(
            row[
                "pushes"
            ]
            for row
            in summary_rows
        )

        total_voids = sum(
            row[
                "voids"
            ]
            for row
            in summary_rows
        )

        total_pending = sum(
            row[
                "pending_bets"
            ]
            for row
            in summary_rows
        )

        total_units = sum(
            float(
                row[
                    "net_units"
                ]
            )
            for row
            in summary_rows
        )

        total_games = sum(
            int(
                row[
                    "games"
                ]
            )
            for row
            in summary_rows
        )

        total_roi = (
            total_units
            / total_graded
            if total_graded > 0
            else np.nan
        )

        summary_rows.append(
            {
                "season": season,
                "week": "TOTAL",
                "games": total_games,
                "selected_bets": (
                    total_selected
                ),
                "graded_bets": (
                    total_graded
                ),
                "wins": total_wins,
                "losses": total_losses,
                "pushes": total_pushes,
                "voids": total_voids,
                "pending_bets": (
                    total_pending
                ),
                "net_units": round(
                    total_units,
                    6,
                ),
                "roi_per_graded_bet": (
                    round(
                        total_roi,
                        6,
                    )
                    if math.isfinite(
                        total_roi
                    )
                    else ""
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows,
        columns=[
            "season",
            "week",
            "games",
            "selected_bets",
            "graded_bets",
            "wins",
            "losses",
            "pushes",
            "voids",
            "pending_bets",
            "net_units",
            "roi_per_graded_bet",
        ],
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / (
            f"season_{season}_"
            "summary.csv"
        )
    )

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    summary.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8",
    )

    os.replace(
        temporary_path,
        output_path,
    )

    print(
        f"WROTE {output_path} "
        f"| weeks={len(summary_rows) - (1 if summary_rows else 0)}"
    )

    return output_path


def main() -> int:
    args = parse_args()

    season = get_season(
        args.season
    )

    print(
        "grade_picks.py "
        f"version={SCRIPT_VERSION}"
    )

    print(
        f"season={season}"
    )

    if not args.picks_dir.is_dir():
        raise FileNotFoundError(
            "Missing picks directory: "
            f"{args.picks_dir}"
        )

    if args.week is not None:
        pick_files = [
            args.picks_dir
            / (
                f"week_{args.week}_"
                "CFB_picks.csv"
            )
        ]

    else:
        pick_files = sorted(
            args.picks_dir.glob(
                "week_*_CFB_picks.csv"
            ),
            key=lambda path: int(
                PICKS_FILE_RE.fullmatch(
                    path.name
                ).group(
                    1
                )
            )
            if PICKS_FILE_RE.fullmatch(
                path.name
            )
            else 10_000,
        )

    processed = 0

    for picks_path in pick_files:
        if not picks_path.is_file():
            continue

        match = PICKS_FILE_RE.fullmatch(
            picks_path.name
        )

        if match is None:
            continue

        week = int(
            match.group(
                1
            )
        )

        preview = pd.read_csv(
            picks_path,
            dtype=str,
            nrows=1,
            encoding="utf-8-sig",
        )

        if preview.empty:
            continue

        file_season = parse_float(
            preview.iloc[0].get(
                "season",
                "",
            )
        )

        if (
            file_season is None
            or int(
                file_season
            )
            != season
        ):
            continue

        grade_week(
            picks_path=(
                picks_path
            ),
            results_dir=(
                args.results_dir
            ),
            output_dir=(
                args.output_dir
            ),
            season=season,
            week=week,
        )

        processed += 1

    build_season_summary(
        args.output_dir,
        season,
    )

    print(
        f"graded_pick_files={processed}"
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
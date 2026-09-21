#!/usr/bin/env python3
"""
Step 15 CFB candidate enrichment engine.

READS:
  docs/win/football/cfb/config/settings.yaml
  docs/win/football/cfb/01_merge/week_{week}_CFB_enriched.csv
  docs/win/football/cfb/00_intake/schedule/weekly/
      week_{week}_CFB_weekly_schedule.csv

WRITES:
  docs/win/football/cfb/02_select/week_{week}_CFB_selected.csv

Weather/travel are NOT loaded or adjusted here. They are already incorporated
upstream into predicted_margin and predicted_total by the projection scripts.

This step does NOT apply betting filters or choose a bet. It preserves the
existing enriched input columns and appends raw candidate metrics for every
available side:

  moneyline: HOME / AWAY
  spread:    HOME / AWAY
  total:     OVER / UNDER

The existing final selection columns are retained for downstream compatibility,
but this step leaves them unselected and marks them as DEFERRED_TO_FILTER.

The *_implied_probability candidate columns contain the no-vig fair market
probability.

EV and full Kelly use the actual offered sportsbook odds.
Kelly is full Kelly capped at settings.yaml selection_defaults.max_kelly.
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

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
CFB_ROOT = SCRIPT_DIR.parents[1]
REPORT_ROOT = CFB_ROOT / "errors"

DEFAULT_SETTINGS_PATH = (
    CFB_ROOT
    / "config"
    / "settings.yaml"
)

CURRENT_WEEK_CONFIG_PATH = (
    CFB_ROOT
    / "config"
    / "current_week.yaml"
)

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


SCRIPT_VERSION = (
    "cfb-selections-v3-rebuild-all-games-2026-09-21"
)

PROBABILITY_EPS = 1e-6
SPREAD_LINE_TOLERANCE = 1e-9
CALCULATION_TOLERANCE = 1e-8

PREDICTION_COLUMNS = [
    "predicted_margin",
    "predicted_total",
    "predicted_home_score",
    "predicted_away_score",
    "home_win_probability",
    "away_win_probability",
    "home_cover_probability",
    "away_cover_probability",
    "over_probability",
    "under_probability",
]

SELECTION_COLUMNS = [
    "ml_selected",
    "ml_selection",
    "ml_selection_reason",
    "ml_odds_american",
    "ml_model_probability",
    "ml_implied_probability",
    "ml_edge",
    "ml_ev",
    "ml_full_kelly",
    "ml_kelly",
    "spread_selected",
    "spread_selection",
    "spread_selection_reason",
    "spread_line",
    "spread_odds_american",
    "spread_model_probability",
    "spread_implied_probability",
    "spread_edge",
    "spread_ev",
    "spread_full_kelly",
    "spread_kelly",
    "total_selected",
    "total_selection",
    "total_selection_reason",
    "total_line",
    "total_odds_american",
    "total_model_probability",
    "total_implied_probability",
    "total_edge",
    "total_ev",
    "total_full_kelly",
    "total_kelly",
]

CANDIDATE_COLUMNS = [
    "ml_home_available",
    "ml_home_odds_american",
    "ml_home_model_probability",
    "ml_home_implied_probability",
    "ml_home_edge",
    "ml_home_ev",
    "ml_home_full_kelly",
    "ml_home_kelly",
    "ml_away_available",
    "ml_away_odds_american",
    "ml_away_model_probability",
    "ml_away_implied_probability",
    "ml_away_edge",
    "ml_away_ev",
    "ml_away_full_kelly",
    "ml_away_kelly",
    "spread_home_available",
    "spread_home_line",
    "spread_home_odds_american",
    "spread_home_model_probability",
    "spread_home_implied_probability",
    "spread_home_edge",
    "spread_home_ev",
    "spread_home_full_kelly",
    "spread_home_kelly",
    "spread_away_available",
    "spread_away_line",
    "spread_away_odds_american",
    "spread_away_model_probability",
    "spread_away_implied_probability",
    "spread_away_edge",
    "spread_away_ev",
    "spread_away_full_kelly",
    "spread_away_kelly",
    "total_over_available",
    "total_over_line",
    "total_over_odds_american",
    "total_over_model_probability",
    "total_over_implied_probability",
    "total_over_edge",
    "total_over_ev",
    "total_over_full_kelly",
    "total_over_kelly",
    "total_under_available",
    "total_under_line",
    "total_under_odds_american",
    "total_under_model_probability",
    "total_under_implied_probability",
    "total_under_edge",
    "total_under_ev",
    "total_under_full_kelly",
    "total_under_kelly",
]

SEASON_TYPE_ALIASES = {
    "1": "pre",
    "2": "reg",
    "3": "post",
    "reg": "reg",
    "regular": "reg",
    "regularseason": "reg",
    "pre": "pre",
    "preseason": "pre",
    "post": "post",
    "postseason": "post",
    "playoff": "post",
    "playoffs": "post",
}


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


def parse_float(value: Any) -> float | None:
    text = clean(value)

    if not text:
        return None

    try:
        number = float(text)

    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def parse_int(value: Any) -> int | None:
    number = parse_float(value)

    if number is None or not float(number).is_integer():
        return None

    return int(number)


def normalize_game_id(value: Any) -> str:
    return re.sub(
        r"\.0$",
        "",
        clean(value),
    )


def normalize_season_type(value: Any) -> str:
    raw = clean(value)
    numeric = parse_float(raw)

    if numeric is not None and float(numeric).is_integer():
        key = str(int(numeric))

        if key in SEASON_TYPE_ALIASES:
            return SEASON_TYPE_ALIASES[key]

    key = re.sub(
        r"[\s_-]+",
        "",
        raw.casefold(),
    )

    return SEASON_TYPE_ALIASES.get(
        key,
        key,
    )


def normalize_bookmaker(
    value: Any,
) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        clean(
            value
        ).casefold(),
    )


def normal_cdf(
    z: float,
) -> float:
    probability = 0.5 * (
        1.0
        + math.erf(
            z
            / math.sqrt(
                2.0
            )
        )
    )

    return float(
        np.clip(
            probability,
            PROBABILITY_EPS,
            1.0
            - PROBABILITY_EPS,
        )
    )


def required_numeric(
    row: pd.Series,
    column: str,
) -> float:
    value = parse_float(
        row.get(
            column,
            "",
        )
    )

    if value is None:
        fail(
            f"game_id={clean(row.get('game_id'))}: "
            f"{column} must be finite; "
            f"found {row.get(column)!r}"
        )

    return value


def resolve_target(
    current_week: dict[str, Any],
    season_override: int | None,
    week_override: int | None,
) -> tuple[
    int,
    int,
    str,
]:
    configured_season = parse_int(
        current_week.get(
            "season"
        )
    )

    configured_week = parse_int(
        current_week.get(
            "week"
        )
    )

    season_type = normalize_season_type(
        current_week.get(
            "season_type"
        )
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

    if (
        season is None
        or season < 1900
    ):
        fail(
            "Invalid target season from "
            f"current_week.yaml/CLI: {season!r}"
        )

    if (
        week is None
        or week <= 0
    ):
        fail(
            "Invalid target week from "
            f"current_week.yaml/CLI: {week!r}"
        )

    if season_type not in {
        "reg",
        "pre",
        "post",
    }:
        fail(
            "Invalid target season_type from "
            "current_week.yaml: "
            f"{current_week.get('season_type')!r}"
        )

    return (
        season,
        week,
        season_type,
    )


def spread_model_probabilities(
    row: pd.Series,
    home_line: float,
) -> tuple[
    float,
    float,
]:
    predicted_margin = required_numeric(
        row,
        "predicted_margin",
    )

    margin_sd = required_numeric(
        row,
        "probability_margin_sd",
    )

    if margin_sd <= 0:
        fail(
            f"game_id={row['game_id']}: "
            "probability_margin_sd must be > 0"
        )

    home_probability = normal_cdf(
        (
            predicted_margin
            + home_line
        )
        / margin_sd
    )

    return (
        home_probability,
        1.0
        - home_probability,
    )


def total_model_probabilities(
    row: pd.Series,
    total_line: float,
) -> tuple[
    float,
    float,
]:
    predicted_total = required_numeric(
        row,
        "predicted_total",
    )

    total_sd = required_numeric(
        row,
        "probability_total_sd",
    )

    if total_sd <= 0:
        fail(
            f"game_id={row['game_id']}: "
            "probability_total_sd must be > 0"
        )

    over_probability = normal_cdf(
        (
            predicted_total
            - total_line
        )
        / total_sd
    )

    return (
        over_probability,
        1.0
        - over_probability,
    )


def count_line_movements(
    working: pd.DataFrame,
    projection_column: str,
    current_column: str,
) -> int:
    count = 0

    for _, row in working.iterrows():
        current = parse_float(
            row.get(
                current_column
            )
        )

        if current is None:
            continue

        projected = parse_float(
            row.get(
                projection_column
            )
        )

        if (
            projected is None
            or not math.isclose(
                projected,
                current,
                rel_tol=0.0,
                abs_tol=SPREAD_LINE_TOLERANCE,
            )
        ):
            count += 1

    return count



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
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        fail(
            f"{label} must contain a YAML mapping: {path}"
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
            f"{label} contains no data rows: {path}"
        )

    return df


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{label} missing required columns: {missing}"
        )


def validate_unique_game_ids(
    df: pd.DataFrame,
    label: str,
) -> None:
    ids = df[
        "game_id"
    ].map(
        normalize_game_id
    )

    if ids.eq("").any():
        fail(
            f"{label} contains blank game_id values"
        )

    if ids.duplicated().any():
        examples = ids[
            ids.duplicated(False)
        ].head(10).tolist()

        fail(
            f"{label} contains duplicate game_id values: {examples}"
        )

    df["game_id"] = ids


def american_to_decimal(
    odds: float,
) -> float:
    if odds == 0:
        fail(
            "American odds cannot be 0"
        )

    if odds > 0:
        return 1.0 + odds / 100.0

    return 1.0 + 100.0 / abs(odds)


def american_implied_probability(
    odds: float,
) -> float:
    return (
        1.0
        / american_to_decimal(odds)
    )


def no_vig_probabilities(
    first_odds: float,
    second_odds: float,
) -> tuple[float, float]:
    first_raw = (
        american_implied_probability(
            first_odds
        )
    )

    second_raw = (
        american_implied_probability(
            second_odds
        )
    )

    total_raw = (
        first_raw
        + second_raw
    )

    if (
        not math.isfinite(total_raw)
        or total_raw <= 0
    ):
        fail(
            "Unable to calculate no-vig probabilities from odds "
            f"{first_odds!r}, {second_odds!r}"
        )

    first_fair = (
        first_raw
        / total_raw
    )

    second_fair = (
        second_raw
        / total_raw
    )

    if not 0.0 <= first_fair <= 1.0:
        fail(
            f"Invalid first no-vig probability: {first_fair}"
        )

    if not 0.0 <= second_fair <= 1.0:
        fail(
            f"Invalid second no-vig probability: {second_fair}"
        )

    return (
        first_fair,
        second_fair,
    )


def calculate_metrics(
    model_probability: float,
    odds_american: float,
    fair_market_probability: float,
) -> dict[str, float]:
    if not 0.0 <= model_probability <= 1.0:
        fail(
            f"Model probability outside [0,1]: {model_probability}"
        )

    if not 0.0 <= fair_market_probability <= 1.0:
        fail(
            "Fair market probability outside [0,1]: "
            f"{fair_market_probability}"
        )

    decimal_odds = (
        american_to_decimal(
            odds_american
        )
    )

    net_win = (
        decimal_odds
        - 1.0
    )

    loss_probability = (
        1.0
        - model_probability
    )

    edge = (
        model_probability
        - fair_market_probability
    )

    ev = (
        model_probability
        * net_win
        - loss_probability
    )

    raw_kelly = (
        (
            net_win
            * model_probability
            - loss_probability
        )
        / net_win
    )

    return {
        "implied_probability":
            fair_market_probability,
        "edge":
            edge,
        "ev":
            ev,
        "full_kelly":
            max(
                0.0,
                raw_kelly,
            ),
    }


def numeric_probability(
    row: pd.Series,
    column: str,
) -> float:
    value = parse_float(
        row[
            column
        ]
    )

    if (
        value is None
        or not 0.0 <= value <= 1.0
    ):
        fail(
            f"game_id={row['game_id']}: "
            f"{column} must be a finite probability in [0,1]; "
            f"found {row[column]!r}"
        )

    return value


def odds_value(
    row: pd.Series,
    column: str,
) -> float | None:
    value = parse_float(
        row.get(
            column,
            "",
        )
    )

    if value is None or value == 0:
        return None

    return value


def make_candidate(
    selection: str,
    model_probability: float,
    odds_american: float,
    fair_market_probability: float,
    *,
    line: float | None = None,
) -> dict[str, Any]:
    return {
        "selection":
            selection,
        "line":
            line,
        "odds_american":
            odds_american,
        "model_probability":
            model_probability,
        **calculate_metrics(
            model_probability,
            odds_american,
            fair_market_probability,
        ),
    }


def deferred_market(
    prefix: str,
    reason: str,
    *,
    line: float | None = None,
) -> dict[str, Any]:
    output = {
        f"{prefix}_selected":
            0,
        f"{prefix}_selection":
            "",
        f"{prefix}_selection_reason":
            reason,
        f"{prefix}_odds_american":
            np.nan,
        f"{prefix}_model_probability":
            np.nan,
        f"{prefix}_implied_probability":
            np.nan,
        f"{prefix}_edge":
            np.nan,
        f"{prefix}_ev":
            np.nan,
        f"{prefix}_full_kelly":
            np.nan,
        f"{prefix}_kelly":
            np.nan,
    }

    if prefix in {
        "spread",
        "total",
    }:
        output[
            f"{prefix}_line"
        ] = (
            np.nan
            if line is None
            else line
        )

    return output


def blank_candidate(
    prefix: str,
    *,
    line: float | None = None,
) -> dict[str, Any]:
    output = {
        f"{prefix}_available":
            0,
        f"{prefix}_odds_american":
            np.nan,
        f"{prefix}_model_probability":
            np.nan,
        f"{prefix}_implied_probability":
            np.nan,
        f"{prefix}_edge":
            np.nan,
        f"{prefix}_ev":
            np.nan,
        f"{prefix}_full_kelly":
            np.nan,
        f"{prefix}_kelly":
            np.nan,
    }

    if (
        prefix.startswith(
            "spread_"
        )
        or prefix.startswith(
            "total_"
        )
    ):
        output[
            f"{prefix}_line"
        ] = (
            np.nan
            if line is None
            else line
        )

    return output


def candidate_columns(
    prefix: str,
    candidate: dict[str, Any],
    *,
    include_line: bool,
) -> dict[str, Any]:
    output = {
        f"{prefix}_available":
            1,
        f"{prefix}_odds_american":
            candidate[
                "odds_american"
            ],
        f"{prefix}_model_probability":
            candidate[
                "model_probability"
            ],
        f"{prefix}_implied_probability":
            candidate[
                "implied_probability"
            ],
        f"{prefix}_edge":
            candidate[
                "edge"
            ],
        f"{prefix}_ev":
            candidate[
                "ev"
            ],
        f"{prefix}_full_kelly":
            candidate[
                "full_kelly"
            ],
        f"{prefix}_kelly":
            candidate[
                "full_kelly"
            ],
    }

    if include_line:
        output[
            f"{prefix}_line"
        ] = candidate[
            "line"
        ]

    return output


def empty_candidate_set(
    reason: str,
) -> dict[str, Any]:
    return {
        **deferred_market(
            "ml",
            reason,
        ),
        **deferred_market(
            "spread",
            reason,
        ),
        **deferred_market(
            "total",
            reason,
        ),
        **blank_candidate(
            "ml_home"
        ),
        **blank_candidate(
            "ml_away"
        ),
        **blank_candidate(
            "spread_home"
        ),
        **blank_candidate(
            "spread_away"
        ),
        **blank_candidate(
            "total_over"
        ),
        **blank_candidate(
            "total_under"
        ),
    }


def evaluate_moneyline(
    row: pd.Series,
) -> dict[str, Any]:
    home_odds = odds_value(
        row,
        "sched_home_moneyline_american",
    )

    away_odds = odds_value(
        row,
        "sched_away_moneyline_american",
    )

    if (
        home_odds is None
        or away_odds is None
    ):
        return {
            **deferred_market(
                "ml",
                "CURRENT_LINE_MISSING",
            ),
            **blank_candidate(
                "ml_home"
            ),
            **blank_candidate(
                "ml_away"
            ),
        }

    (
        home_fair,
        away_fair,
    ) = no_vig_probabilities(
        home_odds,
        away_odds,
    )

    home_candidate = make_candidate(
        "HOME",
        numeric_probability(
            row,
            "home_win_probability",
        ),
        home_odds,
        home_fair,
    )

    away_candidate = make_candidate(
        "AWAY",
        numeric_probability(
            row,
            "away_win_probability",
        ),
        away_odds,
        away_fair,
    )

    return {
        **deferred_market(
            "ml",
            "DEFERRED_TO_FILTER",
        ),
        **candidate_columns(
            "ml_home",
            home_candidate,
            include_line=False,
        ),
        **candidate_columns(
            "ml_away",
            away_candidate,
            include_line=False,
        ),
    }


def evaluate_spread(
    row: pd.Series,
) -> dict[str, Any]:
    home_line = parse_float(
        row.get(
            "sched_home_spread",
            "",
        )
    )

    away_line = parse_float(
        row.get(
            "sched_away_spread",
            "",
        )
    )

    home_odds = odds_value(
        row,
        "sched_home_spread_american",
    )

    away_odds = odds_value(
        row,
        "sched_away_spread_american",
    )

    if (
        home_line is None
        and away_line is None
    ):
        return {
            **deferred_market(
                "spread",
                "CURRENT_LINE_MISSING",
            ),
            **blank_candidate(
                "spread_home"
            ),
            **blank_candidate(
                "spread_away"
            ),
        }

    if (
        home_line is None
        or away_line is None
    ):
        fail(
            f"game_id={row['game_id']}: "
            "spread line pair is partial; "
            f"home={row.get('sched_home_spread')!r} "
            f"away={row.get('sched_away_spread')!r}"
        )

    if not math.isclose(
        home_line
        + away_line,
        0.0,
        rel_tol=0.0,
        abs_tol=SPREAD_LINE_TOLERANCE,
    ):
        fail(
            f"game_id={row['game_id']}: "
            "home/away spread lines are not opposites; "
            f"home={home_line} away={away_line}"
        )

    if (
        home_odds is None
        or away_odds is None
    ):
        return {
            **deferred_market(
                "spread",
                "CURRENT_LINE_MISSING",
                line=home_line,
            ),
            **blank_candidate(
                "spread_home",
                line=home_line,
            ),
            **blank_candidate(
                "spread_away",
                line=away_line,
            ),
        }

    (
        home_fair,
        away_fair,
    ) = no_vig_probabilities(
        home_odds,
        away_odds,
    )

    (
        home_model_probability,
        away_model_probability,
    ) = spread_model_probabilities(
        row,
        home_line,
    )

    home_candidate = make_candidate(
        "HOME",
        home_model_probability,
        home_odds,
        home_fair,
        line=home_line,
    )

    away_candidate = make_candidate(
        "AWAY",
        away_model_probability,
        away_odds,
        away_fair,
        line=away_line,
    )

    return {
        **deferred_market(
            "spread",
            "DEFERRED_TO_FILTER",
        ),
        **candidate_columns(
            "spread_home",
            home_candidate,
            include_line=True,
        ),
        **candidate_columns(
            "spread_away",
            away_candidate,
            include_line=True,
        ),
    }



def evaluate_total(
    row: pd.Series,
) -> dict[str, Any]:
    total_line = parse_float(
        row.get(
            "sched_total",
            "",
        )
    )

    over_odds = odds_value(
        row,
        "sched_over_american",
    )

    under_odds = odds_value(
        row,
        "sched_under_american",
    )

    if (
        total_line is None
        or over_odds is None
        or under_odds is None
    ):
        return {
            **deferred_market(
                "total",
                "CURRENT_LINE_MISSING",
                line=total_line,
            ),
            **blank_candidate(
                "total_over",
                line=total_line,
            ),
            **blank_candidate(
                "total_under",
                line=total_line,
            ),
        }

    (
        over_fair,
        under_fair,
    ) = no_vig_probabilities(
        over_odds,
        under_odds,
    )

    (
        over_model_probability,
        under_model_probability,
    ) = total_model_probabilities(
        row,
        total_line,
    )

    over_candidate = make_candidate(
        "OVER",
        over_model_probability,
        over_odds,
        over_fair,
        line=total_line,
    )

    under_candidate = make_candidate(
        "UNDER",
        under_model_probability,
        under_odds,
        under_fair,
        line=total_line,
    )

    return {
        **deferred_market(
            "total",
            "DEFERRED_TO_FILTER",
            line=total_line,
        ),
        **candidate_columns(
            "total_over",
            over_candidate,
            include_line=True,
        ),
        **candidate_columns(
            "total_under",
            under_candidate,
            include_line=True,
        ),
    }



def validate_probability_pairs(
    df: pd.DataFrame,
) -> None:
    pairs = [
        (
            "home_win_probability",
            "away_win_probability",
            "moneyline",
        ),
        (
            "home_cover_probability",
            "away_cover_probability",
            "spread",
        ),
        (
            "over_probability",
            "under_probability",
            "total",
        ),
    ]

    for (
        first,
        second,
        label,
    ) in pairs:
        a = pd.to_numeric(
            df[first],
            errors="coerce",
        )

        b = pd.to_numeric(
            df[second],
            errors="coerce",
        )

        if (
            a.isna().any()
            or b.isna().any()
        ):
            fail(
                f"{label} probability columns contain "
                "blank/non-numeric values"
            )

        if (
            (
                (a < 0)
                | (a > 1)
                | (b < 0)
                | (b > 1)
            ).any()
        ):
            fail(
                f"{label} probability outside [0,1]"
            )

        if not np.allclose(
            a.to_numpy(
                dtype=float
            )
            + b.to_numpy(
                dtype=float
            ),
            1.0,
            rtol=0,
            atol=1e-9,
        ):
            fail(
                f"{label} complementary probabilities "
                "do not sum to 1"
            )


def validate_settings(
    settings: dict[str, Any],
) -> str:
    forbidden_keys = [
        key
        for key in (
            "season",
            "week",
            "season_type",
        )
        if key in settings
    ]

    if forbidden_keys:
        fail(
            "settings.yaml must not define "
            "season, week, or season_type. "
            "Pipeline target configuration belongs "
            "in current_week.yaml. "
            f"Found: {forbidden_keys}"
        )

    sportsbook = clean(
        settings.get(
            "sportsbook"
        )
    )

    if not sportsbook:
        fail(
            "settings.yaml sportsbook is required"
        )

    odds_format = clean(
        settings.get(
            "odds_format",
            "american",
        )
    ).casefold()

    if odds_format != "american":
        fail(
            "selections.py requires odds_format: american"
        )

    return sportsbook


def validate_combined(
    df: pd.DataFrame,
    season: int,
    week: int,
    season_type: str,
    label: str,
) -> None:
    require_columns(
        df,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "away_team",
            "home_team",
            *PREDICTION_COLUMNS,
        ],
        label,
    )

    validate_unique_game_ids(
        df,
        label,
    )

    seasons = {
        parse_int(
            value
        )
        for value in df[
            "season"
        ]
    }

    weeks = {
        parse_int(
            value
        )
        for value in df[
            "week"
        ]
    }

    types = {
        normalize_season_type(
            value
        )
        for value in df[
            "season_type"
        ]
    }

    if seasons != {
        season
    }:
        fail(
            f"{label}: expected only season={season}; "
            f"found {seasons}"
        )

    if weeks != {
        week
    }:
        fail(
            f"{label}: expected only week={week}; "
            f"found {weeks}"
        )

    if types != {
        season_type
    }:
        fail(
            f"{label}: expected season_type={season_type!r}; "
            f"found {types}"
        )

    validate_probability_pairs(
        df
    )


def merge_schedule(
    combined: pd.DataFrame,
    schedule: pd.DataFrame,
    season: int,
    week: int,
    season_type: str,
    sportsbook: str,
) -> pd.DataFrame:
    require_columns(
        schedule,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "away_team",
            "home_team",
            "neutral_site",
            "roof",
            "bookmaker",
            "home_moneyline_american",
            "away_moneyline_american",
            "home_spread",
            "away_spread",
            "home_spread_american",
            "away_spread_american",
            "total",
            "over_american",
            "under_american",
            "odds_available",
        ],
        "weekly schedule",
    )

    validate_unique_game_ids(
        schedule,
        "weekly schedule",
    )

    season_values = pd.to_numeric(
        schedule[
            "season"
        ],
        errors="coerce",
    )

    week_values = pd.to_numeric(
        schedule[
            "week"
        ],
        errors="coerce",
    )

    type_values = schedule[
        "season_type"
    ].map(
        normalize_season_type
    )

    schedule = schedule.loc[
        season_values.eq(
            season
        )
        & week_values.eq(
            week
        )
        & type_values.eq(
            season_type
        )
    ].copy()

    if schedule.empty:
        fail(
            "Weekly schedule has no rows for "
            f"season={season}, "
            f"week={week}, "
            f"season_type={season_type}"
        )

    configured_book = normalize_bookmaker(
        sportsbook
    )

    odds_available = pd.to_numeric(
        schedule[
            "odds_available"
        ],
        errors="coerce",
    ).fillna(
        0
    )

    bad_book = (
        schedule[
            "bookmaker"
        ]
        .map(
            normalize_bookmaker
        )
        .ne(
            configured_book
        )
        & odds_available.eq(
            1
        )
    )

    if bad_book.any():
        examples = (
            schedule.loc[
                bad_book,
                [
                    "game_id",
                    "bookmaker",
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
            "Weekly schedule bookmaker does not match settings "
            f"sportsbook {sportsbook!r}: {examples}"
        )

    base_ids = set(
        combined[
            "game_id"
        ].map(
            normalize_game_id
        )
    )

    schedule_ids = set(
        schedule[
            "game_id"
        ].map(
            normalize_game_id
        )
    )

    missing = sorted(
        base_ids
        - schedule_ids
    )

    unexpected = sorted(
        schedule_ids
        - base_ids
    )

    if (
        missing
        or unexpected
    ):
        fail(
            "Weekly schedule game coverage does not match "
            "the projected input; "
            f"missing_count={len(missing)} "
            f"unexpected_count={len(unexpected)} "
            f"missing_examples={missing[:10]} "
            f"unexpected_examples={unexpected[:10]}"
        )

    schedule_lookup = schedule.set_index(
        "game_id",
        drop=False,
    )

    identity_mismatches: list[
        dict[
            str,
            str,
        ]
    ] = []

    for _, row in combined.iterrows():
        game_id = normalize_game_id(
            row.get(
                "game_id"
            )
        )

        schedule_row = schedule_lookup.loc[
            game_id
        ]

        expected_away = clean(
            row.get(
                "away_team"
            )
        )

        expected_home = clean(
            row.get(
                "home_team"
            )
        )

        actual_away = clean(
            schedule_row.get(
                "away_team"
            )
        )

        actual_home = clean(
            schedule_row.get(
                "home_team"
            )
        )

        if (
            actual_away != expected_away
            or actual_home != expected_home
        ):
            identity_mismatches.append(
                {
                    "game_id":
                        game_id,
                    "projected_away":
                        expected_away,
                    "schedule_away":
                        actual_away,
                    "projected_home":
                        expected_home,
                    "schedule_home":
                        actual_home,
                }
            )

    if identity_mismatches:
        fail(
            "Weekly schedule team identity does not match "
            "the projected input; "
            f"count={len(identity_mismatches)} "
            f"examples={identity_mismatches[:10]}"
        )

    columns = [
        "game_id",
        "neutral_site",
        "roof",
        "bookmaker",
        "home_moneyline_american",
        "away_moneyline_american",
        "home_spread",
        "away_spread",
        "home_spread_american",
        "away_spread_american",
        "total",
        "over_american",
        "under_american",
        "odds_available",
    ]

    source = schedule[
        columns
    ].copy()

    source = source.rename(
        columns={
            column:
                f"sched_{column}"
            for column in columns
            if column != "game_id"
        }
    )

    return combined.merge(
        source,
        on="game_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )



def build_output(
    original: pd.DataFrame,
    working: pd.DataFrame,
    max_kelly: float,
) -> pd.DataFrame:
    candidate_rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for _, row in working.iterrows():
        odds_available = (
            parse_int(
                row.get(
                    "sched_odds_available",
                    "",
                )
            )
            or 0
        )

        if odds_available != 1:
            result = empty_candidate_set(
                "CURRENT_ODDS_UNAVAILABLE"
            )

        else:
            result = {
                **evaluate_moneyline(
                    row
                ),
                **evaluate_spread(
                    row
                ),
                **evaluate_total(
                    row
                ),
            }

        candidate_rows.append(
            {
                "game_id":
                    row[
                        "game_id"
                    ],
                **result,
            }
        )

    appended_columns = (
        SELECTION_COLUMNS
        + CANDIDATE_COLUMNS
    )

    candidate_frame = pd.DataFrame(
        candidate_rows,
        columns=[
            "game_id",
            *appended_columns,
        ],
    )

    for prefix in [
        "ml_home",
        "ml_away",
        "spread_home",
        "spread_away",
        "total_over",
        "total_under",
    ]:
        candidate_frame[
            f"{prefix}_kelly"
        ] = (
            pd.to_numeric(
                candidate_frame[
                    f"{prefix}_full_kelly"
                ],
                errors="coerce",
            )
            .clip(
                lower=0.0,
                upper=max_kelly,
            )
        )

    if len(
        candidate_frame
    ) != len(
        original
    ):
        fail(
            "Internal candidate row-count mismatch"
        )

    validate_unique_game_ids(
        candidate_frame,
        "candidate results",
    )

    original_ids = set(
        original[
            "game_id"
        ]
    )

    candidate_ids = set(
        candidate_frame[
            "game_id"
        ]
    )

    if candidate_ids != original_ids:
        missing_ids = sorted(
            original_ids
            - candidate_ids
        )

        extra_ids = sorted(
            candidate_ids
            - original_ids
        )

        fail(
            "Candidate game_id mismatch: "
            f"missing={missing_ids[:10]} "
            f"extra={extra_ids[:10]}"
        )

    candidate_frame = (
        original[
            [
                "game_id"
            ]
        ]
        .merge(
            candidate_frame,
            on="game_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
    )

    output = original.copy()

    for column in appended_columns:
        output[
            column
        ] = candidate_frame[
            column
        ].to_numpy()

    return output



def assert_close(
    actual: float,
    expected: float,
    label: str,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=CALCULATION_TOLERANCE,
    ):
        fail(
            f"{label} mismatch: "
            f"actual={actual} expected={expected}"
        )


def candidate_available(
    row: pd.Series,
    prefix: str,
) -> int:
    value = parse_int(
        row.get(
            f"{prefix}_available"
        )
    )

    if value not in {
        0,
        1,
    }:
        fail(
            f"game_id={row['game_id']}: "
            f"{prefix}_available must be 0 or 1"
        )

    return value


def validate_candidate_side(
    row: pd.Series,
    prefix: str,
    expected_model_probability: float,
    expected_fair_probability: float,
    max_kelly: float,
) -> None:
    odds = required_numeric(
        row,
        f"{prefix}_odds_american",
    )

    if odds == 0:
        fail(
            f"game_id={row['game_id']}: "
            f"{prefix}_odds_american cannot be 0"
        )

    model_probability = required_numeric(
        row,
        f"{prefix}_model_probability",
    )

    implied_probability = required_numeric(
        row,
        f"{prefix}_implied_probability",
    )

    edge = required_numeric(
        row,
        f"{prefix}_edge",
    )

    ev = required_numeric(
        row,
        f"{prefix}_ev",
    )

    full_kelly = required_numeric(
        row,
        f"{prefix}_full_kelly",
    )

    kelly = required_numeric(
        row,
        f"{prefix}_kelly",
    )

    assert_close(
        model_probability,
        expected_model_probability,
        (
            f"game_id={row['game_id']} "
            f"{prefix} model probability"
        ),
    )

    assert_close(
        implied_probability,
        expected_fair_probability,
        (
            f"game_id={row['game_id']} "
            f"{prefix} no-vig probability"
        ),
    )

    expected_metrics = calculate_metrics(
        expected_model_probability,
        odds,
        expected_fair_probability,
    )

    assert_close(
        edge,
        expected_metrics[
            "edge"
        ],
        (
            f"game_id={row['game_id']} "
            f"{prefix} edge"
        ),
    )

    assert_close(
        ev,
        expected_metrics[
            "ev"
        ],
        (
            f"game_id={row['game_id']} "
            f"{prefix} EV"
        ),
    )

    assert_close(
        full_kelly,
        expected_metrics[
            "full_kelly"
        ],
        (
            f"game_id={row['game_id']} "
            f"{prefix} full Kelly"
        ),
    )

    expected_kelly = min(
        expected_metrics[
            "full_kelly"
        ],
        max_kelly,
    )

    assert_close(
        kelly,
        expected_kelly,
        (
            f"game_id={row['game_id']} "
            f"{prefix} capped Kelly"
        ),
    )


def validate_candidate_math(
    df: pd.DataFrame,
    max_kelly: float,
) -> None:
    for _, row in df.iterrows():
        game_id = clean(
            row.get(
                "game_id"
            )
        )

        ml_home_available = candidate_available(
            row,
            "ml_home",
        )

        ml_away_available = candidate_available(
            row,
            "ml_away",
        )

        if (
            ml_home_available
            != ml_away_available
        ):
            fail(
                f"game_id={game_id}: "
                "moneyline side availability mismatch"
            )

        if ml_home_available == 1:
            home_odds = required_numeric(
                row,
                "ml_home_odds_american",
            )

            away_odds = required_numeric(
                row,
                "ml_away_odds_american",
            )

            (
                home_fair,
                away_fair,
            ) = no_vig_probabilities(
                home_odds,
                away_odds,
            )

            home_model = numeric_probability(
                row,
                "home_win_probability",
            )

            away_model = numeric_probability(
                row,
                "away_win_probability",
            )

            validate_candidate_side(
                row,
                "ml_home",
                home_model,
                home_fair,
                max_kelly,
            )

            validate_candidate_side(
                row,
                "ml_away",
                away_model,
                away_fair,
                max_kelly,
            )

        spread_home_available = candidate_available(
            row,
            "spread_home",
        )

        spread_away_available = candidate_available(
            row,
            "spread_away",
        )

        if (
            spread_home_available
            != spread_away_available
        ):
            fail(
                f"game_id={game_id}: "
                "spread side availability mismatch"
            )

        if spread_home_available == 1:
            home_line = required_numeric(
                row,
                "spread_home_line",
            )

            away_line = required_numeric(
                row,
                "spread_away_line",
            )

            if not math.isclose(
                home_line
                + away_line,
                0.0,
                rel_tol=0.0,
                abs_tol=SPREAD_LINE_TOLERANCE,
            ):
                fail(
                    f"game_id={game_id}: "
                    "serialized spread lines are not opposites"
                )

            home_odds = required_numeric(
                row,
                "spread_home_odds_american",
            )

            away_odds = required_numeric(
                row,
                "spread_away_odds_american",
            )

            (
                home_fair,
                away_fair,
            ) = no_vig_probabilities(
                home_odds,
                away_odds,
            )

            (
                home_model,
                away_model,
            ) = spread_model_probabilities(
                row,
                home_line,
            )

            validate_candidate_side(
                row,
                "spread_home",
                home_model,
                home_fair,
                max_kelly,
            )

            validate_candidate_side(
                row,
                "spread_away",
                away_model,
                away_fair,
                max_kelly,
            )

        total_over_available = candidate_available(
            row,
            "total_over",
        )

        total_under_available = candidate_available(
            row,
            "total_under",
        )

        if (
            total_over_available
            != total_under_available
        ):
            fail(
                f"game_id={game_id}: "
                "total side availability mismatch"
            )

        if total_over_available == 1:
            over_line = required_numeric(
                row,
                "total_over_line",
            )

            under_line = required_numeric(
                row,
                "total_under_line",
            )

            if not math.isclose(
                over_line,
                under_line,
                rel_tol=0.0,
                abs_tol=SPREAD_LINE_TOLERANCE,
            ):
                fail(
                    f"game_id={game_id}: "
                    "serialized total lines disagree"
                )

            over_odds = required_numeric(
                row,
                "total_over_odds_american",
            )

            under_odds = required_numeric(
                row,
                "total_under_odds_american",
            )

            (
                over_fair,
                under_fair,
            ) = no_vig_probabilities(
                over_odds,
                under_odds,
            )

            (
                over_model,
                under_model,
            ) = total_model_probabilities(
                row,
                over_line,
            )

            validate_candidate_side(
                row,
                "total_over",
                over_model,
                over_fair,
                max_kelly,
            )

            validate_candidate_side(
                row,
                "total_under",
                under_model,
                under_fair,
                max_kelly,
            )


def validate_output_frame(
    output: pd.DataFrame,
    original: pd.DataFrame,
    expected_columns: list[str],
    max_kelly: float,
) -> None:
    if list(
        output.columns
    ) != expected_columns:
        fail(
            "Final candidate column order/integrity check failed"
        )

    if len(
        output
    ) != len(
        original
    ):
        fail(
            "Final candidate row count does not match input"
        )

    validate_unique_game_ids(
        output,
        "candidate output",
    )

    expected_ids = original[
        "game_id"
    ].map(
        normalize_game_id
    ).tolist()

    actual_ids = output[
        "game_id"
    ].map(
        normalize_game_id
    ).tolist()

    if actual_ids != expected_ids:
        fail(
            "game_id order changed during candidate processing"
        )

    expected_away = original[
        "away_team"
    ].map(
        clean
    ).tolist()

    actual_away = output[
        "away_team"
    ].map(
        clean
    ).tolist()

    if actual_away != expected_away:
        fail(
            "away_team changed during candidate processing"
        )

    expected_home = original[
        "home_team"
    ].map(
        clean
    ).tolist()

    actual_home = output[
        "home_team"
    ].map(
        clean
    ).tolist()

    if actual_home != expected_home:
        fail(
            "home_team changed during candidate processing"
        )

    validate_probability_pairs(
        output
    )

    validate_candidate_math(
        output,
        max_kelly,
    )


def write_atomic_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    original: pd.DataFrame,
    expected_columns: list[str],
    max_kelly: float,
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
            df.to_csv(
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

        validate_output_frame(
            serialized,
            original,
            expected_columns,
            max_kelly,
        )

        new_bytes = temporary.read_bytes()

        if (
            path.is_file()
            and path.read_bytes()
            == new_bytes
        ):
            temporary.unlink(
                missing_ok=True
            )

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



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--week",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def run(
    report: PipelineReporter,
    args: argparse.Namespace,
) -> int:
    settings_path = args.settings.resolve()

    report.add_input(
        settings_path
    )

    report.add_input(
        CURRENT_WEEK_CONFIG_PATH
    )

    settings = read_yaml(
        settings_path,
        "settings config",
    )

    current_week = read_yaml(
        CURRENT_WEEK_CONFIG_PATH,
        "current-week config",
    )

    (
        season,
        week,
        season_type,
    ) = resolve_target(
        current_week,
        args.season,
        args.week,
    )

    selection_defaults = settings.get(
        "selection_defaults"
    )

    if not isinstance(
        selection_defaults,
        dict,
    ):
        fail(
            "settings.yaml must contain selection_defaults"
        )

    max_kelly = parse_float(
        selection_defaults.get(
            "max_kelly"
        )
    )

    if (
        max_kelly is None
        or max_kelly < 0
    ):
        fail(
            "settings.yaml "
            "selection_defaults.max_kelly "
            "must be a non-negative number"
        )

    sportsbook = validate_settings(
        settings,
    )

    report.season = season
    report.week = week

    input_path = (
        args.input.resolve()
        if args.input is not None
        else (
            CFB_ROOT
            / "01_merge"
            / f"week_{week}_CFB_enriched.csv"
        )
    )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else (
            CFB_ROOT
            / "02_select"
            / f"week_{week}_CFB_selected.csv"
        )
    )

    schedule_path = (
        CFB_ROOT
        / "00_intake"
        / "schedule"
        / "weekly"
        / f"week_{week}_CFB_weekly_schedule.csv"
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
            "season_type":
                season_type,
            "sportsbook":
                sportsbook,
            "max_kelly":
                float(
                    max_kelly
                ),
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
            "output_modified":
                False,
        }
    )

    if output_path == input_path:
        fail(
            "Candidate output path must differ "
            "from the input path; selections.py "
            "will not overwrite a file it reads."
        )

    combined = read_csv(
        input_path,
        "projected combined enriched file",
    )

    prior_output_columns = [
        column
        for column in (
            SELECTION_COLUMNS
            + CANDIDATE_COLUMNS
        )
        if column in combined.columns
    ]

    if prior_output_columns:
        combined = combined.drop(
            columns=prior_output_columns
        )

    validate_combined(
        combined,
        season,
        week,
        season_type,
        str(
            input_path
        ),
    )

    require_columns(
        combined,
        [
            "home_spread",
            "total",
            "probability_margin_sd",
            "probability_total_sd",
        ],
        "projected combined enriched file",
    )

    report.set_rows(
        rows_in=len(
            combined
        ),
    )

    schedule = read_csv(
        schedule_path,
        "weekly schedule",
    )

    working = merge_schedule(
        combined.copy(),
        schedule,
        season,
        week,
        season_type,
        sportsbook,
    )

    spread_line_repriced_games = count_line_movements(
        working,
        "home_spread",
        "sched_home_spread",
    )

    total_line_repriced_games = count_line_movements(
        working,
        "total",
        "sched_total",
    )

    output = build_output(
        combined,
        working,
        max_kelly,
    )

    expected_columns = (
        list(
            combined.columns
        )
        + SELECTION_COLUMNS
        + CANDIDATE_COLUMNS
    )

    validate_output_frame(
        output,
        combined,
        expected_columns,
        max_kelly,
    )

    output_modified = write_atomic_csv(
        output,
        output_path,
        original=combined,
        expected_columns=expected_columns,
        max_kelly=max_kelly,
    )

    candidate_counts: dict[
        str,
        int,
    ] = {}

    for prefix in [
        "ml_home",
        "ml_away",
        "spread_home",
        "spread_away",
        "total_over",
        "total_under",
    ]:
        candidate_counts[
            prefix
        ] = int(
            pd.to_numeric(
                output[
                    f"{prefix}_available"
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .sum()
        )

    odds_available_games = int(
        pd.to_numeric(
            working[
                "sched_odds_available"
            ],
            errors="coerce",
        )
        .fillna(
            0
        )
        .eq(
            1
        )
        .sum()
    )

    ml_current_line_missing = int(
        output[
            "ml_selection_reason"
        ].eq(
            "CURRENT_LINE_MISSING"
        ).sum()
    )

    spread_current_line_missing = int(
        output[
            "spread_selection_reason"
        ].eq(
            "CURRENT_LINE_MISSING"
        ).sum()
    )

    total_current_line_missing = int(
        output[
            "total_selection_reason"
        ].eq(
            "CURRENT_LINE_MISSING"
        ).sum()
    )

    current_odds_unavailable_games = int(
        output[
            "ml_selection_reason"
        ].eq(
            "CURRENT_ODDS_UNAVAILABLE"
        ).sum()
    )

    report.set_rows(
        rows_out=len(
            output
        ),
    )

    report.update_details(
        {
            "candidate_game_count":
                len(
                    output
                ),
            "odds_available_games":
                odds_available_games,
            "current_odds_unavailable_games":
                current_odds_unavailable_games,
            "moneyline_current_line_missing_games":
                ml_current_line_missing,
            "spread_current_line_missing_games":
                spread_current_line_missing,
            "total_current_line_missing_games":
                total_current_line_missing,
            "spread_probability_recomputed_games":
                candidate_counts[
                    "spread_home"
                ],
            "total_probability_recomputed_games":
                candidate_counts[
                    "total_over"
                ],
            "spread_line_repriced_games":
                spread_line_repriced_games,
            "total_line_repriced_games":
                total_line_repriced_games,
            "ml_home_candidates":
                candidate_counts[
                    "ml_home"
                ],
            "ml_away_candidates":
                candidate_counts[
                    "ml_away"
                ],
            "spread_home_candidates":
                candidate_counts[
                    "spread_home"
                ],
            "spread_away_candidates":
                candidate_counts[
                    "spread_away"
                ],
            "total_over_candidates":
                candidate_counts[
                    "total_over"
                ],
            "total_under_candidates":
                candidate_counts[
                    "total_under"
                ],
            "output_modified":
                output_modified,
        }
    )

    print(
        "selections.py "
        f"version={SCRIPT_VERSION}"
    )

    print(
        "CFB candidate enrichment complete: "
        f"season={season} "
        f"week={week} "
        f"games={len(output)}"
    )

    print(
        "spread_probability_recomputed_games="
        f"{candidate_counts['spread_home']}"
    )

    print(
        "total_probability_recomputed_games="
        f"{candidate_counts['total_over']}"
    )

    print(
        "spread_line_repriced_games="
        f"{spread_line_repriced_games}"
    )

    print(
        "total_line_repriced_games="
        f"{total_line_repriced_games}"
    )

    for prefix in [
        "ml_home",
        "ml_away",
        "spread_home",
        "spread_away",
        "total_over",
        "total_under",
    ]:
        print(
            f"{prefix}_candidates="
            f"{candidate_counts[prefix]}"
        )

    print(
        "output_modified="
        f"{'yes' if output_modified else 'no'}"
    )

    print(
        f"Updated: {output_path}"
    )

    return 0


def main() -> int:
    args = parse_args()

    with PipelineReporter(
        script=__file__,
        stage="02_select",
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
            "selection_scope":
                "candidate_enrichment",
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

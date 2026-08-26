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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CFB_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SETTINGS_PATH = CFB_ROOT / "config/settings.yaml"

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


def normalize_bookmaker(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        clean(value).casefold(),
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

    if any(
        value is None
        for value in [
            home_line,
            away_line,
            home_odds,
            away_odds,
        ]
    ):
        return {
            **deferred_market(
                "spread",
                "CURRENT_LINE_MISSING",
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

    home_candidate = make_candidate(
        "HOME",
        numeric_probability(
            row,
            "home_cover_probability",
        ),
        home_odds,
        home_fair,
        line=home_line,
    )

    away_candidate = make_candidate(
        "AWAY",
        numeric_probability(
            row,
            "away_cover_probability",
        ),
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

    over_candidate = make_candidate(
        "OVER",
        numeric_probability(
            row,
            "over_probability",
        ),
        over_odds,
        over_fair,
        line=total_line,
    )

    under_candidate = make_candidate(
        "UNDER",
        numeric_probability(
            row,
            "under_probability",
        ),
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
    season_override: int | None,
    week_override: int | None,
) -> tuple[
    int,
    int,
    str,
    str,
]:
    season = (
        season_override
        if season_override is not None
        else parse_int(
            settings.get(
                "season"
            )
        )
    )

    week = (
        week_override
        if week_override is not None
        else parse_int(
            settings.get(
                "week"
            )
        )
    )

    if season is None or season < 1900:
        fail(
            f"Invalid season: "
            f"{settings.get('season')!r}"
        )

    if week is None or week <= 0:
        fail(
            f"Invalid week: "
            f"{settings.get('week')!r}"
        )

    season_type = normalize_season_type(
        settings.get(
            "season_type",
            "reg",
        )
    )

    if season_type not in {
        "reg",
        "pre",
        "post",
    }:
        fail(
            "Unsupported season_type: "
            f"{settings.get('season_type')!r}"
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

    return (
        season,
        week,
        season_type,
        sportsbook,
    )


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

    configured_book = (
        normalize_bookmaker(
            sportsbook
        )
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
        ]
    )

    schedule_ids = set(
        schedule[
            "game_id"
        ]
    )

    missing = sorted(
        base_ids
        - schedule_ids
    )

    if missing:
        fail(
            "Weekly schedule missing "
            f"{len(missing)} projected games; "
            f"examples={missing[:10]}"
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


def write_atomic_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix
        + ".tmp"
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

    args = parser.parse_args()

    settings = read_yaml(
        args.settings.resolve(),
        "settings config",
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

    (
        season,
        week,
        season_type,
        sportsbook,
    ) = validate_settings(
        settings,
        args.season,
        args.week,
    )

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

    schedule_path = (
        CFB_ROOT
        / "00_intake"
        / "schedule"
        / "weekly"
        / f"week_{week}_CFB_weekly_schedule.csv"
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

    if list(
        output.columns
    ) != expected_columns:
        fail(
            "Final candidate column "
            "order/integrity check failed"
        )

    if (
        output[
            "game_id"
        ].tolist()
        != combined[
            "game_id"
        ].tolist()
    ):
        fail(
            "game_id order changed during "
            "candidate processing"
        )

    if (
        output[
            "away_team"
        ].tolist()
        != combined[
            "away_team"
        ].tolist()
    ):
        fail(
            "away_team changed during "
            "candidate processing"
        )

    if (
        output[
            "home_team"
        ].tolist()
        != combined[
            "home_team"
        ].tolist()
    ):
        fail(
            "home_team changed during "
            "candidate processing"
        )

    write_atomic_csv(
        output,
        output_path,
    )

    print(
        "CFB candidate enrichment complete: "
        f"season={season} "
        f"week={week} "
        f"games={len(output)}"
    )

    for column, label in [
        (
            "ml_home_available",
            "ml_home",
        ),
        (
            "ml_away_available",
            "ml_away",
        ),
        (
            "spread_home_available",
            "spread_home",
        ),
        (
            "spread_away_available",
            "spread_away",
        ),
        (
            "total_over_available",
            "total_over",
        ),
        (
            "total_under_available",
            "total_under",
        ),
    ]:
        count = int(
            pd.to_numeric(
                output[
                    column
                ],
                errors="coerce",
            )
            .fillna(
                0
            )
            .sum()
        )

        print(
            f"{label}_candidates={count}"
        )

    print(
        f"Updated: {output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
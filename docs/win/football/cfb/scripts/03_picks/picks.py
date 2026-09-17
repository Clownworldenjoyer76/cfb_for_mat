#!/usr/bin/env python3
"""
CFB selection layer.

READS:
  docs/win/football/cfb/02_select/*CFB_selected.csv
  docs/win/football/cfb/config/markets.yaml

WRITES:
  docs/win/football/cfb/03_picks/*CFB_picks.csv

Behavior:
- Preserves every input column.
- Does not modify raw candidate columns.
- Evaluates every enabled side against selection_defaults plus all configured bands.
- Applies spread.max_spread_abs and total.min_total/max_total.
- If multiple sides qualify in one market, uses pick_preference:
    best_ev, best_prob, or best_kelly.
- Applies optional cross-market moneyline_vs_spread rules from markets.yaml.
- Overwrites only the existing final selection columns.
- Selection-time Kelly is min(full_kelly, resolved max_kelly), so markets.yaml
  controls Kelly independently of any upstream candidate Kelly cap.
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

DEFAULT_INPUT_DIR = CFB_ROOT / "02_select"
DEFAULT_MARKETS_PATH = CFB_ROOT / "config/markets.yaml"
DEFAULT_OUTPUT_DIR = CFB_ROOT / "03_picks"
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
    "cfb-picks-v2-target-validation-reporter-2026-09-16"
)

SELECTED_FILE_RE = re.compile(
    r"^week_(\d+)_CFB_selected\.csv$"
)

CALCULATION_TOLERANCE = 1e-8
PROBABILITY_PAIR_TOLERANCE = 1e-6
LINE_TOLERANCE = 1e-9

THRESHOLD_KEYS = {
    "min_ev",
    "min_edge",
    "min_kelly",
    "max_kelly",
    "min_odds_american",
    "max_odds_american",
    "min_model_prob",
    "max_model_prob",
}

BAND_TO_METRIC = {
    "odds_bands": "odds_american",
    "edge_bands": "edge",
    "ev_bands": "ev",
    "kelly_bands": "kelly",
    "prob_bands": "model_probability",
    "line_bands": "line",
}

PICK_METRIC = {
    "best_ev": "ev",
    "best_prob": "model_probability",
    "best_kelly": "kelly",
}

CROSS_MARKET_MODES = {
    "exclusive",
    "allow_both",
}

CROSS_MARKET_PREFERENCES = {
    "moneyline",
    "spread",
    *PICK_METRIC.keys(),
}

MARKETS = {
    "moneyline": {
        "output_prefix": "ml",
        "sides": {
            "home": ("ml_home", "HOME"),
            "away": ("ml_away", "AWAY"),
        },
        "market_extras": set(),
        "side_bands": {
            "odds_bands",
            "edge_bands",
            "ev_bands",
            "kelly_bands",
            "prob_bands",
        },
    },
    "spread": {
        "output_prefix": "spread",
        "sides": {
            "home": ("spread_home", "HOME"),
            "away": ("spread_away", "AWAY"),
        },
        "market_extras": {
            "max_spread_abs",
        },
        "side_bands": set(BAND_TO_METRIC),
    },
    "total": {
        "output_prefix": "total",
        "sides": {
            "over": ("total_over", "OVER"),
            "under": ("total_under", "UNDER"),
        },
        "market_extras": {
            "min_total",
            "max_total",
        },
        "side_bands": set(BAND_TO_METRIC),
    },
}


def fail(message: str) -> None:
    raise RuntimeError(message)


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


def integer_value(
    value: Any,
    label: str,
) -> int:
    text = clean(
        value
    )

    if not text:
        fail(
            f"{label} is required"
        )

    try:
        parsed = float(
            text
        )

    except (
        TypeError,
        ValueError,
    ):
        fail(
            f"{label} must be an integer; "
            f"found {value!r}"
        )

    if (
        not math.isfinite(
            parsed
        )
        or not parsed.is_integer()
    ):
        fail(
            f"{label} must be an integer; "
            f"found {value!r}"
        )

    return int(
        parsed
    )


def load_current_week_config(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        fail(
            "Missing current-week config: "
            f"{path}"
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
            "current_week.yaml must contain "
            "a YAML mapping"
        )

    return data


def resolve_current_target(
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


def resolve_input_files(
    input_dir: Path,
    pattern: str | None,
    default_week: int,
) -> tuple[
    list[Path],
    bool,
    str,
]:
    if pattern is None:
        resolved_pattern = (
            f"week_{default_week}_CFB_selected.csv"
        )

        explicit_pattern = False

    else:
        resolved_pattern = clean(
            pattern
        )

        if not resolved_pattern:
            fail(
                "--pattern cannot be blank"
            )

        explicit_pattern = True

    input_files = sorted(
        path
        for path
        in input_dir.glob(
            resolved_pattern
        )
        if path.is_file()
    )

    if not input_files:
        fail(
            "No input files matched "
            f"{resolved_pattern!r} "
            f"in {input_dir}"
        )

    for path in input_files:
        selected_file_week(
            path
        )

    return (
        input_files,
        explicit_pattern,
        resolved_pattern,
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


def american_to_decimal(
    odds: float,
) -> float:
    if odds == 0:
        fail(
            "American odds cannot be 0"
        )

    if odds > 0:
        return (
            1.0
            + odds
            / 100.0
        )

    return (
        1.0
        + 100.0
        / abs(
            odds
        )
    )


def raw_implied_probability(
    odds: float,
) -> float:
    return (
        1.0
        / american_to_decimal(
            odds
        )
    )


def no_vig_probabilities(
    first_odds: float,
    second_odds: float,
) -> tuple[
    float,
    float,
]:
    first_raw = raw_implied_probability(
        first_odds
    )

    second_raw = raw_implied_probability(
        second_odds
    )

    total = (
        first_raw
        + second_raw
    )

    if (
        not math.isfinite(
            total
        )
        or total <= 0
    ):
        fail(
            "Unable to calculate no-vig "
            "probabilities"
        )

    return (
        first_raw
        / total,
        second_raw
        / total,
    )


def expected_candidate_metrics(
    model_probability: float,
    odds_american: float,
    fair_probability: float,
) -> dict[
    str,
    float,
]:
    decimal_odds = american_to_decimal(
        odds_american
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
        - fair_probability
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


def assert_close(
    actual: float,
    expected: float,
    label: str,
    tolerance: float = CALCULATION_TOLERANCE,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        fail(
            f"{label} mismatch: "
            f"actual={actual} "
            f"expected={expected}"
        )



def number(
    value: Any,
    label: str,
) -> float:
    text = clean(value)

    if not text:
        fail(
            f"{label} is required"
        )

    try:
        result = float(text)
    except (TypeError, ValueError):
        fail(
            f"{label} must be numeric; "
            f"found {value!r}"
        )

    if not math.isfinite(result):
        fail(
            f"{label} must be finite; "
            f"found {value!r}"
        )

    return result


def optional_number(
    value: Any,
) -> float | None:
    text = clean(value)

    if not text:
        return None

    try:
        result = float(text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def boolean(
    value: Any,
    label: str,
) -> bool:
    if isinstance(value, bool):
        return value

    if (
        isinstance(
            value,
            (int, np.integer),
        )
        and value in {0, 1}
    ):
        return bool(value)

    text = clean(value).casefold()

    if text in {
        "true",
        "yes",
        "y",
        "1",
        "on",
    }:
        return True

    if text in {
        "false",
        "no",
        "n",
        "0",
        "off",
    }:
        return False

    fail(
        f"{label} must be true/false; "
        f"found {value!r}"
    )


def require_mapping(
    value: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(
            f"{label} must be a YAML mapping"
        )

    return value


def reject_unknown(
    mapping: dict[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(
        set(mapping)
        - allowed
    )

    if unknown:
        fail(
            f"{label} contains unsupported "
            f"keys: {unknown}"
        )


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        fail(
            f"Missing markets config: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(handle)

    return require_mapping(
        data,
        "markets.yaml",
    )


def load_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.is_file():
        fail(
            f"Missing input file: {path}"
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
            f"Input contains no rows: {path}"
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
            f"{label} missing required "
            f"columns: {missing}"
        )


def validate_thresholds(
    values: dict[str, float],
    label: str,
) -> None:
    if (
        values["min_kelly"]
        > values["max_kelly"]
    ):
        fail(
            f"{label}: min_kelly cannot "
            "exceed max_kelly"
        )

    if (
        values["min_odds_american"]
        > values["max_odds_american"]
    ):
        fail(
            f"{label}: min_odds_american "
            "cannot exceed max_odds_american"
        )

    if (
        values["min_model_prob"]
        > values["max_model_prob"]
    ):
        fail(
            f"{label}: min_model_prob cannot "
            "exceed max_model_prob"
        )

    if (
        values["min_kelly"] < 0
        or values["max_kelly"] < 0
    ):
        fail(
            f"{label}: Kelly limits "
            "cannot be negative"
        )

    if not (
        0
        <= values["min_model_prob"]
        <= 1
    ):
        fail(
            f"{label}: min_model_prob "
            "must be in [0,1]"
        )

    if not (
        0
        <= values["max_model_prob"]
        <= 1
    ):
        fail(
            f"{label}: max_model_prob "
            "must be in [0,1]"
        )


def thresholds(
    mapping: dict[str, Any],
    label: str,
    base: dict[str, float] | None = None,
    require_all: bool = False,
) -> dict[str, float]:
    result = dict(
        base or {}
    )

    if require_all:
        missing = sorted(
            THRESHOLD_KEYS
            - set(mapping)
        )

        if missing:
            fail(
                f"{label} missing required "
                f"keys: {missing}"
            )

    for key in THRESHOLD_KEYS:
        if key in mapping:
            result[key] = number(
                mapping[key],
                f"{label}.{key}",
            )

    missing = sorted(
        THRESHOLD_KEYS
        - set(result)
    )

    if missing:
        fail(
            f"{label} missing threshold "
            f"values: {missing}"
        )

    validate_thresholds(
        result,
        label,
    )

    return result


def bands(
    value: Any,
    label: str,
) -> list[tuple[float, float]]:
    if (
        not isinstance(value, list)
        or not value
    ):
        fail(
            f"{label} must be a non-empty "
            "list of [min, max] bands"
        )

    result: list[
        tuple[float, float]
    ] = []

    for index, item in enumerate(value):
        if (
            not isinstance(
                item,
                (list, tuple),
            )
            or len(item) != 2
        ):
            fail(
                f"{label}[{index}] "
                "must be [min, max]"
            )

        low = number(
            item[0],
            f"{label}[{index}][0]",
        )

        high = number(
            item[1],
            f"{label}[{index}][1]",
        )

        if low > high:
            fail(
                f"{label}[{index}] has "
                "min greater than max"
            )

        result.append(
            (
                low,
                high,
            )
        )

    return result


def matches_band(
    value: float,
    configured: list[
        tuple[float, float]
    ],
) -> bool:
    return any(
        low <= value <= high
        for low, high
        in configured
    )


def normalize_config(
    raw: dict[str, Any],
) -> dict[str, Any]:
    reject_unknown(
        raw,
        {
            "selection_defaults",
            "selection_rules",
            "markets",
        },
        "markets.yaml",
    )

    defaults_raw = require_mapping(
        raw.get(
            "selection_defaults"
        ),
        "markets.yaml.selection_defaults",
    )

    reject_unknown(
        defaults_raw,
        THRESHOLD_KEYS,
        "markets.yaml.selection_defaults",
    )

    defaults = thresholds(
        defaults_raw,
        "markets.yaml.selection_defaults",
        require_all=True,
    )

    selection_rules_raw = require_mapping(
        raw.get(
            "selection_rules",
            {},
        ),
        "markets.yaml.selection_rules",
    )

    reject_unknown(
        selection_rules_raw,
        {
            "moneyline_vs_spread",
        },
        "markets.yaml.selection_rules",
    )

    moneyline_vs_spread_raw = require_mapping(
        selection_rules_raw.get(
            "moneyline_vs_spread",
            {},
        ),
        (
            "markets.yaml.selection_rules."
            "moneyline_vs_spread"
        ),
    )

    reject_unknown(
        moneyline_vs_spread_raw,
        {
            "mode",
            "preference",
        },
        (
            "markets.yaml.selection_rules."
            "moneyline_vs_spread"
        ),
    )

    cross_market_mode = clean(
        moneyline_vs_spread_raw.get(
            "mode",
            "allow_both",
        )
    ).casefold()

    if cross_market_mode not in CROSS_MARKET_MODES:
        fail(
            "markets.yaml.selection_rules."
            "moneyline_vs_spread.mode must be "
            f"one of {sorted(CROSS_MARKET_MODES)}"
        )

    cross_market_preference = clean(
        moneyline_vs_spread_raw.get(
            "preference",
            "best_prob",
        )
    ).casefold()

    if (
        cross_market_preference
        not in CROSS_MARKET_PREFERENCES
    ):
        fail(
            "markets.yaml.selection_rules."
            "moneyline_vs_spread.preference "
            "must be one of "
            f"{sorted(CROSS_MARKET_PREFERENCES)}"
        )

    markets_raw = require_mapping(
        raw.get("markets"),
        "markets.yaml.markets",
    )

    reject_unknown(
        markets_raw,
        set(MARKETS),
        "markets.yaml.markets",
    )

    output: dict[str, Any] = {
        "selection_defaults": defaults,
        "selection_rules": {
            "moneyline_vs_spread": {
                "mode": cross_market_mode,
                "preference": (
                    cross_market_preference
                ),
            },
        },
        "markets": {},
    }

    for (
        market_name,
        spec,
    ) in MARKETS.items():
        market_label = (
            "markets.yaml.markets."
            f"{market_name}"
        )

        market_raw = require_mapping(
            markets_raw.get(
                market_name
            ),
            market_label,
        )

        allowed_market = (
            {
                "enabled",
                "pick_preference",
            }
            | THRESHOLD_KEYS
            | set(spec["sides"])
            | set(
                spec["market_extras"]
            )
        )

        reject_unknown(
            market_raw,
            allowed_market,
            market_label,
        )

        preference = clean(
            market_raw.get(
                "pick_preference",
                "best_prob",
            )
        ).casefold()

        if preference not in PICK_METRIC:
            fail(
                f"{market_label}."
                "pick_preference must be "
                f"one of {sorted(PICK_METRIC)}"
            )

        market_thresholds = thresholds(
            market_raw,
            market_label,
            base=defaults,
        )

        normalized: dict[str, Any] = {
            "enabled": boolean(
                market_raw.get(
                    "enabled",
                    True,
                ),
                f"{market_label}.enabled",
            ),
            "pick_preference": (
                preference
            ),
            "thresholds": (
                market_thresholds
            ),
            "sides": {},
        }

        if market_name == "spread":
            value = number(
                market_raw.get(
                    "max_spread_abs",
                    100.0,
                ),
                (
                    f"{market_label}."
                    "max_spread_abs"
                ),
            )

            if value < 0:
                fail(
                    f"{market_label}."
                    "max_spread_abs "
                    "cannot be negative"
                )

            normalized[
                "max_spread_abs"
            ] = value

        if market_name == "total":
            min_total = number(
                market_raw.get(
                    "min_total",
                    0.0,
                ),
                f"{market_label}.min_total",
            )

            max_total = number(
                market_raw.get(
                    "max_total",
                    100.0,
                ),
                f"{market_label}.max_total",
            )

            if min_total > max_total:
                fail(
                    f"{market_label}."
                    "min_total cannot exceed "
                    "max_total"
                )

            normalized[
                "min_total"
            ] = min_total

            normalized[
                "max_total"
            ] = max_total

        for side_name in spec["sides"]:
            side_label = (
                f"{market_label}."
                f"{side_name}"
            )

            side_raw = require_mapping(
                market_raw.get(
                    side_name
                ),
                side_label,
            )

            allowed_side = (
                {"enabled"}
                | THRESHOLD_KEYS
                | set(
                    spec["side_bands"]
                )
            )

            reject_unknown(
                side_raw,
                allowed_side,
                side_label,
            )

            side_thresholds = (
                thresholds(
                    side_raw,
                    side_label,
                    base=market_thresholds,
                )
            )

            side_bands: dict[
                str,
                list[
                    tuple[
                        float,
                        float,
                    ]
                ],
            ] = {}

            for key in spec[
                "side_bands"
            ]:
                if key in side_raw:
                    side_bands[key] = bands(
                        side_raw[key],
                        f"{side_label}.{key}",
                    )

            normalized[
                "sides"
            ][side_name] = {
                "enabled": boolean(
                    side_raw.get(
                        "enabled",
                        True,
                    ),
                    (
                        f"{side_label}."
                        "enabled"
                    ),
                ),
                "thresholds": (
                    side_thresholds
                ),
                "bands": (
                    side_bands
                ),
            }

        output[
            "markets"
        ][market_name] = normalized

    return output


def selection_columns() -> list[str]:
    return [
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


def candidate_columns() -> list[str]:
    result: list[str] = []

    for (
        market_name,
        spec,
    ) in MARKETS.items():
        for (
            prefix,
            _,
        ) in spec[
            "sides"
        ].values():
            result.extend(
                [
                    f"{prefix}_available",
                    f"{prefix}_odds_american",
                    f"{prefix}_model_probability",
                    f"{prefix}_implied_probability",
                    f"{prefix}_edge",
                    f"{prefix}_ev",
                    f"{prefix}_full_kelly",
                    f"{prefix}_kelly",
                ]
            )

            if market_name in {
                "spread",
                "total",
            }:
                result.append(
                    f"{prefix}_line"
                )

    return result


def validate_input(
    df: pd.DataFrame,
    path: Path,
    season: int,
    week: int,
) -> None:
    require_columns(
        df,
        [
            "season",
            "week",
            "game_id",
            "away_team",
            "home_team",
            *selection_columns(),
            *candidate_columns(),
        ],
        str(
            path
        ),
    )

    filename_week = selected_file_week(
        path
    )

    if filename_week != week:
        fail(
            f"{path}: filename week={filename_week} "
            f"does not match expected week={week}"
        )

    normalized_ids = df[
        "game_id"
    ].map(
        normalize_game_id
    )

    if normalized_ids.eq(
        ""
    ).any():
        fail(
            f"{path} contains "
            "blank game_id values"
        )

    if normalized_ids.duplicated().any():
        examples = normalized_ids[
            normalized_ids.duplicated(
                False
            )
        ].head(
            10
        ).tolist()

        fail(
            f"{path} contains duplicate "
            f"game_id values: {examples}"
        )

    df[
        "game_id"
    ] = normalized_ids

    observed_seasons = {
        integer_value(
            value,
            f"{path}: season",
        )
        for value in df[
            "season"
        ]
    }

    observed_weeks = {
        integer_value(
            value,
            f"{path}: week",
        )
        for value in df[
            "week"
        ]
    }

    if observed_seasons != {
        season
    }:
        fail(
            f"{path}: expected only season={season}; "
            f"found {sorted(observed_seasons)}"
        )

    if observed_weeks != {
        week
    }:
        fail(
            f"{path}: expected only week={week}; "
            f"found {sorted(observed_weeks)}"
        )


def validate_schedule_alignment(
    selected: pd.DataFrame,
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

    schedule_ids = schedule[
        "game_id"
    ].map(
        normalize_game_id
    )

    if schedule_ids.eq(
        ""
    ).any():
        fail(
            f"{schedule_path}: blank game_id found"
        )

    if schedule_ids.duplicated().any():
        examples = schedule_ids[
            schedule_ids.duplicated(
                False
            )
        ].head(
            10
        ).tolist()

        fail(
            f"{schedule_path}: duplicate game_id "
            f"values: {examples}"
        )

    schedule = schedule.copy()

    schedule[
        "game_id"
    ] = schedule_ids

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

    selected_ids = set(
        selected[
            "game_id"
        ]
    )

    target_ids = set(
        schedule[
            "game_id"
        ]
    )

    missing = sorted(
        target_ids
        - selected_ids
    )

    unexpected = sorted(
        selected_ids
        - target_ids
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

    schedule_lookup = schedule.set_index(
        "game_id",
        drop=False,
    )

    mismatches: list[
        dict[
            str,
            str,
        ]
    ] = []

    for _, row in selected.iterrows():
        game_id = row[
            "game_id"
        ]

        schedule_row = schedule_lookup.loc[
            game_id
        ]

        selected_away = clean(
            row.get(
                "away_team"
            )
        )

        selected_home = clean(
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
            selected_away
            != schedule_away
            or selected_home
            != schedule_home
        ):
            mismatches.append(
                {
                    "game_id":
                        game_id,
                    "selected_away":
                        selected_away,
                    "schedule_away":
                        schedule_away,
                    "selected_home":
                        selected_home,
                    "schedule_home":
                        schedule_home,
                }
            )

    if mismatches:
        fail(
            "Selected input team identity does not "
            "match target weekly schedule; "
            f"count={len(mismatches)} "
            f"examples={mismatches[:10]}"
        )


def candidate_numeric(
    row: pd.Series,
    prefix: str,
    metric: str,
) -> float:
    column = (
        f"{prefix}_{metric}"
    )

    value = optional_number(
        row.get(
            column,
            "",
        )
    )

    if value is None:
        fail(
            f"game_id={row['game_id']}: "
            f"available candidate {prefix} "
            f"has blank/non-numeric {column}"
        )

    return value


def validate_candidate_pair(
    df: pd.DataFrame,
    first_prefix: str,
    second_prefix: str,
    *,
    market_name: str,
) -> None:
    for _, row in df.iterrows():
        first_available = is_available(
            row,
            first_prefix,
        )

        second_available = is_available(
            row,
            second_prefix,
        )

        if (
            first_available
            != second_available
        ):
            fail(
                f"game_id={row['game_id']}: "
                f"{market_name} candidate availability "
                "does not match across both sides"
            )

        if not first_available:
            continue

        first_odds = candidate_numeric(
            row,
            first_prefix,
            "odds_american",
        )

        second_odds = candidate_numeric(
            row,
            second_prefix,
            "odds_american",
        )

        if (
            first_odds == 0
            or second_odds == 0
        ):
            fail(
                f"game_id={row['game_id']}: "
                f"{market_name} candidate odds "
                "cannot be zero"
            )

        first_model = candidate_numeric(
            row,
            first_prefix,
            "model_probability",
        )

        second_model = candidate_numeric(
            row,
            second_prefix,
            "model_probability",
        )

        if not (
            0.0
            <= first_model
            <= 1.0
        ):
            fail(
                f"game_id={row['game_id']}: "
                f"{first_prefix}_model_probability "
                "outside [0,1]"
            )

        if not (
            0.0
            <= second_model
            <= 1.0
        ):
            fail(
                f"game_id={row['game_id']}: "
                f"{second_prefix}_model_probability "
                "outside [0,1]"
            )

        assert_close(
            first_model
            + second_model,
            1.0,
            (
                f"game_id={row['game_id']} "
                f"{market_name} model probability pair"
            ),
            PROBABILITY_PAIR_TOLERANCE,
        )

        (
            first_fair,
            second_fair,
        ) = no_vig_probabilities(
            first_odds,
            second_odds,
        )

        for (
            prefix,
            model_probability,
            fair_probability,
            odds,
        ) in [
            (
                first_prefix,
                first_model,
                first_fair,
                first_odds,
            ),
            (
                second_prefix,
                second_model,
                second_fair,
                second_odds,
            ),
        ]:
            stored_fair = candidate_numeric(
                row,
                prefix,
                "implied_probability",
            )

            stored_edge = candidate_numeric(
                row,
                prefix,
                "edge",
            )

            stored_ev = candidate_numeric(
                row,
                prefix,
                "ev",
            )

            stored_full_kelly = candidate_numeric(
                row,
                prefix,
                "full_kelly",
            )

            stored_candidate_kelly = (
                candidate_numeric(
                    row,
                    prefix,
                    "kelly",
                )
            )

            assert_close(
                stored_fair,
                fair_probability,
                (
                    f"game_id={row['game_id']} "
                    f"{prefix} no-vig probability"
                ),
            )

            metrics = expected_candidate_metrics(
                model_probability,
                odds,
                fair_probability,
            )

            assert_close(
                stored_edge,
                metrics[
                    "edge"
                ],
                (
                    f"game_id={row['game_id']} "
                    f"{prefix} edge"
                ),
            )

            assert_close(
                stored_ev,
                metrics[
                    "ev"
                ],
                (
                    f"game_id={row['game_id']} "
                    f"{prefix} EV"
                ),
            )

            assert_close(
                stored_full_kelly,
                metrics[
                    "full_kelly"
                ],
                (
                    f"game_id={row['game_id']} "
                    f"{prefix} full Kelly"
                ),
            )

            if (
                stored_candidate_kelly < 0
                or stored_candidate_kelly
                > stored_full_kelly
                + CALCULATION_TOLERANCE
            ):
                fail(
                    f"game_id={row['game_id']}: "
                    f"{prefix}_kelly is inconsistent "
                    "with full Kelly"
                )

        if market_name == "spread":
            first_line = candidate_numeric(
                row,
                first_prefix,
                "line",
            )

            second_line = candidate_numeric(
                row,
                second_prefix,
                "line",
            )

            if not math.isclose(
                first_line
                + second_line,
                0.0,
                rel_tol=0.0,
                abs_tol=LINE_TOLERANCE,
            ):
                fail(
                    f"game_id={row['game_id']}: "
                    "spread candidate lines are "
                    "not exact opposites"
                )

        if market_name == "total":
            first_line = candidate_numeric(
                row,
                first_prefix,
                "line",
            )

            second_line = candidate_numeric(
                row,
                second_prefix,
                "line",
            )

            if not math.isclose(
                first_line,
                second_line,
                rel_tol=0.0,
                abs_tol=LINE_TOLERANCE,
            ):
                fail(
                    f"game_id={row['game_id']}: "
                    "total candidate lines disagree"
                )


def validate_candidate_arithmetic(
    df: pd.DataFrame,
) -> None:
    validate_candidate_pair(
        df,
        "ml_home",
        "ml_away",
        market_name="moneyline",
    )

    validate_candidate_pair(
        df,
        "spread_home",
        "spread_away",
        market_name="spread",
    )

    validate_candidate_pair(
        df,
        "total_over",
        "total_under",
        market_name="total",
    )



def is_available(
    row: pd.Series,
    prefix: str,
) -> bool:
    value = optional_number(
        row.get(
            f"{prefix}_available",
            "",
        )
    )

    if value not in {
        0.0,
        1.0,
    }:
        fail(
            f"game_id={row['game_id']}: "
            f"{prefix}_available must be "
            "0 or 1; found "
            f"{row.get(f'{prefix}_available')!r}"
        )

    return value == 1.0


def candidate(
    row: pd.Series,
    market_name: str,
    side_name: str,
    prefix: str,
    selection: str,
    side_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    if not side_cfg["enabled"]:
        return None

    if not is_available(
        row,
        prefix,
    ):
        return None

    result: dict[str, Any] = {
        "side_name": side_name,
        "selection": selection,
        "prefix": prefix,
    }

    for metric in [
        "odds_american",
        "model_probability",
        "implied_probability",
        "edge",
        "ev",
        "full_kelly",
    ]:
        column = (
            f"{prefix}_{metric}"
        )

        value = optional_number(
            row.get(
                column,
                "",
            )
        )

        if value is None:
            fail(
                f"game_id={row['game_id']}: "
                "available candidate "
                f"{prefix} has blank/"
                f"non-numeric {column}"
            )

        result[metric] = value

    if not (
        0
        <= result[
            "model_probability"
        ]
        <= 1
    ):
        fail(
            f"game_id={row['game_id']}: "
            f"{prefix}_model_probability "
            "outside [0,1]"
        )

    if not (
        0
        <= result[
            "implied_probability"
        ]
        <= 1
    ):
        fail(
            f"game_id={row['game_id']}: "
            f"{prefix}_implied_probability "
            "outside [0,1]"
        )

    if result["full_kelly"] < 0:
        fail(
            f"game_id={row['game_id']}: "
            f"{prefix}_full_kelly "
            "cannot be negative"
        )

    resolved = side_cfg[
        "thresholds"
    ]

    result["kelly"] = min(
        result["full_kelly"],
        resolved["max_kelly"],
    )

    if market_name in {
        "spread",
        "total",
    }:
        line = optional_number(
            row.get(
                f"{prefix}_line",
                "",
            )
        )

        if line is None:
            fail(
                f"game_id={row['game_id']}: "
                "available candidate "
                f"{prefix} has blank/"
                "non-numeric "
                f"{prefix}_line"
            )

        result["line"] = line

    else:
        result["line"] = None

    return result


def qualification_failures(
    item: dict[str, Any],
    market_name: str,
    market_cfg: dict[str, Any],
    side_cfg: dict[str, Any],
) -> list[str]:
    limits = side_cfg[
        "thresholds"
    ]

    failures: list[
        str
    ] = []

    if (
        item[
            "ev"
        ]
        < limits[
            "min_ev"
        ]
    ):
        failures.append(
            "min_ev"
        )

    if (
        item[
            "edge"
        ]
        < limits[
            "min_edge"
        ]
    ):
        failures.append(
            "min_edge"
        )

    if (
        item[
            "kelly"
        ]
        < limits[
            "min_kelly"
        ]
    ):
        failures.append(
            "min_kelly"
        )

    if not (
        limits[
            "min_odds_american"
        ]
        <= item[
            "odds_american"
        ]
        <= limits[
            "max_odds_american"
        ]
    ):
        failures.append(
            "odds_range"
        )

    if not (
        limits[
            "min_model_prob"
        ]
        <= item[
            "model_probability"
        ]
        <= limits[
            "max_model_prob"
        ]
    ):
        failures.append(
            "model_probability_range"
        )

    if (
        market_name == "spread"
        and (
            abs(
                item[
                    "line"
                ]
            )
            > market_cfg[
                "max_spread_abs"
            ]
        )
    ):
        failures.append(
            "max_spread_abs"
        )

    if (
        market_name == "total"
        and not (
            market_cfg[
                "min_total"
            ]
            <= item[
                "line"
            ]
            <= market_cfg[
                "max_total"
            ]
        )
    ):
        failures.append(
            "total_range"
        )

    for (
        band_name,
        configured,
    ) in side_cfg[
        "bands"
    ].items():
        metric = BAND_TO_METRIC[
            band_name
        ]

        value = item[
            metric
        ]

        if (
            value is None
            or not matches_band(
                value,
                configured,
            )
        ):
            failures.append(
                f"band:{band_name}"
            )

    return failures


def qualifies(
    item: dict[str, Any],
    market_name: str,
    market_cfg: dict[str, Any],
    side_cfg: dict[str, Any],
) -> bool:
    return not qualification_failures(
        item,
        market_name,
        market_cfg,
        side_cfg,
    )



def choose(
    items: list[
        dict[str, Any]
    ],
    preference: str,
) -> dict[str, Any]:
    primary = PICK_METRIC[
        preference
    ]

    def ranking(
        item: dict[str, Any],
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        return (
            float(
                item[primary]
            ),
            float(
                item[
                    "model_probability"
                ]
            ),
            float(
                item["ev"]
            ),
            float(
                item["kelly"]
            ),
        )

    return max(
        items,
        key=ranking,
    )


def empty_selection(
    prefix: str,
    reason: str,
) -> dict[str, Any]:
    result = {
        f"{prefix}_selected": 0,
        f"{prefix}_selection": "",
        f"{prefix}_selection_reason": (
            reason
        ),
        f"{prefix}_odds_american": (
            np.nan
        ),
        f"{prefix}_model_probability": (
            np.nan
        ),
        f"{prefix}_implied_probability": (
            np.nan
        ),
        f"{prefix}_edge": np.nan,
        f"{prefix}_ev": np.nan,
        f"{prefix}_full_kelly": (
            np.nan
        ),
        f"{prefix}_kelly": np.nan,
    }

    if prefix in {
        "spread",
        "total",
    }:
        result[
            f"{prefix}_line"
        ] = np.nan

    return result


def selected_values(
    prefix: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    result = {
        f"{prefix}_selected": 1,
        f"{prefix}_selection": (
            item["selection"]
        ),
        f"{prefix}_selection_reason": (
            "SELECTED_BY_MARKETS_YAML"
        ),
        f"{prefix}_odds_american": (
            item["odds_american"]
        ),
        f"{prefix}_model_probability": (
            item[
                "model_probability"
            ]
        ),
        f"{prefix}_implied_probability": (
            item[
                "implied_probability"
            ]
        ),
        f"{prefix}_edge": (
            item["edge"]
        ),
        f"{prefix}_ev": (
            item["ev"]
        ),
        f"{prefix}_full_kelly": (
            item["full_kelly"]
        ),
        f"{prefix}_kelly": (
            item["kelly"]
        ),
    }

    if prefix in {
        "spread",
        "total",
    }:
        result[
            f"{prefix}_line"
        ] = item["line"]

    return result


def evaluate_market(
    row: pd.Series,
    market_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = MARKETS[
        market_name
    ]

    market_cfg = config[
        "markets"
    ][market_name]

    output_prefix = spec[
        "output_prefix"
    ]

    if not market_cfg["enabled"]:
        return empty_selection(
            output_prefix,
            "MARKET_DISABLED",
        )

    qualifying: list[
        dict[str, Any]
    ] = []

    for (
        side_name,
        (
            candidate_prefix,
            selection,
        ),
    ) in spec[
        "sides"
    ].items():
        side_cfg = market_cfg[
            "sides"
        ][side_name]

        item = candidate(
            row,
            market_name,
            side_name,
            candidate_prefix,
            selection,
            side_cfg,
        )

        if (
            item is not None
            and qualifies(
                item,
                market_name,
                market_cfg,
                side_cfg,
            )
        ):
            qualifying.append(
                item
            )

    if not qualifying:
        return empty_selection(
            output_prefix,
            "NO_QUALIFYING_CANDIDATE",
        )

    winner = choose(
        qualifying,
        market_cfg[
            "pick_preference"
        ],
    )

    return selected_values(
        output_prefix,
        winner,
    )


def selected_from_updates(
    updates: dict[str, Any],
    prefix: str,
) -> bool:
    value = optional_number(
        updates.get(
            f"{prefix}_selected",
            0,
        )
    )

    return value == 1.0


def resolve_moneyline_vs_spread(
    updates: dict[str, Any],
    config: dict[str, Any],
) -> None:
    rule = config[
        "selection_rules"
    ][
        "moneyline_vs_spread"
    ]

    if rule["mode"] == "allow_both":
        return

    if not (
        selected_from_updates(
            updates,
            "ml",
        )
        and selected_from_updates(
            updates,
            "spread",
        )
    ):
        return

    preference = rule[
        "preference"
    ]

    if preference == "moneyline":
        winner = "ml"

    elif preference == "spread":
        winner = "spread"

    else:
        metric = PICK_METRIC[
            preference
        ]

        metric_columns = {
            "model_probability": (
                "model_probability"
            ),
            "ev": "ev",
            "kelly": "kelly",
        }

        suffix = metric_columns[
            metric
        ]

        def market_ranking(
            prefix: str,
        ) -> tuple[
            float,
            float,
            float,
            float,
            int,
        ]:
            primary = optional_number(
                updates.get(
                    f"{prefix}_{suffix}",
                    "",
                )
            )

            model_prob = optional_number(
                updates.get(
                    f"{prefix}_model_probability",
                    "",
                )
            )

            ev = optional_number(
                updates.get(
                    f"{prefix}_ev",
                    "",
                )
            )

            kelly = optional_number(
                updates.get(
                    f"{prefix}_kelly",
                    "",
                )
            )

            if (
                primary is None
                or model_prob is None
                or ev is None
                or kelly is None
            ):
                fail(
                    "Cannot resolve "
                    "moneyline_vs_spread: "
                    "selected market is missing "
                    "ranking metrics"
                )

            # Final deterministic tie-break favors
            # moneyline when all configured metrics
            # are exactly equal.
            tie_break = (
                1
                if prefix == "ml"
                else 0
            )

            return (
                float(primary),
                float(model_prob),
                float(ev),
                float(kelly),
                tie_break,
            )

        winner = max(
            ("ml", "spread"),
            key=market_ranking,
        )

    loser = (
        "spread"
        if winner == "ml"
        else "ml"
    )

    updates.update(
        empty_selection(
            loser,
            (
                "EXCLUDED_BY_"
                "MONEYLINE_VS_SPREAD_RULE"
            ),
        )
    )


def build_updates_for_row(
    row: pd.Series,
    config: dict[str, Any],
) -> dict[
    str,
    Any,
]:
    updates: dict[
        str,
        Any,
    ] = {}

    for market_name in MARKETS:
        updates.update(
            evaluate_market(
                row,
                market_name,
                config,
            )
        )

    resolve_moneyline_vs_spread(
        updates,
        config,
    )

    return updates


def analyze_source(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[
    str,
    Any,
]:
    candidate_available_counts: dict[
        str,
        int,
    ] = {}

    qualifying_side_counts: dict[
        str,
        int,
    ] = {}

    rejection_counts: dict[
        str,
        dict[
            str,
            int,
        ],
    ] = {}

    for (
        market_name,
        spec,
    ) in MARKETS.items():
        market_cfg = config[
            "markets"
        ][
            market_name
        ]

        for (
            side_name,
            (
                prefix,
                selection,
            ),
        ) in spec[
            "sides"
        ].items():
            candidate_available_counts[
                prefix
            ] = 0

            qualifying_side_counts[
                prefix
            ] = 0

            rejection_counts[
                prefix
            ] = {}

            side_cfg = market_cfg[
                "sides"
            ][
                side_name
            ]

            for _, row in df.iterrows():
                reasons: list[
                    str
                ] = []

                if not market_cfg[
                    "enabled"
                ]:
                    reasons = [
                        "market_disabled"
                    ]

                elif not side_cfg[
                    "enabled"
                ]:
                    reasons = [
                        "side_disabled"
                    ]

                elif not is_available(
                    row,
                    prefix,
                ):
                    reasons = [
                        "candidate_unavailable"
                    ]

                else:
                    candidate_available_counts[
                        prefix
                    ] += 1

                    item = candidate(
                        row,
                        market_name,
                        side_name,
                        prefix,
                        selection,
                        side_cfg,
                    )

                    if item is None:
                        fail(
                            "Internal candidate analysis "
                            f"failure for {prefix}"
                        )

                    reasons = qualification_failures(
                        item,
                        market_name,
                        market_cfg,
                        side_cfg,
                    )

                    if not reasons:
                        qualifying_side_counts[
                            prefix
                        ] += 1

                for reason in reasons:
                    rejection_counts[
                        prefix
                    ][
                        reason
                    ] = (
                        rejection_counts[
                            prefix
                        ].get(
                            reason,
                            0,
                        )
                        + 1
                    )

    return {
        "candidate_available_counts":
            candidate_available_counts,
        "qualifying_side_counts":
            qualifying_side_counts,
        "rejection_counts":
            rejection_counts,
    }


def values_equivalent(
    actual: Any,
    expected: Any,
) -> bool:
    expected_text = clean(
        expected
    )

    actual_text = clean(
        actual
    )

    if not expected_text:
        return not actual_text

    if isinstance(
        expected,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        try:
            expected_number = float(
                expected
            )

        except (
            TypeError,
            ValueError,
        ):
            return (
                actual_text
                == expected_text
            )

        if not math.isfinite(
            expected_number
        ):
            return not actual_text

        actual_number = optional_number(
            actual
        )

        if actual_number is None:
            return False

        return math.isclose(
            actual_number,
            expected_number,
            rel_tol=0.0,
            abs_tol=CALCULATION_TOLERANCE,
        )

    return (
        actual_text
        == expected_text
    )


def validate_output_frame(
    output: pd.DataFrame,
    source: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    if list(
        output.columns
    ) != list(
        source.columns
    ):
        fail(
            "Column order changed while "
            "building picks output"
        )

    if len(
        output
    ) != len(
        source
    ):
        fail(
            "Picks output row count changed"
        )

    require_columns(
        output,
        selection_columns(),
        "picks output",
    )

    expected_ids = source[
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
            "game_id order changed while "
            "building picks output"
        )

    non_selection = [
        column
        for column in source.columns
        if column
        not in selection_columns()
    ]

    source_non_selection = source[
        non_selection
    ].copy()

    output_non_selection = output[
        non_selection
    ].copy()

    for column in non_selection:
        source_non_selection[
            column
        ] = source_non_selection[
            column
        ].map(
            clean
        )

        output_non_selection[
            column
        ] = output_non_selection[
            column
        ].map(
            clean
        )

    if not output_non_selection.equals(
        source_non_selection
    ):
        fail(
            "Non-selection columns changed "
            "while building picks output"
        )

    for position in range(
        len(
            source
        )
    ):
        source_row = source.iloc[
            position
        ]

        output_row = output.iloc[
            position
        ]

        expected_updates = build_updates_for_row(
            source_row,
            config,
        )

        for column in selection_columns():
            if column not in expected_updates:
                fail(
                    "Internal expected selection "
                    f"column missing: {column}"
                )

            if not values_equivalent(
                output_row.get(
                    column
                ),
                expected_updates[
                    column
                ],
            ):
                fail(
                    f"game_id={source_row['game_id']}: "
                    f"serialized selection field "
                    f"{column} does not match "
                    "the configured selection result"
                )


def publish_atomic_csv(
    output: pd.DataFrame,
    source: pd.DataFrame,
    output_path: Path,
    config: dict[str, Any],
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

        validate_output_frame(
            serialized,
            source,
            config,
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


def process_file(
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
    *,
    season: int,
    week: int,
    schedule_path: Path,
) -> tuple[
    pd.DataFrame,
    bool,
    dict[
        str,
        Any,
    ],
]:
    df = load_csv(
        input_path
    )

    validate_input(
        df,
        input_path,
        season,
        week,
    )

    schedule = load_csv(
        schedule_path
    )

    validate_schedule_alignment(
        df,
        schedule,
        schedule_path,
        season,
        week,
    )

    validate_candidate_arithmetic(
        df
    )

    source_stats = analyze_source(
        df,
        config,
    )

    output = df.copy()

    for column in selection_columns():
        output[
            column
        ] = output[
            column
        ].astype(
            object
        )

    for index, row in df.iterrows():
        updates = build_updates_for_row(
            row,
            config,
        )

        for (
            column,
            value,
        ) in updates.items():
            output.at[
                index,
                column,
            ] = value

    validate_output_frame(
        output,
        df,
        config,
    )

    output_modified = publish_atomic_csv(
        output,
        df,
        output_path,
        config,
    )

    final_pick_counts = {
        "ml":
            int(
                pd.to_numeric(
                    output[
                        "ml_selected"
                    ],
                    errors="coerce",
                )
                .fillna(
                    0
                )
                .sum()
            ),
        "spread":
            int(
                pd.to_numeric(
                    output[
                        "spread_selected"
                    ],
                    errors="coerce",
                )
                .fillna(
                    0
                )
                .sum()
            ),
        "total":
            int(
                pd.to_numeric(
                    output[
                        "total_selected"
                    ],
                    errors="coerce",
                )
                .fillna(
                    0
                )
                .sum()
            ),
    }

    exclusion_reason = (
        "EXCLUDED_BY_"
        "MONEYLINE_VS_SPREAD_RULE"
    )

    cross_market_exclusions = int(
        output[
            "ml_selection_reason"
        ].map(
            clean
        ).eq(
            exclusion_reason
        ).sum()
        + output[
            "spread_selection_reason"
        ].map(
            clean
        ).eq(
            exclusion_reason
        ).sum()
    )

    stats = {
        **source_stats,
        "final_pick_counts":
            final_pick_counts,
        "cross_market_exclusions":
            cross_market_exclusions,
    }

    return (
        output,
        output_modified,
        stats,
    )



def output_name(
    input_path: Path,
) -> str:
    suffix = (
        "CFB_selected.csv"
    )

    if not input_path.name.endswith(
        suffix
    ):
        fail(
            "Unexpected input filename: "
            f"{input_path.name}"
        )

    return (
        input_path.name[
            :-len(suffix)
        ]
        + "CFB_picks.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
            "Target week when --pattern is omitted. "
            "Defaults to config/current_week.yaml."
        ),
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--markets",
        type=Path,
        default=DEFAULT_MARKETS_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--pattern",
        default=None,
        help=(
            "Explicit selected-file glob. "
            "Omit for target-week-only processing. "
            "A wildcard pattern is the explicit "
            "multiweek/replay mechanism."
        ),
    )

    return parser.parse_args()


def merge_nested_counts(
    destination: dict[
        str,
        dict[
            str,
            int,
        ],
    ],
    source: dict[
        str,
        dict[
            str,
            int,
        ],
    ],
) -> None:
    for (
        prefix,
        counts,
    ) in source.items():
        target_counts = destination.setdefault(
            prefix,
            {},
        )

        for (
            reason,
            count,
        ) in counts.items():
            target_counts[
                reason
            ] = (
                target_counts.get(
                    reason,
                    0,
                )
                + int(
                    count
                )
            )


def run(
    report: PipelineReporter,
    args: argparse.Namespace,
) -> int:
    input_dir = args.input_dir.resolve()
    markets_path = args.markets.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        fail(
            "Missing input directory: "
            f"{input_dir}"
        )

    if (
        args.pattern is not None
        and args.week is not None
    ):
        fail(
            "--week and --pattern are mutually "
            "exclusive. Use --week for a single "
            "target week or --pattern for explicit "
            "file selection/replay."
        )

    current_week = load_current_week_config(
        CURRENT_WEEK_CONFIG_PATH
    )

    (
        season,
        default_week,
    ) = resolve_current_target(
        current_week,
        args.season,
        args.week,
    )

    config = normalize_config(
        load_yaml(
            markets_path
        )
    )

    (
        input_files,
        explicit_pattern,
        resolved_pattern,
    ) = resolve_input_files(
        input_dir,
        args.pattern,
        default_week,
    )

    weeks = [
        selected_file_week(
            path
        )
        for path in input_files
    ]

    unique_weeks = sorted(
        set(
            weeks
        )
    )

    if (
        not explicit_pattern
        and unique_weeks
        != [
            default_week
        ]
    ):
        fail(
            "Default picks execution resolved "
            "outside the configured target week"
        )

    report.season = season

    if len(
        unique_weeks
    ) == 1:
        report.week = unique_weeks[
            0
        ]

    report.add_input(
        CURRENT_WEEK_CONFIG_PATH
    )

    report.add_input(
        markets_path
    )

    aggregate_candidate_counts: dict[
        str,
        int,
    ] = {}

    aggregate_qualifying_counts: dict[
        str,
        int,
    ] = {}

    aggregate_rejections: dict[
        str,
        dict[
            str,
            int,
        ],
    ] = {}

    totals = {
        "games":
            0,
        "ml":
            0,
        "spread":
            0,
        "total":
            0,
        "cross_market_exclusions":
            0,
        "modified_files":
            0,
    }

    output_modified_by_file: dict[
        str,
        bool,
    ] = {}

    for (
        input_path,
        week,
    ) in zip(
        input_files,
        weeks,
        strict=True,
    ):
        schedule_path = weekly_schedule_path(
            week
        )

        output_path = (
            output_dir
            / output_name(
                input_path
            )
        )

        if output_path == input_path:
            fail(
                "Picks output path must differ "
                "from selected input path"
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

        (
            output,
            output_modified,
            stats,
        ) = process_file(
            input_path,
            output_path,
            config,
            season=season,
            week=week,
            schedule_path=schedule_path,
        )

        counts = stats[
            "final_pick_counts"
        ]

        totals[
            "games"
        ] += len(
            output
        )

        totals[
            "ml"
        ] += int(
            counts[
                "ml"
            ]
        )

        totals[
            "spread"
        ] += int(
            counts[
                "spread"
            ]
        )

        totals[
            "total"
        ] += int(
            counts[
                "total"
            ]
        )

        totals[
            "cross_market_exclusions"
        ] += int(
            stats[
                "cross_market_exclusions"
            ]
        )

        if output_modified:
            totals[
                "modified_files"
            ] += 1

        output_modified_by_file[
            output_path.name
        ] = bool(
            output_modified
        )

        for (
            prefix,
            count,
        ) in stats[
            "candidate_available_counts"
        ].items():
            aggregate_candidate_counts[
                prefix
            ] = (
                aggregate_candidate_counts.get(
                    prefix,
                    0,
                )
                + int(
                    count
                )
            )

        for (
            prefix,
            count,
        ) in stats[
            "qualifying_side_counts"
        ].items():
            aggregate_qualifying_counts[
                prefix
            ] = (
                aggregate_qualifying_counts.get(
                    prefix,
                    0,
                )
                + int(
                    count
                )
            )

        merge_nested_counts(
            aggregate_rejections,
            stats[
                "rejection_counts"
            ],
        )

        print(
            f"Processed: "
            f"{input_path.name} -> "
            f"{output_path.name} "
            f"games={len(output)} "
            f"ml_picks={counts['ml']} "
            f"spread_picks={counts['spread']} "
            f"total_picks={counts['total']} "
            "output_modified="
            f"{'yes' if output_modified else 'no'}"
        )

    report.set_rows(
        rows_in=totals[
            "games"
        ],
        rows_out=totals[
            "games"
        ],
    )

    report.update_details(
        {
            "script_version":
                SCRIPT_VERSION,
            "markets_path":
                str(
                    markets_path
                ),
            "input_dir":
                str(
                    input_dir
                ),
            "output_dir":
                str(
                    output_dir
                ),
            "resolved_pattern":
                resolved_pattern,
            "explicit_pattern":
                explicit_pattern,
            "multiweek_mode":
                len(
                    unique_weeks
                )
                > 1,
            "weeks_processed":
                unique_weeks,
            "files_processed":
                len(
                    input_files
                ),
            "games":
                totals[
                    "games"
                ],
            "ml_picks":
                totals[
                    "ml"
                ],
            "spread_picks":
                totals[
                    "spread"
                ],
            "total_picks":
                totals[
                    "total"
                ],
            "cross_market_exclusions":
                totals[
                    "cross_market_exclusions"
                ],
            "candidate_available_counts":
                aggregate_candidate_counts,
            "qualifying_side_counts":
                aggregate_qualifying_counts,
            "rejection_counts":
                aggregate_rejections,
            "modified_files":
                totals[
                    "modified_files"
                ],
            "output_modified":
                totals[
                    "modified_files"
                ]
                > 0,
            "output_modified_by_file":
                output_modified_by_file,
        }
    )

    print(
        "picks.py "
        f"version={SCRIPT_VERSION}"
    )

    print(
        "CFB selection layer complete: "
        f"files={len(input_files)} "
        f"games={totals['games']} "
        f"ml_picks={totals['ml']} "
        f"spread_picks={totals['spread']} "
        f"total_picks={totals['total']} "
        "cross_market_exclusions="
        f"{totals['cross_market_exclusions']} "
        "modified_files="
        f"{totals['modified_files']}"
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
            "selection_scope":
                "pick_filter",
        },
    ) as report:
        return run(
            report,
            args,
        )



if __name__ == "__main__":
    raise SystemExit(main())
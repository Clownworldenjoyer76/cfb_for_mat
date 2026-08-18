#!/usr/bin/env python3
"""
NFL selection layer.

READS:
  docs/win/football/nfl/02_select/*NFL_selected.csv
  docs/win/football/nfl/config/markets.yaml

WRITES:
  docs/win/football/nfl/03_picks/*NFL_picks.csv

Behavior:
- Preserves every input column.
- Does not modify raw candidate columns.
- Evaluates every enabled side against selection_defaults plus all configured bands.
- Applies spread.max_spread_abs and total.min_total/max_total.
- If multiple sides qualify in one market, uses pick_preference:
    best_ev, best_prob, or best_kelly.
- Overwrites only the existing final selection columns.
- Selection-time Kelly is min(full_kelly, resolved max_kelly), so markets.yaml
  controls Kelly independently of any upstream candidate Kelly cap.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_INPUT_DIR = NFL_ROOT / "02_select"
DEFAULT_MARKETS_PATH = NFL_ROOT / "config/markets.yaml"
DEFAULT_OUTPUT_DIR = NFL_ROOT / "03_picks"
DEFAULT_PATTERN = "*NFL_selected.csv"

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
) -> None:
    require_columns(
        df,
        [
            "game_id",
            *selection_columns(),
            *candidate_columns(),
        ],
        str(path),
    )

    game_ids = df[
        "game_id"
    ].map(clean)

    if game_ids.eq("").any():
        fail(
            f"{path} contains "
            "blank game_id values"
        )

    if game_ids.duplicated().any():
        examples = game_ids[
            game_ids.duplicated(False)
        ].head(10).tolist()

        fail(
            f"{path} contains duplicate "
            f"game_id values: {examples}"
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


def qualifies(
    item: dict[str, Any],
    market_name: str,
    market_cfg: dict[str, Any],
    side_cfg: dict[str, Any],
) -> bool:
    limits = side_cfg[
        "thresholds"
    ]

    if (
        item["ev"]
        < limits["min_ev"]
    ):
        return False

    if (
        item["edge"]
        < limits["min_edge"]
    ):
        return False

    if (
        item["kelly"]
        < limits["min_kelly"]
    ):
        return False

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
        return False

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
        return False

    if market_name == "spread":
        if (
            abs(item["line"])
            > market_cfg[
                "max_spread_abs"
            ]
        ):
            return False

    if market_name == "total":
        if not (
            market_cfg[
                "min_total"
            ]
            <= item["line"]
            <= market_cfg[
                "max_total"
            ]
        ):
            return False

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
            return False

    return True


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


def process_file(
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    df = load_csv(
        input_path
    )

    validate_input(
        df,
        input_path,
    )

    output = df.copy()
    for column in selection_columns():
        output[column] = output[column].astype(object)

    original_columns = list(
        df.columns
    )

    for index, row in df.iterrows():
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

        for (
            column,
            value,
        ) in updates.items():
            output.at[
                index,
                column,
            ] = value

    if (
        list(output.columns)
        != original_columns
    ):
        fail(
            "Column order changed while "
            f"processing {input_path}"
        )

    if (
        output[
            "game_id"
        ].tolist()
        != df[
            "game_id"
        ].tolist()
    ):
        fail(
            "game_id order changed while "
            f"processing {input_path}"
        )

    non_selection = [
        column
        for column
        in original_columns
        if column
        not in selection_columns()
    ]

    if not output[
        non_selection
    ].equals(
        df[non_selection]
    ):
        fail(
            "Non-selection columns changed "
            f"while processing {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    output.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        output_path,
    )

    return output


def output_name(
    input_path: Path,
) -> str:
    suffix = (
        "NFL_selected.csv"
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
        + "NFL_picks.csv"
    )


def main() -> int:
    parser = (
        argparse.ArgumentParser()
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
        default=DEFAULT_PATTERN,
    )

    args = parser.parse_args()

    input_dir = (
        args.input_dir.resolve()
    )

    markets_path = (
        args.markets.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    if not input_dir.is_dir():
        fail(
            "Missing input directory: "
            f"{input_dir}"
        )

    config = normalize_config(
        load_yaml(
            markets_path
        )
    )

    input_files = sorted(
        path
        for path
        in input_dir.glob(
            args.pattern
        )
        if path.is_file()
    )

    if not input_files:
        fail(
            "No input files matched "
            f"{args.pattern!r} "
            f"in {input_dir}"
        )

    totals = {
        "games": 0,
        "ml": 0,
        "spread": 0,
        "total": 0,
    }

    for input_path in input_files:
        output_path = (
            output_dir
            / output_name(
                input_path
            )
        )

        output = process_file(
            input_path,
            output_path,
            config,
        )

        counts = {
            "ml": int(
                pd.to_numeric(
                    output[
                        "ml_selected"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            ),
            "spread": int(
                pd.to_numeric(
                    output[
                        "spread_selected"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            ),
            "total": int(
                pd.to_numeric(
                    output[
                        "total_selected"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            ),
        }

        totals[
            "games"
        ] += len(output)

        totals[
            "ml"
        ] += counts["ml"]

        totals[
            "spread"
        ] += counts["spread"]

        totals[
            "total"
        ] += counts["total"]

        print(
            f"Processed: "
            f"{input_path.name} -> "
            f"{output_path.name} "
            f"games={len(output)} "
            f"ml_picks={counts['ml']} "
            f"spread_picks="
            f"{counts['spread']} "
            f"total_picks="
            f"{counts['total']}"
        )

    print(
        "NFL selection layer complete: "
        f"files={len(input_files)} "
        f"games={totals['games']} "
        f"ml_picks={totals['ml']} "
        f"spread_picks="
        f"{totals['spread']} "
        f"total_picks="
        f"{totals['total']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

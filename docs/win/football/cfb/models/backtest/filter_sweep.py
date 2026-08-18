#!/usr/bin/env python3
"""
Fast filter/threshold sweep using run_backtest.py's already-generated
walkforward_probabilities.csv.

No model retraining is performed. The current selections.py implementation is imported
read-only and replayed for every threshold combination. Arguments that are omitted use
the exact currently resolved threshold for that market; therefore this script never
invents a threshold grid on the user's behalf.

Every write is restricted to docs/win/football/nfl/models/backtest/.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import math
import sys
from typing import Any

import numpy as np
import pandas as pd

import run_backtest as bt


OUTPUT_PATH = bt.BACKTEST_DIR / "filter_sweep.csv"
THRESHOLD_KEYS = [
    "min_ev", "min_edge", "min_kelly", "max_kelly",
    "min_odds_american", "max_odds_american",
    "min_model_prob", "max_model_prob",
]
SWEEP_COLUMNS = [
    "market", *THRESHOLD_KEYS,
    "bets", "wins", "losses", "pushes", "decisions", "win_rate_pct",
    "flat_risk_units", "flat_net_units", "flat_roi_pct",
    "kelly_risk_units", "kelly_net_units", "kelly_roi_pct",
]


def threshold_values(args: argparse.Namespace, baseline: dict[str, float]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for key in THRESHOLD_KEYS:
        supplied = getattr(args, key)
        values = supplied if supplied is not None and len(supplied) else [baseline[key]]
        unique: list[float] = []
        for value in values:
            value = float(value)
            if not math.isfinite(value):
                bt.fail(f"{key} contains non-finite value")
            if value not in unique:
                unique.append(value)
        result[key] = unique
    return result


def grade_selected(row: pd.Series, market: str, selected: dict[str, Any]) -> dict[str, Any] | None:
    prefix = {"moneyline": "ml", "spread": "spread", "total": "total"}[market]
    if int(selected.get(f"{prefix}_selected", 0)) != 1:
        return None
    if market == "moneyline":
        graded = bt.grade_moneyline(row, selected)
        return {
            "result": graded["ml_result"],
            "flat_profit": graded["ml_flat_profit_units"],
            "kelly_risk": graded["ml_kelly_risk_units"],
            "kelly_profit": graded["ml_kelly_profit_units"],
        }
    if market == "spread":
        graded = bt.grade_spread(row, selected)
        return {
            "result": graded["spread_result"],
            "flat_profit": graded["spread_flat_profit_units"],
            "kelly_risk": graded["spread_kelly_risk_units"],
            "kelly_profit": graded["spread_kelly_profit_units"],
        }
    graded = bt.grade_total(row, selected)
    return {
        "result": graded["total_result"],
        "flat_profit": graded["total_flat_profit_units"],
        "kelly_risk": graded["total_kelly_risk_units"],
        "kelly_profit": graded["total_kelly_profit_units"],
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    bets = len(results)
    wins = sum(r["result"] == "WIN" for r in results)
    losses = sum(r["result"] == "LOSS" for r in results)
    pushes = sum(r["result"] == "PUSH" for r in results)
    decisions = wins + losses
    flat_net = float(sum(float(r["flat_profit"]) for r in results))
    kelly_risk = float(sum(float(r["kelly_risk"]) for r in results))
    kelly_net = float(sum(float(r["kelly_profit"]) for r in results))
    return {
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "decisions": decisions,
        "win_rate_pct": 100.0 * wins / decisions if decisions else np.nan,
        "flat_risk_units": float(bets),
        "flat_net_units": flat_net,
        "flat_roi_pct": 100.0 * flat_net / bets if bets else np.nan,
        "kelly_risk_units": kelly_risk,
        "kelly_net_units": kelly_net,
        "kelly_roi_pct": 100.0 * kelly_net / kelly_risk if kelly_risk > 0 else np.nan,
    }


def run_market_sweep(
    df: pd.DataFrame,
    market: str,
    settings: dict[str, Any],
    market_document: dict[str, Any],
    selections: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    markets = market_document.get("markets")
    if not isinstance(markets, dict) or not isinstance(markets.get(market), dict):
        bt.fail(f"markets.yaml missing {market}")
    baseline = selections.resolve_thresholds(settings, market, markets[market])
    grid = threshold_values(args, baseline)
    keys = THRESHOLD_KEYS
    combinations = itertools.product(*(grid[key] for key in keys))

    rows: list[dict[str, Any]] = []
    for combo_number, values in enumerate(combinations, start=1):
        thresholds = dict(zip(keys, values))
        settings_copy = copy.deepcopy(settings)
        markets_copy = copy.deepcopy(market_document)
        target = markets_copy["markets"][market]
        for key, value in thresholds.items():
            target[key] = float(value)

        selections.resolve_thresholds(settings_copy, market, target)

        graded_results: list[dict[str, Any]] = []
        for _, source_row in df.iterrows():
            selected = bt.evaluate_single_market(
                source_row, market, settings_copy, markets_copy, selections
            )
            graded = grade_selected(source_row, market, selected)
            if graded is not None:
                graded_results.append(graded)

        rows.append({"market": market, **thresholds, **summarize(graded_results)})
        if combo_number == 1 or combo_number % 50 == 0:
            print(f"filter sweep: market={market} combinations_completed={combo_number}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--market",
        choices=["moneyline", "spread", "total", "all"],
        default="all",
    )
    for key in THRESHOLD_KEYS:
        parser.add_argument(
            "--" + key.replace("_", "-"),
            dest=key,
            type=float,
            nargs="+",
            default=None,
            help=f"Explicit values to test for {key}; omitted = current resolved value.",
        )
    args = parser.parse_args()

    if not bt.PROBABILITIES_PATH.is_file():
        bt.fail(
            f"Missing {bt.PROBABILITIES_PATH}. Run run_backtest.py successfully first."
        )

    bt.ensure_write_path(OUTPUT_PATH)
    selections = bt.load_module(bt.SELECTIONS_PATH, "nfl_filter_sweep_selections_readonly")
    settings = bt.read_yaml(bt.SETTINGS_PATH)
    market_document = bt.read_yaml(bt.MARKETS_PATH)
    df = pd.read_csv(bt.PROBABILITIES_PATH, dtype={"game_id": str}, low_memory=False)
    missing = [
        c for c in [
            *bt.PREDICTION_OUTPUT_COLUMNS,
            *bt.PROBABILITY_COLUMNS,
        ] if c not in df.columns
    ]
    if missing:
        bt.fail(f"walkforward_probabilities.csv missing required columns: {missing}")

    markets = ["moneyline", "spread", "total"] if args.market == "all" else [args.market]
    output_rows: list[dict[str, Any]] = []
    for market in markets:
        output_rows.extend(
            run_market_sweep(df, market, settings, market_document, selections, args)
        )

    output = pd.DataFrame(output_rows, columns=SWEEP_COLUMNS)
    bt.atomic_write_csv(output, OUTPUT_PATH)
    print(f"Wrote {len(output)} filter combinations to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

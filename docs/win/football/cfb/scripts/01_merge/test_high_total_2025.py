#!/usr/bin/env python3
"""
test_high_total_2025.py

Tests the frozen high-total subgroup signal found by audit_market_signal_v1.py
against the untouched 2025 validation season.

Frozen using 2022-2024 only:

    High-total definition:
        sportsbook total >= 60

    Margin correction:
        sportsbook implied home margin + 1.7184 points

    Total correction:
        sportsbook total - 2.2500 points

This script DOES NOT fit or tune anything on 2025.

Input:
    docs/win/football/cfb/data/score_model_v4/
        validation_2025_predictions_v4.csv

Output:
    docs/win/football/cfb/data/signal_audit_v1/
        high_total_2025_verification.csv
        high_total_2025_metrics.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


HIGH_TOTAL_THRESHOLD = 60.0

# Frozen from 2022-2024 signal audit.
MARGIN_CORRECTION = 1.7184
TOTAL_CORRECTION = -2.2500


def find_cfb_root() -> Path:
    """
    Script is expected at:
        docs/win/football/cfb/scripts/01_merge/test_high_total_2025.py
    """
    return Path(__file__).resolve().parents[2]


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    mask = actual.notna() & predicted.notna()
    if not mask.any():
        return float("nan")

    return float(
        np.mean(
            np.abs(
                actual.loc[mask].astype(float).to_numpy()
                - predicted.loc[mask].astype(float).to_numpy()
            )
        )
    )


def mean_error(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Positive value means actual result was higher than prediction.
    """
    mask = actual.notna() & predicted.notna()
    if not mask.any():
        return float("nan")

    return float(
        np.mean(
            actual.loc[mask].astype(float).to_numpy()
            - predicted.loc[mask].astype(float).to_numpy()
        )
    )


def bootstrap_mae_improvement(
    actual: np.ndarray,
    baseline: np.ndarray,
    corrected: np.ndarray,
    iterations: int = 10000,
    seed: int = 2026,
) -> tuple[float, float]:
    """
    Bootstrap 95% confidence interval for:

        baseline MAE - corrected MAE

    Positive = correction improved MAE.
    """
    if len(actual) < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = len(actual)

    improvements = np.empty(iterations, dtype=float)

    for i in range(iterations):
        idx = rng.integers(0, n, size=n)

        base_mae = np.mean(
            np.abs(actual[idx] - baseline[idx])
        )

        corrected_mae = np.mean(
            np.abs(actual[idx] - corrected[idx])
        )

        improvements[i] = base_mae - corrected_mae

    low, high = np.percentile(improvements, [2.5, 97.5])

    return float(low), float(high)


def main() -> None:
    cfb_root = find_cfb_root()

    input_path = (
        cfb_root
        / "data"
        / "score_model_v4"
        / "validation_2025_predictions_v4.csv"
    )

    output_dir = (
        cfb_root
        / "data"
        / "signal_audit_v1"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "high_total_2025_verification.csv"
    metrics_path = output_dir / "high_total_2025_metrics.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing V4 validation file:\n{input_path}"
        )

    print(f"CFB root: {cfb_root}")
    print(f"Loading: {input_path}")

    df = pd.read_csv(input_path)

    required = [
        "game_id",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
        "home_spread",
        "total",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            "Missing required columns from validation file: "
            + ", ".join(missing)
        )

    numeric_cols = [
        "away_score",
        "home_score",
        "home_spread",
        "total",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---------------------------------------------------------
    # Untouched 2025 high-total subgroup
    # ---------------------------------------------------------

    high = df[
        df["total"].notna()
        & (df["total"] >= HIGH_TOTAL_THRESHOLD)
        & df["home_score"].notna()
        & df["away_score"].notna()
    ].copy()

    if high.empty:
        raise RuntimeError(
            f"No 2025 games found with total >= {HIGH_TOTAL_THRESHOLD:.1f}"
        )

    # Actual outcomes.
    high["actual_margin"] = (
        high["home_score"] - high["away_score"]
    )

    high["actual_total"] = (
        high["home_score"] + high["away_score"]
    )

    # Sportsbook spread convention:
    #
    # home_spread = -7 means home favored by 7.
    #
    # Therefore sportsbook implied HOME scoring margin is +7.
    high["market_margin"] = -high["home_spread"]

    high["market_total"] = high["total"]

    # Frozen corrections from 2022-2024.
    high["corrected_margin"] = (
        high["market_margin"] + MARGIN_CORRECTION
    )

    high["corrected_total"] = (
        high["market_total"] + TOTAL_CORRECTION
    )

    # Errors.
    high["market_margin_abs_error"] = np.abs(
        high["actual_margin"] - high["market_margin"]
    )

    high["corrected_margin_abs_error"] = np.abs(
        high["actual_margin"] - high["corrected_margin"]
    )

    high["market_total_abs_error"] = np.abs(
        high["actual_total"] - high["market_total"]
    )

    high["corrected_total_abs_error"] = np.abs(
        high["actual_total"] - high["corrected_total"]
    )

    high["margin_improvement"] = (
        high["market_margin_abs_error"]
        - high["corrected_margin_abs_error"]
    )

    high["total_improvement"] = (
        high["market_total_abs_error"]
        - high["corrected_total_abs_error"]
    )

    # ---------------------------------------------------------
    # Margin metrics
    # ---------------------------------------------------------

    margin_rows = high.dropna(
        subset=[
            "actual_margin",
            "market_margin",
            "corrected_margin",
        ]
    ).copy()

    market_margin_mae = mae(
        margin_rows["actual_margin"],
        margin_rows["market_margin"],
    )

    corrected_margin_mae = mae(
        margin_rows["actual_margin"],
        margin_rows["corrected_margin"],
    )

    margin_improvement = (
        market_margin_mae - corrected_margin_mae
    )

    raw_margin_bias = mean_error(
        margin_rows["actual_margin"],
        margin_rows["market_margin"],
    )

    corrected_margin_bias = mean_error(
        margin_rows["actual_margin"],
        margin_rows["corrected_margin"],
    )

    margin_ci_low, margin_ci_high = bootstrap_mae_improvement(
        margin_rows["actual_margin"].to_numpy(float),
        margin_rows["market_margin"].to_numpy(float),
        margin_rows["corrected_margin"].to_numpy(float),
    )

    # ---------------------------------------------------------
    # Total metrics
    # ---------------------------------------------------------

    total_rows = high.dropna(
        subset=[
            "actual_total",
            "market_total",
            "corrected_total",
        ]
    ).copy()

    market_total_mae = mae(
        total_rows["actual_total"],
        total_rows["market_total"],
    )

    corrected_total_mae = mae(
        total_rows["actual_total"],
        total_rows["corrected_total"],
    )

    total_improvement = (
        market_total_mae - corrected_total_mae
    )

    raw_total_bias = mean_error(
        total_rows["actual_total"],
        total_rows["market_total"],
    )

    corrected_total_bias = mean_error(
        total_rows["actual_total"],
        total_rows["corrected_total"],
    )

    total_ci_low, total_ci_high = bootstrap_mae_improvement(
        total_rows["actual_total"].to_numpy(float),
        total_rows["market_total"].to_numpy(float),
        total_rows["corrected_total"].to_numpy(float),
    )

    # ---------------------------------------------------------
    # Save detailed games
    # ---------------------------------------------------------

    detail_cols = [
        "game_id",
        "game_date",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
        "home_spread",
        "market_margin",
        "corrected_margin",
        "actual_margin",
        "market_margin_abs_error",
        "corrected_margin_abs_error",
        "margin_improvement",
        "market_total",
        "corrected_total",
        "actual_total",
        "market_total_abs_error",
        "corrected_total_abs_error",
        "total_improvement",
    ]

    detail_cols = [
        c for c in detail_cols
        if c in high.columns
    ]

    high[detail_cols].to_csv(
        detail_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------

    metrics = pd.DataFrame(
        [
            {
                "target": "margin",
                "threshold": HIGH_TOTAL_THRESHOLD,
                "games": len(margin_rows),
                "frozen_correction": MARGIN_CORRECTION,
                "market_mae": market_margin_mae,
                "corrected_mae": corrected_margin_mae,
                "mae_improvement": margin_improvement,
                "raw_market_bias": raw_margin_bias,
                "corrected_bias": corrected_margin_bias,
                "bootstrap_ci_low": margin_ci_low,
                "bootstrap_ci_high": margin_ci_high,
                "confirmed_positive_mae": margin_improvement > 0,
                "confirmed_ci_above_zero": margin_ci_low > 0,
            },
            {
                "target": "total",
                "threshold": HIGH_TOTAL_THRESHOLD,
                "games": len(total_rows),
                "frozen_correction": TOTAL_CORRECTION,
                "market_mae": market_total_mae,
                "corrected_mae": corrected_total_mae,
                "mae_improvement": total_improvement,
                "raw_market_bias": raw_total_bias,
                "corrected_bias": corrected_total_bias,
                "bootstrap_ci_low": total_ci_low,
                "bootstrap_ci_high": total_ci_high,
                "confirmed_positive_mae": total_improvement > 0,
                "confirmed_ci_above_zero": total_ci_low > 0,
            },
        ]
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Console report
    # ---------------------------------------------------------

    print()
    print(
        f"2025 HIGH-TOTAL VERIFICATION "
        f"(market total >= {HIGH_TOTAL_THRESHOLD:.1f})"
    )

    print()
    print(f"Games: {len(high)}")

    print()
    print("MARGIN")
    print(
        f"  frozen correction: "
        f"{MARGIN_CORRECTION:+.4f} toward home"
    )
    print(
        f"  2025 raw market bias: "
        f"{raw_margin_bias:+.4f}"
    )
    print(
        f"  market MAE: "
        f"{market_margin_mae:.4f}"
    )
    print(
        f"  corrected MAE: "
        f"{corrected_margin_mae:.4f}"
    )
    print(
        f"  improvement: "
        f"{margin_improvement:+.4f}"
    )
    print(
        f"  bootstrap 95% CI: "
        f"({margin_ci_low:+.4f}, {margin_ci_high:+.4f})"
    )

    print()
    print("TOTAL")
    print(
        f"  frozen correction: "
        f"{TOTAL_CORRECTION:+.4f}"
    )
    print(
        f"  2025 raw market bias: "
        f"{raw_total_bias:+.4f}"
    )
    print(
        f"  market MAE: "
        f"{market_total_mae:.4f}"
    )
    print(
        f"  corrected MAE: "
        f"{corrected_total_mae:.4f}"
    )
    print(
        f"  improvement: "
        f"{total_improvement:+.4f}"
    )
    print(
        f"  bootstrap 95% CI: "
        f"({total_ci_low:+.4f}, {total_ci_high:+.4f})"
    )

    print()
    print("RESULT")

    margin_positive = margin_improvement > 0
    total_positive = total_improvement > 0

    margin_strict = margin_ci_low > 0
    total_strict = total_ci_low > 0

    if margin_strict and total_strict:
        print(
            "  BOTH FROZEN SIGNALS STRICTLY CONFIRMED ON 2025"
        )
    elif total_strict:
        print(
            "  TOTAL SIGNAL STRICTLY CONFIRMED ON 2025"
        )
    elif margin_strict:
        print(
            "  MARGIN SIGNAL STRICTLY CONFIRMED ON 2025"
        )
    elif margin_positive or total_positive:
        print(
            "  SOME 2025 MAE IMPROVEMENT, "
            "BUT NOT STATISTICALLY CONFIRMED"
        )
    else:
        print(
            "  HIGH-TOTAL SIGNAL FAILED 2025"
        )

    print()
    print(f"Game detail: {detail_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Validate v4 forecasts and build probability/error calibration."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from v4_common import (
    BACKTEST_DIR, MODELS_DIR, apply_platt, classification_metrics,
    empirical_probability_greater, fit_platt, regression_metrics, write_json,
)

INPUT_PATH = BACKTEST_DIR / "step13_market_independent_backtest_v4.csv"
OUTPUT_CSV = BACKTEST_DIR / "step14_market_independent_probabilities_v4.csv"
OUTPUT_JSON = MODELS_DIR / "step14_market_independent_validation_v4.json"
ERROR_JSON = MODELS_DIR / "step14_error_distributions_v4.json"
VARIANTS = ["core", "augmented"]


def reg_summary(df: pd.DataFrame, target: str, pred_col: str) -> dict:
    mask = df[target].notna() & df[pred_col].notna()
    return {"n": int(mask.sum()), **regression_metrics(df.loc[mask, target].to_numpy(float), df.loc[mask, pred_col].to_numpy(float))}


def cls_summary(df: pd.DataFrame, target: str, p_col: str) -> dict:
    mask = df[target].notna() & df[p_col].notna()
    return {"n": int(mask.sum()), **classification_metrics(df.loc[mask, target].to_numpy(float), df.loc[mask, p_col].to_numpy(float))}


def pick_reg(df: pd.DataFrame, target: str, prefix: str) -> str:
    return min(VARIANTS, key=lambda v: reg_summary(df, target, f"{prefix}_{v}")["rmse"])


def pick_cls(df: pd.DataFrame) -> str:
    return min(VARIANTS, key=lambda v: cls_summary(df, "actual_home_win", f"raw_home_win_probability_{v}")["logloss"])


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    frame = pd.DataFrame({"y": y, "p": p}).dropna().sort_values("p").reset_index(drop=True)
    if frame.empty:
        return []
    groups = pd.qcut(np.arange(len(frame)), q=min(bins, len(frame)), duplicates="drop")
    out = []
    for _, g in frame.groupby(groups, observed=True):
        out.append({"n": len(g), "mean_probability": float(g["p"].mean()), "observed_rate": float(g["y"].mean())})
    return out


def main() -> int:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    if len(seasons) < 2:
        raise RuntimeError("Need at least two OOF seasons")

    raw_variant_metrics = {
        "margin": {v: reg_summary(df, "actual_margin", f"predicted_margin_{v}") for v in VARIANTS},
        "total": {v: reg_summary(df, "actual_total", f"predicted_total_{v}") for v in VARIANTS},
        "moneyline": {v: cls_summary(df, "actual_home_win", f"raw_home_win_probability_{v}") for v in VARIANTS},
    }
    selected = {
        "margin": pick_reg(df, "actual_margin", "predicted_margin"),
        "total": pick_reg(df, "actual_total", "predicted_total"),
        "moneyline": pick_cls(df),
    }
    df["selected_predicted_margin"] = df[f"predicted_margin_{selected['margin']}"]
    df["selected_predicted_total"] = df[f"predicted_total_{selected['total']}"]
    df["selected_raw_home_win_probability"] = df[f"raw_home_win_probability_{selected['moneyline']}"]

    naive = {
        "margin": reg_summary(df, "actual_margin", "naive_margin_prediction"),
        "total": reg_summary(df, "actual_total", "naive_total_prediction"),
        "moneyline": cls_summary(df, "actual_home_win", "naive_home_win_probability"),
    }
    closing = {
        "margin": reg_summary(df, "actual_margin", "closing_spread_line"),
        "total": reg_summary(df, "actual_total", "closing_total_line"),
        "moneyline": cls_summary(df, "actual_home_win", "closing_home_win_no_vig_probability"),
    }
    selected_raw = {
        "margin": reg_summary(df, "actual_margin", "selected_predicted_margin"),
        "total": reg_summary(df, "actual_total", "selected_predicted_total"),
        "moneyline": cls_summary(df, "actual_home_win", "selected_raw_home_win_probability"),
    }

    nested_rows = []
    nested_meta = []
    for season in seasons[1:]:
        prior = df[df["season"] < season].copy()
        current = df[df["season"] == season].copy()
        mv = pick_reg(prior, "actual_margin", "predicted_margin")
        tv = pick_reg(prior, "actual_total", "predicted_total")
        wv = pick_cls(prior)
        current["chrono_predicted_margin"] = current[f"predicted_margin_{mv}"]
        current["chrono_predicted_total"] = current[f"predicted_total_{tv}"]
        current["chrono_raw_home_win_probability"] = current[f"raw_home_win_probability_{wv}"]

        win_prior = prior[["actual_home_win", f"raw_home_win_probability_{wv}"]].dropna()
        intercept, slope = fit_platt(win_prior[f"raw_home_win_probability_{wv}"].to_numpy(float), win_prior["actual_home_win"].to_numpy(float))
        current["chrono_home_win_probability"] = apply_platt(current["chrono_raw_home_win_probability"].to_numpy(float), intercept, slope)

        margin_errors = (prior["actual_margin"] - prior[f"predicted_margin_{mv}"]).dropna().to_numpy(float)
        total_errors = (prior["actual_total"] - prior[f"predicted_total_{tv}"]).dropna().to_numpy(float)
        current["chrono_home_cover_probability"] = empirical_probability_greater(
            margin_errors, current["closing_spread_line"].to_numpy(float) - current["chrono_predicted_margin"].to_numpy(float)
        )
        current["chrono_over_probability"] = empirical_probability_greater(
            total_errors, current["closing_total_line"].to_numpy(float) - current["chrono_predicted_total"].to_numpy(float)
        )
        nested_meta.append({
            "season": season,
            "selected_variants": {"margin": mv, "total": tv, "moneyline": wv},
            "platt": {"intercept": intercept, "slope": slope, "fit_rows": len(win_prior)},
            "residual_rows": {"margin": len(margin_errors), "total": len(total_errors)},
        })
        nested_rows.append(current)

    chrono = pd.concat(nested_rows, ignore_index=True)
    chronological = {
        "margin": reg_summary(chrono, "actual_margin", "chrono_predicted_margin"),
        "total": reg_summary(chrono, "actual_total", "chrono_predicted_total"),
        "moneyline": cls_summary(chrono, "actual_home_win", "chrono_home_win_probability"),
        "spread_vs_closing": cls_summary(chrono, "actual_home_cover", "chrono_home_cover_probability"),
        "total_vs_closing": cls_summary(chrono, "actual_over", "chrono_over_probability"),
    }
    chrono_baselines = {
        "margin_naive": reg_summary(chrono, "actual_margin", "naive_margin_prediction"),
        "total_naive": reg_summary(chrono, "actual_total", "naive_total_prediction"),
        "moneyline_naive": cls_summary(chrono, "actual_home_win", "naive_home_win_probability"),
        "margin_closing": reg_summary(chrono, "actual_margin", "closing_spread_line"),
        "total_closing": reg_summary(chrono, "actual_total", "closing_total_line"),
        "moneyline_closing": cls_summary(chrono, "actual_home_win", "closing_home_win_no_vig_probability"),
        "spread_closing": cls_summary(chrono, "actual_home_cover", "closing_home_cover_no_vig_probability"),
        "total_price_closing": cls_summary(chrono, "actual_over", "closing_over_no_vig_probability"),
    }

    season_comparison = []
    for season in seasons[1:]:
        s = chrono[chrono["season"] == season]
        row = {
            "season": season,
            "margin_model_rmse": reg_summary(s, "actual_margin", "chrono_predicted_margin")["rmse"],
            "margin_naive_rmse": reg_summary(s, "actual_margin", "naive_margin_prediction")["rmse"],
            "total_model_rmse": reg_summary(s, "actual_total", "chrono_predicted_total")["rmse"],
            "total_naive_rmse": reg_summary(s, "actual_total", "naive_total_prediction")["rmse"],
            "moneyline_model_logloss": cls_summary(s, "actual_home_win", "chrono_home_win_probability")["logloss"],
            "moneyline_naive_logloss": cls_summary(s, "actual_home_win", "naive_home_win_probability")["logloss"],
        }
        row["margin_beats_naive"] = row["margin_model_rmse"] < row["margin_naive_rmse"]
        row["total_beats_naive"] = row["total_model_rmse"] < row["total_naive_rmse"]
        row["moneyline_beats_naive"] = row["moneyline_model_logloss"] < row["moneyline_naive_logloss"]
        season_comparison.append(row)

    def gate(target: str) -> dict:
        if target in ("margin", "total"):
            overall = chronological[target]["rmse"] < chrono_baselines[f"{target}_naive"]["rmse"]
            count = sum(bool(r[f"{target}_beats_naive"]) for r in season_comparison)
        else:
            overall = chronological["moneyline"]["logloss"] < chrono_baselines["moneyline_naive"]["logloss"]
            count = sum(bool(r["moneyline_beats_naive"]) for r in season_comparison)
        return {"beats_naive_overall": bool(overall), "seasons_beating_naive": int(count), "seasons_evaluated": len(season_comparison), "pass": bool(overall and count >= 2)}

    gates = {target: gate(target) for target in ("margin", "total", "moneyline")}
    overall_candidate = all(g["pass"] for g in gates.values())

    win_fit = df[["actual_home_win", "selected_raw_home_win_probability"]].dropna()
    final_intercept, final_slope = fit_platt(win_fit["selected_raw_home_win_probability"].to_numpy(float), win_fit["actual_home_win"].to_numpy(float))
    df["final_calibrated_home_win_probability"] = apply_platt(df["selected_raw_home_win_probability"].to_numpy(float), final_intercept, final_slope)
    margin_errors = (df["actual_margin"] - df["selected_predicted_margin"]).dropna().to_numpy(float)
    total_errors = (df["actual_total"] - df["selected_predicted_total"]).dropna().to_numpy(float)
    df["final_home_cover_probability_at_closing_line"] = empirical_probability_greater(
        margin_errors, df["closing_spread_line"].to_numpy(float) - df["selected_predicted_margin"].to_numpy(float)
    )
    df["final_over_probability_at_closing_line"] = empirical_probability_greater(
        total_errors, df["closing_total_line"].to_numpy(float) - df["selected_predicted_total"].to_numpy(float)
    )

    closing_diagnostic = {
        "warning": "Closing-market diagnostic only; not live decision-time betting performance.",
        "spread": cls_summary(df, "actual_home_cover", "final_home_cover_probability_at_closing_line"),
        "spread_market": cls_summary(df, "actual_home_cover", "closing_home_cover_no_vig_probability"),
        "total": cls_summary(df, "actual_over", "final_over_probability_at_closing_line"),
        "total_market": cls_summary(df, "actual_over", "closing_over_no_vig_probability"),
    }

    write_json(ERROR_JSON, {
        "candidate_version": "v4_market_independent_outcomes",
        "selected_variants": selected,
        "moneyline_platt": {"intercept": final_intercept, "slope": final_slope, "fit_rows": len(win_fit)},
        "margin_error_distribution": {
            "n": len(margin_errors), "mean": float(np.mean(margin_errors)), "std": float(np.std(margin_errors, ddof=1)),
            "quantiles": {str(q): float(np.quantile(margin_errors, q)) for q in (0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99)},
            "errors": [float(x) for x in margin_errors],
        },
        "total_error_distribution": {
            "n": len(total_errors), "mean": float(np.mean(total_errors)), "std": float(np.std(total_errors, ddof=1)),
            "quantiles": {str(q): float(np.quantile(total_errors, q)) for q in (0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99)},
            "errors": [float(x) for x in total_errors],
        },
        "inference_formula": {
            "home_cover_probability": "P(OOF margin_error > live_spread_line - predicted_margin), half-weight equality",
            "over_probability": "P(OOF total_error > live_total_line - predicted_total), half-weight equality",
            "moneyline_probability": "Platt(raw CatBoost home-win probability)",
        },
    })

    write_json(OUTPUT_JSON, {
        "step": 14,
        "candidate_version": "v4_market_independent_outcomes",
        "production_cutover": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "oof_seasons": seasons,
        "final_selected_variants": selected,
        "raw_variant_metrics": raw_variant_metrics,
        "selected_raw_metrics": selected_raw,
        "naive_baselines": naive,
        "closing_market_benchmarks": closing,
        "nested_chronological_metrics": chronological,
        "nested_chronological_baselines": chrono_baselines,
        "nested_fold_metadata": nested_meta,
        "season_comparison": season_comparison,
        "gates": gates,
        "overall_candidate_pass": overall_candidate,
        "closing_market_diagnostic": closing_diagnostic,
        "calibration_bins": {
            "moneyline": calibration_bins(win_fit["actual_home_win"].to_numpy(float), df.loc[win_fit.index, "final_calibrated_home_win_probability"].to_numpy(float)),
            "spread_closing": calibration_bins(df["actual_home_cover"].to_numpy(float), df["final_home_cover_probability_at_closing_line"].to_numpy(float)),
            "total_closing": calibration_bins(df["actual_over"].to_numpy(float), df["final_over_probability_at_closing_line"].to_numpy(float)),
        },
        "historical_roi_claim_allowed": False,
        "next_stage": "If forecast gates pass, deploy in shadow/forward-validation mode against preserved 2026 sportsbook snapshots.",
    })
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"selected={selected}")
    print(f"gates={gates}")
    print(f"overall_candidate_pass={overall_candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

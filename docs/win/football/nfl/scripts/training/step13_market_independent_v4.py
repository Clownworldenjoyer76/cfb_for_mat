#!/usr/bin/env python3
"""Expanding-season chronological OOF backtest for market-independent v4."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from v4_common import (
    BACKTEST_DIR, DEFAULT_START_SEASON, build_feature_variants,
    discover_training_seasons, infer_feature_types, no_vig_positive,
    predict_classifier, predict_regressor, prepare_matrix, read_inputs,
    target_home_win, train_classifier, train_regressor,
)

OUTPUT_PATH = BACKTEST_DIR / "step13_market_independent_backtest_v4.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=None)
    args = parser.parse_args()

    seasons = discover_training_seasons(args.start_season, args.end_season)
    if len(seasons) < 2:
        raise RuntimeError("Need at least two seasons for chronological OOF")
    raw, _ = read_inputs(seasons)
    raw["season_num"] = pd.to_numeric(raw["season"], errors="raise").astype(int)
    variants, _ = build_feature_variants(list(raw.columns))

    prepared = {}
    types = {}
    for variant, features in variants.items():
        categorical, numeric = infer_feature_types(raw, features)
        prepared[variant] = prepare_matrix(raw, features, categorical, numeric)
        types[variant] = [features.index(c) for c in categorical]

    margin = raw["margin"].astype(float)
    total = raw["total_points"].astype(float)
    home_win = target_home_win(raw)
    output_frames = []

    for holdout in seasons[1:]:
        train_idx = raw.index[raw["season_num"] < holdout]
        test_idx = raw.index[raw["season_num"] == holdout]
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise RuntimeError(f"Empty fold for {holdout}")

        fold = pd.DataFrame({
            "game_id": raw.loc[test_idx, "game_id"].values,
            "season": holdout,
            "week": pd.to_numeric(raw.loc[test_idx, "week"], errors="coerce").values,
            "gameday": raw.loc[test_idx, "gameday"].values,
            "away_team": raw.loc[test_idx, "away_team"].values,
            "home_team": raw.loc[test_idx, "home_team"].values,
            "actual_margin": margin.loc[test_idx].values,
            "actual_total": total.loc[test_idx].values,
            "actual_home_win": home_win.loc[test_idx].values,
            "closing_spread_line": pd.to_numeric(raw.loc[test_idx, "spread_line"], errors="coerce").values,
            "closing_total_line": pd.to_numeric(raw.loc[test_idx, "total_line"], errors="coerce").values,
            "closing_away_moneyline": pd.to_numeric(raw.loc[test_idx, "away_moneyline"], errors="coerce").values,
            "closing_home_moneyline": pd.to_numeric(raw.loc[test_idx, "home_moneyline"], errors="coerce").values,
            "closing_away_spread_odds": pd.to_numeric(raw.loc[test_idx, "away_spread_odds"], errors="coerce").values,
            "closing_home_spread_odds": pd.to_numeric(raw.loc[test_idx, "home_spread_odds"], errors="coerce").values,
            "closing_under_odds": pd.to_numeric(raw.loc[test_idx, "under_odds"], errors="coerce").values,
            "closing_over_odds": pd.to_numeric(raw.loc[test_idx, "over_odds"], errors="coerce").values,
        }, index=test_idx)

        fold["naive_margin_prediction"] = float(margin.loc[train_idx].mean())
        fold["naive_total_prediction"] = float(total.loc[train_idx].mean())
        fold["naive_home_win_probability"] = float(home_win.loc[train_idx].dropna().mean())
        fold["closing_home_win_no_vig_probability"] = no_vig_positive(raw.loc[test_idx, "home_moneyline"], raw.loc[test_idx, "away_moneyline"])
        fold["closing_home_cover_no_vig_probability"] = no_vig_positive(raw.loc[test_idx, "home_spread_odds"], raw.loc[test_idx, "away_spread_odds"])
        fold["closing_over_no_vig_probability"] = no_vig_positive(raw.loc[test_idx, "over_odds"], raw.loc[test_idx, "under_odds"])

        spread_diff = fold["actual_margin"] - fold["closing_spread_line"]
        total_diff = fold["actual_total"] - fold["closing_total_line"]
        fold["actual_home_cover"] = np.where(spread_diff > 0, 1.0, np.where(spread_diff < 0, 0.0, np.nan))
        fold["actual_over"] = np.where(total_diff > 0, 1.0, np.where(total_diff < 0, 0.0, np.nan))

        for variant, X in prepared.items():
            cat_indices = types[variant]
            margin_model = train_regressor(X.loc[train_idx], margin.loc[train_idx], cat_indices)
            total_model = train_regressor(X.loc[train_idx], total.loc[train_idx], cat_indices)
            win_train = train_idx[home_win.loc[train_idx].notna()]
            win_model = train_classifier(X.loc[win_train], home_win.loc[win_train], cat_indices)
            fold[f"predicted_margin_{variant}"] = predict_regressor(margin_model, X.loc[test_idx], cat_indices)
            fold[f"predicted_total_{variant}"] = predict_regressor(total_model, X.loc[test_idx], cat_indices)
            fold[f"raw_home_win_probability_{variant}"] = predict_classifier(win_model, X.loc[test_idx], cat_indices)

        output_frames.append(fold.reset_index(drop=True))
        print(f"holdout={holdout} train_rows={len(train_idx)} test_rows={len(test_idx)}")

    output = pd.concat(output_frames, ignore_index=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(output)} rows to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

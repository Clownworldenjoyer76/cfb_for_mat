#!/usr/bin/env python3
"""Train full-history market-independent NFL v4 candidate models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from v4_common import (
    CLASSIFIER_PARAMS, DEFAULT_START_SEASON, MODELS_DIR, REGRESSOR_PARAMS,
    build_feature_variants, discover_training_seasons, infer_feature_types,
    prepare_matrix, read_inputs, target_home_win, train_classifier,
    train_regressor, write_json,
)

SCHEMA_PATH = MODELS_DIR / "step11_market_independent_feature_schema_v4.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=None)
    args = parser.parse_args()

    seasons = discover_training_seasons(args.start_season, args.end_season)
    raw, hashes = read_inputs(seasons)
    variants, families = build_feature_variants(list(raw.columns))
    margin = raw["margin"].astype(float)
    total = raw["total_points"].astype(float)
    home_win = target_home_win(raw)

    model_files = {}
    variant_schema = {}
    for variant, features in variants.items():
        categorical, numeric = infer_feature_types(raw, features)
        X = prepare_matrix(raw, features, categorical, numeric)
        cat_indices = [features.index(c) for c in categorical]

        margin_model = train_regressor(X, margin, cat_indices)
        total_model = train_regressor(X, total, cat_indices)
        win_mask = home_win.notna()
        win_model = train_classifier(X.loc[win_mask], home_win.loc[win_mask], cat_indices)

        paths = {
            "margin": MODELS_DIR / f"step11_margin_model_{variant}_v4.cbm",
            "total": MODELS_DIR / f"step11_total_model_{variant}_v4.cbm",
            "moneyline": MODELS_DIR / f"step11_moneyline_model_{variant}_v4.cbm",
        }
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        margin_model.save_model(str(paths["margin"]))
        total_model.save_model(str(paths["total"]))
        win_model.save_model(str(paths["moneyline"]))
        model_files[variant] = {k: p.name for k, p in paths.items()}
        variant_schema[variant] = {
            "feature_count": len(features),
            "feature_order": features,
            "categorical_features": categorical,
            "numeric_features": numeric,
        }
        print(f"{variant}: features={len(features)} rows={len(raw)}")

    write_json(SCHEMA_PATH, {
        "step": 11,
        "candidate_version": "v4_market_independent_outcomes",
        "production_cutover": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_seasons": seasons,
        "training_rows": len(raw),
        "targets": {
            "margin": "home_score - away_score",
            "total": "home_score + away_score",
            "moneyline": "home win; ties excluded",
        },
        "market_policy": {
            "market_inputs_used_for_training": False,
            "market_targets_used_for_training": False,
            "historical_market_usage": "benchmark only in Step 13/14",
        },
        "feature_families": families,
        "variants": variant_schema,
        "model_files": model_files,
        "regressor_params": REGRESSOR_PARAMS,
        "classifier_params": CLASSIFIER_PARAMS,
        "input_sha256": hashes,
    })
    print(f"Wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

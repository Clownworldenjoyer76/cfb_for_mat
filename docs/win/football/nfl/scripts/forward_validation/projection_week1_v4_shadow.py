#!/usr/bin/env python3
"""Run market-independent v4 in shadow mode for 2026 Week 1.

This script intentionally reuses the production Week 1 feature-construction
helper because that code already resolves prior-season team/QB stats, current
injuries/depth, schedule context, weather, travel, DRAT and EPRED. The helper
constructs its legacy matrix, but v4 receives ONLY the exact feature subset in
step11_market_independent_feature_schema_v4.json.

Sportsbook lines/prices are used only after model prediction to calculate live
market-relative probabilities/edges for forward validation. This script does
not modify production projection files, selection files, thresholds, or model
artifacts.
"""
from __future__ import annotations

import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

SEASON = 2026
WEEK = 1
MISSING_CAT = "__MISSING__"
PROB_EPS = 1e-6


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError(f"Could not locate repository root from {here}")


def nfl_root() -> Path:
    return repo_root() / "docs/win/football/nfl"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_week1_helper():
    path = nfl_root() / "scripts/01_merge/projection_week1.py"
    spec = importlib.util.spec_from_file_location("legacy_week1_projection_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def american_implied(value: object) -> float:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(odds) or odds == 0:
        return math.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def american_decimal(value: object) -> float:
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(odds) or odds == 0:
        return math.nan
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / (-odds)


def no_vig_positive(positive_odds: object, negative_odds: object) -> float:
    pos = american_implied(positive_odds)
    neg = american_implied(negative_odds)
    if not math.isfinite(pos) or not math.isfinite(neg) or pos + neg <= 0:
        return math.nan
    return pos / (pos + neg)


def quoted_ev(probability: float, american_odds: object) -> float:
    dec = american_decimal(american_odds)
    if not math.isfinite(probability) or not math.isfinite(dec):
        return math.nan
    return probability * dec - 1.0


def apply_platt(raw_probability: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    p = np.clip(np.asarray(raw_probability, dtype=float), PROB_EPS, 1.0 - PROB_EPS)
    logit = np.log(p / (1.0 - p))
    z = np.clip(intercept + slope * logit, -35.0, 35.0)
    return np.clip(1.0 / (1.0 + np.exp(-z)), PROB_EPS, 1.0 - PROB_EPS)


def empirical_probability_greater(errors: list[float], thresholds: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(errors, dtype=float))
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("OOF error distribution is empty/non-finite")
    thresholds = np.asarray(thresholds, dtype=float)
    left = np.searchsorted(values, thresholds, side="left")
    right = np.searchsorted(values, thresholds, side="right")
    return ((len(values) - right) + 0.5 * (right - left)) / len(values)


def prepare_v4_matrix(full_features: pd.DataFrame, variant_schema: dict) -> pd.DataFrame:
    feature_order = list(variant_schema["feature_order"])
    categorical = set(variant_schema["categorical_features"])
    numeric = set(variant_schema["numeric_features"])
    missing = [c for c in feature_order if c not in full_features.columns]
    if missing:
        raise RuntimeError(f"Week 1 helper did not construct v4 features: {missing}")
    if categorical | numeric != set(feature_order) or categorical & numeric:
        raise RuntimeError("Invalid v4 feature type partition")

    X = full_features[feature_order].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical:
        text = X[c].astype(str).str.strip()
        X[c] = text.mask(text.str.casefold().isin({"", "nan", "none", "null", "<na>", "nat"}), MISSING_CAT)
    return X


def load_regressor(path: Path) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def load_classifier(path: Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def latest_capture_metadata(root: Path) -> dict:
    snapshot_dir = root / "00_intake/odds/raw/snapshots"
    candidates = sorted(snapshot_dir.glob("*_nfl_odds.json")) if snapshot_dir.exists() else []
    source_kind = "immutable_snapshot"
    if not candidates:
        candidates = sorted((root / "00_intake/odds/raw").glob("*_nfl_odds.json"))
        source_kind = "legacy_current_file"
    if not candidates:
        return {"source_kind": "none", "path": None, "snapshot_id": None, "fetched_at": None}
    path = candidates[-1]
    payload = load_json(path)
    return {
        "source_kind": source_kind,
        "path": str(path.relative_to(root)),
        "snapshot_id": payload.get("snapshot_id"),
        "fetched_at": payload.get("fetched_at"),
    }


def main() -> int:
    root = nfl_root()
    helper = load_week1_helper()

    input_path = root / "00_intake/predictions/enriched/combined/week_1_NFL_enriched.csv"
    original = helper.read_csv(input_path)
    helper.validate_week1_base(original, SEASON, str(input_path))

    # Reuse existing Week 1 source resolution, then discard all non-v4 fields.
    legacy_schema = helper.load_schema(root)
    full_features = helper.prepare_model_features(root, original.copy(), legacy_schema)

    v4_schema_path = root / "models/step11_market_independent_feature_schema_v4.json"
    validation_path = root / "models/step14_market_independent_validation_v4.json"
    errors_path = root / "models/step14_error_distributions_v4.json"
    v4_schema = load_json(v4_schema_path)
    validation = load_json(validation_path)
    errors = load_json(errors_path)

    if v4_schema.get("market_policy", {}).get("market_inputs_used_for_training") is not False:
        raise RuntimeError("v4 schema does not certify market-free training")
    if validation.get("overall_candidate_pass") is not True:
        raise RuntimeError("v4 historical forecast gate did not pass")

    selected = dict(validation["final_selected_variants"])
    predictions: dict[str, np.ndarray] = {}

    for target in ("margin", "total", "moneyline"):
        variant = selected[target]
        variant_schema = v4_schema["variants"][variant]
        X = prepare_v4_matrix(full_features, variant_schema)
        expected = list(variant_schema["feature_order"])

        if target == "margin":
            model = load_regressor(root / f"models/step11_margin_model_{variant}_v4.cbm")
            if list(model.feature_names_) != expected:
                raise RuntimeError("v4 margin model feature order mismatch")
            predictions[target] = np.asarray(model.predict(X), dtype=float)
        elif target == "total":
            model = load_regressor(root / f"models/step11_total_model_{variant}_v4.cbm")
            if list(model.feature_names_) != expected:
                raise RuntimeError("v4 total model feature order mismatch")
            predictions[target] = np.asarray(model.predict(X), dtype=float)
        else:
            model = load_classifier(root / f"models/step11_moneyline_model_{variant}_v4.cbm")
            if list(model.feature_names_) != expected:
                raise RuntimeError("v4 moneyline model feature order mismatch")
            raw_ml = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
            platt = errors["moneyline_platt"]
            predictions[target] = apply_platt(raw_ml, float(platt["intercept"]), float(platt["slope"]))
            predictions["moneyline_raw"] = raw_ml

    predicted_margin = predictions["margin"]
    predicted_total = predictions["total"]
    home_win = predictions["moneyline"]
    away_win = 1.0 - home_win

    spread_line = pd.to_numeric(full_features["spread_line"], errors="coerce").to_numpy(dtype=float)
    total_line = pd.to_numeric(full_features["total_line"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(spread_line).all() or not np.isfinite(total_line).all():
        raise RuntimeError("Live Week 1 spread/total lines are missing")

    home_cover = empirical_probability_greater(
        errors["margin_error_distribution"]["errors"], spread_line - predicted_margin
    )
    away_cover = 1.0 - home_cover
    over = empirical_probability_greater(
        errors["total_error_distribution"]["errors"], total_line - predicted_total
    )
    under = 1.0 - over

    output = original.copy()
    output["v4_shadow_only"] = True
    output["v4_predicted_margin"] = predicted_margin
    output["v4_predicted_total"] = predicted_total
    output["v4_predicted_home_score"] = (predicted_total + predicted_margin) / 2.0
    output["v4_predicted_away_score"] = (predicted_total - predicted_margin) / 2.0
    output["v4_raw_home_win_probability"] = predictions["moneyline_raw"]
    output["v4_home_win_probability"] = home_win
    output["v4_away_win_probability"] = away_win
    output["v4_home_cover_probability"] = home_cover
    output["v4_away_cover_probability"] = away_cover
    output["v4_over_probability"] = over
    output["v4_under_probability"] = under
    output["v4_live_spread_line_training_sign"] = spread_line
    output["v4_live_total_line"] = total_line

    for idx in output.index:
        hm = full_features.at[idx, "home_moneyline"]
        am = full_features.at[idx, "away_moneyline"]
        hs = full_features.at[idx, "home_spread_odds"]
        aws = full_features.at[idx, "away_spread_odds"]
        ov = full_features.at[idx, "over_odds"]
        un = full_features.at[idx, "under_odds"]

        fair_home_ml = no_vig_positive(hm, am)
        fair_home_spread = no_vig_positive(hs, aws)
        fair_over = no_vig_positive(ov, un)

        output.at[idx, "v4_market_fair_home_win_probability"] = fair_home_ml
        output.at[idx, "v4_market_fair_home_cover_probability"] = fair_home_spread
        output.at[idx, "v4_market_fair_over_probability"] = fair_over
        output.at[idx, "v4_home_ml_fair_edge"] = home_win[idx] - fair_home_ml
        output.at[idx, "v4_away_ml_fair_edge"] = away_win[idx] - (1.0 - fair_home_ml)
        output.at[idx, "v4_home_spread_fair_edge"] = home_cover[idx] - fair_home_spread
        output.at[idx, "v4_away_spread_fair_edge"] = away_cover[idx] - (1.0 - fair_home_spread)
        output.at[idx, "v4_over_fair_edge"] = over[idx] - fair_over
        output.at[idx, "v4_under_fair_edge"] = under[idx] - (1.0 - fair_over)
        output.at[idx, "v4_home_ml_quoted_ev"] = quoted_ev(home_win[idx], hm)
        output.at[idx, "v4_away_ml_quoted_ev"] = quoted_ev(away_win[idx], am)
        output.at[idx, "v4_home_spread_quoted_ev"] = quoted_ev(home_cover[idx], hs)
        output.at[idx, "v4_away_spread_quoted_ev"] = quoted_ev(away_cover[idx], aws)
        output.at[idx, "v4_over_quoted_ev"] = quoted_ev(over[idx], ov)
        output.at[idx, "v4_under_quoted_ev"] = quoted_ev(under[idx], un)

    capture = latest_capture_metadata(root)
    generated_at = datetime.now(timezone.utc).isoformat()
    output["v4_shadow_generated_at"] = generated_at
    output["v4_odds_capture_source_kind"] = capture["source_kind"]
    output["v4_odds_capture_source"] = capture["path"] or ""
    output["v4_odds_capture_fetched_at"] = capture["fetched_at"] or ""

    out_dir = root / "forward_validation/v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{SEASON}_week_{WEEK}_v4_shadow.csv"
    meta_path = out_dir / f"{SEASON}_week_{WEEK}_v4_shadow_metadata.json"
    output.to_csv(csv_path, index=False, encoding="utf-8-sig")

    metadata = {
        "candidate_version": "v4_market_independent_outcomes",
        "shadow_only": True,
        "season": SEASON,
        "week": WEEK,
        "generated_at": generated_at,
        "games": len(output),
        "selected_variants": selected,
        "market_inputs_used_by_model": False,
        "market_used_post_prediction": True,
        "probability_method": {
            "moneyline": "OOF Platt calibration of direct home-win classifier",
            "spread": "empirical OOF margin-error distribution evaluated at current live line",
            "total": "empirical OOF total-error distribution evaluated at current live line",
        },
        "odds_capture": capture,
        "historical_validation_overall_candidate_pass": True,
        "production_cutover": False,
        "production_files_modified": False,
        "output": str(csv_path.relative_to(root)),
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} games={len(output)}")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

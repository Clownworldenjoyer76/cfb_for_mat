#!/usr/bin/env python3
"""Shared production inference for the market-independent NFL v4 models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

MISSING_CAT = "__MISSING__"
PROB_EPS = 1e-6
OUTPUT_COLUMNS = [
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


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing v4 production artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def apply_platt(raw_probability: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    p = np.clip(np.asarray(raw_probability, dtype=float), PROB_EPS, 1.0 - PROB_EPS)
    logit = np.log(p / (1.0 - p))
    z = np.clip(intercept + slope * logit, -35.0, 35.0)
    return np.clip(1.0 / (1.0 + np.exp(-z)), PROB_EPS, 1.0 - PROB_EPS)


def empirical_probability_greater(errors: list[float], thresholds: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(errors, dtype=float))
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("V4 OOF error distribution is empty or non-finite")
    thresholds = np.asarray(thresholds, dtype=float)
    if not np.isfinite(thresholds).all():
        raise ValueError("V4 live probability thresholds contain non-finite values")
    left = np.searchsorted(values, thresholds, side="left")
    right = np.searchsorted(values, thresholds, side="right")
    return ((len(values) - right) + 0.5 * (right - left)) / len(values)


def validate_probability_pair(a: np.ndarray, b: np.ndarray, label: str) -> None:
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError(f"{label}: non-finite probability values")
    if ((a < 0.0) | (a > 1.0) | (b < 0.0) | (b > 1.0)).any():
        raise RuntimeError(f"{label}: probabilities outside [0, 1]")
    if not np.allclose(a + b, 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{label}: complementary probabilities do not sum to 1")


def prepare_v4_matrix(full_features: pd.DataFrame, variant_schema: dict) -> pd.DataFrame:
    feature_order = list(variant_schema["feature_order"])
    categorical = set(variant_schema["categorical_features"])
    numeric = set(variant_schema["numeric_features"])
    if categorical & numeric or categorical | numeric != set(feature_order):
        raise RuntimeError("Invalid v4 numeric/categorical feature partition")

    missing = [column for column in feature_order if column not in full_features.columns]
    if missing:
        raise RuntimeError(f"Feature builder did not construct required v4 features: {missing}")

    matrix = full_features[feature_order].copy()
    for column in numeric:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    for column in categorical:
        matrix[column] = matrix[column].map(clean_text).replace("", MISSING_CAT).astype(str)
    return matrix


def load_regressor(path: Path) -> CatBoostRegressor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing v4 model: {path}")
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def load_classifier(path: Path) -> CatBoostClassifier:
    if not path.is_file():
        raise FileNotFoundError(f"Missing v4 model: {path}")
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def apply_v4_production_models(
    root: Path,
    original: pd.DataFrame,
    full_features: pd.DataFrame,
) -> pd.DataFrame:
    manifest = load_json(root / "models/production_model.json")
    if manifest.get("active_model") != "v4_market_independent_outcomes":
        raise RuntimeError(
            f"Production manifest does not activate v4: {manifest.get('active_model')!r}"
        )

    schema = load_json(root / "models/step11_market_independent_feature_schema_v4.json")
    validation = load_json(root / "models/step14_market_independent_validation_v4.json")
    errors = load_json(root / "models/step14_error_distributions_v4.json")

    market_policy = schema.get("market_policy", {})
    if market_policy.get("market_inputs_used_for_training") is not False:
        raise RuntimeError("V4 schema does not certify market-independent training inputs")
    if market_policy.get("market_targets_used_for_training") is not False:
        raise RuntimeError("V4 schema does not certify market-independent training targets")
    if validation.get("overall_candidate_pass") is not True:
        raise RuntimeError("V4 historical forecast gate did not pass")

    selected = dict(validation.get("final_selected_variants", {}))
    if set(selected) != {"margin", "total", "moneyline"}:
        raise RuntimeError(f"Invalid v4 selected variant map: {selected}")

    predictions: dict[str, np.ndarray] = {}
    for target in ("margin", "total", "moneyline"):
        variant = selected[target]
        try:
            variant_schema = schema["variants"][variant]
        except KeyError as exc:
            raise RuntimeError(f"V4 schema missing selected variant {variant!r}") from exc

        matrix = prepare_v4_matrix(full_features, variant_schema)
        expected = list(variant_schema["feature_order"])

        if target == "margin":
            model = load_regressor(root / f"models/step11_margin_model_{variant}_v4.cbm")
            if list(model.feature_names_) != expected:
                raise RuntimeError("V4 margin model feature order mismatch")
            predictions[target] = np.asarray(model.predict(matrix), dtype=float)
        elif target == "total":
            model = load_regressor(root / f"models/step11_total_model_{variant}_v4.cbm")
            if list(model.feature_names_) != expected:
                raise RuntimeError("V4 total model feature order mismatch")
            predictions[target] = np.asarray(model.predict(matrix), dtype=float)
        else:
            model = load_classifier(root / f"models/step11_moneyline_model_{variant}_v4.cbm")
            if list(model.feature_names_) != expected:
                raise RuntimeError("V4 moneyline model feature order mismatch")
            raw_home_win = np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
            platt = errors["moneyline_platt"]
            predictions[target] = apply_platt(
                raw_home_win,
                float(platt["intercept"]),
                float(platt["slope"]),
            )

    predicted_margin = predictions["margin"]
    predicted_total = predictions["total"]
    home_win = predictions["moneyline"]
    if not np.isfinite(predicted_margin).all() or not np.isfinite(predicted_total).all():
        raise RuntimeError("V4 model produced non-finite score predictions")
    if len(predicted_margin) != len(original) or len(predicted_total) != len(original):
        raise RuntimeError("V4 prediction row count differs from projection input")

    spread_line = pd.to_numeric(full_features["spread_line"], errors="coerce").to_numpy(dtype=float)
    total_line = pd.to_numeric(full_features["total_line"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(spread_line).all():
        raise RuntimeError("Current spread_line is missing/non-finite")
    if not np.isfinite(total_line).all():
        raise RuntimeError("Current total_line is missing/non-finite")

    away_win = 1.0 - home_win
    home_cover = empirical_probability_greater(
        errors["margin_error_distribution"]["errors"],
        spread_line - predicted_margin,
    )
    away_cover = 1.0 - home_cover
    over = empirical_probability_greater(
        errors["total_error_distribution"]["errors"],
        total_line - predicted_total,
    )
    under = 1.0 - over

    validate_probability_pair(home_win, away_win, "moneyline")
    validate_probability_pair(home_cover, away_cover, "spread")
    validate_probability_pair(over, under, "total")

    predicted_home_score = (predicted_total + predicted_margin) / 2.0
    predicted_away_score = (predicted_total - predicted_margin) / 2.0

    output = original.copy()
    output["predicted_margin"] = predicted_margin
    output["predicted_total"] = predicted_total
    output["predicted_home_score"] = predicted_home_score
    output["predicted_away_score"] = predicted_away_score
    output["home_win_probability"] = home_win
    output["away_win_probability"] = away_win
    output["home_cover_probability"] = home_cover
    output["away_cover_probability"] = away_cover
    output["over_probability"] = over
    output["under_probability"] = under

    expected_columns = [*original.columns.tolist(), *OUTPUT_COLUMNS]
    if output.columns.tolist() != expected_columns:
        raise RuntimeError("Final output is not original columns plus the exact 10 production prediction columns")
    if output["game_id"].tolist() != original["game_id"].tolist():
        raise RuntimeError("Final v4 output game_id order changed")
    if not output["home_team"].equals(original["home_team"]):
        raise RuntimeError("Final v4 output home_team values changed")
    if not output["away_team"].equals(original["away_team"]):
        raise RuntimeError("Final v4 output away_team values changed")
    return output

#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

SCRIPT_DIR = Path(__file__).resolve().parent
NFL_ROOT = SCRIPT_DIR.parents[1]
TRAINING_DIR = NFL_ROOT / "training"
MODELS_DIR = NFL_ROOT / "models"
BACKTEST_DIR = TRAINING_DIR / "backtests"
HISTORICAL_FILE_PATTERN = re.compile(r"^historical_core_(\d{4})\.csv$")
DEFAULT_START_SEASON = 2021
MISSING_CATEGORY = "__MISSING__"
BLANK_TOKENS = {"", "nan", "none", "null", "<na>", "nat"}
PROB_EPS = 1e-6

TEAM_METRICS = [
    "off_epa_per_play", "def_epa_per_play", "off_success_rate",
    "def_success_rate", "yards_per_play", "yards_per_play_allowed",
    "points_per_drive", "points_per_drive_allowed", "red_zone_td_rate",
    "red_zone_td_rate_allowed", "early_down_epa", "third_down_conversion_rate",
]
QB_METRICS = [
    "epa_per_play", "cpoe", "air_yards", "sack_rate",
    "interception_rate", "fumble_rate",
]
SCHEDULE_FEATURES = [
    "game_type", "week", "weekday", "gametime", "away_team", "home_team",
    "location", "away_rest", "home_rest", "div_game", "roof", "surface",
    "temp", "wind", "stadium_id", "stadium", "hist_surface",
    "hist_weather_icon", "hist_temperature", "hist_precip_probability",
    "hist_precip_type", "hist_wind_speed", "hist_wind_bearing", "rest_diff",
    "miles_traveled", "time_zones_crossed", "east_to_west", "west_to_east",
    "international_flag", "neutral_site_flag",
]
EXTERNAL_SAFE_EXACT = ["drat_away_prob", "drat_home_prob"]

MARKET_EXACT = {
    "away_moneyline", "home_moneyline", "spread_line", "away_spread_odds",
    "home_spread_odds", "total_line", "under_odds", "over_odds",
    "hist_odds_total", "hist_home_spread", "hist_away_spread",
    "drat_away_moneyline", "drat_home_moneyline", "drat_away_spread",
    "drat_home_spread",
}
OUTCOME_EXACT = {
    "away_score", "home_score", "margin", "total_points", "home_win",
    "home_ats_margin", "home_ats_result", "total_result", "result", "total",
}
IDENTITY_EXACT = {
    "game_id", "season", "gameday", "away_qb_id", "home_qb_id",
    "away_qb_name", "home_qb_name", "away_coach", "home_coach",
}
FORBIDDEN_PREFIXES = ("ml_", "ats_", "totals_")
FORCED_CATEGORICAL = {
    "game_type", "weekday", "gametime", "away_team", "home_team", "location",
    "roof", "surface", "stadium_id", "stadium", "hist_surface",
    "hist_weather_icon", "hist_precip_type",
}

REGRESSOR_PARAMS = {
    "loss_function": "RMSE", "eval_metric": "RMSE", "iterations": 450,
    "learning_rate": 0.025, "depth": 5, "l2_leaf_reg": 25.0,
    "random_strength": 1.5, "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0, "one_hot_max_size": 10,
    "max_ctr_complexity": 1, "random_seed": 42, "thread_count": -1,
    "allow_writing_files": False, "verbose": False,
}
CLASSIFIER_PARAMS = {
    "loss_function": "Logloss", "eval_metric": "Logloss", "iterations": 350,
    "learning_rate": 0.02, "depth": 4, "l2_leaf_reg": 35.0,
    "random_strength": 2.0, "bootstrap_type": "Bayesian",
    "bagging_temperature": 1.0, "one_hot_max_size": 10,
    "max_ctr_complexity": 1, "random_seed": 42, "thread_count": -1,
    "allow_writing_files": False, "verbose": False,
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_training_seasons(start_season: int = DEFAULT_START_SEASON, end_season: int | None = None) -> list[int]:
    available = sorted({
        int(match.group(1))
        for path in TRAINING_DIR.iterdir()
        if path.is_file() and (match := HISTORICAL_FILE_PATTERN.fullmatch(path.name))
    })
    if not available or start_season not in available:
        fail(f"Missing historical season files; available={available}")
    resolved_end = max(available) if end_season is None else int(end_season)
    seasons = list(range(start_season, resolved_end + 1))
    missing = [season for season in seasons if season not in available]
    if missing:
        fail(f"Historical season files must be contiguous; missing={missing}")
    return seasons


def read_inputs(seasons: list[int]) -> tuple[pd.DataFrame, dict[str, str]]:
    frames = []
    hashes: dict[str, str] = {}
    reference_columns = None
    required = {
        "game_id", "season", "margin", "total_points", "spread_line", "total_line",
        "away_moneyline", "home_moneyline", "away_spread_odds", "home_spread_odds",
        "under_odds", "over_odds",
    }
    for season in seasons:
        path = TRAINING_DIR / f"historical_core_{season}.csv"
        if not path.exists():
            fail(f"Missing input: {path}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig", low_memory=False)
        if frame.empty or len(frame.columns) != len(set(frame.columns)):
            fail(f"{path}: empty or duplicate columns")
        missing = sorted(required - set(frame.columns))
        if missing:
            fail(f"{path}: missing required columns: {missing}")
        if reference_columns is None:
            reference_columns = list(frame.columns)
        elif list(frame.columns) != reference_columns:
            fail(f"{path}: schema/order differs across seasons")
        season_values = pd.to_numeric(frame["season"], errors="coerce")
        if season_values.isna().any() or not (season_values.astype(int) == season).all():
            fail(f"{path}: contains wrong-season rows")
        frames.append(frame)
        hashes[path.name] = sha256_file(path)
    raw = pd.concat(frames, ignore_index=True)
    ids = raw["game_id"].astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        fail("game_id must be populated and globally unique")
    return raw, hashes


def team_feature_names() -> list[str]:
    return [c for metric in TEAM_METRICS for c in (f"home_{metric}", f"away_{metric}", f"{metric}_diff")]


def qb_feature_names() -> list[str]:
    return [c for metric in QB_METRICS for c in (f"home_qb_{metric}", f"away_qb_{metric}", f"qb_{metric}_diff")]


def is_forbidden(column: str) -> bool:
    return (
        column in MARKET_EXACT or column in OUTCOME_EXACT or column in IDENTITY_EXACT
        or column.startswith(FORBIDDEN_PREFIXES) or column.endswith("_result")
    )


def build_feature_variants(columns: list[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    column_set = set(columns)
    team = team_feature_names()
    qb = qb_feature_names()
    schedule = list(SCHEDULE_FEATURES)
    depth_injury = [c for c in columns if ("inj_" in c or "depth_starter_changes" in c) and not is_forbidden(c)]
    core = list(dict.fromkeys(team + qb + schedule + depth_injury))

    epred = [c for c in columns if c.startswith("epred_") and not is_forbidden(c)]
    drat = [c for c in EXTERNAL_SAFE_EXACT if c in column_set and not is_forbidden(c)]
    augmented = list(dict.fromkeys(core + drat + epred))

    missing = [c for c in core if c not in column_set]
    if missing:
        fail(f"Required core v4 features missing: {missing}")

    for variant, features in {"core": core, "augmented": augmented}.items():
        bad = [c for c in features if is_forbidden(c)]
        if bad:
            fail(f"Forbidden feature entered {variant}: {bad}")
        marketish = [c for c in features if any(token in c.lower() for token in ("moneyline", "spread", "total_line", "odds_total", "_odds"))]
        if marketish:
            fail(f"Market-like feature entered {variant}: {marketish}")

    families = {
        "team_lagged": team,
        "qb_lagged": qb,
        "schedule_rest_venue_weather_travel": schedule,
        "depth_injury": depth_injury,
        "external_drat_probability": drat,
        "external_epred": epred,
    }
    return {"core": core, "augmented": augmented}, families


def looks_numeric(series: pd.Series) -> bool:
    text = series.astype(str).str.strip()
    nonblank = text[~text.str.casefold().isin(BLANK_TOKENS)]
    if nonblank.empty:
        return True
    converted = pd.to_numeric(nonblank.str.replace("%", "", regex=False), errors="coerce")
    return bool(converted.notna().all())


def infer_feature_types(raw: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    categorical, numeric = [], []
    for c in features:
        if c in FORCED_CATEGORICAL or not looks_numeric(raw[c]):
            categorical.append(c)
        else:
            numeric.append(c)
    return categorical, numeric


def prepare_matrix(raw: pd.DataFrame, features: list[str], categorical: list[str], numeric: list[str]) -> pd.DataFrame:
    matrix = raw[features].copy()
    for c in numeric:
        text = matrix[c].astype(str).str.strip().str.replace("%", "", regex=False)
        blank = text.str.casefold().isin(BLANK_TOKENS)
        values = pd.to_numeric(text.mask(blank, np.nan), errors="coerce")
        if ((~blank) & values.isna()).any():
            fail(f"Numeric conversion failed for {c}")
        matrix[c] = values
    for c in categorical:
        text = matrix[c].astype(str).str.strip()
        matrix[c] = text.mask(text.str.casefold().isin(BLANK_TOKENS), MISSING_CATEGORY)
    return matrix


def numeric(raw: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(raw[column], errors="coerce")
    if values.isna().any():
        fail(f"Invalid numeric values in {column}")
    return values.astype(float)


def target_home_win(raw: pd.DataFrame) -> pd.Series:
    margin = numeric(raw, "margin")
    return pd.Series(np.where(margin > 0, 1.0, np.where(margin < 0, 0.0, np.nan)), index=raw.index)


def train_regressor(X: pd.DataFrame, y: pd.Series, cat_indices: list[int]) -> CatBoostRegressor:
    model = CatBoostRegressor(**REGRESSOR_PARAMS)
    model.fit(Pool(X, label=y, cat_features=cat_indices, feature_names=list(X.columns)), verbose=False)
    return model


def train_classifier(X: pd.DataFrame, y: pd.Series, cat_indices: list[int]) -> CatBoostClassifier:
    labels = y.astype(int)
    if set(labels.unique()) != {0, 1}:
        fail("Classifier target lacks both classes")
    model = CatBoostClassifier(**CLASSIFIER_PARAMS)
    model.fit(Pool(X, label=labels, cat_features=cat_indices, feature_names=list(X.columns)), verbose=False)
    return model


def predict_regressor(model: CatBoostRegressor, X: pd.DataFrame, cat_indices: list[int]) -> np.ndarray:
    return np.asarray(model.predict(Pool(X, cat_features=cat_indices, feature_names=list(X.columns))), dtype=float)


def predict_classifier(model: CatBoostClassifier, X: pd.DataFrame, cat_indices: list[int]) -> np.ndarray:
    p = np.asarray(model.predict_proba(Pool(X, cat_features=cat_indices, feature_names=list(X.columns)))[:, 1], dtype=float)
    return np.clip(p, PROB_EPS, 1.0 - PROB_EPS)


def american_implied(values: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x) & (x != 0)
    pos = ok & (x > 0)
    neg = ok & (x < 0)
    out[pos] = 100.0 / (x[pos] + 100.0)
    out[neg] = (-x[neg]) / ((-x[neg]) + 100.0)
    return out


def no_vig_positive(positive_odds: pd.Series | np.ndarray, negative_odds: pd.Series | np.ndarray) -> np.ndarray:
    pos = american_implied(positive_odds)
    neg = american_implied(negative_odds)
    denom = pos + neg
    out = np.full_like(pos, np.nan, dtype=float)
    ok = np.isfinite(denom) & (denom > 0)
    out[ok] = pos[ok] / denom[ok]
    return out


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    return {"mae": float(np.mean(np.abs(y - pred))), "rmse": float(np.sqrt(np.mean((y - pred) ** 2)))}


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), PROB_EPS, 1 - PROB_EPS)
    return {
        "brier": float(np.mean((y - p) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(np.mean((p >= 0.5) == y.astype(int))),
    }


def fit_platt(raw_p: np.ndarray, y: np.ndarray, max_iter: int = 100) -> tuple[float, float]:
    p = np.clip(np.asarray(raw_p, float), PROB_EPS, 1 - PROB_EPS)
    y = np.asarray(y, float)
    x = np.log(p / (1 - p))
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.array([0.0, 1.0], dtype=float)
    ridge = np.diag([1e-8, 1e-6])
    for _ in range(max_iter):
        z = X @ beta
        q = 1 / (1 + np.exp(-np.clip(z, -35, 35)))
        grad = X.T @ (q - y) + ridge @ beta
        w = np.clip(q * (1 - q), 1e-8, None)
        hess = X.T @ (X * w[:, None]) + ridge
        beta_new = beta - np.linalg.solve(hess, grad)
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            beta = beta_new
            break
        beta = beta_new
    return float(beta[0]), float(beta[1])


def apply_platt(raw_p: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    p = np.clip(np.asarray(raw_p, float), PROB_EPS, 1 - PROB_EPS)
    x = np.log(p / (1 - p))
    z = intercept + slope * x
    return np.clip(1 / (1 + np.exp(-np.clip(z, -35, 35))), PROB_EPS, 1 - PROB_EPS)


def empirical_probability_greater(errors: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    errors = np.sort(np.asarray(errors, float))
    thresholds = np.asarray(thresholds, float)
    left = np.searchsorted(errors, thresholds, side="left")
    right = np.searchsorted(errors, thresholds, side="right")
    return ((len(errors) - right) + 0.5 * (right - left)) / len(errors)

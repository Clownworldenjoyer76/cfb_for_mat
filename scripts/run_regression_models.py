# scripts/run_regression_models.py

import os
import sys
import re
import datetime as dt
import pandas as pd
import numpy as np

from typing import Tuple, Optional, Sequence, List, Dict

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib


# ======================
# I/O & Config
# ======================
INPUT_CSV    = "data/modeling_dataset.csv"
OUT_METRICS  = "data/model_regression_metrics.csv"
OUT_LOG      = "logs_model_regression.txt"
OUT_COEFFS   = "data/model_regression_coeffs.csv"
MODEL_DIR    = "models"

# Allow explicit override, e.g. TARGET_COL=points_scored
TARGET_OVERRIDE = os.environ.get("TARGET_COL", "").strip() or None

# Columns never sent to model
ID_COLUMNS = [
    "game_id", "team", "opponent", "home_team", "away_team",
    "venue", "venue_name"
]

# Datetime columns to encode if present
DATETIME_COLUMNS = ["start_date", "kickoff", "game_datetime"]

# Candidate targets (first present & numeric wins if no override provided)
TARGET_CANDIDATES = [
    "ts_points", "ts_points_per_game", "eff_points",
    "points_scored", "points_for", "eff_off_overall_ppa"
]

RANDOM_STATE = 42


# ======================
# Preprocessing helpers
# ======================
def encode_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Expand known datetime columns into numeric components and drop originals."""
    for col in DATETIME_COLUMNS:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[f"{col}_year"]  = s.dt.year
            df[f"{col}_month"] = s.dt.month
            df[f"{col}_day"]   = s.dt.day
            df[f"{col}_hour"]  = s.dt.hour
            df[f"{col}_dow"]   = s.dt.weekday
            df.drop(columns=[col], inplace=True)
    return df


def select_target(df: pd.DataFrame) -> str:
    """Pick a valid numeric target. Fail fast if none."""
    if TARGET_OVERRIDE:
        if TARGET_OVERRIDE not in df.columns:
            raise ValueError(f"TARGET_COL override '{TARGET_OVERRIDE}' not found in columns.")
        if not pd.api.types.is_numeric_dtype(df[TARGET_OVERRIDE]):
            raise ValueError(f"TARGET_COL '{TARGET_OVERRIDE}' is not numeric.")
        return TARGET_OVERRIDE

    for c in TARGET_CANDIDATES:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c

    raise ValueError(
        "No valid target found. Expected one of: "
        + ", ".join(TARGET_CANDIDATES)
        + " or set TARGET_COL env var to a numeric column."
    )


def _exclusion_patterns_for_target(target_col: str) -> List[re.Pattern]:
    """
    Rules to drop feature families that are likely to leak the target.
    For `eff_off_*_ppa` targets, exclude ALL offensive PPA features, both per-down and cumulative.
    """
    pats: List[str] = []
    if re.fullmatch(r"eff_off_.*_ppa", target_col):
        # Exclude any offensive PPA metric siblings (includes cum_ variants and down splits)
        pats.append(r"^eff_off_.*_ppa$")
    elif target_col == "eff_off_overall_ppa":
        pats.append(r"^eff_off_.*_ppa$")

    # Add more rules here for other target families if needed

    return [re.compile(p) for p in pats]


def _apply_feature_exclusions(X: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns in X that match exclusion patterns for the given target."""
    pats = _exclusion_patterns_for_target(target_col)
    if not pats:
        return X, []

    drop_cols: List[str] = []
    for c in X.columns:
        for p in pats:
            if p.search(c):
                drop_cols.append(c)
                break

    drop_cols = sorted(set(drop_cols))
    if drop_cols:
        X = X.drop(columns=drop_cols, errors="ignore")

    return X, drop_cols


def build_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Drop IDs, expand datetimes, keep numerics, drop all-NaN cols, split X/y, apply exclusions."""
    df = df.copy()

    # Drop identifier-ish columns if present
    drop_cols = [c for c in ID_COLUMNS if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Encode datetimes into numeric parts
    df = encode_datetime(df)

    # Target validations
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"Target column is not numeric: {target_col}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Keep only numeric features
    X = X.select_dtypes(include=["number"])

    # Drop columns that are entirely NaN to avoid imputer errors
    if X.shape[1] > 0:
        X = X.loc[:, X.notna().any(axis=0)]

    # Apply exclusion rules based on target (to prevent leakage)
    X, excluded = _apply_feature_exclusions(X, target_col)

    if X.shape[0] == 0:
        raise ValueError("Dataset has zero rows after preprocessing.")
    if X.shape[1] == 0:
        raise ValueError("No numeric features available after preprocessing (all excluded?).")

    return X, y, excluded


# ======================
# Splitting logic
# ======================
def temporal_split_indices(df_raw: pd.DataFrame, test_frac: float = 0.2) -> Optional[Tuple[Sequence[int], Sequence[int]]]:
    """
    If both 'season' and 'week' exist, perform a temporal split:
      - sort by season, week, (stable original index)
      - take the last portion as test set
    Returns (train_label_index, test_label_index) as **label indices** that match df_raw.index.
    """
    if {"season", "week"}.issubset(df_raw.columns):
        tmp = df_raw.reset_index(drop=False).rename(columns={"index": "__row"})
        tmp.sort_values(["season", "week", "__row"], inplace=True)
        n = len(tmp)
        if n == 0:
            return [], []
        split = int(n * (1 - test_frac))
        # Return original index labels (not positions)
        train_labels = tmp.iloc[:split]["__row"].tolist()
        test_labels  = tmp.iloc[split:]["__row"].tolist()
        return train_labels, test_labels
    return None


# ======================
# Modeling
# ======================
def make_pipelines() -> Dict[str, Pipeline]:
    linear = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("model", LinearRegression())
    ])

    ridge = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))
    ])

    lasso = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("model", Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000))
    ])

    return {
        "linear": linear,
        "ridge": ridge,
        "lasso": lasso
    }


def extract_coefficients(pipe: Pipeline, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame of coefficients for linear-like models.
    Coefficients are with respect to the transformed feature space; since our
    transformers (imputer/scaler) keep the same dimensionality/order, we map
    them back to the original numeric feature names.
    """
    model = pipe.named_steps.get("model")
    if model is None or not hasattr(model, "coef_"):
        return None

    coefs = model.coef_
    if isinstance(coefs, np.ndarray) and coefs.ndim > 1:
        coefs = coefs[0]

    df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def train_and_eval(X: pd.DataFrame,
                   y: pd.Series,
                   raw_df: pd.DataFrame) -> Tuple[list, int, int, Dict[str, pd.DataFrame], str]:
    """
    Train/eval three models. Prefer temporal split if season/week present,
    else fall back to random split with fixed seed.
    Also returns per-model coefficient DataFrames and split_mode.
    """
    # Choose split strategy
    idxs = temporal_split_indices(raw_df)
    if idxs:
        train_labels, test_labels = idxs
        if not train_labels or not test_labels:
            raise ValueError("Temporal split produced empty train or test indices.")
        X_train, X_test = X.loc[train_labels], X.loc[test_labels]
        y_train, y_test = y.loc[train_labels], y.loc[test_labels]
        split_mode = "temporal(season,week)"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        split_mode = "random"

    models = make_pipelines()
    metrics = []
    coeffs_by_model: Dict[str, pd.DataFrame] = {}

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
        mae  = float(mean_absolute_error(y_test, y_pred))
        r2   = float(r2_score(y_test, y_pred))

        metrics.append({
            "model": name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_features": int(X.shape[1]),
            "split_mode": split_mode
        })

        joblib.dump(pipe, os.path.join(MODEL_DIR, f"{name}.pkl"))

        coef_df = extract_coefficients(pipe, list(X.columns))
        if coef_df is not None:
            coef_df.insert(0, "model", name)
            coef_df["n_features"] = int(X.shape[1])
            coef_df["split_mode"] = split_mode
            coeffs_by_model[name] = coef_df

    return metrics, X.shape[1], len(y), coeffs_by_model, split_mode


# ======================
# Main
# ======================
def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input dataset not found: {INPUT_CSV}")

    df_raw = pd.read_csv(INPUT_CSV)
    if df_raw.shape[0] == 0:
        raise ValueError("Input dataset is empty.")

    # Resolve target
    target_col = select_target(df_raw)

    # Drop rows with NaN target (prevents 'Input y contains NaN')
    if df_raw[target_col].isna().any():
        n_missing = int(df_raw[target_col].isna().sum())
        print(f"[WARN] Dropping {n_missing} rows with NaN target values in '{target_col}'")
        df_raw = df_raw.dropna(subset=[target_col])
        if df_raw.shape[0] == 0:
            raise ValueError(f"All rows dropped due to NaN target '{target_col}'.")

    # Build features (+ exclusions)
    X, y, excluded = build_features(df_raw, target_col)

    # Train & evaluate (+ collect coeffs)
    metrics, n_feats, n_rows, coeffs_by_model, split_mode = train_and_eval(X, y, df_raw)

    # Ensure output dirs
    for p in [OUT_METRICS, OUT_LOG, OUT_COEFFS]:
        d = os.path.dirname(p) or "."
        os.makedirs(d, exist_ok=True)

    # Write metrics CSV
    pd.DataFrame(metrics).to_csv(OUT_METRICS, index=False)

    # Write coefficients CSV (concat available models)
    if coeffs_by_model:
        coeff_all = pd.concat(list(coeffs_by_model.values()), ignore_index=True)
        coeff_all.to_csv(OUT_COEFFS, index=False)
    else:
        pd.DataFrame(columns=["model","feature","coef","abs_coef","rank","n_features","split_mode"]).to_csv(OUT_COEFFS, index=False)

    # Write log
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write(f"snapshot_utc={dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        f.write(f"input_rows={n_rows}\n")
        f.write(f"features={n_feats}\n")
        f.write(f"target={target_col}\n")
        if excluded:
            f.write(f"excluded_features={len(excluded)} -> {', '.join(excluded[:20])}{' ...' if len(excluded)>20 else ''}\n")
        for m in metrics:
            f.write(
                f"{m['model']}: rmse={m['rmse']:.6f}, mae={m['mae']:.6f}, "
                f"r2={m['r2']:.6f}, n_train={m['n_train']}, n_test={m['n_test']}, "
                f"n_features={m['n_features']}, split_mode={m['split_mode']}\n"
            )

    print("Regression models complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[REGRESSION_ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise

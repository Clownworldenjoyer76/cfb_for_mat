# scripts/run_regression_models.py

import os
import sys
import datetime as dt
import pandas as pd
import numpy as np

from typing import Tuple, Optional, Sequence

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
INPUT_CSV   = "data/modeling_dataset.csv"
OUT_METRICS = "data/model_regression_metrics.csv"
OUT_LOG     = "logs_model_regression.txt"
MODEL_DIR   = "models"

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


def build_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Drop IDs, expand datetimes, keep numerics, drop all-NaN cols, split X/y."""
    # Defensive copy
    df = df.copy()

    # Drop identifiers if present
    drop_cols = [c for c in ID_COLUMNS if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Encode datetimes
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

    if X.shape[0] == 0:
        raise ValueError("Dataset has zero rows after preprocessing.")
    if X.shape[1] == 0:
        raise ValueError("No numeric features available after preprocessing.")

    return X, y


# ======================
# Splitting logic
# ======================
def temporal_split_indices(df_raw: pd.DataFrame, test_frac: float = 0.2) -> Optional[Tuple[Sequence[int], Sequence[int]]]:
    """
    If both 'season' and 'week' exist, perform a temporal split:
      - sort by season, week, (stable index)
      - take the last portion as test set
    Returns (train_index, test_index) or None if not possible.
    """
    if {"season", "week"}.issubset(df_raw.columns):
        tmp = df_raw.reset_index(drop=False).rename(columns={"index": "__row"})
        tmp.sort_values(["season", "week", "__row"], inplace=True)
        n = len(tmp)
        if n == 0:
            return [], []
        split = int(n * (1 - test_frac))
        train_idx = tmp.iloc[:split]["__row"].tolist()
        test_idx  = tmp.iloc[split:]["__row"].tolist()
        return train_idx, test_idx
    return None


# ======================
# Modeling
# ======================
def make_pipelines():
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


def train_and_eval(X: pd.DataFrame,
                   y: pd.Series,
                   raw_df: pd.DataFrame) -> Tuple[list, int, int]:
    """
    Train/eval three models. Prefer temporal split if season/week present,
    else fall back to random split with fixed seed.
    """
    # Choose split strategy
    idxs = temporal_split_indices(raw_df)
    if idxs:
        train_idx, test_idx = idxs
        if not train_idx or not test_idx:
            raise ValueError("Temporal split produced empty train or test indices.")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        split_mode = "temporal(season,week)"
    else:
        # Fall back to random split (fixed seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        split_mode = "random"

    models = make_pipelines()
    metrics = []

    # Ensure model dir exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # RMSE compatible with older sklearn (no 'squared' kwarg)
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

    return metrics, X.shape[1], len(y)


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

    # ---- NEW: drop rows with NaN in target (fixes 'Input y contains NaN') ----
    if df_raw[target_col].isna().any():
        n_missing = int(df_raw[target_col].isna().sum())
        print(f"[WARN] Dropping {n_missing} rows with NaN target values in '{target_col}'")
        df_raw = df_raw.dropna(subset=[target_col])
        if df_raw.shape[0] == 0:
            raise ValueError(f"All rows dropped due to NaN target '{target_col}'.")

    # Build features
    X, y = build_features(df_raw, target_col)

    # Train & evaluate
    metrics, n_feats, n_rows = train_and_eval(X, y, df_raw)

    # Ensure output dirs
    out_metrics_dir = os.path.dirname(OUT_METRICS) or "."
    out_log_dir     = os.path.dirname(OUT_LOG) or "."
    os.makedirs(out_metrics_dir, exist_ok=True)
    os.makedirs(out_log_dir, exist_ok=True)

    # Write metrics CSV
    pd.DataFrame(metrics).to_csv(OUT_METRICS, index=False)

    # Write log
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write(f"snapshot_utc={dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        f.write(f"input_rows={n_rows}\n")
        f.write(f"features={n_feats}\n")
        f.write(f"target={target_col}\n")
        # include split mode per model (duplicated but explicit)
        for m in metrics:
            f.write(
                f"{m['model']}: rmse={m['rmse']:.6f}, mae={m['mae']:.6f}, "
                f"r2={m['r2']:.6f}, n_train={m['n_train']}, n_test={m['n_test']}, "
                f"n_features={m['n_features']}, split_mode={m['split_mode']}\n"
            )

    print("Regression models complete.")


if __name__ == "__main__":
    # Make failures loud and clear in CI
    try:
        main()
    except Exception as e:
        print(f"[REGRESSION_ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise

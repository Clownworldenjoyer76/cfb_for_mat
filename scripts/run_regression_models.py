import os
import datetime as dt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib


# I/O
INPUT_CSV   = "data/modeling_dataset.csv"
OUT_METRICS = "data/model_regression_metrics.csv"
OUT_LOG     = "logs_model_regression.txt"
MODEL_DIR   = "models"

# Columns never sent to model
ID_COLUMNS = [
    "game_id", "team", "opponent", "home_team", "away_team",
    "venue", "venue_name"
]

# Datetime columns to encode if present
DATETIME_COLUMNS = ["start_date", "kickoff", "game_datetime"]


def encode_datetime(df: pd.DataFrame) -> pd.DataFrame:
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
    candidates = [
        "ts_points", "ts_points_per_game", "eff_points",
        "points_scored", "points_for"
    ]
    for c in candidates:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return "week"


def build_features(df: pd.DataFrame, target_col: str):
    drop_cols = [c for c in ID_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = encode_datetime(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"Target column is not numeric: {target_col}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Keep only numeric features
    X = X.select_dtypes(include=["number"])

    # DROP columns that are entirely NaN to avoid SimpleImputer "no observed values" error
    X = X.loc[:, X.notna().any(axis=0)]

    if X.shape[1] == 0:
        raise ValueError("No numeric features available after preprocessing.")

    return X, y


def train_and_eval(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    linear = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("model", LinearRegression())
    ])

    ridge = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("model", Ridge(alpha=1.0, random_state=42))
    ])

    lasso = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("model", Lasso(alpha=0.01, random_state=42, max_iter=10000))
    ])

    models = {
        "linear": linear,
        "ridge": ridge,
        "lasso": lasso
    }

    metrics = []
    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # RMSE compatible with older sklearn (no 'squared' kwarg)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        metrics.append({
            "model": name,
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_features": int(X.shape[1]))
        )

        joblib.dump(pipe, os.path.join(MODEL_DIR, f"{name}.pkl"))

    return metrics, X.shape[1], len(y)


def main():
    df = pd.read_csv(INPUT_CSV)

    target_col = select_target(df)
    X, y = build_features(df, target_col)

    metrics, n_feats, n_rows = train_and_eval(X, y)

    pd.DataFrame(metrics).to_csv(OUT_METRICS, index=False)

    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write(f"snapshot_utc={dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        f.write(f"input_rows={n_rows}\n")
        f.write(f"features={n_feats}\n")
        f.write(f"target={target_col}\n")
        for m in metrics:
            f.write(
                f"{m['model']}: rmse={m['rmse']:.6f}, mae={m['mae']:.6f}, "
                f"r2={m['r2']:.6f}, n_train={m['n_train']}, n_test={m['n_test']}, "
                f"n_features={m['n_features']}\n"
            )
    print("Regression models complete.")


if __name__ == "__main__":
    main()

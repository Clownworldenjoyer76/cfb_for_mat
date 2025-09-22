#!/usr/bin/env python3
"""
CFB Regression Runner (strict (game_id, team) outcome join)

- Loads data/modeling_dataset.csv
- Attaches outcome targets from a per-team scores file that is UNIQUE on (game_id, team)
- Builds features with numeric + safe one-hot categoricals
- Excludes leaky columns via EXCLUDE_TARGET_REGEX (keeps the actual target)
- Temporal split by (season, week)
- Trains LinearRegression, RidgeCV, LassoCV
- Drops rows with NaN in target before training
- Writes:
    logs_model_regression.txt
    data/model_regression_metrics.csv
    data/model_regression_coeffs.csv
    models/<model>_model.pkl
"""

from pathlib import Path
import os, sys, re, math
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
LOG_FILE   = Path("logs_model_regression.txt")
METRICS_CSV = DATA_DIR / "model_regression_metrics.csv"
COEFFS_CSV  = DATA_DIR / "model_regression_coeffs.csv"
INPUT_CSV   = DATA_DIR / "modeling_dataset.csv"

SCORES_CSV = os.environ.get("SCORES_CSV", str(DATA_DIR / "game_scores_clean.csv"))
TARGET_COL = os.environ.get("TARGET_COL", "eff_off_overall_ppa")
EXCLUDE_TARGET_REGEX = os.environ.get("EXCLUDE_TARGET_REGEX", None)

def log(msg: str):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

def attach_outcome_targets(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Already enriched?
    if {"points_scored","points_allowed","margin"}.issubset({c.lower() for c in df_raw.columns}):
        return df_raw
    if not Path(SCORES_CSV).exists():
        raise FileNotFoundError(f"{SCORES_CSV} not found")

    scores = pd.read_csv(SCORES_CSV)
    if not {"game_id","team","points_scored"}.issubset({c.lower() for c in scores.columns}):
        raise KeyError("scores file must have game_id, team, points_scored")

    colmap = {c.lower(): c for c in scores.columns}
    s = scores.rename(columns={
        colmap["game_id"]: "game_id",
        colmap["team"]: "team",
        colmap["points_scored"]: "points_scored",
    })
    if "points_allowed" not in s.columns:
        s["points_allowed"] = np.nan
    if "margin" not in s.columns:
        s["margin"] = s["points_scored"] - s["points_allowed"]

    s = s.sort_values(["game_id","team"]).drop_duplicates(subset=["game_id","team"], keep="first")
    merged = df_raw.merge(s[["game_id","team","points_scored","points_allowed","margin"]],
                          on=["game_id","team"], how="left", validate="m:1")
    log(f"outcome_join_mode=strict(game_id,team)")
    log(f"outcome_merge_missing_rows={merged['points_scored'].isna().sum()}")

    return merged

def select_target_column(df: pd.DataFrame, target_override: str) -> str:
    if target_override in df.columns:
        return target_override
    for c in df.columns:
        if c.lower() == target_override.lower():
            return c
    raise ValueError(f"TARGET_COL '{target_override}' not found")

def exclude_leaky_features(X: pd.DataFrame, target_col: str, pattern: str | None):
    dropped = []
    if pattern:
        rx = re.compile(pattern)
        for col in list(X.columns):
            if col != target_col and rx.search(col):
                dropped.append(col)
        X = X.drop(columns=dropped, errors="ignore")
    return X, dropped

def build_features(df: pd.DataFrame, target_col: str):
    y = pd.to_numeric(df[target_col], errors="coerce")
    drop_cols = {target_col,"home_points","away_points","points_allowed","margin"}
    meta_like = {"game_id","start_date","wx_start_date"}
    drop_cols |= (set(df.columns) & meta_like)

    num_df = df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=[np.number])
    cat_df = df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=["object","category","bool"])
    keep_cats = [c for c in cat_df.columns if c in ("team","opponent","conference","st_conference")]
    cat_df = cat_df[keep_cats].fillna("NA")
    if not cat_df.empty:
        cat_oh = pd.get_dummies(cat_df, prefix=cat_df.columns, drop_first=True)
        X = pd.concat([num_df, cat_oh], axis=1)
    else:
        X = num_df.copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y

def fit_and_eval(X_train, y_train, X_test, y_test, model_name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)
    return rmse, mae, r2, pred, model

def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(INPUT_CSV)
    df_raw = attach_outcome_targets(df_raw)
    target = select_target_column(df_raw, TARGET_COL)

    # Drop rows with NaN target
    before = len(df_raw)
    df_raw = df_raw.dropna(subset=[target])
    after = len(df_raw)
    dropped = before - after
    log(f"dropped_NaN_targets={dropped}")

    X_all, y_all = build_features(df_raw, target)
    X_all, dropped_feats = exclude_leaky_features(X_all, target, EXCLUDE_TARGET_REGEX)

    df_all = df_raw.loc[X_all.index]
    df_all = df_all.assign(__row_id=np.arange(len(df_all)))
    if {"season","week"}.issubset(df_all.columns):
        df_all_sorted = df_all.sort_values(["season","week","__row_id"], kind="mergesort")
        split_mode = "temporal(season,week)"
    else:
        df_all_sorted = df_all.sort_values(["__row_id"], kind="mergesort")
        split_mode = "index"

    X_all = X_all.loc[df_all_sorted.index]
    y_all = y_all.loc[df_all_sorted.index]

    cut = int(len(df_all_sorted) * 0.8)
    train_idx = df_all_sorted.index[:cut]
    test_idx  = df_all_sorted.index[cut:]

    X_train, y_train = X_all.loc[train_idx], y_all.loc[train_idx]
    X_test,  y_test  = X_all.loc[test_idx],  y_all.loc[test_idx]

    ridge_alphas = np.logspace(-3, 3, 13)
    lasso_alphas = np.logspace(-3, 1, 9)
    models = {
        "linear": Pipeline([("scaler", StandardScaler(with_mean=False)), ("lin", LinearRegression())]),
        "ridge":  Pipeline([("scaler", StandardScaler(with_mean=False)), ("rid", RidgeCV(alphas=ridge_alphas))]),
        "lasso":  Pipeline([("scaler", StandardScaler(with_mean=False)), ("las", LassoCV(alphas=lasso_alphas, max_iter=5000, cv=5))]),
    }

    metrics_rows, coeff_rows = [], []
    for name, pipe in models.items():
        rmse, mae, r2, pred, fitted = fit_and_eval(X_train, y_train, X_test, y_test, name, pipe)
        log(f"{name}: rmse={rmse:.6f}, mae={mae:.6f}, r2={r2:.6f}, "
            f"n_train={len(X_train)}, n_test={len(X_test)}, n_features={X_train.shape[1]}, split_mode={split_mode}")
        metrics_rows.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})

        out_path = MODELS_DIR / f"{name}_model.pkl"
        joblib.dump(fitted, out_path)

        try:
            if name == "linear": est = fitted.named_steps["lin"]
            elif name == "ridge": est = fitted.named_steps["rid"]
            else: est = fitted.named_steps["las"]
            feature_names = list(X_train.columns)
            coefs = np.ravel(est.coef_)
            for feat, coef in zip(feature_names, coefs):
                coeff_rows.append({"model": name, "feature": feat, "coefficient": float(coef)})
            if hasattr(est, "intercept_"):
                coeff_rows.append({"model": name, "feature": "__intercept__", "coefficient": float(np.atleast_1d(est.intercept_)[0])})
        except Exception as e:
            log(f"WARNING: coef extraction failed for {name}: {e}")

    pd.DataFrame(metrics_rows).to_csv(METRICS_CSV, index=False)
    pd.DataFrame(coeff_rows).to_csv(COEFFS_CSV, index=False)

if __name__ == "__main__":
    if EXCLUDE_TARGET_REGEX is None and TARGET_COL.lower() in {"points_scored","points","pts","score","margin"}:
        EXCLUDE_TARGET_REGEX = rf"(?i)^(?!{re.escape(TARGET_COL)}$).*(points?|scores?|margin|spread)$"
    try:
        main()
    except Exception as e:
        err = f"ERROR: {type(e).__name__}: {e}"
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(err + "\n")
        print(err)
        sys.exit(1)

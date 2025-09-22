#!/usr/bin/env python3
"""
Train ML models (XGBoost, LightGBM) for points_scored with strict (game_id, team) outcome join.

- Loads data/modeling_dataset.csv
- Attaches outcome targets from a per-team scores file UNIQUE on (game_id, team)
- Drops rows with NaN target and logs dropped_NaN_targets=#
- Builds features: numeric + selected one-hot categoricals
- Temporal split by (season, week) if present, else index split
- Trains:
    * XGBRegressor
    * LGBMRegressor
- Writes:
    logs_ml_models.txt
    data/ml_model_metrics.csv
    data/ml_feature_importance.csv
    models/xgb_model.pkl
    models/lgbm_model.pkl   (if LightGBM installed)

Env (optional):
    TARGET_COL             (default: points_scored)
    EXCLUDE_TARGET_REGEX   (e.g., (?i)^(?!points_scored$).*(points?|scores?|margin|spread)$)
    SCORES_CSV             (default: data/game_scores_clean.csv)
    XGB_N_ESTIMATORS       (default: 600)
    XGB_MAX_DEPTH          (default: 6)
    XGB_LEARNING_RATE      (default: 0.05)
    LGBM_N_ESTIMATORS      (default: 800)
    LGBM_NUM_LEAVES        (default: 63)
    LGBM_LEARNING_RATE     (default: 0.05)
"""

from __future__ import annotations

from pathlib import Path
import os, sys, re, math
import numpy as np
import pandas as pd
import joblib

# Sklearn bits
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Tree libs
try:
    from xgboost import XGBRegressor
except Exception as e:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

# ---- Paths & env ----
DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
LOG_FILE   = Path("logs_ml_models.txt")

INPUT_CSV  = DATA_DIR / "modeling_dataset.csv"
SCORES_CSV = Path(os.environ.get("SCORES_CSV", str(DATA_DIR / "game_scores_clean.csv")))
TARGET_COL = os.environ.get("TARGET_COL", "points_scored")
EXCLUDE_TARGET_REGEX = os.environ.get("EXCLUDE_TARGET_REGEX")

# XGB params
XGB_N_ESTIMATORS  = int(os.environ.get("XGB_N_ESTIMATORS", "600"))
XGB_MAX_DEPTH     = int(os.environ.get("XGB_MAX_DEPTH", "6"))
XGB_LEARNING_RATE = float(os.environ.get("XGB_LEARNING_RATE", "0.05"))

# LGBM params
LGBM_N_ESTIMATORS  = int(os.environ.get("LGBM_N_ESTIMATORS", "800"))
LGBM_NUM_LEAVES    = int(os.environ.get("LGBM_NUM_LEAVES", "63"))
LGBM_LEARNING_RATE = float(os.environ.get("LGBM_LEARNING_RATE", "0.05"))

METRICS_CSV = DATA_DIR / "ml_model_metrics.csv"
IMP_CSV     = DATA_DIR / "ml_feature_importance.csv"


# ---- Logging ----
def log(msg: str):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


# ---- Outcome join (strict (game_id, team)) ----
def attach_outcome_targets(df_raw: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower() for c in df_raw.columns}
    if {"points_scored", "points_allowed", "margin"}.issubset(lower):
        return df_raw

    if not SCORES_CSV.exists():
        raise FileNotFoundError(f"{SCORES_CSV} not found (expected clean per-team scores).")

    s = pd.read_csv(SCORES_CSV)
    cols = {c.lower(): c for c in s.columns}
    need = {"game_id", "team", "points_scored"}
    if not need.issubset({c.lower() for c in s.columns}):
        raise KeyError(f"{SCORES_CSV} must have {need}")

    s = s.rename(columns={
        cols["game_id"]: "game_id",
        cols["team"]: "team",
        cols["points_scored"]: "points_scored",
    })
    if "points_allowed" not in s.columns:
        s["points_allowed"] = np.nan
    if "margin" not in s.columns:
        s["margin"] = s["points_scored"] - s["points_allowed"]

    # Enforce uniqueness on (game_id, team)
    s = s.sort_values(["game_id","team"]).drop_duplicates(subset=["game_id","team"], keep="first")

    for req in ("game_id","team"):
        if req not in df_raw.columns:
            raise KeyError(f"Modeling dataset missing required '{req}' for strict join.")

    merged = df_raw.merge(
        s[["game_id","team","points_scored","points_allowed","margin"]],
        on=["game_id","team"], how="left", validate="m:1"
    )

    log("outcome_join_mode=strict(game_id,team)")
    log(f"outcome_merge_missing_rows={merged['points_scored'].isna().sum()}")
    return merged


# ---- Target/feature building ----
def select_target_column(df: pd.DataFrame, target_override: str) -> str:
    if target_override in df.columns:
        return target_override
    for c in df.columns:
        if c.lower() == target_override.lower():
            return c
    # Fall back to common outcomes
    for alias in ("points_scored","points","pts","score","team_points","points_for","pf","margin"):
        for c in df.columns:
            if c.lower() == alias:
                return c
    raise ValueError(f"TARGET_COL '{target_override}' not found in columns.")

def exclude_leaky_features(X: pd.DataFrame, target_col: str, pattern: str | None):
    dropped = []
    if pattern:
        rx = re.compile(pattern)
        for col in list(X.columns):
            if col == target_col:
                continue
            if rx.search(col):
                dropped.append(col)
        X = X.drop(columns=dropped, errors="ignore")
    return X, dropped

def build_features(df: pd.DataFrame, target_col: str):
    y = pd.to_numeric(df[target_col], errors="coerce")

    # Remove obvious leakage/meta cols
    drop_cols = {target_col, "home_points","away_points","points_allowed","margin"}
    drop_cols |= set(["game_id","start_date","wx_start_date"]) & set(df.columns)

    num_df = df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=[np.number])
    cat_df = df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=["object","category","bool"])

    keep_cats = [c for c in cat_df.columns if c in ("team","opponent","conference","st_conference","is_home")]
    cat_df = cat_df[keep_cats].fillna("NA")
    if not cat_df.empty:
        cat_oh = pd.get_dummies(cat_df, prefix=cat_df.columns, drop_first=True)
        X = pd.concat([num_df, cat_oh], axis=1)
    else:
        X = num_df.copy()

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y


# ---- Split ----
def temporal_order(df: pd.DataFrame):
    df = df.assign(__row_id=np.arange(len(df)))
    if {"season","week"}.issubset(df.columns):
        return df.sort_values(["season","week","__row_id"], kind="mergesort"), "temporal(season,week)"
    return df.sort_values(["__row_id"], kind="mergesort"), "index"


# ---- Train/eval helpers ----
def eval_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2


def fit_xgb(X_train, y_train):
    if XGBRegressor is None:
        raise RuntimeError("xgboost is not available. Ensure it is installed.")
    model = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        n_jobs=0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def fit_lgbm(X_train, y_train):
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is not available. Ensure it is installed.")
    model = LGBMRegressor(
        n_estimators=LGBM_N_ESTIMATORS,
        num_leaves=LGBM_NUM_LEAVES,
        learning_rate=LGBM_LEARNING_RATE,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="regression",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def main():
    # reset log
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_CSV}")

    df_raw = pd.read_csv(INPUT_CSV)
    df_raw = attach_outcome_targets(df_raw)

    target = select_target_column(df_raw, TARGET_COL)

    # Drop NaN target rows
    before = len(df_raw)
    df_raw = df_raw.dropna(subset=[target])
    dropped = before - len(df_raw)
    log(f"dropped_NaN_targets={dropped}")

    # Build features
    X_all, y_all = build_features(df_raw, target)
    X_all, dropped_feats = exclude_leaky_features(X_all, target, EXCLUDE_TARGET_REGEX)

    # Order + split
    df_ord, split_mode = temporal_order(df_raw.loc[X_all.index])
    X_all = X_all.loc[df_ord.index]
    y_all = y_all.loc[df_ord.index]

    n_total = len(df_ord)
    cut = int(n_total * 0.8)
    idx_train = df_ord.index[:cut]
    idx_test  = df_ord.index[cut:]

    X_train, y_train = X_all.loc[idx_train], y_all.loc[idx_train]
    X_test,  y_test  = X_all.loc[idx_test],  y_all.loc[idx_test]

    metrics_rows = []
    imp_rows = []

    # XGBoost
    try:
        xgb = fit_xgb(X_train, y_train)
        pred = xgb.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, pred)
        log(f"xgboost: rmse={rmse:.6f}, mae={mae:.6f}, r2={r2:.6f}, n_train={len(X_train)}, n_test={len(X_test)}, n_features={X_train.shape[1]}, split_mode={split_mode}")
        metrics_rows.append({"model":"xgboost","rmse":rmse,"mae":mae,"r2":r2})

        # Importances
        if hasattr(xgb, "feature_importances_"):
            for feat, imp in zip(X_train.columns, xgb.feature_importances_):
                imp_rows.append({"model":"xgboost","feature":feat,"importance":float(imp)})
        joblib.dump(xgb, MODELS_DIR / "xgb_model.pkl")
    except Exception as e:
        log(f"WARNING: xgboost failed: {e}")

    # LightGBM
    try:
        lgbm = fit_lgbm(X_train, y_train)
        pred = lgbm.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, pred)
        log(f"lightgbm: rmse={rmse:.6f}, mae={mae:.6f}, r2={r2:.6f}, n_train={len(X_train)}, n_test={len(X_test)}, n_features={X_train.shape[1]}, split_mode={split_mode}")
        metrics_rows.append({"model":"lightgbm","rmse":rmse,"mae":mae,"r2":r2})

        if hasattr(lgbm, "feature_importances_"):
            for feat, imp in zip(X_train.columns, lgbm.feature_importances_):
                imp_rows.append({"model":"lightgbm","feature":feat,"importance":float(imp)})
        joblib.dump(lgbm, MODELS_DIR / "lgbm_model.pkl")
    except Exception as e:
        log(f"WARNING: lightgbm failed: {e}")

    # Persist outputs
    pd.DataFrame(metrics_rows).to_csv(METRICS_CSV, index=False)
    pd.DataFrame(imp_rows).to_csv(IMP_CSV, index=False)

    # Header
    hdr = [
        f"snapshot_utc={pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"input_rows={len(df_raw)}",
        f"features={X_train.shape[1]}",
        f"target={target}",
        f"excluded_features={len(dropped_feats)}",
        f"scores_csv={SCORES_CSV}",
    ]
    existing = LOG_FILE.read_text() if LOG_FILE.exists() else ""
    LOG_FILE.write_text("\n".join(hdr) + "\n" + existing)


if __name__ == "__main__":
    # default leak guard for outcomes
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

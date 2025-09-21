#!/usr/bin/env python3
"""
CFB Regression Runner (strict (game_id, team) outcome join)

- Loads data/modeling_dataset.csv
- Attaches outcome targets from a per-team scores file that is UNIQUE on (game_id, team)
  (matches output of clean_scores_unique_by_id.py -> data/game_scores_clean.csv)
- Builds features with numeric + safe one-hot categoricals
- Excludes leaky columns via EXCLUDE_TARGET_REGEX (keeps the actual target)
- Temporal split by (season, week)
- Trains LinearRegression, RidgeCV, LassoCV
- Writes:
    logs_model_regression.txt
    data/model_regression_metrics.csv
    data/model_regression_coeffs.csv
    models/<model>_model.pkl

Env:
    TARGET_COL            default eff_off_overall_ppa (Checklist 2.5 uses points_scored)
    EXCLUDE_TARGET_REGEX  optional leak guard
    SCORES_CSV            optional explicit path to scores (defaults to data/game_scores_clean.csv)
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

# ---- Paths & constants ----
DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
LOG_FILE   = Path("logs_model_regression.txt")
METRICS_CSV = DATA_DIR / "model_regression_metrics.csv"
COEFFS_CSV  = DATA_DIR / "model_regression_coeffs.csv"
INPUT_CSV   = DATA_DIR / "modeling_dataset.csv"

# ---- Environment ----
SCORES_CSV = os.environ.get("SCORES_CSV", str(DATA_DIR / "game_scores_clean.csv"))
TARGET_COL = os.environ.get("TARGET_COL", "eff_off_overall_ppa")
EXCLUDE_TARGET_REGEX = os.environ.get("EXCLUDE_TARGET_REGEX", None)

# ---- Logging ----
def log(msg: str):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

# ---- Helpers ----
def to_bool_home(v):
    if isinstance(v, (int, float, np.integer, np.floating)):
        if pd.isna(v): return None
        return int(v) == 1
    s = str(v).strip().lower()
    if s in {"1","true","t","yes","y","home","h"}: return True
    if s in {"0","false","f","no","n","away","a"}: return False
    return None

# ---- Outcome attach (STRICT on (game_id, team)) ----
def attach_outcome_targets(df_raw: pd.DataFrame) -> pd.DataFrame:
    lower_cols = {c.lower() for c in df_raw.columns}
    if {"points_scored","points_allowed","margin"}.issubset(lower_cols):
        return df_raw

    if not Path(SCORES_CSV).exists():
        raise FileNotFoundError(
            f"Scores file not found at {SCORES_CSV}. Ensure clean step wrote data/game_scores_clean.csv "
            "or set SCORES_CSV to a per-team file unique on (game_id, team)."
        )

    s = pd.read_csv(SCORES_CSV)

    # Map common casings to canonical names
    cmap = {c.lower(): c for c in s.columns}
    def have(*names): return all(n in cmap for n in names)

    # Expect long per-team with at least game_id, team, points_scored
    if not have("game_id", "team"):
        raise KeyError(f"{SCORES_CSV} must include 'game_id' and 'team' columns for strict join.")

    # points_scored may need to be constructed from home/away splits
    if "points_scored" in cmap:
        s = s.rename(columns={cmap["points_scored"]: "points_scored"})
    else:
        # Build from wide columns + is_home if available
        hp = cmap.get("home_points")
        ap = cmap.get("away_points")
        is_home_col = cmap.get("is_home")
        if hp and ap and is_home_col:
            ih = s[is_home_col].map(to_bool_home)
            s["points_scored"]  = pd.to_numeric(s[hp], errors="coerce").where(ih, pd.to_numeric(s[ap], errors="coerce"))
            s["points_allowed"] = pd.to_numeric(s[ap], errors="coerce").where(ih, pd.to_numeric(s[hp], errors="coerce"))
        else:
            raise KeyError(
                f"{SCORES_CSV} lacks 'points_scored' and cannot compute it without "
                "'home_points','away_points','is_home'."
            )

    # Ensure points_allowed exists; compute if possible
    if "points_allowed" not in s.columns:
        hp = cmap.get("home_points")
        ap = cmap.get("away_points")
        is_home_col = cmap.get("is_home")
        if hp and ap and is_home_col:
            ih = s[is_home_col].map(to_bool_home)
            s["points_allowed"] = pd.to_numeric(s[ap], errors="coerce").where(ih, pd.to_numeric(s[hp], errors="coerce"))
        else:
            s["points_allowed"] = np.nan

    # Carry optional columns for logging/ordering
    for want in ("season","week"):
        if want in cmap and want not in s.columns:
            s[want] = s[cmap[want]]

    # Canonicalize keys
    s = s.rename(columns={cmap["game_id"]: "game_id", cmap["team"]: "team"})
    s["team"] = s["team"].astype(str).str.strip()
    s["game_id"] = s["game_id"]

    # Uniqueness on (game_id, team)
    s = s.sort_values(["game_id", "team"]).drop_duplicates(subset=["game_id","team"], keep="first")
    dup_ct = s.duplicated(subset=["game_id","team"]).sum()
    if dup_ct:
        raise ValueError(f"Scores file still has duplicates on (game_id, team): {dup_ct} rows.")

    # Modeling dataset must have keys
    for req in ("game_id","team"):
        if req not in df_raw.columns:
            raise KeyError(f"Modeling dataset is missing required column '{req}' for strict join.")

    # Merge STRICT m:1
    cols_keep = ["game_id","team","points_scored","points_allowed"]
    if "margin" in s.columns:
        cols_keep.append("margin")
    merged = df_raw.merge(s[cols_keep], on=["game_id","team"], how="left", validate="m:1")
    join_mode = "strict(game_id,team)"

    # Compute margin if missing
    if "margin" not in merged.columns or merged["margin"].isna().any():
        merged["margin"] = merged["points_scored"] - merged["points_allowed"]

    log(f"outcome_join_mode={join_mode}")
    log(f"outcome_merge_missing_rows={merged['points_scored'].isna().sum()}")

    # Persist enriched dataset for inspection
    try:
        INPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(INPUT_CSV, index=False)
    except Exception as e:
        log(f"WARNING: failed to persist enriched dataset: {e}")

    return merged

# ---- Feature building / exclusions ----
def select_target_column(df: pd.DataFrame, target_override: str) -> str:
    if target_override in df.columns:
        return target_override
    for c in df.columns:
        if c.lower() == target_override.lower():
            return c
    for alias in ("points_scored","points","pts","score","team_points","points_for","pf","margin"):
        for c in df.columns:
            if c.lower() == alias:
                return c
    raise ValueError(f"TARGET_COL '{target_override}' not found. First 80 cols: {list(df.columns)[:80]}")

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

    drop_cols = {target_col, "home_points","away_points","points_allowed","margin"}
    meta_like = {"game_id","start_date","wx_start_date"}
    drop_cols |= (set(df.columns) & meta_like)

    num_df = df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=[np.number])
    cat_df = df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=["object","category","bool"])

    keep_cats = [c for c in cat_df.columns if c in ("team","opponent","conference","st_conference")]
    cat_df = cat_df[keep_cats].fillna("NA")
    if not cat_df.empty:
        cat_oh = pd.get_dummies(cat_df, prefix=[c for c in cat_df.columns], drop_first=True)
        X = pd.concat([num_df, cat_oh], axis=1)
    else:
        X = num_df.copy()

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y

# ---- Temporal split ----
def temporal_train_test_split(df: pd.DataFrame, test_frac: float = 0.2):
    if not {"season","week"}.issubset(df.columns):
        n = len(df)
        cut = int(n * (1 - test_frac))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy(), "index"
    sdf = df.sort_values(["season","week"], kind="mergesort").reset_index(drop=True)
    n = len(sdf)
    cut = int(n * (1 - test_frac))
    return sdf.iloc[:cut].copy(), sdf.iloc[cut:].copy(), "temporal(season,week)"

# ---- Fit/eval ----
def fit_and_eval(X_train, y_train, X_test, y_test, model_name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)
    return rmse, mae, r2, pred, model

def main():
    # reset logs
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_CSV}")
    df_raw = pd.read_csv(INPUT_CSV)

    # Attach outcome targets (strict)
    df_raw = attach_outcome_targets(df_raw)

    # Target
    target = select_target_column(df_raw, TARGET_COL)

    # Features
    X_all, y_all = build_features(df_raw, target)

    # Leak guard
    X_all, dropped = exclude_leaky_features(X_all, target, EXCLUDE_TARGET_REGEX)

    # Align/sort temporally
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

    n_total = len(df_all_sorted)
    cut = int(n_total * 0.8)
    train_idx = df_all_sorted.index[:cut]
    test_idx  = df_all_sorted.index[cut:]

    X_train, y_train = X_all.loc[train_idx], y_all.loc[train_idx]
    X_test,  y_test  = X_all.loc[test_idx],  y_all.loc[test_idx]

    # Pipelines
    ridge_alphas = np.logspace(-3, 3, 13)
    lasso_alphas = np.logspace(-3, 1, 9)
    models = {
        "linear": Pipeline([("scaler", StandardScaler(with_mean=False)), ("lin", LinearRegression())]),
        "ridge":  Pipeline([("scaler", StandardScaler(with_mean=False)), ("rid", RidgeCV(alphas=ridge_alphas))]),  # removed store_cv_values
        "lasso":  Pipeline([("scaler", StandardScaler(with_mean=False)), ("las", LassoCV(alphas=lasso_alphas, max_iter=5000, cv=5, n_jobs=None))]),
    }

    # Fit/Eval
    metrics_rows = []
    coeff_rows = []

    for name, pipe in models.items():
        rmse, mae, r2, pred, fitted = fit_and_eval(X_train, y_train, X_test, y_test, name, pipe)
        log(f"{name}: rmse={rmse:.6f}, mae={mae:.6f}, r2={r2:.6f}, "
            f"n_train={len(X_train)}, n_test={len(X_test)}, n_features={X_train.shape[1]}, split_mode={split_mode}")
        metrics_rows.append({
            "model": name, "rmse": rmse, "mae": mae, "r2": r2,
            "n_train": len(X_train), "n_test": len(X_test),
            "n_features": X_train.shape[1], "split_mode": split_mode
        })

        # Save model
        out_path = MODELS_DIR / f"{name}_model.pkl"
        joblib.dump(fitted, out_path)

        # Coefficients (best-effort)
        try:
            if name == "linear":
                est = fitted.named_steps["lin"]
            elif name == "ridge":
                est = fitted.named_steps["rid"]
            else:
                est = fitted.named_steps["las"]
            feature_names = list(X_train.columns)
            coefs = np.ravel(est.coef_) if hasattr(est, "coef_") else np.zeros(len(feature_names))
            for feat, coef in zip(feature_names, coefs):
                coeff_rows.append({"model": name, "feature": feat, "coefficient": float(coef)})
            if hasattr(est, "intercept_"):
                coeff_rows.append({"model": name, "feature": "__intercept__", "coefficient": float(np.atleast_1d(est.intercept_)[0])})
        except Exception as e:
            log(f"WARNING: failed to extract coefficients for {name}: {e}")

    # Write metrics and coeffs
    pd.DataFrame(metrics_rows).to_csv(METRICS_CSV, index=False)
    pd.DataFrame(coeff_rows).to_csv(COEFFS_CSV, index=False)

    # Header summary
    excluded_count = len(dropped)
    excluded_list = ", ".join(sorted(dropped)) if excluded_count else ""
    hdr = [
        f"snapshot_utc={pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"input_rows={len(df_raw)}",
        f"features={X_train.shape[1]}",
        f"target={target}",
        f"excluded_features={excluded_count}" + (f" -> {excluded_list}" if excluded_count else ""),
        f"scores_csv={SCORES_CSV}",
    ]
    existing = LOG_FILE.read_text() if LOG_FILE.exists() else ""
    LOG_FILE.write_text("\n".join(hdr) + "\n" + existing)

if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Default leak-guard for outcome targets
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

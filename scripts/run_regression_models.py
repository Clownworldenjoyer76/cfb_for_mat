#!/usr/bin/env python3
"""
CFB Regression Runner (historical-friendly, numeric-only features)

- Loads data/modeling_dataset.csv
- Attaches outcome targets (points_scored, points_allowed, margin) from scores
- (Strict) joins on (game_id, team) if available; logs missing rows count
- Drops rows with NaN in the selected TARGET_COL (logs dropped_NaN_targets)
- Builds FEATURES as **numeric-only** columns after removing IDs/meta + leaks
- Temporal split by (season, week)
- Trains LinearRegression, RidgeCV, LassoCV
- Writes:
    logs_model_regression.txt
    data/model_regression_metrics.csv
    data/model_regression_coeffs.csv
    models/<model>_model.pkl

Environment variables:
    TARGET_COL            (default: eff_off_overall_ppa; set to points_scored in CI)
    EXCLUDE_TARGET_REGEX  (regex to drop leaky columns; keeps the actual target)
    SCORES_GLOB / SCORES_CSV  (optional paths to locate scores file)
"""

from pathlib import Path
import os, sys, re, glob, math
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ---- Paths & constants ----
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
LOG_FILE = Path("logs_model_regression.txt")
METRICS_CSV = DATA_DIR / "model_regression_metrics.csv"
COEFFS_CSV = DATA_DIR / "model_regression_coeffs.csv"
INPUT_CSV = DATA_DIR / "modeling_dataset.csv"

# ---- Environment ----
SCORES_GLOB = os.environ.get("SCORES_GLOB")
SCORES_CSV = os.environ.get("SCORES_CSV")  # explicit scores path preferred by workflow
TARGET_COL = os.environ.get("TARGET_COL", "eff_off_overall_ppa")
EXCLUDE_TARGET_REGEX = os.environ.get("EXCLUDE_TARGET_REGEX")

# ---- Logging helpers ----
def _log_start():
    if LOG_FILE.exists():
        LOG_FILE.unlink()
def log(msg: str):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{msg}\n")
    print(msg)

# ---- Scores discovery & normalization ----
def _find_scores_file():
    if SCORES_CSV and Path(SCORES_CSV).exists():
        return SCORES_CSV
    if SCORES_GLOB:
        cands = sorted(glob.glob(SCORES_GLOB))
    else:
        roots = ["data", "data/raw", "docs/data", "docs/data/final"]
        pats  = ["*game_scores_clean*.csv", "*scores_clean*.csv", "*games*.csv", "*scores*.csv"]
        cands = []
        for r in roots:
            for p in pats:
                cands.extend(sorted(glob.glob(str(Path(r) / p))))
    for p in cands:
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        cols = {c.lower() for c in df.columns}
        if {"game_id","team"}.issubset(cols) and (
            {"points_scored"}.issubset(cols) or {"home_points","away_points"}.issubset(cols)
        ):
            return p
    return None

def _normalize_scores(df_scores: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df_scores.columns}
    def col(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    # Prefer already-expanded per-(game_id,team)
    game_id = col("game_id")
    team    = col("team")
    ps      = col("points_scored")

    if game_id and team and ps:
        out = df_scores[[game_id, team, ps]].copy()
        out.columns = ["game_id","team","points_scored"]
        return out

    # Otherwise build per-team rows from home/away
    home_team = col("home_team","team_home","home")
    away_team = col("away_team","team_away","away")
    hp        = col("home_points","home_score","points_home","score_home")
    ap        = col("away_points","away_score","points_away","score_away")
    if not all([game_id, home_team, away_team, hp, ap]):
        raise ValueError("Scores file lacks required columns to derive points_scored.")
    a = df_scores[[game_id, home_team, hp]].copy()
    a.columns = ["game_id","team","points_scored"]
    b = df_scores[[game_id, away_team, ap]].copy()
    b.columns = ["game_id","team","points_scored"]
    out = pd.concat([a,b], ignore_index=True)
    return out

def _attach_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    cl = {c.lower() for c in df.columns}
    if "points_scored" in cl:
        return df

    scores_path = _find_scores_file()
    if not scores_path:
        raise FileNotFoundError("Could not locate scores file (expects per-(game_id,team) with points_scored).")
    s_raw = pd.read_csv(scores_path)
    s = _normalize_scores(s_raw)

    # strict join on (game_id, team)
    merged = df.merge(
        s, on=["game_id","team"], how="left", validate="m:1"
    )
    # report missing
    miss = int(merged["points_scored"].isna().sum())
    log(f"outcome_join_mode=strict(game_id,team)")
    log(f"outcome_merge_missing_rows={miss}")

    return merged

# ---- Feature building / exclusions ----
META_DROP = {
    "game_id","season","week","team","opponent","start_date","wx_start_date",
    "wx_source_endpoint","eff_source_meta","ts_season_type","eff_season_type",
    "st_conference","ts_conference","eff_conference"
}

def _select_target(df: pd.DataFrame, target_name: str) -> str:
    if target_name in df.columns:
        return target_name
    for c in df.columns:
        if c.lower() == target_name.lower():
            return c
    raise ValueError(f"TARGET_COL '{target_name}' not found in dataset.")

def _exclude_leaks(df: pd.DataFrame, target_col: str, pattern: str | None) -> pd.DataFrame:
    if not pattern:
        return df
    rx = re.compile(pattern)
    keep = []
    dropped = []
    for c in df.columns:
        if c == target_col:
            keep.append(c)
            continue
        if rx.search(c):
            dropped.append(c)
        else:
            keep.append(c)
    if dropped:
        log(f"excluded_by_regex_count={len(dropped)}")
        log("excluded_by_regex_sample=" + ",".join(sorted(dropped)[:30]))
    return df[keep]

def _build_numeric_features(df_all: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    # Drop obvious non-features first (meta IDs etc.)
    base = df_all.drop(columns=[c for c in df_all.columns if c in META_DROP], errors="ignore")

    # Drop the target itself and any derived targets if present
    base = base.drop(columns=[target_col, "points_allowed", "margin", "home_points", "away_points"], errors="ignore")

    # Apply regex-based exclusions (leak guard)
    base = _exclude_leaks(base, target_col, EXCLUDE_TARGET_REGEX)

    # Keep numeric-only columns
    X = base.select_dtypes(include=["number", "float", "int"]).copy()

    # Final NA handling
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # y as numeric
    y = pd.to_numeric(df_all[target_col], errors="coerce")

    # Logging about features
    log(f"feature_candidates_after_drops={base.shape[1]}")
    log(f"numeric_features_used={X.shape[1]}")
    if X.shape[1] > 0:
        log("numeric_features_sample=" + ",".join(list(X.columns)[:30]))
    else:
        log("numeric_features_sample=")

    return X, y

# ---- Split ----
def _temporal_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, test_frac: float = 0.2):
    if {"season","week"}.issubset(df.columns):
        order = df[["season","week"]].apply(pd.to_numeric, errors="coerce")
        idx = np.lexsort((order["week"].values, order["season"].values))
        ordered_index = df.index.values[idx]
    else:
        ordered_index = df.index.values

    n = len(ordered_index)
    cut = int(n * (1 - test_frac))
    train_idx = ordered_index[:cut]
    test_idx = ordered_index[cut:]
    return X.loc[train_idx], y.loc[train_idx], X.loc[test_idx], y.loc[test_idx], ("temporal(season,week)" if {"season","week"}.issubset(df.columns) else "index")

# ---- Fit/eval ----
def _fit_eval(X_train, y_train, X_test, y_test, model_name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)
    return rmse, mae, r2, pred, model

def main():
    _log_start()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_CSV}")

    df_raw = pd.read_csv(INPUT_CSV)

    # Attach outcomes
    df = _attach_outcomes(df_raw)

    # Select target and drop rows with NaN target
    target = _select_target(df, TARGET_COL)
    dropped_before = int(df[target].isna().sum())
    if dropped_before:
        log(f"pre_drop_NaN_targets={dropped_before}")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    log(f"dropped_NaN_targets={dropped_before}")

    # Build numeric-only features
    X_all, y_all = _build_numeric_features(df, target)

    # Guard: must have some features
    if X_all.shape[1] == 0:
        raise ValueError("No numeric features remain after exclusions. Adjust EXCLUDE_TARGET_REGEX or dataset.")

    # Guard: must have some rows
    if X_all.shape[0] == 0:
        raise ValueError("No rows left after target drop; check outcome merge and TARGET_COL.")

    # Split
    X_train, y_train, X_test, y_test, split_mode = _temporal_split(df, X_all, y_all)

    # Pipelines
    ridge_alphas = np.logspace(-3, 3, 13)
    lasso_alphas = np.logspace(-3, 1, 9)

    models = {
        "linear": Pipeline([("scaler", StandardScaler(with_mean=False)), ("lin", LinearRegression())]),
        "ridge":  Pipeline([("scaler", StandardScaler(with_mean=False)), ("rid", RidgeCV(alphas=ridge_alphas))]),
        "lasso":  Pipeline([("scaler", StandardScaler(with_mean=False)), ("las", LassoCV(alphas=lasso_alphas, max_iter=5000, cv=5))]),
    }

    # Fit/Eval
    metrics_rows = []
    coeff_rows = []

    for name, pipe in models.items():
        rmse, mae, r2, pred, fitted = _fit_eval(X_train, y_train, X_test, y_test, name, pipe)
        log(f"{name}: rmse={rmse:.6f}, mae={mae:.6f}, r2={r2:.6f}, n_train={len(X_train)}, n_test={len(X_test)}, n_features={X_train.shape[1]}, split_mode={split_mode}")
        metrics_rows.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2, "n_train": len(X_train), "n_test": len(X_test), "n_features": X_train.shape[1], "split_mode": split_mode})

        # Save model
        out_path = MODELS_DIR / f"{name}_model.pkl"
        joblib.dump(fitted, out_path)

        # Coefficients (linear models only)
        try:
            if name == "linear":
                est = fitted.named_steps["lin"]
            elif name == "ridge":
                est = fitted.named_steps["rid"]
            else:
                est = fitted.named_steps["las"]
            feature_names = list(X_train.columns)
            coefs = np.ravel(getattr(est, "coef_", np.zeros(len(feature_names))))
            for feat, coef in zip(feature_names, coefs):
                coeff_rows.append({"model": name, "feature": feat, "coefficient": float(coef)})
            if hasattr(est, "intercept_"):
                coeff_rows.append({"model": name, "feature": "__intercept__", "coefficient": float(np.atleast_1d(est.intercept_)[0])})
        except Exception as e:
            log(f"WARNING: failed to extract coefficients for {name}: {e}")

    pd.DataFrame(metrics_rows).to_csv(METRICS_CSV, index=False)
    pd.DataFrame(coeff_rows).to_csv(COEFFS_CSV, index=False)

    # Summary header
    hdr = [
        f"snapshot_utc={pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"input_rows={len(df)}",
        f"features={X_all.shape[1]}",
        f"target={target}",
        f"scores_csv={_find_scores_file() or 'unknown'}",
    ]
    existing = LOG_FILE.read_text() if LOG_FILE.exists() else ""
    LOG_FILE.write_text("\n".join(hdr) + "\n" + existing)

if __name__ == "__main__":
    # Default leak-guard if not provided and target is an outcome-ish field
    if EXCLUDE_TARGET_REGEX is None and TARGET_COL.lower() in ("points_scored","points","pts","score","margin"):
        EXCLUDE_TARGET_REGEX = r"(?i)^(?!{}$).*(points?|scores?|margin|spread)$".format(re.escape(TARGET_COL))
    try:
        main()
    except Exception as e:
        err = f"ERROR: {type(e).__name__}: {e}"
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(err + "\n")
        print(err)
        sys.exit(1)

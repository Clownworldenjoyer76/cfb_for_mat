#!/usr/bin/env python3
"""
CFB Regression Runner (complete)

- Loads data/modeling_dataset.csv
- Attaches outcome targets (points_scored, points_allowed, margin) from a scores file
- Builds features with numeric + safe one-hot categoricals
- Excludes leaky columns via EXCLUDE_TARGET_REGEX (keeps the actual target)
- Temporal split by (season, week)
- Trains LinearRegression, RidgeCV, LassoCV
- Writes:
    logs_model_regression.txt
    data/model_regression_metrics.csv
    data/model_regression_coeffs.csv
    models/<model>_model.pkl

Env vars honored:
    TARGET_COL            (default: eff_off_overall_ppa; set to points_scored per Checklist 2.5)
    EXCLUDE_TARGET_REGEX  (default auto-leak-guard when target is points/score/margin)
    SCORES_CSV            (optional) explicit path to scores file (preferred)
    SCORES_GLOB           (optional) glob to locate scores file, e.g., 'data/raw/*games*.csv'
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
SCRIPT_DIR = Path.cwd()
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
LOG_FILE = Path("logs_model_regression.txt")
METRICS_CSV = Path("data/model_regression_metrics.csv")
COEFFS_CSV = Path("data/model_regression_coeffs.csv")
INPUT_CSV = Path("data/modeling_dataset.csv")  # exact per your repo

# ---- Environment ----
SCORES_CSV = os.environ.get("SCORES_CSV")       # <— NEW: explicit override supported
SCORES_GLOB = os.environ.get("SCORES_GLOB")     # legacy/optional discovery
TARGET_COL = os.environ.get("TARGET_COL", "eff_off_overall_ppa")
EXCLUDE_TARGET_REGEX = os.environ.get("EXCLUDE_TARGET_REGEX", None)

# ---- Logging helpers ----
def log(msg: str):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{msg}\n")
    print(msg)

def reset_logs():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

# ---- Scores discovery & normalization ----
def find_scores_file():
    # Prefer explicit SCORES_CSV if provided and exists
    if SCORES_CSV and Path(SCORES_CSV).exists():
        return SCORES_CSV

    if SCORES_GLOB:
        cands = sorted(glob.glob(SCORES_GLOB))
    else:
        # Broad but deterministic search
        cands = []
        roots = ["data/raw", "data", "docs/data", "docs/data/final"]
        pats  = ["*games*.csv", "*scores*.csv", "*results*.csv", "*team_points*.csv"]
        for r in roots:
            for p in pats:
                cands.extend(sorted(glob.glob(str(Path(r) / p))))
    # De-duplicate preserving order
    seen, ordered = set(), []
    for c in cands:
        if c not in seen and Path(c).exists():
            ordered.append(c); seen.add(c)
    # Return the first readable candidate that has enough columns
    for path in ordered:
        try:
            df = pd.read_csv(path, nrows=5)
        except Exception:
            continue
        cols = {c.lower() for c in df.columns}
        if {"game_id","team","points_scored"}.issubset(cols):
            return path  # already long, per-team
        has_teams   = {"home_team","away_team"}.issubset(cols) or {"team_home","team_away"}.issubset(cols)
        has_points  = ({"home_points","away_points"}.issubset(cols)
                       or {"home_score","away_score"}.issubset(cols)
                       or {"points_home","points_away"}.issubset(cols)
                       or {"score_home","score_away"}.issubset(cols))
        if has_points and has_teams:
            return path  # wide home/away
    return None


def normalize_points_cols(df_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Return a standardized wide table with columns:
      [game_id?, season?, week?, home_team, away_team, home_points, away_points]
    """
    cols = {c.lower(): c for c in df_scores.columns}
    def col(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    game_id   = col("game_id")
    season    = col("season")
    week      = col("week")
    home_team = col("home_team","team_home","home")
    away_team = col("away_team","team_away","away")
    hp        = col("home_points","home_score","points_home","score_home")
    ap        = col("away_points","away_score","points_away","score_away")

    if any(x is None for x in [home_team, away_team, hp, ap]):
        raise ValueError("Scores file missing required team/points columns. Need home/away team and points/score.")

    out = pd.DataFrame({
        "home_team": df_scores[home_team].astype(str),
        "away_team": df_scores[away_team].astype(str),
        "home_points": pd.to_numeric(df_scores[hp], errors="coerce"),
        "away_points": pd.to_numeric(df_scores[ap], errors="coerce"),
    })
    if game_id is not None: out["game_id"] = df_scores[game_id]
    if season is not None:  out["season"]  = df_scores[season]
    if week is not None:    out["week"]    = df_scores[week]
    return out


def to_bool_home(v):
    if isinstance(v, (int, float)):
        if np.isnan(v): return None
        return int(v) == 1
    s = str(v).strip().lower()
    if s in ("1","true","t","yes","y","home","h"): return True
    if s in ("0","false","f","no","n","away","a"): return False
    return None


def attach_outcome_targets(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich df_raw with points_scored / points_allowed / margin.

    Supports TWO score-file shapes:

    1) LONG per-team (preferred for our cleaner):
         columns include: game_id, team, points_scored (and usually points_allowed, is_home, etc.)
       -> merge on (game_id, team) with validate='m:1'

    2) WIDE per-game (home/away columns):
         columns: home_team, away_team, home_points, away_points, [game_id? season? week?]
       -> if df_raw has game_id, merge on game_id; otherwise fallback to season/week + team/opponent + is_home
    """
    # Already present?
    cl = {c.lower() for c in df_raw.columns}
    if {"points_scored","points_allowed","margin"}.issubset(cl):
        return df_raw

    scores_path = find_scores_file()
    if not scores_path:
        raise FileNotFoundError(
            "Could not locate a scores file. Set SCORES_CSV or SCORES_GLOB."
        )

    df_scores = pd.read_csv(scores_path)
    scores_cols = {c.lower() for c in df_scores.columns}

    # --- CASE 1: LONG per-team (game_id, team, points_scored) ---
    if {"game_id","team","points_scored"}.issubset(scores_cols):
        # Enforce types and uniqueness
        # (use case-insensitive mapping to actual column names)
        colmap = {c.lower(): c for c in df_scores.columns}
        s = df_scores.rename(columns={
            colmap["game_id"]: "game_id",
            colmap["team"]: "team",
            colmap["points_scored"]: "points_scored",
        })

        if "points_allowed" in scores_cols:
            s = s.rename(columns={colmap["points_allowed"]: "points_allowed"})
        else:
            # If only points_scored present, try to compute from margin if available later; else set NA
            s["points_allowed"] = np.nan

        if "margin" in scores_cols:
            s = s.rename(columns={colmap["margin"]: "margin"})
        else:
            s["margin"] = s["points_scored"] - s["points_allowed"]

        # Ensure uniqueness on (game_id, team)
        s = s.sort_values(["game_id","team"]).drop_duplicates(subset=["game_id","team"], keep="first")
        dups = s.duplicated(subset=["game_id","team"]).sum()
        if dups:
            raise ValueError("Clean scores are not unique by (game_id, team) after de-duplication.")

        # Require matching keys in modeling dataset
        for req in ("game_id","team"):
            if req not in df_raw.columns:
                raise KeyError(f"Modeling dataset lacks '{req}' required to merge per-team scores.")

        merged = df_raw.merge(
            s[["game_id","team","points_scored","points_allowed","margin"]],
            on=["game_id","team"],
            how="left",
            validate="m:1"
        )
        join_mode = "per-team(game_id,team)"

        # If points_allowed was NA in scores, but home/away exists in df_raw, try to backfill
        if merged["points_allowed"].isna().any():
            if {"home_points","away_points","is_home"}.issubset({c.lower() for c in merged.columns}):
                ih = merged["is_home"].map(to_bool_home)
                pa = merged["away_points"].where(ih, merged["home_points"])
                merged["points_allowed"] = merged["points_allowed"].fillna(pa)
            # recompute margin if still NA
        merged["margin"] = merged["points_scored"] - merged["points_allowed"]

    else:
        # --- CASE 2: WIDE per-game ---
        s_wide = normalize_points_cols(df_scores)

        # Try merge by game_id first
        if "game_id" in df_raw.columns and "game_id" in s_wide.columns:
            merged = df_raw.merge(s_wide, on="game_id", how="left", validate="m:1")
            join_mode = "per-game(game_id)"
        else:
            # Fallback by season/week + home/away teams derived from is_home
            for req in ("season","week","team","opponent","is_home"):
                if req not in df_raw.columns:
                    raise KeyError(f"Expected column '{req}' in modeling_dataset for outcome join (fallback mode).")
            is_home_bool = df_raw["is_home"].map(to_bool_home)
            if is_home_bool.isna().any():
                raise ValueError("Could not interpret some 'is_home' values; expected 1/0, True/False, or H/A text.")

            dfj = df_raw.copy()
            dfj["_home_side"] = is_home_bool
            dfj["join_home_team"] = np.where(dfj["_home_side"], dfj["team"], dfj["opponent"]).astype(str)
            dfj["join_away_team"] = np.where(dfj["_home_side"], dfj["opponent"], dfj["team"]).astype(str)

            if "season" not in s_wide.columns or "week" not in s_wide.columns:
                raise KeyError("Scores file lacked season/week; cannot fallback-join without game_id.")
            s2 = s_wide.rename(columns={"home_team":"join_home_team","away_team":"join_away_team"})
            key_cols = ["season","week","join_home_team","join_away_team"]
            merged = dfj.merge(s2, on=key_cols, how="left", validate="m:1")
            join_mode = "per-game(season,week,home/away teams)"

        # Compute per-team outcomes using is_home
        is_home_bool2 = merged["is_home"].map(to_bool_home)
        merged["points_scored"]  = merged["home_points"].where(is_home_bool2, merged["away_points"])
        merged["points_allowed"] = merged["away_points"].where(is_home_bool2, merged["home_points"])
        merged["margin"] = merged["points_scored"] - merged["points_allowed"]

    # Log info
    log(f"outcome_join_mode={join_mode}")
    miss = merged["points_scored"].isna().sum()
    log(f"outcome_merge_missing_rows={miss}")

    # Persist enriched dataset in place for future runs
    try:
        INPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(INPUT_CSV, index=False)
    except Exception as e:
        log(f"WARNING: failed to persist enriched dataset: {e}")

    return merged

# ---- Feature building / exclusions ----
def select_target_column(df: pd.DataFrame, target_override: str) -> str:
    # Exact or case-insensitive match
    if target_override in df.columns:
        return target_override
    for c in df.columns:
        if c.lower() == target_override.lower():
            return c
    # Try common aliases (mainly to smooth outcome adoption)
    common = ["points_scored","points","pts","score","team_points","points_for","pf","home_points","away_points","margin"]
    for cand in common:
        if cand in df.columns: return cand
        for c in df.columns:
            if c.lower() == cand.lower(): return c
    raise ValueError(f"TARGET_COL '{target_override}' not found. Available columns (first 80): {list(df.columns)[:80]}")

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
    # Identify target y
    y = pd.to_numeric(df[target_col], errors="coerce")

    # Drop obvious non-features and target/derived
    drop_cols = {target_col, "home_points","away_points","points_allowed","margin"}
    meta_like = {"game_id","start_date","wx_start_date"}
    drop_cols |= {c for c in df.columns if c in meta_like}

    # Separate numeric and categorical
    num_df = df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=[np.number])
    cat_df = df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=["object","category","bool"])

    # Keep a limited set of categoricals
    keep_cats = [c for c in cat_df.columns if c in ("team","opponent","conference","st_conference")]
    cat_df = cat_df[keep_cats].fillna("NA")
    if not cat_df.empty:
        cat_oh = pd.get_dummies(cat_df, prefix=[c for c in cat_df.columns], drop_first=True)
        X = pd.concat([num_df, cat_oh], axis=1)
    else:
        X = num_df.copy()

    # Final NA handling
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y

# ---- Train/test split ----
def temporal_train_test_split(df: pd.DataFrame, test_frac: float = 0.2):
    if not {"season","week"}.issubset(df.columns):
        n = len(df)
        cut = int(n * (1 - test_frac))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy(), "index"
    sdf = df.sort_values(["season","week"], kind="mergesort").reset_index(drop_df=True)
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
    # Reset logs
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # Load base dataset
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_CSV}")
    df_raw = pd.read_csv(INPUT_CSV)

    # Attach outcome targets if missing
    df_raw = attach_outcome_targets(df_raw)

    # Select target
    target = select_target_column(df_raw, TARGET_COL)

    # Build features
    X_all, y_all = build_features(df_raw, target)

    # Exclude leaky features
    X_all, dropped = exclude_leaky_features(X_all, target, EXCLUDE_TARGET_REGEX)

    # Ensure alignment and sort temporally
    df_all = df_raw.loc[X_all.index]  # align
    df_all = df_all.assign(__row_id=np.arange(len(df_all)))
    df_all_sorted = df_all.sort_values(["season","week","__row_id"], kind="mergesort")
    X_all = X_all.loc[df_all_sorted.index]
    y_all = y_all.loc[df_all_sorted.index]

    n_total = len(df_all_sorted)
    cut = int(n_total * 0.8)
    train_idx = df_all_sorted.index[:cut]
    test_idx  = df_all_sorted.index[cut:]

    X_train, y_train = X_all.loc[train_idx], y_all.loc[train_idx]
    X_test,  y_test  = X_all.loc[test_idx],  y_all.loc[test_idx]
    split_mode = "temporal(season,week)" if {"season","week"}.issubset(df_raw.columns) else "index"

    # Pipelines
    ridge_alphas = np.logspace(-3, 3, 13)
    lasso_alphas = np.logspace(-3, 1, 9)

    models = {
        "linear": Pipeline([("scaler", StandardScaler(with_mean=False)), ("lin", LinearRegression())]),
        "ridge":  Pipeline([("scaler", StandardScaler(with_mean=False)), ("rid", RidgeCV(alphas=ridge_alphas, store_cv_values=False))]),
        "lasso":  Pipeline([("scaler", StandardScaler(with_mean=False)), ("las", LassoCV(alphas=lasso_alphas, max_iter=5000, cv=5, n_jobs=None))]),
    }

    # Fit/Eval
    metrics_rows = []
    coeff_rows = []

    for name, pipe in models.items():
        rmse, mae, r2, pred, fitted = fit_and_eval(X_train, y_train, X_test, y_test, name, pipe)
        # Logging
        log(f"{name}: rmse={rmse:.6f}, mae={mae:.6f}, r2={r2:.6f}, n_train={len(X_train)}, n_test={len(X_test)}, n_features={X_train.shape[1]}, split_mode={split_mode}")
        metrics_rows.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2, "n_train": len(X_train), "n_test": len(X_test), "n_features": X_train.shape[1], "split_mode": split_mode})

        # Save model
        out_path = MODELS_DIR / f"{name}_model.pkl"}
        joblib.dump(fitted, out_path)

        # Extract coefficients
        try:
            if name == "linear":
                est = fitted.named_steps["lin"]
            elif name == "ridge":
                est = fitted.named_steps["rid"]
            else:
                est = fitted.named_steps["las"]
            feature_names = list(X_train.columns)
            coefs = est.coef_.ravel() if hasattr(est.coef_, "ravel") else np.array(est.coef_)
            for feat, coef in zip(feature_names, coefs):
                coeff_rows.append({"model": name, "feature": feat, "coefficient": float(coef)})
            if hasattr(est, "intercept_"):
                coeff_rows.append({"model": name, "feature": "__intercept__", "coefficient": float(np.atleast_1d(est.intercept_)[0])})
        except Exception as e:
            log(f"WARNING: failed to extract coefficients for {name}: {e}")

    # Write metrics and coeffs
    pd.DataFrame(metrics_rows).to_csv(METRICS_CSV, index=False)
    pd.DataFrame(coeff_rows).to_csv(COEFFS_CSV, index=False)

    # Prepend summary header to logs (matches your style)
    excluded_count = len(dropped)
    excluded_list = ", ".join(sorted(dropped)) if excluded_count else ""
    hdr = [
        f"snapshot_utc={pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"input_rows={len(df_raw)}",
        f"features={X_train.shape[1]}",
        f"target={target}",
        f"excluded_features={excluded_count}" + (f" -> {excluded_list}" if excluded_count else ""),
    ]
    existing = LOG_FILE.read_text() if LOG_FILE.exists() else ""
    LOG_FILE.write_text("\n".join(hdr) + "\n" + existing)

if __name__ == "__main__":
    # Ensure dirs exist
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # Default leak-guard if not provided and target is an outcome
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

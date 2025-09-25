#!/usr/bin/env python3
"""
CFB Regression Runner (prefers in-dataset target; falls back to scores join)

What it does
------------
- Loads data/modeling_dataset.csv
- If TARGET_COL (e.g., points_scored) already exists in the dataset, use it directly
  (outcome_join_mode=direct(modeling_dataset)).
- Otherwise, tries to locate a scores CSV and strict-join on (game_id, team_key)
  after name normalization and alias application.
- Drops rows with NaN in TARGET_COL; logs dropped_NaN_targets and other counters.
- Builds numeric-only features (drops IDs/meta + leak columns via EXCLUDE_TARGET_REGEX).
- Temporal split (season, week).
- Trains Linear, RidgeCV, LassoCV.
- Writes:
    logs_model_regression.txt
    data/model_regression_metrics.csv
    data/model_regression_coeffs.csv
    models/*.pkl
"""

from __future__ import annotations

from pathlib import Path
import os, sys, re, glob, math, unicodedata
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ---- Paths & env ----
DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
LOG_FILE   = Path("logs_model_regression.txt")
METRICS_CSV = DATA_DIR / "model_regression_metrics.csv"
COEFFS_CSV  = DATA_DIR / "model_regression_coeffs.csv"
INPUT_CSV   = DATA_DIR / "modeling_dataset.csv"
ALIASES_CSV = Path("mappings/team_aliases.csv")

SCORES_CSV_ENV = os.environ.get("SCORES_CSV")      # optional explicit path
SCORES_GLOB    = os.environ.get("SCORES_GLOB")     # optional glob
TARGET_COL     = os.environ.get("TARGET_COL", "eff_off_overall_ppa")
EXCLUDE_TARGET_REGEX = os.environ.get("EXCLUDE_TARGET_REGEX")

# ---- logging ----
def _reset_log():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

def log(msg: str):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")
    print(msg)

# ---- name normalization / aliases ----
_norm_rx = re.compile(r"[^A-Z0-9]")

def std_name(obj) -> str:
    if pd.isna(obj):
        return ""
    s = str(obj)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    return _norm_rx.sub("", s)

def load_alias_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    cols = {c.lower(): c for c in df.columns}
    if "cfbd_name" in cols and "alias" in cols:
        m = {}
        for _, r in df.iterrows():
            src = str(r[cols["cfbd_name"]]).strip()
            tgt = str(r[cols["alias"]]).strip()
            if src and tgt:
                m[std_name(src)] = tgt  # map by normalized source
        return m
    return {}

def apply_alias(series: pd.Series, alias_map: dict[str, str]) -> pd.Series:
    if not alias_map:
        return series
    key = series.map(std_name)
    mapped = key.map(alias_map)
    return mapped.where(mapped.notna(), series)

# ---- scores discovery and normalization (fallback only) ----
def find_scores_file() -> str | None:
    # 1) explicit
    if SCORES_CSV_ENV and Path(SCORES_CSV_ENV).exists():
        return SCORES_CSV_ENV
    # 2) glob
    cands: list[str] = []
    if SCORES_GLOB:
        cands.extend(sorted(glob.glob(SCORES_GLOB)))
    # 3) common locations
    roots = ["data", "data/raw", "docs/data", "docs/data/final"]
    pats  = ["*game_scores_clean*.csv", "*scores_clean*.csv", "*game_scores*.csv", "*games*.csv", "*scores*.csv"]
    for r in roots:
        for p in pats:
            cands.extend(sorted(glob.glob(str(Path(r) / p))))
    # de-dupe while preserving order
    seen, ordered = set(), []
    for p in cands:
        if p not in seen and Path(p).is_file():
            ordered.append(p)
            seen.add(p)
    for p in ordered:
        try:
            df = pd.read_csv(p, nrows=3)
        except Exception:
            continue
        cols = {c.lower() for c in df.columns}
        if {"game_id"}.issubset(cols) and (
            {"team", "points_scored"}.issubset(cols) or {"home_team", "away_team"}.issubset(cols)
        ):
            return p
    return None

def normalize_scores_to_team_rows(df_scores: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    cols = {c.lower(): c for c in df_scores.columns}

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    gid  = col("game_id", "id")
    team = col("team", "school", "team_name")
    ps   = col("points_scored")

    # Already per-team?
    if gid and team and ps:
        out = df_scores[[gid, team, ps]].copy()
        out.columns = ["game_id", "team", "points_scored"]
        out["team"] = apply_alias(out["team"], alias_map)
        out["team_key"] = out["team"].map(std_name)
        out = out.drop_duplicates(subset=["game_id", "team_key"], keep="first")
        return out

    # Expand from home/away points
    hteam = col("home_team", "team_home", "home")
    ateam = col("away_team", "team_away", "away")
    hp    = col("home_points", "home_score", "points_home", "score_home")
    ap    = col("away_points", "away_score", "points_away", "score_away")
    if not all([gid, hteam, ateam, hp, ap]):
        raise ValueError("Scores file missing required columns to derive per-team points.")
    a = df_scores[[gid, hteam, hp]].copy()
    a.columns = ["game_id", "team", "points_scored"]
    b = df_scores[[gid, ateam, ap]].copy()
    b.columns = ["game_id", "team", "points_scored"]
    out = pd.concat([a, b], ignore_index=True)
    out["team"] = apply_alias(out["team"], alias_map)
    out["team_key"] = out["team"].map(std_name)
    out = out.drop_duplicates(subset=["game_id", "team_key"], keep="first")
    return out

# ---- feature building ----
META_DROP = {
    "game_id", "season", "week", "team", "opponent", "start_date", "wx_start_date",
    "wx_source_endpoint", "eff_source_meta", "ts_season_type", "eff_season_type",
    "st_conference", "ts_conference", "eff_conference"
}

def select_target(df: pd.DataFrame, name: str) -> str:
    if name in df.columns:
        return name
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    raise ValueError(f"TARGET_COL '{name}' not found in dataset.")

def exclude_leaks(df: pd.DataFrame, target_col: str, pattern: str | None) -> tuple[pd.DataFrame, int, list[str]]:
    if not pattern:
        return df, 0, []
    rx = re.compile(pattern)
    keep, dropped = [], []
    for c in df.columns:
        if c == target_col:
            keep.append(c)
            continue
        if rx.search(c):
            dropped.append(c)
        else:
            keep.append(c)
    return df[keep], len(dropped), dropped[:30]

def build_numeric_features(df_all: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    base = df_all.drop(columns=[c for c in df_all.columns if c in META_DROP], errors="ignore")
    base = base.drop(columns=[target_col, "points_allowed", "margin", "home_points", "away_points"], errors="ignore")
    base, n_dropped, dropped_sample = exclude_leaks(base, target_col, EXCLUDE_TARGET_REGEX)
    if n_dropped:
        log(f"excluded_by_regex_count={n_dropped}")
        log("excluded_by_regex_sample=" + ",".join(dropped_sample))
    X = base.select_dtypes(include=["number", "float", "int"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.to_numeric(df_all[target_col], errors="coerce")
    log(f"feature_candidates_after_drops={base.shape[1]}")
    log(f"numeric_features_used={X.shape[1]}")
    log("numeric_features_sample=" + ",".join(list(X.columns)[:30]))
    return X, y

# ---- split / fit ----
def temporal_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, test_frac=0.2):
    if {"season", "week"}.issubset(df.columns):
        order = df[["season", "week"]].apply(pd.to_numeric, errors="coerce")
        idx = np.lexsort((order["week"].values, order["season"].values))
        ordered = df.index.values[idx]
    else:
        ordered = df.index.values
    n = len(ordered)
    cut = int(n * (1 - test_frac))
    return (
        X.loc[ordered[:cut]],
        y.loc[ordered[:cut]],
        X.loc[ordered[cut:]],
        y.loc[ordered[cut:]],
        ("temporal(season,week)" if {"season", "week"}.issubset(df.columns) else "index"),
    )

def fit_eval(Xt, yt, Xv, yv, name, pipe):
    pipe.fit(Xt, yt)
    pred = pipe.predict(Xv)
    rmse = math.sqrt(mean_squared_error(yv, pred))
    mae  = mean_absolute_error(yv, pred)
    r2   = r2_score(yv, pred)
    return rmse, mae, r2, pipe

# ---- main ----
def main():
    _reset_log()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_CSV}")
    dfm = pd.read_csv(INPUT_CSV)

    # alias-map and normalized keys for modeling side (for fallback join)
    alias_map = load_alias_map(ALIASES_CSV)
    if "team" not in dfm.columns:
        raise KeyError("modeling_dataset missing required column 'team'.")
    dfm["team_alias"] = apply_alias(dfm["team"], alias_map)
    dfm["team_key"]   = dfm["team_alias"].map(std_name)

    # Determine outcome source
    direct_mode_possible = TARGET_COL.lower() in {c.lower() for c in dfm.columns}
    merged = dfm.copy()
    scores_path = None

    if direct_mode_possible:
        # Target is already present in modeling dataset
        log("outcome_join_mode=direct(modeling_dataset)")
        log("outcome_merge_missing_rows=0")
    else:
        # Fallback: locate scores and strict-join on (game_id, team_key)
        scores_path = find_scores_file()
        if not scores_path:
            raise FileNotFoundError(
                "No scores file found and TARGET_COL not present in modeling_dataset. "
                "Provide SCORES_CSV/SCORES_GLOB or include the target in the dataset."
            )
        s_raw = pd.read_csv(scores_path)
        s = normalize_scores_to_team_rows(s_raw, alias_map)

        if "game_id" not in dfm.columns:
            raise KeyError("modeling_dataset missing 'game_id' required for fallback join.")

        merged = dfm.merge(
            s[["game_id", "team_key", "points_scored"]],
            on=["game_id", "team_key"],
            how="left",
            validate="m:1",
        )
        miss = int(merged["points_scored"].isna().sum())
        log(f"scores_csv={scores_path}")
        log("outcome_join_mode=strict(game_id,team_key)")
        log(f"outcome_merge_missing_rows={miss}")

    # Select/verify target
    target = select_target(merged, TARGET_COL)

    # Drop NaN targets
    pre_drop = int(pd.to_numeric(merged[target], errors="coerce").isna().sum())
    log(f"pre_drop_NaN_targets={pre_drop}")
    merged = merged.dropna(subset=[target]).reset_index(drop=True)
    log(f"dropped_NaN_targets={pre_drop}")

    # Build features
    X, y = build_numeric_features(merged, target)
    if X.shape[0] == 0:
        raise ValueError("No rows left after target drop; check outcome merge and TARGET_COL.")
    if X.shape[1] == 0:
        raise ValueError("No numeric features remain after exclusions. Adjust EXCLUDE_TARGET_REGEX or dataset.")

    # Split
    Xtr, ytr, Xte, yte, split_mode = temporal_split(merged, X, y)

    # Models
    ridge_alphas = np.logspace(-3, 3, 13)
    lasso_alphas = np.logspace(-3, 1, 9)
    models = {
        "linear": Pipeline([("scaler", StandardScaler(with_mean=False)), ("lin", LinearRegression())]),
        "ridge" : Pipeline([("scaler", StandardScaler(with_mean=False)), ("rid", RidgeCV(alphas=ridge_alphas))]),
        "lasso" : Pipeline([("scaler", StandardScaler(with_mean=False)), ("las", LassoCV(alphas=lasso_alphas, max_iter=5000, cv=5))]),
    }

    # Fit/eval + save
    rows_metrics, rows_coeffs = [], []
    for name, pipe in models.items():
        rmse, mae, r2, fitted = fit_eval(Xtr, ytr, Xte, yte, name, pipe)
        log(
            f"{name}: rmse={rmse:.6f}, mae={mae:.6f}, r2={r2:.6f}, "
            f"n_train={len(Xtr)}, n_test={len(Xte)}, n_features={Xtr.shape[1]}, split_mode={split_mode}"
        )
        rows_metrics.append(
            {
                "model": name,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "n_train": len(Xtr),
                "n_test": len(Xte),
                "n_features": Xtr.shape[1],
                "split_mode": split_mode,
            }
        )
        joblib.dump(fitted, MODELS_DIR / f"{name}_model.pkl")

        # coefficients
        try:
            if name == "linear":
                est = fitted.named_steps["lin"]
            elif name == "ridge":
                est = fitted.named_steps["rid"]
            else:
                est = fitted.named_steps["las"]
            coefs = np.ravel(getattr(est, "coef_", np.zeros(Xtr.shape[1])))
            for feat, coef in zip(Xtr.columns, coefs):
                rows_coeffs.append({"model": name, "feature": feat, "coefficient": float(coef)})
            if hasattr(est, "intercept_"):
                rows_coeffs.append(
                    {"model": name, "feature": "__intercept__", "coefficient": float(np.atleast_1d(est.intercept_)[0])}
                )
        except Exception as e:
            log(f"WARNING: failed to extract coefficients for {name}: {e}")

    pd.DataFrame(rows_metrics).to_csv(METRICS_CSV, index=False)
    pd.DataFrame(rows_coeffs).to_csv(COEFFS_CSV, index=False)

    # summary header
    hdr = [
        f"snapshot_utc={pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"input_rows={len(merged)}",
        f"features={X.shape[1]}",
        f"target={target}",
        f"scores_csv={scores_path if scores_path else ''}",
    ]
    existing = LOG_FILE.read_text() if LOG_FILE.exists() else ""
    LOG_FILE.write_text("\n".join(hdr) + "\n" + existing)

if __name__ == "__main__":
    # Default leak guard for outcome targets
    if EXCLUDE_TARGET_REGEX is None and TARGET_COL.lower() in ("points_scored", "points", "pts", "score", "margin"):
        EXCLUDE_TARGET_REGEX = r"(?i)^(home_points|away_points|points_allowed|total_points|.*margin.*|.*spread.*)$"
    try:
        main()
    except Exception as e:
        err = f"ERROR: {type(e).__name__}: {e}"
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(err + "\n")
        print(err)
        sys.exit(1)

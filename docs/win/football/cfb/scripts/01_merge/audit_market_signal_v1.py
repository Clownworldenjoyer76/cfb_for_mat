#!/usr/bin/env python3
"""
audit_market_signal_v1.py

Standalone CFB market-error signal audit.

Purpose
-------
This is NOT a prediction model. It audits whether information already present in
this repo / SportsDataverse has stable, forward-season ability to explain the
sportsbook's scoring errors.

Required input (created by train_score_model_v4.py):
    docs/win/football/cfb/data/score_model_v4/
        training_team_rows_2021_2025_v4.parquet

Optional existing inputs (read-only):
    docs/win/football/cfb/data/historical_features/{season}_travel_weather.csv
    docs/win/football/cfb/00_intake/pbp/{season}_pbp.parquet

Outputs (only):
    docs/win/football/cfb/data/signal_audit_v1/

Validation policy
-----------------
* Feature screening / ranking uses ONLY 2021-2024 information.
* Forward validation years are 2022, 2023, 2024.
* 2025 is never used to rank, tune, or choose signals.
* After the pre-2025 shortlist is frozen, 2025 is used once for verification.

The audit tests three sportsbook-error targets:
    team_score_error = actual team points - sportsbook implied team points
    margin_error     = actual home margin - sportsbook implied home margin
    total_error      = actual total - sportsbook total

For each numeric feature, two deliberately simple correction forms are tested:
    linear  - one-variable ridge-shrunk residual correction
    bins    - train-only quintile residual means, shrunk toward zero

The point is not to maximize fit. The point is to find signals that repeat across
future seasons and survive an untouched 2025 check.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_SEASONS = [2021, 2022, 2023, 2024]
CV_YEARS = [2022, 2023, 2024]
FINAL_YEAR = 2025
TARGETS = ["team_score_error", "margin_error", "total_error"]
TOP_PER_TARGET = 25
MIN_PRE2025_ROWS = 250
MIN_FOLD_ROWS = 100
LINEAR_RIDGE_ALPHA = 100.0
BIN_COUNT = 5
BIN_SHRINK_N = 150.0
BOOTSTRAP_REPS = 1000
RNG_SEED = 20260829

TARGET_META = {
    "team_score_error": {
        "base_col": "market_team_score",
        "actual_col": "actual_team_score",
        "level": "team",
    },
    "margin_error": {
        "base_col": "market_home_margin",
        "actual_col": "actual_home_margin",
        "level": "game",
    },
    "total_error": {
        "base_col": "market_total_game",
        "actual_col": "actual_total",
        "level": "game",
    },
}

NON_FEATURE_COLUMNS = {
    "season", "week", "game_id", "game_date",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "team", "opponent", "team_id", "opponent_id", "team_side",
    "actual_team_score", "actual_opp_score", "target_score_residual",
    "actual_home_score", "actual_away_score", "actual_home_margin",
    "actual_total", "team_score_error", "margin_error", "total_error",
}

WEATHER_TOKENS = (
    "temperature", "wind", "gust", "humidity", "rain", "snow",
    "precip", "weather_code",
)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit pregame features for stable sportsbook-error signal.")
    p.add_argument("--cfb-root", type=str, default=None,
                   help="Optional docs/win/football/cfb path. Auto-detected from script location by default.")
    p.add_argument("--top", type=int, default=TOP_PER_TARGET,
                   help=f"Frozen pre-2025 shortlist size per target. Default: {TOP_PER_TARGET}.")
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPS,
                   help=f"2025 bootstrap repetitions. Default: {BOOTSTRAP_REPS}.")
    return p.parse_args()


def resolve_cfb_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    # Intended repo path: docs/win/football/cfb/scripts/01_merge/this_file.py
    return Path(__file__).resolve().parents[2]


def norm_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def safe_num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def sign_int(x: float, tol: float = 1e-12) -> int:
    if not math.isfinite(float(x)) or abs(float(x)) <= tol:
        return 0
    return 1 if x > 0 else -1


def source_for_feature(name: str) -> str:
    n = name.lower()
    if n.startswith("qb_") or "_qb_" in n:
        return "qb_pbp"
    if n.startswith("weather_") or any(tok in n for tok in WEATHER_TOKENS):
        return "weather_observed"
    if n.startswith("travel_") or "miles_traveled" in n or "time_zone" in n or "time_zones" in n:
        return "travel"
    if "_sum_" in n:
        return "sdv_weekly_summary"
    if "_rat_" in n:
        return "sdv_weekly_ratings"
    if "_fpi_" in n:
        return "espn_weekly_fpi"
    if "_pre_" in n:
        return "sdv_preseason"
    if n.startswith("espn_"):
        return "espn_game_predictor"
    if n.startswith("market_"):
        return "sportsbook_context"
    return "context"


def deployable_feature(name: str) -> bool:
    # Historical weather file is observed/reanalysis weather, not a historical
    # point-in-time forecast. It is useful diagnostically but should not be
    # promoted as a deployable signal without a matching forecast backtest.
    return source_for_feature(name) != "weather_observed"


# ---------------------------------------------------------------------------
# Load v4 matrix and form game-level targets
# ---------------------------------------------------------------------------


def load_v4_matrix(cfb_root: Path) -> pd.DataFrame:
    path = cfb_root / "data" / "score_model_v4" / "training_team_rows_2021_2025_v4.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required V4 training matrix: {path}\n"
            "Run train_score_model_v4.py first."
        )
    df = pd.read_parquet(path)
    required = {
        "season", "week", "game_id", "team_side",
        "market_team_score", "market_opp_score",
        "actual_team_score", "actual_opp_score",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"V4 training matrix missing required columns: {missing}")

    df = df.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df["game_id"] = df["game_id"].map(norm_id)
    for c in ["market_team_score", "market_opp_score", "actual_team_score", "actual_opp_score"]:
        df[c] = safe_num_series(df[c])
    df["team_score_error"] = df["actual_team_score"] - df["market_team_score"]
    return df


def build_game_rows(team_rows: pd.DataFrame) -> pd.DataFrame:
    home = team_rows[team_rows["team_side"].astype(str).str.lower().eq("home")].copy()
    if home.empty:
        raise ValueError("No home-side rows found in V4 matrix.")

    home["actual_home_score"] = home["actual_team_score"]
    home["actual_away_score"] = home["actual_opp_score"]
    home["actual_home_margin"] = home["actual_home_score"] - home["actual_away_score"]
    home["actual_total"] = home["actual_home_score"] + home["actual_away_score"]
    home["market_home_margin"] = home["market_team_score"] - home["market_opp_score"]
    home["market_total_game"] = home["market_team_score"] + home["market_opp_score"]
    home["margin_error"] = home["actual_home_margin"] - home["market_home_margin"]
    home["total_error"] = home["actual_total"] - home["market_total_game"]
    return home


# ---------------------------------------------------------------------------
# Optional repo features: travel/weather
# ---------------------------------------------------------------------------


def load_travel_weather(cfb_root: Path) -> pd.DataFrame:
    root = cfb_root / "data" / "historical_features"
    frames: list[pd.DataFrame] = []
    for season in range(2021, 2026):
        path = root / f"{season}_travel_weather.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        if "game_id" not in df.columns:
            continue
        df = df.copy()
        df["game_id"] = df["game_id"].map(norm_id)
        df["season"] = pd.to_numeric(df.get("season", season), errors="coerce").fillna(season).astype(int)
        keep = ["season", "game_id"]
        for c in df.columns:
            if c in keep:
                continue
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() >= max(20, int(0.05 * len(df))):
                df[c] = x
                keep.append(c)
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame(columns=["season", "game_id"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.drop_duplicates(["season", "game_id"], keep="last")

    # Prefix all non-key fields so audit provenance is explicit.
    rename = {c: f"travel_weather_{c}" for c in out.columns if c not in {"season", "game_id"}}
    return out.rename(columns=rename)


def add_travel_weather(team_rows: pd.DataFrame, game_rows: pd.DataFrame, tw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tw.empty:
        return team_rows, game_rows
    keys = ["season", "game_id"]
    team = team_rows.merge(tw, on=keys, how="left")
    game = game_rows.merge(tw, on=keys, how="left")

    # Perspective-aware travel fields for the team-score audit.
    ah = "travel_weather_away_miles_traveled"
    hh = "travel_weather_home_miles_traveled"
    if ah in team.columns and hh in team.columns:
        away = safe_num_series(team[ah])
        home = safe_num_series(team[hh])
        is_home = team["team_side"].astype(str).str.lower().eq("home")
        team["travel_team_miles"] = np.where(is_home, home, away)
        team["travel_opp_miles"] = np.where(is_home, away, home)
        team["travel_team_minus_opp_miles"] = team["travel_team_miles"] - team["travel_opp_miles"]
    return team, game


# ---------------------------------------------------------------------------
# Optional repo features: leakage-safe prior-QB continuity from local PBP
# ---------------------------------------------------------------------------


def pbp_schema_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        try:
            return list(pd.read_parquet(path).columns)
        except Exception:
            return []


def build_qb_state(cfb_root: Path, team_rows: pd.DataFrame) -> pd.DataFrame:
    """Build pre-week QB continuity state from prior PBP only.

    No current-game passer is used. Week 1 receives previous-season primary-QB
    information only; Week 2+ may also use current-season pass attempts through
    the prior week.
    """
    pbp_dir = cfb_root / "00_intake" / "pbp"
    count_frames: list[pd.DataFrame] = []

    for season in range(2020, 2026):
        path = pbp_dir / f"{season}_pbp.parquet"
        if not path.exists():
            continue
        cols = pbp_schema_columns(path)
        if not cols:
            continue
        passer_col = next((c for c in ["passer_player_id", "passer_player_name", "passer"] if c in cols), None)
        team_col = next((c for c in ["pos_team", "posteam"] if c in cols), None)
        pass_col = next((c for c in ["pass_attempt", "pass", "pass_flag"] if c in cols), None)
        if not passer_col or not team_col or "week" not in cols:
            continue
        use = ["week", team_col, passer_col]
        if pass_col:
            use.append(pass_col)
        try:
            p = pd.read_parquet(path, columns=use)
        except Exception:
            continue
        p = p.copy()
        p["week"] = pd.to_numeric(p["week"], errors="coerce")
        p["team"] = p[team_col].map(clean_text)
        p["passer"] = p[passer_col].map(clean_text)
        if pass_col:
            v = p[pass_col]
            if pd.api.types.is_bool_dtype(v):
                mask = v.fillna(False)
            else:
                numv = pd.to_numeric(v, errors="coerce")
                if numv.notna().any():
                    mask = numv.fillna(0).ne(0)
                else:
                    mask = v.astype(str).str.lower().isin({"true", "1", "yes"})
            p = p[mask]
        p = p[p["week"].notna() & p["team"].ne("") & p["passer"].ne("")]
        if p.empty:
            continue
        counts = (
            p.groupby(["week", "team", "passer"]).size()
            .rename("attempts").reset_index()
        )
        counts["season"] = season
        count_frames.append(counts[["season", "week", "team", "passer", "attempts"]])

    if not count_frames:
        return pd.DataFrame(columns=["season", "week", "team"])

    counts_all = pd.concat(count_frames, ignore_index=True)
    counts_all["season"] = pd.to_numeric(counts_all["season"], errors="coerce").astype(int)
    counts_all["week"] = pd.to_numeric(counts_all["week"], errors="coerce").astype(int)
    counts_all["attempts"] = pd.to_numeric(counts_all["attempts"], errors="coerce").fillna(0.0)

    season_totals = (
        counts_all.groupby(["season", "team", "passer"], as_index=False)["attempts"].sum()
    )
    prev_primary: dict[tuple[int, str], tuple[str, float, float]] = {}
    for (season, team), grp in season_totals.groupby(["season", "team"]):
        total = float(grp["attempts"].sum())
        best = grp.loc[grp["attempts"].idxmax()]
        att = float(best["attempts"])
        prev_primary[(int(season), str(team))] = (
            str(best["passer"]), att, att / total if total > 0 else np.nan
        )

    combos = team_rows[["season", "week", "team"]].drop_duplicates().copy()
    combos["season"] = pd.to_numeric(combos["season"], errors="coerce").astype(int)
    combos["week"] = pd.to_numeric(combos["week"], errors="coerce").astype(int)
    combos["team"] = combos["team"].map(clean_text)

    grouped_counts = {
        (int(season), str(team)): grp.sort_values("week")
        for (season, team), grp in counts_all.groupby(["season", "team"])
    }

    out_rows: list[dict[str, Any]] = []
    for (season, team), cgrp in combos.groupby(["season", "team"]):
        season = int(season)
        team = str(team)
        previous = prev_primary.get((season - 1, team))
        wcounts = grouped_counts.get((season, team))
        by_week: dict[int, list[tuple[str, float]]] = {}
        if wcounts is not None:
            for w, wg in wcounts.groupby("week"):
                by_week[int(w)] = [(str(r["passer"]), float(r["attempts"])) for _, r in wg.iterrows()]

        cumulative: dict[str, float] = {}
        last_week_added = 0
        for week in sorted(int(w) for w in cgrp["week"].unique()):
            for prior_week in sorted(w for w in by_week if last_week_added < w < week):
                for passer, att in by_week[prior_week]:
                    cumulative[passer] = cumulative.get(passer, 0.0) + att
                last_week_added = max(last_week_added, prior_week)

            cur_total = float(sum(cumulative.values()))
            cur_primary = ""
            cur_primary_att = 0.0
            if cumulative:
                cur_primary, cur_primary_att = max(cumulative.items(), key=lambda z: z[1])

            out_rows.append({
                "season": season,
                "week": week,
                "team": team,
                "qb_prior_season_primary_attempts": previous[1] if previous else np.nan,
                "qb_prior_season_primary_share": previous[2] if previous else np.nan,
                "qb_current_prior_attempts": cur_total,
                "qb_current_primary_share": cur_primary_att / cur_total if cur_total > 0 else np.nan,
                "qb_current_num_passers": float(sum(1 for v in cumulative.values() if v >= 5.0)),
                "qb_current_primary_same_as_prev": (
                    1.0 if previous and cur_primary and cur_primary == previous[0]
                    else 0.0 if previous and cur_primary
                    else np.nan
                ),
            })
    return pd.DataFrame(out_rows)

def add_qb_state(team_rows: pd.DataFrame, game_rows: pd.DataFrame, qb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if qb.empty:
        return team_rows, game_rows
    team = team_rows.merge(qb, on=["season", "week", "team"], how="left")

    # Game rows are home-perspective. Join home and away QB state separately.
    qcols = [c for c in qb.columns if c not in {"season", "week", "team"}]
    home_q = qb.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in qcols}})
    away_q = qb.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in qcols}})
    game = game_rows.merge(home_q, on=["season", "week", "home_team"], how="left")
    game = game.merge(away_q, on=["season", "week", "away_team"], how="left")
    for c in qcols:
        hc = f"home_{c}"
        ac = f"away_{c}"
        if hc in game.columns and ac in game.columns:
            game[f"diff_{c}"] = safe_num_series(game[hc]) - safe_num_series(game[ac])
    return team, game


# ---------------------------------------------------------------------------
# Feature freezing
# ---------------------------------------------------------------------------


def candidate_numeric_features(df: pd.DataFrame) -> list[str]:
    pre = df[df["season"].isin(TRAIN_SEASONS)].copy()
    out: list[str] = []
    for c in df.columns:
        if c in NON_FEATURE_COLUMNS:
            continue
        x = safe_num_series(pre[c])
        if x.notna().sum() < MIN_PRE2025_ROWS:
            continue
        if x.dropna().nunique() <= 1:
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# One-feature correction methods
# ---------------------------------------------------------------------------


@dataclass
class LinearFit:
    mean_x: float
    sd_x: float
    mean_y: float
    coef: float


def fit_linear(x: np.ndarray, y: np.ndarray) -> LinearFit:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 20:
        return LinearFit(0.0, 1.0, 0.0, 0.0)
    mx = float(np.mean(x))
    sx = float(np.std(x))
    if not math.isfinite(sx) or sx < 1e-12:
        return LinearFit(mx, 1.0, float(np.mean(y)), 0.0)
    z = (x - mx) / sx
    my = float(np.mean(y))
    yc = y - my
    coef = float(np.sum(z * yc) / (np.sum(z * z) + LINEAR_RIDGE_ALPHA))
    return LinearFit(mx, sx, my, coef)


def predict_linear(fit: LinearFit, x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), fit.mean_y, dtype=float)
    mask = np.isfinite(x)
    out[mask] = fit.mean_y + fit.coef * ((x[mask] - fit.mean_x) / fit.sd_x)
    return out


@dataclass
class BinFit:
    edges: np.ndarray
    values: np.ndarray
    fallback: float


def fit_bins(x: np.ndarray, y: np.ndarray) -> BinFit:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    fallback = float(np.mean(y)) if len(y) else 0.0
    if len(x) < 50 or len(np.unique(x)) < 4:
        return BinFit(np.array([-np.inf, np.inf]), np.array([fallback]), fallback)

    qs = np.linspace(0, 1, BIN_COUNT + 1)
    raw = np.quantile(x, qs)
    edges = np.unique(raw)
    if len(edges) < 3:
        return BinFit(np.array([-np.inf, np.inf]), np.array([fallback]), fallback)
    edges[0] = -np.inf
    edges[-1] = np.inf
    idx = np.digitize(x, edges[1:-1], right=True)
    values = np.zeros(len(edges) - 1, dtype=float)
    for b in range(len(values)):
        yy = y[idx == b]
        if len(yy) == 0:
            values[b] = fallback
        else:
            raw_mean = float(np.mean(yy))
            shrink = len(yy) / (len(yy) + BIN_SHRINK_N)
            values[b] = shrink * raw_mean + (1.0 - shrink) * fallback
    return BinFit(edges, values, fallback)


def predict_bins(fit: BinFit, x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), fit.fallback, dtype=float)
    mask = np.isfinite(x)
    if mask.any():
        idx = np.digitize(x[mask], fit.edges[1:-1], right=True)
        idx = np.clip(idx, 0, len(fit.values) - 1)
        out[mask] = fit.values[idx]
    return out


def fit_method(method: str, x: np.ndarray, y: np.ndarray) -> Any:
    if method == "linear":
        return fit_linear(x, y)
    if method == "bins":
        return fit_bins(x, y)
    raise ValueError(method)


def predict_method(method: str, fit: Any, x: np.ndarray) -> np.ndarray:
    if method == "linear":
        return predict_linear(fit, x)
    if method == "bins":
        return predict_bins(fit, x)
    raise ValueError(method)


def fit_direction(method: str, fit: Any) -> int:
    if method == "linear":
        return sign_int(fit.coef)
    # For bins, use endpoint difference as a simple direction descriptor.
    if len(fit.values) < 2:
        return 0
    return sign_int(float(fit.values[-1] - fit.values[0]))


# ---------------------------------------------------------------------------
# Forward-season audit
# ---------------------------------------------------------------------------


def pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(float)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(float)
    return pearson_safe(xr, yr)


def evaluate_feature_method(df: pd.DataFrame, feature: str, target: str, method: str) -> dict[str, Any] | None:
    work = df[["season", feature, target]].copy()
    work[feature] = safe_num_series(work[feature])
    work[target] = safe_num_series(work[target])

    fold_rows: list[dict[str, Any]] = []
    pooled_y: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    pooled_x: list[np.ndarray] = []

    for year in CV_YEARS:
        tr = work[(work["season"] < year) & work["season"].isin(TRAIN_SEASONS)]
        va = work[work["season"] == year]
        tr = tr.dropna(subset=[feature, target])
        va = va.dropna(subset=[feature, target])
        if len(tr) < MIN_FOLD_ROWS or len(va) < MIN_FOLD_ROWS:
            continue
        xtr = tr[feature].to_numpy(float)
        ytr = tr[target].to_numpy(float)
        xva = va[feature].to_numpy(float)
        yva = va[target].to_numpy(float)

        fit = fit_method(method, xtr, ytr)
        pred = predict_method(method, fit, xva)
        base_mae = float(np.mean(np.abs(yva)))
        cand_mae = float(np.mean(np.abs(yva - pred)))
        improvement = base_mae - cand_mae
        fold_rows.append({
            "year": year,
            "n": len(va),
            "base_mae": base_mae,
            "candidate_mae": cand_mae,
            "improvement": improvement,
            "direction": fit_direction(method, fit),
            "pearson": pearson_safe(xva, yva),
            "spearman": spearman_safe(xva, yva),
        })
        pooled_y.append(yva)
        pooled_pred.append(pred)
        pooled_x.append(xva)

    if len(fold_rows) < 2:
        return None

    y = np.concatenate(pooled_y)
    pred = np.concatenate(pooled_pred)
    x = np.concatenate(pooled_x)
    pooled_base = float(np.mean(np.abs(y)))
    pooled_cand = float(np.mean(np.abs(y - pred)))
    improvements = [r["improvement"] for r in fold_rows]
    dirs = [r["direction"] for r in fold_rows if r["direction"] != 0]
    corr_signs = [sign_int(r["spearman"]) for r in fold_rows if math.isfinite(r["spearman"]) and sign_int(r["spearman"]) != 0]
    direction_consistent = len(dirs) >= 2 and len(set(dirs)) == 1
    corr_consistent = len(corr_signs) >= 2 and len(set(corr_signs)) == 1
    fold_wins = sum(1 for z in improvements if z > 0)

    row: dict[str, Any] = {
        "target": target,
        "feature": feature,
        "source": source_for_feature(feature),
        "deployable": deployable_feature(feature),
        "method": method,
        "cv_folds": len(fold_rows),
        "cv_rows": len(y),
        "cv_base_mae": pooled_base,
        "cv_candidate_mae": pooled_cand,
        "cv_improvement": pooled_base - pooled_cand,
        "mean_fold_improvement": float(np.mean(improvements)),
        "worst_fold_improvement": float(np.min(improvements)),
        "fold_wins": fold_wins,
        "direction_consistent": direction_consistent,
        "correlation_sign_consistent": corr_consistent,
        "cv_pearson": pearson_safe(x, y),
        "cv_spearman": spearman_safe(x, y),
    }
    for r in fold_rows:
        yk = r["year"]
        row[f"improvement_{yk}"] = r["improvement"]
        row[f"direction_{yk}"] = r["direction"]
        row[f"spearman_{yk}"] = r["spearman"]
    row["stable_cv"] = bool(
        row["cv_improvement"] > 0
        and fold_wins >= 2
        and direction_consistent
    )
    return row


def audit_all_features(team_rows: pd.DataFrame, game_rows: pd.DataFrame) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    for target in TARGETS:
        df = team_rows if TARGET_META[target]["level"] == "team" else game_rows
        features = candidate_numeric_features(df)
        print(f"Auditing {target}: {len(features)} numeric features x 2 methods...")
        for i, feature in enumerate(features, start=1):
            for method in ["linear", "bins"]:
                row = evaluate_feature_method(df, feature, target, method)
                if row is not None:
                    results.append(row)
            if i % 200 == 0 or i == len(features):
                print(f"  {i}/{len(features)} features")
    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results)
    out = out.sort_values(["target", "stable_cv", "cv_improvement"], ascending=[True, False, False])
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Freeze shortlist without 2025
# ---------------------------------------------------------------------------


def freeze_shortlist(all_cv: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for target in TARGETS:
        part = all_cv[(all_cv["target"] == target) & (all_cv["deployable"] == True)].copy()  # noqa: E712
        # One method per feature, selected entirely from 2022-2024 CV.
        part = part.sort_values(["feature", "cv_improvement"], ascending=[True, False])
        part = part.drop_duplicates("feature", keep="first")
        stable = part[part["stable_cv"] == True].copy()  # noqa: E712
        stable = stable.sort_values(
            ["fold_wins", "cv_improvement", "worst_fold_improvement"],
            ascending=[False, False, False],
        )
        if len(stable) < top_n:
            # Fill remaining slots with strongest positive CV signals, but label
            # them unstable so 2025 cannot retroactively make them "stable".
            extra = part[(part["cv_improvement"] > 0) & (~part["feature"].isin(stable["feature"]))].copy()
            extra = extra.sort_values(["fold_wins", "cv_improvement"], ascending=[False, False])
            stable = pd.concat([stable, extra.head(top_n - len(stable))], ignore_index=True)
        else:
            stable = stable.head(top_n)
        stable["pre2025_rank"] = np.arange(1, len(stable) + 1)
        rows.append(stable)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Untouched 2025 verification
# ---------------------------------------------------------------------------


def bootstrap_improvement(y: np.ndarray, pred: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    diff = np.abs(y) - np.abs(y - pred)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 20 or reps <= 0:
        return np.nan, np.nan
    means = np.empty(reps, dtype=float)
    n = len(diff)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(diff[idx]))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def verify_shortlist(shortlist: pd.DataFrame, team_rows: pd.DataFrame, game_rows: pd.DataFrame, reps: int) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, Any]] = []
    for _, s in shortlist.iterrows():
        target = str(s["target"])
        feature = str(s["feature"])
        method = str(s["method"])
        df = team_rows if TARGET_META[target]["level"] == "team" else game_rows
        if feature not in df.columns:
            continue
        tr = df[df["season"].isin(TRAIN_SEASONS)][[feature, target]].copy().dropna()
        va = df[df["season"] == FINAL_YEAR][[feature, target]].copy().dropna()
        if len(tr) < MIN_FOLD_ROWS or len(va) < MIN_FOLD_ROWS:
            continue
        xtr = safe_num_series(tr[feature]).to_numpy(float)
        ytr = safe_num_series(tr[target]).to_numpy(float)
        xva = safe_num_series(va[feature]).to_numpy(float)
        yva = safe_num_series(va[target]).to_numpy(float)
        mask_tr = np.isfinite(xtr) & np.isfinite(ytr)
        mask_va = np.isfinite(xva) & np.isfinite(yva)
        xtr, ytr = xtr[mask_tr], ytr[mask_tr]
        xva, yva = xva[mask_va], yva[mask_va]
        fit = fit_method(method, xtr, ytr)
        pred = predict_method(method, fit, xva)
        base = float(np.mean(np.abs(yva)))
        cand = float(np.mean(np.abs(yva - pred)))
        imp = base - cand
        lo, hi = bootstrap_improvement(yva, pred, reps, rng)
        cv_dir = 0
        dirs = [int(s.get(f"direction_{year}", 0)) for year in CV_YEARS if pd.notna(s.get(f"direction_{year}", np.nan))]
        dirs = [d for d in dirs if d != 0]
        if dirs and len(set(dirs)) == 1:
            cv_dir = dirs[0]
        final_dir = fit_direction(method, fit)
        rows.append({
            "target": target,
            "pre2025_rank": int(s["pre2025_rank"]),
            "feature": feature,
            "source": s["source"],
            "method": method,
            "stable_cv": bool(s["stable_cv"]),
            "cv_improvement": float(s["cv_improvement"]),
            "cv_fold_wins": int(s["fold_wins"]),
            "n_2025": len(yva),
            "market_mae_2025": base,
            "corrected_mae_2025": cand,
            "improvement_2025": imp,
            "bootstrap_low_2025": lo,
            "bootstrap_high_2025": hi,
            "cv_direction": cv_dir,
            "fit_direction_2021_2024": final_dir,
            "direction_preserved": bool(cv_dir != 0 and final_dir == cv_dir),
            "positive_2025": bool(imp > 0),
            "bootstrap_positive_2025": bool(math.isfinite(lo) and lo > 0),
            "confirmed_strict": bool(
                bool(s["stable_cv"])
                and imp > 0
                and cv_dir != 0
                and final_dir == cv_dir
                and math.isfinite(lo)
                and lo > 0
            ),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "confirmed_strict", "improvement_2025"], ascending=[True, False, False])
    return out


# ---------------------------------------------------------------------------
# Subgroup market-bias audit
# ---------------------------------------------------------------------------


def add_subgroup_flags(game_rows: pd.DataFrame) -> pd.DataFrame:
    g = game_rows.copy()
    week = safe_num_series(g["week"])
    spread = safe_num_series(g.get("market_home_margin", pd.Series(index=g.index, dtype=float))).abs()
    total = safe_num_series(g.get("market_total_game", pd.Series(index=g.index, dtype=float)))
    neutral = safe_num_series(g.get("neutral_site", pd.Series(index=g.index, dtype=float))).fillna(0)

    g["sg_week1"] = week.eq(1)
    g["sg_weeks1_4"] = week.between(1, 4)
    g["sg_weeks5_plus"] = week.ge(5)
    g["sg_close_spread_3_5"] = spread.le(3.5)
    g["sg_large_spread_14"] = spread.ge(14)
    g["sg_huge_spread_21"] = spread.ge(21)
    g["sg_low_total_45"] = total.le(45)
    g["sg_high_total_60"] = total.ge(60)
    g["sg_neutral"] = neutral.eq(1)

    for col in ["travel_weather_away_miles_traveled", "travel_weather_home_miles_traveled"]:
        if col not in g.columns:
            g[col] = np.nan
    away_m = safe_num_series(g["travel_weather_away_miles_traveled"])
    home_m = safe_num_series(g["travel_weather_home_miles_traveled"])
    g["sg_away_travel_1000"] = away_m.ge(1000)
    g["sg_travel_gap_1000"] = (away_m - home_m).abs().ge(1000)

    wind_col = next((c for c in g.columns if c.lower().endswith("wind_speed") or "wind_speed" in c.lower()), None)
    if wind_col:
        g["sg_high_wind_8ms"] = safe_num_series(g[wind_col]).ge(8)
    else:
        g["sg_high_wind_8ms"] = False

    # FPI coverage mismatch is a transparent proxy for FBS-vs-non-FBS / data
    # coverage mismatch. It is deliberately not mislabeled as exact FBS/FCS.
    own_fpi_cols = [c for c in g.columns if c.startswith("own_fpi_")]
    opp_fpi_cols = [c for c in g.columns if c.startswith("opp_fpi_")]
    if own_fpi_cols and opp_fpi_cols:
        own_has = g[own_fpi_cols].notna().any(axis=1)
        opp_has = g[opp_fpi_cols].notna().any(axis=1)
        g["sg_one_team_missing_fpi"] = own_has ^ opp_has
    else:
        g["sg_one_team_missing_fpi"] = False

    talent_col = next((c for c in g.columns if c.startswith("diff_pre_") and "talent_composite" in c.lower()), None)
    if talent_col:
        td = safe_num_series(g[talent_col]).abs()
        threshold = float(td[g["season"].isin(TRAIN_SEASONS)].quantile(0.75)) if td.notna().any() else np.nan
        g["sg_large_talent_gap_q75"] = td.ge(threshold) if math.isfinite(threshold) else False
    else:
        g["sg_large_talent_gap_q75"] = False
    return g


def subgroup_forward_test(game_rows: pd.DataFrame) -> pd.DataFrame:
    g = add_subgroup_flags(game_rows)
    subgroup_cols = [c for c in g.columns if c.startswith("sg_")]
    rows: list[dict[str, Any]] = []
    for target in ["margin_error", "total_error"]:
        for sg in subgroup_cols:
            fold_imps: list[float] = []
            fold_bias: list[float] = []
            fold_ns: list[int] = []
            for year in CV_YEARS:
                tr = g[(g["season"] < year) & g["season"].isin(TRAIN_SEASONS) & g[sg].fillna(False)]
                va = g[(g["season"] == year) & g[sg].fillna(False)]
                ytr = safe_num_series(tr[target]).dropna().to_numpy(float)
                yva = safe_num_series(va[target]).dropna().to_numpy(float)
                if len(ytr) < 50 or len(yva) < 30:
                    continue
                # Conservative prior-subgroup-bias correction shrunk toward 0.
                raw_bias = float(np.mean(ytr))
                shrink = len(ytr) / (len(ytr) + 200.0)
                correction = shrink * raw_bias
                base = float(np.mean(np.abs(yva)))
                cand = float(np.mean(np.abs(yva - correction)))
                fold_imps.append(base - cand)
                fold_bias.append(float(np.mean(yva)))
                fold_ns.append(len(yva))
            if len(fold_imps) < 2:
                continue
            signs = [sign_int(x) for x in fold_bias if sign_int(x) != 0]
            rows.append({
                "target": target,
                "subgroup": sg,
                "cv_folds": len(fold_imps),
                "cv_rows": int(sum(fold_ns)),
                "mean_market_bias": float(np.average(fold_bias, weights=fold_ns)),
                "bias_sign_consistent": bool(len(signs) >= 2 and len(set(signs)) == 1),
                "mean_fold_improvement": float(np.mean(fold_imps)),
                "worst_fold_improvement": float(np.min(fold_imps)),
                "fold_wins": int(sum(1 for z in fold_imps if z > 0)),
                "stable_subgroup_bias": bool(
                    sum(1 for z in fold_imps if z > 0) >= 2
                    and len(signs) >= 2
                    and len(set(signs)) == 1
                    and np.mean(fold_imps) > 0
                ),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "stable_subgroup_bias", "mean_fold_improvement"], ascending=[True, False, False])
    return out


# ---------------------------------------------------------------------------
# Summary / outputs
# ---------------------------------------------------------------------------


def build_summary(all_cv: pd.DataFrame, shortlist: pd.DataFrame, verify: pd.DataFrame, subgroup: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "validation_policy": {
            "feature_selection_years": TRAIN_SEASONS,
            "forward_cv_years": CV_YEARS,
            "untouched_final_year": FINAL_YEAR,
            "final_year_used_for_selection": False,
        },
        "targets": {},
        "strict_confirmed_signals": [],
        "stable_subgroup_biases": [],
    }
    for target in TARGETS:
        cvp = all_cv[all_cv["target"] == target] if not all_cv.empty else pd.DataFrame()
        vp = verify[verify["target"] == target] if not verify.empty else pd.DataFrame()
        result["targets"][target] = {
            "feature_method_tests": int(len(cvp)),
            "stable_cv_tests": int(cvp["stable_cv"].sum()) if not cvp.empty else 0,
            "shortlisted": int((shortlist["target"] == target).sum()) if not shortlist.empty else 0,
            "positive_2025": int(vp["positive_2025"].sum()) if not vp.empty else 0,
            "strict_confirmed_2025": int(vp["confirmed_strict"].sum()) if not vp.empty else 0,
        }
    if not verify.empty:
        cols = ["target", "feature", "source", "method", "cv_improvement", "improvement_2025", "bootstrap_low_2025", "bootstrap_high_2025"]
        result["strict_confirmed_signals"] = verify[verify["confirmed_strict"]][cols].to_dict("records")
    if not subgroup.empty:
        cols = ["target", "subgroup", "mean_market_bias", "mean_fold_improvement", "fold_wins"]
        result["stable_subgroup_biases"] = subgroup[subgroup["stable_subgroup_bias"]][cols].to_dict("records")
    return result


def print_top_results(verify: pd.DataFrame, subgroup: pd.DataFrame) -> None:
    print("\n2025 VERIFICATION OF FROZEN SHORTLIST:")
    if verify.empty:
        print("  no shortlist signals had enough 2025 coverage")
    else:
        for target in TARGETS:
            p = verify[verify["target"] == target].sort_values("improvement_2025", ascending=False).head(5)
            if p.empty:
                continue
            print(f"  {target}:")
            for _, r in p.iterrows():
                tag = "CONFIRMED" if bool(r["confirmed_strict"]) else "not confirmed"
                print(
                    f"    {r['feature']} [{r['method']}] "
                    f"CV={r['cv_improvement']:+.4f} 2025={r['improvement_2025']:+.4f} "
                    f"CI=({r['bootstrap_low_2025']:+.4f},{r['bootstrap_high_2025']:+.4f}) {tag}"
                )
    if not subgroup.empty:
        stable = subgroup[subgroup["stable_subgroup_bias"]].head(10)
        print("\nSTABLE PRE-2025 SUBGROUP BIASES:")
        if stable.empty:
            print("  none")
        else:
            for _, r in stable.iterrows():
                print(
                    f"  {r['target']} {r['subgroup']}: "
                    f"bias={r['mean_market_bias']:+.4f} "
                    f"CV improvement={r['mean_fold_improvement']:+.4f} wins={int(r['fold_wins'])}/{int(r['cv_folds'])}"
                )


def main() -> int:
    args = parse_args()
    cfb_root = resolve_cfb_root(args.cfb_root)
    out_dir = cfb_root / "data" / "signal_audit_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CFB root: {cfb_root}")
    print("Loading V4 point-in-time training matrix...")
    team_rows = load_v4_matrix(cfb_root)
    game_rows = build_game_rows(team_rows)
    print(f"  team rows={len(team_rows)} games={len(game_rows)}")

    print("Adding optional historical travel/weather context...")
    tw = load_travel_weather(cfb_root)
    team_rows, game_rows = add_travel_weather(team_rows, game_rows, tw)
    print(f"  travel/weather games={len(tw)}")

    print("Adding optional leakage-safe prior-QB continuity from local PBP...")
    qb = build_qb_state(cfb_root, team_rows)
    team_rows, game_rows = add_qb_state(team_rows, game_rows, qb)
    print(f"  QB state rows={len(qb)}")

    print("\nRunning forward-season feature audit using 2022-2024 only...")
    all_cv = audit_all_features(team_rows, game_rows)
    if all_cv.empty:
        raise RuntimeError("No feature audit results were produced.")

    print("Freezing strongest pre-2025 signals...")
    shortlist = freeze_shortlist(all_cv, max(1, int(args.top)))

    print("Running untouched 2025 verification of frozen shortlist...")
    verify = verify_shortlist(shortlist, team_rows, game_rows, max(0, int(args.bootstrap)))

    print("Auditing predefined game subgroups for repeated market bias...")
    subgroup = subgroup_forward_test(game_rows)

    all_cv_path = out_dir / "all_feature_cv_2022_2024.csv"
    shortlist_path = out_dir / "shortlist_frozen_pre2025.csv"
    verify_path = out_dir / "verification_2025.csv"
    subgroup_path = out_dir / "subgroup_signal_cv_2022_2024.csv"
    summary_path = out_dir / "signal_audit_summary.json"

    all_cv.to_csv(all_cv_path, index=False)
    shortlist.to_csv(shortlist_path, index=False)
    verify.to_csv(verify_path, index=False)
    subgroup.to_csv(subgroup_path, index=False)

    summary = build_summary(all_cv, shortlist, verify, subgroup)
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    print_top_results(verify, subgroup)

    strict_n = int(verify["confirmed_strict"].sum()) if not verify.empty else 0
    print("\nAUDIT RESULT:")
    if strict_n > 0:
        print(f"  STRICTLY CONFIRMED SIGNALS: {strict_n}")
        print("  These are candidates for a narrowly targeted model; they are not yet proof of betting profitability.")
    else:
        print("  NO STRICTLY CONFIRMED FEATURE SIGNALS")
        print("  Do not build another full score model from these inputs unless a subgroup result justifies a narrower test.")

    print(f"\nAll feature CV: {all_cv_path}")
    print(f"Frozen shortlist: {shortlist_path}")
    print(f"2025 verification: {verify_path}")
    print(f"Subgroup CV: {subgroup_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)

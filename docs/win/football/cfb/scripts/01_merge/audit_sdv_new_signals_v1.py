#!/usr/bin/env python3
"""
audit_sdv_new_signals_v1.py

Three-part leakage-safe audit of SportsDataverse information that was not
properly tested by CFB score-model V1-V5:

1) QB execution: prior-week CPOE + QBR EPA.
2) Play-calling tendency: prior-week xPass / Pass OE.
3) Training-history length: the same narrow SDV ratings model trained on
   recent seasons versus the full 2004-2024 history.

The sportsbook remains the baseline. Model/weight selection uses 2022-2024
only. 2025 is untouched until choices are frozen.

Reads existing V5 historical market/team rows and local PBP when possible.
If local PBP predates CPOE/Pass OE, official SportsDataverse PBP parquets are
cached under the audit output directory and used instead. No sportsdataverse
Python package upgrade is required.

Writes only under:
    docs/win/football/cfb/data/sdv_new_signal_audit_v1/
"""

from __future__ import annotations

import bisect
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "sdv-new-signals-v1-2026-08-29"
BASE_SEASONS = [2021, 2022, 2023, 2024, 2025]
CV_SEASONS = [2022, 2023, 2024]
FINAL_SEASON = 2025
LONG_START = 2004
RANDOM_STATE = 20260829
RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0]
CORRECTION_WEIGHTS = [0.0, 0.25, 0.50, 0.75, 1.0]
DIRECT_BLEND_WEIGHTS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0]
BOOTSTRAP_N = 1500

SDV_RELEASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
PBP_URL = SDV_RELEASE + "/espn_cfb_pbp/play_by_play_{season}.parquet"
RATINGS_URL = SDV_RELEASE + "/cfb_ratings_weekly/cfb_ratings_weekly_{season}.parquet"
SCHEDULE_URL = SDV_RELEASE + "/cfb_schedules/cfb_schedules_{season}.parquet"

QB_RAW = ["cpoe", "qbr_epa"]
PASSOE_RAW = ["pass_oe", "xpass"]


def root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def output_dir(root: Path) -> Path:
    p = root / "data" / "sdv_new_signal_audit_v1"
    p.mkdir(parents=True, exist_ok=True)
    (p / "cache").mkdir(parents=True, exist_ok=True)
    return p


def norm_id(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    if s.endswith(".0"):
        try:
            return str(int(float(s)))
        except Exception:
            pass
    return s


def num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def first_existing(cols: set[str], names: list[str]) -> str | None:
    for name in names:
        if name in cols:
            return name
    return None


def download(url: str, dest: Path, timeout: int = 300) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"downloading {url}")
    try:
        with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq
        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        return set(pd.read_parquet(path).columns)


# ---------------------------------------------------------------------------
# Existing market baseline from V5
# ---------------------------------------------------------------------------


def load_market_games(root: Path) -> pd.DataFrame:
    path = root / "data" / "score_model_v5" / "training_team_rows_2021_2025_v5.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Missing V5 team matrix. Expected: "
            "docs/win/football/cfb/data/score_model_v5/training_team_rows_2021_2025_v5.parquet"
        )
    d = pd.read_parquet(path)
    required = {
        "season", "week", "game_id", "team_id", "home_team_id", "away_team_id",
        "market_team_margin", "market_total", "actual_team_score", "actual_opp_score",
    }
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"V5 team matrix missing required columns: {missing}")

    d["game_id"] = d["game_id"].map(norm_id)
    d["team_id"] = d["team_id"].map(norm_id)
    d["home_team_id"] = d["home_team_id"].map(norm_id)
    d["away_team_id"] = d["away_team_id"].map(norm_id)
    d["season"] = pd.to_numeric(d["season"], errors="coerce").astype("Int64")
    d["week"] = pd.to_numeric(d["week"], errors="coerce").astype("Int64")

    home = d[d["team_id"] == d["home_team_id"]].copy()
    if home.empty and "team_side" in d.columns:
        home = d[d["team_side"].astype(str).str.lower().eq("home")].copy()
    if home.empty:
        raise ValueError("Could not identify home-team rows in V5 team matrix.")

    home = home.sort_values(["season", "week", "game_id"]).drop_duplicates("game_id", keep="last")
    home["market_margin"] = num_series(home["market_team_margin"])
    home["market_total"] = num_series(home["market_total"])
    home["actual_home"] = num_series(home["actual_team_score"])
    home["actual_away"] = num_series(home["actual_opp_score"])
    home["actual_margin"] = home["actual_home"] - home["actual_away"]
    home["actual_total"] = home["actual_home"] + home["actual_away"]

    keep = [
        "season", "week", "game_id", "home_team_id", "away_team_id",
        "market_margin", "market_total", "actual_home", "actual_away",
        "actual_margin", "actual_total",
    ]
    for c in ("home_team", "away_team"):
        if c in home.columns:
            keep.append(c)
    out = home[keep].copy()
    out = out[out["season"].isin(BASE_SEASONS)].copy()
    return out


# ---------------------------------------------------------------------------
# PBP source + prior-week signal construction
# ---------------------------------------------------------------------------


def choose_pbp_source(root: Path, out: Path, season: int) -> tuple[Path, str, set[str]]:
    local = root / "00_intake" / "pbp" / f"{season}_pbp.parquet"
    required_signal = {"cpoe", "qbr_epa", "pass_oe", "xpass"}
    if local.exists():
        cols = parquet_columns(local)
        if required_signal.issubset(cols):
            return local, "local", cols

    cached = out / "cache" / f"official_play_by_play_{season}.parquet"
    if not cached.exists():
        download(PBP_URL.format(season=season), cached)
    cols = parquet_columns(cached)
    missing = sorted(required_signal - cols)
    if missing:
        raise ValueError(f"Official SDV PBP {season} missing expected signal columns: {missing}")
    return cached, "official", cols


def load_signal_pbp(path: Path, season: int, cols: set[str]) -> pd.DataFrame:
    season_col = first_existing(cols, ["season"])
    week_col = first_existing(cols, ["week"])
    game_col = first_existing(cols, ["game_id"])
    if not week_col or not game_col:
        raise ValueError(f"PBP {path.name} lacks week/game_id.")

    direct_team = first_existing(cols, ["pos_team_id", "possession_team_id", "offense_team_id"])
    pos_team = first_existing(cols, ["pos_team", "possession_team"])
    is_home = first_existing(cols, ["is_home", "pos_team_is_home", "possession_is_home"])
    home_id = first_existing(cols, ["homeTeamId", "home_team_id", "home_id"])
    away_id = first_existing(cols, ["awayTeamId", "away_team_id", "away_id"])

    needed = [c for c in [season_col, week_col, game_col, direct_team, pos_team, is_home, home_id, away_id,
                           "cpoe", "qbr_epa", "pass_oe", "xpass", "action_play", "pass_flag",
                           "passer_player_id", "weight"] if c and c in cols]
    needed = list(dict.fromkeys(needed))
    d = pd.read_parquet(path, columns=needed)
    if season_col is None:
        d["season"] = season
    elif season_col != "season":
        d["season"] = d[season_col]
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["week"] = pd.to_numeric(d[week_col], errors="coerce")
    d["game_id"] = d[game_col].map(norm_id)

    if direct_team:
        d["team_id"] = d[direct_team].map(norm_id)
    else:
        d["team_id"] = ""

    # If pos_team itself is numeric, it is an ESPN team id in the current SDV release.
    if (d["team_id"] == "").all() and pos_team:
        pnum = pd.to_numeric(d[pos_team], errors="coerce")
        if pnum.notna().mean() >= 0.90:
            d["team_id"] = pnum.map(norm_id)

    # Native local PBP reliably carries possession-side is_home plus home/away ids.
    if (d["team_id"] == "").mean() > 0.20 and is_home and home_id and away_id:
        ih = d[is_home]
        if not pd.api.types.is_bool_dtype(ih):
            ih = ih.astype(str).str.lower().map({"true": True, "1": True, "yes": True,
                                                 "false": False, "0": False, "no": False})
        h = d[home_id].map(norm_id)
        a = d[away_id].map(norm_id)
        derived = pd.Series(np.where(ih.fillna(False), h, a), index=d.index)
        d.loc[d["team_id"].eq(""), "team_id"] = derived[d["team_id"].eq("")]

    d = d[d["team_id"].ne("") & d["week"].notna()].copy()
    if d.empty:
        raise ValueError(f"Could not assign possession team ids in {path.name}.")

    for c in ["cpoe", "qbr_epa", "pass_oe", "xpass"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["season", "week", "game_id", "team_id", "cpoe", "qbr_epa", "pass_oe", "xpass"]]


def aggregate_team_week(pbp: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "week", "team_id"]
    rows = []
    for key, g in pbp.groupby(keys, sort=False):
        row = {"season": int(key[0]), "week": int(key[1]), "team_id": str(key[2])}
        for metric in ["cpoe", "qbr_epa", "pass_oe", "xpass"]:
            x = pd.to_numeric(g[metric], errors="coerce")
            x = x[np.isfinite(x)]
            row[f"{metric}_sum"] = float(x.sum()) if len(x) else 0.0
            row[f"{metric}_n"] = int(len(x))
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_metric(records: list[dict[str, Any]], metric: str) -> float:
    s = sum(float(r.get(f"{metric}_sum", 0.0)) for r in records)
    n = sum(int(r.get(f"{metric}_n", 0)) for r in records)
    return s / n if n > 0 else float("nan")


def build_signal_index(team_week: pd.DataFrame) -> dict[tuple[int, str], tuple[list[int], list[dict[str, Any]]]]:
    idx: dict[tuple[int, str], tuple[list[int], list[dict[str, Any]]]] = {}
    if team_week.empty:
        return idx
    for (season, team_id), g in team_week.groupby(["season", "team_id"], sort=False):
        g = g.sort_values("week")
        records = g.to_dict("records")
        idx[(int(season), str(team_id))] = ([int(r["week"]) for r in records], records)
    return idx


def snapshot(index: dict[tuple[int, str], tuple[list[int], list[dict[str, Any]]]],
             season: int, team_id: str, week: int) -> dict[str, float]:
    key = (int(season), str(team_id))
    records: list[dict[str, Any]] = []
    if key in index:
        weeks, recs = index[key]
        cut = bisect.bisect_left(weeks, int(week))
        records = recs[:cut]
    source_prior = False
    if not records:
        prev = (int(season) - 1, str(team_id))
        if prev in index:
            records = index[prev][1]
            source_prior = True

    out: dict[str, float] = {"used_prior_season": float(source_prior)}
    last3 = records[-3:] if records else []
    for metric in ["cpoe", "qbr_epa", "pass_oe", "xpass"]:
        out[f"{metric}_st"] = weighted_metric(records, metric) if records else float("nan")
        out[f"{metric}_last3"] = weighted_metric(last3, metric) if last3 else float("nan")
        out[f"{metric}_n"] = float(sum(int(r.get(f"{metric}_n", 0)) for r in records)) if records else 0.0
    return out


def attach_pregame_signals(games: pd.DataFrame, team_week: pd.DataFrame) -> pd.DataFrame:
    idx = build_signal_index(team_week)
    rows = []
    for r in games.itertuples(index=False):
        h = snapshot(idx, int(r.season), str(r.home_team_id), int(r.week))
        a = snapshot(idx, int(r.season), str(r.away_team_id), int(r.week))
        row = r._asdict()
        for k in sorted(set(h) | set(a)):
            hv = float(h.get(k, float("nan")))
            av = float(a.get(k, float("nan")))
            row[f"home_{k}"] = hv
            row[f"away_{k}"] = av
            row[f"diff_{k}"] = hv - av if math.isfinite(hv) and math.isfinite(av) else float("nan")
            row[f"sum_{k}"] = hv + av if math.isfinite(hv) and math.isfinite(av) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Models / CV
# ---------------------------------------------------------------------------


def ridge(alpha: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])


def mae(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(p, float))))


def select_residual(games: pd.DataFrame, features: list[str], target: str, market: str) -> dict[str, Any]:
    candidates = []
    for alpha in RIDGE_ALPHAS:
        fold_cache = []
        for test_season in CV_SEASONS:
            tr = games[(games["season"] < test_season) & (games["season"] >= 2021)].copy()
            te = games[games["season"] == test_season].copy()
            tr = tr[np.isfinite(pd.to_numeric(tr[target], errors="coerce")) & np.isfinite(pd.to_numeric(tr[market], errors="coerce"))]
            te = te[np.isfinite(pd.to_numeric(te[target], errors="coerce")) & np.isfinite(pd.to_numeric(te[market], errors="coerce"))]
            if len(tr) < 100 or len(te) < 100:
                continue
            model = ridge(alpha)
            y = tr[target].to_numpy(float) - tr[market].to_numpy(float)
            model.fit(tr[features], y)
            corr = model.predict(te[features])
            fold_cache.append((test_season, te[target].to_numpy(float), te[market].to_numpy(float), corr))
        for w in CORRECTION_WEIGHTS:
            improvements = []
            all_a, all_m, all_p = [], [], []
            for season, a, m, c in fold_cache:
                p = m + float(w) * c
                improvements.append(mae(a, m) - mae(a, p))
                all_a.append(a); all_m.append(m); all_p.append(p)
            if not all_a:
                continue
            aa = np.concatenate(all_a); mm = np.concatenate(all_m); pp = np.concatenate(all_p)
            candidates.append({
                "alpha": alpha,
                "weight": w,
                "cv_improvement": mae(aa, mm) - mae(aa, pp),
                "fold_wins": int(sum(x > 0 for x in improvements)),
                "folds": len(improvements),
            })
    if not candidates:
        raise ValueError("Residual CV produced no valid folds.")
    candidates.sort(key=lambda x: (x["cv_improvement"], x["fold_wins"], -x["weight"]), reverse=True)
    return candidates[0] | {"all_candidates": candidates}


def fit_final_residual(games: pd.DataFrame, features: list[str], target: str, market: str,
                       choice: dict[str, Any]) -> pd.Series:
    tr = games[(games["season"] < FINAL_SEASON) & (games["season"] >= 2021)].copy()
    te = games[games["season"] == FINAL_SEASON].copy()
    tr = tr[np.isfinite(pd.to_numeric(tr[target], errors="coerce")) & np.isfinite(pd.to_numeric(tr[market], errors="coerce"))]
    model = ridge(float(choice["alpha"]))
    model.fit(tr[features], tr[target].to_numpy(float) - tr[market].to_numpy(float))
    pred = te[market].to_numpy(float) + float(choice["weight"]) * model.predict(te[features])
    return pd.Series(pred, index=te.index)


def bootstrap_ci(delta: np.ndarray, n: int = BOOTSTRAP_N) -> tuple[float, float]:
    x = np.asarray(delta, float)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(RANDOM_STATE)
    vals = np.empty(n, dtype=float)
    for i in range(n):
        vals[i] = float(np.mean(x[rng.integers(0, len(x), len(x))]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def evaluate_predictions(df: pd.DataFrame, margin_col: str, total_col: str, label: str) -> dict[str, Any]:
    d = df.copy()
    needed = ["actual_margin", "actual_total", "actual_home", "actual_away", "market_margin", "market_total", margin_col, total_col]
    mask = np.ones(len(d), dtype=bool)
    for c in needed:
        mask &= np.isfinite(pd.to_numeric(d[c], errors="coerce").to_numpy(float))
    d = d.loc[mask].copy()
    if d.empty:
        raise ValueError(f"No evaluable rows for {label}")

    am = d["actual_margin"].to_numpy(float)
    at = d["actual_total"].to_numpy(float)
    mm = d["market_margin"].to_numpy(float)
    mt = d["market_total"].to_numpy(float)
    pm = d[margin_col].to_numpy(float)
    pt = d[total_col].to_numpy(float)

    mh = (mt + mm) / 2.0; ma = (mt - mm) / 2.0
    ph = (pt + pm) / 2.0; pa = (pt - pm) / 2.0
    ah = d["actual_home"].to_numpy(float); aa = d["actual_away"].to_numpy(float)

    market_margin_err = np.abs(am - mm); pred_margin_err = np.abs(am - pm)
    market_total_err = np.abs(at - mt); pred_total_err = np.abs(at - pt)
    market_score_err = (np.abs(ah - mh) + np.abs(aa - ma)) / 2.0
    pred_score_err = (np.abs(ah - ph) + np.abs(aa - pa)) / 2.0

    actual_home_win = am > 0
    market_correct = (mm > 0) == actual_home_win
    pred_correct = (pm > 0) == actual_home_win

    m_ci = bootstrap_ci(market_margin_err - pred_margin_err)
    t_ci = bootstrap_ci(market_total_err - pred_total_err)
    s_ci = bootstrap_ci(market_score_err - pred_score_err)
    w_ci = bootstrap_ci(pred_correct.astype(float) - market_correct.astype(float))

    return {
        "test": label,
        "n": len(d),
        "market_margin_mae": float(market_margin_err.mean()),
        "candidate_margin_mae": float(pred_margin_err.mean()),
        "margin_improvement": float((market_margin_err - pred_margin_err).mean()),
        "margin_ci_low": m_ci[0], "margin_ci_high": m_ci[1],
        "market_total_mae": float(market_total_err.mean()),
        "candidate_total_mae": float(pred_total_err.mean()),
        "total_improvement": float((market_total_err - pred_total_err).mean()),
        "total_ci_low": t_ci[0], "total_ci_high": t_ci[1],
        "market_team_score_mae": float(market_score_err.mean()),
        "candidate_team_score_mae": float(pred_score_err.mean()),
        "team_score_improvement": float((market_score_err - pred_score_err).mean()),
        "team_score_ci_low": s_ci[0], "team_score_ci_high": s_ci[1],
        "market_winner_accuracy": float(market_correct.mean()),
        "candidate_winner_accuracy": float(pred_correct.mean()),
        "winner_accuracy_delta": float(pred_correct.mean() - market_correct.mean()),
        "winner_ci_low": w_ci[0], "winner_ci_high": w_ci[1],
    }


def print_result(r: dict[str, Any]) -> None:
    print(f"\n{r['test']} — 2025 (n={r['n']})")
    print(f"  margin MAE: {r['candidate_margin_mae']:.4f} vs market {r['market_margin_mae']:.4f} | improvement {r['margin_improvement']:+.4f} | CI=({r['margin_ci_low']:+.4f},{r['margin_ci_high']:+.4f})")
    print(f"  total MAE:  {r['candidate_total_mae']:.4f} vs market {r['market_total_mae']:.4f} | improvement {r['total_improvement']:+.4f} | CI=({r['total_ci_low']:+.4f},{r['total_ci_high']:+.4f})")
    print(f"  score MAE:  {r['candidate_team_score_mae']:.4f} vs market {r['market_team_score_mae']:.4f} | improvement {r['team_score_improvement']:+.4f} | CI=({r['team_score_ci_low']:+.4f},{r['team_score_ci_high']:+.4f})")
    print(f"  winner:     {100*r['candidate_winner_accuracy']:.2f}% vs market {100*r['market_winner_accuracy']:.2f}% | delta {100*r['winner_accuracy_delta']:+.2f} pp")


# ---------------------------------------------------------------------------
# Test 1/2: prior-week QB execution and Pass OE
# ---------------------------------------------------------------------------


def run_signal_test(games: pd.DataFrame, feature_cols: list[str], label: str) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    margin_choice = select_residual(games, feature_cols, "actual_margin", "market_margin")
    total_choice = select_residual(games, feature_cols, "actual_total", "market_total")

    d = games.copy()
    d[f"pred_margin_{label}"] = np.nan
    d[f"pred_total_{label}"] = np.nan
    m = fit_final_residual(d, feature_cols, "actual_margin", "market_margin", margin_choice)
    t = fit_final_residual(d, feature_cols, "actual_total", "market_total", total_choice)
    d.loc[m.index, f"pred_margin_{label}"] = m
    d.loc[t.index, f"pred_total_{label}"] = t

    final = d[d["season"] == FINAL_SEASON].copy()
    res = evaluate_predictions(final, f"pred_margin_{label}", f"pred_total_{label}", label)
    res.update({
        "margin_cv_improvement": margin_choice["cv_improvement"],
        "margin_cv_fold_wins": margin_choice["fold_wins"],
        "margin_alpha": margin_choice["alpha"],
        "margin_weight": margin_choice["weight"],
        "total_cv_improvement": total_choice["cv_improvement"],
        "total_cv_fold_wins": total_choice["fold_wins"],
        "total_alpha": total_choice["alpha"],
        "total_weight": total_choice["weight"],
    })
    cv_rows = []
    for target_name, choice in [("margin", margin_choice), ("total", total_choice)]:
        for x in choice["all_candidates"]:
            cv_rows.append({"test": label, "target": target_name, **x})
    return res, final, cv_rows


# ---------------------------------------------------------------------------
# Test 3: recent-vs-long SDV ratings history
# ---------------------------------------------------------------------------

RATING_COLS = ["adj_off_epa", "adj_def_epa", "adj_st_epa", "adj_net", "fei_off", "fei_def", "fei_net", "games", "off_pace"]


def ensure_small_sdv(out: Path, dataset: str, season: int) -> Path:
    if dataset == "ratings":
        url = RATINGS_URL.format(season=season)
        name = f"cfb_ratings_weekly_{season}.parquet"
    elif dataset == "schedule":
        url = SCHEDULE_URL.format(season=season)
        name = f"cfb_schedules_{season}.parquet"
    else:
        raise ValueError(dataset)
    return download(url, out / "cache" / name)


def build_long_history_frame(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_games = []
    coverage = []
    for season in range(LONG_START, FINAL_SEASON + 1):
        rp = ensure_small_sdv(out, "ratings", season)
        sp = ensure_small_sdv(out, "schedule", season)
        ratings = pd.read_parquet(rp)
        sched = pd.read_parquet(sp)
        coverage.append({"source": "ratings_weekly", "season": season, "rows": len(ratings), "columns": len(ratings.columns)})
        coverage.append({"source": "schedule", "season": season, "rows": len(sched), "columns": len(sched.columns)})

        need_r = {"team_id", "through_week", "adj_net", "adj_off_epa", "adj_def_epa", "off_pace"}
        miss = need_r - set(ratings.columns)
        if miss:
            raise ValueError(f"ratings {season} missing {sorted(miss)}")
        ratings["team_id"] = ratings["team_id"].map(norm_id)
        ratings["through_week"] = pd.to_numeric(ratings["through_week"], errors="coerce")
        for c in RATING_COLS:
            if c not in ratings.columns:
                ratings[c] = np.nan
            ratings[c] = pd.to_numeric(ratings[c], errors="coerce")

        required_s = {"game_id", "week", "home_id", "away_id", "home_points", "away_points", "neutral_site"}
        miss = required_s - set(sched.columns)
        if miss:
            raise ValueError(f"schedule {season} missing {sorted(miss)}")
        s = sched.copy()
        if "completed" in s.columns:
            s = s[s["completed"].fillna(False)].copy()
        s = s[pd.to_numeric(s["home_points"], errors="coerce").notna() & pd.to_numeric(s["away_points"], errors="coerce").notna()].copy()
        s["week"] = pd.to_numeric(s["week"], errors="coerce")
        s = s[s["week"] >= 2].copy()
        s["game_id"] = s["game_id"].map(norm_id)
        s["home_id"] = s["home_id"].map(norm_id)
        s["away_id"] = s["away_id"].map(norm_id)
        s["asof"] = s["week"] - 1

        rkeep = ["team_id", "through_week"] + RATING_COLS
        hr = ratings[rkeep].rename(columns={"team_id": "home_id", "through_week": "asof", **{c: f"h_{c}" for c in RATING_COLS}})
        ar = ratings[rkeep].rename(columns={"team_id": "away_id", "through_week": "asof", **{c: f"a_{c}" for c in RATING_COLS}})
        g = s.merge(hr, on=["home_id", "asof"], how="inner").merge(ar, on=["away_id", "asof"], how="inner")
        g["season"] = season
        g["actual_margin"] = pd.to_numeric(g["home_points"], errors="coerce") - pd.to_numeric(g["away_points"], errors="coerce")
        g["actual_total"] = pd.to_numeric(g["home_points"], errors="coerce") + pd.to_numeric(g["away_points"], errors="coerce")
        g["neutral"] = g["neutral_site"].fillna(False).astype(float)
        for c in RATING_COLS:
            g[f"diff_{c}"] = g[f"h_{c}"] - g[f"a_{c}"]
            g[f"sum_{c}"] = g[f"h_{c}"] + g[f"a_{c}"]
        g["pace_product"] = g["h_off_pace"] * g["a_off_pace"]
        g["min_games"] = np.minimum(g["h_games"], g["a_games"])
        all_games.append(g)
        print(f"history {season}: {len(g)} rated completed games")
    return pd.concat(all_games, ignore_index=True), pd.DataFrame(coverage)


LONG_FEATURES = [
    "neutral", "min_games", "pace_product",
    "h_adj_net", "a_adj_net", "diff_adj_net",
    "h_adj_off_epa", "a_adj_off_epa", "diff_adj_off_epa", "sum_adj_off_epa",
    "h_adj_def_epa", "a_adj_def_epa", "diff_adj_def_epa", "sum_adj_def_epa",
    "h_adj_st_epa", "a_adj_st_epa", "diff_adj_st_epa",
    "h_fei_net", "a_fei_net", "diff_fei_net",
    "h_off_pace", "a_off_pace", "sum_off_pace",
]


def long_cv_predictions(hist: pd.DataFrame, market: pd.DataFrame, mode: str, target: str,
                        alpha: float) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    out = []
    for test_season in CV_SEASONS:
        if mode == "recent":
            tr = hist[(hist["season"] >= 2021) & (hist["season"] < test_season)].copy()
        else:
            tr = hist[(hist["season"] >= LONG_START) & (hist["season"] < test_season)].copy()
        te = hist[hist["season"] == test_season].copy()
        mk = market[market["season"] == test_season][["game_id", "market_margin", "market_total"]]
        te = te.merge(mk, on="game_id", how="inner")
        if len(tr) < 300 or len(te) < 100:
            continue
        model = ridge(alpha)
        model.fit(tr[LONG_FEATURES], tr[target].to_numpy(float))
        direct = model.predict(te[LONG_FEATURES])
        market_col = "market_margin" if target == "actual_margin" else "market_total"
        out.append((test_season, te[target].to_numpy(float), te[market_col].to_numpy(float), direct))
    return out


def select_long(hist: pd.DataFrame, market: pd.DataFrame, mode: str, target: str) -> dict[str, Any]:
    cand = []
    for alpha in RIDGE_ALPHAS:
        folds = long_cv_predictions(hist, market, mode, target, alpha)
        for w in DIRECT_BLEND_WEIGHTS:
            impr = []; aa=[]; mm=[]; pp=[]
            for season, a, m, direct in folds:
                p = m + float(w) * (direct - m)
                impr.append(mae(a, m) - mae(a, p))
                aa.append(a); mm.append(m); pp.append(p)
            if not aa:
                continue
            A=np.concatenate(aa); M=np.concatenate(mm); P=np.concatenate(pp)
            cand.append({"alpha": alpha, "weight": w,
                         "cv_improvement": mae(A,M)-mae(A,P),
                         "fold_wins": int(sum(x>0 for x in impr)), "folds": len(impr)})
    if not cand:
        raise ValueError(f"No long-history CV candidates for {mode}/{target}")
    cand.sort(key=lambda x:(x["cv_improvement"],x["fold_wins"],-x["weight"]), reverse=True)
    return cand[0] | {"all_candidates": cand}


def fit_long_final(hist: pd.DataFrame, market: pd.DataFrame, mode: str, target: str,
                   choice: dict[str, Any]) -> pd.DataFrame:
    if mode == "recent":
        tr = hist[(hist["season"] >= 2021) & (hist["season"] < FINAL_SEASON)].copy()
    else:
        tr = hist[(hist["season"] >= LONG_START) & (hist["season"] < FINAL_SEASON)].copy()
    te = hist[hist["season"] == FINAL_SEASON].copy()
    mkcols = ["game_id", "season", "week", "home_team_id", "away_team_id", "market_margin", "market_total",
              "actual_home", "actual_away", "actual_margin", "actual_total"]
    mk = market[market["season"] == FINAL_SEASON][mkcols].copy()
    te = te.merge(mk, on="game_id", how="inner", suffixes=("_sdv", ""))
    model = ridge(float(choice["alpha"]))
    model.fit(tr[LONG_FEATURES], tr[target].to_numpy(float))
    direct = model.predict(te[LONG_FEATURES])
    market_col = "market_margin" if target == "actual_margin" else "market_total"
    pred = te[market_col].to_numpy(float) + float(choice["weight"]) * (direct - te[market_col].to_numpy(float))
    return pd.DataFrame({"game_id": te["game_id"].values, "pred": pred})


def run_history_mode(hist: pd.DataFrame, market: pd.DataFrame, mode: str) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    mc = select_long(hist, market, mode, "actual_margin")
    tc = select_long(hist, market, mode, "actual_total")
    mp = fit_long_final(hist, market, mode, "actual_margin", mc).rename(columns={"pred": f"pred_margin_history_{mode}"})
    tp = fit_long_final(hist, market, mode, "actual_total", tc).rename(columns={"pred": f"pred_total_history_{mode}"})
    final = market[market["season"] == FINAL_SEASON].copy().merge(mp, on="game_id", how="inner").merge(tp, on="game_id", how="inner")
    label = f"history_{mode}"
    res = evaluate_predictions(final, f"pred_margin_history_{mode}", f"pred_total_history_{mode}", label)
    res.update({
        "margin_cv_improvement": mc["cv_improvement"], "margin_cv_fold_wins": mc["fold_wins"], "margin_alpha": mc["alpha"], "margin_weight": mc["weight"],
        "total_cv_improvement": tc["cv_improvement"], "total_cv_fold_wins": tc["fold_wins"], "total_alpha": tc["alpha"], "total_weight": tc["weight"],
    })
    cv = []
    for target_name, choice in [("margin", mc), ("total", tc)]:
        for x in choice["all_candidates"]:
            cv.append({"test": label, "target": target_name, **x})
    return res, final, cv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    root = root_dir()
    out = output_dir(root)
    print(f"SDV new-signal audit: {VERSION}")
    print("2022-2024 = selection only; 2025 = frozen verification")

    market = load_market_games(root)
    print(f"market games loaded: {len(market)}")

    team_weeks = []
    coverage = []
    print("\nLoading CPOE/QBR/Pass-OE PBP...")
    for season in BASE_SEASONS:
        src, source_type, cols = choose_pbp_source(root, out, season)
        pbp = load_signal_pbp(src, season, cols)
        tw = aggregate_team_week(pbp)
        team_weeks.append(tw)
        coverage.append({"source": f"pbp_{source_type}", "season": season, "rows": len(pbp), "columns": len(cols), "path": str(src)})
        print(f"  {season}: source={source_type} plays={len(pbp)} team-weeks={len(tw)}")
    team_week = pd.concat(team_weeks, ignore_index=True)
    signal_games = attach_pregame_signals(market, team_week)

    qb_features = [
        c for c in signal_games.columns
        if any(token in c for token in ["cpoe_st", "cpoe_last3", "qbr_epa_st", "qbr_epa_last3"])
        and c.startswith(("home_", "away_", "diff_", "sum_"))
    ]
    passoe_features = [
        c for c in signal_games.columns
        if any(token in c for token in ["pass_oe_st", "pass_oe_last3", "xpass_st", "xpass_last3"])
        and c.startswith(("home_", "away_", "diff_", "sum_"))
    ]
    if not qb_features or not passoe_features:
        raise ValueError("Failed to construct QB or Pass-OE features.")

    results = []
    predictions = []
    cv_rows = []

    print("\nTEST 1: QB execution (CPOE + QBR EPA)")
    qb_res, qb_pred, qb_cv = run_signal_test(signal_games, qb_features, "qb_execution")
    results.append(qb_res); predictions.append(qb_pred); cv_rows.extend(qb_cv)
    print(f"  frozen margin: alpha={qb_res['margin_alpha']} weight={qb_res['margin_weight']} CV={qb_res['margin_cv_improvement']:+.4f}")
    print(f"  frozen total:  alpha={qb_res['total_alpha']} weight={qb_res['total_weight']} CV={qb_res['total_cv_improvement']:+.4f}")
    print_result(qb_res)

    print("\nTEST 2: Pass OE / xPass")
    po_res, po_pred, po_cv = run_signal_test(signal_games, passoe_features, "pass_oe")
    results.append(po_res); predictions.append(po_pred); cv_rows.extend(po_cv)
    print(f"  frozen margin: alpha={po_res['margin_alpha']} weight={po_res['margin_weight']} CV={po_res['margin_cv_improvement']:+.4f}")
    print(f"  frozen total:  alpha={po_res['total_alpha']} weight={po_res['total_weight']} CV={po_res['total_cv_improvement']:+.4f}")
    print_result(po_res)

    print("\nTEST 3: training-history length (same ratings model, recent vs 2004+)")
    hist, hist_cov = build_long_history_frame(out)
    coverage.extend(hist_cov.to_dict("records"))
    recent_res, recent_pred, recent_cv = run_history_mode(hist, market, "recent")
    long_res, long_pred, long_cv = run_history_mode(hist, market, "long")
    results.extend([recent_res, long_res]); predictions.extend([recent_pred, long_pred]); cv_rows.extend(recent_cv + long_cv)
    print_result(recent_res)
    print_result(long_res)

    delta_long_vs_recent = {
        "margin_mae_delta_long_minus_recent": long_res["candidate_margin_mae"] - recent_res["candidate_margin_mae"],
        "total_mae_delta_long_minus_recent": long_res["candidate_total_mae"] - recent_res["candidate_total_mae"],
        "team_score_mae_delta_long_minus_recent": long_res["candidate_team_score_mae"] - recent_res["candidate_team_score_mae"],
        "winner_accuracy_delta_long_minus_recent": long_res["candidate_winner_accuracy"] - recent_res["candidate_winner_accuracy"],
    }
    print("\nLONG HISTORY VS RECENT-ONLY (negative MAE delta = long history better)")
    print(f"  margin MAE delta: {delta_long_vs_recent['margin_mae_delta_long_minus_recent']:+.4f}")
    print(f"  total MAE delta:  {delta_long_vs_recent['total_mae_delta_long_minus_recent']:+.4f}")
    print(f"  score MAE delta:  {delta_long_vs_recent['team_score_mae_delta_long_minus_recent']:+.4f}")
    print(f"  winner delta:     {100*delta_long_vs_recent['winner_accuracy_delta_long_minus_recent']:+.2f} pp")

    result_df = pd.DataFrame(results)
    result_df.to_csv(out / "verification_2025.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(out / "cv_candidates_pre2025.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "source_coverage.csv", index=False)

    # Merge prediction columns by game_id without duplicating baseline columns.
    pred_out = market[market["season"] == FINAL_SEASON].copy()
    for p in predictions:
        add = p[[c for c in p.columns if c == "game_id" or c.startswith("pred_")]].drop_duplicates("game_id")
        pred_out = pred_out.merge(add, on="game_id", how="left")
    pred_out.to_csv(out / "predictions_2025.csv", index=False)

    summary = {
        "version": VERSION,
        "selection_seasons": CV_SEASONS,
        "final_season": FINAL_SEASON,
        "tests": results,
        "long_vs_recent": delta_long_vs_recent,
        "qb_features": qb_features,
        "passoe_features": passoe_features,
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFINAL AUDIT TABLE")
    for r in results:
        confirmed = (
            r["team_score_improvement"] > 0
            and r["team_score_ci_low"] > 0
            and r["margin_improvement"] >= 0
            and r["total_improvement"] >= 0
        )
        print(f"  {r['test']}: {'CONFIRMED' if confirmed else 'NOT CONFIRMED'} | score {r['team_score_improvement']:+.4f} | margin {r['margin_improvement']:+.4f} | total {r['total_improvement']:+.4f} | winner {100*r['winner_accuracy_delta']:+.2f} pp")

    print(f"\nResults: {out / 'verification_2025.csv'}")
    print(f"Predictions: {out / 'predictions_2025.csv'}")
    print(f"Summary: {out / 'audit_summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)

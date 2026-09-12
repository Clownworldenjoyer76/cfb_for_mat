#!/usr/bin/env python3
"""
audit_market_movement_consensus_v1.py

Standalone CFB market-movement / multi-book consensus signal audit.

This is NOT another score model.

Purpose
-------
Test whether historical opening-to-closing movement, multi-book disagreement,
and SportsDataverse consensus lines contain stable information about errors in
the repo's historical sportsbook baseline.

Historical outcomes / repo baseline:
    docs/win/football/cfb/data/score_model_v4/
        training_team_rows_2021_2025_v4.parquet

External pregame market history (downloaded once and cached locally):
    https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/
        betting/parquet/cfb_line_odds.parquet

SportsDataverse odds columns used:
    game_id, season, week, game_desc, market_type, abbr,
    lines, opening_lines, odds, opening_odds, book

Validation policy
-----------------
* 2021 is available as initial training history.
* 2022, 2023, 2024 are chronological forward validation years.
* Features/corrections are ranked and frozen without using 2025.
* 2025 is checked only after the shortlist is frozen.
* Tiny gains are NOT called useful. A strict pre-2025 feature signal requires:
    - at least +0.10 MAE improvement vs the raw sportsbook baseline,
    - positive incremental improvement vs a train-only bias correction,
    - improvement in all 3 forward validation seasons,
    - consistent fitted direction.
* Strict 2025 confirmation additionally requires:
    - at least +0.10 MAE improvement,
    - positive improvement vs the train-only bias correction,
    - bootstrap 95% lower bound above zero vs the raw sportsbook baseline.

Primary targets
---------------
    margin_error = actual home margin - repo sportsbook implied home margin
    total_error  = actual total - repo sportsbook total

Outputs only
------------
    docs/win/football/cfb/data/market_movement_consensus_audit_v1/
        cache/cfb_line_odds.parquet
        odds_coverage.csv
        market_features_2021_2025.csv
        direct_line_comparison_by_season.csv
        direct_candidate_cv_2022_2024.csv
        direct_candidate_frozen_pre2025.csv
        direct_candidate_verification_2025.csv
        feature_cv_2022_2024.csv
        shortlist_frozen_pre2025.csv
        verification_2025.csv
        audit_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_SEASONS = [2021, 2022, 2023, 2024]
CV_YEARS = [2022, 2023, 2024]
FINAL_YEAR = 2025
TARGETS = ["margin_error", "total_error"]

MIN_FOLD_ROWS = 100
MIN_PRE2025_ROWS = 300
MIN_MEANINGFUL_MAE = 0.10
MIN_INCREMENTAL_OVER_BIAS = 0.0
TOP_PER_TARGET = 10
BOOTSTRAP_REPS = 5000
RNG_SEED = 20260829

LINEAR_RIDGE_ALPHA = 100.0
BIN_COUNT = 5
BIN_SHRINK_N = 150.0

ALPHA_GRID = [0.25, 0.50, 0.75, 1.00]

ODDS_URL = (
    "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/"
    "betting/parquet/cfb_line_odds.parquet"
)

BOOK_ALIASES = {
    "draftkings": ("draftkings", "draft king"),
    "espnbet": ("espnbet", "espn bet"),
    "fanduel": ("fanduel", "fan duel"),
    "caesars": ("caesars", "williamhill", "william hill"),
    "betmgm": ("betmgm", "bet mgm", "mgm"),
}

ID_COLUMNS = {
    "game_id", "season", "week", "game_desc",
    "away_name", "home_name",
}


# ---------------------------------------------------------------------------
# CLI / filesystem
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit CFB opening/closing market movement and multi-book consensus against sportsbook errors."
    )
    p.add_argument(
        "--cfb-root",
        type=str,
        default=None,
        help="Optional docs/win/football/cfb path. Auto-detected from script location by default.",
    )
    p.add_argument(
        "--refresh-odds",
        action="store_true",
        help="Redownload the SportsDataverse historical line file.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=TOP_PER_TARGET,
        help=f"Frozen feature shortlist size per target. Default: {TOP_PER_TARGET}.",
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=BOOTSTRAP_REPS,
        help=f"2025 bootstrap repetitions. Default: {BOOTSTRAP_REPS}.",
    )
    return p.parse_args()


def resolve_cfb_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    # Intended path:
    # docs/win/football/cfb/scripts/01_merge/audit_market_movement_consensus_v1.py
    return Path(__file__).resolve().parents[2]


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "<na>"} else text


def norm_id(value: Any) -> str:
    text = clean_text(value)
    return re.sub(r"\.0$", "", text)


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def sign_int(value: float, tol: float = 1e-12) -> int:
    try:
        v = float(value)
    except Exception:
        return 0
    if not math.isfinite(v) or abs(v) <= tol:
        return 0
    return 1 if v > 0 else -1


def book_key(value: Any) -> str:
    raw = clean_text(value).casefold()
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    for key, aliases in BOOK_ALIASES.items():
        for alias in aliases:
            if re.sub(r"[^a-z0-9]+", "", alias.casefold()) in compact:
                return key
    return compact[:40] if compact else "unknown"


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - pred[mask])))


def mean_error(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(actual[mask] - pred[mask]))


def pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(float)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(float)
    return pearson_safe(xr, yr)


# ---------------------------------------------------------------------------
# Download + load SportsDataverse historical line file
# ---------------------------------------------------------------------------


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".download")
    if tmp.exists():
        tmp.unlink()

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 CFB-market-consensus-audit/1.0",
            "Accept": "application/octet-stream,*/*",
        },
    )

    with urllib.request.urlopen(req, timeout=90, context=ssl_context()) as response:
        with tmp.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    if not tmp.exists() or tmp.stat().st_size < 100_000:
        raise RuntimeError(f"Historical odds download looks incomplete: {tmp}")

    os.replace(tmp, path)


def ensure_odds_file(cache_path: Path, refresh: bool) -> None:
    if cache_path.exists() and cache_path.stat().st_size >= 100_000 and not refresh:
        return
    print("Downloading SportsDataverse multi-book historical odds...")
    print(f"  {ODDS_URL}")
    download_file(ODDS_URL, cache_path)
    print(f"  cached: {cache_path} ({cache_path.stat().st_size / 1_000_000:.2f} MB)")


def load_raw_odds(cache_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(cache_path)
    required = {
        "game_id", "season", "game_desc", "market_type", "abbr",
        "lines", "opening_lines", "odds", "opening_odds", "book",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "SportsDataverse cfb_line_odds schema changed; missing columns: "
            + ", ".join(missing)
        )

    df = df.copy()
    df["game_id"] = df["game_id"].map(norm_id)
    df["season"] = safe_num(df["season"]).astype("Int64")
    if "week" in df.columns:
        df["week"] = safe_num(df["week"]).astype("Int64")
    else:
        df["week"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df["market_type"] = df["market_type"].map(clean_text).str.casefold()
    df["abbr"] = df["abbr"].map(clean_text)
    df["book"] = df["book"].map(clean_text)
    df["book_key"] = df["book"].map(book_key)
    for c in ["lines", "opening_lines", "odds", "opening_odds"]:
        df[c] = safe_num(df[c])

    df = df[
        df["season"].isin(TRAIN_SEASONS + [FINAL_YEAR])
        & df["game_id"].ne("")
    ].copy()
    return df


# ---------------------------------------------------------------------------
# Resolve home side and aggregate book-level / consensus features
# ---------------------------------------------------------------------------


def split_game_desc(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = series.fillna("").astype(str).str.split("@", n=1, expand=True)
    away = parts[0].str.strip() if parts.shape[1] >= 1 else pd.Series("", index=series.index)
    home = parts[1].str.strip() if parts.shape[1] >= 2 else pd.Series("", index=series.index)
    return away, home


def infer_abbr_team_map(spread: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    """
    Infer odds abbreviation -> team name from game_desc co-occurrences.

    For a given abbreviation, its own team appears in every game carrying that
    abbreviation while opponents vary. We only keep unambiguous modal mappings.
    """
    work = spread[["abbr", "game_desc"]].dropna().copy()
    away, home = split_game_desc(work["game_desc"])
    work["away_name"] = away
    work["home_name"] = home

    stacked = pd.concat(
        [
            work[["abbr", "away_name"]].rename(columns={"away_name": "candidate_name"}),
            work[["abbr", "home_name"]].rename(columns={"home_name": "candidate_name"}),
        ],
        ignore_index=True,
    )
    stacked = stacked[
        stacked["abbr"].map(clean_text).ne("")
        & stacked["candidate_name"].map(clean_text).ne("")
    ].copy()

    counts = (
        stacked.groupby(["abbr", "candidate_name"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["abbr", "count", "candidate_name"], ascending=[True, False, True])
    )

    audit_rows: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for abbr, part in counts.groupby("abbr", sort=False):
        part = part.reset_index(drop=True)
        best = int(part.loc[0, "count"])
        second = int(part.loc[1, "count"]) if len(part) > 1 else 0
        name = clean_text(part.loc[0, "candidate_name"])
        unambiguous = best > second
        if unambiguous and name:
            mapping[str(abbr)] = name
        audit_rows.append(
            {
                "abbr": abbr,
                "mapped_team": name,
                "best_count": best,
                "second_count": second,
                "unambiguous": unambiguous,
            }
        )

    return mapping, pd.DataFrame(audit_rows)


def q25(s: pd.Series) -> float:
    return float(s.quantile(0.25)) if s.notna().any() else float("nan")


def q75(s: pd.Series) -> float:
    return float(s.quantile(0.75)) if s.notna().any() else float("nan")


def nan_range(s: pd.Series) -> float:
    x = safe_num(s).dropna()
    return float(x.max() - x.min()) if len(x) else float("nan")


def direction_disagreement(s: pd.Series) -> float:
    x = safe_num(s).dropna().to_numpy(float)
    x = x[np.abs(x) > 1e-12]
    if len(x) < 2:
        return 0.0 if len(x) == 1 else float("nan")
    p = float(np.mean(x > 0))
    return min(p, 1.0 - p)


def aggregate_series_frame(
    book_rows: pd.DataFrame,
    value_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    if book_rows.empty:
        return pd.DataFrame(columns=["game_id", "season"])

    rows: list[dict[str, Any]] = []
    for (gid, season), part in book_rows.groupby(["game_id", "season"], dropna=False):
        row: dict[str, Any] = {
            "game_id": gid,
            "season": season,
            f"{prefix}_book_count": int(part["book_key"].nunique()),
        }
        for col in value_cols:
            x = safe_num(part[col]).dropna()
            name = f"{prefix}_{col}"
            row[f"{name}_median"] = float(x.median()) if len(x) else np.nan
            row[f"{name}_mean"] = float(x.mean()) if len(x) else np.nan
            row[f"{name}_std"] = float(x.std(ddof=0)) if len(x) > 1 else (0.0 if len(x) == 1 else np.nan)
            row[f"{name}_range"] = float(x.max() - x.min()) if len(x) else np.nan
            row[f"{name}_q25"] = float(x.quantile(0.25)) if len(x) else np.nan
            row[f"{name}_q75"] = float(x.quantile(0.75)) if len(x) else np.nan
            row[f"{name}_iqr"] = (
                row[f"{name}_q75"] - row[f"{name}_q25"]
                if math.isfinite(row[f"{name}_q75"]) and math.isfinite(row[f"{name}_q25"])
                else np.nan
            )
            row[f"{name}_direction_disagreement"] = direction_disagreement(x)
        rows.append(row)
    return pd.DataFrame(rows)


def build_market_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = raw.copy()
    raw["game_id"] = raw["game_id"].map(norm_id)
    raw["season"] = safe_num(raw["season"]).astype("Int64")
    if "book_key" not in raw.columns:
        raw["book_key"] = raw["book"].map(book_key)

    spread = raw[
        raw["market_type"].eq("spread")
        & raw["game_desc"].astype(str).str.contains("@", regex=False, na=False)
    ].copy()

    if spread.empty:
        raise RuntimeError("SportsDataverse odds file contains no spread rows for 2021-2025.")

    abbr_map, abbr_audit = infer_abbr_team_map(spread)

    away, home = split_game_desc(spread["game_desc"])
    spread["away_name"] = away
    spread["home_name"] = home
    spread["mapped_team"] = spread["abbr"].map(abbr_map)
    spread["is_home"] = spread["mapped_team"].eq(spread["home_name"])

    home_spread = spread[spread["is_home"]].copy()
    home_spread["close_home_margin"] = -home_spread["lines"]
    home_spread["open_home_margin"] = -home_spread["opening_lines"]
    home_spread["home_margin_move"] = home_spread["close_home_margin"] - home_spread["open_home_margin"]

    # Collapse duplicate side rows inside a book before measuring book disagreement.
    spread_by_book = (
        home_spread.groupby(["game_id", "season", "week", "game_desc", "book_key"], dropna=False)
        .agg(
            close_home_margin=("close_home_margin", "median"),
            open_home_margin=("open_home_margin", "median"),
            home_margin_move=("home_margin_move", "median"),
            close_home_spread_odds=("odds", "median"),
            open_home_spread_odds=("opening_odds", "median"),
        )
        .reset_index()
    )

    totals = raw[raw["market_type"].eq("total")].copy()
    totals["close_total"] = totals["lines"]
    totals["open_total"] = totals["opening_lines"]
    totals["total_move"] = totals["close_total"] - totals["open_total"]

    # Over/under rows can duplicate the same numeric total. Median within book removes that duplication.
    total_by_book = (
        totals.groupby(["game_id", "season", "week", "game_desc", "book_key"], dropna=False)
        .agg(
            close_total=("close_total", "median"),
            open_total=("open_total", "median"),
            total_move=("total_move", "median"),
        )
        .reset_index()
    )

    spread_cons = aggregate_series_frame(
        spread_by_book,
        ["close_home_margin", "open_home_margin", "home_margin_move"],
        "spread",
    )
    total_cons = aggregate_series_frame(
        total_by_book,
        ["close_total", "open_total", "total_move"],
        "total",
    )

    # Canonical short names used by the audit.
    if not spread_cons.empty:
        spread_cons = spread_cons.rename(
            columns={
                "spread_close_home_margin_median": "consensus_close_home_margin",
                "spread_open_home_margin_median": "consensus_open_home_margin",
                "spread_home_margin_move_median": "consensus_home_margin_move",
                "spread_close_home_margin_std": "spread_close_book_std",
                "spread_close_home_margin_range": "spread_close_book_range",
                "spread_close_home_margin_iqr": "spread_close_book_iqr",
                "spread_open_home_margin_std": "spread_open_book_std",
                "spread_open_home_margin_range": "spread_open_book_range",
                "spread_open_home_margin_iqr": "spread_open_book_iqr",
                "spread_home_margin_move_std": "spread_move_book_std",
                "spread_home_margin_move_range": "spread_move_book_range",
                "spread_home_margin_move_iqr": "spread_move_book_iqr",
                "spread_close_home_margin_direction_disagreement": "spread_close_direction_disagreement",
                "spread_home_margin_move_direction_disagreement": "spread_move_direction_disagreement",
            }
        )

    if not total_cons.empty:
        total_cons = total_cons.rename(
            columns={
                "total_close_total_median": "consensus_close_total",
                "total_open_total_median": "consensus_open_total",
                "total_total_move_median": "consensus_total_move",
                "total_close_total_std": "total_close_book_std",
                "total_close_total_range": "total_close_book_range",
                "total_close_total_iqr": "total_close_book_iqr",
                "total_open_total_std": "total_open_book_std",
                "total_open_total_range": "total_open_book_range",
                "total_open_total_iqr": "total_open_book_iqr",
                "total_total_move_std": "total_move_book_std",
                "total_total_move_range": "total_move_book_range",
                "total_total_move_iqr": "total_move_book_iqr",
                "total_close_total_direction_disagreement": "total_close_direction_disagreement",
                "total_total_move_direction_disagreement": "total_move_direction_disagreement",
            }
        )

    game = spread_cons.merge(total_cons, on=["game_id", "season"], how="outer", suffixes=("", "_totaldup"))

    # Add game descriptions / week from whichever book table is available.
    identity = pd.concat(
        [
            spread_by_book[["game_id", "season", "week", "game_desc"]],
            total_by_book[["game_id", "season", "week", "game_desc"]],
        ],
        ignore_index=True,
    )
    identity = identity.drop_duplicates(["game_id", "season"], keep="first")
    game = game.merge(identity, on=["game_id", "season"], how="left")

    # Recognized individual books: direct close/open line and their distance from consensus.
    for key in BOOK_ALIASES:
        sb = spread_by_book[spread_by_book["book_key"].eq(key)][
            ["game_id", "season", "close_home_margin", "open_home_margin", "home_margin_move"]
        ].copy()
        sb = sb.groupby(["game_id", "season"], as_index=False).median(numeric_only=True)
        sb = sb.rename(
            columns={
                "close_home_margin": f"{key}_close_home_margin",
                "open_home_margin": f"{key}_open_home_margin",
                "home_margin_move": f"{key}_home_margin_move",
            }
        )
        game = game.merge(sb, on=["game_id", "season"], how="left")

        tb = total_by_book[total_by_book["book_key"].eq(key)][
            ["game_id", "season", "close_total", "open_total", "total_move"]
        ].copy()
        tb = tb.groupby(["game_id", "season"], as_index=False).median(numeric_only=True)
        tb = tb.rename(
            columns={
                "close_total": f"{key}_close_total",
                "open_total": f"{key}_open_total",
                "total_move": f"{key}_total_move",
            }
        )
        game = game.merge(tb, on=["game_id", "season"], how="left")

    for c in game.columns:
        if c not in ID_COLUMNS and c not in {"week_totaldup", "game_desc_totaldup"}:
            if game[c].dtype.kind not in "biufc":
                continue
            game[c] = safe_num(game[c])

    # Coverage table by season / market and top book counts.
    cov_rows: list[dict[str, Any]] = []
    for season in sorted(raw["season"].dropna().astype(int).unique()):
        part = raw[raw["season"].eq(season)]
        for market in ["spread", "total", "moneyline"]:
            p = part[part["market_type"].eq(market)]
            cov_rows.append(
                {
                    "season": season,
                    "market_type": market,
                    "rows": int(len(p)),
                    "games": int(p["game_id"].nunique()),
                    "books": int(p["book_key"].nunique()),
                    "opening_line_nonnull": int(p["opening_lines"].notna().sum()),
                    "closing_line_nonnull": int(p["lines"].notna().sum()),
                }
            )
    coverage = pd.DataFrame(cov_rows)

    return game, coverage, abbr_audit


# ---------------------------------------------------------------------------
# Load repo historical baseline / outcomes
# ---------------------------------------------------------------------------


def load_repo_games(cfb_root: Path) -> pd.DataFrame:
    path = cfb_root / "data" / "score_model_v4" / "training_team_rows_2021_2025_v4.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required V4 training matrix: {path}\nRun train_score_model_v4.py first."
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

    home = df[df["team_side"].astype(str).str.casefold().eq("home")].copy()
    if home.empty:
        raise RuntimeError("V4 training matrix contains no home-side rows.")

    home["season"] = safe_num(home["season"]).astype("Int64")
    home["week"] = safe_num(home["week"]).astype("Int64")
    home["game_id"] = home["game_id"].map(norm_id)
    for c in ["market_team_score", "market_opp_score", "actual_team_score", "actual_opp_score"]:
        home[c] = safe_num(home[c])

    home["actual_home_score"] = home["actual_team_score"]
    home["actual_away_score"] = home["actual_opp_score"]
    home["actual_home_margin"] = home["actual_home_score"] - home["actual_away_score"]
    home["actual_total"] = home["actual_home_score"] + home["actual_away_score"]

    home["repo_market_home_margin"] = home["market_team_score"] - home["market_opp_score"]
    home["repo_market_total"] = home["market_team_score"] + home["market_opp_score"]
    home["margin_error"] = home["actual_home_margin"] - home["repo_market_home_margin"]
    home["total_error"] = home["actual_total"] - home["repo_market_total"]

    keep = [
        "season", "week", "game_id",
        "actual_home_score", "actual_away_score", "actual_home_margin", "actual_total",
        "repo_market_home_margin", "repo_market_total", "margin_error", "total_error",
    ]
    for c in ["game_date", "home_team", "away_team", "home_team_id", "away_team_id"]:
        if c in home.columns:
            keep.append(c)

    return home[keep].drop_duplicates(["season", "game_id"], keep="last").copy()


def merge_market_repo(repo: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    merged = repo.merge(market, on=["season", "game_id"], how="left", suffixes=("", "_odds"))

    # Core disagreement / movement features relative to the repo's chosen sportsbook line.
    merged["consensus_close_minus_repo_margin"] = (
        safe_num(merged.get("consensus_close_home_margin", pd.Series(index=merged.index, dtype=float)))
        - merged["repo_market_home_margin"]
    )
    merged["consensus_open_minus_repo_margin"] = (
        safe_num(merged.get("consensus_open_home_margin", pd.Series(index=merged.index, dtype=float)))
        - merged["repo_market_home_margin"]
    )
    merged["consensus_close_minus_repo_total"] = (
        safe_num(merged.get("consensus_close_total", pd.Series(index=merged.index, dtype=float)))
        - merged["repo_market_total"]
    )
    merged["consensus_open_minus_repo_total"] = (
        safe_num(merged.get("consensus_open_total", pd.Series(index=merged.index, dtype=float)))
        - merged["repo_market_total"]
    )

    for c in [
        "consensus_close_minus_repo_margin", "consensus_open_minus_repo_margin",
        "consensus_close_minus_repo_total", "consensus_open_minus_repo_total",
        "consensus_home_margin_move", "consensus_total_move",
    ]:
        if c in merged.columns:
            merged[f"abs_{c}"] = safe_num(merged[c]).abs()

    # Recognized book vs consensus disagreement.
    for key in BOOK_ALIASES:
        if f"{key}_close_home_margin" in merged.columns:
            merged[f"{key}_minus_consensus_margin"] = (
                safe_num(merged[f"{key}_close_home_margin"])
                - safe_num(merged.get("consensus_close_home_margin", pd.Series(index=merged.index, dtype=float)))
            )
        if f"{key}_close_total" in merged.columns:
            merged[f"{key}_minus_consensus_total"] = (
                safe_num(merged[f"{key}_close_total"])
                - safe_num(merged.get("consensus_close_total", pd.Series(index=merged.index, dtype=float)))
            )

    return merged


# ---------------------------------------------------------------------------
# Direct line-source comparisons
# ---------------------------------------------------------------------------


def direct_line_comparison_by_season(games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = [
        ("repo_market", "repo_market_home_margin", "repo_market_total"),
        ("consensus_close", "consensus_close_home_margin", "consensus_close_total"),
        ("consensus_open", "consensus_open_home_margin", "consensus_open_total"),
    ]
    for season in TRAIN_SEASONS + [FINAL_YEAR]:
        part = games[games["season"].eq(season)]
        for source, mcol, tcol in sources:
            if mcol not in part.columns or tcol not in part.columns:
                continue
            for target, actual_col, pred_col in [
                ("margin", "actual_home_margin", mcol),
                ("total", "actual_total", tcol),
            ]:
                w = part[[actual_col, pred_col]].dropna()
                if w.empty:
                    continue
                a = w[actual_col].to_numpy(float)
                p = w[pred_col].to_numpy(float)
                rows.append(
                    {
                        "season": season,
                        "source": source,
                        "target": target,
                        "games": len(w),
                        "mae": mae(a, p),
                        "mean_error_actual_minus_pred": mean_error(a, p),
                    }
                )
    return pd.DataFrame(rows)


def candidate_specs() -> list[dict[str, str]]:
    return [
        {
            "target": "margin_error",
            "name": "blend_consensus_close_margin",
            "delta_col": "consensus_close_minus_repo_margin",
        },
        {
            "target": "margin_error",
            "name": "blend_consensus_open_margin",
            "delta_col": "consensus_open_minus_repo_margin",
        },
        {
            "target": "margin_error",
            "name": "follow_open_to_close_margin_move",
            "delta_col": "consensus_home_margin_move",
        },
        {
            "target": "total_error",
            "name": "blend_consensus_close_total",
            "delta_col": "consensus_close_minus_repo_total",
        },
        {
            "target": "total_error",
            "name": "blend_consensus_open_total",
            "delta_col": "consensus_open_minus_repo_total",
        },
        {
            "target": "total_error",
            "name": "follow_open_to_close_total_move",
            "delta_col": "consensus_total_move",
        },
    ]


def evaluate_direct_candidate(games: pd.DataFrame, spec: dict[str, str], alpha: float) -> dict[str, Any] | None:
    target = spec["target"]
    delta_col = spec["delta_col"]
    if delta_col not in games.columns:
        return None

    folds: list[dict[str, Any]] = []
    pooled_y: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    for year in CV_YEARS:
        va = games[games["season"].eq(year)][[target, delta_col]].dropna()
        if len(va) < MIN_FOLD_ROWS:
            continue
        y = va[target].to_numpy(float)
        pred = alpha * va[delta_col].to_numpy(float)
        base = float(np.mean(np.abs(y)))
        cand = float(np.mean(np.abs(y - pred)))
        folds.append(
            {
                "year": year,
                "n": len(y),
                "base_mae": base,
                "candidate_mae": cand,
                "improvement": base - cand,
            }
        )
        pooled_y.append(y)
        pooled_pred.append(pred)

    if len(folds) < 2:
        return None

    y = np.concatenate(pooled_y)
    pred = np.concatenate(pooled_pred)
    out: dict[str, Any] = {
        "target": target,
        "candidate": spec["name"],
        "delta_col": delta_col,
        "alpha": alpha,
        "cv_folds": len(folds),
        "cv_rows": len(y),
        "cv_market_mae": float(np.mean(np.abs(y))),
        "cv_candidate_mae": float(np.mean(np.abs(y - pred))),
        "cv_improvement": float(np.mean(np.abs(y)) - np.mean(np.abs(y - pred))),
        "fold_wins": sum(1 for f in folds if f["improvement"] > 0),
        "worst_fold_improvement": min(f["improvement"] for f in folds),
    }
    for f in folds:
        out[f"improvement_{f['year']}"] = f["improvement"]
    out["strict_pre2025"] = bool(
        out["cv_improvement"] >= MIN_MEANINGFUL_MAE
        and out["fold_wins"] == len(folds)
        and len(folds) == len(CV_YEARS)
    )
    return out


def direct_candidate_cv(games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in candidate_specs():
        for alpha in ALPHA_GRID:
            r = evaluate_direct_candidate(games, spec, alpha)
            if r is not None:
                rows.append(r)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["target", "strict_pre2025", "cv_improvement", "fold_wins"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
    return out


def freeze_direct_candidates(cv: pd.DataFrame) -> pd.DataFrame:
    if cv.empty:
        return cv.copy()
    # Alpha chosen using pre-2025 CV only. Keep one alpha per named candidate,
    # then the strongest two candidates per target.
    selected = (
        cv.sort_values(["candidate", "cv_improvement"], ascending=[True, False])
        .drop_duplicates("candidate", keep="first")
        .sort_values(["target", "strict_pre2025", "cv_improvement"], ascending=[True, False, False])
        .groupby("target", as_index=False, group_keys=False)
        .head(2)
        .copy()
    )
    selected["pre2025_rank"] = selected.groupby("target").cumcount() + 1
    return selected.reset_index(drop=True)


def bootstrap_vs_market(y: np.ndarray, pred: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    diff = np.abs(y) - np.abs(y - pred)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 20 or reps <= 0:
        return np.nan, np.nan
    n = len(diff)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(diff[idx]))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def verify_direct_candidates(frozen: pd.DataFrame, games: pd.DataFrame, reps: int) -> pd.DataFrame:
    if frozen.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(RNG_SEED + 1)
    rows: list[dict[str, Any]] = []
    for _, s in frozen.iterrows():
        target = str(s["target"])
        delta_col = str(s["delta_col"])
        alpha = float(s["alpha"])
        va = games[games["season"].eq(FINAL_YEAR)][[target, delta_col]].dropna()
        if len(va) < MIN_FOLD_ROWS:
            continue
        y = va[target].to_numpy(float)
        pred = alpha * va[delta_col].to_numpy(float)
        base = float(np.mean(np.abs(y)))
        cand = float(np.mean(np.abs(y - pred)))
        imp = base - cand
        lo, hi = bootstrap_vs_market(y, pred, reps, rng)
        rows.append(
            {
                "target": target,
                "pre2025_rank": int(s["pre2025_rank"]),
                "candidate": s["candidate"],
                "delta_col": delta_col,
                "alpha": alpha,
                "strict_pre2025": bool(s["strict_pre2025"]),
                "cv_improvement": float(s["cv_improvement"]),
                "n_2025": len(y),
                "market_mae_2025": base,
                "candidate_mae_2025": cand,
                "improvement_2025": imp,
                "bootstrap_low_2025": lo,
                "bootstrap_high_2025": hi,
                "confirmed_strict": bool(
                    bool(s["strict_pre2025"])
                    and imp >= MIN_MEANINGFUL_MAE
                    and math.isfinite(lo)
                    and lo > 0
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simple one-feature residual corrections
# ---------------------------------------------------------------------------


@dataclass
class LinearFit:
    mean_x: float
    sd_x: float
    bias: float
    coef: float


def fit_linear(x: np.ndarray, y: np.ndarray) -> LinearFit:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(y) == 0:
        return LinearFit(0.0, 1.0, 0.0, 0.0)
    mx = float(np.mean(x))
    sx = float(np.std(x))
    bias = float(np.mean(y))
    if not math.isfinite(sx) or sx < 1e-12:
        return LinearFit(mx, 1.0, bias, 0.0)
    z = (x - mx) / sx
    yc = y - bias
    coef = float(np.sum(z * yc) / (np.sum(z * z) + LINEAR_RIDGE_ALPHA))
    return LinearFit(mx, sx, bias, coef)


def predict_linear(fit: LinearFit, x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), fit.bias, dtype=float)
    mask = np.isfinite(x)
    out[mask] = fit.bias + fit.coef * ((x[mask] - fit.mean_x) / fit.sd_x)
    return out


@dataclass
class BinFit:
    edges: np.ndarray
    values: np.ndarray
    bias: float


def fit_bins(x: np.ndarray, y: np.ndarray) -> BinFit:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    bias = float(np.mean(y)) if len(y) else 0.0
    if len(x) < 50 or len(np.unique(x)) < 4:
        return BinFit(np.array([-np.inf, np.inf]), np.array([bias]), bias)
    raw = np.quantile(x, np.linspace(0, 1, BIN_COUNT + 1))
    edges = np.unique(raw)
    if len(edges) < 3:
        return BinFit(np.array([-np.inf, np.inf]), np.array([bias]), bias)
    edges[0] = -np.inf
    edges[-1] = np.inf
    idx = np.digitize(x, edges[1:-1], right=True)
    vals = np.zeros(len(edges) - 1, dtype=float)
    for b in range(len(vals)):
        yy = y[idx == b]
        if len(yy) == 0:
            vals[b] = bias
        else:
            shrink = len(yy) / (len(yy) + BIN_SHRINK_N)
            vals[b] = shrink * float(np.mean(yy)) + (1.0 - shrink) * bias
    return BinFit(edges, vals, bias)


def predict_bins(fit: BinFit, x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), fit.bias, dtype=float)
    mask = np.isfinite(x)
    if mask.any():
        idx = np.digitize(x[mask], fit.edges[1:-1], right=True)
        idx = np.clip(idx, 0, len(fit.values) - 1)
        out[mask] = fit.values[idx]
    return out


def fit_method(method: str, x: np.ndarray, y: np.ndarray) -> Any:
    return fit_linear(x, y) if method == "linear" else fit_bins(x, y)


def predict_method(method: str, fit: Any, x: np.ndarray) -> np.ndarray:
    return predict_linear(fit, x) if method == "linear" else predict_bins(fit, x)


def fit_direction(method: str, fit: Any) -> int:
    if method == "linear":
        return sign_int(fit.coef)
    if len(fit.values) < 2:
        return 0
    return sign_int(float(fit.values[-1] - fit.values[0]))


def market_feature_columns(games: pd.DataFrame) -> list[str]:
    blocked = {
        "actual_home_score", "actual_away_score", "actual_home_margin", "actual_total",
        "repo_market_home_margin", "repo_market_total", "margin_error", "total_error",
    }
    prefixes = (
        "consensus_", "spread_", "total_", "abs_consensus_",
        "draftkings_", "espnbet_", "fanduel_", "caesars_", "betmgm_",
    )
    cols: list[str] = []
    for c in games.columns:
        if c in blocked or c in ID_COLUMNS:
            continue
        if not c.startswith(prefixes):
            continue
        x = safe_num(games[c])
        pre = x[games["season"].isin(TRAIN_SEASONS)]
        if pre.notna().sum() < MIN_PRE2025_ROWS:
            continue
        if pre.dropna().nunique() < 3:
            continue
        cols.append(c)
    return sorted(set(cols))


def evaluate_feature(games: pd.DataFrame, feature: str, target: str, method: str) -> dict[str, Any] | None:
    work = games[["season", feature, target]].copy()
    work[feature] = safe_num(work[feature])
    work[target] = safe_num(work[target])

    folds: list[dict[str, Any]] = []
    pooled_y: list[np.ndarray] = []
    pooled_pred: list[np.ndarray] = []
    pooled_bias_pred: list[np.ndarray] = []
    pooled_x: list[np.ndarray] = []

    for year in CV_YEARS:
        tr = work[(work["season"] < year) & work["season"].isin(TRAIN_SEASONS)].dropna(subset=[feature, target])
        va = work[work["season"].eq(year)].dropna(subset=[feature, target])
        if len(tr) < MIN_FOLD_ROWS or len(va) < MIN_FOLD_ROWS:
            continue

        xtr = tr[feature].to_numpy(float)
        ytr = tr[target].to_numpy(float)
        xva = va[feature].to_numpy(float)
        yva = va[target].to_numpy(float)

        fit = fit_method(method, xtr, ytr)
        pred = predict_method(method, fit, xva)
        bias = float(np.mean(ytr))
        bias_pred = np.full(len(yva), bias, dtype=float)

        market_mae = float(np.mean(np.abs(yva)))
        bias_mae = float(np.mean(np.abs(yva - bias_pred)))
        candidate_mae = float(np.mean(np.abs(yva - pred)))
        folds.append(
            {
                "year": year,
                "n": len(yva),
                "market_mae": market_mae,
                "bias_mae": bias_mae,
                "candidate_mae": candidate_mae,
                "market_improvement": market_mae - candidate_mae,
                "bias_incremental": bias_mae - candidate_mae,
                "direction": fit_direction(method, fit),
                "spearman": spearman_safe(xva, yva),
            }
        )
        pooled_y.append(yva)
        pooled_pred.append(pred)
        pooled_bias_pred.append(bias_pred)
        pooled_x.append(xva)

    if len(folds) < 2:
        return None

    y = np.concatenate(pooled_y)
    pred = np.concatenate(pooled_pred)
    bias_pred = np.concatenate(pooled_bias_pred)
    x = np.concatenate(pooled_x)
    market_mae = float(np.mean(np.abs(y)))
    bias_mae = float(np.mean(np.abs(y - bias_pred)))
    candidate_mae = float(np.mean(np.abs(y - pred)))
    directions = [f["direction"] for f in folds if f["direction"] != 0]
    fold_wins = sum(1 for f in folds if f["market_improvement"] > 0)
    bias_wins = sum(1 for f in folds if f["bias_incremental"] > 0)
    direction_consistent = len(directions) == len(folds) and len(set(directions)) == 1

    row: dict[str, Any] = {
        "target": target,
        "feature": feature,
        "method": method,
        "cv_folds": len(folds),
        "cv_rows": len(y),
        "cv_market_mae": market_mae,
        "cv_bias_only_mae": bias_mae,
        "cv_candidate_mae": candidate_mae,
        "cv_improvement_vs_market": market_mae - candidate_mae,
        "cv_incremental_vs_bias": bias_mae - candidate_mae,
        "fold_wins_vs_market": fold_wins,
        "fold_wins_vs_bias": bias_wins,
        "worst_fold_improvement_vs_market": min(f["market_improvement"] for f in folds),
        "direction_consistent": direction_consistent,
        "cv_pearson": pearson_safe(x, y),
        "cv_spearman": spearman_safe(x, y),
    }
    for f in folds:
        year = f["year"]
        row[f"improvement_{year}"] = f["market_improvement"]
        row[f"incremental_vs_bias_{year}"] = f["bias_incremental"]
        row[f"direction_{year}"] = f["direction"]
        row[f"spearman_{year}"] = f["spearman"]

    row["stable_cv"] = bool(
        len(folds) == len(CV_YEARS)
        and row["cv_improvement_vs_market"] >= MIN_MEANINGFUL_MAE
        and row["cv_incremental_vs_bias"] > MIN_INCREMENTAL_OVER_BIAS
        and fold_wins == len(CV_YEARS)
        and direction_consistent
    )
    return row


def audit_features(games: pd.DataFrame) -> pd.DataFrame:
    features = market_feature_columns(games)
    print(f"Auditing {len(features)} market-derived features x 2 targets x 2 simple methods...")
    rows: list[dict[str, Any]] = []
    for i, feature in enumerate(features, start=1):
        for target in TARGETS:
            for method in ["linear", "bins"]:
                r = evaluate_feature(games, feature, target, method)
                if r is not None:
                    rows.append(r)
        if i % 20 == 0 or i == len(features):
            print(f"  {i}/{len(features)} features")
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["target", "stable_cv", "cv_improvement_vs_market", "cv_incremental_vs_bias"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
    return out


def freeze_feature_shortlist(cv: pd.DataFrame, top: int) -> pd.DataFrame:
    if cv.empty:
        return cv.copy()
    rows: list[pd.DataFrame] = []
    for target in TARGETS:
        part = cv[cv["target"].eq(target)].copy()
        # One method per feature, chosen before 2025.
        part = (
            part.sort_values(["feature", "cv_improvement_vs_market"], ascending=[True, False])
            .drop_duplicates("feature", keep="first")
        )
        stable = part[part["stable_cv"] == True].copy()  # noqa: E712
        stable = stable.sort_values(
            ["cv_improvement_vs_market", "cv_incremental_vs_bias", "worst_fold_improvement_vs_market"],
            ascending=[False, False, False],
        )
        if len(stable) < top:
            extra = part[
                (~part["feature"].isin(stable["feature"]))
                & (part["cv_improvement_vs_market"] > 0)
            ].copy()
            extra = extra.sort_values(
                ["fold_wins_vs_market", "cv_improvement_vs_market", "cv_incremental_vs_bias"],
                ascending=[False, False, False],
            )
            stable = pd.concat([stable, extra.head(top - len(stable))], ignore_index=True)
        else:
            stable = stable.head(top)
        stable["pre2025_rank"] = np.arange(1, len(stable) + 1)
        rows.append(stable)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def bootstrap_feature_vs_market_and_bias(
    y: np.ndarray,
    pred: np.ndarray,
    bias_pred: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    market_diff = np.abs(y) - np.abs(y - pred)
    bias_diff = np.abs(y - bias_pred) - np.abs(y - pred)
    mask = np.isfinite(market_diff) & np.isfinite(bias_diff)
    market_diff = market_diff[mask]
    bias_diff = bias_diff[mask]
    if len(market_diff) < 20 or reps <= 0:
        return np.nan, np.nan, np.nan, np.nan
    n = len(market_diff)
    m = np.empty(reps, dtype=float)
    b = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        m[i] = float(np.mean(market_diff[idx]))
        b[i] = float(np.mean(bias_diff[idx]))
    return (
        float(np.quantile(m, 0.025)),
        float(np.quantile(m, 0.975)),
        float(np.quantile(b, 0.025)),
        float(np.quantile(b, 0.975)),
    )


def verify_features(frozen: pd.DataFrame, games: pd.DataFrame, reps: int) -> pd.DataFrame:
    if frozen.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict[str, Any]] = []
    for _, s in frozen.iterrows():
        target = str(s["target"])
        feature = str(s["feature"])
        method = str(s["method"])
        tr = games[games["season"].isin(TRAIN_SEASONS)][[feature, target]].dropna()
        va = games[games["season"].eq(FINAL_YEAR)][[feature, target]].dropna()
        if len(tr) < MIN_PRE2025_ROWS or len(va) < MIN_FOLD_ROWS:
            continue

        xtr = tr[feature].to_numpy(float)
        ytr = tr[target].to_numpy(float)
        xva = va[feature].to_numpy(float)
        yva = va[target].to_numpy(float)
        fit = fit_method(method, xtr, ytr)
        pred = predict_method(method, fit, xva)
        bias = float(np.mean(ytr))
        bias_pred = np.full(len(yva), bias, dtype=float)

        market_mae = float(np.mean(np.abs(yva)))
        bias_mae = float(np.mean(np.abs(yva - bias_pred)))
        candidate_mae = float(np.mean(np.abs(yva - pred)))
        imp = market_mae - candidate_mae
        inc = bias_mae - candidate_mae
        mlo, mhi, blo, bhi = bootstrap_feature_vs_market_and_bias(
            yva, pred, bias_pred, reps, rng
        )

        cv_dirs = [
            int(s.get(f"direction_{year}", 0))
            for year in CV_YEARS
            if pd.notna(s.get(f"direction_{year}", np.nan))
        ]
        cv_dirs = [d for d in cv_dirs if d != 0]
        cv_dir = cv_dirs[0] if len(cv_dirs) == len(CV_YEARS) and len(set(cv_dirs)) == 1 else 0
        final_fit_dir = fit_direction(method, fit)

        rows.append(
            {
                "target": target,
                "pre2025_rank": int(s["pre2025_rank"]),
                "feature": feature,
                "method": method,
                "stable_cv": bool(s["stable_cv"]),
                "cv_improvement_vs_market": float(s["cv_improvement_vs_market"]),
                "cv_incremental_vs_bias": float(s["cv_incremental_vs_bias"]),
                "n_2025": len(yva),
                "market_mae_2025": market_mae,
                "bias_only_mae_2025": bias_mae,
                "candidate_mae_2025": candidate_mae,
                "improvement_vs_market_2025": imp,
                "incremental_vs_bias_2025": inc,
                "bootstrap_market_low_2025": mlo,
                "bootstrap_market_high_2025": mhi,
                "bootstrap_bias_low_2025": blo,
                "bootstrap_bias_high_2025": bhi,
                "cv_direction": cv_dir,
                "fit_direction_2021_2024": final_fit_dir,
                "direction_preserved": bool(cv_dir != 0 and cv_dir == final_fit_dir),
                "confirmed_strict": bool(
                    bool(s["stable_cv"])
                    and imp >= MIN_MEANINGFUL_MAE
                    and inc > MIN_INCREMENTAL_OVER_BIAS
                    and cv_dir != 0
                    and cv_dir == final_fit_dir
                    and math.isfinite(mlo)
                    and mlo > 0
                ),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["target", "confirmed_strict", "improvement_vs_market_2025"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Coverage / console reporting
# ---------------------------------------------------------------------------


def matched_coverage(games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season in TRAIN_SEASONS + [FINAL_YEAR]:
        p = games[games["season"].eq(season)]
        rows.append(
            {
                "season": season,
                "repo_games": len(p),
                "consensus_spread_games": int(p.get("consensus_close_home_margin", pd.Series(index=p.index)).notna().sum()),
                "opening_spread_games": int(p.get("consensus_open_home_margin", pd.Series(index=p.index)).notna().sum()),
                "consensus_total_games": int(p.get("consensus_close_total", pd.Series(index=p.index)).notna().sum()),
                "opening_total_games": int(p.get("consensus_open_total", pd.Series(index=p.index)).notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def print_direct_summary(direct: pd.DataFrame) -> None:
    if direct.empty:
        print("No direct line-source comparison rows available.")
        return
    print("\nDIRECT LINE SOURCE MAE BY SEASON:")
    for season in TRAIN_SEASONS + [FINAL_YEAR]:
        part = direct[direct["season"].eq(season)]
        if part.empty:
            continue
        print(f"  {season}:")
        for target in ["margin", "total"]:
            t = part[part["target"].eq(target)]
            vals = []
            for _, r in t.iterrows():
                vals.append(f"{r['source']}={r['mae']:.4f} (n={int(r['games'])})")
            if vals:
                print(f"    {target}: " + " | ".join(vals))


def print_feature_verification(verify: pd.DataFrame) -> None:
    print("\n2025 VERIFICATION OF FROZEN MARKET FEATURES:")
    if verify.empty:
        print("  no frozen features had enough 2025 coverage")
        return
    for target in TARGETS:
        part = verify[verify["target"].eq(target)]
        if part.empty:
            continue
        print(f"  {target}:")
        for _, r in part.head(10).iterrows():
            status = "CONFIRMED" if bool(r["confirmed_strict"]) else "not confirmed"
            print(
                f"    {r['feature']} [{r['method']}] "
                f"CV={r['cv_improvement_vs_market']:+.4f} "
                f"2025={r['improvement_vs_market_2025']:+.4f} "
                f"vs_bias={r['incremental_vs_bias_2025']:+.4f} "
                f"CI=({r['bootstrap_market_low_2025']:+.4f},{r['bootstrap_market_high_2025']:+.4f}) "
                f"{status}"
            )


def print_direct_verification(verify: pd.DataFrame) -> None:
    print("\n2025 VERIFICATION OF FROZEN DIRECT LINE CANDIDATES:")
    if verify.empty:
        print("  no direct candidates had enough 2025 coverage")
        return
    for target in TARGETS:
        part = verify[verify["target"].eq(target)]
        if part.empty:
            continue
        print(f"  {target}:")
        for _, r in part.iterrows():
            status = "CONFIRMED" if bool(r["confirmed_strict"]) else "not confirmed"
            print(
                f"    {r['candidate']} alpha={r['alpha']:.2f} "
                f"CV={r['cv_improvement']:+.4f} "
                f"2025={r['improvement_2025']:+.4f} "
                f"CI=({r['bootstrap_low_2025']:+.4f},{r['bootstrap_high_2025']:+.4f}) "
                f"{status}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfb_root = resolve_cfb_root(args.cfb_root)
    out_dir = cfb_root / "data" / "market_movement_consensus_audit_v1"
    cache_path = out_dir / "cache" / "cfb_line_odds.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CFB root: {cfb_root}")
    print("Loading repo historical outcomes / sportsbook baseline...")
    repo = load_repo_games(cfb_root)
    print(f"  repo games={len(repo)}")

    ensure_odds_file(cache_path, refresh=bool(args.refresh_odds))
    print("Loading SportsDataverse multi-book opening/closing lines...")
    raw = load_raw_odds(cache_path)
    seasons = sorted(raw["season"].dropna().astype(int).unique().tolist())
    print(
        f"  odds rows={len(raw)} games={raw['game_id'].nunique()} "
        f"seasons={seasons} books={raw['book_key'].nunique()}"
    )

    print("Building per-game opening/closing consensus and disagreement features...")
    market, source_coverage, abbr_audit = build_market_features(raw)
    games = merge_market_repo(repo, market)
    coverage = matched_coverage(games)
    print("  matched coverage:")
    for _, r in coverage.iterrows():
        print(
            f"    {int(r['season'])}: repo={int(r['repo_games'])} "
            f"close_spread={int(r['consensus_spread_games'])} "
            f"open_spread={int(r['opening_spread_games'])} "
            f"close_total={int(r['consensus_total_games'])} "
            f"open_total={int(r['opening_total_games'])}"
        )

    # Save coverage before any audit.
    combined_coverage = source_coverage.copy()
    atomic_csv(combined_coverage, out_dir / "odds_coverage.csv")
    atomic_csv(abbr_audit, out_dir / "abbr_team_mapping_audit.csv")
    atomic_csv(games, out_dir / "market_features_2021_2025.csv")

    direct = direct_line_comparison_by_season(games)
    atomic_csv(direct, out_dir / "direct_line_comparison_by_season.csv")
    print_direct_summary(direct)

    print("\nRunning pre-2025 direct consensus/movement candidate audit...")
    direct_cv = direct_candidate_cv(games)
    direct_frozen = freeze_direct_candidates(direct_cv)
    direct_verify = verify_direct_candidates(direct_frozen, games, max(0, int(args.bootstrap)))
    atomic_csv(direct_cv, out_dir / "direct_candidate_cv_2022_2024.csv")
    atomic_csv(direct_frozen, out_dir / "direct_candidate_frozen_pre2025.csv")
    atomic_csv(direct_verify, out_dir / "direct_candidate_verification_2025.csv")

    print("Running pre-2025 feature signal audit...")
    feature_cv = audit_features(games)
    frozen = freeze_feature_shortlist(feature_cv, max(1, int(args.top)))
    print("Freezing strongest pre-2025 market features...")
    verify = verify_features(frozen, games, max(0, int(args.bootstrap)))

    atomic_csv(feature_cv, out_dir / "feature_cv_2022_2024.csv")
    atomic_csv(frozen, out_dir / "shortlist_frozen_pre2025.csv")
    atomic_csv(verify, out_dir / "verification_2025.csv")

    print_direct_verification(direct_verify)
    print_feature_verification(verify)

    strict_direct = (
        direct_verify[direct_verify.get("confirmed_strict", pd.Series(dtype=bool)) == True].copy()  # noqa: E712
        if not direct_verify.empty
        else pd.DataFrame()
    )
    strict_features = (
        verify[verify.get("confirmed_strict", pd.Series(dtype=bool)) == True].copy()  # noqa: E712
        if not verify.empty
        else pd.DataFrame()
    )

    summary = {
        "policy": {
            "train_seasons": TRAIN_SEASONS,
            "forward_cv_years": CV_YEARS,
            "untouched_final_year": FINAL_YEAR,
            "min_fold_rows": MIN_FOLD_ROWS,
            "min_pre2025_rows": MIN_PRE2025_ROWS,
            "meaningful_mae_threshold": MIN_MEANINGFUL_MAE,
        },
        "repo_games": int(len(repo)),
        "odds_rows_2021_2025": int(len(raw)),
        "odds_games_2021_2025": int(raw["game_id"].nunique()),
        "odds_books_2021_2025": int(raw["book_key"].nunique()),
        "matched_coverage": coverage.to_dict("records"),
        "pre2025_feature_rows_tested": int(len(feature_cv)),
        "frozen_feature_rows": int(len(frozen)),
        "strict_feature_confirmations_2025": int(len(strict_features)),
        "strict_direct_confirmations_2025": int(len(strict_direct)),
        "strict_feature_names": (
            strict_features[["target", "feature", "method", "improvement_vs_market_2025"]].to_dict("records")
            if not strict_features.empty
            else []
        ),
        "strict_direct_names": (
            strict_direct[["target", "candidate", "alpha", "improvement_2025"]].to_dict("records")
            if not strict_direct.empty
            else []
        ),
    }
    atomic_json(summary, out_dir / "audit_summary.json")

    print("\nAUDIT RESULT:")
    if strict_direct.empty and strict_features.empty:
        print("  NO STRICTLY CONFIRMED MARKET-MOVEMENT / CONSENSUS SIGNALS")
        print("  Do not build another score model from these signals.")
    else:
        print(
            f"  STRICTLY CONFIRMED: direct={len(strict_direct)} "
            f"feature={len(strict_features)}"
        )
        print("  Only confirmed signals should advance to a narrow correction-layer test.")

    print(f"\nMarket features: {out_dir / 'market_features_2021_2025.csv'}")
    print(f"Direct comparison: {out_dir / 'direct_line_comparison_by_season.csv'}")
    print(f"Direct CV: {out_dir / 'direct_candidate_cv_2022_2024.csv'}")
    print(f"Direct 2025: {out_dir / 'direct_candidate_verification_2025.csv'}")
    print(f"Feature CV: {out_dir / 'feature_cv_2022_2024.csv'}")
    print(f"Frozen shortlist: {out_dir / 'shortlist_frozen_pre2025.csv'}")
    print(f"Feature 2025: {out_dir / 'verification_2025.csv'}")
    print(f"Summary: {out_dir / 'audit_summary.json'}")


if __name__ == "__main__":
    main()

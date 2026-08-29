#!/usr/bin/env python3
"""
train_score_model_v5.py

Standalone CFB score-model experiment focused on two information classes not
properly tested by V1-V4:

1. Leakage-safe player / quarterback continuity inferred from local native
   SportsDataverse PBP.  A game can only use players observed in earlier weeks
   of the current season plus prior-season production.  The game being
   predicted is never used to determine its own QB or returning players.

2. Matchup-specific PBP interactions: pass offense vs pass defense, rush
   offense vs rush defense, sack pressure, interception/turnover pressure,
   explosive-play creation/prevention, third-down performance, red-zone
   performance, pass tendency and pace.

The script also keeps the useful point-in-time/preseason SDV sources from V4
(talent, returning production, recruiting projection, weekly ratings/FEI and
historical contemporaneous FPI), but deliberately DOES NOT feed the prior V4
model all 375 weekly summary columns.  The goal is a smaller, matchup-focused
feature set rather than another 1,300-column search.

Leakage controls
----------------
* Current-season PBP features use only weeks strictly less than the game week.
* Week 1 PBP strength falls back to the previous season final profile.
* QB/player continuity never looks at the current game.  Week 1 has no
  current-season player-return flag; it uses prior-season concentration plus
  preseason aggregate returning-production/talent data.
* Weekly SDV ratings use through_week <= game_week - 1; Week 1 uses the prior
  season final snapshot.
* Historical FPI uses only contemporaneous, in-sequence snapshots whose AS-OF
  date is strictly before the game date.
* Model/group/blend selection uses forward validation seasons 2022-2024 only.
* 2025 is untouched until every feature group/model/blend choice is frozen.
* If the frozen model does not materially beat the market in 2025, current
  lined games deploy the sportsbook-implied score rather than the failed model.

Repository policy
-----------------
This script READS the existing repository and WRITES ONLY under:

    docs/win/football/cfb/data/score_model_v5/

It never modifies projection.py, projection_week1.py, config, intake, picks,
historical inputs, workflows, or any other existing repository file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import requests

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


SCRIPT_VERSION = "cfb-player-matchup-team-score-v5-2026-08-29"
HISTORICAL_SEASONS = [2021, 2022, 2023, 2024, 2025]
CV_SEASONS = [2022, 2023, 2024]
FINAL_SEASON = 2025
SEASON_TYPE = 2
RANDOM_STATE = 20260829

PROVIDER_BY_SEASON = {
    2021: "draftkings",
    2022: "draftkings",
    2023: "draftkings",
    2024: "espnbet",
    2025: "espnbet",
}

# A model is not called useful for microscopic changes.
CV_MIN_TEAM_SCORE_IMPROVEMENT = 0.10
CV_REQUIRED_FOLD_WINS = 2
FINAL_MIN_TEAM_SCORE_IMPROVEMENT = 0.20
FINAL_MAX_MARGIN_DEGRADATION = 0.10
FINAL_MAX_TOTAL_DEGRADATION = 0.10
FINAL_MAX_WINNER_ACCURACY_DEGRADATION = 0.005

RESIDUAL_ALPHA_GRID = np.round(np.arange(0.0, 1.51, 0.05), 2)
SCORE_MIN, SCORE_MAX = 0.0, 100.0

SDV_RELEASE_BASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
SDV_DATASETS = {
    "ratings_weekly": {
        "tag": "cfb_ratings_weekly",
        "filename": "cfb_ratings_weekly_{season}.parquet",
    },
    "team_summaries_weekly": {
        "tag": "cfb_team_summaries_weekly",
        "filename": "cfb_team_summaries_weekly_{season}.parquet",
    },
    "fpi_weekly": {
        "tag": "cfb_fpi_weekly",
        "filename": "cfb_fpi_weekly_{season}.parquet",
    },
    "team_talent": {
        "tag": "cfb_team_talent",
        "filename": "cfb_team_talent_{season}.parquet",
    },
    "returning_production": {
        "tag": "cfb_returning_production",
        "filename": "cfb_returning_production_{season}.parquet",
    },
    "recruiting_proj": {
        "tag": "cfb_recruiting_proj",
        "filename": "cfb_recruiting_proj_{season}.parquet",
    },
}

IDENTIFIER_COLUMNS = {
    "season",
    "week",
    "through_week",
    "team_id",
    "game_id",
    "id",
    "run_date_time_key",
}

METADATA_COLUMNS = {
    "season",
    "week",
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "team",
    "opponent",
    "team_id",
    "opponent_id",
    "team_side",
    "actual_team_score",
    "actual_opp_score",
    "target_score_residual",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train CFB team-score residual model from player continuity and matchup-specific PBP features."
    )
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--predict-week", type=int, default=None)
    p.add_argument(
        "--fast",
        action="store_true",
        help="Smaller model search for troubleshooting.",
    )
    p.add_argument(
        "--refresh-sdv",
        action="store_true",
        help="Redownload SportsDataverse cache assets even when cached.",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Use existing score_model_v5/sdv_cache only; never access network.",
    )
    return p.parse_args()


def cfb_root() -> Path:
    return Path(__file__).resolve().parents[2]


def out_dir(root: Path) -> Path:
    path = root / "data" / "score_model_v5"
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    s = str(value).strip()
    if s.casefold() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return s


def norm_id(value: Any) -> str:
    s = clean_text(value)
    if not s:
        return ""
    if re.fullmatch(r"[-+]?\d+\.0+", s):
        s = s.split(".", 1)[0]
    return s


def norm_name(value: Any) -> str:
    s = clean_text(value)
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def num(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    mapped = (
        s.astype(str)
        .str.strip()
        .str.casefold()
        .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
    )
    return mapped.fillna(False)


def safe_read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def require_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def infer_predict_week(root: Path, season: int, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    weekly = root / "00_intake" / "schedule" / "weekly"
    weeks: list[int] = []
    for path in weekly.glob("week_*_CFB_weekly_schedule.csv"):
        m = re.search(r"week_(\d+)_", path.name)
        if m:
            weeks.append(int(m.group(1)))
    if weeks:
        return max(weeks)
    raise ValueError("Could not infer prediction week; pass --predict-week.")


# ---------------------------------------------------------------------------
# SportsDataverse release cache
# ---------------------------------------------------------------------------


def sdv_cache_path(cache_dir: Path, dataset: str, season: int) -> Path:
    spec = SDV_DATASETS[dataset]
    return cache_dir / spec["filename"].format(season=season)


def sdv_url(dataset: str, season: int) -> str:
    spec = SDV_DATASETS[dataset]
    filename = spec["filename"].format(season=season)
    return f"{SDV_RELEASE_BASE}/{spec['tag']}/{filename}"


def download_file(url: str, dest: Path, timeout: int = 180) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
            if r.status_code == 404:
                return False
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp, dest)
        return True
    except Exception as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        print(f"  WARNING download failed: {url} ({type(exc).__name__}: {exc})")
        return False


def ensure_sdv_asset(
    cache_dir: Path,
    dataset: str,
    season: int,
    *,
    refresh: bool,
    no_download: bool,
) -> Path | None:
    path = sdv_cache_path(cache_dir, dataset, season)
    if path.exists() and path.stat().st_size > 0 and not refresh:
        return path

    # V5 can reuse the already-downloaded V4 cache without copying it.
    legacy_dir = cache_dir.parent.parent / "score_model_v4" / "sdv_cache"
    legacy = sdv_cache_path(legacy_dir, dataset, season)
    if legacy.exists() and legacy.stat().st_size > 0 and not refresh:
        return legacy

    if no_download:
        if path.exists():
            return path
        return legacy if legacy.exists() else None
    url = sdv_url(dataset, season)
    print(f"  downloading {dataset} {season}...")
    ok = download_file(url, path)
    if not ok:
        print(f"  unavailable: {dataset} {season}")
        return None
    return path


def load_sdv_assets(
    cache_dir: Path,
    prediction_season: int,
    *,
    refresh: bool,
    no_download: bool,
) -> tuple[dict[str, dict[int, pd.DataFrame]], pd.DataFrame]:
    stores: dict[str, dict[int, pd.DataFrame]] = {k: {} for k in SDV_DATASETS}
    coverage: list[dict[str, Any]] = []

    # Weekly sources need 2020 to provide a leakage-safe Week-1 prior for 2021.
    weekly_seasons = list(range(2020, prediction_season + 1))
    preseason_seasons = list(range(2021, prediction_season + 1))

    for dataset in SDV_DATASETS:
        seasons = weekly_seasons if dataset in {"ratings_weekly", "team_summaries_weekly"} else preseason_seasons
        if dataset == "fpi_weekly":
            seasons = preseason_seasons
        print(f"Loading SDV {dataset}...")
        for season in seasons:
            path = ensure_sdv_asset(
                cache_dir,
                dataset,
                season,
                refresh=refresh,
                no_download=no_download,
            )
            if path is None or not path.exists():
                coverage.append({"dataset": dataset, "season": season, "rows": 0, "columns": 0, "status": "missing"})
                continue
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                print(f"  WARNING could not read {path.name}: {exc}")
                coverage.append({"dataset": dataset, "season": season, "rows": 0, "columns": 0, "status": "read_error"})
                continue
            if "season" not in df.columns:
                df["season"] = season
            stores[dataset][season] = df
            coverage.append({
                "dataset": dataset,
                "season": season,
                "rows": len(df),
                "columns": len(df.columns),
                "status": "ok",
            })
            print(f"  {season}: rows={len(df)} cols={len(df.columns)}")

    return stores, pd.DataFrame(coverage)


# ---------------------------------------------------------------------------
# Local repository inputs
# ---------------------------------------------------------------------------


def extract_historical_results(root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for season in HISTORICAL_SEASONS:
        path = root / "00_intake" / "pbp" / f"{season}_pbp.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing historical PBP: {path}")
        required = [
            "game_id", "sequenceNumber", "end.homeScore", "end.awayScore",
            "homeTeamId", "awayTeamId",
        ]
        try:
            df = pd.read_parquet(path, columns=required)
        except Exception as exc:
            raise ValueError(
                f"Could not read required final-score/id columns from {path}: {exc}"
            ) from exc
        df["game_id"] = df["game_id"].map(norm_id)
        df["sequenceNumber_num"] = pd.to_numeric(df["sequenceNumber"], errors="coerce")
        df["home_score"] = pd.to_numeric(df["end.homeScore"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["end.awayScore"], errors="coerce")
        df["home_team_id"] = df["homeTeamId"].map(norm_id)
        df["away_team_id"] = df["awayTeamId"].map(norm_id)
        df = df.sort_values(["game_id", "sequenceNumber_num"], na_position="first")
        last = (
            df.groupby("game_id", as_index=False)
            .agg(
                home_score=("home_score", "last"),
                away_score=("away_score", "last"),
                home_team_id=("home_team_id", "last"),
                away_team_id=("away_team_id", "last"),
            )
        )
        last["season"] = season
        rows.append(last)
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["home_score", "away_score"])
    return out


def load_historical_market(root: Path, season: int) -> pd.DataFrame:
    provider = PROVIDER_BY_SEASON[season]
    path = (
        root
        / "data"
        / "historical_betting"
        / "cache"
        / f"{season}_{provider}_espn_market_predictor.csv"
    )
    df = safe_read_csv(path)
    if df.empty:
        raise FileNotFoundError(f"Missing historical market cache: {path}")
    require_columns(df, ["game_id", "home_spread", "total"], str(path))
    df["game_id"] = df["game_id"].map(norm_id)
    keep = [
        "game_id", "home_spread", "total", "espn_home_ptdiff", "espn_away_ptdiff",
        "espn_home_game_projection", "espn_away_game_projection", "odds_status", "predictor_status",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    # One row per game; provider cache is expected unique, but protect against duplicates.
    df = df.drop_duplicates("game_id", keep="last")
    return df


def load_historical_games(root: Path, results: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in HISTORICAL_SEASONS:
        path = root / "00_intake" / "schedule" / f"{season}_schedule.csv"
        schedule = safe_read_csv(path)
        if schedule.empty:
            raise FileNotFoundError(f"Missing historical schedule: {path}")
        require_columns(
            schedule,
            ["season", "season_type", "week", "game_id", "game_date", "away_team", "home_team", "neutral_site"],
            str(path),
        )
        schedule["season"] = pd.to_numeric(schedule["season"], errors="coerce")
        schedule["season_type"] = pd.to_numeric(schedule["season_type"], errors="coerce")
        schedule["week"] = pd.to_numeric(schedule["week"], errors="coerce")
        schedule["game_id"] = schedule["game_id"].map(norm_id)
        schedule = schedule[(schedule["season"] == season) & (schedule["season_type"] == SEASON_TYPE)].copy()
        schedule["game_date"] = pd.to_datetime(schedule["game_date"], errors="coerce")

        r = results[results["season"] == season].drop(columns=["season"])
        market = load_historical_market(root, season)
        merged = schedule.merge(r, on="game_id", how="inner").merge(market, on="game_id", how="left")
        frames.append(merged)

    games = pd.concat(frames, ignore_index=True)
    games["home_spread"] = pd.to_numeric(games.get("home_spread"), errors="coerce")
    games["total"] = pd.to_numeric(games.get("total"), errors="coerce")
    games["neutral_site"] = pd.to_numeric(games["neutral_site"], errors="coerce").fillna(0).astype(int)
    games["actual_margin"] = games["home_score"] - games["away_score"]
    games["actual_total"] = games["home_score"] + games["away_score"]
    games["market_margin"] = -games["home_spread"]
    games["market_home_score"] = (games["total"] + games["market_margin"]) / 2.0
    games["market_away_score"] = (games["total"] - games["market_margin"]) / 2.0
    return games


def build_team_alias_map(root: Path) -> dict[str, str]:
    path = root / "config" / "mapping" / "team_map.csv"
    df = safe_read_csv(path, encoding="utf-8-sig")
    if df.empty:
        return {}
    out: dict[str, str] = {}
    name_cols = [
        c for c in ["canonical_team", "alias", "location", "shortDisplayName", "team_slug", "team_name"]
        if c in df.columns
    ]
    if "team_id" not in df.columns:
        return {}
    for _, row in df.iterrows():
        tid = norm_id(row.get("team_id"))
        if not tid:
            continue
        for c in name_cols:
            key = norm_name(row.get(c))
            if key and key not in out:
                out[key] = tid
    return out


def current_games(root: Path, season: int, week: int, alias_map: dict[str, str]) -> pd.DataFrame:
    enriched_path = root / "01_merge" / f"week_{week}_CFB_enriched.csv"
    weekly_path = root / "00_intake" / "schedule" / "weekly" / f"week_{week}_CFB_weekly_schedule.csv"

    if enriched_path.exists():
        df = safe_read_csv(enriched_path)
    else:
        df = safe_read_csv(weekly_path)
    if df.empty:
        raise FileNotFoundError(f"Missing current enriched/weekly schedule for week {week}")

    require_columns(df, ["game_id", "away_team", "home_team"], "current games")
    df["game_id"] = df["game_id"].map(norm_id)
    if "season" not in df.columns:
        df["season"] = season
    if "week" not in df.columns:
        df["week"] = week
    if "game_date" not in df.columns:
        df["game_date"] = pd.NaT
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if "neutral_site" not in df.columns:
        if "neutral_site_flag" in df.columns:
            df["neutral_site"] = df["neutral_site_flag"]
        else:
            df["neutral_site"] = 0
    df["neutral_site"] = pd.to_numeric(df["neutral_site"], errors="coerce").fillna(0).astype(int)

    if "home_team_id" not in df.columns:
        df["home_team_id"] = ""
    if "away_team_id" not in df.columns:
        df["away_team_id"] = ""
    df["home_team_id"] = df["home_team_id"].map(norm_id)
    df["away_team_id"] = df["away_team_id"].map(norm_id)

    missing_home = df["home_team_id"].eq("")
    missing_away = df["away_team_id"].eq("")
    if missing_home.any():
        df.loc[missing_home, "home_team_id"] = df.loc[missing_home, "home_team"].map(
            lambda x: alias_map.get(norm_name(x), "")
        )
    if missing_away.any():
        df.loc[missing_away, "away_team_id"] = df.loc[missing_away, "away_team"].map(
            lambda x: alias_map.get(norm_name(x), "")
        )

    for col in ["home_spread", "total"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["market_margin"] = -df["home_spread"]
    df["market_home_score"] = (df["total"] + df["market_margin"]) / 2.0
    df["market_away_score"] = (df["total"] - df["market_margin"]) / 2.0

    return df.drop_duplicates("game_id", keep="last").copy()


# ---------------------------------------------------------------------------
# Point-in-time SDV feature preparation and lookup
# ---------------------------------------------------------------------------


def numeric_feature_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().any():
            cols.append(c)
    return cols


@dataclass
class FeatureStore:
    preseason: dict[tuple[int, str], dict[str, float]]
    ratings: dict[tuple[int, str], pd.DataFrame]
    summaries: dict[tuple[int, str], pd.DataFrame]
    fpi: dict[tuple[int, str], pd.DataFrame]
    source_columns: dict[str, list[str]]


def prepare_feature_store(stores: dict[str, dict[int, pd.DataFrame]]) -> FeatureStore:
    preseason: dict[tuple[int, str], dict[str, float]] = {}
    source_columns: dict[str, list[str]] = {"pre": [], "rat": [], "sum": [], "fpi": []}

    # Merge three preseason-known sources by season/team.
    pre_parts: list[pd.DataFrame] = []
    pre_specs = [
        ("team_talent", "talent"),
        ("returning_production", "return"),
        ("recruiting_proj", "recruit"),
    ]
    for dataset, short in pre_specs:
        season_parts: list[pd.DataFrame] = []
        dataset_feature_names: set[str] = set()
        for season, raw in stores.get(dataset, {}).items():
            if raw.empty or "team_id" not in raw.columns:
                continue
            df = raw.copy()
            df["season"] = pd.to_numeric(df.get("season", season), errors="coerce").fillna(season).astype(int)
            df["team_id"] = df["team_id"].map(norm_id)
            feature_cols = numeric_feature_columns(df, IDENTIFIER_COLUMNS | {"team"})
            rename = {c: f"pre_{short}_{c}" for c in feature_cols}
            part = df[["season", "team_id"] + feature_cols].rename(columns=rename)
            part = part.drop_duplicates(["season", "team_id"], keep="last")
            season_parts.append(part)
            dataset_feature_names.update(rename.values())
        if season_parts:
            pre_parts.append(pd.concat(season_parts, ignore_index=True, sort=False))
            source_columns["pre"].extend(sorted(dataset_feature_names))

    if pre_parts:
        merged = pre_parts[0]
        for part in pre_parts[1:]:
            merged = merged.merge(part, on=["season", "team_id"], how="outer")
        for _, row in merged.iterrows():
            key = (int(row["season"]), norm_id(row["team_id"]))
            vals = {c: num(row[c]) for c in merged.columns if c not in {"season", "team_id"}}
            preseason[key] = vals

    ratings: dict[tuple[int, str], pd.DataFrame] = {}
    for season, raw in stores.get("ratings_weekly", {}).items():
        if raw.empty or "team_id" not in raw.columns or "through_week" not in raw.columns:
            continue
        df = raw.copy()
        df["season"] = pd.to_numeric(df.get("season", season), errors="coerce").fillna(season).astype(int)
        df["team_id"] = df["team_id"].map(norm_id)
        df["through_week"] = pd.to_numeric(df["through_week"], errors="coerce")
        feature_cols = numeric_feature_columns(df, IDENTIFIER_COLUMNS)
        rename = {c: f"rat_{c}" for c in feature_cols}
        source_columns["rat"].extend(rename.values())
        df = df[["season", "team_id", "through_week"] + feature_cols].rename(columns=rename)
        for tid, grp in df.groupby("team_id"):
            ratings[(season, norm_id(tid))] = grp.sort_values("through_week").reset_index(drop=True)

    summaries: dict[tuple[int, str], pd.DataFrame] = {}
    for season, raw in stores.get("team_summaries_weekly", {}).items():
        if raw.empty or "team_id" not in raw.columns or "through_week" not in raw.columns:
            continue
        df = raw.copy()
        df["season"] = pd.to_numeric(df.get("season", season), errors="coerce").fillna(season).astype(int)
        df["team_id"] = df["team_id"].map(norm_id)
        df["through_week"] = pd.to_numeric(df["through_week"], errors="coerce")
        # Keep all numeric descriptive measures.  Rank columns are intentionally
        # retained; they are point-in-time and may encode nonlinear national context.
        feature_cols = numeric_feature_columns(df, IDENTIFIER_COLUMNS)
        rename = {c: f"sum_{c}" for c in feature_cols}
        source_columns["sum"].extend(rename.values())
        df = df[["season", "team_id", "through_week"] + feature_cols].rename(columns=rename)
        for tid, grp in df.groupby("team_id"):
            summaries[(season, norm_id(tid))] = grp.sort_values("through_week").reset_index(drop=True)

    fpi: dict[tuple[int, str], pd.DataFrame] = {}
    for season, raw in stores.get("fpi_weekly", {}).items():
        if raw.empty or "team_id" not in raw.columns:
            continue
        df = raw.copy()
        df["team_id"] = df["team_id"].map(norm_id)
        if "snapshot_out_of_sequence" in df.columns:
            df = df[~to_bool_series(df["snapshot_out_of_sequence"])].copy()
        if "snapshot_is_contemporaneous" in df.columns:
            df = df[to_bool_series(df["snapshot_is_contemporaneous"])].copy()
        if "run_date_time_key" in df.columns:
            run_text = df["run_date_time_key"].astype(str).str.replace(r"\.0$", "", regex=True).str[:8]
            df["asof_date"] = pd.to_datetime(run_text, format="%Y%m%d", errors="coerce")
        else:
            df["asof_date"] = pd.NaT
        feature_cols = numeric_feature_columns(
            df,
            IDENTIFIER_COLUMNS | {"snapshot_out_of_sequence", "snapshot_is_contemporaneous", "asof_date"},
        )
        rename = {c: f"fpi_{c}" for c in feature_cols}
        source_columns["fpi"].extend(rename.values())
        use = df[["team_id", "asof_date"] + feature_cols].rename(columns=rename)
        use = use.dropna(subset=["asof_date"])
        for tid, grp in use.groupby("team_id"):
            fpi[(season, norm_id(tid))] = grp.sort_values("asof_date").reset_index(drop=True)

    for k in source_columns:
        source_columns[k] = sorted(set(source_columns[k]))

    return FeatureStore(
        preseason=preseason,
        ratings=ratings,
        summaries=summaries,
        fpi=fpi,
        source_columns=source_columns,
    )


def row_numeric_dict(row: pd.Series, skip: set[str]) -> dict[str, float]:
    return {c: num(row[c]) for c in row.index if c not in skip}


def weekly_lookup(
    store: dict[tuple[int, str], pd.DataFrame],
    season: int,
    team_id: str,
    game_week: int,
) -> dict[str, float]:
    tid = norm_id(team_id)
    if not tid:
        return {}
    if game_week > 1:
        grp = store.get((season, tid))
        if grp is not None and not grp.empty:
            eligible = grp[pd.to_numeric(grp["through_week"], errors="coerce") <= game_week - 1]
            if not eligible.empty:
                row = eligible.iloc[-1]
                return row_numeric_dict(row, {"season", "team_id", "through_week"})
    # Week 1 or missing current-season prior: previous season final snapshot.
    grp = store.get((season - 1, tid))
    if grp is not None and not grp.empty:
        row = grp.iloc[-1]
        return row_numeric_dict(row, {"season", "team_id", "through_week"})
    return {}


def fpi_lookup(
    store: dict[tuple[int, str], pd.DataFrame],
    season: int,
    team_id: str,
    game_date: pd.Timestamp,
) -> dict[str, float]:
    tid = norm_id(team_id)
    if not tid or pd.isna(game_date):
        return {}
    grp = store.get((season, tid))
    if grp is None or grp.empty:
        return {}
    # Strict date inequality is deliberately conservative: no same-day snapshot
    # can accidentally contain information after kickoff.
    eligible = grp[grp["asof_date"] < pd.Timestamp(game_date).normalize()]
    if eligible.empty:
        return {}
    row = eligible.iloc[-1]
    return row_numeric_dict(row, {"team_id", "asof_date"})


def source_triplet(prefix: str, own: dict[str, float], opp: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(own) | set(opp))
    out: dict[str, float] = {}
    for key in keys:
        # key already contains its source prefix (pre_, rat_, sum_, fpi_).
        o = own.get(key, float("nan"))
        p = opp.get(key, float("nan"))
        out[f"own_{key}"] = o
        out[f"opp_{key}"] = p
        out[f"diff_{key}"] = o - p if math.isfinite(o) and math.isfinite(p) else float("nan")
    return out


def predictor_values(game: pd.Series, home: bool) -> tuple[float, float]:
    if home:
        team_pt = num(game.get("espn_home_ptdiff"))
        opp_pt = num(game.get("espn_away_ptdiff"))
        team_prob = num(game.get("espn_home_game_projection"))
        if not math.isfinite(team_prob):
            team_prob = num(game.get("espn_home_prob"))
        opp_prob = num(game.get("espn_away_game_projection"))
        if not math.isfinite(opp_prob):
            opp_prob = num(game.get("espn_away_prob"))
    else:
        team_pt = num(game.get("espn_away_ptdiff"))
        opp_pt = num(game.get("espn_home_ptdiff"))
        team_prob = num(game.get("espn_away_game_projection"))
        if not math.isfinite(team_prob):
            team_prob = num(game.get("espn_away_prob"))
        opp_prob = num(game.get("espn_home_game_projection"))
        if not math.isfinite(opp_prob):
            opp_prob = num(game.get("espn_home_prob"))
    return team_pt, opp_pt, team_prob, opp_prob


def build_team_rows(games: pd.DataFrame, features: FeatureStore, pbp_store: PBPFeatureStore, include_targets: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, g in games.iterrows():
        season = int(num(g.get("season")))
        week = int(num(g.get("week")))
        game_date = pd.to_datetime(g.get("game_date"), errors="coerce")
        neutral = int(num(g.get("neutral_site")) if math.isfinite(num(g.get("neutral_site"))) else 0)

        for home in [True, False]:
            side = "home" if home else "away"
            opp_side = "away" if home else "home"
            team_id = norm_id(g.get(f"{side}_team_id"))
            opp_id = norm_id(g.get(f"{opp_side}_team_id"))
            team_name = clean_text(g.get(f"{side}_team"))
            opp_name = clean_text(g.get(f"{opp_side}_team"))

            market_team_score = num(g.get(f"market_{side}_score"))
            market_opp_score = num(g.get(f"market_{opp_side}_score"))
            market_margin_home = num(g.get("market_margin"))
            market_team_margin = market_margin_home if home else -market_margin_home
            market_total = num(g.get("total"))
            espn_team_pt, espn_opp_pt, espn_team_prob, espn_opp_prob = predictor_values(g, home)

            own_pre = features.preseason.get((season, team_id), {})
            opp_pre = features.preseason.get((season, opp_id), {})
            own_rat = weekly_lookup(features.ratings, season, team_id, week)
            opp_rat = weekly_lookup(features.ratings, season, opp_id, week)
            own_sum = weekly_lookup(features.summaries, season, team_id, week)
            opp_sum = weekly_lookup(features.summaries, season, opp_id, week)
            own_fpi = fpi_lookup(features.fpi, season, team_id, game_date)
            opp_fpi = fpi_lookup(features.fpi, season, opp_id, game_date)
            own_pbp = pregame_pbp_profile(pbp_store, season, team_id, week)
            opp_pbp = pregame_pbp_profile(pbp_store, season, opp_id, week)
            own_player = pregame_player_profile(pbp_store, season, team_id, week)
            opp_player = pregame_player_profile(pbp_store, season, opp_id, week)

            row: dict[str, Any] = {
                "season": season,
                "week": week,
                "game_id": norm_id(g.get("game_id")),
                "game_date": game_date,
                "home_team": clean_text(g.get("home_team")),
                "away_team": clean_text(g.get("away_team")),
                "home_team_id": norm_id(g.get("home_team_id")),
                "away_team_id": norm_id(g.get("away_team_id")),
                "team": team_name,
                "opponent": opp_name,
                "team_id": team_id,
                "opponent_id": opp_id,
                "team_side": side,
                "team_is_home": 1.0 if home else 0.0,
                "neutral_site": float(neutral),
                "week_num": float(week),
                "week_sin": math.sin(2.0 * math.pi * min(max(week, 1), 16) / 16.0),
                "week_cos": math.cos(2.0 * math.pi * min(max(week, 1), 16) / 16.0),
                "market_team_score": market_team_score,
                "market_opp_score": market_opp_score,
                "market_team_margin": market_team_margin,
                "market_total": market_total,
                "market_abs_margin": abs(market_team_margin) if math.isfinite(market_team_margin) else float("nan"),
                "espn_team_ptdiff": espn_team_pt,
                "espn_opp_ptdiff": espn_opp_pt,
                "espn_ptdiff_edge": espn_team_pt - market_team_margin
                    if math.isfinite(espn_team_pt) and math.isfinite(market_team_margin) else float("nan"),
                "espn_team_prob": espn_team_prob,
                "espn_opp_prob": espn_opp_prob,
                "has_preseason": float(bool(own_pre) and bool(opp_pre)),
                "has_ratings": float(bool(own_rat) and bool(opp_rat)),
                "has_summaries": float(bool(own_sum) and bool(opp_sum)),
                "has_fpi": float(bool(own_fpi) and bool(opp_fpi)),
                "has_pbp_matchup": float(bool(own_pbp) and bool(opp_pbp)),
                "has_player_continuity": float(bool(own_player)),
            }
            row.update(source_triplet("pre", own_pre, opp_pre))
            row.update(source_triplet("rat", own_rat, opp_rat))
            # Weekly summaries are joined for compatibility/auditing but V5 feature
            # groups deliberately do not consume the full 375-column summary block.
            row.update(source_triplet("sum", own_sum, opp_sum))
            row.update(source_triplet("fpi", own_fpi, opp_fpi))
            row.update(source_triplet("pbp", own_pbp, opp_pbp))
            row.update(pbp_matchup_features(own_pbp, opp_pbp))
            row.update(player_triplet(own_player, opp_player))

            if include_targets:
                actual_team = num(g.get(f"{side}_score"))
                actual_opp = num(g.get(f"{opp_side}_score"))
                row["actual_team_score"] = actual_team
                row["actual_opp_score"] = actual_opp
                row["target_score_residual"] = (
                    actual_team - market_team_score
                    if math.isfinite(actual_team) and math.isfinite(market_team_score)
                    else float("nan")
                )
            rows.append(row)

    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# Leakage-safe local PBP matchup + player continuity features
# ---------------------------------------------------------------------------


@dataclass
class PBPFeatureStore:
    team_games: dict[tuple[int, str], pd.DataFrame]
    player_usage: dict[tuple[int, str], pd.DataFrame]
    coverage: pd.DataFrame


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _bool_col(df: pd.DataFrame, names: list[str]) -> pd.Series:
    col = _first_existing(df, names)
    if col is None:
        return pd.Series(False, index=df.index, dtype=bool)
    return to_bool_series(df[col]).fillna(False)


def _num_col(df: pd.DataFrame, names: list[str]) -> pd.Series:
    col = _first_existing(df, names)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _player_key(df: pd.DataFrame, role: str) -> pd.Series:
    id_names = [
        f"{role}_player_id", f"{role}PlayerId", f"{role}.id", f"{role}_id",
    ]
    name_names = [
        f"{role}_player_name", f"{role}PlayerName", f"{role}.displayName", f"{role}_name",
    ]
    id_col = _first_existing(df, id_names)
    name_col = _first_existing(df, name_names)
    out = pd.Series("", index=df.index, dtype=object)
    if id_col is not None:
        ids = df[id_col].map(norm_id)
        out = np.where(ids != "", "id:" + ids.astype(str), "")
        out = pd.Series(out, index=df.index, dtype=object)
    if name_col is not None:
        names = df[name_col].map(norm_name)
        use = (out == "") & (names != "")
        out.loc[use] = "name:" + names.loc[use].astype(str)
    return out


def _assign_team_ids(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    # Prefer native possession-team IDs if they are present.
    pos_id_col = _first_existing(df, ["pos_team_id", "posTeamId", "pos_team.id"])
    def_id_col = _first_existing(df, ["def_pos_team_id", "defPosTeamId", "def_pos_team.id"])
    if pos_id_col is not None:
        off = df[pos_id_col].map(norm_id)
    else:
        pos = df.get("pos_team", pd.Series("", index=df.index)).map(clean_text)
        home_name = df.get("homeTeamName", pd.Series("", index=df.index)).map(clean_text)
        away_name = df.get("awayTeamName", pd.Series("", index=df.index)).map(clean_text)
        home_id = df.get("homeTeamId", pd.Series("", index=df.index)).map(norm_id)
        away_id = df.get("awayTeamId", pd.Series("", index=df.index)).map(norm_id)

        # Some SDV schema versions store pos_team directly as a numeric team id.
        pos_as_id = pos.map(norm_id)
        numeric_match = (pos_as_id == home_id) | (pos_as_id == away_id)
        off = pd.Series("", index=df.index, dtype=object)
        off.loc[numeric_match] = pos_as_id.loc[numeric_match]

        pnorm = pos.map(norm_name)
        hnorm = home_name.map(norm_name)
        anorm = away_name.map(norm_name)
        off.loc[(off == "") & (pnorm == hnorm)] = home_id.loc[(off == "") & (pnorm == hnorm)]
        off.loc[(off == "") & (pnorm == anorm)] = away_id.loc[(off == "") & (pnorm == anorm)]

        # Required native `is_home` is the final fallback.
        if "is_home" in df.columns:
            ih = to_bool_series(df["is_home"]).fillna(False)
            missing = off == ""
            off.loc[missing & ih] = home_id.loc[missing & ih]
            off.loc[missing & ~ih] = away_id.loc[missing & ~ih]

    if def_id_col is not None:
        deff = df[def_id_col].map(norm_id)
    else:
        home_id = df.get("homeTeamId", pd.Series("", index=df.index)).map(norm_id)
        away_id = df.get("awayTeamId", pd.Series("", index=df.index)).map(norm_id)
        deff = pd.Series("", index=df.index, dtype=object)
        deff.loc[off == home_id] = away_id.loc[off == home_id]
        deff.loc[off == away_id] = home_id.loc[off == away_id]
    return off.astype(str), deff.astype(str)


def _safe_div_series(numer: pd.Series, denom: pd.Series) -> pd.Series:
    n = pd.to_numeric(numer, errors="coerce").astype(float)
    d = pd.to_numeric(denom, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=n.index, dtype=float)
    ok = d > 0
    out.loc[ok] = n.loc[ok] / d.loc[ok]
    return out


def _extract_team_game_and_players(df: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    d = df.copy()
    d["game_id"] = d["game_id"].map(norm_id)
    d["week"] = pd.to_numeric(d.get("week"), errors="coerce")
    off_id, def_id = _assign_team_ids(d)
    d["off_id"] = off_id
    d["def_id"] = def_id
    d = d[(d["game_id"] != "") & (d["off_id"] != "") & d["week"].notna()].copy()
    d["week"] = d["week"].astype(int)

    epa = _num_col(d, ["EPA", "epa"])
    yards = _num_col(d, ["statYardage", "yards_gained", "yards"])
    down = _num_col(d, ["down"])
    ytg = _num_col(d, ["start.yardsToEndzone", "yards_to_goal", "start_yardsToEndzone"])

    passf = _bool_col(d, ["pass_flag", "pass", "pass_attempt"])
    rushf = _bool_col(d, ["rush_flag", "rush"])
    scrim = _bool_col(d, ["scrimmage_play"])
    if not scrim.any():
        scrim = passf | rushf
    sack = _bool_col(d, ["sack_flag", "sack", "sack_vec"])
    interception = _bool_col(d, ["interception_flag", "int", "interception"])
    fumble_lost = _bool_col(d, ["fumble_lost_flag", "fumble_lost"])
    success = _bool_col(d, ["EPA_success", "success"])
    touchdown = _bool_col(d, ["touchdown", "td_play", "offense_score_play"])
    third = down.eq(3)
    rz = ytg.le(20) & ytg.notna()

    d["scrim_n"] = scrim.astype(int)
    d["pass_n"] = passf.astype(int)
    d["rush_n"] = rushf.astype(int)
    d["epa_scrim_sum"] = epa.where(scrim, 0.0).fillna(0.0)
    d["epa_scrim_n"] = (scrim & epa.notna()).astype(int)
    d["pass_epa_sum"] = epa.where(passf, 0.0).fillna(0.0)
    d["pass_epa_n"] = (passf & epa.notna()).astype(int)
    d["rush_epa_sum"] = epa.where(rushf, 0.0).fillna(0.0)
    d["rush_epa_n"] = (rushf & epa.notna()).astype(int)
    d["pass_success_n"] = (passf & success).astype(int)
    d["rush_success_n"] = (rushf & success).astype(int)
    d["expl_pass_n"] = (passf & yards.ge(20)).astype(int)
    d["expl_rush_n"] = (rushf & yards.ge(10)).astype(int)
    d["sack_n"] = (passf & sack).astype(int)
    d["int_n"] = (passf & interception).astype(int)
    d["fumble_lost_n"] = (scrim & fumble_lost).astype(int)
    d["turnover_n"] = (scrim & (interception | fumble_lost)).astype(int)
    d["third_n"] = (scrim & third).astype(int)
    d["third_success_n"] = (scrim & third & success).astype(int)
    d["rz_n"] = (scrim & rz).astype(int)
    d["rz_success_n"] = (scrim & rz & success).astype(int)
    d["td_n"] = (scrim & touchdown).astype(int)

    drive_col = _first_existing(d, ["drive.id", "drive_id", "driveId"])
    if drive_col is not None:
        d["drive_key"] = d[drive_col].map(clean_text)
    else:
        d["drive_key"] = ""

    group_cols = ["game_id", "week", "off_id"]
    agg = d.groupby(group_cols, as_index=False).agg(
        opponent_id=("def_id", "first"),
        scrim=("scrim_n", "sum"),
        passes=("pass_n", "sum"),
        rushes=("rush_n", "sum"),
        epa_scrim_sum=("epa_scrim_sum", "sum"),
        epa_scrim_n=("epa_scrim_n", "sum"),
        pass_epa_sum=("pass_epa_sum", "sum"),
        pass_epa_n=("pass_epa_n", "sum"),
        rush_epa_sum=("rush_epa_sum", "sum"),
        rush_epa_n=("rush_epa_n", "sum"),
        pass_success_n=("pass_success_n", "sum"),
        rush_success_n=("rush_success_n", "sum"),
        expl_pass_n=("expl_pass_n", "sum"),
        expl_rush_n=("expl_rush_n", "sum"),
        sacks=("sack_n", "sum"),
        interceptions=("int_n", "sum"),
        fumbles_lost=("fumble_lost_n", "sum"),
        turnovers=("turnover_n", "sum"),
        third_downs=("third_n", "sum"),
        third_successes=("third_success_n", "sum"),
        rz_plays=("rz_n", "sum"),
        rz_successes=("rz_success_n", "sum"),
        touchdowns=("td_n", "sum"),
        drives=("drive_key", lambda x: x[x != ""].nunique()),
    )
    agg = agg.rename(columns={"off_id": "team_id"})
    agg["season"] = season

    agg["off_epa"] = _safe_div_series(agg["epa_scrim_sum"], agg["epa_scrim_n"])
    agg["off_pass_epa"] = _safe_div_series(agg["pass_epa_sum"], agg["pass_epa_n"])
    agg["off_rush_epa"] = _safe_div_series(agg["rush_epa_sum"], agg["rush_epa_n"])
    agg["off_pass_success"] = _safe_div_series(agg["pass_success_n"], agg["passes"])
    agg["off_rush_success"] = _safe_div_series(agg["rush_success_n"], agg["rushes"])
    agg["off_expl_pass"] = _safe_div_series(agg["expl_pass_n"], agg["passes"])
    agg["off_expl_rush"] = _safe_div_series(agg["expl_rush_n"], agg["rushes"])
    agg["off_sack_allowed"] = _safe_div_series(agg["sacks"], agg["passes"])
    agg["off_int_rate"] = _safe_div_series(agg["interceptions"], agg["passes"])
    agg["off_fumble_lost_rate"] = _safe_div_series(agg["fumbles_lost"], agg["scrim"])
    agg["off_turnover_rate"] = _safe_div_series(agg["turnovers"], agg["scrim"])
    agg["off_third_success"] = _safe_div_series(agg["third_successes"], agg["third_downs"])
    agg["off_rz_success"] = _safe_div_series(agg["rz_successes"], agg["rz_plays"])
    agg["off_td_rate"] = _safe_div_series(agg["touchdowns"], agg["scrim"])
    agg["off_pass_rate"] = _safe_div_series(agg["passes"], agg["scrim"])
    agg["off_plays"] = agg["scrim"].astype(float)
    agg["off_drives"] = agg["drives"].astype(float)

    off_metrics = [
        "off_epa", "off_pass_epa", "off_rush_epa", "off_pass_success", "off_rush_success",
        "off_expl_pass", "off_expl_rush", "off_sack_allowed", "off_int_rate",
        "off_fumble_lost_rate", "off_turnover_rate", "off_third_success", "off_rz_success",
        "off_td_rate", "off_pass_rate", "off_plays", "off_drives",
    ]
    opp = agg[["game_id", "team_id"] + off_metrics].copy()
    opp = opp.rename(columns={"team_id": "opponent_id", **{c: "def_" + c[4:] for c in off_metrics}})
    team_games = agg[["season", "week", "game_id", "team_id", "opponent_id"] + off_metrics].merge(
        opp, on=["game_id", "opponent_id"], how="left"
    )

    # Player usage long table.  IDs are preferred; names are a fallback.
    passer = _player_key(d, "passer")
    rusher = _player_key(d, "rusher")
    receiver = _player_key(d, "receiver")
    target = _bool_col(d, ["target", "target_flag"])
    if not target.any():
        target = passf & receiver.ne("")

    player_parts: list[pd.DataFrame] = []
    for role, key, mask in [
        ("pass", passer, passf),
        ("rush", rusher, rushf),
        ("recv", receiver, target),
    ]:
        p = pd.DataFrame({
            "season": season,
            "week": d["week"].to_numpy(),
            "team_id": d["off_id"].to_numpy(),
            "player": key.to_numpy(),
            "role": role,
            "epa": epa.to_numpy(),
            "use": mask.astype(int).to_numpy(),
        }, index=d.index)
        p = p[(p["use"] > 0) & p["player"].astype(str).ne("")].copy()
        if not p.empty:
            player_parts.append(p)
    if player_parts:
        players = pd.concat(player_parts, ignore_index=True)
        players = players.groupby(["season", "week", "team_id", "player", "role"], as_index=False).agg(
            usage=("use", "sum"),
            epa_sum=("epa", "sum"),
        )
    else:
        players = pd.DataFrame(columns=["season", "week", "team_id", "player", "role", "usage", "epa_sum"])

    coverage = {
        "season": season,
        "plays": len(d),
        "games": d["game_id"].nunique(),
        "team_games": len(team_games),
        "pass_plays": int(passf.sum()),
        "passer_identified": int((passf & passer.ne("")).sum()),
        "rush_plays": int(rushf.sum()),
        "rusher_identified": int((rushf & rusher.ne("")).sum()),
        "targets": int(target.sum()),
        "receiver_identified": int((target & receiver.ne("")).sum()),
    }
    return team_games, players, coverage


def build_pbp_feature_store(root: Path, prediction_season: int) -> PBPFeatureStore:
    team_games: dict[tuple[int, str], pd.DataFrame] = {}
    player_usage: dict[tuple[int, str], pd.DataFrame] = {}
    coverage_rows: list[dict[str, Any]] = []

    seasons = list(range(2021, prediction_season + 1))
    for season in seasons:
        path = root / "00_intake" / "pbp" / f"{season}_pbp.parquet"
        if not path.exists():
            continue
        print(f"Extracting focused PBP/player features: {season}...")
        raw = pd.read_parquet(path)
        tg, pu, cov = _extract_team_game_and_players(raw, season)
        coverage_rows.append(cov)
        for tid, grp in tg.groupby("team_id"):
            team_games[(season, norm_id(tid))] = grp.sort_values(["week", "game_id"]).reset_index(drop=True)
        if not pu.empty:
            for tid, grp in pu.groupby("team_id"):
                player_usage[(season, norm_id(tid))] = grp.sort_values(["week", "role", "player"]).reset_index(drop=True)
        pass_cov = cov["passer_identified"] / max(cov["pass_plays"], 1)
        rush_cov = cov["rusher_identified"] / max(cov["rush_plays"], 1)
        recv_cov = cov["receiver_identified"] / max(cov["targets"], 1)
        print(
            f"  team-games={len(tg)} player-rows={len(pu)} "
            f"ID coverage pass={pass_cov:.1%} rush={rush_cov:.1%} recv={recv_cov:.1%}"
        )
    return PBPFeatureStore(
        team_games=team_games,
        player_usage=player_usage,
        coverage=pd.DataFrame(coverage_rows),
    )


PBP_PROFILE_METRICS = [
    "off_epa", "off_pass_epa", "off_rush_epa", "off_pass_success", "off_rush_success",
    "off_expl_pass", "off_expl_rush", "off_sack_allowed", "off_int_rate",
    "off_fumble_lost_rate", "off_turnover_rate", "off_third_success", "off_rz_success",
    "off_td_rate", "off_pass_rate", "off_plays", "off_drives",
    "def_epa", "def_pass_epa", "def_rush_epa", "def_pass_success", "def_rush_success",
    "def_expl_pass", "def_expl_rush", "def_sack_allowed", "def_int_rate",
    "def_fumble_lost_rate", "def_turnover_rate", "def_third_success", "def_rz_success",
    "def_td_rate", "def_pass_rate", "def_plays", "def_drives",
]


def _profile_from_games(grp: pd.DataFrame, suffix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if grp is None or grp.empty:
        return out
    out[f"games_{suffix}"] = float(len(grp))
    for c in PBP_PROFILE_METRICS:
        if c not in grp.columns:
            continue
        x = pd.to_numeric(grp[c], errors="coerce")
        out[f"{c}_{suffix}"] = float(x.mean()) if x.notna().any() else float("nan")
    return out


def pregame_pbp_profile(store: PBPFeatureStore, season: int, team_id: str, game_week: int) -> dict[str, float]:
    tid = norm_id(team_id)
    out: dict[str, float] = {}
    current = store.team_games.get((season, tid))
    prior = None
    if current is not None and not current.empty:
        prior = current[pd.to_numeric(current["week"], errors="coerce") < game_week].copy()
    prev = store.team_games.get((season - 1, tid))

    if prior is not None and not prior.empty:
        out.update(_profile_from_games(prior, "mean"))
        out.update(_profile_from_games(prior.tail(3), "last3"))
        out["using_current_season"] = 1.0
    elif prev is not None and not prev.empty:
        # Week 1 / no current-season prior game: use previous season as the
        # active matchup profile, while retaining a source flag.
        out.update(_profile_from_games(prev, "mean"))
        out.update(_profile_from_games(prev.tail(3), "last3"))
        out["using_current_season"] = 0.0
    else:
        out["using_current_season"] = float("nan")

    if prev is not None and not prev.empty:
        out.update(_profile_from_games(prev, "prev"))
    return out


def _role_summary(df: pd.DataFrame, role: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["player", "usage", "epa_sum"])
    x = df[df["role"] == role].copy()
    if x.empty:
        return pd.DataFrame(columns=["player", "usage", "epa_sum"])
    return x.groupby("player", as_index=False).agg(usage=("usage", "sum"), epa_sum=("epa_sum", "sum"))


def _usage_hhi(role_df: pd.DataFrame) -> float:
    if role_df.empty:
        return float("nan")
    total = pd.to_numeric(role_df["usage"], errors="coerce").sum()
    if not math.isfinite(float(total)) or total <= 0:
        return float("nan")
    shares = pd.to_numeric(role_df["usage"], errors="coerce").fillna(0.0).to_numpy(float) / float(total)
    return float(np.sum(shares * shares))


def pregame_player_profile(store: PBPFeatureStore, season: int, team_id: str, game_week: int) -> dict[str, float]:
    tid = norm_id(team_id)
    prev_all = store.player_usage.get((season - 1, tid))
    cur_all = store.player_usage.get((season, tid))
    if prev_all is None or prev_all.empty:
        return {}
    cur = None
    if cur_all is not None and not cur_all.empty:
        cur = cur_all[pd.to_numeric(cur_all["week"], errors="coerce") < game_week].copy()

    out: dict[str, float] = {}
    for role in ["pass", "rush", "recv"]:
        prev = _role_summary(prev_all, role)
        current = _role_summary(cur, role) if cur is not None else pd.DataFrame(columns=prev.columns)
        if prev.empty:
            continue
        prev = prev.sort_values("usage", ascending=False).reset_index(drop=True)
        prev_total = float(pd.to_numeric(prev["usage"], errors="coerce").sum())
        if prev_total <= 0:
            continue
        prev_top = clean_text(prev.iloc[0]["player"])
        prev_top_share = float(prev.iloc[0]["usage"]) / prev_total
        out[f"{role}_prev_total_usage"] = prev_total
        out[f"{role}_prev_top_share"] = prev_top_share
        out[f"{role}_prev_hhi"] = _usage_hhi(prev)
        prev_epa = float(pd.to_numeric(prev["epa_sum"], errors="coerce").sum())
        out[f"{role}_prev_epa_per_use"] = prev_epa / prev_total if prev_total > 0 else float("nan")

        if current is None or current.empty:
            out[f"{role}_current_total_usage"] = float("nan")
            out[f"{role}_current_players"] = float("nan")
            out[f"{role}_current_top_share"] = float("nan")
            out[f"{role}_return_fraction"] = float("nan")
            out[f"{role}_prev_top_seen"] = float("nan")
            out[f"{role}_same_primary"] = float("nan")
            continue

        current = current.sort_values("usage", ascending=False).reset_index(drop=True)
        cur_total = float(pd.to_numeric(current["usage"], errors="coerce").sum())
        cur_players = set(current["player"].astype(str))
        current_top = clean_text(current.iloc[0]["player"])
        out[f"{role}_current_total_usage"] = cur_total
        out[f"{role}_current_players"] = float(len(cur_players))
        out[f"{role}_current_top_share"] = float(current.iloc[0]["usage"]) / cur_total if cur_total > 0 else float("nan")
        prev_returned = prev[prev["player"].astype(str).isin(cur_players)]
        returned_usage = float(pd.to_numeric(prev_returned["usage"], errors="coerce").sum())
        out[f"{role}_return_fraction"] = returned_usage / prev_total
        out[f"{role}_prev_top_seen"] = 1.0 if prev_top in cur_players else 0.0
        out[f"{role}_same_primary"] = 1.0 if current_top == prev_top else 0.0
        cur_epa = float(pd.to_numeric(current["epa_sum"], errors="coerce").sum())
        out[f"{role}_current_epa_per_use"] = cur_epa / cur_total if cur_total > 0 else float("nan")

    # Plain-language QB aliases make the important continuity variables easy to inspect.
    out["qb_prev_dominance"] = out.get("pass_prev_top_share", float("nan"))
    out["qb_return_seen"] = out.get("pass_prev_top_seen", float("nan"))
    out["qb_same_primary"] = out.get("pass_same_primary", float("nan"))
    out["qb_current_stability"] = out.get("pass_current_top_share", float("nan"))
    if math.isfinite(out.get("qb_same_primary", float("nan"))):
        out["qb_change_flag"] = 1.0 - out["qb_same_primary"]
    else:
        out["qb_change_flag"] = float("nan")
    return out


def pbp_matchup_features(own: dict[str, float], opp: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}

    def pair(name: str, off_key: str, def_key: str, *, lower_off_bad: bool = False) -> None:
        a = own.get(off_key, float("nan"))
        b = opp.get(def_key, float("nan"))
        if math.isfinite(a) and math.isfinite(b):
            out[f"match_{name}_mean"] = (a + b) / 2.0
            out[f"match_{name}_gap"] = a - b
            out[f"match_{name}_product"] = a * b
        else:
            out[f"match_{name}_mean"] = float("nan")
            out[f"match_{name}_gap"] = float("nan")
            out[f"match_{name}_product"] = float("nan")

    for suffix in ["mean", "last3"]:
        pair(f"pass_epa_{suffix}", f"off_pass_epa_{suffix}", f"def_pass_epa_{suffix}")
        pair(f"rush_epa_{suffix}", f"off_rush_epa_{suffix}", f"def_rush_epa_{suffix}")
        pair(f"pass_success_{suffix}", f"off_pass_success_{suffix}", f"def_pass_success_{suffix}")
        pair(f"rush_success_{suffix}", f"off_rush_success_{suffix}", f"def_rush_success_{suffix}")
        pair(f"expl_pass_{suffix}", f"off_expl_pass_{suffix}", f"def_expl_pass_{suffix}")
        pair(f"expl_rush_{suffix}", f"off_expl_rush_{suffix}", f"def_expl_rush_{suffix}")
        pair(f"third_{suffix}", f"off_third_success_{suffix}", f"def_third_success_{suffix}")
        pair(f"redzone_{suffix}", f"off_rz_success_{suffix}", f"def_rz_success_{suffix}")
        pair(f"td_rate_{suffix}", f"off_td_rate_{suffix}", f"def_td_rate_{suffix}")

        # Pressure/turnovers are rates where both a high offense value (allowed/committed)
        # and high opponent-defense value (generated/allowed in its opponent offense row)
        # indicate matchup stress; the model sees mean/gap/product rather than a hand-set weight.
        pair(f"sack_pressure_{suffix}", f"off_sack_allowed_{suffix}", f"def_sack_allowed_{suffix}")
        pair(f"int_pressure_{suffix}", f"off_int_rate_{suffix}", f"def_int_rate_{suffix}")
        pair(f"turnover_pressure_{suffix}", f"off_turnover_rate_{suffix}", f"def_turnover_rate_{suffix}")

        op = own.get(f"off_plays_{suffix}", float("nan"))
        dp = opp.get(f"def_plays_{suffix}", float("nan"))
        if math.isfinite(op) and math.isfinite(dp):
            out[f"match_plays_{suffix}"] = (op + dp) / 2.0
        od = own.get(f"off_drives_{suffix}", float("nan"))
        dd = opp.get(f"def_drives_{suffix}", float("nan"))
        if math.isfinite(od) and math.isfinite(dd):
            out[f"match_drives_{suffix}"] = (od + dd) / 2.0
    return out


def player_triplet(own: dict[str, float], opp: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = sorted(set(own) | set(opp))
    for k in keys:
        a = own.get(k, float("nan"))
        b = opp.get(k, float("nan"))
        out[f"own_player_{k}"] = a
        out[f"opp_player_{k}"] = b
        out[f"diff_player_{k}"] = a - b if math.isfinite(a) and math.isfinite(b) else float("nan")
    return out


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def model_specs(fast: bool) -> dict[str, Callable[[], Pipeline]]:
    def ridge(alpha: float) -> Pipeline:
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])

    def tree_pipe(model: Any) -> Pipeline:
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("model", model),
        ])

    specs: dict[str, Callable[[], Pipeline]] = {
        "ridge_10": lambda: ridge(10.0),
        "ridge_100": lambda: ridge(100.0),
        "ridge_1000": lambda: ridge(1000.0),
        "extra": lambda: tree_pipe(
            ExtraTreesRegressor(
                n_estimators=220 if fast else 550,
                max_features=0.65,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }
    if fast:
        return specs

    specs["rf"] = lambda: tree_pipe(
        RandomForestRegressor(
            n_estimators=450,
            max_features=0.60,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    specs["hist"] = lambda: tree_pipe(
        HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=450,
            max_leaf_nodes=15,
            l2_regularization=10.0,
            random_state=RANDOM_STATE,
        )
    )
    if XGBRegressor is not None:
        specs["xgb"] = lambda: tree_pipe(
            XGBRegressor(
                n_estimators=700,
                learning_rate=0.03,
                max_depth=3,
                min_child_weight=8,
                subsample=0.85,
                colsample_bytree=0.70,
                reg_alpha=1.0,
                reg_lambda=12.0,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )
        )
    if LGBMRegressor is not None:
        specs["lgbm"] = lambda: tree_pipe(
            LGBMRegressor(
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=35,
                subsample=0.85,
                colsample_bytree=0.70,
                reg_alpha=1.0,
                reg_lambda=12.0,
                objective="regression_l1",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            )
        )
    return specs


def candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in METADATA_COLUMNS:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        # Every model input is numeric; convert source fields defensively.
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().any():
            df[c] = conv
            cols.append(c)
    return cols


def freeze_features(df_pre2025: pd.DataFrame, cols: list[str]) -> list[str]:
    frozen: list[str] = []
    n = max(len(df_pre2025), 1)
    for c in cols:
        if c not in df_pre2025.columns:
            continue
        x = pd.to_numeric(df_pre2025[c], errors="coerce")
        # Availability-only filter; no outcome is used here.
        if x.notna().sum() < max(20, int(0.02 * n)):
            continue
        if x.dropna().nunique() <= 1:
            continue
        frozen.append(c)
    return frozen


def build_feature_groups(all_cols: list[str]) -> dict[str, list[str]]:
    context = [c for c in all_cols if c in {
        "team_is_home", "neutral_site", "week_num", "week_sin", "week_cos",
        "market_team_score", "market_opp_score", "market_team_margin", "market_total", "market_abs_margin",
        "has_preseason", "has_ratings", "has_fpi", "has_pbp_matchup", "has_player_continuity",
    }]
    espn = [c for c in all_cols if c.startswith("espn_")]
    pre = [c for c in all_cols if "_pre_" in c]
    rat = [c for c in all_cols if "_rat_" in c]
    fpi = [c for c in all_cols if "_fpi_" in c]
    pbp = [c for c in all_cols if "_pbp_" in c or c.startswith("match_")]
    player = [c for c in all_cols if "_player_" in c]

    def uniq(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    # Deliberately omit the enormous _sum_ block from every V5 model group.
    # V4 already showed that feeding all weekly summary columns did not generalize.
    return {
        "market_only": uniq(context),
        "market_espn": uniq(context + espn),
        "matchup": uniq(context + espn + pbp),
        "continuity": uniq(context + espn + pre + player),
        "matchup_continuity": uniq(context + espn + pre + pbp + player),
        "focused_sdv_matchup": uniq(context + espn + pre + rat + fpi + pbp),
        "focused_sdv_matchup_continuity": uniq(context + espn + pre + rat + fpi + pbp + player),
    }


def choose_alpha(y_actual: np.ndarray, base: np.ndarray, raw_resid: np.ndarray) -> tuple[float, float]:
    best_alpha = 0.0
    best_mae = mean_absolute_error(y_actual, base)
    for alpha in RESIDUAL_ALPHA_GRID:
        pred = np.clip(base + alpha * raw_resid, SCORE_MIN, SCORE_MAX)
        mae = mean_absolute_error(y_actual, pred)
        if mae < best_mae - 1e-12:
            best_mae = mae
            best_alpha = float(alpha)
    return best_alpha, best_mae


def forward_select(
    rows: pd.DataFrame,
    groups: dict[str, list[str]],
    specs: dict[str, Callable[[], Pipeline]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    lined = rows.dropna(subset=["actual_team_score", "target_score_residual", "market_team_score"]).copy()
    records: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for group_name, features in groups.items():
        if not features:
            continue
        for model_name, factory in specs.items():
            oof_parts: list[pd.DataFrame] = []
            failed = False
            for val_season in CV_SEASONS:
                train = lined[lined["season"] < val_season]
                val = lined[lined["season"] == val_season]
                if train.empty or val.empty:
                    failed = True
                    break
                try:
                    model = factory()
                    model.fit(train[features], train["target_score_residual"].astype(float))
                    raw = model.predict(val[features])
                except Exception as exc:
                    print(f"    failed {group_name}|{model_name}|{val_season}: {type(exc).__name__}: {exc}")
                    failed = True
                    break
                part = val[["season", "game_id", "team_side", "actual_team_score", "market_team_score"]].copy()
                part["raw_resid"] = raw
                oof_parts.append(part)
            if failed or not oof_parts:
                continue

            oof = pd.concat(oof_parts, ignore_index=True)
            y = oof["actual_team_score"].to_numpy(float)
            base = oof["market_team_score"].to_numpy(float)
            raw = oof["raw_resid"].to_numpy(float)
            alpha, model_mae = choose_alpha(y, base, raw)
            market_mae = mean_absolute_error(y, base)
            improvement = market_mae - model_mae
            fold_wins = 0
            fold_metrics: dict[int, tuple[float, float]] = {}
            for s in CV_SEASONS:
                fold = oof[oof["season"] == s]
                yy = fold["actual_team_score"].to_numpy(float)
                bb = fold["market_team_score"].to_numpy(float)
                rr = fold["raw_resid"].to_numpy(float)
                pm = mean_absolute_error(yy, np.clip(bb + alpha * rr, SCORE_MIN, SCORE_MAX))
                bm = mean_absolute_error(yy, bb)
                fold_metrics[s] = (pm, bm)
                if pm < bm:
                    fold_wins += 1

            rec: dict[str, Any] = {
                "feature_group": group_name,
                "model": model_name,
                "feature_count": len(features),
                "alpha": alpha,
                "cv_model_mae": model_mae,
                "cv_market_mae": market_mae,
                "cv_improvement": improvement,
                "fold_wins": fold_wins,
            }
            for s, (pm, bm) in fold_metrics.items():
                rec[f"{s}_model_mae"] = pm
                rec[f"{s}_market_mae"] = bm
                rec[f"{s}_improvement"] = bm - pm
            records.append(rec)
            print(
                f"  {group_name}|{model_name}: MAE={model_mae:.4f} "
                f"market={market_mae:.4f} improvement={improvement:+.4f} "
                f"alpha={alpha:.2f} wins={fold_wins}/3"
            )
            if best is None or model_mae < best["cv_model_mae"] - 1e-12:
                best = rec.copy()

    if best is None:
        raise RuntimeError("No model candidate completed forward validation.")
    best["cv_accepted"] = bool(
        best["cv_improvement"] >= CV_MIN_TEAM_SCORE_IMPROVEMENT
        and best["fold_wins"] >= CV_REQUIRED_FOLD_WINS
        and best["alpha"] > 0
    )
    return best, pd.DataFrame(records).sort_values("cv_model_mae").reset_index(drop=True)


def fit_model(factory: Callable[[], Pipeline], rows: pd.DataFrame, features: list[str], target: str) -> Pipeline:
    model = factory()
    model.fit(rows[features], rows[target].astype(float))
    return model


def team_to_game_predictions(team_rows: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    home = team_rows[team_rows["team_side"] == "home"].copy()
    away = team_rows[team_rows["team_side"] == "away"].copy()
    home = home[["game_id", pred_col]].rename(columns={pred_col: "pred_home_score"})
    away = away[["game_id", pred_col]].rename(columns={pred_col: "pred_away_score"})
    return home.merge(away, on="game_id", how="inner")


def game_metrics(games: pd.DataFrame, pred_home_col: str, pred_away_col: str) -> dict[str, float]:
    use = games.dropna(subset=["home_score", "away_score", pred_home_col, pred_away_col]).copy()
    ph = use[pred_home_col].to_numpy(float)
    pa = use[pred_away_col].to_numpy(float)
    ah = use["home_score"].to_numpy(float)
    aa = use["away_score"].to_numpy(float)
    pred_margin = ph - pa
    actual_margin = ah - aa
    pred_total = ph + pa
    actual_total = ah + aa
    non_tie = actual_margin != 0
    winner = float(np.mean(np.sign(pred_margin[non_tie]) == np.sign(actual_margin[non_tie]))) if non_tie.any() else float("nan")
    return {
        "games": int(len(use)),
        "team_score_mae": float((mean_absolute_error(ah, ph) + mean_absolute_error(aa, pa)) / 2.0),
        "home_score_mae": float(mean_absolute_error(ah, ph)),
        "away_score_mae": float(mean_absolute_error(aa, pa)),
        "margin_mae": float(mean_absolute_error(actual_margin, pred_margin)),
        "total_mae": float(mean_absolute_error(actual_total, pred_total)),
        "winner_accuracy": winner,
    }


def final_2025_validation(
    games: pd.DataFrame,
    team_rows: pd.DataFrame,
    best: dict[str, Any],
    groups: dict[str, list[str]],
    specs: dict[str, Callable[[], Pipeline]],
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features = groups[best["feature_group"]]
    train = team_rows[(team_rows["season"] < FINAL_SEASON)].dropna(
        subset=["target_score_residual", "market_team_score", "actual_team_score"]
    )
    val = team_rows[(team_rows["season"] == FINAL_SEASON)].dropna(
        subset=["target_score_residual", "market_team_score", "actual_team_score"]
    ).copy()
    model = fit_model(specs[best["model"]], train, features, "target_score_residual")
    raw = model.predict(val[features])
    alpha = float(best["alpha"])
    val["candidate_team_score"] = np.clip(
        val["market_team_score"].to_numpy(float) + alpha * raw,
        SCORE_MIN,
        SCORE_MAX,
    )

    pred_game = team_to_game_predictions(val, "candidate_team_score")
    vgames = games[games["season"] == FINAL_SEASON].copy().merge(pred_game, on="game_id", how="inner")
    vgames["market_pred_home"] = vgames["market_home_score"]
    vgames["market_pred_away"] = vgames["market_away_score"]

    cand = game_metrics(vgames, "pred_home_score", "pred_away_score")
    market = game_metrics(vgames, "market_pred_home", "market_pred_away")
    metrics = {
        "candidate": cand,
        "market": market,
        "improvement": {
            "team_score_mae": market["team_score_mae"] - cand["team_score_mae"],
            "margin_mae": market["margin_mae"] - cand["margin_mae"],
            "total_mae": market["total_mae"] - cand["total_mae"],
            "winner_accuracy": cand["winner_accuracy"] - market["winner_accuracy"],
        },
    }
    validated = bool(
        best["cv_accepted"]
        and metrics["improvement"]["team_score_mae"] >= FINAL_MIN_TEAM_SCORE_IMPROVEMENT
        and metrics["improvement"]["margin_mae"] >= -FINAL_MAX_MARGIN_DEGRADATION
        and metrics["improvement"]["total_mae"] >= -FINAL_MAX_TOTAL_DEGRADATION
        and metrics["improvement"]["winner_accuracy"] >= -FINAL_MAX_WINNER_ACCURACY_DEGRADATION
    )
    metrics["validated"] = validated

    out_cols = [
        "game_id", "game_date", "away_team", "home_team", "away_score", "home_score",
        "home_spread", "total", "market_home_score", "market_away_score",
        "pred_home_score", "pred_away_score",
    ]
    out_cols = [c for c in out_cols if c in vgames.columns]
    return model, val, vgames[out_cols].copy(), metrics


def direct_fallback_model(
    team_rows: pd.DataFrame,
    all_features: list[str],
) -> tuple[Pipeline, list[str]]:
    # No-market fallback is deliberately simple and separately labeled.  It is
    # not used to decide whether V5 beat the sportsbook.
    features = [
        c for c in all_features
        if not c.startswith("market_") and c not in {"market_team_score", "market_opp_score", "market_team_margin", "market_total", "market_abs_margin"}
    ]
    train = team_rows.dropna(subset=["actual_team_score"]).copy()
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=100.0)),
    ])
    model.fit(train[features], train["actual_team_score"].astype(float))
    return model, features


def predict_current(
    current: pd.DataFrame,
    current_rows: pd.DataFrame,
    residual_model: Pipeline,
    residual_features: list[str],
    alpha: float,
    direct_model: Pipeline,
    direct_features: list[str],
    validated: bool,
) -> pd.DataFrame:
    rows = current_rows.copy()
    has_market = rows["market_team_score"].notna()
    rows["candidate_team_score"] = np.nan
    if has_market.any():
        raw = residual_model.predict(rows.loc[has_market, residual_features])
        rows.loc[has_market, "candidate_team_score"] = np.clip(
            rows.loc[has_market, "market_team_score"].to_numpy(float) + alpha * raw,
            SCORE_MIN,
            SCORE_MAX,
        )
    missing_market = ~has_market
    if missing_market.any():
        direct = direct_model.predict(rows.loc[missing_market, direct_features])
        rows.loc[missing_market, "candidate_team_score"] = np.clip(direct, SCORE_MIN, SCORE_MAX)

    pred = team_to_game_predictions(rows, "candidate_team_score")
    out = current.merge(pred, on="game_id", how="left")
    out = out.rename(columns={"pred_home_score": "candidate_home_score", "pred_away_score": "candidate_away_score"})
    out["candidate_margin"] = out["candidate_home_score"] - out["candidate_away_score"]
    out["candidate_total"] = out["candidate_home_score"] + out["candidate_away_score"]

    line_ok = out["market_home_score"].notna() & out["market_away_score"].notna()
    if validated:
        out["deployed_home_score"] = out["candidate_home_score"]
        out["deployed_away_score"] = out["candidate_away_score"]
        out["deployment_source"] = np.where(line_ok, "VALIDATED_V5", "DIRECT_FALLBACK_NO_MARKET")
    else:
        out["deployed_home_score"] = np.where(line_ok, out["market_home_score"], out["candidate_home_score"])
        out["deployed_away_score"] = np.where(line_ok, out["market_away_score"], out["candidate_away_score"])
        out["deployment_source"] = np.where(line_ok, "MARKET_BASELINE_V5_REJECTED", "DIRECT_FALLBACK_NO_MARKET")

    out["deployed_margin"] = out["deployed_home_score"] - out["deployed_away_score"]
    out["deployed_total"] = out["deployed_home_score"] + out["deployed_away_score"]
    out["model_status"] = "VALIDATED" if validated else "REJECTED"
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    root = cfb_root()
    output = out_dir(root)
    cache_dir = output / "sdv_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    week = infer_predict_week(root, args.season, args.predict_week)

    print(f"CFB root: {root}")
    print(f"Prediction season/week: {args.season}/{week}")
    print("Loading SportsDataverse point-in-time/preseason assets...")
    stores, coverage = load_sdv_assets(
        cache_dir,
        args.season,
        refresh=args.refresh_sdv,
        no_download=args.no_download,
    )
    coverage_path = output / "sdv_source_coverage_v5.csv"
    coverage.to_csv(coverage_path, index=False)

    historical_ok = coverage[
        coverage["season"].isin(HISTORICAL_SEASONS)
        & coverage["dataset"].isin(["ratings_weekly", "team_summaries_weekly", "fpi_weekly", "team_talent", "returning_production"])
        & coverage["status"].eq("ok")
    ]
    if historical_ok.empty:
        raise RuntimeError(
            "No new historical SportsDataverse V5 assets were available. "
            "Run with internet access or populate data/score_model_v5/sdv_cache/."
        )

    print("Preparing SDV feature lookups...")
    feature_store = prepare_feature_store(stores)
    print(
        "  source columns: "
        f"pre={len(feature_store.source_columns['pre'])} "
        f"ratings={len(feature_store.source_columns['rat'])} "
        f"fpi={len(feature_store.source_columns['fpi'])} "
        f"summaries={len(feature_store.source_columns['sum'])}"
    )

    print("Building leakage-safe local PBP matchup/player feature store...")
    pbp_store = build_pbp_feature_store(root, args.season)
    player_cov_path = output / "player_pbp_coverage_v5.csv"
    pbp_store.coverage.to_csv(player_cov_path, index=False)

    print("Extracting historical final scores/team IDs from local PBP...")
    results = extract_historical_results(root)
    print(f"  completed games from PBP: {len(results)}")

    print("Building historical game table + sportsbook baseline...")
    games = load_historical_games(root, results)
    print(f"  regular-season matched games: {len(games)}")
    print(f"  games with spread+total: {int((games['home_spread'].notna() & games['total'].notna()).sum())}")

    print("Joining leakage-safe SDV + matchup/player snapshots and building team-score rows...")
    team_rows = build_team_rows(games, feature_store, pbp_store, include_targets=True)
    print(f"  historical team rows: {len(team_rows)}")

    all_cols = candidate_feature_columns(team_rows)
    pre2025 = team_rows[team_rows["season"] <= 2024]
    frozen = freeze_features(pre2025, all_cols)
    groups = build_feature_groups(frozen)
    print(f"  frozen numeric features: {len(frozen)}")
    for name, cols in groups.items():
        print(f"    {name}: {len(cols)}")

    specs = model_specs(args.fast)
    print(f"  model configurations: {len(specs)}")
    print("Selecting matchup/player team-score residual model with 2022-2024 forward validation...")
    best, cv = forward_select(team_rows, groups, specs)
    print(
        f"BEST: {best['feature_group']}|{best['model']} "
        f"alpha={best['alpha']:.2f} improvement={best['cv_improvement']:+.4f} "
        f"fold_wins={best['fold_wins']}/3 cv_accepted={best['cv_accepted']}"
    )

    cv_path = output / "cv_model_results_v5.csv"
    cv.to_csv(cv_path, index=False)

    print("Running frozen 2025 FINAL validation...")
    pre2025_model, val_rows, val_games, metrics = final_2025_validation(
        games, team_rows, best, groups, specs
    )
    cand = metrics["candidate"]
    market = metrics["market"]
    imp = metrics["improvement"]
    print(
        f"  team_score_mae: candidate={cand['team_score_mae']:.4f} "
        f"market={market['team_score_mae']:.4f} improvement={imp['team_score_mae']:+.4f}"
    )
    print(
        f"  margin_mae: candidate={cand['margin_mae']:.4f} "
        f"market={market['margin_mae']:.4f} improvement={imp['margin_mae']:+.4f}"
    )
    print(
        f"  total_mae: candidate={cand['total_mae']:.4f} "
        f"market={market['total_mae']:.4f} improvement={imp['total_mae']:+.4f}"
    )
    print(
        f"  winner_accuracy: candidate={cand['winner_accuracy']:.4f} "
        f"market={market['winner_accuracy']:.4f} delta={imp['winner_accuracy']:+.4f}"
    )
    print(f"  MODEL STATUS: {'VALIDATED' if metrics['validated'] else 'REJECTED'}")

    val_path = output / "validation_2025_predictions_v5.csv"
    val_games.to_csv(val_path, index=False)
    metrics_path = output / "validation_2025_metrics_v5.csv"
    metric_rows = []
    for metric in ["team_score_mae", "home_score_mae", "away_score_mae", "margin_mae", "total_mae", "winner_accuracy"]:
        metric_rows.append({
            "metric": metric,
            "candidate": cand.get(metric),
            "market": market.get(metric),
            "improvement": imp.get(metric, (cand.get(metric, np.nan) - market.get(metric, np.nan))),
        })
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)

    # Save training matrix efficiently; it is useful for auditing source joins.
    matrix_path = output / "training_team_rows_2021_2025_v5.parquet"
    team_rows.to_parquet(matrix_path, index=False)

    print("Refitting selected residual model on 2021-2025 for current predictions...")
    selected_features = groups[best["feature_group"]]
    lined_all = team_rows.dropna(subset=["target_score_residual", "market_team_score", "actual_team_score"])
    final_residual_model = fit_model(specs[best["model"]], lined_all, selected_features, "target_score_residual")

    direct_model, direct_features = direct_fallback_model(team_rows, frozen)

    alias_map = build_team_alias_map(root)
    current = current_games(root, args.season, week, alias_map)
    current_rows = build_team_rows(current, feature_store, pbp_store, include_targets=False)
    # Ensure every frozen column exists in current rows even when a 2026 source is unavailable.
    needed = list(dict.fromkeys(list(selected_features) + list(direct_features)))
    missing = [c for c in needed if c not in current_rows.columns]
    if missing:
        current_rows = pd.concat(
            [current_rows, pd.DataFrame(np.nan, index=current_rows.index, columns=missing)],
            axis=1,
        )
    current_rows[needed] = current_rows[needed].apply(pd.to_numeric, errors="coerce")

    predictions = predict_current(
        current,
        current_rows,
        final_residual_model,
        selected_features,
        float(best["alpha"]),
        direct_model,
        direct_features,
        bool(metrics["validated"]),
    )

    pred_path = output / f"week_{week}_CFB_trained_score_predictions_v5.csv"
    predictions.to_csv(pred_path, index=False)

    model_path = output / "score_model_v5.joblib"
    payload = {
        "script_version": SCRIPT_VERSION,
        "best": best,
        "validated_2025": bool(metrics["validated"]),
        "selected_features": selected_features,
        "residual_model": final_residual_model,
        "direct_features": direct_features,
        "direct_model": direct_model,
        "final_metrics": metrics,
    }
    joblib.dump(payload, model_path)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "historical_seasons": HISTORICAL_SEASONS,
        "cv_seasons": CV_SEASONS,
        "final_validation_season": FINAL_SEASON,
        "prediction_season": args.season,
        "prediction_week": week,
        "best_candidate": best,
        "final_2025": metrics,
        "model_status": "VALIDATED" if metrics["validated"] else "REJECTED",
        "frozen_feature_count": len(frozen),
        "selected_feature_count": len(selected_features),
        "source_feature_counts": {k: len(v) for k, v in feature_store.source_columns.items()},
        "pbp_player_coverage_rows": int(len(pbp_store.coverage)),
        "leakage_controls": {
            "weekly_ratings": "game week W uses through_week <= W-1; Week 1 uses prior-season final",
            "weekly_summaries": "game week W uses through_week <= W-1; Week 1 uses prior-season final",
            "fpi": "contemporaneous + in-sequence + asof_date strictly before game_date",
            "preseason": "same-season talent/returning/recruiting projection only",
            "local_pbp_matchup": "current season uses only weeks < game week; Week 1 falls back to prior-season final profile",
            "player_continuity": "current-game players are never used; current-season player set uses only weeks < game week",
            "2025": "not used in model/group/alpha selection",
        },
        "outputs": {
            "predictions": str(pred_path),
            "training_matrix": str(matrix_path),
            "cv": str(cv_path),
            "validation_predictions": str(val_path),
            "validation_metrics": str(metrics_path),
            "model": str(model_path),
            "sdv_coverage": str(coverage_path),
            "player_pbp_coverage": str(player_cov_path),
        },
    }
    manifest_path = output / "score_model_manifest_v5.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"Current predictions: {pred_path}")
    print(f"Training matrix: {matrix_path}")
    print(f"CV results: {cv_path}")
    print(f"2025 predictions: {val_path}")
    print(f"2025 metrics: {metrics_path}")
    print(f"Saved model: {model_path}")
    print(f"Manifest: {manifest_path}")
    print(f"SDV coverage: {coverage_path}")
    print(f"Player/PBP coverage: {player_cov_path}")


if __name__ == "__main__":
    main()

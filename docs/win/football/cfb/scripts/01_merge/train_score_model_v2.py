#!/usr/bin/env python3
"""
train_score_model_v2.py

A leakage-safe CFB score-model trainer that reads the existing repository and
writes ONLY under:

    docs/win/football/cfb/data/score_model_v2/

It never modifies the existing projection, selection, picks, config, intake,
or historical files.

What is different from the old trainer
--------------------------------------
1. Builds game-level efficiency features directly from the native PBP files.
2. Uses ALL prior games available at the start of each week, not only the most
   recent weekly team-stat row.
3. Blocks same-week result leakage: every game in week N is predicted from data
   through week N-1 only.
4. Carries prior-season performance into Week 1 explicitly and separately from
   current-season performance.
5. Builds first-order opponent-adjusted efficiency histories from prior games.
6. Tests feature groups and multiple model families using forward seasons only:
       train through 2021 -> validate 2022
       train through 2022 -> validate 2023
       train through 2023 -> validate 2024
7. Freezes model/feature/blend choices BEFORE examining 2025.
8. Uses 2025 only as the final truth test.
9. Calls a model VALIDATED only if it materially improves score accuracy. Tiny
   gains do not pass.
10. Historical point-in-time FPI and injury snapshots are not in the repo, so
    they are not fabricated into training. Current-only FPI/injuries are also
    excluded from the learned score model to preserve train/predict parity.

Historical inputs
-----------------
    docs/win/football/cfb/00_intake/schedule/{2021..2025}_schedule.csv
    docs/win/football/cfb/00_intake/pbp/{2021..2025}_pbp.parquet
    docs/win/football/cfb/data/historical_betting/cache/
        {season}_{provider}_espn_market_predictor.csv
    docs/win/football/cfb/data/historical_features/{season}_travel_weather.csv
    docs/win/football/cfb/config/mapping/team_map.csv

Optional current-week inputs
----------------------------
    docs/win/football/cfb/00_intake/schedule/weekly/week_{week}_CFB_weekly_schedule.csv
    docs/win/football/cfb/01_merge/week_{week}_CFB_enriched.csv
    docs/win/football/cfb/data/travel/{season}_week_{week}_travel.csv
    docs/win/football/cfb/data/weather/week_{week}_CFB_weekly_weather.csv
    docs/win/football/cfb/00_intake/pbp/{season}_pbp.parquet
        (completed PRIOR weeks only; target week is never used)

Outputs
-------
    docs/win/football/cfb/data/score_model_v2/pbp_team_game_features_2021_2025.csv
    docs/win/football/cfb/data/score_model_v2/training_games_2021_2025_v2.csv
    docs/win/football/cfb/data/score_model_v2/cv_model_results_v2.csv
    docs/win/football/cfb/data/score_model_v2/feature_group_results_v2.csv
    docs/win/football/cfb/data/score_model_v2/validation_2025_predictions_v2.csv
    docs/win/football/cfb/data/score_model_v2/validation_2025_metrics_v2.csv
    docs/win/football/cfb/data/score_model_v2/feature_columns_v2.csv
    docs/win/football/cfb/data/score_model_v2/score_model_v2.joblib
    docs/win/football/cfb/data/score_model_v2/score_model_manifest_v2.json
    docs/win/football/cfb/data/score_model_v2/week_{week}_CFB_trained_score_predictions_v2.csv
        (when --predict-week is supplied)

The model is deliberately conservative. If 2025 does not show a material gain,
it is marked REJECTED and lined current games remain on the market baseline.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
import unicodedata
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
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


SCRIPT_VERSION = "cfb-trained-score-v2-2026-08-29"
HISTORICAL_SEASONS = [2021, 2022, 2023, 2024, 2025]
CV_VALIDATION_SEASONS = [2022, 2023, 2024]
FINAL_VALIDATION_SEASON = 2025
SEASON_TYPE = 2
RANDOM_STATE = 20260829

PROVIDER_BY_SEASON = {
    2021: "draftkings",
    2022: "draftkings",
    2023: "draftkings",
    2024: "espnbet",
    2025: "espnbet",
}

# A candidate must show more than noise in forward-season testing before 2025.
CV_MIN_TARGET_MAE_IMPROVEMENT = 0.10
CV_REQUIRED_FOLD_WINS = 2
CV_MAX_SINGLE_FOLD_DEGRADATION = 0.35

# Final deployment standards. These are acceptance thresholds, not fitted
# hyperparameters. They prevent tiny gains from being called success.
FINAL_MIN_TARGET_MAE_IMPROVEMENT = 0.10
FINAL_MIN_TEAM_SCORE_MAE_IMPROVEMENT = 0.20
FINAL_MAX_OTHER_TARGET_DEGRADATION = 0.10
FINAL_MAX_WINNER_ACCURACY_DEGRADATION = 0.005

BLEND_GRID = np.round(np.arange(0.0, 1.51, 0.05), 2)

# Native columns guaranteed by the repo's pull_pbp.py / pull_team_stats.py.
PBP_REQUIRED = [
    "season",
    "week",
    "game_id",
    "sequenceNumber",
    "pos_team",
    "def_pos_team",
    "EPA",
    "EPA_success",
    "statYardage",
    "down",
    "start.yardsToEndzone",
    "start.homeScore",
    "start.awayScore",
    "end.homeScore",
    "end.awayScore",
    "drive.id",
    "first_down_created",
    "touchdown",
    "offense_score_play",
    "defense_score_play",
    "pass_td",
    "rush_td",
    "scrimmage_play",
    "is_home",
]

# Optional native SportsDataverse/cfbfastR fields. The script inspects the
# parquet schema and uses only fields actually present.
OPTIONAL_FLAG_ALIASES = {
    "pass_flag": ["pass", "pass_attempt", "qb_dropback", "dropback"],
    "rush_flag": ["rush", "rush_attempt"],
    "sack_flag": ["sack", "sack_play"],
    "interception_flag": ["interception", "interception_thrown", "int"],
    "fumble_lost_flag": ["fumble_lost", "fumble_lost_play"],
    "turnover_flag": ["turnover", "turnover_play"],
}

# Metrics stored for each team's game history. Defensive metrics are the
# opponent's corresponding offense metric in that game.
HISTORY_METRICS = [
    "points_for",
    "points_against",
    "margin",
    "total",
    "score_vs_market",
    "allowed_vs_market",
    "margin_vs_market",
    "total_vs_market",
    "off_epa",
    "def_epa_allowed",
    "off_success",
    "def_success_allowed",
    "off_ypp",
    "def_ypp_allowed",
    "off_explosive10",
    "def_explosive10_allowed",
    "off_explosive20",
    "def_explosive20_allowed",
    "off_negative_rate",
    "def_negative_rate_allowed",
    "off_first_down_rate",
    "def_first_down_rate_allowed",
    "off_early_down_epa",
    "def_early_down_epa_allowed",
    "off_third_down_rate",
    "def_third_down_rate_allowed",
    "off_redzone_td_rate",
    "def_redzone_td_rate_allowed",
    "off_points_per_drive",
    "def_points_per_drive_allowed",
    "off_scoring_drive_rate",
    "def_scoring_drive_rate_allowed",
    "off_plays_per_drive",
    "def_plays_per_drive_allowed",
    "off_drives",
    "off_pass_rate",
    "off_pass_epa",
    "def_pass_epa_allowed",
    "off_rush_epa",
    "def_rush_epa_allowed",
    "off_sack_rate",
    "def_sack_rate_allowed",
    "off_turnover_rate",
    "def_turnover_rate_allowed",
    # First-order opponent-adjusted metrics created at update time.
    "adj_off_epa",
    "adj_def_epa_allowed",
    "adj_off_success",
    "adj_def_success_allowed",
    "adj_off_ypp",
    "adj_def_ypp_allowed",
    "adj_off_ppd",
    "adj_def_ppd_allowed",
]

KEY_VARIABILITY_METRICS = {
    "points_for",
    "points_against",
    "margin",
    "total",
    "off_epa",
    "def_epa_allowed",
    "off_points_per_drive",
    "def_points_per_drive_allowed",
}

MATCHUP_PAIRS = [
    ("off_epa", "def_epa_allowed", "epa"),
    ("off_success", "def_success_allowed", "success"),
    ("off_ypp", "def_ypp_allowed", "ypp"),
    ("off_explosive10", "def_explosive10_allowed", "explosive10"),
    ("off_first_down_rate", "def_first_down_rate_allowed", "first_down"),
    ("off_early_down_epa", "def_early_down_epa_allowed", "early_down_epa"),
    ("off_third_down_rate", "def_third_down_rate_allowed", "third_down"),
    ("off_redzone_td_rate", "def_redzone_td_rate_allowed", "redzone"),
    ("off_points_per_drive", "def_points_per_drive_allowed", "ppd"),
    ("off_pass_epa", "def_pass_epa_allowed", "pass_epa"),
    ("off_rush_epa", "def_rush_epa_allowed", "rush_epa"),
    ("off_sack_rate", "def_sack_rate_allowed", "sack_rate"),
    ("off_turnover_rate", "def_turnover_rate_allowed", "turnover_rate"),
    ("adj_off_epa", "adj_def_epa_allowed", "adj_epa"),
    ("adj_off_success", "adj_def_success_allowed", "adj_success"),
    ("adj_off_ypp", "adj_def_ypp_allowed", "adj_ypp"),
    ("adj_off_ppd", "adj_def_ppd_allowed", "adj_ppd"),
]

WINDOWS = ("cur", "last3", "last5", "prev")

META_COLUMNS = {
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "home_final",
    "away_final",
    "actual_margin",
    "actual_total",
    "market_implied_home_score",
    "market_implied_away_score",
}

TARGET_COLUMNS = {
    "actual_margin",
    "actual_total",
    "home_final",
    "away_final",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate CFB score model v2.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--predict-week", type=int, default=None)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Smaller model search for troubleshooting. Full search is default.",
    )
    parser.add_argument(
        "--rebuild-pbp-features",
        action="store_true",
        help="Ignore the v2 PBP feature cache and rebuild it from parquet.",
    )
    return parser.parse_args()


def resolve_cfb_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "docs" / "win" / "football" / "cfb"
        if candidate.is_dir():
            return candidate
    for parent in here.parents:
        if parent.name == "cfb":
            return parent
    raise RuntimeError(f"Cannot resolve CFB root from {here}")


CFB_ROOT = resolve_cfb_root()
SCHEDULE_DIR = CFB_ROOT / "00_intake" / "schedule"
WEEKLY_SCHEDULE_DIR = SCHEDULE_DIR / "weekly"
PBP_DIR = CFB_ROOT / "00_intake" / "pbp"
HIST_CACHE_DIR = CFB_ROOT / "data" / "historical_betting" / "cache"
HIST_FEATURE_DIR = CFB_ROOT / "data" / "historical_features"
TRAVEL_DIR = CFB_ROOT / "data" / "travel"
WEATHER_DIR = CFB_ROOT / "data" / "weather"
ENRICHED_DIR = CFB_ROOT / "01_merge"
TEAM_MAP_PATH = CFB_ROOT / "config" / "mapping" / "team_map.csv"
OUT_DIR = CFB_ROOT / "data" / "score_model_v2"


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def gid(value: Any) -> str:
    text = clean(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def fnum(value: Any) -> float | None:
    text = clean(value).replace(",", "").replace("%", "")
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _finite_numbers(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
        if math.isfinite(number):
            out.append(number)
    return out


def safe_mean(values: Iterable[Any]) -> float | None:
    nums = _finite_numbers(values)
    return float(sum(nums) / len(nums)) if nums else None


def safe_std(values: Iterable[Any]) -> float | None:
    nums = _finite_numbers(values)
    if len(nums) < 2:
        return None
    arr = np.asarray(nums, dtype=float)
    return float(arr.std(ddof=0))


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    result = numeric.eq(1)
    unresolved = numeric.isna()
    if unresolved.any():
        text = series.astype(str).str.strip().str.casefold()
        mapped = text.isin({"true", "t", "yes", "y", "1"})
        result = result.where(~unresolved, mapped)
    return result.fillna(False).astype(bool)


def read_csv(path: Path, required: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"{path} missing columns: {missing}")
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if path.parent != OUT_DIR:
        raise RuntimeError(f"Refusing to write outside {OUT_DIR}: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_json(data: dict[str, Any], path: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if path.parent != OUT_DIR:
        raise RuntimeError(f"Refusing to write outside {OUT_DIR}: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


class TeamResolver:
    def __init__(self, path: Path):
        df = read_csv(path, ["canonical_team"])
        aliases: dict[str, set[str]] = defaultdict(set)
        candidate_columns = [
            c
            for c in [
                "canonical_team",
                "alias",
                "location",
                "shortDisplayName",
                "team_slug",
                "team_name",
                "nickname",
            ]
            if c in df.columns
        ]
        for _, row in df.iterrows():
            canonical = clean(row.get("canonical_team"))
            if not canonical:
                continue
            for col in candidate_columns:
                key = norm(row.get(col))
                if key:
                    aliases[key].add(canonical)
        self.mapping = {
            key: next(iter(values)) for key, values in aliases.items() if len(values) == 1
        }

    def resolve(self, value: Any) -> str:
        text = clean(value)
        return self.mapping.get(norm(text), text)


def american_implied(odds: Any) -> float | None:
    o = fnum(odds)
    if o is None or (-100 < o < 100):
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def load_schedule(season: int) -> pd.DataFrame:
    path = SCHEDULE_DIR / f"{season}_schedule.csv"
    df = read_csv(
        path,
        ["season", "season_type", "week", "game_id", "away_team", "home_team"],
    ).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["season_type"] = pd.to_numeric(df["season_type"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["game_id"] = df["game_id"].map(gid)
    df = df[
        df["season"].eq(season)
        & df["season_type"].eq(SEASON_TYPE)
        & df["week"].notna()
        & df["game_id"].ne("")
    ].copy()
    df["week"] = df["week"].astype(int)
    if df["game_id"].duplicated().any():
        raise RuntimeError(f"Duplicate game_id in {path}")
    return df


def load_historical_market(season: int) -> pd.DataFrame:
    provider = PROVIDER_BY_SEASON[season]
    path = HIST_CACHE_DIR / f"{season}_{provider}_espn_market_predictor.csv"
    df = read_csv(path, ["game_id"]).copy()
    df["game_id"] = df["game_id"].map(gid)
    return df.drop_duplicates("game_id", keep="last")


def load_historical_travel_weather(season: int) -> pd.DataFrame:
    path = HIST_FEATURE_DIR / f"{season}_travel_weather.csv"
    df = read_csv(path, ["game_id"]).copy()
    df["game_id"] = df["game_id"].map(gid)
    return df.drop_duplicates("game_id", keep="last")


def first_present(columns: set[str], aliases: list[str]) -> str | None:
    for name in aliases:
        if name in columns:
            return name
    return None


def parquet_columns(path: Path) -> set[str]:
    # Prefer pyarrow metadata because it avoids loading the file just to inspect
    # columns. Fall back to pandas if pyarrow metadata access is unavailable.
    try:
        import pyarrow.parquet as pq  # type: ignore
        return set(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return set(pd.read_parquet(path).columns)


def pbp_read_columns(path: Path) -> tuple[list[str], dict[str, str]]:
    columns = parquet_columns(path)
    missing = [c for c in PBP_REQUIRED if c not in columns]
    if missing:
        raise RuntimeError(f"{path} missing required native PBP columns: {missing}")
    aliases: dict[str, str] = {}
    selected = list(PBP_REQUIRED)
    for standard, candidates in OPTIONAL_FLAG_ALIASES.items():
        source = first_present(columns, candidates)
        if source is not None:
            aliases[standard] = source
            if source not in selected:
                selected.append(source)
    return selected, aliases


def extract_side_offense_summary(
    df: pd.DataFrame,
    optional_aliases: dict[str, str],
) -> pd.DataFrame:
    work = df.copy()
    work["game_id"] = work["game_id"].map(gid)
    work["sequenceNumber"] = pd.to_numeric(work["sequenceNumber"], errors="coerce")
    work["is_home_flag"] = bool_series(work["is_home"])
    work["scrimmage_flag"] = bool_series(work["scrimmage_play"])
    work["epa"] = pd.to_numeric(work["EPA"], errors="coerce")
    work["success"] = pd.to_numeric(work["EPA_success"], errors="coerce")
    work["yards"] = pd.to_numeric(work["statYardage"], errors="coerce")
    work["down_num"] = pd.to_numeric(work["down"], errors="coerce")
    work["yardline_100"] = pd.to_numeric(work["start.yardsToEndzone"], errors="coerce")
    work["first_down_flag"] = pd.to_numeric(work["first_down_created"], errors="coerce").fillna(0).gt(0)
    work["touchdown_flag"] = bool_series(work["touchdown"])
    work["offense_score_flag"] = bool_series(work["offense_score_play"])

    for standard, source in optional_aliases.items():
        work[standard] = bool_series(work[source])
    for standard in OPTIONAL_FLAG_ALIASES:
        if standard not in work.columns:
            work[standard] = False

    # Build a robust turnover flag from any native turnover indicators present.
    work["turnover_any"] = (
        work["turnover_flag"]
        | work["interception_flag"]
        | work["fumble_lost_flag"]
    )

    valid = work[
        work["scrimmage_flag"]
        & work["game_id"].ne("")
        & work["epa"].notna()
    ].copy()
    if valid.empty:
        raise RuntimeError("No valid scrimmage plays found in PBP")

    valid["explosive10"] = valid["yards"].ge(10).astype(float)
    valid["explosive20"] = valid["yards"].ge(20).astype(float)
    valid["negative"] = valid["yards"].lt(0).astype(float)
    valid["first_down_num"] = valid["first_down_flag"].astype(float)
    valid["early_epa_value"] = valid["epa"].where(valid["down_num"].isin([1, 2]))
    valid["third_conversion"] = valid["first_down_num"].where(valid["down_num"].eq(3))
    valid["pass_epa_value"] = valid["epa"].where(valid["pass_flag"])
    valid["rush_epa_value"] = valid["epa"].where(valid["rush_flag"])
    valid["pass_num"] = valid["pass_flag"].astype(float)
    valid["sack_num"] = valid["sack_flag"].astype(float)
    valid["turnover_num"] = valid["turnover_any"].astype(float)

    keys = ["game_id", "is_home_flag"]
    side = (
        valid.groupby(keys, dropna=False)
        .agg(
            off_plays=("epa", "size"),
            off_epa=("epa", "mean"),
            off_epa_std=("epa", "std"),
            off_success=("success", "mean"),
            off_ypp=("yards", "mean"),
            off_explosive10=("explosive10", "mean"),
            off_explosive20=("explosive20", "mean"),
            off_negative_rate=("negative", "mean"),
            off_first_down_rate=("first_down_num", "mean"),
            off_early_down_epa=("early_epa_value", "mean"),
            off_third_down_rate=("third_conversion", "mean"),
            off_pass_rate=("pass_num", "mean"),
            off_pass_epa=("pass_epa_value", "mean"),
            off_rush_epa=("rush_epa_value", "mean"),
            off_sack_rate=("sack_num", "mean"),
            off_turnover_rate=("turnover_num", "mean"),
        )
        .reset_index()
    )

    # Drive-level scoring and red-zone features.
    drives = work[
        work["game_id"].ne("")
        & work["drive.id"].notna()
    ].copy()
    drives["start_home"] = pd.to_numeric(drives["start.homeScore"], errors="coerce")
    drives["start_away"] = pd.to_numeric(drives["start.awayScore"], errors="coerce")
    drives["end_home"] = pd.to_numeric(drives["end.homeScore"], errors="coerce")
    drives["end_away"] = pd.to_numeric(drives["end.awayScore"], errors="coerce")
    drives["off_start_score"] = np.where(
        drives["is_home_flag"], drives["start_home"], drives["start_away"]
    )
    drives["off_end_score"] = np.where(
        drives["is_home_flag"], drives["end_home"], drives["end_away"]
    )
    drives = drives.sort_values(
        ["game_id", "is_home_flag", "drive.id", "sequenceNumber"], kind="stable"
    )
    dkeys = ["game_id", "is_home_flag", "drive.id"]
    drive_summary = (
        drives.groupby(dkeys, dropna=False)
        .agg(
            drive_start_score=("off_start_score", "first"),
            drive_end_score=("off_end_score", "last"),
            min_yardline=("yardline_100", "min"),
            scrimmage_plays=("scrimmage_flag", "sum"),
            offense_td=("touchdown_flag", lambda s: bool(s.any())),
        )
        .reset_index()
    )
    drive_summary["drive_points"] = (
        pd.to_numeric(drive_summary["drive_end_score"], errors="coerce")
        - pd.to_numeric(drive_summary["drive_start_score"], errors="coerce")
    )
    drive_summary.loc[drive_summary["drive_points"].lt(0), "drive_points"] = 0.0
    drive_summary.loc[drive_summary["drive_points"].gt(8), "drive_points"] = 8.0
    drive_summary["scoring_drive"] = drive_summary["drive_points"].gt(0).astype(float)
    drive_summary["redzone_drive"] = pd.to_numeric(
        drive_summary["min_yardline"], errors="coerce"
    ).le(20)
    drive_summary["redzone_td"] = (
        drive_summary["redzone_drive"] & drive_summary["drive_points"].ge(6)
    ).astype(float)

    drive_agg = (
        drive_summary.groupby(keys, dropna=False)
        .agg(
            off_drives=("drive.id", "nunique"),
            off_points_per_drive=("drive_points", "mean"),
            off_scoring_drive_rate=("scoring_drive", "mean"),
            off_plays_per_drive=("scrimmage_plays", "mean"),
        )
        .reset_index()
    )
    rz = drive_summary[drive_summary["redzone_drive"]].copy()
    if rz.empty:
        rz_agg = pd.DataFrame(columns=keys + ["off_redzone_td_rate"])
    else:
        rz_agg = (
            rz.groupby(keys, dropna=False)
            .agg(off_redzone_td_rate=("redzone_td", "mean"))
            .reset_index()
        )
    side = side.merge(drive_agg, on=keys, how="left").merge(rz_agg, on=keys, how="left")

    # Optional native flags are not guaranteed in every historical parquet.
    # Missing schema fields must stay missing, not masquerade as zero rates.
    if "pass_flag" not in optional_aliases:
        side["off_pass_rate"] = np.nan
        side["off_pass_epa"] = np.nan
    if "rush_flag" not in optional_aliases:
        side["off_rush_epa"] = np.nan
    if "sack_flag" not in optional_aliases:
        side["off_sack_rate"] = np.nan
    if not any(
        name in optional_aliases
        for name in ("turnover_flag", "interception_flag", "fumble_lost_flag")
    ):
        side["off_turnover_rate"] = np.nan
    return side


def extract_finals(df: pd.DataFrame) -> pd.DataFrame:
    work = df[["game_id", "sequenceNumber", "end.homeScore", "end.awayScore"]].copy()
    work["game_id"] = work["game_id"].map(gid)
    work["sequenceNumber"] = pd.to_numeric(work["sequenceNumber"], errors="coerce")
    work["home_final"] = pd.to_numeric(work["end.homeScore"], errors="coerce")
    work["away_final"] = pd.to_numeric(work["end.awayScore"], errors="coerce")
    work = work.dropna(subset=["home_final", "away_final"]).sort_values(
        ["game_id", "sequenceNumber"], kind="stable"
    )
    return (
        work.groupby("game_id", as_index=False)
        .tail(1)[["game_id", "home_final", "away_final"]]
        .drop_duplicates("game_id", keep="last")
    )


def build_team_game_features_for_season(
    season: int,
    schedule: pd.DataFrame,
    resolver: TeamResolver,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    path = PBP_DIR / f"{season}_pbp.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    read_columns, aliases = pbp_read_columns(path)
    df = pd.read_parquet(path, columns=read_columns)
    side = extract_side_offense_summary(df, aliases)
    finals = extract_finals(df)

    schedule_lookup = schedule[["game_id", "home_team", "away_team", "week"]].copy()
    schedule_lookup["home_team"] = schedule_lookup["home_team"].map(resolver.resolve)
    schedule_lookup["away_team"] = schedule_lookup["away_team"].map(resolver.resolve)

    home = side[side["is_home_flag"].eq(True)].drop(columns=["is_home_flag"])
    away = side[side["is_home_flag"].eq(False)].drop(columns=["is_home_flag"])
    home = home.add_prefix("home_").rename(columns={"home_game_id": "game_id"})
    away = away.add_prefix("away_").rename(columns={"away_game_id": "game_id"})
    games = schedule_lookup.merge(home, on="game_id", how="left").merge(
        away, on="game_id", how="left"
    ).merge(finals, on="game_id", how="inner")

    rows: list[dict[str, Any]] = []
    off_metric_names = [c[len("home_"):] for c in home.columns if c.startswith("home_off_")]
    for _, g in games.iterrows():
        for is_home in (True, False):
            side_prefix = "home" if is_home else "away"
            opp_prefix = "away" if is_home else "home"
            team = g["home_team"] if is_home else g["away_team"]
            opponent = g["away_team"] if is_home else g["home_team"]
            row: dict[str, Any] = {
                "season": season,
                "week": int(g["week"]),
                "game_id": g["game_id"],
                "team": team,
                "opponent": opponent,
                "is_home": int(is_home),
                "points_for": float(g["home_final"] if is_home else g["away_final"]),
                "points_against": float(g["away_final"] if is_home else g["home_final"]),
            }
            for metric in off_metric_names:
                row[metric] = fnum(g.get(f"{side_prefix}_{metric}"))
                # Convert opponent offense into this team's defensive allowed metric.
                if metric.startswith("off_"):
                    def_name = "def_" + metric[len("off_"):] + "_allowed"
                    row[def_name] = fnum(g.get(f"{opp_prefix}_{metric}"))
            rows.append(row)
    team_games = pd.DataFrame(rows)
    return team_games, finals, aliases


def pbp_cache_signature(seasons: list[int]) -> dict[str, Any]:
    sig: dict[str, Any] = {}
    for season in seasons:
        path = PBP_DIR / f"{season}_pbp.parquet"
        if path.is_file():
            stat = path.stat()
            sig[str(season)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return sig


def load_or_build_historical_team_games(
    resolver: TeamResolver,
    schedules: dict[int, pd.DataFrame],
    rebuild: bool,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[str, dict[str, str]]]:
    cache_path = OUT_DIR / "pbp_team_game_features_2021_2025.csv"
    cache_meta = OUT_DIR / "pbp_team_game_features_manifest_v2.json"
    current_sig = pbp_cache_signature(HISTORICAL_SEASONS)

    if not rebuild and cache_path.is_file() and cache_meta.is_file():
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            if meta.get("pbp_signature") == current_sig and meta.get("script_version") == SCRIPT_VERSION:
                cached = read_csv(cache_path)
                cached["game_id"] = cached["game_id"].map(gid)
                cached["season"] = pd.to_numeric(cached["season"], errors="coerce")
                cached["week"] = pd.to_numeric(cached["week"], errors="coerce")
                finals_by_season: dict[int, pd.DataFrame] = {}
                for season in HISTORICAL_SEASONS:
                    s = cached[pd.to_numeric(cached["season"], errors="coerce").eq(season)]
                    g = s[s["is_home"].astype(str).isin({"1", "1.0", "True", "true"})].copy()
                    if g.empty:
                        # is_home normally reads back numeric.
                        g = s[pd.to_numeric(s["is_home"], errors="coerce").eq(1)].copy()
                    finals_by_season[season] = g[["game_id", "points_for", "points_against"]].rename(
                        columns={"points_for": "home_final", "points_against": "away_final"}
                    )
                return cached, finals_by_season, meta.get("optional_aliases", {})
        except Exception:
            pass

    frames: list[pd.DataFrame] = []
    finals_by_season: dict[int, pd.DataFrame] = {}
    alias_manifest: dict[str, dict[str, str]] = {}
    for season in HISTORICAL_SEASONS:
        print(f"Extracting PBP game features: {season}...")
        team_games, finals, aliases = build_team_game_features_for_season(
            season, schedules[season], resolver
        )
        frames.append(team_games)
        finals_by_season[season] = finals
        alias_manifest[str(season)] = aliases
        print(
            f"  team-game rows={len(team_games)} games={team_games['game_id'].nunique()} "
            f"optional_flags={sorted(aliases)}"
        )
    combined = pd.concat(frames, ignore_index=True)
    write_csv(combined, cache_path)
    write_json(
        {
            "script_version": SCRIPT_VERSION,
            "pbp_signature": current_sig,
            "optional_aliases": alias_manifest,
        },
        cache_meta,
    )
    return combined, finals_by_season, alias_manifest


@dataclass
class TeamHistory:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def subset(self, season: int | None = None, last_n: int | None = None) -> list[dict[str, Any]]:
        data = self.rows
        if season is not None:
            data = [r for r in data if int(r["season"]) == int(season)]
        if last_n is not None:
            data = data[-last_n:]
        return data

    def last_week(self) -> tuple[int, int] | None:
        if not self.rows:
            return None
        r = self.rows[-1]
        return int(r["season"]), int(r["week"])


def metric_mean(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [r.get(metric) for r in rows]
    return safe_mean(values)


def history_snapshot(history: TeamHistory, season: int, week: int) -> dict[str, Any]:
    current = history.subset(season=season)
    previous = history.subset(season=season - 1)
    last3 = history.subset(last_n=3)
    last5 = history.subset(last_n=5)
    snap: dict[str, Any] = {
        "games_current": float(len(current)),
        "games_previous": float(len(previous)),
        "games_history": float(len(history.rows)),
    }
    last = history.last_week()
    if last is None:
        snap["weeks_since_last_game"] = None
    elif last[0] == season:
        snap["weeks_since_last_game"] = float(max(0, week - last[1]))
    else:
        snap["weeks_since_last_game"] = None

    for metric in HISTORY_METRICS:
        snap[f"cur_{metric}"] = metric_mean(current, metric)
        snap[f"last3_{metric}"] = metric_mean(last3, metric)
        snap[f"last5_{metric}"] = metric_mean(last5, metric)
        snap[f"prev_{metric}"] = metric_mean(previous, metric)
        if metric in KEY_VARIABILITY_METRICS:
            snap[f"cur_sd_{metric}"] = safe_std([r.get(metric) for r in current])
            snap[f"last5_sd_{metric}"] = safe_std([r.get(metric) for r in last5])
    return snap


def add_history_pair(
    features: dict[str, Any],
    home_snap: dict[str, Any],
    away_snap: dict[str, Any],
) -> None:
    keys = sorted(set(home_snap) | set(away_snap))
    for key in keys:
        hv = fnum(home_snap.get(key))
        av = fnum(away_snap.get(key))
        features[f"home_hist_{key}"] = hv
        features[f"away_hist_{key}"] = av
        features[f"diff_hist_{key}"] = None if hv is None or av is None else hv - av

    # Explicit offense-vs-opponent-defense matchup features for each time window.
    for window in WINDOWS:
        for offense_metric, defense_metric, label in MATCHUP_PAIRS:
            h_off = fnum(home_snap.get(f"{window}_{offense_metric}"))
            a_off = fnum(away_snap.get(f"{window}_{offense_metric}"))
            h_def = fnum(home_snap.get(f"{window}_{defense_metric}"))
            a_def = fnum(away_snap.get(f"{window}_{defense_metric}"))
            home_match = None if h_off is None or a_def is None else h_off - a_def
            away_match = None if a_off is None or h_def is None else a_off - h_def
            features[f"matchup_{window}_home_{label}"] = home_match
            features[f"matchup_{window}_away_{label}"] = away_match
            features[f"matchup_{window}_diff_{label}"] = (
                None if home_match is None or away_match is None else home_match - away_match
            )


def market_features(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    home_spread = fnum(row.get("home_spread"))
    total = fnum(row.get("total"))
    home_ml = fnum(row.get("home_moneyline_american"))
    away_ml = fnum(row.get("away_moneyline_american"))
    market_margin = None if home_spread is None else -home_spread
    return {
        "market_margin": market_margin,
        "market_total": total,
        "market_abs_margin": None if market_margin is None else abs(market_margin),
        "market_home_ml_implied": american_implied(home_ml),
        "market_away_ml_implied": american_implied(away_ml),
        "market_home_spread_price": fnum(row.get("home_spread_american")),
        "market_away_spread_price": fnum(row.get("away_spread_american")),
        "market_over_price": fnum(row.get("over_american")),
        "market_under_price": fnum(row.get("under_american")),
        "market_spread_available": 1.0 if home_spread is not None else 0.0,
        "market_total_available": 1.0 if total is not None else 0.0,
    }


def espn_features(row: pd.Series | dict[str, Any], historical: bool) -> dict[str, Any]:
    if historical:
        home_prob = fnum(row.get("espn_home_game_projection"))
        away_prob = fnum(row.get("espn_away_game_projection"))
    else:
        home_prob = fnum(row.get("espn_home_prob"))
        away_prob = fnum(row.get("espn_away_prob"))
        if home_prob is None:
            home_prob = fnum(row.get("espn_home_game_projection"))
        if away_prob is None:
            away_prob = fnum(row.get("espn_away_game_projection"))
    hdiff = fnum(row.get("espn_home_ptdiff"))
    adiff = fnum(row.get("espn_away_ptdiff"))
    return {
        "espn_home_ptdiff": hdiff,
        "espn_away_ptdiff": adiff,
        "espn_home_probability": home_prob,
        "espn_away_probability": away_prob,
        "espn_available": 1.0 if hdiff is not None or home_prob is not None else 0.0,
    }


def context_features(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    neutral = fnum(row.get("neutral_site"))
    if neutral is None:
        neutral = fnum(row.get("neutral_site_flag"))
    surface = norm(row.get("surface"))
    roof = norm(row.get("roof") or row.get("roof_type"))
    date = pd.to_datetime(clean(row.get("game_date")), errors="coerce")
    return {
        "context_week": fnum(row.get("week")),
        "context_neutral": neutral,
        "context_grass": 1.0 if "grass" in surface else (0.0 if surface else None),
        "context_turf": 1.0 if "turf" in surface or "artificial" in surface else (0.0 if surface else None),
        "context_open_air": 1.0 if "open" in roof else (0.0 if roof else None),
        "context_dome": 1.0 if "dome" in roof or "indoor" in roof or "closed" in roof else (0.0 if roof else None),
        "context_month": float(date.month) if not pd.isna(date) else None,
    }


def travel_weather_features(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    # Historical files and current travel/weather files use the same core names.
    fields = [
        "away_miles_traveled",
        "home_miles_traveled",
        "away_time_zone_change_hours",
        "home_time_zone_change_hours",
        "away_time_zones_crossed",
        "home_time_zones_crossed",
        "away_east_to_west",
        "away_west_to_east",
        "home_east_to_west",
        "home_west_to_east",
        "international_flag",
        "temperature",
        "wind_speed",
        "wind_gust",
        "precip_probability",
        "precipitation",
        "rain_flag",
        "snow_flag",
        "humidity",
        "dome_flag",
        "retractable_roof_flag",
        "open_air_flag",
    ]
    out = {f"tw_{name}": fnum(row.get(name)) for name in fields}
    away_miles = fnum(row.get("away_miles_traveled"))
    home_miles = fnum(row.get("home_miles_traveled"))
    out["tw_net_miles_1000"] = (
        None if away_miles is None or home_miles is None else (away_miles - home_miles) / 1000.0
    )
    away_tz = fnum(row.get("away_time_zones_crossed"))
    home_tz = fnum(row.get("home_time_zones_crossed"))
    out["tw_net_time_zones"] = None if away_tz is None or home_tz is None else away_tz - home_tz
    return out


def disagreement_features(features: dict[str, Any]) -> None:
    market_margin = fnum(features.get("market_margin"))
    market_total = fnum(features.get("market_total"))
    espn_margin = fnum(features.get("espn_home_ptdiff"))
    features["disagree_espn_market_margin"] = (
        None if espn_margin is None or market_margin is None else espn_margin - market_margin
    )
    # PBP form-vs-market disagreement is already represented by historical
    # market residual histories; expose a few direct summary differences.
    h_margin_form = fnum(features.get("home_hist_cur_margin"))
    a_margin_form = fnum(features.get("away_hist_cur_margin"))
    if h_margin_form is not None and a_margin_form is not None and market_margin is not None:
        features["disagree_form_market_margin"] = (h_margin_form - a_margin_form) - market_margin
    else:
        features["disagree_form_market_margin"] = None
    h_total = fnum(features.get("home_hist_cur_total"))
    a_total = fnum(features.get("away_hist_cur_total"))
    if h_total is not None and a_total is not None and market_total is not None:
        features["disagree_form_market_total"] = ((h_total + a_total) / 2.0) - market_total
    else:
        features["disagree_form_market_total"] = None


def pregame_feature_row(
    game: pd.Series | dict[str, Any],
    home_snap: dict[str, Any],
    away_snap: dict[str, Any],
    historical: bool,
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    features.update(market_features(game))
    features.update(espn_features(game, historical=historical))
    features.update(context_features(game))
    features.update(travel_weather_features(game))
    add_history_pair(features, home_snap, away_snap)
    disagreement_features(features)
    return features


def make_team_performance(
    team_row: pd.Series,
    opponent_row: pd.Series,
    game: pd.Series,
    team_is_home: bool,
    own_snap: dict[str, Any],
    opp_snap: dict[str, Any],
) -> dict[str, Any]:
    points_for = float(team_row["points_for"])
    points_against = float(team_row["points_against"])
    margin = points_for - points_against
    total_actual = points_for + points_against
    market_margin = fnum(game.get("market_margin"))
    market_total = fnum(game.get("market_total"))
    team_market_margin = market_margin if team_is_home else (None if market_margin is None else -market_margin)
    expected_team_score = None
    expected_opp_score = None
    if team_market_margin is not None and market_total is not None:
        expected_team_score = (market_total + team_market_margin) / 2.0
        expected_opp_score = (market_total - team_market_margin) / 2.0

    perf: dict[str, Any] = {
        "season": int(game["season"]),
        "week": int(game["week"]),
        "points_for": points_for,
        "points_against": points_against,
        "margin": margin,
        "total": total_actual,
        "score_vs_market": None if expected_team_score is None else points_for - expected_team_score,
        "allowed_vs_market": None if expected_opp_score is None else points_against - expected_opp_score,
        "margin_vs_market": None if team_market_margin is None else margin - team_market_margin,
        "total_vs_market": None if market_total is None else total_actual - market_total,
    }

    # Copy offense metrics and their already-built defensive allowed counterparts.
    for metric in HISTORY_METRICS:
        if metric in perf:
            continue
        if metric.startswith("adj_"):
            continue
        if metric in team_row.index:
            perf[metric] = fnum(team_row.get(metric))
        elif metric.startswith("def_") and metric in team_row.index:
            perf[metric] = fnum(team_row.get(metric))
        else:
            perf.setdefault(metric, None)

    # First-order opponent adjustment uses only the opponent's PRE-GAME history.
    adjustments = [
        ("off_epa", "def_epa_allowed", "adj_off_epa"),
        ("def_epa_allowed", "off_epa", "adj_def_epa_allowed"),
        ("off_success", "def_success_allowed", "adj_off_success"),
        ("def_success_allowed", "off_success", "adj_def_success_allowed"),
        ("off_ypp", "def_ypp_allowed", "adj_off_ypp"),
        ("def_ypp_allowed", "off_ypp", "adj_def_ypp_allowed"),
        ("off_points_per_drive", "def_points_per_drive_allowed", "adj_off_ppd"),
        ("def_points_per_drive_allowed", "off_points_per_drive", "adj_def_ppd_allowed"),
    ]
    for own_metric, opp_metric, out_metric in adjustments:
        observed = fnum(perf.get(own_metric))
        # Prefer current-season opponent mean, then previous season, then last5.
        expected = fnum(opp_snap.get(f"cur_{opp_metric}"))
        if expected is None:
            expected = fnum(opp_snap.get(f"prev_{opp_metric}"))
        if expected is None:
            expected = fnum(opp_snap.get(f"last5_{opp_metric}"))
        perf[out_metric] = None if observed is None or expected is None else observed - expected
    return perf


def build_historical_base_games(
    schedules: dict[int, pd.DataFrame],
    finals_by_season: dict[int, pd.DataFrame],
    resolver: TeamResolver,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in HISTORICAL_SEASONS:
        schedule = schedules[season].copy()
        schedule["home_team"] = schedule["home_team"].map(resolver.resolve)
        schedule["away_team"] = schedule["away_team"].map(resolver.resolve)
        market = load_historical_market(season)
        tw = load_historical_travel_weather(season)
        frame = schedule.merge(finals_by_season[season], on="game_id", how="inner", validate="one_to_one")
        frame = frame.merge(market, on="game_id", how="left", validate="one_to_one", suffixes=("", "_market"))
        frame = frame.merge(tw, on="game_id", how="left", validate="one_to_one", suffixes=("", "_tw"))
        frame["season"] = season
        m = market_features(frame.iloc[0]) if len(frame) else {}
        # Vectorized core market targets.
        frame["home_spread"] = pd.to_numeric(frame.get("home_spread"), errors="coerce")
        frame["total"] = pd.to_numeric(frame.get("total"), errors="coerce")
        frame["market_margin"] = -frame["home_spread"]
        frame["market_total"] = frame["total"]
        frame["actual_margin"] = pd.to_numeric(frame["home_final"], errors="coerce") - pd.to_numeric(frame["away_final"], errors="coerce")
        frame["actual_total"] = pd.to_numeric(frame["home_final"], errors="coerce") + pd.to_numeric(frame["away_final"], errors="coerce")
        frame["market_implied_home_score"] = (frame["market_total"] + frame["market_margin"]) / 2.0
        frame["market_implied_away_score"] = (frame["market_total"] - frame["market_margin"]) / 2.0
        frames.append(frame)
    games = pd.concat(frames, ignore_index=True)
    games = games.sort_values(["season", "week", "game_id"], kind="stable").reset_index(drop=True)
    return games


def build_training_matrix(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, TeamHistory]]:
    histories: dict[str, TeamHistory] = defaultdict(TeamHistory)
    lookup = team_games.copy()
    lookup["game_id"] = lookup["game_id"].map(gid)
    lookup["is_home_num"] = pd.to_numeric(lookup["is_home"], errors="coerce")
    tg_lookup = {
        (str(r["game_id"]), int(r["is_home_num"])): r
        for _, r in lookup.dropna(subset=["is_home_num"]).iterrows()
    }

    output_rows: list[dict[str, Any]] = []
    # Critical leakage rule: snapshot an entire week first, THEN update histories.
    for (season, week), week_games in games.groupby(["season", "week"], sort=True):
        staged_updates: list[tuple[str, dict[str, Any]]] = []
        for _, game in week_games.iterrows():
            home_team = clean(game["home_team"])
            away_team = clean(game["away_team"])
            home_snap = history_snapshot(histories[home_team], int(season), int(week))
            away_snap = history_snapshot(histories[away_team], int(season), int(week))
            features = pregame_feature_row(game, home_snap, away_snap, historical=True)
            out = {
                "season": int(season),
                "season_type": SEASON_TYPE,
                "week": int(week),
                "game_id": gid(game["game_id"]),
                "game_date": clean(game.get("game_date")),
                "game_time": clean(game.get("game_time")),
                "away_team": away_team,
                "home_team": home_team,
                "home_final": fnum(game.get("home_final")),
                "away_final": fnum(game.get("away_final")),
                "actual_margin": fnum(game.get("actual_margin")),
                "actual_total": fnum(game.get("actual_total")),
                "market_implied_home_score": fnum(game.get("market_implied_home_score")),
                "market_implied_away_score": fnum(game.get("market_implied_away_score")),
                **features,
            }
            output_rows.append(out)

            game_id = gid(game["game_id"])
            hrow = tg_lookup.get((game_id, 1))
            arow = tg_lookup.get((game_id, 0))
            if hrow is None or arow is None:
                continue
            hperf = make_team_performance(hrow, arow, game, True, home_snap, away_snap)
            aperf = make_team_performance(arow, hrow, game, False, away_snap, home_snap)
            staged_updates.append((home_team, hperf))
            staged_updates.append((away_team, aperf))

        # Only after every game in the week has its pregame row.
        for team, perf in staged_updates:
            histories[team].add(perf)

    return pd.DataFrame(output_rows), histories


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = META_COLUMNS | TARGET_COLUMNS
    columns: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= max(20, int(len(df) * 0.02)):
            df[col] = converted
            columns.append(col)
    return sorted(columns)


def feature_group_map(all_features: list[str]) -> dict[str, list[str]]:
    def starts(prefixes: tuple[str, ...]) -> list[str]:
        return [c for c in all_features if c.startswith(prefixes)]

    market = starts(("market_",))
    espn = starts(("espn_", "disagree_espn_"))
    pbp = starts(("home_hist_", "away_hist_", "diff_hist_", "matchup_", "disagree_form_"))
    context = starts(("context_",))
    tw = starts(("tw_",))

    def uniq(*groups: list[str]) -> list[str]:
        return sorted(set(itertools.chain.from_iterable(groups)))

    groups = {
        "market_only": uniq(market, context),
        "market_espn": uniq(market, espn, context),
        "market_pbp": uniq(market, pbp, context),
        "market_espn_pbp": uniq(market, espn, pbp, context),
        "full": uniq(market, espn, pbp, context, tw),
        "no_market_pbp": uniq(pbp, context, tw),
        "no_market_espn_pbp": uniq(espn, pbp, context, tw),
    }
    return {name: cols for name, cols in groups.items() if cols}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    scale: bool = False


def candidate_models(fast: bool) -> list[ModelSpec]:
    specs = [
        ModelSpec("ridge_10", Ridge(alpha=10.0), True),
        ModelSpec("ridge_100", Ridge(alpha=100.0), True),
        ModelSpec(
            "extra_2",
            ExtraTreesRegressor(
                n_estimators=250 if fast else 700,
                min_samples_leaf=2,
                max_features=0.75,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "extra_5",
            ExtraTreesRegressor(
                n_estimators=250 if fast else 700,
                min_samples_leaf=5,
                max_features=1.0,
                random_state=RANDOM_STATE + 1,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "rf_4",
            RandomForestRegressor(
                n_estimators=200 if fast else 600,
                min_samples_leaf=4,
                max_features=0.70,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "hist_15",
            HistGradientBoostingRegressor(
                learning_rate=0.035,
                max_iter=180 if fast else 500,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=10.0,
                random_state=RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "hist_31",
            HistGradientBoostingRegressor(
                learning_rate=0.025,
                max_iter=200 if fast else 650,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=25.0,
                random_state=RANDOM_STATE + 1,
            ),
        ),
    ]
    if not fast:
        specs.append(
            ModelSpec(
                "huber",
                HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=3000, tol=1e-5),
                True,
            )
        )
    if XGBRegressor is not None:
        specs.extend(
            [
                ModelSpec(
                    "xgb_d2",
                    XGBRegressor(
                        n_estimators=250 if fast else 700,
                        max_depth=2,
                        learning_rate=0.03,
                        min_child_weight=20,
                        subsample=0.85,
                        colsample_bytree=0.75,
                        reg_alpha=1.0,
                        reg_lambda=25.0,
                        objective="reg:absoluteerror",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
                ModelSpec(
                    "xgb_d3",
                    XGBRegressor(
                        n_estimators=250 if fast else 800,
                        max_depth=3,
                        learning_rate=0.025,
                        min_child_weight=30,
                        subsample=0.85,
                        colsample_bytree=0.70,
                        reg_alpha=2.0,
                        reg_lambda=40.0,
                        objective="reg:absoluteerror",
                        random_state=RANDOM_STATE + 1,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        )
    if LGBMRegressor is not None:
        specs.extend(
            [
                ModelSpec(
                    "lgb_7",
                    LGBMRegressor(
                        n_estimators=250 if fast else 800,
                        learning_rate=0.025,
                        num_leaves=7,
                        max_depth=4,
                        min_child_samples=45,
                        subsample=0.85,
                        colsample_bytree=0.75,
                        reg_alpha=1.0,
                        reg_lambda=25.0,
                        objective="mae",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
                ModelSpec(
                    "lgb_15",
                    LGBMRegressor(
                        n_estimators=250 if fast else 900,
                        learning_rate=0.02,
                        num_leaves=15,
                        max_depth=5,
                        min_child_samples=55,
                        subsample=0.85,
                        colsample_bytree=0.70,
                        reg_alpha=2.0,
                        reg_lambda=40.0,
                        objective="mae",
                        random_state=RANDOM_STATE + 1,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    return specs


def make_pipeline(spec: ModelSpec) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if spec.scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", clone(spec.estimator)))
    return Pipeline(steps)


@dataclass
class CandidateConfig:
    name: str
    group: str
    model_name: str
    objective: str
    feature_columns: list[str]
    alpha: float = 1.0
    ensemble_members: list["CandidateConfig"] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "model_name": self.model_name,
            "objective": self.objective,
            "alpha": self.alpha,
            "feature_count": len(self.feature_columns),
            "ensemble_members": None if not self.ensemble_members else [m.name for m in self.ensemble_members],
        }


def fit_raw_candidate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    config: CandidateConfig,
    model_specs: dict[str, ModelSpec],
    target: str,
    baseline_col: str,
) -> tuple[Any, np.ndarray]:
    if config.ensemble_members:
        fitted = []
        member_preds = []
        for member in config.ensemble_members:
            model, pred = fit_raw_candidate(train, valid, member, model_specs, target, baseline_col)
            fitted.append((member, model))
            member_preds.append(pred)
        return fitted, np.mean(np.vstack(member_preds), axis=0)

    spec = model_specs[config.model_name]
    model = make_pipeline(spec)
    X_train = train[config.feature_columns]
    X_valid = valid[config.feature_columns]
    y_train = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
    if config.objective == "residual":
        baseline_train = pd.to_numeric(train[baseline_col], errors="coerce").to_numpy(float)
        y_fit = y_train - baseline_train
    else:
        y_fit = y_train
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_fit)
        pred = np.asarray(model.predict(X_valid), dtype=float)
    if config.objective == "residual":
        pred = pd.to_numeric(valid[baseline_col], errors="coerce").to_numpy(float) + pred
    return model, pred


def optimize_blend(actual: np.ndarray, baseline: np.ndarray, raw_pred: np.ndarray) -> tuple[float, float]:
    best_alpha = 0.0
    best_mae = mean_absolute_error(actual, baseline)
    for alpha in BLEND_GRID:
        pred = baseline + float(alpha) * (raw_pred - baseline)
        mae = mean_absolute_error(actual, pred)
        if mae < best_mae - 1e-12:
            best_mae = float(mae)
            best_alpha = float(alpha)
    return best_alpha, best_mae


def forward_cv_search(
    df: pd.DataFrame,
    target: str,
    baseline_col: str,
    groups: dict[str, list[str]],
    specs: list[ModelSpec],
    fast: bool,
) -> tuple[CandidateConfig, pd.DataFrame, pd.DataFrame, bool]:
    model_specs = {s.name: s for s in specs}
    individual_records: list[dict[str, Any]] = []
    oof_predictions: dict[str, pd.Series] = {}
    configs: dict[str, CandidateConfig] = {}

    market_groups = [g for g in groups if g.startswith("market") or g == "full"]
    objectives = ["direct", "residual"] if not fast else ["residual", "direct"]

    for group_name in market_groups:
        columns = groups[group_name]
        for spec in specs:
            for objective in objectives:
                name = f"{group_name}|{spec.name}|{objective}"
                config = CandidateConfig(name, group_name, spec.name, objective, columns)
                configs[name] = config
                fold_rows = []
                pred_series = pd.Series(index=df.index, dtype=float)
                actual_series = pd.Series(index=df.index, dtype=float)
                baseline_series = pd.Series(index=df.index, dtype=float)

                for val_season in CV_VALIDATION_SEASONS:
                    train_mask = (
                        pd.to_numeric(df["season"], errors="coerce").lt(val_season)
                        & pd.to_numeric(df[target], errors="coerce").notna()
                        & pd.to_numeric(df[baseline_col], errors="coerce").notna()
                    )
                    valid_mask = (
                        pd.to_numeric(df["season"], errors="coerce").eq(val_season)
                        & pd.to_numeric(df[target], errors="coerce").notna()
                        & pd.to_numeric(df[baseline_col], errors="coerce").notna()
                    )
                    train = df.loc[train_mask].copy()
                    valid = df.loc[valid_mask].copy()
                    if len(train) < 300 or len(valid) < 100:
                        continue
                    try:
                        _, raw_pred = fit_raw_candidate(
                            train, valid, config, model_specs, target, baseline_col
                        )
                    except Exception as exc:
                        fold_rows = []
                        break
                    idx = valid.index
                    pred_series.loc[idx] = raw_pred
                    actual_series.loc[idx] = pd.to_numeric(valid[target], errors="coerce").to_numpy(float)
                    baseline_series.loc[idx] = pd.to_numeric(valid[baseline_col], errors="coerce").to_numpy(float)
                    fold_rows.append({"validation_season": val_season, "rows": len(valid)})

                valid_idx = actual_series.dropna().index.intersection(pred_series.dropna().index).intersection(baseline_series.dropna().index)
                if len(valid_idx) < 300 or len(fold_rows) < 2:
                    continue
                actual = actual_series.loc[valid_idx].to_numpy(float)
                baseline = baseline_series.loc[valid_idx].to_numpy(float)
                raw = pred_series.loc[valid_idx].to_numpy(float)
                alpha, blended_mae = optimize_blend(actual, baseline, raw)
                blended = baseline + alpha * (raw - baseline)
                oof_predictions[name] = pd.Series(blended, index=valid_idx)
                config.alpha = alpha
                baseline_mae = mean_absolute_error(actual, baseline)

                fold_improvements = []
                fold_wins = 0
                worst_degradation = 0.0
                for fold in fold_rows:
                    year = fold["validation_season"]
                    idx = df.index[pd.to_numeric(df["season"], errors="coerce").eq(year)].intersection(valid_idx)
                    a = pd.to_numeric(df.loc[idx, target], errors="coerce").to_numpy(float)
                    b = pd.to_numeric(df.loc[idx, baseline_col], errors="coerce").to_numpy(float)
                    p = oof_predictions[name].loc[idx].to_numpy(float)
                    bmae = mean_absolute_error(a, b)
                    pmae = mean_absolute_error(a, p)
                    imp = bmae - pmae
                    fold_improvements.append(imp)
                    if imp > 0:
                        fold_wins += 1
                    worst_degradation = max(worst_degradation, -imp)

                individual_records.append(
                    {
                        "target": target,
                        "candidate": name,
                        "group": group_name,
                        "model": spec.name,
                        "objective": objective,
                        "blend_alpha": alpha,
                        "rows": len(valid_idx),
                        "cv_mae": blended_mae,
                        "market_mae": baseline_mae,
                        "improvement_points": baseline_mae - blended_mae,
                        "fold_wins": fold_wins,
                        "worst_fold_degradation": worst_degradation,
                        "fold_improvement_2022": fold_improvements[0] if len(fold_improvements) > 0 else np.nan,
                        "fold_improvement_2023": fold_improvements[1] if len(fold_improvements) > 1 else np.nan,
                        "fold_improvement_2024": fold_improvements[2] if len(fold_improvements) > 2 else np.nan,
                        "feature_count": len(columns),
                        "is_ensemble": 0,
                    }
                )

    if not individual_records:
        raise RuntimeError(f"No CV candidates completed for {target}")

    indiv_df = pd.DataFrame(individual_records).sort_values(
        ["cv_mae", "worst_fold_degradation", "feature_count"], kind="stable"
    )

    # Build simple equal-weight ensembles from the best distinct individual models.
    top_names = []
    seen_model_family = set()
    for _, row in indiv_df.iterrows():
        name = row["candidate"]
        family = row["model"]
        if family in seen_model_family:
            continue
        top_names.append(name)
        seen_model_family.add(family)
        if len(top_names) >= (3 if fast else 5):
            break

    ensemble_records = []
    for size in ([2] if fast else [2, 3]):
        for members in itertools.combinations(top_names, size):
            common = None
            for member in members:
                idx = oof_predictions[member].dropna().index
                common = idx if common is None else common.intersection(idx)
            if common is None or len(common) < 300:
                continue
            raw = np.mean(np.vstack([oof_predictions[m].loc[common].to_numpy(float) for m in members]), axis=0)
            actual = pd.to_numeric(df.loc[common, target], errors="coerce").to_numpy(float)
            baseline = pd.to_numeric(df.loc[common, baseline_col], errors="coerce").to_numpy(float)
            alpha, mae = optimize_blend(actual, baseline, raw)
            blended = baseline + alpha * (raw - baseline)
            ensemble_name = "ensemble:" + "+".join(members)
            member_configs = [configs[m] for m in members]
            config = CandidateConfig(
                name=ensemble_name,
                group="ensemble",
                model_name="ensemble_mean",
                objective="mixed",
                feature_columns=sorted(set(itertools.chain.from_iterable(m.feature_columns for m in member_configs))),
                alpha=alpha,
                ensemble_members=member_configs,
            )
            configs[ensemble_name] = config
            oof_predictions[ensemble_name] = pd.Series(blended, index=common)
            baseline_mae = mean_absolute_error(actual, baseline)
            fold_improvements = []
            fold_wins = 0
            worst_degradation = 0.0
            for year in CV_VALIDATION_SEASONS:
                idx = df.index[pd.to_numeric(df["season"], errors="coerce").eq(year)].intersection(common)
                if len(idx) == 0:
                    continue
                a = pd.to_numeric(df.loc[idx, target], errors="coerce").to_numpy(float)
                b = pd.to_numeric(df.loc[idx, baseline_col], errors="coerce").to_numpy(float)
                p = oof_predictions[ensemble_name].loc[idx].to_numpy(float)
                imp = mean_absolute_error(a, b) - mean_absolute_error(a, p)
                fold_improvements.append(imp)
                if imp > 0:
                    fold_wins += 1
                worst_degradation = max(worst_degradation, -imp)
            ensemble_records.append(
                {
                    "target": target,
                    "candidate": ensemble_name,
                    "group": "ensemble",
                    "model": "ensemble_mean",
                    "objective": "mixed",
                    "blend_alpha": alpha,
                    "rows": len(common),
                    "cv_mae": mae,
                    "market_mae": baseline_mae,
                    "improvement_points": baseline_mae - mae,
                    "fold_wins": fold_wins,
                    "worst_fold_degradation": worst_degradation,
                    "fold_improvement_2022": fold_improvements[0] if len(fold_improvements) > 0 else np.nan,
                    "fold_improvement_2023": fold_improvements[1] if len(fold_improvements) > 1 else np.nan,
                    "fold_improvement_2024": fold_improvements[2] if len(fold_improvements) > 2 else np.nan,
                    "feature_count": len(config.feature_columns),
                    "is_ensemble": 1,
                }
            )

    all_df = pd.concat([indiv_df, pd.DataFrame(ensemble_records)], ignore_index=True).sort_values(
        ["cv_mae", "worst_fold_degradation", "feature_count"], kind="stable"
    )
    best_row = all_df.iloc[0]
    best = configs[best_row["candidate"]]
    accepted = bool(
        float(best_row["improvement_points"]) >= CV_MIN_TARGET_MAE_IMPROVEMENT
        and int(best_row["fold_wins"]) >= CV_REQUIRED_FOLD_WINS
        and float(best_row["worst_fold_degradation"]) <= CV_MAX_SINGLE_FOLD_DEGRADATION
        and best.alpha > 0.0
    )

    # Aggregate feature-group diagnostics using the best candidate in each group.
    group_results = (
        all_df[~all_df["group"].eq("ensemble")]
        .sort_values(["group", "cv_mae"], kind="stable")
        .groupby("group", as_index=False)
        .head(1)
        .copy()
    )
    group_results["target"] = target
    return best, all_df, group_results, accepted


def fit_selection(
    train: pd.DataFrame,
    config: CandidateConfig,
    specs: list[ModelSpec],
    target: str,
    baseline_col: str,
) -> Any:
    model_specs = {s.name: s for s in specs}
    # Use the same function with train as a dummy valid set; fitted object is all we need.
    fitted, _ = fit_raw_candidate(train, train.iloc[:1].copy(), config, model_specs, target, baseline_col)
    return fitted


def predict_fitted_raw(
    fitted: Any,
    config: CandidateConfig,
    frame: pd.DataFrame,
    baseline_col: str,
) -> np.ndarray:
    if config.ensemble_members:
        preds = []
        for member, (_, model) in zip(config.ensemble_members, fitted):
            preds.append(predict_fitted_raw(model, member, frame, baseline_col))
        # Member predictions already include their own member alpha if this helper
        # applied it, so use a dedicated raw-member path below instead.
        # This branch is replaced by explicit member raw prediction logic.
        raise RuntimeError("Unexpected nested ensemble prediction path")
    model = fitted
    pred = np.asarray(model.predict(frame[config.feature_columns]), dtype=float)
    if config.objective == "residual":
        pred = pd.to_numeric(frame[baseline_col], errors="coerce").to_numpy(float) + pred
    return pred


def predict_selection(
    fitted: Any,
    config: CandidateConfig,
    frame: pd.DataFrame,
    baseline_col: str,
) -> np.ndarray:
    baseline = pd.to_numeric(frame[baseline_col], errors="coerce").to_numpy(float)
    if config.ensemble_members:
        member_raw = []
        for member, (_, model) in zip(config.ensemble_members, fitted):
            raw = np.asarray(model.predict(frame[member.feature_columns]), dtype=float)
            if member.objective == "residual":
                raw = baseline + raw
            member_pred = baseline + member.alpha * (raw - baseline)
            member_raw.append(member_pred)
        raw_pred = np.mean(np.vstack(member_raw), axis=0)
    else:
        raw_pred = np.asarray(fitted.predict(frame[config.feature_columns]), dtype=float)
        if config.objective == "residual":
            raw_pred = baseline + raw_pred
    return baseline + config.alpha * (raw_pred - baseline)


def no_market_search(
    df: pd.DataFrame,
    target: str,
    groups: dict[str, list[str]],
    specs: list[ModelSpec],
    fast: bool,
) -> tuple[CandidateConfig, pd.DataFrame]:
    usable_groups = [g for g in ["no_market_espn_pbp", "no_market_pbp"] if g in groups]
    records = []
    configs: dict[str, CandidateConfig] = {}
    for group in usable_groups:
        for spec in specs:
            name = f"{group}|{spec.name}|direct"
            config = CandidateConfig(name, group, spec.name, "direct", groups[group], alpha=1.0)
            configs[name] = config
            fold_maes = []
            total_rows = 0
            for year in CV_VALIDATION_SEASONS:
                train = df[
                    pd.to_numeric(df["season"], errors="coerce").lt(year)
                    & pd.to_numeric(df[target], errors="coerce").notna()
                ].copy()
                valid = df[
                    pd.to_numeric(df["season"], errors="coerce").eq(year)
                    & pd.to_numeric(df[target], errors="coerce").notna()
                ].copy()
                if len(train) < 300 or len(valid) < 100:
                    continue
                model = make_pipeline(spec)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(train[config.feature_columns], pd.to_numeric(train[target], errors="coerce").to_numpy(float))
                    pred = np.asarray(model.predict(valid[config.feature_columns]), dtype=float)
                fold_maes.append(mean_absolute_error(pd.to_numeric(valid[target], errors="coerce"), pred))
                total_rows += len(valid)
            if fold_maes:
                records.append(
                    {
                        "target": target,
                        "candidate": name,
                        "group": group,
                        "model": spec.name,
                        "cv_mae": float(np.mean(fold_maes)),
                        "worst_fold_mae": float(np.max(fold_maes)),
                        "rows": total_rows,
                        "feature_count": len(config.feature_columns),
                    }
                )
    if not records:
        raise RuntimeError(f"No no-market candidates completed for {target}")
    results = pd.DataFrame(records).sort_values(["cv_mae", "worst_fold_mae", "feature_count"], kind="stable")
    return configs[results.iloc[0]["candidate"]], results


def fit_direct_config(train: pd.DataFrame, config: CandidateConfig, specs: list[ModelSpec], target: str) -> Any:
    spec_map = {s.name: s for s in specs}
    model = make_pipeline(spec_map[config.model_name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train[config.feature_columns], pd.to_numeric(train[target], errors="coerce").to_numpy(float))
    return model


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(actual, pred)))


def winner_accuracy(actual_margin: np.ndarray, pred_margin: np.ndarray) -> float:
    actual_sign = np.sign(actual_margin)
    pred_sign = np.sign(pred_margin)
    mask = actual_sign != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(actual_sign[mask] == pred_sign[mask]))


def validation_metrics_table(
    validation: pd.DataFrame,
    candidate_margin: np.ndarray,
    candidate_total: np.ndarray,
    deploy_candidate_margin: bool,
    deploy_candidate_total: bool,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray]:
    actual_margin = pd.to_numeric(validation["actual_margin"], errors="coerce").to_numpy(float)
    actual_total = pd.to_numeric(validation["actual_total"], errors="coerce").to_numpy(float)
    actual_home = pd.to_numeric(validation["home_final"], errors="coerce").to_numpy(float)
    actual_away = pd.to_numeric(validation["away_final"], errors="coerce").to_numpy(float)
    market_margin = pd.to_numeric(validation["market_margin"], errors="coerce").to_numpy(float)
    market_total = pd.to_numeric(validation["market_total"], errors="coerce").to_numpy(float)

    frozen_margin = candidate_margin if deploy_candidate_margin else market_margin
    frozen_total = candidate_total if deploy_candidate_total else market_total
    frozen_total = np.maximum(frozen_total, np.abs(frozen_margin))
    candidate_total_clipped = np.maximum(candidate_total, np.abs(candidate_margin))
    market_total_clipped = np.maximum(market_total, np.abs(market_margin))

    market_home = (market_total_clipped + market_margin) / 2.0
    market_away = (market_total_clipped - market_margin) / 2.0
    candidate_home = (candidate_total_clipped + candidate_margin) / 2.0
    candidate_away = (candidate_total_clipped - candidate_margin) / 2.0
    frozen_home = (frozen_total + frozen_margin) / 2.0
    frozen_away = (frozen_total - frozen_margin) / 2.0

    def team_mae(home_pred: np.ndarray, away_pred: np.ndarray) -> float:
        return float(np.mean(np.concatenate([np.abs(actual_home - home_pred), np.abs(actual_away - away_pred)])))

    market_team_mae = team_mae(market_home, market_away)
    candidate_team_mae = team_mae(candidate_home, candidate_away)
    frozen_team_mae = team_mae(frozen_home, frozen_away)

    rows = [
        {
            "metric": "margin",
            "candidate": mean_absolute_error(actual_margin, candidate_margin),
            "frozen_cv_choice": mean_absolute_error(actual_margin, frozen_margin),
            "market": mean_absolute_error(actual_margin, market_margin),
        },
        {
            "metric": "total",
            "candidate": mean_absolute_error(actual_total, candidate_total),
            "frozen_cv_choice": mean_absolute_error(actual_total, frozen_total),
            "market": mean_absolute_error(actual_total, market_total),
        },
        {
            "metric": "average_team_score",
            "candidate": candidate_team_mae,
            "frozen_cv_choice": frozen_team_mae,
            "market": market_team_mae,
        },
        {
            "metric": "winner_accuracy",
            "candidate": winner_accuracy(actual_margin, candidate_margin),
            "frozen_cv_choice": winner_accuracy(actual_margin, frozen_margin),
            "market": winner_accuracy(actual_margin, market_margin),
        },
    ]
    metrics = pd.DataFrame(rows)
    metrics["improvement_vs_market"] = np.where(
        metrics["metric"].eq("winner_accuracy"),
        metrics["frozen_cv_choice"] - metrics["market"],
        metrics["market"] - metrics["frozen_cv_choice"],
    )

    gates = {
        "candidate_margin_mae": float(mean_absolute_error(actual_margin, candidate_margin)),
        "market_margin_mae": float(mean_absolute_error(actual_margin, market_margin)),
        "candidate_total_mae": float(mean_absolute_error(actual_total, candidate_total)),
        "market_total_mae": float(mean_absolute_error(actual_total, market_total)),
        "frozen_team_score_mae": frozen_team_mae,
        "market_team_score_mae": market_team_mae,
        "team_score_improvement": market_team_mae - frozen_team_mae,
        "frozen_winner_accuracy": winner_accuracy(actual_margin, frozen_margin),
        "market_winner_accuracy": winner_accuracy(actual_margin, market_margin),
    }
    return metrics, gates, frozen_margin, frozen_total


def load_current_prior_team_games(
    season: int,
    target_week: int,
    resolver: TeamResolver,
) -> pd.DataFrame:
    path = PBP_DIR / f"{season}_pbp.parquet"
    if not path.is_file() or target_week <= 1:
        return pd.DataFrame()
    schedule = load_schedule(season)
    prior_schedule = schedule[schedule["week"].lt(target_week)].copy()
    if prior_schedule.empty:
        return pd.DataFrame()
    team_games, _, _ = build_team_game_features_for_season(season, prior_schedule, resolver)
    return team_games[pd.to_numeric(team_games["week"], errors="coerce").lt(target_week)].copy()


def update_histories_for_current_prior_weeks(
    histories: dict[str, TeamHistory],
    season: int,
    target_week: int,
    team_games: pd.DataFrame,
    resolver: TeamResolver,
) -> None:
    if team_games.empty:
        return
    schedule = load_schedule(season)
    schedule = schedule[schedule["week"].lt(target_week)].copy()
    schedule["home_team"] = schedule["home_team"].map(resolver.resolve)
    schedule["away_team"] = schedule["away_team"].map(resolver.resolve)

    # Current historical market snapshots for prior weeks may not be retained in
    # the same exact weekly schedule path after the week changes. Use market-less
    # performance updates if unavailable; PBP efficiency still updates correctly.
    tg = team_games.copy()
    tg["is_home_num"] = pd.to_numeric(tg["is_home"], errors="coerce")
    lookup = {
        (gid(r["game_id"]), int(r["is_home_num"])): r
        for _, r in tg.dropna(subset=["is_home_num"]).iterrows()
    }
    for week in sorted(schedule["week"].unique()):
        week_games = schedule[schedule["week"].eq(week)]
        staged = []
        for _, game in week_games.iterrows():
            game = game.copy()
            game["market_margin"] = np.nan
            game["market_total"] = np.nan
            home = game["home_team"]
            away = game["away_team"]
            hs = history_snapshot(histories[home], season, int(week))
            aas = history_snapshot(histories[away], season, int(week))
            hrow = lookup.get((gid(game["game_id"]), 1))
            arow = lookup.get((gid(game["game_id"]), 0))
            if hrow is None or arow is None:
                continue
            staged.append((home, make_team_performance(hrow, arow, game, True, hs, aas)))
            staged.append((away, make_team_performance(arow, hrow, game, False, aas, hs)))
        for team, perf in staged:
            histories[team].add(perf)


def build_current_frame(
    season: int,
    week: int,
    histories: dict[str, TeamHistory],
    resolver: TeamResolver,
) -> pd.DataFrame:
    weekly_path = WEEKLY_SCHEDULE_DIR / f"week_{week}_CFB_weekly_schedule.csv"
    enriched_path = ENRICHED_DIR / f"week_{week}_CFB_enriched.csv"
    travel_path = TRAVEL_DIR / f"{season}_week_{week}_travel.csv"
    weather_path = WEATHER_DIR / f"week_{week}_CFB_weekly_weather.csv"

    weekly = read_csv(weekly_path, ["game_id", "away_team", "home_team", "home_spread", "total"]).copy()
    weekly["game_id"] = weekly["game_id"].map(gid)
    enriched = read_csv(enriched_path, ["game_id"]).copy() if enriched_path.is_file() else pd.DataFrame(columns=["game_id"])
    travel = read_csv(travel_path, ["game_id"]).copy() if travel_path.is_file() else pd.DataFrame(columns=["game_id"])
    weather = read_csv(weather_path, ["game_id"]).copy() if weather_path.is_file() else pd.DataFrame(columns=["game_id"])
    for frame in [enriched, travel, weather]:
        if "game_id" in frame.columns:
            frame["game_id"] = frame["game_id"].map(gid)

    merged = weekly.copy()
    if not enriched.empty:
        merged = merged.merge(enriched, on="game_id", how="left", suffixes=("", "_enriched"))
    if not travel.empty:
        merged = merged.merge(travel, on="game_id", how="left", suffixes=("", "_travel"))
    if not weather.empty:
        merged = merged.merge(weather, on="game_id", how="left", suffixes=("", "_weather"))

    rows = []
    for _, game in merged.iterrows():
        home = resolver.resolve(game["home_team"])
        away = resolver.resolve(game["away_team"])
        hs = history_snapshot(histories[home], season, week)
        aas = history_snapshot(histories[away], season, week)
        features = pregame_feature_row(game, hs, aas, historical=False)
        rows.append(
            {
                "season": season,
                "season_type": SEASON_TYPE,
                "week": week,
                "game_id": gid(game["game_id"]),
                "game_date": clean(game.get("game_date")),
                "game_time": clean(game.get("game_time")),
                "away_team": away,
                "home_team": home,
                **features,
            }
        )
    return pd.DataFrame(rows)


def ensure_feature_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resolver = TeamResolver(TEAM_MAP_PATH)

    schedules = {season: load_schedule(season) for season in HISTORICAL_SEASONS}
    print("Building game-level PBP features...")
    team_games, finals_by_season, optional_aliases = load_or_build_historical_team_games(
        resolver, schedules, rebuild=args.rebuild_pbp_features
    )

    print("Building leakage-safe week-start training matrix...")
    base_games = build_historical_base_games(schedules, finals_by_season, resolver)
    training, histories = build_training_matrix(base_games, team_games)
    # Freeze the usable feature schema before the untouched 2025 final test.
    feature_probe = training[pd.to_numeric(training["season"], errors="coerce").le(2024)].copy()
    all_features = numeric_feature_columns(feature_probe)
    for col in all_features:
        training[col] = pd.to_numeric(training[col], errors="coerce")
    groups = feature_group_map(all_features)
    training_path = OUT_DIR / "training_games_2021_2025_v2.csv"
    write_csv(training, training_path)
    print(f"Historical games: {len(training)}")
    print(f"Numeric pregame features: {len(all_features)}")

    specs = candidate_models(args.fast)
    print(f"Model families/configurations: {len(specs)}")
    print("Selecting margin model with forward-season validation...")
    margin_sel, margin_cv, margin_groups, margin_cv_ok = forward_cv_search(
        training, "actual_margin", "market_margin", groups, specs, args.fast
    )
    print(
        f"  best={margin_sel.name} alpha={margin_sel.alpha:.2f} "
        f"cv_accepted={margin_cv_ok}"
    )

    print("Selecting total model with forward-season validation...")
    total_sel, total_cv, total_groups, total_cv_ok = forward_cv_search(
        training, "actual_total", "market_total", groups, specs, args.fast
    )
    print(
        f"  best={total_sel.name} alpha={total_sel.alpha:.2f} "
        f"cv_accepted={total_cv_ok}"
    )

    print("Selecting no-market fallbacks...")
    fallback_margin_sel, fallback_margin_cv = no_market_search(
        training, "actual_margin", groups, specs, args.fast
    )
    fallback_total_sel, fallback_total_cv = no_market_search(
        training, "actual_total", groups, specs, args.fast
    )

    cv_results = pd.concat([margin_cv, total_cv], ignore_index=True)
    write_csv(cv_results, OUT_DIR / "cv_model_results_v2.csv")
    feature_group_results = pd.concat([margin_groups, total_groups], ignore_index=True)
    write_csv(feature_group_results, OUT_DIR / "feature_group_results_v2.csv")

    # Freeze choices now. Everything below this point uses 2025 only as final validation.
    train_2024 = training[pd.to_numeric(training["season"], errors="coerce").le(2024)].copy()
    valid_2025 = training[
        pd.to_numeric(training["season"], errors="coerce").eq(2025)
        & pd.to_numeric(training["market_margin"], errors="coerce").notna()
        & pd.to_numeric(training["market_total"], errors="coerce").notna()
    ].copy()
    if len(valid_2025) < 300:
        raise RuntimeError(f"Too few 2025 lined validation games: {len(valid_2025)}")

    margin_fit = fit_selection(
        train_2024[pd.to_numeric(train_2024["market_margin"], errors="coerce").notna()].copy(),
        margin_sel,
        specs,
        "actual_margin",
        "market_margin",
    )
    total_fit = fit_selection(
        train_2024[pd.to_numeric(train_2024["market_total"], errors="coerce").notna()].copy(),
        total_sel,
        specs,
        "actual_total",
        "market_total",
    )
    candidate_margin = predict_selection(margin_fit, margin_sel, valid_2025, "market_margin")
    candidate_total = predict_selection(total_fit, total_sel, valid_2025, "market_total")

    actual_margin = pd.to_numeric(valid_2025["actual_margin"], errors="coerce").to_numpy(float)
    actual_total = pd.to_numeric(valid_2025["actual_total"], errors="coerce").to_numpy(float)
    market_margin = pd.to_numeric(valid_2025["market_margin"], errors="coerce").to_numpy(float)
    market_total = pd.to_numeric(valid_2025["market_total"], errors="coerce").to_numpy(float)

    margin_improvement = mean_absolute_error(actual_margin, market_margin) - mean_absolute_error(actual_margin, candidate_margin)
    total_improvement = mean_absolute_error(actual_total, market_total) - mean_absolute_error(actual_total, candidate_total)
    margin_final_ok = bool(margin_cv_ok and margin_improvement >= FINAL_MIN_TARGET_MAE_IMPROVEMENT)
    total_final_ok = bool(total_cv_ok and total_improvement >= FINAL_MIN_TARGET_MAE_IMPROVEMENT)

    metrics, gates, frozen_margin, frozen_total = validation_metrics_table(
        valid_2025,
        candidate_margin,
        candidate_total,
        margin_final_ok,
        total_final_ok,
    )
    score_improvement = float(gates["team_score_improvement"])
    frozen_margin_mae = mean_absolute_error(actual_margin, frozen_margin)
    frozen_total_mae = mean_absolute_error(actual_total, frozen_total)
    market_margin_mae = mean_absolute_error(actual_margin, market_margin)
    market_total_mae = mean_absolute_error(actual_total, market_total)
    winner_delta = gates["frozen_winner_accuracy"] - gates["market_winner_accuracy"]

    overall_validated = bool(
        score_improvement >= FINAL_MIN_TEAM_SCORE_MAE_IMPROVEMENT
        and frozen_margin_mae <= market_margin_mae + FINAL_MAX_OTHER_TARGET_DEGRADATION
        and frozen_total_mae <= market_total_mae + FINAL_MAX_OTHER_TARGET_DEGRADATION
        and winner_delta >= -FINAL_MAX_WINNER_ACCURACY_DEGRADATION
        and (margin_final_ok or total_final_ok)
    )

    # Final 2025 audit predictions.
    validation_output = valid_2025[
        [
            "season",
            "week",
            "game_id",
            "away_team",
            "home_team",
            "home_final",
            "away_final",
            "market_margin",
            "market_total",
        ]
    ].copy()
    validation_output["candidate_margin"] = candidate_margin
    validation_output["candidate_total"] = candidate_total
    validation_output["frozen_cv_margin"] = frozen_margin
    validation_output["frozen_cv_total"] = frozen_total
    validation_output["market_home_score"] = (market_total + market_margin) / 2.0
    validation_output["market_away_score"] = (market_total - market_margin) / 2.0
    validation_output["candidate_home_score"] = (np.maximum(candidate_total, np.abs(candidate_margin)) + candidate_margin) / 2.0
    validation_output["candidate_away_score"] = (np.maximum(candidate_total, np.abs(candidate_margin)) - candidate_margin) / 2.0
    validation_output["frozen_home_score"] = (np.maximum(frozen_total, np.abs(frozen_margin)) + frozen_margin) / 2.0
    validation_output["frozen_away_score"] = (np.maximum(frozen_total, np.abs(frozen_margin)) - frozen_margin) / 2.0
    write_csv(validation_output, OUT_DIR / "validation_2025_predictions_v2.csv")
    write_csv(metrics, OUT_DIR / "validation_2025_metrics_v2.csv")

    print("2025 FINAL validation (never used to choose features/models/blends):")
    print(
        f"  margin MAE: candidate={mean_absolute_error(actual_margin, candidate_margin):.4f} "
        f"market={market_margin_mae:.4f} improvement={margin_improvement:.4f} pass={margin_final_ok}"
    )
    print(
        f"  total MAE: candidate={mean_absolute_error(actual_total, candidate_total):.4f} "
        f"market={market_total_mae:.4f} improvement={total_improvement:.4f} pass={total_final_ok}"
    )
    print(
        f"  team-score MAE: frozen={gates['frozen_team_score_mae']:.4f} "
        f"market={gates['market_team_score_mae']:.4f} improvement={score_improvement:.4f}"
    )
    print(
        f"  winner accuracy: frozen={gates['frozen_winner_accuracy']:.4f} "
        f"market={gates['market_winner_accuracy']:.4f} delta={winner_delta:+.4f}"
    )
    print(f"  MODEL STATUS: {'VALIDATED' if overall_validated else 'REJECTED'}")

    # Refit frozen model choices on all 2021-2025 data only if they passed their
    # own target validation. Fallbacks are always fitted for games with no line.
    lined_margin_train = training[pd.to_numeric(training["market_margin"], errors="coerce").notna()].copy()
    lined_total_train = training[pd.to_numeric(training["market_total"], errors="coerce").notna()].copy()
    final_margin_fit = fit_selection(lined_margin_train, margin_sel, specs, "actual_margin", "market_margin") if margin_final_ok else None
    final_total_fit = fit_selection(lined_total_train, total_sel, specs, "actual_total", "market_total") if total_final_ok else None
    fallback_margin_fit = fit_direct_config(training, fallback_margin_sel, specs, "actual_margin")
    fallback_total_fit = fit_direct_config(training, fallback_total_sel, specs, "actual_total")

    feature_rows = []
    configs_to_record = [
        ("margin", margin_sel),
        ("total", total_sel),
        ("fallback_margin", fallback_margin_sel),
        ("fallback_total", fallback_total_sel),
    ]
    for label, cfg in configs_to_record:
        members = cfg.ensemble_members or [cfg]
        for member in members:
            for col in member.feature_columns:
                feature_rows.append(
                    {"target": label, "candidate": member.name, "feature": col}
                )
    write_csv(pd.DataFrame(feature_rows).drop_duplicates(), OUT_DIR / "feature_columns_v2.csv")

    payload = {
        "script_version": SCRIPT_VERSION,
        "overall_validated": overall_validated,
        "margin_final_ok": margin_final_ok,
        "total_final_ok": total_final_ok,
        "margin_selection": margin_sel,
        "total_selection": total_sel,
        "fallback_margin_selection": fallback_margin_sel,
        "fallback_total_selection": fallback_total_sel,
        "margin_model": final_margin_fit,
        "total_model": final_total_fit,
        "fallback_margin_model": fallback_margin_fit,
        "fallback_total_model": fallback_total_fit,
        "all_features": all_features,
    }
    model_path = OUT_DIR / "score_model_v2.joblib"
    joblib.dump(payload, model_path)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "historical_seasons": HISTORICAL_SEASONS,
        "cv_validation_seasons": CV_VALIDATION_SEASONS,
        "final_validation_season": FINAL_VALIDATION_SEASON,
        "historical_games": int(len(training)),
        "pbp_team_game_rows": int(len(team_games)),
        "numeric_feature_count": len(all_features),
        "optional_pbp_fields_used": optional_aliases,
        "margin_selection": margin_sel.to_dict(),
        "total_selection": total_sel.to_dict(),
        "fallback_margin_selection": fallback_margin_sel.to_dict(),
        "fallback_total_selection": fallback_total_sel.to_dict(),
        "cv_acceptance": {
            "margin": margin_cv_ok,
            "total": total_cv_ok,
            "min_target_mae_improvement": CV_MIN_TARGET_MAE_IMPROVEMENT,
            "required_fold_wins": CV_REQUIRED_FOLD_WINS,
            "max_single_fold_degradation": CV_MAX_SINGLE_FOLD_DEGRADATION,
        },
        "validation_2025": {
            **{k: float(v) for k, v in gates.items()},
            "margin_candidate_improvement": float(margin_improvement),
            "total_candidate_improvement": float(total_improvement),
            "margin_pass": margin_final_ok,
            "total_pass": total_final_ok,
            "overall_status": "VALIDATED" if overall_validated else "REJECTED",
        },
        "deployment_gate": {
            "minimum_team_score_mae_improvement_points": FINAL_MIN_TEAM_SCORE_MAE_IMPROVEMENT,
            "minimum_target_mae_improvement_points": FINAL_MIN_TARGET_MAE_IMPROVEMENT,
            "maximum_other_target_degradation_points": FINAL_MAX_OTHER_TARGET_DEGRADATION,
            "maximum_winner_accuracy_degradation": FINAL_MAX_WINNER_ACCURACY_DEGRADATION,
        },
        "excluded": {
            "historical_fpi": "No point-in-time 2021-2025 FPI history exists in the repo.",
            "historical_injuries": "No point-in-time 2021-2025 injury history exists in the repo.",
            "old_projection_outputs": "Excluded; this model trains from raw pregame inputs and PBP history.",
            "ev_kelly": "Excluded; these are betting outputs, not score-prediction inputs.",
        },
        "write_scope": str(OUT_DIR),
    }
    write_json(manifest, OUT_DIR / "score_model_manifest_v2.json")

    if args.predict_week is not None:
        # Reuse histories after the 2025 training matrix, then add only completed
        # PRIOR 2026 weeks. Target-week PBP is never used.
        prior_team_games = load_current_prior_team_games(
            args.season, args.predict_week, resolver
        )
        update_histories_for_current_prior_weeks(
            histories, args.season, args.predict_week, prior_team_games, resolver
        )
        current = build_current_frame(args.season, args.predict_week, histories, resolver)
        needed = sorted(
            set(
                itertools.chain(
                    margin_sel.feature_columns,
                    total_sel.feature_columns,
                    fallback_margin_sel.feature_columns,
                    fallback_total_sel.feature_columns,
                )
            )
        )
        ensure_feature_columns(current, needed)

        market_margin_current = pd.to_numeric(current["market_margin"], errors="coerce").to_numpy(float)
        market_total_current = pd.to_numeric(current["market_total"], errors="coerce").to_numpy(float)
        pred_margin = np.full(len(current), np.nan)
        pred_total = np.full(len(current), np.nan)
        candidate_margin_current = np.full(len(current), np.nan)
        candidate_total_current = np.full(len(current), np.nan)
        margin_source = np.empty(len(current), dtype=object)
        total_source = np.empty(len(current), dtype=object)

        margin_has_market = np.isfinite(market_margin_current)
        total_has_market = np.isfinite(market_total_current)

        if final_margin_fit is not None and margin_has_market.any():
            subset = current.loc[margin_has_market]
            candidate_margin_current[margin_has_market] = predict_selection(
                final_margin_fit, margin_sel, subset, "market_margin"
            )
        if final_total_fit is not None and total_has_market.any():
            subset = current.loc[total_has_market]
            candidate_total_current[total_has_market] = predict_selection(
                final_total_fit, total_sel, subset, "market_total"
            )

        for i in range(len(current)):
            if margin_has_market[i]:
                if overall_validated and margin_final_ok and np.isfinite(candidate_margin_current[i]):
                    pred_margin[i] = candidate_margin_current[i]
                    margin_source[i] = "TRAINED_V2"
                else:
                    pred_margin[i] = market_margin_current[i]
                    margin_source[i] = "MARKET_BASELINE"
            else:
                row = current.iloc[[i]]
                pred_margin[i] = float(fallback_margin_fit.predict(row[fallback_margin_sel.feature_columns])[0])
                margin_source[i] = "NO_MARKET_FALLBACK"

            if total_has_market[i]:
                if overall_validated and total_final_ok and np.isfinite(candidate_total_current[i]):
                    pred_total[i] = candidate_total_current[i]
                    total_source[i] = "TRAINED_V2"
                else:
                    pred_total[i] = market_total_current[i]
                    total_source[i] = "MARKET_BASELINE"
            else:
                row = current.iloc[[i]]
                pred_total[i] = float(fallback_total_fit.predict(row[fallback_total_sel.feature_columns])[0])
                total_source[i] = "NO_MARKET_FALLBACK"

        pred_total = np.maximum(pred_total, np.abs(pred_margin))
        output = current[
            ["season", "season_type", "week", "game_id", "game_date", "game_time", "away_team", "home_team"]
        ].copy()
        output["model_status"] = "VALIDATED" if overall_validated else "REJECTED"
        output["margin_source"] = margin_source
        output["total_source"] = total_source
        output["market_margin"] = market_margin_current
        output["market_total"] = market_total_current
        output["candidate_trained_margin"] = candidate_margin_current
        output["candidate_trained_total"] = candidate_total_current
        output["predicted_margin"] = pred_margin
        output["predicted_total"] = pred_total
        output["predicted_home_score"] = (pred_total + pred_margin) / 2.0
        output["predicted_away_score"] = (pred_total - pred_margin) / 2.0
        prediction_path = OUT_DIR / f"week_{args.predict_week}_CFB_trained_score_predictions_v2.csv"
        write_csv(output, prediction_path)
        print(f"Current predictions: {prediction_path}")

    print(f"Training matrix: {training_path}")
    print(f"CV results: {OUT_DIR / 'cv_model_results_v2.csv'}")
    print(f"Feature groups: {OUT_DIR / 'feature_group_results_v2.csv'}")
    print(f"2025 predictions: {OUT_DIR / 'validation_2025_predictions_v2.csv'}")
    print(f"2025 metrics: {OUT_DIR / 'validation_2025_metrics_v2.csv'}")
    print(f"Saved model: {model_path}")
    print(f"Manifest: {OUT_DIR / 'score_model_manifest_v2.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

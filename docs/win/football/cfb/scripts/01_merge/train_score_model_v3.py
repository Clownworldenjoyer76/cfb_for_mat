#!/usr/bin/env python3
"""
train_score_model_v3.py

Standalone CFB score-model experiment combining two materially different ideas:

1. Predict possessions (drives) and scoring efficiency (effective points/drive)
   separately, then reconstruct each team's score.
2. Maintain dynamic opponent-adjusted team ratings that update only AFTER an
   entire week is complete, so no same-week result can leak into another game.

The script READS the existing repository and WRITES ONLY under:

    docs/win/football/cfb/data/score_model_v3/

It never modifies projection.py, projection_week1.py, selections.py, picks.py,
config files, intake files, historical files, or workflows.

Historical model-selection protocol
-----------------------------------
* 2021 is the initial training season.
* Forward validation chooses models using:
      train <= 2021 -> validate 2022
      train <= 2022 -> validate 2023
      train <= 2023 -> validate 2024
* 2025 is completely untouched until all model, feature-group, residual/direct,
  and market-blend choices are frozen.
* 2025 is then used once as the final validation season.

Core targets
------------
PACE:
    actual_pace = mean(home offensive drives, away offensive drives)

TEAM SCORING EFFICIENCY:
    actual_effective_ppd = final team points / offensive drives

The use of final points / offensive drives intentionally includes defensive and
special-teams scoring in the effective scoring rate. That lets reconstructed
scores match the actual scoreboard target rather than only offensive points.

Dynamic ratings
---------------
The rating book carries opponent-adjusted offsets for:
* effective points/drive offense and defense
* EPA/play offense and defense
* success rate offense and defense
* yards/play offense and defense
* turnover rate offense and defense
* team pace

A team's observed performance is judged against the opponent's PRE-GAME rating.
Ratings are updated only after every game in that week has been snapshotted.
Prior-season ratings are regressed toward the league average at the next season.

Historical learned inputs
-------------------------
* Native SportsDataverse PBP, 2021-2025
* Historical sportsbook spread/total and ESPN predictor cache
* Deterministic travel features from historical travel/weather files
* Schedule/venue context

Historical weather observations are NOT used as learned features because the
repo does not contain point-in-time historical weather forecasts. Using actual
observed weather to train a pregame model would be leakage.

Current prediction inputs
-------------------------
* weekly schedule
* current enriched file (for ESPN predictor when available)
* current deterministic travel file
* current season PBP from COMPLETED PRIOR weeks only

Outputs
-------
    docs/win/football/cfb/data/score_model_v3/
        pbp_team_game_features_2021_2025_v3.csv
        training_games_2021_2025_v3.csv
        training_ppd_rows_2021_2025_v3.csv
        pace_cv_results_v3.csv
        ppd_cv_results_v3.csv
        score_cv_results_v3.csv
        validation_2025_predictions_v3.csv
        validation_2025_metrics_v3.csv
        score_model_v3.joblib
        score_model_manifest_v3.json
        week_{week}_CFB_trained_score_predictions_v3.csv

A model is marked VALIDATED only if the frozen 2025 result materially improves
average team-score MAE without materially degrading margin/total/winner accuracy.
If rejected, current lined games keep the market baseline in the DEPLOYED score
columns, while the component-model candidate remains visible for inspection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import unicodedata
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


SCRIPT_VERSION = "cfb-score-components-dynamic-ratings-v3-2026-08-29"
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

# These are validation standards, not predictive weights.
CV_MIN_TEAM_SCORE_IMPROVEMENT = 0.10
CV_REQUIRED_SCORE_FOLD_WINS = 2
FINAL_MIN_TEAM_SCORE_IMPROVEMENT = 0.20
FINAL_MAX_MARGIN_DEGRADATION = 0.10
FINAL_MAX_TOTAL_DEGRADATION = 0.10
FINAL_MAX_WINNER_ACCURACY_DEGRADATION = 0.005

# The only predictive blending weight is selected from 2022-2024 forward OOF
# predictions, then frozen before 2025 is examined.
MARKET_BLEND_GRID = np.round(np.arange(0.0, 1.51, 0.05), 2)

# Prior-season strength carry is structural shrinkage, not a fitted game weight.
# A second candidate value is tested by forward CV via the rating-cache build.
DEFAULT_PRIOR_CARRY = 0.55

# Very wide physical guards prevent pathological tree extrapolation. These are
# not tuned against outcomes.
PACE_MIN, PACE_MAX = 6.0, 20.0
PPD_MIN, PPD_MAX = 0.0, 8.0
SCORE_MIN, SCORE_MAX = 0.0, 100.0

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

OPTIONAL_FLAG_ALIASES = {
    "pass_flag": ["pass", "pass_attempt", "qb_dropback", "dropback"],
    "rush_flag": ["rush", "rush_attempt"],
    "sack_flag": ["sack", "sack_play"],
    "interception_flag": ["interception", "interception_thrown", "int"],
    "fumble_lost_flag": ["fumble_lost", "fumble_lost_play"],
    "turnover_flag": ["turnover", "turnover_play"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train possession + scoring-efficiency CFB model with dynamic ratings."
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--predict-week", type=int, default=None)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a smaller model search for troubleshooting. Full search is default.",
    )
    parser.add_argument(
        "--rebuild-pbp-features",
        action="store_true",
        help="Ignore the v3 PBP team-game cache and rebuild from Parquet.",
    )
    parser.add_argument(
        "--prior-carry",
        type=float,
        default=DEFAULT_PRIOR_CARRY,
        help="Fraction of prior-season dynamic rating retained into the next season.",
    )
    return parser.parse_args()


def resolve_cfb_root() -> Path:
    here = Path(__file__).resolve()
    # Works when script lives in docs/win/football/cfb/scripts/01_merge/.
    for parent in [here.parent, *here.parents]:
        if parent.name == "cfb" and (parent / "00_intake").is_dir():
            return parent
        candidate = parent / "docs" / "win" / "football" / "cfb"
        if candidate.is_dir():
            return candidate
    raise RuntimeError(f"Cannot resolve CFB repo root from {here}")


CFB_ROOT = resolve_cfb_root()
SCHEDULE_DIR = CFB_ROOT / "00_intake" / "schedule"
WEEKLY_SCHEDULE_DIR = SCHEDULE_DIR / "weekly"
PBP_DIR = CFB_ROOT / "00_intake" / "pbp"
HIST_CACHE_DIR = CFB_ROOT / "data" / "historical_betting" / "cache"
HIST_FEATURE_DIR = CFB_ROOT / "data" / "historical_features"
TRAVEL_DIR = CFB_ROOT / "data" / "travel"
ENRICHED_DIR = CFB_ROOT / "01_merge"
TEAM_MAP_PATH = CFB_ROOT / "config" / "mapping" / "team_map.csv"
OUT_DIR = CFB_ROOT / "data" / "score_model_v3"


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
    return number if math.isfinite(number) else None


def finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        num = fnum(value)
        if num is not None:
            out.append(num)
    return out


def safe_mean(values: Iterable[Any]) -> float | None:
    nums = finite(values)
    return float(np.mean(nums)) if nums else None


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


def load_historical_travel(season: int) -> pd.DataFrame:
    # File also contains historical observed weather. Only deterministic travel
    # columns are consumed later.
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
    work["first_down_flag"] = pd.to_numeric(
        work["first_down_created"], errors="coerce"
    ).fillna(0).gt(0)
    work["touchdown_flag"] = bool_series(work["touchdown"])

    for standard, source in optional_aliases.items():
        work[standard] = bool_series(work[source])
    for standard in OPTIONAL_FLAG_ALIASES:
        if standard not in work.columns:
            work[standard] = False

    work["turnover_any"] = (
        work["turnover_flag"] | work["interception_flag"] | work["fumble_lost_flag"]
    )

    valid = work[
        work["scrimmage_flag"] & work["game_id"].ne("") & work["epa"].notna()
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

    drives = work[work["game_id"].ne("") & work["drive.id"].notna()].copy()
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
    games = (
        schedule_lookup.merge(home, on="game_id", how="left")
        .merge(away, on="game_id", how="left")
        .merge(finals, on="game_id", how="inner")
    )

    rows: list[dict[str, Any]] = []
    off_metric_names = [c[len("home_"):] for c in home.columns if c.startswith("home_off_")]
    for _, game in games.iterrows():
        for is_home in (True, False):
            side_prefix = "home" if is_home else "away"
            opp_prefix = "away" if is_home else "home"
            team = game["home_team"] if is_home else game["away_team"]
            opponent = game["away_team"] if is_home else game["home_team"]
            row: dict[str, Any] = {
                "season": season,
                "week": int(game["week"]),
                "game_id": gid(game["game_id"]),
                "team": team,
                "opponent": opponent,
                "is_home": int(is_home),
                "points_for": float(game["home_final"] if is_home else game["away_final"]),
                "points_against": float(game["away_final"] if is_home else game["home_final"]),
            }
            for metric in off_metric_names:
                row[metric] = fnum(game.get(f"{side_prefix}_{metric}"))
                if metric.startswith("off_"):
                    def_name = "def_" + metric[len("off_"):] + "_allowed"
                    row[def_name] = fnum(game.get(f"{opp_prefix}_{metric}"))
            drives = fnum(row.get("off_drives"))
            row["effective_ppd"] = (
                None if drives is None or drives <= 0 else row["points_for"] / drives
            )
            rows.append(row)
    return pd.DataFrame(rows), finals, aliases


def pbp_signature(seasons: list[int]) -> dict[str, Any]:
    sig: dict[str, Any] = {}
    for season in seasons:
        path = PBP_DIR / f"{season}_pbp.parquet"
        st = path.stat()
        sig[str(season)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    return sig


def load_or_build_historical_team_games(
    resolver: TeamResolver,
    schedules: dict[int, pd.DataFrame],
    rebuild: bool,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[str, Any]]:
    cache_path = OUT_DIR / "pbp_team_game_features_2021_2025_v3.csv"
    meta_path = OUT_DIR / "pbp_team_game_features_2021_2025_v3.meta.json"
    current_sig = pbp_signature(HISTORICAL_SEASONS)

    if not rebuild and cache_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("pbp_signature") == current_sig:
                cached = read_csv(cache_path)
                cached["game_id"] = cached["game_id"].map(gid)
                finals_by_season: dict[int, pd.DataFrame] = {}
                for season in HISTORICAL_SEASONS:
                    # Finals can be recovered from team rows without reopening PBP.
                    s = cached[pd.to_numeric(cached["season"], errors="coerce").eq(season)]
                    h = s[pd.to_numeric(s["is_home"], errors="coerce").eq(1)][
                        ["game_id", "points_for", "points_against"]
                    ].copy()
                    h = h.rename(columns={"points_for": "home_final", "points_against": "away_final"})
                    finals_by_season[season] = h.drop_duplicates("game_id", keep="last")
                print("Using cached PBP team-game features.")
                return cached, finals_by_season, meta.get("optional_aliases", {})
        except Exception:
            pass

    frames: list[pd.DataFrame] = []
    finals_by_season: dict[int, pd.DataFrame] = {}
    aliases_manifest: dict[str, Any] = {}
    for season in HISTORICAL_SEASONS:
        print(f"Extracting PBP game features: {season}...")
        tg, finals, aliases = build_team_game_features_for_season(
            season, schedules[season], resolver
        )
        frames.append(tg)
        finals_by_season[season] = finals
        aliases_manifest[str(season)] = aliases
        print(
            f"  team-game rows={len(tg)} games={tg['game_id'].nunique()} "
            f"optional_flags={sorted(aliases)}"
        )
    combined = pd.concat(frames, ignore_index=True)
    write_csv(combined, cache_path)
    write_json(
        {
            "script_version": SCRIPT_VERSION,
            "pbp_signature": current_sig,
            "optional_aliases": aliases_manifest,
        },
        meta_path,
    )
    return combined, finals_by_season, aliases_manifest


# ---------------------------------------------------------------------------
# Dynamic ratings
# ---------------------------------------------------------------------------

RATING_METRICS = ["ppd", "epa", "success", "ypp", "turnover"]


@dataclass
class TeamRating:
    off: dict[str, float] = field(default_factory=lambda: {m: 0.0 for m in RATING_METRICS})
    defense: dict[str, float] = field(default_factory=lambda: {m: 0.0 for m in RATING_METRICS})
    pace: float = 0.0
    current_games: int = 0
    previous_games: int = 0
    total_games: int = 0


@dataclass
class LeagueLevel:
    ppd: float = 2.25
    epa: float = 0.0
    success: float = 0.45
    ypp: float = 5.5
    turnover: float = 0.02
    pace: float = 11.5


class DynamicRatingBook:
    def __init__(self, prior_carry: float):
        if not 0.0 <= prior_carry <= 1.0:
            raise ValueError("prior_carry must be between 0 and 1")
        self.prior_carry = float(prior_carry)
        self.teams: dict[str, TeamRating] = defaultdict(TeamRating)
        self.league = LeagueLevel()
        self.current_season: int | None = None

    def transition_to(self, season: int) -> None:
        if self.current_season is None:
            self.current_season = season
            return
        if season == self.current_season:
            return
        if season < self.current_season:
            raise RuntimeError("DynamicRatingBook cannot move backward in time")
        while self.current_season < season:
            for state in self.teams.values():
                for metric in RATING_METRICS:
                    state.off[metric] *= self.prior_carry
                    state.defense[metric] *= self.prior_carry
                state.pace *= self.prior_carry
                state.previous_games = state.current_games
                state.current_games = 0
            self.current_season += 1

    @staticmethod
    def update_alpha(current_games: int) -> float:
        # Fast adaptation early, stabilizing as current-season evidence grows.
        return float(max(0.12, min(0.42, 0.42 / (1.0 + 0.12 * current_games))))

    def snapshot(self, team: str, opponent: str) -> dict[str, float]:
        t = self.teams[team]
        o = self.teams[opponent]
        expected_ppd = self.league.ppd + t.off["ppd"] + o.defense["ppd"]
        expected_pace = self.league.pace + (t.pace + o.pace) / 2.0
        out: dict[str, float] = {
            "rating_expected_ppd": expected_ppd,
            "rating_expected_pace": expected_pace,
            "rating_team_pace_offset": t.pace,
            "rating_opp_pace_offset": o.pace,
            "rating_team_games_current": float(t.current_games),
            "rating_team_games_previous": float(t.previous_games),
            "rating_team_games_total": float(t.total_games),
            "rating_opp_games_current": float(o.current_games),
            "rating_opp_games_previous": float(o.previous_games),
            "rating_opp_games_total": float(o.total_games),
            "rating_team_reliability": min(1.0, (t.current_games + self.prior_carry * t.previous_games) / 6.0),
            "rating_opp_reliability": min(1.0, (o.current_games + self.prior_carry * o.previous_games) / 6.0),
            "rating_league_ppd": self.league.ppd,
            "rating_league_pace": self.league.pace,
            "rating_league_epa": self.league.epa,
            "rating_league_success": self.league.success,
            "rating_league_ypp": self.league.ypp,
            "rating_league_turnover": self.league.turnover,
        }
        for metric in RATING_METRICS:
            out[f"rating_team_off_{metric}"] = t.off[metric]
            out[f"rating_team_def_{metric}"] = t.defense[metric]
            out[f"rating_opp_off_{metric}"] = o.off[metric]
            out[f"rating_opp_def_{metric}"] = o.defense[metric]
            out[f"rating_matchup_{metric}"] = t.off[metric] + o.defense[metric]
        return out

    def stage_game_updates(
        self,
        home_team: str,
        away_team: str,
        home_row: pd.Series,
        away_row: pd.Series,
    ) -> tuple[list[tuple[str, dict[str, float]]], dict[str, float]]:
        h = self.teams[home_team]
        a = self.teams[away_team]

        h_ppd = fnum(home_row.get("effective_ppd"))
        a_ppd = fnum(away_row.get("effective_ppd"))
        h_drives = fnum(home_row.get("off_drives"))
        a_drives = fnum(away_row.get("off_drives"))
        pace = safe_mean([h_drives, a_drives])

        observed = {
            "ppd": (h_ppd, a_ppd),
            "epa": (fnum(home_row.get("off_epa")), fnum(away_row.get("off_epa"))),
            "success": (fnum(home_row.get("off_success")), fnum(away_row.get("off_success"))),
            "ypp": (fnum(home_row.get("off_ypp")), fnum(away_row.get("off_ypp"))),
            "turnover": (
                fnum(home_row.get("off_turnover_rate")),
                fnum(away_row.get("off_turnover_rate")),
            ),
        }
        league_vals = {
            "ppd": self.league.ppd,
            "epa": self.league.epa,
            "success": self.league.success,
            "ypp": self.league.ypp,
            "turnover": self.league.turnover,
        }

        def signals(own: TeamRating, opp: TeamRating, own_idx: int, opp_idx: int) -> dict[str, float]:
            sig: dict[str, float] = {}
            for metric in RATING_METRICS:
                own_obs = observed[metric][own_idx]
                opp_obs = observed[metric][opp_idx]
                league = league_vals[metric]
                if own_obs is not None:
                    sig[f"off_{metric}"] = own_obs - (league + opp.defense[metric])
                if opp_obs is not None:
                    sig[f"def_{metric}"] = opp_obs - (league + opp.off[metric])
            if pace is not None:
                sig["pace"] = pace - self.league.pace
            return sig

        updates = [
            (home_team, signals(h, a, 0, 1)),
            (away_team, signals(a, h, 1, 0)),
        ]
        league_obs = {
            "ppd": safe_mean([h_ppd, a_ppd]),
            "epa": safe_mean(observed["epa"]),
            "success": safe_mean(observed["success"]),
            "ypp": safe_mean(observed["ypp"]),
            "turnover": safe_mean(observed["turnover"]),
            "pace": pace,
        }
        return updates, {k: v for k, v in league_obs.items() if v is not None}

    def apply_week(
        self,
        staged_updates: list[tuple[str, dict[str, float]]],
        league_observations: list[dict[str, float]],
    ) -> None:
        for team, sig in staged_updates:
            state = self.teams[team]
            alpha = self.update_alpha(state.current_games)
            for metric in RATING_METRICS:
                off_key = f"off_{metric}"
                def_key = f"def_{metric}"
                if off_key in sig:
                    state.off[metric] = (1.0 - alpha) * state.off[metric] + alpha * sig[off_key]
                if def_key in sig:
                    state.defense[metric] = (
                        (1.0 - alpha) * state.defense[metric] + alpha * sig[def_key]
                    )
            if "pace" in sig:
                state.pace = (1.0 - alpha) * state.pace + alpha * sig["pace"]
            state.current_games += 1
            state.total_games += 1

        # League level updates happen after the whole week. A modest EWMA lets
        # scoring environment drift across years without using future games.
        if league_observations:
            beta = 0.12
            for metric in ["ppd", "epa", "success", "ypp", "turnover", "pace"]:
                value = safe_mean(obs.get(metric) for obs in league_observations)
                if value is None:
                    continue
                old = getattr(self.league, metric)
                setattr(self.league, metric, (1.0 - beta) * old + beta * value)


# ---------------------------------------------------------------------------
# Pregame feature construction
# ---------------------------------------------------------------------------

def market_game_features(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    home_spread = fnum(row.get("home_spread"))
    total = fnum(row.get("total"))
    home_ml = fnum(row.get("home_moneyline_american"))
    away_ml = fnum(row.get("away_moneyline_american"))
    margin = None if home_spread is None else -home_spread
    home_score = None if margin is None or total is None else (total + margin) / 2.0
    away_score = None if margin is None or total is None else (total - margin) / 2.0
    return {
        "market_margin": margin,
        "market_total": total,
        "market_abs_margin": None if margin is None else abs(margin),
        "market_home_implied_score": home_score,
        "market_away_implied_score": away_score,
        "market_home_ml_implied": american_implied(home_ml),
        "market_away_ml_implied": american_implied(away_ml),
        "market_available": 1.0 if margin is not None and total is not None else 0.0,
    }


def espn_game_features(row: pd.Series | dict[str, Any], historical: bool) -> dict[str, Any]:
    if historical:
        hp = fnum(row.get("espn_home_game_projection"))
        ap = fnum(row.get("espn_away_game_projection"))
    else:
        hp = fnum(row.get("espn_home_prob"))
        ap = fnum(row.get("espn_away_prob"))
        if hp is None:
            hp = fnum(row.get("espn_home_game_projection"))
        if ap is None:
            ap = fnum(row.get("espn_away_game_projection"))
    hdiff = fnum(row.get("espn_home_ptdiff"))
    adiff = fnum(row.get("espn_away_ptdiff"))
    return {
        "espn_home_probability": hp,
        "espn_away_probability": ap,
        "espn_home_ptdiff": hdiff,
        "espn_away_ptdiff": adiff,
        "espn_available": 1.0 if hp is not None or hdiff is not None else 0.0,
    }


def context_features(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    neutral = fnum(row.get("neutral_site"))
    surface = norm(row.get("surface"))
    roof = norm(row.get("roof") or row.get("roof_type"))
    return {
        "context_week": fnum(row.get("week")),
        "context_neutral": neutral,
        "context_grass": 1.0 if "grass" in surface else (0.0 if surface else None),
        "context_turf": 1.0 if "turf" in surface or "artificial" in surface else (0.0 if surface else None),
        "context_open_air": 1.0 if "open" in roof else (0.0 if roof else None),
        "context_dome": 1.0 if "dome" in roof or "indoor" in roof or "closed" in roof else (0.0 if roof else None),
    }


def travel_features(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    # Deterministic travel only. No historical observed weather columns.
    raw_names = [
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
    ]
    out = {f"travel_{name}": fnum(row.get(name)) for name in raw_names}
    away_miles = fnum(row.get("away_miles_traveled"))
    home_miles = fnum(row.get("home_miles_traveled"))
    away_tz = fnum(row.get("away_time_zones_crossed"))
    home_tz = fnum(row.get("home_time_zones_crossed"))
    a_e2w = fnum(row.get("away_east_to_west"))
    h_e2w = fnum(row.get("home_east_to_west"))
    a_w2e = fnum(row.get("away_west_to_east"))
    h_w2e = fnum(row.get("home_west_to_east"))
    out["travel_net_miles_1000"] = (
        None if away_miles is None or home_miles is None
        else (away_miles - home_miles) / 1000.0
    )
    out["travel_net_time_zones"] = (
        None if away_tz is None or home_tz is None else away_tz - home_tz
    )
    out["travel_net_east_to_west"] = (
        None if a_e2w is None or h_e2w is None else a_e2w - h_e2w
    )
    out["travel_net_west_to_east"] = (
        None if a_w2e is None or h_w2e is None else a_w2e - h_w2e
    )
    out["travel_international"] = fnum(row.get("international_flag"))
    return out


def side_specific_features(
    game_features: dict[str, Any],
    rating_features: dict[str, Any],
    is_home: bool,
) -> dict[str, Any]:
    out = dict(rating_features)
    out["context_is_home"] = 1.0 if is_home else 0.0
    for key, value in game_features.items():
        if key.startswith(("context_", "travel_")):
            out[key] = value

    # Convert market/ESPN into the offense team's perspective.
    pace_base = fnum(rating_features.get("rating_expected_pace"))
    market_team_score = fnum(
        game_features.get("market_home_implied_score" if is_home else "market_away_implied_score")
    )
    out["market_team_implied_score"] = market_team_score
    out["market_opponent_implied_score"] = fnum(
        game_features.get("market_away_implied_score" if is_home else "market_home_implied_score")
    )
    out["market_total"] = fnum(game_features.get("market_total"))
    out["market_abs_margin"] = fnum(game_features.get("market_abs_margin"))
    out["market_team_implied_ppd"] = (
        None if market_team_score is None or pace_base is None or pace_base <= 0
        else market_team_score / pace_base
    )
    out["market_available"] = fnum(game_features.get("market_available"))

    out["espn_team_probability"] = fnum(
        game_features.get("espn_home_probability" if is_home else "espn_away_probability")
    )
    out["espn_opponent_probability"] = fnum(
        game_features.get("espn_away_probability" if is_home else "espn_home_probability")
    )
    out["espn_team_ptdiff"] = fnum(
        game_features.get("espn_home_ptdiff" if is_home else "espn_away_ptdiff")
    )
    out["espn_available"] = fnum(game_features.get("espn_available"))
    return out


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
        travel = load_historical_travel(season)
        frame = schedule.merge(
            finals_by_season[season], on="game_id", how="inner", validate="one_to_one"
        )
        frame = frame.merge(
            market, on="game_id", how="left", validate="one_to_one", suffixes=("", "_market")
        )
        frame = frame.merge(
            travel, on="game_id", how="left", validate="one_to_one", suffixes=("", "_travel")
        )
        frame["season"] = season
        frames.append(frame)
    games = pd.concat(frames, ignore_index=True)
    return games.sort_values(["season", "week", "game_id"], kind="stable").reset_index(drop=True)


def build_training_matrices(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    prior_carry: float,
) -> tuple[pd.DataFrame, pd.DataFrame, DynamicRatingBook]:
    book = DynamicRatingBook(prior_carry=prior_carry)
    tg = team_games.copy()
    tg["game_id"] = tg["game_id"].map(gid)
    tg["is_home_num"] = pd.to_numeric(tg["is_home"], errors="coerce")
    lookup = {
        (gid(r["game_id"]), int(r["is_home_num"])): r
        for _, r in tg.dropna(subset=["is_home_num"]).iterrows()
    }

    game_rows: list[dict[str, Any]] = []
    ppd_rows: list[dict[str, Any]] = []

    for (season, week), week_games in games.groupby(["season", "week"], sort=True):
        season = int(season)
        week = int(week)
        book.transition_to(season)
        staged_updates: list[tuple[str, dict[str, float]]] = []
        league_obs: list[dict[str, float]] = []

        for _, game in week_games.iterrows():
            game_id = gid(game["game_id"])
            home_team = clean(game["home_team"])
            away_team = clean(game["away_team"])
            hrow = lookup.get((game_id, 1))
            arow = lookup.get((game_id, 0))
            if hrow is None or arow is None:
                continue
            h_drives = fnum(hrow.get("off_drives"))
            a_drives = fnum(arow.get("off_drives"))
            h_ppd = fnum(hrow.get("effective_ppd"))
            a_ppd = fnum(arow.get("effective_ppd"))
            actual_pace = safe_mean([h_drives, a_drives])
            if actual_pace is None or h_ppd is None or a_ppd is None:
                continue

            h_rating = book.snapshot(home_team, away_team)
            a_rating = book.snapshot(away_team, home_team)

            gfeat: dict[str, Any] = {}
            gfeat.update(market_game_features(game))
            gfeat.update(espn_game_features(game, historical=True))
            gfeat.update(context_features(game))
            gfeat.update(travel_features(game))

            home_final = fnum(game.get("home_final"))
            away_final = fnum(game.get("away_final"))
            actual_margin = None if home_final is None or away_final is None else home_final - away_final
            actual_total = None if home_final is None or away_final is None else home_final + away_final

            pace_features = {
                "rating_expected_pace": h_rating["rating_expected_pace"],
                "rating_home_pace_offset": h_rating["rating_team_pace_offset"],
                "rating_away_pace_offset": a_rating["rating_team_pace_offset"],
                "rating_home_games_current": h_rating["rating_team_games_current"],
                "rating_away_games_current": a_rating["rating_team_games_current"],
                "rating_home_games_previous": h_rating["rating_team_games_previous"],
                "rating_away_games_previous": a_rating["rating_team_games_previous"],
                "rating_home_reliability": h_rating["rating_team_reliability"],
                "rating_away_reliability": a_rating["rating_team_reliability"],
                "rating_league_pace": h_rating["rating_league_pace"],
                "rating_home_off_ppd": h_rating["rating_team_off_ppd"],
                "rating_away_off_ppd": a_rating["rating_team_off_ppd"],
                "rating_home_def_ppd": h_rating["rating_team_def_ppd"],
                "rating_away_def_ppd": a_rating["rating_team_def_ppd"],
                **{k: v for k, v in gfeat.items() if k.startswith(("market_", "espn_", "context_", "travel_"))},
            }
            game_rows.append(
                {
                    "row_id": game_id,
                    "season": season,
                    "week": week,
                    "game_id": game_id,
                    "game_date": clean(game.get("game_date")),
                    "away_team": away_team,
                    "home_team": home_team,
                    "home_final": home_final,
                    "away_final": away_final,
                    "actual_margin": actual_margin,
                    "actual_total": actual_total,
                    "actual_pace": actual_pace,
                    "actual_home_ppd": h_ppd,
                    "actual_away_ppd": a_ppd,
                    **pace_features,
                }
            )

            for is_home, team, opponent, rating, actual_ppd in [
                (True, home_team, away_team, h_rating, h_ppd),
                (False, away_team, home_team, a_rating, a_ppd),
            ]:
                ppd_rows.append(
                    {
                        "row_id": f"{game_id}:{1 if is_home else 0}",
                        "season": season,
                        "week": week,
                        "game_id": game_id,
                        "is_home": 1 if is_home else 0,
                        "team": team,
                        "opponent": opponent,
                        "actual_ppd": actual_ppd,
                        **side_specific_features(gfeat, rating, is_home),
                    }
                )

            updates, obs = book.stage_game_updates(home_team, away_team, hrow, arow)
            staged_updates.extend(updates)
            league_obs.append(obs)

        # No game in a week sees another result from that same week.
        book.apply_week(staged_updates, league_obs)

    return pd.DataFrame(game_rows), pd.DataFrame(ppd_rows), book


# ---------------------------------------------------------------------------
# Model search
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    scale: bool = False


def model_specs(fast: bool) -> list[ModelSpec]:
    specs: list[ModelSpec] = [
        ModelSpec("ridge_25", Ridge(alpha=25.0), True),
        ModelSpec("ridge_100", Ridge(alpha=100.0), True),
        ModelSpec(
            "extra_4",
            ExtraTreesRegressor(
                n_estimators=250 if fast else 700,
                min_samples_leaf=4,
                max_features=0.80,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            "rf_6",
            RandomForestRegressor(
                n_estimators=220 if fast else 600,
                min_samples_leaf=6,
                max_features=0.75,
                random_state=RANDOM_STATE + 1,
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
                l2_regularization=15.0,
                random_state=RANDOM_STATE,
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
        specs.append(
            ModelSpec(
                "xgb_d2",
                XGBRegressor(
                    n_estimators=250 if fast else 700,
                    max_depth=2,
                    learning_rate=0.03,
                    min_child_weight=20,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    reg_alpha=1.0,
                    reg_lambda=30.0,
                    objective="reg:absoluteerror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            )
        )
    if LGBMRegressor is not None:
        specs.append(
            ModelSpec(
                "lgb_15",
                LGBMRegressor(
                    n_estimators=250 if fast else 700,
                    learning_rate=0.025,
                    num_leaves=15,
                    min_child_samples=40,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    reg_alpha=1.0,
                    reg_lambda=25.0,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                ),
            )
        )
    return specs


def make_pipeline(spec: ModelSpec) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if spec.scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", clone(spec.estimator)))
    return Pipeline(steps)


def usable_numeric_features(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= max(30, int(len(df) * 0.02)):
            cols.append(col)
    return sorted(cols)


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    rating = [c for c in columns if c.startswith("rating_")]
    context = [c for c in columns if c.startswith("context_")]
    market = [c for c in columns if c.startswith("market_")]
    espn = [c for c in columns if c.startswith("espn_")]
    travel = [c for c in columns if c.startswith("travel_")]

    def uniq(*parts: list[str]) -> list[str]:
        return sorted(set().union(*[set(p) for p in parts]))

    groups = {
        "ratings": uniq(rating, context),
        "ratings_market": uniq(rating, context, market),
        "ratings_market_espn": uniq(rating, context, market, espn),
        "full": uniq(rating, context, market, espn, travel),
    }
    return {name: cols for name, cols in groups.items() if cols}


@dataclass(frozen=True)
class ModelChoice:
    spec_name: str
    group_name: str
    features: tuple[str, ...]
    mode: str  # direct | residual

    @property
    def name(self) -> str:
        return f"{self.group_name}|{self.spec_name}|{self.mode}"


def find_spec(specs: list[ModelSpec], name: str) -> ModelSpec:
    for spec in specs:
        if spec.name == name:
            return spec
    raise KeyError(name)


def fit_choice(
    df: pd.DataFrame,
    choice: ModelChoice,
    specs: list[ModelSpec],
    target: str,
    baseline: str,
) -> Pipeline:
    work = df.copy()
    X = work[list(choice.features)].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(work[target], errors="coerce")
    if choice.mode == "residual":
        base = pd.to_numeric(work[baseline], errors="coerce")
        y = y - base
    mask = y.notna()
    if choice.mode == "residual":
        mask &= pd.to_numeric(work[baseline], errors="coerce").notna()
    pipe = make_pipeline(find_spec(specs, choice.spec_name))
    pipe.fit(X.loc[mask], y.loc[mask])
    return pipe


def predict_choice(
    fitted: Pipeline,
    choice: ModelChoice,
    df: pd.DataFrame,
    baseline: str,
) -> np.ndarray:
    X = df[list(choice.features)].apply(pd.to_numeric, errors="coerce")
    raw = np.asarray(fitted.predict(X), dtype=float)
    if choice.mode == "residual":
        base = pd.to_numeric(df[baseline], errors="coerce").to_numpy(float)
        return base + raw
    return raw


def forward_component_search(
    df: pd.DataFrame,
    target: str,
    baseline: str,
    specs: list[ModelSpec],
    groups: dict[str, list[str]],
    component_name: str,
) -> tuple[ModelChoice, pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, Any]] = []
    best_key: tuple[float, int, float] | None = None
    best_choice: ModelChoice | None = None
    best_oof: pd.DataFrame | None = None

    for group_name, features in groups.items():
        for spec in specs:
            for mode in ("residual", "direct"):
                choice = ModelChoice(spec.name, group_name, tuple(features), mode)
                fold_rows: list[dict[str, Any]] = []
                oof_parts: list[pd.DataFrame] = []
                for val_season in CV_VALIDATION_SEASONS:
                    train = df[pd.to_numeric(df["season"], errors="coerce").lt(val_season)].copy()
                    valid = df[pd.to_numeric(df["season"], errors="coerce").eq(val_season)].copy()
                    valid_target = pd.to_numeric(valid[target], errors="coerce")
                    valid_base = pd.to_numeric(valid[baseline], errors="coerce")
                    mask = valid_target.notna() & valid_base.notna()
                    valid = valid.loc[mask].copy()
                    if len(train) < 200 or len(valid) < 100:
                        continue
                    try:
                        fit = fit_choice(train, choice, specs, target, baseline)
                        pred = predict_choice(fit, choice, valid, baseline)
                    except Exception as exc:
                        fold_rows = []
                        oof_parts = []
                        break
                    actual = pd.to_numeric(valid[target], errors="coerce").to_numpy(float)
                    base = pd.to_numeric(valid[baseline], errors="coerce").to_numpy(float)
                    pred = np.asarray(pred, dtype=float)
                    mae = mean_absolute_error(actual, pred)
                    base_mae = mean_absolute_error(actual, base)
                    fold_rows.append(
                        {
                            "component": component_name,
                            "choice": choice.name,
                            "validation_season": val_season,
                            "rows": len(valid),
                            "mae": mae,
                            "baseline_mae": base_mae,
                            "improvement": base_mae - mae,
                        }
                    )
                    part = valid[["row_id", "season", "game_id"]].copy()
                    if "is_home" in valid.columns:
                        part["is_home"] = valid["is_home"].values
                    part["actual"] = actual
                    part["baseline"] = base
                    part["prediction"] = pred
                    oof_parts.append(part)

                if len(fold_rows) != len(CV_VALIDATION_SEASONS):
                    continue
                fold_df = pd.DataFrame(fold_rows)
                mean_mae = float(fold_df["mae"].mean())
                mean_base = float(fold_df["baseline_mae"].mean())
                wins = int((fold_df["improvement"] > 0).sum())
                mean_improvement = mean_base - mean_mae
                summary = {
                    "component": component_name,
                    "choice": choice.name,
                    "group": group_name,
                    "model": spec.name,
                    "mode": mode,
                    "feature_count": len(features),
                    "mean_mae": mean_mae,
                    "mean_baseline_mae": mean_base,
                    "mean_improvement": mean_improvement,
                    "fold_wins": wins,
                }
                results.append(summary)
                # Prefer lower MAE, then more fold wins, then simpler feature count.
                key = (mean_mae, -wins, float(len(features)))
                if best_key is None or key < best_key:
                    best_key = key
                    best_choice = choice
                    best_oof = pd.concat(oof_parts, ignore_index=True)

    if best_choice is None or best_oof is None:
        raise RuntimeError(f"No valid forward-CV candidates for {component_name}")
    result_df = pd.DataFrame(results).sort_values(
        ["mean_mae", "fold_wins", "feature_count"], ascending=[True, False, True]
    )
    return best_choice, result_df, best_oof


def clip_pace(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), PACE_MIN, PACE_MAX)


def clip_ppd(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), PPD_MIN, PPD_MAX)


def clip_score(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), SCORE_MIN, SCORE_MAX)


def assemble_score_predictions(
    games: pd.DataFrame,
    pace_predictions: pd.DataFrame,
    ppd_predictions: pd.DataFrame,
) -> pd.DataFrame:
    pace = pace_predictions[["game_id", "season", "prediction"]].rename(
        columns={"prediction": "predicted_pace"}
    )
    ppd = ppd_predictions.copy()
    ppd["is_home"] = pd.to_numeric(ppd["is_home"], errors="coerce").astype(int)
    hp = ppd[ppd["is_home"].eq(1)][["game_id", "prediction"]].rename(
        columns={"prediction": "predicted_home_ppd"}
    )
    ap = ppd[ppd["is_home"].eq(0)][["game_id", "prediction"]].rename(
        columns={"prediction": "predicted_away_ppd"}
    )
    merged = games.merge(pace, on=["game_id", "season"], how="inner").merge(
        hp, on="game_id", how="inner"
    ).merge(ap, on="game_id", how="inner")
    merged["predicted_pace"] = clip_pace(
        pd.to_numeric(merged["predicted_pace"], errors="coerce").to_numpy(float)
    )
    merged["predicted_home_ppd"] = clip_ppd(
        pd.to_numeric(merged["predicted_home_ppd"], errors="coerce").to_numpy(float)
    )
    merged["predicted_away_ppd"] = clip_ppd(
        pd.to_numeric(merged["predicted_away_ppd"], errors="coerce").to_numpy(float)
    )
    merged["component_home_score"] = clip_score(
        merged["predicted_pace"].to_numpy(float) * merged["predicted_home_ppd"].to_numpy(float)
    )
    merged["component_away_score"] = clip_score(
        merged["predicted_pace"].to_numpy(float) * merged["predicted_away_ppd"].to_numpy(float)
    )
    merged["component_margin"] = merged["component_home_score"] - merged["component_away_score"]
    merged["component_total"] = merged["component_home_score"] + merged["component_away_score"]
    return merged


def team_score_mae(df: pd.DataFrame, home_col: str, away_col: str) -> float:
    h = np.abs(
        pd.to_numeric(df["home_final"], errors="coerce").to_numpy(float)
        - pd.to_numeric(df[home_col], errors="coerce").to_numpy(float)
    )
    a = np.abs(
        pd.to_numeric(df["away_final"], errors="coerce").to_numpy(float)
        - pd.to_numeric(df[away_col], errors="coerce").to_numpy(float)
    )
    return float(np.mean(np.concatenate([h, a])))


def choose_market_blend(score_oof: pd.DataFrame) -> tuple[float, pd.DataFrame, bool]:
    lined = score_oof[
        pd.to_numeric(score_oof["market_home_implied_score"], errors="coerce").notna()
        & pd.to_numeric(score_oof["market_away_implied_score"], errors="coerce").notna()
    ].copy()
    if len(lined) < 500:
        raise RuntimeError(f"Too few lined OOF games for score blend: {len(lined)}")

    rows: list[dict[str, Any]] = []
    best_alpha = 0.0
    best_mae = math.inf
    for alpha in MARKET_BLEND_GRID:
        lined["blend_home"] = (
            lined["market_home_implied_score"]
            + alpha * (lined["component_home_score"] - lined["market_home_implied_score"])
        )
        lined["blend_away"] = (
            lined["market_away_implied_score"]
            + alpha * (lined["component_away_score"] - lined["market_away_implied_score"])
        )
        mae = team_score_mae(lined, "blend_home", "blend_away")
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)

    fold_wins = 0
    for season in CV_VALIDATION_SEASONS:
        fold = lined[pd.to_numeric(lined["season"], errors="coerce").eq(season)].copy()
        if fold.empty:
            continue
        fold["frozen_home"] = (
            fold["market_home_implied_score"]
            + best_alpha * (fold["component_home_score"] - fold["market_home_implied_score"])
        )
        fold["frozen_away"] = (
            fold["market_away_implied_score"]
            + best_alpha * (fold["component_away_score"] - fold["market_away_implied_score"])
        )
        model_mae = team_score_mae(fold, "frozen_home", "frozen_away")
        market_mae = team_score_mae(
            fold, "market_home_implied_score", "market_away_implied_score"
        )
        improvement = market_mae - model_mae
        if improvement > 0:
            fold_wins += 1
        rows.append(
            {
                "validation_season": season,
                "alpha": best_alpha,
                "games": len(fold),
                "model_team_score_mae": model_mae,
                "market_team_score_mae": market_mae,
                "improvement": improvement,
            }
        )

    market_mae = team_score_mae(
        lined, "market_home_implied_score", "market_away_implied_score"
    )
    overall_improvement = market_mae - best_mae
    accepted = bool(
        best_alpha > 0
        and overall_improvement >= CV_MIN_TEAM_SCORE_IMPROVEMENT
        and fold_wins >= CV_REQUIRED_SCORE_FOLD_WINS
    )
    rows.append(
        {
            "validation_season": "ALL_2022_2024",
            "alpha": best_alpha,
            "games": len(lined),
            "model_team_score_mae": best_mae,
            "market_team_score_mae": market_mae,
            "improvement": overall_improvement,
            "fold_wins": fold_wins,
            "accepted": accepted,
        }
    )
    return best_alpha, pd.DataFrame(rows), accepted


def winner_accuracy(actual_margin: np.ndarray, predicted_margin: np.ndarray) -> float:
    actual = np.sign(np.asarray(actual_margin, dtype=float))
    pred = np.sign(np.asarray(predicted_margin, dtype=float))
    mask = actual != 0
    return float(np.mean(actual[mask] == pred[mask])) if mask.any() else float("nan")


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def evaluate_2025(
    valid: pd.DataFrame,
    alpha: float,
    cv_accepted: bool,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    lined = valid[
        pd.to_numeric(valid["market_home_implied_score"], errors="coerce").notna()
        & pd.to_numeric(valid["market_away_implied_score"], errors="coerce").notna()
    ].copy()
    if len(lined) < 300:
        raise RuntimeError(f"Too few lined 2025 validation games: {len(lined)}")

    lined["candidate_home_score"] = clip_score(
        lined["market_home_implied_score"].to_numpy(float)
        + alpha * (
            lined["component_home_score"].to_numpy(float)
            - lined["market_home_implied_score"].to_numpy(float)
        )
    )
    lined["candidate_away_score"] = clip_score(
        lined["market_away_implied_score"].to_numpy(float)
        + alpha * (
            lined["component_away_score"].to_numpy(float)
            - lined["market_away_implied_score"].to_numpy(float)
        )
    )
    lined["candidate_margin"] = lined["candidate_home_score"] - lined["candidate_away_score"]
    lined["candidate_total"] = lined["candidate_home_score"] + lined["candidate_away_score"]

    actual_margin = pd.to_numeric(lined["actual_margin"], errors="coerce").to_numpy(float)
    actual_total = pd.to_numeric(lined["actual_total"], errors="coerce").to_numpy(float)
    market_margin = pd.to_numeric(lined["market_margin"], errors="coerce").to_numpy(float)
    market_total = pd.to_numeric(lined["market_total"], errors="coerce").to_numpy(float)
    cand_margin = lined["candidate_margin"].to_numpy(float)
    cand_total = lined["candidate_total"].to_numpy(float)

    market_team_mae = team_score_mae(
        lined, "market_home_implied_score", "market_away_implied_score"
    )
    candidate_team_mae = team_score_mae(
        lined, "candidate_home_score", "candidate_away_score"
    )
    margin_market_mae = mean_absolute_error(actual_margin, market_margin)
    margin_candidate_mae = mean_absolute_error(actual_margin, cand_margin)
    total_market_mae = mean_absolute_error(actual_total, market_total)
    total_candidate_mae = mean_absolute_error(actual_total, cand_total)
    market_win = winner_accuracy(actual_margin, market_margin)
    cand_win = winner_accuracy(actual_margin, cand_margin)

    team_improvement = market_team_mae - candidate_team_mae
    margin_delta = margin_candidate_mae - margin_market_mae
    total_delta = total_candidate_mae - total_market_mae
    winner_delta = cand_win - market_win

    validated = bool(
        cv_accepted
        and team_improvement >= FINAL_MIN_TEAM_SCORE_IMPROVEMENT
        and margin_delta <= FINAL_MAX_MARGIN_DEGRADATION
        and total_delta <= FINAL_MAX_TOTAL_DEGRADATION
        and winner_delta >= -FINAL_MAX_WINNER_ACCURACY_DEGRADATION
    )

    metrics = pd.DataFrame(
        [
            {
                "metric": "team_score_mae",
                "candidate": candidate_team_mae,
                "market": market_team_mae,
                "improvement": team_improvement,
            },
            {
                "metric": "margin_mae",
                "candidate": margin_candidate_mae,
                "market": margin_market_mae,
                "improvement": margin_market_mae - margin_candidate_mae,
            },
            {
                "metric": "total_mae",
                "candidate": total_candidate_mae,
                "market": total_market_mae,
                "improvement": total_market_mae - total_candidate_mae,
            },
            {
                "metric": "winner_accuracy",
                "candidate": cand_win,
                "market": market_win,
                "improvement": winner_delta,
            },
            {
                "metric": "margin_rmse",
                "candidate": rmse(actual_margin, cand_margin),
                "market": rmse(actual_margin, market_margin),
                "improvement": rmse(actual_margin, market_margin) - rmse(actual_margin, cand_margin),
            },
            {
                "metric": "total_rmse",
                "candidate": rmse(actual_total, cand_total),
                "market": rmse(actual_total, market_total),
                "improvement": rmse(actual_total, market_total) - rmse(actual_total, cand_total),
            },
        ]
    )
    gates = {
        "cv_accepted": cv_accepted,
        "frozen_market_blend_alpha": alpha,
        "team_score_improvement": team_improvement,
        "margin_degradation": margin_delta,
        "total_degradation": total_delta,
        "winner_accuracy_delta": winner_delta,
        "validated": validated,
    }
    return metrics, gates, lined


# ---------------------------------------------------------------------------
# Current-season ratings and predictions
# ---------------------------------------------------------------------------

def update_book_with_current_prior_weeks(
    book: DynamicRatingBook,
    season: int,
    target_week: int,
    resolver: TeamResolver,
) -> None:
    book.transition_to(season)
    if target_week <= 1:
        return
    pbp_path = PBP_DIR / f"{season}_pbp.parquet"
    if not pbp_path.is_file():
        print("No current-season PBP file found; using carried ratings only.")
        return
    schedule = load_schedule(season)
    schedule = schedule[schedule["week"].lt(target_week)].copy()
    if schedule.empty:
        return
    tg, _, _ = build_team_game_features_for_season(season, schedule, resolver)
    if tg.empty:
        return
    tg["game_id"] = tg["game_id"].map(gid)
    tg["is_home_num"] = pd.to_numeric(tg["is_home"], errors="coerce")
    lookup = {
        (gid(r["game_id"]), int(r["is_home_num"])): r
        for _, r in tg.dropna(subset=["is_home_num"]).iterrows()
    }
    schedule["home_team"] = schedule["home_team"].map(resolver.resolve)
    schedule["away_team"] = schedule["away_team"].map(resolver.resolve)
    for week in sorted(schedule["week"].unique()):
        staged: list[tuple[str, dict[str, float]]] = []
        league_obs: list[dict[str, float]] = []
        for _, game in schedule[schedule["week"].eq(week)].iterrows():
            hrow = lookup.get((gid(game["game_id"]), 1))
            arow = lookup.get((gid(game["game_id"]), 0))
            if hrow is None or arow is None:
                continue
            updates, obs = book.stage_game_updates(
                game["home_team"], game["away_team"], hrow, arow
            )
            staged.extend(updates)
            league_obs.append(obs)
        book.apply_week(staged, league_obs)


def build_current_frames(
    season: int,
    week: int,
    book: DynamicRatingBook,
    resolver: TeamResolver,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly_path = WEEKLY_SCHEDULE_DIR / f"week_{week}_CFB_weekly_schedule.csv"
    enriched_path = ENRICHED_DIR / f"week_{week}_CFB_enriched.csv"
    travel_path = TRAVEL_DIR / f"{season}_week_{week}_travel.csv"

    weekly = read_csv(
        weekly_path, ["game_id", "away_team", "home_team", "home_spread", "total"]
    ).copy()
    weekly["game_id"] = weekly["game_id"].map(gid)
    enriched = (
        read_csv(enriched_path, ["game_id"]).copy()
        if enriched_path.is_file()
        else pd.DataFrame(columns=["game_id"])
    )
    travel = (
        read_csv(travel_path, ["game_id"]).copy()
        if travel_path.is_file()
        else pd.DataFrame(columns=["game_id"])
    )
    for frame in (enriched, travel):
        if "game_id" in frame.columns:
            frame["game_id"] = frame["game_id"].map(gid)

    merged = weekly.copy()
    if not enriched.empty:
        # Avoid duplicate market fields from enriched; weekly schedule is source of truth.
        duplicate = [
            c for c in enriched.columns if c != "game_id" and c in merged.columns
        ]
        enriched = enriched.drop(columns=duplicate, errors="ignore")
        merged = merged.merge(enriched, on="game_id", how="left")
    if not travel.empty:
        duplicate = [c for c in travel.columns if c != "game_id" and c in merged.columns]
        travel = travel.drop(columns=duplicate, errors="ignore")
        merged = merged.merge(travel, on="game_id", how="left")

    game_rows: list[dict[str, Any]] = []
    ppd_rows: list[dict[str, Any]] = []
    for _, game in merged.iterrows():
        home = resolver.resolve(game["home_team"])
        away = resolver.resolve(game["away_team"])
        h_rating = book.snapshot(home, away)
        a_rating = book.snapshot(away, home)
        gfeat: dict[str, Any] = {}
        gfeat.update(market_game_features(game))
        gfeat.update(espn_game_features(game, historical=False))
        gfeat.update(context_features(game))
        gfeat.update(travel_features(game))

        pace_features = {
            "rating_expected_pace": h_rating["rating_expected_pace"],
            "rating_home_pace_offset": h_rating["rating_team_pace_offset"],
            "rating_away_pace_offset": a_rating["rating_team_pace_offset"],
            "rating_home_games_current": h_rating["rating_team_games_current"],
            "rating_away_games_current": a_rating["rating_team_games_current"],
            "rating_home_games_previous": h_rating["rating_team_games_previous"],
            "rating_away_games_previous": a_rating["rating_team_games_previous"],
            "rating_home_reliability": h_rating["rating_team_reliability"],
            "rating_away_reliability": a_rating["rating_team_reliability"],
            "rating_league_pace": h_rating["rating_league_pace"],
            "rating_home_off_ppd": h_rating["rating_team_off_ppd"],
            "rating_away_off_ppd": a_rating["rating_team_off_ppd"],
            "rating_home_def_ppd": h_rating["rating_team_def_ppd"],
            "rating_away_def_ppd": a_rating["rating_team_def_ppd"],
            **{k: v for k, v in gfeat.items() if k.startswith(("market_", "espn_", "context_", "travel_"))},
        }
        game_rows.append(
            {
                "row_id": gid(game["game_id"]),
                "season": season,
                "week": week,
                "game_id": gid(game["game_id"]),
                "game_date": clean(game.get("game_date")),
                "game_time": clean(game.get("game_time")),
                "away_team": away,
                "home_team": home,
                **pace_features,
            }
        )
        for is_home, team, opponent, rating in [
            (True, home, away, h_rating),
            (False, away, home, a_rating),
        ]:
            ppd_rows.append(
                {
                    "row_id": f"{gid(game['game_id'])}:{1 if is_home else 0}",
                    "season": season,
                    "week": week,
                    "game_id": gid(game["game_id"]),
                    "is_home": 1 if is_home else 0,
                    "team": team,
                    "opponent": opponent,
                    **side_specific_features(gfeat, rating, is_home),
                }
            )
    return pd.DataFrame(game_rows), pd.DataFrame(ppd_rows)


def ensure_features(frame: pd.DataFrame, features: Iterable[str]) -> None:
    for col in features:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")


def predict_component_frames(
    game_frame: pd.DataFrame,
    ppd_frame: pd.DataFrame,
    pace_fit: Pipeline,
    pace_choice: ModelChoice,
    ppd_fit: Pipeline,
    ppd_choice: ModelChoice,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_features(game_frame, pace_choice.features)
    ensure_features(ppd_frame, ppd_choice.features)
    pace_pred = clip_pace(
        predict_choice(pace_fit, pace_choice, game_frame, "rating_expected_pace")
    )
    ppd_pred = clip_ppd(
        predict_choice(ppd_fit, ppd_choice, ppd_frame, "rating_expected_ppd")
    )
    p1 = game_frame[["row_id", "season", "game_id"]].copy()
    p1["prediction"] = pace_pred
    p2 = ppd_frame[["row_id", "season", "game_id", "is_home"]].copy()
    p2["prediction"] = ppd_pred
    scores = assemble_score_predictions(game_frame, p1, p2)
    return p1, p2, scores


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resolver = TeamResolver(TEAM_MAP_PATH)

    schedules = {season: load_schedule(season) for season in HISTORICAL_SEASONS}
    print("Building game-level PBP features...")
    team_games, finals_by_season, optional_aliases = load_or_build_historical_team_games(
        resolver, schedules, rebuild=args.rebuild_pbp_features
    )

    print("Building dynamic opponent-adjusted ratings and component targets...")
    base_games = build_historical_base_games(schedules, finals_by_season, resolver)
    training_games, training_ppd, final_book = build_training_matrices(
        base_games, team_games, prior_carry=args.prior_carry
    )
    write_csv(training_games, OUT_DIR / "training_games_2021_2025_v3.csv")
    write_csv(training_ppd, OUT_DIR / "training_ppd_rows_2021_2025_v3.csv")
    print(f"Historical games with component targets: {len(training_games)}")
    print(f"Historical team-PPD rows: {len(training_ppd)}")

    # Freeze feature schemas using <=2024 only. 2025 cannot influence feature inclusion.
    game_probe = training_games[pd.to_numeric(training_games["season"], errors="coerce").le(2024)].copy()
    ppd_probe = training_ppd[pd.to_numeric(training_ppd["season"], errors="coerce").le(2024)].copy()
    pace_excluded = {
        "row_id", "season", "week", "game_id", "game_date", "away_team", "home_team",
        "home_final", "away_final", "actual_margin", "actual_total", "actual_pace",
        "actual_home_ppd", "actual_away_ppd",
    }
    ppd_excluded = {
        "row_id", "season", "week", "game_id", "is_home", "team", "opponent", "actual_ppd",
    }
    pace_features = usable_numeric_features(game_probe, pace_excluded)
    ppd_features = usable_numeric_features(ppd_probe, ppd_excluded)
    for c in pace_features:
        training_games[c] = pd.to_numeric(training_games[c], errors="coerce")
    for c in ppd_features:
        training_ppd[c] = pd.to_numeric(training_ppd[c], errors="coerce")
    pace_groups = feature_groups(pace_features)
    ppd_groups = feature_groups(ppd_features)

    specs = model_specs(args.fast)
    print(f"PACE pregame features: {len(pace_features)}")
    print(f"PPD pregame features: {len(ppd_features)}")
    print(f"Model configurations: {len(specs)}")

    print("Selecting possession/pace model with 2022-2024 forward validation...")
    pace_choice, pace_cv, pace_oof = forward_component_search(
        training_games,
        target="actual_pace",
        baseline="rating_expected_pace",
        specs=specs,
        groups=pace_groups,
        component_name="PACE",
    )
    print(f"  best pace={pace_choice.name}")
    write_csv(pace_cv, OUT_DIR / "pace_cv_results_v3.csv")

    print("Selecting team scoring-efficiency model with 2022-2024 forward validation...")
    ppd_choice, ppd_cv, ppd_oof = forward_component_search(
        training_ppd,
        target="actual_ppd",
        baseline="rating_expected_ppd",
        specs=specs,
        groups=ppd_groups,
        component_name="EFFECTIVE_PPD",
    )
    print(f"  best ppd={ppd_choice.name}")
    write_csv(ppd_cv, OUT_DIR / "ppd_cv_results_v3.csv")

    score_oof_games = training_games[
        pd.to_numeric(training_games["season"], errors="coerce").isin(CV_VALIDATION_SEASONS)
    ].copy()
    score_oof = assemble_score_predictions(score_oof_games, pace_oof, ppd_oof)
    alpha, score_cv, cv_accepted = choose_market_blend(score_oof)
    write_csv(score_cv, OUT_DIR / "score_cv_results_v3.csv")
    all_cv = score_cv[score_cv["validation_season"].astype(str).eq("ALL_2022_2024")].iloc[0]
    print(
        "Frozen score blend from 2022-2024: "
        f"alpha={alpha:.2f} improvement={float(all_cv['improvement']):.4f} "
        f"fold_wins={int(fnum(all_cv.get('fold_wins')) or 0)} accepted={cv_accepted}"
    )

    # Everything above is frozen. 2025 is examined for the first time below.
    train_game_2024 = training_games[pd.to_numeric(training_games["season"], errors="coerce").le(2024)].copy()
    train_ppd_2024 = training_ppd[pd.to_numeric(training_ppd["season"], errors="coerce").le(2024)].copy()
    valid_game_2025 = training_games[pd.to_numeric(training_games["season"], errors="coerce").eq(2025)].copy()
    valid_ppd_2025 = training_ppd[pd.to_numeric(training_ppd["season"], errors="coerce").eq(2025)].copy()

    pace_fit_2024 = fit_choice(
        train_game_2024, pace_choice, specs, "actual_pace", "rating_expected_pace"
    )
    ppd_fit_2024 = fit_choice(
        train_ppd_2024, ppd_choice, specs, "actual_ppd", "rating_expected_ppd"
    )
    _, _, score_2025 = predict_component_frames(
        valid_game_2025.copy(), valid_ppd_2025.copy(),
        pace_fit_2024, pace_choice, ppd_fit_2024, ppd_choice,
    )
    metrics, gates, validation_2025 = evaluate_2025(score_2025, alpha, cv_accepted)
    write_csv(validation_2025, OUT_DIR / "validation_2025_predictions_v3.csv")
    write_csv(metrics, OUT_DIR / "validation_2025_metrics_v3.csv")

    metric_lookup = {r["metric"]: r for _, r in metrics.iterrows()}
    print("2025 FINAL validation (not used for model or blend selection):")
    for metric_name in ["team_score_mae", "margin_mae", "total_mae", "winner_accuracy"]:
        r = metric_lookup[metric_name]
        print(
            f"  {metric_name}: candidate={float(r['candidate']):.4f} "
            f"market={float(r['market']):.4f} improvement={float(r['improvement']):+.4f}"
        )
    print(f"  MODEL STATUS: {'VALIDATED' if gates['validated'] else 'REJECTED'}")

    # Refit frozen architecture on all 2021-2025 only AFTER final validation.
    pace_fit_all = fit_choice(
        training_games, pace_choice, specs, "actual_pace", "rating_expected_pace"
    )
    ppd_fit_all = fit_choice(
        training_ppd, ppd_choice, specs, "actual_ppd", "rating_expected_ppd"
    )

    model_bundle = {
        "script_version": SCRIPT_VERSION,
        "pace_choice": pace_choice,
        "ppd_choice": ppd_choice,
        "pace_model": pace_fit_all,
        "ppd_model": ppd_fit_all,
        "market_blend_alpha": alpha,
        "cv_accepted": cv_accepted,
        "validated_2025": bool(gates["validated"]),
        "prior_carry": args.prior_carry,
    }
    model_path = OUT_DIR / "score_model_v3.joblib"
    tmp_model = model_path.with_suffix(".joblib.tmp")
    joblib.dump(model_bundle, tmp_model)
    os.replace(tmp_model, model_path)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "historical_seasons": HISTORICAL_SEASONS,
        "cv_validation_seasons": CV_VALIDATION_SEASONS,
        "final_validation_season": FINAL_VALIDATION_SEASON,
        "architecture": "dynamic opponent-adjusted ratings -> pace + effective PPD -> scores",
        "pace_choice": pace_choice.name,
        "ppd_choice": ppd_choice.name,
        "pace_features": list(pace_choice.features),
        "ppd_features": list(ppd_choice.features),
        "market_blend_alpha": alpha,
        "cv_accepted": cv_accepted,
        "validation_gates": gates,
        "prior_carry": args.prior_carry,
        "historical_weather_used": False,
        "optional_pbp_aliases": optional_aliases,
    }
    write_json(manifest, OUT_DIR / "score_model_manifest_v3.json")

    if args.predict_week is not None:
        week = int(args.predict_week)
        current_book = copy.deepcopy(final_book)
        update_book_with_current_prior_weeks(
            current_book, args.season, week, resolver
        )
        current_games, current_ppd = build_current_frames(
            args.season, week, current_book, resolver
        )
        _, _, current_scores = predict_component_frames(
            current_games.copy(), current_ppd.copy(),
            pace_fit_all, pace_choice, ppd_fit_all, ppd_choice,
        )
        current_scores["candidate_home_score"] = np.where(
            pd.to_numeric(current_scores["market_home_implied_score"], errors="coerce").notna(),
            current_scores["market_home_implied_score"]
            + alpha * (
                current_scores["component_home_score"]
                - current_scores["market_home_implied_score"]
            ),
            current_scores["component_home_score"],
        )
        current_scores["candidate_away_score"] = np.where(
            pd.to_numeric(current_scores["market_away_implied_score"], errors="coerce").notna(),
            current_scores["market_away_implied_score"]
            + alpha * (
                current_scores["component_away_score"]
                - current_scores["market_away_implied_score"]
            ),
            current_scores["component_away_score"],
        )
        current_scores["candidate_home_score"] = clip_score(current_scores["candidate_home_score"].to_numpy(float))
        current_scores["candidate_away_score"] = clip_score(current_scores["candidate_away_score"].to_numpy(float))
        current_scores["candidate_margin"] = current_scores["candidate_home_score"] - current_scores["candidate_away_score"]
        current_scores["candidate_total"] = current_scores["candidate_home_score"] + current_scores["candidate_away_score"]
        current_scores["model_validated_2025"] = 1 if gates["validated"] else 0

        market_available = (
            pd.to_numeric(current_scores["market_home_implied_score"], errors="coerce").notna()
            & pd.to_numeric(current_scores["market_away_implied_score"], errors="coerce").notna()
        )
        if gates["validated"]:
            current_scores["deployed_home_score"] = current_scores["candidate_home_score"]
            current_scores["deployed_away_score"] = current_scores["candidate_away_score"]
            current_scores["deployment_source"] = "VALIDATED_COMPONENT_MODEL"
        else:
            current_scores["deployed_home_score"] = np.where(
                market_available,
                current_scores["market_home_implied_score"],
                current_scores["component_home_score"],
            )
            current_scores["deployed_away_score"] = np.where(
                market_available,
                current_scores["market_away_implied_score"],
                current_scores["component_away_score"],
            )
            current_scores["deployment_source"] = np.where(
                market_available, "MARKET_BASELINE_MODEL_REJECTED", "COMPONENT_FALLBACK_NO_MARKET"
            )
        current_scores["deployed_margin"] = current_scores["deployed_home_score"] - current_scores["deployed_away_score"]
        current_scores["deployed_total"] = current_scores["deployed_home_score"] + current_scores["deployed_away_score"]

        keep = [
            "season", "week", "game_id", "game_date", "game_time", "away_team", "home_team",
            "market_margin", "market_total", "market_away_implied_score", "market_home_implied_score",
            "predicted_pace", "predicted_away_ppd", "predicted_home_ppd",
            "component_away_score", "component_home_score", "component_margin", "component_total",
            "candidate_away_score", "candidate_home_score", "candidate_margin", "candidate_total",
            "deployed_away_score", "deployed_home_score", "deployed_margin", "deployed_total",
            "model_validated_2025", "deployment_source",
        ]
        keep = [c for c in keep if c in current_scores.columns]
        current_out = current_scores[keep].copy()
        current_path = OUT_DIR / f"week_{week}_CFB_trained_score_predictions_v3.csv"
        write_csv(current_out, current_path)
        print(f"Current predictions: {current_path}")

    print(f"Training game matrix: {OUT_DIR / 'training_games_2021_2025_v3.csv'}")
    print(f"Training PPD matrix: {OUT_DIR / 'training_ppd_rows_2021_2025_v3.csv'}")
    print(f"PACE CV: {OUT_DIR / 'pace_cv_results_v3.csv'}")
    print(f"PPD CV: {OUT_DIR / 'ppd_cv_results_v3.csv'}")
    print(f"Score CV: {OUT_DIR / 'score_cv_results_v3.csv'}")
    print(f"2025 predictions: {OUT_DIR / 'validation_2025_predictions_v3.csv'}")
    print(f"2025 metrics: {OUT_DIR / 'validation_2025_metrics_v3.csv'}")
    print(f"Saved model: {model_path}")
    print(f"Manifest: {OUT_DIR / 'score_model_manifest_v3.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

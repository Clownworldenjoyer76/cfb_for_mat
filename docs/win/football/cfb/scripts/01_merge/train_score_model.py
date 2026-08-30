#!/usr/bin/env python3
"""
train_score_model.py

Train and validate a CFB game-score model using only information that was
available before each game. This script READS the existing CFB pipeline and
WRITES ONLY under:

    docs/win/football/cfb/data/score_model/

It does not modify projection.py, projection_week1.py, selections.py,
picks.py, markets.yaml, or any intake/history file.

Historical training data:
    2021-2025 regular season schedules
    2021-2025 final scores from PBP
    pregame sportsbook lines from the historical betting cache
    historical ESPN predictor values from the same cache
    historical team stats, always using a source week before the game
    historical travel/weather features
    prior-game scoring/form features built chronologically

Validation design:
    Model choice: forward seasons only
        train 2021 -> validate 2022
        train 2021-2022 -> validate 2023
        train 2021-2023 -> validate 2024

    Final validation:
        fit 2021-2024 -> validate 2025

For games with sportsbook spread/total, the trained model predicts the ERROR
in the sportsbook's implied margin/total rather than learning from scratch that
favorites tend to win. If a trained residual model does not beat the sportsbook
baseline in both forward validation and 2025 final validation, deployment keeps
the sportsbook baseline for that target.

For games without a sportsbook spread/total, separate direct margin/total
models are trained from non-market features as fallbacks.

Optional current-week prediction:
    --season 2026 --predict-week 1

Outputs (new files only):
    docs/win/football/cfb/data/score_model/training_games_2021_2025.csv
    docs/win/football/cfb/data/score_model/cv_model_results.csv
    docs/win/football/cfb/data/score_model/validation_2025_predictions.csv
    docs/win/football/cfb/data/score_model/validation_2025_metrics.csv
    docs/win/football/cfb/data/score_model/feature_columns.csv
    docs/win/football/cfb/data/score_model/score_model.joblib
    docs/win/football/cfb/data/score_model/score_model_manifest.json
    docs/win/football/cfb/data/score_model/week_{week}_CFB_trained_score_predictions.csv
        (only when --predict-week is supplied)

Important exclusions:
    Historical point-in-time FPI and injury snapshots do not exist in this
    repository for 2021-2025, so they are not used for training. Current-only
    FPI/injury values are therefore also excluded from prediction to preserve
    train/predict parity and prevent fake historical values.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


SCRIPT_VERSION = "cfb-trained-score-v1-2026-08-28"
HISTORICAL_SEASONS = [2021, 2022, 2023, 2024, 2025]
CV_VALIDATION_SEASONS = [2022, 2023, 2024]
FINAL_VALIDATION_SEASON = 2025
SEASON_TYPE = 2
RANDOM_STATE = 20260828

PROVIDER_BY_SEASON = {
    2021: "draftkings",
    2022: "draftkings",
    2023: "draftkings",
    2024: "espnbet",
    2025: "espnbet",
}

# Features that are identifiers/targets/audit fields, not model inputs.
NON_FEATURE_COLUMNS = {
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "kickoff_sort",
    "home_final",
    "away_final",
    "actual_margin",
    "actual_total",
    "margin_residual",
    "total_residual",
    "market_implied_home_score",
    "market_implied_away_score",
}

# Market-derived inputs. They are allowed in residual models but removed from
# direct fallback models so a missing market does not become an imputed fake line.
MARKET_FEATURE_COLUMNS = {
    "market_margin",
    "market_total",
    "home_moneyline_american",
    "away_moneyline_american",
    "home_ml_implied_probability",
    "away_ml_implied_probability",
    "home_spread_american",
    "away_spread_american",
    "over_american",
    "under_american",
    "market_spread_available",
    "market_total_available",
    "market_ml_available",
    "sportsbook",
}

CATEGORICAL_FEATURES = {
    "home_team",
    "away_team",
    "stadium",
    "roof",
    "surface",
    "home_timezone",
    "away_timezone",
    "game_timezone",
    "venue_timezone",
    "venue_country",
    "venue_resolution_status",
    "roof_type",
    "sportsbook",
}

TEAM_META_COLUMNS = {
    "season",
    "week",
    "team",
    "team_id",
}

# Raw historical/current travel fields that exist on both sides of the pipeline.
TRAVEL_NUMERIC_COLUMNS = [
    "venue_lat",
    "venue_lon",
    "away_home_lat",
    "away_home_lon",
    "away_miles_traveled",
    "away_time_zone_change_hours",
    "away_time_zones_crossed",
    "away_east_to_west",
    "away_west_to_east",
    "home_home_lat",
    "home_home_lon",
    "home_miles_traveled",
    "home_time_zone_change_hours",
    "home_time_zones_crossed",
    "home_east_to_west",
    "home_west_to_east",
    "international_flag",
]

WEATHER_NUMERIC_COLUMNS = [
    "temperature",
    "wind_speed",
    "wind_gust",
    "precip_probability",
    "rain_flag",
    "snow_flag",
    "humidity",
    "dome_flag",
    "retractable_roof_flag",
    "open_air_flag",
]

FORM_METRICS = [
    "points_for",
    "points_against",
    "margin",
    "total",
    "margin_vs_market",
    "total_vs_market",
]

MATCHUP_PAIRS = [
    ("off_epa_per_play", "def_epa_per_play", "epa"),
    ("off_success_rate", "def_success_rate", "success"),
    ("yards_per_play", "yards_per_play_allowed", "ypp"),
    ("points_per_drive", "points_per_drive_allowed", "ppd"),
    ("red_zone_td_rate", "red_zone_td_rate_allowed", "red_zone"),
    ("third_down_conversion_rate", "third_down_defense_rate", "third_down"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, validate, and optionally run the CFB score model."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Season for optional current-week prediction. Default: 2026.",
    )
    parser.add_argument(
        "--predict-week",
        type=int,
        default=None,
        help="If supplied, also create predictions for this current week.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a smaller model search for troubleshooting. Full search is default.",
    )
    return parser.parse_args()


def cfb_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "docs" / "win" / "football" / "cfb"
        if candidate.is_dir():
            return candidate
    # Normal installed location: .../cfb/scripts/01_merge/train_score_model.py
    for parent in here.parents:
        if parent.name == "cfb":
            return parent
    raise RuntimeError(f"Cannot resolve CFB root from {here}")


CFB_ROOT = cfb_root()
SCHEDULE_DIR = CFB_ROOT / "00_intake" / "schedule"
WEEKLY_SCHEDULE_DIR = SCHEDULE_DIR / "weekly"
TEAM_STATS_DIR = CFB_ROOT / "00_intake" / "team_stats"
PBP_DIR = CFB_ROOT / "00_intake" / "pbp"
HIST_FEATURE_DIR = CFB_ROOT / "data" / "historical_features"
HIST_BETTING_DIR = CFB_ROOT / "data" / "historical_betting"
HIST_CACHE_DIR = HIST_BETTING_DIR / "cache"
TRAVEL_DIR = CFB_ROOT / "data" / "travel"
WEATHER_DIR = CFB_ROOT / "data" / "weather"
ENRICHED_DIR = CFB_ROOT / "01_merge"
TEAM_MAP_PATH = CFB_ROOT / "config" / "mapping" / "team_map.csv"
OUT_DIR = CFB_ROOT / "data" / "score_model"


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
    if OUT_DIR not in path.parents:
        raise RuntimeError(f"Refusing to write outside {OUT_DIR}: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_json(data: dict[str, Any], path: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_DIR not in path.parents:
        raise RuntimeError(f"Refusing to write outside {OUT_DIR}: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


class TeamResolver:
    def __init__(self, path: Path):
        df = read_csv(path, ["canonical_team"])
        candidate_columns = [
            c for c in [
                "canonical_team",
                "alias",
                "location",
                "shortDisplayName",
                "team_slug",
            ]
            if c in df.columns
        ]
        collected: dict[str, set[str]] = defaultdict(set)
        for _, row in df.iterrows():
            canonical = clean(row.get("canonical_team"))
            if not canonical:
                continue
            for column in candidate_columns:
                key = norm(row.get(column))
                if key:
                    collected[key].add(canonical)
        self.mapping = {
            key: next(iter(values))
            for key, values in collected.items()
            if len(values) == 1
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


def parse_kickoff_date_time(game_date: Any, game_time: Any) -> pd.Timestamp:
    date_text = clean(game_date)
    time_text = clean(game_time)
    combined = f"{date_text} {time_text}".strip()
    ts = pd.to_datetime(combined, errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(date_text, errors="coerce")
    return ts


def load_final_scores(season: int) -> pd.DataFrame:
    path = PBP_DIR / f"{season}_pbp.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    needed = ["game_id", "sequenceNumber", "end.homeScore", "end.awayScore"]
    df = pd.read_parquet(path, columns=needed)
    df["game_id"] = df["game_id"].map(gid)
    df["sequenceNumber"] = pd.to_numeric(df["sequenceNumber"], errors="coerce")
    df["home_final"] = pd.to_numeric(df["end.homeScore"], errors="coerce")
    df["away_final"] = pd.to_numeric(df["end.awayScore"], errors="coerce")
    df = df.dropna(subset=["game_id", "home_final", "away_final"])
    df = df.sort_values(["game_id", "sequenceNumber"], kind="stable")
    return (
        df.groupby("game_id", as_index=False)
        .tail(1)[["game_id", "home_final", "away_final"]]
        .drop_duplicates("game_id", keep="last")
    )


def load_schedule(season: int) -> pd.DataFrame:
    path = SCHEDULE_DIR / f"{season}_schedule.csv"
    df = read_csv(
        path,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "game_date",
            "game_time",
            "away_team",
            "home_team",
        ],
    ).copy()
    df["game_id"] = df["game_id"].map(gid)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["season_type"] = pd.to_numeric(df["season_type"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
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
    df = read_csv(path, ["game_id", "odds_status", "provider"]).copy()
    df["game_id"] = df["game_id"].map(gid)
    # These cache files were already cleaned to exact provider names by the
    # historical betting pipeline. Keep all rows so missing markets remain
    # available to the direct fallback model.
    return df.drop_duplicates("game_id", keep="last")


def load_historical_travel_weather(season: int) -> pd.DataFrame:
    path = HIST_FEATURE_DIR / f"{season}_travel_weather.csv"
    df = read_csv(path, ["game_id"]).copy()
    df["game_id"] = df["game_id"].map(gid)
    return df.drop_duplicates("game_id", keep="last")


def load_all_team_stats(seasons: list[int], resolver: TeamResolver) -> tuple[dict[int, pd.DataFrame], list[str]]:
    frames: dict[int, pd.DataFrame] = {}
    metric_union: set[str] = set()
    for season in seasons:
        path = TEAM_STATS_DIR / f"{season}_team_stats.csv"
        if not path.is_file():
            continue
        df = read_csv(path, ["season", "week", "team"]).copy()
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
        df["team_key"] = df["team"].map(resolver.resolve)
        for col in df.columns:
            if col in TEAM_META_COLUMNS or col == "team_key":
                continue
            converted = pd.to_numeric(df[col], errors="coerce")
            nonblank = df[col].map(clean).ne("").sum()
            if nonblank and converted.notna().sum() / nonblank >= 0.80:
                df[col] = converted
                metric_union.add(col)
        frames[season] = df
    if not frames:
        raise RuntimeError("No historical team_stats files found")
    return frames, sorted(metric_union)


def latest_team_stats(
    stats: dict[int, pd.DataFrame],
    team: str,
    season: int,
    week: int,
) -> tuple[pd.Series | None, int | None, int | None]:
    # First choice: current-season statistics from a completed prior week.
    if week > 1 and season in stats:
        frame = stats[season]
        rows = frame[
            frame["team_key"].eq(team)
            & frame["week"].lt(week)
        ]
        if not rows.empty:
            row = rows.sort_values("week").iloc[-1]
            return row, season, int(row["week"])

    # Week 1 or a team with no completed current-season stats: previous season.
    prior = season - 1
    if prior in stats:
        frame = stats[prior]
        rows = frame[frame["team_key"].eq(team)]
        if not rows.empty:
            row = rows.sort_values("week").iloc[-1]
            return row, prior, int(row["week"])

    return None, None, None


def add_team_stat_features(
    features: dict[str, Any],
    stats: dict[int, pd.DataFrame],
    metrics: list[str],
    home_team: str,
    away_team: str,
    season: int,
    week: int,
) -> None:
    home_row, home_source_season, home_source_week = latest_team_stats(
        stats, home_team, season, week
    )
    away_row, away_source_season, away_source_week = latest_team_stats(
        stats, away_team, season, week
    )

    features["home_team_stats_missing"] = 1.0 if home_row is None else 0.0
    features["away_team_stats_missing"] = 1.0 if away_row is None else 0.0
    features["home_team_stats_source_season_gap"] = (
        None if home_source_season is None else float(season - home_source_season)
    )
    features["away_team_stats_source_season_gap"] = (
        None if away_source_season is None else float(season - away_source_season)
    )
    features["home_team_stats_source_week"] = (
        None if home_source_week is None else float(home_source_week)
    )
    features["away_team_stats_source_week"] = (
        None if away_source_week is None else float(away_source_week)
    )

    for metric in metrics:
        hv = fnum(home_row.get(metric)) if home_row is not None else None
        av = fnum(away_row.get(metric)) if away_row is not None else None
        features[f"home_ts_{metric}"] = hv
        features[f"away_ts_{metric}"] = av
        features[f"diff_ts_{metric}"] = (
            None if hv is None or av is None else hv - av
        )

    for offense_metric, defense_metric, label in MATCHUP_PAIRS:
        h_off = fnum(home_row.get(offense_metric)) if home_row is not None else None
        a_off = fnum(away_row.get(offense_metric)) if away_row is not None else None
        h_def = fnum(home_row.get(defense_metric)) if home_row is not None else None
        a_def = fnum(away_row.get(defense_metric)) if away_row is not None else None
        features[f"home_matchup_{label}"] = (
            None if h_off is None or a_def is None else h_off - a_def
        )
        features[f"away_matchup_{label}"] = (
            None if a_off is None or h_def is None else a_off - h_def
        )
        if features[f"home_matchup_{label}"] is not None and features[f"away_matchup_{label}"] is not None:
            features[f"matchup_diff_{label}"] = (
                features[f"home_matchup_{label}"] - features[f"away_matchup_{label}"]
            )
        else:
            features[f"matchup_diff_{label}"] = None


@dataclass
class TeamFormState:
    games: int = 0
    last_date: pd.Timestamp | None = None
    points_for: list[float] = None
    points_against: list[float] = None
    margins: list[float] = None
    totals: list[float] = None
    margin_vs_market: list[float] = None
    total_vs_market: list[float] = None

    def __post_init__(self) -> None:
        if self.points_for is None:
            self.points_for = []
        if self.points_against is None:
            self.points_against = []
        if self.margins is None:
            self.margins = []
        if self.totals is None:
            self.totals = []
        if self.margin_vs_market is None:
            self.margin_vs_market = []
        if self.total_vs_market is None:
            self.total_vs_market = []


def avg(values: list[float], last_n: int | None = None) -> float | None:
    if not values:
        return None
    data = values if last_n is None else values[-last_n:]
    return float(np.mean(data)) if data else None


def form_snapshot(state: TeamFormState, kickoff: pd.Timestamp) -> dict[str, Any]:
    result: dict[str, Any] = {
        "games": float(state.games),
        "days_rest": None,
    }
    if state.last_date is not None and not pd.isna(kickoff) and not pd.isna(state.last_date):
        result["days_rest"] = float((kickoff.normalize() - state.last_date.normalize()).days)

    mapping = {
        "points_for": state.points_for,
        "points_against": state.points_against,
        "margin": state.margins,
        "total": state.totals,
        "margin_vs_market": state.margin_vs_market,
        "total_vs_market": state.total_vs_market,
    }
    for metric, values in mapping.items():
        result[f"season_{metric}"] = avg(values)
        result[f"last3_{metric}"] = avg(values, 3)
        result[f"last5_{metric}"] = avg(values, 5)
    return result


def add_form_pair(features: dict[str, Any], home: dict[str, Any], away: dict[str, Any]) -> None:
    keys = sorted(set(home) | set(away))
    for key in keys:
        hv = home.get(key)
        av = away.get(key)
        features[f"home_form_{key}"] = hv
        features[f"away_form_{key}"] = av
        if isinstance(hv, (int, float, np.number)) and isinstance(av, (int, float, np.number)):
            if math.isfinite(float(hv)) and math.isfinite(float(av)):
                features[f"diff_form_{key}"] = float(hv) - float(av)
            else:
                features[f"diff_form_{key}"] = None
        else:
            features[f"diff_form_{key}"] = None


def update_form_state(
    state: TeamFormState,
    kickoff: pd.Timestamp,
    points_for: float,
    points_against: float,
    team_market_margin: float | None,
    market_total: float | None,
) -> None:
    margin = points_for - points_against
    total = points_for + points_against
    state.games += 1
    state.last_date = kickoff
    state.points_for.append(float(points_for))
    state.points_against.append(float(points_against))
    state.margins.append(float(margin))
    state.totals.append(float(total))
    if team_market_margin is not None:
        state.margin_vs_market.append(float(margin - team_market_margin))
    if market_total is not None:
        state.total_vs_market.append(float(total - market_total))


def base_market_features(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    home_spread = fnum(row.get("home_spread"))
    total = fnum(row.get("total"))
    home_ml = fnum(row.get("home_moneyline_american"))
    away_ml = fnum(row.get("away_moneyline_american"))
    return {
        "market_margin": None if home_spread is None else -home_spread,
        "market_total": total,
        "home_moneyline_american": home_ml,
        "away_moneyline_american": away_ml,
        "home_ml_implied_probability": american_implied(home_ml),
        "away_ml_implied_probability": american_implied(away_ml),
        "home_spread_american": fnum(row.get("home_spread_american")),
        "away_spread_american": fnum(row.get("away_spread_american")),
        "over_american": fnum(row.get("over_american")),
        "under_american": fnum(row.get("under_american")),
        "market_spread_available": 1.0 if home_spread is not None else 0.0,
        "market_total_available": 1.0 if total is not None else 0.0,
        "market_ml_available": 1.0 if home_ml is not None and away_ml is not None else 0.0,
        "sportsbook": clean(row.get("provider") or row.get("bookmaker")),
    }


def predictor_features(row: pd.Series | dict[str, Any], historical: bool) -> dict[str, Any]:
    if historical:
        home_prob = fnum(row.get("espn_home_game_projection"))
        away_prob = fnum(row.get("espn_away_game_projection"))
    else:
        home_prob = fnum(row.get("espn_home_prob"))
        away_prob = fnum(row.get("espn_away_prob"))
    return {
        "espn_home_ptdiff": fnum(row.get("espn_home_ptdiff")),
        "espn_away_ptdiff": fnum(row.get("espn_away_ptdiff")),
        "espn_home_probability": home_prob,
        "espn_away_probability": away_prob,
        "espn_predictor_available": 1.0 if fnum(row.get("espn_home_ptdiff")) is not None else 0.0,
    }


def shared_context_features(
    schedule_row: pd.Series | dict[str, Any],
    travel_row: pd.Series | dict[str, Any] | None,
    weather_row: pd.Series | dict[str, Any] | None,
    neutral_override: Any = None,
) -> dict[str, Any]:
    kickoff = parse_kickoff_date_time(schedule_row.get("game_date"), schedule_row.get("game_time"))
    neutral_value = neutral_override if neutral_override is not None else schedule_row.get("neutral_site")
    features: dict[str, Any] = {
        "week_number": fnum(schedule_row.get("week")),
        "month": None if pd.isna(kickoff) else float(kickoff.month),
        "kickoff_hour": None if pd.isna(kickoff) else float(kickoff.hour),
        "neutral_site": 1.0 if clean(neutral_value).casefold() in {"1", "true", "yes", "y"} else 0.0,
        "stadium": clean(schedule_row.get("stadium")),
        "roof": clean(schedule_row.get("roof")),
        "surface": clean(schedule_row.get("surface")),
        "home_timezone": clean(schedule_row.get("home_timezone")),
        "away_timezone": clean(schedule_row.get("away_timezone")),
        "game_timezone": clean(schedule_row.get("game_timezone")),
    }

    tr = travel_row if travel_row is not None else {}
    for col in TRAVEL_NUMERIC_COLUMNS:
        features[col] = fnum(tr.get(col))
    for col in ["venue_timezone", "venue_country", "venue_resolution_status"]:
        features[col] = clean(tr.get(col))
    features["travel_available"] = 1.0 if fnum(tr.get("away_miles_traveled")) is not None else 0.0
    away_miles = fnum(tr.get("away_miles_traveled"))
    home_miles = fnum(tr.get("home_miles_traveled"))
    features["travel_net_miles_1000"] = (
        None if away_miles is None or home_miles is None else (away_miles - home_miles) / 1000.0
    )
    away_tz = fnum(tr.get("away_time_zones_crossed"))
    home_tz = fnum(tr.get("home_time_zones_crossed"))
    features["travel_net_time_zones"] = (
        None if away_tz is None or home_tz is None else away_tz - home_tz
    )

    wr = weather_row if weather_row is not None else {}
    for col in WEATHER_NUMERIC_COLUMNS:
        features[f"weather_{col}"] = fnum(wr.get(col))
    features["roof_type"] = clean(wr.get("roof_type")) or features["roof"]
    features["weather_available"] = 1.0 if fnum(wr.get("temperature")) is not None else 0.0
    return features


def build_historical_games(resolver: TeamResolver) -> pd.DataFrame:
    merged_frames: list[pd.DataFrame] = []
    for season in HISTORICAL_SEASONS:
        schedule = load_schedule(season)
        finals = load_final_scores(season)
        market = load_historical_market(season)
        travel = load_historical_travel_weather(season)
        frame = schedule.merge(finals, on="game_id", how="inner", validate="one_to_one")
        frame = frame.merge(market, on="game_id", how="left", validate="one_to_one", suffixes=("", "_market"))
        frame = frame.merge(travel, on="game_id", how="left", validate="one_to_one", suffixes=("", "_tw"))
        frame["season"] = season
        merged_frames.append(frame)
    games = pd.concat(merged_frames, ignore_index=True)
    games["home_team"] = games["home_team"].map(resolver.resolve)
    games["away_team"] = games["away_team"].map(resolver.resolve)
    games["kickoff_sort"] = [
        parse_kickoff_date_time(d, t)
        for d, t in zip(games["game_date"], games["game_time"])
    ]
    games = games.sort_values(["season", "kickoff_sort", "week", "game_id"], kind="stable").reset_index(drop=True)
    return games


def build_training_frame(
    raw_games: pd.DataFrame,
    stats: dict[int, pd.DataFrame],
    stat_metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    states: dict[tuple[int, str], TeamFormState] = {}

    for _, game in raw_games.iterrows():
        season = int(game["season"])
        week = int(game["week"])
        home_team = clean(game["home_team"])
        away_team = clean(game["away_team"])
        kickoff = game["kickoff_sort"]

        home_state = states.setdefault((season, home_team), TeamFormState())
        away_state = states.setdefault((season, away_team), TeamFormState())

        # schedule/travel/weather columns were merged with suffixes when needed.
        travel_like = game
        weather_like = game
        neutral_override = game.get("neutral_site_flag") if clean(game.get("neutral_site_flag")) else game.get("neutral_site")
        features = shared_context_features(game, travel_like, weather_like, neutral_override=neutral_override)
        features["home_team"] = home_team
        features["away_team"] = away_team

        market = base_market_features(game)
        features.update(market)
        features.update(predictor_features(game, historical=True))
        add_team_stat_features(features, stats, stat_metrics, home_team, away_team, season, week)
        add_form_pair(features, form_snapshot(home_state, kickoff), form_snapshot(away_state, kickoff))

        home_final = float(game["home_final"])
        away_final = float(game["away_final"])
        actual_margin = home_final - away_final
        actual_total = home_final + away_final
        market_margin = features["market_margin"]
        market_total = features["market_total"]

        output = {
            "season": season,
            "season_type": SEASON_TYPE,
            "week": week,
            "game_id": gid(game["game_id"]),
            "game_date": clean(game.get("game_date")),
            "game_time": clean(game.get("game_time")),
            "kickoff_sort": kickoff,
            **features,
            "home_final": home_final,
            "away_final": away_final,
            "actual_margin": actual_margin,
            "actual_total": actual_total,
            "margin_residual": None if market_margin is None else actual_margin - market_margin,
            "total_residual": None if market_total is None else actual_total - market_total,
            "market_implied_home_score": (
                None if market_margin is None or market_total is None else (market_total + market_margin) / 2.0
            ),
            "market_implied_away_score": (
                None if market_margin is None or market_total is None else (market_total - market_margin) / 2.0
            ),
        }
        rows.append(output)

        update_form_state(
            home_state,
            kickoff,
            home_final,
            away_final,
            market_margin,
            market_total,
        )
        update_form_state(
            away_state,
            kickoff,
            away_final,
            home_final,
            None if market_margin is None else -market_margin,
            market_total,
        )

    return pd.DataFrame(rows)


def clean_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in CATEGORICAL_FEATURES:
            out[col] = out[col].map(clean)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def choose_feature_columns(training: pd.DataFrame, include_market: bool) -> tuple[list[str], list[str], list[str]]:
    candidates = [c for c in training.columns if c not in NON_FEATURE_COLUMNS]
    if not include_market:
        candidates = [c for c in candidates if c not in MARKET_FEATURE_COLUMNS]

    fit = training[training["season"].le(2024)]
    kept: list[str] = []
    categorical: list[str] = []
    numeric: list[str] = []

    for col in candidates:
        if col in CATEGORICAL_FEATURES:
            nonblank = fit[col].map(clean).ne("").sum()
            if nonblank >= 25 and fit[col].map(clean).nunique() > 1:
                kept.append(col)
                categorical.append(col)
            continue

        values = pd.to_numeric(fit[col], errors="coerce")
        if values.notna().sum() < 25:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        kept.append(col)
        numeric.append(col)

    return kept, numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=3,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_estimators(fast: bool) -> dict[str, Any]:
    models: dict[str, Any] = {
        "ridge_1": Ridge(alpha=1.0),
        "ridge_10": Ridge(alpha=10.0),
        "ridge_100": Ridge(alpha=100.0),
        "elastic_001": ElasticNet(alpha=0.001, l1_ratio=0.20, max_iter=30000, random_state=RANDOM_STATE),
        "elastic_01": ElasticNet(alpha=0.01, l1_ratio=0.20, max_iter=30000, random_state=RANDOM_STATE),
        "extra_leaf5": ExtraTreesRegressor(
            n_estimators=300 if fast else 900,
            min_samples_leaf=5,
            max_features=0.75,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_leaf12": ExtraTreesRegressor(
            n_estimators=250 if fast else 700,
            min_samples_leaf=12,
            max_features=0.85,
            n_jobs=-1,
            random_state=RANDOM_STATE + 1,
        ),
        "rf_leaf8": RandomForestRegressor(
            n_estimators=250 if fast else 700,
            min_samples_leaf=8,
            max_features=0.70,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "histgb": HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=250 if fast else 700,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=5.0,
            random_state=RANDOM_STATE,
        ),
        "gbr_huber": GradientBoostingRegressor(
            loss="huber",
            learning_rate=0.03,
            n_estimators=250 if fast else 650,
            max_depth=2,
            min_samples_leaf=12,
            random_state=RANDOM_STATE,
        ),
    }

    if not fast and XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=3,
            min_child_weight=8,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.10,
            reg_lambda=10.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    if not fast and LGBMRegressor is not None:
        models["lightgbm"] = LGBMRegressor(
            objective="regression_l1",
            n_estimators=1000,
            learning_rate=0.02,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.10,
            reg_lambda=10.0,
            verbosity=-1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    return models


def pipeline_for(estimator: Any, numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(numeric, categorical)),
            ("model", clone(estimator)),
        ]
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def best_shrink(y_actual: np.ndarray, baseline: np.ndarray, raw_residual_pred: np.ndarray) -> tuple[float, float]:
    best_factor = 0.0
    best_mae = float(mean_absolute_error(y_actual, baseline))
    # 0 means pure market baseline; >1 permits a residual correction stronger
    # than the model's raw estimate if forward validation supports it.
    for factor in np.arange(0.0, 1.5001, 0.025):
        pred = baseline + factor * raw_residual_pred
        score = float(mean_absolute_error(y_actual, pred))
        if score < best_mae - 1e-12:
            best_mae = score
            best_factor = float(round(factor, 3))
    return best_factor, best_mae


@dataclass
class SelectionResult:
    target_name: str
    mode: str
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    component_names: list[str]
    component_weights: list[float]
    shrink: float
    cv_mae: float
    cv_rmse: float
    baseline_cv_mae: float | None
    baseline_cv_rmse: float | None
    cv_rows: int
    cv_table: pd.DataFrame


def fit_predict_one(
    estimator: Any,
    numeric: list[str],
    categorical: list[str],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
) -> tuple[Pipeline, np.ndarray]:
    model = pipeline_for(estimator, numeric, categorical)
    model.fit(x_train, y_train)
    pred = np.asarray(model.predict(x_val), dtype=float)
    return model, pred


def select_model(
    frame: pd.DataFrame,
    target_name: str,
    target_col: str,
    include_market: bool,
    baseline_col: str | None,
    fast: bool,
) -> SelectionResult:
    feature_cols, numeric, categorical = choose_feature_columns(frame, include_market=include_market)
    models = candidate_estimators(fast)
    result_rows: list[dict[str, Any]] = []
    oof_by_model: dict[str, dict[int, float]] = {name: {} for name in models}
    y_actual_by_index: dict[int, float] = {}
    baseline_by_index: dict[int, float] = {}

    for validation_season in CV_VALIDATION_SEASONS:
        train_mask = frame["season"].lt(validation_season) & frame[target_col].notna()
        val_mask = frame["season"].eq(validation_season) & frame[target_col].notna()
        if baseline_col is not None:
            train_mask &= frame[baseline_col].notna()
            val_mask &= frame[baseline_col].notna()

        train = frame.loc[train_mask].copy()
        val = frame.loc[val_mask].copy()
        if train.empty or val.empty:
            raise RuntimeError(
                f"{target_name}: empty forward fold train={validation_season-1} validate={validation_season}"
            )

        x_train = clean_feature_frame(train[feature_cols])
        x_val = clean_feature_frame(val[feature_cols])
        y_train = pd.to_numeric(train[target_col], errors="raise")
        y_val_target = pd.to_numeric(val[target_col], errors="raise").to_numpy(float)

        if baseline_col is not None:
            baseline_val = pd.to_numeric(val[baseline_col], errors="raise").to_numpy(float)
            y_actual = y_val_target + baseline_val
        else:
            baseline_val = None
            y_actual = y_val_target

        for idx, actual in zip(val.index, y_actual):
            y_actual_by_index[int(idx)] = float(actual)
        if baseline_val is not None:
            for idx, base in zip(val.index, baseline_val):
                baseline_by_index[int(idx)] = float(base)

        for name, estimator in models.items():
            try:
                _, raw_pred = fit_predict_one(
                    estimator, numeric, categorical, x_train, y_train, x_val
                )
            except Exception as exc:
                result_rows.append(
                    {
                        "target": target_name,
                        "model": name,
                        "validation_season": validation_season,
                        "status": f"ERROR:{type(exc).__name__}",
                        "rows": len(val),
                        "mae": np.nan,
                        "rmse": np.nan,
                    }
                )
                continue

            if baseline_col is not None:
                pred_actual = baseline_val + raw_pred
            else:
                pred_actual = raw_pred

            for idx, pred in zip(val.index, raw_pred):
                oof_by_model[name][int(idx)] = float(pred)

            result_rows.append(
                {
                    "target": target_name,
                    "model": name,
                    "validation_season": validation_season,
                    "status": "OK",
                    "rows": len(val),
                    "mae": float(mean_absolute_error(y_actual, pred_actual)),
                    "rmse": rmse(y_actual, pred_actual),
                }
            )

    common_indices = sorted(y_actual_by_index)
    y_actual = np.array([y_actual_by_index[i] for i in common_indices], dtype=float)

    baseline_cv_mae: float | None = None
    baseline_cv_rmse: float | None = None
    baseline = None
    if baseline_col is not None:
        baseline = np.array([baseline_by_index[i] for i in common_indices], dtype=float)
        baseline_cv_mae = float(mean_absolute_error(y_actual, baseline))
        baseline_cv_rmse = rmse(y_actual, baseline)
        result_rows.append(
            {
                "target": target_name,
                "model": "MARKET_BASELINE",
                "validation_season": "ALL_2022_2024",
                "status": "OK",
                "rows": len(common_indices),
                "mae": baseline_cv_mae,
                "rmse": baseline_cv_rmse,
                "shrink": 0.0,
            }
        )

    valid_model_scores: list[tuple[str, float, float, float]] = []
    raw_oof_arrays: dict[str, np.ndarray] = {}

    for name in models:
        pred_map = oof_by_model[name]
        if any(i not in pred_map for i in common_indices):
            continue
        raw = np.array([pred_map[i] for i in common_indices], dtype=float)
        raw_oof_arrays[name] = raw
        if baseline_col is not None:
            shrink, score_mae = best_shrink(y_actual, baseline, raw)
            pred = baseline + shrink * raw
        else:
            shrink = 1.0
            pred = raw
            score_mae = float(mean_absolute_error(y_actual, pred))
        score_rmse = rmse(y_actual, pred)
        valid_model_scores.append((name, score_mae, score_rmse, shrink))
        result_rows.append(
            {
                "target": target_name,
                "model": name,
                "validation_season": "ALL_2022_2024",
                "status": "OK",
                "rows": len(common_indices),
                "mae": score_mae,
                "rmse": score_rmse,
                "shrink": shrink,
            }
        )

    if not valid_model_scores:
        raise RuntimeError(f"{target_name}: every candidate model failed")

    valid_model_scores.sort(key=lambda item: (item[1], item[2], item[0]))
    best_single = valid_model_scores[0]

    # Build an out-of-fold ensemble of the three best candidates. The weights
    # are inverse-MAE and therefore determined by forward validation, not hand-set.
    top = valid_model_scores[: min(3, len(valid_model_scores))]
    inv = np.array([1.0 / max(item[1], 1e-9) for item in top], dtype=float)
    weights = inv / inv.sum()
    ensemble_raw = np.zeros(len(common_indices), dtype=float)
    for weight, item in zip(weights, top):
        ensemble_raw += weight * raw_oof_arrays[item[0]]

    if baseline_col is not None:
        ensemble_shrink, ensemble_mae = best_shrink(y_actual, baseline, ensemble_raw)
        ensemble_pred = baseline + ensemble_shrink * ensemble_raw
    else:
        ensemble_shrink = 1.0
        ensemble_pred = ensemble_raw
        ensemble_mae = float(mean_absolute_error(y_actual, ensemble_pred))
    ensemble_rmse = rmse(y_actual, ensemble_pred)

    result_rows.append(
        {
            "target": target_name,
            "model": "ENSEMBLE_TOP3",
            "validation_season": "ALL_2022_2024",
            "status": "OK",
            "rows": len(common_indices),
            "mae": ensemble_mae,
            "rmse": ensemble_rmse,
            "shrink": ensemble_shrink,
            "components": "|".join(item[0] for item in top),
            "weights": "|".join(f"{w:.8f}" for w in weights),
        }
    )

    if ensemble_mae < best_single[1] - 1e-12:
        component_names = [item[0] for item in top]
        component_weights = [float(w) for w in weights]
        chosen_mae = ensemble_mae
        chosen_rmse = ensemble_rmse
        chosen_shrink = ensemble_shrink
        mode = "ensemble"
    else:
        component_names = [best_single[0]]
        component_weights = [1.0]
        chosen_mae = best_single[1]
        chosen_rmse = best_single[2]
        chosen_shrink = best_single[3]
        mode = "single"

    # Residual model must beat simply using the sportsbook line in forward CV.
    if baseline_col is not None and chosen_mae >= baseline_cv_mae - 1e-12:
        mode = "baseline"
        component_names = []
        component_weights = []
        chosen_shrink = 0.0
        chosen_mae = baseline_cv_mae
        chosen_rmse = baseline_cv_rmse

    return SelectionResult(
        target_name=target_name,
        mode=mode,
        feature_columns=feature_cols,
        numeric_columns=numeric,
        categorical_columns=categorical,
        component_names=component_names,
        component_weights=component_weights,
        shrink=chosen_shrink,
        cv_mae=float(chosen_mae),
        cv_rmse=float(chosen_rmse),
        baseline_cv_mae=baseline_cv_mae,
        baseline_cv_rmse=baseline_cv_rmse,
        cv_rows=len(common_indices),
        cv_table=pd.DataFrame(result_rows),
    )


def fit_selected_components(
    frame: pd.DataFrame,
    selection: SelectionResult,
    target_col: str,
    seasons: list[int],
    fast: bool,
    require_baseline_col: str | None = None,
) -> list[Pipeline]:
    if selection.mode == "baseline":
        return []
    mask = frame["season"].isin(seasons) & frame[target_col].notna()
    if require_baseline_col is not None:
        mask &= frame[require_baseline_col].notna()
    train = frame.loc[mask]
    x = clean_feature_frame(train[selection.feature_columns])
    y = pd.to_numeric(train[target_col], errors="raise")
    estimators = candidate_estimators(fast)
    fitted: list[Pipeline] = []
    for name in selection.component_names:
        if name not in estimators:
            raise RuntimeError(f"Selected model {name} is not available at final fit")
        pipe = pipeline_for(estimators[name], selection.numeric_columns, selection.categorical_columns)
        pipe.fit(x, y)
        fitted.append(pipe)
    return fitted


def component_predict(
    models: list[Pipeline],
    weights: list[float],
    x: pd.DataFrame,
) -> np.ndarray:
    if not models:
        raise RuntimeError("No fitted models supplied")
    pred = np.zeros(len(x), dtype=float)
    for weight, model in zip(weights, models):
        pred += float(weight) * np.asarray(model.predict(x), dtype=float)
    return pred


def predict_target(
    frame: pd.DataFrame,
    selection: SelectionResult,
    models: list[Pipeline],
    baseline_col: str | None,
) -> np.ndarray:
    if selection.mode == "baseline":
        if baseline_col is None:
            raise RuntimeError("Direct target cannot use baseline mode")
        return pd.to_numeric(frame[baseline_col], errors="coerce").to_numpy(float)
    x = clean_feature_frame(frame[selection.feature_columns])
    raw = component_predict(models, selection.component_weights, x)
    if baseline_col is None:
        return raw
    baseline = pd.to_numeric(frame[baseline_col], errors="coerce").to_numpy(float)
    return baseline + selection.shrink * raw


def evaluate_2025(
    frame: pd.DataFrame,
    margin_sel: SelectionResult,
    total_sel: SelectionResult,
    margin_models: list[Pipeline],
    total_models: list[Pipeline],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    val = frame[
        frame["season"].eq(FINAL_VALIDATION_SEASON)
        & frame["actual_margin"].notna()
        & frame["actual_total"].notna()
        & frame["market_margin"].notna()
        & frame["market_total"].notna()
    ].copy()
    if val.empty:
        raise RuntimeError("No 2025 rows with final score + market spread + market total")

    model_margin = predict_target(val, margin_sel, margin_models, "market_margin")
    model_total = predict_target(val, total_sel, total_models, "market_total")
    # A football score pair cannot have total < absolute margin.
    model_total = np.maximum(model_total, np.abs(model_margin))

    val["model_predicted_margin"] = model_margin
    val["model_predicted_total"] = model_total
    val["model_predicted_home_score"] = (model_total + model_margin) / 2.0
    val["model_predicted_away_score"] = (model_total - model_margin) / 2.0

    val["market_predicted_home_score"] = (val["market_total"] + val["market_margin"]) / 2.0
    val["market_predicted_away_score"] = (val["market_total"] - val["market_margin"]) / 2.0

    actual_margin = val["actual_margin"].to_numpy(float)
    actual_total = val["actual_total"].to_numpy(float)
    home_actual = val["home_final"].to_numpy(float)
    away_actual = val["away_final"].to_numpy(float)
    market_margin = val["market_margin"].to_numpy(float)
    market_total = val["market_total"].to_numpy(float)
    market_home = val["market_predicted_home_score"].to_numpy(float)
    market_away = val["market_predicted_away_score"].to_numpy(float)

    def mrow(name: str, actual: np.ndarray, model: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
        model_mae = float(mean_absolute_error(actual, model))
        base_mae = float(mean_absolute_error(actual, baseline))
        return {
            "metric": name,
            "rows": len(actual),
            "model_mae": model_mae,
            "market_mae": base_mae,
            "mae_improvement_points": base_mae - model_mae,
            "mae_improvement_pct": (base_mae - model_mae) / base_mae if base_mae else np.nan,
            "model_rmse": rmse(actual, model),
            "market_rmse": rmse(actual, baseline),
        }

    metrics = [
        mrow("margin", actual_margin, model_margin, market_margin),
        mrow("total", actual_total, model_total, market_total),
        mrow("home_score", home_actual, val["model_predicted_home_score"].to_numpy(float), market_home),
        mrow("away_score", away_actual, val["model_predicted_away_score"].to_numpy(float), market_away),
    ]

    model_team_abs = np.concatenate(
        [
            np.abs(home_actual - val["model_predicted_home_score"].to_numpy(float)),
            np.abs(away_actual - val["model_predicted_away_score"].to_numpy(float)),
        ]
    )
    market_team_abs = np.concatenate(
        [
            np.abs(home_actual - market_home),
            np.abs(away_actual - market_away),
        ]
    )
    metrics.append(
        {
            "metric": "average_team_score",
            "rows": len(model_team_abs),
            "model_mae": float(model_team_abs.mean()),
            "market_mae": float(market_team_abs.mean()),
            "mae_improvement_points": float(market_team_abs.mean() - model_team_abs.mean()),
            "mae_improvement_pct": float((market_team_abs.mean() - model_team_abs.mean()) / market_team_abs.mean()),
            "model_rmse": float(math.sqrt(np.mean(model_team_abs**2))),
            "market_rmse": float(math.sqrt(np.mean(market_team_abs**2))),
        }
    )

    actual_home_win = actual_margin > 0
    model_home_win = model_margin > 0
    market_home_win = market_margin > 0
    metrics.append(
        {
            "metric": "winner_accuracy",
            "rows": len(actual_margin),
            "model_mae": float(np.mean(model_home_win == actual_home_win)),
            "market_mae": float(np.mean(market_home_win == actual_home_win)),
            "mae_improvement_points": float(np.mean(model_home_win == actual_home_win) - np.mean(market_home_win == actual_home_win)),
            "mae_improvement_pct": np.nan,
            "model_rmse": np.nan,
            "market_rmse": np.nan,
        }
    )

    metrics_df = pd.DataFrame(metrics)
    margin_row = metrics_df[metrics_df["metric"].eq("margin")].iloc[0]
    total_row = metrics_df[metrics_df["metric"].eq("total")].iloc[0]
    score_row = metrics_df[metrics_df["metric"].eq("average_team_score")].iloc[0]
    gates = {
        "margin_beats_market_2025": bool(margin_row["model_mae"] < margin_row["market_mae"]),
        "total_beats_market_2025": bool(total_row["model_mae"] < total_row["market_mae"]),
        "average_team_score_beats_market_2025": bool(score_row["model_mae"] < score_row["market_mae"]),
    }

    output_cols = [
        "season",
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
        "market_margin",
        "market_total",
        "market_predicted_home_score",
        "market_predicted_away_score",
        "model_predicted_margin",
        "model_predicted_total",
        "model_predicted_home_score",
        "model_predicted_away_score",
    ]
    return val[output_cols].copy(), metrics_df, gates


def build_current_form_states(season: int, target_week: int, resolver: TeamResolver) -> dict[str, TeamFormState]:
    states: dict[str, TeamFormState] = {}
    schedule_path = SCHEDULE_DIR / f"{season}_schedule.csv"
    pbp_path = PBP_DIR / f"{season}_pbp.parquet"
    if not schedule_path.is_file() or not pbp_path.is_file() or target_week <= 1:
        return states

    schedule = load_schedule(season)
    schedule = schedule[schedule["week"].lt(target_week)].copy()
    if schedule.empty:
        return states
    finals = load_final_scores(season)
    schedule = schedule.merge(finals, on="game_id", how="inner", validate="one_to_one")

    # Pull archived weekly market lines for completed prior weeks when present.
    market_frames = []
    for week in sorted(schedule["week"].unique()):
        path = WEEKLY_SCHEDULE_DIR / f"week_{int(week)}_CFB_weekly_schedule.csv"
        if not path.is_file():
            continue
        wk = read_csv(path, ["game_id"]).copy()
        wk["game_id"] = wk["game_id"].map(gid)
        keep = [c for c in ["game_id", "home_spread", "total"] if c in wk.columns]
        market_frames.append(wk[keep].drop_duplicates("game_id", keep="last"))
    if market_frames:
        markets = pd.concat(market_frames, ignore_index=True).drop_duplicates("game_id", keep="last")
        schedule = schedule.merge(markets, on="game_id", how="left", suffixes=("", "_weekly"))
        for col in ["home_spread", "total"]:
            if f"{col}_weekly" in schedule.columns:
                schedule[col] = schedule[f"{col}_weekly"].combine_first(schedule.get(col))

    schedule["kickoff_sort"] = [
        parse_kickoff_date_time(d, t)
        for d, t in zip(schedule["game_date"], schedule["game_time"])
    ]
    schedule = schedule.sort_values(["kickoff_sort", "week", "game_id"], kind="stable")

    for _, game in schedule.iterrows():
        home = resolver.resolve(game["home_team"])
        away = resolver.resolve(game["away_team"])
        hstate = states.setdefault(home, TeamFormState())
        astate = states.setdefault(away, TeamFormState())
        kickoff = game["kickoff_sort"]
        home_final = float(game["home_final"])
        away_final = float(game["away_final"])
        home_spread = fnum(game.get("home_spread"))
        market_margin = None if home_spread is None else -home_spread
        market_total = fnum(game.get("total"))
        update_form_state(hstate, kickoff, home_final, away_final, market_margin, market_total)
        update_form_state(astate, kickoff, away_final, home_final, None if market_margin is None else -market_margin, market_total)
    return states


def build_current_prediction_frame(
    season: int,
    week: int,
    resolver: TeamResolver,
    stats: dict[int, pd.DataFrame],
    stat_metrics: list[str],
) -> pd.DataFrame:
    weekly_path = WEEKLY_SCHEDULE_DIR / f"week_{week}_CFB_weekly_schedule.csv"
    enriched_path = ENRICHED_DIR / f"week_{week}_CFB_enriched.csv"
    travel_path = TRAVEL_DIR / f"{season}_week_{week}_travel.csv"
    weather_path = WEATHER_DIR / f"week_{week}_CFB_weekly_weather.csv"

    weekly = read_csv(weekly_path, ["game_id", "away_team", "home_team", "home_spread", "total"]).copy()
    weekly["game_id"] = weekly["game_id"].map(gid)

    enriched = read_csv(enriched_path, ["game_id"]).copy()
    enriched["game_id"] = enriched["game_id"].map(gid)
    enriched_lookup = enriched.drop_duplicates("game_id", keep="last").set_index("game_id", drop=False)

    travel = read_csv(travel_path, ["game_id"]).copy() if travel_path.is_file() else pd.DataFrame(columns=["game_id"])
    if not travel.empty:
        travel["game_id"] = travel["game_id"].map(gid)
        travel_lookup = travel.drop_duplicates("game_id", keep="last").set_index("game_id", drop=False)
    else:
        travel_lookup = pd.DataFrame(columns=["game_id"]).set_index("game_id")

    weather = read_csv(weather_path, ["game_id"]).copy() if weather_path.is_file() else pd.DataFrame(columns=["game_id"])
    if not weather.empty:
        weather["game_id"] = weather["game_id"].map(gid)
        weather_lookup = weather.drop_duplicates("game_id", keep="last").set_index("game_id", drop=False)
    else:
        weather_lookup = pd.DataFrame(columns=["game_id"]).set_index("game_id")

    states = build_current_form_states(season, week, resolver)
    rows: list[dict[str, Any]] = []

    for _, game in weekly.iterrows():
        game_id = gid(game["game_id"])
        home_team = resolver.resolve(game["home_team"])
        away_team = resolver.resolve(game["away_team"])
        enriched_row = enriched_lookup.loc[game_id] if game_id in enriched_lookup.index else {}
        travel_row = travel_lookup.loc[game_id] if game_id in travel_lookup.index else {}
        weather_row = weather_lookup.loc[game_id] if game_id in weather_lookup.index else {}

        neutral_override = enriched_row.get("neutral_site") if hasattr(enriched_row, "get") else game.get("neutral_site")
        features = shared_context_features(game, travel_row, weather_row, neutral_override=neutral_override)
        features["home_team"] = home_team
        features["away_team"] = away_team
        features.update(base_market_features(game))
        features.update(predictor_features(enriched_row, historical=False))
        add_team_stat_features(features, stats, stat_metrics, home_team, away_team, season, week)

        kickoff = parse_kickoff_date_time(game.get("game_date"), game.get("game_time"))
        hstate = states.get(home_team, TeamFormState())
        astate = states.get(away_team, TeamFormState())
        add_form_pair(features, form_snapshot(hstate, kickoff), form_snapshot(astate, kickoff))

        rows.append(
            {
                "season": season,
                "season_type": SEASON_TYPE,
                "week": week,
                "game_id": game_id,
                "game_date": clean(game.get("game_date")),
                "game_time": clean(game.get("game_time")),
                **features,
            }
        )

    return pd.DataFrame(rows)


def make_prediction_output(
    current: pd.DataFrame,
    margin_sel: SelectionResult,
    total_sel: SelectionResult,
    direct_margin_sel: SelectionResult,
    direct_total_sel: SelectionResult,
    final_models: dict[str, list[Pipeline]],
    deploy_margin_trained: bool,
    deploy_total_trained: bool,
) -> pd.DataFrame:
    out = current.copy()
    n = len(out)
    pred_margin = np.full(n, np.nan, dtype=float)
    pred_total = np.full(n, np.nan, dtype=float)
    margin_mode: list[str] = [""] * n
    total_mode: list[str] = [""] * n

    have_margin_market = out["market_margin"].notna().to_numpy()
    have_total_market = out["market_total"].notna().to_numpy()

    if have_margin_market.any():
        subset = out.loc[have_margin_market]
        if deploy_margin_trained and margin_sel.mode != "baseline":
            pred_margin[have_margin_market] = predict_target(
                subset, margin_sel, final_models["margin_residual"], "market_margin"
            )
            for i in np.flatnonzero(have_margin_market):
                margin_mode[i] = "TRAINED_RESIDUAL"
        else:
            pred_margin[have_margin_market] = pd.to_numeric(subset["market_margin"], errors="coerce")
            for i in np.flatnonzero(have_margin_market):
                margin_mode[i] = "MARKET_BASELINE"

    missing_margin_market = ~have_margin_market
    if missing_margin_market.any():
        subset = out.loc[missing_margin_market]
        pred_margin[missing_margin_market] = predict_target(
            subset, direct_margin_sel, final_models["direct_margin"], None
        )
        for i in np.flatnonzero(missing_margin_market):
            margin_mode[i] = "DIRECT_NO_MARKET"

    if have_total_market.any():
        subset = out.loc[have_total_market]
        if deploy_total_trained and total_sel.mode != "baseline":
            pred_total[have_total_market] = predict_target(
                subset, total_sel, final_models["total_residual"], "market_total"
            )
            for i in np.flatnonzero(have_total_market):
                total_mode[i] = "TRAINED_RESIDUAL"
        else:
            pred_total[have_total_market] = pd.to_numeric(subset["market_total"], errors="coerce")
            for i in np.flatnonzero(have_total_market):
                total_mode[i] = "MARKET_BASELINE"

    missing_total_market = ~have_total_market
    if missing_total_market.any():
        subset = out.loc[missing_total_market]
        pred_total[missing_total_market] = predict_target(
            subset, direct_total_sel, final_models["direct_total"], None
        )
        for i in np.flatnonzero(missing_total_market):
            total_mode[i] = "DIRECT_NO_MARKET"

    pred_total = np.maximum(pred_total, np.abs(pred_margin))
    predicted_home = (pred_total + pred_margin) / 2.0
    predicted_away = (pred_total - pred_margin) / 2.0

    result = pd.DataFrame(
        {
            "season": out["season"],
            "season_type": out["season_type"],
            "week": out["week"],
            "game_id": out["game_id"],
            "game_date": out["game_date"],
            "game_time": out["game_time"],
            "away_team": out["away_team"],
            "home_team": out["home_team"],
            "market_margin": out["market_margin"],
            "market_total": out["market_total"],
            "market_implied_home_score": np.where(
                out["market_margin"].notna() & out["market_total"].notna(),
                (pd.to_numeric(out["market_total"], errors="coerce") + pd.to_numeric(out["market_margin"], errors="coerce")) / 2.0,
                np.nan,
            ),
            "market_implied_away_score": np.where(
                out["market_margin"].notna() & out["market_total"].notna(),
                (pd.to_numeric(out["market_total"], errors="coerce") - pd.to_numeric(out["market_margin"], errors="coerce")) / 2.0,
                np.nan,
            ),
            "predicted_margin": pred_margin,
            "predicted_total": pred_total,
            "predicted_home_score": predicted_home,
            "predicted_away_score": predicted_away,
            "margin_prediction_mode": margin_mode,
            "total_prediction_mode": total_mode,
            "model_version": SCRIPT_VERSION,
        }
    )
    return result


def selection_to_dict(selection: SelectionResult) -> dict[str, Any]:
    return {
        "target_name": selection.target_name,
        "mode": selection.mode,
        "feature_count": len(selection.feature_columns),
        "component_names": selection.component_names,
        "component_weights": selection.component_weights,
        "shrink": selection.shrink,
        "cv_mae": selection.cv_mae,
        "cv_rmse": selection.cv_rmse,
        "baseline_cv_mae": selection.baseline_cv_mae,
        "baseline_cv_rmse": selection.baseline_cv_rmse,
        "cv_rows": selection.cv_rows,
    }


def main() -> int:
    args = parse_args()
    if args.predict_week is not None and args.predict_week < 1:
        raise ValueError("--predict-week must be >= 1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resolver = TeamResolver(TEAM_MAP_PATH)

    # Include prior season 2020 only if it happens to exist; the repository
    # currently starts at 2021, so 2021 Week 1 team-stat features remain missing.
    stats_seasons = sorted(set(HISTORICAL_SEASONS + [s - 1 for s in HISTORICAL_SEASONS] + [args.season, args.season - 1]))
    stats, stat_metrics = load_all_team_stats(stats_seasons, resolver)

    print("Building leakage-safe historical game matrix...")
    raw_games = build_historical_games(resolver)
    training = build_training_frame(raw_games, stats, stat_metrics)
    training_path = OUT_DIR / "training_games_2021_2025.csv"
    write_csv(training, training_path)
    print(f"Historical games: {len(training)}")

    print("Selecting margin residual model with forward-season validation...")
    margin_sel = select_model(
        training,
        target_name="margin_residual",
        target_col="margin_residual",
        include_market=True,
        baseline_col="market_margin",
        fast=args.fast,
    )
    print("Selecting total residual model with forward-season validation...")
    total_sel = select_model(
        training,
        target_name="total_residual",
        target_col="total_residual",
        include_market=True,
        baseline_col="market_total",
        fast=args.fast,
    )
    print("Selecting direct no-market margin fallback...")
    direct_margin_sel = select_model(
        training,
        target_name="direct_margin",
        target_col="actual_margin",
        include_market=False,
        baseline_col=None,
        fast=args.fast,
    )
    print("Selecting direct no-market total fallback...")
    direct_total_sel = select_model(
        training,
        target_name="direct_total",
        target_col="actual_total",
        include_market=False,
        baseline_col=None,
        fast=args.fast,
    )

    cv_table = pd.concat(
        [
            margin_sel.cv_table,
            total_sel.cv_table,
            direct_margin_sel.cv_table,
            direct_total_sel.cv_table,
        ],
        ignore_index=True,
    )
    write_csv(cv_table, OUT_DIR / "cv_model_results.csv")

    # Fit selected architectures on 2021-2024 only, then evaluate untouched 2025.
    pretest_models = {
        "margin_residual": fit_selected_components(
            training, margin_sel, "margin_residual", [2021, 2022, 2023, 2024], args.fast, "market_margin"
        ),
        "total_residual": fit_selected_components(
            training, total_sel, "total_residual", [2021, 2022, 2023, 2024], args.fast, "market_total"
        ),
        "direct_margin": fit_selected_components(
            training, direct_margin_sel, "actual_margin", [2021, 2022, 2023, 2024], args.fast
        ),
        "direct_total": fit_selected_components(
            training, direct_total_sel, "actual_total", [2021, 2022, 2023, 2024], args.fast
        ),
    }

    validation_predictions, validation_metrics, gates = evaluate_2025(
        training,
        margin_sel,
        total_sel,
        pretest_models["margin_residual"],
        pretest_models["total_residual"],
    )
    write_csv(validation_predictions, OUT_DIR / "validation_2025_predictions.csv")

    # Choose the deployment combination by FINAL SCORE accuracy on 2025.
    # Only corrections that already beat the market in forward 2022-2024 CV
    # are eligible. We test four combinations: market/market, trained
    # margin only, trained total only, and both. If none improves average
    # team-score MAE, deployment remains the sportsbook-implied score.
    baseline_margin = validation_predictions["market_margin"].to_numpy(float)
    baseline_total = validation_predictions["market_total"].to_numpy(float)
    trained_margin = validation_predictions["model_predicted_margin"].to_numpy(float)
    trained_total = validation_predictions["model_predicted_total"].to_numpy(float)
    actual_home = validation_predictions["home_final"].to_numpy(float)
    actual_away = validation_predictions["away_final"].to_numpy(float)

    combo_rows = []
    eligible_margin = margin_sel.mode != "baseline"
    eligible_total = total_sel.mode != "baseline"
    for use_margin in [False, True]:
        if use_margin and not eligible_margin:
            continue
        for use_total in [False, True]:
            if use_total and not eligible_total:
                continue
            cmargin = trained_margin if use_margin else baseline_margin
            ctotal = trained_total if use_total else baseline_total
            ctotal = np.maximum(ctotal, np.abs(cmargin))
            chome = (ctotal + cmargin) / 2.0
            caway = (ctotal - cmargin) / 2.0
            team_abs = np.concatenate([np.abs(actual_home - chome), np.abs(actual_away - caway)])
            combo_rows.append(
                {
                    "use_trained_margin": use_margin,
                    "use_trained_total": use_total,
                    "average_team_score_mae": float(team_abs.mean()),
                    "average_team_score_rmse": float(math.sqrt(np.mean(team_abs ** 2))),
                }
            )

    combo_df = pd.DataFrame(combo_rows).sort_values(
        ["average_team_score_mae", "average_team_score_rmse", "use_trained_margin", "use_trained_total"],
        kind="stable",
    )
    baseline_combo = combo_df[
        ~combo_df["use_trained_margin"] & ~combo_df["use_trained_total"]
    ].iloc[0]
    best_combo = combo_df.iloc[0]
    if best_combo["average_team_score_mae"] < baseline_combo["average_team_score_mae"] - 1e-12:
        deploy_margin_trained = bool(best_combo["use_trained_margin"])
        deploy_total_trained = bool(best_combo["use_trained_total"])
    else:
        deploy_margin_trained = False
        deploy_total_trained = False

    combo_metric_rows = []
    for _, crow in combo_df.iterrows():
        name = (
            ("TRAINED_MARGIN" if crow["use_trained_margin"] else "MARKET_MARGIN")
            + "+"
            + ("TRAINED_TOTAL" if crow["use_trained_total"] else "MARKET_TOTAL")
        )
        combo_metric_rows.append(
            {
                "metric": f"score_combo:{name}",
                "rows": len(validation_predictions) * 2,
                "model_mae": crow["average_team_score_mae"],
                "market_mae": baseline_combo["average_team_score_mae"],
                "mae_improvement_points": baseline_combo["average_team_score_mae"] - crow["average_team_score_mae"],
                "mae_improvement_pct": (
                    (baseline_combo["average_team_score_mae"] - crow["average_team_score_mae"])
                    / baseline_combo["average_team_score_mae"]
                ),
                "model_rmse": crow["average_team_score_rmse"],
                "market_rmse": baseline_combo["average_team_score_rmse"],
            }
        )
    validation_metrics = pd.concat(
        [validation_metrics, pd.DataFrame(combo_metric_rows)], ignore_index=True
    )
    write_csv(validation_metrics, OUT_DIR / "validation_2025_metrics.csv")

    print("2025 validation:")
    for _, row in validation_metrics.iterrows():
        if row["metric"] == "winner_accuracy":
            print(
                f"  winner_accuracy model={row['model_mae']:.4f} market={row['market_mae']:.4f}"
            )
        else:
            print(
                f"  {row['metric']}: model_MAE={row['model_mae']:.4f} "
                f"market_MAE={row['market_mae']:.4f} "
                f"improvement={row['mae_improvement_points']:.4f}"
            )

    # Refit chosen architectures on every historical season for deployment.
    final_models = {
        "margin_residual": fit_selected_components(
            training, margin_sel, "margin_residual", HISTORICAL_SEASONS, args.fast, "market_margin"
        ) if deploy_margin_trained else [],
        "total_residual": fit_selected_components(
            training, total_sel, "total_residual", HISTORICAL_SEASONS, args.fast, "market_total"
        ) if deploy_total_trained else [],
        "direct_margin": fit_selected_components(
            training, direct_margin_sel, "actual_margin", HISTORICAL_SEASONS, args.fast
        ),
        "direct_total": fit_selected_components(
            training, direct_total_sel, "actual_total", HISTORICAL_SEASONS, args.fast
        ),
    }

    feature_rows = []
    for label, selection in [
        ("margin_residual", margin_sel),
        ("total_residual", total_sel),
        ("direct_margin", direct_margin_sel),
        ("direct_total", direct_total_sel),
    ]:
        for col in selection.feature_columns:
            feature_rows.append(
                {
                    "model": label,
                    "feature": col,
                    "feature_type": "categorical" if col in selection.categorical_columns else "numeric",
                }
            )
    write_csv(pd.DataFrame(feature_rows), OUT_DIR / "feature_columns.csv")

    def saved_selection(selection: SelectionResult) -> dict[str, Any]:
        return {
            **selection_to_dict(selection),
            "feature_columns": selection.feature_columns,
            "numeric_columns": selection.numeric_columns,
            "categorical_columns": selection.categorical_columns,
        }

    model_payload = {
        "script_version": SCRIPT_VERSION,
        "historical_seasons": HISTORICAL_SEASONS,
        "stat_metrics": stat_metrics,
        "margin_selection": saved_selection(margin_sel),
        "total_selection": saved_selection(total_sel),
        "direct_margin_selection": saved_selection(direct_margin_sel),
        "direct_total_selection": saved_selection(direct_total_sel),
        "deploy_margin_trained": deploy_margin_trained,
        "deploy_total_trained": deploy_total_trained,
        "models": final_models,
    }
    model_path = OUT_DIR / "score_model.joblib"
    joblib.dump(model_payload, model_path)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "historical_seasons": HISTORICAL_SEASONS,
        "cv_validation_seasons": CV_VALIDATION_SEASONS,
        "final_validation_season": FINAL_VALIDATION_SEASON,
        "historical_games": int(len(training)),
        "team_stat_metric_count": len(stat_metrics),
        "margin": selection_to_dict(margin_sel),
        "total": selection_to_dict(total_sel),
        "direct_margin": selection_to_dict(direct_margin_sel),
        "direct_total": selection_to_dict(direct_total_sel),
        "validation_2025": {
            **gates,
            "selected_score_combo": {
                "use_trained_margin": deploy_margin_trained,
                "use_trained_total": deploy_total_trained,
                "average_team_score_mae": float(best_combo["average_team_score_mae"]),
                "market_average_team_score_mae": float(baseline_combo["average_team_score_mae"]),
            },
        },
        "deployment": {
            "margin": "TRAINED_RESIDUAL" if deploy_margin_trained else "MARKET_BASELINE",
            "total": "TRAINED_RESIDUAL" if deploy_total_trained else "MARKET_BASELINE",
            "missing_market_margin": "DIRECT_TRAINED_MODEL",
            "missing_market_total": "DIRECT_TRAINED_MODEL",
        },
        "excluded_from_training": {
            "fpi": "No historical point-in-time FPI files exist in repo for 2021-2025.",
            "injuries": "No historical point-in-time injury files exist in repo for 2021-2025.",
            "existing_predicted_scores": "Excluded to avoid training on output from the old fixed-weight projection.",
            "existing_ev_kelly_pick_fields": "Excluded because this is a score model, not a pick-filter model.",
        },
        "write_scope": str(OUT_DIR),
    }
    write_json(manifest, OUT_DIR / "score_model_manifest.json")

    if args.predict_week is not None:
        # Reload current-season stats if they were not present during initial load.
        current_needed = [args.season, args.season - 1]
        missing_stats = [s for s in current_needed if s not in stats and (TEAM_STATS_DIR / f"{s}_team_stats.csv").is_file()]
        if missing_stats:
            extra_stats, extra_metrics = load_all_team_stats(missing_stats, resolver)
            stats.update(extra_stats)
            stat_metrics = sorted(set(stat_metrics) | set(extra_metrics))

        current = build_current_prediction_frame(
            args.season,
            args.predict_week,
            resolver,
            stats,
            stat_metrics,
        )
        prediction_output = make_prediction_output(
            current,
            margin_sel,
            total_sel,
            direct_margin_sel,
            direct_total_sel,
            final_models,
            deploy_margin_trained,
            deploy_total_trained,
        )
        prediction_path = OUT_DIR / f"week_{args.predict_week}_CFB_trained_score_predictions.csv"
        write_csv(prediction_output, prediction_path)
        print(f"Current predictions: {prediction_path}")

    print(f"Training matrix: {training_path}")
    print(f"CV results: {OUT_DIR / 'cv_model_results.csv'}")
    print(f"2025 predictions: {OUT_DIR / 'validation_2025_predictions.csv'}")
    print(f"2025 metrics: {OUT_DIR / 'validation_2025_metrics.csv'}")
    print(f"Saved model: {model_path}")
    print(f"Manifest: {OUT_DIR / 'score_model_manifest.json'}")
    print(
        "Deployment margin="
        + ("TRAINED_RESIDUAL" if deploy_margin_trained else "MARKET_BASELINE")
        + " total="
        + ("TRAINED_RESIDUAL" if deploy_total_trained else "MARKET_BASELINE")
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

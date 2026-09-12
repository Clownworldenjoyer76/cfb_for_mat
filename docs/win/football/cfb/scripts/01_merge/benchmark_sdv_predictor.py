#!/usr/bin/env python3
"""
benchmark_sdv_predictor.py

Benchmarks the CURRENT SportsDataverse CFB pregame winner predictor without
requiring the local `sportsdataverse` Python package.

Data source (official SportsDataverse release assets):
  cfb_schedules/cfb_schedules_{season}.parquet
  cfb_ratings_weekly/cfb_ratings_weekly_{season}.parquet

Leakage boundary:
  A game in week W uses only the ratings snapshot through week W-1.
  Week 1 is excluded because there is no current-season W-1 snapshot.

Current SDV winner model reproduced here exactly from cfb_game_predict.py and
cfb_prediction_constants.py as published August 2026:
  expected_margin = slope(games_played) * (home_adj_net - away_adj_net) + HFA
  home_win_prob = Phi(expected_margin / margin_sd)

For straight-up winner accuracy, Phi is monotonic, so the predicted winner is
simply home when expected_margin > 0 and away otherwise.

The slope uses the SMALLER of the two teams' games counts, matching SDV's
cfb_predict_games() implementation.

Outputs:
  docs/win/football/cfb/data/sdv_predictor_benchmark/
    sdv_game_predictions_2022_2025.csv
    sdv_accuracy_by_season.csv
    sdv_accuracy_summary.csv
    cache/*.parquet
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests


SEASONS = [2022, 2023, 2024, 2025]
MIN_WEEK = 2

# Current SportsDataverse "modern" constants, verified from
# sportsdataverse/cfb/cfb_prediction_constants.py on 2026-08-29.
HFA_POINTS = 3.0365
MARGIN_SD = 18.7894
NET_POINTS_SCALE = 24.6578
SLOPE_BY_GAMES = (
    (0, 3, 10.6201),
    (4, 5, 26.0576),
    (6, 7, 42.0032),
    (8, 20, 54.4874),
)

SDV_RELEASE_BASE = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
)
SCHEDULE_URL = SDV_RELEASE_BASE + "/cfb_schedules/cfb_schedules_{season}.parquet"
RATINGS_URL = (
    SDV_RELEASE_BASE
    + "/cfb_ratings_weekly/cfb_ratings_weekly_{season}.parquet"
)

REQUEST_TIMEOUT = 90
SCRIPT_VERSION = "2.0-direct-formula-no-package"
USER_AGENT = "cfb-sdv-benchmark/2.0"


def cfb_root() -> Path:
    # docs/win/football/cfb/scripts/01_merge/benchmark_sdv_predictor.py
    return Path(__file__).resolve().parents[2]


def download_parquet(url: str, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return pd.read_parquet(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    try:
        frame = pd.read_parquet(io.BytesIO(response.content))
    except Exception as exc:
        raise RuntimeError(
            f"Downloaded {url}, but could not read it as parquet: {exc}"
        ) from exc

    frame.to_parquet(cache_path, index=False)
    return frame


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    text = series.astype("string").str.strip().str.lower()
    return text.isin({"true", "t", "1", "yes", "y"})


def slope_for_games(games_played: pd.Series) -> pd.Series:
    gp = pd.to_numeric(games_played, errors="coerce")
    slope = pd.Series(NET_POINTS_SCALE, index=gp.index, dtype=float)

    for lo, hi, value in SLOPE_BY_GAMES:
        mask = gp.between(lo, hi, inclusive="both")
        slope.loc[mask] = value

    return slope


def load_season(season: int, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading official SDV data for {season}...")

    schedule = download_parquet(
        SCHEDULE_URL.format(season=season),
        cache_dir / f"cfb_schedules_{season}.parquet",
    )
    ratings = download_parquet(
        RATINGS_URL.format(season=season),
        cache_dir / f"cfb_ratings_weekly_{season}.parquet",
    )

    require_columns(
        schedule,
        {
            "game_id",
            "season",
            "week",
            "home_id",
            "away_id",
            "neutral_site",
            "home_points",
            "away_points",
        },
        f"{season} schedule",
    )
    require_columns(
        ratings,
        {
            "season",
            "team_id",
            "through_week",
            "adj_net",
            "games",
        },
        f"{season} weekly ratings",
    )

    schedule = to_numeric(
        schedule,
        [
            "game_id",
            "season",
            "week",
            "home_id",
            "away_id",
            "home_points",
            "away_points",
        ],
    )
    ratings = to_numeric(
        ratings,
        ["season", "team_id", "through_week", "adj_net", "games"],
    )

    schedule["neutral_site"] = to_bool(schedule["neutral_site"])

    schedule = schedule[
        schedule["home_points"].notna()
        & schedule["away_points"].notna()
        & schedule["week"].notna()
        & (schedule["week"] >= MIN_WEEK)
    ].copy()

    ratings = ratings[
        ratings["team_id"].notna()
        & ratings["through_week"].notna()
        & ratings["adj_net"].notna()
        & ratings["games"].notna()
    ].copy()

    for col in ("game_id", "season", "week", "home_id", "away_id"):
        schedule[col] = schedule[col].astype("int64")

    for col in ("season", "team_id", "through_week", "games"):
        ratings[col] = ratings[col].astype("int64")

    return schedule, ratings


def predict_season(season: int, cache_dir: Path) -> pd.DataFrame:
    schedule, ratings = load_season(season, cache_dir)

    home_ratings = ratings[
        ["season", "through_week", "team_id", "adj_net", "games"]
    ].rename(
        columns={
            "through_week": "asof_week",
            "team_id": "home_id",
            "adj_net": "home_adj_net",
            "games": "home_games",
        }
    )

    away_ratings = ratings[
        ["season", "through_week", "team_id", "adj_net", "games"]
    ].rename(
        columns={
            "through_week": "asof_week",
            "team_id": "away_id",
            "adj_net": "away_adj_net",
            "games": "away_games",
        }
    )

    games = schedule.copy()
    games["asof_week"] = games["week"] - 1

    games = games.merge(
        home_ratings,
        on=["season", "asof_week", "home_id"],
        how="inner",
        validate="many_to_one",
    )
    games = games.merge(
        away_ratings,
        on=["season", "asof_week", "away_id"],
        how="inner",
        validate="many_to_one",
    )

    if games.empty:
        raise RuntimeError(f"{season}: no games matched SDV week W-1 ratings.")

    games["binding_games"] = games[["home_games", "away_games"]].min(axis=1)
    games["sdv_slope"] = slope_for_games(games["binding_games"])

    hfa = np.where(games["neutral_site"].astype(bool), 0.0, HFA_POINTS)
    games["sdv_exp_margin"] = (
        games["sdv_slope"]
        * (games["home_adj_net"] - games["away_adj_net"])
        + hfa
    )

    # For winner accuracy this is exactly equivalent to home_win_prob > 0.5.
    games["sdv_home_pick"] = games["sdv_exp_margin"] > 0.0
    games["actual_home_win"] = games["home_points"] > games["away_points"]
    games["sdv_winner_correct"] = (
        games["sdv_home_pick"] == games["actual_home_win"]
    )
    games["actual_margin"] = games["home_points"] - games["away_points"]
    games["sdv_margin_abs_error"] = (
        games["sdv_exp_margin"] - games["actual_margin"]
    ).abs()

    keep = [
        "season",
        "week",
        "game_id",
        "home_id",
        "away_id",
    ]
    for optional in ("home_team", "away_team", "season_type", "neutral_site"):
        if optional in games.columns:
            keep.append(optional)
    keep += [
        "home_points",
        "away_points",
        "asof_week",
        "home_adj_net",
        "away_adj_net",
        "home_games",
        "away_games",
        "binding_games",
        "sdv_slope",
        "sdv_exp_margin",
        "sdv_home_pick",
        "actual_home_win",
        "sdv_winner_correct",
        "actual_margin",
        "sdv_margin_abs_error",
    ]

    return games[keep].sort_values(["week", "game_id"]).reset_index(drop=True)


def accuracy_row(season: int | str, df: pd.DataFrame) -> dict[str, object]:
    n = int(len(df))
    correct = int(df["sdv_winner_correct"].astype(bool).sum())
    return {
        "season": season,
        "games": n,
        "correct_winners": correct,
        "winner_accuracy_pct": (100.0 * correct / n) if n else np.nan,
        "margin_mae": float(df["sdv_margin_abs_error"].mean()) if n else np.nan,
    }


def main() -> int:
    print(f"SDV benchmark version: {SCRIPT_VERSION}")
    print("Mode: official SDV release parquets + published SDV formula; no sportsdataverse import")
    print()

    root = cfb_root()
    out_dir = root / "data" / "sdv_predictor_benchmark"
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    season_rows: list[dict[str, object]] = []

    try:
        for season in SEASONS:
            result = predict_season(season, cache_dir)
            all_frames.append(result)
            row = accuracy_row(season, result)
            season_rows.append(row)
            print(
                f"{season}: {row['correct_winners']}/{row['games']} correct = "
                f"{row['winner_accuracy_pct']:.2f}% | "
                f"margin MAE={row['margin_mae']:.4f}"
            )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    games = pd.concat(all_frames, ignore_index=True)
    by_season = pd.DataFrame(season_rows)
    overall = accuracy_row("2022-2025", games)
    summary = pd.DataFrame([overall])

    game_path = out_dir / "sdv_game_predictions_2022_2025.csv"
    season_path = out_dir / "sdv_accuracy_by_season.csv"
    summary_path = out_dir / "sdv_accuracy_summary.csv"

    games.to_csv(game_path, index=False)
    by_season.to_csv(season_path, index=False)
    summary.to_csv(summary_path, index=False)

    print()
    print("SDV CURRENT PREBUILT PREDICTOR — STRAIGHT-UP WINNER ACCURACY")
    print(f"  games: {overall['games']}")
    print(f"  correct: {overall['correct_winners']}")
    print(f"  WIN %: {overall['winner_accuracy_pct']:.2f}%")
    print(f"  margin MAE: {overall['margin_mae']:.4f}")
    print()
    print("By season:")
    for row in season_rows:
        print(
            f"  {int(row['season'])}: {row['correct_winners']}/{row['games']} = "
            f"{row['winner_accuracy_pct']:.2f}%"
        )
    print()
    print(f"Game predictions: {game_path}")
    print(f"Season results: {season_path}")
    print(f"Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

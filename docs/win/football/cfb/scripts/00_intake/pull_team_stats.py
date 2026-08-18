#!/usr/bin/env python3
# docs/win/football/nfl/scripts/00_intake/pull_team_stats.py
#
# Builds weekly NFL team-strength stats from the local nflverse PBP intake file.
#
# Input:
#   docs/win/football/nfl/00_intake/pbp/{season}_pbp.csv.gz
#
# Output:
#   docs/win/football/nfl/00_intake/team_stats/{season}_team_stats.csv
#
# Summary/error log:
#   docs/win/football/nfl/errors/00_intake/pull_team_stats.txt

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "season",
    "week",
    "team",
    "off_epa_per_play",
    "def_epa_per_play",
    "off_success_rate",
    "def_success_rate",
    "yards_per_play",
    "yards_per_play_allowed",
    "points_per_drive",
    "points_per_drive_allowed",
    "red_zone_td_rate",
    "red_zone_td_rate_allowed",
    "early_down_epa",
    "third_down_conversion_rate",
]

SCRIMMAGE_PLAY_TYPES = {"pass", "run"}


class RunLog:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.lines: list[str] = []

    def write_line(self, message: str = "", *, stderr: bool = False) -> None:
        self.lines.append(message)
        print(message, file=sys.stderr if stderr else sys.stdout)

    def save(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def resolve_paths() -> tuple[Path, Path, Path, Path]:
    nfl_root = Path(__file__).resolve().parents[2]

    pbp_dir = nfl_root / "00_intake" / "pbp"
    output_dir = nfl_root / "00_intake" / "team_stats"
    error_dir = nfl_root / "errors" / "00_intake"

    return nfl_root, pbp_dir, output_dir, error_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build NFL weekly team stats from local nflverse PBP."
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="NFL season. If omitted, NFL_SEASON environment variable is used.",
    )
    return parser.parse_args()


def get_season(cli_season: str | None) -> str:
    if cli_season:
        return str(cli_season)

    env_season = os.getenv("NFL_SEASON")
    if env_season:
        return str(env_season)

    raise SystemExit("Missing season. Pass --season or set NFL_SEASON.")


def write_empty_output(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_path, index=False)


def read_pbp(pbp_path: Path) -> pd.DataFrame:
    if not pbp_path.exists():
        raise FileNotFoundError(f"PBP input file not found: {pbp_path}")

    try:
        return pd.read_csv(pbp_path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "season",
        "week",
        "epa",
        "success",
        "yards_gained",
        "down",
        "yardline_100",
        "posteam_score",
        "posteam_score_post",
        "touchdown",
        "pass_touchdown",
        "rush_touchdown",
        "third_down_converted",
        "third_down_failed",
        "first_down",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def get_drive_column(df: pd.DataFrame) -> str:
    if "fixed_drive" in df.columns:
        return "fixed_drive"
    if "drive" in df.columns:
        return "drive"

    raise ValueError("Missing required drive column: fixed_drive or drive")


def build_valid_scrimmage_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        pbp,
        [
            "season",
            "week",
            "posteam",
            "defteam",
            "epa",
            "success",
            "yards_gained",
            "down",
        ],
        "scrimmage-play team stats",
    )

    df = pbp.copy()

    mask = (
        df["season"].notna()
        & df["week"].notna()
        & df["posteam"].notna()
        & df["defteam"].notna()
        & df["epa"].notna()
    )

    if "play_type" in df.columns:
        mask &= df["play_type"].isin(SCRIMMAGE_PLAY_TYPES)

    return df.loc[mask].copy()


def build_offense_stats(valid_plays: pd.DataFrame) -> pd.DataFrame:
    off = (
        valid_plays.groupby(["season", "week", "posteam"], dropna=False)
        .agg(
            off_epa_per_play=("epa", "mean"),
            off_success_rate=("success", "mean"),
            yards_per_play=("yards_gained", "mean"),
        )
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    early_downs = valid_plays[valid_plays["down"].isin([1, 2])].copy()

    if early_downs.empty:
        early = pd.DataFrame(columns=["season", "week", "team", "early_down_epa"])
    else:
        early = (
            early_downs.groupby(["season", "week", "posteam"], dropna=False)
            .agg(early_down_epa=("epa", "mean"))
            .reset_index()
            .rename(columns={"posteam": "team"})
        )

    return off.merge(early, on=["season", "week", "team"], how="outer")


def build_defense_stats(valid_plays: pd.DataFrame) -> pd.DataFrame:
    return (
        valid_plays.groupby(["season", "week", "defteam"], dropna=False)
        .agg(
            def_epa_per_play=("epa", "mean"),
            def_success_rate=("success", "mean"),
            yards_per_play_allowed=("yards_gained", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )


def build_third_down_stats(pbp: pd.DataFrame, valid_plays: pd.DataFrame) -> pd.DataFrame:
    if {"third_down_converted", "third_down_failed"}.issubset(pbp.columns):
        require_columns(
            pbp,
            ["season", "week", "posteam"],
            "third-down conversion rate",
        )

        third = pbp[
            pbp["season"].notna()
            & pbp["week"].notna()
            & pbp["posteam"].notna()
            & (
                (pbp["third_down_converted"] == 1)
                | (pbp["third_down_failed"] == 1)
            )
        ].copy()

        if third.empty:
            return pd.DataFrame(
                columns=["season", "week", "team", "third_down_conversion_rate"]
            )

        third["third_down_conversion_flag"] = np.where(
            third["third_down_converted"] == 1, 1.0, 0.0
        )

    elif {"down", "first_down"}.issubset(valid_plays.columns):
        third = valid_plays[
            valid_plays["posteam"].notna()
            & valid_plays["down"].eq(3)
        ].copy()

        if third.empty:
            return pd.DataFrame(
                columns=["season", "week", "team", "third_down_conversion_rate"]
            )

        third["third_down_conversion_flag"] = np.where(
            third["first_down"] == 1, 1.0, 0.0
        )

    else:
        return pd.DataFrame(
            columns=["season", "week", "team", "third_down_conversion_rate"]
        )

    return (
        third.groupby(["season", "week", "posteam"], dropna=False)
        .agg(third_down_conversion_rate=("third_down_conversion_flag", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )


def build_drive_points_stats(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    drive_col = get_drive_column(pbp)

    require_columns(
        pbp,
        [
            "season",
            "week",
            "game_id",
            drive_col,
            "posteam",
            "defteam",
            "posteam_score",
            "posteam_score_post",
        ],
        "points per drive",
    )

    sort_columns = ["season", "week", "game_id", drive_col]
    if "play_id" in pbp.columns:
        sort_columns.append("play_id")

    drives = pbp[
        pbp["season"].notna()
        & pbp["week"].notna()
        & pbp["game_id"].notna()
        & pbp[drive_col].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
    ].copy()

    if drives.empty:
        empty_off = pd.DataFrame(columns=["season", "week", "team", "points_per_drive"])
        empty_def = pd.DataFrame(
            columns=["season", "week", "team", "points_per_drive_allowed"]
        )
        return empty_off, empty_def

    drives = drives.sort_values(sort_columns)

    drive_keys = ["season", "week", "game_id", drive_col, "posteam", "defteam"]

    drive_scores = (
        drives.groupby(drive_keys, dropna=False)
        .agg(
            drive_start_score=("posteam_score", "first"),
            drive_end_score=("posteam_score_post", "last"),
        )
        .reset_index()
    )

    drive_scores["drive_points"] = (
        drive_scores["drive_end_score"] - drive_scores["drive_start_score"]
    )

    drive_scores.loc[drive_scores["drive_points"] < 0, "drive_points"] = 0

    off_points = (
        drive_scores.groupby(["season", "week", "posteam"], dropna=False)
        .agg(points_per_drive=("drive_points", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    def_points = (
        drive_scores.groupby(["season", "week", "defteam"], dropna=False)
        .agg(points_per_drive_allowed=("drive_points", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    return off_points, def_points


def add_offensive_touchdown_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"touchdown", "td_team"}.issubset(df.columns):
        df["offensive_touchdown_flag"] = np.where(
            (df["touchdown"] == 1) & (df["td_team"] == df["posteam"]),
            1.0,
            0.0,
        )
        return df

    touchdown_parts = []

    if "pass_touchdown" in df.columns:
        touchdown_parts.append(df["pass_touchdown"].fillna(0).eq(1))

    if "rush_touchdown" in df.columns:
        touchdown_parts.append(df["rush_touchdown"].fillna(0).eq(1))

    if touchdown_parts:
        flag = touchdown_parts[0]
        for part in touchdown_parts[1:]:
            flag = flag | part

        df["offensive_touchdown_flag"] = np.where(flag, 1.0, 0.0)
        return df

    if "touchdown" in df.columns:
        df["offensive_touchdown_flag"] = np.where(df["touchdown"] == 1, 1.0, 0.0)
        return df

    df["offensive_touchdown_flag"] = np.nan
    return df


def build_red_zone_stats(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    drive_col = get_drive_column(pbp)

    require_columns(
        pbp,
        [
            "season",
            "week",
            "game_id",
            drive_col,
            "posteam",
            "defteam",
            "yardline_100",
        ],
        "red-zone touchdown rate",
    )

    df = pbp[
        pbp["season"].notna()
        & pbp["week"].notna()
        & pbp["game_id"].notna()
        & pbp[drive_col].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
    ].copy()

    if df.empty:
        empty_off = pd.DataFrame(columns=["season", "week", "team", "red_zone_td_rate"])
        empty_def = pd.DataFrame(
            columns=["season", "week", "team", "red_zone_td_rate_allowed"]
        )
        return empty_off, empty_def

    df = add_offensive_touchdown_flag(df)

    drive_keys = ["season", "week", "game_id", drive_col, "posteam", "defteam"]

    red_zone_trips = (
        df[df["yardline_100"].between(0, 20, inclusive="both")]
        [drive_keys]
        .drop_duplicates()
    )

    if red_zone_trips.empty:
        empty_off = pd.DataFrame(columns=["season", "week", "team", "red_zone_td_rate"])
        empty_def = pd.DataFrame(
            columns=["season", "week", "team", "red_zone_td_rate_allowed"]
        )
        return empty_off, empty_def

    td_by_drive = (
        df.groupby(drive_keys, dropna=False)
        .agg(red_zone_drive_td=("offensive_touchdown_flag", "max"))
        .reset_index()
    )

    trips = red_zone_trips.merge(td_by_drive, on=drive_keys, how="left")
    trips["red_zone_drive_td"] = trips["red_zone_drive_td"].fillna(0)

    off_rz = (
        trips.groupby(["season", "week", "posteam"], dropna=False)
        .agg(red_zone_td_rate=("red_zone_drive_td", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )

    def_rz = (
        trips.groupby(["season", "week", "defteam"], dropna=False)
        .agg(red_zone_td_rate_allowed=("red_zone_drive_td", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    return off_rz, def_rz


def merge_stat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    result: pd.DataFrame | None = None

    for frame in frames:
        if frame is None or frame.empty:
            continue

        if result is None:
            result = frame.copy()
        else:
            result = result.merge(frame, on=["season", "week", "team"], how="outer")

    if result is None:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for col in OUTPUT_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan

    result = result[OUTPUT_COLUMNS]
    result = result.sort_values(["season", "week", "team"]).reset_index(drop=True)

    return result


def build_team_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    pbp = coerce_numeric_columns(pbp)

    valid_plays = build_valid_scrimmage_plays(pbp)

    if valid_plays.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    offense_stats = build_offense_stats(valid_plays)
    defense_stats = build_defense_stats(valid_plays)
    third_down_stats = build_third_down_stats(pbp, valid_plays)
    off_points, def_points = build_drive_points_stats(pbp)
    off_rz, def_rz = build_red_zone_stats(pbp)

    team_stats = merge_stat_frames(
        [
            offense_stats,
            defense_stats,
            off_points,
            def_points,
            off_rz,
            def_rz,
            third_down_stats,
        ]
    )

    return team_stats


def run() -> int:
    args = parse_args()
    season = get_season(args.season)

    _, pbp_dir, output_dir, error_dir = resolve_paths()

    pbp_path = pbp_dir / f"{season}_pbp.csv.gz"
    output_path = output_dir / f"{season}_team_stats.csv"
    log_path = error_dir / "pull_team_stats.txt"

    log = RunLog(log_path)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)

        log.write_line("=" * 80)
        log.write_line(
            f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] "
            f"pull_team_stats.py started | season={season}"
        )
        log.write_line(f"input={pbp_path}")
        log.write_line(f"output={output_path}")
        log.write_line(f"log={log_path}")

        pbp = read_pbp(pbp_path)

        if pbp.empty:
            write_empty_output(output_path)
            log.write_line("pbp_rows=0")
            log.write_line("output_rows=0")
            log.write_line("output_columns=15")
            log.write_line("status=empty_pbp_written")
            log.write_line("=" * 80)
            return 0

        team_stats = build_team_stats(pbp)
        team_stats.to_csv(output_path, index=False)

        log.write_line(f"pbp_rows={len(pbp)}")
        log.write_line(f"output_rows={len(team_stats)}")
        log.write_line(f"output_columns={len(team_stats.columns)}")
        log.write_line("status=success")
        log.write_line("=" * 80)

        return 0

    except Exception as exc:
        log.write_line("=" * 80, stderr=True)
        log.write_line("pull_team_stats.py failed", stderr=True)
        log.write_line(f"error={exc}", stderr=True)
        log.write_line(traceback.format_exc(), stderr=True)
        log.write_line("status=failed", stderr=True)
        log.write_line("=" * 80, stderr=True)
        return 1

    finally:
        log.save()


if __name__ == "__main__":
    raise SystemExit(run())

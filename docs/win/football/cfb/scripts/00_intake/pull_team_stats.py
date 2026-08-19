#!/usr/bin/env python3
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

SDV_REQUIRED_COLUMNS = [
    "season",
    "week",
    "game_id",
    "sequenceNumber",
    "pos_team",
    "def_pos_team",
    "homeTeamId",
    "awayTeamId",
    "homeTeamName",
    "awayTeamName",
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
    cfb_root = Path(__file__).resolve().parents[2]
    pbp_dir = cfb_root / "00_intake" / "pbp"
    output_dir = cfb_root / "00_intake" / "team_stats"
    error_dir = cfb_root / "errors" / "00_intake"
    return cfb_root, pbp_dir, output_dir, error_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CFB weekly team stats from native SportsDataverse PBP."
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="CFB season. If omitted, CFB_SEASON environment variable is used.",
    )
    parser.add_argument(
        "--pbp-path",
        type=str,
        default=None,
        help=(
            "Optional explicit PBP path. Supports .parquet and .csv/.csv.gz. "
            "If omitted, 00_intake/pbp/{season}_pbp.parquet is used."
        ),
    )
    return parser.parse_args()


def get_season(cli_season: str | None) -> str:
    if cli_season:
        return str(cli_season)
    env_season = os.getenv("CFB_SEASON")
    if env_season:
        return str(env_season)
    raise SystemExit("Missing season. Pass --season or set CFB_SEASON.")


def write_empty_output(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_path, index=False)


def read_pbp(pbp_path: Path) -> pd.DataFrame:
    if not pbp_path.exists():
        raise FileNotFoundError(f"PBP input file not found: {pbp_path}")

    suffixes = [s.lower() for s in pbp_path.suffixes]

    if pbp_path.suffix.lower() == ".parquet":
        return pd.read_parquet(pbp_path)

    if ".csv" in suffixes:
        try:
            return pd.read_csv(pbp_path, low_memory=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    raise ValueError(
        f"Unsupported PBP format: {pbp_path}. Expected .parquet, .csv, or .csv.gz."
    )


def require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def _full_team_name(name: object, mascot: object = None) -> str | None:
    if pd.isna(name):
        return None

    base = str(name).strip()
    if not base:
        return None

    if mascot is None or pd.isna(mascot):
        return base

    nick = str(mascot).strip()
    if not nick:
        return base

    if base.casefold().endswith(nick.casefold()):
        return base

    return f"{base} {nick}"


def _native_team_names(pbp: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Published SportsDataverse season Parquets currently store pos_team and
    def_pos_team as full display-name strings (for example, "Ohio State Buckeyes").
    Use those native values directly.
    """
    posteam = pbp["pos_team"].astype("object")
    defteam = pbp["def_pos_team"].astype("object")
    return posteam, defteam


def _home_away_display_names(
    pbp: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    if "homeTeamMascot" in pbp.columns:
        home_names = pd.Series(
            [
                _full_team_name(name, mascot)
                for name, mascot in zip(
                    pbp["homeTeamName"],
                    pbp["homeTeamMascot"],
                )
            ],
            index=pbp.index,
            dtype="object",
        )
    else:
        home_names = pbp["homeTeamName"].astype("object")

    if "awayTeamMascot" in pbp.columns:
        away_names = pd.Series(
            [
                _full_team_name(name, mascot)
                for name, mascot in zip(
                    pbp["awayTeamName"],
                    pbp["awayTeamMascot"],
                )
            ],
            index=pbp.index,
            dtype="object",
        )
    else:
        away_names = pbp["awayTeamName"].astype("object")

    return home_names, away_names

def _offense_score_from_is_home(
    is_home: pd.Series,
    home_score: pd.Series,
    away_score: pd.Series,
) -> pd.Series:
    """
    Return the score for the possession team at play start using the native
    SportsDataverse is_home flag. This avoids reconstructing or matching team
    names for score perspective.
    """
    home_flag = is_home.fillna(False).astype(bool)

    home_vals = pd.to_numeric(home_score, errors="coerce")
    away_vals = pd.to_numeric(away_score, errors="coerce")

    return pd.Series(
        np.where(home_flag, home_vals, away_vals),
        index=is_home.index,
        dtype="float64",
    )

def adapt_sportsdataverse_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        pbp,
        SDV_REQUIRED_COLUMNS,
        "native SportsDataverse CFB PBP",
    )

    out = pd.DataFrame(index=pbp.index)

    posteam, defteam = _native_team_names(pbp)

    out["season"] = pd.to_numeric(pbp["season"], errors="coerce")
    out["week"] = pd.to_numeric(pbp["week"], errors="coerce")
    out["game_id"] = pbp["game_id"]
    out["espn_sequence_number"] = pd.to_numeric(
        pbp["sequenceNumber"],
        errors="coerce",
    )

    out["posteam"] = posteam
    out["defteam"] = defteam

    out["epa"] = pd.to_numeric(pbp["EPA"], errors="coerce")
    out["success"] = pd.to_numeric(pbp["EPA_success"], errors="coerce")
    out["yards_gained"] = pd.to_numeric(pbp["statYardage"], errors="coerce")
    out["down"] = pd.to_numeric(pbp["down"], errors="coerce")
    out["yardline_100"] = pd.to_numeric(
        pbp["start.yardsToEndzone"],
        errors="coerce",
    )

    out["drive"] = pbp["drive.id"]
    out["first_down"] = pd.to_numeric(
        pbp["first_down_created"],
        errors="coerce",
    )

    out["touchdown"] = pd.to_numeric(pbp["touchdown"], errors="coerce")
    out["pass_touchdown"] = pd.to_numeric(pbp["pass_td"], errors="coerce")
    out["rush_touchdown"] = pd.to_numeric(pbp["rush_td"], errors="coerce")
    out["scrimmage_play"] = pbp["scrimmage_play"].fillna(False).astype(bool)

    out["posteam_score"] = _offense_score_from_is_home(
        pbp["is_home"],
        pbp["start.homeScore"],
        pbp["start.awayScore"],
    )
    out["posteam_score_post"] = _offense_score_from_is_home(
        pbp["is_home"],
        pbp["end.homeScore"],
        pbp["end.awayScore"],
    )

    offense_score = pbp["offense_score_play"].fillna(False).astype(bool)
    defense_score = pbp["defense_score_play"].fillna(False).astype(bool)
    touchdown = pbp["touchdown"].fillna(False).astype(bool)

    out["td_team"] = pd.NA
    out.loc[touchdown & offense_score, "td_team"] = out.loc[
        touchdown & offense_score, "posteam"
    ]
    out.loc[touchdown & defense_score, "td_team"] = out.loc[
        touchdown & defense_score, "defteam"
    ]

    return out


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
            "scrimmage_play",
        ],
        "scrimmage-play team stats",
    )

    mask = (
        pbp["season"].notna()
        & pbp["week"].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
        & pbp["epa"].notna()
        & pbp["scrimmage_play"].eq(True)
    )

    return pbp.loc[mask].copy()


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


def build_third_down_stats(valid_plays: pd.DataFrame) -> pd.DataFrame:
    third = valid_plays[
        valid_plays["posteam"].notna()
        & valid_plays["down"].eq(3)
    ].copy()

    if third.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "team",
                "third_down_conversion_rate",
            ]
        )

    third["third_down_conversion_flag"] = np.where(
        third["first_down"].eq(1),
        1.0,
        0.0,
    )

    return (
        third.groupby(["season", "week", "posteam"], dropna=False)
        .agg(third_down_conversion_rate=("third_down_conversion_flag", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )


def build_drive_points_stats(
    pbp: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        pbp,
        [
            "season",
            "week",
            "game_id",
            "drive",
            "posteam",
            "defteam",
            "posteam_score",
            "posteam_score_post",
            "espn_sequence_number",
        ],
        "points per drive",
    )

    drives = pbp[
        pbp["season"].notna()
        & pbp["week"].notna()
        & pbp["game_id"].notna()
        & pbp["drive"].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
    ].copy()

    if drives.empty:
        empty_off = pd.DataFrame(
            columns=["season", "week", "team", "points_per_drive"]
        )
        empty_def = pd.DataFrame(
            columns=["season", "week", "team", "points_per_drive_allowed"]
        )
        return empty_off, empty_def

    drives = drives.sort_values(
        ["season", "week", "game_id", "drive", "espn_sequence_number"]
    )

    drive_keys = [
        "season",
        "week",
        "game_id",
        "drive",
        "posteam",
        "defteam",
    ]

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

    same_team = (
        df["td_team"].fillna("").astype(str)
        == df["posteam"].fillna("").astype(str)
    )

    df["offensive_touchdown_flag"] = np.where(
        df["touchdown"].eq(1)
        & df["posteam"].notna()
        & same_team,
        1.0,
        0.0,
    )
    return df


def build_red_zone_stats(
    pbp: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        pbp,
        [
            "season",
            "week",
            "game_id",
            "drive",
            "posteam",
            "defteam",
            "yardline_100",
            "touchdown",
            "td_team",
        ],
        "red-zone touchdown rate",
    )

    df = pbp[
        pbp["season"].notna()
        & pbp["week"].notna()
        & pbp["game_id"].notna()
        & pbp["drive"].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
    ].copy()

    if df.empty:
        empty_off = pd.DataFrame(
            columns=["season", "week", "team", "red_zone_td_rate"]
        )
        empty_def = pd.DataFrame(
            columns=["season", "week", "team", "red_zone_td_rate_allowed"]
        )
        return empty_off, empty_def

    df = add_offensive_touchdown_flag(df)

    drive_keys = [
        "season",
        "week",
        "game_id",
        "drive",
        "posteam",
        "defteam",
    ]

    red_zone_trips = (
        df[df["yardline_100"].between(0, 20, inclusive="both")]
        [drive_keys]
        .drop_duplicates()
    )

    if red_zone_trips.empty:
        empty_off = pd.DataFrame(
            columns=["season", "week", "team", "red_zone_td_rate"]
        )
        empty_def = pd.DataFrame(
            columns=["season", "week", "team", "red_zone_td_rate_allowed"]
        )
        return empty_off, empty_def

    td_by_drive = (
        df.groupby(drive_keys, dropna=False)
        .agg(red_zone_drive_td=("offensive_touchdown_flag", "max"))
        .reset_index()
    )

    trips = red_zone_trips.merge(
        td_by_drive,
        on=drive_keys,
        how="left",
    )
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
            result = result.merge(
                frame,
                on=["season", "week", "team"],
                how="outer",
            )

    if result is None:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for col in OUTPUT_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan

    result = result[OUTPUT_COLUMNS]
    result = result.sort_values(
        ["season", "week", "team"]
    ).reset_index(drop=True)

    return result


def build_team_stats(native_pbp: pd.DataFrame) -> pd.DataFrame:
    if native_pbp.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    pbp = adapt_sportsdataverse_pbp(native_pbp)

    valid_plays = build_valid_scrimmage_plays(pbp)

    if valid_plays.empty:
        raise ValueError(
            "No valid scrimmage plays after SportsDataverse adaptation. "
            f"adapted_rows={len(pbp)}"
        )

    offense_stats = build_offense_stats(valid_plays)
    defense_stats = build_defense_stats(valid_plays)
    third_down_stats = build_third_down_stats(valid_plays)
    off_points, def_points = build_drive_points_stats(pbp)
    off_rz, def_rz = build_red_zone_stats(pbp)

    return merge_stat_frames(
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


def run() -> int:
    args = parse_args()
    season = get_season(args.season)

    _, pbp_dir, output_dir, error_dir = resolve_paths()

    pbp_path = (
        Path(args.pbp_path).expanduser()
        if args.pbp_path
        else pbp_dir / f"{season}_pbp.parquet"
    )

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
        log.write_line("source=sportsdataverse_native_cfb_pbp")
        log.write_line(f"input={pbp_path}")
        log.write_line(f"output={output_path}")
        log.write_line(f"log={log_path}")

        pbp = read_pbp(pbp_path)

        if pbp.empty:
            write_empty_output(output_path)
            log.write_line("pbp_rows=0")
            log.write_line("output_rows=0")
            log.write_line(f"output_columns={len(OUTPUT_COLUMNS)}")
            log.write_line("status=empty_pbp_written")
            log.write_line("=" * 80)
            return 0

        require_columns(
            pbp,
            SDV_REQUIRED_COLUMNS,
            "native SportsDataverse CFB PBP",
        )

        team_stats = build_team_stats(pbp)

        if len(pbp) > 0 and team_stats.empty:
            raise ValueError(
                "PBP contained rows but team-stat output was empty. "
                "Refusing to report success."
            )

        team_stats.to_csv(output_path, index=False)

        log.write_line(f"pbp_rows={len(pbp)}")
        log.write_line(f"pbp_columns={len(pbp.columns)}")
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

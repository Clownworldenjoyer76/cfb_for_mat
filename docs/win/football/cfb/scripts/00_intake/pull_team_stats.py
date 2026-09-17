#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


PBP_DIR = CFB_ROOT / "00_intake" / "pbp"
OUTPUT_DIR = CFB_ROOT / "00_intake" / "team_stats"
REPORT_ROOT = CFB_ROOT / "errors"

SCRIPT_VERSION = "cfb-team-stats-v3-denominator-counts-2026-09-17"

METRIC_COLUMNS = [
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

METRIC_COUNT_COLUMNS = {
    metric: f"{metric}_count"
    for metric in METRIC_COLUMNS
}

COUNT_COLUMNS = [
    METRIC_COUNT_COLUMNS[metric]
    for metric in METRIC_COLUMNS
]

OUTPUT_COLUMNS = [
    "season",
    "week",
    "team",
    *METRIC_COLUMNS,
    *COUNT_COLUMNS,
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
    raw = (
        str(cli_season).strip()
        if cli_season is not None
        else str(os.getenv("CFB_SEASON", "")).strip()
    )

    if not raw:
        raise ValueError(
            "Missing season. Pass --season or set CFB_SEASON."
        )

    try:
        season = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Season must be an integer; found {raw!r}"
        ) from exc

    if season < 2000 or season > 2100:
        raise ValueError(
            f"Season is outside the supported range: {season}"
        )

    return str(season)


def write_team_stats_atomic(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        frame.to_csv(
            temp_path,
            index=False,
        )

        os.replace(
            temp_path,
            output_path,
        )

    finally:
        try:
            temp_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass

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


def validate_pbp_season(
    pbp: pd.DataFrame,
    requested_season: str,
) -> None:
    try:
        expected_season = int(str(requested_season).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Configured season must be an integer; found {requested_season!r}"
        ) from exc

    season_values = pd.to_numeric(
        pbp["season"],
        errors="coerce",
    )

    invalid_mask = season_values.isna() | season_values.mod(1).ne(0)

    if invalid_mask.any():
        examples = (
            pbp.loc[invalid_mask, "season"]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            "PBP contains blank, non-numeric, or non-integer season values: "
            f"{examples}"
        )

    observed_seasons = sorted(
        season_values.astype(int).unique().tolist()
    )

    if observed_seasons != [expected_season]:
        raise ValueError(
            "PBP season does not match requested output season. "
            f"requested_season={expected_season}, "
            f"observed_seasons={observed_seasons}"
        )


def validate_pbp_integrity(
    pbp: pd.DataFrame,
) -> None:
    week_values = pd.to_numeric(
        pbp["week"],
        errors="coerce",
    )

    invalid_week = (
        week_values.isna()
        | week_values.mod(1).ne(0)
        | week_values.le(0)
    )

    if invalid_week.any():
        examples = (
            pbp.loc[invalid_week, "week"]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            "PBP contains blank, non-numeric, non-integer, "
            f"or non-positive week values: {examples}"
        )

    game_ids = (
        pbp["game_id"]
        .astype("string")
        .str.strip()
    )

    invalid_game_id = (
        game_ids.isna()
        | game_ids.fillna("").eq("")
    )

    if invalid_game_id.any():
        raise ValueError(
            "PBP contains blank game_id values"
        )

    sequence_numbers = pd.to_numeric(
        pbp["sequenceNumber"],
        errors="coerce",
    )

    if sequence_numbers.isna().any():
        examples = (
            pbp.loc[
                sequence_numbers.isna(),
                "sequenceNumber",
            ]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            "PBP contains blank or non-numeric "
            f"sequenceNumber values: {examples}"
        )


def _clean_team_series(
    series: pd.Series,
) -> pd.Series:
    result = (
        series
        .astype("string")
        .str.strip()
    )

    normalized = result.str.casefold()

    invalid = (
        result.isna()
        | normalized.isin(
            {
                "",
                "nan",
                "none",
                "null",
                "<na>",
            }
        )
    )

    return result.mask(
        invalid,
        pd.NA,
    ).astype("object")


def _coerce_boolean_series(
    series: pd.Series,
    label: str,
) -> pd.Series:
    def convert(value: object) -> bool:
        if value is None or pd.isna(value):
            return False

        if isinstance(
            value,
            (bool, np.bool_),
        ):
            return bool(value)

        if isinstance(
            value,
            (int, np.integer),
        ):
            if int(value) in {0, 1}:
                return bool(int(value))

        if isinstance(
            value,
            (float, np.floating),
        ):
            number = float(value)

            if np.isfinite(number) and number in {0.0, 1.0}:
                return bool(int(number))

        text = str(value).strip().casefold()

        if text in {
            "true",
            "1",
            "1.0",
            "yes",
            "y",
        }:
            return True

        if text in {
            "false",
            "0",
            "0.0",
            "no",
            "n",
            "",
            "nan",
            "none",
            "null",
            "<na>",
        }:
            return False

        raise ValueError(
            f"{label} contains unsupported boolean value {value!r}"
        )

    return series.map(
        convert
    ).astype(bool)


def _native_team_names(
    pbp: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    posteam = _clean_team_series(
        pbp["pos_team"]
    )

    defteam = _clean_team_series(
        pbp["def_pos_team"]
    )

    return posteam, defteam

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
    home_flag = _coerce_boolean_series(
        is_home,
        "is_home",
    )

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

    touchdown_flags = _coerce_boolean_series(
        pbp["touchdown"],
        "touchdown",
    )
    out["touchdown"] = touchdown_flags.astype(float)
    out["pass_touchdown"] = pd.to_numeric(pbp["pass_td"], errors="coerce")
    out["rush_touchdown"] = pd.to_numeric(pbp["rush_td"], errors="coerce")
    out["scrimmage_play"] = _coerce_boolean_series(
        pbp["scrimmage_play"],
        "scrimmage_play",
    )

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

    offense_score = _coerce_boolean_series(
        pbp["offense_score_play"],
        "offense_score_play",
    )
    defense_score = _coerce_boolean_series(
        pbp["defense_score_play"],
        "defense_score_play",
    )
    touchdown = touchdown_flags

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



def build_metric_counts(
    pbp: pd.DataFrame,
    valid_plays: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "season",
        "week",
        "team",
    ]

    frames: list[pd.DataFrame] = []

    offense = (
        valid_plays.groupby(
            [
                "season",
                "week",
                "posteam",
            ],
            dropna=False,
        )
        .agg(
            off_epa_per_play_count=(
                "epa",
                "count",
            ),
            off_success_rate_count=(
                "success",
                "count",
            ),
            yards_per_play_count=(
                "yards_gained",
                "count",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "posteam": "team"
            }
        )
    )
    frames.append(
        offense
    )

    defense = (
        valid_plays.groupby(
            [
                "season",
                "week",
                "defteam",
            ],
            dropna=False,
        )
        .agg(
            def_epa_per_play_count=(
                "epa",
                "count",
            ),
            def_success_rate_count=(
                "success",
                "count",
            ),
            yards_per_play_allowed_count=(
                "yards_gained",
                "count",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "defteam": "team"
            }
        )
    )
    frames.append(
        defense
    )

    early = valid_plays[
        valid_plays[
            "down"
        ].isin(
            [1, 2]
        )
    ].copy()

    if not early.empty:
        early_counts = (
            early.groupby(
                [
                    "season",
                    "week",
                    "posteam",
                ],
                dropna=False,
            )
            .agg(
                early_down_epa_count=(
                    "epa",
                    "count",
                )
            )
            .reset_index()
            .rename(
                columns={
                    "posteam": "team"
                }
            )
        )
        frames.append(
            early_counts
        )

    third = valid_plays[
        valid_plays[
            "posteam"
        ].notna()
        & valid_plays[
            "down"
        ].eq(3)
    ].copy()

    if not third.empty:
        third_counts = (
            third.groupby(
                [
                    "season",
                    "week",
                    "posteam",
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name=(
                    "third_down_conversion_rate_count"
                )
            )
            .rename(
                columns={
                    "posteam": "team"
                }
            )
        )
        frames.append(
            third_counts
        )

    drives = pbp[
        pbp[
            "season"
        ].notna()
        & pbp[
            "week"
        ].notna()
        & pbp[
            "game_id"
        ].notna()
        & pbp[
            "drive"
        ].notna()
        & pbp[
            "posteam"
        ].notna()
        & pbp[
            "defteam"
        ].notna()
    ].copy()

    if not drives.empty:
        drives = drives.sort_values(
            [
                "season",
                "week",
                "game_id",
                "drive",
                "espn_sequence_number",
            ]
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
            drives.groupby(
                drive_keys,
                dropna=False,
            )
            .agg(
                drive_start_score=(
                    "posteam_score",
                    "first",
                ),
                drive_end_score=(
                    "posteam_score_post",
                    "last",
                ),
            )
            .reset_index()
        )

        drive_scores[
            "drive_points"
        ] = (
            drive_scores[
                "drive_end_score"
            ]
            - drive_scores[
                "drive_start_score"
            ]
        )

        drive_scores.loc[
            drive_scores[
                "drive_points"
            ] < 0,
            "drive_points",
        ] = 0

        off_drives = (
            drive_scores.groupby(
                [
                    "season",
                    "week",
                    "posteam",
                ],
                dropna=False,
            )
            .agg(
                points_per_drive_count=(
                    "drive_points",
                    "count",
                )
            )
            .reset_index()
            .rename(
                columns={
                    "posteam": "team"
                }
            )
        )

        def_drives = (
            drive_scores.groupby(
                [
                    "season",
                    "week",
                    "defteam",
                ],
                dropna=False,
            )
            .agg(
                points_per_drive_allowed_count=(
                    "drive_points",
                    "count",
                )
            )
            .reset_index()
            .rename(
                columns={
                    "defteam": "team"
                }
            )
        )

        frames.extend(
            [
                off_drives,
                def_drives,
            ]
        )

    rz = pbp[
        pbp[
            "season"
        ].notna()
        & pbp[
            "week"
        ].notna()
        & pbp[
            "game_id"
        ].notna()
        & pbp[
            "drive"
        ].notna()
        & pbp[
            "posteam"
        ].notna()
        & pbp[
            "defteam"
        ].notna()
    ].copy()

    if not rz.empty:
        drive_keys = [
            "season",
            "week",
            "game_id",
            "drive",
            "posteam",
            "defteam",
        ]

        red_zone_trips = (
            rz[
                rz[
                    "yardline_100"
                ].between(
                    0,
                    20,
                    inclusive="both",
                )
            ][
                drive_keys
            ]
            .drop_duplicates()
        )

        if not red_zone_trips.empty:
            off_rz = (
                red_zone_trips.groupby(
                    [
                        "season",
                        "week",
                        "posteam",
                    ],
                    dropna=False,
                )
                .size()
                .reset_index(
                    name=(
                        "red_zone_td_rate_count"
                    )
                )
                .rename(
                    columns={
                        "posteam": "team"
                    }
                )
            )

            def_rz = (
                red_zone_trips.groupby(
                    [
                        "season",
                        "week",
                        "defteam",
                    ],
                    dropna=False,
                )
                .size()
                .reset_index(
                    name=(
                        "red_zone_td_rate_allowed_count"
                    )
                )
                .rename(
                    columns={
                        "defteam": "team"
                    }
                )
            )

            frames.extend(
                [
                    off_rz,
                    def_rz,
                ]
            )

    result: pd.DataFrame | None = None

    for frame in frames:
        if frame.empty:
            continue

        if result is None:
            result = frame.copy()
        else:
            result = result.merge(
                frame,
                on=keys,
                how="outer",
            )

    if result is None:
        return pd.DataFrame(
            columns=[
                *keys,
                *COUNT_COLUMNS,
            ]
        )

    for column in COUNT_COLUMNS:
        if column not in result.columns:
            result[
                column
            ] = np.nan

    return result[
        [
            *keys,
            *COUNT_COLUMNS,
        ]
    ]


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


def validate_team_stats_output(
    team_stats: pd.DataFrame,
    requested_season: str,
) -> None:
    if team_stats.empty:
        raise ValueError(
            "Team-stat output is empty"
        )

    if list(team_stats.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            "Team-stat output columns do not match "
            "the required output schema"
        )

    expected_season = int(
        str(requested_season).strip()
    )

    season_values = pd.to_numeric(
        team_stats["season"],
        errors="coerce",
    )

    invalid_season = (
        season_values.isna()
        | season_values.mod(1).ne(0)
    )

    if invalid_season.any():
        raise ValueError(
            "Team-stat output contains invalid season values"
        )

    observed_seasons = sorted(
        season_values.astype(int).unique().tolist()
    )

    if observed_seasons != [expected_season]:
        raise ValueError(
            "Team-stat output season mismatch. "
            f"requested_season={expected_season}, "
            f"observed_seasons={observed_seasons}"
        )

    week_values = pd.to_numeric(
        team_stats["week"],
        errors="coerce",
    )

    invalid_week = (
        week_values.isna()
        | week_values.mod(1).ne(0)
        | week_values.le(0)
    )

    if invalid_week.any():
        raise ValueError(
            "Team-stat output contains invalid week values"
        )

    team_values = (
        team_stats["team"]
        .astype("string")
        .str.strip()
    )

    invalid_team = (
        team_values.isna()
        | team_values.fillna("").eq("")
    )

    if invalid_team.any():
        raise ValueError(
            "Team-stat output contains blank team values"
        )

    key_frame = pd.DataFrame(
        {
            "season": season_values.astype(int),
            "week": week_values.astype(int),
            "team": team_values,
        }
    )

    duplicate_mask = key_frame.duplicated(
        ["season", "week", "team"],
        keep=False,
    )

    if duplicate_mask.any():
        examples = (
            key_frame.loc[
                duplicate_mask,
                ["season", "week", "team"],
            ]
            .drop_duplicates()
            .head(10)
            .astype(str)
            .agg("/".join, axis=1)
            .tolist()
        )

        raise ValueError(
            "Team-stat output contains duplicate "
            f"(season, week, team) rows: {examples}"
        )

    metric_columns = METRIC_COLUMNS

    for column in metric_columns:
        numeric = pd.to_numeric(
            team_stats[column],
            errors="coerce",
        )

        invalid_numeric = (
            team_stats[column].notna()
            & numeric.isna()
        )

        if invalid_numeric.any():
            raise ValueError(
                f"Team-stat output contains non-numeric {column}"
            )

        values = numeric.dropna().to_numpy(
            dtype=float
        )

        if (
            values.size
            and not np.isfinite(values).all()
        ):
            raise ValueError(
                f"Team-stat output contains non-finite {column}"
            )


    for (
        metric,
        count_column,
    ) in METRIC_COUNT_COLUMNS.items():
        counts = pd.to_numeric(
            team_stats[
                count_column
            ],
            errors="coerce",
        )

        invalid = (
            counts.notna()
            & (
                counts.lt(0)
                | counts.mod(1).ne(0)
            )
        )

        if invalid.any():
            raise ValueError(
                "Team-stat output contains invalid "
                f"{count_column}"
            )

        metric_values = pd.to_numeric(
            team_stats[
                metric
            ],
            errors="coerce",
        )

        metric_present = (
            metric_values.notna()
        )

        positive_count = (
            counts.fillna(
                0.0
            ).gt(0)
        )

        if (
            metric_present
            != positive_count
        ).any():
            raise ValueError(
                "Team-stat metric/count contract "
                f"failed for {metric}"
            )

    rate_columns = [
        "off_success_rate",
        "def_success_rate",
        "red_zone_td_rate",
        "red_zone_td_rate_allowed",
        "third_down_conversion_rate",
    ]

    for column in rate_columns:
        numeric = pd.to_numeric(
            team_stats[column],
            errors="coerce",
        ).dropna()

        if (
            numeric.lt(0).any()
            or numeric.gt(1).any()
        ):
            raise ValueError(
                f"Team-stat output {column} is outside [0, 1]"
            )

    for column in [
        "points_per_drive",
        "points_per_drive_allowed",
    ]:
        numeric = pd.to_numeric(
            team_stats[column],
            errors="coerce",
        ).dropna()

        if numeric.lt(0).any():
            raise ValueError(
                f"Team-stat output {column} contains negative values"
            )

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

    metric_counts = build_metric_counts(
        pbp,
        valid_plays,
    )

    return merge_stat_frames(
        [
            offense_stats,
            defense_stats,
            off_points,
            def_points,
            off_rz,
            def_rz,
            third_down_stats,
            metric_counts,
        ]
    )


def run() -> int:
    args = parse_args()
    season = get_season(
        args.season
    )

    pbp_path = (
        Path(args.pbp_path)
        .expanduser()
        .resolve()
        if args.pbp_path
        else PBP_DIR / f"{season}_pbp.parquet"
    )

    output_path = (
        OUTPUT_DIR
        / f"{season}_team_stats.csv"
    )

    with PipelineReporter(
        script=__file__,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        season=int(season),
        extra_context={
            "script_version": SCRIPT_VERSION,
            "source": "sportsdataverse_native_cfb_pbp",
        },
    ) as report:
        report.add_input(
            pbp_path
        )
        report.add_output(
            output_path
        )

        report.update_details(
            {
                "explicit_pbp_path": bool(
                    args.pbp_path
                ),
                "output_modified": False,
            }
        )

        pbp = read_pbp(
            pbp_path
        )

        report.set_rows(
            rows_in=len(pbp),
        )

        report.set_detail(
            "pbp_columns",
            len(pbp.columns),
        )

        if pbp.empty:
            raise RuntimeError(
                "PBP input is empty; refusing to overwrite "
                f"team-stat output: {output_path}"
            )

        require_columns(
            pbp,
            SDV_REQUIRED_COLUMNS,
            "native SportsDataverse CFB PBP",
        )

        validate_pbp_season(
            pbp,
            season,
        )

        validate_pbp_integrity(
            pbp,
        )

        week_values = pd.to_numeric(
            pbp["week"],
            errors="raise",
        ).astype(int)

        game_ids = (
            pbp["game_id"]
            .astype("string")
            .str.strip()
        )

        report.update_details(
            {
                "games_in_input": int(
                    game_ids.nunique()
                ),
                "weeks_in_input": sorted(
                    week_values.unique().tolist()
                ),
            }
        )

        team_stats = build_team_stats(
            pbp
        )

        validate_team_stats_output(
            team_stats,
            season,
        )

        write_team_stats_atomic(
            team_stats,
            output_path,
        )

        output_weeks = sorted(
            pd.to_numeric(
                team_stats["week"],
                errors="raise",
            )
            .astype(int)
            .unique()
            .tolist()
        )

        report.set_rows(
            rows_out=len(team_stats),
        )

        report.update_details(
            {
                "output_modified": True,
                "output_columns": len(
                    team_stats.columns
                ),
                "team_week_rows": len(
                    team_stats
                ),
                "teams_in_output": int(
                    team_stats["team"].nunique()
                ),
                "weeks_in_output": output_weeks,
            }
        )

        print("pull_team_stats.py completed")
        print(f"season={season}")
        print(f"pbp_rows={len(pbp)}")
        print(
            f"games_in_input="
            f"{game_ids.nunique()}"
        )
        print(
            f"team_stat_rows="
            f"{len(team_stats)}"
        )
        print(f"output={output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(
        run()
    )
#!/usr/bin/env python3
# docs/win/football/nfl/scripts/00_intake/pull_pbp.py
#
# Pulls nflverse play-by-play data for one NFL season.
#
# Output:
#   docs/win/football/nfl/00_intake/pbp/{season}_pbp.csv.gz
#
# Intended downstream calculations:
#   EPA/play
#   Success rate
#   Early-down EPA
#   Third-down conversion rate
#   Red-zone TD rate
#   Points/drive
#   Yards/play
#   QB EPA/play
#   QB CPOE
#   Sacks
#   Turnovers

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE_DIR = Path("docs/win/football/nfl")
SETTINGS_FILE = BASE_DIR / "config/settings.yaml"
PBP_DIR = BASE_DIR / "00_intake/pbp"
ERROR_DIR = BASE_DIR / "errors/00_intake"
LOG_FILE = ERROR_DIR / "pull_pbp.txt"


# ─────────────────────────────────────────────
# COLUMNS NEEDED BY DOWNSTREAM FEATURE STEPS
# ─────────────────────────────────────────────

FEATURE_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "old_game_id",
    "play_id",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "side_of_field",
    "yardline_100",
    "game_date",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "qtr",
    "down",
    "ydstogo",
    "ydsnet",
    "desc",
    "play_type",
    "yards_gained",
    "epa",
    "success",
    "wp",
    "wpa",
    "cp",
    "cpoe",
    "qb_epa",
    "pass",
    "rush",
    "sack",
    "interception",
    "fumble",
    "fumble_lost",
    "turnover",
    "touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "first_down",
    "third_down_converted",
    "third_down_failed",
    "fourth_down_converted",
    "fourth_down_failed",
    "series",
    "series_success",
    "drive",
    "fixed_drive",
    "fixed_drive_result",
    "drive_real_start_time",
    "drive_play_count",
    "drive_time_of_possession",
    "drive_first_downs",
    "drive_inside20",
    "drive_ended_with_score",
    "drive_quarter_start",
    "drive_quarter_end",
    "drive_yards_penalized",
    "posteam_score",
    "defteam_score",
    "score_differential",
    "total_home_score",
    "total_away_score",
    "passer_player_id",
    "passer_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "receiver_player_id",
    "receiver_player_name",
]


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def ensure_dirs() -> None:
    PBP_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    ensure_dirs()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{now_stamp()}] {message}\n")


# ─────────────────────────────────────────────
# SETTINGS / CLI
# ─────────────────────────────────────────────

def read_settings() -> dict[str, Any]:
    if not SETTINGS_FILE.exists():
        return {}
    if yaml is None:
        return {}
    with SETTINGS_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def get_season(args: argparse.Namespace) -> int:
    if args.season is not None:
        return int(args.season)

    settings = read_settings()
    season = settings.get("season")
    if season not in (None, ""):
        return int(season)

    raise ValueError(
        "Missing season. Provide --season or set season in "
        "docs/win/football/nfl/config/settings.yaml."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull nflverse play-by-play data for one NFL season."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="NFL season to pull. If omitted, reads config/settings.yaml season.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "nflreadpy", "nfl_data_py"],
        default="auto",
        help="Data loader to use. Default: auto.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────

def _to_pandas(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if hasattr(data, "to_pandas"):
        return data.to_pandas()

    return pd.DataFrame(data)


def load_with_nflreadpy(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    try:
        data = nfl.load_pbp(seasons=[season])
    except TypeError:
        data = nfl.load_pbp([season])

    return _to_pandas(data)


def load_with_nfl_data_py(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl

    try:
        data = nfl.import_pbp_data([season], columns=None, downcast=True, cache=False)
    except TypeError:
        data = nfl.import_pbp_data([season], downcast=True, cache=False)

    return _to_pandas(data)


def load_pbp(season: int, source: str) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []

    if source in {"auto", "nflreadpy"}:
        try:
            return load_with_nflreadpy(season), "nflreadpy"
        except Exception as exc:
            errors.append(f"nflreadpy failed: {repr(exc)}")
            if source == "nflreadpy":
                raise RuntimeError("; ".join(errors)) from exc

    if source in {"auto", "nfl_data_py"}:
        try:
            return load_with_nfl_data_py(season), "nfl_data_py"
        except Exception as exc:
            errors.append(f"nfl_data_py failed: {repr(exc)}")
            if source == "nfl_data_py":
                raise RuntimeError("; ".join(errors)) from exc

    raise RuntimeError("; ".join(errors) if errors else "No PBP source attempted.")


# ─────────────────────────────────────────────
# NORMALIZATION / OUTPUT
# ─────────────────────────────────────────────

def add_derived_spec_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "turnover" not in df.columns:
        if "interception" in df.columns and "fumble_lost" in df.columns:
            interception = pd.to_numeric(df["interception"], errors="coerce").fillna(0)
            fumble_lost = pd.to_numeric(df["fumble_lost"], errors="coerce").fillna(0)
            df["turnover"] = ((interception == 1) | (fumble_lost == 1)).astype(int)

    return df


def clean_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = add_derived_spec_columns(df)

    sort_cols = [col for col in ["game_id", "play_id"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable")

    ordered = [col for col in FEATURE_COLUMNS if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]
    df = df[ordered + remaining]

    return df


def missing_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in FEATURE_COLUMNS if col not in df.columns]


def write_pbp(df: pd.DataFrame, season: int) -> Path:
    output_file = PBP_DIR / f"{season}_pbp.csv.gz"
    df.to_csv(output_file, index=False, compression="gzip")
    return output_file


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> int:
    ensure_dirs()
    args = parse_args()

    try:
        season = get_season(args)
        log("=" * 80)
        log(f"pull_pbp.py started | season={season} | source={args.source}")

        df, used_source = load_pbp(season=season, source=args.source)
        df = clean_for_csv(df)

        output_file = write_pbp(df=df, season=season)
        missing = missing_feature_columns(df)

        log(f"source_used={used_source}")
        log(f"rows={len(df)} columns={len(df.columns)}")
        log(f"output={output_file}")
        if missing:
            log("missing_feature_columns=" + ",".join(missing))
        else:
            log("missing_feature_columns=none")
        log("pull_pbp.py completed")

        print("nfl pull_pbp completed")
        print(f"season: {season}")
        print(f"source_used: {used_source}")
        print(f"rows: {len(df)}")
        print(f"columns: {len(df.columns)}")
        print(f"output: {output_file}")
        if missing:
            print("missing_feature_columns: " + ",".join(missing))
        else:
            print("missing_feature_columns: none")

        return 0

    except Exception as exc:
        log(f"ERROR: {repr(exc)}")
        log(traceback.format_exc())
        print("nfl pull_pbp failed", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

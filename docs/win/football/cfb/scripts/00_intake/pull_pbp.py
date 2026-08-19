#!/usr/bin/env python3
# docs/win/football/cfb/scripts/00_intake/pull_pbp.py
#
# Builds native SportsDataverse/cfbfastR college-football play-by-play from the
# local season schedule.
#
# Processing source:
#   sportsdataverse.cfb.CFBPlayProcess
#
# Per-game processing:
#   proc = CFBPlayProcess(gameId=game_id)
#   proc.espn_cfb_pbp()
#   result = proc.run_processing_pipeline()
#   plays = result["plays"]
#
# Schedule input:
#   docs/win/football/cfb/00_intake/schedule/{season}_schedule.csv
#
# Output:
#   docs/win/football/cfb/00_intake/pbp/{season}_pbp.parquet
#
# Design:
#   * preserves the native SportsDataverse play schema; no column renaming
#   * uses SportsDataverse EPA/WP/WPA/CP/CPOE/QBR/etc.
#   * only stores completed games
#   * incremental by default: already-stored completed games are not reprocessed
#   * --game-id explicitly refreshes/replaces the requested game(s)
#   * --refresh reprocesses all eligible games and replaces their stored rows
#   * writes Parquet atomically
#
# The historical 2021-2025 PBP files use the same native schema family.

from __future__ import annotations

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from sportsdataverse.cfb import CFBPlayProcess
except ImportError as exc:  # pragma: no cover
    CFBPlayProcess = None
    SPORTSDATAVERSE_IMPORT_ERROR: Exception | None = exc
else:
    SPORTSDATAVERSE_IMPORT_ERROR = None


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parents[2]

SETTINGS_FILE = BASE_DIR / "config" / "settings.yaml"
SCHEDULE_DIR = BASE_DIR / "00_intake" / "schedule"
PBP_DIR = BASE_DIR / "00_intake" / "pbp"
ERROR_DIR = BASE_DIR / "errors" / "00_intake"
LOG_FILE = ERROR_DIR / "pull_pbp.txt"

EASTERN = ZoneInfo("America/New_York")

# Keep concurrency conservative. SportsDataverse itself performs ESPN network
# work and XGBoost model inference inside each game process.
DEFAULT_WORKERS = 3

# These are not a replacement schema. They are only invariants required by the
# downstream team-stat pipeline and by safe incremental season assembly.
REQUIRED_NATIVE_COLUMNS = [
    "season",
    "week",
    "game_id",
    "id",
    "sequenceNumber",
    "status_type_completed",
    "homeTeamId",
    "awayTeamId",
    "homeTeamName",
    "awayTeamName",
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


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def now_stamp() -> str:
    return datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S %Z")


def ensure_dirs() -> None:
    PBP_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)


def reset_log() -> None:
    ensure_dirs()
    LOG_FILE.write_text("", encoding="utf-8")


def log(message: str) -> None:
    ensure_dirs()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{now_stamp()}] {message.rstrip()}\n")


# ─────────────────────────────────────────────
# SETTINGS / CLI
# ─────────────────────────────────────────────

def read_settings() -> dict[str, Any]:
    if not SETTINGS_FILE.exists() or yaml is None:
        return {}

    with SETTINGS_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data if isinstance(data, dict) else {}


def get_season(args: argparse.Namespace) -> int:
    if args.season is not None:
        return int(args.season)

    settings = read_settings()
    season = settings.get("season")

    if season not in (None, ""):
        return int(season)

    env_season = os.getenv("CFB_SEASON")
    if env_season:
        return int(env_season)

    raise ValueError(
        "Missing season. Provide --season, set season in "
        "docs/win/football/cfb/config/settings.yaml, or set CFB_SEASON."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build native SportsDataverse CFB PBP from the local season schedule."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="CFB season to process.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent game processes. Default: {DEFAULT_WORKERS}.",
    )

    parser.add_argument(
        "--game-id",
        action="append",
        default=None,
        help=(
            "Optional ESPN game id to process. Repeat for multiple games. "
            "Explicit game ids are reprocessed even if already stored."
        ),
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Reprocess all eligible games and replace their stored rows. "
            "Without this flag, the normal season run is incremental."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Process and validate games but do not modify the season Parquet. "
            "Useful for one-game smoke tests."
        ),
    )

    return parser.parse_args()


# ─────────────────────────────────────────────
# LOCAL SCHEDULE
# ─────────────────────────────────────────────

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if text.casefold() in {"", "none", "nan", "null", "<na>"}:
        return ""

    return text


def load_schedule(season: int) -> dict[str, dict[str, str]]:
    schedule_path = SCHEDULE_DIR / f"{season}_schedule.csv"

    if not schedule_path.exists():
        raise FileNotFoundError(f"Missing season schedule: {schedule_path}")

    schedule = pd.read_csv(schedule_path, dtype=str, keep_default_na=False)

    required = {
        "season",
        "season_type",
        "week",
        "game_id",
        "game_date",
        "away_team",
        "home_team",
    }

    missing = sorted(required - set(schedule.columns))
    if missing:
        raise ValueError(
            f"{schedule_path} missing required columns: {missing}"
        )

    schedule = schedule[
        schedule["season"].astype(str).str.strip().eq(str(season))
    ].copy()

    if schedule.empty:
        raise RuntimeError(
            f"No season={season} games found in {schedule_path}"
        )

    schedule["game_id"] = schedule["game_id"].astype(str).str.strip()
    schedule = schedule[schedule["game_id"].ne("")].copy()

    if schedule["game_id"].duplicated().any():
        duplicates = sorted(
            schedule.loc[
                schedule["game_id"].duplicated(keep=False),
                "game_id",
            ].unique()
        )
        raise ValueError(
            "Duplicate game_id values in schedule: "
            + ", ".join(duplicates[:20])
        )

    return {
        row["game_id"]: {
            column: clean_text(row[column])
            for column in schedule.columns
        }
        for _, row in schedule.iterrows()
    }


def parse_game_date(value: Any):
    text = clean_text(value)
    if not text:
        return None

    # Current local schedule uses YYYY-MM-DD. Accept an ISO timestamp too.
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None

    return parsed.date()


def schedule_game_is_future(row: dict[str, str]) -> bool:
    game_date = parse_game_date(row.get("game_date"))
    if game_date is None:
        return False

    return game_date > datetime.now(EASTERN).date()


# ─────────────────────────────────────────────
# SPORTSDATAVERSE PROCESSING
# ─────────────────────────────────────────────

def sportsdataverse_version() -> str:
    try:
        return package_version("sportsdataverse")
    except PackageNotFoundError:
        return "not-installed"
    except Exception:
        return "unknown"


def require_sportsdataverse() -> None:
    if CFBPlayProcess is not None:
        return

    detail = (
        f": {SPORTSDATAVERSE_IMPORT_ERROR}"
        if SPORTSDATAVERSE_IMPORT_ERROR is not None
        else ""
    )
    raise RuntimeError(
        "sportsdataverse is required. Install it with "
        "`python -m pip install --upgrade sportsdataverse pyarrow polars xgboost`"
        + detail
    )


def require_native_columns(
    df: pd.DataFrame,
    context: str,
) -> None:
    missing = [
        column
        for column in REQUIRED_NATIVE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{context} missing required native columns: {missing}"
        )


def game_is_completed(df: pd.DataFrame) -> bool:
    if "status_type_completed" not in df.columns:
        return False

    values = df["status_type_completed"].dropna()

    if values.empty:
        return False

    # Current native output is bool. This fallback also handles string/binary
    # representations without treating arbitrary non-empty strings as True.
    if pd.api.types.is_bool_dtype(values.dtype):
        return bool(values.all())

    normalized = (
        values.astype(str)
        .str.strip()
        .str.casefold()
        .map(
            {
                "true": True,
                "1": True,
                "yes": True,
                "false": False,
                "0": False,
                "no": False,
            }
        )
    )

    return bool(normalized.notna().all() and normalized.all())


def validate_processed_game(
    df: pd.DataFrame,
    game_id: int,
    season: int,
) -> None:
    if df.empty:
        raise ValueError("processor returned zero plays")

    require_native_columns(
        df,
        f"game_id={game_id}",
    )

    game_ids = pd.to_numeric(df["game_id"], errors="coerce")
    if game_ids.isna().any() or not game_ids.eq(game_id).all():
        observed = sorted(
            {
                int(value)
                for value in game_ids.dropna().unique()
            }
        )
        raise ValueError(
            f"game_id mismatch: requested={game_id} observed={observed}"
        )

    seasons = pd.to_numeric(df["season"], errors="coerce")
    if seasons.isna().any() or not seasons.eq(season).all():
        observed = sorted(
            {
                int(value)
                for value in seasons.dropna().unique()
            }
        )
        raise ValueError(
            f"season mismatch: requested={season} observed={observed}"
        )

    if df.duplicated(["game_id", "id"]).any():
        duplicate_count = int(
            df.duplicated(["game_id", "id"], keep=False).sum()
        )
        raise ValueError(
            f"duplicate native play ids within game: {duplicate_count} rows"
        )


def process_one_game(
    task: tuple[int, int],
) -> tuple[int, pd.DataFrame | None, str]:
    """
    Worker entry point. Kept at module scope so it is picklable on Windows.
    """
    game_id, season = task

    try:
        if CFBPlayProcess is None:
            return game_id, None, "sportsdataverse import unavailable"

        proc = CFBPlayProcess(gameId=game_id)
        proc.espn_cfb_pbp()
        result = proc.run_processing_pipeline()

        if not isinstance(result, dict):
            return (
                game_id,
                None,
                f"processor returned {type(result).__name__}, expected dict",
            )

        plays = result.get("plays") or []

        if not isinstance(plays, list):
            return (
                game_id,
                None,
                f"result['plays'] returned {type(plays).__name__}, expected list",
            )

        if not plays:
            return game_id, None, "no plays returned"

        df = pd.DataFrame(plays)

        validate_processed_game(
            df,
            game_id=game_id,
            season=season,
        )

        if not game_is_completed(df):
            return game_id, None, "game not completed"

        return game_id, df, ""

    except Exception as exc:
        return (
            game_id,
            None,
            f"{type(exc).__name__}: {exc}",
        )


def process_games(
    game_ids: list[int],
    season: int,
    workers: int,
) -> tuple[list[pd.DataFrame], list[tuple[int, str]]]:
    if not game_ids:
        return [], []

    frames: list[pd.DataFrame] = []
    skipped: list[tuple[int, str]] = []

    tasks = [(game_id, season) for game_id in game_ids]

    # Direct execution is useful for one-game smoke tests and avoids Windows
    # process-spawn overhead when concurrency cannot help.
    if workers == 1 or len(tasks) == 1:
        for index, task in enumerate(tasks, start=1):
            game_id, frame, reason = process_one_game(task)

            if frame is not None and not frame.empty:
                frames.append(frame)
                print(
                    f"game={game_id} plays={len(frame)} "
                    f"columns={len(frame.columns)} "
                    f"completed={index}/{len(tasks)}"
                )
            else:
                skipped.append((game_id, reason))
                print(
                    f"game={game_id} skipped={reason} "
                    f"completed={index}/{len(tasks)}"
                )

        return frames, skipped

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_game = {
            executor.submit(process_one_game, task): task[0]
            for task in tasks
        }

        completed = 0

        for future in as_completed(future_to_game):
            requested_game_id = future_to_game[future]
            completed += 1

            try:
                game_id, frame, reason = future.result()
            except Exception as exc:
                game_id = requested_game_id
                frame = None
                reason = f"worker failed: {type(exc).__name__}: {exc}"

            if frame is not None and not frame.empty:
                frames.append(frame)
                print(
                    f"game={game_id} plays={len(frame)} "
                    f"columns={len(frame.columns)} "
                    f"completed={completed}/{len(tasks)}"
                )
            else:
                skipped.append((game_id, reason))
                print(
                    f"game={game_id} skipped={reason} "
                    f"completed={completed}/{len(tasks)}"
                )

    return frames, skipped


# ─────────────────────────────────────────────
# EXISTING / SEASON ASSEMBLY
# ─────────────────────────────────────────────

def read_existing_pbp(output_file: Path) -> pd.DataFrame:
    if not output_file.exists():
        return pd.DataFrame()

    existing = pd.read_parquet(output_file)

    if existing.empty:
        return existing

    require_native_columns(
        existing,
        f"existing PBP {output_file}",
    )

    return existing


def game_ids_in_frame(df: pd.DataFrame) -> set[int]:
    if df.empty or "game_id" not in df.columns:
        return set()

    values = pd.to_numeric(df["game_id"], errors="coerce").dropna()
    return {int(value) for value in values.unique()}


def combine_season_pbp(
    existing: pd.DataFrame,
    new_frames: list[pd.DataFrame],
) -> tuple[pd.DataFrame, set[int]]:
    if not new_frames:
        return existing.copy(), set()

    replacement_ids: set[int] = set()

    for frame in new_frames:
        replacement_ids.update(game_ids_in_frame(frame))

    if existing.empty:
        base = pd.DataFrame()
    else:
        existing_game_ids = pd.to_numeric(
            existing["game_id"],
            errors="coerce",
        )
        base = existing.loc[
            ~existing_game_ids.isin(replacement_ids)
        ].copy()

    # The current processor's column order is the canonical order for this run.
    # Preserve it first; retain any older columns afterward if package versions
    # differ across incremental runs.
    current_columns: list[str] = []
    seen: set[str] = set()

    for frame in new_frames:
        for column in frame.columns:
            if column not in seen:
                current_columns.append(column)
                seen.add(column)

    if not base.empty:
        for column in base.columns:
            if column not in seen:
                current_columns.append(column)
                seen.add(column)

    combined = pd.concat(
        [base, *new_frames],
        ignore_index=True,
        sort=False,
    )

    combined = combined.reindex(columns=current_columns)

    sort_columns = [
        column
        for column in [
            "game_id",
            "game_play_number",
            "sequenceNumber",
            "id",
        ]
        if column in combined.columns
    ]

    if sort_columns:
        combined = combined.sort_values(
            sort_columns,
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)

    return combined, replacement_ids


def validate_season_pbp(
    df: pd.DataFrame,
    season: int,
) -> None:
    if df.empty:
        return

    require_native_columns(
        df,
        f"season={season} combined PBP",
    )

    seasons = pd.to_numeric(df["season"], errors="coerce")
    if seasons.isna().any() or not seasons.eq(season).all():
        observed = sorted(
            {
                int(value)
                for value in seasons.dropna().unique()
            }
        )
        raise ValueError(
            f"combined PBP contains wrong season values: {observed}"
        )

    if df.duplicated(["game_id", "id"]).any():
        duplicate_count = int(
            df.duplicated(["game_id", "id"], keep=False).sum()
        )
        raise ValueError(
            "combined PBP has duplicate (game_id, id) rows: "
            f"{duplicate_count}"
        )

    completed = df["status_type_completed"].dropna()

    if not completed.empty:
        if pd.api.types.is_bool_dtype(completed.dtype):
            if not bool(completed.all()):
                raise ValueError(
                    "combined PBP contains rows from incomplete games"
                )
        else:
            normalized = (
                completed.astype(str)
                .str.strip()
                .str.casefold()
            )
            bad = ~normalized.isin({"true", "1", "yes"})
            if bad.any():
                raise ValueError(
                    "combined PBP contains rows from incomplete games"
                )


def write_pbp_atomic(
    df: pd.DataFrame,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_output = output_file.with_name(
        output_file.name + f".{os.getpid()}.tmp"
    )

    try:
        df.to_parquet(
            tmp_output,
            index=False,
            engine="pyarrow",
        )
        os.replace(tmp_output, output_file)
    finally:
        try:
            tmp_output.unlink(missing_ok=True)
        except OSError:
            pass


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> int:
    ensure_dirs()
    reset_log()
    args = parse_args()

    try:
        require_sportsdataverse()

        season = get_season(args)

        if args.workers < 1:
            raise ValueError("--workers must be at least 1")

        output_file = PBP_DIR / f"{season}_pbp.parquet"

        log("=" * 80)
        log(
            f"pull_pbp.py started | season={season} "
            f"| source=sportsdataverse.CFBPlayProcess "
            f"| sportsdataverse_version={sportsdataverse_version()} "
            f"| workers={args.workers} "
            f"| refresh={args.refresh} "
            f"| dry_run={args.dry_run}"
        )

        schedule = load_schedule(season)
        existing = read_existing_pbp(output_file)
        existing_game_ids = game_ids_in_frame(existing)

        requested_game_ids = {
            int(game_id)
            for game_id in (args.game_id or [])
        }

        if requested_game_ids:
            schedule_ids_as_int = {
                int(game_id)
                for game_id in schedule
            }

            missing_ids = sorted(
                requested_game_ids - schedule_ids_as_int
            )

            if missing_ids:
                raise ValueError(
                    "Requested game id(s) not present in local schedule: "
                    + ", ".join(str(game_id) for game_id in missing_ids)
                )

            # Explicit requests are always refreshes and may be used for a
            # completed game even if it is already stored.
            process_ids = sorted(requested_game_ids)
            future_games_skipped = 0
            already_stored_skipped = 0

        else:
            eligible_rows = [
                row
                for row in schedule.values()
                if not schedule_game_is_future(row)
            ]

            future_games_skipped = (
                len(schedule) - len(eligible_rows)
            )

            eligible_ids = sorted(
                int(row["game_id"])
                for row in eligible_rows
            )

            if args.refresh:
                process_ids = eligible_ids
                already_stored_skipped = 0
            else:
                process_ids = [
                    game_id
                    for game_id in eligible_ids
                    if game_id not in existing_game_ids
                ]
                already_stored_skipped = (
                    len(eligible_ids) - len(process_ids)
                )

        log(f"schedule_games_total={len(schedule)}")
        log(f"existing_games={len(existing_game_ids)}")
        log(f"existing_rows={len(existing)}")
        log(f"future_games_skipped={future_games_skipped}")
        log(f"already_stored_skipped={already_stored_skipped}")
        log(f"games_to_process={len(process_ids)}")

        if not process_ids:
            log("status=no_new_games")
            log(f"output={output_file}")
            log("=" * 80)

            print("cfb pull_pbp completed")
            print(f"season: {season}")
            print("source_used: sportsdataverse.CFBPlayProcess")
            print(f"sportsdataverse_version: {sportsdataverse_version()}")
            print("games_processed: 0")
            print("games_skipped: 0")
            print(f"future_games_skipped: {future_games_skipped}")
            print(f"already_stored_skipped: {already_stored_skipped}")
            print(f"rows: {len(existing)}")
            print(f"columns: {len(existing.columns)}")
            print(f"output: {output_file}")
            print("status: no_new_games")
            return 0

        new_frames, skipped = process_games(
            game_ids=process_ids,
            season=season,
            workers=args.workers,
        )

        if args.dry_run:
            if new_frames:
                dry_run_df, dry_run_ids = combine_season_pbp(
                    existing=pd.DataFrame(),
                    new_frames=new_frames,
                )
                validate_season_pbp(
                    dry_run_df,
                    season=season,
                )

                log(f"dry_run_games={len(dry_run_ids)}")
                log(f"dry_run_rows={len(dry_run_df)}")
                log(f"dry_run_columns={len(dry_run_df.columns)}")
                log(f"games_skipped={len(skipped)}")

                for game_id, reason in sorted(skipped):
                    log(f"SKIPPED game_id={game_id} reason={reason}")

                log("status=dry_run_success")
                log("=" * 80)

                print("cfb pull_pbp dry run completed")
                print(f"season: {season}")
                print("source_used: sportsdataverse.CFBPlayProcess")
                print(f"sportsdataverse_version: {sportsdataverse_version()}")
                print(f"games_processed: {len(dry_run_ids)}")
                print(f"games_skipped: {len(skipped)}")
                print(f"rows: {len(dry_run_df)}")
                print(f"columns: {len(dry_run_df.columns)}")
                print("output_modified: no")
                print("status: dry_run_success")
                return 0

            log(f"games_skipped={len(skipped)}")
            for game_id, reason in sorted(skipped):
                log(f"SKIPPED game_id={game_id} reason={reason}")
            log("status=dry_run_no_completed_games")
            log("=" * 80)

            print("cfb pull_pbp dry run completed")
            print(f"season: {season}")
            print("source_used: sportsdataverse.CFBPlayProcess")
            print(f"sportsdataverse_version: {sportsdataverse_version()}")
            print("games_processed: 0")
            print(f"games_skipped: {len(skipped)}")
            print("rows: 0")
            print("columns: 0")
            print("output_modified: no")
            print("status: dry_run_no_completed_games")
            return 0

        combined, replacement_ids = combine_season_pbp(
            existing=existing,
            new_frames=new_frames,
        )

        validate_season_pbp(
            combined,
            season=season,
        )

        # Do not create a meaningless zero-column/zero-row Parquet before the
        # first completed game exists.
        if combined.empty and existing.empty:
            log("status=no_completed_games")
            for game_id, reason in sorted(skipped):
                log(f"SKIPPED game_id={game_id} reason={reason}")
            log("=" * 80)

            print("cfb pull_pbp completed")
            print(f"season: {season}")
            print("source_used: sportsdataverse.CFBPlayProcess")
            print(f"sportsdataverse_version: {sportsdataverse_version()}")
            print("games_processed: 0")
            print(f"games_skipped: {len(skipped)}")
            print(f"future_games_skipped: {future_games_skipped}")
            print("rows: 0")
            print("columns: 0")
            print("status: no_completed_games")
            return 0

        # Only rewrite when at least one game was successfully processed.
        if new_frames:
            write_pbp_atomic(
                combined,
                output_file=output_file,
            )

        final_game_ids = game_ids_in_frame(combined)

        log(f"games_processed={len(replacement_ids)}")
        log(f"games_skipped={len(skipped)}")
        log(f"games_in_output={len(final_game_ids)}")
        log(f"rows={len(combined)}")
        log(f"columns={len(combined.columns)}")
        log(f"output={output_file}")

        for game_id, reason in sorted(skipped):
            log(f"SKIPPED game_id={game_id} reason={reason}")

        log("status=success")
        log("=" * 80)

        print("cfb pull_pbp completed")
        print(f"season: {season}")
        print("source_used: sportsdataverse.CFBPlayProcess")
        print(f"sportsdataverse_version: {sportsdataverse_version()}")
        print(f"games_processed: {len(replacement_ids)}")
        print(f"games_skipped: {len(skipped)}")
        print(f"games_in_output: {len(final_game_ids)}")
        print(f"future_games_skipped: {future_games_skipped}")
        print(f"already_stored_skipped: {already_stored_skipped}")
        print(f"rows: {len(combined)}")
        print(f"columns: {len(combined.columns)}")
        print(f"output: {output_file}")
        print("status: success")

        return 0

    except Exception as exc:
        log(f"ERROR: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        print("cfb pull_pbp failed", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

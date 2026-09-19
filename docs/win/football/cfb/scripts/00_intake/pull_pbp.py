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
#   proc = CFBPlayProcess(gameId=game_id, join_participants=False)
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

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
BASE_DIR = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

CURRENT_WEEK_FILE = BASE_DIR / "config" / "current_week.yaml"
SCHEDULE_DIR = BASE_DIR / "00_intake" / "schedule"
PBP_DIR = BASE_DIR / "00_intake" / "pbp"
REPORT_ROOT = BASE_DIR / "errors"

EASTERN = ZoneInfo("America/New_York")

IMPLEMENTATION_VERSION = "sportsdataverse_v5_2026-09-12"

# Keep concurrency conservative. SportsDataverse itself performs ESPN network
# work and XGBoost model inference inside each game process.
DEFAULT_WORKERS = 1

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
# SETTINGS / CLI
# ──────────────────────────────────────────────

def read_current_week() -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read "
            "docs/win/football/cfb/config/current_week.yaml"
        )

    if not CURRENT_WEEK_FILE.exists():
        raise FileNotFoundError(
            f"Missing CFB current-week config: {CURRENT_WEEK_FILE}"
        )

    with CURRENT_WEEK_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"{CURRENT_WEEK_FILE} must contain a YAML mapping"
        )

    return data


def get_season(args: argparse.Namespace) -> int:
    config = read_current_week()
    configured_season = config.get("season")

    if configured_season in (None, ""):
        raise ValueError(
            f"{CURRENT_WEEK_FILE} is missing required season"
        )

    try:
        season = int(configured_season)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{CURRENT_WEEK_FILE} has invalid season={configured_season!r}"
        ) from exc

    if args.season is not None and int(args.season) != season:
        raise ValueError(
            f"--season={args.season} does not match "
            f"{CURRENT_WEEK_FILE} season={season}"
        )

    return season

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
        help=(
            "Optional consistency check. Must match "
            "config/current_week.yaml."
        ),
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

    invalid_game_dates: list[str] = []

    for _, row in schedule.iterrows():
        try:
            parse_game_date(row["game_date"])
        except ValueError as exc:
            invalid_game_dates.append(
                f"game_id={row['game_id']}: {exc}"
            )

    if invalid_game_dates:
        raise ValueError(
            f"{schedule_path} contains invalid game_date values: "
            + "; ".join(invalid_game_dates[:20])
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
        raise ValueError("game_date is blank")

    # Current local schedule uses YYYY-MM-DD. Accept an ISO timestamp too.
    parsed = pd.to_datetime(text, errors="coerce")

    if pd.isna(parsed):
        raise ValueError(f"invalid game_date {text!r}")

    return parsed.date()

def schedule_game_is_future(row: dict[str, str]) -> bool:
    game_date = parse_game_date(row.get("game_date"))


    # Automatic PBP processing is limited to games dated before today.
    # Games scheduled today or later are excluded.
    return game_date >= datetime.now(EASTERN).date()


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
) -> tuple[int, pd.DataFrame | None, str, str]:
    """
    Worker entry point. Kept at module scope so it is picklable on Windows.

    disposition:
      processed = completed game returned valid PBP
      skipped   = expected nonfatal state
      failed    = processor, worker, or validation failure
    """
    game_id, season = task

    print(
        f"game={game_id} status=starting join_participants=false",
        flush=True,
    )

    try:
        if CFBPlayProcess is None:
            return (
                game_id,
                None,
                "failed",
                "sportsdataverse import unavailable",
            )

        proc = CFBPlayProcess(gameId=game_id, join_participants=False)

        print(
            f"game={game_id} stage=espn_cfb_pbp starting",
            flush=True,
        )
        proc.espn_cfb_pbp()
        print(
            f"game={game_id} stage=espn_cfb_pbp complete",
            flush=True,
        )

        print(
            f"game={game_id} stage=processing_pipeline starting",
            flush=True,
        )
        result = proc.run_processing_pipeline()
        print(
            f"game={game_id} stage=processing_pipeline complete",
            flush=True,
        )

        if not isinstance(result, dict):
            return (
                game_id,
                None,
                "failed",
                f"processor returned {type(result).__name__}, expected dict",
            )

        plays = result.get("plays") or []

        if not isinstance(plays, list):
            return (
                game_id,
                None,
                "failed",
                f"result['plays'] returned {type(plays).__name__}, expected list",
            )

        if not plays:
            return (
                game_id,
                None,
                "skipped",
                "no plays returned",
            )

        df = pd.DataFrame(plays)

        validate_processed_game(
            df,
            game_id=game_id,
            season=season,
        )

        if not game_is_completed(df):
            return (
                game_id,
                None,
                "skipped",
                "game not completed",
            )

        return (
            game_id,
            df,
            "processed",
            "",
        )

    except Exception as exc:
        return (
            game_id,
            None,
            "failed",
            f"{type(exc).__name__}: {exc}",
        )



def _process_games_parallel(
    *,
    tasks: list[tuple[int, int]],
    workers: int,
    frames: list[pd.DataFrame],
    skipped: list[tuple[int, str]],
    failures: list[tuple[int, str]],
) -> None:
    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
        future_to_game = {
            executor.submit(
                process_one_game,
                task,
            ): task[0]
            for task in tasks
        }

        completed = 0

        for future in as_completed(
            future_to_game
        ):
            requested_game_id = future_to_game[future]
            completed += 1

            try:
                (
                    game_id,
                    frame,
                    disposition,
                    reason,
                ) = future.result()

            except Exception as exc:
                game_id = requested_game_id
                frame = None
                disposition = "failed"
                reason = (
                    "worker failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            if disposition == "processed":
                if frame is None or frame.empty:
                    failure_reason = (
                        "processor reported processed "
                        "but returned no frame"
                    )
                    failures.append(
                        (game_id, failure_reason)
                    )
                    print(
                        f"game={game_id} failed={failure_reason} "
                        f"completed={completed}/{len(tasks)}"
                    )
                else:
                    frames.append(frame)
                    print(
                        f"game={game_id} plays={len(frame)} "
                        f"columns={len(frame.columns)} "
                        f"completed={completed}/{len(tasks)}"
                    )

            elif disposition == "skipped":
                skipped.append((game_id, reason))
                print(
                    f"game={game_id} skipped={reason} "
                    f"completed={completed}/{len(tasks)}"
                )

            elif disposition == "failed":
                failures.append((game_id, reason))
                print(
                    f"game={game_id} failed={reason} "
                    f"completed={completed}/{len(tasks)}"
                )

            else:
                failure_reason = (
                    f"unexpected disposition={disposition!r}"
                )
                failures.append((game_id, failure_reason))
                print(
                    f"game={game_id} failed={failure_reason} "
                    f"completed={completed}/{len(tasks)}"
                )


def process_games(
    game_ids: list[int],
    season: int,
    workers: int,
) -> tuple[
    list[pd.DataFrame],
    list[tuple[int, str]],
    list[tuple[int, str]],
]:
    if not game_ids:
        return [], [], []

    frames: list[pd.DataFrame] = []
    skipped: list[tuple[int, str]] = []
    failures: list[tuple[int, str]] = []

    tasks = [(game_id, season) for game_id in game_ids]

    if workers == 1 or len(tasks) == 1:
        for index, task in enumerate(tasks, start=1):
            game_id, frame, disposition, reason = process_one_game(task)

            if disposition == "processed":
                if frame is None or frame.empty:
                    failure_reason = (
                        "processor reported processed but returned no frame"
                    )
                    failures.append((game_id, failure_reason))
                    print(
                        f"game={game_id} failed={failure_reason} "
                        f"completed={index}/{len(tasks)}"
                    )
                else:
                    frames.append(frame)
                    print(
                        f"game={game_id} plays={len(frame)} "
                        f"columns={len(frame.columns)} "
                        f"completed={index}/{len(tasks)}"
                    )

            elif disposition == "skipped":
                skipped.append((game_id, reason))
                print(
                    f"game={game_id} skipped={reason} "
                    f"completed={index}/{len(tasks)}"
                )

            elif disposition == "failed":
                failures.append((game_id, reason))
                print(
                    f"game={game_id} failed={reason} "
                    f"completed={index}/{len(tasks)}"
                )

            else:
                failure_reason = (
                    f"unexpected disposition={disposition!r}"
                )
                failures.append((game_id, failure_reason))
                print(
                    f"game={game_id} failed={failure_reason} "
                    f"completed={index}/{len(tasks)}"
                )

        return frames, skipped, failures

    _process_games_parallel(
        tasks=tasks,
        workers=workers,
        frames=frames,
        skipped=skipped,
        failures=failures,
    )

    return frames, skipped, failures

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


def _select_pbp_process_ids(
    *,
    requested_game_ids: set[int],
    schedule: dict[str, dict[str, str]],
    existing_game_ids: set[int],
    refresh: bool,
) -> tuple[list[int], int, int]:
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
                + ", ".join(
                    str(game_id)
                    for game_id in missing_ids
                )
            )

        return sorted(requested_game_ids), 0, 0

    eligible_rows = [
        row
        for row in schedule.values()
        if not schedule_game_is_future(row)
    ]

    future_games_skipped = len(schedule) - len(eligible_rows)

    eligible_ids = sorted(
        int(row["game_id"])
        for row in eligible_rows
    )

    if refresh:
        return eligible_ids, future_games_skipped, 0

    process_ids = [
        game_id
        for game_id in eligible_ids
        if game_id not in existing_game_ids
    ]

    already_stored_skipped = len(eligible_ids) - len(process_ids)

    return process_ids, future_games_skipped, already_stored_skipped


def _raise_on_pbp_failures(
    *,
    failures: list[tuple[int, str]],
    report: PipelineReporter,
    new_frames: list[pd.DataFrame],
    skipped: list[tuple[int, str]],
    skipped_details: list[dict[str, object]],
    existing: pd.DataFrame,
) -> None:
    if not failures:
        return

    failure_details = [
        {
            "game_id": game_id,
            "reason": reason,
        }
        for game_id, reason in sorted(failures)
    ]

    report.update_details(
        {
            "run_status": "failed",
            "games_processed_before_failure": len(new_frames),
            "games_skipped": len(skipped),
            "games_failed": len(failures),
            "skipped_games": skipped_details,
            "failed_games": failure_details,
            "output_modified": False,
        }
    )

    report.set_rows(rows_out=len(existing))

    failure_detail = "; ".join(
        f"game_id={game_id}: {reason}"
        for game_id, reason in sorted(failures)
    )

    raise RuntimeError(
        f"{len(failures)} game(s) failed PBP processing: "
        f"{failure_detail}"
    )


def main() -> int:
    args = parse_args()

    with PipelineReporter(
        script=__file__,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        extra_context={
            "implementation": IMPLEMENTATION_VERSION,
            "source": "sportsdataverse.CFBPlayProcess",
        },
    ) as report:
        report.add_input(CURRENT_WEEK_FILE)

        report.update_details(
            {
                "workers": args.workers,
                "join_participants": False,
                "refresh": args.refresh,
                "dry_run": args.dry_run,
                "output_modified": False,
            }
        )

        require_sportsdataverse()

        season = get_season(args)
        current_week_config = read_current_week()

        report.season = season

        configured_week = current_week_config.get("week")
        if configured_week not in (None, ""):
            report.week = configured_week

        report.set_detail(
            "sportsdataverse_version",
            sportsdataverse_version(),
        )

        if args.workers < 1:
            raise ValueError("--workers must be at least 1")

        schedule_path = SCHEDULE_DIR / f"{season}_schedule.csv"
        output_file = PBP_DIR / f"{season}_pbp.parquet"

        report.add_input(schedule_path)

        if output_file.exists():
            report.add_input(output_file)

        report.add_output(output_file)

        schedule = load_schedule(season)
        existing = read_existing_pbp(output_file)

        validate_season_pbp(
            existing,
            season=season,
        )

        existing_game_ids = game_ids_in_frame(existing)

        requested_game_ids = {
            int(game_id)
            for game_id in (args.game_id or [])
        }

        report.set_detail(
            "requested_game_ids",
            sorted(requested_game_ids),
        )

        (
            process_ids,
            future_games_skipped,
            already_stored_skipped,
        ) = _select_pbp_process_ids(
            requested_game_ids=requested_game_ids,
            schedule=schedule,
            existing_game_ids=existing_game_ids,
            refresh=args.refresh,
        )

        report.set_rows(
            rows_in=len(existing),
        )

        report.update_details(
            {
                "schedule_games_total": len(schedule),
                "existing_games": len(existing_game_ids),
                "existing_rows": len(existing),
                "future_games_skipped": future_games_skipped,
                "already_stored_skipped": already_stored_skipped,
                "games_attempted": len(process_ids),
            }
        )

        if not process_ids:
            report.update_details(
                {
                    "run_status": "no_new_games",
                    "games_processed": 0,
                    "games_skipped": 0,
                    "games_failed": 0,
                    "games_in_output": len(existing_game_ids),
                    "final_rows": len(existing),
                    "final_columns": len(existing.columns),
                    "output_modified": False,
                }
            )

            report.set_rows(
                rows_out=len(existing),
            )

            print("cfb pull_pbp completed")
            print(f"implementation: {IMPLEMENTATION_VERSION}")
            print(f"season: {season}")
            print("source_used: sportsdataverse.CFBPlayProcess")
            print(
                f"sportsdataverse_version: "
                f"{sportsdataverse_version()}"
            )
            print("games_processed: 0")
            print("games_skipped: 0")
            print(
                f"future_games_skipped: "
                f"{future_games_skipped}"
            )
            print(
                f"already_stored_skipped: "
                f"{already_stored_skipped}"
            )
            print(f"rows: {len(existing)}")
            print(f"columns: {len(existing.columns)}")
            print(f"output: {output_file}")
            print("status: no_new_games")

            return 0

        new_frames, skipped, failures = process_games(
            game_ids=process_ids,
            season=season,
            workers=args.workers,
        )

        skipped_details = [
            {
                "game_id": game_id,
                "reason": reason,
            }
            for game_id, reason in sorted(skipped)
        ]

        _raise_on_pbp_failures(
            failures=failures,
            report=report,
            new_frames=new_frames,
            skipped=skipped,
            skipped_details=skipped_details,
            existing=existing,
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

                report.update_details(
                    {
                        "run_status": "dry_run_success",
                        "games_processed": len(dry_run_ids),
                        "games_skipped": len(skipped),
                        "games_failed": 0,
                        "skipped_games": skipped_details,
                        "dry_run_rows": len(dry_run_df),
                        "dry_run_columns": len(dry_run_df.columns),
                        "output_modified": False,
                    }
                )

                report.set_rows(
                    rows_out=len(dry_run_df),
                )

                print("cfb pull_pbp dry run completed")
                print(f"implementation: {IMPLEMENTATION_VERSION}")
                print(f"season: {season}")
                print("source_used: sportsdataverse.CFBPlayProcess")
                print(
                    f"sportsdataverse_version: "
                    f"{sportsdataverse_version()}"
                )
                print(f"games_processed: {len(dry_run_ids)}")
                print(f"games_skipped: {len(skipped)}")
                print(f"rows: {len(dry_run_df)}")
                print(f"columns: {len(dry_run_df.columns)}")
                print("output_modified: no")
                print("status: dry_run_success")

                return 0

            report.update_details(
                {
                    "run_status": "dry_run_no_completed_games",
                    "games_processed": 0,
                    "games_skipped": len(skipped),
                    "games_failed": 0,
                    "skipped_games": skipped_details,
                    "dry_run_rows": 0,
                    "dry_run_columns": 0,
                    "output_modified": False,
                }
            )

            report.set_rows(
                rows_out=0,
            )

            print("cfb pull_pbp dry run completed")
            print(f"implementation: {IMPLEMENTATION_VERSION}")
            print(f"season: {season}")
            print("source_used: sportsdataverse.CFBPlayProcess")
            print(
                f"sportsdataverse_version: "
                f"{sportsdataverse_version()}"
            )
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
            report.update_details(
                {
                    "run_status": "no_completed_games",
                    "games_processed": 0,
                    "games_skipped": len(skipped),
                    "games_failed": 0,
                    "skipped_games": skipped_details,
                    "games_in_output": 0,
                    "final_rows": 0,
                    "final_columns": 0,
                    "output_modified": False,
                }
            )

            report.set_rows(
                rows_out=0,
            )

            print("cfb pull_pbp completed")
            print(f"implementation: {IMPLEMENTATION_VERSION}")
            print(f"season: {season}")
            print("source_used: sportsdataverse.CFBPlayProcess")
            print(
                f"sportsdataverse_version: "
                f"{sportsdataverse_version()}"
            )
            print("games_processed: 0")
            print(f"games_skipped: {len(skipped)}")
            print(
                f"future_games_skipped: "
                f"{future_games_skipped}"
            )
            print("rows: 0")
            print("columns: 0")
            print("status: no_completed_games")

            return 0

        output_modified = False

        # Only rewrite when at least one game was successfully processed.
        if new_frames:
            write_pbp_atomic(
                combined,
                output_file=output_file,
            )
            output_modified = True

        final_game_ids = game_ids_in_frame(combined)

        report.update_details(
            {
                "run_status": "success",
                "games_processed": len(replacement_ids),
                "games_skipped": len(skipped),
                "games_failed": 0,
                "skipped_games": skipped_details,
                "games_in_output": len(final_game_ids),
                "final_rows": len(combined),
                "final_columns": len(combined.columns),
                "output_modified": output_modified,
            }
        )

        report.set_rows(
            rows_out=len(combined),
        )

        print("cfb pull_pbp completed")
        print(f"implementation: {IMPLEMENTATION_VERSION}")
        print(f"season: {season}")
        print("source_used: sportsdataverse.CFBPlayProcess")
        print(
            f"sportsdataverse_version: "
            f"{sportsdataverse_version()}"
        )
        print(f"games_processed: {len(replacement_ids)}")
        print(f"games_skipped: {len(skipped)}")
        print(f"games_in_output: {len(final_game_ids)}")
        print(
            f"future_games_skipped: "
            f"{future_games_skipped}"
        )
        print(
            f"already_stored_skipped: "
            f"{already_stored_skipped}"
        )
        print(f"rows: {len(combined)}")
        print(f"columns: {len(combined.columns)}")
        print(f"output: {output_file}")
        print("status: success")

        return 0

if __name__ == "__main__":
    raise SystemExit(main())
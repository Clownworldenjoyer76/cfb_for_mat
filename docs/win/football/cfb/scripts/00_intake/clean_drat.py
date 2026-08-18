#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCRIPT_PATH = Path(__file__).resolve()
NFL_ROOT = SCRIPT_PATH.parents[2]

DRAT_RAW_DIR = NFL_ROOT / "00_intake" / "predictions" / "drat" / "raw"
DRAT_CLEAN_DIR = NFL_ROOT / "00_intake" / "predictions" / "drat" / "clean"
SCHEDULE_WEEKLY_DIR = NFL_ROOT / "00_intake" / "schedule" / "weekly"
LOG_PATH = NFL_ROOT / "errors" / "00_intake" / "clean_drat.txt"

HISTORICAL_FILENAME_RE = re.compile(
    r"^(?P<season>\d{4})_wk(?P<week>\d{2})_odds\.csv$",
    re.IGNORECASE,
)

EXPECTED_INPUT_HEADERS = [
    "season",
    "week",
    "game_id",
    "commence_time_utc",
    "home_team",
    "away_team",
    "book",
    "spread_home",
    "spread_away",
    "total",
    "moneyline_home",
    "moneyline_away",
    "updated_at_utc",
    "is_consensus",
    "game_date",
    "game_time",
    "home_prob",
    "away_prob",
    "spread_home_odds",
    "spread_away_odds",
    "total_over",
    "total_under",
    "total_odds_over",
    "total_odds_under",
    "away_projected_score",
    "home_projected_score",
    "total_projected_score",
]

OUTPUT_HEADERS = [
    "season",
    "week",
    "game_id",
    "commence_time_utc",
    "home_team",
    "away_team",
    "spread_home",
    "spread_away",
    "total",
    "moneyline_home",
    "moneyline_away",
    "updated_at_utc",
    "game_date",
    "game_time",
    "home_prob",
    "away_prob",
    "spread_home_odds",
    "spread_away_odds",
    "total_over",
    "total_under",
    "total_odds_over",
    "total_odds_under",
    "away_projected_score",
    "home_projected_score",
    "total_projected_score",
]

SCHEDULE_REQUIRED_HEADERS = [
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
]

MatchKey = tuple[str, str, str, str]
ScheduleIndex = dict[MatchKey, set[str]]


class RunLog:
    def __init__(self) -> None:
        self.info_lines: list[str] = []
        self.warning_lines: list[str] = []
        self.error_lines: list[str] = []

    def info(self, message: str) -> None:
        self.info_lines.append(message)
        print(f"INFO: {message}")

    def warning(self, message: str) -> None:
        self.warning_lines.append(message)
        print(f"WARNING: {message}")

    def error(self, message: str) -> None:
        self.error_lines.append(message)
        print(f"ERROR: {message}", file=sys.stderr)

    @property
    def has_errors(self) -> bool:
        return bool(self.error_lines)

    def write(
        self,
        *,
        schedule_files: int,
        schedule_rows: int,
        schedule_keys: int,
        historical_found: int,
        historical_ignored_2025: int,
        historical_succeeded: int,
        historical_failed: int,
        latest_succeeded: int,
        latest_failed: int,
        rows_written: int,
    ) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        lines = [
            "clean_drat.py",
            "=" * 80,
            f"Log written UTC: {started}",
            "",
            "Paths",
            "-" * 80,
            f"DRAT raw:       {DRAT_RAW_DIR}",
            f"DRAT clean:     {DRAT_CLEAN_DIR}",
            f"Schedule input: {SCHEDULE_WEEKLY_DIR}",
            f"Log:            {LOG_PATH}",
            "",
            "Summary",
            "-" * 80,
            f"Schedule files read:            {schedule_files}",
            f"Schedule rows read:             {schedule_rows}",
            f"Schedule match keys loaded:     {schedule_keys}",
            f"Historical DRAT files found:    {historical_found}",
            f"2025 historical files ignored: {historical_ignored_2025}",
            f"Historical files succeeded:    {historical_succeeded}",
            f"Historical files failed:       {historical_failed}",
            f"latest.csv succeeded:           {latest_succeeded}",
            f"latest.csv failed:              {latest_failed}",
            f"Rows written:                   {rows_written}",
            f"Warnings:                       {len(self.warning_lines)}",
            f"Errors:                         {len(self.error_lines)}",
            "",
        ]

        if self.info_lines:
            lines.extend(
                [
                    "Details",
                    "-" * 80,
                    *self.info_lines,
                    "",
                ]
            )

        if self.warning_lines:
            lines.extend(
                [
                    "Warnings",
                    "-" * 80,
                    *self.warning_lines,
                    "",
                ]
            )

        if self.error_lines:
            lines.extend(
                [
                    "Errors",
                    "-" * 80,
                    *self.error_lines,
                    "",
                ]
            )

        lines.extend(
            [
                "Result",
                "-" * 80,
                "FAILED" if self.has_errors else "SUCCESS",
                "",
            ]
        )

        LOG_PATH.write_text("\n".join(lines), encoding="utf-8")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_number(value: object) -> str:
    """
    Normalize integer-like season/week values for matching only.

    Examples:
        "01"   -> "1"
        "1"    -> "1"
        "2026" -> "2026"

    Non-integer values are returned stripped and unchanged.
    """
    text = clean_text(value)

    if not text:
        return ""

    try:
        return str(int(text))
    except ValueError:
        return text


def make_match_key(row: dict[str, str]) -> MatchKey:
    return (
        normalize_number(row.get("season")),
        normalize_number(row.get("week")),
        clean_text(row.get("home_team")),
        clean_text(row.get("away_team")),
    )


def describe_match_key(key: MatchKey) -> str:
    season, week, home_team, away_team = key
    return (
        f"season={season}, week={week}, "
        f"home_team={home_team!r}, away_team={away_team!r}"
    )


def read_csv_rows(
    path: Path,
    required_headers: Iterable[str],
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        fieldnames = [clean_text(field) for field in reader.fieldnames]
        required = list(required_headers)

        missing = [header for header in required if header not in fieldnames]
        if missing:
            raise ValueError(
                "Missing required header(s): " + ", ".join(missing)
            )

        rows = list(reader)

    return fieldnames, rows


def build_schedule_index(
    log: RunLog,
) -> tuple[ScheduleIndex | None, int, int]:
    if not SCHEDULE_WEEKLY_DIR.exists():
        log.error(
            f"Schedule directory does not exist: {SCHEDULE_WEEKLY_DIR}"
        )
        return None, 0, 0

    schedule_files = sorted(SCHEDULE_WEEKLY_DIR.glob("*.csv"))

    if not schedule_files:
        log.error(
            f"No schedule CSV files found in: {SCHEDULE_WEEKLY_DIR}"
        )
        return None, 0, 0

    schedule_index: ScheduleIndex = {}
    schedule_rows_read = 0
    load_failed = False

    for schedule_path in schedule_files:
        try:
            _, rows = read_csv_rows(
                schedule_path,
                SCHEDULE_REQUIRED_HEADERS,
            )
        except Exception as exc:
            log.error(
                f"Could not read schedule file {schedule_path}: {exc}"
            )
            load_failed = True
            continue

        file_rows_loaded = 0

        for row_number, row in enumerate(rows, start=2):
            schedule_rows_read += 1

            key = make_match_key(row)
            game_id = clean_text(row.get("game_id"))

            season, week, home_team, away_team = key

            if not season or not week or not home_team or not away_team:
                log.warning(
                    f"Skipping schedule row with incomplete match fields: "
                    f"{schedule_path} row {row_number}; "
                    f"{describe_match_key(key)}"
                )
                continue

            if not game_id:
                log.warning(
                    f"Skipping schedule row with blank game_id: "
                    f"{schedule_path} row {row_number}; "
                    f"{describe_match_key(key)}"
                )
                continue

            schedule_index.setdefault(key, set()).add(game_id)
            file_rows_loaded += 1

        log.info(
            f"Schedule loaded: {schedule_path.name} "
            f"({file_rows_loaded} usable rows)"
        )

    if load_failed:
        log.error(
            "Schedule index could not be trusted because one or more "
            "schedule files failed validation."
        )
        return None, len(schedule_files), schedule_rows_read

    if not schedule_index:
        log.error("Schedule index contains no usable game records.")
        return None, len(schedule_files), schedule_rows_read

    ambiguous_keys = {
        key: game_ids
        for key, game_ids in schedule_index.items()
        if len(game_ids) > 1
    }

    if ambiguous_keys:
        log.warning(
            f"Schedule data contains {len(ambiguous_keys)} match key(s) "
            "with multiple game_id values. A DRAT row using one of those "
            "keys will fail rather than choosing a game_id arbitrarily."
        )

    return schedule_index, len(schedule_files), schedule_rows_read


def write_clean_csv(
    output_path: Path,
    rows: list[dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_HEADERS,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        header: clean_text(row.get(header))
                        for header in OUTPUT_HEADERS
                    }
                )

        temporary_path.replace(output_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def transform_drat_rows(
    *,
    source_path: Path,
    rows: list[dict[str, str]],
    schedule_index: ScheduleIndex,
    log: RunLog,
    expected_season: str | None = None,
    expected_week: str | None = None,
) -> list[dict[str, str]] | None:
    output_rows: list[dict[str, str]] = []
    file_errors: list[str] = []

    normalized_expected_season = (
        normalize_number(expected_season)
        if expected_season is not None
        else None
    )
    normalized_expected_week = (
        normalize_number(expected_week)
        if expected_week is not None
        else None
    )

    for row_number, row in enumerate(rows, start=2):
        key = make_match_key(row)
        season, week, home_team, away_team = key

        if not season or not week or not home_team or not away_team:
            file_errors.append(
                f"{source_path} row {row_number}: "
                "missing one or more schedule-match values; "
                f"{describe_match_key(key)}"
            )
            continue

        if (
            normalized_expected_season is not None
            and season != normalized_expected_season
        ):
            file_errors.append(
                f"{source_path} row {row_number}: "
                f"row season={season!r} does not match filename "
                f"season={normalized_expected_season!r}."
            )
            continue

        if (
            normalized_expected_week is not None
            and week != normalized_expected_week
        ):
            file_errors.append(
                f"{source_path} row {row_number}: "
                f"row week={week!r} does not match filename "
                f"week={normalized_expected_week!r}."
            )
            continue

        schedule_game_ids = schedule_index.get(key)

        if not schedule_game_ids:
            file_errors.append(
                f"{source_path} row {row_number}: "
                "no schedule match found for "
                f"{describe_match_key(key)}."
            )
            continue

        if len(schedule_game_ids) != 1:
            file_errors.append(
                f"{source_path} row {row_number}: "
                "schedule match is ambiguous for "
                f"{describe_match_key(key)}; "
                f"game_id values={sorted(schedule_game_ids)!r}."
            )
            continue

        schedule_game_id = next(iter(schedule_game_ids))

        clean_row = {
            header: clean_text(row.get(header))
            for header in OUTPUT_HEADERS
        }

        # Always replace the imported DRAT game_id with the schedule game_id.
        clean_row["game_id"] = schedule_game_id

        output_rows.append(clean_row)

    if file_errors:
        for message in file_errors:
            log.error(message)

        log.error(
            f"{source_path.name}: clean output was not written because "
            f"{len(file_errors)} row error(s) were found."
        )
        return None

    return output_rows


def process_historical_file(
    source_path: Path,
    schedule_index: ScheduleIndex,
    log: RunLog,
) -> tuple[bool, int]:
    match = HISTORICAL_FILENAME_RE.fullmatch(source_path.name)

    if match is None:
        log.error(
            f"Historical filename does not match expected pattern: "
            f"{source_path.name}"
        )
        return False, 0

    season = match.group("season")
    week_text = match.group("week")
    week_number = int(week_text)

    output_path = (
        DRAT_CLEAN_DIR
        / f"{season}_week_{week_number}_drat.csv"
    )

    try:
        fieldnames, rows = read_csv_rows(
            source_path,
            EXPECTED_INPUT_HEADERS,
        )
    except Exception as exc:
        log.error(f"Could not read {source_path}: {exc}")
        return False, 0

    extra_headers = [
        header
        for header in fieldnames
        if header not in EXPECTED_INPUT_HEADERS
    ]

    if extra_headers:
        log.warning(
            f"{source_path.name}: extra input header(s) will be ignored: "
            + ", ".join(extra_headers)
        )

    clean_rows = transform_drat_rows(
        source_path=source_path,
        rows=rows,
        schedule_index=schedule_index,
        log=log,
        expected_season=season,
        expected_week=str(week_number),
    )

    if clean_rows is None:
        return False, 0

    try:
        write_clean_csv(output_path, clean_rows)
    except Exception as exc:
        log.error(
            f"Could not write clean output {output_path}: {exc}"
        )
        return False, 0

    log.info(
        f"Historical DRAT cleaned: {source_path.name} -> "
        f"{output_path.name} ({len(clean_rows)} rows)"
    )

    return True, len(clean_rows)


def process_latest_file(
    source_path: Path,
    schedule_index: ScheduleIndex,
    log: RunLog,
) -> tuple[bool, int]:
    output_path = DRAT_CLEAN_DIR / "latest.csv"

    try:
        fieldnames, rows = read_csv_rows(
            source_path,
            EXPECTED_INPUT_HEADERS,
        )
    except Exception as exc:
        log.error(f"Could not read {source_path}: {exc}")
        return False, 0

    extra_headers = [
        header
        for header in fieldnames
        if header not in EXPECTED_INPUT_HEADERS
    ]

    if extra_headers:
        log.warning(
            f"{source_path.name}: extra input header(s) will be ignored: "
            + ", ".join(extra_headers)
        )

    clean_rows = transform_drat_rows(
        source_path=source_path,
        rows=rows,
        schedule_index=schedule_index,
        log=log,
    )

    if clean_rows is None:
        return False, 0

    try:
        write_clean_csv(output_path, clean_rows)
    except Exception as exc:
        log.error(
            f"Could not write clean output {output_path}: {exc}"
        )
        return False, 0

    log.info(
        f"Latest DRAT cleaned: {source_path.name} -> "
        f"{output_path.name} ({len(clean_rows)} rows)"
    )

    return True, len(clean_rows)


def main() -> int:
    log = RunLog()

    schedule_files = 0
    schedule_rows = 0
    schedule_keys = 0

    historical_found = 0
    historical_ignored_2025 = 0
    historical_succeeded = 0
    historical_failed = 0

    latest_succeeded = 0
    latest_failed = 0
    rows_written = 0

    try:
        DRAT_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        if not DRAT_RAW_DIR.exists():
            log.error(
                f"DRAT raw directory does not exist: {DRAT_RAW_DIR}"
            )
        else:
            schedule_index, schedule_files, schedule_rows = (
                build_schedule_index(log)
            )

            if schedule_index is not None:
                schedule_keys = len(schedule_index)

                raw_csv_files = sorted(DRAT_RAW_DIR.glob("*.csv"))

                historical_files: list[Path] = []
                latest_path: Path | None = None

                for source_path in raw_csv_files:
                    filename_lower = source_path.name.lower()

                    if filename_lower == "latest.csv":
                        latest_path = source_path
                        continue

                    if source_path.name.startswith("2025"):
                        historical_ignored_2025 += 1
                        log.info(
                            f"Ignored 2025 DRAT file: "
                            f"{source_path.name}"
                        )
                        continue

                    if HISTORICAL_FILENAME_RE.fullmatch(source_path.name):
                        historical_files.append(source_path)
                        continue

                    log.warning(
                        f"Ignored unrecognized CSV in DRAT raw directory: "
                        f"{source_path.name}"
                    )

                historical_found = len(historical_files)

                if not historical_files:
                    log.warning(
                        "No non-2025 historical DRAT files matching "
                        "YYYY_wkNN_odds.csv were found."
                    )

                for source_path in historical_files:
                    success, row_count = process_historical_file(
                        source_path,
                        schedule_index,
                        log,
                    )

                    if success:
                        historical_succeeded += 1
                        rows_written += row_count
                    else:
                        historical_failed += 1

                if latest_path is None:
                    latest_failed += 1
                    log.error(
                        f"Required latest.csv was not found at: "
                        f"{DRAT_RAW_DIR / 'latest.csv'}"
                    )
                else:
                    success, row_count = process_latest_file(
                        latest_path,
                        schedule_index,
                        log,
                    )

                    if success:
                        latest_succeeded += 1
                        rows_written += row_count
                    else:
                        latest_failed += 1

    except Exception as exc:
        log.error(
            f"Unexpected fatal error: "
            f"{type(exc).__name__}: {exc}"
        )

    try:
        log.write(
            schedule_files=schedule_files,
            schedule_rows=schedule_rows,
            schedule_keys=schedule_keys,
            historical_found=historical_found,
            historical_ignored_2025=historical_ignored_2025,
            historical_succeeded=historical_succeeded,
            historical_failed=historical_failed,
            latest_succeeded=latest_succeeded,
            latest_failed=latest_failed,
            rows_written=rows_written,
        )
    except Exception as exc:
        print(
            f"ERROR: Could not write log file {LOG_PATH}: {exc}",
            file=sys.stderr,
        )
        return 1

    return 1 if log.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())

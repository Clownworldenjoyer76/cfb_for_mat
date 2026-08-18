#!/usr/bin/env python3
"""
Step 5 Split: split the combined historical NFL training table by season.

Runs immediately after Step 5.

READS:
  docs/win/football/nfl/training/historical_core_2021_2025.csv

WRITES:
  docs/win/football/nfl/training/historical_core_2021.csv
  docs/win/football/nfl/training/historical_core_2022.csv
  docs/win/football/nfl/training/historical_core_2023.csv
  docs/win/football/nfl/training/historical_core_2024.csv
  docs/win/football/nfl/training/historical_core_2025.csv

The split is based strictly on the existing `season` column.

The script validates that:
  - the source contains a `season` column;
  - every row belongs to exactly one expected season;
  - only seasons 2021-2025 are present;
  - every output has the exact same columns as the source;
  - every output row has the correct number of fields;
  - every output row has the correct season;
  - the combined output row count exactly matches the source row count;
  - no source rows are lost during the split.

The original combined file is NOT modified or deleted.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys


NFL_ROOT = Path("docs/win/football/nfl")

TRAINING_DIR = (
    NFL_ROOT / "training"
)

SOURCE_PATH = (
    TRAINING_DIR / "historical_core_2021_2025.csv"
)

EXPECTED_SEASONS = [
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
]

OUTPUT_PATHS = {
    season: (
        TRAINING_DIR
        / f"historical_core_{season}.csv"
    )
    for season in EXPECTED_SEASONS
}

TEMP_PATHS = {
    season: (
        TRAINING_DIR
        / f"historical_core_{season}.step5_split.tmp.csv"
    )
    for season in EXPECTED_SEASONS
}


def increase_csv_field_limit() -> None:
    """
    Increase Python's CSV field-size limit as far as the
    current platform safely allows.
    """
    limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def normalize_season(
    value: str,
    row_number: int,
) -> str:
    """
    Normalize and validate a season value.

    Accepts values such as:
      2021
      "2021"
      " 2021 "

    Rejects blank, malformed, or unexpected seasons.
    """
    text = str(value).strip()

    if not text:
        raise ValueError(
            f"Row {row_number}: blank season value."
        )

    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Row {row_number}: invalid season value: "
            f"{text!r}"
        ) from exc

    if not numeric.is_integer():
        raise ValueError(
            f"Row {row_number}: non-integer season value: "
            f"{text!r}"
        )

    season = str(
        int(numeric)
    )

    if season not in EXPECTED_SEASONS:
        raise ValueError(
            f"Row {row_number}: unexpected season "
            f"{season!r}. Expected one of: "
            + ", ".join(EXPECTED_SEASONS)
        )

    return season


def remove_temp_files() -> None:
    """
    Remove any temporary split files left by this run.
    """
    for path in TEMP_PATHS.values():
        if path.exists():
            path.unlink()


def validate_source_header(
    header: list[str],
) -> int:
    """
    Validate the source header and return the index of
    the season column.
    """
    if not header:
        raise ValueError(
            "Source training file has no header."
        )

    duplicate_columns = sorted(
        {
            column
            for column in header
            if header.count(column) > 1
        }
    )

    if duplicate_columns:
        raise ValueError(
            "Source training file contains duplicate "
            "column names: "
            + ", ".join(duplicate_columns)
        )

    if "season" not in header:
        raise ValueError(
            "Source training file is missing required "
            "column: season"
        )

    return header.index(
        "season"
    )


def split_source() -> tuple[
    list[str],
    dict[str, int],
    int,
]:
    """
    Stream the combined training file and route every
    data row into the correct season-specific temporary
    output file.
    """
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing source training file: "
            f"{SOURCE_PATH}"
        )

    if not SOURCE_PATH.is_file():
        raise RuntimeError(
            f"Source path is not a file: "
            f"{SOURCE_PATH}"
        )

    TRAINING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    remove_temp_files()

    row_counts = {
        season: 0
        for season in EXPECTED_SEASONS
    }

    total_rows = 0

    handles: dict[str, object] = {}
    writers: dict[str, csv.writer] = {}

    try:
        with SOURCE_PATH.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as source_handle:
            reader = csv.reader(
                source_handle
            )

            try:
                header = next(
                    reader
                )
            except StopIteration as exc:
                raise ValueError(
                    "Source training file is empty."
                ) from exc

            season_index = (
                validate_source_header(
                    header
                )
            )

            expected_column_count = len(
                header
            )

            for season in EXPECTED_SEASONS:
                temp_path = TEMP_PATHS[
                    season
                ]

                handle = temp_path.open(
                    "w",
                    encoding="utf-8",
                    newline="",
                )

                handles[season] = handle

                writer = csv.writer(
                    handle,
                    lineterminator="\n",
                )

                writers[season] = writer

                writer.writerow(
                    header
                )

            for row_number, row in enumerate(
                reader,
                start=2,
            ):
                if len(row) != expected_column_count:
                    raise ValueError(
                        f"Row {row_number}: expected "
                        f"{expected_column_count} columns, "
                        f"found {len(row)}."
                    )

                season = normalize_season(
                    row[season_index],
                    row_number,
                )

                writers[season].writerow(
                    row
                )

                row_counts[season] += 1
                total_rows += 1

    finally:
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass

    if total_rows == 0:
        raise ValueError(
            "Source training file contains no data rows."
        )

    return (
        header,
        row_counts,
        total_rows,
    )


def validate_split_file(
    season: str,
    expected_header: list[str],
    expected_rows: int,
) -> int:
    """
    Re-read one temporary split file and independently
    validate its header, row width, season values, and
    row count.
    """
    path = TEMP_PATHS[
        season
    ]

    if not path.exists():
        raise FileNotFoundError(
            f"Missing temporary split file: {path}"
        )

    actual_rows = 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.reader(
            handle
        )

        try:
            header = next(
                reader
            )
        except StopIteration as exc:
            raise ValueError(
                f"{path}: output file is empty."
            ) from exc

        if header != expected_header:
            raise ValueError(
                f"{path}: output columns do not exactly "
                "match the source columns."
            )

        season_index = header.index(
            "season"
        )

        expected_column_count = len(
            expected_header
        )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            if len(row) != expected_column_count:
                raise ValueError(
                    f"{path}: row {row_number} expected "
                    f"{expected_column_count} columns, "
                    f"found {len(row)}."
                )

            row_season = normalize_season(
                row[season_index],
                row_number,
            )

            if row_season != season:
                raise ValueError(
                    f"{path}: row {row_number} belongs "
                    f"to season {row_season}, expected "
                    f"{season}."
                )

            actual_rows += 1

    if actual_rows != expected_rows:
        raise RuntimeError(
            f"{path}: row-count mismatch. "
            f"Expected {expected_rows}, "
            f"found {actual_rows}."
        )

    return actual_rows


def publish_split_files() -> None:
    """
    Atomically replace the five final season files with
    the fully validated temporary outputs.
    """
    for season in EXPECTED_SEASONS:
        temp_path = TEMP_PATHS[
            season
        ]

        output_path = OUTPUT_PATHS[
            season
        ]

        temp_path.replace(
            output_path
        )


def main() -> int:
    increase_csv_field_limit()

    print(
        f"Reading: {SOURCE_PATH}"
    )

    try:
        (
            source_header,
            expected_counts,
            source_total_rows,
        ) = split_source()

        validated_counts: dict[str, int] = {}

        for season in EXPECTED_SEASONS:
            validated_counts[season] = (
                validate_split_file(
                    season=season,
                    expected_header=source_header,
                    expected_rows=expected_counts[
                        season
                    ],
                )
            )

        validated_total_rows = sum(
            validated_counts.values()
        )

        if (
            validated_total_rows
            != source_total_rows
        ):
            raise RuntimeError(
                "Split validation failed: total row "
                "count does not match source. "
                f"Source={source_total_rows}, "
                f"split={validated_total_rows}"
            )

        publish_split_files()

    except Exception:
        remove_temp_files()
        raise

    print(
        f"Source columns: {len(source_header)}"
    )

    print(
        f"Source rows: {source_total_rows}"
    )

    for season in EXPECTED_SEASONS:
        path = OUTPUT_PATHS[
            season
        ]

        size_bytes = path.stat().st_size

        size_mib = (
            size_bytes
            / (1024 * 1024)
        )

        print(
            f"{season}: "
            f"{validated_counts[season]} rows | "
            f"{size_mib:.2f} MiB | "
            f"{path}"
        )

    print(
        "Split row validation passed: "
        f"{validated_total_rows}/"
        f"{source_total_rows}"
    )

    print(
        "Original combined file preserved: "
        f"{SOURCE_PATH}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise

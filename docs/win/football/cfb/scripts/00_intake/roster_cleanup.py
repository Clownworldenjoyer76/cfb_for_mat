#!/usr/bin/env python3
"""
roster_cleanup.py

Validate the configured-season raw ESPN CFB roster and publish the cleaned
roster_master.csv using the intentionally preserved roster-master schema.

Inputs:
    docs/win/football/cfb/config/current_week.yaml
    docs/win/football/cfb/data/master/team_master.csv
    docs/win/football/cfb/data/raw/raw_roster.csv

Output:
    docs/win/football/cfb/data/master/roster_master.csv
"""

from __future__ import annotations

import csv
import os
import sys
import uuid
from pathlib import Path

import yaml


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


CURRENT_WEEK_CONFIG_PATH = CFB_ROOT / "config" / "current_week.yaml"
TEAM_MASTER_PATH = CFB_ROOT / "data" / "master" / "team_master.csv"
INPUT_PATH = CFB_ROOT / "data" / "raw" / "raw_roster.csv"
OUTPUT_PATH = CFB_ROOT / "data" / "master" / "roster_master.csv"
REPORT_ROOT = CFB_ROOT / "errors"

SCRIPT_VERSION = "cfb-roster-cleanup-v2-2026-09-15"

KEEP_COLUMNS = [
    "age",
    "alternateIds.sdr",
    "birthPlace.city",
    "birthPlace.country",
    "birthPlace.state",
    "college.abbrev",
    "college.guid",
    "college.id",
    "college.name",
    "college.shortName",
    "contract.active",
    "contract.bonus",
    "contract.optionType",
    "contract.salary",
    "contract.salaryRemaining",
    "contract.season.endDate",
    "contract.season.startDate",
    "contract.season.year",
    "contract.signedThrough",
    "dateOfBirth",
    "debutYear",
    "displayHeight",
    "displayName",
    "displayWeight",
    "experience.years",
    "firstName",
    "fullName",
    "guid",
    "hand.abbreviation",
    "hand.displayValue",
    "hand.type",
    "headshot.alt",
    "headshot.href",
    "height",
    "id",
    "injuries.0.date",
    "injuries.0.status",
    "jersey",
    "lastName",
    "position.abbreviation",
    "position.displayName",
    "position.id",
    "position.leaf",
    "position.name",
    "position.parent.abbreviation",
    "position.parent.displayName",
    "position.parent.id",
    "position.parent.leaf",
    "position.parent.name",
    "shortName",
    "slug",
    "status.abbreviation",
    "status.id",
    "status.name",
    "status.type",
    "team_id",
    "uid",
    "weight",
]

REQUIRED_INPUT_COLUMNS = [
    "season",
    "season_type",
    "id",
    "displayName",
    "team_id",
]


def load_current_week() -> tuple[int, int, int]:
    if not CURRENT_WEEK_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing current-week config: {CURRENT_WEEK_CONFIG_PATH}"
        )

    with CURRENT_WEEK_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Current-week config must contain a YAML mapping")

    values: dict[str, int] = {}

    for key in ("season", "season_type", "week"):
        if key not in payload:
            raise ValueError(
                f"Current-week config missing required key: {key}"
            )

        raw = payload.get(key)

        if isinstance(raw, bool):
            raise ValueError(
                f"Current-week config {key} must be an integer"
            )

        try:
            values[key] = int(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Current-week config {key} must be an integer"
            ) from exc

    if values["season"] < 2000:
        raise ValueError(
            f"Invalid configured season: {values['season']}"
        )

    if values["season_type"] < 1:
        raise ValueError(
            f"Invalid configured season_type: {values['season_type']}"
        )

    if values["week"] < 1:
        raise ValueError(
            f"Invalid configured week: {values['week']}"
        )

    return (
        values["season"],
        values["season_type"],
        values["week"],
    )


def duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)

    return sorted(duplicates)


def load_authoritative_team_ids() -> set[str]:
    if not TEAM_MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing team master: {TEAM_MASTER_PATH}"
        )

    with TEAM_MASTER_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        if "team_id" not in fieldnames:
            raise ValueError(
                "team_master.csv missing required column: team_id"
            )

        team_ids: set[str] = set()

        for index, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    "Malformed team_master.csv row with extra fields at "
                    f"CSV line {index}"
                )

            team_id = str(row.get("team_id") or "").strip()

            if not team_id:
                raise ValueError(
                    f"team_master.csv has blank team_id at CSV line {index}"
                )

            if not team_id.isdigit() or int(team_id) <= 0:
                raise ValueError(
                    "team_master.csv has invalid team_id at "
                    f"CSV line {index}: {team_id!r}"
                )

            team_ids.add(team_id)

    if not team_ids:
        raise ValueError(
            "team_master.csv contains no authoritative team IDs"
        )

    return team_ids


def _stage1_read_raw_roster_input() -> tuple[list[dict[str, str]], list[str]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing raw roster input: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        if not fieldnames:
            raise ValueError("raw_roster.csv has no header")

        duplicate_headers = duplicate_values(fieldnames)
        if duplicate_headers:
            raise ValueError(
                "raw_roster.csv contains duplicate header columns: "
                f"{duplicate_headers}"
            )

        input_columns = set(fieldnames)
        missing_required = [
            column
            for column in REQUIRED_INPUT_COLUMNS
            if column not in input_columns
        ]
        if missing_required:
            raise ValueError(
                "raw_roster.csv missing required columns: "
                f"{missing_required}"
            )

        missing_optional = [
            column
            for column in KEEP_COLUMNS
            if column not in input_columns
        ]
        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError("raw_roster.csv contains no athlete rows")

    return raw_rows, missing_optional


def _stage1_validate_raw_roster_structure(
    row: dict[str, str],
    *,
    row_number: int,
) -> None:
    if None in row:
        raise ValueError(
            "raw_roster.csv contains a malformed row with extra "
            f"fields at CSV line {row_number}"
        )
    missing_cells = [
        key
        for key, value in row.items()
        if key is not None and value is None
    ]
    if missing_cells:
        raise ValueError(
            "raw_roster.csv contains a structurally incomplete row "
            f"at CSV line {row_number}; missing cells for "
            f"{missing_cells[:20]}"
        )


def _stage1_validate_raw_roster_values(
    row: dict[str, str],
    *,
    row_number: int,
    season: int,
    season_type: int,
    authoritative_team_ids: set[str],
) -> tuple[str, str]:
    row_season = str(row.get("season") or "").strip()
    row_season_type = str(row.get("season_type") or "").strip()
    athlete_id = str(row.get("id") or "").strip()
    display_name = str(row.get("displayName") or "").strip()
    team_id = str(row.get("team_id") or "").strip()
    if row_season != str(season):
        raise ValueError(
            "raw_roster.csv season mismatch at CSV line "
            f"{row_number}: expected={season}, actual={row_season!r}"
        )
    if row_season_type != str(season_type):
        raise ValueError(
            "raw_roster.csv season_type mismatch at CSV line "
            f"{row_number}: expected={season_type}, actual={row_season_type!r}"
        )
    if not athlete_id:
        raise ValueError(
            "raw_roster.csv has blank athlete id at CSV line "
            f"{row_number}"
        )
    if not display_name:
        raise ValueError(
            "raw_roster.csv has blank displayName at CSV line "
            f"{row_number}"
        )
    if not team_id:
        raise ValueError(
            "raw_roster.csv has blank team_id at CSV line "
            f"{row_number}"
        )
    if team_id not in authoritative_team_ids:
        raise ValueError(
            "raw_roster.csv references a non-authoritative team at "
            f"CSV line {row_number}: team_id={team_id!r}"
        )
    return athlete_id, team_id


def _stage1_validate_raw_roster_row(
    row: dict[str, str],
    *,
    row_number: int,
    season: int,
    season_type: int,
    authoritative_team_ids: set[str],
) -> tuple[dict[str, str], str, str]:
    _stage1_validate_raw_roster_structure(row, row_number=row_number)
    athlete_id, team_id = _stage1_validate_raw_roster_values(
        row,
        row_number=row_number,
        season=season,
        season_type=season_type,
        authoritative_team_ids=authoritative_team_ids,
    )
    cleaned = {
        column: str(row.get(column) or "")
        for column in KEEP_COLUMNS
    }
    return cleaned, athlete_id, team_id



def _stage1_record_roster_identity(
    *,
    athlete_id: str,
    team_id: str,
    seen_keys: set[tuple[str, str]],
    athlete_team_assignments: dict[str, str],
) -> None:
    key = (team_id, athlete_id)
    if key in seen_keys:
        raise ValueError(
            "raw_roster.csv contains duplicate athlete key: "
            f"{key}"
        )

    prior_team = athlete_team_assignments.get(athlete_id)
    if prior_team is not None and prior_team != team_id:
        raise ValueError(
            "raw_roster.csv assigns one athlete to multiple teams: "
            f"athlete_id={athlete_id}, teams={prior_team},{team_id}"
        )

    seen_keys.add(key)
    athlete_team_assignments[athlete_id] = team_id


def _stage1_validate_roster_team_coverage(
    authoritative_team_ids: set[str],
    represented_team_ids: set[str],
) -> None:
    missing_teams = sorted(
        authoritative_team_ids - represented_team_ids,
        key=int,
    )
    foreign_teams = sorted(
        represented_team_ids - authoritative_team_ids,
        key=int,
    )
    if missing_teams or foreign_teams:
        raise ValueError(
            "raw_roster.csv team coverage does not match the "
            "authoritative team universe. "
            f"missing={missing_teams[:50]}, foreign={foreign_teams[:50]}"
        )


def load_and_validate_raw_roster(
    *,
    season: int,
    season_type: int,
    authoritative_team_ids: set[str],
) -> tuple[
    list[dict[str, str]],
    list[str],
    set[str],
    set[str],
]:
    raw_rows, missing_optional = _stage1_read_raw_roster_input()
    cleaned_rows: list[dict[str, str]] = []
    represented_team_ids: set[str] = set()
    unique_athlete_ids: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    athlete_team_assignments: dict[str, str] = {}

    for row_number, row in enumerate(raw_rows, start=2):
        cleaned, athlete_id, team_id = _stage1_validate_raw_roster_row(
            row,
            row_number=row_number,
            season=season,
            season_type=season_type,
            authoritative_team_ids=authoritative_team_ids,
        )
        _stage1_record_roster_identity(
            athlete_id=athlete_id,
            team_id=team_id,
            seen_keys=seen_keys,
            athlete_team_assignments=athlete_team_assignments,
        )
        represented_team_ids.add(team_id)
        unique_athlete_ids.add(athlete_id)
        cleaned_rows.append(cleaned)

    _stage1_validate_roster_team_coverage(
        authoritative_team_ids,
        represented_team_ids,
    )
    return (
        cleaned_rows,
        missing_optional,
        represented_team_ids,
        unique_athlete_ids,
    )



def temporary_path(final_path: Path) -> Path:
    return final_path.with_name(
        f".{final_path.name}.{uuid.uuid4().hex}.tmp"
    )


def write_staged_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=KEEP_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_serialized_roster_rows(
    rows: list[dict[str, str]],
    *,
    authoritative_team_ids: set[str],
    represented_team_ids: set[str],
    seen_keys: set[tuple[str, str]],
    athlete_team_assignments: dict[str, str],
) -> None:
    for row_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(
                "Serialized roster_master.csv contains a malformed row "
                f"at CSV line {row_number}"
            )

        athlete_id = str(row.get("id") or "").strip()
        display_name = str(row.get("displayName") or "").strip()
        team_id = str(row.get("team_id") or "").strip()

        if not athlete_id:
            raise ValueError(
                "Serialized roster_master.csv has blank athlete id at "
                f"CSV line {row_number}"
            )

        if not display_name:
            raise ValueError(
                "Serialized roster_master.csv has blank displayName at "
                f"CSV line {row_number}"
            )

        if not team_id:
            raise ValueError(
                "Serialized roster_master.csv has blank team_id at "
                f"CSV line {row_number}"
            )

        if team_id not in authoritative_team_ids:
            raise ValueError(
                "Serialized roster_master.csv references a "
                f"non-authoritative team_id={team_id!r}"
            )

        key = (team_id, athlete_id)

        if key in seen_keys:
            raise ValueError(
                "Serialized roster_master.csv contains duplicate "
                f"athlete key: {key}"
            )

        prior_team = athlete_team_assignments.get(athlete_id)

        if prior_team is not None and prior_team != team_id:
            raise ValueError(
                "Serialized roster_master.csv assigns one athlete to "
                "multiple teams: "
                f"athlete_id={athlete_id}, "
                f"teams={prior_team},{team_id}"
            )

        seen_keys.add(key)
        athlete_team_assignments[athlete_id] = team_id
        represented_team_ids.add(team_id)


def validate_staged_csv(
    path: Path,
    *,
    expected_rows: list[dict[str, str]],
    authoritative_team_ids: set[str],
) -> None:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)
        actual_fieldnames = reader.fieldnames or []

        if actual_fieldnames != KEEP_COLUMNS:
            raise ValueError(
                "Serialized roster_master.csv header mismatch"
            )

        rows = list(reader)

    if len(rows) != len(expected_rows):
        raise ValueError(
            "Serialized roster_master.csv row count mismatch: "
            f"expected={len(expected_rows)}, actual={len(rows)}"
        )

    if rows != expected_rows:
        raise ValueError(
            "Serialized roster_master.csv does not exactly match the "
            "validated cleaned rows"
        )

    represented_team_ids: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    athlete_team_assignments: dict[str, str] = {}

    _validate_serialized_roster_rows(
        rows,
        authoritative_team_ids=authoritative_team_ids,
        represented_team_ids=represented_team_ids,
        seen_keys=seen_keys,
        athlete_team_assignments=athlete_team_assignments,
    )

    if represented_team_ids != authoritative_team_ids:
        missing_teams = sorted(
            authoritative_team_ids - represented_team_ids,
            key=int,
        )
        foreign_teams = sorted(
            represented_team_ids - authoritative_team_ids,
            key=int,
        )

        raise ValueError(
            "Serialized roster_master.csv team coverage mismatch. "
            f"missing={missing_teams[:50]}, "
            f"foreign={foreign_teams[:50]}"
        )


def publish_atomic(
    rows: list[dict[str, str]],
    authoritative_team_ids: set[str],
) -> bool:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = temporary_path(OUTPUT_PATH)

    try:
        write_staged_csv(temp_path, rows)

        validate_staged_csv(
            temp_path,
            expected_rows=rows,
            authoritative_team_ids=authoritative_team_ids,
        )

        if (
            OUTPUT_PATH.exists()
            and OUTPUT_PATH.read_bytes()
            == temp_path.read_bytes()
        ):
            return False

        os.replace(temp_path, OUTPUT_PATH)
        return True

    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


def run(report: PipelineReporter) -> int:
    season, season_type, week = load_current_week()

    report.season = season
    report.week = week
    report.set_detail("season_type", season_type)

    authoritative_team_ids = load_authoritative_team_ids()

    (
        cleaned_rows,
        missing_optional,
        represented_team_ids,
        unique_athlete_ids,
    ) = load_and_validate_raw_roster(
        season=season,
        season_type=season_type,
        authoritative_team_ids=authoritative_team_ids,
    )

    output_modified = publish_atomic(
        cleaned_rows,
        authoritative_team_ids,
    )

    report.set_rows(
        rows_in=len(cleaned_rows),
        rows_out=len(cleaned_rows),
    )

    report.update_details(
        {
            "authoritative_team_count": len(
                authoritative_team_ids
            ),
            "represented_team_count": len(
                represented_team_ids
            ),
            "unique_athlete_count": len(
                unique_athlete_ids
            ),
            "output_column_count": len(KEEP_COLUMNS),
            "output_columns": KEEP_COLUMNS,
            "missing_optional_column_count": len(
                missing_optional
            ),
            "missing_optional_columns": missing_optional,
            "duplicate_athlete_key_count": 0,
            "athlete_team_conflict_count": 0,
            "output_path": str(OUTPUT_PATH),
            "output_modified": output_modified,
        }
    )

    return 0


def main() -> int:
    with PipelineReporter(
        script=SCRIPT_PATH,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        extra_context={
            "script_version": SCRIPT_VERSION,
            "source": "raw_roster.csv",
        },
    ) as report:
        report.add_input(CURRENT_WEEK_CONFIG_PATH)
        report.add_input(TEAM_MASTER_PATH)
        report.add_input(INPUT_PATH)
        report.add_output(OUTPUT_PATH)

        return run(report)


if __name__ == "__main__":
    raise SystemExit(main())

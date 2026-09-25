"""Shared low-level helpers for CFB pipeline scripts."""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import pandas as pd
import yaml


CFB_ROOT = Path(__file__).resolve().parent.parent

_NULL_TEXT = {
    "",
    "nan",
    "none",
    "null",
    "<na>",
    "nat",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    return "" if text.casefold() in _NULL_TEXT else text


def finite_float(value: Any) -> Optional[float]:
    text = clean_text(value)
    if not text:
        return None

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_yaml(
    path: Path,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"{label} must contain a YAML mapping: {path}"
        )

    return data


def require_columns(
    frame: Any,
    columns: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]
    if missing:
        raise RuntimeError(
            f"{label}: missing required columns: {missing}"
        )


def _required_integer(value: Any, label: str) -> int:
    text = clean_text(value)
    if not text:
        raise RuntimeError(f"{label} is required")

    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{label} must be numeric; found {value!r}"
        ) from exc

    if not math.isfinite(number) or not number.is_integer():
        raise RuntimeError(
            f"{label} must be an integer; found {value!r}"
        )

    return int(number)


def resolve_target(
    current_week: dict[str, Any],
    season_override: Optional[int],
    week_override: Optional[int],
) -> tuple[int, int]:
    configured_season = _required_integer(
        current_week.get("season"),
        "current_week.season",
    )
    configured_week = _required_integer(
        current_week.get("week"),
        "current_week.week",
    )

    season = (
        int(season_override)
        if season_override is not None
        else configured_season
    )
    week = (
        int(week_override)
        if week_override is not None
        else configured_week
    )

    if season < 1900:
        raise RuntimeError(f"Invalid target season: {season}")
    if week <= 0:
        raise RuntimeError(f"Invalid target week: {week}")

    return season, week


def normalize_game_id(value: Any) -> str:
    text = clean_text(value)
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def validate_game_ids(
    frame: Any,
    label: str,
) -> None:
    game_ids = frame["game_id"].map(normalize_game_id)

    if game_ids.eq("").any():
        raise RuntimeError(f"{label}: blank game_id found")

    duplicates = (
        game_ids[
            game_ids.duplicated(keep=False)
        ]
        .drop_duplicates()
        .tolist()
    )

    if duplicates:
        raise RuntimeError(
            f"{label}: duplicate game_id values: {duplicates[:10]}"
        )

    frame["game_id"] = game_ids


def weekly_schedule_path(week: int) -> Path:
    return (
        CFB_ROOT
        / "00_intake"
        / "schedule"
        / "weekly"
        / f"week_{week}_CFB_weekly_schedule.csv"
    )


def load_current_week_config(
    path: Path,
) -> tuple[int, int, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing current-week config: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            "Current-week config must contain a YAML mapping"
        )

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


def read_required_csv(
    path: Path,
    required_columns: list[str],
    label: str,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        missing = [
            column
            for column in required_columns
            if column not in fieldnames
        ]
        if missing:
            raise ValueError(
                f"{label} missing columns: {missing}"
            )

        rows = list(reader)

    return rows


def format_number(number: Optional[float]) -> str:
    if number is None:
        return ""
    if number.is_integer():
        return str(int(number))
    return str(number)


def format_american(number: Optional[float]) -> str:
    if number is None or number == 0:
        return ""
    return str(int(round(number)))

def add_projection_core_arguments(parser: Any) -> None:
    parser.add_argument(
        "--home-field",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--drives-per-team",
        type=float,
        default=11.5,
    )
    parser.add_argument(
        "--market-margin-weight",
        type=float,
        default=0.36,
    )
    parser.add_argument(
        "--fpi-margin-weight",
        type=float,
        default=0.28,
    )
    parser.add_argument(
        "--espn-margin-weight",
        type=float,
        default=0.20,
    )


def print_projection_source_counts(frame: Any) -> None:
    print(
        "with_market_spread="
        f"{int(pd.to_numeric(frame['market_home_margin'], errors='coerce').notna().sum())}"
    )
    print(
        "with_fpi="
        f"{int(pd.to_numeric(frame['fpi_home_margin'], errors='coerce').notna().sum())}"
    )
    print(
        "with_espn="
        f"{int(pd.to_numeric(frame['espn_home_margin'], errors='coerce').notna().sum())}"
    )


def print_projection_adjustment_counts(frame: Any) -> None:
    print(
        "with_market_total="
        f"{int(pd.to_numeric(frame['market_total'], errors='coerce').notna().sum())}"
    )
    print(
        "fresh_injury_adjustments="
        f"{int(pd.to_numeric(frame['injury_margin_adjustment'], errors='coerce').fillna(0).abs().gt(0).sum())}"
    )
    print(
        "travel_adjustments="
        f"{int(pd.to_numeric(frame['travel_margin_adjustment'], errors='coerce').fillna(0).abs().gt(0).sum())}"
    )
    print(
        "weather_adjustments="
        f"{int(pd.to_numeric(frame['weather_total_adjustment'], errors='coerce').fillna(0).abs().gt(0).sum())}"
    )


def normalized_frame_pair(
    serialized: Any,
    expected: Any,
    columns: list[str],
    cleaner: Any,
) -> tuple[Any, Any]:
    left = serialized.reset_index(
        drop=True
    ).copy()
    right = expected.reset_index(
        drop=True
    ).copy()

    for column in columns:
        left[column] = left[column].map(cleaner)
        right[column] = right[column].map(cleaner)

    return left, right


def write_csv_rows_durable(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    *,
    project_columns: bool = False,
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()

        if project_columns:
            writer.writerows(
                {
                    column: row.get(
                        column,
                        "",
                    )
                    for column in fieldnames
                }
                for row in rows
            )
        else:
            writer.writerows(rows)

        handle.flush()
        os.fsync(
            handle.fileno()
        )


def write_atomic_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp_path = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        write_csv_rows_durable(
            temp_path,
            rows,
            fieldnames,
            project_columns=True,
        )
        os.replace(
            temp_path,
            path,
        )
    finally:
        try:
            temp_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass

def validate_target_columns(
    frame: Any,
    label: str,
    season: int,
    week: int,
    integer_parser: Any,
) -> None:
    seasons = {
        integer_parser(value, f"{label}: season")
        for value in frame["season"]
    }
    weeks = {
        integer_parser(value, f"{label}: week")
        for value in frame["week"]
    }
    if seasons != {season}:
        raise RuntimeError(
            f"{label}: expected only season={season}; "
            f"found {sorted(seasons)}"
        )
    if weeks != {week}:
        raise RuntimeError(
            f"{label}: expected only week={week}; "
            f"found {sorted(weeks)}"
        )


def team_identity_values(
    row: Any,
    schedule_row: Any,
    cleaner: Any,
) -> tuple[str, str, str, str]:
    return (
        cleaner(row.get("away_team")),
        cleaner(row.get("home_team")),
        cleaner(schedule_row.get("away_team")),
        cleaner(schedule_row.get("home_team")),
    )


def require_csv_fieldnames(
    fieldnames: list[str],
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(
            f"{label} missing required columns: {missing}"
        )


def stage_dataframe_csv(
    path: Path,
    frame: Any,
) -> Any:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        frame.to_csv(
            handle,
            index=False,
            lineterminator="\n",
        )
        handle.flush()
        os.fsync(handle.fileno())

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )

def resolve_weekly_report_target(
    report: Any,
    config_path: Path,
    season_override: Optional[int],
    week_override: Optional[int],
) -> tuple[int, int]:
    current_week = read_yaml(
        config_path,
        "current-week config",
    )
    season, week = resolve_target(
        current_week,
        season_override,
        week_override,
    )
    report.season = season
    report.week = week
    return season, week


def register_report_paths(
    report: Any,
    *,
    inputs: tuple[Path, ...],
    output: Path,
) -> None:
    for path in inputs:
        report.add_input(path)
    report.add_output(output)


def prepare_schedule_coverage(
    schedule: Any,
    normalized_schedule_ids: Any,
    candidate: Any,
    label: str,
    season: int,
    week: int,
    integer_parser: Any,
) -> tuple[Any, list[str], list[str]]:
    prepared = schedule.copy()
    prepared["game_id"] = normalized_schedule_ids

    validate_target_columns(
        prepared,
        label,
        season,
        week,
        integer_parser,
    )

    candidate_ids = set(candidate["game_id"])
    target_ids = set(prepared["game_id"])

    return (
        prepared,
        sorted(target_ids - candidate_ids),
        sorted(candidate_ids - target_ids),
    )

def require_schedule_coverage(
    candidate: Any,
    schedule: Any,
    message: str,
) -> Any:
    candidate_ids = set(candidate["game_id"])
    target_ids = set(schedule["game_id"])

    missing = sorted(target_ids - candidate_ids)
    unexpected = sorted(candidate_ids - target_ids)

    if missing or unexpected:
        raise RuntimeError(
            f"{message}; "
            f"missing_count={len(missing)} "
            f"unexpected_count={len(unexpected)} "
            f"missing_examples={missing[:10]} "
            f"unexpected_examples={unexpected[:10]}"
        )

    return schedule.set_index(
        "game_id",
        drop=False,
    )

"""Shared low-level helpers for CFB pipeline scripts."""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
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
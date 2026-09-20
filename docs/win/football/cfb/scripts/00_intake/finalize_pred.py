#!/usr/bin/env python3
"""Finalize ESPN predictor data for the configured CFB target week."""

from __future__ import annotations

import csv
import os
import re
import sys
import uuid
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import yaml

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter

CONFIG_PATH = CFB_ROOT / "config" / "current_week.yaml"
CLEAN_DIR = CFB_ROOT / "00_intake" / "predictions" / "clean"
WEEKLY_DIR = CFB_ROOT / "00_intake" / "schedule" / "weekly"
OUTPUT_DIR = CFB_ROOT / "00_intake" / "predictions" / "final"
REPORT_ROOT = CFB_ROOT / "errors"

SCRIPT_VERSION = "cfb-finalize-pred-v2-2026-09-16"
ESPN_MARGIN_SYMMETRY_TOLERANCE = Decimal("0.25")
TWO_PLACES = Decimal("0.01")

OUT_HEADERS = [
    "game_id", "game_date", "game_time", "home_team", "away_team",
    "matchupQuality", "home_prob", "away_prob", "tie_prob",
    "away_projected_pts", "home_projected_pts", "total_projected_pts",
    "home_PtDiff", "away_PtDiff", "home_rating", "away_rating",
    "game_name", "season", "season_type", "week", "sport", "league",
]

SCHEDULE_REQUIRED_COLUMNS = {
    "season", "season_type", "week", "game_id", "game_date", "game_time",
    "away_team", "home_team", "total", "odds_available", "odds_missing_reason",
}


class FinalizePredictionValidationError(RuntimeError):
    pass


def text(value: object) -> str:
    return "" if value is None else str(value).strip()


def positive_int(value: object, *, label: str) -> int:
    value_text = text(value)

    if not re.fullmatch(r"\d+", value_text):
        raise FinalizePredictionValidationError(
            f"{label} must be a positive integer: {value!r}"
        )

    parsed = int(value_text)

    if parsed <= 0:
        raise FinalizePredictionValidationError(
            f"{label} must be positive: {parsed}"
        )

    return parsed


def finite_decimal(value: object, *, label: str) -> Decimal:
    value_text = text(value)

    if not value_text:
        raise FinalizePredictionValidationError(f"{label} is blank")

    try:
        number = Decimal(value_text)
    except InvalidOperation as exc:
        raise FinalizePredictionValidationError(
            f"{label} is not numeric: {value_text!r}"
        ) from exc

    if not number.is_finite():
        raise FinalizePredictionValidationError(
            f"{label} is not finite: {value_text!r}"
        )

    return number


def probability(value: object, *, label: str) -> Decimal:
    number = finite_decimal(value, label=label)

    if not Decimal("0") <= number <= Decimal("1"):
        raise FinalizePredictionValidationError(
            f"{label} must be between 0 and 1: {number}"
        )

    return number


def percentage(value: object, *, label: str) -> Decimal:
    number = finite_decimal(value, label=label)

    if not Decimal("0") <= number <= Decimal("100"):
        raise FinalizePredictionValidationError(
            f"{label} must be between 0 and 100: {number}"
        )

    return number


def fmt2(number: Decimal) -> str:
    return format(number.quantize(TWO_PLACES), ".2f")


def validate_date(value: object, *, label: str) -> str:
    value_text = text(value)

    if not value_text:
        raise FinalizePredictionValidationError(f"{label} is blank")

    try:
        datetime.strptime(value_text, "%Y-%m-%d")
    except ValueError as exc:
        raise FinalizePredictionValidationError(
            f"{label} must use YYYY-MM-DD: {value_text!r}"
        ) from exc

    return value_text


def validate_time(value: object, *, label: str) -> str:
    value_text = text(value)

    if not value_text:
        raise FinalizePredictionValidationError(f"{label} is blank")

    try:
        datetime.strptime(value_text, "%H:%M")
    except ValueError as exc:
        raise FinalizePredictionValidationError(
            f"{label} must use HH:MM: {value_text!r}"
        ) from exc

    return value_text


def binary_int(value: object, *, label: str) -> int:
    value_text = text(value)

    if value_text not in {"0", "1"}:
        raise FinalizePredictionValidationError(
            f"{label} must be 0 or 1: {value_text!r}"
        )

    return int(value_text)


def load_config() -> tuple[int, int, int]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing current-week config: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise FinalizePredictionValidationError(
            "current_week.yaml must contain a mapping"
        )

    values: dict[str, int] = {}

    for key in ("season", "season_type", "week"):
        if key not in payload:
            raise FinalizePredictionValidationError(
                f"current_week.yaml missing required key: {key}"
            )

        values[key] = positive_int(
            payload.get(key),
            label=f"current_week.{key}",
        )

    if values["season"] < 2000:
        raise FinalizePredictionValidationError(
            f"Invalid configured season: {values['season']}"
        )

    return values["season"], values["season_type"], values["week"]


def target_paths(
    season: int,
    season_type: int,
    week: int,
) -> tuple[Path, Path, Path]:
    clean_path = CLEAN_DIR / f"{season}_{season_type}_{week}_predictions.csv"
    schedule_path = WEEKLY_DIR / f"week_{week}_CFB_weekly_schedule.csv"
    output_path = OUTPUT_DIR / f"{season}_{season_type}_{week}_clean_predictions.csv"
    return clean_path, schedule_path, output_path


def load_schedule(
    path: Path,
    *,
    season: int,
    season_type: int,
    week: int,
) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Target weekly schedule not found: {path}")

    games: dict[str, dict[str, str]] = {}

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(
            SCHEDULE_REQUIRED_COLUMNS
            - set(reader.fieldnames or [])
        )

        if missing:
            raise FinalizePredictionValidationError(
                f"Target weekly schedule missing required columns: {missing}"
            )

        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise FinalizePredictionValidationError(
                    f"Malformed weekly-schedule row at CSV line {line_number}"
                )

            row_target = (
                positive_int(
                    row.get("season"),
                    label=f"schedule season line {line_number}",
                ),
                positive_int(
                    row.get("season_type"),
                    label=f"schedule season_type line {line_number}",
                ),
                positive_int(
                    row.get("week"),
                    label=f"schedule week line {line_number}",
                ),
            )

            if row_target != (
                season,
                season_type,
                week,
            ):
                raise FinalizePredictionValidationError(
                    "Weekly schedule target mismatch at CSV line "
                    f"{line_number}: expected={season}/{season_type}/{week}, "
                    f"actual={row_target[0]}/{row_target[1]}/{row_target[2]}"
                )

            game_id = str(
                positive_int(
                    row.get("game_id"),
                    label=f"schedule game_id line {line_number}",
                )
            )

            if game_id in games:
                raise FinalizePredictionValidationError(
                    f"Duplicate target game_id in weekly schedule: {game_id}"
                )

            home_team = text(row.get("home_team"))
            away_team = text(row.get("away_team"))

            if not home_team or not away_team:
                raise FinalizePredictionValidationError(
                    f"Blank team identity for game_id={game_id}"
                )

            game_date = validate_date(
                row.get("game_date"),
                label=f"game_date for game_id={game_id}",
            )

            game_time = validate_time(
                row.get("game_time"),
                label=f"game_time for game_id={game_id}",
            )

            odds_available = binary_int(
                row.get("odds_available"),
                label=f"odds_available for game_id={game_id}",
            )

            total_text = text(row.get("total"))

            if total_text:
                total_value = finite_decimal(
                    total_text,
                    label=f"total for game_id={game_id}",
                )

                if total_value <= 0:
                    raise FinalizePredictionValidationError(
                        f"total must be positive for game_id={game_id}: "
                        f"{total_value}"
                    )

            games[game_id] = {
                "game_id": game_id,
                "game_date": game_date,
                "game_time": game_time,
                "home_team": home_team,
                "away_team": away_team,
                "game_name": f"{away_team} at {home_team}",
                "total": total_text,
                "odds_available": str(odds_available),
                "odds_missing_reason": text(
                    row.get("odds_missing_reason")
                ),
            }

    if not games:
        raise FinalizePredictionValidationError(
            "Target weekly schedule contains no games"
        )

    return games


def load_clean_rows(
    path: Path,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Target clean prediction file not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != OUT_HEADERS:
            raise FinalizePredictionValidationError(
                "Target clean prediction header mismatch: "
                f"expected={OUT_HEADERS}, actual={reader.fieldnames}"
            )

        rows: list[dict[str, str]] = []

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            if None in row:
                raise FinalizePredictionValidationError(
                    "Malformed clean prediction row at "
                    f"CSV line {line_number}"
                )

            rows.append({
                key: text(value)
                for key, value in row.items()
            })

    if not rows:
        raise FinalizePredictionValidationError(
            "Target clean prediction file contains no rows"
        )

    return rows


def _validate_clean_prediction_rows(
    rows: list[dict[str, str]],
    *,
    season: int,
    season_type: int,
    week: int,
    schedule: dict[str, dict[str, str]],
    seen: set[str],
    margins: dict[str, Decimal],
) -> None:
    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        if list(row.keys()) != OUT_HEADERS:
            raise FinalizePredictionValidationError(
                f"Clean prediction schema mismatch at row {row_number}"
            )

        row_target = (
            positive_int(
                row.get("season"),
                label=f"clean season row {row_number}",
            ),
            positive_int(
                row.get("season_type"),
                label=f"clean season_type row {row_number}",
            ),
            positive_int(
                row.get("week"),
                label=f"clean week row {row_number}",
            ),
        )

        if row_target != (
            season,
            season_type,
            week,
        ):
            raise FinalizePredictionValidationError(
                f"Clean prediction target mismatch at row {row_number}"
            )

        game_id = str(
            positive_int(
                row.get("game_id"),
                label=f"clean game_id row {row_number}",
            )
        )

        if game_id in seen:
            raise FinalizePredictionValidationError(
                f"Duplicate clean prediction game_id={game_id}"
            )

        target = schedule.get(game_id)

        if target is None:
            raise FinalizePredictionValidationError(
                f"Clean prediction contains foreign game_id={game_id}"
            )

        seen.add(game_id)

        for field in (
            "home_team",
            "away_team",
            "game_name",
        ):
            if text(row.get(field)) != target[field]:
                raise FinalizePredictionValidationError(
                    f"Clean prediction {field} mismatch for "
                    f"game_id={game_id}: expected={target[field]!r}, "
                    f"actual={text(row.get(field))!r}"
                )

        if text(row.get("sport")) != "football":
            raise FinalizePredictionValidationError(
                f"Unexpected sport for game_id={game_id}: "
                f"{row.get('sport')!r}"
            )

        if text(row.get("league")) != "college-football":
            raise FinalizePredictionValidationError(
                f"Unexpected league for game_id={game_id}: "
                f"{row.get('league')!r}"
            )

        for field in (
            "game_date",
            "game_time",
            "away_projected_pts",
            "home_projected_pts",
            "total_projected_pts",
        ):
            if text(row.get(field)):
                raise FinalizePredictionValidationError(
                    f"Clean-stage {field} must be blank "
                    f"for game_id={game_id}"
                )

        percentage(
            row.get("matchupQuality"),
            label=f"matchupQuality for game_id={game_id}",
        )

        home_prob = probability(
            row.get("home_prob"),
            label=f"home_prob for game_id={game_id}",
        )

        away_prob = probability(
            row.get("away_prob"),
            label=f"away_prob for game_id={game_id}",
        )

        if (
            home_prob
            + away_prob
            != Decimal("1.0000")
        ):
            raise FinalizePredictionValidationError(
                "home_prob + away_prob must equal 1.0000 for "
                f"game_id={game_id}: home={home_prob}, "
                f"away={away_prob}"
            )

        tie_text = text(
            row.get("tie_prob")
        )

        if tie_text:
            probability(
                tie_text,
                label=f"tie_prob for game_id={game_id}",
            )

        home_ptdiff = finite_decimal(
            row.get("home_PtDiff"),
            label=f"home_PtDiff for game_id={game_id}",
        )

        away_ptdiff = finite_decimal(
            row.get("away_PtDiff"),
            label=f"away_PtDiff for game_id={game_id}",
        )

        mismatch = abs(
            home_ptdiff
            + away_ptdiff
        )

        if (
            mismatch
            > ESPN_MARGIN_SYMMETRY_TOLERANCE
        ):
            raise FinalizePredictionValidationError(
                "ESPN point-differential asymmetry exceeds "
                f"tolerance for game_id={game_id}: "
                f"home={home_ptdiff}, away={away_ptdiff}, "
                f"mismatch={mismatch}, "
                f"tolerance={ESPN_MARGIN_SYMMETRY_TOLERANCE}"
            )

        finite_decimal(
            row.get("home_rating"),
            label=f"home_rating for game_id={game_id}",
        )

        finite_decimal(
            row.get("away_rating"),
            label=f"away_rating for game_id={game_id}",
        )

        margins[game_id] = mismatch


def validate_clean_rows(
    rows: list[dict[str, str]],
    *,
    season: int,
    season_type: int,
    week: int,
    schedule: dict[str, dict[str, str]],
) -> dict[str, Decimal]:
    if len(rows) != len(schedule):
        raise FinalizePredictionValidationError(
            "Clean prediction row-count mismatch: "
            f"expected={len(schedule)}, actual={len(rows)}"
        )

    seen: set[str] = set()
    margins: dict[str, Decimal] = {}

    _validate_clean_prediction_rows(
        rows,
        season=season,
        season_type=season_type,
        week=week,
        schedule=schedule,
        seen=seen,
        margins=margins,
    )

    if seen != set(schedule):
        raise FinalizePredictionValidationError(
            "Clean prediction game coverage does not "
            "match target schedule"
        )

    return margins


def build_final_rows(
    clean_rows: list[dict[str, str]],
    *,
    schedule: dict[str, dict[str, str]],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    int,
]:
    final_rows: list[dict[str, str]] = []
    missing_totals: list[dict[str, str]] = []
    projected_score_games = 0

    for row in clean_rows:
        game_id = text(
            row.get("game_id")
        )

        target = schedule[
            game_id
        ]

        rec = {
            header: text(
                row.get(header)
            )
            for header in OUT_HEADERS
        }

        rec["game_date"] = target[
            "game_date"
        ]

        rec["game_time"] = target[
            "game_time"
        ]

        total_text = target[
            "total"
        ]

        if not total_text:
            rec["total_projected_pts"] = ""
            rec["home_projected_pts"] = ""
            rec["away_projected_pts"] = ""

            missing_totals.append({
                "game_id": game_id,
                "odds_available": target[
                    "odds_available"
                ],
                "odds_missing_reason": (
                    target[
                        "odds_missing_reason"
                    ]
                    or "total_market_unavailable"
                ),
            })

        else:
            total = finite_decimal(
                total_text,
                label=(
                    f"total for game_id={game_id}"
                ),
            )

            home_ptdiff = finite_decimal(
                rec["home_PtDiff"],
                label=(
                    "home_PtDiff for "
                    f"game_id={game_id}"
                ),
            )

            away_ptdiff = finite_decimal(
                rec["away_PtDiff"],
                label=(
                    "away_PtDiff for "
                    f"game_id={game_id}"
                ),
            )

            home_projected = (
                total + home_ptdiff
            ) / Decimal("2")

            away_projected = (
                total + away_ptdiff
            ) / Decimal("2")

            sum_diff = abs(
                (
                    home_projected
                    + away_projected
                )
                - total
            )

            margin_diff = abs(
                (
                    home_projected
                    - away_projected
                )
                - home_ptdiff
            )

            if (
                sum_diff
                > ESPN_MARGIN_SYMMETRY_TOLERANCE
                or margin_diff
                > ESPN_MARGIN_SYMMETRY_TOLERANCE
            ):
                raise FinalizePredictionValidationError(
                    "Projected-score consistency failure "
                    f"for game_id={game_id}: "
                    f"sum_diff={sum_diff}, "
                    f"margin_diff={margin_diff}"
                )

            rec[
                "total_projected_pts"
            ] = fmt2(total)

            rec[
                "home_projected_pts"
            ] = fmt2(
                home_projected
            )

            rec[
                "away_projected_pts"
            ] = fmt2(
                away_projected
            )

            projected_score_games += 1

        final_rows.append(
            rec
        )

    return (
        final_rows,
        missing_totals,
        projected_score_games,
    )



def _require_final_prediction_row_count(
    rows: list[dict[str, str]],
    schedule: dict[str, dict[str, str]],
) -> None:
    if len(rows) != len(schedule):
        raise FinalizePredictionValidationError(
            "Final prediction row-count mismatch: "
            f"expected={len(schedule)}, actual={len(rows)}"
        )


def _validate_final_prediction_schema(
    row: dict[str, str],
    *,
    row_number: int,
) -> None:
    if list(row.keys()) != OUT_HEADERS:
        raise FinalizePredictionValidationError(
            f"Final prediction schema mismatch at row {row_number}"
        )


def _validate_final_prediction_identity(
    row: dict[str, str],
    *,
    game_id: str,
) -> None:
    if (
        text(row.get("sport"))
        != "football"
        or text(row.get("league"))
        != "college-football"
    ):
        raise FinalizePredictionValidationError(
            "Final sport/league mismatch "
            f"for game_id={game_id}"
        )


def _validate_final_prediction_coverage(
    seen: set[str],
    schedule: dict[str, dict[str, str]],
) -> None:
    if seen != set(schedule):
        raise FinalizePredictionValidationError(
            "Final prediction game coverage does not "
            "match target schedule"
        )


def validate_final_rows(
    rows: list[dict[str, str]],
    *,
    season: int,
    season_type: int,
    week: int,
    schedule: dict[str, dict[str, str]],
) -> None:
    _require_final_prediction_row_count(
        rows,
        schedule,
    )

    seen: set[str] = set()

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        _validate_final_prediction_schema(
            row,
            row_number=row_number,
        )

        game_id = str(
            positive_int(
                row.get("game_id"),
                label=f"final game_id row {row_number}",
            )
        )

        if game_id in seen:
            raise FinalizePredictionValidationError(
                f"Duplicate final prediction game_id={game_id}"
            )

        target = schedule.get(
            game_id
        )

        if target is None:
            raise FinalizePredictionValidationError(
                f"Final prediction contains foreign game_id={game_id}"
            )

        seen.add(
            game_id
        )

        row_target = (
            positive_int(
                row.get("season"),
                label=f"final season {game_id}",
            ),
            positive_int(
                row.get("season_type"),
                label=f"final season_type {game_id}",
            ),
            positive_int(
                row.get("week"),
                label=f"final week {game_id}",
            ),
        )

        if row_target != (
            season,
            season_type,
            week,
        ):
            raise FinalizePredictionValidationError(
                f"Final prediction target mismatch for game_id={game_id}"
            )

        for field in (
            "home_team",
            "away_team",
            "game_name",
            "game_date",
            "game_time",
        ):
            if text(row.get(field)) != target[field]:
                raise FinalizePredictionValidationError(
                    f"Final prediction {field} mismatch "
                    f"for game_id={game_id}"
                )

        percentage(
            row.get("matchupQuality"),
            label=(
                "final matchupQuality for "
                f"game_id={game_id}"
            ),
        )

        home_prob = probability(
            row.get("home_prob"),
            label=(
                f"final home_prob for game_id={game_id}"
            ),
        )

        away_prob = probability(
            row.get("away_prob"),
            label=(
                f"final away_prob for game_id={game_id}"
            ),
        )

        if (
            home_prob
            + away_prob
            != Decimal("1.0000")
        ):
            raise FinalizePredictionValidationError(
                "Final probabilities do not sum "
                f"to 1.0000 for game_id={game_id}"
            )

        tie_text = text(
            row.get("tie_prob")
        )

        if tie_text:
            probability(
                tie_text,
                label=(
                    f"final tie_prob for game_id={game_id}"
                ),
            )

        home_ptdiff = finite_decimal(
            row.get("home_PtDiff"),
            label=(
                f"final home_PtDiff for game_id={game_id}"
            ),
        )

        away_ptdiff = finite_decimal(
            row.get("away_PtDiff"),
            label=(
                f"final away_PtDiff for game_id={game_id}"
            ),
        )

        if (
            abs(
                home_ptdiff
                + away_ptdiff
            )
            > ESPN_MARGIN_SYMMETRY_TOLERANCE
        ):
            raise FinalizePredictionValidationError(
                "Final point-differential asymmetry "
                f"for game_id={game_id}"
            )

        finite_decimal(
            row.get("home_rating"),
            label=(
                f"final home_rating for game_id={game_id}"
            ),
        )

        finite_decimal(
            row.get("away_rating"),
            label=(
                f"final away_rating for game_id={game_id}"
            ),
        )

        total_text = target[
            "total"
        ]

        if not total_text:
            for field in (
                "total_projected_pts",
                "home_projected_pts",
                "away_projected_pts",
            ):
                if text(row.get(field)):
                    raise FinalizePredictionValidationError(
                        f"Final {field} must be blank "
                        "without market total for "
                        f"game_id={game_id}"
                    )

        else:
            total = finite_decimal(
                total_text,
                label=(
                    "schedule total for "
                    f"game_id={game_id}"
                ),
            )

            expected_total = fmt2(
                total
            )

            expected_home = fmt2(
                (
                    total
                    + home_ptdiff
                )
                / Decimal("2")
            )

            expected_away = fmt2(
                (
                    total
                    + away_ptdiff
                )
                / Decimal("2")
            )

            for field, expected in (
                (
                    "total_projected_pts",
                    expected_total,
                ),
                (
                    "home_projected_pts",
                    expected_home,
                ),
                (
                    "away_projected_pts",
                    expected_away,
                ),
            ):
                if text(
                    row.get(field)
                ) != expected:
                    raise FinalizePredictionValidationError(
                        f"Final {field} mismatch "
                        f"for game_id={game_id}: "
                        f"expected={expected}, "
                        f"actual={text(row.get(field))!r}"
                    )

        _validate_final_prediction_identity(
            row,
            game_id=game_id,
        )

    _validate_final_prediction_coverage(
        seen,
        schedule,
    )


def read_staged_rows(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames != OUT_HEADERS:
            raise FinalizePredictionValidationError(
                "Staged final output header mismatch: "
                f"{reader.fieldnames}"
            )

        rows: list[dict[str, str]] = []

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            if None in row:
                raise FinalizePredictionValidationError(
                    "Malformed staged final row "
                    f"at CSV line {line_number}"
                )

            rows.append({
                key: text(value)
                for key, value in row.items()
            })

    return rows


def publish_atomic(
    rows: list[dict[str, str]],
    path: Path,
    *,
    season: int,
    season_type: int,
    week: int,
    schedule: dict[str, dict[str, str]],
) -> tuple[bool, int]:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with temp_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=OUT_HEADERS,
                extrasaction="raise",
            )

            writer.writeheader()
            writer.writerows(rows)

            handle.flush()
            os.fsync(
                handle.fileno()
            )

        staged_rows = read_staged_rows(
            temp_path
        )

        validate_final_rows(
            staged_rows,
            season=season,
            season_type=season_type,
            week=week,
            schedule=schedule,
        )

        output_modified = (
            not path.exists()
            or path.read_bytes()
            != temp_path.read_bytes()
        )

        if output_modified:
            os.replace(
                temp_path,
                path,
            )
        else:
            temp_path.unlink()

        return (
            output_modified,
            len(staged_rows),
        )

    except Exception:
        try:
            temp_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        raise


def main() -> None:
    with PipelineReporter(
        script=__file__,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        extra_context={
            "script_version": SCRIPT_VERSION,
            "source": (
                "configured clean ESPN predictions "
                "+ weekly schedule"
            ),
        },
    ) as report:
        report.add_input(
            CONFIG_PATH
        )

        (
            season,
            season_type,
            week,
        ) = load_config()

        report.season = season
        report.week = week

        (
            clean_path,
            schedule_path,
            output_path,
        ) = target_paths(
            season,
            season_type,
            week,
        )

        report.add_input(
            clean_path
        )
        report.add_input(
            schedule_path
        )
        report.add_output(
            output_path
        )

        report.update_details({
            "season_type": season_type,
            "target_clean_prediction_path": (
                clean_path
            ),
            "target_schedule_path": (
                schedule_path
            ),
            "output_path": output_path,
            "output_columns": OUT_HEADERS,
            "output_modified": False,
            "margin_symmetry_tolerance": str(
                ESPN_MARGIN_SYMMETRY_TOLERANCE
            ),
        })

        schedule = load_schedule(
            schedule_path,
            season=season,
            season_type=season_type,
            week=week,
        )

        clean_rows = load_clean_rows(
            clean_path
        )

        report.set_rows(
            rows_in=len(clean_rows)
        )

        margin_mismatches = (
            validate_clean_rows(
                clean_rows,
                season=season,
                season_type=season_type,
                week=week,
                schedule=schedule,
            )
        )

        (
            final_rows,
            missing_totals,
            projected_score_games,
        ) = build_final_rows(
            clean_rows,
            schedule=schedule,
        )

        validate_final_rows(
            final_rows,
            season=season,
            season_type=season_type,
            week=week,
            schedule=schedule,
        )

        (
            output_modified,
            staged_rows,
        ) = publish_atomic(
            final_rows,
            output_path,
            season=season,
            season_type=season_type,
            week=week,
            schedule=schedule,
        )

        report.set_rows(
            rows_out=staged_rows
        )

        report.update_details({
            "target_schedule_rows": (
                len(schedule)
            ),
            "clean_prediction_rows": (
                len(clean_rows)
            ),
            "final_rows_expected": (
                len(schedule)
            ),
            "final_rows_produced": (
                len(final_rows)
            ),
            "staged_rows_validated": (
                staged_rows
            ),
            "games_with_projected_scores": (
                projected_score_games
            ),
            "missing_total_count": (
                len(missing_totals)
            ),
            "missing_total_game_ids": [
                item["game_id"]
                for item in missing_totals
            ],
            "missing_total_details": (
                missing_totals
            ),
            "margin_consistency_failure_count": 0,
            "maximum_margin_asymmetry": str(
                max(
                    margin_mismatches.values(),
                    default=Decimal("0"),
                )
            ),
            "output_modified": (
                output_modified
            ),
        })

        if missing_totals:
            report.warning(
                "Market total unavailable for one or more "
                "target games; projected score fields "
                "left blank",
                count=len(
                    missing_totals
                ),
                game_ids=[
                    item["game_id"]
                    for item in missing_totals
                ],
            )

        print(
            "finalize_pred.py "
            f"version={SCRIPT_VERSION} "
            f"target={season}/{season_type}/{week} "
            f"rows={staged_rows} "
            f"missing_totals={len(missing_totals)} "
            f"output_modified={output_modified}"
        )


if __name__ == "__main__":
    main()

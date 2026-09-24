#!/usr/bin/env python3
"""
Clean ESPN predictor data for the configured CFB target week.

The configured season/type/week and weekly schedule are authoritative.
Validation failures are fatal and prevent partial or stale-target publication.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import uuid
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

import yaml

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter
from pipeline_shared import write_csv_rows_durable
from type_support import ScalarValue

CONFIG_PATH = CFB_ROOT / "config" / "current_week.yaml"
WEEKLY_SCHEDULE_DIR = CFB_ROOT / "00_intake" / "schedule" / "weekly"
RAW_DIR = CFB_ROOT / "00_intake" / "predictions" / "e_predictions"
OUTPUT_DIR = CFB_ROOT / "00_intake" / "predictions" / "clean"
REPORT_ROOT = CFB_ROOT / "errors"

SCRIPT_VERSION = "cfb-clean-e-pred-v2-2026-09-16"

OUT_HEADERS = [
    "game_id", "game_date", "game_time", "home_team", "away_team",
    "matchupQuality", "home_prob", "away_prob", "tie_prob",
    "away_projected_pts", "home_projected_pts", "total_projected_pts",
    "home_PtDiff", "away_PtDiff", "home_rating", "away_rating",
    "game_name", "season", "season_type", "week", "sport", "league",
]

RAW_REQUIRED_COLUMNS = {
    "season", "season_type", "week", "game_id", "game_name", "home_away",
    "team_id", "gameProjection", "matchupQuality", "oppSeasonStrengthFbsRank",
    "oppSeasonStrengthRating", "teamChanceLoss", "teamChanceTie",
    "teamPredPtDiff",
}

SCHEDULE_REQUIRED_COLUMNS = {
    "season", "season_type", "week", "game_id", "away_team", "home_team",
}

VALID_SIDES = {"homeTeam", "awayTeam"}
PROBABILITY_QUANTUM = Decimal("0.0001")
PERCENT_SUM_TOLERANCE = Decimal("0.000001")
TIE_MATCH_TOLERANCE = Decimal("0.000001")


class CleanPredictionValidationError(RuntimeError):
    pass


def text(value: ScalarValue) -> str:
    return "" if value is None else str(value).strip()


def normalize_name(value: ScalarValue) -> str:
    return " ".join(text(value).split()).casefold()


def positive_int(value: ScalarValue, *, label: str) -> int:
    value_text = text(value)

    if not re.fullmatch(r"\d+", value_text):
        raise CleanPredictionValidationError(
            f"{label} must be a positive integer: {value!r}"
        )

    parsed = int(value_text)

    if parsed <= 0:
        raise CleanPredictionValidationError(
            f"{label} must be positive: {parsed}"
        )

    return parsed


def finite_decimal(value: ScalarValue, *, label: str) -> Decimal:
    value_text = text(value)

    if not value_text:
        raise CleanPredictionValidationError(
            f"{label} is blank"
        )

    try:
        number = Decimal(value_text)
    except InvalidOperation as exc:
        raise CleanPredictionValidationError(
            f"{label} is not numeric: {value_text!r}"
        ) from exc

    if not number.is_finite():
        raise CleanPredictionValidationError(
            f"{label} is not finite: {value_text!r}"
        )

    return number


def percent_decimal(value: ScalarValue, *, label: str) -> Decimal:
    number = finite_decimal(
        value,
        label=label,
    )

    if not Decimal("0") <= number <= Decimal("100"):
        raise CleanPredictionValidationError(
            f"{label} must be between 0 and 100: {number}"
        )

    return number


def optional_percent(value: ScalarValue, *, label: str) -> Decimal | None:
    if not text(value):
        return None

    return percent_decimal(
        value,
        label=label,
    )


def quantized_probability(percent: Decimal) -> Decimal:
    return (
        percent / Decimal("100")
    ).quantize(
        PROBABILITY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def serialize_probability(percent: Decimal) -> str:
    return format(
        quantized_probability(percent),
        ".4f",
    )


def serialize_probability_pair(
    home_percent: Decimal,
    away_percent: Decimal,
) -> tuple[str, str, bool]:
    raw_sum_diff = abs(
        home_percent
        + away_percent
        - Decimal("100")
    )

    if raw_sum_diff > PERCENT_SUM_TOLERANCE:
        raise CleanPredictionValidationError(
            "Home/away gameProjection does not sum to 100 percent: "
            f"home={home_percent}, away={away_percent}, "
            f"difference={raw_sum_diff}"
        )

    home_exact = home_percent / Decimal("100")
    away_exact = away_percent / Decimal("100")

    home_prob = quantized_probability(
        home_percent
    )
    away_prob = quantized_probability(
        away_percent
    )

    delta = (
        Decimal("1.0000")
        - (home_prob + away_prob)
    )

    adjusted = delta != 0

    if adjusted:
        home_candidate = home_prob + delta
        away_candidate = away_prob + delta

        home_error = (
            abs(home_candidate - home_exact)
            + abs(away_prob - away_exact)
        )
        away_error = (
            abs(home_prob - home_exact)
            + abs(away_candidate - away_exact)
        )

        if home_error <= away_error:
            home_prob = home_candidate
        else:
            away_prob = away_candidate

    if not Decimal("0") <= home_prob <= Decimal("1"):
        raise CleanPredictionValidationError(
            "Rounded home probability outside [0, 1]: "
            f"{home_prob}"
        )

    if not Decimal("0") <= away_prob <= Decimal("1"):
        raise CleanPredictionValidationError(
            "Rounded away probability outside [0, 1]: "
            f"{away_prob}"
        )

    if home_prob + away_prob != Decimal("1.0000"):
        raise CleanPredictionValidationError(
            "Serialized home/away probabilities do not "
            "sum to 1.0000"
        )

    return (
        format(home_prob, ".4f"),
        format(away_prob, ".4f"),
        adjusted,
    )


def load_config() -> tuple[int, int, int]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing current-week config: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise CleanPredictionValidationError(
            "current_week.yaml must contain a mapping"
        )

    values: dict[str, int] = {}

    for key in (
        "season",
        "season_type",
        "week",
    ):
        if key not in payload:
            raise CleanPredictionValidationError(
                "current_week.yaml missing required key: "
                f"{key}"
            )

        values[key] = positive_int(
            payload.get(key),
            label=f"current_week.{key}",
        )

    if values["season"] < 2000:
        raise CleanPredictionValidationError(
            f"Invalid configured season: {values['season']}"
        )

    return (
        values["season"],
        values["season_type"],
        values["week"],
    )


def target_paths(
    season: int,
    season_type: int,
    week: int,
) -> tuple[Path, Path, Path]:
    schedule = (
        WEEKLY_SCHEDULE_DIR
        / f"week_{week}_CFB_weekly_schedule.csv"
    )

    raw = (
        RAW_DIR
        / f"{season}_{season_type}_{week}_e_predictions.csv"
    )

    output = (
        OUTPUT_DIR
        / f"{season}_{season_type}_{week}_predictions.csv"
    )

    return schedule, raw, output


def load_schedule(
    path: Path,
    *,
    season: int,
    season_type: int,
    week: int,
) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Target weekly schedule not found: {path}"
        )

    games: dict[str, dict[str, str]] = {}

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        missing = sorted(
            SCHEDULE_REQUIRED_COLUMNS
            - set(reader.fieldnames or [])
        )

        if missing:
            raise CleanPredictionValidationError(
                "Target weekly schedule missing required "
                f"columns: {missing}"
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            if None in row:
                raise CleanPredictionValidationError(
                    "Malformed weekly-schedule row at "
                    f"CSV line {line_number}"
                )

            row_season = positive_int(
                row.get("season"),
                label=(
                    "weekly schedule season at "
                    f"CSV line {line_number}"
                ),
            )

            row_type = positive_int(
                row.get("season_type"),
                label=(
                    "weekly schedule season_type at "
                    f"CSV line {line_number}"
                ),
            )

            row_week = positive_int(
                row.get("week"),
                label=(
                    "weekly schedule week at "
                    f"CSV line {line_number}"
                ),
            )

            if (
                row_season,
                row_type,
                row_week,
            ) != (
                season,
                season_type,
                week,
            ):
                raise CleanPredictionValidationError(
                    "Weekly schedule target mismatch at "
                    f"CSV line {line_number}: "
                    f"expected={season}/{season_type}/{week}, "
                    f"actual={row_season}/{row_type}/{row_week}"
                )

            game_id = str(
                positive_int(
                    row.get("game_id"),
                    label=(
                        "weekly schedule game_id at "
                        f"CSV line {line_number}"
                    ),
                )
            )

            if game_id in games:
                raise CleanPredictionValidationError(
                    "Duplicate target game_id in weekly schedule: "
                    f"{game_id}"
                )

            away_team = text(
                row.get("away_team")
            )
            home_team = text(
                row.get("home_team")
            )

            if not away_team or not home_team:
                raise CleanPredictionValidationError(
                    "Blank team identity for "
                    f"game_id={game_id}"
                )

            if (
                normalize_name(away_team)
                == normalize_name(home_team)
            ):
                raise CleanPredictionValidationError(
                    "Identical home/away team for "
                    f"game_id={game_id}"
                )

            games[game_id] = {
                "game_id": game_id,
                "away_team": away_team,
                "home_team": home_team,
                "game_name": (
                    f"{away_team} at {home_team}"
                ),
            }

    if not games:
        raise CleanPredictionValidationError(
            "Target weekly schedule contains no games"
        )

    return games


def load_raw_rows(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            "Target ESPN prediction file not found: "
            f"{path}"
        )

    rows: list[dict[str, str]] = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)

        missing = sorted(
            RAW_REQUIRED_COLUMNS
            - set(reader.fieldnames or [])
        )

        if missing:
            raise CleanPredictionValidationError(
                "Target ESPN prediction file missing "
                f"required columns: {missing}"
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            if None in row:
                raise CleanPredictionValidationError(
                    "Malformed ESPN prediction row at "
                    f"CSV line {line_number}"
                )

            rows.append({
                key: text(value)
                for key, value in row.items()
            })

    if not rows:
        raise CleanPredictionValidationError(
            "Target ESPN prediction file contains no rows"
        )

    return rows


def validate_raw_row(
    row: dict[str, str],
    *,
    line_number: int,
    season: int,
    season_type: int,
    week: int,
    schedule: dict[str, dict[str, str]],
) -> tuple[str, str]:
    row_target = (
        positive_int(
            row.get("season"),
            label=(
                f"season at CSV line {line_number}"
            ),
        ),
        positive_int(
            row.get("season_type"),
            label=(
                f"season_type at CSV line {line_number}"
            ),
        ),
        positive_int(
            row.get("week"),
            label=(
                f"week at CSV line {line_number}"
            ),
        ),
    )

    if row_target != (
        season,
        season_type,
        week,
    ):
        raise CleanPredictionValidationError(
            "Prediction target mismatch at CSV line "
            f"{line_number}: "
            f"expected={season}/{season_type}/{week}, "
            f"actual={row_target[0]}/"
            f"{row_target[1]}/{row_target[2]}"
        )

    game_id = str(
        positive_int(
            row.get("game_id"),
            label=(
                f"game_id at CSV line {line_number}"
            ),
        )
    )

    if game_id not in schedule:
        raise CleanPredictionValidationError(
            "Prediction contains foreign "
            f"game_id={game_id}"
        )

    side = text(
        row.get("home_away")
    )

    if side not in VALID_SIDES:
        raise CleanPredictionValidationError(
            "Invalid home_away for "
            f"game_id={game_id}: {side!r}"
        )

    positive_int(
        row.get("team_id"),
        label=(
            f"team_id for game_id={game_id} "
            f"side={side}"
        ),
    )

    expected_name = (
        schedule[game_id]["game_name"]
    )

    if (
        normalize_name(row.get("game_name"))
        != normalize_name(expected_name)
    ):
        raise CleanPredictionValidationError(
            "Prediction game_name mismatch for "
            f"game_id={game_id}: "
            f"expected={expected_name!r}, "
            f"actual={text(row.get('game_name'))!r}"
        )

    percent_decimal(
        row.get("gameProjection"),
        label=(
            "gameProjection for "
            f"game_id={game_id} side={side}"
        ),
    )

    percent_decimal(
        row.get("matchupQuality"),
        label=(
            "matchupQuality for "
            f"game_id={game_id} side={side}"
        ),
    )

    rank = finite_decimal(
        row.get("oppSeasonStrengthFbsRank"),
        label=(
            "oppSeasonStrengthFbsRank for "
            f"game_id={game_id} side={side}"
        ),
    )

    if rank < 0:
        raise CleanPredictionValidationError(
            "Negative oppSeasonStrengthFbsRank for "
            f"game_id={game_id} side={side}"
        )

    finite_decimal(
        row.get("oppSeasonStrengthRating"),
        label=(
            "oppSeasonStrengthRating for "
            f"game_id={game_id} side={side}"
        ),
    )

    percent_decimal(
        row.get("teamChanceLoss"),
        label=(
            "teamChanceLoss for "
            f"game_id={game_id} side={side}"
        ),
    )

    optional_percent(
        row.get("teamChanceTie"),
        label=(
            "teamChanceTie for "
            f"game_id={game_id} side={side}"
        ),
    )

    finite_decimal(
        row.get("teamPredPtDiff"),
        label=(
            "teamPredPtDiff for "
            f"game_id={game_id} side={side}"
        ),
    )

    return game_id, side


def build_clean_rows(
    raw_rows: list[dict[str, str]],
    *,
    season: int,
    season_type: int,
    week: int,
    schedule: dict[str, dict[str, str]],
) -> tuple[
    list[dict[str, str]],
    dict[str, int],
]:
    expected_raw_rows = len(schedule) * 2

    if len(raw_rows) != expected_raw_rows:
        raise CleanPredictionValidationError(
            "Target ESPN prediction row-count mismatch: "
            f"expected={expected_raw_rows}, "
            f"actual={len(raw_rows)}"
        )

    grouped: dict[
        str,
        dict[str, dict[str, str]],
    ] = {}

    game_order: list[str] = []
    home_side_count = 0
    away_side_count = 0

    for line_number, row in enumerate(
        raw_rows,
        start=2,
    ):
        game_id, side = validate_raw_row(
            row,
            line_number=line_number,
            season=season,
            season_type=season_type,
            week=week,
            schedule=schedule,
        )

        if game_id not in grouped:
            grouped[game_id] = {}
            game_order.append(game_id)

        sides = grouped[game_id]

        if side in sides:
            raise CleanPredictionValidationError(
                "Duplicate prediction side for "
                f"game_id={game_id}: {side}"
            )

        sides[side] = row

        home_side_count += int(
            side == "homeTeam"
        )
        away_side_count += int(
            side == "awayTeam"
        )

    expected_ids = set(schedule)
    actual_ids = set(grouped)

    if actual_ids != expected_ids:
        raise CleanPredictionValidationError(
            "Prediction game coverage mismatch: "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"extra={sorted(actual_ids - expected_ids)}"
        )

    clean_rows: list[
        dict[str, str]
    ] = []

    pair_adjustments = 0

    for game_id in game_order:
        target = schedule[game_id]
        sides = grouped[game_id]

        if set(sides) != VALID_SIDES:
            raise CleanPredictionValidationError(
                "Incomplete prediction sides for "
                f"game_id={game_id}: "
                f"actual={sorted(sides)}"
            )

        home = sides["homeTeam"]
        away = sides["awayTeam"]

        home_matchup = percent_decimal(
            home["matchupQuality"],
            label=(
                "home matchupQuality for "
                f"game_id={game_id}"
            ),
        )

        away_matchup = percent_decimal(
            away["matchupQuality"],
            label=(
                "away matchupQuality for "
                f"game_id={game_id}"
            ),
        )

        if home_matchup != away_matchup:
            raise CleanPredictionValidationError(
                "matchupQuality mismatch for "
                f"game_id={game_id}: "
                f"home={home_matchup}, "
                f"away={away_matchup}"
            )

        home_percent = percent_decimal(
            home["gameProjection"],
            label=(
                "home gameProjection for "
                f"game_id={game_id}"
            ),
        )

        away_percent = percent_decimal(
            away["gameProjection"],
            label=(
                "away gameProjection for "
                f"game_id={game_id}"
            ),
        )

        try:
            (
                home_prob,
                away_prob,
                adjusted,
            ) = serialize_probability_pair(
                home_percent,
                away_percent,
            )

        except CleanPredictionValidationError as exc:
            raise CleanPredictionValidationError(
                f"{exc} for game_id={game_id}"
            ) from exc

        pair_adjustments += int(
            adjusted
        )

        home_tie = optional_percent(
            home.get("teamChanceTie"),
            label=(
                "home teamChanceTie for "
                f"game_id={game_id}"
            ),
        )

        away_tie = optional_percent(
            away.get("teamChanceTie"),
            label=(
                "away teamChanceTie for "
                f"game_id={game_id}"
            ),
        )

        if (
            (home_tie is None)
            != (away_tie is None)
        ):
            raise CleanPredictionValidationError(
                "teamChanceTie presence mismatch for "
                f"game_id={game_id}: "
                f"home={home_tie}, away={away_tie}"
            )

        tie_prob = ""

        if (
            home_tie is not None
            and away_tie is not None
        ):
            if (
                abs(home_tie - away_tie)
                > TIE_MATCH_TOLERANCE
            ):
                raise CleanPredictionValidationError(
                    "teamChanceTie mismatch for "
                    f"game_id={game_id}: "
                    f"home={home_tie}, "
                    f"away={away_tie}"
                )

            home_tie_prob = (
                serialize_probability(
                    home_tie
                )
            )

            away_tie_prob = (
                serialize_probability(
                    away_tie
                )
            )

            if home_tie_prob != away_tie_prob:
                raise CleanPredictionValidationError(
                    "Serialized teamChanceTie mismatch for "
                    f"game_id={game_id}"
                )

            tie_prob = home_tie_prob

        home_ptdiff = text(
            home["teamPredPtDiff"]
        )

        away_ptdiff = text(
            away["teamPredPtDiff"]
        )

        home_rating = text(
            away["oppSeasonStrengthRating"]
        )

        away_rating = text(
            home["oppSeasonStrengthRating"]
        )

        clean_rows.append({
            "game_id": game_id,
            "game_date": "",
            "game_time": "",
            "home_team": target["home_team"],
            "away_team": target["away_team"],
            "matchupQuality": text(
                home["matchupQuality"]
            ),
            "home_prob": home_prob,
            "away_prob": away_prob,
            "tie_prob": tie_prob,
            "away_projected_pts": "",
            "home_projected_pts": "",
            "total_projected_pts": "",
            "home_PtDiff": home_ptdiff,
            "away_PtDiff": away_ptdiff,
            "home_rating": home_rating,
            "away_rating": away_rating,
            "game_name": target["game_name"],
            "season": str(season),
            "season_type": str(
                season_type
            ),
            "week": str(week),
            "sport": "football",
            "league": "college-football",
        })

    return clean_rows, {
        "raw_unique_game_count": len(
            grouped
        ),
        "home_side_count": (
            home_side_count
        ),
        "away_side_count": (
            away_side_count
        ),
        "duplicate_game_side_count": 0,
        "malformed_row_count": 0,
        "incomplete_game_count": 0,
        "probability_consistency_failure_count": 0,
        "probability_pair_rounding_adjustment_count": (
            pair_adjustments
        ),
    }


def _validate_staged_output_row(
    row: dict[str, str],
    *,
    line_number: int,
    season: int,
    season_type: int,
    week: int,
    schedule: dict[str, dict[str, str]],
    seen: set[str],
) -> None:
    if None in row:
        raise CleanPredictionValidationError(
            "Malformed staged output row at "
            f"CSV line {line_number}"
        )

    game_id = str(
        positive_int(
            row.get("game_id"),
            label=(
                "staged game_id at "
                f"CSV line {line_number}"
            ),
        )
    )

    if game_id in seen:
        raise CleanPredictionValidationError(
            "Duplicate staged "
            f"game_id={game_id}"
        )

    if game_id not in schedule:
        raise CleanPredictionValidationError(
            "Foreign staged "
            f"game_id={game_id}"
        )

    seen.add(game_id)
    target = schedule[game_id]

    row_target = (
        positive_int(
            row.get("season"),
            label=(
                f"staged season {game_id}"
            ),
        ),
        positive_int(
            row.get("season_type"),
            label=(
                f"staged season_type {game_id}"
            ),
        ),
        positive_int(
            row.get("week"),
            label=(
                f"staged week {game_id}"
            ),
        ),
    )

    if row_target != (
        season,
        season_type,
        week,
    ):
        raise CleanPredictionValidationError(
            "Staged target mismatch for "
            f"game_id={game_id}"
        )

    for field, expected in (
        (
            "home_team",
            target["home_team"],
        ),
        (
            "away_team",
            target["away_team"],
        ),
        (
            "game_name",
            target["game_name"],
        ),
        (
            "sport",
            "football",
        ),
        (
            "league",
            "college-football",
        ),
    ):
        if text(row.get(field)) != expected:
            raise CleanPredictionValidationError(
                "Staged "
                f"{field} mismatch for "
                f"game_id={game_id}"
            )

    home_prob = finite_decimal(
        row.get("home_prob"),
        label=(
            "staged home_prob for "
            f"game_id={game_id}"
        ),
    )

    away_prob = finite_decimal(
        row.get("away_prob"),
        label=(
            "staged away_prob for "
            f"game_id={game_id}"
        ),
    )

    if not (
        Decimal("0")
        <= home_prob
        <= Decimal("1")
    ):
        raise CleanPredictionValidationError(
            "Staged home_prob outside [0,1] "
            f"for game_id={game_id}"
        )

    if not (
        Decimal("0")
        <= away_prob
        <= Decimal("1")
    ):
        raise CleanPredictionValidationError(
            "Staged away_prob outside [0,1] "
            f"for game_id={game_id}"
        )

    if (
        home_prob + away_prob
        != Decimal("1.0000")
    ):
        raise CleanPredictionValidationError(
            "Staged probabilities do not sum "
            "to 1.0000 for "
            f"game_id={game_id}"
        )

    tie_value = text(
        row.get("tie_prob")
    )

    if tie_value:
        tie_prob = finite_decimal(
            tie_value,
            label=(
                "staged tie_prob for "
                f"game_id={game_id}"
            ),
        )

        if not (
            Decimal("0")
            <= tie_prob
            <= Decimal("1")
        ):
            raise CleanPredictionValidationError(
                "Staged tie_prob outside [0,1] "
                f"for game_id={game_id}"
            )

    for field in (
        "matchupQuality",
        "home_PtDiff",
        "away_PtDiff",
        "home_rating",
        "away_rating",
    ):
        finite_decimal(
            row.get(field),
            label=(
                f"staged {field} for "
                f"game_id={game_id}"
            ),
        )

    for field in (
        "game_date",
        "game_time",
        "away_projected_pts",
        "home_projected_pts",
        "total_projected_pts",
    ):
        if text(row.get(field)):
            raise CleanPredictionValidationError(
                f"Staged {field} must be blank "
                f"for game_id={game_id}"
            )



def _validate_staged_output_rows(
    rows: list[dict[str, str]],
    *,
    season: int,
    season_type: int,
    week: int,
    schedule: dict[str, dict[str, str]],
    seen: set[str],
) -> None:
    for line_number, row in enumerate(
        rows,
        start=2,
    ):
        _validate_staged_output_row(
            row,
            line_number=line_number,
            season=season,
            season_type=season_type,
            week=week,
            schedule=schedule,
            seen=seen,
        )


def _validate_staged_output_row_count(
    rows: list[dict[str, str]],
    schedule: dict[str, dict[str, str]],
) -> None:
    if len(rows) != len(schedule):
        raise CleanPredictionValidationError(
            "Staged clean output row-count mismatch: "
            f"expected={len(schedule)}, "
            f"actual={len(rows)}"
        )


def _validate_staged_output_coverage(
    seen: set[str],
    schedule: dict[str, dict[str, str]],
) -> None:
    if seen != set(schedule):
        raise CleanPredictionValidationError(
            "Staged clean output game coverage mismatch"
        )


def validate_staged_output(
    path: Path,
    *,
    season: int,
    season_type: int,
    week: int,
    schedule: dict[str, dict[str, str]],
) -> int:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames != OUT_HEADERS:
            raise CleanPredictionValidationError(
                "Staged clean output header mismatch: "
                f"{reader.fieldnames}"
            )

        rows = list(reader)

    _validate_staged_output_row_count(
        rows,
        schedule,
    )

    seen: set[str] = set()

    _validate_staged_output_rows(
        rows,
        season=season,
        season_type=season_type,
        week=week,
        schedule=schedule,
        seen=seen,
    )

    _validate_staged_output_coverage(
        seen,
        schedule,
    )

    return len(rows)


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
        (
            f".{path.name}."
            f"{uuid.uuid4().hex}.tmp"
        )
    )

    try:
        write_csv_rows_durable(
            temp_path,
            rows,
            OUT_HEADERS,
        )

        staged_rows = (
            validate_staged_output(
                temp_path,
                season=season,
                season_type=season_type,
                week=week,
                schedule=schedule,
            )
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
            staged_rows,
        )

    except Exception:
        try:
            temp_path.unlink(
                missing_ok=True
            )
        except OSError:
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
                "validated ESPN predictor raw output"
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
            schedule_path,
            raw_path,
            output_path,
        ) = target_paths(
            season,
            season_type,
            week,
        )

        report.add_input(
            schedule_path
        )
        report.add_input(
            raw_path
        )
        report.add_output(
            output_path
        )

        report.update_details({
            "season_type": season_type,
            "target_schedule_path": (
                schedule_path
            ),
            "target_raw_path": raw_path,
            "output_path": output_path,
            "output_columns": OUT_HEADERS,
            "output_modified": False,
        })

        schedule = load_schedule(
            schedule_path,
            season=season,
            season_type=season_type,
            week=week,
        )

        expected_raw_rows = (
            len(schedule) * 2
        )

        report.update_details({
            "target_game_count": (
                len(schedule)
            ),
            "raw_rows_expected": (
                expected_raw_rows
            ),
        })

        raw_rows = load_raw_rows(
            raw_path
        )

        report.set_rows(
            rows_in=len(raw_rows)
        )

        report.set_detail(
            "raw_rows_read",
            len(raw_rows),
        )

        (
            clean_rows,
            metrics,
        ) = build_clean_rows(
            raw_rows,
            season=season,
            season_type=season_type,
            week=week,
            schedule=schedule,
        )

        report.update_details(
            metrics
        )

        report.update_details({
            "clean_rows_expected": (
                len(schedule)
            ),
            "clean_rows_produced": (
                len(clean_rows)
            ),
        })

        (
            output_modified,
            staged_rows,
        ) = publish_atomic(
            clean_rows,
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
            "staged_rows_validated": (
                staged_rows
            ),
            "output_modified": (
                output_modified
            ),
        })

        print(
            "clean_e_pred.py "
            f"version={SCRIPT_VERSION} "
            f"target={season}/{season_type}/{week} "
            f"raw_rows={len(raw_rows)} "
            f"games={staged_rows} "
            f"output_modified={output_modified}"
        )


if __name__ == "__main__":
    main()

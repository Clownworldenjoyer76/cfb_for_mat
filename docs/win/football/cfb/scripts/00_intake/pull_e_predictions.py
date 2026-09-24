#!/usr/bin/env python3
"""
pull_e_predictions.py

Pull ESPN predictor data for the configured CFB target week.

Inputs:
    docs/win/football/cfb/config/current_week.yaml
    docs/win/football/cfb/00_intake/schedule/weekly/
        week_{week}_CFB_weekly_schedule.csv

Source:
    https://sports.core.api.espn.com/v2/sports/football/
        leagues/college-football/events/{game_id}/
        competitions/{game_id}/predictor

Output:
    docs/win/football/cfb/00_intake/predictions/e_predictions/
        {season}_{season_type}_{week}_e_predictions.csv

The configured target week is authoritative. A target predictor request or
validation failure is fatal and prevents publication of a partial output.
"""

from __future__ import annotations

from http.client import HTTPException

import csv
import json
import math
import os
import re
import sys
import urllib.parse
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request

import yaml


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_ROOT = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from http_security import open_https
from pipeline_reporter import PipelineReporter
from type_support import ScalarValue


CONFIG_PATH = CFB_ROOT / "config" / "current_week.yaml"
WEEKLY_SCHEDULE_DIR = CFB_ROOT / "00_intake" / "schedule" / "weekly"
OUTPUT_DIR = (
    CFB_ROOT
    / "00_intake"
    / "predictions"
    / "e_predictions"
)
REPORT_ROOT = CFB_ROOT / "errors"

SCRIPT_VERSION = "cfb-espn-predictor-v2-2026-09-15"

PREDICTOR_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/football/"
    "leagues/college-football/events/{game_id}/"
    "competitions/{game_id}/predictor"
)

ESPN_CORE_HOST = "sports.core.api.espn.com"

TEAM_REF_PATTERN = re.compile(
    r"/teams/(\d+)(?:/|$)"
)

OUTPUT_HEADER = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_name",
    "home_away",
    "team_id",
    "gameProjection",
    "matchupQuality",
    "oppSeasonStrengthFbsRank",
    "oppSeasonStrengthRating",
    "teamChanceLoss",
    "teamChanceTie",
    "teamPredPtDiff",
]

REQUIRED_SCHEDULE_COLUMNS = {
    "season",
    "season_type",
    "week",
    "game_id",
    "away_team",
    "home_team",
}

REQUIRED_NUMERIC_STATS = {
    "gameProjection",
    "matchupQuality",
    "oppSeasonStrengthFbsRank",
    "oppSeasonStrengthRating",
    "teamChanceLoss",
    "teamPredPtDiff",
}

OUTPUT_STATS = [
    "gameProjection",
    "matchupQuality",
    "oppSeasonStrengthFbsRank",
    "oppSeasonStrengthRating",
    "teamChanceLoss",
    "teamChanceTie",
    "teamPredPtDiff",
]

UNRESOLVED_TEAM_NAMES = {
    "tbd",
    "tba",
    "to be determined",
}

@dataclass
class RuntimeState:
    request_count: int = 0
    request_success_count: int = 0
    request_failures: list[dict[str, ScalarValue]] = field(
        default_factory=list
    )
    predictor_response_count: int = 0
    complete_game_count: int = 0
    incomplete_details: list[dict[str, ScalarValue]] = field(
        default_factory=list
    )
    duplicate_game_side_count: int = 0
    duplicate_stat_name_count: int = 0


class PredictorValidationError(RuntimeError):
    pass


class PredictorRequestError(RuntimeError):
    pass


def parse_positive_int(
    value: ScalarValue,
    *,
    label: str,
) -> int:
    if isinstance(value, bool):
        raise PredictorValidationError(
            f"{label} must be a positive integer, not boolean"
        )

    text = str(
        value or ""
    ).strip()

    if not re.fullmatch(
        r"\d+",
        text,
    ):
        raise PredictorValidationError(
            f"{label} must be a positive integer: {value!r}"
        )

    parsed = int(text)

    if parsed <= 0:
        raise PredictorValidationError(
            f"{label} must be positive: {parsed}"
        )

    return parsed


def parse_positive_int_text(
    value: ScalarValue,
    *,
    label: str,
) -> str:
    return str(
        parse_positive_int(
            value,
            label=label,
        )
    )


def scalar_text(
    value: ScalarValue,
    *,
    label: str,
) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        (
            dict,
            list,
            tuple,
            set,
        ),
    ):
        raise PredictorValidationError(
            f"{label} must be scalar"
        )

    if isinstance(value, bool):
        raise PredictorValidationError(
            f"{label} must not be boolean"
        )

    return str(value).strip()


def finite_number(
    value: ScalarValue,
    *,
    label: str,
) -> float:
    text = scalar_text(
        value,
        label=label,
    )

    if not text:
        raise PredictorValidationError(
            f"{label} is blank"
        )

    try:
        number = float(text)
    except ValueError as exc:
        raise PredictorValidationError(
            f"{label} is not numeric: {text!r}"
        ) from exc

    if not math.isfinite(number):
        raise PredictorValidationError(
            f"{label} is not finite: {text!r}"
        )

    return number


def validate_percent(
    value: ScalarValue,
    *,
    label: str,
) -> float:
    number = finite_number(
        value,
        label=label,
    )

    if (
        number < 0.0
        or number > 100.0
    ):
        raise PredictorValidationError(
            f"{label} must be between 0 and 100: {number}"
        )

    return number


def normalize_name(
    value: ScalarValue,
) -> str:
    return " ".join(
        str(value or "").strip().split()
    ).casefold()


def load_config() -> tuple[int, int, int]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing current-week config: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = yaml.safe_load(
            handle
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "current_week.yaml must contain a mapping"
        )

    values: dict[str, int] = {}

    for key in (
        "season",
        "season_type",
        "week",
    ):
        if key not in payload:
            raise ValueError(
                f"current_week.yaml missing required key: {key}"
            )

        values[key] = parse_positive_int(
            payload.get(key),
            label=f"current_week.{key}",
        )

    if values["season"] < 2000:
        raise ValueError(
            f"Invalid configured season: {values['season']}"
        )

    return (
        values["season"],
        values["season_type"],
        values["week"],
    )


def target_schedule_path(
    week: int,
) -> Path:
    return (
        WEEKLY_SCHEDULE_DIR
        / f"week_{week}_CFB_weekly_schedule.csv"
    )


def output_path(
    *,
    season: int,
    season_type: int,
    week: int,
) -> Path:
    return (
        OUTPUT_DIR
        / (
            f"{season}_{season_type}_{week}_"
            "e_predictions.csv"
        )
    )


def unresolved_team_name(
    value: str,
) -> bool:
    return normalize_name(
        value
    ) in UNRESOLVED_TEAM_NAMES



def _require_target_games_path(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Target weekly schedule not found: {path}"
        )


def _load_target_game_rows(
    reader: csv.DictReader,
    games: dict[str, dict[str, str]],
    *,
    season: int,
    season_type: int,
    week: int,
) -> None:
    for line_number, row in enumerate(
        reader,
        start=2,
    ):
        if None in row:
            raise PredictorValidationError(
                "Malformed weekly-schedule row at "
                f"CSV line {line_number}"
            )

        row_season = parse_positive_int(
            row.get("season"),
            label=(
                "weekly schedule season at "
                f"CSV line {line_number}"
            ),
        )

        row_season_type = parse_positive_int(
            row.get("season_type"),
            label=(
                "weekly schedule season_type at "
                f"CSV line {line_number}"
            ),
        )

        row_week = parse_positive_int(
            row.get("week"),
            label=(
                "weekly schedule week at "
                f"CSV line {line_number}"
            ),
        )

        if (
            row_season != season
            or row_season_type != season_type
            or row_week != week
        ):
            raise PredictorValidationError(
                "Weekly schedule target mismatch at "
                f"CSV line {line_number}: "
                f"expected={season}/{season_type}/{week}, "
                f"actual={row_season}/"
                f"{row_season_type}/{row_week}"
            )

        game_id = parse_positive_int_text(
            row.get("game_id"),
            label=(
                "weekly schedule game_id at "
                f"CSV line {line_number}"
            ),
        )

        if game_id in games:
            raise PredictorValidationError(
                "Duplicate target game_id in weekly schedule: "
                f"{game_id}"
            )

        away_team = str(
            row.get("away_team") or ""
        ).strip()

        home_team = str(
            row.get("home_team") or ""
        ).strip()

        if not away_team:
            raise PredictorValidationError(
                "Blank away_team for "
                f"game_id={game_id}"
            )

        if not home_team:
            raise PredictorValidationError(
                "Blank home_team for "
                f"game_id={game_id}"
            )

        if unresolved_team_name(
            away_team
        ):
            raise PredictorValidationError(
                "Unresolved away_team for "
                f"game_id={game_id}: {away_team!r}"
            )

        if unresolved_team_name(
            home_team
        ):
            raise PredictorValidationError(
                "Unresolved home_team for "
                f"game_id={game_id}: {home_team!r}"
            )

        if (
            normalize_name(away_team)
            == normalize_name(home_team)
        ):
            raise PredictorValidationError(
                "Weekly schedule has identical home/away "
                f"team for game_id={game_id}"
            )

        games[game_id] = {
            "season": str(season),
            "season_type": str(
                season_type
            ),
            "week": str(week),
            "game_id": game_id,
            "away_team": away_team,
            "home_team": home_team,
        }


def _require_target_games(
    games: dict[str, dict[str, str]],
) -> None:
    if not games:
        raise PredictorValidationError(
            "Target weekly schedule contains no games"
        )


def load_target_games(
    path: Path,
    *,
    season: int,
    season_type: int,
    week: int,
) -> dict[str, dict[str, str]]:
    _require_target_games_path(path)

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        fieldnames = reader.fieldnames or []

        missing_columns = sorted(
            REQUIRED_SCHEDULE_COLUMNS
            - set(fieldnames)
        )

        if missing_columns:
            raise PredictorValidationError(
                "Target weekly schedule missing required columns: "
                f"{missing_columns}"
            )

        games: dict[
            str,
            dict[str, str],
        ] = {}

        _load_target_game_rows(
            reader,
            games,
            season=season,
            season_type=season_type,
            week=week,
        )

    _require_target_games(games)

    return games


def predictor_url(
    game_id: str,
) -> str:
    return PREDICTOR_URL_TEMPLATE.format(
        game_id=game_id
    )


def request_failure(
    *,
    state: RuntimeState,
    game_id: str,
    url: str,
    error: str,
    http_status: Optional[int] = None,
) -> PredictorRequestError:
    detail: dict[str, ScalarValue] = {
        "game_id": game_id,
        "url": url,
        "error": error,
    }

    if http_status is not None:
        detail[
            "http_status"
        ] = http_status

    state.request_failures.append(
        detail
    )

    return PredictorRequestError(
        f"Predictor request failed for game_id={game_id}: {error}"
    )


def fetch_predictor(
    game_id: str,
    *,
    state: RuntimeState,
    timeout: int = 20,
) -> dict:

    url = predictor_url(
        game_id
    )

    parsed = urllib.parse.urlparse(
        url
    )

    if (
        parsed.scheme != "https"
        or parsed.hostname != ESPN_CORE_HOST
    ):
        raise PredictorValidationError(
            f"Unexpected predictor URL: {url!r}"
        )

    state.request_count += 1

    request = Request(
        url,
        headers={
            "User-Agent": "cfb-espn-predictor/2.0",
            "Accept": "application/json",
        },
    )

    try:
        with open_https(
            request,
            allowed_hosts={ESPN_CORE_HOST},
            timeout=timeout,
        ) as response:
            status = int(
                response.status
            )

            body = (
                response.read()
                .decode("utf-8")
            )

    except HTTPError as exc:
        error_body = ""

        try:
            error_body = (
                exc.read()
                .decode(
                    "utf-8",
                    errors="replace",
                )
            )
        except (HTTPException, OSError, UnicodeError, ValueError):
            pass

        raise request_failure(
            state=state,
            game_id=game_id,
            url=url,
            http_status=exc.code,
            error=(
                error_body[:1000]
                or str(exc)
            ),
        ) from exc

    except URLError as exc:
        raise request_failure(
            state=state,
            game_id=game_id,
            url=url,
            error=str(exc),
        ) from exc

    except Exception as exc:
        raise request_failure(
            state=state,
            game_id=game_id,
            url=url,
            error=str(exc),
        ) from exc

    if (
        status < 200
        or status >= 300
    ):
        raise request_failure(
            state=state,
            game_id=game_id,
            url=url,
            http_status=status,
            error=(
                f"unexpected HTTP status {status}"
            ),
        )

    try:
        payload = json.loads(
            body
        )
    except Exception as exc:
        raise request_failure(
            state=state,
            game_id=game_id,
            url=url,
            http_status=status,
            error=(
                f"malformed JSON: {exc}"
            ),
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        raise request_failure(
            state=state,
            game_id=game_id,
            url=url,
            http_status=status,
            error=(
                "JSON root is not an object"
            ),
        )

    state.request_success_count += 1
    state.predictor_response_count += 1

    return payload


def extract_team_id(
    side_data: dict,
    *,
    game_id: str,
    side: str,
) -> str:
    team = side_data.get(
        "team"
    )

    if not isinstance(
        team,
        dict,
    ):
        raise PredictorValidationError(
            f"{side}.team is not an object "
            f"for game_id={game_id}"
        )

    ref = str(
        team.get("$ref") or ""
    ).strip()

    if not ref:
        raise PredictorValidationError(
            f"{side}.team.$ref is blank "
            f"for game_id={game_id}"
        )

    parsed = urllib.parse.urlparse(
        ref
    )

    if (
        parsed.scheme not in {
            "http",
            "https",
        }
        or parsed.hostname != ESPN_CORE_HOST
    ):
        raise PredictorValidationError(
            f"Unexpected ESPN team reference "
            f"for game_id={game_id}, side={side}: {ref!r}"
        )

    match = TEAM_REF_PATTERN.search(
        parsed.path
    )

    if not match:
        raise PredictorValidationError(
            "Could not extract team_id from ESPN team reference "
            f"for game_id={game_id}, side={side}: {ref!r}"
        )

    return parse_positive_int_text(
        match.group(1),
        label=(
            f"team_id for game_id={game_id}, side={side}"
        ),
    )


def parse_statistics(
    side_data: dict,
    *,
    game_id: str,
    side: str,
    state: RuntimeState,
) -> dict[str, str]:

    statistics = side_data.get(
        "statistics"
    )

    if not isinstance(
        statistics,
        list,
    ):
        raise PredictorValidationError(
            f"{side}.statistics is not a list "
            f"for game_id={game_id}"
        )

    stats: dict[str, str] = {}

    for stat_index, stat in enumerate(
        statistics
    ):
        if not isinstance(
            stat,
            dict,
        ):
            raise PredictorValidationError(
                "Predictor statistics contains non-object entry "
                f"for game_id={game_id}, side={side}, "
                f"stat_index={stat_index}"
            )

        name = str(
            stat.get("name") or ""
        ).strip()

        if not name:
            raise PredictorValidationError(
                "Predictor statistic has blank name "
                f"for game_id={game_id}, side={side}, "
                f"stat_index={stat_index}"
            )

        value = scalar_text(
            stat.get("value"),
            label=(
                f"{name} for game_id={game_id}, side={side}"
            ),
        )

        if name in stats:
            state.duplicate_stat_name_count += 1

            if stats[name] != value:
                raise PredictorValidationError(
                    "Conflicting duplicate predictor statistic "
                    f"for game_id={game_id}, side={side}, "
                    f"name={name!r}, "
                    f"first={stats[name]!r}, second={value!r}"
                )

            continue

        stats[
            name
        ] = value

    missing = sorted(
        name
        for name in REQUIRED_NUMERIC_STATS
        if not str(
            stats.get(name) or ""
        ).strip()
    )

    if missing:
        raise PredictorValidationError(
            "Predictor response missing required statistics "
            f"for game_id={game_id}, side={side}: {missing}"
        )

    validate_percent(
        stats["gameProjection"],
        label=(
            f"gameProjection for game_id={game_id}, side={side}"
        ),
    )

    validate_percent(
        stats["matchupQuality"],
        label=(
            f"matchupQuality for game_id={game_id}, side={side}"
        ),
    )

    rank = finite_number(
        stats[
            "oppSeasonStrengthFbsRank"
        ],
        label=(
            "oppSeasonStrengthFbsRank for "
            f"game_id={game_id}, side={side}"
        ),
    )

    if rank < 0:
        raise PredictorValidationError(
            "oppSeasonStrengthFbsRank must be non-negative "
            f"for game_id={game_id}, side={side}: {rank}"
        )

    finite_number(
        stats[
            "oppSeasonStrengthRating"
        ],
        label=(
            "oppSeasonStrengthRating for "
            f"game_id={game_id}, side={side}"
        ),
    )

    validate_percent(
        stats["teamChanceLoss"],
        label=(
            f"teamChanceLoss for game_id={game_id}, side={side}"
        ),
    )

    tie = str(
        stats.get(
            "teamChanceTie",
            "",
        )
        or ""
    ).strip()

    if tie:
        validate_percent(
            tie,
            label=(
                f"teamChanceTie for game_id={game_id}, side={side}"
            ),
        )

    finite_number(
        stats["teamPredPtDiff"],
        label=(
            f"teamPredPtDiff for game_id={game_id}, side={side}"
        ),
    )

    return stats


def validate_predictor_response(
    predictor: dict,
    *,
    target: dict[str, str],
    state: RuntimeState,
) -> list[dict[str, str]]:

    game_id = target[
        "game_id"
    ]

    game_name = str(
        predictor.get("name") or ""
    ).strip()

    if not game_name:
        raise PredictorValidationError(
            f"Predictor response has blank name for game_id={game_id}"
        )

    expected_name = (
        f"{target['away_team']} at {target['home_team']}"
    )

    if (
        normalize_name(game_name)
        != normalize_name(expected_name)
    ):
        raise PredictorValidationError(
            "Predictor matchup name does not match target schedule "
            f"for game_id={game_id}: "
            f"expected={expected_name!r}, actual={game_name!r}"
        )

    rows: list[
        dict[str, str]
    ] = []

    team_ids: set[str] = set()

    for side in (
        "homeTeam",
        "awayTeam",
    ):
        if side not in predictor:
            raise PredictorValidationError(
                f"Predictor response missing {side} "
                f"for game_id={game_id}"
            )

        side_data = predictor.get(
            side
        )

        if not isinstance(
            side_data,
            dict,
        ):
            raise PredictorValidationError(
                f"Predictor {side} is not an object "
                f"for game_id={game_id}"
            )

        team_id = extract_team_id(
            side_data,
            game_id=game_id,
            side=side,
        )

        if team_id in team_ids:
            raise PredictorValidationError(
                "Predictor response has duplicate team_id "
                f"for game_id={game_id}: {team_id}"
            )

        team_ids.add(
            team_id
        )

        stats = parse_statistics(
            side_data,
            game_id=game_id,
            side=side,
            state=state,
        )

        row = {
            "season": target[
                "season"
            ],
            "season_type": target[
                "season_type"
            ],
            "week": target[
                "week"
            ],
            "game_id": game_id,
            "game_name": game_name,
            "home_away": side,
            "team_id": team_id,
        }

        for stat_name in OUTPUT_STATS:
            row[
                stat_name
            ] = str(
                stats.get(
                    stat_name,
                    "",
                )
                or ""
            ).strip()

        rows.append(
            row
        )

    if len(rows) != 2:
        raise PredictorValidationError(
            "Predictor response did not produce exactly two rows "
            f"for game_id={game_id}"
        )

    state.complete_game_count += 1

    return rows



def _validate_output_row_count(
    rows: list[dict[str, str]],
    expected_rows: int,
) -> None:
    if len(rows) != expected_rows:
        raise PredictorValidationError(
            "Predictor output row-count mismatch: "
            f"expected={expected_rows}, actual={len(rows)}"
        )


def _validate_output_target_metadata(
    *,
    row_season: int,
    row_type: int,
    row_week: int,
    season: int,
    season_type: int,
    week: int,
    row_index: int,
) -> None:
    if (
        row_season != season
        or row_type != season_type
        or row_week != week
    ):
        raise PredictorValidationError(
            "Predictor output target metadata mismatch at "
            f"row_index={row_index}"
        )


def _validate_output_game_coverage(
    missing_ids: list[str],
    foreign_ids: list[str],
) -> None:
    if missing_ids or foreign_ids:
        raise PredictorValidationError(
            "Predictor output game coverage mismatch: "
            f"missing={missing_ids[:50]}, "
            f"foreign={foreign_ids[:50]}"
        )


def validate_output_rows(
    rows: list[dict[str, str]],
    *,
    targets: dict[str, dict[str, str]],
    season: int,
    season_type: int,
    week: int,
    state: RuntimeState,
) -> None:

    expected_rows = (
        len(targets)
        * 2
    )

    _validate_output_row_count(
        rows,
        expected_rows,
    )

    grouped: dict[
        str,
        dict[str, dict[str, str]],
    ] = {}

    for row_index, row in enumerate(
        rows
    ):
        if list(
            row.keys()
        ) != OUTPUT_HEADER:
            raise PredictorValidationError(
                "Predictor output schema mismatch at "
                f"row_index={row_index}"
            )

        row_season = parse_positive_int(
            row.get("season"),
            label=(
                "output season at "
                f"row_index={row_index}"
            ),
        )

        row_type = parse_positive_int(
            row.get("season_type"),
            label=(
                "output season_type at "
                f"row_index={row_index}"
            ),
        )

        row_week = parse_positive_int(
            row.get("week"),
            label=(
                "output week at "
                f"row_index={row_index}"
            ),
        )

        _validate_output_target_metadata(
            row_season=row_season,
            row_type=row_type,
            row_week=row_week,
            season=season,
            season_type=season_type,
            week=week,
            row_index=row_index,
        )

        game_id = parse_positive_int_text(
            row.get("game_id"),
            label=(
                "output game_id at "
                f"row_index={row_index}"
            ),
        )

        target = targets.get(
            game_id
        )

        if target is None:
            raise PredictorValidationError(
                "Predictor output contains foreign game_id: "
                f"{game_id}"
            )

        game_name = str(
            row.get("game_name") or ""
        ).strip()

        expected_name = (
            f"{target['away_team']} at {target['home_team']}"
        )

        if (
            normalize_name(game_name)
            != normalize_name(expected_name)
        ):
            raise PredictorValidationError(
                "Predictor output game_name mismatch "
                f"for game_id={game_id}"
            )

        side = str(
            row.get("home_away") or ""
        ).strip()

        if side not in {
            "homeTeam",
            "awayTeam",
        }:
            raise PredictorValidationError(
                "Predictor output has invalid home_away "
                f"for game_id={game_id}: {side!r}"
            )

        team_id = parse_positive_int_text(
            row.get("team_id"),
            label=(
                f"output team_id for game_id={game_id}, side={side}"
            ),
        )

        validate_percent(
            row.get("gameProjection"),
            label=(
                f"output gameProjection for game_id={game_id}, side={side}"
            ),
        )

        validate_percent(
            row.get("matchupQuality"),
            label=(
                f"output matchupQuality for game_id={game_id}, side={side}"
            ),
        )

        rank = finite_number(
            row.get(
                "oppSeasonStrengthFbsRank"
            ),
            label=(
                "output oppSeasonStrengthFbsRank for "
                f"game_id={game_id}, side={side}"
            ),
        )

        if rank < 0:
            raise PredictorValidationError(
                "Output oppSeasonStrengthFbsRank is negative "
                f"for game_id={game_id}, side={side}"
            )

        finite_number(
            row.get(
                "oppSeasonStrengthRating"
            ),
            label=(
                "output oppSeasonStrengthRating for "
                f"game_id={game_id}, side={side}"
            ),
        )

        validate_percent(
            row.get("teamChanceLoss"),
            label=(
                f"output teamChanceLoss for game_id={game_id}, side={side}"
            ),
        )

        tie = str(
            row.get(
                "teamChanceTie"
            )
            or ""
        ).strip()

        if tie:
            validate_percent(
                tie,
                label=(
                    f"output teamChanceTie for game_id={game_id}, side={side}"
                ),
            )

        finite_number(
            row.get("teamPredPtDiff"),
            label=(
                f"output teamPredPtDiff for game_id={game_id}, side={side}"
            ),
        )

        game_sides = grouped.setdefault(
            game_id,
            {},
        )

        if side in game_sides:
            state.duplicate_game_side_count += 1

            raise PredictorValidationError(
                "Duplicate predictor game/side row: "
                f"game_id={game_id}, side={side}"
            )

        game_sides[
            side
        ] = {
            "team_id": team_id,
        }

    output_game_ids = set(
        grouped
    )

    target_game_ids = set(
        targets
    )

    missing_ids = sorted(
        target_game_ids
        - output_game_ids,
        key=int,
    )

    foreign_ids = sorted(
        output_game_ids
        - target_game_ids,
        key=int,
    )

    _validate_output_game_coverage(
        missing_ids,
        foreign_ids,
    )

    for game_id, sides in (
        grouped.items()
    ):
        if set(
            sides
        ) != {
            "homeTeam",
            "awayTeam",
        }:
            raise PredictorValidationError(
                "Predictor output does not contain exactly "
                f"homeTeam/awayTeam for game_id={game_id}"
            )

        if (
            sides[
                "homeTeam"
            ][
                "team_id"
            ]
            == sides[
                "awayTeam"
            ][
                "team_id"
            ]
        ):
            raise PredictorValidationError(
                "Predictor output has identical home/away "
                f"team_id for game_id={game_id}"
            )


def temp_path_for(
    final_path: Path,
) -> Path:
    return final_path.with_name(
        f".{final_path.name}."
        f"{uuid.uuid4().hex}.tmp"
    )


def write_staged_csv(
    path: Path,
    *,
    rows: list[dict[str, str]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_HEADER,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

        handle.flush()
        os.fsync(
            handle.fileno()
        )


def read_csv_exact(
    path: Path,
) -> tuple[
    list[str],
    list[dict[str, str]],
]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        header = (
            reader.fieldnames
            or []
        )

        rows = list(
            reader
        )

    return (
        header,
        rows,
    )


def publish_atomic(
    final_path: Path,
    *,
    rows: list[dict[str, str]],
    targets: dict[str, dict[str, str]],
    season: int,
    season_type: int,
    week: int,
    state: RuntimeState,
) -> bool:
    final_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = temp_path_for(
        final_path
    )

    try:
        write_staged_csv(
            temp_path,
            rows=rows,
        )

        (
            staged_header,
            staged_rows,
        ) = read_csv_exact(
            temp_path
        )

        if staged_header != OUTPUT_HEADER:
            raise PredictorValidationError(
                "Serialized predictor header mismatch"
            )

        if staged_rows != rows:
            raise PredictorValidationError(
                "Serialized predictor rows do not exactly "
                "match validated rows"
            )

        validate_output_rows(
            staged_rows,
            targets=targets,
            season=season,
            season_type=season_type,
            week=week,
            state=state,
        )

        if (
            final_path.exists()
            and final_path.read_bytes()
            == temp_path.read_bytes()
        ):
            return False

        os.replace(
            temp_path,
            final_path,
        )

        return True

    finally:
        try:
            temp_path.unlink(
                missing_ok=True
            )
        except OSError:
            pass


def update_report(
    report: PipelineReporter,
    state: RuntimeState,
    *,
    schedule_path: Optional[Path],
    final_path: Optional[Path],
    target_count: int,
    rows: list[dict[str, str]],
    output_modified: Optional[bool],
) -> None:
    expected_rows = (
        target_count * 2
    )

    details: dict[str, object] = {
        "target_schedule_path": (
            str(schedule_path)
            if schedule_path is not None
            else ""
        ),
        "target_game_count": target_count,
        "espn_request_count": state.request_count,
        "espn_request_success_count": (
            state.request_success_count
        ),
        "espn_request_failure_count": len(
            state.request_failures
        ),
        "espn_request_failures": (
            state.request_failures
        ),
        "predictor_response_count": (
            state.predictor_response_count
        ),
        "complete_game_count": (
            state.complete_game_count
        ),
        "incomplete_predictor_count": len(
            state.incomplete_details
        ),
        "incomplete_predictor_details": (
            state.incomplete_details
        ),
        "raw_rows_expected": expected_rows,
        "raw_rows_produced": len(
            rows
        ),
        "duplicate_game_side_count": (
            state.duplicate_game_side_count
        ),
        "duplicate_stat_name_count": (
            state.duplicate_stat_name_count
        ),
        "output_columns": OUTPUT_HEADER,
        "output_path": (
            str(final_path)
            if final_path is not None
            else ""
        ),
    }

    if output_modified is not None:
        details[
            "output_modified"
        ] = output_modified

    report.update_details(
        details
    )


def run(
    report: PipelineReporter,
) -> int:
    state = RuntimeState()



    schedule_path: Optional[Path] = None
    final_path: Optional[Path] = None

    targets: dict[
        str,
        dict[str, str],
    ] = {}

    rows: list[
        dict[str, str]
    ] = []

    output_modified: Optional[bool] = None

    try:
        (
            season,
            season_type,
            week,
        ) = load_config()

        report.season = season
        report.week = week

        report.set_detail(
            "season_type",
            season_type,
        )

        schedule_path = (
            target_schedule_path(
                week
            )
        )

        final_path = output_path(
            season=season,
            season_type=season_type,
            week=week,
        )

        report.add_input(
            schedule_path
        )

        report.add_output(
            final_path
        )

        targets = load_target_games(
            schedule_path,
            season=season,
            season_type=season_type,
            week=week,
        )

        for game_id in sorted(
            targets,
            key=int,
        ):
            target = targets[
                game_id
            ]

            try:
                predictor = fetch_predictor(
                    game_id,
                    state=state,
                )

            except PredictorRequestError as exc:
                state.incomplete_details.append(
                    {
                        "game_id": game_id,
                        "kind": "request_failure",
                        "error": str(exc),
                    }
                )
                continue

            try:
                game_rows = (
                    validate_predictor_response(
                        predictor,
                        target=target,
                        state=state,
                    )
                )

            except PredictorValidationError as exc:
                state.incomplete_details.append(
                    {
                        "game_id": game_id,
                        "kind": "validation_failure",
                        "error": str(exc),
                    }
                )
                continue

            rows.extend(
                game_rows
            )

        rows.sort(
            key=lambda row: (
                int(
                    row["game_id"]
                ),
                0
                if row[
                    "home_away"
                ] == "homeTeam"
                else 1,
            )
        )

        report.set_rows(
            rows_in=len(targets),
            rows_out=len(rows),
        )

        if state.incomplete_details:
            failed_game_ids = [
                str(
                    detail[
                        "game_id"
                    ]
                )
                for detail
                in state.incomplete_details
            ]

            raise RuntimeError(
                "Target ESPN predictor collection is incomplete; "
                "refusing partial publication. "
                f"target_games={len(targets)}, "
                f"complete_games={state.complete_game_count}, "
                f"failed_games={len(state.incomplete_details)}, "
                f"game_ids={failed_game_ids[:50]}"
            )

        validate_output_rows(
            rows,
            targets=targets,
            season=season,
            season_type=season_type,
            week=week,
            state=state,
        )

        output_modified = (
            publish_atomic(
                final_path,
                rows=rows,
                targets=targets,
                season=season,
                season_type=season_type,
                week=week,
                state=state,
            )
        )

        update_report(
            report,
            state,
            schedule_path=schedule_path,
            final_path=final_path,
            target_count=len(
                targets
            ),
            rows=rows,
            output_modified=output_modified,
        )

        print(
            "ESPN predictor target complete: "
            f"season={season} "
            f"season_type={season_type} "
            f"week={week} "
            f"games={len(targets)} "
            f"rows={len(rows)} "
            f"output_modified={output_modified}"
        )

        return 0

    except Exception:
        report.set_rows(
            rows_in=len(
                targets
            ),
            rows_out=len(
                rows
            ),
        )

        update_report(
            report,
            state,
            schedule_path=schedule_path,
            final_path=final_path,
            target_count=len(
                targets
            ),
            rows=rows,
            output_modified=output_modified,
        )

        raise


def main() -> int:
    with PipelineReporter(
        script=SCRIPT_PATH,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        extra_context={
            "script_version": SCRIPT_VERSION,
            "source": "ESPN Core API predictor",
        },
    ) as report:
        report.add_input(
            CONFIG_PATH
        )

        return run(
            report
        )

    raise RuntimeError("context manager unexpectedly suppressed an exception")

if __name__ == "__main__":
    raise SystemExit(
        main()
    )

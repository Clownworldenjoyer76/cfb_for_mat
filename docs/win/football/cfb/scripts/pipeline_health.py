#!/usr/bin/env python3
"""CFB pipeline health reporter."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


SCRIPT_VERSION = "cfb-pipeline-health-v3-history-coverage-2026-09-16"

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
BASE = SCRIPTS_DIR.parent
REPO_ROOT = BASE.parents[3]

CONFIG = BASE / "config/current_week.yaml"
OUTPUT = BASE / "pipeline_health.json"
LOG = BASE / "errors/pipeline_health.txt"
REPORT_ROOT = BASE / "errors"
FRONTEND_OUTPUT = (
    REPO_ROOT / "frontend/data/pipeline_health/cfb.json"
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


NY = ZoneInfo("America/New_York")

JOB_STATUS_ENV = "CFB_PIPELINE_JOB_STATUS"
SEASON_ENV = "CFB_SEASON"
WEEK_ENV = "CFB_WEEK"

TRUE_FLAGS = {"1", "1.0", "true", "yes", "y"}
FALSE_FLAGS = {"0", "0.0", "false", "no", "n"}

CURRENT_WEEK_KEYS = (
    "weekly_schedule",
    "predictions",
    "merged",
    "candidates",
    "picks",
    "clean_picks",
    "all_games",
    "selected",
    "locked",
)

STAGE_NAMES = {
    "season_schedule": "Season Schedule",
    "weekly_schedule": "Weekly Schedule",
    "predictions": "Final Predictions",
    "merged": "Projection",
    "candidates": "Betting Candidates",
    "picks": "Market Picks",
    "clean_picks": "Clean Weekly Picks",
    "all_games": "All Games Picks",
    "selected": "Final Selected Picks",
    "locked": "Locked Selected Picks",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {
        "", "nan", "none", "null", "<na>", "nat"
    } else text


def clean_id(value: Any) -> str:
    text = clean(value)
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def parse_int(value: Any, label: str) -> int:
    text = clean(value)
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
            f"{label} must be a whole number; found {value!r}"
        )

    return int(number)


def strict_flag(value: Any, label: str) -> bool:
    text = clean(value).casefold()

    if text in TRUE_FLAGS:
        return True
    if text in FALSE_FLAGS:
        return False

    raise RuntimeError(
        f"{label} must be a valid 0/1 flag; found {value!r}"
    )


def load_target() -> tuple[str, int, int]:
    if not CONFIG.is_file():
        raise RuntimeError(f"Missing config: {CONFIG}")

    data = yaml.safe_load(
        CONFIG.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            f"{CONFIG} must contain a YAML mapping"
        )

    season = parse_int(data.get("season"), "config season")
    season_type = parse_int(
        data.get("season_type"),
        "config season_type",
    )
    week = parse_int(data.get("week"), "config week")

    if season < 1900 or season_type <= 0 or week <= 0:
        raise RuntimeError(
            "Configured season/season_type/week is invalid"
        )

    env_season = clean(os.getenv(SEASON_ENV))
    env_week = clean(os.getenv(WEEK_ENV))

    if env_season:
        parsed = parse_int(env_season, SEASON_ENV)
        if parsed != season:
            raise RuntimeError(
                f"{SEASON_ENV}={parsed} does not match config season={season}"
            )

    if env_week:
        parsed = parse_int(env_week, WEEK_ENV)
        if parsed != week:
            raise RuntimeError(
                f"{WEEK_ENV}={parsed} does not match config week={week}"
            )

    return str(season), season_type, week


def build_paths(
    season: str,
    season_type: int,
    week: int,
) -> dict[str, Path]:
    selected = (
        BASE
        / "03_picks/selected"
        / f"week_{week}_CFB_select_picks.csv"
    )

    return {
        "season_schedule":
            BASE / f"00_intake/schedule/{season}_schedule.csv",
        "weekly_schedule":
            BASE
            / "00_intake/schedule/weekly"
            / f"week_{week}_CFB_weekly_schedule.csv",
        "predictions":
            BASE
            / "00_intake/predictions/final"
            / f"{season}_{season_type}_{week}_clean_predictions.csv",
        "merged":
            BASE / f"01_merge/week_{week}_CFB_enriched.csv",
        "candidates":
            BASE / f"02_select/week_{week}_CFB_selected.csv",
        "picks":
            BASE / f"03_picks/week_{week}_CFB_picks.csv",
        "clean_picks":
            BASE
            / "03_picks/cleaned"
            / f"week_{week}_CFB_clean_picks.csv",
        "all_games":
            BASE
            / "03_picks/all_games"
            / f"all_week_{week}_CFB_picks.csv",
        "selected":
            selected,
        "locked":
            selected.parent
            / "locked"
            / selected.name,
    }


def read_csv_state(path: Path) -> dict[str, Any]:
    result = {
        "exists": path.is_file(),
        "fields": [],
        "rows": [],
        "error": "",
    }

    if not result["exists"]:
        return result

    if path.stat().st_size == 0:
        result["error"] = "empty file"
        return result

    try:
        with path.open(
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(
                handle,
                strict=True,
            )

            fields = list(reader.fieldnames or [])

            if not fields:
                raise RuntimeError("missing CSV header")

            rows = list(reader)

            for line, row in enumerate(rows, start=2):
                if (
                    None in row
                    or any(value is None for value in row.values())
                ):
                    raise RuntimeError(
                        f"malformed row at line {line}"
                    )

            result["fields"] = fields
            result["rows"] = rows

    except Exception as exc:
        result["error"] = str(exc)

    return result


def validate_stage(
    key: str,
    path: Path,
    season: str,
    season_type: int,
    week: int,
) -> tuple[dict[str, Any], list[dict]]:
    state = read_csv_state(path)

    stage = {
        "name": STAGE_NAMES[key],
        "path": str(path),
        "exists": state["exists"],
        "status": "STATUS: MISSING",
    }

    if not state["exists"]:
        return stage, []

    if state["error"]:
        stage["status"] = "STATUS: INVALID"
        stage["error"] = state["error"]
        return stage, []

    fields = set(state["fields"])
    rows = state["rows"]

    required = {"season", "game_id"}

    if key != "all_games":
        required.add("season_type")

    if key != "season_schedule":
        required.add("week")

    if key == "weekly_schedule":
        required.update(
            {"away_team", "home_team", "odds_available"}
        )

    if key in {"selected", "locked"}:
        required.update(
            {"ml_selected", "spread_selected", "total_selected"}
        )

    missing = required - fields

    if missing:
        stage["status"] = "STATUS: INVALID"
        stage["error"] = (
            "missing required columns: "
            + ", ".join(sorted(missing))
        )
        return stage, []

    allow_header_only = key in {"selected", "locked"}

    if not rows and not allow_header_only:
        stage["status"] = "STATUS: INVALID"
        stage["error"] = "contains no data rows"
        return stage, []

    seen: set[str] = set()

    try:
        for line, row in enumerate(rows, start=2):
            row_season = parse_int(
                row.get("season"),
                f"{path} line {line}: season",
            )
            if row_season != int(season):
                raise RuntimeError(
                    f"wrong season at line {line}"
                )

            if key != "all_games":
                row_type = parse_int(
                    row.get("season_type"),
                    f"{path} line {line}: season_type",
                )

                if row_type != season_type:
                    raise RuntimeError(
                        f"wrong season_type at line {line}"
                    )

            if key != "season_schedule":
                row_week = parse_int(
                    row.get("week"),
                    f"{path} line {line}: week",
                )
                if row_week != week:
                    raise RuntimeError(
                        f"wrong week at line {line}"
                    )

            game_id = clean_id(row.get("game_id"))

            if not game_id:
                raise RuntimeError(
                    f"blank game_id at line {line}"
                )

            if game_id in seen:
                raise RuntimeError(
                    f"duplicate game_id {game_id}"
                )

            seen.add(game_id)

            if key == "weekly_schedule":
                if (
                    not clean(row.get("away_team"))
                    or not clean(row.get("home_team"))
                ):
                    raise RuntimeError(
                        f"blank team identity at line {line}"
                    )

                strict_flag(
                    row.get("odds_available"),
                    f"{path} line {line}: odds_available",
                )

            if key in {"selected", "locked"}:
                for flag in (
                    "ml_selected",
                    "spread_selected",
                    "total_selected",
                ):
                    strict_flag(
                        row.get(flag),
                        f"{path} line {line}: {flag}",
                    )

    except Exception as exc:
        stage["status"] = "STATUS: INVALID"
        stage["error"] = str(exc)
        return stage, rows

    stage["status"] = "STATUS: SUCCESS"
    stage["rows"] = len(rows)

    return stage, rows


def ids(rows: list[dict]) -> set[str]:
    return {
        game_id
        for row in rows
        if (game_id := clean_id(row.get("game_id")))
    }


def team_mismatches(
    schedule_rows: list[dict],
    rows: list[dict],
) -> list[str]:
    schedule = {
        clean_id(row.get("game_id")): (
            clean(row.get("away_team")),
            clean(row.get("home_team")),
        )
        for row in schedule_rows
        if clean_id(row.get("game_id"))
    }

    mismatches = []

    for row in rows:
        game_id = clean_id(row.get("game_id"))

        if game_id not in schedule:
            continue

        if "away_team" not in row or "home_team" not in row:
            continue

        away = clean(row.get("away_team"))
        home = clean(row.get("home_team"))

        if not away or not home:
            continue

        if (away, home) != schedule[game_id]:
            mismatches.append(game_id)

    return sorted(set(mismatches))


def selected_bet_count(
    rows: list[dict],
    label: str,
) -> int:
    total = 0

    for line, row in enumerate(rows, start=2):
        for flag in (
            "ml_selected",
            "spread_selected",
            "total_selected",
        ):
            if strict_flag(
                row.get(flag),
                f"{label} line {line}: {flag}",
            ):
                total += 1

    return total




def build_history_paths(
    season: str,
    season_type: int,
    week: int,
) -> dict[str, Any]:
    return {
        "pbp":
            BASE
            / "00_intake"
            / "pbp"
            / f"{season}_pbp.parquet",
        "team_stats":
            BASE
            / "00_intake"
            / "team_stats"
            / f"{season}_team_stats.csv",
        "results": [
            (
                BASE
                / "04_final_results"
                / "results"
                / (
                    f"{season}_{season_type}_"
                    f"{prior_week}.csv"
                )
            )
            for prior_week
            in range(
                1,
                week,
            )
        ],
    }


def _history_integer_column(
    frame: pd.DataFrame,
    column: str,
    label: str,
) -> pd.Series:
    if column not in frame.columns:
        raise RuntimeError(
            f"{label} missing required column: {column}"
        )

    values = pd.to_numeric(
        frame[
            column
        ],
        errors="coerce",
    )

    invalid = (
        values.isna()
        | values.mod(1).ne(0)
    )

    if invalid.any():
        examples = (
            frame.loc[
                invalid,
                column,
            ]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            f"{label} contains invalid integer "
            f"{column} values: {examples}"
        )

    return values.astype(
        int
    )


def _read_history_csv(
    path: Path,
    required: set[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing historical coverage input: {path}"
        )

    try:
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            low_memory=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read historical coverage input "
            f"{path}: {exc}"
        ) from exc

    missing = sorted(
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{path} missing required columns: {missing}"
        )

    return frame


def validate_historical_coverage(
    history_paths: dict[str, Any],
    *,
    season: str,
    season_type: int,
    target_week: int,
) -> dict[str, Any]:
    expected_completed: dict[
        str,
        int,
    ] = {}

    result_paths = list(
        history_paths[
            "results"
        ]
    )

    for expected_week, path in enumerate(
        result_paths,
        start=1,
    ):
        results = _read_history_csv(
            path,
            {
                "season",
                "season_type",
                "week",
                "game_id",
                "completed",
            },
        )

        result_season = _history_integer_column(
            results,
            "season",
            str(path),
        )

        result_type = _history_integer_column(
            results,
            "season_type",
            str(path),
        )

        result_week = _history_integer_column(
            results,
            "week",
            str(path),
        )

        if not result_season.eq(
            int(
                season
            )
        ).all():
            raise RuntimeError(
                f"{path}: wrong season"
            )

        if not result_type.eq(
            season_type
        ).all():
            raise RuntimeError(
                f"{path}: wrong season_type"
            )

        if not result_week.eq(
            expected_week
        ).all():
            raise RuntimeError(
                f"{path}: wrong week"
            )

        game_ids = results[
            "game_id"
        ].map(
            clean_id
        )

        if game_ids.eq(
            ""
        ).any():
            raise RuntimeError(
                f"{path}: blank game_id"
            )

        if game_ids.duplicated().any():
            duplicates = (
                game_ids[
                    game_ids.duplicated(
                        False
                    )
                ]
                .drop_duplicates()
                .head(10)
                .tolist()
            )

            raise RuntimeError(
                f"{path}: duplicate game_id values: "
                f"{duplicates}"
            )

        for line, (
            game_id,
            completed_raw,
        ) in enumerate(
            zip(
                game_ids,
                results[
                    "completed"
                ],
                strict=True,
            ),
            start=2,
        ):
            completed = strict_flag(
                completed_raw,
                (
                    f"{path} line "
                    f"{line}: completed"
                ),
            )

            if not completed:
                continue

            if game_id in expected_completed:
                raise RuntimeError(
                    "Completed game_id appears in multiple "
                    f"prior result weeks: {game_id}"
                )

            expected_completed[
                game_id
            ] = expected_week

    if target_week > 1 and not expected_completed:
        raise RuntimeError(
            "No completed games found in prior-week "
            "final-score results"
        )

    pbp_path = Path(
        history_paths[
            "pbp"
        ]
    )

    if not pbp_path.is_file():
        raise FileNotFoundError(
            f"Missing current-season PBP: {pbp_path}"
        )

    try:
        pbp = pd.read_parquet(
            pbp_path,
            columns=[
                "season",
                "week",
                "game_id",
                "homeTeamId",
                "awayTeamId",
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read PBP coverage input "
            f"{pbp_path}: {exc}"
        ) from exc

    if (
        target_week > 1
        and pbp.empty
    ):
        raise RuntimeError(
            f"PBP coverage input is empty: {pbp_path}"
        )

    pbp_game_ids: set[str] = set()
    expected_team_week_pairs: set[
        tuple[
            int,
            str,
        ]
    ] = set()

    if not pbp.empty:
        pbp_season = _history_integer_column(
            pbp,
            "season",
            str(pbp_path),
        )

        pbp_week = _history_integer_column(
            pbp,
            "week",
            str(pbp_path),
        )

        if not pbp_season.eq(
            int(
                season
            )
        ).all():
            raise RuntimeError(
                f"{pbp_path}: wrong PBP season"
            )

        if pbp_week.le(
            0
        ).any():
            raise RuntimeError(
                f"{pbp_path}: non-positive PBP week"
            )

        prior_pbp = pbp.loc[
            pbp_week.lt(
                target_week
            )
        ].copy()

        prior_pbp[
            "_history_week"
        ] = pbp_week.loc[
            prior_pbp.index
        ].astype(
            int
        )

        prior_pbp[
            "_history_game_id"
        ] = prior_pbp[
            "game_id"
        ].map(
            clean_id
        )

        if prior_pbp[
            "_history_game_id"
        ].eq(
            ""
        ).any():
            raise RuntimeError(
                f"{pbp_path}: blank prior PBP game_id"
            )

        for game_id, group in prior_pbp.groupby(
            "_history_game_id",
            sort=False,
        ):
            weeks = {
                int(
                    value
                )
                for value
                in group[
                    "_history_week"
                ].tolist()
            }

            home_ids = {
                clean_id(
                    value
                )
                for value
                in group[
                    "homeTeamId"
                ]
                if clean_id(
                    value
                )
            }

            away_ids = {
                clean_id(
                    value
                )
                for value
                in group[
                    "awayTeamId"
                ]
                if clean_id(
                    value
                )
            }

            if (
                len(
                    weeks
                ) != 1
                or len(
                    home_ids
                ) != 1
                or len(
                    away_ids
                ) != 1
            ):
                raise RuntimeError(
                    "PBP game identity is not stable for "
                    f"game_id={game_id}; "
                    f"weeks={sorted(weeks)}, "
                    f"home_ids={sorted(home_ids)}, "
                    f"away_ids={sorted(away_ids)}"
                )

            week_value = next(
                iter(
                    weeks
                )
            )

            home_id = next(
                iter(
                    home_ids
                )
            )

            away_id = next(
                iter(
                    away_ids
                )
            )

            if home_id == away_id:
                raise RuntimeError(
                    "PBP home/away team IDs are identical "
                    f"for game_id={game_id}"
                )

            pbp_game_ids.add(
                game_id
            )

            expected_team_week_pairs.add(
                (
                    week_value,
                    home_id,
                )
            )

            expected_team_week_pairs.add(
                (
                    week_value,
                    away_id,
                )
            )

    expected_game_ids = set(
        expected_completed
    )

    missing_pbp = sorted(
        expected_game_ids
        - pbp_game_ids
    )

    unexpected_pbp = sorted(
        pbp_game_ids
        - expected_game_ids
    )

    if (
        missing_pbp
        or unexpected_pbp
    ):
        raise RuntimeError(
            "Completed-game PBP coverage mismatch. "
            f"expected={len(expected_game_ids)} "
            f"actual={len(pbp_game_ids)} "
            f"missing_count={len(missing_pbp)} "
            f"unexpected_count={len(unexpected_pbp)} "
            f"missing_examples={missing_pbp[:10]} "
            f"unexpected_examples={unexpected_pbp[:10]}"
        )

    team_stats_path = Path(
        history_paths[
            "team_stats"
        ]
    )

    stats = _read_history_csv(
        team_stats_path,
        {
            "season",
            "week",
            "team",
        },
    )

    stats_season = _history_integer_column(
        stats,
        "season",
        str(
            team_stats_path
        ),
    )

    stats_week = _history_integer_column(
        stats,
        "week",
        str(
            team_stats_path
        ),
    )

    if not stats_season.eq(
        int(
            season
        )
    ).all():
        raise RuntimeError(
            f"{team_stats_path}: wrong team-stat season"
        )

    prior_stats = stats.loc[
        stats_week.lt(
            target_week
        )
    ].copy()

    prior_stat_week = stats_week.loc[
        prior_stats.index
    ].astype(
        int
    )

    team_ids = prior_stats[
        "team"
    ].map(
        clean_id
    )

    if team_ids.eq(
        ""
    ).any():
        raise RuntimeError(
            f"{team_stats_path}: blank team ID"
        )

    pair_rows = list(
        zip(
            prior_stat_week.tolist(),
            team_ids.tolist(),
            strict=True,
        )
    )

    if len(
        pair_rows
    ) != len(
        set(
            pair_rows
        )
    ):
        raise RuntimeError(
            "Current-season team stats contain duplicate "
            "(week, team) rows"
        )

    actual_team_week_pairs = set(
        pair_rows
    )

    missing_stats = sorted(
        expected_team_week_pairs
        - actual_team_week_pairs
    )

    unexpected_stats = sorted(
        actual_team_week_pairs
        - expected_team_week_pairs
    )

    if (
        missing_stats
        or unexpected_stats
    ):
        raise RuntimeError(
            "Team-stat coverage does not exactly match "
            "completed PBP team/week coverage. "
            f"expected={len(expected_team_week_pairs)} "
            f"actual={len(actual_team_week_pairs)} "
            f"missing_count={len(missing_stats)} "
            f"unexpected_count={len(unexpected_stats)} "
            f"missing_examples={missing_stats[:10]} "
            f"unexpected_examples={unexpected_stats[:10]}"
        )

    return {
        "completed_result_games":
            len(
                expected_game_ids
            ),
        "pbp_prior_games":
            len(
                pbp_game_ids
            ),
        "expected_team_week_rows":
            len(
                expected_team_week_pairs
            ),
        "team_stats_team_week_rows":
            len(
                actual_team_week_pairs
            ),
        "result_files": [
            str(
                path
            )
            for path
            in result_paths
        ],
        "pbp_path":
            str(
                pbp_path
            ),
        "team_stats_path":
            str(
                team_stats_path
            ),
    }


def health_log(payload: dict[str, Any]) -> str:
    lines = [
        f"=== CFB PIPELINE HEALTH {payload['generated_at_utc']} ===",
        f"status={payload['status']}",
        f"workflow_status={payload['workflow']['status']}",
        f"season={payload['season']}",
        f"season_type={payload['season_type']}",
        f"week={payload['report_week']}",
        f"fatal_errors={len(payload['fatal_errors'])}",
        f"warnings={len(payload['warnings'])}",
    ]

    lines.extend(
        f"FATAL: {item}"
        for item in payload["fatal_errors"]
    )
    lines.extend(
        f"WARNING: {item}"
        for item in payload["warnings"]
    )

    return "\n".join(lines) + "\n"


def publish_outputs(
    payload: dict[str, Any],
) -> bool:
    json_text = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    contents = {
        OUTPUT: json_text.encode("utf-8"),
        FRONTEND_OUTPUT: json_text.encode("utf-8"),
        LOG: health_log(payload).encode("utf-8"),
    }

    token = uuid.uuid4().hex
    temps: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    modified = False

    try:
        for final, content in contents.items():
            final.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temp = final.with_name(
                f".{final.name}.{token}.tmp"
            )

            with temp.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())

            temps[final] = temp

            if (
                not final.is_file()
                or final.read_bytes() != content
            ):
                modified = True

        output_payload = json.loads(
            temps[OUTPUT].read_text(encoding="utf-8")
        )
        frontend_payload = json.loads(
            temps[FRONTEND_OUTPUT].read_text(encoding="utf-8")
        )

        if output_payload != payload:
            raise RuntimeError(
                "staged pipeline_health.json failed validation"
            )

        if frontend_payload != payload:
            raise RuntimeError(
                "staged frontend health JSON failed validation"
            )

        if (
            temps[OUTPUT].read_bytes()
            != temps[FRONTEND_OUTPUT].read_bytes()
        ):
            raise RuntimeError(
                "health JSON outputs are not byte-identical"
            )

        if not modified:
            return False

        for final in contents:
            backup = final.with_name(
                f".{final.name}.{token}.bak"
            )

            if final.exists():
                os.replace(final, backup)
                backups[final] = backup

        try:
            for final, temp in temps.items():
                os.replace(temp, final)

        except Exception:
            for final in contents:
                final.unlink(missing_ok=True)

            for final, backup in backups.items():
                if backup.exists():
                    os.replace(backup, final)

            raise

        return True

    finally:
        for temp in temps.values():
            temp.unlink(missing_ok=True)

        for backup in backups.values():
            backup.unlink(missing_ok=True)


def unresolved_paths() -> dict[str, Path]:
    return {
        key: BASE / f"UNRESOLVED/{key}.csv"
        for key in STAGE_NAMES
    }


def main() -> int:
    now_utc = datetime.now(UTC)
    now_ny = datetime.now(NY)
    job_status = (
        clean(os.getenv(JOB_STATUS_ENV))
        or "unknown"
    ).lower()

    fatal_errors: list[str] = []
    warnings: list[str] = []

    season = ""
    season_type = 0
    week = 0

    with PipelineReporter(
        script=__file__,
        stage="pipeline_health",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        season=None,
        week=None,
        extra_context={
            "script_version": SCRIPT_VERSION,
        },
    ) as report:
        report.add_input(CONFIG)

        try:
            season, season_type, week = load_target()
        except Exception as exc:
            fatal_errors.append(
                f"CFB target configuration invalid: {exc}"
            )

        report.season = season or None
        report.week = week or None

        paths = (
            build_paths(season, season_type, week)
            if season and season_type and week
            else unresolved_paths()
        )

        stages: list[dict[str, Any]] = []
        rows_by_key: dict[str, list[dict]] = {}

        history_paths: dict[str, Any] = {}
        history_coverage: dict[str, Any] = {
            "completed_result_games": 0,
            "pbp_prior_games": 0,
            "expected_team_week_rows": 0,
            "team_stats_team_week_rows": 0,
        }

        if season and season_type and week:
            for key, path in paths.items():
                report.add_input(path)

                stage, rows = validate_stage(
                    key,
                    path,
                    season,
                    season_type,
                    week,
                )

                stages.append(stage)
                rows_by_key[key] = rows
        else:
            for key, path in paths.items():
                stages.append(
                    {
                        "name": STAGE_NAMES[key],
                        "path": str(path),
                        "exists": False,
                        "status": "STATUS: INVALID",
                        "error": "target unresolved",
                    }
                )
                rows_by_key[key] = []

        if season and season_type and week:
            history_paths = build_history_paths(
                season,
                season_type,
                week,
            )

            report.add_input(
                history_paths[
                    "pbp"
                ]
            )

            report.add_input(
                history_paths[
                    "team_stats"
                ]
            )

            for result_path in history_paths[
                "results"
            ]:
                report.add_input(
                    result_path
                )

            all_history_paths = [
                history_paths[
                    "pbp"
                ],
                history_paths[
                    "team_stats"
                ],
                *history_paths[
                    "results"
                ],
            ]

            try:
                history_coverage = (
                    validate_historical_coverage(
                        history_paths,
                        season=season,
                        season_type=season_type,
                        target_week=week,
                    )
                )

                stages.append(
                    {
                        "name":
                            "Completed History Coverage",
                        "path":
                            str(
                                history_paths[
                                    "pbp"
                                ]
                            ),
                        "exists":
                            True,
                        "status":
                            "STATUS: SUCCESS",
                        "rows":
                            history_coverage[
                                "completed_result_games"
                            ],
                    }
                )

            except Exception as exc:
                stages.append(
                    {
                        "name":
                            "Completed History Coverage",
                        "path":
                            str(
                                history_paths[
                                    "pbp"
                                ]
                            ),
                        "exists":
                            all(
                                Path(
                                    path
                                ).is_file()
                                for path
                                in all_history_paths
                            ),
                        "status":
                            "STATUS: INVALID",
                        "error":
                            str(
                                exc
                            ),
                    }
                )

                fatal_errors.append(
                    "CFB completed-history coverage "
                    f"is invalid: {exc}"
                )

        unhealthy = [
            stage["name"]
            for stage in stages
            if stage["status"] != "STATUS: SUCCESS"
        ]

        if unhealthy:
            fatal_errors.append(
                "CFB required output(s) missing or invalid: "
                + ", ".join(unhealthy)
            )

        if job_status != "success":
            fatal_errors.append(
                f"CFB workflow status is {job_status.upper()}"
            )

        weekly_rows = rows_by_key.get(
            "weekly_schedule",
            [],
        )
        prediction_rows = rows_by_key.get(
            "predictions",
            [],
        )
        merged_rows = rows_by_key.get(
            "merged",
            [],
        )
        all_games_rows = rows_by_key.get(
            "all_games",
            [],
        )
        selected_rows = rows_by_key.get(
            "selected",
            [],
        )
        locked_rows = rows_by_key.get(
            "locked",
            [],
        )

        weekly_ids = ids(weekly_rows)
        prediction_ids = ids(prediction_rows)
        merged_ids = ids(merged_rows)
        all_games_ids = ids(all_games_rows)
        selected_ids = ids(selected_rows)
        locked_ids = ids(locked_rows)

        valid_status = {
            stage["name"]: stage["status"]
            for stage in stages
        }

        sportsbook_ids: set[str] = set()

        if (
            valid_status.get("Weekly Schedule")
            == "STATUS: SUCCESS"
        ):
            sportsbook_ids = {
                clean_id(row.get("game_id"))
                for row in weekly_rows
                if strict_flag(
                    row.get("odds_available"),
                    "weekly odds_available",
                )
            }

        missing_predictions = (
            sorted(weekly_ids - prediction_ids)
            if prediction_rows
            else []
        )

        missing_sportsbook = sorted(
            weekly_ids - sportsbook_ids
        )

        predictions_not_merged = (
            sorted(prediction_ids - merged_ids)
            if prediction_rows and merged_rows
            else []
        )

        scheduled_missing_merged = (
            sorted(weekly_ids - merged_ids)
            if weekly_rows and merged_rows
            else []
        )

        merged_not_scheduled = (
            sorted(merged_ids - weekly_ids)
            if weekly_rows and merged_rows
            else []
        )

        scheduled_missing_all_games = (
            sorted(weekly_ids - all_games_ids)
            if weekly_rows and all_games_rows
            else []
        )

        all_games_not_scheduled = (
            sorted(all_games_ids - weekly_ids)
            if weekly_rows and all_games_rows
            else []
        )

        selected_not_scheduled = sorted(
            selected_ids - weekly_ids
        )

        locked_not_scheduled = sorted(
            locked_ids - weekly_ids
        )

        if missing_predictions:
            warnings.append(
                f"CFB Week {week}: "
                f"{len(missing_predictions)} scheduled game(s) "
                "missing final predictions"
            )

        if missing_sportsbook:
            warnings.append(
                f"CFB Week {week}: "
                f"{len(missing_sportsbook)} scheduled game(s) "
                "do not have sportsbook odds"
            )

        if predictions_not_merged:
            warnings.append(
                f"CFB Week {week}: "
                f"{len(predictions_not_merged)} predicted game(s) "
                "are not present in the projection output"
            )

        exact_coverage_failures = {
            "scheduled_missing_merged":
                scheduled_missing_merged,
            "merged_not_scheduled":
                merged_not_scheduled,
            "scheduled_missing_all_games":
                scheduled_missing_all_games,
            "all_games_not_scheduled":
                all_games_not_scheduled,
            "selected_not_scheduled":
                selected_not_scheduled,
            "locked_not_scheduled":
                locked_not_scheduled,
        }

        if any(exact_coverage_failures.values()):
            fatal_errors.append(
                "CFB current-week output coverage does not "
                "match the target schedule"
            )

        identity: dict[str, Any] = {}

        for key in CURRENT_WEEK_KEYS:
            stage_rows = rows_by_key.get(key, [])

            mismatches = team_mismatches(
                weekly_rows,
                stage_rows,
            )

            identity[
                f"{key}_team_mismatch_game_ids"
            ] = mismatches

            if mismatches:
                fatal_errors.append(
                    f"CFB {STAGE_NAMES[key]} team identity "
                    "does not match weekly schedule"
                )

        selected_games = len(selected_ids)
        locked_games = len(locked_ids)

        selected_bets = (
            selected_bet_count(
                selected_rows,
                "selected output",
            )
            if selected_rows
            else 0
        )

        locked_bets = (
            selected_bet_count(
                locked_rows,
                "locked output",
            )
            if locked_rows
            else 0
        )

        if (
            valid_status.get("Final Selected Picks")
            == "STATUS: SUCCESS"
            and valid_status.get("Locked Selected Picks")
            == "STATUS: SUCCESS"
            and (
                selected_ids != locked_ids
                or selected_bets != locked_bets
            )
        ):
            fatal_errors.append(
                "CFB selected and locked outputs disagree"
            )

        counts = {
            "scheduled_games": len(weekly_ids),
            "prediction_games": len(prediction_ids),
            "sportsbook_games": len(sportsbook_ids),
            "merged_games": len(merged_ids),
            "all_games_games": len(all_games_ids),
            "selected_games": selected_games,
            "selected_bets": selected_bets,
            "locked_games": locked_games,
            "locked_bets": locked_bets,
            "completed_prior_games":
                history_coverage.get(
                    "completed_result_games",
                    0,
                ),
            "pbp_prior_games":
                history_coverage.get(
                    "pbp_prior_games",
                    0,
                ),
            "expected_team_week_rows":
                history_coverage.get(
                    "expected_team_week_rows",
                    0,
                ),
            "team_stats_team_week_rows":
                history_coverage.get(
                    "team_stats_team_week_rows",
                    0,
                ),
        }

        league = {
            "in_season": True,
            "season": season,
            "season_type": season_type,
            "week": week,
            "paths": {
                key: str(path)
                for key, path in paths.items()
            },
            "counts": counts,
            "identity": identity,
            "history_paths": {
                "pbp":
                    str(
                        history_paths.get(
                            "pbp",
                            "",
                        )
                    ),
                "team_stats":
                    str(
                        history_paths.get(
                            "team_stats",
                            "",
                        )
                    ),
                "results": [
                    str(
                        path
                    )
                    for path
                    in history_paths.get(
                        "results",
                        []
                    )
                ],
            },
            "completed_history_coverage":
                history_coverage,
            "coverage": {
                "scheduled_missing_predictions":
                    missing_predictions,
                "scheduled_missing_sportsbook":
                    missing_sportsbook,
                "predictions_not_merged":
                    predictions_not_merged,
                **exact_coverage_failures,
            },
        }

        status = (
            "failed"
            if fatal_errors
            else "warning"
            if warnings
            else "healthy"
        )

        payload = {
            "schema_version": 1,
            "sport": "cfb",
            "generated_at_utc": now_utc.isoformat(),
            "game_date_new_york":
                now_ny.strftime("%Y_%m_%d"),
            "season": season,
            "season_type": season_type,
            "report_week": week,
            "status": status,
            "fatal_errors": fatal_errors,
            "warnings": warnings,
            "leagues": {
                "cfb": league,
            },
            "stage_status": stages,
            "workflow": {
                "name": "CFB Pipeline",
                "status": job_status,
                "run_id":
                    clean(os.getenv("GITHUB_RUN_ID")),
                "run_attempt":
                    clean(os.getenv("GITHUB_RUN_ATTEMPT")),
                "sha":
                    clean(os.getenv("GITHUB_SHA")),
                "ref_name":
                    clean(os.getenv("GITHUB_REF_NAME")),
            },
        }

        output_modified = publish_outputs(
            payload
        )

        report.add_output(OUTPUT)
        report.add_output(LOG)
        report.add_output(FRONTEND_OUTPUT)

        report.set_rows(
            rows_in=len(weekly_ids),
            rows_out=selected_games,
        )

        report.update_details(
            {
                "script_version":
                    SCRIPT_VERSION,
                "season_type":
                    season_type,
                "workflow_status":
                    job_status,
                "health_status":
                    status,
                "counts":
                    counts,
                "fatal_errors":
                    len(fatal_errors),
                "warnings":
                    len(warnings),
                "output_modified":
                    output_modified,
            }
        )

        for warning in warnings:
            report.warning(warning)

        for fatal in fatal_errors:
            report.error(fatal)

        print(
            "CFB pipeline health: "
            f"{status} "
            f"season={season or 'unresolved'} "
            f"week={week or 'unresolved'} "
            f"selected_bets={selected_bets} "
            f"locked_bets={locked_bets}"
        )

    return 1 if fatal_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

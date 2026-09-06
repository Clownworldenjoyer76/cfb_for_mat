#!/usr/bin/env python3
# docs/win/football/cfb/scripts/pipeline_health.py
"""CFB pipeline health reporter.

Writes:
    docs/win/football/cfb/pipeline_health.json
    docs/win/football/cfb/errors/pipeline_health.txt
    frontend/data/pipeline_health/cfb.json
"""
from __future__ import annotations

import csv
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


BASE = Path("docs/win/football/cfb")
OUTPUT = BASE / "pipeline_health.json"
LOG = BASE / "errors/pipeline_health.txt"
FRONTEND_OUTPUT = Path("frontend/data/pipeline_health/cfb.json")

NY = ZoneInfo("America/New_York")

JOB_STATUS_ENV = "CFB_PIPELINE_JOB_STATUS"
SEASON_ENV = "CFB_SEASON"
WEEK_ENV = "CFB_WEEK"


def clean(value) -> str:
    return "" if value is None else str(value).strip()


def clean_id(value) -> str:
    text = clean(value)
    if not text:
        return ""

    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass

    return text


def read_rows(path: Path) -> list[dict]:
    if not path.exists() or not path.is_file():
        return []

    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def unique_game_ids(rows: list[dict]) -> set[str]:
    return {
        game_id
        for row in rows
        if (game_id := clean_id(row.get("game_id")))
    }


def duplicate_game_ids(rows: list[dict]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for row in rows:
        game_id = clean_id(row.get("game_id"))
        if not game_id:
            continue

        if game_id in seen:
            duplicates.add(game_id)
        else:
            seen.add(game_id)

    return sorted(duplicates)


def resolve_week() -> int:
    raw = clean(os.getenv(WEEK_ENV))

    if raw:
        try:
            week = int(raw)
        except ValueError:
            week = 0

        if week > 0:
            return week

    weekly_dir = BASE / "00_intake/schedule/weekly"
    weeks = []

    for path in weekly_dir.glob("week_*_CFB_weekly_schedule.csv"):
        match = re.fullmatch(
            r"week_(\d+)_CFB_weekly_schedule\.csv",
            path.name,
        )
        if match:
            weeks.append(int(match.group(1)))

    if not weeks:
        raise RuntimeError(
            f"No CFB week could be resolved from {WEEK_ENV} or {weekly_dir}"
        )

    return max(weeks)


def resolve_season() -> str:
    value = clean(os.getenv(SEASON_ENV))
    return value or str(datetime.now(NY).year)


def build_paths(
    season: str,
    week: int,
    schedule_rows: list[dict],
) -> dict[str, Path]:
    season_type = ""

    for row in schedule_rows:
        season_type = clean(row.get("season_type"))
        if season_type:
            break

    prediction_name = (
        f"{season}_{season_type}_{week}_clean_predictions.csv"
        if season_type
        else f"{season}_2_{week}_clean_predictions.csv"
    )

    return {
        "season_schedule": BASE / f"00_intake/schedule/{season}_schedule.csv",
        "weekly_schedule": (
            BASE
            / "00_intake/schedule/weekly"
            / f"week_{week}_CFB_weekly_schedule.csv"
        ),
        "predictions": (
            BASE
            / "00_intake/predictions/final"
            / prediction_name
        ),
        "merged": BASE / f"01_merge/week_{week}_CFB_enriched.csv",
        "candidates": BASE / f"02_select/week_{week}_CFB_selected.csv",
        "picks": BASE / f"03_picks/week_{week}_CFB_picks.csv",
        "clean_picks": (
            BASE
            / "03_picks/cleaned"
            / f"week_{week}_CFB_clean_picks.csv"
        ),
        "all_games": (
            BASE
            / "03_picks/all_games"
            / f"all_week_{week}_CFB_picks.csv"
        ),
        "selected": (
            BASE
            / "03_picks/selected"
            / f"week_{week}_CFB_select_picks.csv"
        ),
    }


def stage_row(name: str, path: Path) -> dict:
    exists = path.exists() and path.is_file()
    nonempty = exists and path.stat().st_size > 0

    return {
        "name": name,
        "path": str(path),
        "exists": exists,
        "status": "STATUS: SUCCESS" if nonempty else "STATUS: MISSING",
    }


def build_stage_status(paths: dict[str, Path]) -> list[dict]:
    return [
        stage_row("Season Schedule", paths["season_schedule"]),
        stage_row("Weekly Schedule", paths["weekly_schedule"]),
        stage_row("Final Predictions", paths["predictions"]),
        stage_row("Projection", paths["merged"]),
        stage_row("Betting Candidates", paths["candidates"]),
        stage_row("Market Picks", paths["picks"]),
        stage_row("Clean Weekly Picks", paths["clean_picks"]),
        stage_row("All Games Picks", paths["all_games"]),
        stage_row("Final Selected Picks", paths["selected"]),
    ]


def write_log(payload: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"=== CFB PIPELINE HEALTH {payload['generated_at_utc']} ===",
        f"status={payload['status']}",
        f"workflow_status={payload['workflow']['status']}",
        f"season={payload['season']}",
        f"week={payload['report_week']}",
        f"fatal_errors={len(payload['fatal_errors'])}",
        f"warnings={len(payload['warnings'])}",
    ]

    for fatal in payload["fatal_errors"]:
        lines.append(f"FATAL: {fatal}")

    for warning in payload["warnings"]:
        lines.append(f"WARNING: {warning}")

    LOG.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    now_utc = datetime.now(UTC)
    now_ny = datetime.now(NY)

    job_status = clean(os.getenv(JOB_STATUS_ENV)) or "unknown"
    season = resolve_season()

    fatal_errors: list[str] = []
    warnings: list[str] = []

    try:
        week = resolve_week()
    except Exception as exc:
        week = 0
        fatal_errors.append(f"CFB: {exc}")

    weekly_path = (
        BASE
        / "00_intake/schedule/weekly"
        / f"week_{week}_CFB_weekly_schedule.csv"
    )

    schedule_rows = read_rows(weekly_path) if week else []

    paths = (
        build_paths(season, week, schedule_rows)
        if week
        else {
            "season_schedule": BASE / f"00_intake/schedule/{season}_schedule.csv",
            "weekly_schedule": weekly_path,
            "predictions": BASE / "00_intake/predictions/final/UNRESOLVED.csv",
            "merged": BASE / "01_merge/UNRESOLVED.csv",
            "candidates": BASE / "02_select/UNRESOLVED.csv",
            "picks": BASE / "03_picks/UNRESOLVED.csv",
            "clean_picks": BASE / "03_picks/cleaned/UNRESOLVED.csv",
            "all_games": BASE / "03_picks/all_games/UNRESOLVED.csv",
            "selected": BASE / "03_picks/selected/UNRESOLVED.csv",
        }
    )

    rows = {
        key: read_rows(path)
        for key, path in paths.items()
        if key != "season_schedule"
    }

    weekly_rows = rows.get("weekly_schedule", [])
    prediction_rows = rows.get("predictions", [])
    merged_rows = rows.get("merged", [])
    selected_rows = rows.get("selected", [])

    weekly_ids = unique_game_ids(weekly_rows)
    prediction_ids = unique_game_ids(prediction_rows)
    merged_ids = unique_game_ids(merged_rows)

    sportsbook_ids = {
        clean_id(row.get("game_id"))
        for row in weekly_rows
        if clean(row.get("odds_available")) == "1"
        and clean_id(row.get("game_id"))
    }

    missing_predictions = sorted(
        weekly_ids - prediction_ids
    )

    missing_sportsbook = sorted(
        weekly_ids - sportsbook_ids
    )

    predictions_not_merged = sorted(
        prediction_ids - merged_ids
    )

    weekly_duplicates = duplicate_game_ids(
        weekly_rows
    )

    prediction_duplicates = duplicate_game_ids(
        prediction_rows
    )

    merged_duplicates = duplicate_game_ids(
        merged_rows
    )

    blank_weekly_ids = sum(
        1
        for row in weekly_rows
        if not clean_id(row.get("game_id"))
    )

    counts = {
        "scheduled_games": len(weekly_ids),
        "prediction_games": len(prediction_ids),
        "sportsbook_games": len(sportsbook_ids),
        "merged_games": len(merged_ids),
        "selected_bets": len(selected_rows),
        "locked_bets": 0,
    }

    league = {
        "in_season": True,
        "season": season,
        "week": week,
        "paths": {
            key: str(path)
            for key, path in paths.items()
        },
        "counts": counts,
        "identity": {
            "weekly_duplicate_game_ids": weekly_duplicates,
            "prediction_duplicate_game_ids": prediction_duplicates,
            "merged_duplicate_game_ids": merged_duplicates,
            "blank_weekly_game_ids": blank_weekly_ids,
        },
        "coverage": {
            "scheduled_missing_predictions": missing_predictions,
            "scheduled_missing_sportsbook": missing_sportsbook,
            "predictions_not_merged": predictions_not_merged,
        },
    }

    stage_status = build_stage_status(paths)

    if job_status.lower() != "success":
        fatal_errors.append(
            f"CFB workflow status is {job_status.upper()}"
        )

    if job_status.lower() == "success":
        missing_required = [
            row["name"]
            for row in stage_status
            if row["status"] != "STATUS: SUCCESS"
        ]

        if missing_required:
            fatal_errors.append(
                "CFB required current-week output(s) missing: "
                + ", ".join(missing_required)
            )

        if (
            weekly_duplicates
            or prediction_duplicates
            or merged_duplicates
        ):
            fatal_errors.append(
                "CFB duplicate game_id values detected "
                "in current-week data"
            )

        if blank_weekly_ids:
            fatal_errors.append(
                f"CFB weekly schedule has "
                f"{blank_weekly_ids} row(s) with blank game_id"
            )

    if missing_predictions:
        warnings.append(
            f"CFB Week {week}: "
            f"{len(missing_predictions)} scheduled game(s) "
            f"missing final predictions"
        )

    if missing_sportsbook:
        warnings.append(
            f"CFB Week {week}: "
            f"{len(missing_sportsbook)} scheduled game(s) "
            f"do not have sportsbook odds"
        )

    if predictions_not_merged:
        warnings.append(
            f"CFB Week {week}: "
            f"{len(predictions_not_merged)} predicted game(s) "
            f"are not present in the projection output"
        )

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
        "game_date_new_york": now_ny.strftime("%Y_%m_%d"),
        "season": season,
        "report_week": week,
        "status": status,
        "fatal_errors": fatal_errors,
        "warnings": warnings,
        "leagues": {
            "cfb": league,
        },
        "stage_status": stage_status,
        "workflow": {
            "name": "CFB Pipeline",
            "status": job_status.lower(),
            "run_id": clean(os.getenv("GITHUB_RUN_ID")),
            "run_attempt": clean(os.getenv("GITHUB_RUN_ATTEMPT")),
            "sha": clean(os.getenv("GITHUB_SHA")),
            "ref_name": clean(os.getenv("GITHUB_REF_NAME")),
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FRONTEND_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    OUTPUT.write_text(
        text,
        encoding="utf-8",
    )

    FRONTEND_OUTPUT.write_text(
        text,
        encoding="utf-8",
    )

    write_log(payload)

    print(f"CFB pipeline health: {status}")
    print(f"season: {season}")
    print(f"week: {week}")
    print(f"output: {OUTPUT}")
    print(f"frontend: {FRONTEND_OUTPUT}")

    for warning in warnings:
        print(f"WARNING: {warning}")

    for fatal in fatal_errors:
        print(f"FATAL: {fatal}")

    return 1 if fatal_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
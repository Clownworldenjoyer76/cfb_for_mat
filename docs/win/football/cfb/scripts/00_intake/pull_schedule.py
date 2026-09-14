#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import os
import sys
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
CFB_DIR = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_reporter import PipelineReporter


CONFIG_FILE = CFB_DIR / "config" / "current_week.yaml"

TEAM_MAP_FILE = (
    CFB_DIR
    / "config"
    / "mapping"
    / "team_map.csv"
)

STADIUM_MAP_FILE = (
    CFB_DIR
    / "config"
    / "mapping"
    / "stadium_map.csv"
)

OUTPUT_DIR = (
    CFB_DIR
    / "00_intake"
    / "schedule"
)

UPDATES_DIR = (
    OUTPUT_DIR
    / "updates"
)

REPORT_ROOT = (
    CFB_DIR
    / "errors"
)

TEAM_MAP_REQUIRED_COLUMNS = {
    "team_id",
    "canonical_team",
}

STADIUM_MAP_REQUIRED_COLUMNS = {
    "team",
    "stadium",
    "venue_full_name",
    "roof_type",
    "surface",
    "timezone",
}

OUTPUT_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "neutral_site",
    "stadium",
    "roof",
    "surface",
    "home_timezone",
    "away_timezone",
    "game_timezone",
]


def clean(value: Any) -> str:
    if value is None:
        return ""

    value = str(value).strip()

    if value.lower() in {
        "none",
        "null",
        "nan",
    }:
        return ""

    return value


def lookup_key(value: Any) -> str:
    return clean(value).casefold()


def load_pipeline_config(
    path: Path,
) -> tuple[int, int, int]:
    if not path.is_file():
        raise RuntimeError(
            f"Missing required file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(
            handle
        )

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"Invalid YAML mapping: {path}"
        )

    try:
        season = int(
            data["season"]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            f"Invalid or missing season in {path}"
        ) from exc

    try:
        season_type = int(
            data["season_type"]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            f"Invalid or missing season_type in {path}"
        ) from exc

    if not 2000 <= season <= 2099:
        raise RuntimeError(
            f"Invalid season in {path}: {season}"
        )

    if season_type <= 0:
        raise RuntimeError(
            f"Invalid season_type in {path}: "
            f"{season_type}"
        )

    try:
        week = int(
            data["week"]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            f"Invalid or missing week in {path}"
        ) from exc

    if week <= 0:
        raise RuntimeError(
            f"Invalid week in {path}: {week}"
        )

    return (
        season,
        season_type,
        week,
    )


def read_csv(
    path: Path,
    *,
    required_columns: set[str] | None = None,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(
            f"Missing required file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames is None:
            raise RuntimeError(
                f"Missing CSV header: {path}"
            )

        fieldnames = {
            clean(name)
            for name in reader.fieldnames
        }

        missing = (
            set(
                required_columns
                or set()
            )
            - fieldnames
        )

        if missing:
            raise RuntimeError(
                f"{path}: missing required "
                f"columns: {sorted(missing)}"
            )

        rows: list[
            dict[str, str]
        ] = []

        for (
            line_number,
            row,
        ) in enumerate(
            reader,
            start=2,
        ):
            if None in row:
                raise RuntimeError(
                    f"{path}: malformed CSV "
                    f"row at line {line_number}"
                )

            rows.append(
                {
                    clean(key): clean(value)
                    for key, value
                    in row.items()
                }
            )

        return rows


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        path.with_name(
            f".{path.name}."
            f"{uuid.uuid4().hex}.tmp"
        )
    )

    try:
        with temp_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_COLUMNS,
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        column: clean(
                            row.get(column)
                        )
                        for column
                        in OUTPUT_COLUMNS
                    }
                )

            handle.flush()
            os.fsync(
                handle.fileno()
            )

        os.replace(
            temp_path,
            path,
        )

    except Exception:
        try:
            temp_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        raise


def build_team_maps(
    rows: list[dict[str, str]],
) -> tuple[
    list[str],
    dict[str, str],
]:
    team_ids: list[str] = []
    seen_ids: set[str] = set()

    team_lookup: dict[
        str,
        str,
    ] = {}

    for row in rows:
        team_id = clean(
            row.get(
                "team_id"
            )
        )

        canonical = clean(
            row.get(
                "canonical_team"
            )
        )

        if (
            team_id
            and team_id
            not in seen_ids
        ):
            team_ids.append(
                team_id
            )

            seen_ids.add(
                team_id
            )

        if not canonical:
            continue

        candidates = [
            row.get("team_id"),
            row.get(
                "canonical_team"
            ),
            row.get("team_abbr"),
            row.get("alias"),
            row.get("location"),
            row.get("team_name"),
            row.get("team_slug"),
            row.get("nickname"),
            row.get(
                "shortDisplayName"
            ),
        ]

        for candidate in candidates:
            candidate = clean(
                candidate
            )

            if candidate:
                team_lookup[
                    lookup_key(
                        candidate
                    )
                ] = canonical

    if not team_ids:
        raise RuntimeError(
            "No team_id values found "
            "in team_map.csv"
        )

    return (
        team_ids,
        team_lookup,
    )


def build_stadium_maps(
    rows: list[dict[str, str]],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    by_team: dict[
        str,
        dict[str, str],
    ] = {}

    by_stadium: dict[
        str,
        dict[str, str],
    ] = {}

    for row in rows:
        team = clean(
            row.get("team")
        )

        stadium = clean(
            row.get("stadium")
        )

        venue_full_name = clean(
            row.get(
                "venue_full_name"
            )
        )

        if team:
            by_team[
                lookup_key(team)
            ] = row

        if stadium:
            by_stadium[
                lookup_key(stadium)
            ] = row

        if venue_full_name:
            by_stadium[
                lookup_key(
                    venue_full_name
                )
            ] = row

    return (
        by_team,
        by_stadium,
    )


def fetch_schedule(
    team_id: str,
    *,
    season: int,
    season_type: int,
    report: PipelineReporter,
) -> dict[str, Any] | None:
    url = (
        "https://site.api.espn.com/"
        "apis/site/v2/sports/football/"
        "college-football/"
        f"teams/{team_id}/schedule"
        f"?season={season}"
        f"&seasontype={season_type}"
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64)"
            ),
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:
            payload = json.loads(
                response
                .read()
                .decode("utf-8")
            )

        if not isinstance(
            payload,
            dict,
        ):
            report.warning(
                "ESPN schedule response "
                "was not a JSON object",
                team_id=team_id,
            )

            return None

        return payload

    except Exception as exc:
        report.warning(
            "ESPN schedule fetch failed",
            team_id=team_id,
            error_type=(
                type(exc).__name__
            ),
            error=str(exc),
        )

        return None


def first_competition(
    event: dict[str, Any],
) -> dict[str, Any]:
    competitions = event.get(
        "competitions"
    )

    if (
        isinstance(
            competitions,
            list,
        )
        and competitions
        and isinstance(
            competitions[0],
            dict,
        )
    ):
        return competitions[0]

    return {}


def competitor_team(
    competition: dict[str, Any],
    home_away: str,
) -> dict[str, Any]:
    competitors = competition.get(
        "competitors"
    )

    if not isinstance(
        competitors,
        list,
    ):
        return {}

    for competitor in competitors:
        if not isinstance(
            competitor,
            dict,
        ):
            continue

        if (
            clean(
                competitor.get(
                    "homeAway"
                )
            ).casefold()
            != home_away.casefold()
        ):
            continue

        team = competitor.get(
            "team"
        )

        if isinstance(
            team,
            dict,
        ):
            return team

    return {}


def canonical_team(
    team: dict[str, Any],
    team_lookup: dict[str, str],
) -> str:
    candidates = [
        team.get("id"),
        team.get("displayName"),
        team.get(
            "shortDisplayName"
        ),
        team.get(
            "abbreviation"
        ),
        team.get("location"),
        team.get("nickname"),
    ]

    for candidate in candidates:
        mapped = team_lookup.get(
            lookup_key(
                candidate
            )
        )

        if mapped:
            return mapped

    return (
        clean(
            team.get(
                "displayName"
            )
        )
        or clean(
            team.get(
                "shortDisplayName"
            )
        )
        or clean(
            team.get(
                "location"
            )
        )
    )


def parse_game_datetime(
    raw_date: str,
    game_timezone: str,
    *,
    game_id: str,
    report: PipelineReporter,
) -> tuple[str, str]:
    if not raw_date:
        return "", ""

    dt = datetime.fromisoformat(
        raw_date.replace(
            "Z",
            "+00:00",
        )
    )

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=timezone.utc
        )

    if game_timezone:
        try:
            dt = dt.astimezone(
                ZoneInfo(
                    game_timezone
                )
            )

        except Exception as exc:
            report.warning(
                "Invalid game timezone; "
                "UTC fallback used",
                game_id=game_id,
                game_timezone=(
                    game_timezone
                ),
                error_type=(
                    type(exc).__name__
                ),
                error=str(exc),
            )

            dt = dt.astimezone(
                timezone.utc
            )

    else:
        dt = dt.astimezone(
            timezone.utc
        )

    return (
        dt.strftime(
            "%Y-%m-%d"
        ),
        dt.strftime(
            "%H:%M"
        ),
    )


def event_to_row(
    event: dict[str, Any],
    team_lookup: dict[str, str],
    stadium_by_team: dict[
        str,
        dict[str, str],
    ],
    stadium_by_stadium: dict[
        str,
        dict[str, str],
    ],
    *,
    season: int,
    season_type: int,
    report: PipelineReporter,
) -> dict[str, str] | None:
    game_id = clean(
        event.get("id")
    )

    if not game_id:
        return None

    competition = (
        first_competition(
            event
        )
    )

    home_obj = competitor_team(
        competition,
        "home",
    )

    away_obj = competitor_team(
        competition,
        "away",
    )

    home_team = canonical_team(
        home_obj,
        team_lookup,
    )

    away_team = canonical_team(
        away_obj,
        team_lookup,
    )

    if (
        not home_team
        or not away_team
    ):
        report.warning(
            "Schedule event skipped "
            "because home or away "
            "team was missing",
            game_id=game_id,
        )

        return None

    venue = competition.get(
        "venue"
    )

    if not isinstance(
        venue,
        dict,
    ):
        venue = {}

    espn_stadium = clean(
        venue.get(
            "fullName"
        )
    )

    neutral_value = (
        competition.get(
            "neutralSite"
        )
    )

    neutral_site = (
        "1"
        if neutral_value is True
        else "0"
        if neutral_value is False
        else ""
    )

    stadium_row: dict[
        str,
        str,
    ] = {}

    if neutral_site == "1":
        stadium_row = (
            stadium_by_stadium.get(
                lookup_key(
                    espn_stadium
                ),
                {},
            )
        )

    if not stadium_row:
        stadium_row = (
            stadium_by_team.get(
                lookup_key(
                    home_team
                ),
                {},
            )
        )

    if not stadium_row:
        stadium_row = (
            stadium_by_stadium.get(
                lookup_key(
                    espn_stadium
                ),
                {},
            )
        )

    stadium = (
        clean(
            stadium_row.get(
                "stadium"
            )
        )
        or espn_stadium
    )

    roof = clean(
        stadium_row.get(
            "roof_type"
        )
    )

    surface = clean(
        stadium_row.get(
            "surface"
        )
    )

    game_timezone = clean(
        stadium_row.get(
            "timezone"
        )
    )

    home_stadium_row = (
        stadium_by_team.get(
            lookup_key(
                home_team
            ),
            {},
        )
    )

    away_stadium_row = (
        stadium_by_team.get(
            lookup_key(
                away_team
            ),
            {},
        )
    )

    home_timezone = clean(
        home_stadium_row.get(
            "timezone"
        )
    )

    away_timezone = clean(
        away_stadium_row.get(
            "timezone"
        )
    )

    game_date, game_time = (
        parse_game_datetime(
            clean(
                event.get("date")
            ),
            game_timezone,
            game_id=game_id,
            report=report,
        )
    )

    season_obj = event.get(
        "season"
    )

    event_season = ""

    if isinstance(
        season_obj,
        dict,
    ):
        event_season = clean(
            season_obj.get(
                "year"
            )
        )

    season_type_obj = (
        event.get(
            "seasonType"
        )
    )

    event_season_type = ""

    if isinstance(
        season_type_obj,
        dict,
    ):
        event_season_type = (
            clean(
                season_type_obj.get(
                    "type"
                )
            )
            or clean(
                season_type_obj.get(
                    "id"
                )
            )
        )

    week_obj = event.get(
        "week"
    )

    week = ""

    if isinstance(
        week_obj,
        dict,
    ):
        week = clean(
            week_obj.get(
                "number"
            )
        )

    return {
        "season": (
            event_season
            or str(season)
        ),
        "season_type": (
            event_season_type
            or str(season_type)
        ),
        "week": week,
        "game_id": game_id,
        "game_date": game_date,
        "game_time": game_time,
        "away_team": away_team,
        "home_team": home_team,
        "neutral_site": neutral_site,
        "stadium": stadium,
        "roof": roof,
        "surface": surface,
        "home_timezone": (
            home_timezone
        ),
        "away_timezone": (
            away_timezone
        ),
        "game_timezone": (
            game_timezone
        ),
    }


def read_existing_schedule(
    output_file: Path,
) -> list[dict[str, str]]:
    if not output_file.exists():
        return []

    with output_file.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        if reader.fieldnames is None:
            return []

        return [
            {
                column: clean(
                    row.get(column)
                )
                for column
                in OUTPUT_COLUMNS
            }
            for row in reader
        ]


def sort_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            clean(
                row.get(
                    "game_date"
                )
            ),
            clean(
                row.get(
                    "game_time"
                )
            ),
            clean(
                row.get(
                    "game_id"
                )
            ),
        ),
    )


def main() -> None:
    with PipelineReporter(
        script=__file__,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
    ) as report:
        report.add_input(
            CONFIG_FILE
        )

        report.add_input(
            TEAM_MAP_FILE
        )

        report.add_input(
            STADIUM_MAP_FILE
        )

        (
            season,
            season_type,
            week,
        ) = load_pipeline_config(
            CONFIG_FILE
        )

        report.season = season
        report.week = week

        report.set_detail(
            "season_type",
            season_type,
        )

        output_file = (
            OUTPUT_DIR
            / f"{season}_schedule.csv"
        )

        team_rows = read_csv(
            TEAM_MAP_FILE,
            required_columns=(
                TEAM_MAP_REQUIRED_COLUMNS
            ),
        )

        stadium_rows = read_csv(
            STADIUM_MAP_FILE,
            required_columns=(
                STADIUM_MAP_REQUIRED_COLUMNS
            ),
        )

        (
            team_ids,
            team_lookup,
        ) = build_team_maps(
            team_rows
        )

        (
            stadium_by_team,
            stadium_by_stadium,
        ) = build_stadium_maps(
            stadium_rows
        )

        pulled: dict[
            str,
            dict[str, str],
        ] = {}

        successful_team_pulls = 0
        failed_team_pulls = 0
        invalid_team_payloads = 0
        empty_team_pulls = 0
        total_events_seen = 0
        skipped_events = 0

        for team_id in team_ids:
            data = fetch_schedule(
                team_id,
                season=season,
                season_type=(
                    season_type
                ),
                report=report,
            )

            if data is None:
                failed_team_pulls += 1
                continue

            events = data.get(
                "events"
            )

            if not isinstance(
                events,
                list,
            ):
                (
                    invalid_team_payloads
                ) += 1

                report.warning(
                    "ESPN schedule response "
                    "contained invalid "
                    "events data",
                    team_id=team_id,
                )

                continue

            if not events:
                empty_team_pulls += 1
                continue

            (
                successful_team_pulls
            ) += 1

            total_events_seen += len(
                events
            )

            for event in events:
                if not isinstance(
                    event,
                    dict,
                ):
                    skipped_events += 1
                    continue

                row = event_to_row(
                    event,
                    team_lookup,
                    stadium_by_team,
                    stadium_by_stadium,
                    season=season,
                    season_type=(
                        season_type
                    ),
                    report=report,
                )

                if row is None:
                    skipped_events += 1
                    continue

                pulled[
                    row["game_id"]
                ] = row

        report.update_details(
            {
                "team_map_rows": (
                    len(team_rows)
                ),
                "stadium_map_rows": (
                    len(stadium_rows)
                ),
                "team_ids": (
                    len(team_ids)
                ),
                "successful_team_pulls": (
                    successful_team_pulls
                ),
                "failed_team_pulls": (
                    failed_team_pulls
                ),
                "invalid_team_payloads": (
                    invalid_team_payloads
                ),
                "empty_team_pulls": (
                    empty_team_pulls
                ),
                "events_seen": (
                    total_events_seen
                ),
                "skipped_events": (
                    skipped_events
                ),
                "unique_games_pulled": (
                    len(pulled)
                ),
            }
        )

        if not pulled:
            raise RuntimeError(
                "ESPN returned zero usable "
                "schedule rows. Existing "
                "schedule was NOT overwritten."
            )

        pulled_rows = sort_rows(
            list(
                pulled.values()
            )
        )

        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%d_%H%M%S"
        )

        update_file = (
            UPDATES_DIR
            / (
                f"{season}_schedule_"
                f"{timestamp}.csv"
            )
        )

        write_csv_atomic(
            update_file,
            pulled_rows,
        )

        report.add_output(
            update_file
        )

        existing_rows = (
            read_existing_schedule(
                output_file
            )
        )

        merged: dict[
            str,
            dict[str, str],
        ] = {}

        existing_by_id: dict[
            str,
            dict[str, str],
        ] = {}

        for row in existing_rows:
            game_id = clean(
                row.get(
                    "game_id"
                )
            )

            if game_id:
                existing_by_id[
                    game_id
                ] = row

                merged[
                    game_id
                ] = row

        added_games = 0
        updated_games = 0
        unchanged_games = 0

        for (
            game_id,
            row,
        ) in pulled.items():
            existing_row = (
                existing_by_id.get(
                    game_id
                )
            )

            if existing_row is None:
                added_games += 1

            elif all(
                clean(
                    existing_row.get(
                        column
                    )
                )
                == clean(
                    row.get(
                        column
                    )
                )
                for column
                in OUTPUT_COLUMNS
            ):
                unchanged_games += 1

            else:
                updated_games += 1

            merged[
                game_id
            ] = row

        preserved_games = len(
            set(existing_by_id)
            - set(pulled)
        )

        output_rows = sort_rows(
            list(
                merged.values()
            )
        )

        if not output_rows:
            raise RuntimeError(
                "No schedule rows "
                "available to write."
            )

        missing_stadium = sum(
            1
            for row in output_rows
            if not clean(
                row.get("stadium")
            )
        )

        missing_surface = sum(
            1
            for row in output_rows
            if not clean(
                row.get("surface")
            )
        )

        missing_roof = sum(
            1
            for row in output_rows
            if not clean(
                row.get("roof")
            )
        )

        missing_home_timezone = sum(
            1
            for row in output_rows
            if not clean(
                row.get(
                    "home_timezone"
                )
            )
        )

        missing_away_timezone = sum(
            1
            for row in output_rows
            if not clean(
                row.get(
                    "away_timezone"
                )
            )
        )

        missing_game_timezone = sum(
            1
            for row in output_rows
            if not clean(
                row.get(
                    "game_timezone"
                )
            )
        )

        write_csv_atomic(
            output_file,
            output_rows,
        )

        report.add_output(
            output_file
        )

        report.set_rows(
            rows_in=(
                total_events_seen
            ),
            rows_out=(
                len(output_rows)
            ),
        )

        report.update_details(
            {
                "existing_schedule_rows": (
                    len(existing_rows)
                ),
                "schedule_rows_written": (
                    len(output_rows)
                ),
                "schedule_games_added": (
                    added_games
                ),
                "schedule_games_updated": (
                    updated_games
                ),
                "schedule_games_unchanged": (
                    unchanged_games
                ),
                "schedule_games_preserved": (
                    preserved_games
                ),
                "missing_stadium": (
                    missing_stadium
                ),
                "missing_surface": (
                    missing_surface
                ),
                "missing_roof": (
                    missing_roof
                ),
                "missing_home_timezone": (
                    missing_home_timezone
                ),
                "missing_away_timezone": (
                    missing_away_timezone
                ),
                "missing_game_timezone": (
                    missing_game_timezone
                ),
            }
        )

        print(
            f"Wrote {len(output_rows)} "
            f"schedule rows to "
            f"{output_file}"
        )

        print(
            f"Pulled {len(pulled_rows)} "
            f"unique ESPN games"
        )


if __name__ == "__main__":
    main()
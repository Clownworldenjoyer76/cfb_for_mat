#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import sys
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


YEAR = 2026

CFB_DIR = Path(__file__).resolve().parents[2]

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

OUTPUT_FILE = (
    OUTPUT_DIR
    / f"{YEAR}_schedule.csv"
)

UPDATES_DIR = (
    OUTPUT_DIR
    / "updates"
)

ERROR_DIR = (
    CFB_DIR
    / "errors"
    / "00_intake"
)

LOG_FILE = (
    ERROR_DIR
    / "pull_schedule.txt"
)

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


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

UPDATES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ERROR_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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


def reset_log() -> None:
    LOG_FILE.write_text(
        "",
        encoding="utf-8",
    )


def log(message: str) -> None:
    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            message.rstrip()
            + "\n"
        )


def fatal(message: str) -> None:
    log(
        f"ERROR: {message}"
    )

    raise RuntimeError(
        message
    )


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        fatal(
            f"Missing required file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            fatal(
                f"Missing CSV header: {path}"
            )

        return [
            {
                clean(k): clean(v)
                for k, v
                in row.items()
            }
            for row in reader
        ]


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
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
            row.get("team_id")
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
            row.get("canonical_team"),
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
        fatal(
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
) -> dict[str, Any]:
    url = (
        "https://site.api.espn.com/"
        "apis/site/v2/sports/football/"
        "college-football/"
        f"teams/{team_id}/schedule"
        f"?season={YEAR}"
        "&seasontype=2"
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
            return json.loads(
                response
                .read()
                .decode("utf-8")
            )

    except Exception as e:
        log(
            f"TEAM_ID={team_id} "
            f"FETCH_ERROR={repr(e)}"
        )

        return {}


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
        team.get("shortDisplayName"),
        team.get("abbreviation"),
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
            team.get("displayName")
        )
        or clean(
            team.get(
                "shortDisplayName"
            )
        )
        or clean(
            team.get("location")
        )
    )


def parse_game_datetime(
    raw_date: str,
    game_timezone: str,
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
        except Exception:
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
        log(
            f"SKIP game_id={game_id} "
            "missing home/away team"
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

    neutral_value = competition.get(
        "neutralSite"
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
        )
    )

    season_obj = event.get(
        "season"
    )

    season = ""

    if isinstance(
        season_obj,
        dict,
    ):
        season = clean(
            season_obj.get(
                "year"
            )
        )

    season_type_obj = (
        event.get(
            "seasonType"
        )
    )

    season_type = ""

    if isinstance(
        season_type_obj,
        dict,
    ):
        season_type = (
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
            season
            or str(YEAR)
        ),
        "season_type": (
            season_type
            or "2"
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
        "home_timezone": home_timezone,
        "away_timezone": away_timezone,
        "game_timezone": game_timezone,
    }


def read_existing_schedule() -> list[
    dict[str, str]
]:
    if not OUTPUT_FILE.exists():
        return []

    with OUTPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

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
    reset_log()

    try:
        team_rows = read_csv(
            TEAM_MAP_FILE
        )

        stadium_rows = read_csv(
            STADIUM_MAP_FILE
        )

        team_ids, team_lookup = (
            build_team_maps(
                team_rows
            )
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
        total_events_seen = 0

        for team_id in team_ids:
            data = fetch_schedule(
                team_id
            )

            events = data.get(
                "events"
            )

            if not isinstance(
                events,
                list,
            ):
                log(
                    f"TEAM_ID={team_id} "
                    "events=INVALID"
                )
                continue

            log(
                f"TEAM_ID={team_id} "
                f"events={len(events)}"
            )

            if not events:
                continue

            successful_team_pulls += 1
            total_events_seen += len(
                events
            )

            for event in events:
                if not isinstance(
                    event,
                    dict,
                ):
                    continue

                row = event_to_row(
                    event,
                    team_lookup,
                    stadium_by_team,
                    stadium_by_stadium,
                )

                if row is None:
                    continue

                game_id = row[
                    "game_id"
                ]

                pulled[
                    game_id
                ] = row

        if not pulled:
            fatal(
                "ESPN returned zero usable "
                "2026 CFB schedule rows. "
                "Existing schedule was NOT overwritten."
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
                f"{YEAR}_schedule_"
                f"{timestamp}.csv"
            )
        )

        write_csv(
            update_file,
            pulled_rows,
        )

        existing_rows = (
            read_existing_schedule()
        )

        merged: dict[
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
                merged[
                    game_id
                ] = row

        for game_id, row in (
            pulled.items()
        ):
            merged[
                game_id
            ] = row

        output_rows = sort_rows(
            list(
                merged.values()
            )
        )

        if not output_rows:
            fatal(
                "No schedule rows available "
                "to write."
            )

        write_csv(
            OUTPUT_FILE,
            output_rows,
        )

        log(
            f"team_ids={len(team_ids)}"
        )

        log(
            "successful_team_pulls="
            f"{successful_team_pulls}"
        )

        log(
            f"events_seen={total_events_seen}"
        )

        log(
            "unique_games_pulled="
            f"{len(pulled_rows)}"
        )

        log(
            "schedule_rows_written="
            f"{len(output_rows)}"
        )

        print(
            f"Wrote {len(output_rows)} "
            f"schedule rows to "
            f"{OUTPUT_FILE}"
        )

        print(
            f"Pulled {len(pulled_rows)} "
            f"unique ESPN games"
        )

    except Exception:
        log(
            traceback.format_exc()
        )
        raise


if __name__ == "__main__":
    main()

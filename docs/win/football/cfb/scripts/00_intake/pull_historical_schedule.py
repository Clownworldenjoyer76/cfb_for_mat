#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
docs/win/football/cfb/scripts/00_intake/pull_historical_schedule.py

ONE-OFF HISTORICAL CFB SCHEDULE BACKFILL.

Pulls college-football schedules for:
    2021
    2022
    2023
    2024
    2025

using ESPN's team schedule endpoint.

Source:
    https://site.api.espn.com/apis/site/v2/sports/football/
    college-football/teams/{TEAM_ID}/schedule?season={YEAR}

Inputs:
    docs/win/football/cfb/config/mapping/team_map.csv
    docs/win/football/cfb/config/mapping/stadium_map.csv

Outputs:
    docs/win/football/cfb/00_intake/schedule/2021_schedule.csv
    docs/win/football/cfb/00_intake/schedule/2022_schedule.csv
    docs/win/football/cfb/00_intake/schedule/2023_schedule.csv
    docs/win/football/cfb/00_intake/schedule/2024_schedule.csv
    docs/win/football/cfb/00_intake/schedule/2025_schedule.csv

This script intentionally does not touch 2026.
"""

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


SEASONS = [2021, 2022, 2023, 2024, 2025]

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

TEAM_ID_COLUMN = "team_id"
CANONICAL_TEAM_COLUMN = "canonical_team"

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
    / "pull_historical_schedule.txt"
)

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

    text = str(value).strip()

    if text.lower() in {
        "none",
        "nan",
        "null",
    }:
        return ""

    return text


def key(value: Any) -> str:
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

    sys.exit(
        f"ERROR: {message}"
    )


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        fatal(
            f"Missing required file: {path}"
        )

    rows: list[dict[str, str]] = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            fatal(
                f"Missing header row: {path}"
            )

        for row in reader:
            rows.append(
                {
                    clean(k): clean(v)
                    for k, v in row.items()
                }
            )

    return rows


def require_columns(
    rows: list[dict[str, str]],
    required_cols: list[str],
    file_label: str,
) -> None:
    if not rows:
        fatal(
            f"{file_label} has no data rows"
        )

    available = set(
        rows[0].keys()
    )

    missing = [
        col
        for col in required_cols
        if col not in available
    ]

    if missing:
        fatal(
            f"{file_label} missing required columns: "
            f"{missing}"
        )


def build_team_maps(
    team_rows: list[dict[str, str]],
) -> tuple[
    list[str],
    dict[str, str],
]:
    require_columns(
        rows=team_rows,
        required_cols=[
            TEAM_ID_COLUMN,
            CANONICAL_TEAM_COLUMN,
        ],
        file_label=str(
            TEAM_MAP_FILE
        ),
    )

    team_ids: list[str] = []
    seen_team_ids: set[str] = set()

    team_lookup: dict[
        str,
        str,
    ] = {}

    optional_lookup_columns = [
        TEAM_ID_COLUMN,
        "team_abbr",
        "alias",
        "source_name",
        "canonical_team",
        "shortDisplayName",
        "location",
        "team_name",
        "nickname",
        "team_slug",
    ]

    for row_number, row in enumerate(
        team_rows,
        start=2,
    ):
        team_id = clean(
            row.get(
                TEAM_ID_COLUMN
            )
        )

        canonical_team = clean(
            row.get(
                CANONICAL_TEAM_COLUMN
            )
        )

        if not team_id:
            log(
                "WARNING: "
                f"team_map row {row_number} "
                f"missing {TEAM_ID_COLUMN}"
            )

        elif team_id not in seen_team_ids:
            team_ids.append(
                team_id
            )

            seen_team_ids.add(
                team_id
            )

        if not canonical_team:
            log(
                "WARNING: "
                f"team_map row {row_number} "
                f"missing "
                f"{CANONICAL_TEAM_COLUMN}"
            )

            continue

        for col in optional_lookup_columns:
            value = clean(
                row.get(col)
            )

            if value:
                team_lookup[
                    key(value)
                ] = canonical_team

    if not team_ids:
        fatal(
            "No team IDs found in "
            f"{TEAM_MAP_FILE}"
        )

    return (
        team_ids,
        team_lookup,
    )


def build_stadium_maps(
    stadium_rows: list[
        dict[str, str]
    ],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    require_columns(
        rows=stadium_rows,
        required_cols=[
            "team",
            "stadium",
            "timezone",
            "surface",
            "roof_type",
            "venue_id",
        ],
        file_label=str(
            STADIUM_MAP_FILE
        ),
    )

    by_team: dict[
        str,
        dict[str, str],
    ] = {}

    by_stadium: dict[
        str,
        dict[str, str],
    ] = {}

    by_venue_id: dict[
        str,
        dict[str, str],
    ] = {}

    for row_number, row in enumerate(
        stadium_rows,
        start=2,
    ):
        team_value = clean(
            row.get("team")
        )

        stadium_value = clean(
            row.get("stadium")
        )

        venue_id_value = clean(
            row.get("venue_id")
        )

        if team_value:
            by_team[
                key(team_value)
            ] = row

        else:
            log(
                "WARNING: "
                f"stadium_map row {row_number} "
                "missing team"
            )

        if stadium_value:
            by_stadium[
                key(stadium_value)
            ] = row

        if venue_id_value:
            by_venue_id[
                key(venue_id_value)
            ] = row

    return (
        by_team,
        by_stadium,
        by_venue_id,
    )


def fetch_team_schedule(
    team_id: str,
    season: int,
) -> dict[str, Any] | None:
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/football/"
        f"college-football/teams/{team_id}/schedule"
        f"?season={season}"
    )

    request = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:
            body = (
                response
                .read()
                .decode("utf-8")
            )

            return json.loads(
                body
            )

    except urllib.error.HTTPError as e:
        log(
            "WARNING: "
            f"season={season} "
            f"TEAM_ID={team_id} "
            f"HTTP error "
            f"{e.code} {e.reason}"
        )

        return None

    except urllib.error.URLError as e:
        log(
            "WARNING: "
            f"season={season} "
            f"TEAM_ID={team_id} "
            f"URL error "
            f"{e.reason}"
        )

        return None

    except Exception as e:
        log(
            "WARNING: "
            f"season={season} "
            f"TEAM_ID={team_id} "
            f"fetch failed: {e}"
        )

        return None


def get_first_competition(
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
    ):
        first = competitions[0]

        if isinstance(
            first,
            dict,
        ):
            return first

    return {}


def get_team_by_home_away(
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
            == home_away.casefold()
        ):
            team = competitor.get(
                "team"
            )

            if isinstance(
                team,
                dict,
            ):
                return team

    return {}


def map_team_name(
    team: dict[str, Any],
    team_lookup: dict[str, str],
    game_id: str,
    side: str,
) -> str:
    candidates = [
        team.get("id"),
        team.get("displayName"),
        team.get("abbreviation"),
        team.get("shortDisplayName"),
        team.get("name"),
        team.get("location"),
        team.get("nickname"),
        team.get("slug"),
    ]

    for candidate in candidates:
        mapped = team_lookup.get(
            key(candidate)
        )

        if mapped:
            return mapped

    fallback = (
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
        or clean(
            team.get("name")
        )
    )

    log(
        "WARNING: unmapped historical team; "
        "using ESPN fallback "
        f"game_id={game_id} "
        f"side={side} "
        f"id={clean(team.get('id'))} "
        f"displayName="
        f"{clean(team.get('displayName'))} "
        f"fallback={fallback}"
    )

    return fallback


def get_bool_text(
    value: Any,
) -> str:
    if isinstance(
        value,
        bool,
    ):
        return (
            "1"
            if value
            else "0"
        )

    text = clean(
        value
    ).casefold()

    if text in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return "1"

    if text in {
        "false",
        "0",
        "no",
        "n",
    }:
        return "0"

    return ""


def get_stadium_row(
    home_team: str,
    espn_stadium: str,
    espn_venue_id: str,
    neutral_site: str,
    stadium_by_team: dict[
        str,
        dict[str, str],
    ],
    stadium_by_stadium: dict[
        str,
        dict[str, str],
    ],
    stadium_by_venue_id: dict[
        str,
        dict[str, str],
    ],
    game_id: str,
) -> dict[str, str]:
    if neutral_site == "1":
        if espn_venue_id:
            match = (
                stadium_by_venue_id.get(
                    key(espn_venue_id)
                )
            )

            if match:
                return match

        match = (
            stadium_by_stadium.get(
                key(espn_stadium)
            )
        )

        if match:
            return match

        return {}

    home_match = (
        stadium_by_team.get(
            key(home_team)
        )
    )

    if home_match:
        return home_match

    if espn_venue_id:
        venue_match = (
            stadium_by_venue_id.get(
                key(espn_venue_id)
            )
        )

        if venue_match:
            return venue_match

    stadium_match = (
        stadium_by_stadium.get(
            key(espn_stadium)
        )
    )

    if stadium_match:
        return stadium_match

    return {}


def get_team_timezone(
    team: str,
    stadium_by_team: dict[
        str,
        dict[str, str],
    ],
) -> str:
    row = stadium_by_team.get(
        key(team),
        {},
    )

    return clean(
        row.get("timezone")
    )


def parse_event_datetime(
    raw_date: str,
    game_timezone: str,
) -> tuple[str, str]:
    if not raw_date:
        return "", ""

    try:
        dt_utc = datetime.fromisoformat(
            raw_date.replace(
                "Z",
                "+00:00",
            )
        )

        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(
                tzinfo=timezone.utc
            )

    except Exception:
        return "", ""

    if game_timezone:
        try:
            dt_local = (
                dt_utc.astimezone(
                    ZoneInfo(
                        game_timezone
                    )
                )
            )

        except Exception:
            dt_local = (
                dt_utc.astimezone(
                    timezone.utc
                )
            )

    else:
        dt_local = (
            dt_utc.astimezone(
                timezone.utc
            )
        )

    return (
        dt_local.strftime(
            "%Y-%m-%d"
        ),
        dt_local.strftime(
            "%H:%M"
        ),
    )


def build_row(
    event: dict[str, Any],
    requested_season: int,
    team_lookup: dict[str, str],
    stadium_by_team: dict[
        str,
        dict[str, str],
    ],
    stadium_by_stadium: dict[
        str,
        dict[str, str],
    ],
    stadium_by_venue_id: dict[
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
        get_first_competition(
            event
        )
    )

    home_team_obj = (
        get_team_by_home_away(
            competition,
            "home",
        )
    )

    away_team_obj = (
        get_team_by_home_away(
            competition,
            "away",
        )
    )

    home_team = (
        map_team_name(
            home_team_obj,
            team_lookup,
            game_id,
            "home",
        )
        if home_team_obj
        else ""
    )

    away_team = (
        map_team_name(
            away_team_obj,
            team_lookup,
            game_id,
            "away",
        )
        if away_team_obj
        else ""
    )

    neutral_site = (
        get_bool_text(
            competition.get(
                "neutralSite"
            )
        )
    )

    venue = competition.get(
        "venue"
    )

    if not isinstance(
        venue,
        dict,
    ):
        venue = {}

    espn_stadium = clean(
        venue.get("fullName")
    )

    espn_venue_id = clean(
        venue.get("id")
    )

    stadium_row = (
        get_stadium_row(
            home_team=home_team,
            espn_stadium=espn_stadium,
            espn_venue_id=espn_venue_id,
            neutral_site=neutral_site,
            stadium_by_team=stadium_by_team,
            stadium_by_stadium=stadium_by_stadium,
            stadium_by_venue_id=stadium_by_venue_id,
            game_id=game_id,
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

    home_timezone = (
        get_team_timezone(
            home_team,
            stadium_by_team,
        )
    )

    away_timezone = (
        get_team_timezone(
            away_team,
            stadium_by_team,
        )
    )

    game_timezone = clean(
        stadium_row.get(
            "timezone"
        )
    )

    game_date, game_time = (
        parse_event_datetime(
            raw_date=clean(
                event.get("date")
            ),
            game_timezone=(
                game_timezone
            ),
        )
    )

    season = ""
    season_obj = event.get(
        "season"
    )

    if isinstance(
        season_obj,
        dict,
    ):
        season = clean(
            season_obj.get(
                "year"
            )
        )

    if not season:
        season = str(
            requested_season
        )

    season_type = ""
    season_type_obj = (
        event.get(
            "seasonType"
        )
    )

    if isinstance(
        season_type_obj,
        dict,
    ):
        season_type = (
            clean(
                season_type_obj.get(
                    "abbreviation"
                )
            )
        )

    week = ""
    week_obj = event.get(
        "week"
    )

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
        "season": season,
        "season_type": season_type,
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


def write_csv(
    path: Path,
    rows: list[
        dict[str, str]
    ],
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
                    col: clean(
                        row.get(col)
                    )
                    for col
                    in OUTPUT_COLUMNS
                }
            )


def sort_key(
    row: dict[str, str],
) -> tuple[
    str,
    int,
    str,
]:
    week_text = clean(
        row.get("week")
    )

    try:
        week = int(
            week_text
        )
    except ValueError:
        week = 999

    return (
        clean(
            row.get("game_date")
        ),
        week,
        clean(
            row.get("game_id")
        ),
    )


def pull_season(
    season: int,
    team_ids: list[str],
    team_lookup: dict[str, str],
    stadium_by_team: dict[
        str,
        dict[str, str],
    ],
    stadium_by_stadium: dict[
        str,
        dict[str, str],
    ],
    stadium_by_venue_id: dict[
        str,
        dict[str, str],
    ],
    timestamp: str,
) -> int:
    rows_by_game_id: dict[
        str,
        dict[str, str],
    ] = {}

    api_calls_attempted = 0
    api_calls_succeeded = 0
    events_seen = 0
    duplicates_seen = 0

    print(
        f"Pulling {season}..."
    )

    log(
        ""
    )

    log(
        "=" * 80
    )

    log(
        f"START SEASON={season}"
    )

    log(
        "=" * 80
    )

    for team_id in team_ids:
        api_calls_attempted += 1

        data = fetch_team_schedule(
            team_id,
            season,
        )

        if not data:
            continue

        api_calls_succeeded += 1

        events = data.get(
            "events"
        )

        if not isinstance(
            events,
            list,
        ):
            log(
                "WARNING: "
                f"season={season} "
                f"TEAM_ID={team_id} "
                "response missing events list"
            )

            continue

        log(
            f"season={season} "
            f"TEAM_ID={team_id} "
            f"events_returned={len(events)}"
        )

        for event in events:
            if not isinstance(
                event,
                dict,
            ):
                continue

            events_seen += 1

            row = build_row(
                event=event,
                requested_season=season,
                team_lookup=team_lookup,
                stadium_by_team=stadium_by_team,
                stadium_by_stadium=stadium_by_stadium,
                stadium_by_venue_id=stadium_by_venue_id,
            )

            if row is None:
                continue

            # Keep only the requested season.
            if (
                clean(
                    row.get("season")
                )
                != str(season)
            ):
                continue

            game_id = row[
                "game_id"
            ]

            if (
                game_id
                in rows_by_game_id
            ):
                duplicates_seen += 1

            rows_by_game_id[
                game_id
            ] = row

    rows = list(
        rows_by_game_id.values()
    )

    rows.sort(
        key=sort_key
    )

    if not rows:
        fatal(
            "ESPN returned no team-schedule "
            f"games for season={season}"
        )

    output_file = (
        OUTPUT_DIR
        / f"{season}_schedule.csv"
    )

    updates_file = (
        UPDATES_DIR
        / (
            f"{season}_schedule_"
            f"{timestamp}.csv"
        )
    )

    write_csv(
        output_file,
        rows,
    )

    write_csv(
        updates_file,
        rows,
    )

    log(
        f"season={season} "
        f"api_calls_attempted="
        f"{api_calls_attempted}"
    )

    log(
        f"season={season} "
        f"api_calls_succeeded="
        f"{api_calls_succeeded}"
    )

    log(
        f"season={season} "
        f"events_seen="
        f"{events_seen}"
    )

    log(
        f"season={season} "
        f"duplicates_seen="
        f"{duplicates_seen}"
    )

    log(
        f"season={season} "
        f"unique_games="
        f"{len(rows)}"
    )

    log(
        f"season={season} "
        f"output={output_file}"
    )

    print(
        f"{season}: "
        f"{len(rows)} unique games"
    )

    return len(rows)


def main() -> None:
    reset_log()

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    log(
        "pull_historical_schedule.py "
        "started"
    )

    log(
        f"SEASONS={SEASONS}"
    )

    try:
        team_rows = read_csv(
            TEAM_MAP_FILE
        )

        stadium_rows = read_csv(
            STADIUM_MAP_FILE
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
            stadium_by_venue_id,
        ) = build_stadium_maps(
            stadium_rows
        )

        log(
            f"team_ids_found="
            f"{len(team_ids)}"
        )

        total_games = 0

        season_counts: dict[
            int,
            int,
        ] = {}

        for season in SEASONS:
            count = pull_season(
                season=season,
                team_ids=team_ids,
                team_lookup=team_lookup,
                stadium_by_team=stadium_by_team,
                stadium_by_stadium=stadium_by_stadium,
                stadium_by_venue_id=stadium_by_venue_id,
                timestamp=timestamp,
            )

            season_counts[
                season
            ] = count

            total_games += count

        log(
            ""
        )

        log(
            "HISTORICAL SUMMARY"
        )

        for season in SEASONS:
            log(
                f"{season}_games="
                f"{season_counts[season]}"
            )

        log(
            f"total_games="
            f"{total_games}"
        )

        log(
            "pull_historical_schedule.py "
            "finished"
        )

        print()
        print(
            "Historical schedule "
            "backfill complete."
        )

        for season in SEASONS:
            print(
                f"{season}: "
                f"{season_counts[season]} games"
            )

        print(
            f"Total: {total_games} games"
        )

    except SystemExit:
        raise

    except Exception:
        log(
            "ERROR: unhandled exception"
        )

        log(
            traceback.format_exc()
        )

        sys.exit(
            "ERROR: "
            "pull_historical_schedule.py "
            "failed. "
            f"See {LOG_FILE}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
docs/win/football/cfb/scripts/00_intake/pull_schedule.py

Pulls 2026 college-football schedule from ESPN team schedule API.

Source:
  https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{TEAM_ID}/schedule?season=2026

Inputs:
  docs/win/football/cfb/config/mapping/team_map.csv
  docs/win/football/cfb/config/mapping/stadium_map.csv

Main output:
  docs/win/football/cfb/00_intake/schedule/2026_schedule.csv

Per-run pulled output:
  docs/win/football/cfb/00_intake/schedule/updates/2026_schedule_YYYYMMDD_HHMMSS.csv

Summary / warnings:
  docs/win/football/cfb/errors/00_intake/pull_schedule.txt
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


YEAR = 2026

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

TEAM_MAP_FILE = CFB_DIR / "config" / "mapping" / "team_map.csv"
STADIUM_MAP_FILE = CFB_DIR / "config" / "mapping" / "stadium_map.csv"

OUTPUT_DIR = CFB_DIR / "00_intake" / "schedule"
OUTPUT_FILE = OUTPUT_DIR / f"{YEAR}_schedule.csv"

UPDATES_DIR = OUTPUT_DIR / "updates"

ERROR_DIR = CFB_DIR / "errors" / "00_intake"
LOG_FILE = ERROR_DIR / "pull_schedule.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPDATES_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)


def clean(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"none", "nan", "null"}:
        return ""

    return text


def key(value: Any) -> str:
    return clean(value).casefold()


def reset_log() -> None:
    LOG_FILE.write_text("", encoding="utf-8")


def log(message: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def fatal(message: str) -> None:
    log(f"ERROR: {message}")
    sys.exit(f"ERROR: {message}")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        fatal(f"Missing required file: {path}")

    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            fatal(f"Missing header row: {path}")

        for row in reader:
            rows.append({clean(k): clean(v) for k, v in row.items()})

    return rows


def read_existing_output(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            fatal(f"Missing header row: {path}")

        missing = [col for col in OUTPUT_COLUMNS if col not in reader.fieldnames]
        if missing:
            fatal(f"{path} missing required columns: {missing}")

        for row in reader:
            rows.append({col: clean(row.get(col)) for col in OUTPUT_COLUMNS})

    return rows


def require_columns(
    rows: list[dict[str, str]],
    required_cols: list[str],
    file_label: str,
) -> None:
    if not rows:
        fatal(f"{file_label} has no data rows")

    available = set(rows[0].keys())
    missing = [col for col in required_cols if col not in available]

    if missing:
        fatal(f"{file_label} missing required columns: {missing}")


def build_team_maps(
    team_rows: list[dict[str, str]],
) -> tuple[list[str], dict[str, str]]:
    require_columns(
        rows=team_rows,
        required_cols=[TEAM_ID_COLUMN, CANONICAL_TEAM_COLUMN],
        file_label=str(TEAM_MAP_FILE),
    )

    team_ids: list[str] = []
    seen_team_ids: set[str] = set()
    team_lookup: dict[str, str] = {}

    # CFB team_map.csv contains multiple alias rows per team.  Keep the
    # original NFL lookup fields and also consume the CFB alias/name fields.
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

    for row_number, row in enumerate(team_rows, start=2):
        team_id = clean(row.get(TEAM_ID_COLUMN))
        canonical_team = clean(row.get(CANONICAL_TEAM_COLUMN))

        if not team_id:
            log(f"WARNING: team_map row {row_number} missing {TEAM_ID_COLUMN}")
        elif team_id not in seen_team_ids:
            team_ids.append(team_id)
            seen_team_ids.add(team_id)

        if not canonical_team:
            log(f"WARNING: team_map row {row_number} missing {CANONICAL_TEAM_COLUMN}")
            continue

        for col in optional_lookup_columns:
            value = clean(row.get(col))
            if value:
                team_lookup[key(value)] = canonical_team

    if not team_ids:
        fatal(
            f"No TEAM_ID values found in {TEAM_MAP_FILE} "
            f"column {TEAM_ID_COLUMN}"
        )

    return team_ids, team_lookup


def build_stadium_maps(
    stadium_rows: list[dict[str, str]],
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
        file_label=str(STADIUM_MAP_FILE),
    )

    by_team: dict[str, dict[str, str]] = {}
    by_stadium: dict[str, dict[str, str]] = {}
    by_venue_id: dict[str, dict[str, str]] = {}

    for row_number, row in enumerate(stadium_rows, start=2):
        team_value = clean(row.get("team"))
        stadium_value = clean(row.get("stadium"))
        venue_id_value = clean(row.get("venue_id"))

        if team_value:
            by_team[key(team_value)] = row
        else:
            log(f"WARNING: stadium_map row {row_number} missing team")

        if stadium_value:
            by_stadium[key(stadium_value)] = row
        else:
            log(f"WARNING: stadium_map row {row_number} missing stadium")

        if venue_id_value:
            by_venue_id[key(venue_id_value)] = row
        else:
            log(f"WARNING: stadium_map row {row_number} missing venue_id")

    return by_team, by_stadium, by_venue_id


def fetch_team_schedule(team_id: str) -> dict[str, Any] | None:
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/football/"
        f"college-football/teams/{team_id}/schedule?season={YEAR}"
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
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)

    except urllib.error.HTTPError as e:
        log(
            f"WARNING: HTTP error for TEAM_ID={team_id}: "
            f"{e.code} {e.reason}"
        )
        return None

    except urllib.error.URLError as e:
        log(f"WARNING: URL error for TEAM_ID={team_id}: {e.reason}")
        return None

    except Exception as e:
        log(f"WARNING: Fetch failed for TEAM_ID={team_id}: {e}")
        return None


def get_first_competition(event: dict[str, Any]) -> dict[str, Any]:
    competitions = event.get("competitions")

    if isinstance(competitions, list) and competitions:
        first = competitions[0]
        if isinstance(first, dict):
            return first

    return {}


def get_team_by_home_away(
    competition: dict[str, Any],
    home_away: str,
) -> dict[str, Any]:
    competitors = competition.get("competitors")

    if not isinstance(competitors, list):
        return {}

    for competitor in competitors:
        if not isinstance(competitor, dict):
            continue

        if clean(competitor.get("homeAway")).casefold() == home_away.casefold():
            team = competitor.get("team")
            if isinstance(team, dict):
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
        mapped = team_lookup.get(key(candidate))
        if mapped:
            return mapped

    # CFB schedules routinely contain FCS/non-master opponents.  Preserve the
    # ESPN name instead of writing a blank team value, but keep a warning so
    # these can be reviewed or added to team_map.csv later if desired.
    fallback = (
        clean(team.get("displayName"))
        or clean(team.get("shortDisplayName"))
        or clean(team.get("location"))
        or clean(team.get("name"))
    )

    log(
        "WARNING: unmapped team; using ESPN fallback "
        f"game_id={game_id} side={side} "
        f"id={clean(team.get('id'))} "
        f"displayName={clean(team.get('displayName'))} "
        f"abbreviation={clean(team.get('abbreviation'))} "
        f"fallback={fallback}"
    )

    return fallback


def get_bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"

    text = clean(value).casefold()

    if text in {"true", "1", "yes", "y"}:
        return "1"

    if text in {"false", "0", "no", "n"}:
        return "0"

    return ""


def get_stadium_row(
    home_team: str,
    espn_stadium: str,
    espn_venue_id: str,
    neutral_site: str,
    stadium_by_team: dict[str, dict[str, str]],
    stadium_by_stadium: dict[str, dict[str, str]],
    stadium_by_venue_id: dict[str, dict[str, str]],
    game_id: str,
) -> dict[str, str]:
    if neutral_site == "1":
        if espn_venue_id:
            venue_id_match = stadium_by_venue_id.get(key(espn_venue_id))
            if venue_id_match:
                return venue_id_match

        stadium_match = stadium_by_stadium.get(key(espn_stadium))
        if stadium_match:
            return stadium_match

        log(
            "WARNING: neutral-site stadium not mapped "
            f"game_id={game_id} venue_id={espn_venue_id} "
            f"stadium={espn_stadium}"
        )
        return {}

    home_match = stadium_by_team.get(key(home_team))
    if home_match:
        return home_match

    if espn_venue_id:
        venue_id_match = stadium_by_venue_id.get(key(espn_venue_id))
        if venue_id_match:
            return venue_id_match

    stadium_match = stadium_by_stadium.get(key(espn_stadium))
    if stadium_match:
        return stadium_match

    log(
        "WARNING: home team stadium row not mapped "
        f"game_id={game_id} home_team={home_team} "
        f"venue_id={espn_venue_id} stadium={espn_stadium}"
    )

    return {}


def get_team_timezone(
    team: str,
    stadium_by_team: dict[str, dict[str, str]],
    game_id: str,
    side: str,
) -> str:
    row = stadium_by_team.get(key(team), {})

    timezone_value = clean(row.get("timezone"))

    if not timezone_value:
        log(
            f"WARNING: missing {side}_timezone "
            f"game_id={game_id} team={team}"
        )

    return timezone_value


def parse_event_datetime(
    raw_date: str,
    game_timezone: str,
    game_id: str,
) -> tuple[str, str]:
    if not raw_date:
        log(f"WARNING: missing event.date game_id={game_id}")
        return "", ""

    try:
        dt_utc = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))

        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)

    except Exception as e:
        log(
            f"WARNING: could not parse event.date "
            f"game_id={game_id} date={raw_date} error={e}"
        )
        return "", ""

    if game_timezone:
        try:
            dt_local = dt_utc.astimezone(ZoneInfo(game_timezone))
        except Exception as e:
            log(
                "WARNING: invalid game_timezone; using UTC datetime "
                f"game_id={game_id} game_timezone={game_timezone} error={e}"
            )
            dt_local = dt_utc.astimezone(timezone.utc)
    else:
        log(
            f"WARNING: missing game_timezone; using UTC datetime "
            f"game_id={game_id}"
        )
        dt_local = dt_utc.astimezone(timezone.utc)

    return dt_local.strftime("%Y-%m-%d"), dt_local.strftime("%H:%M")


def build_row(
    event: dict[str, Any],
    team_lookup: dict[str, str],
    stadium_by_team: dict[str, dict[str, str]],
    stadium_by_stadium: dict[str, dict[str, str]],
    stadium_by_venue_id: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    game_id = clean(event.get("id"))

    if not game_id:
        log("WARNING: skipped event with missing id")
        return None

    competition = get_first_competition(event)

    home_team_obj = get_team_by_home_away(competition, "home")
    away_team_obj = get_team_by_home_away(competition, "away")

    home_team = (
        map_team_name(home_team_obj, team_lookup, game_id, "home")
        if home_team_obj
        else ""
    )
    away_team = (
        map_team_name(away_team_obj, team_lookup, game_id, "away")
        if away_team_obj
        else ""
    )

    if not home_team:
        log(f"WARNING: missing mapped home_team game_id={game_id}")

    if not away_team:
        log(f"WARNING: missing mapped away_team game_id={game_id}")

    neutral_site = get_bool_text(competition.get("neutralSite"))

    if neutral_site == "":
        log(f"WARNING: missing neutral_site game_id={game_id}")

    venue = competition.get("venue")
    if not isinstance(venue, dict):
        venue = {}

    espn_stadium = clean(venue.get("fullName"))
    espn_venue_id = clean(venue.get("id"))

    stadium_row = get_stadium_row(
        home_team=home_team,
        espn_stadium=espn_stadium,
        espn_venue_id=espn_venue_id,
        neutral_site=neutral_site,
        stadium_by_team=stadium_by_team,
        stadium_by_stadium=stadium_by_stadium,
        stadium_by_venue_id=stadium_by_venue_id,
        game_id=game_id,
    )

    # If a neutral venue is not in stadium_map_cfb.csv, retain ESPN's venue
    # name rather than throwing it away.  Roof/surface/timezone remain blank
    # unless we have an actual mapped venue record.
    stadium = clean(stadium_row.get("stadium")) or espn_stadium
    roof = clean(stadium_row.get("roof_type"))
    surface = clean(stadium_row.get("surface"))

    if not stadium:
        log(f"WARNING: missing stadium game_id={game_id}")

    if not roof:
        log(f"WARNING: missing roof game_id={game_id}")

    if not surface:
        log(f"WARNING: missing surface game_id={game_id}")

    home_timezone = get_team_timezone(
        home_team,
        stadium_by_team,
        game_id,
        "home",
    )
    away_timezone = get_team_timezone(
        away_team,
        stadium_by_team,
        game_id,
        "away",
    )

    game_timezone = clean(stadium_row.get("timezone"))

    if not game_timezone:
        log(f"WARNING: missing game_timezone game_id={game_id}")

    game_date, game_time = parse_event_datetime(
        raw_date=clean(event.get("date")),
        game_timezone=game_timezone,
        game_id=game_id,
    )

    season = ""
    season_obj = event.get("season")
    if isinstance(season_obj, dict):
        season = clean(season_obj.get("year"))

    if not season:
        log(f"WARNING: missing season game_id={game_id}")

    season_type = ""
    season_type_obj = event.get("seasonType")
    if isinstance(season_type_obj, dict):
        season_type = clean(season_type_obj.get("abbreviation"))

    if not season_type:
        log(f"WARNING: missing season_type game_id={game_id}")

    week = ""
    week_obj = event.get("week")
    if isinstance(week_obj, dict):
        week = clean(week_obj.get("number"))

    if not week:
        log(f"WARNING: missing week game_id={game_id}")

    if not game_date:
        log(f"WARNING: missing game_date game_id={game_id}")

    if not game_time:
        log(f"WARNING: missing game_time game_id={game_id}")

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


def rows_equal(a: dict[str, str], b: dict[str, str]) -> bool:
    return all(
        clean(a.get(col)) == clean(b.get(col))
        for col in OUTPUT_COLUMNS
    )


def changed_columns(
    a: dict[str, str],
    b: dict[str, str],
) -> list[str]:
    return [
        col
        for col in OUTPUT_COLUMNS
        if clean(a.get(col)) != clean(b.get(col))
    ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {col: clean(row.get(col)) for col in OUTPUT_COLUMNS}
            )


def get_updates_file() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return UPDATES_DIR / f"{YEAR}_schedule_{timestamp}.csv"


def main() -> None:
    reset_log()

    updates_file = get_updates_file()

    log("pull_schedule.py started")
    log("LEAGUE=college-football")
    log(f"YEAR={YEAR}")
    log(f"TEAM_MAP_FILE={TEAM_MAP_FILE}")
    log(f"STADIUM_MAP_FILE={STADIUM_MAP_FILE}")
    log(f"OUTPUT_FILE={OUTPUT_FILE}")
    log(f"UPDATES_FILE={updates_file}")

    try:
        team_rows = read_csv(TEAM_MAP_FILE)
        stadium_rows = read_csv(STADIUM_MAP_FILE)
        existing_rows = read_existing_output(OUTPUT_FILE)

        team_ids, team_lookup = build_team_maps(team_rows)
        (
            stadium_by_team,
            stadium_by_stadium,
            stadium_by_venue_id,
        ) = build_stadium_maps(stadium_rows)

        log(f"team_ids_found={len(team_ids)}")
        log(f"existing_rows_found={len(existing_rows)}")

        pulled_rows_by_game_id: dict[str, dict[str, str]] = {}
        api_calls_attempted = 0
        api_calls_succeeded = 0
        events_seen = 0
        duplicate_events_seen = 0
        duplicate_events_rewritten = 0

        for team_id in team_ids:
            api_calls_attempted += 1

            data = fetch_team_schedule(team_id)

            if not data:
                continue

            api_calls_succeeded += 1

            events = data.get("events")

            if not isinstance(events, list):
                log(
                    f"WARNING: TEAM_ID={team_id} "
                    f"response missing events list"
                )
                continue

            log(
                f"TEAM_ID={team_id} "
                f"events_returned={len(events)}"
            )

            for event in events:
                if not isinstance(event, dict):
                    log(
                        f"WARNING: TEAM_ID={team_id} "
                        f"skipped non-dict event"
                    )
                    continue

                events_seen += 1

                game_id = clean(event.get("id"))

                if not game_id:
                    log(
                        f"WARNING: TEAM_ID={team_id} "
                        f"skipped event missing id"
                    )
                    continue

                row = build_row(
                    event=event,
                    team_lookup=team_lookup,
                    stadium_by_team=stadium_by_team,
                    stadium_by_stadium=stadium_by_stadium,
                    stadium_by_venue_id=stadium_by_venue_id,
                )

                if row is None:
                    continue

                if game_id in pulled_rows_by_game_id:
                    duplicate_events_seen += 1

                    previous_row = pulled_rows_by_game_id[game_id]
                    if not rows_equal(previous_row, row):
                        duplicate_events_rewritten += 1
                        log(
                            "WARNING: duplicate game_id pulled "
                            "with changed row; latest row kept "
                            f"game_id={game_id} TEAM_ID={team_id} "
                            f"changed_columns="
                            f"{changed_columns(previous_row, row)}"
                        )
                    else:
                        log(
                            "WARNING: duplicate game_id pulled "
                            "with same row "
                            f"game_id={game_id} TEAM_ID={team_id}"
                        )

                pulled_rows_by_game_id[game_id] = row

        pulled_rows = list(pulled_rows_by_game_id.values())
        write_csv(updates_file, pulled_rows)

        existing_rows_by_game_id: dict[str, dict[str, str]] = {}
        duplicate_existing_game_ids = 0

        for row in existing_rows:
            game_id = clean(row.get("game_id"))

            if not game_id:
                log("WARNING: existing output row missing game_id")
                continue

            if game_id in existing_rows_by_game_id:
                duplicate_existing_game_ids += 1
                log(
                    "WARNING: duplicate existing game_id found; "
                    "latest existing row kept "
                    f"game_id={game_id}"
                )

            existing_rows_by_game_id[game_id] = row

        merged_rows_by_game_id: dict[str, dict[str, str]] = dict(
            existing_rows_by_game_id
        )

        added_rows = 0
        updated_rows = 0
        unchanged_rows = 0

        for game_id, pulled_row in pulled_rows_by_game_id.items():
            existing_row = existing_rows_by_game_id.get(game_id)

            if existing_row is None:
                added_rows += 1
                merged_rows_by_game_id[game_id] = pulled_row
                log(f"ADDED: game_id={game_id}")
                continue

            if rows_equal(existing_row, pulled_row):
                unchanged_rows += 1
                continue

            updated_rows += 1
            merged_rows_by_game_id[game_id] = pulled_row
            log(
                f"UPDATED: game_id={game_id} "
                f"changed_columns="
                f"{changed_columns(existing_row, pulled_row)}"
            )

        missing_from_new_pull = 0

        for game_id in existing_rows_by_game_id:
            if game_id not in pulled_rows_by_game_id:
                missing_from_new_pull += 1
                log(
                    f"KEPT_MISSING_FROM_NEW_PULL: "
                    f"game_id={game_id}"
                )

        output_rows = list(merged_rows_by_game_id.values())

        write_csv(OUTPUT_FILE, output_rows)

        log(f"api_calls_attempted={api_calls_attempted}")
        log(f"api_calls_succeeded={api_calls_succeeded}")
        log(f"events_seen={events_seen}")
        log(f"duplicate_events_seen={duplicate_events_seen}")
        log(
            f"duplicate_events_rewritten="
            f"{duplicate_events_rewritten}"
        )
        log(
            f"duplicate_existing_game_ids="
            f"{duplicate_existing_game_ids}"
        )
        log(f"pulled_unique_games={len(pulled_rows)}")
        log(
            f"existing_unique_games="
            f"{len(existing_rows_by_game_id)}"
        )
        log(f"added_rows={added_rows}")
        log(f"updated_rows={updated_rows}")
        log(f"unchanged_rows={unchanged_rows}")
        log(f"missing_from_new_pull={missing_from_new_pull}")
        log(f"main_unique_games_written={len(output_rows)}")
        log("pull_schedule.py finished")

        print(
            f"Wrote {len(output_rows)} rows to {OUTPUT_FILE}"
        )
        print(
            f"Wrote {len(pulled_rows)} pulled rows "
            f"to {updates_file}"
        )
        print(
            f"Summary/warnings written to {LOG_FILE}"
        )

    except SystemExit:
        raise

    except Exception:
        log("ERROR: unhandled exception")
        log(traceback.format_exc())
        sys.exit(
            f"ERROR: pull_schedule.py failed. See {LOG_FILE}"
        )


if __name__ == "__main__":
    main()

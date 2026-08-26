#!/usr/bin/env python3
"""
build_travel.py

Builds:
    docs/win/football/cfb/data/travel/{season}_week_{week}_travel.csv

Reads:
    docs/win/football/cfb/00_intake/schedule/weekly/
        week_{week}_CFB_weekly_schedule.csv

    docs/win/football/cfb/config/mapping/stadium_map.csv

Purpose:
    - Resolve the actual scheduled game venue.
    - Calculate away-team travel from its home stadium to the game venue.
    - Calculate home-team travel from its home stadium to the game venue.
    - Calculate each team's timezone change to the game venue.
    - Correctly handle neutral-site games.

Important:
    - The schedule's stadium field is the source of truth for the game venue.
    - Neutral-site games never default to the listed home team's stadium.
    - If a neutral venue cannot be resolved from stadium_map.csv, travel
      values are left blank instead of using an incorrect venue.

Manual run only.
"""

import csv
import glob
import math
import os
import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo


BASE_DIR = "docs/win/football/cfb"

SCHEDULE_DIR = os.path.join(
    BASE_DIR,
    "00_intake/schedule/weekly",
)

STADIUM_MAP_PATH = os.path.join(
    BASE_DIR,
    "config/mapping/stadium_map.csv",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data/travel",
)


OUTPUT_HEADERS = [
    "game_id",
    "away_team",
    "home_team",
    "stadium",
    "neutral_site_flag",
    "venue_resolution_status",
    "venue_lat",
    "venue_lon",
    "venue_timezone",
    "venue_country",
    "away_home_lat",
    "away_home_lon",
    "away_home_timezone",
    "away_miles_traveled",
    "away_time_zone_change_hours",
    "away_time_zones_crossed",
    "away_east_to_west",
    "away_west_to_east",
    "home_home_lat",
    "home_home_lon",
    "home_home_timezone",
    "home_miles_traveled",
    "home_time_zone_change_hours",
    "home_time_zones_crossed",
    "home_east_to_west",
    "home_west_to_east",
    "international_flag",
]


def clean(value):
    if value is None:
        return ""

    return str(value).strip()


def normalize_key(value):
    text = clean(value)

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        ch
        for ch in text
        if not unicodedata.combining(ch)
    )

    text = text.casefold()

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text,
    )


def strip_parenthetical(value):
    text = clean(value)

    if not text:
        return ""

    return re.sub(
        r"\s*\([^)]*\)\s*$",
        "",
        text,
    ).strip()


def parse_flag(value):
    text = clean(value).casefold()

    if text in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return 1

    return 0


def load_csv(path):
    with open(
        path,
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def load_stadium_maps():
    rows = load_csv(
        STADIUM_MAP_PATH
    )

    team_lookup = {}
    venue_lookup = {}

    for row in rows:

        team_key = normalize_key(
            row.get(
                "team",
                "",
            )
        )

        if team_key:
            team_lookup[
                team_key
            ] = row

        venue_names = {
            clean(
                row.get(
                    "stadium",
                    "",
                )
            ),
            clean(
                row.get(
                    "venue_full_name",
                    "",
                )
            ),
            strip_parenthetical(
                row.get(
                    "stadium",
                    "",
                )
            ),
            strip_parenthetical(
                row.get(
                    "venue_full_name",
                    "",
                )
            ),
        }

        for venue_name in venue_names:

            venue_key = normalize_key(
                venue_name
            )

            if not venue_key:
                continue

            venue_lookup.setdefault(
                venue_key,
                [],
            ).append(
                row
            )

    return (
        team_lookup,
        venue_lookup,
    )


def same_coordinates(
    row_a,
    row_b,
):
    try:
        return (
            abs(
                float(
                    row_a.get(
                        "latitude",
                        "",
                    )
                )
                - float(
                    row_b.get(
                        "latitude",
                        "",
                    )
                )
            )
            < 1e-7
            and
            abs(
                float(
                    row_a.get(
                        "longitude",
                        "",
                    )
                )
                - float(
                    row_b.get(
                        "longitude",
                        "",
                    )
                )
            )
            < 1e-7
        )

    except Exception:
        return False


def dedupe_venue_rows(
    rows,
):
    unique = []

    for row in rows:

        duplicate = False

        for existing in unique:

            if same_coordinates(
                row,
                existing,
            ):
                duplicate = True
                break

        if not duplicate:
            unique.append(
                row
            )

    return unique


def resolve_venue(
    game,
    home_team_row,
    venue_lookup,
    neutral_site_flag,
    log_lines,
):
    game_id = clean(
        game.get(
            "game_id",
            "",
        )
    )

    scheduled_stadium = clean(
        game.get(
            "stadium",
            "",
        )
    )

    game_timezone = clean(
        game.get(
            "game_timezone",
            "",
        )
    )

    venue_candidates = []

    venue_names = {
        scheduled_stadium,
        strip_parenthetical(
            scheduled_stadium
        ),
    }

    for venue_name in venue_names:

        venue_key = normalize_key(
            venue_name
        )

        if venue_key:
            venue_candidates.extend(
                venue_lookup.get(
                    venue_key,
                    [],
                )
            )

    venue_candidates = (
        dedupe_venue_rows(
            venue_candidates
        )
    )

    if len(
        venue_candidates
    ) == 1:

        return (
            venue_candidates[0],
            "resolved_schedule_stadium",
        )

    if (
        len(
            venue_candidates
        )
        > 1
        and game_timezone
    ):

        timezone_matches = [
            row
            for row
            in venue_candidates
            if clean(
                row.get(
                    "timezone",
                    "",
                )
            )
            == game_timezone
        ]

        timezone_matches = (
            dedupe_venue_rows(
                timezone_matches
            )
        )

        if len(
            timezone_matches
        ) == 1:

            return (
                timezone_matches[0],
                "resolved_schedule_stadium_timezone",
            )

    if (
        len(
            venue_candidates
        )
        > 1
        and home_team_row
        is not None
    ):

        home_matches = [
            row
            for row
            in venue_candidates
            if same_coordinates(
                row,
                home_team_row,
            )
        ]

        if len(
            home_matches
        ) == 1:

            return (
                home_matches[0],
                "resolved_schedule_stadium_home_match",
            )

    if len(
        venue_candidates
    ) > 1:

        log_lines.append(
            f"ERROR: game_id={game_id} "
            "ambiguous scheduled venue "
            f"stadium='{scheduled_stadium}' "
            f"candidates={len(venue_candidates)}"
        )

        return (
            None,
            "ambiguous_schedule_stadium",
        )

    if neutral_site_flag == 1:

        log_lines.append(
            f"ERROR: game_id={game_id} "
            "neutral-site venue could not "
            "be resolved from stadium_map.csv; "
            f"stadium='{scheduled_stadium}'"
        )

        return (
            None,
            "unresolved_neutral_venue",
        )

    if home_team_row is not None:

        log_lines.append(
            f"WARNING: game_id={game_id} "
            f"scheduled stadium '{scheduled_stadium}' "
            "not found in stadium_map.csv; "
            "using listed home team's home stadium "
            "because game is non-neutral"
        )

        return (
            home_team_row,
            "resolved_non_neutral_home_fallback",
        )

    log_lines.append(
        f"ERROR: game_id={game_id} "
        "venue could not be resolved; "
        f"stadium='{scheduled_stadium}'"
    )

    return (
        None,
        "unresolved_venue",
    )


def haversine_miles(
    lat1,
    lon1,
    lat2,
    lon2,
):
    radius_miles = 3958.8

    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

    dphi = math.radians(
        lat2 - lat1
    )

    dlambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(
            dphi / 2
        )
        ** 2
        +
        math.cos(
            phi1
        )
        *
        math.cos(
            phi2
        )
        *
        math.sin(
            dlambda / 2
        )
        ** 2
    )

    c = (
        2
        * math.asin(
            math.sqrt(
                a
            )
        )
    )

    return (
        radius_miles
        * c
    )


def utc_offset_hours(
    tz_name,
    game_date,
):
    tz_name = clean(
        tz_name
    )

    game_date = clean(
        game_date
    )

    if (
        not tz_name
        or not game_date
    ):
        return None

    try:
        dt = datetime.strptime(
            f"{game_date} 12:00",
            "%Y-%m-%d %H:%M",
        ).replace(
            tzinfo=ZoneInfo(
                tz_name
            )
        )

        offset = dt.utcoffset()

        if offset is None:
            return None

        return (
            offset.total_seconds()
            / 3600
        )

    except Exception:
        return None


def travel_direction(
    origin_lon,
    destination_lon,
):
    try:
        origin = float(
            origin_lon
        )

        destination = float(
            destination_lon
        )

    except Exception:
        return (
            "",
            "",
        )

    if destination > origin:
        return (
            0,
            1,
        )

    if destination < origin:
        return (
            1,
            0,
        )

    return (
        0,
        0,
    )


def team_travel_values(
    team_row,
    venue_row,
    venue_timezone,
    game_date,
    game_id,
    team_label,
    log_lines,
):
    if team_row is None:

        return {
            "home_lat": "",
            "home_lon": "",
            "home_timezone": "",
            "miles_traveled": "",
            "time_zone_change_hours": "",
            "time_zones_crossed": "",
            "east_to_west": "",
            "west_to_east": "",
        }

    team_lat = clean(
        team_row.get(
            "latitude",
            "",
        )
    )

    team_lon = clean(
        team_row.get(
            "longitude",
            "",
        )
    )

    team_timezone = clean(
        team_row.get(
            "timezone",
            "",
        )
    )

    values = {
        "home_lat": team_lat,
        "home_lon": team_lon,
        "home_timezone": team_timezone,
        "miles_traveled": "",
        "time_zone_change_hours": "",
        "time_zones_crossed": "",
        "east_to_west": "",
        "west_to_east": "",
    }

    if venue_row is None:
        return values

    venue_lat = clean(
        venue_row.get(
            "latitude",
            "",
        )
    )

    venue_lon = clean(
        venue_row.get(
            "longitude",
            "",
        )
    )

    try:
        values[
            "miles_traveled"
        ] = round(
            haversine_miles(
                float(
                    team_lat
                ),
                float(
                    team_lon
                ),
                float(
                    venue_lat
                ),
                float(
                    venue_lon
                ),
            ),
            1,
        )

    except Exception as exc:

        log_lines.append(
            f"ERROR: game_id={game_id} "
            f"failed computing "
            f"{team_label}_miles_traveled: "
            f"{exc}"
        )

    team_offset = (
        utc_offset_hours(
            team_timezone,
            game_date,
        )
    )

    venue_offset = (
        utc_offset_hours(
            venue_timezone,
            game_date,
        )
    )

    if (
        team_offset is not None
        and venue_offset is not None
    ):

        signed_change = (
            venue_offset
            - team_offset
        )

        values[
            "time_zone_change_hours"
        ] = round(
            signed_change,
            1,
        )

        values[
            "time_zones_crossed"
        ] = round(
            abs(
                signed_change
            ),
            1,
        )

    else:

        log_lines.append(
            f"WARNING: game_id={game_id} "
            f"could not compute "
            f"{team_label} timezone change "
            f"(team_timezone='{team_timezone}', "
            f"venue_timezone='{venue_timezone}', "
            f"game_date='{game_date}')"
        )

    (
        east_to_west,
        west_to_east,
    ) = travel_direction(
        team_lon,
        venue_lon,
    )

    values[
        "east_to_west"
    ] = east_to_west

    values[
        "west_to_east"
    ] = west_to_east

    return values


def build_row(
    game,
    team_lookup,
    venue_lookup,
    log_lines,
):
    game_id = clean(
        game.get(
            "game_id",
            "",
        )
    )

    away_team = clean(
        game.get(
            "away_team",
            "",
        )
    )

    home_team = clean(
        game.get(
            "home_team",
            "",
        )
    )

    scheduled_stadium = clean(
        game.get(
            "stadium",
            "",
        )
    )

    game_date = clean(
        game.get(
            "game_date",
            "",
        )
    )

    neutral_site_flag = (
        parse_flag(
            game.get(
                "neutral_site",
                "",
            )
        )
    )

    away_team_row = (
        team_lookup.get(
            normalize_key(
                away_team
            )
        )
    )

    home_team_row = (
        team_lookup.get(
            normalize_key(
                home_team
            )
        )
    )

    if away_team_row is None:

        log_lines.append(
            f"ERROR: game_id={game_id} "
            "no stadium_map match "
            f"for away_team='{away_team}'"
        )

    if home_team_row is None:

        log_lines.append(
            f"ERROR: game_id={game_id} "
            "no stadium_map match "
            f"for home_team='{home_team}'"
        )

    (
        venue_row,
        venue_resolution_status,
    ) = resolve_venue(
        game,
        home_team_row,
        venue_lookup,
        neutral_site_flag,
        log_lines,
    )

    venue_lat = clean(
        venue_row.get(
            "latitude",
            "",
        )
        if venue_row
        else ""
    )

    venue_lon = clean(
        venue_row.get(
            "longitude",
            "",
        )
        if venue_row
        else ""
    )

    venue_timezone = clean(
        game.get(
            "game_timezone",
            "",
        )
    )

    if (
        not venue_timezone
        and venue_row
        is not None
    ):

        venue_timezone = clean(
            venue_row.get(
                "timezone",
                "",
            )
        )

    venue_country = clean(
        venue_row.get(
            "venue_country",
            "",
        )
        if venue_row
        else ""
    )

    away_values = (
        team_travel_values(
            away_team_row,
            venue_row,
            venue_timezone,
            game_date,
            game_id,
            "away",
            log_lines,
        )
    )

    home_values = (
        team_travel_values(
            home_team_row,
            venue_row,
            venue_timezone,
            game_date,
            game_id,
            "home",
            log_lines,
        )
    )

    if venue_country:

        international_flag = (
            0
            if venue_country.upper()
            == "USA"
            else 1
        )

    else:
        international_flag = ""

    return {
        "game_id": game_id,
        "away_team": away_team,
        "home_team": home_team,
        "stadium": scheduled_stadium,
        "neutral_site_flag": neutral_site_flag,
        "venue_resolution_status": (
            venue_resolution_status
        ),
        "venue_lat": venue_lat,
        "venue_lon": venue_lon,
        "venue_timezone": venue_timezone,
        "venue_country": venue_country,
        "away_home_lat": (
            away_values[
                "home_lat"
            ]
        ),
        "away_home_lon": (
            away_values[
                "home_lon"
            ]
        ),
        "away_home_timezone": (
            away_values[
                "home_timezone"
            ]
        ),
        "away_miles_traveled": (
            away_values[
                "miles_traveled"
            ]
        ),
        "away_time_zone_change_hours": (
            away_values[
                "time_zone_change_hours"
            ]
        ),
        "away_time_zones_crossed": (
            away_values[
                "time_zones_crossed"
            ]
        ),
        "away_east_to_west": (
            away_values[
                "east_to_west"
            ]
        ),
        "away_west_to_east": (
            away_values[
                "west_to_east"
            ]
        ),
        "home_home_lat": (
            home_values[
                "home_lat"
            ]
        ),
        "home_home_lon": (
            home_values[
                "home_lon"
            ]
        ),
        "home_home_timezone": (
            home_values[
                "home_timezone"
            ]
        ),
        "home_miles_traveled": (
            home_values[
                "miles_traveled"
            ]
        ),
        "home_time_zone_change_hours": (
            home_values[
                "time_zone_change_hours"
            ]
        ),
        "home_time_zones_crossed": (
            home_values[
                "time_zones_crossed"
            ]
        ),
        "home_east_to_west": (
            home_values[
                "east_to_west"
            ]
        ),
        "home_west_to_east": (
            home_values[
                "west_to_east"
            ]
        ),
        "international_flag": (
            international_flag
        ),
    }


def process_week(
    season,
    week,
    schedule_path,
    team_lookup,
    venue_lookup,
    log_lines,
):
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{season}_week_{week}_travel.csv",
    )

    schedule_rows = load_csv(
        schedule_path
    )

    output_rows = [
        build_row(
            game,
            team_lookup,
            venue_lookup,
            log_lines,
        )
        for game
        in schedule_rows
    ]

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_HEADERS,
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    resolved = sum(
        1
        for row
        in output_rows
        if (
            row["venue_lat"]
            and row["venue_lon"]
        )
    )

    neutral_games = sum(
        1
        for row
        in output_rows
        if row[
            "neutral_site_flag"
        ]
        == 1
    )

    unresolved_neutral = sum(
        1
        for row
        in output_rows
        if (
            row[
                "neutral_site_flag"
            ]
            == 1
            and not row[
                "venue_lat"
            ]
        )
    )

    print(
        f"Wrote {len(output_rows)} rows "
        f"to {output_path} | "
        f"venues_resolved={resolved} | "
        f"neutral_games={neutral_games} | "
        f"unresolved_neutral="
        f"{unresolved_neutral}"
    )


def main():
    log_lines = []

    (
        team_lookup,
        venue_lookup,
    ) = load_stadium_maps()

    schedule_files = sorted(
        glob.glob(
            os.path.join(
                SCHEDULE_DIR,
                "week_*_CFB_weekly_schedule.csv",
            )
        )
    )

    if not schedule_files:

        print(
            "WARNING: no weekly schedule "
            f"files found in {SCHEDULE_DIR}"
        )

        return

    for schedule_path in schedule_files:

        filename = os.path.basename(
            schedule_path
        )

        match = re.fullmatch(
            r"week_(\d+)_CFB_weekly_schedule\.csv",
            filename,
        )

        if not match:

            log_lines.append(
                "WARNING: skipped "
                "unrecognized file name: "
                f"{filename}"
            )

            continue

        week = int(
            match.group(1)
        )

        rows = load_csv(
            schedule_path
        )

        if not rows:

            log_lines.append(
                "WARNING: skipped empty "
                f"weekly schedule: {schedule_path}"
            )

            continue

        season = clean(
            rows[0].get(
                "season",
                "",
            )
        )

        if not season:

            log_lines.append(
                "ERROR: weekly schedule "
                f"missing season: {schedule_path}"
            )

            continue

        process_week(
            season,
            week,
            schedule_path,
            team_lookup,
            venue_lookup,
            log_lines,
        )

    if log_lines:

        print(
            "Issues encountered:"
        )

        for line in log_lines:
            print(
                line
            )


if __name__ == "__main__":
    main()
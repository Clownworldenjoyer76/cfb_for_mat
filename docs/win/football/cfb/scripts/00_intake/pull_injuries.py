#!/usr/bin/env python3
"""
pull_injuries.py

Pull the configured CFB injury report from ESPN's site API, validate provider
season/team/player identities, and atomically publish the season injury file.

Inputs:
    docs/win/football/cfb/config/current_week.yaml
    docs/win/football/cfb/data/master/league_master.csv
    docs/win/football/cfb/config/mapping/team_map.csv

Output:
    docs/win/football/cfb/00_intake/injuries/{season}_injuries.csv

This script is an automated CFB pipeline intake step.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import urllib.parse
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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


CURRENT_WEEK_CONFIG_PATH = CFB_ROOT / "config" / "current_week.yaml"
LEAGUE_MASTER_PATH = CFB_ROOT / "data" / "master" / "league_master.csv"
TEAM_MAP_PATH = CFB_ROOT / "config" / "mapping" / "team_map.csv"
OUTPUT_DIR = CFB_ROOT / "00_intake" / "injuries"
REPORT_ROOT = CFB_ROOT / "errors"

SCRIPT_VERSION = "cfb-pull-injuries-v4-stale-zero-current-2026-09-15"

# Keep intake freshness aligned with the projection's default
# injury-report freshness window. ESPN's payload timestamp is used
# as the reference clock so runner clock drift cannot change results.
MAX_REPORT_AGE_DAYS = 60.0

INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/"
    "college-football/injuries"
)

ESPN_SITE_HOST = "site.api.espn.com"
ESPN_WEB_HOSTS = {
    "espn.com",
    "www.espn.com",
}

PLAYER_ID_PATTERN = re.compile(
    r"/id/(\d+)(?:/|$)"
)

OUTPUT_HEADERS = [
    "season",
    "team",
    "player_id",
    "player_name",
    "position",
    "game_status",
    "report_date",
]

@dataclass
class RuntimeState:
    request_count: int = 0
    request_failures: list[dict[str, str]] = field(
        default_factory=list
    )
    provider_status: str = ""
    provider_timestamp: str = ""
    provider_timestamp_utc: datetime | None = None
    provider_season: int | None = None
    provider_season_type: int | None = None
    provider_team_group_count: int = 0
    provider_team_groups_with_injuries: int = 0
    provider_team_groups_without_injuries: int = 0
    raw_injury_count: int = 0
    fresh_injury_count: int = 0
    stale_injury_count: int = 0
    duplicate_injury_id_count: int = 0
    duplicate_output_identity_count: int = 0
    missing_player_id_count: int = 0
    report_year_mismatch_count: int = 0
    foreign_team_ids: set[str] = field(
        default_factory=set
    )
    provider_team_ids: set[str] = field(
        default_factory=set
    )
    report_dates_utc: list[datetime] = field(
        default_factory=list
    )


class InjuryValidationError(RuntimeError):
    pass


def parse_positive_int_text(
    value: object,
    *,
    label: str,
) -> str:
    if isinstance(value, bool):
        raise InjuryValidationError(
            f"{label} must be a positive integer, not boolean"
        )

    text = str(value or "").strip()

    if not re.fullmatch(r"\d+", text):
        raise InjuryValidationError(
            f"{label} must be a positive integer: {value!r}"
        )

    parsed = int(text)

    if parsed <= 0:
        raise InjuryValidationError(
            f"{label} must be positive: {parsed}"
        )

    return str(parsed)


def parse_positive_int(
    value: object,
    *,
    label: str,
) -> int:
    return int(
        parse_positive_int_text(
            value,
            label=label,
        )
    )


def load_current_week() -> tuple[int, int, int]:
    if not CURRENT_WEEK_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing current-week config: {CURRENT_WEEK_CONFIG_PATH}"
        )

    with CURRENT_WEEK_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            "Current-week config must contain a YAML mapping"
        )

    values: dict[str, int] = {}

    for key in (
        "season",
        "season_type",
        "week",
    ):
        if key not in payload:
            raise ValueError(
                f"Current-week config missing required key: {key}"
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


def load_authoritative_team_ids(
    *,
    season: int,
    season_type: int,
) -> list[str]:
    if not LEAGUE_MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing league master: {LEAGUE_MASTER_PATH}"
        )

    with LEAGUE_MASTER_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        required = {
            "team_id",
            "season",
            "season_type",
        }

        missing = sorted(
            required - set(fieldnames)
        )

        if missing:
            raise ValueError(
                "league_master.csv missing required columns: "
                f"{missing}"
            )

        team_ids: set[str] = set()

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            if None in row:
                raise ValueError(
                    "league_master.csv contains malformed row at "
                    f"CSV line {row_number}"
                )

            team_id = parse_positive_int_text(
                row.get("team_id"),
                label=(
                    "league_master.csv team_id at "
                    f"CSV line {row_number}"
                ),
            )

            row_season = parse_positive_int(
                row.get("season"),
                label=(
                    "league_master.csv season at "
                    f"CSV line {row_number}"
                ),
            )

            row_season_type = parse_positive_int(
                row.get("season_type"),
                label=(
                    "league_master.csv season_type at "
                    f"CSV line {row_number}"
                ),
            )

            if row_season != season:
                raise ValueError(
                    "league_master.csv season mismatch at "
                    f"CSV line {row_number}: "
                    f"expected={season}, actual={row_season}"
                )

            if row_season_type != season_type:
                raise ValueError(
                    "league_master.csv season_type mismatch at "
                    f"CSV line {row_number}: "
                    f"expected={season_type}, "
                    f"actual={row_season_type}"
                )

            if team_id in team_ids:
                raise ValueError(
                    "league_master.csv contains duplicate "
                    f"team_id={team_id}"
                )

            team_ids.add(team_id)

    if not team_ids:
        raise ValueError(
            "league_master.csv contains no authoritative teams"
        )

    return sorted(
        team_ids,
        key=int,
    )



def _validate_canonical_team_name_uniqueness(
    canonical_by_id: dict[str, str],
) -> None:
    inverse: dict[str, str] = {}

    for team_id, canonical in canonical_by_id.items():
        prior_id = inverse.get(
            canonical
        )

        if prior_id is not None and prior_id != team_id:
            raise ValueError(
                "team_map.csv maps one canonical team name to "
                "multiple authoritative IDs: "
                f"canonical={canonical!r}, "
                f"team_ids={prior_id},{team_id}"
            )

        inverse[
            canonical
        ] = team_id


def load_canonical_team_names(
    authoritative_team_ids: list[str],
) -> dict[str, str]:
    if not TEAM_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Missing team map: {TEAM_MAP_PATH}"
        )

    authoritative_set = set(
        authoritative_team_ids
    )

    canonical_by_id: dict[str, str] = {}

    with TEAM_MAP_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        required = {
            "team_id",
            "canonical_team",
        }

        missing = sorted(
            required - set(fieldnames)
        )

        if missing:
            raise ValueError(
                "team_map.csv missing required columns: "
                f"{missing}"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            if None in row:
                raise ValueError(
                    "team_map.csv contains malformed row at "
                    f"CSV line {row_number}"
                )

            raw_team_id = str(
                row.get("team_id") or ""
            ).strip()

            if not raw_team_id:
                continue

            team_id = parse_positive_int_text(
                raw_team_id,
                label=(
                    "team_map.csv team_id at "
                    f"CSV line {row_number}"
                ),
            )

            if team_id not in authoritative_set:
                continue

            canonical = str(
                row.get("canonical_team") or ""
            ).strip()

            if not canonical:
                raise ValueError(
                    "team_map.csv has blank canonical_team for "
                    f"authoritative team_id={team_id} at "
                    f"CSV line {row_number}"
                )

            prior = canonical_by_id.get(
                team_id
            )

            if (
                prior is not None
                and prior != canonical
            ):
                raise ValueError(
                    "team_map.csv has conflicting canonical names for "
                    f"team_id={team_id}: "
                    f"{prior!r} vs {canonical!r}"
                )

            canonical_by_id[
                team_id
            ] = canonical

    missing_ids = sorted(
        authoritative_set
        - set(canonical_by_id),
        key=int,
    )

    if missing_ids:
        raise ValueError(
            "team_map.csv is missing canonical mappings for "
            f"authoritative team IDs: {missing_ids[:50]}"
        )

    _validate_canonical_team_name_uniqueness(
        canonical_by_id
    )

    return canonical_by_id


def output_path_for_season(
    season: int,
) -> Path:
    return (
        OUTPUT_DIR
        / f"{season}_injuries.csv"
    )


def injuries_url(
    *,
    season: int,
    season_type: int,
) -> str:
    query = urllib.parse.urlencode(
        {
            "season": season,
            "seasontype": season_type,
        }
    )

    return (
        f"{INJURIES_URL}?{query}"
    )


def validate_injuries_url(
    url: str,
) -> str:
    text = str(
        url or ""
    ).strip()

    parsed = urllib.parse.urlparse(
        text
    )

    if parsed.scheme != "https":
        raise InjuryValidationError(
            "Injuries URL must use HTTPS: "
            f"{text!r}"
        )

    if parsed.hostname != ESPN_SITE_HOST:
        raise InjuryValidationError(
            "Unexpected injuries URL host: "
            f"{text!r}"
        )

    return text


def fetch_json(
    url: str,
    *,
    state: RuntimeState,
    timeout: int = 30,
) -> dict:

    url = validate_injuries_url(
        url
    )

    state.request_count += 1

    request = Request(
        url,
        headers={
            "User-Agent": "cfb-pull-injuries/2.0",
            "Accept": "application/json",
        },
    )

    try:
        with open_https(
            request,
            allowed_hosts={ESPN_SITE_HOST},
            timeout=timeout,
        ) as response:
            status = response.status
            body = (
                response.read()
                .decode("utf-8")
            )

    except HTTPError as exc:
        error_body = ""

        try:
            error_body = (
                exc.read()
                .decode("utf-8")
            )
        except Exception:
            pass

        failure = {
            "url": url,
            "http_status": str(
                exc.code
            ),
            "error": (
                error_body
                or str(exc)
            ),
        }

        state.request_failures.append(
            failure
        )

        raise RuntimeError(
            "ESPN injuries request failed: "
            f"status={exc.code}, "
            f"error={failure['error']}"
        ) from exc

    except URLError as exc:
        failure = {
            "url": url,
            "http_status": "",
            "error": str(exc),
        }

        state.request_failures.append(
            failure
        )

        raise RuntimeError(
            f"ESPN injuries request failed: {exc}"
        ) from exc

    except Exception as exc:
        failure = {
            "url": url,
            "http_status": "",
            "error": str(exc),
        }

        state.request_failures.append(
            failure
        )

        raise RuntimeError(
            f"ESPN injuries request failed: {exc}"
        ) from exc

    if (
        status < 200
        or status >= 300
    ):
        failure = {
            "url": url,
            "http_status": str(
                status
            ),
            "error": body,
        }

        state.request_failures.append(
            failure
        )

        raise RuntimeError(
            "ESPN injuries request returned "
            f"unexpected status: {status}"
        )

    try:
        payload = json.loads(
            body
        )

    except Exception as exc:
        failure = {
            "url": url,
            "http_status": str(
                status
            ),
            "error": (
                f"JSON parse failed: {exc}"
            ),
        }

        state.request_failures.append(
            failure
        )

        raise RuntimeError(
            "ESPN injuries request returned malformed JSON"
        ) from exc

    if not isinstance(
        payload,
        dict,
    ):
        failure = {
            "url": url,
            "http_status": str(
                status
            ),
            "error": (
                "JSON root is not an object"
            ),
        }

        state.request_failures.append(
            failure
        )

        raise RuntimeError(
            "ESPN injuries request returned non-object JSON"
        )

    return payload


def parse_timestamp(
    value: object,
    *,
    label: str,
) -> tuple[str, datetime]:
    text = str(
        value or ""
    ).strip()

    if not text:
        raise InjuryValidationError(
            f"{label} is blank"
        )

    normalized = text

    if normalized.endswith("Z"):
        normalized = (
            normalized[:-1]
            + "+00:00"
        )

    try:
        parsed = datetime.fromisoformat(
            normalized
        )

    except ValueError as exc:
        raise InjuryValidationError(
            f"{label} is not a valid ISO timestamp: {text!r}"
        ) from exc

    if parsed.tzinfo is None:
        raise InjuryValidationError(
            f"{label} lacks timezone information: {text!r}"
        )

    return (
        text,
        parsed.astimezone(
            timezone.utc
        ),
    )


def report_age_days(
    report_date_utc: datetime,
    *,
    state: RuntimeState,
) -> float:
    if state.provider_timestamp_utc is None:
        raise InjuryValidationError(
            "Provider timestamp is unavailable for "
            "injury freshness validation"
        )

    age_seconds = (
        state.provider_timestamp_utc
        - report_date_utc
    ).total_seconds()

    # Small provider clock skew must not make a report stale.
    if age_seconds < 0:
        return 0.0

    return (
        age_seconds
        / 86400.0
    )


def validate_provider_envelope(
    data: dict,
    *,
    season: int,
    season_type: int,
    state: RuntimeState,
) -> list[dict]:

    status = str(
        data.get("status") or ""
    ).strip()

    state.provider_status = status

    if (
        status
        and status.casefold() != "success"
    ):
        raise InjuryValidationError(
            "ESPN injuries payload status "
            f"is not success: {status!r}"
        )

    provider_timestamp = str(
        data.get("timestamp") or ""
    ).strip()

    if not provider_timestamp:
        raise InjuryValidationError(
            "ESPN injuries payload timestamp is required "
            "for freshness validation"
        )

    (
        _,
        provider_timestamp_utc,
    ) = parse_timestamp(
        provider_timestamp,
        label=(
            "ESPN injuries payload timestamp"
        ),
    )

    state.provider_timestamp = (
        provider_timestamp
    )
    state.provider_timestamp_utc = (
        provider_timestamp_utc
    )

    season_obj = data.get(
        "season"
    )

    if not isinstance(
        season_obj,
        dict,
    ):
        raise InjuryValidationError(
            "ESPN injuries payload season "
            "field is not an object"
        )

    provider_season = (
        parse_positive_int(
            season_obj.get("year"),
            label=(
                "ESPN injuries payload season.year"
            ),
        )
    )

    state.provider_season = provider_season

    if provider_season != season:
        raise InjuryValidationError(
            "ESPN injuries payload season mismatch: "
            f"expected={season}, "
            f"actual={provider_season}"
        )

    raw_type = season_obj.get(
        "type"
    )

    if raw_type not in (
        None,
        "",
    ):
        provider_type = (
            parse_positive_int(
                raw_type,
                label=(
                    "ESPN injuries payload season.type"
                ),
            )
        )

        state.provider_season_type = (
            provider_type
        )

        if provider_type != season_type:
            raise InjuryValidationError(
                "ESPN injuries payload season_type mismatch: "
                f"expected={season_type}, "
                f"actual={provider_type}"
            )

    injuries = data.get(
        "injuries"
    )

    if not isinstance(
        injuries,
        list,
    ):
        raise InjuryValidationError(
            "ESPN injuries payload injuries "
            "field is not a list"
        )

    return injuries


def extract_playercard_ids(
    athlete: dict,
    *,
    context: str,
) -> set[str]:
    links = athlete.get(
        "links"
    )

    if links is None:
        return set()

    if not isinstance(
        links,
        list,
    ):
        raise InjuryValidationError(
            "Athlete links field is not a list "
            f"for {context}"
        )

    playercard_ids: set[str] = set()

    for link_index, link in enumerate(
        links
    ):
        if not isinstance(
            link,
            dict,
        ):
            raise InjuryValidationError(
                "Athlete links contains non-object "
                f"entry for {context}, "
                f"link_index={link_index}"
            )

        rel = link.get(
            "rel"
        )

        if rel is None:
            continue

        if not isinstance(
            rel,
            list,
        ):
            raise InjuryValidationError(
                "Athlete link rel is not a list "
                f"for {context}, "
                f"link_index={link_index}"
            )

        if "playercard" not in rel:
            continue

        href = str(
            link.get("href") or ""
        ).strip()

        parsed = urllib.parse.urlparse(
            href
        )

        if parsed.scheme not in {
            "http",
            "https",
        }:
            raise InjuryValidationError(
                "Playercard link has unsupported scheme "
                f"for {context}: {href!r}"
            )

        if parsed.hostname not in (
            ESPN_WEB_HOSTS
        ):
            raise InjuryValidationError(
                "Playercard link has unexpected host "
                f"for {context}: {href!r}"
            )

        match = PLAYER_ID_PATTERN.search(
            parsed.path
        )

        if not match:
            raise InjuryValidationError(
                "Playercard link does not contain athlete id "
                f"for {context}: {href!r}"
            )

        playercard_ids.add(
            parse_positive_int_text(
                match.group(1),
                label=(
                    f"playercard athlete id for {context}"
                ),
            )
        )

    return playercard_ids


def extract_player_id(
    athlete: dict,
    *,
    context: str,
    state: RuntimeState,
) -> str:

    direct_id = str(
        athlete.get("id") or ""
    ).strip()

    parsed_direct = ""

    if direct_id:
        parsed_direct = (
            parse_positive_int_text(
                direct_id,
                label=f"athlete.id for {context}",
            )
        )

    playercard_ids = (
        extract_playercard_ids(
            athlete,
            context=context,
        )
    )

    if len(playercard_ids) > 1:
        raise InjuryValidationError(
            "Athlete has conflicting playercard IDs for "
            f"{context}: {sorted(playercard_ids)}"
        )

    playercard_id = (
        next(iter(playercard_ids))
        if playercard_ids
        else ""
    )

    if (
        parsed_direct
        and playercard_id
        and parsed_direct != playercard_id
    ):
        raise InjuryValidationError(
            "Athlete direct ID and playercard ID disagree for "
            f"{context}: direct={parsed_direct}, "
            f"playercard={playercard_id}"
        )

    player_id = (
        parsed_direct
        or playercard_id
    )

    if not player_id:
        state.missing_player_id_count += 1

        raise InjuryValidationError(
            f"Athlete has no usable player ID for {context}"
        )

    return player_id


def validate_athlete_team(
    athlete: dict,
    *,
    team_id: str,
    context: str,
) -> None:
    if "team" not in athlete:
        return

    team_obj = athlete.get(
        "team"
    )

    if team_obj is None:
        return

    if not isinstance(
        team_obj,
        dict,
    ):
        raise InjuryValidationError(
            f"Athlete team field is not an object for {context}"
        )

    raw_athlete_team_id = str(
        team_obj.get("id") or ""
    ).strip()

    if not raw_athlete_team_id:
        return

    athlete_team_id = (
        parse_positive_int_text(
            raw_athlete_team_id,
            label=(
                f"athlete team.id for {context}"
            ),
        )
    )

    if athlete_team_id != team_id:
        raise InjuryValidationError(
            "Athlete team identity mismatch for "
            f"{context}: "
            f"group_team_id={team_id}, "
            f"athlete_team_id={athlete_team_id}"
        )


def position_abbreviation(
    athlete: dict,
    *,
    context: str,
) -> str:
    if "position" not in athlete:
        return ""

    position = athlete.get(
        "position"
    )

    if position is None:
        return ""

    if not isinstance(
        position,
        dict,
    ):
        raise InjuryValidationError(
            "Athlete position field is not an object "
            f"for {context}"
        )

    return str(
        position.get("abbreviation") or ""
    ).strip()



def _require_injury_team_entry(
    team_entry: object,
    *,
    team_index: int,
) -> None:
    if not isinstance(team_entry, dict):
        raise InjuryValidationError(
            "ESPN injuries collection contains "
            "non-object team entry at "
            f"team_index={team_index}"
        )


def _require_authoritative_injury_team(
    team_id: str,
    authoritative_set: set[str],
    *,
    state: RuntimeState,
) -> None:
    if team_id not in authoritative_set:
        state.foreign_team_ids.add(
            team_id
        )

        raise InjuryValidationError(
            "ESPN injuries payload contains "
            f"foreign team_id={team_id}"
        )


def _require_injury_list(
    injuries: object,
    *,
    team_id: str,
) -> None:
    if not isinstance(injuries, list):
        raise InjuryValidationError(
            "ESPN injuries team entry injuries "
            "field is not a list for "
            f"team_id={team_id}"
        )


def _record_injury_group_presence(
    injuries: list,
    *,
    state: RuntimeState,
) -> None:
    if injuries:
        state.provider_team_groups_with_injuries += 1
    else:
        state.provider_team_groups_without_injuries += 1


def _require_injury_object(
    injury: object,
    *,
    team_id: str,
    injury_index: int,
) -> None:
    if not isinstance(injury, dict):
        raise InjuryValidationError(
            "ESPN injuries team group contains "
            "non-object injury for "
            f"team_id={team_id}, "
            f"injury_index={injury_index}"
        )


def _normalize_injury_id(
    injury_id: str,
    *,
    context: str,
) -> str:
    if not injury_id:
        return injury_id

    return parse_positive_int_text(
        injury_id,
        label=f"injury.id for {context}",
    )


def _require_injury_athlete(
    athlete: object,
    *,
    context: str,
) -> None:
    if not isinstance(athlete, dict):
        raise InjuryValidationError(
            "Injury athlete field is not "
            f"an object for {context}"
        )


def build_rows(
    team_entries: list[dict],
    *,
    season: int,
    authoritative_team_ids: list[str],
    canonical_by_id: dict[str, str],
    state: RuntimeState,
) -> list[dict[str, str]]:

    authoritative_set = set(
        authoritative_team_ids
    )

    rows: list[dict[str, str]] = []

    injury_id_records: dict[
        str,
        dict[str, str],
    ] = {}

    output_identity_rows: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ] = {}

    state.provider_team_group_count = len(
        team_entries
    )

    for team_index, team_entry in enumerate(
        team_entries
    ):
        _require_injury_team_entry(
            team_entry,
            team_index=team_index,
        )

        team_id = parse_positive_int_text(
            team_entry.get("id"),
            label=(
                "ESPN injuries team.id at "
                f"team_index={team_index}"
            ),
        )

        state.provider_team_ids.add(
            team_id
        )

        _require_authoritative_injury_team(
            team_id,
            authoritative_set,
            state=state,
        )

        injuries = team_entry.get(
            "injuries"
        )

        _require_injury_list(
            injuries,
            team_id=team_id,
        )

        state.raw_injury_count += len(
            injuries
        )

        _record_injury_group_presence(
            injuries,
            state=state,
        )

        canonical_team = (
            canonical_by_id[
                team_id
            ]
        )

        for injury_index, injury in enumerate(
            injuries
        ):
            _require_injury_object(
                injury,
                team_id=team_id,
                injury_index=injury_index,
            )

            context = (
                f"team_id={team_id}, "
                f"injury_index={injury_index}"
            )

            injury_id = str(
                injury.get("id") or ""
            ).strip()

            injury_id = _normalize_injury_id(
                injury_id,
                context=context,
            )

            athlete = injury.get(
                "athlete"
            )

            _require_injury_athlete(
                athlete,
                context=context,
            )

            validate_athlete_team(
                athlete,
                team_id=team_id,
                context=context,
            )

            player_id = extract_player_id(
                athlete,
                context=context,
                state=state,
            )

            player_name = str(
                athlete.get("displayName") or ""
            ).strip()

            if not player_name:
                raise InjuryValidationError(
                    "Athlete displayName is blank "
                    f"for {context}"
                )

            position = position_abbreviation(
                athlete,
                context=context,
            )

            game_status = str(
                injury.get("status") or ""
            ).strip()

            if not game_status:
                raise InjuryValidationError(
                    "Injury status is blank "
                    f"for {context}"
                )

            (
                report_date,
                report_date_utc,
            ) = parse_timestamp(
                injury.get("date"),
                label=(
                    f"injury date for {context}"
                ),
            )

            state.report_dates_utc.append(
                report_date_utc
            )

            if (
                report_date_utc.year
                != season
            ):
                state.report_year_mismatch_count += 1

            age_days = report_age_days(
                report_date_utc,
                state=state,
            )

            if age_days > MAX_REPORT_AGE_DAYS:
                state.stale_injury_count += 1
                continue

            state.fresh_injury_count += 1

            provider_identity = (
                team_id,
                player_id,
                game_status,
                report_date,
            )

            row = {
                "season": str(season),
                "team": canonical_team,
                "player_id": player_id,
                "player_name": player_name,
                "position": position,
                "game_status": game_status,
                "report_date": report_date,
            }

            if injury_id:
                prior_provider_row = (
                    injury_id_records.get(
                        injury_id
                    )
                )

                if (
                    prior_provider_row
                    is not None
                ):
                    state.duplicate_injury_id_count += 1

                    if (
                        prior_provider_row
                        != row
                    ):
                        raise InjuryValidationError(
                            "Conflicting duplicate provider "
                            "injury ID: "
                            f"injury_id={injury_id}, "
                            f"first={prior_provider_row}, "
                            f"second={row}"
                        )

                    continue

                injury_id_records[
                    injury_id
                ] = row.copy()

            output_identity = (
                provider_identity
            )

            prior_row = (
                output_identity_rows.get(
                    output_identity
                )
            )

            if prior_row is not None:
                state.duplicate_output_identity_count += 1

                if prior_row != row:
                    raise InjuryValidationError(
                        "Conflicting duplicate injury "
                        "output identity: "
                        f"identity={output_identity}, "
                        f"first={prior_row}, "
                        f"second={row}"
                    )

                continue

            output_identity_rows[
                output_identity
            ] = row

            rows.append(
                row
            )

    rows.sort(
        key=lambda row: (
            row["team"].casefold(),
            row["player_name"].casefold(),
            int(row["player_id"]),
            row["report_date"],
            row["game_status"].casefold(),
        )
    )

    return rows


def validate_output_rows(
    rows: list[dict[str, str]],
    *,
    season: int,
    canonical_by_id: dict[str, str],
    state: RuntimeState,
) -> None:
    canonical_to_id = {
        canonical: team_id
        for team_id, canonical
        in canonical_by_id.items()
    }

    seen_identity: set[
        tuple[str, str, str, str]
    ] = set()

    for row_index, row in enumerate(
        rows
    ):
        if list(row.keys()) != OUTPUT_HEADERS:
            raise InjuryValidationError(
                "Injury output row schema mismatch at "
                f"row_index={row_index}"
            )

        row_season = parse_positive_int(
            row.get("season"),
            label=(
                "injury output season at "
                f"row_index={row_index}"
            ),
        )

        if row_season != season:
            raise InjuryValidationError(
                "Injury output season mismatch at "
                f"row_index={row_index}: "
                f"expected={season}, "
                f"actual={row_season}"
            )

        team = str(
            row.get("team") or ""
        ).strip()

        if team not in canonical_to_id:
            raise InjuryValidationError(
                "Injury output contains "
                "non-authoritative team at "
                f"row_index={row_index}: "
                f"{team!r}"
            )

        team_id = canonical_to_id[
            team
        ]

        player_id = (
            parse_positive_int_text(
                row.get("player_id"),
                label=(
                    "injury output player_id at "
                    f"row_index={row_index}"
                ),
            )
        )

        player_name = str(
            row.get("player_name") or ""
        ).strip()

        game_status = str(
            row.get("game_status") or ""
        ).strip()

        if not player_name:
            raise InjuryValidationError(
                "Injury output contains blank "
                "player_name at "
                f"row_index={row_index}"
            )

        if not game_status:
            raise InjuryValidationError(
                "Injury output contains blank "
                "game_status at "
                f"row_index={row_index}"
            )

        (
            report_date,
            report_date_utc,
        ) = parse_timestamp(
            row.get("report_date"),
            label=(
                "injury output report_date at "
                f"row_index={row_index}"
            ),
        )

        age_days = report_age_days(
            report_date_utc,
            state=state,
        )

        if age_days > MAX_REPORT_AGE_DAYS:
            raise InjuryValidationError(
                "Injury output contains stale report at "
                f"row_index={row_index}: "
                f"report_date={report_date!r}, "
                f"age_days={age_days:.3f}, "
                f"max_age_days={MAX_REPORT_AGE_DAYS:.3f}"
            )

        identity = (
            team_id,
            player_id,
            game_status,
            report_date,
        )

        if identity in seen_identity:
            raise InjuryValidationError(
                "Injury output contains duplicate "
                f"identity: {identity}"
            )

        seen_identity.add(
            identity
        )


def temporary_path(
    final_path: Path,
) -> Path:
    return final_path.with_name(
        f".{final_path.name}."
        f"{uuid.uuid4().hex}.tmp"
    )


def write_staged_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_HEADERS,
        )

        writer.writeheader()
        writer.writerows(rows)

        handle.flush()
        os.fsync(
            handle.fileno()
        )


def validate_staged_csv(
    path: Path,
    *,
    expected_rows: list[dict[str, str]],
    season: int,
    canonical_by_id: dict[str, str],
    state: RuntimeState,
) -> None:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        fieldnames = (
            reader.fieldnames
            or []
        )

        if fieldnames != OUTPUT_HEADERS:
            raise InjuryValidationError(
                "Serialized injuries CSV "
                "header mismatch"
            )

        rows = list(
            reader
        )

    if rows != expected_rows:
        raise InjuryValidationError(
            "Serialized injuries CSV does not "
            "exactly match validated rows"
        )

    validate_output_rows(
        rows,
        season=season,
        canonical_by_id=canonical_by_id,
        state=state,
    )


def publish_atomic(
    output_path: Path,
    *,
    rows: list[dict[str, str]],
    season: int,
    canonical_by_id: dict[str, str],
    state: RuntimeState,
) -> bool:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = temporary_path(
        output_path
    )

    try:
        write_staged_csv(
            temp_path,
            rows,
        )

        validate_staged_csv(
            temp_path,
            expected_rows=rows,
            season=season,
            canonical_by_id=canonical_by_id,
            state=state,
        )

        if (
            output_path.exists()
            and output_path.read_bytes()
            == temp_path.read_bytes()
        ):
            return False

        os.replace(
            temp_path,
            output_path,
        )

        return True

    finally:
        try:
            temp_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass


def report_date_range(
    state: RuntimeState,
) -> tuple[str, str]:
    if not state.report_dates_utc:
        return "", ""

    minimum = min(
        state.report_dates_utc
    )

    maximum = max(
        state.report_dates_utc
    )

    return (
        minimum.isoformat(),
        maximum.isoformat(),
    )



def _injury_status_counts(
    rows: list[dict[str, str]],
) -> Counter:
    return Counter(
        str(
            row.get("game_status") or ""
        ).strip()
        for row in rows
        if str(
            row.get("game_status") or ""
        ).strip()
    )


def _injury_freshness_outcome(
    state: RuntimeState,
) -> str:
    if (
        state.raw_injury_count > 0
        and state.fresh_injury_count == 0
    ):
        return "all_stale_zero_current"

    if (
        state.fresh_injury_count > 0
        and state.stale_injury_count > 0
    ):
        return "mixed_fresh_and_stale"

    if state.fresh_injury_count > 0:
        return "fresh_only"

    return "provider_zero_injuries"


def _add_injury_output_details(
    details: dict[str, object],
    *,
    output_path: Path | None,
    output_modified: bool | None,
) -> None:
    if output_path is not None:
        details[
            "output_path"
        ] = str(
            output_path
        )

    if output_modified is not None:
        details[
            "output_modified"
        ] = output_modified


def update_report_details(
    report: PipelineReporter,
    state: RuntimeState,
    *,
    authoritative_team_ids: list[str],
    canonical_by_id: dict[str, str],
    rows: list[dict[str, str]],
    output_path: Path | None,
    output_modified: bool | None,
) -> None:
    status_counts = _injury_status_counts(rows)

    represented_names = {
        str(
            row.get("team") or ""
        ).strip()
        for row in rows
        if str(
            row.get("team") or ""
        ).strip()
    }

    canonical_to_id = {
        canonical: team_id
        for team_id, canonical
        in canonical_by_id.items()
    }

    represented_ids = sorted(
        {
            canonical_to_id[name]
            for name in represented_names
            if name in canonical_to_id
        },
        key=int,
    )

    unique_players = {
        str(
            row.get("player_id") or ""
        ).strip()
        for row in rows
        if str(
            row.get("player_id") or ""
        ).strip()
    }

    (
        report_date_min,
        report_date_max,
    ) = report_date_range(
        state
    )

    details: dict[str, object] = {
        "provider_status": state.provider_status,
        "provider_timestamp": state.provider_timestamp,
        "freshness_reference": "espn_payload_timestamp",
        "freshness_reference_utc": (
            state.provider_timestamp_utc.isoformat()
            if state.provider_timestamp_utc is not None
            else ""
        ),
        "max_report_age_days": MAX_REPORT_AGE_DAYS,
        "provider_season": state.provider_season,
        "provider_season_type": state.provider_season_type,
        "espn_request_count": state.request_count,
        "espn_request_failure_count": len(
            state.request_failures
        ),
        "espn_request_failure_details": (
            state.request_failures
        ),
        "authoritative_team_count": len(
            authoritative_team_ids
        ),
        "canonical_team_mapping_count": len(
            canonical_by_id
        ),
        "provider_team_group_count": (
            state.provider_team_group_count
        ),
        "provider_unique_team_count": len(
            state.provider_team_ids
        ),
        "provider_team_groups_with_injuries": (
            state.provider_team_groups_with_injuries
        ),
        "provider_team_groups_without_injuries": (
            state.provider_team_groups_without_injuries
        ),
        "represented_authoritative_team_count": len(
            represented_ids
        ),
        "represented_authoritative_team_ids": (
            represented_ids
        ),
        "foreign_team_count": len(
            state.foreign_team_ids
        ),
        "foreign_team_ids": sorted(
            state.foreign_team_ids,
            key=int,
        ),
        "raw_injury_count": state.raw_injury_count,
        "fresh_injury_count": state.fresh_injury_count,
        "stale_injury_count": state.stale_injury_count,
        "stale_injury_disposition": "excluded_from_output",
        "all_provider_injuries_stale": (
            state.raw_injury_count > 0
            and state.fresh_injury_count == 0
        ),
        "freshness_outcome": (
            _injury_freshness_outcome(
                state
            )
        ),
        "published_injury_count": len(
            rows
        ),
        "unique_player_count": len(
            unique_players
        ),
        "duplicate_provider_injury_id_count": (
            state.duplicate_injury_id_count
        ),
        "duplicate_output_identity_count": (
            state.duplicate_output_identity_count
        ),
        "missing_player_id_count": (
            state.missing_player_id_count
        ),
        "status_counts": dict(
            sorted(
                status_counts.items()
            )
        ),
        "provider_report_date_min": (
            report_date_min
        ),
        "provider_report_date_max": (
            report_date_max
        ),
        "report_year_mismatch_count": (
            state.report_year_mismatch_count
        ),
        "output_columns": OUTPUT_HEADERS,
    }

    _add_injury_output_details(
        details,
        output_path=output_path,
        output_modified=output_modified,
    )

    report.update_details(
        details
    )


def run(
    report: PipelineReporter,
) -> int:
    state = RuntimeState()

    (
        season,
        season_type,
        week,
    ) = load_current_week()

    report.season = season
    report.week = week

    report.set_detail(
        "season_type",
        season_type,
    )

    output_path = (
        output_path_for_season(
            season
        )
    )

    report.add_output(
        output_path
    )

    authoritative_team_ids: list[
        str
    ] = []

    canonical_by_id: dict[
        str,
        str,
    ] = {}

    rows: list[
        dict[str, str]
    ] = []

    output_modified: bool | None = None

    try:
        authoritative_team_ids = (
            load_authoritative_team_ids(
                season=season,
                season_type=season_type,
            )
        )

        canonical_by_id = (
            load_canonical_team_names(
                authoritative_team_ids
            )
        )

        url = injuries_url(
            season=season,
            season_type=season_type,
        )

        data = fetch_json(
            url,
            state=state,
        )

        team_entries = (
            validate_provider_envelope(
                data,
                season=season,
                season_type=season_type,
                state=state,
            )
        )

        rows = build_rows(
            team_entries,
            season=season,
            authoritative_team_ids=(
                authoritative_team_ids
            ),
            canonical_by_id=canonical_by_id,
            state=state,
        )

        if (
            state.raw_injury_count > 0
            and state.fresh_injury_count == 0
        ):
            report.warning(
                "ESPN returned only stale injury records; "
                "all stale records were excluded and zero "
                "current injuries will be published: "
                f"raw={state.raw_injury_count}, "
                f"stale={state.stale_injury_count}, "
                f"fresh={state.fresh_injury_count}, "
                f"max_age_days={MAX_REPORT_AGE_DAYS:.1f}"
            )

        elif state.stale_injury_count:
            report.warning(
                "Excluded stale ESPN injury records: "
                f"stale={state.stale_injury_count}, "
                f"fresh={state.fresh_injury_count}, "
                f"max_age_days={MAX_REPORT_AGE_DAYS:.1f}"
            )

        validate_output_rows(
            rows,
            season=season,
            canonical_by_id=canonical_by_id,
            state=state,
        )

        output_modified = publish_atomic(
            output_path,
            rows=rows,
            season=season,
            canonical_by_id=canonical_by_id,
            state=state,
        )

        report.set_rows(
            rows_in=state.raw_injury_count,
            rows_out=len(rows),
        )

        update_report_details(
            report,
            state,
            authoritative_team_ids=(
                authoritative_team_ids
            ),
            canonical_by_id=canonical_by_id,
            rows=rows,
            output_path=output_path,
            output_modified=output_modified,
        )

        return 0

    except Exception:
        report.set_rows(
            rows_in=state.raw_injury_count,
            rows_out=len(rows),
        )

        update_report_details(
            report,
            state,
            authoritative_team_ids=(
                authoritative_team_ids
            ),
            canonical_by_id=canonical_by_id,
            rows=rows,
            output_path=output_path,
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
            "source": "ESPN Site API",
        },
    ) as report:
        report.set_detail(
            "output_modified",
            False,
        )

        report.add_input(
            CURRENT_WEEK_CONFIG_PATH
        )
        report.add_input(
            LEAGUE_MASTER_PATH
        )
        report.add_input(
            TEAM_MAP_PATH
        )

        return run(
            report
        )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )

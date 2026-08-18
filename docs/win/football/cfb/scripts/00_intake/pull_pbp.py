#!/usr/bin/env python3
# docs/win/football/cfb/scripts/00_intake/pull_pbp.py
#
# Pulls ESPN Core-v2 college-football play-by-play for games in the local
# season schedule and normalizes it toward the nflverse PBP contract used by
# the downstream team-stat pipeline.
#
# Primary plays source:
#   https://sports.core.api.espn.com/v2/sports/football/leagues/
#   college-football/events/{game_id}/competitions/{game_id}/plays
#
# Win-probability source:
#   https://sports.core.api.espn.com/v2/sports/football/leagues/
#   college-football/events/{game_id}/competitions/{game_id}/probabilities
#
# Schedule input:
#   docs/win/football/cfb/00_intake/schedule/{season}_schedule.csv
#
# Team-id mapping:
#   docs/win/football/cfb/data/master/team_master.csv
#
# Roster lookup (optional; used to hydrate participant names):
#   docs/win/football/cfb/data/master/roster_master.csv
#
# Output:
#   docs/win/football/cfb/00_intake/pbp/{season}_pbp.csv.gz
#
# Important compatibility note:
# ESPN does not publish nflverse/cfbfastR EPA, completion probability, CPOE,
# or QB EPA as native Core-v2 play fields.  This script therefore:
#   * derives an explicit state-transition EPA estimate for scrimmage plays,
#   * defines success as EPA > 0,
#   * uses ESPN's own probability feed for WP/WPA where available,
#   * sets QB EPA equal to play EPA on pass/sack plays,
#   * leaves CP/CPOE blank rather than fabricating them.
#
# The EPA estimator is deliberately isolated in estimate_expected_points()
# and calculate_epa() so a trained CFB EP model can replace it later without
# changing the normalized PBP schema or downstream code.

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# ─────────────────────────────────────────────
# PATHS / SOURCE
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parents[2]

SETTINGS_FILE = BASE_DIR / "config" / "settings.yaml"
SCHEDULE_DIR = BASE_DIR / "00_intake" / "schedule"
TEAM_MASTER_FILE = BASE_DIR / "data" / "master" / "team_master.csv"
ROSTER_MASTER_FILE = BASE_DIR / "data" / "master" / "roster_master.csv"

PBP_DIR = BASE_DIR / "00_intake" / "pbp"
ERROR_DIR = BASE_DIR / "errors" / "00_intake"
LOG_FILE = ERROR_DIR / "pull_pbp.txt"

PLAYS_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/"
    "college-football/events/{game_id}/competitions/{game_id}/plays"
)

PROBABILITIES_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/"
    "college-football/events/{game_id}/competitions/{game_id}/probabilities"
)

DEFAULT_WORKERS = 6
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3


# ─────────────────────────────────────────────
# NFLVERSE-COMPATIBILITY CONTRACT
# ─────────────────────────────────────────────

FEATURE_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "old_game_id",
    "play_id",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "side_of_field",
    "yardline_100",
    "game_date",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "qtr",
    "down",
    "ydstogo",
    "ydsnet",
    "desc",
    "play_type",
    "yards_gained",
    "epa",
    "success",
    "wp",
    "wpa",
    "cp",
    "cpoe",
    "qb_epa",
    "pass",
    "rush",
    "sack",
    "interception",
    "fumble",
    "fumble_lost",
    "turnover",
    "touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "first_down",
    "third_down_converted",
    "third_down_failed",
    "fourth_down_converted",
    "fourth_down_failed",
    "series",
    "series_success",
    "drive",
    "fixed_drive",
    "fixed_drive_result",
    "drive_real_start_time",
    "drive_play_count",
    "drive_time_of_possession",
    "drive_first_downs",
    "drive_inside20",
    "drive_ended_with_score",
    "drive_quarter_start",
    "drive_quarter_end",
    "drive_yards_penalized",
    "posteam_score",
    "defteam_score",
    "score_differential",
    "total_home_score",
    "total_away_score",
    "passer_player_id",
    "passer_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "receiver_player_id",
    "receiver_player_name",
]

# pull_team_stats.py consumes these nflverse columns even though the current
# NFL pull_pbp FEATURE_COLUMNS list does not explicitly include all of them.
DOWNSTREAM_REQUIRED_EXTRA_COLUMNS = [
    "posteam_score_post",
    "defteam_score_post",
    "score_differential_post",
    "td_team",
]

SOURCE_AUDIT_COLUMNS = [
    "espn_sequence_number",
    "espn_type_id",
    "espn_type_text",
    "espn_start_team_id",
    "espn_end_team_id",
    "espn_scoring_team_id",
    "espn_scoring_play",
    "espn_score_value",
    "espn_is_penalty",
    "espn_is_turnover",
    "espn_start_yard_line",
    "espn_end_yard_line",
    "espn_end_down",
    "espn_end_distance",
    "espn_end_yards_to_endzone",
    "espn_wallclock",
    "espn_modified",
    "espn_drive_ref",
    "epa_source",
    "wp_source",
]

OUTPUT_COLUMNS = (
    FEATURE_COLUMNS
    + DOWNSTREAM_REQUIRED_EXTRA_COLUMNS
    + SOURCE_AUDIT_COLUMNS
)


# ─────────────────────────────────────────────
# ESPN PLAY-TYPE TAXONOMY
# ─────────────────────────────────────────────

PASS_TYPE_TEXT = {
    "pass",
    "pass reception",
    "pass incompletion",
    "pass completion",
    "passing touchdown",
    "pass reception touchdown",
    "sack",
    "sack touchdown",
    "interception",
    "interception return",
    "interception return touchdown",
    "pass interception",
    "pass interception return",
    "pass interception return touchdown",
}

RUSH_TYPE_TEXT = {
    "rush",
    "rushing touchdown",
}

TOUCHDOWN_TYPE_TEXT = {
    "blocked punt touchdown",
    "blocked field goal touchdown",
    "missed field goal return touchdown",
    "fumble recovery (opponent) touchdown",
    "fumble recovery (own) touchdown",
    "fumble return touchdown",
    "interception return touchdown",
    "pass interception return touchdown",
    "punt touchdown",
    "punt return touchdown",
    "punt team fumble recovery touchdown",
    "sack touchdown",
    "uncategorized touchdown",
    "kickoff team fumble recovery touchdown",
    "kickoff return touchdown",
    "kickoff touchdown",
    "passing touchdown",
    "rushing touchdown",
    "pass reception touchdown",
}

PUNT_TYPE_TEXT = {
    "punt",
    "punt touchdown",
    "punt return touchdown",
    "punt team fumble recovery",
    "punt team fumble recovery touchdown",
    "blocked punt",
    "blocked punt touchdown",
    "blocked punt (safety)",
    "punt (safety)",
}

FIELD_GOAL_GOOD_TYPES = {
    "field goal good",
}

TURNOVER_ON_DOWNS_HINTS = {
    "turnover on downs",
}


# ─────────────────────────────────────────────
# EXPECTED-POINTS COMPATIBILITY ESTIMATOR
# ─────────────────────────────────────────────

# Approximate first-and-10 EP anchors, expressed as distance from the
# opponent goal line.  Linear interpolation plus down/distance penalties
# produces a stable state value.  This is intentionally isolated so it can
# later be replaced by a trained CFB model without touching the output schema.
FIRST_DOWN_EP_ANCHORS = [
    (1.0, 6.4),
    (10.0, 5.4),
    (20.0, 4.5),
    (30.0, 3.6),
    (40.0, 2.8),
    (50.0, 2.1),
    (60.0, 1.5),
    (70.0, 0.9),
    (80.0, 0.3),
    (90.0, -0.4),
    (99.0, -1.2),
]


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def ensure_dirs() -> None:
    PBP_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)


def reset_log() -> None:
    ensure_dirs()
    LOG_FILE.write_text("", encoding="utf-8")


def log(message: str) -> None:
    ensure_dirs()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{now_stamp()}] {message.rstrip()}\n")


# ─────────────────────────────────────────────
# GENERIC HELPERS
# ─────────────────────────────────────────────

def clean(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if text.casefold() in {"", "none", "nan", "null", "<na>"}:
        return ""

    return text


def to_int(value: Any, default: int | None = None) -> int | None:
    text = clean(value)
    if not text:
        return default

    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float | None = None) -> float | None:
    text = clean(value)
    if not text:
        return default

    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def bool_int(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0

    text = clean(value).casefold()

    if text in {"1", "true", "yes", "y"}:
        return 1

    return 0


def get_ref(value: Any) -> str:
    if isinstance(value, dict):
        return clean(value.get("$ref"))

    return ""


def id_from_ref(ref_url: Any, entity: str | None = None) -> str:
    ref = clean(ref_url)
    if not ref:
        return ""

    if entity:
        match = re.search(
            rf"/{re.escape(entity)}/([^/?#]+)",
            ref,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)

    path = urllib.parse.urlparse(ref).path.rstrip("/")
    if not path:
        return ""

    return path.split("/")[-1]


def object_id(value: Any, entity: str | None = None) -> str:
    if isinstance(value, dict):
        direct = clean(value.get("id"))
        if direct:
            return direct

        ref = clean(value.get("$ref"))
        if ref:
            return id_from_ref(ref, entity=entity)

    return clean(value) if not isinstance(value, (dict, list)) else ""


def nested_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def seconds_from_clock(clock: Any) -> int | None:
    if isinstance(clock, dict):
        seconds = to_int(clock.get("value"))
        if seconds is not None:
            return seconds

        text = clean(clock.get("displayValue"))
    else:
        text = clean(clock)

    if not text:
        return None

    match = re.match(r"^(\d+):(\d{2})$", text)
    if not match:
        return None

    return int(match.group(1)) * 60 + int(match.group(2))


def game_seconds_remaining(period: int | None, clock_seconds: int | None) -> int | None:
    if period is None or clock_seconds is None:
        return None

    if period == 1:
        return 2700 + clock_seconds
    if period == 2:
        return 1800 + clock_seconds
    if period == 3:
        return 900 + clock_seconds
    if period == 4:
        return clock_seconds

    # College overtime is untimed.
    if period > 4:
        return 0

    return None


def half_seconds_remaining(period: int | None, clock_seconds: int | None) -> int | None:
    if period is None or clock_seconds is None:
        return None

    if period in {1, 3}:
        return 900 + clock_seconds
    if period in {2, 4}:
        return clock_seconds
    if period > 4:
        return 0

    return None


def add_query_params(url: str, **params: Any) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)

    for key, value in params.items():
        query[key] = [str(value)]

    return urllib.parse.urlunparse(
        parsed._replace(query=urllib.parse.urlencode(query, doseq=True))
    )


# ─────────────────────────────────────────────
# SETTINGS / CLI
# ─────────────────────────────────────────────

def read_settings() -> dict[str, Any]:
    if not SETTINGS_FILE.exists() or yaml is None:
        return {}

    with SETTINGS_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data if isinstance(data, dict) else {}


def get_season(args: argparse.Namespace) -> int:
    if args.season is not None:
        return int(args.season)

    settings = read_settings()
    season = settings.get("season")

    if season not in (None, ""):
        return int(season)

    env_season = os.getenv("CFB_SEASON")
    if env_season:
        return int(env_season)

    raise ValueError(
        "Missing season. Provide --season, set season in "
        "docs/win/football/cfb/config/settings.yaml, or set CFB_SEASON."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull ESPN Core-v2 CFB play-by-play and normalize it toward "
            "the nflverse schema used by downstream calculations."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="CFB season to pull.",
    )

    parser.add_argument(
        "--source",
        choices=["espn"],
        default="espn",
        help="PBP source. Currently ESPN Core-v2 only.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent game pulls. Default: {DEFAULT_WORKERS}.",
    )

    parser.add_argument(
        "--game-id",
        action="append",
        default=None,
        help=(
            "Optional ESPN game id to pull. Repeat for multiple games. "
            "If omitted, all games in the local season schedule are considered."
        ),
    )

    return parser.parse_args()


# ─────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────

def fetch_json(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; cfb_for_mat CFB data pipeline)"
            ),
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.espn.com",
            "Referer": "https://www.espn.com/",
        },
        method="GET",
    )

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json_loads(response.read().decode("utf-8"))

            if not isinstance(payload, dict):
                raise RuntimeError(f"Expected JSON object from {url}")

            return payload

        except urllib.error.HTTPError as exc:
            last_error = exc

            # No need to retry permanent "not available" responses.
            if exc.code in {400, 404}:
                raise

        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc

        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 4))

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"Unknown fetch failure: {url}")


def json_loads(text: str) -> Any:
    import json
    return json.loads(text)


def fetch_collection(base_url: str) -> list[dict[str, Any]]:
    first_url = add_query_params(
        base_url,
        limit=1000,
        page=1,
        lang="en",
        region="us",
    )
    first = fetch_json(first_url)

    items = [
        item
        for item in first.get("items", [])
        if isinstance(item, dict)
    ]

    page_count = to_int(first.get("pageCount"), 1) or 1

    for page in range(2, page_count + 1):
        page_url = add_query_params(
            base_url,
            limit=1000,
            page=page,
            lang="en",
            region="us",
        )
        data = fetch_json(page_url)

        items.extend(
            item
            for item in data.get("items", [])
            if isinstance(item, dict)
        )

    return items


# ─────────────────────────────────────────────
# LOCAL INPUTS
# ─────────────────────────────────────────────

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(f"Missing header row: {path}")

        return [
            {clean(key): clean(value) for key, value in row.items()}
            for row in reader
        ]


def load_schedule(season: int) -> dict[str, dict[str, str]]:
    schedule_path = SCHEDULE_DIR / f"{season}_schedule.csv"
    rows = read_csv_rows(schedule_path)

    required = {
        "season",
        "season_type",
        "week",
        "game_id",
        "game_date",
        "away_team",
        "home_team",
    }

    available = set(rows[0].keys()) if rows else set()
    missing = sorted(required - available)

    if missing:
        raise ValueError(
            f"{schedule_path} missing required columns: {missing}"
        )

    games: dict[str, dict[str, str]] = {}

    for row in rows:
        if clean(row.get("season")) != str(season):
            continue

        game_id = clean(row.get("game_id"))
        if not game_id:
            continue

        games[game_id] = row

    if not games:
        raise RuntimeError(
            f"No season={season} games found in {schedule_path}"
        )

    return games


def schedule_game_is_future(row: dict[str, str]) -> bool:
    game_date = clean(row.get("game_date"))

    if not game_date:
        return False

    try:
        parsed = datetime.strptime(game_date, "%Y-%m-%d").date()
    except ValueError:
        return False

    return parsed > datetime.now(timezone.utc).date()


def load_team_name_to_id() -> dict[str, str]:
    if not TEAM_MASTER_FILE.exists():
        return {}

    rows = read_csv_rows(TEAM_MASTER_FILE)
    lookup: dict[str, str] = {}

    for row in rows:
        team_id = clean(row.get("team_id"))
        if not team_id:
            continue

        for field in [
            "canonical_team",
            "team",
            "shortDisplayName",
            "location",
            "team_name",
            "nickname",
            "team_slug",
            "team_abbr",
            "alias",
        ]:
            value = clean(row.get(field))
            if value:
                lookup[value.casefold()] = team_id

    return lookup


def load_team_id_to_name() -> dict[str, str]:
    if not TEAM_MASTER_FILE.exists():
        return {}

    rows = read_csv_rows(TEAM_MASTER_FILE)
    lookup: dict[str, str] = {}

    for row in rows:
        team_id = clean(row.get("team_id"))
        if not team_id:
            continue

        canonical = (
            clean(row.get("canonical_team"))
            or clean(row.get("team"))
            or clean(row.get("shortDisplayName"))
            or clean(row.get("location"))
        )

        if canonical and team_id not in lookup:
            lookup[team_id] = canonical

    return lookup


def load_roster_lookup() -> dict[str, str]:
    if not ROSTER_MASTER_FILE.exists():
        return {}

    try:
        rows = read_csv_rows(ROSTER_MASTER_FILE)
    except Exception:
        return {}

    lookup: dict[str, str] = {}

    for row in rows:
        player_id = clean(row.get("id"))
        name = (
            clean(row.get("displayName"))
            or clean(row.get("fullName"))
            or clean(row.get("shortName"))
        )

        if player_id and name and player_id not in lookup:
            lookup[player_id] = name

    return lookup


# ─────────────────────────────────────────────
# TEAM / PARTICIPANT EXTRACTION
# ─────────────────────────────────────────────

def participant_type_text(value: Any) -> str:
    if isinstance(value, dict):
        direct = (
            clean(value.get("text"))
            or clean(value.get("name"))
            or clean(value.get("type"))
            or clean(value.get("id"))
        )

        if direct:
            return direct

        ref = clean(value.get("$ref"))
        if ref:
            return id_from_ref(ref)

        return ""

    return clean(value)


def extract_team_participants(play: dict[str, Any]) -> dict[str, str]:
    result = {
        "offense": "",
        "defense": "",
    }

    raw = play.get("teamParticipants")

    if not isinstance(raw, list):
        return result

    for item in raw:
        if not isinstance(item, dict):
            continue

        type_text = participant_type_text(item.get("type")).casefold()

        team_obj = item.get("team")
        team_id = object_id(team_obj, entity="teams")

        if not team_id:
            team_id = clean(item.get("teamId"))

        if not team_id:
            team_id = id_from_ref(item.get("$ref"), entity="teams")

        if not team_id:
            continue

        if "offense" in type_text:
            result["offense"] = team_id
        elif "defense" in type_text:
            result["defense"] = team_id

    return result


def extract_player_participants(
    play: dict[str, Any],
    roster_lookup: dict[str, str],
) -> dict[str, str]:
    out = {
        "passer_player_id": "",
        "passer_player_name": "",
        "rusher_player_id": "",
        "rusher_player_name": "",
        "receiver_player_id": "",
        "receiver_player_name": "",
    }

    participants = play.get("participants")

    if not isinstance(participants, list):
        return out

    for item in participants:
        if not isinstance(item, dict):
            continue

        type_text = participant_type_text(item.get("type")).casefold()

        athlete = item.get("athlete")
        athlete_id = object_id(athlete, entity="athletes")

        if not athlete_id:
            athlete_id = clean(item.get("athleteId"))

        if not athlete_id:
            athlete_id = id_from_ref(item.get("$ref"), entity="athletes")

        athlete_name = ""

        if isinstance(athlete, dict):
            athlete_name = (
                clean(athlete.get("displayName"))
                or clean(athlete.get("fullName"))
                or clean(athlete.get("shortName"))
            )

        if not athlete_name and athlete_id:
            athlete_name = roster_lookup.get(athlete_id, "")

        role = ""

        if "passer" in type_text:
            role = "passer"
        elif "rusher" in type_text:
            role = "rusher"
        elif "receiver" in type_text or "target" in type_text:
            role = "receiver"

        if role and not out[f"{role}_player_id"]:
            out[f"{role}_player_id"] = athlete_id
            out[f"{role}_player_name"] = athlete_name

    return out


# ─────────────────────────────────────────────
# PLAY CLASSIFICATION / STATE
# ─────────────────────────────────────────────

def normalize_play_type(espn_type_text: Any) -> str:
    value = clean(espn_type_text)
    lower = value.casefold()

    if lower in PASS_TYPE_TEXT:
        return "pass"

    if lower in RUSH_TYPE_TEXT:
        return "run"

    if "pass" in lower or "sack" in lower or "interception" in lower:
        return "pass"

    if "rush" in lower:
        return "run"

    if "punt" in lower:
        return "punt"

    if "kickoff" in lower:
        return "kickoff"

    if "field goal" in lower:
        return "field_goal"

    if "extra point" in lower or "pat" in lower:
        return "extra_point"

    if "two-point" in lower or "2pt" in lower:
        return "two_point_attempt"

    if "penalty" in lower:
        return "no_play"

    if "timeout" in lower:
        return "no_play"

    if not value:
        return ""

    return re.sub(r"[^a-z0-9]+", "_", lower).strip("_")


def interpolate_first_down_ep(yards_to_goal: float) -> float:
    ytg = max(1.0, min(99.0, float(yards_to_goal)))

    anchors = FIRST_DOWN_EP_ANCHORS

    if ytg <= anchors[0][0]:
        return anchors[0][1]

    if ytg >= anchors[-1][0]:
        return anchors[-1][1]

    for index in range(1, len(anchors)):
        x0, y0 = anchors[index - 1]
        x1, y1 = anchors[index]

        if x0 <= ytg <= x1:
            weight = (ytg - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)

    return 0.0


def estimate_expected_points(
    down: Any,
    distance: Any,
    yards_to_goal: Any,
) -> float | None:
    down_num = to_int(down)
    distance_num = to_float(distance)
    ytg = to_float(yards_to_goal)

    if down_num is None or ytg is None or down_num < 1 or down_num > 4:
        return None

    if distance_num is None or distance_num <= 0:
        distance_num = min(10.0, max(1.0, ytg))

    base = interpolate_first_down_ep(ytg)

    extra_to_go = max(distance_num - 3.0, 0.0)

    if down_num == 1:
        penalty = 0.0
    elif down_num == 2:
        penalty = 0.10 + 0.040 * extra_to_go
    elif down_num == 3:
        penalty = 0.45 + 0.080 * extra_to_go
    else:
        penalty = 1.20 + 0.120 * extra_to_go

    # Goal-to-go states should not be over-penalized by distance.
    if distance_num >= ytg:
        penalty *= 0.70

    return max(-6.5, min(6.8, base - penalty))


def is_touchdown_type(type_text: str, score_value: Any) -> bool:
    if type_text.casefold() in TOUCHDOWN_TYPE_TEXT:
        return True

    return to_int(score_value, 0) == 6


def scoring_side(
    scoring_team_id: str,
    offense_id: str,
    defense_id: str,
) -> str:
    if scoring_team_id and scoring_team_id == offense_id:
        return "offense"

    if scoring_team_id and scoring_team_id == defense_id:
        return "defense"

    return ""


def calculate_epa(row: dict[str, Any]) -> float | None:
    if row.get("play_type") not in {"pass", "run"}:
        return None

    ep_before = estimate_expected_points(
        row.get("down"),
        row.get("ydstogo"),
        row.get("yardline_100"),
    )

    if ep_before is None:
        return None

    touchdown = to_int(row.get("touchdown"), 0) or 0

    if touchdown == 1:
        td_team = clean(row.get("td_team"))
        posteam = clean(row.get("posteam"))

        if td_team and posteam:
            ep_after = 7.0 if td_team == posteam else -7.0
        else:
            # Offensive TD is the safer fallback for ordinary pass/run TDs.
            ep_after = 7.0

        return ep_after - ep_before

    type_text = clean(row.get("espn_type_text")).casefold()

    if type_text in FIELD_GOAL_GOOD_TYPES:
        return 3.0 - ep_before

    if "safety" in type_text:
        scoring_team = clean(row.get("td_team"))
        posteam = clean(row.get("posteam"))
        ep_after = 2.0 if scoring_team and scoring_team == posteam else -2.0
        return ep_after - ep_before

    end_down = row.get("espn_end_down")
    end_distance = row.get("espn_end_distance")
    end_ytg = row.get("espn_end_yards_to_endzone")

    ep_end = estimate_expected_points(
        end_down,
        end_distance,
        end_ytg,
    )

    if ep_end is None:
        return None

    offense_id = clean(row.get("espn_start_team_id"))
    end_team_id = clean(row.get("espn_end_team_id"))

    turnover = to_int(row.get("turnover"), 0) or 0

    if turnover == 1:
        ep_end *= -1.0
    elif offense_id and end_team_id and offense_id != end_team_id:
        ep_end *= -1.0

    return ep_end - ep_before


# ─────────────────────────────────────────────
# WIN PROBABILITY
# ─────────────────────────────────────────────

def build_probability_map(
    probability_items: list[dict[str, Any]],
) -> dict[str, float]:
    result: dict[str, float] = {}

    for item in probability_items:
        play_id = object_id(item.get("play"), entity="plays")

        if not play_id:
            play_id = clean(item.get("playId"))

        if not play_id:
            play_id = id_from_ref(item.get("playRef"), entity="plays")

        home_wp = to_float(
            item.get("homeWinPercentage")
            if "homeWinPercentage" in item
            else item.get("home_win_percentage")
        )

        if home_wp is not None and home_wp > 1:
            home_wp /= 100.0

        if play_id and home_wp is not None:
            result[play_id] = home_wp

    return result


# ─────────────────────────────────────────────
# GAME NORMALIZATION
# ─────────────────────────────────────────────

def team_name(
    team_id: str,
    preferred_name: str,
    team_id_to_name: dict[str, str],
) -> str:
    if preferred_name:
        return preferred_name

    if team_id:
        return team_id_to_name.get(team_id, team_id)

    return ""


def infer_game_team_ids(
    schedule_row: dict[str, str],
    plays: list[dict[str, Any]],
    name_to_id: dict[str, str],
) -> tuple[str, str]:
    home_name = clean(schedule_row.get("home_team"))
    away_name = clean(schedule_row.get("away_team"))

    home_id = name_to_id.get(home_name.casefold(), "")
    away_id = name_to_id.get(away_name.casefold(), "")

    observed: list[str] = []

    for play in plays:
        team_parts = extract_team_participants(play)

        candidates = [
            team_parts.get("offense", ""),
            team_parts.get("defense", ""),
            object_id(nested_dict(play.get("start")).get("team"), entity="teams"),
            object_id(nested_dict(play.get("end")).get("team"), entity="teams"),
            object_id(play.get("team"), entity="teams"),
        ]

        for candidate in candidates:
            candidate = clean(candidate)
            if candidate and candidate not in observed:
                observed.append(candidate)

    if not home_id and away_id:
        others = [team_id for team_id in observed if team_id != away_id]
        if others:
            home_id = others[0]

    if not away_id and home_id:
        others = [team_id for team_id in observed if team_id != home_id]
        if others:
            away_id = others[0]

    if not home_id and not away_id and len(observed) >= 2:
        # Last-resort only.  The local schedule normally guarantees at least
        # one mapped FBS team, so this should be rare.
        home_id = observed[0]
        away_id = observed[1]

    return home_id, away_id


def normalize_game(
    schedule_row: dict[str, str],
    plays: list[dict[str, Any]],
    probability_items: list[dict[str, Any]],
    name_to_id: dict[str, str],
    team_id_to_name: dict[str, str],
    roster_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    if not plays:
        return []

    game_id = clean(schedule_row.get("game_id"))
    season = clean(schedule_row.get("season"))
    season_type = clean(schedule_row.get("season_type"))
    week = clean(schedule_row.get("week"))
    game_date = clean(schedule_row.get("game_date"))
    home_name = clean(schedule_row.get("home_team"))
    away_name = clean(schedule_row.get("away_team"))

    home_id, away_id = infer_game_team_ids(
        schedule_row=schedule_row,
        plays=plays,
        name_to_id=name_to_id,
    )

    probability_map = build_probability_map(probability_items)

    rows: list[dict[str, Any]] = []

    for play_index, play in enumerate(plays, start=1):
        start = nested_dict(play.get("start"))
        end = nested_dict(play.get("end"))
        period_obj = nested_dict(play.get("period"))
        clock_obj = play.get("clock")

        play_id = clean(play.get("id")) or clean(play.get("sequenceNumber"))
        sequence_number = (
            to_int(play.get("sequenceNumber"))
            or to_int(play.get("sequence"))
            or play_index
        )

        type_obj = nested_dict(play.get("type"))
        type_id = clean(type_obj.get("id"))
        type_text = clean(type_obj.get("text")) or clean(type_obj.get("name"))
        normalized_type = normalize_play_type(type_text)

        period = (
            to_int(period_obj.get("number"))
            or to_int(period_obj.get("value"))
        )

        clock_seconds = seconds_from_clock(clock_obj)

        start_team_id = object_id(start.get("team"), entity="teams")
        end_team_id = object_id(end.get("team"), entity="teams")

        team_parts = extract_team_participants(play)
        offense_id = team_parts.get("offense") or start_team_id
        defense_id = team_parts.get("defense", "")

        if not defense_id:
            if offense_id == home_id:
                defense_id = away_id
            elif offense_id == away_id:
                defense_id = home_id

        # If start.team is absent but teamParticipants supplies offense, keep
        # the compatibility audit field aligned with possession.
        if not start_team_id:
            start_team_id = offense_id

        posteam = ""
        defteam = ""

        if offense_id == home_id:
            posteam = home_name
            defteam = away_name
        elif offense_id == away_id:
            posteam = away_name
            defteam = home_name
        else:
            posteam = team_name(
                offense_id,
                "",
                team_id_to_name,
            )
            defteam = team_name(
                defense_id,
                "",
                team_id_to_name,
            )

        scoring_team_id = object_id(play.get("team"), entity="teams")

        scoring_play = bool_int(play.get("scoringPlay"))
        score_value = to_int(play.get("scoreValue"), 0) or 0
        touchdown = 1 if is_touchdown_type(type_text, score_value) else 0

        type_lower = type_text.casefold()

        # Core-v2 normally identifies the scoring team directly.  For rare
        # plays where it does not, infer only from unambiguous TD taxonomy.
        if touchdown and not scoring_team_id:
            offensive_td = (
                type_lower in {
                    "passing touchdown",
                    "pass reception touchdown",
                    "rushing touchdown",
                    "fumble recovery (own) touchdown",
                }
            )

            defensive_td = any(
                marker in type_lower
                for marker in [
                    "interception return touchdown",
                    "fumble recovery (opponent) touchdown",
                    "fumble return touchdown",
                    "blocked punt touchdown",
                    "blocked field goal touchdown",
                    "sack touchdown",
                ]
            )

            if offensive_td:
                scoring_team_id = offense_id
            elif defensive_td:
                scoring_team_id = defense_id

        scoring_team_name = ""

        if scoring_team_id:
            if scoring_team_id == home_id:
                scoring_team_name = home_name
            elif scoring_team_id == away_id:
                scoring_team_name = away_name
            else:
                scoring_team_name = team_id_to_name.get(
                    scoring_team_id,
                    scoring_team_id,
                )

        turnover = bool_int(play.get("isTurnover"))

        if turnover == 0 and (
            "interception" in type_lower
            or "fumble recovery (opponent)" in type_lower
        ):
            turnover = 1

        description = clean(play.get("text")) or clean(play.get("shortText"))
        desc_lower = description.casefold()

        interception = int(
            "interception" in type_lower
            or (
                normalized_type == "pass"
                and "intercepted" in desc_lower
            )
        )

        fumble = int(
            "fumble" in type_lower
            or "fumble" in desc_lower
        )

        fumble_lost = int(fumble == 1 and turnover == 1)
        sack = int("sack" in type_lower)

        pass_flag = int(normalized_type == "pass")
        rush_flag = int(normalized_type == "run")

        start_down = to_int(start.get("down"))
        start_distance = to_float(start.get("distance"))
        start_yardline = to_float(start.get("yardLine"))
        start_ytg = to_float(start.get("yardsToEndzone"))

        end_down = to_int(end.get("down"))
        end_distance = to_float(end.get("distance"))
        end_yardline = to_float(end.get("yardLine"))
        end_ytg = to_float(end.get("yardsToEndzone"))

        yards_gained = to_float(play.get("statYardage"))

        same_possession_after = (
            bool(offense_id)
            and bool(end_team_id)
            and offense_id == end_team_id
            and turnover == 0
        )

        first_down = 0

        if touchdown == 1 and scoring_team_id == offense_id:
            first_down = 1
        elif (
            normalized_type in {"pass", "run"}
            and turnover == 0
            and end_down == 1
            and (
                not end_team_id
                or end_team_id == offense_id
            )
        ):
            first_down = 1
        elif (
            normalized_type in {"pass", "run"}
            and turnover == 0
            and yards_gained is not None
            and start_distance is not None
            and yards_gained >= start_distance
        ):
            first_down = 1

        third_converted = int(start_down == 3 and first_down == 1)
        third_failed = int(
            start_down == 3
            and normalized_type in {"pass", "run"}
            and third_converted == 0
        )

        fourth_converted = int(start_down == 4 and first_down == 1)
        fourth_failed = int(
            start_down == 4
            and normalized_type in {"pass", "run"}
            and fourth_converted == 0
        )

        if start_ytg is None:
            side_of_field = ""
        elif abs(start_ytg - 50.0) < 0.001:
            side_of_field = "MID"
        elif start_ytg < 50:
            side_of_field = defteam
        else:
            side_of_field = posteam

        drive_ref = get_ref(play.get("drive"))
        drive_id = object_id(play.get("drive"), entity="drives")

        if not drive_id:
            drive_id = id_from_ref(drive_ref, entity="drives")

        player_fields = extract_player_participants(
            play,
            roster_lookup=roster_lookup,
        )

        row: dict[str, Any] = {
            "season": season,
            "season_type": season_type,
            "week": week,
            "game_id": game_id,
            "old_game_id": game_id,
            "play_id": play_id,
            "home_team": home_name,
            "away_team": away_name,
            "posteam": posteam,
            "defteam": defteam,
            "side_of_field": side_of_field,
            "yardline_100": start_ytg,
            "game_date": game_date,
            "game_seconds_remaining": game_seconds_remaining(
                period,
                clock_seconds,
            ),
            "half_seconds_remaining": half_seconds_remaining(
                period,
                clock_seconds,
            ),
            "qtr": period,
            "down": start_down,
            "ydstogo": start_distance,
            "ydsnet": "",
            "desc": description,
            "play_type": normalized_type,
            "yards_gained": yards_gained,
            "epa": "",
            "success": "",
            "wp": "",
            "wpa": "",
            "cp": "",
            "cpoe": "",
            "qb_epa": "",
            "pass": pass_flag,
            "rush": rush_flag,
            "sack": sack,
            "interception": interception,
            "fumble": fumble,
            "fumble_lost": fumble_lost,
            "turnover": turnover,
            "touchdown": touchdown,
            "pass_touchdown": int(
                touchdown == 1 and normalized_type == "pass"
                and scoring_team_id == offense_id
            ),
            "rush_touchdown": int(
                touchdown == 1 and normalized_type == "run"
                and scoring_team_id == offense_id
            ),
            "first_down": first_down,
            "third_down_converted": third_converted,
            "third_down_failed": third_failed,
            "fourth_down_converted": fourth_converted,
            "fourth_down_failed": fourth_failed,
            "series": "",
            "series_success": "",
            "drive": drive_id,
            "fixed_drive": "",
            "fixed_drive_result": "",
            "drive_real_start_time": "",
            "drive_play_count": "",
            "drive_time_of_possession": "",
            "drive_first_downs": "",
            "drive_inside20": "",
            "drive_ended_with_score": "",
            "drive_quarter_start": "",
            "drive_quarter_end": "",
            "drive_yards_penalized": 0,
            "posteam_score": "",
            "defteam_score": "",
            "score_differential": "",
            "total_home_score": to_int(play.get("homeScore")),
            "total_away_score": to_int(play.get("awayScore")),
            "posteam_score_post": "",
            "defteam_score_post": "",
            "score_differential_post": "",
            "td_team": scoring_team_name if touchdown else "",
            "espn_sequence_number": sequence_number,
            "espn_type_id": type_id,
            "espn_type_text": type_text,
            "espn_start_team_id": offense_id or start_team_id,
            "espn_end_team_id": end_team_id,
            "espn_scoring_team_id": scoring_team_id,
            "espn_scoring_play": scoring_play,
            "espn_score_value": score_value,
            "espn_is_penalty": bool_int(play.get("isPenalty")),
            "espn_is_turnover": bool_int(play.get("isTurnover")),
            "espn_start_yard_line": start_yardline,
            "espn_end_yard_line": end_yardline,
            "espn_end_down": end_down,
            "espn_end_distance": end_distance,
            "espn_end_yards_to_endzone": end_ytg,
            "espn_wallclock": clean(play.get("wallclock")),
            "espn_modified": clean(play.get("modified")),
            "espn_drive_ref": drive_ref,
            "epa_source": "cfb_state_transition_estimate",
            "wp_source": "espn_core_probabilities",
        }

        row.update(player_fields)

        # Temporary raw value used only while deriving WP/WPA.
        row["_home_wp_after"] = probability_map.get(play_id)

        rows.append(row)

    rows.sort(
        key=lambda row: (
            to_int(row.get("espn_sequence_number"), 10**9) or 10**9,
            clean(row.get("play_id")),
        )
    )

    add_score_columns(
        rows=rows,
        home_id=home_id,
        away_id=away_id,
    )

    add_drive_fallbacks(rows)
    add_series_columns(rows)
    add_drive_summary_columns(rows)
    add_epa_columns(rows)
    add_wp_columns(rows, home_id=home_id, away_id=away_id)

    for row in rows:
        row.pop("_home_wp_after", None)

    return rows


def add_score_columns(
    rows: list[dict[str, Any]],
    home_id: str,
    away_id: str,
) -> None:
    previous_home_score = 0
    previous_away_score = 0

    for row in rows:
        post_home = to_int(row.get("total_home_score"))
        post_away = to_int(row.get("total_away_score"))

        if post_home is None:
            post_home = previous_home_score

        if post_away is None:
            post_away = previous_away_score

        offense_id = clean(row.get("espn_start_team_id"))

        if offense_id == home_id:
            pre_for = previous_home_score
            pre_against = previous_away_score
            post_for = post_home
            post_against = post_away

        elif offense_id == away_id:
            pre_for = previous_away_score
            pre_against = previous_home_score
            post_for = post_away
            post_against = post_home

        else:
            # Leave possession-oriented scores blank when possession is
            # unresolved rather than assigning the wrong side.
            pre_for = None
            pre_against = None
            post_for = None
            post_against = None

        row["posteam_score"] = pre_for
        row["defteam_score"] = pre_against

        if pre_for is not None and pre_against is not None:
            row["score_differential"] = pre_for - pre_against

        row["posteam_score_post"] = post_for
        row["defteam_score_post"] = post_against

        if post_for is not None and post_against is not None:
            row["score_differential_post"] = post_for - post_against

        row["total_home_score"] = post_home
        row["total_away_score"] = post_away

        previous_home_score = post_home
        previous_away_score = post_away


def add_drive_fallbacks(rows: list[dict[str, Any]]) -> None:
    drive_number_by_id: dict[str, int] = {}
    next_drive_number = 1

    synthetic_drive_id = ""
    previous_posteam = ""

    for row in rows:
        raw_drive = clean(row.get("drive"))
        posteam = clean(row.get("posteam"))
        play_type = clean(row.get("play_type"))

        if raw_drive:
            if raw_drive not in drive_number_by_id:
                drive_number_by_id[raw_drive] = next_drive_number
                next_drive_number += 1

            row["fixed_drive"] = drive_number_by_id[raw_drive]
            synthetic_drive_id = ""

        elif posteam and play_type in {"pass", "run"}:
            if not synthetic_drive_id or posteam != previous_posteam:
                synthetic_drive_id = (
                    f"{clean(row.get('game_id'))}_synthetic_{next_drive_number}"
                )
                drive_number_by_id[synthetic_drive_id] = next_drive_number
                next_drive_number += 1

            row["drive"] = synthetic_drive_id
            row["fixed_drive"] = drive_number_by_id[synthetic_drive_id]

        else:
            row["fixed_drive"] = ""

        if posteam:
            previous_posteam = posteam


def add_series_columns(rows: list[dict[str, Any]]) -> None:
    series_number = 0
    previous_drive = None

    for row in rows:
        drive = clean(row.get("drive"))
        down = to_int(row.get("down"))

        if drive != previous_drive:
            series_number += 1
        elif down == 1:
            series_number += 1

        row["series"] = series_number
        previous_drive = drive

    success_by_series: dict[int, int] = {}

    for row in rows:
        series = to_int(row.get("series"))
        if series is None:
            continue

        success_by_series[series] = max(
            success_by_series.get(series, 0),
            to_int(row.get("first_down"), 0) or 0,
            to_int(row.get("touchdown"), 0) or 0,
        )

    for row in rows:
        series = to_int(row.get("series"))
        row["series_success"] = (
            success_by_series.get(series, 0)
            if series is not None
            else ""
        )


def format_possession_time(seconds: int | None) -> str:
    if seconds is None:
        return ""

    seconds = max(0, int(seconds))
    minutes, remaining = divmod(seconds, 60)
    return f"{minutes}:{remaining:02d}"


def drive_result(group: list[dict[str, Any]]) -> str:
    if not group:
        return ""

    for row in reversed(group):
        type_text = clean(row.get("espn_type_text")).casefold()
        desc = clean(row.get("desc")).casefold()

        if to_int(row.get("touchdown"), 0) == 1:
            return "Touchdown"

        if type_text in FIELD_GOAL_GOOD_TYPES:
            return "Field goal"

        if "punt" in type_text:
            return "Punt"

        if to_int(row.get("turnover"), 0) == 1:
            return "Turnover"

        if any(hint in desc for hint in TURNOVER_ON_DOWNS_HINTS):
            return "Turnover on downs"

        if "end of" in type_text or "end of" in desc:
            return "End of half"

    return "Other"


def add_drive_summary_columns(rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        drive_key = clean(row.get("drive"))
        if not drive_key:
            continue

        groups.setdefault(drive_key, []).append(row)

    for group in groups.values():
        scrimmage = [
            row for row in group
            if row.get("play_type") in {"pass", "run"}
        ]

        ydsnet = sum(
            to_float(row.get("yards_gained"), 0.0) or 0.0
            for row in scrimmage
        )

        first_downs = sum(
            to_int(row.get("first_down"), 0) or 0
            for row in group
        )

        inside20 = int(
            any(
                (
                    to_float(row.get("yardline_100")) is not None
                    and 0
                    <= (to_float(row.get("yardline_100")) or 0)
                    <= 20
                )
                for row in group
            )
        )

        offense_score = int(
            any(
                (
                    to_int(row.get("touchdown"), 0) == 1
                    and clean(row.get("td_team"))
                    and clean(row.get("td_team")) == clean(row.get("posteam"))
                )
                or clean(row.get("espn_type_text")).casefold()
                in FIELD_GOAL_GOOD_TYPES
                for row in group
            )
        )

        first_seconds = to_int(group[0].get("game_seconds_remaining"))
        last_seconds = to_int(group[-1].get("game_seconds_remaining"))

        elapsed = None

        if (
            first_seconds is not None
            and last_seconds is not None
            and first_seconds >= last_seconds
        ):
            elapsed = first_seconds - last_seconds

        result = drive_result(group)
        start_wallclock = clean(group[0].get("espn_wallclock"))

        for row in group:
            row["ydsnet"] = ydsnet
            row["fixed_drive_result"] = result
            row["drive_real_start_time"] = start_wallclock
            row["drive_play_count"] = len(scrimmage)
            row["drive_time_of_possession"] = format_possession_time(elapsed)
            row["drive_first_downs"] = first_downs
            row["drive_inside20"] = inside20
            row["drive_ended_with_score"] = offense_score
            row["drive_quarter_start"] = group[0].get("qtr", "")
            row["drive_quarter_end"] = group[-1].get("qtr", "")


def add_epa_columns(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        epa = calculate_epa(row)

        if epa is None or not math.isfinite(epa):
            row["epa"] = ""
            row["success"] = ""
            row["qb_epa"] = ""
            continue

        epa = round(float(epa), 6)

        row["epa"] = epa
        row["success"] = int(epa > 0)

        if row.get("play_type") == "pass" or to_int(row.get("sack"), 0) == 1:
            row["qb_epa"] = epa


def add_wp_columns(
    rows: list[dict[str, Any]],
    home_id: str,
    away_id: str,
) -> None:
    previous_home_wp: float | None = None

    for row in rows:
        current_home_wp = to_float(row.get("_home_wp_after"))
        offense_id = clean(row.get("espn_start_team_id"))

        if (
            previous_home_wp is not None
            and offense_id in {home_id, away_id}
        ):
            row["wp"] = (
                previous_home_wp
                if offense_id == home_id
                else 1.0 - previous_home_wp
            )

        if (
            previous_home_wp is not None
            and current_home_wp is not None
            and offense_id in {home_id, away_id}
        ):
            home_delta = current_home_wp - previous_home_wp
            row["wpa"] = (
                home_delta
                if offense_id == home_id
                else -home_delta
            )

        if current_home_wp is not None:
            previous_home_wp = current_home_wp


# ─────────────────────────────────────────────
# GAME PULL
# ─────────────────────────────────────────────

def pull_one_game(
    schedule_row: dict[str, str],
    name_to_id: dict[str, str],
    team_id_to_name: dict[str, str],
    roster_lookup: dict[str, str],
) -> tuple[str, list[dict[str, Any]], str]:
    game_id = clean(schedule_row.get("game_id"))

    plays_url = PLAYS_URL_TEMPLATE.format(game_id=game_id)

    try:
        plays = fetch_collection(plays_url)
    except urllib.error.HTTPError as exc:
        if exc.code in {400, 404}:
            return game_id, [], f"plays unavailable HTTP {exc.code}"
        return game_id, [], f"plays HTTP error {exc.code}: {exc.reason}"
    except Exception as exc:
        return game_id, [], f"plays fetch failed: {exc}"

    if not plays:
        return game_id, [], "no plays returned"

    probability_items: list[dict[str, Any]] = []

    probability_url = PROBABILITIES_URL_TEMPLATE.format(game_id=game_id)

    try:
        probability_items = fetch_collection(probability_url)
    except Exception:
        # WP/WPA are compatibility enrichments, not required to preserve the
        # downstream team-stat calculations.  Do not discard otherwise valid
        # PBP when ESPN has no probability feed for a game.
        probability_items = []

    try:
        rows = normalize_game(
            schedule_row=schedule_row,
            plays=plays,
            probability_items=probability_items,
            name_to_id=name_to_id,
            team_id_to_name=team_id_to_name,
            roster_lookup=roster_lookup,
        )
    except Exception as exc:
        return game_id, [], f"normalization failed: {exc}"

    return game_id, rows, ""


# ─────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────

def clean_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = df.copy()

    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    # Keep compatibility columns first and any future audit/source columns
    # after them without silently discarding new information.
    ordered = [column for column in OUTPUT_COLUMNS if column in df.columns]
    remaining = [column for column in df.columns if column not in ordered]

    df = df[ordered + remaining]

    sort_columns = [
        column
        for column in ["game_id", "espn_sequence_number", "play_id"]
        if column in df.columns
    ]

    if sort_columns:
        df = df.sort_values(sort_columns, kind="stable")

    return df


def write_pbp(df: pd.DataFrame, season: int) -> Path:
    output_file = PBP_DIR / f"{season}_pbp.csv.gz"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_output = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    df.to_csv(
        tmp_output,
        index=False,
        compression="gzip",
    )

    os.replace(tmp_output, output_file)

    return output_file


def missing_feature_columns(df: pd.DataFrame) -> list[str]:
    required = FEATURE_COLUMNS + DOWNSTREAM_REQUIRED_EXTRA_COLUMNS
    return [column for column in required if column not in df.columns]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> int:
    ensure_dirs()
    reset_log()
    args = parse_args()

    try:
        season = get_season(args)

        if args.workers < 1:
            raise ValueError("--workers must be at least 1")

        log("=" * 80)
        log(
            f"pull_pbp.py started | season={season} "
            f"| source={args.source} | workers={args.workers}"
        )

        schedule = load_schedule(season)
        name_to_id = load_team_name_to_id()
        team_id_to_name = load_team_id_to_name()
        roster_lookup = load_roster_lookup()

        selected_game_ids = set(args.game_id or [])

        if selected_game_ids:
            missing_ids = sorted(
                game_id
                for game_id in selected_game_ids
                if game_id not in schedule
            )

            if missing_ids:
                raise ValueError(
                    "Requested game id(s) not present in local schedule: "
                    + ", ".join(missing_ids)
                )

            schedule_rows = [
                schedule[game_id]
                for game_id in sorted(selected_game_ids)
            ]
            future_games_skipped = 0
        else:
            all_schedule_rows = list(schedule.values())
            schedule_rows = [
                row
                for row in all_schedule_rows
                if not schedule_game_is_future(row)
            ]
            future_games_skipped = (
                len(all_schedule_rows) - len(schedule_rows)
            )

        log(f"schedule_games_eligible={len(schedule_rows)}")
        log(f"future_games_skipped={future_games_skipped}")
        log(f"team_name_id_mappings={len(name_to_id)}")
        log(f"roster_player_mappings={len(roster_lookup)}")

        all_rows: list[dict[str, Any]] = []
        skipped: list[tuple[str, str]] = []

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_game = {
                executor.submit(
                    pull_one_game,
                    schedule_row,
                    name_to_id,
                    team_id_to_name,
                    roster_lookup,
                ): clean(schedule_row.get("game_id"))
                for schedule_row in schedule_rows
            }

            completed = 0

            for future in as_completed(future_to_game):
                game_id = future_to_game[future]
                completed += 1

                try:
                    pulled_game_id, rows, reason = future.result()
                except Exception as exc:
                    pulled_game_id = game_id
                    rows = []
                    reason = f"worker failed: {exc}"

                if rows:
                    all_rows.extend(rows)
                    print(
                        f"game={pulled_game_id} "
                        f"plays={len(rows)} "
                        f"completed={completed}/{len(schedule_rows)}"
                    )
                else:
                    skipped.append((pulled_game_id, reason))
                    print(
                        f"game={pulled_game_id} skipped={reason} "
                        f"completed={completed}/{len(schedule_rows)}"
                    )

        df = pd.DataFrame(all_rows)
        df = clean_for_csv(df)

        output_file = write_pbp(df=df, season=season)

        missing = missing_feature_columns(df)

        scrimmage = (
            df[df["play_type"].isin(["pass", "run"])]
            if not df.empty and "play_type" in df.columns
            else pd.DataFrame()
        )

        epa_non_null = (
            int(
                pd.to_numeric(
                    scrimmage["epa"],
                    errors="coerce",
                ).notna().sum()
            )
            if not scrimmage.empty and "epa" in scrimmage.columns
            else 0
        )

        wp_non_null = (
            int(
                pd.to_numeric(
                    df["wp"],
                    errors="coerce",
                ).notna().sum()
            )
            if not df.empty and "wp" in df.columns
            else 0
        )

        log(f"games_with_pbp={len(schedule_rows) - len(skipped)}")
        log(f"future_games_skipped={future_games_skipped}")
        log(f"games_skipped={len(skipped)}")
        log(f"rows={len(df)}")
        log(f"columns={len(df.columns)}")
        log(f"scrimmage_rows={len(scrimmage)}")
        log(f"scrimmage_epa_rows={epa_non_null}")
        log(f"wp_rows={wp_non_null}")
        log(f"output={output_file}")

        if missing:
            log("missing_feature_columns=" + ",".join(missing))
        else:
            log("missing_feature_columns=none")

        for game_id, reason in sorted(skipped):
            log(f"SKIPPED game_id={game_id} reason={reason}")

        log("pull_pbp.py completed")
        log("=" * 80)

        print("cfb pull_pbp completed")
        print(f"season: {season}")
        print("source_used: espn_core_v2")
        print(f"games_with_pbp: {len(schedule_rows) - len(skipped)}")
        print(f"games_skipped: {len(skipped)}")
        print(f"future_games_skipped: {future_games_skipped}")
        print(f"rows: {len(df)}")
        print(f"columns: {len(df.columns)}")
        print(f"scrimmage_epa_rows: {epa_non_null}")
        print(f"wp_rows: {wp_non_null}")
        print(f"output: {output_file}")

        if missing:
            print("missing_feature_columns: " + ",".join(missing))
        else:
            print("missing_feature_columns: none")

        return 0

    except Exception as exc:
        log(f"ERROR: {repr(exc)}")
        log(traceback.format_exc())
        print("cfb pull_pbp failed", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

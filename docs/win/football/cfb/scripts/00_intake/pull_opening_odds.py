#!/usr/bin/env python3
# docs/win/football/cfb/scripts/00_intake/pull_opening_odds.py

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
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
WEEKLY_DIR = CFB_ROOT / "00_intake" / "schedule" / "weekly"
OPENERS_DIR = CFB_ROOT / "00_intake" / "odds" / "openers"
REPORT_ROOT = CFB_ROOT / "errors"

ESPN_BASE = (
    "https://sports.core.api.espn.com/v2/sports/football/"
    "leagues/college-football"
)
ESPN_CORE_HOST = "sports.core.api.espn.com"
SCRIPT_VERSION = "cfb-opening-odds-v2-2026-09-15"

LEGACY_OUTPUT_COLUMNS = [
    "game_id",
    "odds_provider_game_id",
    "market_type",
    "bet_side",
    "opening_line",
    "opening_odds_american",
    "opening_timestamp",
    "bookmaker",
    "opening_spread",
    "current_spread",
    "spread_movement",
    "opening_total",
    "current_total",
    "total_movement",
    "opening_moneyline",
    "current_moneyline",
    "moneyline_movement",
    "opener_status",
    "opener_missing_reason",
    "opener_http_status",
]

OUTPUT_COLUMNS = LEGACY_OUTPUT_COLUMNS[:7] + [
    "opening_captured_at",
] + LEGACY_OUTPUT_COLUMNS[7:]

WEEKLY_REQUIRED_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "odds_provider_game_id",
    "kickoff_utc",
    "game_locked",
    "away_team",
    "home_team",
    "bookmaker",
    "home_moneyline_american",
    "away_moneyline_american",
    "home_spread",
    "away_spread",
    "total",
    "odds_available",
    "odds_missing_reason",
]

VALID_MARKET_SIDES = {
    "h2h": {"home", "away"},
    "spreads": {"home", "away"},
    "totals": {"over", "under"},
}

VALID_STATUSES = {
    "ok",
    "missing",
    "error",
}

NUMERIC_FIELDS = {
    "opening_line",
    "opening_odds_american",
    "opening_spread",
    "current_spread",
    "spread_movement",
    "opening_total",
    "current_total",
    "total_movement",
    "opening_moneyline",
    "current_moneyline",
    "moneyline_movement",
}


class HardFetchError(RuntimeError):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
        raw = payload.get(key)

        if isinstance(raw, bool):
            raise ValueError(
                f"Current-week config {key} must be an integer"
            )

        try:
            values[key] = int(
                str(raw).strip()
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Current-week config {key} must be an integer"
            ) from exc

    if values["season"] < 2000:
        raise ValueError(
            f"Invalid season: {values['season']}"
        )

    if values["season_type"] < 1:
        raise ValueError(
            f"Invalid season_type: {values['season_type']}"
        )

    if values["week"] < 1:
        raise ValueError(
            f"Invalid week: {values['week']}"
        )

    return (
        values["season"],
        values["season_type"],
        values["week"],
    )


def read_csv(
    path: Path,
    required_columns: list[str],
    label: str,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        missing = [
            column
            for column in required_columns
            if column not in fieldnames
        ]

        if missing:
            raise ValueError(
                f"{label} missing columns: {missing}"
            )

        return list(reader)


def parse_aware_iso(
    value: object,
    label: str,
) -> datetime:
    text = str(
        value or ""
    ).strip()

    if not text:
        raise ValueError(
            f"{label} is blank"
        )

    if text.endswith("Z"):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        parsed = datetime.fromisoformat(
            text
        )
    except ValueError as exc:
        raise ValueError(
            f"{label} is not a valid ISO timestamp: {value!r}"
        ) from exc

    if parsed.tzinfo is None:
        raise ValueError(
            f"{label} must include a timezone: {value!r}"
        )

    return parsed.astimezone(
        timezone.utc
    )


def validate_weekly_rows(
    rows: list[dict[str, str]],
    season: int,
    season_type: int,
    week: int,
) -> None:
    if not rows:
        raise ValueError(
            "Weekly schedule is empty"
        )

    seen: set[str] = set()

    for index, row in enumerate(rows):
        game_id = str(
            row.get(
                "game_id",
                "",
            )
        ).strip()

        provider_game_id = str(
            row.get(
                "odds_provider_game_id",
                "",
            )
        ).strip()

        home_team = str(
            row.get(
                "home_team",
                "",
            )
        ).strip()

        away_team = str(
            row.get(
                "away_team",
                "",
            )
        ).strip()

        if not game_id:
            raise ValueError(
                f"Weekly schedule row {index} has blank game_id"
            )

        if game_id in seen:
            raise ValueError(
                "Weekly schedule contains duplicate "
                f"game_id={game_id}"
            )

        seen.add(game_id)

        if provider_game_id != game_id:
            raise ValueError(
                "Weekly schedule odds_provider_game_id "
                "must equal game_id for "
                f"game_id={game_id}: "
                f"{provider_game_id!r}"
            )

        if str(
            row.get(
                "season",
                "",
            )
        ).strip() != str(season):
            raise ValueError(
                "Weekly schedule season mismatch "
                f"for game_id={game_id}"
            )

        if str(
            row.get(
                "season_type",
                "",
            )
        ).strip() != str(season_type):
            raise ValueError(
                "Weekly schedule season_type mismatch "
                f"for game_id={game_id}"
            )

        if str(
            row.get(
                "week",
                "",
            )
        ).strip() != str(week):
            raise ValueError(
                "Weekly schedule week mismatch "
                f"for game_id={game_id}"
            )

        if not home_team or not away_team:
            raise ValueError(
                "Weekly schedule has blank team "
                f"for game_id={game_id}"
            )

        parse_aware_iso(
            row.get(
                "kickoff_utc",
                "",
            ),
            f"kickoff_utc game_id={game_id}",
        )

        game_locked = str(
            row.get(
                "game_locked",
                "",
            )
        ).strip()

        if game_locked not in {
            "0",
            "1",
        }:
            raise ValueError(
                "Weekly schedule has invalid "
                "game_locked for "
                f"game_id={game_id}: "
                f"{game_locked!r}"
            )

        odds_available = str(
            row.get(
                "odds_available",
                "",
            )
        ).strip()

        if odds_available not in {
            "0",
            "1",
        }:
            raise ValueError(
                "Weekly schedule has invalid "
                "odds_available for "
                f"game_id={game_id}: "
                f"{odds_available!r}"
            )

        bookmaker = canonical_bookmaker(
            row.get(
                "bookmaker",
                "",
            )
        )

        if (
            odds_available == "1"
            and not bookmaker
        ):
            raise ValueError(
                "Weekly schedule has odds_available=1 "
                "with blank bookmaker for "
                f"game_id={game_id}"
            )


def build_url(
    path: str,
    params: dict[str, object] | None = None,
) -> str:
    url = f"{ESPN_BASE}{path}"

    if params:
        return (
            f"{url}?"
            f"{urlencode(params)}"
        )

    return url


def http_get_json(
    url: str,
) -> tuple[
    int | None,
    object | None,
    str,
]:
    request = Request(
        url,
        headers={
            "User-Agent": "cfb-pull-opening-odds/4.0",
            "Accept": "application/json",
        },
    )

    try:
        with open_https(
            request,
            allowed_hosts={ESPN_CORE_HOST},
            timeout=45,
        ) as response:
            status = response.status
            body = (
                response.read()
                .decode("utf-8")
            )

    except HTTPError as exc:
        body = ""

        try:
            body = (
                exc.read()
                .decode("utf-8")
            )
        except Exception:
            pass

        return (
            exc.code,
            None,
            body or str(exc),
        )

    except URLError as exc:
        return (
            None,
            None,
            str(exc),
        )

    except Exception as exc:
        return (
            None,
            None,
            str(exc),
        )

    if (
        status < 200
        or status >= 300
    ):
        return (
            status,
            None,
            body,
        )

    try:
        return (
            status,
            json.loads(body),
            "",
        )
    except Exception as exc:
        return (
            status,
            None,
            f"JSON parse failed: {exc}",
        )


def to_float(
    value: object,
) -> float | None:
    if (
        value is None
        or isinstance(
            value,
            bool,
        )
    ):
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.casefold() in {
        "even",
        "ev",
        "evens",
    }:
        return 100.0

    try:
        number = float(text)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(number):
        return None

    return number


def clean_number(
    value: object,
) -> str:
    number = to_float(value)

    if number is None:
        return ""

    if number.is_integer():
        return str(
            int(number)
        )

    return str(number)


def normalize_american(
    value: object,
) -> str:
    number = to_float(value)

    if (
        number is None
        or number == 0
    ):
        return ""

    return str(
        int(
            round(number)
        )
    )


def numeric_movement(
    current_value: object,
    opening_value: object,
) -> str:
    current = to_float(
        current_value
    )

    opening = to_float(
        opening_value
    )

    if (
        current is None
        or opening is None
    ):
        return ""

    movement = (
        current
        - opening
    )

    if movement.is_integer():
        return str(
            int(movement)
        )

    return str(
        round(
            movement,
            4,
        )
    )


def normalize_provider_timestamp(
    value: object,
) -> str:
    text = str(
        value or ""
    ).strip()

    if not text:
        return ""

    number = to_float(text)

    if number is not None:
        if number > 1_000_000_000_000:
            return datetime.fromtimestamp(
                number / 1000,
                tz=timezone.utc,
            ).isoformat()

        if number > 1_000_000_000:
            return datetime.fromtimestamp(
                number,
                tz=timezone.utc,
            ).isoformat()

    parsed = parse_aware_iso(
        text,
        "ESPN opening timestamp",
    )

    return parsed.isoformat()


def bookmaker_key(
    value: object,
) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(
            value or ""
        )
        .strip()
        .lower(),
    )


def canonical_bookmaker(
    value: object,
) -> str:
    text = str(
        value or ""
    ).strip()

    key = bookmaker_key(text)

    if key == "draftkings":
        return "DraftKings"

    if key == "fanduel":
        return "FanDuel"

    return text


def fetch_ref(
    ref: str,
    label: str,
) -> dict:
    if ref.startswith(
        "http://sports.core.api.espn.com/"
    ):
        ref = (
            "https://sports.core.api.espn.com/"
            + ref[
                len("http://sports.core.api.espn.com/"):
            ]
        )

    (
        status,
        payload,
        error,
    ) = http_get_json(ref)

    if (
        status is None
        or status < 200
        or status >= 300
        or payload is None
    ):
        raise HardFetchError(
            f"{label} fetch failed: "
            f"status={status!r}, "
            f"ref={ref}, "
            f"error={error}"
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise HardFetchError(
            f"{label} returned "
            f"non-object JSON: {ref}"
        )

    return payload


def provider_info(
    odds_item: dict,
) -> dict[str, object]:
    provider = odds_item.get(
        "provider"
    )

    if not isinstance(
        provider,
        dict,
    ):
        return {
            "id": "",
            "name": "",
            "priority": None,
        }

    provider_data = provider

    if (
        provider.get("$ref")
        and not (
            provider.get("id")
            or provider.get("name")
            or provider.get(
                "displayName"
            )
            or provider.get(
                "shortName"
            )
        )
    ):
        provider_data = fetch_ref(
            str(
                provider["$ref"]
            ),
            "provider reference",
        )

    return {
        "id": str(
            provider_data.get(
                "id",
                "",
            )
        ).strip(),
        "name": canonical_bookmaker(
            provider_data.get("name")
            or provider_data.get(
                "displayName"
            )
            or provider_data.get(
                "shortName"
            )
            or ""
        ),
        "priority": to_float(
            provider_data.get(
                "priority"
            )
        ),
    }


def resolve_collection_items(
    collection: object,
) -> list[dict]:
    if not isinstance(
        collection,
        dict,
    ):
        raise HardFetchError(
            "ESPN odds response was "
            "not a JSON object"
        )

    items = collection.get(
        "items",
        [],
    )

    if not isinstance(
        items,
        list,
    ):
        raise HardFetchError(
            "ESPN odds response items "
            "field was not a list"
        )

    resolved: list[dict] = []

    for index, item in enumerate(
        items
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise HardFetchError(
                "ESPN odds response contains "
                "non-object item at index "
                f"{index}"
            )

        if (
            item.get("$ref")
            and not (
                item.get("provider")
                or item.get(
                    "homeTeamOdds"
                )
                or item.get(
                    "awayTeamOdds"
                )
                or item.get("open")
                or item.get("current")
                or item.get(
                    "overUnder"
                ) is not None
            )
        ):
            item = fetch_ref(
                str(
                    item["$ref"]
                ),
                "odds item reference",
            )

        resolved.append(item)

    return resolved


def select_odds_item(
    items: list[dict],
    bookmaker_name: str,
) -> tuple[
    dict | None,
    str,
]:
    if not items:
        return (
            None,
            "no_current_odds_items",
        )

    desired_key = bookmaker_key(
        bookmaker_name
    )

    if desired_key:
        for item in items:
            info = provider_info(item)
            candidate = (
                info["name"]
                or info["id"]
            )

            if (
                bookmaker_key(
                    candidate
                )
                == desired_key
            ):
                return (
                    item,
                    "",
                )

        return (
            None,
            "bookmaker_not_found",
        )

    ranked: list[
        tuple[
            float,
            int,
            dict,
        ]
    ] = []

    for index, item in enumerate(
        items
    ):
        info = provider_info(item)
        priority = info[
            "priority"
        ]

        rank = (
            float(priority)
            if priority is not None
            else 1_000_000.0
        )

        ranked.append(
            (
                rank,
                index,
                item,
            )
        )

    ranked.sort(
        key=lambda value: (
            value[0],
            value[1],
        )
    )

    return (
        ranked[0][2],
        "",
    )


def fetch_current_odds(
    game_id: str,
    bookmaker_name: str,
) -> tuple[
    dict | None,
    int | str,
    str,
    str,
]:
    path = (
        f"/events/{game_id}/"
        f"competitions/{game_id}/odds"
    )

    url = build_url(
        path,
        {
            "limit": 100,
            "lang": "en",
            "region": "us",
        },
    )

    (
        status,
        collection,
        error,
    ) = http_get_json(url)

    if status == 404:
        return (
            None,
            404,
            "not_found",
            url,
        )

    if (
        status is None
        or status < 200
        or status >= 300
        or collection is None
    ):
        raise HardFetchError(
            "Opening-odds request failed "
            f"for game_id={game_id}: "
            f"status={status!r}, "
            f"error={error}"
        )

    items = resolve_collection_items(
        collection
    )

    (
        selected,
        missing_reason,
    ) = select_odds_item(
        items,
        bookmaker_name,
    )

    if selected is None:
        return (
            None,
            status,
            missing_reason,
            url,
        )

    return (
        selected,
        status,
        "",
        url,
    )


def market_block(
    parent: object,
    snapshot: str,
) -> dict:
    if not isinstance(
        parent,
        dict,
    ):
        return {}

    block = parent.get(
        snapshot
    )

    return (
        block
        if isinstance(
            block,
            dict,
        )
        else {}
    )


def market_object(
    block: object,
    market: str,
) -> dict:
    if not isinstance(
        block,
        dict,
    ):
        return {}

    obj = block.get(market)

    return (
        obj
        if isinstance(
            obj,
            dict,
        )
        else {}
    )


def market_value(
    block: object,
    market: str,
) -> str:
    obj = market_object(
        block,
        market,
    )

    if not obj:
        return ""

    return clean_number(
        obj.get("american")
        or obj.get(
            "alternateDisplayValue"
        )
        or obj.get("value")
    )


def market_american(
    block: object,
    market: str,
) -> str:
    obj = market_object(
        block,
        market,
    )

    if not obj:
        return ""

    return normalize_american(
        obj.get("american")
        or obj.get(
            "alternateDisplayValue"
        )
        or obj.get("value")
    )


def first_block_timestamp(
    block: object,
) -> str:
    if not isinstance(
        block,
        dict,
    ):
        return ""

    for key in (
        "timestamp",
        "lastUpdated",
        "lastUpdate",
        "updated",
        "date",
    ):
        value = block.get(key)

        if (
            value is not None
            and str(
                value
            ).strip()
        ):
            return normalize_provider_timestamp(
                value
            )

    for value in block.values():
        if not isinstance(
            value,
            dict,
        ):
            continue

        for key in (
            "timestamp",
            "lastUpdated",
            "lastUpdate",
            "updated",
            "date",
        ):
            candidate = value.get(
                key
            )

            if (
                candidate is not None
                and str(
                    candidate
                ).strip()
            ):
                return normalize_provider_timestamp(
                    candidate
                )

    return ""


def infer_opening_favorite(
    home_open: dict,
    away_open: dict,
    home_moneyline: str,
    away_moneyline: str,
) -> str:
    home_ml = to_float(
        home_moneyline
    )

    away_ml = to_float(
        away_moneyline
    )

    if (
        home_ml is not None
        and away_ml is not None
    ):
        if (
            home_ml < 0 < away_ml
        ):
            return "home"

        if (
            away_ml < 0 < home_ml
        ):
            return "away"

    home_favorite = (
        home_open.get(
            "favorite"
        )
        is True
    )

    away_favorite = (
        away_open.get(
            "favorite"
        )
        is True
    )

    if (
        home_favorite
        and not away_favorite
    ):
        return "home"

    if (
        away_favorite
        and not home_favorite
    ):
        return "away"

    return ""


def normalize_opening_spreads(
    home_open: dict,
    away_open: dict,
    home_moneyline: str,
    away_moneyline: str,
) -> tuple[str, str]:
    home_num = to_float(
        market_value(
            home_open,
            "pointSpread",
        )
    )

    away_num = to_float(
        market_value(
            away_open,
            "pointSpread",
        )
    )

    if (
        home_num is None
        and away_num is None
    ):
        return (
            "",
            "",
        )

    favorite = infer_opening_favorite(
        home_open,
        away_open,
        home_moneyline,
        away_moneyline,
    )

    if favorite:
        magnitude = abs(
            home_num
            if home_num is not None
            else away_num
        )

        if favorite == "home":
            return (
                clean_number(
                    -magnitude
                ),
                clean_number(
                    magnitude
                ),
            )

        return (
            clean_number(
                magnitude
            ),
            clean_number(
                -magnitude
            ),
        )

    if (
        home_num is not None
        and away_num is not None
    ):
        if abs(
            home_num
            + away_num
        ) > 0.000001:
            raise ValueError(
                "ESPN opening spreads are "
                "not symmetric: "
                f"home={home_num}, "
                f"away={away_num}"
            )

        return (
            clean_number(
                home_num
            ),
            clean_number(
                away_num
            ),
        )

    if home_num is not None:
        return (
            clean_number(
                home_num
            ),
            clean_number(
                -home_num
            ),
        )

    return (
        clean_number(
            -away_num
        ),
        clean_number(
            away_num
        ),
    )


def get_opening(
    odds_item: dict,
) -> dict[str, str]:
    game_open = market_block(
        odds_item,
        "open",
    )

    home_team = (
        odds_item.get(
            "homeTeamOdds"
        )
        if isinstance(
            odds_item.get(
                "homeTeamOdds"
            ),
            dict,
        )
        else {}
    )

    away_team = (
        odds_item.get(
            "awayTeamOdds"
        )
        if isinstance(
            odds_item.get(
                "awayTeamOdds"
            ),
            dict,
        )
        else {}
    )

    home_open = market_block(
        home_team,
        "open",
    )

    away_open = market_block(
        away_team,
        "open",
    )

    home_moneyline = (
        market_american(
            home_open,
            "moneyLine",
        )
    )

    away_moneyline = (
        market_american(
            away_open,
            "moneyLine",
        )
    )

    (
        home_spread,
        away_spread,
    ) = normalize_opening_spreads(
        home_open,
        away_open,
        home_moneyline,
        away_moneyline,
    )

    timestamp = (
        first_block_timestamp(
            game_open
        )
        or first_block_timestamp(
            home_open
        )
        or first_block_timestamp(
            away_open
        )
    )

    return {
        "home_moneyline": (
            home_moneyline
        ),
        "away_moneyline": (
            away_moneyline
        ),
        "home_spread": (
            home_spread
        ),
        "away_spread": (
            away_spread
        ),
        "home_spread_odds": (
            market_american(
                home_open,
                "spread",
            )
        ),
        "away_spread_odds": (
            market_american(
                away_open,
                "spread",
            )
        ),
        "total": market_value(
            game_open,
            "total",
        ),
        "over_odds": (
            market_american(
                game_open,
                "over",
            )
        ),
        "under_odds": (
            market_american(
                game_open,
                "under",
            )
        ),
        "timestamp": timestamp,
    }


def row_has_required_opening(
    row: dict[str, str],
) -> bool:
    market_type = str(
        row.get(
            "market_type",
            "",
        )
    ).strip()

    if market_type == "h2h":
        return bool(
            str(
                row.get(
                    "opening_moneyline",
                    "",
                )
            ).strip()
            or str(
                row.get(
                    "opening_odds_american",
                    "",
                )
            ).strip()
        )

    if market_type == "spreads":
        return bool(
            str(
                row.get(
                    "opening_spread",
                    "",
                )
            ).strip()
            or str(
                row.get(
                    "opening_line",
                    "",
                )
            ).strip()
        )

    if market_type == "totals":
        return bool(
            str(
                row.get(
                    "opening_total",
                    "",
                )
            ).strip()
            or str(
                row.get(
                    "opening_line",
                    "",
                )
            ).strip()
        )

    return False


def status_fields(
    value: object,
    missing_reason: str,
    http_status: object,
) -> dict[str, str]:
    if str(
        value or ""
    ).strip():
        return {
            "opener_status": "ok",
            "opener_missing_reason": "",
            "opener_http_status": str(
                http_status or ""
            ),
        }

    return {
        "opener_status": "missing",
        "opener_missing_reason": (
            missing_reason
        ),
        "opener_http_status": str(
            http_status or ""
        ),
    }


def base_row(
    weekly_row: dict[str, str],
    market_type: str,
    bet_side: str,
    bookmaker: str,
) -> dict[str, str]:
    game_id = str(
        weekly_row.get(
            "game_id",
            "",
        )
    ).strip()

    return {
        "game_id": game_id,
        "odds_provider_game_id": (
            game_id
        ),
        "market_type": market_type,
        "bet_side": bet_side,
        "opening_line": "",
        "opening_odds_american": "",
        "opening_timestamp": "",
        "opening_captured_at": "",
        "bookmaker": (
            canonical_bookmaker(
                bookmaker
            )
        ),
        "opening_spread": "",
        "current_spread": "",
        "spread_movement": "",
        "opening_total": "",
        "current_total": "",
        "total_movement": "",
        "opening_moneyline": "",
        "current_moneyline": "",
        "moneyline_movement": "",
        "opener_status": "",
        "opener_missing_reason": "",
        "opener_http_status": "",
    }


def build_game_rows(
    weekly_row: dict[str, str],
    opening: dict[str, str],
    bookmaker: str,
    http_status: object,
    request_missing_reason: str,
    captured_at: str,
) -> list[dict[str, str]]:
    rows: list[
        dict[str, str]
    ] = []

    provider_missing = bool(
        request_missing_reason
    )

    for side in (
        "home",
        "away",
    ):
        opening_value = (
            opening.get(
                f"{side}_moneyline",
                "",
            )
        )

        current_value = str(
            weekly_row.get(
                f"{side}_moneyline_american",
                "",
            )
        ).strip()

        reason = (
            request_missing_reason
            if provider_missing
            else (
                f"opening_{side}_"
                "moneyline_missing"
            )
        )

        row = base_row(
            weekly_row,
            "h2h",
            side,
            bookmaker,
        )

        row.update(
            {
                "opening_odds_american": (
                    opening_value
                ),
                "opening_timestamp": (
                    opening.get(
                        "timestamp",
                        "",
                    )
                ),
                "opening_captured_at": (
                    captured_at
                    if opening_value
                    else ""
                ),
                "opening_moneyline": (
                    opening_value
                ),
                "current_moneyline": (
                    current_value
                ),
                "moneyline_movement": (
                    numeric_movement(
                        current_value,
                        opening_value,
                    )
                ),
                **status_fields(
                    opening_value,
                    reason,
                    http_status,
                ),
            }
        )

        rows.append(row)

    for side in (
        "home",
        "away",
    ):
        opening_spread = (
            opening.get(
                f"{side}_spread",
                "",
            )
        )

        opening_odds = (
            opening.get(
                f"{side}_spread_odds",
                "",
            )
        )

        current_spread = str(
            weekly_row.get(
                f"{side}_spread",
                "",
            )
        ).strip()

        reason = (
            request_missing_reason
            if provider_missing
            else (
                f"opening_{side}_"
                "spread_missing"
            )
        )

        row = base_row(
            weekly_row,
            "spreads",
            side,
            bookmaker,
        )

        row.update(
            {
                "opening_line": (
                    opening_spread
                ),
                "opening_odds_american": (
                    opening_odds
                ),
                "opening_timestamp": (
                    opening.get(
                        "timestamp",
                        "",
                    )
                ),
                "opening_captured_at": (
                    captured_at
                    if opening_spread
                    else ""
                ),
                "opening_spread": (
                    opening_spread
                ),
                "current_spread": (
                    current_spread
                ),
                "spread_movement": (
                    numeric_movement(
                        current_spread,
                        opening_spread,
                    )
                ),
                **status_fields(
                    opening_spread,
                    reason,
                    http_status,
                ),
            }
        )

        rows.append(row)

    opening_total = (
        opening.get(
            "total",
            "",
        )
    )

    current_total = str(
        weekly_row.get(
            "total",
            "",
        )
    ).strip()

    for side in (
        "over",
        "under",
    ):
        opening_odds = (
            opening.get(
                f"{side}_odds",
                "",
            )
        )

        reason = (
            request_missing_reason
            if provider_missing
            else "opening_total_missing"
        )

        row = base_row(
            weekly_row,
            "totals",
            side,
            bookmaker,
        )

        row.update(
            {
                "opening_line": (
                    opening_total
                ),
                "opening_odds_american": (
                    opening_odds
                ),
                "opening_timestamp": (
                    opening.get(
                        "timestamp",
                        "",
                    )
                ),
                "opening_captured_at": (
                    captured_at
                    if opening_total
                    else ""
                ),
                "opening_total": (
                    opening_total
                ),
                "current_total": (
                    current_total
                ),
                "total_movement": (
                    numeric_movement(
                        current_total,
                        opening_total,
                    )
                ),
                **status_fields(
                    opening_total,
                    reason,
                    http_status,
                ),
            }
        )

        rows.append(row)

    return rows


def build_opening_rows(
    weekly_rows: list[
        dict[str, str]
    ],
    captured_at: str,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, int],
]:
    output_rows: list[
        dict[str, str]
    ] = []

    hard_failures: list[
        dict[str, str]
    ] = []

    provider_counts: dict[
        str,
        int,
    ] = {}

    for weekly_row in weekly_rows:
        game_id = str(
            weekly_row[
                "game_id"
            ]
        ).strip()

        weekly_bookmaker = (
            canonical_bookmaker(
                weekly_row.get(
                    "bookmaker",
                    "",
                )
            )
        )

        desired_bookmaker = (
            weekly_bookmaker
            if str(
                weekly_row.get(
                    "odds_available",
                    "",
                )
            ).strip() == "1"
            else ""
        )

        try:
            (
                odds_item,
                http_status,
                missing_reason,
                _url,
            ) = fetch_current_odds(
                game_id,
                desired_bookmaker,
            )

        except HardFetchError as exc:
            hard_failures.append(
                {
                    "game_id": game_id,
                    "error_type": (
                        type(exc).__name__
                    ),
                    "message": str(exc),
                }
            )

            continue

        if odds_item is None:
            bookmaker = (
                weekly_bookmaker
            )
            opening: dict[
                str,
                str,
            ] = {}

        else:
            provider = provider_info(
                odds_item
            )

            bookmaker = canonical_bookmaker(
                provider["name"]
                or provider["id"]
                or weekly_bookmaker
            )

            if not bookmaker:
                hard_failures.append(
                    {
                        "game_id": game_id,
                        "error_type": (
                            "ProviderIdentityError"
                        ),
                        "message": (
                            "ESPN opener item has "
                            "no provider identity"
                        ),
                    }
                )

                continue

            if (
                desired_bookmaker
                and bookmaker_key(
                    bookmaker
                )
                != bookmaker_key(
                    desired_bookmaker
                )
            ):
                hard_failures.append(
                    {
                        "game_id": game_id,
                        "error_type": (
                            "BookmakerMismatchError"
                        ),
                        "message": (
                            "Requested bookmaker="
                            f"{desired_bookmaker!r}, "
                            "received bookmaker="
                            f"{bookmaker!r}"
                        ),
                    }
                )

                continue

            provider_counts[
                bookmaker
            ] = (
                provider_counts.get(
                    bookmaker,
                    0,
                )
                + 1
            )

            opening = get_opening(
                odds_item
            )

        output_rows.extend(
            build_game_rows(
                weekly_row,
                opening,
                bookmaker,
                http_status,
                missing_reason,
                captured_at,
            )
        )

    return (
        output_rows,
        hard_failures,
        provider_counts,
    )


def row_key(
    row: dict[str, str],
) -> tuple[
    str,
    str,
    str,
    str,
]:
    return (
        str(
            row.get(
                "game_id",
                "",
            )
        ).strip(),
        str(
            row.get(
                "market_type",
                "",
            )
        ).strip(),
        str(
            row.get(
                "bet_side",
                "",
            )
        ).strip(),
        bookmaker_key(
            row.get(
                "bookmaker",
                "",
            )
        ),
    )


def expected_movement(
    row: dict[str, str],
) -> str:
    market_type = str(
        row.get(
            "market_type",
            "",
        )
    ).strip()

    if market_type == "h2h":
        return numeric_movement(
            row.get(
                "current_moneyline",
                "",
            ),
            row.get(
                "opening_moneyline",
                "",
            ),
        )

    if market_type == "spreads":
        return numeric_movement(
            row.get(
                "current_spread",
                "",
            ),
            row.get(
                "opening_spread",
                "",
            ),
        )

    if market_type == "totals":
        return numeric_movement(
            row.get(
                "current_total",
                "",
            ),
            row.get(
                "opening_total",
                "",
            ),
        )

    return ""



def _validate_opener_row_identity(
    *,
    game_id: str,
    provider_game_id: str,
    market_type: str,
    bet_side: str,
    status: str,
    index: int,
    label: str,
) -> None:
    if not game_id:
        raise ValueError(
            f"{label} row {index} "
            "has blank game_id"
        )

    if provider_game_id != game_id:
        raise ValueError(
            f"{label} row {index} "
            "provider game ID mismatch: "
            f"game_id={game_id}, "
            "odds_provider_game_id="
            f"{provider_game_id!r}"
        )

    if market_type not in VALID_MARKET_SIDES:
        raise ValueError(
            f"{label} row {index} "
            "has invalid market_type="
            f"{market_type!r}"
        )

    if bet_side not in VALID_MARKET_SIDES[market_type]:
        raise ValueError(
            f"{label} row {index} "
            "has invalid bet_side="
            f"{bet_side!r} for "
            f"{market_type}"
        )

    if status not in VALID_STATUSES:
        raise ValueError(
            f"{label} row {index} "
            "has invalid opener_status="
            f"{status!r}"
        )


def _validate_opener_row_status(
    *,
    status: str,
    has_opening: bool,
    bookmaker: str,
    index: int,
    label: str,
) -> None:
    if status == "ok" and not has_opening:
        raise ValueError(
            f"{label} row {index} "
            "status=ok without opening value"
        )

    if has_opening and status != "ok":
        raise ValueError(
            f"{label} row {index} "
            "has opening value but "
            f"status={status!r}"
        )

    if status == "ok" and not bookmaker:
        raise ValueError(
            f"{label} row {index} "
            "status=ok with blank bookmaker"
        )


def validate_opener_row(
    row: dict[str, str],
    index: int,
    label: str,
) -> None:
    game_id = str(
        row.get(
            "game_id",
            "",
        )
    ).strip()

    provider_game_id = str(
        row.get(
            "odds_provider_game_id",
            "",
        )
    ).strip()

    market_type = str(
        row.get(
            "market_type",
            "",
        )
    ).strip()

    bet_side = str(
        row.get(
            "bet_side",
            "",
        )
    ).strip()

    status = str(
        row.get(
            "opener_status",
            "",
        )
    ).strip()

    _validate_opener_row_identity(
        game_id=game_id,
        provider_game_id=provider_game_id,
        market_type=market_type,
        bet_side=bet_side,
        status=status,
        index=index,
        label=label,
    )

    bookmaker = canonical_bookmaker(
        row.get(
            "bookmaker",
            "",
        )
    )

    row["bookmaker"] = bookmaker

    for field in NUMERIC_FIELDS:
        text = str(
            row.get(
                field,
                "",
            )
        ).strip()

        if (
            text
            and to_float(text) is None
        ):
            raise ValueError(
                f"{label} row {index} "
                "has invalid numeric "
                f"{field}={text!r}"
            )

    for field in (
        "opening_odds_american",
        "opening_moneyline",
        "current_moneyline",
    ):
        text = str(
            row.get(
                field,
                "",
            )
        ).strip()

        number = to_float(text)

        if (
            text
            and (
                number is None
                or number == 0
            )
        ):
            raise ValueError(
                f"{label} row {index} "
                "has invalid American odds "
                f"{field}={text!r}"
            )

    opening_timestamp = str(
        row.get(
            "opening_timestamp",
            "",
        )
    ).strip()

    if opening_timestamp:
        parse_aware_iso(
            opening_timestamp,
            (
                f"{label} "
                "opening_timestamp "
                f"row {index}"
            ),
        )

    captured_at = str(
        row.get(
            "opening_captured_at",
            "",
        )
    ).strip()

    if captured_at:
        parse_aware_iso(
            captured_at,
            (
                f"{label} "
                "opening_captured_at "
                f"row {index}"
            ),
        )

    has_opening = (
        row_has_required_opening(
            row
        )
    )

    _validate_opener_row_status(
        status=status,
        has_opening=has_opening,
        bookmaker=bookmaker,
        index=index,
        label=label,
    )

    expected = expected_movement(
        row
    )

    if market_type == "h2h":
        actual = str(
            row.get(
                "moneyline_movement",
                "",
            )
        ).strip()

    elif market_type == "spreads":
        actual = str(
            row.get(
                "spread_movement",
                "",
            )
        ).strip()

    else:
        actual = str(
            row.get(
                "total_movement",
                "",
            )
        ).strip()

    if actual != expected:
        raise ValueError(
            f"{label} row {index} "
            "movement mismatch for "
            f"key={row_key(row)}: "
            f"actual={actual!r}, "
            f"expected={expected!r}"
        )


def read_existing_openers(
    path: Path,
) -> tuple[
    list[dict[str, str]],
    bool,
]:
    if not path.exists():
        return (
            [],
            False,
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        fieldnames = (
            reader.fieldnames
            or []
        )

        if fieldnames == OUTPUT_COLUMNS:
            legacy_schema = False

        elif (
            fieldnames
            == LEGACY_OUTPUT_COLUMNS
        ):
            legacy_schema = True

        else:
            raise ValueError(
                "Existing opener file schema "
                "mismatch. "
                f"expected={OUTPUT_COLUMNS}, "
                "legacy="
                f"{LEGACY_OUTPUT_COLUMNS}, "
                f"actual={fieldnames}"
            )

        rows: list[
            dict[str, str]
        ] = []

        seen: set[
            tuple[
                str,
                str,
                str,
                str,
            ]
        ] = set()

        for index, source in enumerate(
            reader
        ):
            row = {
                column: source.get(
                    column,
                    "",
                )
                for column
                in OUTPUT_COLUMNS
            }

            if legacy_schema:
                row[
                    "opening_captured_at"
                ] = ""

            row["bookmaker"] = (
                canonical_bookmaker(
                    row.get(
                        "bookmaker",
                        "",
                    )
                )
            )

            validate_opener_row(
                row,
                index,
                "existing opener",
            )

            key = row_key(row)

            if key in seen:
                raise ValueError(
                    "Existing opener file "
                    "contains duplicate key: "
                    f"{key}"
                )

            seen.add(key)
            rows.append(row)

    return (
        rows,
        legacy_schema,
    )


def opening_fields_for_market(
    market_type: str,
) -> tuple[str, ...]:
    common = (
        "opening_line",
        "opening_odds_american",
        "opening_timestamp",
        "opening_captured_at",
    )

    if market_type == "h2h":
        return (
            common
            + (
                "opening_moneyline",
            )
        )

    if market_type == "spreads":
        return (
            common
            + (
                "opening_spread",
            )
        )

    if market_type == "totals":
        return (
            common
            + (
                "opening_total",
            )
        )

    raise ValueError(
        "Unsupported market_type="
        f"{market_type!r}"
    )


def merge_row(
    existing: dict[str, str],
    new: dict[str, str],
) -> tuple[
    dict[str, str],
    bool,
    bool,
]:
    merged = dict(existing)

    market_type = str(
        new["market_type"]
    ).strip()

    existing_has_opening = (
        row_has_required_opening(
            existing
        )
    )

    new_has_opening = (
        row_has_required_opening(
            new
        )
    )

    preserved = False
    captured = False

    if existing_has_opening:
        preserved = True

        for field in (
            opening_fields_for_market(
                market_type
            )
        ):
            merged[field] = (
                existing.get(
                    field,
                    "",
                )
            )

        merged[
            "opener_status"
        ] = "ok"

        merged[
            "opener_missing_reason"
        ] = ""

    elif new_has_opening:
        captured = True

        for field in (
            opening_fields_for_market(
                market_type
            )
        ):
            merged[field] = (
                new.get(
                    field,
                    "",
                )
            )

        merged[
            "opener_status"
        ] = "ok"

        merged[
            "opener_missing_reason"
        ] = ""

    else:
        for field in (
            opening_fields_for_market(
                market_type
            )
        ):
            merged[field] = (
                new.get(
                    field,
                    "",
                )
            )

        merged[
            "opener_status"
        ] = new.get(
            "opener_status",
            "missing",
        )

        merged[
            "opener_missing_reason"
        ] = new.get(
            "opener_missing_reason",
            "",
        )

    for field in (
        "game_id",
        "odds_provider_game_id",
        "market_type",
        "bet_side",
        "bookmaker",
        "current_spread",
        "current_total",
        "current_moneyline",
        "opener_http_status",
    ):
        merged[field] = new.get(
            field,
            merged.get(
                field,
                "",
            ),
        )

    merged[
        "spread_movement"
    ] = numeric_movement(
        merged.get(
            "current_spread",
            "",
        ),
        merged.get(
            "opening_spread",
            "",
        ),
    )

    merged[
        "total_movement"
    ] = numeric_movement(
        merged.get(
            "current_total",
            "",
        ),
        merged.get(
            "opening_total",
            "",
        ),
    )

    merged[
        "moneyline_movement"
    ] = numeric_movement(
        merged.get(
            "current_moneyline",
            "",
        ),
        merged.get(
            "opening_moneyline",
            "",
        ),
    )

    return (
        merged,
        preserved,
        captured,
    )


def upsert_rows(
    existing_rows: list[
        dict[str, str]
    ],
    new_rows: list[
        dict[str, str]
    ],
) -> tuple[
    list[dict[str, str]],
    int,
    int,
    int,
]:
    keyed = {
        row_key(row): dict(row)
        for row in existing_rows
    }

    preserved_count = 0
    captured_count = 0
    inserted_count = 0

    real_provider_keys = {
        (
            row["game_id"],
            row["market_type"],
            row["bet_side"],
        )
        for row in new_rows
        if bookmaker_key(
            row.get(
                "bookmaker",
                "",
            )
        )
    }

    for key in list(keyed):
        (
            game_id,
            market_type,
            bet_side,
            bookmaker,
        ) = key

        if (
            not bookmaker
            and (
                game_id,
                market_type,
                bet_side,
            )
            in real_provider_keys
        ):
            del keyed[key]

    for new in new_rows:
        key = row_key(new)
        existing = keyed.get(key)

        if existing is None:
            keyed[key] = dict(new)
            inserted_count += 1

            if row_has_required_opening(
                new
            ):
                captured_count += 1

            continue

        (
            merged,
            preserved,
            captured,
        ) = merge_row(
            existing,
            new,
        )

        keyed[key] = merged

        preserved_count += int(
            preserved
        )

        captured_count += int(
            captured
        )

    rows = list(
        keyed.values()
    )

    rows.sort(
        key=lambda row: (
            row.get(
                "game_id",
                "",
            ),
            row.get(
                "market_type",
                "",
            ),
            row.get(
                "bet_side",
                "",
            ),
            bookmaker_key(
                row.get(
                    "bookmaker",
                    "",
                )
            ),
        )
    )

    return (
        rows,
        preserved_count,
        captured_count,
        inserted_count,
    )


def validate_new_coverage(
    weekly_rows: list[
        dict[str, str]
    ],
    new_rows: list[
        dict[str, str]
    ],
) -> None:
    expected_games = {
        str(
            row["game_id"]
        ).strip()
        for row in weekly_rows
    }

    expected_pairs = {
        (
            game_id,
            market_type,
            side,
        )
        for game_id
        in expected_games
        for market_type, sides
        in VALID_MARKET_SIDES.items()
        for side
        in sides
    }

    observed_pairs = {
        (
            str(
                row.get(
                    "game_id",
                    "",
                )
            ).strip(),
            str(
                row.get(
                    "market_type",
                    "",
                )
            ).strip(),
            str(
                row.get(
                    "bet_side",
                    "",
                )
            ).strip(),
        )
        for row in new_rows
    }

    if (
        len(new_rows)
        != len(
            expected_pairs
        )
    ):
        raise ValueError(
            "New opener coverage row "
            "count mismatch: "
            f"rows={len(new_rows)}, "
            f"expected="
            f"{len(expected_pairs)}"
        )

    if (
        observed_pairs
        != expected_pairs
    ):
        missing = sorted(
            expected_pairs
            - observed_pairs
        )

        extra = sorted(
            observed_pairs
            - expected_pairs
        )

        raise ValueError(
            "New opener coverage mismatch. "
            f"missing={missing[:20]}, "
            f"extra={extra[:20]}"
        )


def validate_final_rows(
    rows: list[
        dict[str, str]
    ],
) -> None:
    seen: set[
        tuple[
            str,
            str,
            str,
            str,
        ]
    ] = set()

    for index, row in enumerate(
        rows
    ):
        if (
            set(row)
            != set(
                OUTPUT_COLUMNS
            )
        ):
            raise ValueError(
                "Final opener row "
                f"{index} does not match "
                "output schema"
            )

        validate_opener_row(
            row,
            index,
            "final opener",
        )

        key = row_key(row)

        if key in seen:
            raise ValueError(
                "Final opener rows contain "
                f"duplicate key: {key}"
            )

        seen.add(key)


def write_csv_atomic(
    path: Path,
    rows: list[
        dict[str, str]
    ],
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
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_COLUMNS,
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        column: row.get(
                            column,
                            "",
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

    finally:
        try:
            temp_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass



def _count_blank_provider_timestamps(
    final_rows: list[dict[str, str]],
) -> int:
    return sum(
        1
        for row in final_rows
        if (
            row_has_required_opening(row)
            and not str(
                row.get(
                    "opening_timestamp",
                    "",
                )
            ).strip()
        )
    )


def _raise_if_opening_fetch_failed(
    hard_failures: list[dict[str, object]],
) -> None:
    if hard_failures:
        raise RuntimeError(
            "One or more ESPN "
            "opening-odds requests failed; "
            "refusing to modify opener history. "
            f"failures={len(hard_failures)}"
        )


def _add_existing_opener_output_input(
    report: PipelineReporter,
    output_path: Path,
) -> None:
    if output_path.exists():
        report.add_input(
            output_path
        )


def main() -> int:
    with PipelineReporter(
        script=__file__,
        stage="00_intake",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        extra_context={
            "script_version": (
                SCRIPT_VERSION
            ),
            "source": (
                "ESPN Core API"
            ),
        },
    ) as report:
        report.add_input(
            CURRENT_WEEK_CONFIG_PATH
        )

        report.set_detail(
            "output_modified",
            False,
        )

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

        weekly_path = (
            WEEKLY_DIR
            / (
                f"week_{week}_"
                "CFB_weekly_schedule.csv"
            )
        )

        report.add_input(
            weekly_path
        )

        weekly_rows = read_csv(
            weekly_path,
            WEEKLY_REQUIRED_COLUMNS,
            "weekly schedule CSV",
        )

        validate_weekly_rows(
            weekly_rows,
            season,
            season_type,
            week,
        )

        output_path = (
            OPENERS_DIR
            / f"{season}_CFB_openers.csv"
        )

        _add_existing_opener_output_input(
            report,
            output_path,
        )

        report.add_output(
            output_path
        )

        (
            existing_rows,
            legacy_schema,
        ) = read_existing_openers(
            output_path
        )

        captured_at = (
            utc_now()
            .isoformat()
        )

        (
            new_rows,
            hard_failures,
            provider_counts,
        ) = build_opening_rows(
            weekly_rows,
            captured_at,
        )

        report.set_rows(
            rows_in=len(
                weekly_rows
            ),
        )

        report.update_details(
            {
                "target_games": len(
                    weekly_rows
                ),
                "games_requested": len(
                    weekly_rows
                ),
                "hard_fetch_failures": len(
                    hard_failures
                ),
                "hard_fetch_failure_details": (
                    hard_failures
                ),
                "provider_game_counts": (
                    provider_counts
                ),
                "existing_rows": len(
                    existing_rows
                ),
                "legacy_schema_migrated": (
                    legacy_schema
                ),
                "new_rows_built": len(
                    new_rows
                ),
            }
        )

        _raise_if_opening_fetch_failed(
            hard_failures
        )

        validate_new_coverage(
            weekly_rows,
            new_rows,
        )

        (
            final_rows,
            preserved_count,
            captured_count,
            inserted_count,
        ) = upsert_rows(
            existing_rows,
            new_rows,
        )

        validate_final_rows(
            final_rows
        )

        ok_rows = sum(
            1
            for row in final_rows
            if row[
                "opener_status"
            ] == "ok"
        )

        missing_rows = sum(
            1
            for row in final_rows
            if row[
                "opener_status"
            ] == "missing"
        )

        error_rows = sum(
            1
            for row in final_rows
            if row[
                "opener_status"
            ] == "error"
        )

        blank_provider_timestamps = (
            _count_blank_provider_timestamps(
                final_rows
            )
        )

        blank_capture_provenance = sum(
            1
            for row in final_rows
            if row_has_required_opening(
                row
            )
            and not str(
                row.get(
                    "opening_captured_at",
                    "",
                )
            ).strip()
        )

        current_week_missing_rows = sum(
            1
            for row in new_rows
            if row[
                "opener_status"
            ] == "missing"
        )

        report.update_details(
            {
                "opening_rows_preserved": (
                    preserved_count
                ),
                "opening_rows_newly_captured": (
                    captured_count
                ),
                "rows_inserted": (
                    inserted_count
                ),
                "final_rows": len(
                    final_rows
                ),
                "final_ok_rows": (
                    ok_rows
                ),
                "final_missing_rows": (
                    missing_rows
                ),
                "final_error_rows": (
                    error_rows
                ),
                "current_week_missing_rows": (
                    current_week_missing_rows
                ),
                "blank_provider_timestamps_on_valid_openers": (
                    blank_provider_timestamps
                ),
                "blank_capture_provenance_on_valid_openers": (
                    blank_capture_provenance
                ),
                "output_modified": False,
            }
        )

        write_csv_atomic(
            output_path,
            final_rows,
        )

        report.set_rows(
            rows_out=len(
                final_rows
            ),
        )

        report.update_details(
            {
                "output_modified": True,
                "output_path": str(
                    output_path
                ),
            }
        )

        print(
            "pull_opening_odds.py completed"
        )

        print(
            f"season={season} "
            f"season_type={season_type} "
            f"week={week}"
        )

        print(
            f"target_games="
            f"{len(weekly_rows)}"
        )

        print(
            f"new_rows_built="
            f"{len(new_rows)}"
        )

        print(
            "opening_rows_preserved="
            f"{preserved_count}"
        )

        print(
            "opening_rows_newly_captured="
            f"{captured_count}"
        )

        print(
            f"final_rows="
            f"{len(final_rows)}"
        )

        print(
            f"final_ok_rows="
            f"{ok_rows}"
        )

        print(
            f"final_missing_rows="
            f"{missing_rows}"
        )

        print(
            f"output={output_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
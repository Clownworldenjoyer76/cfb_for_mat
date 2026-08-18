#!/usr/bin/env python3
# docs/win/football/cfb/scripts/00_intake/pull_opening_odds.py

import csv
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_DIR = Path("docs/win/football/cfb")
WEEKLY_DIR = BASE_DIR / "00_intake" / "schedule" / "weekly"
OPENERS_DIR = BASE_DIR / "00_intake" / "odds" / "openers"
ERROR_DIR = BASE_DIR / "errors" / "00_intake"

ERROR_DIR.mkdir(parents=True, exist_ok=True)
OPENERS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "pull_opening_odds.txt"

ESPN_BASE = (
    "https://sports.core.api.espn.com/v2/sports/football/"
    "leagues/college-football"
)

OUTPUT_COLUMNS = [
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

WEEKLY_REQUIRED_COLUMNS = [
    "season",
    "week",
    "game_id",
    "odds_provider_game_id",
    "away_team",
    "home_team",
    "bookmaker",
    "home_moneyline_american",
    "away_moneyline_american",
    "home_spread",
    "away_spread",
    "total",
    "odds_available",
]


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now_iso()}] {message}\n")


def fail(message):
    log(f"ERROR: {message}")
    raise RuntimeError(message)


def latest_file(directory, pattern, label):
    files = sorted(
        directory.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not files:
        fail(
            f"No {label} found in "
            f"{directory} matching {pattern}"
        )

    return files[0]


def read_csv(path, required_columns, label):
    if not path.exists():
        fail(f"Missing {label}: {path}")

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        missing = [
            column
            for column in required_columns
            if column not in fieldnames
        ]

        if missing:
            fail(
                f"{label} missing columns: "
                f"{missing}"
            )

        return list(reader)


def write_csv(path, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: row.get(column, "")
                    for column in OUTPUT_COLUMNS
                }
            )


def build_url(path, params=None):
    url = f"{ESPN_BASE}{path}"

    if params:
        return f"{url}?{urlencode(params)}"

    return url


def http_get_json(url):
    request = Request(
        url,
        headers={
            "User-Agent": "cfb-pull-opening-odds/3.0",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(
            request,
            timeout=45,
        ) as response:
            status = response.status
            body = response.read().decode("utf-8")

    except HTTPError as exc:
        body = ""

        try:
            body = exc.read().decode("utf-8")
        except Exception:
            pass

        return {
            "_request_failed": True,
            "_http_status": exc.code,
            "_error": body or str(exc),
        }

    except URLError as exc:
        return {
            "_request_failed": True,
            "_http_status": "",
            "_error": str(exc),
        }

    except Exception as exc:
        return {
            "_request_failed": True,
            "_http_status": "",
            "_error": str(exc),
        }

    if status < 200 or status >= 300:
        return {
            "_request_failed": True,
            "_http_status": status,
            "_error": body,
        }

    try:
        return json.loads(body)

    except Exception as exc:
        return {
            "_request_failed": True,
            "_http_status": status,
            "_error": f"JSON parse failed: {exc}",
        }


def to_float(value):
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.lower() in {
        "even",
        "ev",
        "evens",
    }:
        return 100.0

    try:
        return float(text)

    except Exception:
        return None


def clean_number(value):
    number = to_float(value)

    if number is None:
        return ""

    if number.is_integer():
        return str(int(number))

    return str(number)


def normalize_american(value):
    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    if text.lower() in {
        "even",
        "ev",
        "evens",
    }:
        return "100"

    number = to_float(text)

    if number is None:
        return ""

    return str(int(round(number)))


def numeric_movement(
    current_value,
    opening_value,
):
    current = to_float(current_value)
    opening = to_float(opening_value)

    if (
        current is None
        or opening is None
    ):
        return ""

    movement = current - opening

    if movement.is_integer():
        return str(int(movement))

    return str(
        round(
            movement,
            4,
        )
    )


def normalize_timestamp(value):
    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    number = to_float(text)

    if number is not None:
        try:
            if number > 1_000_000_000_000:
                dt = datetime.fromtimestamp(
                    number / 1000,
                    tz=timezone.utc,
                )
                return dt.isoformat()

            if number > 1_000_000_000:
                dt = datetime.fromtimestamp(
                    number,
                    tz=timezone.utc,
                )
                return dt.isoformat()

        except Exception:
            pass

    return text


def bookmaker_key(value):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value or "")
        .strip()
        .lower(),
    )


def canonical_bookmaker(value):
    text = str(
        value or ""
    ).strip()

    key = bookmaker_key(text)

    if key == "draftkings":
        return "DraftKings"

    if key == "fanduel":
        return "FanDuel"

    return text


def fetch_ref(ref):
    if not ref:
        return None

    if ref.startswith("http://"):
        ref = (
            "https://"
            + ref[len("http://"):]
        )

    response = http_get_json(ref)

    if (
        isinstance(response, dict)
        and response.get(
            "_request_failed"
        )
    ):
        log(
            "REF_FETCH_FAILED "
            f"status={response.get('_http_status', '')} "
            f"ref={ref} "
            f"error={response.get('_error', '')}"
        )

        return None

    return response


def provider_info(odds_item):
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
        resolved = fetch_ref(
            provider.get("$ref")
        )

        if isinstance(
            resolved,
            dict,
        ):
            provider_data = resolved

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
    collection,
):
    if not isinstance(
        collection,
        dict,
    ):
        return []

    items = collection.get(
        "items",
        [],
    )

    if not isinstance(
        items,
        list,
    ):
        return []

    resolved = []

    for item in items:
        if not isinstance(
            item,
            dict,
        ):
            continue

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
            fetched = fetch_ref(
                item.get("$ref")
            )

            if isinstance(
                fetched,
                dict,
            ):
                resolved.append(
                    fetched
                )

        else:
            resolved.append(item)

    return resolved


def select_primary_odds_item(
    items,
    bookmaker_name="",
):
    if not items:
        return None

    desired_key = bookmaker_key(
        bookmaker_name
    )

    if desired_key:
        for item in items:
            info = provider_info(item)

            if (
                bookmaker_key(
                    info["name"]
                )
                == desired_key
            ):
                return item

    ranked = []

    for index, item in enumerate(
        items
    ):
        info = provider_info(item)

        priority = info[
            "priority"
        ]

        rank = (
            priority
            if priority is not None
            else 1_000_000
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

    return ranked[0][2]


def fetch_current_odds(
    game_id,
    bookmaker_name="",
):
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

    response = http_get_json(url)

    if (
        isinstance(response, dict)
        and response.get(
            "_request_failed"
        )
    ):
        return (
            None,
            response.get(
                "_http_status",
                "",
            ),
            response.get(
                "_error",
                "",
            ),
        )

    items = resolve_collection_items(
        response
    )

    selected = select_primary_odds_item(
        items,
        bookmaker_name=bookmaker_name,
    )

    if selected is None:
        return (
            None,
            200,
            "no_current_odds_items",
        )

    return (
        selected,
        200,
        "",
    )


def market_block(
    parent,
    snapshot,
):
    if not isinstance(
        parent,
        dict,
    ):
        return {}

    block = parent.get(
        snapshot
    )

    if not isinstance(
        block,
        dict,
    ):
        return {}

    return block


def market_object(
    block,
    market,
):
    if not isinstance(
        block,
        dict,
    ):
        return {}

    obj = block.get(
        market
    )

    if not isinstance(
        obj,
        dict,
    ):
        return {}

    return obj


def market_value(
    block,
    market,
):
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
    block,
    market,
):
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
    )


def first_block_timestamp(
    block,
):
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
            and str(value).strip()
        ):
            return normalize_timestamp(
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
                return normalize_timestamp(
                    candidate
                )

    return ""


def infer_opening_favorite(
    home_open,
    away_open,
    home_moneyline,
    away_moneyline,
):
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
            home_ml < 0
            and away_ml > 0
        ):
            return "home"

        if (
            away_ml < 0
            and home_ml > 0
        ):
            return "away"

    home_favorite = (
        home_open.get("favorite")
        is True
    )

    away_favorite = (
        away_open.get("favorite")
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
    home_open,
    away_open,
    home_moneyline,
    away_moneyline,
):
    raw_home = market_value(
        home_open,
        "pointSpread",
    )

    raw_away = market_value(
        away_open,
        "pointSpread",
    )

    home_num = to_float(
        raw_home
    )

    away_num = to_float(
        raw_away
    )

    if (
        home_num is None
        and away_num is None
    ):
        return "", ""

    favorite = infer_opening_favorite(
        home_open,
        away_open,
        home_moneyline,
        away_moneyline,
    )

    if favorite:
        if home_num is not None:
            magnitude = abs(
                home_num
            )
        else:
            magnitude = abs(
                away_num
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
    odds_item,
):
    if not isinstance(
        odds_item,
        dict,
    ):
        return {}

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

    opening_timestamp = (
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
        "timestamp": (
            opening_timestamp
        ),
    }


def status_fields(
    status,
    reason="",
    http_status="",
):
    return {
        "opener_status": status,
        "opener_missing_reason": reason,
        "opener_http_status": str(
            http_status or ""
        ),
    }


def market_status(
    value,
    http_status,
    request_error,
    missing_reason,
):
    if request_error:
        status = (
            "missing"
            if str(http_status) == "404"
            else "error"
        )

        return status_fields(
            status,
            request_error,
            http_status,
        )

    if (
        str(
            value or ""
        ).strip()
        == ""
    ):
        return status_fields(
            "missing",
            missing_reason,
            http_status,
        )

    return status_fields(
        "ok",
        "",
        http_status,
    )


def base_row(
    weekly_row,
    market_type,
    bet_side,
    bookmaker,
):
    return {
        "game_id": weekly_row.get(
            "game_id",
            "",
        ),
        "odds_provider_game_id": (
            weekly_row.get(
                "odds_provider_game_id",
                "",
            )
            or weekly_row.get(
                "game_id",
                "",
            )
        ),
        "market_type": (
            market_type
        ),
        "bet_side": (
            bet_side
        ),
        "opening_line": "",
        "opening_odds_american": "",
        "opening_timestamp": "",
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


def add_h2h_rows(
    rows,
    weekly_row,
    opening,
    bookmaker,
    http_status,
    request_error,
):
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

        status = market_status(
            opening_value,
            http_status,
            request_error,
            (
                f"opening_{side}_"
                "moneyline_missing"
            ),
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
                **status,
            }
        )

        rows.append(row)


def add_spread_rows(
    rows,
    weekly_row,
    opening,
    bookmaker,
    http_status,
    request_error,
):
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

        status = market_status(
            opening_spread,
            http_status,
            request_error,
            (
                f"opening_{side}_"
                "spread_missing"
            ),
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
                **status,
            }
        )

        rows.append(row)


def add_total_rows(
    rows,
    weekly_row,
    opening,
    bookmaker,
    http_status,
    request_error,
):
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

        status = market_status(
            opening_total,
            http_status,
            request_error,
            "opening_total_missing",
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
                **status,
            }
        )

        rows.append(row)


def build_opening_rows(
    weekly_rows,
):
    output_rows = []

    for weekly_row in weekly_rows:
        if (
            str(
                weekly_row.get(
                    "odds_available",
                    "",
                )
            ).strip()
            != "1"
        ):
            continue

        game_id = str(
            weekly_row.get(
                "game_id",
                "",
            )
        ).strip()

        if not game_id:
            log(
                "SKIP_ROW_BLANK_GAME_ID"
            )
            continue

        weekly_bookmaker = (
            canonical_bookmaker(
                weekly_row.get(
                    "bookmaker",
                    "",
                )
            )
        )

        (
            odds_item,
            http_status,
            request_error,
        ) = fetch_current_odds(
            game_id,
            bookmaker_name=(
                weekly_bookmaker
            ),
        )

        if odds_item is None:
            opening = {}
            bookmaker = (
                weekly_bookmaker
            )

        else:
            opening = get_opening(
                odds_item
            )

            provider = provider_info(
                odds_item
            )

            bookmaker = (
                canonical_bookmaker(
                    provider["name"]
                    or weekly_bookmaker
                    or provider["id"]
                )
            )

        add_h2h_rows(
            output_rows,
            weekly_row,
            opening,
            bookmaker,
            http_status,
            request_error,
        )

        add_spread_rows(
            output_rows,
            weekly_row,
            opening,
            bookmaker,
            http_status,
            request_error,
        )

        add_total_rows(
            output_rows,
            weekly_row,
            opening,
            bookmaker,
            http_status,
            request_error,
        )

    return output_rows


def read_existing_openers(
    path,
):
    if not path.exists():
        return []

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)

        fieldnames = (
            reader.fieldnames or []
        )

        existing_rows = []

        for row in reader:
            normalized = {
                column: row.get(
                    column,
                    "",
                )
                for column in (
                    OUTPUT_COLUMNS
                )
            }

            normalized[
                "bookmaker"
            ] = canonical_bookmaker(
                normalized.get(
                    "bookmaker",
                    "",
                )
            )

            existing_rows.append(
                normalized
            )

        missing = [
            column
            for column in OUTPUT_COLUMNS
            if column not in fieldnames
        ]

        if missing:
            log(
                "Existing opener file "
                "missing columns; "
                "blanks inserted: "
                f"{missing}"
            )

        return existing_rows


def row_has_required_opening(
    row,
):
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


def row_status_rank(
    row,
):
    status = str(
        row.get(
            "opener_status",
            "",
        )
    ).strip()

    has_required_opening = (
        row_has_required_opening(
            row
        )
    )

    if (
        status == "ok"
        and has_required_opening
    ):
        return 3

    if has_required_opening:
        return 2

    if status == "missing":
        return 1

    return 0


def row_key(
    row,
):
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


def upsert_rows(
    existing_rows,
    new_rows,
):
    keyed = {}

    for row in existing_rows:
        row[
            "bookmaker"
        ] = canonical_bookmaker(
            row.get(
                "bookmaker",
                "",
            )
        )

        key = row_key(row)

        existing = keyed.get(
            key
        )

        if (
            existing is None
            or row_status_rank(row)
            > row_status_rank(
                existing
            )
        ):
            keyed[key] = row

    for row in new_rows:
        row[
            "bookmaker"
        ] = canonical_bookmaker(
            row.get(
                "bookmaker",
                "",
            )
        )

        key = row_key(row)

        existing = keyed.get(
            key
        )

        if existing is None:
            keyed[key] = row
            continue

        if (
            row_status_rank(row)
            >= row_status_rank(
                existing
            )
        ):
            keyed[key] = row

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

    return rows


def detect_season(
    weekly_rows,
):
    seasons = sorted(
        {
            str(
                row.get(
                    "season",
                    "",
                )
            ).strip()
            for row in weekly_rows
            if str(
                row.get(
                    "season",
                    "",
                )
            ).strip()
        }
    )

    if len(seasons) != 1:
        fail(
            "Expected exactly one season "
            "in weekly schedule, found: "
            f"{seasons}"
        )

    return seasons[0]


def main():
    LOG_FILE.write_text(
        "",
        encoding="utf-8",
    )

    weekly_path = latest_file(
        WEEKLY_DIR,
        "week_*_CFB_weekly_schedule.csv",
        "weekly schedule CSV",
    )

    log(
        f"Weekly schedule input: "
        f"{weekly_path}"
    )

    weekly_rows = read_csv(
        weekly_path,
        WEEKLY_REQUIRED_COLUMNS,
        "weekly schedule CSV",
    )

    season = detect_season(
        weekly_rows
    )

    output_path = (
        OPENERS_DIR
        / f"{season}_CFB_openers.csv"
    )

    existing_rows = (
        read_existing_openers(
            output_path
        )
    )

    new_rows = build_opening_rows(
        weekly_rows
    )

    final_rows = upsert_rows(
        existing_rows,
        new_rows,
    )

    write_csv(
        output_path,
        final_rows,
    )

    ok_rows = sum(
        1
        for row in final_rows
        if str(
            row.get(
                "opener_status",
                "",
            )
        ).strip()
        == "ok"
    )

    missing_rows = sum(
        1
        for row in final_rows
        if str(
            row.get(
                "opener_status",
                "",
            )
        ).strip()
        == "missing"
    )

    error_rows = sum(
        1
        for row in final_rows
        if str(
            row.get(
                "opener_status",
                "",
            )
        ).strip()
        == "error"
    )

    log(
        f"Weekly rows loaded: "
        f"{len(weekly_rows)}"
    )
    log(
        f"Existing opener rows loaded: "
        f"{len(existing_rows)}"
    )
    log(
        f"New opener rows built: "
        f"{len(new_rows)}"
    )
    log(
        f"Final opener rows written: "
        f"{len(final_rows)}"
    )
    log(
        f"Final opener ok rows: "
        f"{ok_rows}"
    )
    log(
        f"Final opener missing rows: "
        f"{missing_rows}"
    )
    log(
        f"Final opener error rows: "
        f"{error_rows}"
    )
    log(
        f"Output written: "
        f"{output_path}"
    )

    print(
        f"Opening odds written: "
        f"{output_path}"
    )
    print(
        f"Weekly rows loaded: "
        f"{len(weekly_rows)}"
    )
    print(
        f"Existing opener rows loaded: "
        f"{len(existing_rows)}"
    )
    print(
        f"New opener rows built: "
        f"{len(new_rows)}"
    )
    print(
        f"Final opener rows written: "
        f"{len(final_rows)}"
    )
    print(
        f"Final opener ok rows: "
        f"{ok_rows}"
    )
    print(
        f"Final opener missing rows: "
        f"{missing_rows}"
    )
    print(
        f"Final opener error rows: "
        f"{error_rows}"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception:
        log(
            traceback.format_exc()
        )

        print(
            f"ERROR: see {LOG_FILE}",
            file=sys.stderr,
        )

        raise

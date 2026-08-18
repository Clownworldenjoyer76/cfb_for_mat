#!/usr/bin/env python3
# docs/win/football/cfb/scripts/00_intake/pull_opening_odds.py

import csv
import json
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
            "User-Agent": "cfb-pull-opening-odds/2.0",
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

    return str(round(movement, 4))


def fetch_ref(ref):
    if not ref:
        return None

    if ref.startswith("http://"):
        ref = "https://" + ref[len("http://"):]

    response = http_get_json(ref)

    if (
        isinstance(response, dict)
        and response.get("_request_failed")
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
    provider = odds_item.get("provider")

    if not isinstance(provider, dict):
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
        )
    ):
        resolved = fetch_ref(
            provider.get("$ref")
        )

        if isinstance(resolved, dict):
            provider_data = resolved

    return {
        "id": str(
            provider_data.get(
                "id",
                "",
            )
        ).strip(),
        "name": str(
            provider_data.get("name")
            or provider_data.get(
                "displayName"
            )
            or provider_data.get(
                "shortName"
            )
            or ""
        ).strip(),
        "priority": to_float(
            provider_data.get(
                "priority"
            )
        ),
    }


def resolve_collection_items(collection):
    if not isinstance(collection, dict):
        return []

    items = collection.get(
        "items",
        [],
    )

    if not isinstance(items, list):
        return []

    resolved = []

    for item in items:
        if not isinstance(item, dict):
            continue

        if (
            item.get("$ref")
            and not (
                item.get("provider")
                or item.get("homeTeamOdds")
                or item.get("awayTeamOdds")
                or item.get("open")
                or item.get("current")
                or item.get("overUnder") is not None
            )
        ):
            fetched = fetch_ref(
                item.get("$ref")
            )

            if isinstance(fetched, dict):
                resolved.append(fetched)

        else:
            resolved.append(item)

    return resolved


def select_primary_odds_item(
    items,
    bookmaker_name="",
):
    if not items:
        return None

    desired = str(
        bookmaker_name or ""
    ).strip().lower()

    if desired:
        for item in items:
            info = provider_info(item)

            if (
                info["name"]
                .strip()
                .lower()
                == desired
            ):
                return item

    ranked = []

    for index, item in enumerate(items):
        info = provider_info(item)

        priority = info["priority"]

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
        and response.get("_request_failed")
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
    if not isinstance(parent, dict):
        return {}

    block = parent.get(snapshot)

    if not isinstance(block, dict):
        return {}

    return block


def market_value(
    block,
    market,
):
    if not isinstance(block, dict):
        return ""

    obj = block.get(market)

    if not isinstance(obj, dict):
        return ""

    if market in {
        "pointSpread",
        "total",
    }:
        return clean_number(
            obj.get("american")
            or obj.get(
                "alternateDisplayValue"
            )
        )

    return clean_number(
        obj.get("value")
    )


def market_american(
    block,
    market,
):
    if not isinstance(block, dict):
        return ""

    obj = block.get(market)

    if not isinstance(obj, dict):
        return ""

    return normalize_american(
        obj.get("american")
        or obj.get(
            "alternateDisplayValue"
        )
    )


def get_opening(odds_item):
    if not isinstance(odds_item, dict):
        return {}

    game_open = market_block(
        odds_item,
        "open",
    )

    home_team = (
        odds_item.get("homeTeamOdds")
        if isinstance(
            odds_item.get("homeTeamOdds"),
            dict,
        )
        else {}
    )

    away_team = (
        odds_item.get("awayTeamOdds")
        if isinstance(
            odds_item.get("awayTeamOdds"),
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

    return {
        "home_moneyline": market_american(
            home_open,
            "moneyLine",
        ),
        "away_moneyline": market_american(
            away_open,
            "moneyLine",
        ),
        "home_spread": market_value(
            home_open,
            "pointSpread",
        ),
        "away_spread": market_value(
            away_open,
            "pointSpread",
        ),
        "home_spread_odds": market_american(
            home_open,
            "spread",
        ),
        "away_spread_odds": market_american(
            away_open,
            "spread",
        ),
        "total": market_value(
            game_open,
            "total",
        ),
        "over_odds": market_american(
            game_open,
            "over",
        ),
        "under_odds": market_american(
            game_open,
            "under",
        ),
        "timestamp": "",
    }


def has_opening(opening):
    if not isinstance(opening, dict):
        return False

    return any(
        str(
            opening.get(
                field,
                "",
            )
        ).strip()
        for field in [
            "home_moneyline",
            "away_moneyline",
            "home_spread",
            "away_spread",
            "home_spread_odds",
            "away_spread_odds",
            "total",
            "over_odds",
            "under_odds",
        ]
    )


def response_status(
    opening,
    http_status,
    error="",
):
    if error:
        status = (
            "missing"
            if str(http_status) == "404"
            else "error"
        )

        return {
            "opener_status": status,
            "opener_missing_reason": error,
            "opener_http_status": str(
                http_status or ""
            ),
        }

    if not has_opening(opening):
        return {
            "opener_status": "missing",
            "opener_missing_reason": (
                "no_opening_data"
            ),
            "opener_http_status": str(
                http_status or ""
            ),
        }

    return {
        "opener_status": "ok",
        "opener_missing_reason": "",
        "opener_http_status": str(
            http_status or ""
        ),
    }


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
        "market_type": market_type,
        "bet_side": bet_side,
        "opening_line": "",
        "opening_odds_american": "",
        "opening_timestamp": "",
        "bookmaker": bookmaker,
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
    status,
):
    for side in (
        "home",
        "away",
    ):
        opening_value = opening.get(
            f"{side}_moneyline",
            "",
        )

        current_value = str(
            weekly_row.get(
                f"{side}_moneyline_american",
                "",
            )
        ).strip()

        row = base_row(
            weekly_row,
            "h2h",
            side,
            bookmaker,
        )

        row.update(
            {
                "opening_line": "",
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
    status,
):
    for side in (
        "home",
        "away",
    ):
        opening_spread = opening.get(
            f"{side}_spread",
            "",
        )

        opening_odds = opening.get(
            f"{side}_spread_odds",
            "",
        )

        current_spread = str(
            weekly_row.get(
                f"{side}_spread",
                "",
            )
        ).strip()

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
    status,
):
    opening_total = opening.get(
        "total",
        "",
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
        opening_odds = opening.get(
            f"{side}_odds",
            "",
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

        weekly_bookmaker = str(
            weekly_row.get(
                "bookmaker",
                "",
            )
        ).strip()

        (
            odds_item,
            http_status,
            error,
        ) = fetch_current_odds(
            game_id,
            bookmaker_name=weekly_bookmaker,
        )

        if odds_item is None:
            opening = {}
            bookmaker = weekly_bookmaker

        else:
            opening = get_opening(
                odds_item
            )

            provider = provider_info(
                odds_item
            )

            bookmaker = (
                provider["name"]
                or weekly_bookmaker
                or provider["id"]
            )

        status = response_status(
            opening,
            http_status,
            error,
        )

        add_h2h_rows(
            output_rows,
            weekly_row,
            opening,
            bookmaker,
            status,
        )

        add_spread_rows(
            output_rows,
            weekly_row,
            opening,
            bookmaker,
            status,
        )

        add_total_rows(
            output_rows,
            weekly_row,
            opening,
            bookmaker,
            status,
        )

    return output_rows


def read_existing_openers(path):
    if not path.exists():
        return []

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        existing_rows = []

        for row in reader:
            normalized = {}

            for column in OUTPUT_COLUMNS:
                normalized[column] = row.get(
                    column,
                    "",
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
                "missing columns; blanks inserted: "
                f"{missing}"
            )

        return existing_rows


def row_has_opening_data(row):
    return any(
        str(
            row.get(
                column,
                "",
            )
        ).strip()
        for column in [
            "opening_line",
            "opening_odds_american",
            "opening_timestamp",
            "opening_spread",
            "opening_total",
            "opening_moneyline",
        ]
    )


def row_status_rank(row):
    status = str(
        row.get(
            "opener_status",
            "",
        )
    ).strip()

    if status == "ok":
        return 3

    if row_has_opening_data(row):
        return 2

    if status == "missing":
        return 1

    return 0


def upsert_rows(
    existing_rows,
    new_rows,
):
    keyed = {}

    for row in existing_rows:
        key = (
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
            str(
                row.get(
                    "bookmaker",
                    "",
                )
            ).strip(),
        )

        keyed[key] = row

    for row in new_rows:
        key = (
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
            str(
                row.get(
                    "bookmaker",
                    "",
                )
            ).strip(),
        )

        existing = keyed.get(key)

        if existing is None:
            keyed[key] = row
            continue

        if (
            row_status_rank(row)
            >= row_status_rank(existing)
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
            row.get(
                "bookmaker",
                "",
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

#!/usr/bin/env python3
# docs/win/football/nfl/scripts/00_intake/pull_opening_odds.py

import csv
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_DIR = Path("docs/win/football/nfl")

WEEKLY_DIR = BASE_DIR / "00_intake" / "schedule" / "weekly"
OPENERS_DIR = BASE_DIR / "00_intake" / "odds" / "openers"

ERROR_DIR = BASE_DIR / "errors" / "00_intake"
ERROR_DIR.mkdir(parents=True, exist_ok=True)
OPENERS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "pull_opening_odds.txt"

API_KEY_ENV = "API_ODDS"
API_BASE = "https://api.odds-api.io/v3"

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

MARKET_API_NAMES = {
    "h2h": "ML",
    "spreads": "Spread",
    "totals": "Totals",
}


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now_iso()}] {message}\n")


def fail(message):
    log(f"ERROR: {message}")
    raise RuntimeError(message)


def latest_file(directory, pattern, label):
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if not files:
        fail(f"No {label} found in {directory} matching {pattern}")

    return files[0]


def read_csv(path, required_columns, label):
    if not path.exists():
        fail(f"Missing {label}: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            fail(f"{label} missing columns: {missing}")

        return list(reader)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def build_url(path, params):
    return f"{API_BASE}{path}?{urlencode(params)}"


def http_get_json(url):
    request = Request(url, headers={"User-Agent": "nfl-pull-opening-odds/1.0"})

    try:
        with urlopen(request, timeout=45) as response:
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

    if text.lower() in {"even", "ev", "evens"}:
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


def decimal_to_american(value):
    decimal = to_float(value)

    if decimal is None:
        return ""

    if decimal <= 1:
        return ""

    if decimal >= 2:
        american = (decimal - 1) * 100
    else:
        american = -100 / (decimal - 1)

    return str(int(round(american)))


def normalize_odds_to_american(value):
    raw = unwrap_value(value)

    if raw is None:
        return ""

    text = str(raw).strip()

    if not text:
        return ""

    if text.lower() in {"even", "ev", "evens"}:
        return "100"

    number = to_float(text)

    if number is None:
        return ""

    if number < 0:
        return str(int(round(number)))

    if number >= 100:
        return str(int(round(number)))

    return decimal_to_american(number)


def numeric_movement(current_value, opening_value):
    current = to_float(current_value)
    opening = to_float(opening_value)

    if current is None or opening is None:
        return ""

    movement = current - opening

    if movement.is_integer():
        return str(int(movement))

    return str(round(movement, 4))


def same_number(left, right):
    left_num = to_float(left)
    right_num = to_float(right)

    if left_num is None or right_num is None:
        return False

    return abs(left_num - right_num) < 0.0001


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
                dt = datetime.fromtimestamp(number / 1000, tz=timezone.utc)
                return dt.isoformat()
            if number > 1_000_000_000:
                dt = datetime.fromtimestamp(number, tz=timezone.utc)
                return dt.isoformat()
        except Exception:
            pass

    return text


def unwrap_value(value):
    if isinstance(value, dict):
        for key in [
            "american",
            "odds_american",
            "price_american",
            "price",
            "odds",
            "decimal",
            "odds_decimal",
            "value",
        ]:
            if key in value:
                return unwrap_value(value.get(key))

        return ""

    if isinstance(value, list):
        if not value:
            return ""

        return unwrap_value(value[0])

    return value


def first_existing_value(data, keys):
    if not isinstance(data, dict):
        return ""

    for key in keys:
        if key in data:
            value = unwrap_value(data.get(key))

            if str(value).strip() != "":
                return value

    return ""


def movement_request(api_key, event_id, bookmaker, market, market_line):
    params = {
        "apiKey": api_key,
        "eventId": event_id,
        "bookmaker": bookmaker,
        "market": market,
    }

    if market_line != "":
        params["marketLine"] = market_line

    url = build_url("/odds/movements", params)
    response = http_get_json(url)

    if isinstance(response, dict) and response.get("_request_failed"):
        log(
            "MOVEMENT_REQUEST_FAILED "
            f"eventId={event_id} "
            f"bookmaker={bookmaker} "
            f"market={market} "
            f"marketLine={market_line} "
            f"status={response.get('_http_status', '')} "
            f"error={response.get('_error', '')}"
        )

    return response


def get_opening(response):
    if not isinstance(response, dict):
        return {}

    opening = response.get("opening")

    if isinstance(opening, dict):
        return opening

    return {}


def has_opening(response):
    return bool(get_opening(response))


def response_status(response):
    if not isinstance(response, dict):
        return {
            "opener_status": "missing",
            "opener_missing_reason": "invalid_response",
            "opener_http_status": "",
        }

    if response.get("_request_failed"):
        http_status = str(response.get("_http_status", "")).strip()
        error = str(response.get("_error", "")).strip()

        if http_status == "404":
            status = "missing"
        else:
            status = "error"

        return {
            "opener_status": status,
            "opener_missing_reason": error,
            "opener_http_status": http_status,
        }

    if not has_opening(response):
        return {
            "opener_status": "missing",
            "opener_missing_reason": "no_opening_data",
            "opener_http_status": "",
        }

    return {
        "opener_status": "ok",
        "opener_missing_reason": "",
        "opener_http_status": "",
    }


def unique_lines(lines):
    seen = set()
    output = []

    for line in lines:
        text = str(line or "").strip()

        if text in seen:
            continue

        seen.add(text)
        output.append(text)

    return output


def movement_request_with_candidates(api_key, event_id, bookmaker, market, market_lines):
    responses = []

    for market_line in unique_lines(market_lines):
        response = movement_request(
            api_key=api_key,
            event_id=event_id,
            bookmaker=bookmaker,
            market=market,
            market_line=market_line,
        )

        responses.append((response, market_line))

        if has_opening(response):
            return response, market_line

    if responses:
        return responses[-1]

    return {}, ""


def h2h_opening_odds(opening, side):
    if side == "home":
        return normalize_odds_to_american(
            first_existing_value(
                opening,
                [
                    "home",
                    "Home",
                    "home_odds",
                    "homeOdds",
                    "home_price",
                    "homePrice",
                ],
            )
        )

    return normalize_odds_to_american(
        first_existing_value(
            opening,
            [
                "away",
                "Away",
                "away_odds",
                "awayOdds",
                "away_price",
                "awayPrice",
            ],
        )
    )


def spread_opening_odds(opening, side):
    if side == "home":
        return normalize_odds_to_american(
            first_existing_value(
                opening,
                [
                    "home",
                    "Home",
                    "home_odds",
                    "homeOdds",
                    "home_price",
                    "homePrice",
                ],
            )
        )

    return normalize_odds_to_american(
        first_existing_value(
            opening,
            [
                "away",
                "Away",
                "away_odds",
                "awayOdds",
                "away_price",
                "awayPrice",
            ],
        )
    )


def total_opening_odds(opening, side):
    if side == "over":
        return normalize_odds_to_american(
            first_existing_value(
                opening,
                [
                    "over",
                    "Over",
                    "over_odds",
                    "overOdds",
                    "over_price",
                    "overPrice",
                    "o",
                    "O",
                    "home",
                    "Home",
                ],
            )
        )

    return normalize_odds_to_american(
        first_existing_value(
            opening,
            [
                "under",
                "Under",
                "under_odds",
                "underOdds",
                "under_price",
                "underPrice",
                "u",
                "U",
                "away",
                "Away",
            ],
        )
    )


def opening_hdp(opening):
    return clean_number(
        first_existing_value(
            opening,
            [
                "hdp",
                "line",
                "marketLine",
                "market_line",
                "spread",
                "total",
                "points",
            ],
        )
    )


def add_status_fields(row, status):
    row["opener_status"] = status.get("opener_status", "")
    row["opener_missing_reason"] = status.get("opener_missing_reason", "")
    row["opener_http_status"] = status.get("opener_http_status", "")
    return row


def add_h2h_rows(rows, weekly_row, response):
    opening = get_opening(response)
    status = response_status(response)

    opening_timestamp = normalize_timestamp(opening.get("timestamp"))

    home_open = h2h_opening_odds(opening, "home")
    away_open = h2h_opening_odds(opening, "away")

    home_current = str(weekly_row.get("home_moneyline_american", "")).strip()
    away_current = str(weekly_row.get("away_moneyline_american", "")).strip()

    rows.append(
        add_status_fields(
            {
                "game_id": weekly_row.get("game_id", ""),
                "odds_provider_game_id": weekly_row.get("odds_provider_game_id", ""),
                "market_type": "h2h",
                "bet_side": "home",
                "opening_line": "",
                "opening_odds_american": home_open,
                "opening_timestamp": opening_timestamp,
                "bookmaker": weekly_row.get("bookmaker", ""),
                "opening_spread": "",
                "current_spread": "",
                "spread_movement": "",
                "opening_total": "",
                "current_total": "",
                "total_movement": "",
                "opening_moneyline": home_open,
                "current_moneyline": home_current,
                "moneyline_movement": numeric_movement(home_current, home_open),
            },
            status,
        )
    )

    rows.append(
        add_status_fields(
            {
                "game_id": weekly_row.get("game_id", ""),
                "odds_provider_game_id": weekly_row.get("odds_provider_game_id", ""),
                "market_type": "h2h",
                "bet_side": "away",
                "opening_line": "",
                "opening_odds_american": away_open,
                "opening_timestamp": opening_timestamp,
                "bookmaker": weekly_row.get("bookmaker", ""),
                "opening_spread": "",
                "current_spread": "",
                "spread_movement": "",
                "opening_total": "",
                "current_total": "",
                "total_movement": "",
                "opening_moneyline": away_open,
                "current_moneyline": away_current,
                "moneyline_movement": numeric_movement(away_current, away_open),
            },
            status,
        )
    )


def spread_lines_from_opening(opening, weekly_row, market_line_used):
    opening_line = opening_hdp(opening)

    if opening_line == "":
        return "", ""

    home_spread = str(weekly_row.get("home_spread", "")).strip()
    away_spread = str(weekly_row.get("away_spread", "")).strip()

    if same_number(market_line_used, away_spread):
        away_opening_spread = opening_line
        opening_number = to_float(opening_line)

        if opening_number is None:
            return "", away_opening_spread

        home_opening_spread = clean_number(-opening_number)
        return home_opening_spread, away_opening_spread

    home_opening_spread = opening_line
    opening_number = to_float(opening_line)

    if opening_number is None:
        return home_opening_spread, ""

    away_opening_spread = clean_number(-opening_number)
    return home_opening_spread, away_opening_spread


def add_spread_rows(rows, weekly_row, response, market_line_used):
    opening = get_opening(response)
    status = response_status(response)

    opening_timestamp = normalize_timestamp(opening.get("timestamp"))

    opening_home_spread, opening_away_spread = spread_lines_from_opening(opening, weekly_row, market_line_used)

    home_open_odds = spread_opening_odds(opening, "home")
    away_open_odds = spread_opening_odds(opening, "away")

    home_current_spread = str(weekly_row.get("home_spread", "")).strip()
    away_current_spread = str(weekly_row.get("away_spread", "")).strip()

    rows.append(
        add_status_fields(
            {
                "game_id": weekly_row.get("game_id", ""),
                "odds_provider_game_id": weekly_row.get("odds_provider_game_id", ""),
                "market_type": "spreads",
                "bet_side": "home",
                "opening_line": opening_home_spread,
                "opening_odds_american": home_open_odds,
                "opening_timestamp": opening_timestamp,
                "bookmaker": weekly_row.get("bookmaker", ""),
                "opening_spread": opening_home_spread,
                "current_spread": home_current_spread,
                "spread_movement": numeric_movement(home_current_spread, opening_home_spread),
                "opening_total": "",
                "current_total": "",
                "total_movement": "",
                "opening_moneyline": "",
                "current_moneyline": "",
                "moneyline_movement": "",
            },
            status,
        )
    )

    rows.append(
        add_status_fields(
            {
                "game_id": weekly_row.get("game_id", ""),
                "odds_provider_game_id": weekly_row.get("odds_provider_game_id", ""),
                "market_type": "spreads",
                "bet_side": "away",
                "opening_line": opening_away_spread,
                "opening_odds_american": away_open_odds,
                "opening_timestamp": opening_timestamp,
                "bookmaker": weekly_row.get("bookmaker", ""),
                "opening_spread": opening_away_spread,
                "current_spread": away_current_spread,
                "spread_movement": numeric_movement(away_current_spread, opening_away_spread),
                "opening_total": "",
                "current_total": "",
                "total_movement": "",
                "opening_moneyline": "",
                "current_moneyline": "",
                "moneyline_movement": "",
            },
            status,
        )
    )


def add_total_rows(rows, weekly_row, response):
    opening = get_opening(response)
    status = response_status(response)

    opening_timestamp = normalize_timestamp(opening.get("timestamp"))

    opening_total = opening_hdp(opening)
    current_total = str(weekly_row.get("total", "")).strip()

    over_open_odds = total_opening_odds(opening, "over")
    under_open_odds = total_opening_odds(opening, "under")

    rows.append(
        add_status_fields(
            {
                "game_id": weekly_row.get("game_id", ""),
                "odds_provider_game_id": weekly_row.get("odds_provider_game_id", ""),
                "market_type": "totals",
                "bet_side": "over",
                "opening_line": opening_total,
                "opening_odds_american": over_open_odds,
                "opening_timestamp": opening_timestamp,
                "bookmaker": weekly_row.get("bookmaker", ""),
                "opening_spread": "",
                "current_spread": "",
                "spread_movement": "",
                "opening_total": opening_total,
                "current_total": current_total,
                "total_movement": numeric_movement(current_total, opening_total),
                "opening_moneyline": "",
                "current_moneyline": "",
                "moneyline_movement": "",
            },
            status,
        )
    )

    rows.append(
        add_status_fields(
            {
                "game_id": weekly_row.get("game_id", ""),
                "odds_provider_game_id": weekly_row.get("odds_provider_game_id", ""),
                "market_type": "totals",
                "bet_side": "under",
                "opening_line": opening_total,
                "opening_odds_american": under_open_odds,
                "opening_timestamp": opening_timestamp,
                "bookmaker": weekly_row.get("bookmaker", ""),
                "opening_spread": "",
                "current_spread": "",
                "spread_movement": "",
                "opening_total": opening_total,
                "current_total": current_total,
                "total_movement": numeric_movement(current_total, opening_total),
                "opening_moneyline": "",
                "current_moneyline": "",
                "moneyline_movement": "",
            },
            status,
        )
    )


def build_opening_rows(api_key, weekly_rows):
    output_rows = []

    for row in weekly_rows:
        if str(row.get("odds_available", "")).strip() != "1":
            continue

        event_id = str(row.get("odds_provider_game_id", "")).strip()
        bookmaker = str(row.get("bookmaker", "")).strip()

        if not event_id or not bookmaker:
            log(
                "SKIP_ROW_MISSING_EVENT_OR_BOOKMAKER "
                f"game_id={row.get('game_id', '')} "
                f"event_id={event_id} "
                f"bookmaker={bookmaker}"
            )
            continue

        h2h_response, _ = movement_request_with_candidates(
            api_key=api_key,
            event_id=event_id,
            bookmaker=bookmaker,
            market=MARKET_API_NAMES["h2h"],
            market_lines=[""],
        )
        add_h2h_rows(output_rows, row, h2h_response)

        home_spread = str(row.get("home_spread", "")).strip()
        away_spread = str(row.get("away_spread", "")).strip()

        if home_spread or away_spread:
            spread_response, spread_line_used = movement_request_with_candidates(
                api_key=api_key,
                event_id=event_id,
                bookmaker=bookmaker,
                market=MARKET_API_NAMES["spreads"],
                market_lines=[home_spread, away_spread],
            )
            add_spread_rows(output_rows, row, spread_response, spread_line_used)
        else:
            log(
                "SPREAD_ROW_MISSING_CURRENT_SPREAD "
                f"game_id={row.get('game_id', '')} "
                f"event_id={event_id} "
                f"bookmaker={bookmaker}"
            )
            add_spread_rows(output_rows, row, {}, "")

        total = str(row.get("total", "")).strip()

        if total:
            total_response, _ = movement_request_with_candidates(
                api_key=api_key,
                event_id=event_id,
                bookmaker=bookmaker,
                market=MARKET_API_NAMES["totals"],
                market_lines=[total],
            )
            add_total_rows(output_rows, row, total_response)
        else:
            log(
                "TOTAL_ROW_MISSING_CURRENT_TOTAL "
                f"game_id={row.get('game_id', '')} "
                f"event_id={event_id} "
                f"bookmaker={bookmaker}"
            )
            add_total_rows(output_rows, row, {})

    return output_rows


def read_existing_openers(path):
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        existing_rows = []

        for row in reader:
            normalized = {}

            for column in OUTPUT_COLUMNS:
                normalized[column] = row.get(column, "")

            existing_rows.append(normalized)

        missing = [column for column in OUTPUT_COLUMNS if column not in fieldnames]
        if missing:
            log(f"Existing opener file missing columns; blanks inserted: {missing}")

        return existing_rows


def row_has_opening_data(row):
    return any(
        str(row.get(column, "")).strip()
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
    status = str(row.get("opener_status", "")).strip()

    if status == "ok":
        return 3

    if row_has_opening_data(row):
        return 2

    if status == "missing":
        return 1

    return 0


def upsert_rows(existing_rows, new_rows):
    keyed = {}

    for row in existing_rows:
        key = (
            str(row.get("game_id", "")).strip(),
            str(row.get("market_type", "")).strip(),
            str(row.get("bet_side", "")).strip(),
            str(row.get("bookmaker", "")).strip(),
        )
        keyed[key] = row

    for row in new_rows:
        key = (
            str(row.get("game_id", "")).strip(),
            str(row.get("market_type", "")).strip(),
            str(row.get("bet_side", "")).strip(),
            str(row.get("bookmaker", "")).strip(),
        )

        existing = keyed.get(key)

        if existing is None:
            keyed[key] = row
            continue

        if row_status_rank(row) >= row_status_rank(existing):
            keyed[key] = row

    rows = list(keyed.values())

    rows.sort(
        key=lambda row: (
            row.get("game_id", ""),
            row.get("market_type", ""),
            row.get("bet_side", ""),
            row.get("bookmaker", ""),
        )
    )

    return rows


def detect_season(weekly_rows):
    seasons = sorted({str(row.get("season", "")).strip() for row in weekly_rows if str(row.get("season", "")).strip()})

    if len(seasons) != 1:
        fail(f"Expected exactly one season in weekly schedule, found: {seasons}")

    return seasons[0]


def main():
    LOG_FILE.write_text("", encoding="utf-8")

    api_key = os.getenv(API_KEY_ENV, "").strip()

    if not api_key:
        fail(f"Missing environment variable: {API_KEY_ENV}")

    weekly_path = latest_file(WEEKLY_DIR, "week_*_NFL_weekly_schedule.csv", "weekly schedule CSV")

    log(f"Weekly schedule input: {weekly_path}")

    weekly_rows = read_csv(weekly_path, WEEKLY_REQUIRED_COLUMNS, "weekly schedule CSV")
    season = detect_season(weekly_rows)

    output_path = OPENERS_DIR / f"{season}_NFL_openers.csv"

    existing_rows = read_existing_openers(output_path)
    new_rows = build_opening_rows(api_key, weekly_rows)
    final_rows = upsert_rows(existing_rows, new_rows)

    write_csv(output_path, final_rows)

    ok_rows = sum(1 for row in final_rows if str(row.get("opener_status", "")).strip() == "ok")
    missing_rows = sum(1 for row in final_rows if str(row.get("opener_status", "")).strip() == "missing")
    error_rows = sum(1 for row in final_rows if str(row.get("opener_status", "")).strip() == "error")

    log(f"Weekly rows loaded: {len(weekly_rows)}")
    log(f"Existing opener rows loaded: {len(existing_rows)}")
    log(f"New opener rows built: {len(new_rows)}")
    log(f"Final opener rows written: {len(final_rows)}")
    log(f"Final opener ok rows: {ok_rows}")
    log(f"Final opener missing rows: {missing_rows}")
    log(f"Final opener error rows: {error_rows}")
    log(f"Output written: {output_path}")

    print(f"Opening odds written: {output_path}")
    print(f"Weekly rows loaded: {len(weekly_rows)}")
    print(f"New opener rows built: {len(new_rows)}")
    print(f"Final opener rows written: {len(final_rows)}")
    print(f"Final opener ok rows: {ok_rows}")
    print(f"Final opener missing rows: {missing_rows}")
    print(f"Final opener error rows: {error_rows}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        print(f"ERROR: see {LOG_FILE}", file=sys.stderr)
        raise

#!/usr/bin/env python3
# docs/win/football/nfl/scripts/00_intake/pull_odds.py
"""Pull current NFL odds and preserve every point-in-time snapshot.

Compatibility outputs remain unchanged:
  docs/win/football/nfl/00_intake/odds/YYYY_MM_DD_NFL_odds.csv
  docs/win/football/nfl/00_intake/odds/raw/YYYY_MM_DD_nfl_odds.json

Every run also writes immutable timestamped copies under:
  docs/win/football/nfl/00_intake/odds/snapshots/
  docs/win/football/nfl/00_intake/odds/raw/snapshots/

The normalized CSV includes snapshot_id and snapshot_fetched_at so future
training/backtests can make true as-of joins without inferring capture time from
Git history or filesystem metadata.
"""

import csv
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_DIR = Path("docs/win/football/nfl")
ODDS_DIR = BASE_DIR / "00_intake" / "odds"
RAW_ODDS_DIR = ODDS_DIR / "raw"
SNAPSHOT_DIR = ODDS_DIR / "snapshots"
RAW_SNAPSHOT_DIR = RAW_ODDS_DIR / "snapshots"
ERROR_DIR = BASE_DIR / "errors" / "00_intake"

for directory in (RAW_ODDS_DIR, SNAPSHOT_DIR, RAW_SNAPSHOT_DIR, ODDS_DIR, ERROR_DIR):
    directory.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "pull_odds.txt"
API_KEY_ENV = "API_ODDS"
API_BASE = "https://api.odds-api.io/v3"
SPORT = "american-football"
LEAGUE = "usa-nfl"
PREFERRED_BOOKMAKERS = ["DraftKings", "FanDuel"]

OUTPUT_COLUMNS = [
    "snapshot_id",
    "snapshot_fetched_at",
    "game_id",
    "commence_time",
    "home_team",
    "away_team",
    "bookmaker",
    "market_type",
    "bet_side",
    "line",
    "odds_american",
    "odds_decimal",
    "last_update",
    "home_moneyline_american",
    "away_moneyline_american",
    "home_spread",
    "away_spread",
    "home_spread_american",
    "away_spread_american",
    "total",
    "over_american",
    "under_american",
]


def utc_now():
    return datetime.now(timezone.utc)


def utc_now_iso():
    return utc_now().isoformat()


def log(message):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now_iso()}] {message}\n")


def fail(message):
    log(f"ERROR: {message}")
    raise RuntimeError(message)


def build_url(path, params):
    return f"{API_BASE}{path}?{urlencode(params)}"


def http_get_json(url):
    request = Request(url, headers={"User-Agent": "nfl-pull-odds/2.0"})
    with urlopen(request, timeout=45) as response:
        status = response.status
        body = response.read().decode("utf-8")
    if status < 200 or status >= 300:
        fail(f"HTTP {status} from odds provider")
    try:
        return json.loads(body)
    except Exception as exc:
        fail(f"Failed to parse JSON from odds provider: {exc}")


def to_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
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
    if decimal is None or decimal <= 1:
        return ""
    if decimal >= 2:
        american = (decimal - 1) * 100
    else:
        american = -100 / (decimal - 1)
    return str(int(round(american)))


def get_market(bookmaker_markets, market_name):
    for market in bookmaker_markets:
        if str(market.get("name", "")).strip().lower() == market_name.lower():
            return market
    return None


def first_odds_entry(market):
    if not market:
        return {}
    odds = market.get("odds")
    if not isinstance(odds, list) or not odds or not isinstance(odds[0], dict):
        return {}
    return odds[0]


def select_bookmaker(bookmakers):
    if not isinstance(bookmakers, dict):
        return "", []
    for bookmaker in PREFERRED_BOOKMAKERS:
        markets = bookmakers.get(bookmaker)
        if isinstance(markets, list) and markets:
            return bookmaker, markets
    return "", []


def build_current_fields(bookmaker_markets):
    current = {
        "home_moneyline_american": "",
        "away_moneyline_american": "",
        "home_spread": "",
        "away_spread": "",
        "home_spread_american": "",
        "away_spread_american": "",
        "total": "",
        "over_american": "",
        "under_american": "",
    }

    ml_odds = first_odds_entry(get_market(bookmaker_markets, "ML"))
    if ml_odds:
        current["home_moneyline_american"] = decimal_to_american(ml_odds.get("home"))
        current["away_moneyline_american"] = decimal_to_american(ml_odds.get("away"))

    spread_odds = first_odds_entry(get_market(bookmaker_markets, "Spread"))
    if spread_odds:
        hdp = to_float(spread_odds.get("hdp"))
        if hdp is not None:
            current["home_spread"] = clean_number(hdp)
            current["away_spread"] = clean_number(-hdp)
        current["home_spread_american"] = decimal_to_american(spread_odds.get("home"))
        current["away_spread_american"] = decimal_to_american(spread_odds.get("away"))

    totals_odds = first_odds_entry(get_market(bookmaker_markets, "Totals"))
    if totals_odds:
        current["total"] = clean_number(totals_odds.get("hdp"))
        current["over_american"] = decimal_to_american(totals_odds.get("over"))
        current["under_american"] = decimal_to_american(totals_odds.get("under"))

    return current


def add_row(
    rows,
    event,
    bookmaker,
    market_type,
    bet_side,
    line,
    odds_decimal,
    last_update,
    current_fields,
    snapshot_id,
    snapshot_fetched_at,
):
    row = {
        "snapshot_id": snapshot_id,
        "snapshot_fetched_at": snapshot_fetched_at,
        "game_id": str(event.get("id", "")).strip(),
        "commence_time": str(event.get("date", "")).strip(),
        "home_team": str(event.get("home", "")).strip(),
        "away_team": str(event.get("away", "")).strip(),
        "bookmaker": bookmaker,
        "market_type": market_type,
        "bet_side": bet_side,
        "line": clean_number(line),
        "odds_american": decimal_to_american(odds_decimal),
        "odds_decimal": clean_number(odds_decimal),
        "last_update": str(last_update or "").strip(),
    }
    row.update(current_fields)
    rows.append(row)


def normalize_event(event, snapshot_id, snapshot_fetched_at):
    rows = []
    bookmaker, bookmaker_markets = select_bookmaker(event.get("bookmakers"))
    if not bookmaker:
        return rows

    current_fields = build_current_fields(bookmaker_markets)

    ml_market = get_market(bookmaker_markets, "ML")
    ml_odds = first_odds_entry(ml_market)
    if ml_odds:
        last_update = ml_market.get("updatedAt", "")
        for side in ("home", "away"):
            add_row(
                rows, event, bookmaker, "h2h", side, "", ml_odds.get(side),
                last_update, current_fields, snapshot_id, snapshot_fetched_at,
            )

    spread_market = get_market(bookmaker_markets, "Spread")
    spread_odds = first_odds_entry(spread_market)
    if spread_odds:
        last_update = spread_market.get("updatedAt", "")
        hdp = to_float(spread_odds.get("hdp"))
        home_line = hdp if hdp is not None else ""
        away_line = -hdp if hdp is not None else ""
        add_row(
            rows, event, bookmaker, "spreads", "home", home_line,
            spread_odds.get("home"), last_update, current_fields,
            snapshot_id, snapshot_fetched_at,
        )
        add_row(
            rows, event, bookmaker, "spreads", "away", away_line,
            spread_odds.get("away"), last_update, current_fields,
            snapshot_id, snapshot_fetched_at,
        )

    totals_market = get_market(bookmaker_markets, "Totals")
    totals_odds = first_odds_entry(totals_market)
    if totals_odds:
        last_update = totals_market.get("updatedAt", "")
        total_line = totals_odds.get("hdp")
        add_row(
            rows, event, bookmaker, "totals", "over", total_line,
            totals_odds.get("over"), last_update, current_fields,
            snapshot_id, snapshot_fetched_at,
        )
        add_row(
            rows, event, bookmaker, "totals", "under", total_line,
            totals_odds.get("under"), last_update, current_fields,
            snapshot_id, snapshot_fetched_at,
        )

    return rows


def chunked(values, size):
    for index in range(0, len(values), size):
        yield values[index:index + size]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    LOG_FILE.write_text("", encoding="utf-8")
    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        fail(f"Missing environment variable: {API_KEY_ENV}")

    captured_at = utc_now()
    run_date = captured_at.strftime("%Y_%m_%d")
    snapshot_id = captured_at.strftime("%Y_%m_%d_%H%M%S_%f")
    snapshot_fetched_at = captured_at.isoformat()

    # Compatibility/current outputs used by the existing weekly merge.
    raw_path = RAW_ODDS_DIR / f"{run_date}_nfl_odds.json"
    csv_path = ODDS_DIR / f"{run_date}_NFL_odds.csv"

    # Immutable archive outputs used for future as-of training/backtests.
    raw_snapshot_path = RAW_SNAPSHOT_DIR / f"{snapshot_id}_nfl_odds.json"
    csv_snapshot_path = SNAPSHOT_DIR / f"{snapshot_id}_NFL_odds.csv"

    events_url = build_url(
        "/events",
        {"apiKey": api_key, "sport": SPORT, "league": LEAGUE, "limit": 500},
    )
    log(f"Fetching NFL events: sport={SPORT}, league={LEAGUE}")
    events = http_get_json(events_url)
    if not isinstance(events, list):
        fail("Events response was not a JSON list")

    event_ids = [str(event.get("id", "")).strip() for event in events if str(event.get("id", "")).strip()]
    if not event_ids:
        fail("No NFL event IDs returned")

    all_odds = []
    for batch in chunked(event_ids, 10):
        odds_url = build_url(
            "/odds/multi",
            {
                "apiKey": api_key,
                "eventIds": ",".join(batch),
                "bookmakers": ",".join(PREFERRED_BOOKMAKERS),
            },
        )
        log(f"Fetching odds batch: {','.join(batch)}")
        odds_batch = http_get_json(odds_url)
        if isinstance(odds_batch, list):
            all_odds.extend(odds_batch)
        elif isinstance(odds_batch, dict):
            all_odds.append(odds_batch)
        else:
            fail("Odds response was not a JSON list or object")

    raw_payload = {
        "snapshot_id": snapshot_id,
        "fetched_at": snapshot_fetched_at,
        "sport": SPORT,
        "league": LEAGUE,
        "preferred_bookmakers": PREFERRED_BOOKMAKERS,
        "events_url": events_url.replace(api_key, "REDACTED"),
        "events_count": len(events),
        "odds_events_count": len(all_odds),
        "events": events,
        "odds": all_odds,
    }

    rows = []
    for event in all_odds:
        rows.extend(normalize_event(event, snapshot_id, snapshot_fetched_at))

    # Write the exact same payload to compatibility + immutable archive files.
    write_json(raw_path, raw_payload)
    write_json(raw_snapshot_path, raw_payload)
    write_csv(csv_path, rows)
    write_csv(csv_snapshot_path, rows)

    log(f"Snapshot ID: {snapshot_id}")
    log(f"Snapshot fetched at: {snapshot_fetched_at}")
    log(f"Events returned: {len(events)}")
    log(f"Odds events returned: {len(all_odds)}")
    log(f"CSV rows written: {len(rows)}")
    log(f"Current raw JSON written: {raw_path}")
    log(f"Current normalized CSV written: {csv_path}")
    log(f"Archived raw JSON written: {raw_snapshot_path}")
    log(f"Archived normalized CSV written: {csv_snapshot_path}")

    print(f"Snapshot ID: {snapshot_id}")
    print(f"Current raw JSON written: {raw_path}")
    print(f"Current normalized CSV written: {csv_path}")
    print(f"Archived raw JSON written: {raw_snapshot_path}")
    print(f"Archived normalized CSV written: {csv_snapshot_path}")
    print(f"Rows written: {len(rows)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        print(f"ERROR: see {LOG_FILE}", file=sys.stderr)
        raise

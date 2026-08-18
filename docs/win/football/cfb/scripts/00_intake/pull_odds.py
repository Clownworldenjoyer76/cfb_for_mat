#!/usr/bin/env python3
# docs/win/football/cfb/scripts/00_intake/pull_odds.py
"""Pull current CFB odds from ESPN Core and preserve point-in-time snapshots.

Input:
  docs/win/football/cfb/00_intake/schedule/{season}_schedule.csv

ESPN source:
  https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/
  events/{game_id}/competitions/{game_id}/odds

Compatibility/current outputs:
  docs/win/football/cfb/00_intake/odds/YYYY_MM_DD_CFB_odds.csv
  docs/win/football/cfb/00_intake/odds/raw/YYYY_MM_DD_cfb_odds.json

Immutable snapshot outputs:
  docs/win/football/cfb/00_intake/odds/snapshots/
  docs/win/football/cfb/00_intake/odds/raw/snapshots/

No external odds provider or API key is used.
"""

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
from zoneinfo import ZoneInfo


BASE_DIR = Path("docs/win/football/cfb")
SCHEDULE_DIR = BASE_DIR / "00_intake" / "schedule"

ODDS_DIR = BASE_DIR / "00_intake" / "odds"
RAW_ODDS_DIR = ODDS_DIR / "raw"
SNAPSHOT_DIR = ODDS_DIR / "snapshots"
RAW_SNAPSHOT_DIR = RAW_ODDS_DIR / "snapshots"
ERROR_DIR = BASE_DIR / "errors" / "00_intake"

for directory in (
    RAW_ODDS_DIR,
    SNAPSHOT_DIR,
    RAW_SNAPSHOT_DIR,
    ODDS_DIR,
    ERROR_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

LOG_FILE = ERROR_DIR / "pull_odds.txt"

ESPN_BASE = (
    "https://sports.core.api.espn.com/v2/sports/football/"
    "leagues/college-football"
)

SCHEDULE_REQUIRED_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
]

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


def latest_file(directory, pattern, label):
    files = sorted(
        directory.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not files:
        fail(f"No {label} found in {directory} matching {pattern}")

    return files[0]


def read_csv(path, required_columns, label):
    if not path.exists():
        fail(f"Missing {label}: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        missing = [
            column
            for column in required_columns
            if column not in fieldnames
        ]
        if missing:
            fail(f"{label} missing columns: {missing}")

        return list(reader)


def build_url(path, params=None):
    url = f"{ESPN_BASE}{path}"
    if params:
        return f"{url}?{urlencode(params)}"
    return url


def http_get_json(url):
    request = Request(
        url,
        headers={
            "User-Agent": "cfb-espn-pull-odds/1.0",
            "Accept": "application/json",
        },
    )

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

        return exc.code, None, body or str(exc)
    except URLError as exc:
        return None, None, str(exc)
    except Exception as exc:
        return None, None, str(exc)

    if status < 200 or status >= 300:
        return status, None, body

    try:
        return status, json.loads(body), ""
    except Exception as exc:
        return status, None, f"JSON parse failed: {exc}"


def fetch_ref(ref):
    if not ref:
        return None

    status, payload, error = http_get_json(ref)
    if payload is None:
        log(
            "REF_FETCH_FAILED "
            f"status={status or ''} ref={ref} error={error}"
        )
    return payload


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


def normalize_american(value):
    number = to_float(value)
    if number is None:
        return ""

    if number == 0:
        return ""

    return str(int(round(number)))


def american_to_decimal(value):
    american = to_float(value)
    if american is None or american == 0:
        return ""

    if american > 0:
        decimal = 1 + (american / 100)
    else:
        decimal = 1 + (100 / abs(american))

    return clean_number(round(decimal, 6))


def parse_iso_or_date(value):
    text = str(value or "").strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(text)
    except Exception:
        pass

    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except Exception:
        return None


def schedule_kickoff_iso(row):
    game_date = str(row.get("game_date", "")).strip()
    game_time = str(row.get("game_time", "")).strip()
    game_timezone = str(row.get("game_timezone", "")).strip()

    if game_date and game_time and game_timezone:
        try:
            local_dt = datetime.strptime(
                f"{game_date} {game_time}",
                "%Y-%m-%d %H:%M",
            ).replace(
                tzinfo=ZoneInfo(game_timezone)
            )
            return (
                local_dt.astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except Exception:
            pass

    if game_date and game_time:
        return f"{game_date}T{game_time}"

    return game_date


def is_upcoming_or_today(row):
    game_date = str(row.get("game_date", "")).strip()
    parsed = parse_iso_or_date(game_date)

    if parsed is None:
        return True

    return parsed.date() >= utc_now().date()


def provider_info(odds_item):
    provider = odds_item.get("provider")

    if isinstance(provider, dict):
        provider_data = provider

        if provider.get("$ref") and not (
            provider.get("name")
            or provider.get("id")
            or provider.get("priority") is not None
        ):
            resolved = fetch_ref(provider.get("$ref"))
            if isinstance(resolved, dict):
                provider_data = resolved

        return {
            "id": str(provider_data.get("id", "")).strip(),
            "name": str(
                provider_data.get("name")
                or provider_data.get("displayName")
                or provider_data.get("shortName")
                or ""
            ).strip(),
            "priority": to_float(provider_data.get("priority")),
        }

    return {
        "id": "",
        "name": "",
        "priority": None,
    }


def resolve_odds_items(collection):
    if not isinstance(collection, dict):
        return []

    items = collection.get("items", [])
    if not isinstance(items, list):
        return []

    resolved = []

    for item in items:
        if not isinstance(item, dict):
            continue

        if item.get("$ref") and not (
            item.get("provider")
            or item.get("homeTeamOdds")
            or item.get("awayTeamOdds")
            or item.get("overUnder") is not None
            or item.get("spread") is not None
        ):
            fetched = fetch_ref(item.get("$ref"))
            if isinstance(fetched, dict):
                resolved.append(fetched)
        else:
            resolved.append(item)

    return resolved


def select_primary_odds_item(items):
    if not items:
        return None

    ranked = []

    for index, item in enumerate(items):
        info = provider_info(item)
        priority = info["priority"]
        rank = priority if priority is not None else 1_000_000
        ranked.append((rank, index, item))

    ranked.sort(key=lambda value: (value[0], value[1]))
    return ranked[0][2]


def nested_value(data, path):
    current = data

    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    return current


def first_value(data, paths):
    for path in paths:
        value = nested_value(data, path)
        if value is not None and str(value).strip() != "":
            return value

    return None


def parse_details_line(details):
    text = str(details or "").strip()

    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*$", text)
    if not match:
        return None

    return to_float(match.group(1))


def extract_market_values(odds_item):
    home_team_odds = (
        odds_item.get("homeTeamOdds")
        if isinstance(odds_item.get("homeTeamOdds"), dict)
        else {}
    )
    away_team_odds = (
        odds_item.get("awayTeamOdds")
        if isinstance(odds_item.get("awayTeamOdds"), dict)
        else {}
    )

    home_moneyline = normalize_american(
        first_value(
            odds_item,
            [
                ("homeTeamOdds", "moneyLine"),
                ("homeTeamOdds", "moneyline"),
                ("homeMoneyLine",),
                ("homeMoneyline",),
            ],
        )
    )

    away_moneyline = normalize_american(
        first_value(
            odds_item,
            [
                ("awayTeamOdds", "moneyLine"),
                ("awayTeamOdds", "moneyline"),
                ("awayMoneyLine",),
                ("awayMoneyline",),
            ],
        )
    )

    home_spread_odds = normalize_american(
        first_value(
            odds_item,
            [
                ("homeTeamOdds", "spreadOdds"),
                ("homeSpreadOdds",),
            ],
        )
    )

    away_spread_odds = normalize_american(
        first_value(
            odds_item,
            [
                ("awayTeamOdds", "spreadOdds"),
                ("awaySpreadOdds",),
            ],
        )
    )

    total = clean_number(
        first_value(
            odds_item,
            [
                ("overUnder",),
                ("total",),
            ],
        )
    )

    over_american = normalize_american(
        first_value(
            odds_item,
            [
                ("overOdds",),
                ("over",),
            ],
        )
    )

    under_american = normalize_american(
        first_value(
            odds_item,
            [
                ("underOdds",),
                ("under",),
            ],
        )
    )

    direct_home_spread = first_value(
        odds_item,
        [
            ("homeTeamOdds", "spread"),
            ("homeSpread",),
        ],
    )

    direct_away_spread = first_value(
        odds_item,
        [
            ("awayTeamOdds", "spread"),
            ("awaySpread",),
        ],
    )

    home_spread = clean_number(direct_home_spread)
    away_spread = clean_number(direct_away_spread)

    if home_spread == "" and away_spread != "":
        away_num = to_float(away_spread)
        if away_num is not None:
            home_spread = clean_number(-away_num)

    if away_spread == "" and home_spread != "":
        home_num = to_float(home_spread)
        if home_num is not None:
            away_spread = clean_number(-home_num)

    if home_spread == "" and away_spread == "":
        detail_line = parse_details_line(
            odds_item.get("details", "")
        )
        generic_spread = to_float(
            odds_item.get("spread")
        )

        line = (
            detail_line
            if detail_line is not None
            else generic_spread
        )

        home_favorite = bool(
            home_team_odds.get("favorite")
        )
        away_favorite = bool(
            away_team_odds.get("favorite")
        )

        if line is not None:
            if home_favorite and not away_favorite:
                home_spread = clean_number(line)
                away_spread = clean_number(-line)
            elif away_favorite and not home_favorite:
                away_spread = clean_number(line)
                home_spread = clean_number(-line)
            else:
                home_spread = clean_number(line)
                away_spread = clean_number(-line)

    last_update = str(
        first_value(
            odds_item,
            [
                ("lastUpdated",),
                ("lastUpdate",),
                ("updated",),
                ("timestamp",),
                ("date",),
            ],
        )
        or ""
    ).strip()

    return {
        "home_moneyline_american": home_moneyline,
        "away_moneyline_american": away_moneyline,
        "home_spread": home_spread,
        "away_spread": away_spread,
        "home_spread_american": home_spread_odds,
        "away_spread_american": away_spread_odds,
        "total": total,
        "over_american": over_american,
        "under_american": under_american,
        "last_update": last_update,
    }


def add_market_row(
    rows,
    event,
    bookmaker,
    market_type,
    bet_side,
    line,
    odds_american,
    current_fields,
    snapshot_id,
    snapshot_fetched_at,
):
    rows.append(
        {
            "snapshot_id": snapshot_id,
            "snapshot_fetched_at": snapshot_fetched_at,
            "game_id": event["id"],
            "commence_time": event["date"],
            "home_team": event["home"],
            "away_team": event["away"],
            "bookmaker": bookmaker,
            "market_type": market_type,
            "bet_side": bet_side,
            "line": clean_number(line),
            "odds_american": normalize_american(
                odds_american
            ),
            "odds_decimal": american_to_decimal(
                odds_american
            ),
            "last_update": current_fields.get(
                "last_update",
                "",
            ),
            **{
                key: current_fields.get(key, "")
                for key in [
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
            },
        }
    )


def normalize_event_odds(
    event,
    odds_item,
    snapshot_id,
    snapshot_fetched_at,
):
    info = provider_info(odds_item)
    bookmaker = info["name"] or info["id"]
    current = extract_market_values(odds_item)
    rows = []

    if (
        current["home_moneyline_american"]
        or current["away_moneyline_american"]
    ):
        add_market_row(
            rows,
            event,
            bookmaker,
            "h2h",
            "home",
            "",
            current["home_moneyline_american"],
            current,
            snapshot_id,
            snapshot_fetched_at,
        )
        add_market_row(
            rows,
            event,
            bookmaker,
            "h2h",
            "away",
            "",
            current["away_moneyline_american"],
            current,
            snapshot_id,
            snapshot_fetched_at,
        )

    if (
        current["home_spread"]
        or current["away_spread"]
        or current["home_spread_american"]
        or current["away_spread_american"]
    ):
        add_market_row(
            rows,
            event,
            bookmaker,
            "spreads",
            "home",
            current["home_spread"],
            current["home_spread_american"],
            current,
            snapshot_id,
            snapshot_fetched_at,
        )
        add_market_row(
            rows,
            event,
            bookmaker,
            "spreads",
            "away",
            current["away_spread"],
            current["away_spread_american"],
            current,
            snapshot_id,
            snapshot_fetched_at,
        )

    if (
        current["total"]
        or current["over_american"]
        or current["under_american"]
    ):
        add_market_row(
            rows,
            event,
            bookmaker,
            "totals",
            "over",
            current["total"],
            current["over_american"],
            current,
            snapshot_id,
            snapshot_fetched_at,
        )
        add_market_row(
            rows,
            event,
            bookmaker,
            "totals",
            "under",
            current["total"],
            current["under_american"],
            current,
            snapshot_id,
            snapshot_fetched_at,
        )

    return rows, info


def fetch_game_odds(game_id):
    path = (
        f"/events/{game_id}/competitions/"
        f"{game_id}/odds"
    )
    url = build_url(
        path,
        {
            "limit": 100,
            "lang": "en",
            "region": "us",
        },
    )

    status, collection, error = http_get_json(url)

    if collection is None:
        log(
            "ODDS_UNAVAILABLE "
            f"game_id={game_id} status={status or ''} "
            f"error={error}"
        )
        return None, url, status

    items = resolve_odds_items(collection)
    selected = select_primary_odds_item(items)

    if selected is None:
        log(
            f"ODDS_EMPTY game_id={game_id} "
            f"status={status or ''}"
        )
        return None, url, status

    return selected, url, status


def build_schedule_groups(schedule_rows):
    upcoming = [
        row
        for row in schedule_rows
        if str(row.get("game_id", "")).strip()
        and is_upcoming_or_today(row)
    ]

    groups = {}

    for row in upcoming:
        key = (
            str(row.get("season", "")).strip(),
            str(row.get("season_type", "")).strip(),
            str(row.get("week", "")).strip(),
        )
        groups.setdefault(key, []).append(row)

    def group_sort(item):
        _, rows = item
        dates = [
            str(row.get("game_date", "")).strip()
            for row in rows
            if str(row.get("game_date", "")).strip()
        ]
        return min(dates) if dates else "9999-99-99"

    return [
        (key, rows)
        for key, rows in sorted(
            groups.items(),
            key=group_sort,
        )
    ]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

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


def write_json(path, payload):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )


def main():
    LOG_FILE.write_text(
        "",
        encoding="utf-8",
    )

    schedule_path = latest_file(
        SCHEDULE_DIR,
        "*_schedule.csv",
        "CFB schedule CSV",
    )

    schedule_rows = read_csv(
        schedule_path,
        SCHEDULE_REQUIRED_COLUMNS,
        "CFB schedule CSV",
    )

    schedule_groups = build_schedule_groups(
        schedule_rows
    )

    if not schedule_groups:
        fail(
            "No upcoming CFB schedule rows available "
            "for ESPN odds lookup"
        )

    captured_at = utc_now()
    run_date = captured_at.strftime("%Y_%m_%d")
    snapshot_id = captured_at.strftime(
        "%Y_%m_%d_%H%M%S_%f"
    )
    snapshot_fetched_at = (
        captured_at.isoformat()
    )

    raw_path = (
        RAW_ODDS_DIR
        / f"{run_date}_cfb_odds.json"
    )
    csv_path = (
        ODDS_DIR
        / f"{run_date}_CFB_odds.csv"
    )

    raw_snapshot_path = (
        RAW_SNAPSHOT_DIR
        / f"{snapshot_id}_cfb_odds.json"
    )
    csv_snapshot_path = (
        SNAPSHOT_DIR
        / f"{snapshot_id}_CFB_odds.csv"
    )

    selected_week = None
    raw_events = []
    raw_odds = []
    rows = []
    request_urls = []

    for week_key, week_rows in schedule_groups:
        week_events = []
        week_raw_odds = []
        week_csv_rows = []
        week_urls = []

        for schedule_row in week_rows:
            game_id = str(
                schedule_row.get("game_id", "")
            ).strip()

            odds_item, odds_url, status = (
                fetch_game_odds(game_id)
            )
            week_urls.append(
                {
                    "game_id": game_id,
                    "url": odds_url,
                    "status": status,
                }
            )

            if odds_item is None:
                continue

            event = {
                "id": game_id,
                "date": schedule_kickoff_iso(
                    schedule_row
                ),
                "home": str(
                    schedule_row.get(
                        "home_team",
                        "",
                    )
                ).strip(),
                "away": str(
                    schedule_row.get(
                        "away_team",
                        "",
                    )
                ).strip(),
            }

            normalized_rows, provider = (
                normalize_event_odds(
                    event,
                    odds_item,
                    snapshot_id,
                    snapshot_fetched_at,
                )
            )

            if not normalized_rows:
                log(
                    "ODDS_NO_SUPPORTED_MARKETS "
                    f"game_id={game_id}"
                )
                continue

            week_events.append(event)
            week_raw_odds.append(
                {
                    "game_id": game_id,
                    "provider": provider,
                    "odds": odds_item,
                }
            )
            week_csv_rows.extend(
                normalized_rows
            )

        if week_csv_rows:
            selected_week = week_key
            raw_events = week_events
            raw_odds = week_raw_odds
            rows = week_csv_rows
            request_urls = week_urls
            break

    if not rows:
        fail(
            "ESPN returned no supported current odds "
            "for any upcoming CFB schedule week"
        )

    raw_payload = {
        "snapshot_id": snapshot_id,
        "fetched_at": snapshot_fetched_at,
        "sport": "football",
        "league": "college-football",
        "source": "ESPN Core API",
        "schedule_input": str(schedule_path),
        "selected_schedule_group": {
            "season": selected_week[0],
            "season_type": selected_week[1],
            "week": selected_week[2],
        },
        "request_urls": request_urls,
        "events_count": len(raw_events),
        "odds_events_count": len(raw_odds),
        "events": raw_events,
        "odds": raw_odds,
    }

    write_json(
        raw_path,
        raw_payload,
    )
    write_json(
        raw_snapshot_path,
        raw_payload,
    )
    write_csv(
        csv_path,
        rows,
    )
    write_csv(
        csv_snapshot_path,
        rows,
    )

    log(f"Schedule input: {schedule_path}")
    log(f"Selected schedule group: {selected_week}")
    log(f"Snapshot ID: {snapshot_id}")
    log(
        f"Snapshot fetched at: "
        f"{snapshot_fetched_at}"
    )
    log(
        f"ESPN odds events returned: "
        f"{len(raw_odds)}"
    )
    log(
        f"CSV rows written: {len(rows)}"
    )
    log(
        f"Current raw JSON written: "
        f"{raw_path}"
    )
    log(
        f"Current normalized CSV written: "
        f"{csv_path}"
    )
    log(
        f"Archived raw JSON written: "
        f"{raw_snapshot_path}"
    )
    log(
        f"Archived normalized CSV written: "
        f"{csv_snapshot_path}"
    )

    print(
        f"Selected schedule group: "
        f"season={selected_week[0]} "
        f"season_type={selected_week[1]} "
        f"week={selected_week[2]}"
    )
    print(
        f"Snapshot ID: {snapshot_id}"
    )
    print(
        f"Current raw JSON written: "
        f"{raw_path}"
    )
    print(
        f"Current normalized CSV written: "
        f"{csv_path}"
    )
    print(
        f"Archived raw JSON written: "
        f"{raw_snapshot_path}"
    )
    print(
        f"Archived normalized CSV written: "
        f"{csv_snapshot_path}"
    )
    print(
        f"Rows written: {len(rows)}"
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

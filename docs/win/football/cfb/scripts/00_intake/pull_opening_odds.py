#!/usr/bin/env python3
# docs/win/football/cfb/scripts/00_intake/pull_opening_odds.py
"""Pull CFB opening odds from ESPN Core odds movement history.

Input:
  docs/win/football/cfb/00_intake/schedule/weekly/week_*_CFB_weekly_schedule.csv

ESPN sources:
  Current competition odds:
    https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/
    events/{game_id}/competitions/{game_id}/odds

  Odds movement history:
    https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/
    events/{game_id}/competitions/{game_id}/odds/{provider_id}/history/0/movement

Output:
  docs/win/football/cfb/00_intake/odds/openers/{season}_CFB_openers.csv

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


BASE_DIR = Path("docs/win/football/cfb")

WEEKLY_DIR = (
    BASE_DIR
    / "00_intake"
    / "schedule"
    / "weekly"
)
OPENERS_DIR = (
    BASE_DIR
    / "00_intake"
    / "odds"
    / "openers"
)

ERROR_DIR = (
    BASE_DIR
    / "errors"
    / "00_intake"
)
ERROR_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
OPENERS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = (
    ERROR_DIR
    / "pull_opening_odds.txt"
)

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
    return datetime.now(
        timezone.utc
    ).isoformat()


def log(message):
    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"[{utc_now_iso()}] "
            f"{message}\n"
        )


def fail(message):
    log(
        f"ERROR: {message}"
    )
    raise RuntimeError(
        message
    )


def latest_file(
    directory,
    pattern,
    label,
):
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


def read_csv(
    path,
    required_columns,
    label,
):
    if not path.exists():
        fail(
            f"Missing {label}: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)
        fieldnames = (
            reader.fieldnames or []
        )

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
                    column: row.get(
                        column,
                        "",
                    )
                    for column in OUTPUT_COLUMNS
                }
            )


def build_url(
    path,
    params=None,
):
    url = f"{ESPN_BASE}{path}"

    if params:
        return (
            f"{url}?"
            f"{urlencode(params)}"
        )

    return url


def http_get_json(url):
    request = Request(
        url,
        headers={
            "User-Agent": (
                "cfb-espn-pull-opening-odds/1.0"
            ),
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(
            request,
            timeout=45,
        ) as response:
            status = response.status
            body = (
                response
                .read()
                .decode("utf-8")
            )

    except HTTPError as exc:
        body = ""
        try:
            body = (
                exc
                .read()
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


def fetch_ref(ref):
    if not ref:
        return None

    status, payload, error = (
        http_get_json(ref)
    )

    if payload is None:
        log(
            "REF_FETCH_FAILED "
            f"status={status or ''} "
            f"ref={ref} "
            f"error={error}"
        )

    return payload


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
    number = to_float(value)

    if (
        number is None
        or number == 0
    ):
        return ""

    return str(
        int(round(number))
    )


def numeric_movement(
    current_value,
    opening_value,
):
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
        current - opening
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


def timestamp_sort_value(value):
    text = normalize_timestamp(
        value
    )

    if not text:
        return None

    candidate = text

    if candidate.endswith("Z"):
        candidate = (
            candidate[:-1]
            + "+00:00"
        )

    try:
        dt = datetime.fromisoformat(
            candidate
        )

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt.timestamp()

    except Exception:
        return None


def nested_value(
    data,
    path,
):
    current = data

    for key in path:
        if not isinstance(
            current,
            dict,
        ):
            return None

        current = current.get(key)

    return current


def first_value(
    data,
    paths,
):
    for path in paths:
        value = nested_value(
            data,
            path,
        )

        if (
            value is not None
            and str(value).strip() != ""
        ):
            return value

    return None


def parse_details_line(details):
    text = str(
        details or ""
    ).strip()

    match = re.search(
        r"([+-]?\d+(?:\.\d+)?)\s*$",
        text,
    )

    if not match:
        return None

    return to_float(
        match.group(1)
    )


def provider_info(
    odds_item,
):
    provider = odds_item.get(
        "provider"
    )

    if isinstance(
        provider,
        dict,
    ):
        provider_data = provider

        if (
            provider.get("$ref")
            and not (
                provider.get("name")
                or provider.get("id")
                or provider.get(
                    "priority"
                ) is not None
            )
        ):
            resolved = fetch_ref(
                provider.get("$ref")
            )

            if isinstance(
                resolved,
                dict,
            ):
                provider_data = (
                    resolved
                )

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

    return {
        "id": "",
        "name": "",
        "priority": None,
    }


def resolve_collection_items(
    collection,
):
    if isinstance(
        collection,
        list,
    ):
        items = collection
    elif isinstance(
        collection,
        dict,
    ):
        items = collection.get(
            "items",
            [],
        )
    else:
        return []

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
            and len(item) <= 3
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
            resolved.append(
                item
            )

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
            info = provider_info(
                item
            )
            if (
                info["name"]
                .strip()
                .lower()
                == desired
            ):
                return item

    ranked = []

    for index, item in enumerate(
        items
    ):
        info = provider_info(
            item
        )
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


def extract_market_values(
    odds_item,
):
    home_team_odds = (
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

    away_team_odds = (
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

    home_moneyline = (
        normalize_american(
            first_value(
                odds_item,
                [
                    (
                        "homeTeamOdds",
                        "moneyLine",
                    ),
                    (
                        "homeTeamOdds",
                        "moneyline",
                    ),
                    (
                        "homeMoneyLine",
                    ),
                    (
                        "homeMoneyline",
                    ),
                ],
            )
        )
    )

    away_moneyline = (
        normalize_american(
            first_value(
                odds_item,
                [
                    (
                        "awayTeamOdds",
                        "moneyLine",
                    ),
                    (
                        "awayTeamOdds",
                        "moneyline",
                    ),
                    (
                        "awayMoneyLine",
                    ),
                    (
                        "awayMoneyline",
                    ),
                ],
            )
        )
    )

    home_spread_odds = (
        normalize_american(
            first_value(
                odds_item,
                [
                    (
                        "homeTeamOdds",
                        "spreadOdds",
                    ),
                    (
                        "homeSpreadOdds",
                    ),
                ],
            )
        )
    )

    away_spread_odds = (
        normalize_american(
            first_value(
                odds_item,
                [
                    (
                        "awayTeamOdds",
                        "spreadOdds",
                    ),
                    (
                        "awaySpreadOdds",
                    ),
                ],
            )
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

    over_american = (
        normalize_american(
            first_value(
                odds_item,
                [
                    ("overOdds",),
                    ("over",),
                ],
            )
        )
    )

    under_american = (
        normalize_american(
            first_value(
                odds_item,
                [
                    ("underOdds",),
                    ("under",),
                ],
            )
        )
    )

    direct_home_spread = (
        first_value(
            odds_item,
            [
                (
                    "homeTeamOdds",
                    "spread",
                ),
                (
                    "homeSpread",
                ),
            ],
        )
    )

    direct_away_spread = (
        first_value(
            odds_item,
            [
                (
                    "awayTeamOdds",
                    "spread",
                ),
                (
                    "awaySpread",
                ),
            ],
        )
    )

    home_spread = clean_number(
        direct_home_spread
    )
    away_spread = clean_number(
        direct_away_spread
    )

    if (
        home_spread == ""
        and away_spread != ""
    ):
        away_num = to_float(
            away_spread
        )
        if away_num is not None:
            home_spread = (
                clean_number(
                    -away_num
                )
            )

    if (
        away_spread == ""
        and home_spread != ""
    ):
        home_num = to_float(
            home_spread
        )
        if home_num is not None:
            away_spread = (
                clean_number(
                    -home_num
                )
            )

    if (
        home_spread == ""
        and away_spread == ""
    ):
        detail_line = (
            parse_details_line(
                odds_item.get(
                    "details",
                    "",
                )
            )
        )

        generic_spread = (
            to_float(
                odds_item.get(
                    "spread"
                )
            )
        )

        line = (
            detail_line
            if detail_line is not None
            else generic_spread
        )

        home_favorite = bool(
            home_team_odds.get(
                "favorite"
            )
        )
        away_favorite = bool(
            away_team_odds.get(
                "favorite"
            )
        )

        if line is not None:
            if (
                home_favorite
                and not away_favorite
            ):
                home_spread = (
                    clean_number(
                        line
                    )
                )
                away_spread = (
                    clean_number(
                        -line
                    )
                )

            elif (
                away_favorite
                and not home_favorite
            ):
                away_spread = (
                    clean_number(
                        line
                    )
                )
                home_spread = (
                    clean_number(
                        -line
                    )
                )

            else:
                home_spread = (
                    clean_number(
                        line
                    )
                )
                away_spread = (
                    clean_number(
                        -line
                    )
                )

    timestamp = normalize_timestamp(
        first_value(
            odds_item,
            [
                ("timestamp",),
                ("lastUpdated",),
                ("lastUpdate",),
                ("updated",),
                ("date",),
            ],
        )
    )

    return {
        "home_moneyline_american": (
            home_moneyline
        ),
        "away_moneyline_american": (
            away_moneyline
        ),
        "home_spread": home_spread,
        "away_spread": away_spread,
        "home_spread_american": (
            home_spread_odds
        ),
        "away_spread_american": (
            away_spread_odds
        ),
        "total": total,
        "over_american": over_american,
        "under_american": under_american,
        "timestamp": timestamp,
    }


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

    status, collection, error = (
        http_get_json(url)
    )

    if collection is None:
        return (
            None,
            status,
            error,
        )

    items = resolve_collection_items(
        collection
    )

    selected = select_primary_odds_item(
        items,
        bookmaker_name=bookmaker_name,
    )

    if selected is None:
        return (
            None,
            status,
            "no_current_odds_items",
        )

    return (
        selected,
        status,
        "",
    )


def fetch_movement_history(
    game_id,
    provider_id,
):
    path = (
        f"/events/{game_id}/"
        f"competitions/{game_id}/"
        f"odds/{provider_id}/"
        "history/0/movement"
    )

    url = build_url(
        path,
        {
            "limit": 1000,
            "lang": "en",
            "region": "us",
        },
    )

    status, payload, error = (
        http_get_json(url)
    )

    return (
        status,
        payload,
        error,
        url,
    )


def looks_like_odds_payload(
    data,
):
    if not isinstance(
        data,
        dict,
    ):
        return False

    keys = {
        "homeTeamOdds",
        "awayTeamOdds",
        "homeMoneyLine",
        "awayMoneyLine",
        "overUnder",
        "spread",
        "details",
        "homeSpread",
        "awaySpread",
    }

    return bool(
        keys.intersection(
            data.keys()
        )
    )


def unwrap_movement_record(
    record,
):
    if not isinstance(
        record,
        dict,
    ):
        return {}

    if looks_like_odds_payload(
        record
    ):
        return record

    for key in [
        "odds",
        "value",
        "snapshot",
        "current",
        "data",
    ]:
        nested = record.get(
            key
        )

        if (
            isinstance(
                nested,
                dict,
            )
            and looks_like_odds_payload(
                nested
            )
        ):
            output = dict(
                nested
            )

            if "timestamp" not in output:
                output["timestamp"] = (
                    first_value(
                        record,
                        [
                            ("timestamp",),
                            ("date",),
                            ("lastUpdated",),
                            ("lastUpdate",),
                        ],
                    )
                    or ""
                )

            return output

    return record


def movement_records(
    payload,
):
    items = resolve_collection_items(
        payload
    )

    if not items and isinstance(
        payload,
        dict,
    ):
        for key in [
            "movements",
            "history",
            "entries",
        ]:
            candidate = payload.get(
                key
            )
            if isinstance(
                candidate,
                list,
            ):
                items = candidate
                break

    return [
        unwrap_movement_record(
            item
        )
        for item in items
        if isinstance(
            item,
            dict,
        )
    ]


def earliest_movement_record(
    records,
):
    timestamped = []

    for index, record in enumerate(
        records
    ):
        timestamp = first_value(
            record,
            [
                ("timestamp",),
                ("date",),
                ("lastUpdated",),
                ("lastUpdate",),
                ("updated",),
            ],
        )

        sort_value = timestamp_sort_value(
            timestamp
        )

        if sort_value is not None:
            timestamped.append(
                (
                    sort_value,
                    index,
                    record,
                )
            )

    if not timestamped:
        return None

    timestamped.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    return timestamped[0][2]


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
    opening_ts = opening.get(
        "timestamp",
        "",
    )

    for side in [
        "home",
        "away",
    ]:
        row = base_row(
            weekly_row,
            "h2h",
            side,
            bookmaker,
        )

        opening_value = opening.get(
            f"{side}_moneyline_american",
            "",
        )
        current_value = str(
            weekly_row.get(
                f"{side}_moneyline_american",
                "",
            )
        ).strip()

        row.update(
            {
                "opening_odds_american": (
                    opening_value
                ),
                "opening_timestamp": (
                    opening_ts
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

        rows.append(
            row
        )


def add_spread_rows(
    rows,
    weekly_row,
    opening,
    bookmaker,
    status,
):
    opening_ts = opening.get(
        "timestamp",
        "",
    )

    for side in [
        "home",
        "away",
    ]:
        row = base_row(
            weekly_row,
            "spreads",
            side,
            bookmaker,
        )

        opening_spread = opening.get(
            f"{side}_spread",
            "",
        )
        opening_odds = opening.get(
            f"{side}_spread_american",
            "",
        )
        current_spread = str(
            weekly_row.get(
                f"{side}_spread",
                "",
            )
        ).strip()

        row.update(
            {
                "opening_line": (
                    opening_spread
                ),
                "opening_odds_american": (
                    opening_odds
                ),
                "opening_timestamp": (
                    opening_ts
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

        rows.append(
            row
        )


def add_total_rows(
    rows,
    weekly_row,
    opening,
    bookmaker,
    status,
):
    opening_ts = opening.get(
        "timestamp",
        "",
    )
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

    for side in [
        "over",
        "under",
    ]:
        row = base_row(
            weekly_row,
            "totals",
            side,
            bookmaker,
        )

        opening_odds = opening.get(
            f"{side}_american",
            "",
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
                    opening_ts
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

        rows.append(
            row
        )


def empty_opening_values():
    return {
        "home_moneyline_american": "",
        "away_moneyline_american": "",
        "home_spread": "",
        "away_spread": "",
        "home_spread_american": "",
        "away_spread_american": "",
        "total": "",
        "over_american": "",
        "under_american": "",
        "timestamp": "",
    }


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
            current_item,
            current_status,
            current_error,
        ) = fetch_current_odds(
            game_id,
            bookmaker_name=weekly_bookmaker,
        )

        if current_item is None:
            opening = (
                empty_opening_values()
            )
            status = status_fields(
                "error",
                (
                    "current_espn_odds_unavailable: "
                    f"{current_error}"
                ),
                current_status,
            )
            bookmaker = (
                weekly_bookmaker
                or ""
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
            continue

        provider = provider_info(
            current_item
        )
        provider_id = provider[
            "id"
        ]
        bookmaker = (
            provider["name"]
            or weekly_bookmaker
            or provider_id
        )

        if not provider_id:
            opening = (
                empty_opening_values()
            )
            status = status_fields(
                "error",
                "espn_provider_id_missing",
                current_status,
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
            continue

        (
            movement_status,
            movement_payload,
            movement_error,
            movement_url,
        ) = fetch_movement_history(
            game_id,
            provider_id,
        )

        if movement_payload is None:
            log(
                "MOVEMENT_REQUEST_FAILED "
                f"game_id={game_id} "
                f"provider_id={provider_id} "
                f"status={movement_status or ''} "
                f"url={movement_url} "
                f"error={movement_error}"
            )

            opening = (
                empty_opening_values()
            )

            status_name = (
                "missing"
                if movement_status == 404
                else "error"
            )

            status = status_fields(
                status_name,
                (
                    movement_error
                    or "espn_movement_unavailable"
                ),
                movement_status,
            )

        else:
            records = movement_records(
                movement_payload
            )
            earliest = earliest_movement_record(
                records
            )

            if earliest is None:
                opening = (
                    empty_opening_values()
                )
                status = status_fields(
                    "missing",
                    "no_timestamped_espn_movement_data",
                    movement_status,
                )
            else:
                opening = (
                    extract_market_values(
                        earliest
                    )
                )

                if not any(
                    str(
                        opening.get(
                            field,
                            "",
                        )
                    ).strip()
                    for field in [
                        "home_moneyline_american",
                        "away_moneyline_american",
                        "home_spread",
                        "away_spread",
                        "total",
                    ]
                ):
                    status = (
                        status_fields(
                            "missing",
                            "opening_market_not_in_espn_movement",
                            movement_status,
                        )
                    )
                else:
                    status = (
                        status_fields(
                            "ok",
                            "",
                            movement_status,
                        )
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
                for column in OUTPUT_COLUMNS
            }
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
                "missing columns; blanks "
                f"inserted: {missing}"
            )

        return existing_rows


def row_has_opening_data(
    row,
):
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

    if row_has_opening_data(
        row
    ):
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

    new_rows = (
        build_opening_rows(
            weekly_rows
        )
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

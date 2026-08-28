#!/usr/bin/env python3
"""
Wide-open historical CFB betting backtest, 2021-2025.

Purpose
-------
Measure historical results for EVERY available candidate side with no betting
selection thresholds applied.

Sportsbook mapping
------------------
2021-2023: DraftKings
2024-2025: ESPN BET

Candidate sides
---------------
MONEYLINE
    HOME
    AWAY

SPREAD
    HOME
    AWAY

TOTAL
    OVER
    UNDER

Metrics binned exactly as configured below
-------------------------------------------
EV:              0.05
Kelly:           0.05
Win probability: 0.05 (5 percentage points)
American odds:   50
Spread line:     1 point
Total line:      5 points

Wide-open means
---------------
- no minimum EV
- no minimum Kelly
- no minimum win probability
- no minimum edge
- no odds filter
- no line filter
- no best_ev / best_prob / best_kelly side selection
- all available candidate sides are graded

Kelly in this report is FULL KELLY from selections.py, not a capped live Kelly
value. This prevents a live configuration cap from altering the historical
binning study.

Outputs
-------
docs/win/football/cfb/data/historical_betting/
    historical_wide_open_candidates_2021_2025.csv
    historical_wide_open_bins_2021_2025.csv
    historical_wide_open_bins_by_season.csv
    historical_wide_open_game_audit_2021_2025.csv

The binned outputs contain, for each market/side/metric/bin:
    bets
    wins
    losses
    pushes
    win_rate
    profit_units
    roi_per_unit_risk

Historical limitations
----------------------
- Point-in-time historical FPI snapshots are not stored, so FPI is disabled.
- Point-in-time historical injury snapshots are not stored, so injury adjustment is 0.
- --travel-weather-mode current uses the current fitted travel/weather
  coefficients retrospectively.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import ssl
import sys
import time
import urllib.request

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CFB_ROOT = SCRIPT_DIR.parents[1]

SCHEDULE_DIR = CFB_ROOT / "00_intake" / "schedule"
TEAM_STATS_DIR = CFB_ROOT / "00_intake" / "team_stats"
PBP_DIR = CFB_ROOT / "00_intake" / "pbp"
HIST_DIR = CFB_ROOT / "data" / "historical_features"
OUT_DIR = CFB_ROOT / "data" / "historical_betting"
CACHE_DIR = OUT_DIR / "cache"

TEAM_MAP = CFB_ROOT / "config" / "mapping" / "team_map.csv"
STADIUM_MAP = CFB_ROOT / "config" / "mapping" / "stadium_map.csv"
TW_COEFS = CFB_ROOT / "config" / "travel_weather_coefficients.csv"

PROJ_PATH = CFB_ROOT / "scripts" / "01_merge" / "projection_week1.py"
SEL_PATH = CFB_ROOT / "scripts" / "02_select" / "selections.py"

SEASONS = [2021, 2022, 2023, 2024, 2025]
SEASON_TYPE = 2

DEFAULT_PROVIDER_BY_SEASON = {
    2021: "draftkings",
    2022: "draftkings",
    2023: "draftkings",
    2024: "espnbet",
    2025: "espnbet",
}

HOME_FIELD = 2.5
DRIVES = 11.5

MARGIN_WEIGHTS = (
    0.36,
    0.28,
    0.20,
    0.16,
)

TOTAL_MARKET_WEIGHT = 0.75

MARGIN_SD = 14.0
TOTAL_SD = 14.0

ESPN_SYMMETRY_TOL = 0.25

EV_BIN = 0.05
KELLY_BIN = 0.05
PROB_BIN = 0.05
ODDS_BIN = 50.0
SPREAD_LINE_BIN = 1.0
TOTAL_LINE_BIN = 5.0

ODDS_URL = (
    "https://sports.core.api.espn.com/"
    "v2/sports/football/leagues/"
    "college-football/events/{g}/"
    "competitions/{g}/odds"
    "?lang=en&region=us"
)

PRED_URL = (
    "https://sports.core.api.espn.com/"
    "v2/sports/football/leagues/"
    "college-football/events/{g}/"
    "competitions/{g}/predictor"
    "?lang=en&region=us"
)

USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "Chrome/126.0"
)

CANDIDATE_SPECS = [
    {
        "market": "moneyline",
        "side": "HOME",
        "prefix": "ml_home",
        "line_source": None,
    },
    {
        "market": "moneyline",
        "side": "AWAY",
        "prefix": "ml_away",
        "line_source": None,
    },
    {
        "market": "spread",
        "side": "HOME",
        "prefix": "spread_home",
        "line_source": "home_spread",
    },
    {
        "market": "spread",
        "side": "AWAY",
        "prefix": "spread_away",
        "line_source": "away_spread",
    },
    {
        "market": "total",
        "side": "OVER",
        "prefix": "total_over",
        "line_source": "total",
    },
    {
        "market": "total",
        "side": "UNDER",
        "prefix": "total_under",
        "line_source": "total",
    },
]


def load_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module

    spec.loader.exec_module(
        module
    )

    return module


projection = load_module(
    "cfb_projection_hist_wide",
    PROJ_PATH,
)

selections = load_module(
    "cfb_selections_hist_wide",
    SEL_PATH,
)


def clean(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.casefold() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }:
        return ""

    return text


def game_id(value: Any) -> str:
    return re.sub(
        r"\.0$",
        "",
        clean(value),
    )


def norm(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        clean(value).casefold(),
    )


def fnum(value: Any) -> float | None:
    try:
        number = float(
            clean(value)
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        number
    ):
        return None

    return number


def american(value: Any) -> float | None:
    number = fnum(
        value
    )

    if number is None:
        return None

    if (
        number >= 100
        or number <= -100
    ):
        return number

    return None


def read_csv(
    path: Path,
    required: list[str] | None = None,
) -> pd.DataFrame:

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if required:
        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise RuntimeError(
                f"{path} missing columns: "
                f"{missing}"
            )

    return df


def write_csv(
    df: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    df.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        path,
    )


def ssl_ctx():
    try:
        import certifi

        return ssl.create_default_context(
            cafile=certifi.where()
        )

    except Exception:
        return (
            ssl.create_default_context()
        )


SSL_CTX = ssl_ctx()


def fetch_json(
    url: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:

    if url.startswith(
        "http://"
    ):
        url = (
            "https://"
            + url[
                len("http://"):
            ]
        )

    last_error: Exception | None = None

    for attempt in range(
        retries + 1
    ):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent":
                        USER_AGENT,
                    "Accept":
                        "application/json",
                },
            )

            with urllib.request.urlopen(
                request,
                timeout=timeout,
                context=SSL_CTX,
            ) as response:

                payload = json.loads(
                    response
                    .read()
                    .decode(
                        "utf-8"
                    )
                )

            if not isinstance(
                payload,
                dict,
            ):
                raise ValueError(
                    "ESPN response "
                    "is not an object"
                )

            return payload

        except Exception as exc:
            last_error = exc

            if attempt < retries:
                time.sleep(
                    0.35
                    * (
                        attempt
                        + 1
                    )
                )

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "ESPN request failed"
    )


def resolve_item(
    item: Any,
    timeout: int,
    retries: int,
) -> dict[str, Any] | None:

    if not isinstance(
        item,
        dict,
    ):
        return None

    if (
        clean(
            item.get(
                "$ref"
            )
        )
        and not isinstance(
            item.get(
                "provider"
            ),
            dict,
        )
    ):
        try:
            return fetch_json(
                clean(
                    item[
                        "$ref"
                    ]
                ),
                timeout,
                retries,
            )

        except Exception:
            return None

    return item


def provider_info(
    item: dict[str, Any],
) -> tuple[
    str,
    str,
]:

    provider = (
        item.get(
            "provider"
        )
        if isinstance(
            item.get(
                "provider"
            ),
            dict,
        )
        else {}
    )

    name = clean(
        provider.get(
            "name"
        )
    )

    provider_id = clean(
        provider.get(
            "id"
        )
    )

    if not provider_id:
        match = re.search(
            r"/providers/([^/?]+)",
            clean(
                provider.get(
                    "$ref"
                )
            ),
        )

        if match:
            provider_id = (
                match.group(
                    1
                )
            )

    return (
        name,
        provider_id,
    )


def provider_match(
    name: str,
    requested: str,
) -> bool:

    requested_key = norm(
        requested
    )

    actual_key = norm(
        name
    )

    exact_names = {
        "draftkings": {
            "draftkings",
        },
        "espnbet": {
            "espnbet",
        },
        "fanduel": {
            "fanduel",
        },
        "caesars": {
            "caesarssportsbook",
            "caesars",
        },
        "betmgm": {
            "betmgm",
            "mgm",
        },
    }

    return (
        actual_key
        in exact_names.get(
            requested_key,
            {
                requested_key,
            },
        )
    )


def blank_odds(
    status: str,
    available: str = "",
) -> dict[str, Any]:

    return {
        "odds_status":
            status,

        "provider":
            "",

        "provider_id":
            "",

        "available_providers":
            available,

        "home_moneyline_american":
            np.nan,

        "away_moneyline_american":
            np.nan,

        "home_spread":
            np.nan,

        "away_spread":
            np.nan,

        "home_spread_american":
            np.nan,

        "away_spread_american":
            np.nan,

        "total":
            np.nan,

        "over_american":
            np.nan,

        "under_american":
            np.nan,

        "odds_line_date":
            "",
    }


def parse_odds(
    payload: dict[str, Any],
    requested: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:

    raw_items = payload.get(
        "items",
        [],
    )

    if not isinstance(
        raw_items,
        list,
    ):
        raw_items = []

    items = []

    for raw_item in raw_items:
        item = resolve_item(
            raw_item,
            timeout,
            retries,
        )

        if item is not None:
            items.append(
                item
            )

    chosen = None

    for item in items:
        (
            name,
            _,
        ) = provider_info(
            item
        )

        if provider_match(
            name,
            requested,
        ):
            chosen = item
            break

    if chosen is None:
        available = "|".join(
            sorted(
                {
                    provider_info(
                        item
                    )[0]
                    for item in items
                    if provider_info(
                        item
                    )[0]
                }
            )
        )

        return blank_odds(
            "PROVIDER_NOT_AVAILABLE",
            available,
        )

    (
        provider_name,
        provider_id,
    ) = provider_info(
        chosen
    )

    if not provider_match(
        provider_name,
        requested,
    ):
        raise RuntimeError(
            "Internal provider mismatch: "
            f"requested={requested!r} "
            f"resolved={provider_name!r}"
        )

    home = (
        chosen.get(
            "homeTeamOdds"
        )
        if isinstance(
            chosen.get(
                "homeTeamOdds"
            ),
            dict,
        )
        else {}
    )

    away = (
        chosen.get(
            "awayTeamOdds"
        )
        if isinstance(
            chosen.get(
                "awayTeamOdds"
            ),
            dict,
        )
        else {}
    )

    home_spread = fnum(
        chosen.get(
            "spread"
        )
    )

    return {
        "odds_status":
            "OK",

        "provider":
            provider_name,

        "provider_id":
            provider_id,

        "available_providers":
            "",

        "home_moneyline_american":
            american(
                home.get(
                    "moneyLine"
                )
            ),

        "away_moneyline_american":
            american(
                away.get(
                    "moneyLine"
                )
            ),

        "home_spread":
            home_spread,

        "away_spread":
            (
                None
                if home_spread is None
                else -home_spread
            ),

        "home_spread_american":
            american(
                home.get(
                    "spreadOdds"
                )
            ),

        "away_spread_american":
            american(
                away.get(
                    "spreadOdds"
                )
            ),

        "total":
            fnum(
                chosen.get(
                    "overUnder"
                )
            ),

        "over_american":
            american(
                chosen.get(
                    "overOdds"
                )
            ),

        "under_american":
            american(
                chosen.get(
                    "underOdds"
                )
            ),

        "odds_line_date":
            clean(
                chosen.get(
                    "lineDate"
                )
                or chosen.get(
                    "lastUpdated"
                )
            ),
    }


def stat_map(
    side: Any,
) -> dict[str, Any]:

    if not isinstance(
        side,
        dict,
    ):
        return {}

    stats = side.get(
        "statistics",
        side.get(
            "stats",
            [],
        ),
    )

    if not isinstance(
        stats,
        list,
    ):
        return {}

    result = {}

    for stat in stats:
        if not isinstance(
            stat,
            dict,
        ):
            continue

        key = norm(
            stat.get(
                "name"
            )
        )

        if key:
            result[
                key
            ] = stat.get(
                "value"
            )

    return result


def parse_predictor(
    payload: dict[str, Any],
) -> dict[str, Any]:

    home = stat_map(
        payload.get(
            "homeTeam"
        )
    )

    away = stat_map(
        payload.get(
            "awayTeam"
        )
    )

    home_ptdiff = fnum(
        home.get(
            "teampredptdiff"
        )
    )

    away_ptdiff = fnum(
        away.get(
            "teampredptdiff"
        )
    )

    home_projection = fnum(
        home.get(
            "gameprojection"
        )
    )

    away_projection = fnum(
        away.get(
            "gameprojection"
        )
    )

    if (
        home_projection
        is not None
        and home_projection > 1
    ):
        home_projection /= 100.0

    if (
        away_projection
        is not None
        and away_projection > 1
    ):
        away_projection /= 100.0

    if home_ptdiff is None:
        status = (
            "PREDICTOR_MARGIN_MISSING"
        )

    elif (
        away_ptdiff is not None
        and abs(
            home_ptdiff
            + away_ptdiff
        )
        > ESPN_SYMMETRY_TOL
    ):
        status = (
            "PREDICTOR_MARGIN_INCONSISTENT"
        )

    else:
        status = "OK"

    return {
        "predictor_status":
            status,

        "espn_home_ptdiff":
            home_ptdiff,

        "espn_away_ptdiff":
            away_ptdiff,

        "espn_home_game_projection":
            home_projection,

        "espn_away_game_projection":
            away_projection,
    }


def fetch_game(
    gid: str,
    provider: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:

    row = {
        "game_id":
            gid,

        "requested_provider":
            provider,
    }

    try:
        payload = fetch_json(
            ODDS_URL.format(
                g=gid
            ),
            timeout,
            retries,
        )

        row.update(
            parse_odds(
                payload,
                provider,
                timeout,
                retries,
            )
        )

    except Exception as exc:
        row.update(
            blank_odds(
                "FETCH_ERROR:"
                f"{type(exc).__name__}"
            )
        )

    try:
        payload = fetch_json(
            PRED_URL.format(
                g=gid
            ),
            timeout,
            retries,
        )

        row.update(
            parse_predictor(
                payload
            )
        )

    except Exception as exc:
        row.update(
            {
                "predictor_status":
                    "FETCH_ERROR:"
                    f"{type(exc).__name__}",

                "espn_home_ptdiff":
                    np.nan,

                "espn_away_ptdiff":
                    np.nan,

                "espn_home_game_projection":
                    np.nan,

                "espn_away_game_projection":
                    np.nan,
            }
        )

    return row


def provider_for_season(
    season: int,
    override: str | None,
) -> str:

    if clean(
        override
    ):
        return clean(
            override
        )

    if (
        season
        not in DEFAULT_PROVIDER_BY_SEASON
    ):
        raise RuntimeError(
            "No default provider configured "
            f"for season={season}"
        )

    return (
        DEFAULT_PROVIDER_BY_SEASON[
            season
        ]
    )


def espn_cache(
    season: int,
    gids: list[str],
    provider: str,
    workers: int,
    timeout: int,
    retries: int,
    refresh: bool,
) -> pd.DataFrame:

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        CACHE_DIR
        / (
            f"{season}_"
            f"{norm(provider)}_"
            "espn_market_predictor.csv"
        )
    )

    cache = pd.DataFrame()

    if (
        path.is_file()
        and not refresh
    ):
        cache = pd.read_csv(
            path,
            low_memory=False,
        )

        if "game_id" in cache.columns:
            cache[
                "game_id"
            ] = cache[
                "game_id"
            ].map(
                game_id
            )

            cache = (
                cache
                .drop_duplicates(
                    "game_id",
                    keep="last",
                )
            )

        if (
            not cache.empty
            and "odds_status"
            in cache.columns
        ):
            status = (
                cache[
                    "odds_status"
                ]
                .map(
                    clean
                )
            )

            if "provider" in cache.columns:
                provider_ok = (
                    cache[
                        "provider"
                    ]
                    .map(
                        lambda value:
                            provider_match(
                                clean(
                                    value
                                ),
                                provider,
                            )
                    )
                )

            else:
                provider_ok = pd.Series(
                    False,
                    index=cache.index,
                )

            contaminated = (
                status.eq(
                    "OK"
                )
                & ~provider_ok
            )

            contaminated_count = int(
                contaminated.sum()
            )

            if contaminated_count:
                print(
                    f"{season}: removing "
                    f"{contaminated_count} "
                    "contaminated cached "
                    f"{provider} rows"
                )

                cache = cache.loc[
                    ~contaminated
                ].copy()

    have = set(
        cache.get(
            "game_id",
            pd.Series(
                dtype=str
            ),
        ).astype(
            str
        )
    )

    missing = [
        gid
        for gid in gids
        if gid not in have
    ]

    if missing:
        print(
            f"{season}: fetching ESPN "
            "odds/predictor for "
            f"{len(missing)} games "
            f"({provider})"
        )

        rows = []

        with ThreadPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = {
                executor.submit(
                    fetch_game,
                    gid,
                    provider,
                    timeout,
                    retries,
                ):
                    gid

                for gid in missing
            }

            for number, future in enumerate(
                as_completed(
                    futures
                ),
                start=1,
            ):
                gid = futures[
                    future
                ]

                try:
                    rows.append(
                        future.result()
                    )

                except Exception as exc:
                    rows.append(
                        {
                            "game_id":
                                gid,

                            "requested_provider":
                                provider,

                            "odds_status":
                                "UNHANDLED:"
                                f"{type(exc).__name__}",

                            "predictor_status":
                                "UNHANDLED:"
                                f"{type(exc).__name__}",
                        }
                    )

                if (
                    number % 100 == 0
                    or number
                    == len(
                        missing
                    )
                ):
                    print(
                        f"{season}: fetched "
                        f"{number}/"
                        f"{len(missing)}"
                    )

        new = pd.DataFrame(
            rows
        )

        cache = (
            new
            if cache.empty
            else pd.concat(
                [
                    cache,
                    new,
                ],
                ignore_index=True,
            )
        )

        cache[
            "game_id"
        ] = cache[
            "game_id"
        ].map(
            game_id
        )

        cache = (
            cache
            .drop_duplicates(
                "game_id",
                keep="last",
            )
            .sort_values(
                "game_id"
            )
        )

        write_csv(
            cache,
            path,
        )

    if cache.empty:
        return pd.DataFrame(
            columns=[
                "game_id"
            ]
        )

    result = cache[
        cache[
            "game_id"
        ].isin(
            gids
        )
    ].copy()

    if (
        "odds_status"
        in result.columns
    ):
        ok_rows = result[
            result[
                "odds_status"
            ].map(
                clean
            ).eq(
                "OK"
            )
        ]

        if not ok_rows.empty:
            bad_provider = ok_rows[
                ~ok_rows[
                    "provider"
                ]
                .map(
                    lambda value:
                        provider_match(
                            clean(
                                value
                            ),
                            provider,
                        )
                )
            ]

            if not bad_provider.empty:
                examples = (
                    bad_provider[
                        [
                            "game_id",
                            "provider",
                            "provider_id",
                        ]
                    ]
                    .head(
                        10
                    )
                    .to_dict(
                        "records"
                    )
                )

                raise RuntimeError(
                    f"{season}: sportsbook "
                    "validation failed for "
                    f"requested provider "
                    f"{provider!r}: "
                    f"{examples}"
                )

    return result


def load_schedule(
    season: int,
) -> pd.DataFrame:

    path = (
        SCHEDULE_DIR
        / f"{season}_schedule.csv"
    )

    df = read_csv(
        path,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "game_date",
            "game_time",
            "away_team",
            "home_team",
            "neutral_site",
            "stadium",
        ],
    )

    df[
        "game_id"
    ] = df[
        "game_id"
    ].map(
        game_id
    )

    season_values = pd.to_numeric(
        df[
            "season"
        ],
        errors="coerce",
    )

    season_types = pd.to_numeric(
        df[
            "season_type"
        ],
        errors="coerce",
    )

    weeks = pd.to_numeric(
        df[
            "week"
        ],
        errors="coerce",
    )

    df = df[
        season_values.eq(
            season
        )
        & season_types.eq(
            SEASON_TYPE
        )
        & weeks.notna()
        & df[
            "game_id"
        ].ne(
            ""
        )
    ].copy()

    df[
        "week"
    ] = pd.to_numeric(
        df[
            "week"
        ],
        errors="raise",
    ).astype(
        int
    )

    if df[
        "game_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate game_id in "
            f"{path}"
        )

    return df


def load_finals(
    season: int,
) -> pd.DataFrame:

    path = (
        PBP_DIR
        / f"{season}_pbp.parquet"
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    df = pd.read_parquet(
        path,
        columns=[
            "game_id",
            "sequenceNumber",
            "end.homeScore",
            "end.awayScore",
        ],
    )

    df[
        "game_id"
    ] = df[
        "game_id"
    ].map(
        game_id
    )

    df[
        "sequenceNumber"
    ] = pd.to_numeric(
        df[
            "sequenceNumber"
        ],
        errors="coerce",
    )

    df[
        "home_final"
    ] = pd.to_numeric(
        df[
            "end.homeScore"
        ],
        errors="coerce",
    )

    df[
        "away_final"
    ] = pd.to_numeric(
        df[
            "end.awayScore"
        ],
        errors="coerce",
    )

    df = (
        df
        .dropna(
            subset=[
                "home_final",
                "away_final",
            ]
        )
        .sort_values(
            [
                "game_id",
                "sequenceNumber",
            ]
        )
    )

    return (
        df
        .groupby(
            "game_id",
            as_index=False,
        )
        .tail(
            1
        )
        [
            [
                "game_id",
                "home_final",
                "away_final",
            ]
        ]
        .copy()
    )


def load_features(
    season: int,
) -> pd.DataFrame:

    df = read_csv(
        HIST_DIR
        / (
            f"{season}_"
            "travel_weather.csv"
        ),
        [
            "game_id"
        ],
    )

    df[
        "game_id"
    ] = df[
        "game_id"
    ].map(
        game_id
    )

    return (
        df.drop_duplicates(
            "game_id",
            keep="last",
        )
    )


def load_team_stats(
    season: int,
) -> pd.DataFrame:

    return read_csv(
        TEAM_STATS_DIR
        / (
            f"{season}_"
            "team_stats.csv"
        ),
        [
            "season",
            "week",
            "team",
            *projection.TEAM_METRICS,
        ],
    )


def prepare_prior(
    season: int,
    week: int,
    resolver,
    cache: dict[
        int,
        pd.DataFrame,
    ],
):

    if week <= 1:
        source_season = (
            season
            - 1
        )

        min_weeks = 10

        label = (
            "prior_season_"
            f"{source_season}"
        )

        path = (
            TEAM_STATS_DIR
            / (
                f"{source_season}_"
                "team_stats.csv"
            )
        )

        if not path.is_file():
            return (
                None,
                min_weeks,
                "NO_PRIOR_SEASON_TEAM_STATS",
            )

        if source_season not in cache:
            cache[
                source_season
            ] = load_team_stats(
                source_season
            )

        source = cache[
            source_season
        ].copy()

    else:
        min_weeks = 1

        label = (
            "current_season_"
            f"before_week_{week}"
        )

        if season not in cache:
            cache[
                season
            ] = load_team_stats(
                season
            )

        source = cache[
            season
        ].copy()

        source_week = pd.to_numeric(
            source[
                "week"
            ],
            errors="coerce",
        )

        source_season = pd.to_numeric(
            source[
                "season"
            ],
            errors="coerce",
        )

        source = source[
            source_season.eq(
                season
            )
            & source_week.notna()
            & source_week.lt(
                week
            )
        ].copy()

        if source.empty:
            return (
                None,
                min_weeks,
                "NO_CURRENT_SEASON_PRIOR_STATS",
            )

    prior = (
        projection
        .build_prior_table(
            source,
            resolver,
        )
    )

    empty_fpi = pd.DataFrame(
        columns=[
            "team",
            "team_id",
            "fpi",
            "epaoffense",
            "epadefense",
        ]
    )

    prior = (
        projection
        .scale_prior_to_fpi(
            prior,
            empty_fpi,
        )
    )

    return (
        prior,
        min_weeks,
        label,
    )


def prior_for_game(
    prior: pd.DataFrame,
    min_weeks: int,
    home: str,
    away: str,
    hfa: float,
):

    lookup = prior.set_index(
        "team",
        drop=False,
    )

    fallback = prior.mean(
        numeric_only=True
    )

    fallback[
        "prior_team_weeks"
    ] = 0

    fallback[
        "prior_rating"
    ] = 0.0

    home_source = (
        lookup.loc[
            home
        ]
        if home
        in lookup.index
        else None
    )

    away_source = (
        lookup.loc[
            away
        ]
        if away
        in lookup.index
        else None
    )

    home_weeks = (
        int(
            fnum(
                home_source.get(
                    "prior_team_weeks"
                )
            )
            or 0
        )
        if home_source
        is not None
        else 0
    )

    away_weeks = (
        int(
            fnum(
                away_source.get(
                    "prior_team_weeks"
                )
            )
            or 0
        )
        if away_source
        is not None
        else 0
    )

    home_fallback = (
        home_source is None
        or home_weeks
        < min_weeks
    )

    away_fallback = (
        away_source is None
        or away_weeks
        < min_weeks
    )

    home_prior = (
        fallback
        if home_fallback
        else home_source
    )

    away_prior = (
        fallback
        if away_fallback
        else away_source
    )

    if (
        home_fallback
        or away_fallback
    ):
        prior_margin = None

    else:
        prior_margin = (
            float(
                home_prior[
                    "prior_rating"
                ]
            )
            - float(
                away_prior[
                    "prior_rating"
                ]
            )
            + hfa
        )

    prior_total = (
        projection
        .prior_total_estimate(
            home_prior,
            away_prior,
            DRIVES,
        )
    )

    return {
        "home_weeks":
            home_weeks,

        "away_weeks":
            away_weeks,

        "home_fallback":
            int(
                home_fallback
            ),

        "away_fallback":
            int(
                away_fallback
            ),

        "margin":
            prior_margin,

        "total":
            prior_total,
    }


def evaluate_candidates(
    gid: str,
    probabilities: dict[
        str,
        float,
    ],
    market: pd.Series,
) -> dict[str, Any]:

    row = pd.Series(
        {
            "game_id":
                gid,

            **probabilities,

            "sched_home_moneyline_american":
                market.get(
                    "home_moneyline_american"
                ),

            "sched_away_moneyline_american":
                market.get(
                    "away_moneyline_american"
                ),

            "sched_home_spread":
                market.get(
                    "home_spread"
                ),

            "sched_away_spread":
                market.get(
                    "away_spread"
                ),

            "sched_home_spread_american":
                market.get(
                    "home_spread_american"
                ),

            "sched_away_spread_american":
                market.get(
                    "away_spread_american"
                ),

            "sched_total":
                market.get(
                    "total"
                ),

            "sched_over_american":
                market.get(
                    "over_american"
                ),

            "sched_under_american":
                market.get(
                    "under_american"
                ),
        }
    )

    return {
        **selections
        .evaluate_moneyline(
            row
        ),

        **selections
        .evaluate_spread(
            row
        ),

        **selections
        .evaluate_total(
            row
        ),
    }


def win_profit(
    odds: float,
) -> float:

    if odds > 0:
        return (
            odds
            / 100.0
        )

    return (
        100.0
        / abs(
            odds
        )
    )


def grade(
    market: str,
    side: str,
    line: float | None,
    home_final: float,
    away_final: float,
) -> str:

    side = side.upper()

    if market == "moneyline":
        if home_final == away_final:
            return "PUSH"

        if side == "HOME":
            return (
                "WIN"
                if home_final
                > away_final
                else "LOSS"
            )

        if side == "AWAY":
            return (
                "WIN"
                if away_final
                > home_final
                else "LOSS"
            )

    if (
        market == "spread"
        and line is not None
    ):
        if side == "HOME":
            value = (
                home_final
                + line
                - away_final
            )

        elif side == "AWAY":
            value = (
                away_final
                + line
                - home_final
            )

        else:
            return "UNGRADABLE"

        if abs(
            value
        ) < 1e-9:
            return "PUSH"

        return (
            "WIN"
            if value > 0
            else "LOSS"
        )

    if (
        market == "total"
        and line is not None
    ):
        actual_total = (
            home_final
            + away_final
        )

        if abs(
            actual_total
            - line
        ) < 1e-9:
            return "PUSH"

        if side == "OVER":
            return (
                "WIN"
                if actual_total
                > line
                else "LOSS"
            )

        if side == "UNDER":
            return (
                "WIN"
                if actual_total
                < line
                else "LOSS"
            )

    return "UNGRADABLE"


def candidate_row(
    game: dict[str, Any],
    market_row: pd.Series,
    metrics: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any] | None:

    prefix = spec[
        "prefix"
    ]

    market_name = spec[
        "market"
    ]

    side = spec[
        "side"
    ]

    model_probability = fnum(
        metrics.get(
            f"{prefix}_"
            "model_probability"
        )
    )

    ev = fnum(
        metrics.get(
            f"{prefix}_ev"
        )
    )

    kelly = fnum(
        metrics.get(
            f"{prefix}_"
            "full_kelly"
        )
    )

    odds = fnum(
        metrics.get(
            f"{prefix}_"
            "odds_american"
        )
    )

    if odds is None:
        if market_name == "moneyline":
            odds = fnum(
                market_row.get(
                    "home_moneyline_american"
                    if side == "HOME"
                    else "away_moneyline_american"
                )
            )

        elif market_name == "spread":
            odds = fnum(
                market_row.get(
                    "home_spread_american"
                    if side == "HOME"
                    else "away_spread_american"
                )
            )

        elif market_name == "total":
            odds = fnum(
                market_row.get(
                    "over_american"
                    if side == "OVER"
                    else "under_american"
                )
            )

    if spec[
        "line_source"
    ] is None:
        line = None

    else:
        line = fnum(
            metrics.get(
                f"{prefix}_line"
            )
        )

        if line is None:
            line = fnum(
                market_row.get(
                    spec[
                        "line_source"
                    ]
                )
            )

    required = [
        model_probability,
        ev,
        kelly,
        odds,
    ]

    if market_name in {
        "spread",
        "total",
    }:
        required.append(
            line
        )

    if any(
        value is None
        for value in required
    ):
        return None

    result = grade(
        market_name,
        side,
        line,
        float(
            game[
                "home_final"
            ]
        ),
        float(
            game[
                "away_final"
            ]
        ),
    )

    if result == "WIN":
        profit = win_profit(
            float(
                odds
            )
        )

        win_indicator = 1.0

    elif result == "LOSS":
        profit = -1.0
        win_indicator = 0.0

    elif result == "PUSH":
        profit = 0.0
        win_indicator = np.nan

    else:
        return None

    return {
        "season":
            game[
                "season"
            ],

        "week":
            game[
                "week"
            ],

        "game_id":
            game[
                "game_id"
            ],

        "game_date":
            game[
                "game_date"
            ],

        "away_team":
            game[
                "away_team"
            ],

        "home_team":
            game[
                "home_team"
            ],

        "provider":
            game[
                "provider"
            ],

        "market":
            market_name,

        "side":
            side,

        "ev":
            float(
                ev
            ),

        "kelly":
            float(
                kelly
            ),

        "win_probability":
            float(
                model_probability
            ),

        "odds_american":
            float(
                odds
            ),

        "line":
            line,

        "home_final":
            float(
                game[
                    "home_final"
                ]
            ),

        "away_final":
            float(
                game[
                    "away_final"
                ]
            ),

        "result":
            result,

        "win_indicator":
            win_indicator,

        "profit_units":
            profit,
    }


def replay_season(
    season: int,
    schedule: pd.DataFrame,
    finals: pd.DataFrame,
    features: pd.DataFrame,
    espn: pd.DataFrame,
    resolver,
    stadium_lookup,
    coefficients,
    stats_cache,
):

    finals_lookup = (
        finals.set_index(
            "game_id",
            drop=False,
        )
    )

    features_lookup = (
        features.set_index(
            "game_id",
            drop=False,
        )
    )

    market_lookup = (
        espn.set_index(
            "game_id",
            drop=False,
        )
    )

    candidates = []
    audit_rows = []

    for week in sorted(
        schedule[
            "week"
        ].unique()
    ):

        (
            prior,
            min_weeks,
            prior_source,
        ) = prepare_prior(
            season,
            int(
                week
            ),
            resolver,
            stats_cache,
        )

        if prior is None:
            print(
                f"{season} week "
                f"{week}: skipped - "
                f"{prior_source}"
            )

        week_schedule = schedule[
            schedule[
                "week"
            ].eq(
                week
            )
        ]

        for _, sched_row in (
            week_schedule.iterrows()
        ):
            gid = game_id(
                sched_row.get(
                    "game_id"
                )
            )

            if (
                gid not in finals_lookup.index
                or gid not in market_lookup.index
            ):
                continue

            final_row = (
                finals_lookup.loc[
                    gid
                ]
            )

            market_row = (
                market_lookup.loc[
                    gid
                ]
            )

            feature_row = (
                features_lookup.loc[
                    gid
                ]
                if gid
                in features_lookup.index
                else None
            )

            home_team = (
                resolver.resolve(
                    sched_row.get(
                        "home_team"
                    )
                )
            )

            away_team = (
                resolver.resolve(
                    sched_row.get(
                        "away_team"
                    )
                )
            )

            base = {
                "season":
                    season,

                "week":
                    int(
                        week
                    ),

                "game_id":
                    gid,

                "game_date":
                    clean(
                        sched_row.get(
                            "game_date"
                        )
                    ),

                "away_team":
                    away_team,

                "home_team":
                    home_team,

                "provider":
                    clean(
                        market_row.get(
                            "provider"
                        )
                    ),

                "odds_status":
                    clean(
                        market_row.get(
                            "odds_status"
                        )
                    ),

                "predictor_status":
                    clean(
                        market_row.get(
                            "predictor_status"
                        )
                    ),

                "home_final":
                    float(
                        final_row[
                            "home_final"
                        ]
                    ),

                "away_final":
                    float(
                        final_row[
                            "away_final"
                        ]
                    ),
            }

            if prior is None:
                audit_rows.append(
                    {
                        **base,

                        "projection_status":
                            prior_source,

                        "candidate_count":
                            0,
                    }
                )

                continue

            (
                neutral_original,
                neutral,
                neutral_corrected,
                home_stadium_match,
            ) = (
                projection
                .resolve_neutral_site(
                    sched_row,
                    home_team,
                    stadium_lookup,
                )
            )

            home_field = (
                0.0
                if neutral
                else HOME_FIELD
            )

            prior_context = (
                prior_for_game(
                    prior,
                    min_weeks,
                    home_team,
                    away_team,
                    home_field,
                )
            )

            home_spread = fnum(
                market_row.get(
                    "home_spread"
                )
            )

            total_line = fnum(
                market_row.get(
                    "total"
                )
            )

            market_margin = (
                None
                if home_spread is None
                else -home_spread
            )

            espn_home = fnum(
                market_row.get(
                    "espn_home_ptdiff"
                )
            )

            espn_away = fnum(
                market_row.get(
                    "espn_away_ptdiff"
                )
            )

            espn_margin = None

            if (
                espn_home is not None
                and (
                    espn_away is None
                    or abs(
                        espn_home
                        + espn_away
                    )
                    <= ESPN_SYMMETRY_TOL
                )
            ):
                espn_margin = (
                    espn_home
                )

            fpi_margin = None

            (
                blended_margin,
                _,
            ) = (
                projection
                .weighted_blend(
                    [
                        (
                            market_margin,
                            MARGIN_WEIGHTS[
                                0
                            ],
                        ),
                        (
                            fpi_margin,
                            MARGIN_WEIGHTS[
                                1
                            ],
                        ),
                        (
                            espn_margin,
                            MARGIN_WEIGHTS[
                                2
                            ],
                        ),
                        (
                            prior_context[
                                "margin"
                            ],
                            MARGIN_WEIGHTS[
                                3
                            ],
                        ),
                    ]
                )
            )

            if blended_margin is None:
                audit_rows.append(
                    {
                        **base,

                        "projection_status":
                            "NO_MARGIN_COMPONENT",

                        "candidate_count":
                            0,
                    }
                )

                continue

            (
                _,
                travel_adjustment,
                _,
            ) = (
                projection
                .calculate_travel_adjustment(
                    feature_row,
                    coefficients.get(
                        "margin",
                        {},
                    ),
                )
            )

            predicted_margin = (
                float(
                    blended_margin
                )
                + travel_adjustment
            )

            (
                predicted_total,
                _,
            ) = (
                projection
                .weighted_blend(
                    [
                        (
                            total_line,
                            TOTAL_MARKET_WEIGHT,
                        ),
                        (
                            prior_context[
                                "total"
                            ],
                            1.0
                            - TOTAL_MARKET_WEIGHT,
                        ),
                    ]
                )
            )

            if predicted_total is None:
                audit_rows.append(
                    {
                        **base,

                        "projection_status":
                            "NO_TOTAL_COMPONENT",

                        "candidate_count":
                            0,
                    }
                )

                continue

            (
                _,
                _,
                weather_adjustment,
                _,
            ) = (
                projection
                .calculate_weather_adjustment(
                    feature_row,
                    coefficients.get(
                        "total",
                        {},
                    ),
                )
            )

            predicted_total = (
                float(
                    predicted_total
                )
                + weather_adjustment
            )

            predicted_total = max(
                predicted_total,
                abs(
                    predicted_margin
                )
                + 2.0,
            )

            probabilities = (
                projection
                .build_betting_probabilities(
                    predicted_margin,
                    predicted_total,
                    home_spread,
                    total_line,
                    MARGIN_SD,
                    TOTAL_SD,
                )
            )

            metrics = (
                evaluate_candidates(
                    gid,
                    probabilities,
                    market_row,
                )
            )

            game_candidate_count = 0

            for spec in CANDIDATE_SPECS:
                row = candidate_row(
                    base,
                    market_row,
                    metrics,
                    spec,
                )

                if row is not None:
                    candidates.append(
                        row
                    )

                    game_candidate_count += 1

            audit_rows.append(
                {
                    **base,

                    "projection_status":
                        "OK",

                    "candidate_count":
                        game_candidate_count,

                    "neutral_site_original":
                        int(
                            neutral_original
                        ),

                    "neutral_site":
                        int(
                            neutral
                        ),

                    "neutral_site_corrected":
                        int(
                            neutral_corrected
                        ),

                    "home_stadium_match":
                        int(
                            home_stadium_match
                        ),
                }
            )

    return (
        pd.DataFrame(
            audit_rows
        ),
        pd.DataFrame(
            candidates
        ),
    )


def floor_bin(
    value: float,
    width: float,
) -> tuple[
    float,
    float,
]:

    low = (
        math.floor(
            (
                value
                + 1e-12
            )
            / width
        )
        * width
    )

    high = (
        low
        + width
    )

    return (
        low,
        high,
    )


def probability_bin(
    value: float,
) -> tuple[
    float,
    float,
]:

    value = min(
        max(
            value,
            0.0,
        ),
        1.0,
    )

    if abs(
        value
        - 1.0
    ) < 1e-12:
        return (
            1.0
            - PROB_BIN,
            1.0,
        )

    low = (
        math.floor(
            (
                value
                + 1e-12
            )
            / PROB_BIN
        )
        * PROB_BIN
    )

    high = min(
        low
        + PROB_BIN,
        1.0,
    )

    return (
        low,
        high,
    )


def metric_bin(
    metric: str,
    value: float,
    market: str,
) -> tuple[
    float,
    float,
]:

    if metric == "ev":
        return floor_bin(
            value,
            EV_BIN,
        )

    if metric == "kelly":
        return floor_bin(
            value,
            KELLY_BIN,
        )

    if metric == (
        "win_probability"
    ):
        return probability_bin(
            value
        )

    if metric == (
        "odds_american"
    ):
        return floor_bin(
            value,
            ODDS_BIN,
        )

    if metric == "line":
        if market == "spread":
            return floor_bin(
                value,
                SPREAD_LINE_BIN,
            )

        if market == "total":
            return floor_bin(
                value,
                TOTAL_LINE_BIN,
            )

    raise ValueError(
        "Unsupported metric/market bin: "
        f"metric={metric} "
        f"market={market}"
    )


def format_bin_label(
    metric: str,
    low: float,
    high: float,
) -> str:

    if metric in {
        "ev",
        "kelly",
        "win_probability",
    }:
        return (
            f"{low:.2f} "
            f"to {high:.2f}"
        )

    if metric == (
        "odds_american"
    ):
        return (
            f"{int(round(low))} "
            f"to <{int(round(high))}"
        )

    if metric == "line":
        if (
            float(
                low
            ).is_integer()
            and float(
                high
            ).is_integer()
        ):
            return (
                f"{int(low)} "
                f"to <{int(high)}"
            )

        return (
            f"{low:g} "
            f"to <{high:g}"
        )

    return (
        f"{low:g} "
        f"to <{high:g}"
    )


def summarize_bin_group(
    group: pd.DataFrame,
    metric: str,
    market: str,
    side: str,
    low: float,
    high: float,
    season: int | str | None,
) -> dict[str, Any]:

    settled = group[
        group[
            "result"
        ].isin(
            [
                "WIN",
                "LOSS",
            ]
        )
    ]

    bets = len(
        group
    )

    wins = int(
        group[
            "result"
        ].eq(
            "WIN"
        ).sum()
    )

    losses = int(
        group[
            "result"
        ].eq(
            "LOSS"
        ).sum()
    )

    pushes = int(
        group[
            "result"
        ].eq(
            "PUSH"
        ).sum()
    )

    profit = float(
        pd.to_numeric(
            group[
                "profit_units"
            ],
            errors="coerce",
        ).sum()
    )

    result = {
        "market":
            market,

        "side":
            side,

        "metric":
            metric,

        "bin_low":
            low,

        "bin_high":
            high,

        "bin_label":
            format_bin_label(
                metric,
                low,
                high,
            ),

        "bets":
            bets,

        "wins":
            wins,

        "losses":
            losses,

        "pushes":
            pushes,

        "win_rate":
            (
                wins
                / len(
                    settled
                )
                if len(
                    settled
                )
                else np.nan
            ),

        "profit_units":
            profit,

        "roi_per_unit_risk":
            (
                profit
                / bets
                if bets
                else np.nan
            ),

        "avg_metric_value":
            float(
                pd.to_numeric(
                    group[
                        metric
                    ],
                    errors="coerce",
                ).mean()
            ),
    }

    if season is not None:
        result = {
            "season":
                season,

            **result,
        }

    return result


def build_binned_results(
    candidates: pd.DataFrame,
    by_season: bool,
) -> pd.DataFrame:

    rows = []

    for spec in CANDIDATE_SPECS:
        market = spec[
            "market"
        ]

        side = spec[
            "side"
        ]

        base = candidates[
            candidates[
                "market"
            ].eq(
                market
            )
            & candidates[
                "side"
            ].eq(
                side
            )
        ].copy()

        metrics = [
            "ev",
            "kelly",
            "win_probability",
            "odds_american",
        ]

        if market in {
            "spread",
            "total",
        }:
            metrics.append(
                "line"
            )

        if by_season:
            season_groups = [
                (
                    int(
                        season
                    ),
                    frame,
                )
                for (
                    season,
                    frame,
                )
                in base.groupby(
                    "season",
                    sort=True,
                )
            ]

        else:
            season_groups = [
                (
                    None,
                    base,
                )
            ]

        for (
            season,
            frame,
        ) in season_groups:

            for metric in metrics:
                metric_values = (
                    pd.to_numeric(
                        frame[
                            metric
                        ],
                        errors="coerce",
                    )
                )

                valid = frame.loc[
                    metric_values.notna()
                ].copy()

                if valid.empty:
                    continue

                valid[
                    metric
                ] = pd.to_numeric(
                    valid[
                        metric
                    ],
                    errors="coerce",
                )

                valid[
                    "_bin"
                ] = valid[
                    metric
                ].map(
                    lambda value:
                        metric_bin(
                            metric,
                            float(
                                value
                            ),
                            market,
                        )
                )

                for (
                    (
                        low,
                        high,
                    ),
                    group,
                ) in valid.groupby(
                    "_bin",
                    sort=True,
                ):

                    rows.append(
                        summarize_bin_group(
                            group,
                            metric,
                            market,
                            side,
                            float(
                                low
                            ),
                            float(
                                high
                            ),
                            season,
                        )
                    )

    output = pd.DataFrame(
        rows
    )

    if output.empty:
        return output

    if by_season:
        sort_columns = [
            "season",
            "market",
            "side",
            "metric",
            "bin_low",
        ]

    else:
        sort_columns = [
            "market",
            "side",
            "metric",
            "bin_low",
        ]

    return (
        output
        .sort_values(
            sort_columns,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


def print_candidate_counts(
    candidates: pd.DataFrame,
) -> None:

    counts = (
        candidates
        .groupby(
            [
                "market",
                "side",
            ],
            as_index=False,
        )
        .agg(
            bets=(
                "game_id",
                "size",
            ),
            wins=(
                "result",
                lambda s:
                    int(
                        (
                            s
                            == "WIN"
                        ).sum()
                    ),
            ),
            losses=(
                "result",
                lambda s:
                    int(
                        (
                            s
                            == "LOSS"
                        ).sum()
                    ),
            ),
            pushes=(
                "result",
                lambda s:
                    int(
                        (
                            s
                            == "PUSH"
                        ).sum()
                    ),
            ),
            profit_units=(
                "profit_units",
                "sum",
            ),
        )
    )

    settled = (
        candidates[
            candidates[
                "result"
            ].isin(
                [
                    "WIN",
                    "LOSS",
                ]
            )
        ]
        .groupby(
            [
                "market",
                "side",
            ]
        )
        .size()
        .rename(
            "settled"
        )
        .reset_index()
    )

    counts = counts.merge(
        settled,
        on=[
            "market",
            "side",
        ],
        how="left",
    )

    counts[
        "settled"
    ] = (
        counts[
            "settled"
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    counts[
        "win_rate"
    ] = (
        counts[
            "wins"
        ]
        / counts[
            "settled"
        ].replace(
            0,
            np.nan,
        )
    )

    counts[
        "roi_per_unit_risk"
    ] = (
        counts[
            "profit_units"
        ]
        / counts[
            "bets"
        ]
    )

    print(
        "\n=== WIDE-OPEN "
        "CANDIDATE RESULTS ==="
    )

    print(
        counts.to_string(
            index=False
        )
    )


def parse_args():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--provider",
        default=None,
        help=(
            "Optional single sportsbook "
            "override for every season. "
            "If omitted, uses DraftKings "
            "for 2021-2023 and ESPN BET "
            "for 2024-2025."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--refresh-cache",
        action="store_true",
    )

    parser.add_argument(
        "--travel-weather-mode",
        choices=[
            "current",
            "none",
        ],
        default="current",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.workers < 1:
        raise ValueError(
            "--workers must be >= 1"
        )

    if args.timeout < 1:
        raise ValueError(
            "--timeout must be >= 1"
        )

    if args.retries < 0:
        raise ValueError(
            "--retries cannot be negative"
        )

    provider_map = {
        season:
            provider_for_season(
                season,
                args.provider,
            )
        for season
        in SEASONS
    }

    print(
        "seasons=2021-2025"
    )

    print(
        "filters=NONE"
    )

    print(
        "candidate_mode="
        "ALL_AVAILABLE_SIDES"
    )

    print(
        "kelly="
        "FULL_UNCAPPED_KELLY"
    )

    print(
        "provider_map="
        + " | ".join(
            (
                f"{season}:"
                f"{provider_map[season]}"
            )
            for season
            in SEASONS
        )
    )

    print(
        "travel_weather_mode="
        f"{args.travel_weather_mode}"
    )

    print(
        "bins="
        f"ev:{EV_BIN} | "
        f"kelly:{KELLY_BIN} | "
        f"win_probability:{PROB_BIN} | "
        f"odds:{int(ODDS_BIN)} | "
        f"spread_line:{SPREAD_LINE_BIN:g} | "
        f"total_line:{TOTAL_LINE_BIN:g}"
    )

    team_map = (
        projection.read_csv(
            TEAM_MAP,
            [
                "team_id",
                "canonical_team",
            ],
            "team map",
        )
    )

    resolver = (
        projection.TeamResolver(
            team_map
        )
    )

    stadium_lookup = (
        projection
        .build_home_stadium_lookup(
            STADIUM_MAP,
            resolver,
        )
    )

    if (
        args.travel_weather_mode
        == "none"
    ):
        coefficients = {
            "margin":
                {},
            "total":
                {},
        }

    else:
        coefficients = (
            projection
            .load_travel_weather_coefficients(
                TW_COEFS
            )
        )

    stats_cache: dict[
        int,
        pd.DataFrame,
    ] = {}

    candidate_frames = []
    audit_frames = []

    for season in SEASONS:
        provider = (
            provider_map[
                season
            ]
        )

        print(
            f"\n=== {season} "
            f"| {provider} ==="
        )

        schedule = (
            load_schedule(
                season
            )
        )

        finals = (
            load_finals(
                season
            )
        )

        features = (
            load_features(
                season
            )
        )

        eligible_ids = (
            set(
                finals[
                    "game_id"
                ]
            )
            & set(
                features[
                    "game_id"
                ]
            )
        )

        schedule = schedule[
            schedule[
                "game_id"
            ].isin(
                eligible_ids
            )
        ].copy()

        print(
            "eligible_regular_games="
            f"{len(schedule)}"
        )

        market = espn_cache(
            season,
            schedule[
                "game_id"
            ].tolist(),
            provider,
            args.workers,
            args.timeout,
            args.retries,
            args.refresh_cache,
        )

        odds_status = (
            market.get(
                "odds_status",
                pd.Series(
                    dtype=str
                ),
            )
        )

        odds_ok = int(
            odds_status.eq(
                "OK"
            ).sum()
        )

        provider_missing = int(
            odds_status.eq(
                "PROVIDER_NOT_AVAILABLE"
            ).sum()
        )

        fetch_errors = int(
            odds_status
            .astype(
                str
            )
            .str.startswith(
                "FETCH_ERROR"
            )
            .sum()
        )

        actual_ok_providers = sorted(
            {
                clean(
                    value
                )
                for value
                in market.loc[
                    odds_status.eq(
                        "OK"
                    ),
                    "provider",
                ]
                if clean(
                    value
                )
            }
        )

        print(
            f"odds_ok={odds_ok} "
            "provider_missing="
            f"{provider_missing} "
            "fetch_errors="
            f"{fetch_errors}"
        )

        print(
            "actual_ok_providers="
            + (
                "|".join(
                    actual_ok_providers
                )
                if actual_ok_providers
                else "NONE"
            )
        )

        (
            audit,
            candidates,
        ) = replay_season(
            season,
            schedule,
            finals,
            features,
            market,
            resolver,
            stadium_lookup,
            coefficients,
            stats_cache,
        )

        audit_frames.append(
            audit
        )

        candidate_frames.append(
            candidates
        )

        projected = (
            int(
                audit[
                    "projection_status"
                ].eq(
                    "OK"
                ).sum()
            )
            if not audit.empty
            else 0
        )

        print(
            "game_audit_rows="
            f"{len(audit)} "
            "projected="
            f"{projected} "
            "candidate_sides="
            f"{len(candidates)}"
        )

    candidates = pd.concat(
        candidate_frames,
        ignore_index=True,
    )

    audit = pd.concat(
        audit_frames,
        ignore_index=True,
    )

    if candidates.empty:
        raise RuntimeError(
            "Historical replay produced "
            "zero candidate sides"
        )

    bad_results = candidates[
        ~candidates[
            "result"
        ].isin(
            [
                "WIN",
                "LOSS",
                "PUSH",
            ]
        )
    ]

    if not bad_results.empty:
        raise RuntimeError(
            "Ungradable candidate rows "
            "found: "
            f"{bad_results.head(10).to_dict('records')}"
        )

    duplicate_key = [
        "season",
        "game_id",
        "market",
        "side",
    ]

    duplicates = candidates[
        candidates.duplicated(
            duplicate_key,
            keep=False,
        )
    ]

    if not duplicates.empty:
        raise RuntimeError(
            "Duplicate candidate sides "
            "found: "
            f"{duplicates[duplicate_key].head(10).to_dict('records')}"
        )

    for season in SEASONS:
        requested = (
            provider_map[
                season
            ]
        )

        frame = candidates[
            pd.to_numeric(
                candidates[
                    "season"
                ],
                errors="coerce",
            ).eq(
                season
            )
        ]

        bad_provider = frame[
            ~frame[
                "provider"
            ].map(
                lambda value:
                    provider_match(
                        clean(
                            value
                        ),
                        requested,
                    )
            )
        ]

        if not bad_provider.empty:
            raise RuntimeError(
                f"{season}: candidate rows "
                "contain wrong sportsbook: "
                f"{bad_provider[['game_id', 'provider']].head(10).to_dict('records')}"
            )

    combined_bins = (
        build_binned_results(
            candidates,
            by_season=False,
        )
    )

    season_bins = (
        build_binned_results(
            candidates,
            by_season=True,
        )
    )

    outputs = {
        "historical_wide_open_candidates_2021_2025.csv":
            candidates,

        "historical_wide_open_bins_2021_2025.csv":
            combined_bins,

        "historical_wide_open_bins_by_season.csv":
            season_bins,

        "historical_wide_open_game_audit_2021_2025.csv":
            audit,
    }

    for (
        filename,
        dataframe,
    ) in outputs.items():

        write_csv(
            dataframe,
            OUT_DIR
            / filename,
        )

    print_candidate_counts(
        candidates
    )

    print(
        "\n=== OUTPUTS ==="
    )

    for filename in outputs:
        print(
            OUT_DIR
            / filename
        )

    print(
        "markets.yaml_read=no"
    )

    print(
        "markets.yaml_modified=no"
    )

    print(
        "selection_thresholds_applied=NONE"
    )

    if (
        args.travel_weather_mode
        == "current"
    ):
        print(
            "NOTE: current travel/weather "
            "coefficients are retrospective "
            "because they were fitted on "
            "2021-2025."
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        raise
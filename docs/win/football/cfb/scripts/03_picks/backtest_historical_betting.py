#!/usr/bin/env python3
"""
Historical CFB betting backtest, 2021-2025.

Uses the live selections.py + picks.py logic for candidate metrics and final
market selection. Replays the current projection structure with historical
pregame market data, ESPN predictor data, prior-week team stats, and historical
travel/weather, then grades bets from historical PBP finals.

Thresholds are learned on 2021-2024 and evaluated on 2025. markets.yaml is
never modified.

Historical limitations: point-in-time FPI and injury snapshots are not stored
for 2021-2025, so those components use the current projection's missing-data
behavior. --travel-weather-mode current uses today's fitted coefficients
retrospectively; use "none" for a leakage-safer sensitivity run.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
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
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CFB_ROOT = SCRIPT_DIR.parents[1]

SCHEDULE_DIR = (
    CFB_ROOT
    / "00_intake"
    / "schedule"
)

TEAM_STATS_DIR = (
    CFB_ROOT
    / "00_intake"
    / "team_stats"
)

PBP_DIR = (
    CFB_ROOT
    / "00_intake"
    / "pbp"
)

HIST_DIR = (
    CFB_ROOT
    / "data"
    / "historical_features"
)

OUT_DIR = (
    CFB_ROOT
    / "data"
    / "historical_betting"
)

CACHE_DIR = (
    OUT_DIR
    / "cache"
)

TEAM_MAP = (
    CFB_ROOT
    / "config"
    / "mapping"
    / "team_map.csv"
)

STADIUM_MAP = (
    CFB_ROOT
    / "config"
    / "mapping"
    / "stadium_map.csv"
)

MARKETS = (
    CFB_ROOT
    / "config"
    / "markets.yaml"
)

SETTINGS = (
    CFB_ROOT
    / "config"
    / "settings.yaml"
)

TW_COEFS = (
    CFB_ROOT
    / "config"
    / "travel_weather_coefficients.csv"
)

PROJ_PATH = (
    CFB_ROOT
    / "scripts"
    / "01_merge"
    / "projection_week1.py"
)

SEL_PATH = (
    CFB_ROOT
    / "scripts"
    / "02_select"
    / "selections.py"
)

PICKS_PATH = (
    CFB_ROOT
    / "scripts"
    / "03_picks"
    / "picks.py"
)


SEASONS = [
    2021,
    2022,
    2023,
    2024,
    2025,
]

TRAIN = [
    2021,
    2022,
    2023,
    2024,
]

HOLDOUT = 2025

SEASON_TYPE = 2

HOME_FIELD = 2.5
DRIVES = 11.5

MARGIN_WEIGHTS = (
    0.36,  # market
    0.28,  # FPI
    0.20,  # ESPN predictor
    0.16,  # team stats
)

TOTAL_MARKET_WEIGHT = 0.75

MARGIN_SD = 14.0
TOTAL_SD = 14.0

ESPN_SYMMETRY_TOL = 0.25

METRICS = [
    "model_probability",
    "edge",
    "ev",
    "kelly",
]


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

KNOWN_PROVIDER_IDS = {
    "draftkings": {
        "40",
        "41",
    },
    "fanduel": {
        "37",
    },
    "caesars": {
        "38",
    },
    "betmgm": {
        "58",
    },
    "espnbet": {
        "68",
    },
}


def load_module(
    name: str,
    path: Path,
):
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import {path}"
        )

    mod = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        name
    ] = mod

    spec.loader.exec_module(
        mod
    )

    return mod


projection = load_module(
    "cfb_projection_hist",
    PROJ_PATH,
)

selections = load_module(
    "cfb_selections_hist",
    SEL_PATH,
)

picks = load_module(
    "cfb_picks_hist",
    PICKS_PATH,
)


def clean(
    value: Any,
) -> str:
    if value is None:
        return ""

    text = str(
        value
    ).strip()

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


def game_id(
    value: Any,
) -> str:
    return re.sub(
        r"\.0$",
        "",
        clean(
            value
        ),
    )


def norm(
    value: Any,
) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        clean(
            value
        ).casefold(),
    )


def fnum(
    value: Any,
) -> float | None:
    try:
        number = float(
            clean(
                value
            )
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


def inum(
    value: Any,
) -> int | None:
    number = fnum(
        value
    )

    if (
        number is not None
        and float(
            number
        ).is_integer()
    ):
        return int(
            number
        )

    return None


def american(
    value: Any,
) -> float | None:
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
            for column
            in required
            if column
            not in df.columns
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
            request = (
                urllib.request.Request(
                    url,
                    headers={
                        "User-Agent":
                            USER_AGENT,
                        "Accept":
                            "application/json",
                    },
                )
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
    provider_id: str,
    requested: str,
) -> bool:

    key = norm(
        requested
    )

    if norm(
        name
    ) == key:
        return True

    return (
        provider_id
        in KNOWN_PROVIDER_IDS.get(
            key,
            set(),
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
            provider_id,
        ) = provider_info(
            item
        )

        if provider_match(
            name,
            provider_id,
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
            provider_name
            or requested,

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

        if "game_id" in cache:
            cache[
                "game_id"
            ] = cache[
                "game_id"
            ].map(
                game_id
            )

            cache = (
                cache.drop_duplicates(
                    "game_id",
                    keep="last",
                )
            )

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

                for gid
                in missing
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

    return cache[
        cache[
            "game_id"
        ].isin(
            gids
        )
    ].copy()


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

    columns = [
        "game_id",
        "sequenceNumber",
        "end.homeScore",
        "end.awayScore",
    ]

    df = pd.read_parquet(
        path,
        columns=columns,
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


def candidate_selection(
    gid: str,
    probabilities: dict[
        str,
        float,
    ],
    market: pd.Series,
    config: dict[
        str,
        Any,
    ],
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

    candidates = {
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

    pick_row = pd.Series(
        {
            "game_id":
                gid,

            **candidates,
        }
    )

    chosen = {}

    for market_name in picks.MARKETS:
        chosen.update(
            picks.evaluate_market(
                pick_row,
                market_name,
                config,
            )
        )

    return {
        **candidates,
        **chosen,
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
    selection: str,
    line: float | None,
    home_final: float,
    away_final: float,
) -> str:

    side = selection.upper()

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
            value = math.nan

        if math.isfinite(
            value
        ):
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


def bet_row(
    game: dict[str, Any],
    market: str,
    prefix: str,
    chosen: dict[str, Any],
) -> dict[str, Any]:

    odds = fnum(
        chosen.get(
            f"{prefix}_"
            "odds_american"
        )
    )

    line = (
        fnum(
            chosen.get(
                f"{prefix}_line"
            )
        )
        if prefix in {
            "spread",
            "total",
        }
        else None
    )

    if odds is None:
        raise RuntimeError(
            "Selected bet missing "
            f"odds: {game['game_id']} "
            f"{market}"
        )

    result = grade(
        market,
        clean(
            chosen.get(
                f"{prefix}_selection"
            )
        ),
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
            odds
        )
        win = 1.0

    elif result == "LOSS":
        profit = -1.0
        win = 0.0

    elif result == "PUSH":
        profit = 0.0
        win = np.nan

    else:
        profit = np.nan
        win = np.nan

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
            market,

        "selection":
            clean(
                chosen.get(
                    f"{prefix}_selection"
                )
            ),

        "line":
            line,

        "odds_american":
            odds,

        "model_probability":
            fnum(
                chosen.get(
                    f"{prefix}_"
                    "model_probability"
                )
            ),

        "implied_probability":
            fnum(
                chosen.get(
                    f"{prefix}_"
                    "implied_probability"
                )
            ),

        "edge":
            fnum(
                chosen.get(
                    f"{prefix}_edge"
                )
            ),

        "ev":
            fnum(
                chosen.get(
                    f"{prefix}_ev"
                )
            ),

        "full_kelly":
            fnum(
                chosen.get(
                    f"{prefix}_"
                    "full_kelly"
                )
            ),

        "kelly":
            fnum(
                chosen.get(
                    f"{prefix}_kelly"
                )
            ),

        "home_final":
            game[
                "home_final"
            ],

        "away_final":
            game[
                "away_final"
            ],

        "result":
            result,

        "win_indicator":
            win,

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
    config,
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

    games = []
    bets = []

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
                f"{season} week {week}: "
                f"skipped - {prior_source}"
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
                or gid
                not in market_lookup.index
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

            base_record = {
                "season":
                    season,

                "season_type":
                    SEASON_TYPE,

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

                "game_time":
                    clean(
                        sched_row.get(
                            "game_time"
                        )
                    ),

                "away_team":
                    away_team,

                "home_team":
                    home_team,

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
            }

            if prior is None:
                games.append(
                    {
                        **base_record,
                        "projection_status":
                            prior_source,
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

            if espn_home is not None:
                if (
                    espn_away is None
                    or abs(
                        espn_home
                        + espn_away
                    )
                    <= ESPN_SYMMETRY_TOL
                ):
                    espn_margin = (
                        espn_home
                    )

            # No historical point-in-time
            # FPI snapshot is available.
            fpi_margin = None

            (
                blended_margin,
                margin_weights,
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
                games.append(
                    {
                        **base_record,
                        "projection_status":
                            "NO_MARGIN_COMPONENT",
                    }
                )

                continue

            (
                travel_features,
                travel_adjustment,
                travel_used,
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

            # No historical point-in-time
            # injury snapshot is available.
            injury_adjustment = 0.0

            predicted_margin_before_travel = (
                float(
                    blended_margin
                )
                + injury_adjustment
            )

            predicted_margin = (
                predicted_margin_before_travel
                + travel_adjustment
            )

            (
                predicted_total,
                total_weights,
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
                games.append(
                    {
                        **base_record,
                        "projection_status":
                            "NO_TOTAL_COMPONENT",
                    }
                )

                continue

            predicted_total_before_weather = (
                float(
                    predicted_total
                )
            )

            (
                weather_features,
                weather_exposed,
                weather_adjustment,
                weather_used,
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
                predicted_total_before_weather
                + weather_adjustment
            )

            predicted_total = max(
                predicted_total,
                abs(
                    predicted_margin
                )
                + 2.0,
            )

            predicted_home_score = (
                predicted_total
                + predicted_margin
            ) / 2.0

            predicted_away_score = (
                predicted_total
                - predicted_margin
            ) / 2.0

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

            chosen = (
                candidate_selection(
                    gid,
                    probabilities,
                    market_row,
                    config,
                )
            )

            home_final = float(
                final_row[
                    "home_final"
                ]
            )

            away_final = float(
                final_row[
                    "away_final"
                ]
            )

            record = {
                **base_record,

                "provider":
                    clean(
                        market_row.get(
                            "provider"
                        )
                    ),

                "provider_id":
                    clean(
                        market_row.get(
                            "provider_id"
                        )
                    ),

                "projection_status":
                    "OK",

                "prior_source":
                    prior_source,

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

                "home_final":
                    home_final,

                "away_final":
                    away_final,

                "actual_margin":
                    home_final
                    - away_final,

                "actual_total":
                    home_final
                    + away_final,

                "home_spread":
                    home_spread,

                "away_spread":
                    fnum(
                        market_row.get(
                            "away_spread"
                        )
                    ),

                "market_total":
                    total_line,

                "home_moneyline_american":
                    fnum(
                        market_row.get(
                            "home_moneyline_american"
                        )
                    ),

                "away_moneyline_american":
                    fnum(
                        market_row.get(
                            "away_moneyline_american"
                        )
                    ),

                "home_spread_american":
                    fnum(
                        market_row.get(
                            "home_spread_american"
                        )
                    ),

                "away_spread_american":
                    fnum(
                        market_row.get(
                            "away_spread_american"
                        )
                    ),

                "over_american":
                    fnum(
                        market_row.get(
                            "over_american"
                        )
                    ),

                "under_american":
                    fnum(
                        market_row.get(
                            "under_american"
                        )
                    ),

                "market_home_margin":
                    market_margin,

                "espn_home_margin":
                    espn_margin,

                "prior_home_margin":
                    prior_context[
                        "margin"
                    ],

                "home_prior_team_weeks":
                    prior_context[
                        "home_weeks"
                    ],

                "away_prior_team_weeks":
                    prior_context[
                        "away_weeks"
                    ],

                "home_prior_fallback":
                    prior_context[
                        "home_fallback"
                    ],

                "away_prior_fallback":
                    prior_context[
                        "away_fallback"
                    ],

                "market_margin_weight_used":
                    margin_weights[
                        0
                    ],

                "fpi_margin_weight_used":
                    margin_weights[
                        1
                    ],

                "espn_margin_weight_used":
                    margin_weights[
                        2
                    ],

                "team_margin_weight_used":
                    margin_weights[
                        3
                    ],

                "predicted_margin_before_travel":
                    predicted_margin_before_travel,

                "travel_margin_adjustment":
                    travel_adjustment,

                "travel_features_used":
                    travel_used,

                "predicted_margin":
                    predicted_margin,

                "prior_total":
                    prior_context[
                        "total"
                    ],

                "market_total_weight_used":
                    total_weights[
                        0
                    ],

                "team_total_weight_used":
                    total_weights[
                        1
                    ],

                "predicted_total_before_weather":
                    predicted_total_before_weather,

                "weather_exposed":
                    int(
                        weather_exposed
                    ),

                "weather_total_adjustment":
                    weather_adjustment,

                "weather_features_used":
                    weather_used,

                "predicted_total":
                    predicted_total,

                "predicted_home_score":
                    predicted_home_score,

                "predicted_away_score":
                    predicted_away_score,

                **probabilities,
                **travel_features,
                **weather_features,
            }

            for column in (
                picks.selection_columns()
            ):
                record[
                    column
                ] = chosen.get(
                    column,
                    np.nan,
                )

            games.append(
                record
            )

            for (
                market_name,
                prefix,
            ) in [
                (
                    "moneyline",
                    "ml",
                ),
                (
                    "spread",
                    "spread",
                ),
                (
                    "total",
                    "total",
                ),
            ]:
                if inum(
                    chosen.get(
                        f"{prefix}_selected"
                    )
                ) == 1:

                    bets.append(
                        bet_row(
                            record,
                            market_name,
                            prefix,
                            chosen,
                        )
                    )

    return (
        pd.DataFrame(
            games
        ),
        pd.DataFrame(
            bets
        ),
    )


def perf(
    df: pd.DataFrame,
    season: str,
    market: str,
) -> dict[str, Any]:

    graded = df[
        df[
            "result"
        ].isin(
            [
                "WIN",
                "LOSS",
                "PUSH",
            ]
        )
    ]

    settled = graded[
        graded[
            "result"
        ].isin(
            [
                "WIN",
                "LOSS",
            ]
        )
    ]

    bets = len(
        graded
    )

    wins = int(
        graded[
            "result"
        ].eq(
            "WIN"
        ).sum()
    )

    profit = float(
        pd.to_numeric(
            graded[
                "profit_units"
            ],
            errors="coerce",
        ).sum()
    )

    return {
        "season":
            season,

        "market":
            market,

        "bets":
            bets,

        "wins":
            wins,

        "losses":
            int(
                graded[
                    "result"
                ].eq(
                    "LOSS"
                ).sum()
            ),

        "pushes":
            int(
                graded[
                    "result"
                ].eq(
                    "PUSH"
                ).sum()
            ),

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
    }


def summary_table(
    bets: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    seasons = sorted(
        pd.to_numeric(
            bets[
                "season"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(
            int
        )
        .unique()
    )

    for season in seasons:
        frame = bets[
            pd.to_numeric(
                bets[
                    "season"
                ],
                errors="coerce",
            ).eq(
                season
            )
        ]

        rows.append(
            perf(
                frame,
                str(
                    season
                ),
                "ALL",
            )
        )

        for market in [
            "moneyline",
            "spread",
            "total",
        ]:
            rows.append(
                perf(
                    frame[
                        frame[
                            "market"
                        ].eq(
                            market
                        )
                    ],
                    str(
                        season
                    ),
                    market,
                )
            )

    rows.append(
        perf(
            bets,
            "ALL",
            "ALL",
        )
    )

    for market in [
        "moneyline",
        "spread",
        "total",
    ]:
        rows.append(
            perf(
                bets[
                    bets[
                        "market"
                    ].eq(
                        market
                    )
                ],
                "ALL",
                market,
            )
        )

    return pd.DataFrame(
        rows
    )


def corr(
    x: pd.Series,
    y: pd.Series,
    method: str,
) -> float:

    frame = pd.DataFrame(
        {
            "x":
                pd.to_numeric(
                    x,
                    errors="coerce",
                ),

            "y":
                pd.to_numeric(
                    y,
                    errors="coerce",
                ),
        }
    ).dropna()

    if (
        len(
            frame
        ) < 3
        or frame[
            "x"
        ].nunique() < 2
        or frame[
            "y"
        ].nunique() < 2
    ):
        return np.nan

    return float(
        frame[
            "x"
        ].corr(
            frame[
                "y"
            ],
            method=method,
        )
    )


def correlation_table(
    bets: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    groups = [
        (
            "ALL",
            bets,
        ),
        *[
            (
                market,
                bets[
                    bets[
                        "market"
                    ].eq(
                        market
                    )
                ],
            )
            for market in [
                "moneyline",
                "spread",
                "total",
            ]
        ],
    ]

    for (
        market,
        frame,
    ) in groups:

        graded = frame[
            frame[
                "result"
            ].isin(
                [
                    "WIN",
                    "LOSS",
                    "PUSH",
                ]
            )
        ]

        settled = graded[
            graded[
                "result"
            ].isin(
                [
                    "WIN",
                    "LOSS",
                ]
            )
        ]

        for metric in METRICS:
            rows.append(
                {
                    "market":
                        market,

                    "metric":
                        metric,

                    "bets":
                        len(
                            graded
                        ),

                    "settled_bets":
                        len(
                            settled
                        ),

                    "pearson_vs_win":
                        corr(
                            settled[
                                metric
                            ],
                            settled[
                                "win_indicator"
                            ],
                            "pearson",
                        ),

                    "spearman_vs_win":
                        corr(
                            settled[
                                metric
                            ],
                            settled[
                                "win_indicator"
                            ],
                            "spearman",
                        ),

                    "pearson_vs_profit":
                        corr(
                            graded[
                                metric
                            ],
                            graded[
                                "profit_units"
                            ],
                            "pearson",
                        ),

                    "spearman_vs_profit":
                        corr(
                            graded[
                                metric
                            ],
                            graded[
                                "profit_units"
                            ],
                            "spearman",
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def bin_table(
    bets: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for market in [
        "moneyline",
        "spread",
        "total",
    ]:
        base = bets[
            bets[
                "market"
            ].eq(
                market
            )
        ].copy()

        for metric in METRICS:
            frame = base.copy()

            frame[
                metric
            ] = pd.to_numeric(
                frame[
                    metric
                ],
                errors="coerce",
            )

            frame = frame.dropna(
                subset=[
                    metric,
                    "profit_units",
                ]
            )

            if frame.empty:
                continue

            if metric == (
                "model_probability"
            ):
                frame[
                    "bucket"
                ] = pd.cut(
                    frame[
                        metric
                    ],
                    bins=np.arange(
                        0,
                        1.0001,
                        0.05,
                    ),
                    include_lowest=True,
                    duplicates="drop",
                )

            else:
                q = min(
                    5,
                    frame[
                        metric
                    ].nunique(),
                )

                if q >= 2:
                    frame[
                        "bucket"
                    ] = pd.qcut(
                        frame[
                            metric
                        ],
                        q=q,
                        duplicates="drop",
                    )

                else:
                    frame[
                        "bucket"
                    ] = "ALL"

            grouped = frame.groupby(
                "bucket",
                observed=True,
                dropna=True,
            )

            for (
                bucket,
                group,
            ) in grouped:

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

                wins = int(
                    settled[
                        "result"
                    ].eq(
                        "WIN"
                    ).sum()
                )

                profit = float(
                    group[
                        "profit_units"
                    ].sum()
                )

                rows.append(
                    {
                        "market":
                            market,

                        "metric":
                            metric,

                        "bucket":
                            str(
                                bucket
                            ),

                        "bets":
                            len(
                                group
                            ),

                        "wins":
                            wins,

                        "losses":
                            int(
                                settled[
                                    "result"
                                ].eq(
                                    "LOSS"
                                ).sum()
                            ),

                        "pushes":
                            int(
                                group[
                                    "result"
                                ].eq(
                                    "PUSH"
                                ).sum()
                            ),

                        "avg_metric":
                            float(
                                group[
                                    metric
                                ].mean()
                            ),

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
                            profit
                            / len(
                                group
                            ),

                        "calibration_error":
                            (
                                wins
                                / len(
                                    settled
                                )
                                - float(
                                    group[
                                        metric
                                    ].mean()
                                )
                                if (
                                    metric
                                    == "model_probability"
                                    and len(
                                        settled
                                    )
                                )
                                else np.nan
                            ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def validate_uniform_thresholds(
    config: dict[str, Any],
) -> None:

    keys = [
        "min_ev",
        "min_edge",
        "min_kelly",
        "min_model_prob",
    ]

    for (
        market,
        spec,
    ) in picks.MARKETS.items():

        values = []

        for side in spec[
            "sides"
        ]:
            thresholds = (
                config[
                    "markets"
                ][
                    market
                ][
                    "sides"
                ][
                    side
                ][
                    "thresholds"
                ]
            )

            values.append(
                tuple(
                    float(
                        thresholds[
                            key
                        ]
                    )
                    for key
                    in keys
                )
            )

        if len(
            set(
                values
            )
        ) > 1:
            raise RuntimeError(
                f"{market}: "
                "side-specific current "
                "minimum thresholds are "
                "not supported by this "
                "historical learner"
            )


def floors(
    config: dict[str, Any],
    market: str,
) -> dict[str, float]:

    side = next(
        iter(
            picks.MARKETS[
                market
            ][
                "sides"
            ]
        )
    )

    thresholds = (
        config[
            "markets"
        ][
            market
        ][
            "sides"
        ][
            side
        ][
            "thresholds"
        ]
    )

    return {
        "model_probability":
            float(
                thresholds[
                    "min_model_prob"
                ]
            ),

        "edge":
            float(
                thresholds[
                    "min_edge"
                ]
            ),

        "ev":
            float(
                thresholds[
                    "min_ev"
                ]
            ),

        "kelly":
            float(
                thresholds[
                    "min_kelly"
                ]
            ),
    }


def cutoffs(
    series: pd.Series,
    floor: float,
) -> list[float]:

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    values = values[
        values.ge(
            floor
            - 1e-12
        )
    ]

    if values.empty:
        return [
            floor
        ]

    generated = [
        floor
    ]

    for quantile in np.linspace(
        0,
        0.90,
        7,
    ):
        generated.append(
            float(
                values.quantile(
                    quantile
                )
            )
        )

    return sorted(
        {
            round(
                max(
                    floor,
                    value,
                ),
                8,
            )
            for value
            in generated
        }
    )


def profit_stats(
    values: np.ndarray,
) -> dict[str, float]:

    count = len(
        values
    )

    if not count:
        return {
            "bets":
                0,

            "profit":
                0.0,

            "roi":
                np.nan,

            "lcb":
                np.nan,
        }

    profit = float(
        np.sum(
            values
        )
    )

    roi = float(
        np.mean(
            values
        )
    )

    if count >= 2:
        standard_error = (
            float(
                np.std(
                    values,
                    ddof=1,
                )
            )
            / math.sqrt(
                count
            )
        )

        lcb = (
            roi
            - 1.96
            * standard_error
        )

    else:
        lcb = np.nan

    return {
        "bets":
            count,

        "profit":
            profit,

        "roi":
            roi,

        "lcb":
            lcb,
    }


def apply_thresholds(
    df: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:

    mask = np.ones(
        len(
            df
        ),
        dtype=bool,
    )

    for metric in METRICS:
        values = pd.to_numeric(
            df[
                metric
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        mask &= (
            np.isfinite(
                values
            )
            & (
                values
                >= thresholds[
                    metric
                ]
                - 1e-12
            )
        )

    return df.loc[
        mask
    ].copy()


def threshold_market(
    market: str,
    bets: pd.DataFrame,
    config: dict[str, Any],
    min_train: int,
):

    frame = bets[
        bets[
            "market"
        ].eq(
            market
        )
    ].copy()

    frame[
        "season"
    ] = pd.to_numeric(
        frame[
            "season"
        ],
        errors="coerce",
    )

    train = frame[
        frame[
            "season"
        ].isin(
            TRAIN
        )
    ].copy()

    holdout = frame[
        frame[
            "season"
        ].eq(
            HOLDOUT
        )
    ].copy()

    base = floors(
        config,
        market,
    )

    baseline_train = (
        apply_thresholds(
            train,
            base,
        )
    )

    baseline_holdout = (
        apply_thresholds(
            holdout,
            base,
        )
    )

    baseline_train_stats = (
        profit_stats(
            pd.to_numeric(
                baseline_train[
                    "profit_units"
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(
                dtype=float
            )
        )
    )

    baseline_holdout_stats = (
        profit_stats(
            pd.to_numeric(
                baseline_holdout[
                    "profit_units"
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(
                dtype=float
            )
        )
    )

    cutoff_sets = {
        metric:
            cutoffs(
                train[
                    metric
                ],
                base[
                    metric
                ],
            )
        for metric
        in METRICS
    }

    arrays = {
        metric:
            pd.to_numeric(
                train[
                    metric
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )
        for metric
        in METRICS
    }

    profits = pd.to_numeric(
        train[
            "profit_units"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    valid = np.isfinite(
        profits
    )

    rows = []

    combinations = itertools.product(
        cutoff_sets[
            "model_probability"
        ],
        cutoff_sets[
            "edge"
        ],
        cutoff_sets[
            "ev"
        ],
        cutoff_sets[
            "kelly"
        ],
    )

    for (
        probability_cut,
        edge_cut,
        ev_cut,
        kelly_cut,
    ) in combinations:

        mask = (
            valid
            & np.isfinite(
                arrays[
                    "model_probability"
                ]
            )
            & np.isfinite(
                arrays[
                    "edge"
                ]
            )
            & np.isfinite(
                arrays[
                    "ev"
                ]
            )
            & np.isfinite(
                arrays[
                    "kelly"
                ]
            )
        )

        mask &= (
            arrays[
                "model_probability"
            ]
            >= probability_cut
            - 1e-12
        )

        mask &= (
            arrays[
                "edge"
            ]
            >= edge_cut
            - 1e-12
        )

        mask &= (
            arrays[
                "ev"
            ]
            >= ev_cut
            - 1e-12
        )

        mask &= (
            arrays[
                "kelly"
            ]
            >= kelly_cut
            - 1e-12
        )

        stats = profit_stats(
            profits[
                mask
            ]
        )

        if stats[
            "bets"
        ] < min_train:
            continue

        settled_mask = (
            mask
            & train[
                "result"
            ].isin(
                [
                    "WIN",
                    "LOSS",
                ]
            ).to_numpy(
                dtype=bool
            )
        )

        settled = train.loc[
            settled_mask
        ]

        win_rate = (
            float(
                settled[
                    "result"
                ].eq(
                    "WIN"
                ).mean()
            )
            if len(
                settled
            )
            else np.nan
        )

        rows.append(
            {
                "market":
                    market,

                "min_model_prob":
                    probability_cut,

                "min_edge":
                    edge_cut,

                "min_ev":
                    ev_cut,

                "min_kelly":
                    kelly_cut,

                "train_bets":
                    stats[
                        "bets"
                    ],

                "train_win_rate":
                    win_rate,

                "train_profit_units":
                    stats[
                        "profit"
                    ],

                "train_roi":
                    stats[
                        "roi"
                    ],

                "train_roi_lcb95":
                    stats[
                        "lcb"
                    ],
            }
        )

    search = pd.DataFrame(
        rows
    )

    if search.empty:
        best = base.copy()

        reason = (
            "NO_CANDIDATE_MET_"
            "MIN_TRAIN_BETS"
        )

    else:
        search = (
            search
            .sort_values(
                [
                    "train_roi_lcb95",
                    "train_roi",
                    "train_bets",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
                kind="stable",
            )
            .reset_index(
                drop=True
            )
        )

        row = search.iloc[
            0
        ]

        best = {
            "model_probability":
                float(
                    row[
                        "min_model_prob"
                    ]
                ),

            "edge":
                float(
                    row[
                        "min_edge"
                    ]
                ),

            "ev":
                float(
                    row[
                        "min_ev"
                    ]
                ),

            "kelly":
                float(
                    row[
                        "min_kelly"
                    ]
                ),
        }

        reason = (
            "MAX_TRAIN_ROI_LCB95"
        )

    recommended_train = (
        apply_thresholds(
            train,
            best,
        )
    )

    recommended_holdout = (
        apply_thresholds(
            holdout,
            best,
        )
    )

    recommended_train_stats = (
        profit_stats(
            pd.to_numeric(
                recommended_train[
                    "profit_units"
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(
                dtype=float
            )
        )
    )

    recommended_holdout_stats = (
        profit_stats(
            pd.to_numeric(
                recommended_holdout[
                    "profit_units"
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(
                dtype=float
            )
        )
    )

    settled = recommended_holdout[
        recommended_holdout[
            "result"
        ].isin(
            [
                "WIN",
                "LOSS",
            ]
        )
    ]

    holdout_win_rate = (
        float(
            settled[
                "result"
            ].eq(
                "WIN"
            ).mean()
        )
        if len(
            settled
        )
        else np.nan
    )

    changed = any(
        best[
            metric
        ]
        > base[
            metric
        ]
        + 1e-12
        for metric
        in METRICS
    )

    if not changed:
        status = (
            "KEEP_CURRENT"
        )

    elif (
        recommended_holdout_stats[
            "bets"
        ] < 20
    ):
        status = (
            "INSUFFICIENT_HOLDOUT"
        )

    elif (
        recommended_train_stats[
            "lcb"
        ]
        > baseline_train_stats[
            "lcb"
        ]
        and recommended_holdout_stats[
            "roi"
        ]
        >= baseline_holdout_stats[
            "roi"
        ]
    ):
        status = (
            "SUPPORTED"
        )

    else:
        status = (
            "REJECTED_HOLDOUT"
        )

    recommendation = {
        "market":
            market,

        "status":
            status,

        "selection_reason":
            reason,

        "train_seasons":
            "2021-2024",

        "holdout_season":
            2025,

        "base_min_model_prob":
            base[
                "model_probability"
            ],

        "base_min_edge":
            base[
                "edge"
            ],

        "base_min_ev":
            base[
                "ev"
            ],

        "base_min_kelly":
            base[
                "kelly"
            ],

        "recommended_min_model_prob":
            best[
                "model_probability"
            ],

        "recommended_min_edge":
            best[
                "edge"
            ],

        "recommended_min_ev":
            best[
                "ev"
            ],

        "recommended_min_kelly":
            best[
                "kelly"
            ],

        "baseline_train_bets":
            baseline_train_stats[
                "bets"
            ],

        "baseline_train_roi":
            baseline_train_stats[
                "roi"
            ],

        "baseline_train_roi_lcb95":
            baseline_train_stats[
                "lcb"
            ],

        "recommended_train_bets":
            recommended_train_stats[
                "bets"
            ],

        "recommended_train_roi":
            recommended_train_stats[
                "roi"
            ],

        "recommended_train_roi_lcb95":
            recommended_train_stats[
                "lcb"
            ],

        "baseline_holdout_bets":
            baseline_holdout_stats[
                "bets"
            ],

        "baseline_holdout_roi":
            baseline_holdout_stats[
                "roi"
            ],

        "recommended_holdout_bets":
            recommended_holdout_stats[
                "bets"
            ],

        "recommended_holdout_win_rate":
            holdout_win_rate,

        "recommended_holdout_profit_units":
            recommended_holdout_stats[
                "profit"
            ],

        "recommended_holdout_roi":
            recommended_holdout_stats[
                "roi"
            ],
    }

    return (
        search,
        recommendation,
    )


def learn_thresholds(
    bets: pd.DataFrame,
    config: dict[str, Any],
    min_train: int,
):

    validate_uniform_thresholds(
        config
    )

    searches = []
    recommendations = []

    for market in [
        "moneyline",
        "spread",
        "total",
    ]:
        (
            search,
            recommendation,
        ) = threshold_market(
            market,
            bets,
            config,
            min_train,
        )

        searches.append(
            search
        )

        recommendations.append(
            recommendation
        )

    search_output = (
        pd.concat(
            searches,
            ignore_index=True,
        )
        if searches
        else pd.DataFrame()
    )

    recommendation_output = (
        pd.DataFrame(
            recommendations
        )
    )

    return (
        search_output,
        recommendation_output,
    )


def sportsbook_from_settings() -> str:

    if not SETTINGS.is_file():
        return "draftkings"

    with SETTINGS.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = (
            yaml.safe_load(
                handle
            )
            or {}
        )

    return (
        clean(
            data.get(
                "sportsbook"
            )
        )
        or "draftkings"
    )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--provider",
        default=None,
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

    parser.add_argument(
        "--min-train-bets",
        type=int,
        default=50,
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

    if args.min_train_bets < 2:
        raise ValueError(
            "--min-train-bets must be >= 2"
        )

    provider = (
        clean(
            args.provider
        )
        or sportsbook_from_settings()
    )

    print(
        f"provider={provider}"
    )

    print(
        "seasons=2021-2025"
    )

    print(
        "threshold_train=2021-2024"
    )

    print(
        "threshold_holdout=2025"
    )

    print(
        "travel_weather_mode="
        f"{args.travel_weather_mode}"
    )

    config = (
        picks.normalize_config(
            picks.load_yaml(
                MARKETS
            )
        )
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

    stats_cache = {}

    game_frames = []
    bet_frames = []

    for season in SEASONS:
        print(
            f"\n=== {season} ==="
        )

        schedule = load_schedule(
            season
        )

        finals = load_finals(
            season
        )

        features = load_features(
            season
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

        (
            games,
            bets,
        ) = replay_season(
            season,
            schedule,
            finals,
            features,
            market,
            resolver,
            stadium_lookup,
            config,
            coefficients,
            stats_cache,
        )

        game_frames.append(
            games
        )

        bet_frames.append(
            bets
        )

        projected = (
            int(
                games[
                    "projection_status"
                ].eq(
                    "OK"
                ).sum()
            )
            if not games.empty
            else 0
        )

        print(
            "game_audit_rows="
            f"{len(games)} "
            "projected="
            f"{projected} "
            "selected_bets="
            f"{len(bets)}"
        )

    games = pd.concat(
        game_frames,
        ignore_index=True,
    )

    bets = pd.concat(
        bet_frames,
        ignore_index=True,
    )

    if bets.empty:
        raise RuntimeError(
            "Historical replay produced "
            "zero selected bets. Inspect "
            "cache odds_status/provider "
            "coverage."
        )

    bad = bets[
        ~bets[
            "result"
        ].isin(
            [
                "WIN",
                "LOSS",
                "PUSH",
            ]
        )
    ]

    if not bad.empty:
        raise RuntimeError(
            "Ungradable bets found: "
            f"{bad[['game_id', 'market', 'result']].head(10).to_dict('records')}"
        )

    summary = summary_table(
        bets
    )

    correlations = correlation_table(
        bets
    )

    bins = bin_table(
        bets
    )

    (
        threshold_search,
        recommendations,
    ) = learn_thresholds(
        bets,
        config,
        args.min_train_bets,
    )

    outputs = {
        "historical_betting_games_2021_2025.csv":
            games,

        "historical_bets_2021_2025.csv":
            bets,

        "historical_betting_summary.csv":
            summary,

        "historical_metric_correlations.csv":
            correlations,

        "historical_metric_bins.csv":
            bins,

        "historical_threshold_search.csv":
            threshold_search,

        "historical_threshold_recommendations.csv":
            recommendations,
    }

    for (
        name,
        dataframe,
    ) in outputs.items():
        write_csv(
            dataframe,
            OUT_DIR
            / name,
        )

    print(
        "\n=== SUMMARY ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\n=== CORRELATIONS ==="
    )

    print(
        correlations.to_string(
            index=False
        )
    )

    print(
        "\n=== THRESHOLD "
        "RECOMMENDATIONS ==="
    )

    print(
        recommendations.to_string(
            index=False
        )
    )

    print(
        "\noutputs="
        f"{OUT_DIR}"
    )

    print(
        "markets.yaml_modified=no"
    )

    if (
        args.travel_weather_mode
        == "current"
    ):
        print(
            "NOTE: travel/weather "
            "coefficients are retrospective "
            "because the current coefficients "
            "were fitted on 2021-2025."
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
#!/usr/bin/env python3
"""
compare_pick_preferences.py

Historical CFB pick-preference comparison, 2021-2025.

Compares the three supported live pick preferences:

    best_ev
    best_prob
    best_kelly

The comparison is performed independently for:

    moneyline
    spread
    total

It also reports side-level results:

    moneyline: HOME / AWAY
    spread:    HOME / AWAY
    total:     OVER / UNDER

IMPORTANT:
- Uses the CURRENT thresholds and market restrictions from markets.yaml.
- Does NOT modify markets.yaml.
- Uses the same historical projection/backtest machinery as
  backtest_historical_betting.py.
- Uses the same historical sportsbook mapping:
      2021-2023 = DraftKings
      2024-2025 = ESPN BET
- Uses exact sportsbook-name validation from the main backtest.
- Uses cached ESPN historical odds/predictor data when available.
- Defaults to current fitted travel/weather coefficients.
- --travel-weather-mode none can be used as a sensitivity test.

Outputs:

docs/win/football/cfb/data/historical_betting/
    historical_preference_bets_2021_2025.csv
    historical_preference_comparison.csv
    historical_preference_comparison_by_season.csv
    historical_preference_rankings.csv
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import sys

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

BACKTEST_PATH = (
    SCRIPT_DIR
    / "backtest_historical_betting.py"
)

PREFERENCES = [
    "best_ev",
    "best_prob",
    "best_kelly",
]

MARKETS = [
    "moneyline",
    "spread",
    "total",
]


def load_backtest_module():
    if not BACKTEST_PATH.is_file():
        raise FileNotFoundError(
            BACKTEST_PATH
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            "cfb_historical_backtest",
            BACKTEST_PATH,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Unable to import "
            f"{BACKTEST_PATH}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        "cfb_historical_backtest"
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


bt = load_backtest_module()


def clean(
    value: Any,
) -> str:
    return bt.clean(
        value
    )


def numeric(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def config_for_preference(
    base_config: dict[str, Any],
    preference: str,
) -> dict[str, Any]:
    config = copy.deepcopy(
        base_config
    )

    for market in MARKETS:
        config[
            "markets"
        ][
            market
        ][
            "pick_preference"
        ] = preference

    return config


def validate_provider_bets(
    bets: pd.DataFrame,
    provider_map: dict[int, str],
) -> None:
    if bets.empty:
        return

    for season in bt.SEASONS:
        requested = (
            provider_map[
                season
            ]
        )

        frame = bets[
            numeric(
                bets[
                    "season"
                ]
            ).eq(
                season
            )
        ]

        if frame.empty:
            continue

        bad = frame[
            ~frame[
                "provider"
            ].map(
                lambda value:
                    bt.provider_match(
                        clean(
                            value
                        ),
                        requested,
                    )
            )
        ]

        if not bad.empty:
            examples = (
                bad[
                    [
                        "game_id",
                        "provider",
                        "market",
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
                f"{season}: wrong sportsbook "
                f"in selected bets: {examples}"
            )


def performance(
    frame: pd.DataFrame,
) -> dict[str, Any]:
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
    ].copy()

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

    settled_bets = len(
        settled
    )

    wins = int(
        settled[
            "result"
        ].eq(
            "WIN"
        ).sum()
    )

    losses = int(
        settled[
            "result"
        ].eq(
            "LOSS"
        ).sum()
    )

    pushes = int(
        graded[
            "result"
        ].eq(
            "PUSH"
        ).sum()
    )

    profit = float(
        numeric(
            graded[
                "profit_units"
            ]
        ).sum()
    )

    avg_probability = (
        float(
            numeric(
                graded[
                    "model_probability"
                ]
            ).mean()
        )
        if bets
        else np.nan
    )

    avg_edge = (
        float(
            numeric(
                graded[
                    "edge"
                ]
            ).mean()
        )
        if bets
        else np.nan
    )

    avg_ev = (
        float(
            numeric(
                graded[
                    "ev"
                ]
            ).mean()
        )
        if bets
        else np.nan
    )

    avg_kelly = (
        float(
            numeric(
                graded[
                    "kelly"
                ]
            ).mean()
        )
        if bets
        else np.nan
    )

    avg_odds = (
        float(
            numeric(
                graded[
                    "odds_american"
                ]
            ).mean()
        )
        if bets
        else np.nan
    )

    return {
        "bets":
            bets,

        "settled_bets":
            settled_bets,

        "wins":
            wins,

        "losses":
            losses,

        "pushes":
            pushes,

        "win_rate":
            (
                wins
                / settled_bets
                if settled_bets
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

        "avg_model_probability":
            avg_probability,

        "avg_edge":
            avg_edge,

        "avg_ev":
            avg_ev,

        "avg_kelly":
            avg_kelly,

        "avg_odds_american":
            avg_odds,
    }


def period_frames(
    bets: pd.DataFrame,
):
    season_values = numeric(
        bets[
            "season"
        ]
    )

    return [
        (
            "TRAIN_2021_2024",
            bets[
                season_values.isin(
                    bt.TRAIN
                )
            ],
        ),
        (
            "HOLDOUT_2025",
            bets[
                season_values.eq(
                    bt.HOLDOUT
                )
            ],
        ),
        (
            "ALL_2021_2025",
            bets,
        ),
    ]


def build_period_comparison(
    bets: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        period,
        period_frame,
    ) in period_frames(
        bets
    ):
        for preference in PREFERENCES:
            preference_frame = (
                period_frame[
                    period_frame[
                        "pick_preference"
                    ].eq(
                        preference
                    )
                ]
            )

            for market in MARKETS:
                market_frame = (
                    preference_frame[
                        preference_frame[
                            "market"
                        ].eq(
                            market
                        )
                    ]
                )

                rows.append(
                    {
                        "period":
                            period,

                        "market":
                            market,

                        "selection":
                            "ALL",

                        "pick_preference":
                            preference,

                        **performance(
                            market_frame
                        ),
                    }
                )

                selections = sorted(
                    {
                        clean(
                            value
                        )
                        for value
                        in market_frame[
                            "selection"
                        ]
                        if clean(
                            value
                        )
                    }
                )

                for selection in selections:
                    side_frame = (
                        market_frame[
                            market_frame[
                                "selection"
                            ].map(
                                clean
                            ).eq(
                                selection
                            )
                        ]
                    )

                    rows.append(
                        {
                            "period":
                                period,

                            "market":
                                market,

                            "selection":
                                selection,

                            "pick_preference":
                                preference,

                            **performance(
                                side_frame
                            ),
                        }
                    )

    return pd.DataFrame(
        rows
    )


def build_season_comparison(
    bets: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    seasons = sorted(
        numeric(
            bets[
                "season"
            ]
        )
        .dropna()
        .astype(
            int
        )
        .unique()
    )

    for season in seasons:
        season_frame = bets[
            numeric(
                bets[
                    "season"
                ]
            ).eq(
                season
            )
        ]

        for preference in PREFERENCES:
            preference_frame = (
                season_frame[
                    season_frame[
                        "pick_preference"
                    ].eq(
                        preference
                    )
                ]
            )

            for market in MARKETS:
                market_frame = (
                    preference_frame[
                        preference_frame[
                            "market"
                        ].eq(
                            market
                        )
                    ]
                )

                rows.append(
                    {
                        "season":
                            season,

                        "market":
                            market,

                        "selection":
                            "ALL",

                        "pick_preference":
                            preference,

                        **performance(
                            market_frame
                        ),
                    }
                )

                selections = sorted(
                    {
                        clean(
                            value
                        )
                        for value
                        in market_frame[
                            "selection"
                        ]
                        if clean(
                            value
                        )
                    }
                )

                for selection in selections:
                    side_frame = (
                        market_frame[
                            market_frame[
                                "selection"
                            ].map(
                                clean
                            ).eq(
                                selection
                            )
                        ]
                    )

                    rows.append(
                        {
                            "season":
                                season,

                            "market":
                                market,

                            "selection":
                                selection,

                            "pick_preference":
                                preference,

                            **performance(
                                side_frame
                            ),
                        }
                    )

    return pd.DataFrame(
        rows
    )


def build_rankings(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for market in MARKETS:
        market_all = comparison[
            comparison[
                "market"
            ].eq(
                market
            )
            & comparison[
                "selection"
            ].eq(
                "ALL"
            )
        ]

        for preference in PREFERENCES:
            train = market_all[
                market_all[
                    "period"
                ].eq(
                    "TRAIN_2021_2024"
                )
                & market_all[
                    "pick_preference"
                ].eq(
                    preference
                )
            ]

            holdout = market_all[
                market_all[
                    "period"
                ].eq(
                    "HOLDOUT_2025"
                )
                & market_all[
                    "pick_preference"
                ].eq(
                    preference
                )
            ]

            overall = market_all[
                market_all[
                    "period"
                ].eq(
                    "ALL_2021_2025"
                )
                & market_all[
                    "pick_preference"
                ].eq(
                    preference
                )
            ]

            if (
                train.empty
                or holdout.empty
                or overall.empty
            ):
                continue

            train_row = train.iloc[
                0
            ]

            holdout_row = holdout.iloc[
                0
            ]

            overall_row = overall.iloc[
                0
            ]

            rows.append(
                {
                    "market":
                        market,

                    "pick_preference":
                        preference,

                    "train_bets":
                        train_row[
                            "bets"
                        ],

                    "train_win_rate":
                        train_row[
                            "win_rate"
                        ],

                    "train_profit_units":
                        train_row[
                            "profit_units"
                        ],

                    "train_roi":
                        train_row[
                            "roi_per_unit_risk"
                        ],

                    "holdout_bets":
                        holdout_row[
                            "bets"
                        ],

                    "holdout_win_rate":
                        holdout_row[
                            "win_rate"
                        ],

                    "holdout_profit_units":
                        holdout_row[
                            "profit_units"
                        ],

                    "holdout_roi":
                        holdout_row[
                            "roi_per_unit_risk"
                        ],

                    "all_bets":
                        overall_row[
                            "bets"
                        ],

                    "all_win_rate":
                        overall_row[
                            "win_rate"
                        ],

                    "all_profit_units":
                        overall_row[
                            "profit_units"
                        ],

                    "all_roi":
                        overall_row[
                            "roi_per_unit_risk"
                        ],
                }
            )

    ranking = pd.DataFrame(
        rows
    )

    if ranking.empty:
        return ranking

    ranking[
        "holdout_roi_rank"
    ] = (
        ranking
        .groupby(
            "market"
        )[
            "holdout_roi"
        ]
        .rank(
            method="min",
            ascending=False,
        )
    )

    ranking[
        "train_roi_rank"
    ] = (
        ranking
        .groupby(
            "market"
        )[
            "train_roi"
        ]
        .rank(
            method="min",
            ascending=False,
        )
    )

    ranking[
        "all_roi_rank"
    ] = (
        ranking
        .groupby(
            "market"
        )[
            "all_roi"
        ]
        .rank(
            method="min",
            ascending=False,
        )
    )

    ranking[
        "profitable_train"
    ] = (
        numeric(
            ranking[
                "train_roi"
            ]
        ) > 0
    ).astype(
        int
    )

    ranking[
        "profitable_holdout"
    ] = (
        numeric(
            ranking[
                "holdout_roi"
            ]
        ) > 0
    ).astype(
        int
    )

    ranking[
        "profitable_both"
    ] = (
        (
            ranking[
                "profitable_train"
            ].eq(
                1
            )
        )
        & (
            ranking[
                "profitable_holdout"
            ].eq(
                1
            )
        )
    ).astype(
        int
    )

    return (
        ranking
        .sort_values(
            [
                "market",
                "profitable_both",
                "holdout_roi",
                "train_roi",
            ],
            ascending=[
                True,
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


def market_threshold_text(
    config: dict[str, Any],
    market: str,
) -> str:
    sides = (
        config[
            "markets"
        ][
            market
        ][
            "sides"
        ]
    )

    first_side = next(
        iter(
            sides
        )
    )

    threshold = (
        sides[
            first_side
        ][
            "thresholds"
        ]
    )

    return (
        f"min_prob="
        f"{threshold['min_model_prob']:.6f} "
        f"min_edge="
        f"{threshold['min_edge']:.6f} "
        f"min_ev="
        f"{threshold['min_ev']:.6f} "
        f"min_kelly="
        f"{threshold['min_kelly']:.6f} "
        f"max_kelly="
        f"{threshold['max_kelly']:.6f}"
    )


def print_market_table(
    comparison: pd.DataFrame,
    market: str,
) -> None:
    frame = comparison[
        comparison[
            "market"
        ].eq(
            market
        )
        & comparison[
            "selection"
        ].eq(
            "ALL"
        )
    ][
        [
            "period",
            "pick_preference",
            "bets",
            "wins",
            "losses",
            "pushes",
            "win_rate",
            "profit_units",
            "roi_per_unit_risk",
            "avg_model_probability",
            "avg_ev",
            "avg_kelly",
        ]
    ]

    print(
        f"\n=== {market.upper()} "
        "PREFERENCE COMPARISON ==="
    )

    print(
        frame.to_string(
            index=False
        )
    )


def print_side_table(
    comparison: pd.DataFrame,
    market: str,
) -> None:
    frame = comparison[
        comparison[
            "market"
        ].eq(
            market
        )
        & comparison[
            "selection"
        ].ne(
            "ALL"
        )
    ][
        [
            "period",
            "selection",
            "pick_preference",
            "bets",
            "wins",
            "losses",
            "pushes",
            "win_rate",
            "profit_units",
            "roi_per_unit_risk",
        ]
    ]

    print(
        f"\n=== {market.upper()} "
        "SIDE BREAKDOWN ==="
    )

    print(
        frame.to_string(
            index=False
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--provider",
        default=None,
        help=(
            "Optional single sportsbook "
            "override for all seasons. "
            "Default uses DraftKings "
            "2021-2023 and ESPN BET "
            "2024-2025."
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
            bt.provider_for_season(
                season,
                args.provider,
            )
        for season
        in bt.SEASONS
    }

    base_config = (
        bt.picks.normalize_config(
            bt.picks.load_yaml(
                bt.MARKETS
            )
        )
    )

    print(
        "seasons=2021-2025"
    )

    print(
        "train=2021-2024"
    )

    print(
        "holdout=2025"
    )

    print(
        "preferences="
        + "|".join(
            PREFERENCES
        )
    )

    print(
        "provider_map="
        + " | ".join(
            (
                f"{season}:"
                f"{provider_map[season]}"
            )
            for season
            in bt.SEASONS
        )
    )

    print(
        "travel_weather_mode="
        f"{args.travel_weather_mode}"
    )

    print(
        "\nCURRENT MARKET THRESHOLDS"
    )

    for market in MARKETS:
        print(
            f"{market}: "
            f"{market_threshold_text(base_config, market)}"
        )

    team_map = (
        bt.projection.read_csv(
            bt.TEAM_MAP,
            [
                "team_id",
                "canonical_team",
            ],
            "team map",
        )
    )

    resolver = (
        bt.projection.TeamResolver(
            team_map
        )
    )

    stadium_lookup = (
        bt.projection
        .build_home_stadium_lookup(
            bt.STADIUM_MAP,
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
            bt.projection
            .load_travel_weather_coefficients(
                bt.TW_COEFS
            )
        )

    stats_cache: dict[
        int,
        pd.DataFrame,
    ] = {}

    all_bets = []

    for season in bt.SEASONS:
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
            bt.load_schedule(
                season
            )
        )

        finals = (
            bt.load_finals(
                season
            )
        )

        features = (
            bt.load_features(
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

        market_data = (
            bt.espn_cache(
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
        )

        odds_status = (
            market_data.get(
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
                in market_data.loc[
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
            f"provider_missing={provider_missing} "
            f"fetch_errors={fetch_errors}"
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

        for preference in PREFERENCES:
            config = (
                config_for_preference(
                    base_config,
                    preference,
                )
            )

            (
                games,
                bets,
            ) = bt.replay_season(
                season,
                schedule,
                finals,
                features,
                market_data,
                resolver,
                stadium_lookup,
                config,
                coefficients,
                stats_cache,
            )

            if not bets.empty:
                bets = bets.copy()

                bets[
                    "pick_preference"
                ] = preference

                all_bets.append(
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
                f"{preference}: "
                f"projected={projected} "
                f"selected_bets={len(bets)}"
            )

    if not all_bets:
        raise RuntimeError(
            "Preference comparison "
            "produced zero bets"
        )

    bets = pd.concat(
        all_bets,
        ignore_index=True,
    )

    bad_results = bets[
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

    if not bad_results.empty:
        examples = (
            bad_results[
                [
                    "game_id",
                    "market",
                    "selection",
                    "result",
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
            "Ungradable preference bets: "
            f"{examples}"
        )

    validate_provider_bets(
        bets,
        provider_map,
    )

    duplicate_key = [
        "season",
        "game_id",
        "market",
        "pick_preference",
    ]

    duplicates = bets[
        bets.duplicated(
            duplicate_key,
            keep=False,
        )
    ]

    if not duplicates.empty:
        examples = (
            duplicates[
                duplicate_key
                + [
                    "selection"
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
            "Duplicate selected market bets "
            "found within preference: "
            f"{examples}"
        )

    comparison = (
        build_period_comparison(
            bets
        )
    )

    by_season = (
        build_season_comparison(
            bets
        )
    )

    rankings = (
        build_rankings(
            comparison
        )
    )

    bt.OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        (
            "historical_preference_"
            "bets_2021_2025.csv"
        ):
            bets,

        (
            "historical_preference_"
            "comparison.csv"
        ):
            comparison,

        (
            "historical_preference_"
            "comparison_by_season.csv"
        ):
            by_season,

        (
            "historical_preference_"
            "rankings.csv"
        ):
            rankings,
    }

    for (
        filename,
        dataframe,
    ) in outputs.items():
        bt.write_csv(
            dataframe,
            bt.OUT_DIR
            / filename,
        )

    print_market_table(
        comparison,
        "moneyline",
    )

    print_side_table(
        comparison,
        "moneyline",
    )

    print_market_table(
        comparison,
        "spread",
    )

    print_side_table(
        comparison,
        "spread",
    )

    print_market_table(
        comparison,
        "total",
    )

    print_side_table(
        comparison,
        "total",
    )

    print(
        "\n=== PREFERENCE RANKINGS ==="
    )

    print(
        rankings.to_string(
            index=False
        )
    )

    print(
        "\noutputs="
        f"{bt.OUT_DIR}"
    )

    print(
        "markets.yaml_modified=no"
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
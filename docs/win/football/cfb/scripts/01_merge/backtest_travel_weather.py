#!/usr/bin/env python3
"""
backtest_travel_weather.py

Backtests 2021-2025 historical travel/weather features against actual
CFB margins and totals.

Reads:
    docs/win/football/cfb/data/historical_features/{season}_travel_weather.csv
    docs/win/football/cfb/00_intake/pbp/{season}_pbp.parquet

Writes:
    docs/win/football/cfb/config/travel_weather_coefficients.csv
    docs/win/football/cfb/data/historical_features/travel_weather_backtest_games.csv
    docs/win/football/cfb/data/historical_features/travel_weather_backtest_summary.csv

Method:
    1. Extract actual final scores from historical PBP.
    2. Build a pregame scoring baseline using ONLY games already completed
       before each kickoff.
    3. Calculate margin/total residuals versus that baseline.
    4. Test travel features against margin residuals.
    5. Test weather features against total residuals.
    6. Use forward-season out-of-sample testing to select features.
    7. Fit final coefficients using all 2021-2025 eligible games.

No manual travel/weather point adjustments are hard-coded.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SEASONS = list(range(2021, 2026))
MIN_PRIOR_TEAM_GAMES = 3

CFB_ROOT = Path("docs/win/football/cfb")

PBP_DIR = CFB_ROOT / "00_intake" / "pbp"

HISTORICAL_DIR = (
    CFB_ROOT
    / "data"
    / "historical_features"
)

CONFIG_DIR = (
    CFB_ROOT
    / "config"
)

COEFFICIENT_PATH = (
    CONFIG_DIR
    / "travel_weather_coefficients.csv"
)

GAMES_OUTPUT_PATH = (
    HISTORICAL_DIR
    / "travel_weather_backtest_games.csv"
)

SUMMARY_OUTPUT_PATH = (
    HISTORICAL_DIR
    / "travel_weather_backtest_summary.csv"
)


TRAVEL_FEATURES = [
    "travel_net_miles_1000",
    "travel_net_time_zones",
    "travel_net_east_to_west",
    "travel_net_west_to_east",
    "travel_international",
]

WEATHER_FEATURES = [
    "weather_temperature_c",
    "weather_wind_speed_ms",
    "weather_wind_gust_ms",
    "weather_humidity_pct",
    "weather_rain_flag",
    "weather_snow_flag",
]


FEATURE_UNITS = {
    "travel_net_miles_1000": "points_per_1000_miles",
    "travel_net_time_zones": "points_per_timezone_hour",
    "travel_net_east_to_west": "points_per_net_flag",
    "travel_net_west_to_east": "points_per_net_flag",
    "travel_international": "points_per_flag",
    "weather_temperature_c": "points_per_degree_c",
    "weather_wind_speed_ms": "points_per_meter_per_second",
    "weather_wind_gust_ms": "points_per_meter_per_second",
    "weather_humidity_pct": "points_per_percentage_point",
    "weather_rain_flag": "points_per_flag",
    "weather_snow_flag": "points_per_flag",
}


def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def normalize_game_id(series):
    return (
        series
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True,
        )
        .str.strip()
    )


def read_final_scores(season):
    path = (
        PBP_DIR
        / f"{season}_pbp.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing PBP file: {path}"
        )

    required = [
        "game_id",
        "sequenceNumber",
        "end.homeScore",
        "end.awayScore",
    ]

    pbp = pd.read_parquet(
        path,
        columns=required,
    )

    missing = [
        col
        for col in required
        if col not in pbp.columns
    ]

    if missing:
        raise ValueError(
            f"{path} missing columns: {missing}"
        )

    pbp = pbp.copy()

    pbp["game_id"] = normalize_game_id(
        pbp["game_id"]
    )

    pbp["sequenceNumber"] = numeric(
        pbp["sequenceNumber"]
    )

    pbp["home_final"] = numeric(
        pbp["end.homeScore"]
    )

    pbp["away_final"] = numeric(
        pbp["end.awayScore"]
    )

    pbp = pbp.dropna(
        subset=[
            "game_id",
            "home_final",
            "away_final",
        ]
    )

    pbp = pbp.sort_values(
        [
            "game_id",
            "sequenceNumber",
        ],
        kind="stable",
    )

    finals = (
        pbp
        .groupby(
            "game_id",
            as_index=False,
        )
        .tail(1)
        [
            [
                "game_id",
                "home_final",
                "away_final",
            ]
        ]
        .copy()
    )

    finals["season"] = season

    return finals


def read_historical_features(season):
    path = (
        HISTORICAL_DIR
        / f"{season}_travel_weather.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing historical feature file: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if "game_id" not in df.columns:
        raise ValueError(
            f"{path} missing game_id"
        )

    df["game_id"] = normalize_game_id(
        df["game_id"]
    )

    df["season"] = season

    return df


def load_all_games():
    frames = []

    for season in SEASONS:
        print(
            f"Loading {season}..."
        )

        features = (
            read_historical_features(
                season
            )
        )

        scores = (
            read_final_scores(
                season
            )
        )

        merged = features.merge(
            scores[
                [
                    "game_id",
                    "home_final",
                    "away_final",
                ]
            ],
            on="game_id",
            how="inner",
            validate="one_to_one",
        )

        print(
            f"{season}: "
            f"features={len(features)} "
            f"final_scores={len(scores)} "
            f"matched={len(merged)}"
        )

        frames.append(
            merged
        )

    games = pd.concat(
        frames,
        ignore_index=True,
    )

    games["home_final"] = numeric(
        games["home_final"]
    )

    games["away_final"] = numeric(
        games["away_final"]
    )

    games["actual_margin"] = (
        games["home_final"]
        - games["away_final"]
    )

    games["actual_total"] = (
        games["home_final"]
        + games["away_final"]
    )

    return games


def parse_kickoff(df):
    kickoff = pd.to_datetime(
        df.get(
            "kickoff_utc"
        ),
        errors="coerce",
        utc=True,
    )

    if kickoff.notna().any():
        return kickoff

    combined = (
        df["game_date"].astype(str)
        + " "
        + df["game_time"].astype(str)
    )

    return pd.to_datetime(
        combined,
        errors="coerce",
        utc=True,
    )


def build_pregame_baselines(games):
    games = games.copy()

    games["kickoff_dt"] = (
        parse_kickoff(
            games
        )
    )

    games = games.dropna(
        subset=[
            "kickoff_dt",
            "home_final",
            "away_final",
        ]
    )

    games = games.sort_values(
        [
            "kickoff_dt",
            "game_id",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    baseline_home = pd.Series(
        np.nan,
        index=games.index,
        dtype=float,
    )

    baseline_away = pd.Series(
        np.nan,
        index=games.index,
        dtype=float,
    )

    baseline_margin = pd.Series(
        np.nan,
        index=games.index,
        dtype=float,
    )

    baseline_total = pd.Series(
        np.nan,
        index=games.index,
        dtype=float,
    )

    prior_home_advantage = pd.Series(
        np.nan,
        index=games.index,
        dtype=float,
    )

    team_stats = defaultdict(
        lambda: {
            "games": 0,
            "points_for": 0.0,
            "points_against": 0.0,
        }
    )

    hfa_sum = 0.0
    hfa_games = 0

    grouped = games.groupby(
        "kickoff_dt",
        sort=True,
    )

    for _, group in grouped:

        for idx in group.index:

            row = games.loc[
                idx
            ]

            season = int(
                row["season"]
            )

            home_team = str(
                row["home_team"]
            ).strip()

            away_team = str(
                row["away_team"]
            ).strip()

            home_key = (
                season,
                home_team,
            )

            away_key = (
                season,
                away_team,
            )

            home_state = (
                team_stats[
                    home_key
                ]
            )

            away_state = (
                team_stats[
                    away_key
                ]
            )

            if hfa_games > 0:
                hfa = (
                    hfa_sum
                    / hfa_games
                )
            else:
                hfa = 0.0

            prior_home_advantage.loc[
                idx
            ] = hfa

            if (
                home_state["games"]
                < MIN_PRIOR_TEAM_GAMES
                or
                away_state["games"]
                < MIN_PRIOR_TEAM_GAMES
            ):
                continue

            home_pf = (
                home_state[
                    "points_for"
                ]
                / home_state[
                    "games"
                ]
            )

            home_pa = (
                home_state[
                    "points_against"
                ]
                / home_state[
                    "games"
                ]
            )

            away_pf = (
                away_state[
                    "points_for"
                ]
                / away_state[
                    "games"
                ]
            )

            away_pa = (
                away_state[
                    "points_against"
                ]
                / away_state[
                    "games"
                ]
            )

            neutral = int(
                numeric(
                    pd.Series(
                        [
                            row.get(
                                "neutral_site_flag",
                                row.get(
                                    "neutral_site",
                                    0,
                                ),
                            )
                        ]
                    )
                ).fillna(
                    0
                ).iloc[
                    0
                ]
            )

            game_hfa = (
                0.0
                if neutral
                else hfa
            )

            projected_home = (
                (
                    home_pf
                    + away_pa
                )
                / 2.0
                + game_hfa / 2.0
            )

            projected_away = (
                (
                    away_pf
                    + home_pa
                )
                / 2.0
                - game_hfa / 2.0
            )

            baseline_home.loc[
                idx
            ] = projected_home

            baseline_away.loc[
                idx
            ] = projected_away

            baseline_margin.loc[
                idx
            ] = (
                projected_home
                - projected_away
            )

            baseline_total.loc[
                idx
            ] = (
                projected_home
                + projected_away
            )

        # Update results only AFTER every game at this kickoff
        # has received its pregame baseline.
        for idx in group.index:

            row = games.loc[
                idx
            ]

            season = int(
                row["season"]
            )

            home_team = str(
                row["home_team"]
            ).strip()

            away_team = str(
                row["away_team"]
            ).strip()

            home_score = float(
                row["home_final"]
            )

            away_score = float(
                row["away_final"]
            )

            home_key = (
                season,
                home_team,
            )

            away_key = (
                season,
                away_team,
            )

            team_stats[
                home_key
            ]["games"] += 1

            team_stats[
                home_key
            ]["points_for"] += (
                home_score
            )

            team_stats[
                home_key
            ]["points_against"] += (
                away_score
            )

            team_stats[
                away_key
            ]["games"] += 1

            team_stats[
                away_key
            ]["points_for"] += (
                away_score
            )

            team_stats[
                away_key
            ]["points_against"] += (
                home_score
            )

            neutral = int(
                numeric(
                    pd.Series(
                        [
                            row.get(
                                "neutral_site_flag",
                                row.get(
                                    "neutral_site",
                                    0,
                                ),
                            )
                        ]
                    )
                ).fillna(
                    0
                ).iloc[
                    0
                ]
            )

            if not neutral:
                hfa_sum += (
                    home_score
                    - away_score
                )

                hfa_games += 1

    games[
        "baseline_home_score"
    ] = baseline_home

    games[
        "baseline_away_score"
    ] = baseline_away

    games[
        "baseline_margin"
    ] = baseline_margin

    games[
        "baseline_total"
    ] = baseline_total

    games[
        "prior_home_advantage"
    ] = prior_home_advantage

    games[
        "margin_residual"
    ] = (
        games[
            "actual_margin"
        ]
        - games[
            "baseline_margin"
        ]
    )

    games[
        "total_residual"
    ] = (
        games[
            "actual_total"
        ]
        - games[
            "baseline_total"
        ]
    )

    return games


def weather_exposed(row):
    roof = str(
        row.get(
            "roof",
            ""
        )
    ).strip().casefold()

    roof_type = str(
        row.get(
            "roof_type",
            ""
        )
    ).strip().casefold()

    dome_flag = numeric(
        pd.Series(
            [
                row.get(
                    "dome_flag",
                    np.nan,
                )
            ]
        )
    ).iloc[
        0
    ]

    open_air_flag = numeric(
        pd.Series(
            [
                row.get(
                    "open_air_flag",
                    np.nan,
                )
            ]
        )
    ).iloc[
        0
    ]

    closed_terms = (
        "dome",
        "indoor",
        "closed",
    )

    if any(
        term in roof
        for term
        in closed_terms
    ):
        return 0

    if any(
        term in roof_type
        for term
        in closed_terms
    ):
        return 0

    if (
        not pd.isna(
            dome_flag
        )
        and dome_flag == 1
    ):
        return 0

    if (
        "open_air"
        in roof
        or "open air"
        in roof
        or "outdoor"
        in roof
        or
        (
            not pd.isna(
                open_air_flag
            )
            and open_air_flag == 1
        )
    ):
        return 1

    # Unknown roof state:
    # do not attribute outcome to weather.
    return 0


def engineer_features(games):
    df = games.copy()

    away_miles = numeric(
        df.get(
            "away_miles_traveled"
        )
    )

    home_miles = numeric(
        df.get(
            "home_miles_traveled"
        )
    )

    away_tz = numeric(
        df.get(
            "away_time_zones_crossed"
        )
    )

    home_tz = numeric(
        df.get(
            "home_time_zones_crossed"
        )
    )

    away_e2w = numeric(
        df.get(
            "away_east_to_west"
        )
    )

    home_e2w = numeric(
        df.get(
            "home_east_to_west"
        )
    )

    away_w2e = numeric(
        df.get(
            "away_west_to_east"
        )
    )

    home_w2e = numeric(
        df.get(
            "home_west_to_east"
        )
    )

    df[
        "travel_net_miles_1000"
    ] = (
        away_miles
        - home_miles
    ) / 1000.0

    df[
        "travel_net_time_zones"
    ] = (
        away_tz
        - home_tz
    )

    df[
        "travel_net_east_to_west"
    ] = (
        away_e2w
        - home_e2w
    )

    df[
        "travel_net_west_to_east"
    ] = (
        away_w2e
        - home_w2e
    )

    df[
        "travel_international"
    ] = numeric(
        df.get(
            "international_flag"
        )
    )

    exposed = df.apply(
        weather_exposed,
        axis=1,
    )

    df[
        "weather_exposed"
    ] = exposed

    weather_status = (
        df.get(
            "weather_status",
            pd.Series(
                "",
                index=df.index,
            ),
        )
        .astype(str)
        .str.casefold()
    )

    weather_ok = (
        weather_status
        == "ok"
    )

    weather_available = (
        weather_ok
        & (
            exposed == 1
        )
    )

    weather_map = {
        "weather_temperature_c":
            "temperature",
        "weather_wind_speed_ms":
            "wind_speed",
        "weather_wind_gust_ms":
            "wind_gust",
        "weather_humidity_pct":
            "humidity",
        "weather_rain_flag":
            "rain_flag",
        "weather_snow_flag":
            "snow_flag",
    }

    for output_col, source_col in (
        weather_map.items()
    ):

        source = numeric(
            df.get(
                source_col
            )
        )

        df[
            output_col
        ] = np.where(
            weather_available,
            source,
            np.nan,
        )

    return df


def fit_linear(
    df,
    target,
    features,
):
    if not features:
        return (
            0.0,
            {},
        )

    X = df[
        features
    ].to_numpy(
        dtype=float
    )

    y = df[
        target
    ].to_numpy(
        dtype=float
    )

    design = np.column_stack(
        [
            np.ones(
                len(X)
            ),
            X,
        ]
    )

    beta, *_ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )

    intercept = float(
        beta[0]
    )

    coefficients = {
        feature: float(
            value
        )
        for feature, value
        in zip(
            features,
            beta[1:],
        )
    }

    return (
        intercept,
        coefficients,
    )


def predict_linear(
    df,
    intercept,
    coefficients,
):
    prediction = np.full(
        len(df),
        float(intercept),
        dtype=float,
    )

    for feature, coefficient in (
        coefficients.items()
    ):
        prediction += (
            df[
                feature
            ].to_numpy(
                dtype=float
            )
            * coefficient
        )

    return prediction


def error_metrics(
    actual,
    prediction,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    error = (
        actual
        - prediction
    )

    return {
        "mae": float(
            np.mean(
                np.abs(
                    error
                )
            )
        ),
        "rmse": float(
            math.sqrt(
                np.mean(
                    error ** 2
                )
            )
        ),
        "n": int(
            len(
                actual
            )
        ),
    }


def forward_cv(
    df,
    target,
    features,
):
    seasons = sorted(
        int(x)
        for x
        in df[
            "season"
        ].dropna().unique()
    )

    predictions = []
    actuals = []
    fold_rows = []

    for test_season in seasons[
        1:
    ]:

        train = df[
            df[
                "season"
            ]
            < test_season
        ]

        test = df[
            df[
                "season"
            ]
            == test_season
        ]

        if (
            train.empty
            or test.empty
        ):
            continue

        if features:

            (
                intercept,
                coefficients,
            ) = fit_linear(
                train,
                target,
                features,
            )

            prediction = (
                predict_linear(
                    test,
                    intercept,
                    coefficients,
                )
            )

        else:

            # No travel/weather adjustment.
            prediction = np.zeros(
                len(test),
                dtype=float,
            )

        actual = test[
            target
        ].to_numpy(
            dtype=float
        )

        metrics = error_metrics(
            actual,
            prediction,
        )

        fold_rows.append(
            {
                "test_season":
                    test_season,
                "features":
                    "|".join(
                        features
                    ),
                **metrics,
            }
        )

        predictions.extend(
            prediction.tolist()
        )

        actuals.extend(
            actual.tolist()
        )

    if not actuals:
        return (
            {
                "mae": np.nan,
                "rmse": np.nan,
                "n": 0,
            },
            fold_rows,
        )

    return (
        error_metrics(
            actuals,
            predictions,
        ),
        fold_rows,
    )


def select_features(
    df,
    target,
    candidates,
):
    selected = []

    baseline_metrics, _ = (
        forward_cv(
            df,
            target,
            [],
        )
    )

    current_metrics = (
        baseline_metrics
    )

    selection_rows = []

    while True:

        best_feature = None
        best_metrics = None

        for candidate in candidates:

            if candidate in selected:
                continue

            trial_features = (
                selected
                + [
                    candidate
                ]
            )

            metrics, _ = (
                forward_cv(
                    df,
                    target,
                    trial_features,
                )
            )

            if np.isnan(
                metrics["mae"]
            ):
                continue

            improves_mae = (
                metrics["mae"]
                < current_metrics[
                    "mae"
                ]
            )

            improves_rmse = (
                metrics["rmse"]
                < current_metrics[
                    "rmse"
                ]
            )

            if not (
                improves_mae
                and improves_rmse
            ):
                continue

            if (
                best_metrics is None
                or metrics["mae"]
                < best_metrics[
                    "mae"
                ]
            ):
                best_feature = (
                    candidate
                )

                best_metrics = (
                    metrics
                )

        if best_feature is None:
            break

        selected.append(
            best_feature
        )

        selection_rows.append(
            {
                "step":
                    len(
                        selected
                    ),
                "feature":
                    best_feature,
                "mae":
                    best_metrics[
                        "mae"
                    ],
                "rmse":
                    best_metrics[
                        "rmse"
                    ],
            }
        )

        current_metrics = (
            best_metrics
        )

    return {
        "selected":
            selected,
        "baseline_metrics":
            baseline_metrics,
        "final_metrics":
            current_metrics,
        "selection_rows":
            selection_rows,
    }


def prepare_analysis_data(
    games,
    target,
    candidates,
):
    required = (
        [
            "season",
            target,
        ]
        + candidates
    )

    df = games[
        required
    ].copy()

    for col in required:
        df[
            col
        ] = numeric(
            df[
                col
            ]
        )

    df = df.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return df.dropna(
        subset=required
    ).copy()


def weather_centers(
    df,
):
    centers = {}

    for feature in WEATHER_FEATURES:

        value = numeric(
            df[
                feature
            ]
        ).mean()

        centers[
            feature
        ] = (
            float(value)
            if not pd.isna(
                value
            )
            else 0.0
        )

    return centers


def center_weather(
    df,
    centers,
):
    df = df.copy()

    for feature, center in (
        centers.items()
    ):

        df[
            feature
        ] = (
            numeric(
                df[
                    feature
                ]
            )
            - center
        )

    return df


def build_coefficient_rows(
    target_name,
    candidates,
    selected,
    coefficients,
    centers,
    sample_size,
    cv_metrics,
):
    rows = []

    for feature in candidates:

        rows.append(
            {
                "target":
                    target_name,
                "feature":
                    feature,
                "selected":
                    int(
                        feature
                        in selected
                    ),
                "coefficient":
                    (
                        coefficients.get(
                            feature,
                            0.0,
                        )
                        if feature
                        in selected
                        else 0.0
                    ),
                "center":
                    centers.get(
                        feature,
                        0.0,
                    ),
                "units":
                    FEATURE_UNITS.get(
                        feature,
                        "",
                    ),
                "sample_size":
                    sample_size,
                "cv_mae":
                    cv_metrics[
                        "mae"
                    ],
                "cv_rmse":
                    cv_metrics[
                        "rmse"
                    ],
            }
        )

    return rows


def main():
    HISTORICAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    games = load_all_games()

    print(
        f"Matched historical games: "
        f"{len(games)}"
    )

    games = (
        build_pregame_baselines(
            games
        )
    )

    games = engineer_features(
        games
    )

    usable_baseline = (
        games[
            "baseline_margin"
        ].notna()
        & games[
            "baseline_total"
        ].notna()
    )

    games = games[
        usable_baseline
    ].copy()

    print(
        "Games with valid pregame baseline: "
        f"{len(games)}"
    )

    # -------------------------
    # TRAVEL -> MARGIN
    # -------------------------

    travel_df = (
        prepare_analysis_data(
            games,
            "margin_residual",
            TRAVEL_FEATURES,
        )
    )

    travel_result = (
        select_features(
            travel_df,
            "margin_residual",
            TRAVEL_FEATURES,
        )
    )

    travel_selected = (
        travel_result[
            "selected"
        ]
    )

    if travel_selected:

        (
            travel_intercept,
            travel_coefficients,
        ) = fit_linear(
            travel_df,
            "margin_residual",
            travel_selected,
        )

    else:

        travel_intercept = 0.0
        travel_coefficients = {}

    # -------------------------
    # WEATHER -> TOTAL
    # -------------------------

    weather_raw = (
        prepare_analysis_data(
            games,
            "total_residual",
            WEATHER_FEATURES,
        )
    )

    centers = weather_centers(
        weather_raw
    )

    weather_df = center_weather(
        weather_raw,
        centers,
    )

    weather_result = (
        select_features(
            weather_df,
            "total_residual",
            WEATHER_FEATURES,
        )
    )

    weather_selected = (
        weather_result[
            "selected"
        ]
    )

    if weather_selected:

        (
            weather_intercept,
            weather_coefficients,
        ) = fit_linear(
            weather_df,
            "total_residual",
            weather_selected,
        )

    else:

        weather_intercept = 0.0
        weather_coefficients = {}

    # -------------------------
    # COEFFICIENT OUTPUT
    # -------------------------

    coefficient_rows = []

    coefficient_rows.extend(
        build_coefficient_rows(
            "margin",
            TRAVEL_FEATURES,
            travel_selected,
            travel_coefficients,
            {},
            len(
                travel_df
            ),
            travel_result[
                "final_metrics"
            ],
        )
    )

    coefficient_rows.extend(
        build_coefficient_rows(
            "total",
            WEATHER_FEATURES,
            weather_selected,
            weather_coefficients,
            centers,
            len(
                weather_df
            ),
            weather_result[
                "final_metrics"
            ],
        )
    )

    coefficient_df = pd.DataFrame(
        coefficient_rows
    )

    coefficient_df.to_csv(
        COEFFICIENT_PATH,
        index=False,
    )

    # -------------------------
    # BACKTEST SUMMARY
    # -------------------------

    summary_rows = [
        {
            "model":
                "travel_margin",
            "games":
                len(
                    travel_df
                ),
            "selected_features":
                "|".join(
                    travel_selected
                ),
            "baseline_cv_mae":
                travel_result[
                    "baseline_metrics"
                ][
                    "mae"
                ],
            "adjusted_cv_mae":
                travel_result[
                    "final_metrics"
                ][
                    "mae"
                ],
            "mae_improvement":
                (
                    travel_result[
                        "baseline_metrics"
                    ][
                        "mae"
                    ]
                    -
                    travel_result[
                        "final_metrics"
                    ][
                        "mae"
                    ]
                ),
            "baseline_cv_rmse":
                travel_result[
                    "baseline_metrics"
                ][
                    "rmse"
                ],
            "adjusted_cv_rmse":
                travel_result[
                    "final_metrics"
                ][
                    "rmse"
                ],
            "rmse_improvement":
                (
                    travel_result[
                        "baseline_metrics"
                    ][
                        "rmse"
                    ]
                    -
                    travel_result[
                        "final_metrics"
                    ][
                        "rmse"
                    ]
                ),
        },
        {
            "model":
                "weather_total",
            "games":
                len(
                    weather_df
                ),
            "selected_features":
                "|".join(
                    weather_selected
                ),
            "baseline_cv_mae":
                weather_result[
                    "baseline_metrics"
                ][
                    "mae"
                ],
            "adjusted_cv_mae":
                weather_result[
                    "final_metrics"
                ][
                    "mae"
                ],
            "mae_improvement":
                (
                    weather_result[
                        "baseline_metrics"
                    ][
                        "mae"
                    ]
                    -
                    weather_result[
                        "final_metrics"
                    ][
                        "mae"
                    ]
                ),
            "baseline_cv_rmse":
                weather_result[
                    "baseline_metrics"
                ][
                    "rmse"
                ],
            "adjusted_cv_rmse":
                weather_result[
                    "final_metrics"
                ][
                    "rmse"
                ],
            "rmse_improvement":
                (
                    weather_result[
                        "baseline_metrics"
                    ][
                        "rmse"
                    ]
                    -
                    weather_result[
                        "final_metrics"
                    ][
                        "rmse"
                    ]
                ),
        },
    ]

    pd.DataFrame(
        summary_rows
    ).to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    # -------------------------
    # GAME-LEVEL AUDIT OUTPUT
    # -------------------------

    audit_columns = [
        "season",
        "week",
        "game_id",
        "kickoff_dt",
        "away_team",
        "home_team",
        "stadium",
        "neutral_site_flag",
        "home_final",
        "away_final",
        "actual_margin",
        "actual_total",
        "baseline_margin",
        "baseline_total",
        "margin_residual",
        "total_residual",
        "prior_home_advantage",
        "away_miles_traveled",
        "home_miles_traveled",
        "away_time_zones_crossed",
        "home_time_zones_crossed",
        "travel_net_miles_1000",
        "travel_net_time_zones",
        "travel_net_east_to_west",
        "travel_net_west_to_east",
        "travel_international",
        "weather_exposed",
        "temperature",
        "wind_speed",
        "wind_gust",
        "humidity",
        "rain_flag",
        "snow_flag",
        "weather_status",
    ]

    audit_columns = [
        col
        for col in audit_columns
        if col in games.columns
    ]

    games[
        audit_columns
    ].to_csv(
        GAMES_OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "TRAVEL -> MARGIN"
    )
    print(
        "selected_features="
        + (
            ",".join(
                travel_selected
            )
            if travel_selected
            else "NONE"
        )
    )
    print(
        "baseline_cv_mae="
        f"{travel_result['baseline_metrics']['mae']:.4f}"
    )
    print(
        "adjusted_cv_mae="
        f"{travel_result['final_metrics']['mae']:.4f}"
    )
    print(
        "baseline_cv_rmse="
        f"{travel_result['baseline_metrics']['rmse']:.4f}"
    )
    print(
        "adjusted_cv_rmse="
        f"{travel_result['final_metrics']['rmse']:.4f}"
    )

    print()
    print(
        "WEATHER -> TOTAL"
    )
    print(
        "selected_features="
        + (
            ",".join(
                weather_selected
            )
            if weather_selected
            else "NONE"
        )
    )
    print(
        "baseline_cv_mae="
        f"{weather_result['baseline_metrics']['mae']:.4f}"
    )
    print(
        "adjusted_cv_mae="
        f"{weather_result['final_metrics']['mae']:.4f}"
    )
    print(
        "baseline_cv_rmse="
        f"{weather_result['baseline_metrics']['rmse']:.4f}"
    )
    print(
        "adjusted_cv_rmse="
        f"{weather_result['final_metrics']['rmse']:.4f}"
    )

    print()
    print(
        f"WROTE {COEFFICIENT_PATH}"
    )
    print(
        f"WROTE {SUMMARY_OUTPUT_PATH}"
    )
    print(
        f"WROTE {GAMES_OUTPUT_PATH}"
    )
    print(
        "status=success"
    )


if __name__ == "__main__":
    main()
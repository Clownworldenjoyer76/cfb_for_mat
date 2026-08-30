#!/usr/bin/env python3
# docs/win/football/cfb/scripts/01_merge/projection_week1.py
"""
CFB Week 1 cold-start projection.

Produces the downstream selection-engine fields:
    predicted_margin
    predicted_total
    predicted_home_score
    predicted_away_score
    home_win_probability
    away_win_probability
    home_cover_probability
    away_cover_probability
    over_probability
    under_probability

No current-season PBP or game result is used.

Probability model
-----------------
The point forecasts are converted to probabilities with configurable normal
error assumptions:

    actual_margin ~ Normal(predicted_margin, margin_sd)
    actual_total  ~ Normal(predicted_total, total_sd)

Therefore:
    home win  = P(actual_margin > 0)
    home cover = P(actual_margin + home_spread > 0)
    over = P(actual_total > market_total)

Away/under probabilities are exact complements. If the sportsbook spread or
total is unavailable, the corresponding market-dependent probability pair is
set to 0.50/0.50. The downstream candidate engine will still mark that market
unavailable because the line/odds are missing.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


SCRIPT_VERSION = "cfb-week1-v7-game-lock-2026-08-30"
MIN_PRIOR_TEAM_WEEKS = 10
ESPN_MARGIN_SYMMETRY_TOLERANCE = 0.25
DEFAULT_MARGIN_SD = 14.0
DEFAULT_TOTAL_SD = 14.0
PROBABILITY_EPS = 1e-6

TEAM_METRICS = [
    "off_epa_per_play",
    "def_epa_per_play",
    "off_success_rate",
    "def_success_rate",
    "yards_per_play",
    "yards_per_play_allowed",
    "points_per_drive",
    "points_per_drive_allowed",
    "red_zone_td_rate",
    "red_zone_td_rate_allowed",
    "early_down_epa",
    "third_down_conversion_rate",
]

OUTPUT_BASE_COLUMNS = [
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
    "roof",
    "surface",
    "bookmaker",
    "home_spread",
    "total",
]

REQUIRED_PREDICTION_COLUMNS = [
    "predicted_margin",
    "predicted_total",
    "predicted_home_score",
    "predicted_away_score",
    "home_win_probability",
    "away_win_probability",
    "home_cover_probability",
    "away_cover_probability",
    "over_probability",
    "under_probability",
]


TRAVEL_REQUIRED_COLUMNS = [
    "game_id",
    "away_miles_traveled",
    "home_miles_traveled",
    "away_time_zones_crossed",
    "home_time_zones_crossed",
    "away_east_to_west",
    "home_east_to_west",
    "away_west_to_east",
    "home_west_to_east",
    "international_flag",
]

WEATHER_REQUIRED_COLUMNS = [
    "game_id",
    "temperature",
    "wind_speed",
    "wind_gust",
    "humidity",
    "rain_flag",
    "snow_flag",
    "roof",
    "roof_type",
    "dome_flag",
    "retractable_roof_flag",
    "open_air_flag",
    "weather_timestep_utc",
]

COEFFICIENT_REQUIRED_COLUMNS = [
    "target",
    "feature",
    "selected",
    "coefficient",
    "center",
]

SUPPORTED_TRAVEL_FEATURES = {
    "travel_net_miles_1000",
    "travel_net_time_zones",
    "travel_net_east_to_west",
    "travel_net_west_to_east",
    "travel_international",
}

SUPPORTED_WEATHER_FEATURES = {
    "weather_temperature_c",
    "weather_wind_speed_ms",
    "weather_wind_gust_ms",
    "weather_humidity_pct",
    "weather_rain_flag",
    "weather_snow_flag",
}

OUT_STATUS_MULTIPLIER = {
    "out": 1.00,
    "doubtful": 0.75,
    "questionable": 0.25,
}

POSITION_POINT_COST = {
    "QB": 3.00,
    "OL": 0.60,
    "OT": 0.60,
    "LT": 0.60,
    "RT": 0.60,
    "OG": 0.60,
    "LG": 0.60,
    "RG": 0.60,
    "C": 0.60,
    "WR": 0.50,
    "RB": 0.50,
    "HB": 0.50,
    "FB": 0.30,
    "TE": 0.45,
    "DE": 0.40,
    "DT": 0.40,
    "DL": 0.40,
    "NT": 0.40,
    "EDGE": 0.45,
    "LB": 0.40,
    "ILB": 0.40,
    "OLB": 0.40,
    "MLB": 0.40,
    "CB": 0.40,
    "DB": 0.35,
    "S": 0.35,
    "FS": 0.35,
    "SS": 0.35,
    "K": 0.20,
    "PK": 0.20,
    "P": 0.15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CFB Week 1 cold-start projections and betting probabilities."
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Target season. Defaults to CFB_SEASON, then 2026.",
    )

    parser.add_argument(
        "--prior-season",
        type=int,
        default=None,
        help="Prior season used for team-strength priors. Defaults to season-1.",
    )

    parser.add_argument(
        "--week",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--home-field",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--drives-per-team",
        type=float,
        default=11.5,
    )

    parser.add_argument(
        "--market-margin-weight",
        type=float,
        default=0.36,
    )

    parser.add_argument(
        "--fpi-margin-weight",
        type=float,
        default=0.28,
    )

    parser.add_argument(
        "--espn-margin-weight",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--prior-margin-weight",
        type=float,
        default=0.16,
    )

    parser.add_argument(
        "--market-total-weight",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--fresh-injury-days",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--margin-sd",
        type=float,
        default=DEFAULT_MARGIN_SD,
        help=(
            "Margin forecast error SD used for win/cover probabilities. "
            "Calibrate from historical CFB residuals when available."
        ),
    )

    parser.add_argument(
        "--total-sd",
        type=float,
        default=DEFAULT_TOTAL_SD,
        help=(
            "Total forecast error SD used for over/under probabilities. "
            "Calibrate from historical CFB residuals when available."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate projections but do not write output.",
    )

    return parser.parse_args()


def clean(
    value: object,
) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

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


def normalize_key(
    value: object,
) -> str:
    text = unicodedata.normalize(
        "NFKD",
        clean(value),
    )

    text = "".join(
        ch
        for ch in text
        if not unicodedata.combining(
            ch
        )
    )

    text = (
        text.casefold()
        .replace(
            "&",
            " and ",
        )
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def as_bool(
    value: object,
) -> bool:
    return clean(
        value
    ).casefold() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
    }


def as_float(
    value: object,
) -> float | None:
    text = (
        clean(value)
        .replace(
            ",",
            "",
        )
        .replace(
            "%",
            "",
        )
    )

    if not text:
        return None

    try:
        number = float(
            text
        )
    except ValueError:
        return None

    if not math.isfinite(
        number
    ):
        return None

    return number


def repo_cfb_root() -> Path:
    here = Path(
        __file__
    ).resolve()

    for parent in [
        here.parent,
        *here.parents,
    ]:
        candidate = (
            parent
            / "docs"
            / "win"
            / "football"
            / "cfb"
        )

        if candidate.is_dir():
            return candidate

    try:
        candidate = here.parents[
            2
        ]
    except IndexError as exc:
        raise RuntimeError(
            f"Cannot resolve CFB root from {here}"
        ) from exc

    if candidate.name != "cfb":
        raise RuntimeError(
            f"Cannot resolve CFB root from {here}"
        )

    return candidate


def read_csv(
    path: Path,
    required: list[str],
    label: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label}: {path}"
        )

    df = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        low_memory=False,
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label} missing required columns: "
            f"{missing}"
        )

    return df


def normalize_game_id(
    value: object,
) -> str:
    text = clean(
        value
    )

    if text.endswith(
        ".0"
    ):
        text = text[
            :-2
        ]

    return text



def schedule_kickoff_utc(
    row: pd.Series,
) -> datetime | None:
    game_date = clean(
        row.get(
            "game_date"
        )
    )
    game_time = clean(
        row.get(
            "game_time"
        )
    )
    game_timezone = clean(
        row.get(
            "game_timezone"
        )
    )

    if (
        not game_date
        or not game_time
        or not game_timezone
    ):
        return None

    try:
        local_dt = datetime.strptime(
            f"{game_date} {game_time}",
            "%Y-%m-%d %H:%M",
        ).replace(
            tzinfo=ZoneInfo(
                game_timezone
            )
        )
    except Exception:
        return None

    return local_dt.astimezone(
        timezone.utc
    )


def locked_game_ids(
    schedule: pd.DataFrame,
    now_utc: datetime | None = None,
) -> set[str]:
    now_utc = (
        now_utc
        or datetime.now(
            timezone.utc
        )
    )

    locked: set[str] = set()

    for _, row in schedule.iterrows():
        game_id = normalize_game_id(
            row.get(
                "game_id"
            )
        )

        explicit_lock = as_bool(
            row.get(
                "game_locked"
            )
        )

        kickoff = schedule_kickoff_utc(
            row
        )

        time_lock = (
            kickoff is not None
            and now_utc >= kickoff
        )

        if (
            game_id
            and (
                explicit_lock
                or time_lock
            )
        ):
            locked.add(
                game_id
            )

    return locked


def preserve_locked_rows(
    projected: pd.DataFrame,
    schedule: pd.DataFrame,
    existing_output_path: Path,
    label: str,
) -> tuple[pd.DataFrame, int]:
    locked_ids = locked_game_ids(
        schedule
    )

    if not locked_ids:
        return (
            projected,
            0,
        )

    if not existing_output_path.is_file():
        raise RuntimeError(
            f"{label}: {len(locked_ids)} game(s) have already "
            "kicked off but no existing output is available to "
            "preserve. Refusing to create post-kickoff projections. "
            f"game_ids={sorted(locked_ids)[:10]}"
        )

    existing = pd.read_csv(
        existing_output_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if "game_id" not in existing.columns:
        raise RuntimeError(
            f"{label}: existing output has no game_id column: "
            f"{existing_output_path}"
        )

    existing[
        "game_id"
    ] = existing[
        "game_id"
    ].map(
        normalize_game_id
    )

    if existing[
        "game_id"
    ].duplicated().any():
        raise RuntimeError(
            f"{label}: existing output contains duplicate game_id "
            f"values: {existing_output_path}"
        )

    missing_rows = sorted(
        locked_ids
        - set(
            existing[
                "game_id"
            ]
        )
    )

    if missing_rows:
        raise RuntimeError(
            f"{label}: existing output is missing locked games. "
            f"Refusing to rebuild them after kickoff. "
            f"game_ids={missing_rows[:10]}"
        )

    missing_columns = [
        column
        for column in projected.columns
        if column not in existing.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"{label}: existing output cannot safely preserve locked "
            f"rows because columns are missing: {missing_columns}"
        )

    result = projected.astype(object).copy()
    result[
        "game_id"
    ] = result[
        "game_id"
    ].map(
        normalize_game_id
    )

    existing_lookup = existing.set_index(
        "game_id",
        drop=False,
    )

    for game_id in locked_ids:
        mask = result[
            "game_id"
        ].eq(
            game_id
        )

        if not mask.any():
            raise RuntimeError(
                f"{label}: locked game_id={game_id} is not present "
                "in the newly built projection frame"
            )

        prior_row = existing_lookup.loc[
            game_id,
            result.columns,
        ]

        result.loc[
            mask,
            result.columns,
        ] = prior_row.to_numpy()

    validate_probability_output(
        result
    )

    return (
        result,
        len(locked_ids),
    )


def load_game_feature_file(
    path: Path,
    required: list[str],
    label: str,
) -> pd.DataFrame:
    df = read_csv(
        path,
        required,
        label,
    ).copy()

    df[
        "game_id"
    ] = df[
        "game_id"
    ].map(
        normalize_game_id
    )

    df = df[
        df[
            "game_id"
        ].ne(
            ""
        )
    ].copy()

    duplicate_mask = df[
        "game_id"
    ].duplicated(
        keep=False
    )

    if duplicate_mask.any():
        duplicates = (
            df.loc[
                duplicate_mask,
                "game_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"{label} contains duplicate game_id values: "
            f"{duplicates[:10]}"
        )

    return df


def load_travel_weather_coefficients(
    path: Path,
) -> dict[
    str,
    dict[
        str,
        dict[
            str,
            float,
        ],
    ],
]:
    coefficients = read_csv(
        path,
        COEFFICIENT_REQUIRED_COLUMNS,
        "travel/weather coefficients",
    )

    selected = coefficients[
        coefficients[
            "selected"
        ].map(
            as_bool
        )
    ].copy()

    result: dict[
        str,
        dict[
            str,
            dict[
                str,
                float,
            ],
        ],
    ] = {
        "margin": {},
        "total": {},
    }

    for _, row in selected.iterrows():
        target = clean(
            row.get(
                "target"
            )
        ).casefold()

        feature = clean(
            row.get(
                "feature"
            )
        )

        coefficient = as_float(
            row.get(
                "coefficient"
            )
        )

        center = as_float(
            row.get(
                "center"
            )
        )

        if target not in result:
            raise ValueError(
                "Unsupported travel/weather coefficient target: "
                f"{target!r}"
            )

        supported = (
            SUPPORTED_TRAVEL_FEATURES
            if target == "margin"
            else SUPPORTED_WEATHER_FEATURES
        )

        if feature not in supported:
            raise ValueError(
                "Unsupported selected travel/weather feature: "
                f"{feature!r}"
            )

        if coefficient is None:
            raise ValueError(
                "Selected travel/weather coefficient is not numeric: "
                f"target={target} feature={feature}"
            )

        result[
            target
        ][
            feature
        ] = {
            "coefficient": float(
                coefficient
            ),
            "center": float(
                center
                if center is not None
                else 0.0
            ),
        }

    return result


def _series_value(
    row: pd.Series | None,
    column: str,
) -> float | None:
    if row is None:
        return None

    return as_float(
        row.get(
            column
        )
    )


def build_travel_features(
    row: pd.Series | None,
) -> dict[
    str,
    float | None,
]:
    if row is None:
        return {
            feature: None
            for feature
            in SUPPORTED_TRAVEL_FEATURES
        }

    away_miles = _series_value(
        row,
        "away_miles_traveled",
    )

    home_miles = _series_value(
        row,
        "home_miles_traveled",
    )

    away_tz = _series_value(
        row,
        "away_time_zones_crossed",
    )

    home_tz = _series_value(
        row,
        "home_time_zones_crossed",
    )

    away_e2w = _series_value(
        row,
        "away_east_to_west",
    )

    home_e2w = _series_value(
        row,
        "home_east_to_west",
    )

    away_w2e = _series_value(
        row,
        "away_west_to_east",
    )

    home_w2e = _series_value(
        row,
        "home_west_to_east",
    )

    international = _series_value(
        row,
        "international_flag",
    )

    return {
        "travel_net_miles_1000":
            (
                None
                if (
                    away_miles is None
                    or home_miles is None
                )
                else (
                    away_miles
                    - home_miles
                ) / 1000.0
            ),
        "travel_net_time_zones":
            (
                None
                if (
                    away_tz is None
                    or home_tz is None
                )
                else (
                    away_tz
                    - home_tz
                )
            ),
        "travel_net_east_to_west":
            (
                None
                if (
                    away_e2w is None
                    or home_e2w is None
                )
                else (
                    away_e2w
                    - home_e2w
                )
            ),
        "travel_net_west_to_east":
            (
                None
                if (
                    away_w2e is None
                    or home_w2e is None
                )
                else (
                    away_w2e
                    - home_w2e
                )
            ),
        "travel_international":
            international,
    }


def weather_is_exposed(
    row: pd.Series | None,
) -> bool:
    if row is None:
        return False

    roof = clean(
        row.get(
            "roof"
        )
    ).casefold()

    roof_type = clean(
        row.get(
            "roof_type"
        )
    ).casefold()

    dome_flag = _series_value(
        row,
        "dome_flag",
    )

    open_air_flag = _series_value(
        row,
        "open_air_flag",
    )

    if (
        dome_flag == 1.0
        or "dome" in roof
        or "indoor" in roof
        or "closed" in roof
        or "dome" in roof_type
        or "indoor" in roof_type
        or "closed" in roof_type
    ):
        return False

    if open_air_flag == 1.0:
        return True

    return (
        "open_air" in roof
        or "open air" in roof
        or "outdoor" in roof
    )


def build_weather_features(
    row: pd.Series | None,
) -> dict[
    str,
    float | None,
]:
    if row is None:
        return {
            feature: None
            for feature
            in SUPPORTED_WEATHER_FEATURES
        }

    return {
        "weather_temperature_c":
            _series_value(
                row,
                "temperature",
            ),
        "weather_wind_speed_ms":
            _series_value(
                row,
                "wind_speed",
            ),
        "weather_wind_gust_ms":
            _series_value(
                row,
                "wind_gust",
            ),
        "weather_humidity_pct":
            _series_value(
                row,
                "humidity",
            ),
        "weather_rain_flag":
            _series_value(
                row,
                "rain_flag",
            ),
        "weather_snow_flag":
            _series_value(
                row,
                "snow_flag",
            ),
    }


def calculate_feature_adjustment(
    features: dict[
        str,
        float | None,
    ],
    coefficients: dict[
        str,
        dict[
            str,
            float,
        ],
    ],
) -> tuple[
    float,
    int,
]:
    adjustment = 0.0
    used = 0

    for (
        feature,
        settings,
    ) in coefficients.items():
        value = features.get(
            feature
        )

        if value is None:
            continue

        coefficient = float(
            settings[
                "coefficient"
            ]
        )

        center = float(
            settings.get(
                "center",
                0.0,
            )
        )

        adjustment += (
            (
                float(
                    value
                )
                - center
            )
            * coefficient
        )

        used += 1

    return (
        float(
            adjustment
        ),
        used,
    )


def calculate_travel_adjustment(
    row: pd.Series | None,
    coefficients: dict[
        str,
        dict[
            str,
            float,
        ],
    ],
) -> tuple[
    dict[
        str,
        float | None,
    ],
    float,
    int,
]:
    features = build_travel_features(
        row
    )

    adjustment, used = (
        calculate_feature_adjustment(
            features,
            coefficients,
        )
    )

    return (
        features,
        adjustment,
        used,
    )


def calculate_weather_adjustment(
    row: pd.Series | None,
    coefficients: dict[
        str,
        dict[
            str,
            float,
        ],
    ],
) -> tuple[
    dict[
        str,
        float | None,
    ],
    bool,
    float,
    int,
]:
    features = build_weather_features(
        row
    )

    exposed = weather_is_exposed(
        row
    )

    if not exposed:
        return (
            features,
            False,
            0.0,
            0,
        )

    adjustment, used = (
        calculate_feature_adjustment(
            features,
            coefficients,
        )
    )

    return (
        features,
        True,
        adjustment,
        used,
    )


class TeamResolver:
    def __init__(
        self,
        team_map: pd.DataFrame,
    ) -> None:
        required = [
            "team_id",
            "canonical_team",
        ]

        missing = [
            column
            for column in required
            if column not in team_map.columns
        ]

        if missing:
            raise ValueError(
                "team_map.csv missing required columns: "
                f"{missing}"
            )

        self.alias_to_team: dict[
            str,
            str,
        ] = {}

        self.team_to_id: dict[
            str,
            str,
        ] = {}

        self.id_to_team: dict[
            str,
            str,
        ] = {}

        alias_columns = [
            "canonical_team",
            "alias",
            "location",
            "nickname",
            "shortDisplayName",
            "team_slug",
        ]

        for _, row in team_map.iterrows():
            canonical = clean(
                row.get(
                    "canonical_team"
                )
            )

            team_id = clean(
                row.get(
                    "team_id"
                )
            )

            if not canonical:
                continue

            self.team_to_id.setdefault(
                canonical,
                team_id,
            )

            if team_id:
                self.id_to_team.setdefault(
                    team_id,
                    canonical,
                )

            values = [
                canonical
            ]

            for column in alias_columns:
                if column not in team_map.columns:
                    continue

                value = clean(
                    row.get(
                        column
                    )
                )

                if value:
                    values.append(
                        value
                    )

            location = clean(
                row.get(
                    "location"
                )
            )

            nickname = clean(
                row.get(
                    "nickname"
                )
            )

            if (
                location
                and nickname
            ):
                values.append(
                    f"{location} {nickname}"
                )

            for value in values:
                key = normalize_key(
                    value
                )

                if not key:
                    continue

                prior = self.alias_to_team.get(
                    key
                )

                if (
                    prior is None
                    or prior == canonical
                ):
                    self.alias_to_team[
                        key
                    ] = canonical

    def resolve(
        self,
        value: object,
    ) -> str:
        raw = clean(
            value
        )

        if not raw:
            return ""

        return self.alias_to_team.get(
            normalize_key(
                raw
            ),
            raw,
        )

    def team_id(
        self,
        value: object,
    ) -> str:
        return self.team_to_id.get(
            self.resolve(
                value
            ),
            "",
        )


def shrink_metric(
    team_mean: pd.Series,
    team_count: pd.Series,
    global_mean: float,
    strength: float = 2.0,
) -> pd.Series:
    count = pd.to_numeric(
        team_count,
        errors="coerce",
    ).fillna(
        0.0
    )

    weight = count / (
        count
        + strength
    )

    return (
        weight
        * team_mean
        + (
            1.0
            - weight
        )
        * global_mean
    )


def build_prior_table(
    team_stats: pd.DataFrame,
    resolver: TeamResolver,
) -> pd.DataFrame:
    work = team_stats.copy()

    work[
        "team"
    ] = work[
        "team"
    ].map(
        resolver.resolve
    )

    for metric in TEAM_METRICS:
        work[
            metric
        ] = pd.to_numeric(
            work[
                metric
            ],
            errors="coerce",
        )

    work = work[
        work[
            "team"
        ].map(
            clean
        ).ne(
            ""
        )
    ].copy()

    if work.empty:
        raise ValueError(
            "Prior team-stats file has no usable team rows."
        )

    grouped_mean = (
        work.groupby(
            "team",
            as_index=False,
        )[
            TEAM_METRICS
        ]
        .mean()
    )

    grouped_count = (
        work.groupby(
            "team",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "prior_team_weeks"
            }
        )
    )

    prior = grouped_mean.merge(
        grouped_count,
        on="team",
        how="left",
    )

    for metric in TEAM_METRICS:
        global_mean = float(
            work[
                metric
            ].mean(
                skipna=True
            )
        )

        if not math.isfinite(
            global_mean
        ):
            global_mean = 0.0

        prior[
            metric
        ] = shrink_metric(
            prior[
                metric
            ],
            prior[
                "prior_team_weeks"
            ],
            global_mean,
        )

    prior[
        "net_epa"
    ] = (
        prior[
            "off_epa_per_play"
        ]
        - prior[
            "def_epa_per_play"
        ]
    )

    prior[
        "success_edge"
    ] = (
        prior[
            "off_success_rate"
        ]
        - prior[
            "def_success_rate"
        ]
    )

    prior[
        "ypp_edge"
    ] = (
        prior[
            "yards_per_play"
        ]
        - prior[
            "yards_per_play_allowed"
        ]
    )

    prior[
        "ppd_edge"
    ] = (
        prior[
            "points_per_drive"
        ]
        - prior[
            "points_per_drive_allowed"
        ]
    )

    prior[
        "red_zone_edge"
    ] = (
        prior[
            "red_zone_td_rate"
        ]
        - prior[
            "red_zone_td_rate_allowed"
        ]
    )

    strength_parts = {
        "net_epa": 0.30,
        "ppd_edge": 0.25,
        "ypp_edge": 0.15,
        "success_edge": 0.15,
        "red_zone_edge": 0.05,
        "early_down_epa": 0.05,
        "third_down_conversion_rate": 0.05,
    }

    prior[
        "prior_strength_raw"
    ] = 0.0

    for (
        metric,
        weight,
    ) in strength_parts.items():
        values = pd.to_numeric(
            prior[
                metric
            ],
            errors="coerce",
        )

        mean = float(
            values.mean(
                skipna=True
            )
        )

        std = float(
            values.std(
                skipna=True,
                ddof=0,
            )
        )

        if (
            not math.isfinite(
                std
            )
            or std < 1e-9
        ):
            z = pd.Series(
                0.0,
                index=prior.index,
            )

        else:
            z = (
                values.fillna(
                    mean
                )
                - mean
            ) / std

        prior[
            "prior_strength_raw"
        ] += (
            weight
            * z
        )

    raw_mean = float(
        prior[
            "prior_strength_raw"
        ].mean()
    )

    raw_std = float(
        prior[
            "prior_strength_raw"
        ].std(
            ddof=0
        )
    )

    if (
        not math.isfinite(
            raw_std
        )
        or raw_std < 1e-9
    ):
        prior[
            "prior_strength_z"
        ] = 0.0

    else:
        prior[
            "prior_strength_z"
        ] = (
            prior[
                "prior_strength_raw"
            ]
            - raw_mean
        ) / raw_std

    return prior


def load_fpi(
    path: Path,
    resolver: TeamResolver,
) -> pd.DataFrame:
    if not path.is_file():
        print(
            "WARNING: FPI file not found; "
            "FPI component disabled: "
            f"{path}"
        )

        return pd.DataFrame(
            columns=[
                "team",
                "team_id",
                "fpi",
                "epaoffense",
                "epadefense",
            ]
        )

    fpi = read_csv(
        path,
        [
            "team_id",
            "fpi",
        ],
        "team power index",
    )

    fpi[
        "team_id"
    ] = fpi[
        "team_id"
    ].map(
        clean
    )

    fpi[
        "team"
    ] = (
        fpi[
            "team_id"
        ]
        .map(
            resolver.id_to_team
        )
        .fillna(
            ""
        )
    )

    for column in [
        "fpi",
        "epaoffense",
        "epadefense",
    ]:
        if column not in fpi.columns:
            fpi[
                column
            ] = np.nan

        fpi[
            column
        ] = pd.to_numeric(
            fpi[
                column
            ],
            errors="coerce",
        )

    fpi = (
        fpi[
            fpi[
                "team"
            ].ne(
                ""
            )
        ]
        .drop_duplicates(
            "team",
            keep="last",
        )
    )

    return fpi[
        [
            "team",
            "team_id",
            "fpi",
            "epaoffense",
            "epadefense",
        ]
    ]


def scale_prior_to_fpi(
    prior: pd.DataFrame,
    fpi: pd.DataFrame,
) -> pd.DataFrame:
    result = prior.copy()

    fpi_std = float(
        pd.to_numeric(
            fpi.get(
                "fpi"
            ),
            errors="coerce",
        ).std(
            ddof=0
        )
    )

    if (
        not math.isfinite(
            fpi_std
        )
        or fpi_std < 1.0
    ):
        fpi_std = 10.0

    result[
        "prior_rating"
    ] = (
        result[
            "prior_strength_z"
        ]
        * fpi_std
    )

    return result


def load_espn_predictions(
    predictions_dir: Path,
    season: int,
    week: int,
    resolver: TeamResolver,
) -> pd.DataFrame:
    files = sorted(
        predictions_dir.glob(
            f"{season}_*_{week}_clean_predictions.csv"
        )
    )

    if not files:
        print(
            "WARNING: no finalized ESPN prediction file found; "
            "ESPN predictor component disabled: "
            f"{predictions_dir}"
        )

        return pd.DataFrame()

    required = [
        "game_id",
        "home_team",
        "away_team",
        "home_PtDiff",
        "away_PtDiff",
        "home_prob",
        "away_prob",
        "matchupQuality",
    ]

    frames: list[
        pd.DataFrame
    ] = []

    for path in files:
        frame = read_csv(
            path,
            required,
            "finalized ESPN predictions",
        )

        if "season" in frame.columns:
            frame = frame[
                pd.to_numeric(
                    frame[
                        "season"
                    ],
                    errors="coerce",
                ).eq(
                    season
                )
            ].copy()

        if "week" in frame.columns:
            frame = frame[
                pd.to_numeric(
                    frame[
                        "week"
                    ],
                    errors="coerce",
                ).eq(
                    week
                )
            ].copy()

        frames.append(
            frame
        )

    predictions = pd.concat(
        frames,
        ignore_index=True,
    )

    if predictions.empty:
        print(
            "WARNING: finalized ESPN prediction files "
            "contained no matching rows."
        )

        return pd.DataFrame()

    predictions[
        "game_id"
    ] = predictions[
        "game_id"
    ].map(
        clean
    )

    predictions[
        "home_team_resolved"
    ] = predictions[
        "home_team"
    ].map(
        resolver.resolve
    )

    predictions[
        "away_team_resolved"
    ] = predictions[
        "away_team"
    ].map(
        resolver.resolve
    )

    predictions = predictions[
        predictions[
            "game_id"
        ].ne(
            ""
        )
    ].copy()

    duplicate_mask = predictions[
        "game_id"
    ].duplicated(
        keep=False
    )

    if duplicate_mask.any():
        duplicates = predictions.loc[
            duplicate_mask,
            "game_id",
        ].tolist()

        raise ValueError(
            "Duplicate game_id values in finalized ESPN predictions: "
            f"{duplicates[:10]}"
        )

    return predictions


def build_home_stadium_lookup(
    stadium_map_path: Path,
    resolver: TeamResolver,
) -> dict[
    str,
    set[str],
]:
    if not stadium_map_path.is_file():
        print(
            "WARNING: stadium_map.csv not found; "
            "neutral-site sanity correction disabled: "
            f"{stadium_map_path}"
        )

        return {}

    stadium_map = read_csv(
        stadium_map_path,
        [
            "team_id",
            "stadium",
        ],
        "stadium map",
    )

    lookup: dict[
        str,
        set[str],
    ] = {}

    for _, row in stadium_map.iterrows():
        team = ""

        if "team" in stadium_map.columns:
            team = resolver.resolve(
                row.get(
                    "team"
                )
            )

        if not team:
            team = resolver.id_to_team.get(
                clean(
                    row.get(
                        "team_id"
                    )
                ),
                "",
            )

        if not team:
            continue

        names = lookup.setdefault(
            team,
            set(),
        )

        for column in [
            "stadium",
            "venue_full_name",
        ]:
            if column not in stadium_map.columns:
                continue

            key = normalize_key(
                row.get(
                    column
                )
            )

            if key:
                names.add(
                    key
                )

    return lookup


def resolve_neutral_site(
    sched_row: pd.Series,
    home_team: str,
    home_stadium_lookup: dict[
        str,
        set[str],
    ],
) -> tuple[
    bool,
    bool,
    bool,
    bool,
]:
    original_neutral = as_bool(
        sched_row.get(
            "neutral_site"
        )
    )

    stadium_key = normalize_key(
        sched_row.get(
            "stadium"
        )
    )

    home_stadium_match = bool(
        stadium_key
        and stadium_key
        in home_stadium_lookup.get(
            home_team,
            set(),
        )
    )

    corrected = (
        original_neutral
        and home_stadium_match
    )

    effective_neutral = (
        original_neutral
        and not corrected
    )

    return (
        original_neutral,
        effective_neutral,
        corrected,
        home_stadium_match,
    )


def position_cost(
    position: object,
) -> float:
    pos = (
        clean(
            position
        )
        .upper()
        .replace(
            " ",
            "",
        )
    )

    return POSITION_POINT_COST.get(
        pos,
        0.35,
    )


def injury_status_multiplier(
    status: object,
) -> float:
    text = clean(
        status
    ).casefold()

    for (
        key,
        multiplier,
    ) in OUT_STATUS_MULTIPLIER.items():
        if key in text:
            return multiplier

    return 0.0


def build_injury_lookup(
    injuries_path: Path,
    resolver: TeamResolver,
    fresh_days: int,
) -> dict[
    str,
    pd.DataFrame,
]:
    if not injuries_path.is_file():
        print(
            "WARNING: injury file not found; "
            "injury adjustment disabled: "
            f"{injuries_path}"
        )

        return {}

    injuries = read_csv(
        injuries_path,
        [
            "team",
            "position",
            "game_status",
            "report_date",
        ],
        "injuries",
    )

    injuries[
        "team"
    ] = injuries[
        "team"
    ].map(
        resolver.resolve
    )

    injuries[
        "report_ts"
    ] = pd.to_datetime(
        injuries[
            "report_date"
        ],
        errors="coerce",
        utc=True,
    )

    injuries[
        "status_multiplier"
    ] = injuries[
        "game_status"
    ].map(
        injury_status_multiplier
    )

    injuries[
        "position_cost"
    ] = injuries[
        "position"
    ].map(
        position_cost
    )

    injuries[
        "raw_penalty"
    ] = (
        injuries[
            "status_multiplier"
        ]
        * injuries[
            "position_cost"
        ]
    )

    return {
        team:
            group.copy()
        for (
            team,
            group,
        ) in injuries.groupby(
            "team"
        )
    }


def injury_summary_for_game(
    team: str,
    game_date: object,
    injury_lookup: dict[
        str,
        pd.DataFrame,
    ],
    fresh_days: int,
) -> tuple[
    int,
    int,
    int,
    float,
]:
    group = injury_lookup.get(
        team
    )

    if (
        group is None
        or group.empty
    ):
        return (
            0,
            0,
            0,
            0.0,
        )

    game_ts = pd.to_datetime(
        clean(
            game_date
        ),
        errors="coerce",
        utc=True,
    )

    if pd.isna(
        game_ts
    ):
        return (
            0,
            0,
            0,
            0.0,
        )

    age_days = (
        game_ts
        - group[
            "report_ts"
        ]
    ).dt.total_seconds() / 86400.0

    fresh = group[
        group[
            "report_ts"
        ].notna()
        & age_days.ge(
            0
        )
        & age_days.le(
            float(
                fresh_days
            )
        )
        & group[
            "status_multiplier"
        ].gt(
            0
        )
    ].copy()

    if fresh.empty:
        return (
            0,
            0,
            0,
            0.0,
        )

    statuses = fresh[
        "game_status"
    ].map(
        lambda value:
            clean(
                value
            ).casefold()
    )

    out_count = int(
        statuses.str.contains(
            "out",
            regex=False,
        ).sum()
    )

    doubtful_count = int(
        statuses.str.contains(
            "doubtful",
            regex=False,
        ).sum()
    )

    questionable_count = int(
        statuses.str.contains(
            "questionable",
            regex=False,
        ).sum()
    )

    penalty = min(
        7.0,
        float(
            fresh[
                "raw_penalty"
            ].sum()
        ),
    )

    return (
        out_count,
        doubtful_count,
        questionable_count,
        penalty,
    )


def weighted_blend(
    components: list[
        tuple[
            float | None,
            float,
        ]
    ],
) -> tuple[
    float | None,
    list[float],
]:
    valid = [
        (
            value,
            weight,
        )
        for (
            value,
            weight,
        ) in components
        if (
            value is not None
            and math.isfinite(
                float(
                    value
                )
            )
            and weight > 0
        )
    ]

    if not valid:
        return (
            None,
            [
                0.0
                for _ in components
            ],
        )

    weight_sum = sum(
        weight
        for (
            _,
            weight,
        ) in valid
    )

    normalized: list[
        float
    ] = []

    numerator = 0.0

    for (
        value,
        weight,
    ) in components:
        if (
            value is None
            or not math.isfinite(
                float(
                    value
                )
            )
            or weight <= 0
        ):
            normalized.append(
                0.0
            )

            continue

        effective = (
            weight
            / weight_sum
        )

        normalized.append(
            effective
        )

        numerator += (
            float(
                value
            )
            * effective
        )

    return (
        numerator,
        normalized,
    )


def prior_total_estimate(
    home: pd.Series,
    away: pd.Series,
    drives_per_team: float,
) -> float | None:
    values = [
        as_float(
            home.get(
                "points_per_drive"
            )
        ),
        as_float(
            away.get(
                "points_per_drive_allowed"
            )
        ),
        as_float(
            away.get(
                "points_per_drive"
            )
        ),
        as_float(
            home.get(
                "points_per_drive_allowed"
            )
        ),
    ]

    if any(
        value is None
        for value in values
    ):
        return None

    home_ppd = (
        values[
            0
        ]
        + values[
            1
        ]
    ) / 2.0

    away_ppd = (
        values[
            2
        ]
        + values[
            3
        ]
    ) / 2.0

    total = (
        home_ppd
        + away_ppd
    ) * drives_per_team

    return float(
        np.clip(
            total,
            20.0,
            90.0,
        )
    )


def normal_cdf(
    z: float,
) -> float:
    probability = 0.5 * (
        1.0
        + math.erf(
            z
            / math.sqrt(
                2.0
            )
        )
    )

    return float(
        np.clip(
            probability,
            PROBABILITY_EPS,
            1.0
            - PROBABILITY_EPS,
        )
    )


def build_betting_probabilities(
    predicted_margin: float,
    predicted_total: float,
    home_spread: float | None,
    market_total: float | None,
    margin_sd: float,
    total_sd: float,
) -> dict[
    str,
    float,
]:
    home_win = normal_cdf(
        predicted_margin
        / margin_sd
    )

    away_win = (
        1.0
        - home_win
    )

    if home_spread is None:
        home_cover = 0.5
        away_cover = 0.5

    else:
        home_cover = normal_cdf(
            (
                predicted_margin
                + home_spread
            )
            / margin_sd
        )

        away_cover = (
            1.0
            - home_cover
        )

    if market_total is None:
        over = 0.5
        under = 0.5

    else:
        over = normal_cdf(
            (
                predicted_total
                - market_total
            )
            / total_sd
        )

        under = (
            1.0
            - over
        )

    return {
        "home_win_probability":
            home_win,
        "away_win_probability":
            away_win,
        "home_cover_probability":
            home_cover,
        "away_cover_probability":
            away_cover,
        "over_probability":
            over,
        "under_probability":
            under,
    }


def validate_probability_output(
    result: pd.DataFrame,
) -> None:
    missing = [
        column
        for column
        in REQUIRED_PREDICTION_COLUMNS
        if column not in result.columns
    ]

    if missing:
        raise RuntimeError(
            "Projection output missing "
            "downstream-required columns: "
            f"{missing}"
        )

    pairs = [
        (
            "home_win_probability",
            "away_win_probability",
        ),
        (
            "home_cover_probability",
            "away_cover_probability",
        ),
        (
            "over_probability",
            "under_probability",
        ),
    ]

    for (
        first_column,
        second_column,
    ) in pairs:
        first = pd.to_numeric(
            result[
                first_column
            ],
            errors="coerce",
        )

        second = pd.to_numeric(
            result[
                second_column
            ],
            errors="coerce",
        )

        if (
            first.isna().any()
            or second.isna().any()
        ):
            raise RuntimeError(
                "Non-numeric betting probability in "
                f"{first_column}/"
                f"{second_column}"
            )

        if (
            first.lt(
                0.0
            ).any()
            or first.gt(
                1.0
            ).any()
            or second.lt(
                0.0
            ).any()
            or second.gt(
                1.0
            ).any()
        ):
            raise RuntimeError(
                "Betting probability outside [0,1] in "
                f"{first_column}/"
                f"{second_column}"
            )

        if not np.allclose(
            (
                first
                + second
            ).to_numpy(
                dtype=float
            ),
            1.0,
            rtol=0.0,
            atol=2e-6,
        ):
            raise RuntimeError(
                "Complementary probabilities "
                "do not sum to 1 for "
                f"{first_column}/"
                f"{second_column}"
            )


def build_projection(
    schedule: pd.DataFrame,
    prior: pd.DataFrame,
    fpi: pd.DataFrame,
    espn_predictions: pd.DataFrame,
    resolver: TeamResolver,
    home_stadium_lookup: dict[
        str,
        set[str],
    ],
    injury_lookup: dict[
        str,
        pd.DataFrame,
    ],
    travel: pd.DataFrame,
    weather: pd.DataFrame,
    travel_weather_coefficients: dict[
        str,
        dict[
            str,
            dict[
                str,
                float,
            ],
        ],
    ],
    args: argparse.Namespace,
) -> pd.DataFrame:
    prior_lookup = prior.set_index(
        "team",
        drop=False,
    )

    fpi_lookup = (
        fpi.set_index(
            "team",
            drop=False,
        )
        if not fpi.empty
        else None
    )

    espn_lookup = (
        espn_predictions.set_index(
            "game_id",
            drop=False,
        )
        if not espn_predictions.empty
        else None
    )

    travel_lookup = (
        travel.set_index(
            "game_id",
            drop=False,
        )
        if not travel.empty
        else None
    )

    weather_lookup = (
        weather.set_index(
            "game_id",
            drop=False,
        )
        if not weather.empty
        else None
    )

    margin_feature_coefficients = (
        travel_weather_coefficients.get(
            "margin",
            {},
        )
    )

    total_feature_coefficients = (
        travel_weather_coefficients.get(
            "total",
            {},
        )
    )

    output_rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    fallback_prior = prior.mean(
        numeric_only=True
    )

    fallback_prior[
        "prior_team_weeks"
    ] = 0

    fallback_prior[
        "prior_rating"
    ] = 0.0

    for _, sched_row in schedule.iterrows():
        game_id = normalize_game_id(
            sched_row.get(
                "game_id"
            )
        )

        home_team = resolver.resolve(
            sched_row.get(
                "home_team"
            )
        )

        away_team = resolver.resolve(
            sched_row.get(
                "away_team"
            )
        )

        if home_team in prior_lookup.index:
            home_source_prior = prior_lookup.loc[
                home_team
            ]

            home_prior_team_weeks = int(
                home_source_prior[
                    "prior_team_weeks"
                ]
            )

        else:
            home_source_prior = None
            home_prior_team_weeks = 0

        if away_team in prior_lookup.index:
            away_source_prior = prior_lookup.loc[
                away_team
            ]

            away_prior_team_weeks = int(
                away_source_prior[
                    "prior_team_weeks"
                ]
            )

        else:
            away_source_prior = None
            away_prior_team_weeks = 0

        home_prior_fallback = (
            home_source_prior is None
            or home_prior_team_weeks
            < MIN_PRIOR_TEAM_WEEKS
        )

        away_prior_fallback = (
            away_source_prior is None
            or away_prior_team_weeks
            < MIN_PRIOR_TEAM_WEEKS
        )

        home_prior = (
            fallback_prior
            if home_prior_fallback
            else home_source_prior
        )

        away_prior = (
            fallback_prior
            if away_prior_fallback
            else away_source_prior
        )

        (
            original_neutral,
            effective_neutral,
            neutral_corrected,
            home_stadium_match,
        ) = resolve_neutral_site(
            sched_row,
            home_team,
            home_stadium_lookup,
        )

        home_field = (
            0.0
            if effective_neutral
            else float(
                args.home_field
            )
        )

        if (
            home_prior_fallback
            or away_prior_fallback
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
                + home_field
            )

        home_fpi = None
        away_fpi = None

        if fpi_lookup is not None:
            if home_team in fpi_lookup.index:
                home_fpi = as_float(
                    fpi_lookup.loc[
                        home_team
                    ].get(
                        "fpi"
                    )
                )

            if away_team in fpi_lookup.index:
                away_fpi = as_float(
                    fpi_lookup.loc[
                        away_team
                    ].get(
                        "fpi"
                    )
                )

        fpi_margin = None

        if (
            home_fpi is not None
            and away_fpi is not None
        ):
            fpi_margin = (
                home_fpi
                - away_fpi
                + home_field
            )

        home_spread = as_float(
            sched_row.get(
                "home_spread"
            )
        )

        market_margin = (
            -home_spread
            if home_spread is not None
            else None
        )

        espn_match_valid = False
        espn_margin_consistent = False
        espn_home_margin = None
        espn_home_ptdiff = None
        espn_away_ptdiff = None
        espn_home_prob = None
        espn_away_prob = None
        espn_tie_prob = None
        espn_matchup_quality = None

        if (
            espn_lookup is not None
            and game_id in espn_lookup.index
        ):
            espn_row = espn_lookup.loc[
                game_id
            ]

            espn_home_team = resolver.resolve(
                espn_row.get(
                    "home_team"
                )
            )

            espn_away_team = resolver.resolve(
                espn_row.get(
                    "away_team"
                )
            )

            espn_match_valid = (
                espn_home_team
                == home_team
                and espn_away_team
                == away_team
            )

            espn_home_ptdiff = as_float(
                espn_row.get(
                    "home_PtDiff"
                )
            )

            espn_away_ptdiff = as_float(
                espn_row.get(
                    "away_PtDiff"
                )
            )

            espn_home_prob = as_float(
                espn_row.get(
                    "home_prob"
                )
            )

            espn_away_prob = as_float(
                espn_row.get(
                    "away_prob"
                )
            )

            espn_tie_prob = as_float(
                espn_row.get(
                    "tie_prob"
                )
            )

            espn_matchup_quality = as_float(
                espn_row.get(
                    "matchupQuality"
                )
            )

            if espn_home_ptdiff is not None:
                if espn_away_ptdiff is None:
                    espn_margin_consistent = True

                else:
                    espn_margin_consistent = (
                        abs(
                            espn_home_ptdiff
                            + espn_away_ptdiff
                        )
                        <= ESPN_MARGIN_SYMMETRY_TOLERANCE
                    )

            if (
                espn_match_valid
                and espn_margin_consistent
            ):
                espn_home_margin = (
                    espn_home_ptdiff
                )

        (
            blended_margin,
            margin_weights,
        ) = weighted_blend(
            [
                (
                    market_margin,
                    args.market_margin_weight,
                ),
                (
                    fpi_margin,
                    args.fpi_margin_weight,
                ),
                (
                    espn_home_margin,
                    args.espn_margin_weight,
                ),
                (
                    prior_margin,
                    args.prior_margin_weight,
                ),
            ]
        )

        if blended_margin is None:
            raise RuntimeError(
                "No usable margin component "
                f"for game_id={game_id}"
            )

        travel_row = (
            travel_lookup.loc[
                game_id
            ]
            if (
                travel_lookup is not None
                and game_id in travel_lookup.index
            )
            else None
        )

        (
            travel_features,
            travel_margin_adjustment,
            travel_features_used,
        ) = calculate_travel_adjustment(
            travel_row,
            margin_feature_coefficients,
        )

        (
            home_out,
            home_doubtful,
            home_questionable,
            home_injury_penalty,
        ) = injury_summary_for_game(
            home_team,
            sched_row.get(
                "game_date"
            ),
            injury_lookup,
            args.fresh_injury_days,
        )

        (
            away_out,
            away_doubtful,
            away_questionable,
            away_injury_penalty,
        ) = injury_summary_for_game(
            away_team,
            sched_row.get(
                "game_date"
            ),
            injury_lookup,
            args.fresh_injury_days,
        )

        injury_adjustment = (
            away_injury_penalty
            - home_injury_penalty
        )

        predicted_margin_before_travel = (
            blended_margin
            + injury_adjustment
        )

        predicted_margin = (
            predicted_margin_before_travel
            + travel_margin_adjustment
        )

        prior_total = prior_total_estimate(
            home_prior,
            away_prior,
            args.drives_per_team,
        )

        market_total = as_float(
            sched_row.get(
                "total"
            )
        )

        (
            predicted_total,
            total_weights,
        ) = weighted_blend(
            [
                (
                    market_total,
                    args.market_total_weight,
                ),
                (
                    prior_total,
                    max(
                        0.0,
                        1.0
                        - args.market_total_weight,
                    ),
                ),
            ]
        )

        if predicted_total is None:
            raise RuntimeError(
                "No usable total component "
                f"for game_id={game_id}"
            )

        predicted_total_before_weather = float(
            predicted_total
        )

        weather_row = (
            weather_lookup.loc[
                game_id
            ]
            if (
                weather_lookup is not None
                and game_id in weather_lookup.index
            )
            else None
        )

        (
            weather_features,
            weather_exposed,
            weather_total_adjustment,
            weather_features_used,
        ) = calculate_weather_adjustment(
            weather_row,
            total_feature_coefficients,
        )

        predicted_total = (
            predicted_total_before_weather
            + weather_total_adjustment
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

        betting_probabilities = (
            build_betting_probabilities(
                predicted_margin=
                    predicted_margin,
                predicted_total=
                    predicted_total,
                home_spread=
                    home_spread,
                market_total=
                    market_total,
                margin_sd=float(
                    args.margin_sd
                ),
                total_sd=float(
                    args.total_sd
                ),
            )
        )

        rec: dict[
            str,
            object,
        ] = {
            column:
                clean(
                    sched_row.get(
                        column
                    )
                )
            for column
            in OUTPUT_BASE_COLUMNS
        }

        rec.update(
            {
                "neutral_site_original":
                    int(
                        original_neutral
                    ),

                "neutral_site":
                    int(
                        effective_neutral
                    ),

                "neutral_site_corrected":
                    int(
                        neutral_corrected
                    ),

                "home_stadium_match":
                    int(
                        home_stadium_match
                    ),

                "home_field_points":
                    round(
                        home_field,
                        4,
                    ),

                "home_team":
                    home_team,

                "away_team":
                    away_team,

                "home_team_id":
                    resolver.team_id(
                        home_team
                    ),

                "away_team_id":
                    resolver.team_id(
                        away_team
                    ),

                "home_prior_team_weeks":
                    home_prior_team_weeks,

                "away_prior_team_weeks":
                    away_prior_team_weeks,

                "home_prior_fallback":
                    int(
                        home_prior_fallback
                    ),

                "away_prior_fallback":
                    int(
                        away_prior_fallback
                    ),

                "home_prior_rating":
                    round(
                        float(
                            home_prior[
                                "prior_rating"
                            ]
                        ),
                        4,
                    ),

                "away_prior_rating":
                    round(
                        float(
                            away_prior[
                                "prior_rating"
                            ]
                        ),
                        4,
                    ),

                "prior_home_margin":
                    (
                        None
                        if prior_margin is None
                        else round(
                            prior_margin,
                            4,
                        )
                    ),

                "home_fpi":
                    (
                        None
                        if home_fpi is None
                        else round(
                            home_fpi,
                            4,
                        )
                    ),

                "away_fpi":
                    (
                        None
                        if away_fpi is None
                        else round(
                            away_fpi,
                            4,
                        )
                    ),

                "fpi_home_margin":
                    (
                        None
                        if fpi_margin is None
                        else round(
                            fpi_margin,
                            4,
                        )
                    ),

                "espn_prediction_match_valid":
                    int(
                        espn_match_valid
                    ),

                "espn_margin_consistent":
                    int(
                        espn_margin_consistent
                    ),

                "espn_matchup_quality":
                    (
                        None
                        if espn_matchup_quality is None
                        else round(
                            espn_matchup_quality,
                            4,
                        )
                    ),

                "espn_home_prob":
                    (
                        None
                        if espn_home_prob is None
                        else round(
                            espn_home_prob,
                            6,
                        )
                    ),

                "espn_away_prob":
                    (
                        None
                        if espn_away_prob is None
                        else round(
                            espn_away_prob,
                            6,
                        )
                    ),

                "espn_tie_prob":
                    (
                        None
                        if espn_tie_prob is None
                        else round(
                            espn_tie_prob,
                            6,
                        )
                    ),

                "espn_home_ptdiff":
                    (
                        None
                        if espn_home_ptdiff is None
                        else round(
                            espn_home_ptdiff,
                            4,
                        )
                    ),

                "espn_away_ptdiff":
                    (
                        None
                        if espn_away_ptdiff is None
                        else round(
                            espn_away_ptdiff,
                            4,
                        )
                    ),

                "espn_home_margin":
                    (
                        None
                        if espn_home_margin is None
                        else round(
                            espn_home_margin,
                            4,
                        )
                    ),

                "market_home_margin":
                    (
                        None
                        if market_margin is None
                        else round(
                            market_margin,
                            4,
                        )
                    ),

                "margin_weight_market":
                    round(
                        margin_weights[
                            0
                        ],
                        4,
                    ),

                "margin_weight_fpi":
                    round(
                        margin_weights[
                            1
                        ],
                        4,
                    ),

                "margin_weight_espn":
                    round(
                        margin_weights[
                            2
                        ],
                        4,
                    ),

                "margin_weight_prior":
                    round(
                        margin_weights[
                            3
                        ],
                        4,
                    ),

                "home_out_count":
                    home_out,

                "home_doubtful_count":
                    home_doubtful,

                "home_questionable_count":
                    home_questionable,

                "away_out_count":
                    away_out,

                "away_doubtful_count":
                    away_doubtful,

                "away_questionable_count":
                    away_questionable,

                "home_injury_penalty":
                    round(
                        home_injury_penalty,
                        4,
                    ),

                "away_injury_penalty":
                    round(
                        away_injury_penalty,
                        4,
                    ),

                "injury_margin_adjustment":
                    round(
                        injury_adjustment,
                        4,
                    ),

                "travel_data_available":
                    int(
                        travel_row is not None
                    ),

                "away_miles_traveled":
                    (
                        None
                        if travel_row is None
                        else as_float(
                            travel_row.get(
                                "away_miles_traveled"
                            )
                        )
                    ),

                "home_miles_traveled":
                    (
                        None
                        if travel_row is None
                        else as_float(
                            travel_row.get(
                                "home_miles_traveled"
                            )
                        )
                    ),

                "travel_net_miles_1000":
                    travel_features.get(
                        "travel_net_miles_1000"
                    ),

                "travel_net_time_zones":
                    travel_features.get(
                        "travel_net_time_zones"
                    ),

                "travel_net_east_to_west":
                    travel_features.get(
                        "travel_net_east_to_west"
                    ),

                "travel_net_west_to_east":
                    travel_features.get(
                        "travel_net_west_to_east"
                    ),

                "travel_international":
                    travel_features.get(
                        "travel_international"
                    ),

                "travel_features_used":
                    travel_features_used,

                "travel_margin_adjustment":
                    round(
                        travel_margin_adjustment,
                        4,
                    ),

                "predicted_margin_before_travel":
                    round(
                        predicted_margin_before_travel,
                        4,
                    ),

                "prior_total":
                    (
                        None
                        if prior_total is None
                        else round(
                            prior_total,
                            4,
                        )
                    ),

                "market_total":
                    (
                        None
                        if market_total is None
                        else round(
                            market_total,
                            4,
                        )
                    ),

                "total_weight_market":
                    round(
                        total_weights[
                            0
                        ],
                        4,
                    ),

                "total_weight_prior":
                    round(
                        total_weights[
                            1
                        ],
                        4,
                    ),

                "weather_data_available":
                    int(
                        weather_row is not None
                    ),

                "weather_exposed":
                    int(
                        weather_exposed
                    ),

                "weather_temperature_c":
                    weather_features.get(
                        "weather_temperature_c"
                    ),

                "weather_wind_speed_ms":
                    weather_features.get(
                        "weather_wind_speed_ms"
                    ),

                "weather_wind_gust_ms":
                    weather_features.get(
                        "weather_wind_gust_ms"
                    ),

                "weather_humidity_pct":
                    weather_features.get(
                        "weather_humidity_pct"
                    ),

                "weather_rain_flag":
                    weather_features.get(
                        "weather_rain_flag"
                    ),

                "weather_snow_flag":
                    weather_features.get(
                        "weather_snow_flag"
                    ),

                "weather_features_used":
                    weather_features_used,

                "weather_total_adjustment":
                    round(
                        weather_total_adjustment,
                        4,
                    ),

                "predicted_total_before_weather":
                    round(
                        predicted_total_before_weather,
                        4,
                    ),

                "predicted_margin":
                    round(
                        predicted_margin,
                        2,
                    ),

                "predicted_total":
                    round(
                        predicted_total,
                        2,
                    ),

                "predicted_home_score":
                    round(
                        predicted_home_score,
                        2,
                    ),

                "predicted_away_score":
                    round(
                        predicted_away_score,
                        2,
                    ),

                "home_win_probability":
                    round(
                        betting_probabilities[
                            "home_win_probability"
                        ],
                        6,
                    ),

                "away_win_probability":
                    round(
                        betting_probabilities[
                            "away_win_probability"
                        ],
                        6,
                    ),

                "home_cover_probability":
                    round(
                        betting_probabilities[
                            "home_cover_probability"
                        ],
                        6,
                    ),

                "away_cover_probability":
                    round(
                        betting_probabilities[
                            "away_cover_probability"
                        ],
                        6,
                    ),

                "over_probability":
                    round(
                        betting_probabilities[
                            "over_probability"
                        ],
                        6,
                    ),

                "under_probability":
                    round(
                        betting_probabilities[
                            "under_probability"
                        ],
                        6,
                    ),

                "probability_margin_sd":
                    round(
                        float(
                            args.margin_sd
                        ),
                        4,
                    ),

                "probability_total_sd":
                    round(
                        float(
                            args.total_sd
                        ),
                        4,
                    ),

                "spread_probability_line_available":
                    int(
                        home_spread is not None
                    ),

                "total_probability_line_available":
                    int(
                        market_total is not None
                    ),

                "projection_version":
                    SCRIPT_VERSION,
            }
        )

        for metric in TEAM_METRICS:
            rec[
                f"home_prior_{metric}"
            ] = round(
                float(
                    home_prior[
                        metric
                    ]
                ),
                6,
            )

            rec[
                f"away_prior_{metric}"
            ] = round(
                float(
                    away_prior[
                        metric
                    ]
                ),
                6,
            )

        output_rows.append(
            rec
        )

    result = pd.DataFrame(
        output_rows
    )

    if result.empty:
        raise RuntimeError(
            "No Week 1 games could be projected."
        )

    if result[
        "game_id"
    ].duplicated().any():
        duplicates = result.loc[
            result[
                "game_id"
            ].duplicated(
                keep=False
            ),
            "game_id",
        ].tolist()

        raise ValueError(
            "Duplicate game_id values in output: "
            f"{duplicates[:10]}"
        )

    validate_probability_output(
        result
    )

    return result


def validate_args(
    args: argparse.Namespace,
) -> None:
    weights = [
        args.market_margin_weight,
        args.fpi_margin_weight,
        args.espn_margin_weight,
        args.prior_margin_weight,
        args.market_total_weight,
    ]

    if any(
        weight < 0
        for weight in weights
    ):
        raise ValueError(
            "Projection weights cannot be negative."
        )

    if (
        args.market_margin_weight
        + args.fpi_margin_weight
        + args.espn_margin_weight
        + args.prior_margin_weight
        <= 0
    ):
        raise ValueError(
            "At least one margin weight must be positive."
        )

    if not (
        0.0
        <= args.market_total_weight
        <= 1.0
    ):
        raise ValueError(
            "--market-total-weight must be between 0 and 1."
        )

    if args.drives_per_team <= 0:
        raise ValueError(
            "--drives-per-team must be positive."
        )

    if args.fresh_injury_days < 0:
        raise ValueError(
            "--fresh-injury-days cannot be negative."
        )

    if args.home_field < 0:
        raise ValueError(
            "--home-field cannot be negative."
        )

    if (
        not math.isfinite(
            float(
                args.margin_sd
            )
        )
        or args.margin_sd <= 0
    ):
        raise ValueError(
            "--margin-sd must be a positive finite number."
        )

    if (
        not math.isfinite(
            float(
                args.total_sd
            )
        )
        or args.total_sd <= 0
    ):
        raise ValueError(
            "--total-sd must be a positive finite number."
        )


def main() -> None:
    args = parse_args()

    validate_args(
        args
    )

    season = args.season

    if season is None:
        season = int(
            os.getenv(
                "CFB_SEASON",
                "2026",
            )
        )

    prior_season = (
        args.prior_season
        or (
            season
            - 1
        )
    )

    root = repo_cfb_root()

    schedule_path = (
        root
        / "00_intake"
        / "schedule"
        / "weekly"
        / f"week_{args.week}_CFB_weekly_schedule.csv"
    )

    prior_path = (
        root
        / "00_intake"
        / "team_stats"
        / f"{prior_season}_team_stats.csv"
    )

    fpi_path = (
        root
        / "data"
        / "team_power_index"
        / f"team_power_index_{season}.csv"
    )

    predictions_dir = (
        root
        / "00_intake"
        / "predictions"
        / "final"
    )

    injuries_path = (
        root
        / "00_intake"
        / "injuries"
        / f"{season}_injuries.csv"
    )

    team_map_path = (
        root
        / "config"
        / "mapping"
        / "team_map.csv"
    )

    stadium_map_path = (
        root
        / "config"
        / "mapping"
        / "stadium_map.csv"
    )

    travel_path = (
        root
        / "data"
        / "travel"
        / f"{season}_week_{args.week}_travel.csv"
    )

    weather_path = (
        root
        / "data"
        / "weather"
        / f"week_{args.week}_CFB_weekly_weather.csv"
    )

    travel_weather_coefficients_path = (
        root
        / "config"
        / "travel_weather_coefficients.csv"
    )

    output_path = (
        root
        / "01_merge"
        / f"week_{args.week}_CFB_enriched.csv"
    )

    schedule = read_csv(
        schedule_path,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "game_date",
            "game_time",
            "game_timezone",
            "game_locked",
            "away_team",
            "home_team",
            "neutral_site",
            "stadium",
            "home_spread",
            "total",
        ],
        "weekly schedule",
    )

    schedule = schedule[
        pd.to_numeric(
            schedule[
                "season"
            ],
            errors="coerce",
        ).eq(
            season
        )
        & pd.to_numeric(
            schedule[
                "week"
            ],
            errors="coerce",
        ).eq(
            args.week
        )
    ].copy()

    if schedule.empty:
        raise ValueError(
            "No schedule rows for "
            f"season={season} "
            f"week={args.week} "
            f"in {schedule_path}"
        )

    team_map = read_csv(
        team_map_path,
        [
            "team_id",
            "canonical_team",
        ],
        "team map",
    )

    resolver = TeamResolver(
        team_map
    )

    team_stats = read_csv(
        prior_path,
        [
            "season",
            "week",
            "team",
            *TEAM_METRICS,
        ],
        "prior-season team stats",
    )

    team_stats = team_stats[
        pd.to_numeric(
            team_stats[
                "season"
            ],
            errors="coerce",
        ).eq(
            prior_season
        )
    ].copy()

    if team_stats.empty:
        raise ValueError(
            "No prior team-stat rows "
            f"for season={prior_season} "
            f"in {prior_path}"
        )

    prior = build_prior_table(
        team_stats,
        resolver,
    )

    fpi = load_fpi(
        fpi_path,
        resolver,
    )

    prior = scale_prior_to_fpi(
        prior,
        fpi,
    )

    espn_predictions = (
        load_espn_predictions(
            predictions_dir,
            season,
            args.week,
            resolver,
        )
    )

    home_stadium_lookup = (
        build_home_stadium_lookup(
            stadium_map_path,
            resolver,
        )
    )

    injury_lookup = (
        build_injury_lookup(
            injuries_path,
            resolver,
            args.fresh_injury_days,
        )
    )

    travel = load_game_feature_file(
        travel_path,
        TRAVEL_REQUIRED_COLUMNS,
        "weekly travel",
    )

    weather = load_game_feature_file(
        weather_path,
        WEATHER_REQUIRED_COLUMNS,
        "weekly weather",
    )

    travel_weather_coefficients = (
        load_travel_weather_coefficients(
            travel_weather_coefficients_path
        )
    )

    predictions = build_projection(
        schedule,
        prior,
        fpi,
        espn_predictions,
        resolver,
        home_stadium_lookup,
        injury_lookup,
        travel,
        weather,
        travel_weather_coefficients,
        args,
    )

    (
        predictions,
        locked_games_preserved,
    ) = preserve_locked_rows(
        predictions,
        schedule,
        output_path,
        "Week 1 projection",
    )

    print(
        "projection_week1.py "
        f"version={SCRIPT_VERSION}"
    )

    print(
        f"season={season}"
    )

    print(
        f"prior_season={prior_season}"
    )

    print(
        f"week={args.week}"
    )

    print(
        f"games={len(predictions)}"
    )

    print(
        "locked_games_preserved="
        f"{locked_games_preserved}"
    )

    print(
        "min_prior_team_weeks="
        f"{MIN_PRIOR_TEAM_WEEKS}"
    )

    print(
        "probability_margin_sd="
        f"{float(args.margin_sd):.4f}"
    )

    print(
        "probability_total_sd="
        f"{float(args.total_sd):.4f}"
    )

    print(
        "home_prior_fallbacks="
        f"{int(predictions['home_prior_fallback'].sum())}"
    )

    print(
        "away_prior_fallbacks="
        f"{int(predictions['away_prior_fallback'].sum())}"
    )

    print(
        "prior_margin_disabled="
        f"{int(predictions['prior_home_margin'].isna().sum())}"
    )

    print(
        "neutral_site_corrections="
        f"{int(predictions['neutral_site_corrected'].sum())}"
    )

    print(
        "with_market_spread="
        f"{int(predictions['market_home_margin'].notna().sum())}"
    )

    print(
        "with_fpi="
        f"{int(predictions['fpi_home_margin'].notna().sum())}"
    )

    print(
        "with_espn="
        f"{int(predictions['espn_home_margin'].notna().sum())}"
    )

    print(
        "with_prior_margin="
        f"{int(predictions['prior_home_margin'].notna().sum())}"
    )

    print(
        "with_market_total="
        f"{int(predictions['market_total'].notna().sum())}"
    )

    print(
        "fresh_injury_adjustments="
        f"{int(predictions['injury_margin_adjustment'].abs().gt(0).sum())}"
    )

    print(
        "travel_adjustments="
        f"{int(predictions['travel_margin_adjustment'].abs().gt(0).sum())}"
    )

    print(
        "weather_adjustments="
        f"{int(predictions['weather_total_adjustment'].abs().gt(0).sum())}"
    )

    if args.dry_run:
        print(
            "output_modified=no"
        )

        print(
            "status=dry_run_success"
        )

        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(
        f"output={output_path}"
    )

    print(
        "status=success"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )

        raise
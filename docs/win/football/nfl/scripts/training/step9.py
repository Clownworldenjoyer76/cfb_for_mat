#!/usr/bin/env python3
"""
Step 9: add rest difference and historical travel values.

READS ONLY:
  docs/win/football/nfl/data/historic_data/games/games_2010_2025.csv
  docs/win/football/nfl/data/master/team_master.csv

UPDATES IN PLACE:
  docs/win/football/nfl/training/historical_core_2021.csv
  docs/win/football/nfl/training/historical_core_2022.csv
  docs/win/football/nfl/training/historical_core_2023.csv
  docs/win/football/nfl/training/historical_core_2024.csv
  docs/win/football/nfl/training/historical_core_2025.csv

ADDS/REBUILDS:
  rest_diff
  miles_traveled
  time_zones_crossed
  east_to_west
  west_to_east
  international_flag
  neutral_site_flag

Travel calculations mirror the existing build_travel.py conventions:
  - miles_traveled: haversine distance from away-team coordinates to
    home-team coordinates, rounded to one decimal
  - time_zones_crossed: absolute UTC-offset difference on the game date
  - direction flags: based on longitude movement from away to home

Historical-team abbreviation normalization:
  WAS -> WSH
  LA  -> LAR

The historical games source contains domestic home-stadium values for the
seven 2025 NFL international games. Those games are explicitly identified
below so international_flag remains correct.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import math
import re
import sys
import unicodedata

import pandas as pd


NFL_ROOT = Path(
    "docs/win/football/nfl"
)

TRAINING_DIR = (
    NFL_ROOT
    / "training"
)

GAMES_PATH = (
    NFL_ROOT
    / "data/historic_data/games/games_2010_2025.csv"
)

TEAM_MASTER_PATH = (
    NFL_ROOT
    / "data/master/team_master.csv"
)

SEASONS = [
    2021,
    2022,
    2023,
    2024,
    2025,
]

TRAINING_PATHS = {
    season: (
        TRAINING_DIR
        / f"historical_core_{season}.csv"
    )
    for season in SEASONS
}

GENERATED_COLUMNS = [
    "rest_diff",
    "miles_traveled",
    "time_zones_crossed",
    "east_to_west",
    "west_to_east",
    "international_flag",
    "neutral_site_flag",
]

TRAINING_REQUIRED = [
    "season",
    "week",
    "home_team",
    "away_team",
    "home_rest",
    "away_rest",
    "stadium",
]

GAMES_REQUIRED = [
    "season",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "location",
    "stadium",
]

TEAM_MASTER_REQUIRED = [
    "canonical_team",
    "team_abbr",
    "latitude",
    "longitude",
    "timezone",
]

EARTH_RADIUS_MILES = 3958.8

BLANK_VALUES = {
    "",
    "nan",
    "none",
    "<na>",
    "null",
}

TEAM_ABBR_ALIASES = {
    "WAS": "WSH",
    "LA": "LAR",
}

INTERNATIONAL_STADIUMS = {
    "allianz arena",
    "arena corinthians",
    "azteca stadium",
    "corinthians arena",
    "croke park",
    "deutsche bank park",
    "estadio azteca",
    "neo quimica arena",
    "olympic stadium",
    "santiago bernabeu stadium",
    "tottenham hotspur stadium",
    "tottenham stadium",
    "wembley stadium",
}

# Key format:
#   (season, week, home_team, away_team)
#
# The historical games source currently carries the domestic home stadium
# for these 2025 international games, so stadium-name detection alone would
# incorrectly mark them as domestic.
INTERNATIONAL_GAME_KEYS = {
    (
        2025,
        1,
        "LAC",
        "KC",
    ),
    (
        2025,
        4,
        "PIT",
        "MIN",
    ),
    (
        2025,
        5,
        "CLE",
        "MIN",
    ),
    (
        2025,
        6,
        "NYJ",
        "DEN",
    ),
    (
        2025,
        7,
        "JAX",
        "LAR",
    ),
    (
        2025,
        10,
        "IND",
        "ATL",
    ),
    (
        2025,
        11,
        "MIA",
        "WSH",
    ),
}


def read_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}"
        )

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label}: missing required columns: "
            f"{missing}"
        )


def clean(
    value: object,
) -> str:
    if value is None:
        return ""

    text = str(
        value
    ).strip()

    if text.lower() in BLANK_VALUES:
        return ""

    return text


def team_key(
    value: object,
) -> str:
    key = clean(
        value
    ).upper()

    return TEAM_ABBR_ALIASES.get(
        key,
        key,
    )


def stadium_key(
    value: object,
) -> str:
    text = unicodedata.normalize(
        "NFKD",
        clean(
            value
        ),
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(
            character
        )
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text.casefold(),
    )

    return " ".join(
        text.split()
    )


def parse_int(
    value: object,
    label: str,
) -> int:
    text = clean(
        value
    )

    if text == "":
        raise ValueError(
            f"{label}: blank integer value"
        )

    try:
        number = float(
            text
        )

    except ValueError as exc:
        raise ValueError(
            f"{label}: invalid integer value "
            f"{text!r}"
        ) from exc

    if (
        not math.isfinite(
            number
        )
        or abs(
            number
            - round(
                number
            )
        )
        > 1e-9
    ):
        raise ValueError(
            f"{label}: invalid integer value "
            f"{text!r}"
        )

    return int(
        round(
            number
        )
    )


def parse_float(
    value: object,
    label: str,
) -> float:
    text = clean(
        value
    )

    if text == "":
        raise ValueError(
            f"{label}: blank numeric value"
        )

    try:
        number = float(
            text
        )

    except ValueError as exc:
        raise ValueError(
            f"{label}: invalid numeric value "
            f"{text!r}"
        ) from exc

    if not math.isfinite(
        number
    ):
        raise ValueError(
            f"{label}: non-finite numeric value "
            f"{text!r}"
        )

    return number


def format_number(
    value: float,
) -> str:
    if abs(
        value
    ) < 1e-12:
        value = 0.0

    if float(
        value
    ).is_integer():
        return str(
            int(
                value
            )
        )

    return str(
        value
    )


def haversine_miles(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

    dphi = math.radians(
        lat2
        - lat1
    )

    dlambda = math.radians(
        lon2
        - lon1
    )

    a = (
        math.sin(
            dphi / 2
        ) ** 2
        + math.cos(
            phi1
        )
        * math.cos(
            phi2
        )
        * math.sin(
            dlambda / 2
        ) ** 2
    )

    return (
        EARTH_RADIUS_MILES
        * (
            2
            * math.asin(
                math.sqrt(
                    a
                )
            )
        )
    )


def utc_offset_hours(
    timezone_name: str,
    game_date: str,
) -> float:
    try:
        dt = datetime.strptime(
            game_date,
            "%Y-%m-%d",
        ).replace(
            tzinfo=ZoneInfo(
                timezone_name
            )
        )

    except Exception as exc:
        raise ValueError(
            "Could not calculate UTC offset for "
            f"timezone={timezone_name!r}, "
            f"gameday={game_date!r}: "
            f"{exc}"
        ) from exc

    offset = dt.utcoffset()

    if offset is None:
        raise ValueError(
            "No UTC offset for "
            f"timezone={timezone_name!r}, "
            f"gameday={game_date!r}"
        )

    return (
        offset.total_seconds()
        / 3600
    )


def load_team_lookup() -> dict[
    str,
    dict[str, object],
]:
    df = read_csv(
        TEAM_MASTER_PATH
    )

    require_columns(
        df,
        TEAM_MASTER_REQUIRED,
        "team master",
    )

    lookup: dict[
        str,
        dict[str, object],
    ] = {}

    for index, row in df.iterrows():
        abbr = team_key(
            row[
                "team_abbr"
            ]
        )

        if not abbr:
            continue

        canonical_team = clean(
            row[
                "canonical_team"
            ]
        )

        timezone_name = clean(
            row[
                "timezone"
            ]
        )

        if (
            not canonical_team
            or not timezone_name
        ):
            raise ValueError(
                f"{TEAM_MASTER_PATH} "
                f"row {index + 2}: "
                f"incomplete team data "
                f"for {abbr}"
            )

        record = {
            "canonical_team": (
                canonical_team
            ),
            "latitude": (
                parse_float(
                    row[
                        "latitude"
                    ],
                    (
                        f"{TEAM_MASTER_PATH} "
                        f"row {index + 2}: "
                        "latitude"
                    ),
                )
            ),
            "longitude": (
                parse_float(
                    row[
                        "longitude"
                    ],
                    (
                        f"{TEAM_MASTER_PATH} "
                        f"row {index + 2}: "
                        "longitude"
                    ),
                )
            ),
            "timezone": (
                timezone_name
            ),
        }

        existing = lookup.get(
            abbr
        )

        if existing is None:
            lookup[
                abbr
            ] = record

        elif existing != record:
            raise ValueError(
                f"{TEAM_MASTER_PATH}: "
                "conflicting "
                "coordinate/timezone rows "
                f"for team_abbr={abbr}"
            )

    if not lookup:
        raise RuntimeError(
            f"{TEAM_MASTER_PATH}: "
            "no usable NFL team rows found"
        )

    return lookup


def load_games_lookup() -> dict[
    tuple[
        int,
        int,
        str,
        str,
    ],
    dict[str, str],
]:
    df = read_csv(
        GAMES_PATH
    )

    require_columns(
        df,
        GAMES_REQUIRED,
        "historical games",
    )

    lookup: dict[
        tuple[
            int,
            int,
            str,
            str,
        ],
        dict[str, str],
    ] = {}

    for index, row in df.iterrows():
        season_text = clean(
            row[
                "season"
            ]
        )

        week_text = clean(
            row[
                "week"
            ]
        )

        if (
            not season_text
            or not week_text
        ):
            continue

        season = parse_int(
            season_text,
            (
                f"{GAMES_PATH} "
                f"row {index + 2}: "
                "season"
            ),
        )

        if season not in SEASONS:
            continue

        week = parse_int(
            week_text,
            (
                f"{GAMES_PATH} "
                f"row {index + 2}: "
                "week"
            ),
        )

        home_team = team_key(
            row[
                "home_team"
            ]
        )

        away_team = team_key(
            row[
                "away_team"
            ]
        )

        gameday = clean(
            row[
                "gameday"
            ]
        )

        location = clean(
            row[
                "location"
            ]
        )

        stadium = clean(
            row[
                "stadium"
            ]
        )

        if (
            not home_team
            or not away_team
            or not gameday
            or not location
            or not stadium
        ):
            raise ValueError(
                f"{GAMES_PATH} "
                f"row {index + 2}: "
                "incomplete required game data"
            )

        key = (
            season,
            week,
            home_team,
            away_team,
        )

        if key in lookup:
            raise ValueError(
                f"{GAMES_PATH}: "
                f"duplicate game key {key}"
            )

        lookup[
            key
        ] = {
            "gameday": (
                gameday
            ),
            "location": (
                location
            ),
            "stadium": (
                stadium
            ),
        }

    return lookup


def location_to_neutral_flag(
    location: str,
    label: str,
) -> int:
    value = clean(
        location
    ).casefold()

    if value == "neutral":
        return 1

    if value == "home":
        return 0

    raise ValueError(
        f"{label}: "
        "unsupported location value "
        f"{location!r}"
    )


def is_international_game(
    game_key: tuple[
        int,
        int,
        str,
        str,
    ],
    stadium: str,
) -> int:
    if game_key in INTERNATIONAL_GAME_KEYS:
        return 1

    if (
        stadium_key(
            stadium
        )
        in INTERNATIONAL_STADIUMS
    ):
        return 1

    return 0


def process_season(
    season: int,
    games_lookup: dict[
        tuple[
            int,
            int,
            str,
            str,
        ],
        dict[str, str],
    ],
    team_lookup: dict[
        str,
        dict[str, object],
    ],
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    path = (
        TRAINING_PATHS[
            season
        ]
    )

    df = read_csv(
        path
    )

    require_columns(
        df,
        TRAINING_REQUIRED,
        (
            "historical training table "
            f"{season}"
        ),
    )

    if (
        len(
            df.columns
        )
        != len(
            set(
                df.columns
            )
        )
    ):
        raise ValueError(
            f"{path}: "
            "duplicate column names"
        )

    original_rows = len(
        df
    )

    for column in GENERATED_COLUMNS:
        if column in df.columns:
            df = df.drop(
                columns=[
                    column
                ]
            )

    output = {
        column: []
        for column in GENERATED_COLUMNS
    }

    historical_matches = 0
    international_games = 0
    neutral_games = 0

    for index, row in df.iterrows():
        row_season = parse_int(
            row[
                "season"
            ],
            (
                f"{path} "
                f"row {index + 2}: "
                "season"
            ),
        )

        week = parse_int(
            row[
                "week"
            ],
            (
                f"{path} "
                f"row {index + 2}: "
                "week"
            ),
        )

        if row_season != season:
            raise ValueError(
                f"{path} "
                f"row {index + 2}: "
                f"expected season {season}, "
                f"found {row_season}"
            )

        home_team = team_key(
            row[
                "home_team"
            ]
        )

        away_team = team_key(
            row[
                "away_team"
            ]
        )

        game_key = (
            season,
            week,
            home_team,
            away_team,
        )

        game = games_lookup.get(
            game_key
        )

        if game is None:
            raise RuntimeError(
                f"{path} "
                f"row {index + 2}: "
                "historical game not found "
                f"for {game_key}"
            )

        historical_matches += 1

        training_stadium = clean(
            row[
                "stadium"
            ]
        )

        source_stadium = game[
            "stadium"
        ]

        if (
            training_stadium
            and stadium_key(
                training_stadium
            )
            != stadium_key(
                source_stadium
            )
        ):
            raise RuntimeError(
                f"{path} "
                f"row {index + 2}: "
                "stadium mismatch; "
                f"training="
                f"{training_stadium!r}, "
                f"historical="
                f"{source_stadium!r}"
            )

        home_info = (
            team_lookup.get(
                home_team
            )
        )

        away_info = (
            team_lookup.get(
                away_team
            )
        )

        if home_info is None:
            raise RuntimeError(
                f"{path} "
                f"row {index + 2}: "
                "team_master match "
                "not found for "
                f"{home_team}"
            )

        if away_info is None:
            raise RuntimeError(
                f"{path} "
                f"row {index + 2}: "
                "team_master match "
                "not found for "
                f"{away_team}"
            )

        home_rest = parse_float(
            row[
                "home_rest"
            ],
            (
                f"{path} "
                f"row {index + 2}: "
                "home_rest"
            ),
        )

        away_rest = parse_float(
            row[
                "away_rest"
            ],
            (
                f"{path} "
                f"row {index + 2}: "
                "away_rest"
            ),
        )

        rest_diff = (
            home_rest
            - away_rest
        )

        home_lat = float(
            home_info[
                "latitude"
            ]
        )

        home_lon = float(
            home_info[
                "longitude"
            ]
        )

        away_lat = float(
            away_info[
                "latitude"
            ]
        )

        away_lon = float(
            away_info[
                "longitude"
            ]
        )

        miles = round(
            haversine_miles(
                away_lat,
                away_lon,
                home_lat,
                home_lon,
            ),
            1,
        )

        away_offset = (
            utc_offset_hours(
                str(
                    away_info[
                        "timezone"
                    ]
                ),
                game[
                    "gameday"
                ],
            )
        )

        home_offset = (
            utc_offset_hours(
                str(
                    home_info[
                        "timezone"
                    ]
                ),
                game[
                    "gameday"
                ],
            )
        )

        time_zones_crossed = abs(
            home_offset
            - away_offset
        )

        if home_lon > away_lon:
            east_to_west = 0
            west_to_east = 1

        elif home_lon < away_lon:
            east_to_west = 1
            west_to_east = 0

        else:
            east_to_west = 0
            west_to_east = 0

        neutral = (
            location_to_neutral_flag(
                game[
                    "location"
                ],
                (
                    f"{GAMES_PATH}: "
                    f"{game_key}"
                ),
            )
        )

        international = (
            is_international_game(
                game_key,
                source_stadium,
            )
        )

        neutral_games += (
            neutral
        )

        international_games += (
            international
        )

        output[
            "rest_diff"
        ].append(
            format_number(
                rest_diff
            )
        )

        output[
            "miles_traveled"
        ].append(
            f"{miles:.1f}"
        )

        output[
            "time_zones_crossed"
        ].append(
            str(
                float(
                    time_zones_crossed
                )
            )
        )

        output[
            "east_to_west"
        ].append(
            str(
                east_to_west
            )
        )

        output[
            "west_to_east"
        ].append(
            str(
                west_to_east
            )
        )

        output[
            "international_flag"
        ].append(
            str(
                international
            )
        )

        output[
            "neutral_site_flag"
        ].append(
            str(
                neutral
            )
        )

    for column in GENERATED_COLUMNS:
        df[
            column
        ] = output[
            column
        ]

    if len(
        df
    ) != original_rows:
        raise RuntimeError(
            f"{season}: "
            "row count changed "
            "during Step 9"
        )

    if (
        df[
            GENERATED_COLUMNS
        ]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            f"{season}: "
            "generated Step 9 columns "
            "contain null values"
        )

    return (
        df,
        {
            "rows": (
                original_rows
            ),
            "historical_matches": (
                historical_matches
            ),
            "international_games": (
                international_games
            ),
            "neutral_games": (
                neutral_games
            ),
        },
    )


def write_outputs(
    outputs: dict[
        int,
        pd.DataFrame,
    ],
) -> None:
    temp_paths: dict[
        int,
        Path,
    ] = {}

    try:
        for season in SEASONS:
            output_path = (
                TRAINING_PATHS[
                    season
                ]
            )

            temp_path = (
                output_path.with_suffix(
                    ".step9.tmp.csv"
                )
            )

            temp_paths[
                season
            ] = temp_path

            outputs[
                season
            ].to_csv(
                temp_path,
                index=False,
                encoding="utf-8",
                lineterminator="\n",
            )

        for season in SEASONS:
            temp_paths[
                season
            ].replace(
                TRAINING_PATHS[
                    season
                ]
            )

    except Exception:
        for temp_path in (
            temp_paths.values()
        ):
            if temp_path.exists():
                temp_path.unlink()

        raise


def main() -> int:
    team_lookup = (
        load_team_lookup()
    )

    games_lookup = (
        load_games_lookup()
    )

    outputs: dict[
        int,
        pd.DataFrame,
    ] = {}

    results: dict[
        int,
        dict[str, int],
    ] = {}

    for season in SEASONS:
        (
            outputs[
                season
            ],
            results[
                season
            ],
        ) = process_season(
            season,
            games_lookup,
            team_lookup,
        )

    write_outputs(
        outputs
    )

    print(
        "Step 9 complete."
    )

    for season in SEASONS:
        stats = results[
            season
        ]

        print(
            f"{season}: "
            f"rows={stats['rows']}, "
            f"historical_matches="
            f"{stats['historical_matches']}, "
            f"international_games="
            f"{stats['international_games']}, "
            f"neutral_games="
            f"{stats['neutral_games']}"
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

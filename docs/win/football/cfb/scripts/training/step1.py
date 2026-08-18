#!/usr/bin/env python3
"""
Build the historical core NFL modeling table for seasons 2021-2025.

READS ONLY:
  docs/win/football/nfl/data/historic_data/games/games_2010_2025.csv
  docs/win/football/nfl/data/historic_data/predictions/drat/nfl_{season}.csv
  docs/win/football/nfl/data/historic_data/predictions/epred/{season}_predictions.csv
  docs/win/football/nfl/data/historic_data/schedule/master_schedule_{season}.csv

WRITES ONLY:
  docs/win/football/nfl/training/historical_core_2021_2025.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

NFL_ROOT = Path("docs/win/football/nfl")
GAMES_PATH = NFL_ROOT / "data/historic_data/games/games_2010_2025.csv"
DRAT_DIR = NFL_ROOT / "data/historic_data/predictions/drat"
EPRED_DIR = NFL_ROOT / "data/historic_data/predictions/epred"
SCHEDULE_DIR = NFL_ROOT / "data/historic_data/schedule"
OUTPUT_PATH = NFL_ROOT / "training/historical_core_2021_2025.csv"
SEASONS = range(2021, 2026)

GAMES_COLUMNS = [
    "espn", "season", "game_type", "week", "gameday", "weekday", "gametime",
    "away_team", "home_team", "away_score", "home_score", "location", "away_rest",
    "home_rest", "away_moneyline", "home_moneyline", "spread_line", "away_spread_odds",
    "home_spread_odds", "total_line", "under_odds", "over_odds", "div_game", "roof",
    "surface", "temp", "wind", "away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name",
    "away_coach", "home_coach", "stadium_id", "stadium",
]

DRAT_COLUMNS = [
    "game_id", "away_prob", "home_prob", "away_moneyline", "home_moneyline", "away_spread", "home_spread"
]
DRAT_RENAME = {
    "away_prob": "drat_away_prob",
    "home_prob": "drat_home_prob",
    "away_moneyline": "drat_away_moneyline",
    "home_moneyline": "drat_home_moneyline",
    "away_spread": "drat_away_spread",
    "home_spread": "drat_home_spread",
}

EPRED_COLUMNS = [
    "game_id", "matchupQuality", "home_prob", "away_prob", "tie_prob", "away_projected_pts",
    "home_projected_pts", "total_projected_pts", "home_PtDiff", "away_PtDiff", "home_rating", "away_rating"
]
EPRED_RENAME = {
    "matchupQuality": "epred_matchupQuality",
    "home_prob": "epred_home_prob",
    "away_prob": "epred_away_prob",
    "tie_prob": "epred_tie_prob",
    "away_projected_pts": "epred_away_projected_pts",
    "home_projected_pts": "epred_home_projected_pts",
    "total_projected_pts": "epred_total_projected_pts",
    "home_PtDiff": "epred_home_PtDiff",
    "away_PtDiff": "epred_away_PtDiff",
    "home_rating": "epred_home_rating",
    "away_rating": "epred_away_rating",
}

SCHEDULE_COLUMNS = ["game_id", "home_team", "away_team", "season", "season_type", "week"]

FINAL_COLUMNS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime", "away_team", "home_team",
    "away_score", "home_score", "location", "away_rest", "home_rest", "away_moneyline", "home_moneyline",
    "spread_line", "away_spread_odds", "home_spread_odds", "total_line", "under_odds", "over_odds", "div_game",
    "roof", "surface", "temp", "wind", "away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name", "away_coach",
    "home_coach", "stadium_id", "stadium", "drat_away_prob", "drat_home_prob", "drat_away_moneyline",
    "drat_home_moneyline", "drat_away_spread", "drat_home_spread", "epred_matchupQuality", "epred_home_prob",
    "epred_away_prob", "epred_tie_prob", "epred_away_projected_pts", "epred_home_projected_pts",
    "epred_total_projected_pts", "epred_home_PtDiff", "epred_away_PtDiff", "epred_home_rating", "epred_away_rating",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False)


def require_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")


def normalize_game_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def require_unique_game_id(df: pd.DataFrame, label: str) -> None:
    duplicated = df.loc[df["game_id"].duplicated(keep=False), "game_id"].dropna().unique()
    if len(duplicated):
        sample = ", ".join(map(str, duplicated[:10]))
        raise ValueError(f"{label}: duplicate game_id values found: {sample}")


def main() -> int:
    games = read_csv(GAMES_PATH)
    require_columns(games, GAMES_COLUMNS, GAMES_PATH)
    games = games[GAMES_COLUMNS].copy()
    games["season"] = games["season"].astype("string").str.strip()
    games = games[games["season"].isin([str(season) for season in SEASONS])].copy()
    games = games.rename(columns={"espn": "game_id"})
    games["game_id"] = normalize_game_id(games["game_id"])
    games = games[games["game_id"].notna() & games["game_id"].ne("")].copy()
    require_unique_game_id(games, str(GAMES_PATH))

    season_outputs: list[pd.DataFrame] = []

    for season in SEASONS:
        drat_path = DRAT_DIR / f"nfl_{season}.csv"
        epred_path = EPRED_DIR / f"{season}_predictions.csv"
        schedule_path = SCHEDULE_DIR / f"master_schedule_{season}.csv"

        drat = read_csv(drat_path)
        epred = read_csv(epred_path)
        schedule = read_csv(schedule_path)

        require_columns(drat, DRAT_COLUMNS, drat_path)
        require_columns(epred, EPRED_COLUMNS, epred_path)
        require_columns(schedule, SCHEDULE_COLUMNS, schedule_path)

        drat = drat[DRAT_COLUMNS].copy().rename(columns=DRAT_RENAME)
        epred = epred[EPRED_COLUMNS].copy().rename(columns=EPRED_RENAME)
        schedule = schedule[SCHEDULE_COLUMNS].copy()

        for frame in (drat, epred, schedule):
            frame["game_id"] = normalize_game_id(frame["game_id"])

        require_unique_game_id(drat, str(drat_path))
        require_unique_game_id(epred, str(epred_path))
        require_unique_game_id(schedule, str(schedule_path))

        season_games = games[games["season"] == str(season)].copy()
        schedule_ids = schedule[["game_id"]].copy()

        merged = season_games.merge(schedule_ids, on="game_id", how="inner", validate="one_to_one")
        merged = merged.merge(drat, on="game_id", how="inner", validate="one_to_one")
        merged = merged.merge(epred, on="game_id", how="inner", validate="one_to_one")
        merged = merged[FINAL_COLUMNS].copy()
        season_outputs.append(merged)

        print(
            f"{season}: games={len(season_games)} schedule={len(schedule)} "
            f"drat={len(drat)} epred={len(epred)} joined={len(merged)}"
        )

    output = pd.concat(season_outputs, ignore_index=True)
    output["season"] = pd.to_numeric(output["season"], errors="coerce")
    output["week"] = pd.to_numeric(output["week"], errors="coerce")
    output = output.sort_values(["season", "week", "gameday", "gametime", "game_id"], kind="stable").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(output)} rows to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

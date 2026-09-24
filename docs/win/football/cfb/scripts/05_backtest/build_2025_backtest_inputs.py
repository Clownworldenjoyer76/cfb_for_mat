#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CFB_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = CFB_ROOT / "scripts"
REPORT_ROOT = CFB_ROOT / "errors"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPTS_DIR),
    )

from pipeline_reporter import PipelineReporter
from pipeline_shared import (
    clean_text as clean,
    finite_float as as_float,
    load_module,
)


SCRIPT_VERSION = "cfb-backtest-inputs-v1-reporter-2026-09-16"

PROJECTION_PATH = CFB_ROOT / "scripts" / "01_merge" / "projection_week1.py"
SELECTIONS_PATH = CFB_ROOT / "scripts" / "02_select" / "selections.py"
PULL_ODDS_PATH = CFB_ROOT / "scripts" / "00_intake" / "pull_odds.py"
PULL_E_PRED_PATH = CFB_ROOT / "scripts" / "00_intake" / "pull_e_predictions.py"

TEAM_MAP_PATH = CFB_ROOT / "config" / "mapping" / "team_map.csv"
STADIUM_MAP_PATH = CFB_ROOT / "config" / "mapping" / "stadium_map.csv"
COEFFICIENTS_PATH = CFB_ROOT / "config" / "travel_weather_coefficients.csv"
SETTINGS_PATH = CFB_ROOT / "config" / "settings.yaml"

ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/"
    "leagues/college-football/events/{game_id}/competitions/{game_id}/odds"
)


def normalize_prob(value: Any) -> float | None:
    number = as_float(value)
    if number is None:
        return None
    if 1.0 < number <= 100.0:
        number /= 100.0
    return number if 0.0 <= number <= 1.0 else None


def choose_odds_item(pull_odds: ModuleType, items: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, int]:
    for item in items:
        info = pull_odds.provider_info(item)
        name = clean(info.get("name"))
        if "draftkings" in name.casefold().replace(" ", ""):
            return item, name or "DraftKings", 0

    fallback = pull_odds.select_primary_odds_item(items)
    if fallback is None:
        return None, "", 0

    info = pull_odds.provider_info(fallback)
    return fallback, clean(info.get("name")) or "ESPN_PRIMARY", 1


def fetch_game_odds(pull_odds: ModuleType, game_id: str) -> dict[str, Any]:
    url = ODDS_URL.format(game_id=game_id)
    status, payload, error = pull_odds.http_get_json(url)

    if payload is None:
        return {
            "bookmaker": "draftkings",
            "odds_available": 0,
            "odds_missing_reason": f"ESPN_ODDS_UNAVAILABLE_{status or 'ERR'}:{error}",
            "historical_source_bookmaker": "",
            "historical_provider_proxy": 0,
        }

    items = pull_odds.resolve_odds_items(payload)
    item, source_book, proxy = choose_odds_item(pull_odds, items)

    if item is None:
        return {
            "bookmaker": "draftkings",
            "odds_available": 0,
            "odds_missing_reason": "HISTORICAL_ODDS_UNAVAILABLE",
            "historical_source_bookmaker": "",
            "historical_provider_proxy": 0,
        }

    values = pull_odds.extract_market_values(item)

    return {
        "bookmaker": "draftkings",
        "home_moneyline_american": values.get("home_moneyline_american", ""),
        "away_moneyline_american": values.get("away_moneyline_american", ""),
        "home_spread": values.get("home_spread", ""),
        "away_spread": values.get("away_spread", ""),
        "home_spread_american": values.get("home_spread_american", ""),
        "away_spread_american": values.get("away_spread_american", ""),
        "total": values.get("total", ""),
        "over_american": values.get("over_american", ""),
        "under_american": values.get("under_american", ""),
        "odds_last_update": clean(values.get("last_update", "")),
        "odds_available": 1,
        "odds_missing_reason": "",
        "historical_source_bookmaker": source_book,
        "historical_provider_proxy": proxy,
    }


def fetch_predictor(
    pull_e_pred: ModuleType,
    game_id: str,
    season: int,
    week: int,
    home_team: str,
    away_team: str,
) -> dict[str, Any]:
    rows = pull_e_pred.get_predictor_rows(game_id, season, 2, week)

    home = next(
        (r for r in rows if clean(r.get("home_away")).casefold() == "hometeam"),
        None,
    )
    away = next(
        (r for r in rows if clean(r.get("home_away")).casefold() == "awayteam"),
        None,
    )

    if home is None and away is None:
        return {}

    return {
        "game_id": game_id,
        "home_team": home_team,
        "away_team": away_team,
        "home_PtDiff": as_float(home.get("teamPredPtDiff")) if home else None,
        "away_PtDiff": as_float(away.get("teamPredPtDiff")) if away else None,
        "home_prob": normalize_prob(home.get("gameProjection")) if home else None,
        "away_prob": normalize_prob(away.get("gameProjection")) if away else None,
        "tie_prob": None,
        "matchupQuality": (
            as_float(home.get("matchupQuality"))
            if home
            else as_float(away.get("matchupQuality"))
        ),
    }


def build_schedule_frame(
    source_week: pd.DataFrame,
    odds_by_game: dict[str, dict[str, Any]],
    base: ModuleType,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, source in source_week.iterrows():
        game_id = clean(source.get("game_id"))
        row = {column: clean(source.get(column)) for column in base.OUTPUT_BASE_COLUMNS}

        row.update(
            {
                "season": clean(source.get("season")),
                "season_type": clean(source.get("season_type")),
                "week": clean(source.get("week")),
                "game_id": game_id,
                "away_team": clean(source.get("away_team")),
                "home_team": clean(source.get("home_team")),
                "neutral_site": clean(source.get("neutral_site")),
                "stadium": clean(source.get("stadium")),
                "roof": clean(source.get("roof")),
                "surface": clean(source.get("surface")),
                "home_timezone": clean(source.get("home_timezone")),
                "away_timezone": clean(source.get("away_timezone")),
                "game_timezone": clean(source.get("game_timezone")),
                "game_locked": "0",
            }
        )

        row.update(odds_by_game.get(game_id, {}))
        rows.append(row)

    return pd.DataFrame(rows)


def build_prior(
    base: ModuleType,
    current_stats: pd.DataFrame,
    prior_stats: pd.DataFrame,
    resolver: Any,
    season: int,
    week: int,
) -> pd.DataFrame:
    if week == 1:
        source = prior_stats.copy()
        base.MIN_PRIOR_TEAM_WEEKS = 10
    else:
        season_num = pd.to_numeric(current_stats["season"], errors="coerce")
        week_num = pd.to_numeric(current_stats["week"], errors="coerce")
        source = current_stats[
            season_num.eq(season)
            & week_num.notna()
            & week_num.lt(week)
        ].copy()
        base.MIN_PRIOR_TEAM_WEEKS = 1

    if source.empty:
        raise RuntimeError(f"No prior team stats for season={season} week={week}")

    prior = base.build_prior_table(source, resolver)
    return base.scale_prior_to_fpi(prior, pd.DataFrame(columns=["fpi"]))


def projection_args(base: ModuleType) -> Namespace:
    return Namespace(
        home_field=2.5,
        drives_per_team=11.5,
        market_margin_weight=0.36,
        fpi_margin_weight=0.28,
        espn_margin_weight=0.20,
        prior_margin_weight=0.16,
        market_total_weight=0.75,
        fresh_injury_days=60,
        margin_sd=base.DEFAULT_MARGIN_SD,
        total_sd=base.DEFAULT_TOTAL_SD,
    )


def _main_impl() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--prior-season", type=int, default=2024)
    args = parser.parse_args()

    season = int(args.season)
    prior_season = int(args.prior_season)

    base = load_module("cfb_bt_projection_base", PROJECTION_PATH)
    selections = load_module("cfb_bt_selections", SELECTIONS_PATH)
    pull_odds = load_module("cfb_bt_pull_odds", PULL_ODDS_PATH)
    pull_e_pred = load_module("cfb_bt_pull_e_pred", PULL_E_PRED_PATH)

    schedule_path = CFB_ROOT / "00_intake" / "schedule" / f"{season}_schedule.csv"
    current_stats_path = CFB_ROOT / "00_intake" / "team_stats" / f"{season}_team_stats.csv"
    prior_stats_path = CFB_ROOT / "00_intake" / "team_stats" / f"{prior_season}_team_stats.csv"

    for path in [
        schedule_path,
        current_stats_path,
        prior_stats_path,
        TEAM_MAP_PATH,
        STADIUM_MAP_PATH,
        COEFFICIENTS_PATH,
        SETTINGS_PATH,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    schedule = pd.read_csv(schedule_path, dtype=str, encoding="utf-8-sig", low_memory=False)
    current_stats = pd.read_csv(current_stats_path, dtype=str, encoding="utf-8-sig", low_memory=False)
    prior_stats = pd.read_csv(prior_stats_path, dtype=str, encoding="utf-8-sig", low_memory=False)
    team_map = pd.read_csv(TEAM_MAP_PATH, dtype=str, encoding="utf-8-sig", low_memory=False)

    resolver = base.TeamResolver(team_map)
    home_stadium_lookup = base.build_home_stadium_lookup(STADIUM_MAP_PATH, resolver)
    coefficients = base.load_travel_weather_coefficients(COEFFICIENTS_PATH)

    season_num = pd.to_numeric(schedule["season"], errors="coerce")
    type_num = pd.to_numeric(schedule["season_type"], errors="coerce")
    week_num = pd.to_numeric(schedule["week"], errors="coerce")

    regular = schedule[
        season_num.eq(season)
        & type_num.eq(2)
        & week_num.notna()
    ].copy()

    weeks = sorted(
        regular["week"].astype(float).astype(int).unique().tolist()
    )

    output_root = CFB_ROOT / "05_backtest" / "input" / str(season)
    candidate_dir = output_root / "candidates"
    weekly_dir = output_root / "weekly"
    audit_dir = output_root / "audit"

    candidate_dir.mkdir(parents=True, exist_ok=True)
    weekly_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    settings = selections.read_yaml(SETTINGS_PATH, "settings config")
    max_kelly = float(settings["selection_defaults"]["max_kelly"])

    total_games = 0
    total_with_odds = 0

    for week in weeks:
        source_week = regular[
            pd.to_numeric(regular["week"], errors="coerce").eq(week)
        ].copy().reset_index(drop=True)

        odds_by_game: dict[str, dict[str, Any]] = {}
        predictor_rows: list[dict[str, Any]] = []

        for index, source in source_week.iterrows():
            game_id = clean(source.get("game_id"))
            home_team = clean(source.get("home_team"))
            away_team = clean(source.get("away_team"))

            odds = fetch_game_odds(pull_odds, game_id)
            odds_by_game[game_id] = odds

            if int(odds.get("odds_available", 0) or 0) == 1:
                total_with_odds += 1

            predictor = fetch_predictor(
                pull_e_pred,
                game_id,
                season,
                week,
                home_team,
                away_team,
            )

            if predictor:
                predictor_rows.append(predictor)

            print(
                f"week={week} game={index + 1}/{len(source_week)} "
                f"game_id={game_id} odds={odds.get('odds_available', 0)} "
                f"predictor={1 if predictor else 0}"
            )

        weekly = build_schedule_frame(source_week, odds_by_game, base)

        required_schedule_columns = [
            "season",
            "season_type",
            "week",
            "game_id",
            "neutral_site",
            "roof",
            "bookmaker",
            "home_moneyline_american",
            "away_moneyline_american",
            "home_spread",
            "away_spread",
            "home_spread_american",
            "away_spread_american",
            "total",
            "over_american",
            "under_american",
            "odds_available",
            "game_locked",
        ]

        for column in required_schedule_columns:
            if column not in weekly.columns:
                weekly[column] = ""

        weekly_path = weekly_dir / f"week_{week}_CFB_weekly_schedule.csv"
        weekly.to_csv(weekly_path, index=False)

        prior = build_prior(
            base,
            current_stats,
            prior_stats,
            resolver,
            season,
            week,
        )

        espn_predictions = pd.DataFrame(predictor_rows)

        projected = base.build_projection(
            weekly,
            prior,
            pd.DataFrame(
                columns=["team", "team_id", "fpi", "epaoffense", "epadefense"]
            ),
            espn_predictions,
            resolver,
            home_stadium_lookup,
            {},
            pd.DataFrame(),
            pd.DataFrame(),
            coefficients,
            projection_args(base),
        )

        working = selections.merge_schedule(
            projected,
            weekly,
            season,
            week,
            "reg",
            "draftkings",
        )

        candidates = selections.build_output(
            projected,
            working,
            max_kelly,
        )

        # Add audit fields from the historical odds pull.
        audit_market = weekly[
            [
                "game_id",
                "historical_source_bookmaker",
                "historical_provider_proxy",
            ]
        ].copy()

        candidates = candidates.merge(
            audit_market,
            on="game_id",
            how="left",
            validate="one_to_one",
        )

        candidate_path = candidate_dir / f"week_{week}_CFB_selected.csv"
        candidates.to_csv(candidate_path, index=False)

        audit = pd.DataFrame(
            [
                {
                    "season": season,
                    "week": week,
                    "games": len(source_week),
                    "games_with_historical_odds": int(
                        pd.to_numeric(
                            weekly["odds_available"],
                            errors="coerce",
                        ).fillna(0).sum()
                    ),
                    "games_with_espn_predictor": len(espn_predictions),
                    "fpi_component": "DISABLED_NO_POINT_IN_TIME_ARCHIVE",
                    "injury_component": "DISABLED_NO_POINT_IN_TIME_ARCHIVE",
                    "travel_component": "DISABLED",
                    "weather_component": "DISABLED",
                    "odds_note": (
                        "DraftKings used when ESPN retains it; otherwise ESPN primary "
                        "historical provider is used as a proxy."
                    ),
                }
            ]
        )

        audit.to_csv(
            audit_dir / f"week_{week}_audit.csv",
            index=False,
        )

        total_games += len(source_week)
        print(f"wrote={candidate_path} rows={len(candidates)}")

    print("build_2025_backtest_inputs completed")
    print(f"season={season}")
    print(f"weeks={len(weeks)}")
    print(f"games={total_games}")
    print(f"games_with_historical_odds={total_with_odds}")
    print(f"candidate_dir={candidate_dir}")
    print("status=success")
    return 0


def _pipeline_report_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        add_help=False,
    )

    parser.add_argument(
        "--season",
        type=int,
        default=2025,
    )

    parser.add_argument(
        "--prior-season",
        type=int,
        default=2024,
    )

    args, _ = parser.parse_known_args()

    return args


def main() -> int:
    if any(
        arg in {"-h", "--help"}
        for arg in sys.argv[1:]
    ):
        return int(
            _main_impl()
        )

    with PipelineReporter(
        script=__file__,
        stage="05_backtest",
        report_root=REPORT_ROOT,
        pipeline="cfb",
        league="CFB",
        extra_context={
            "script_version": SCRIPT_VERSION,
            "backtest_scope": "historical_input_builder",
        },
    ) as report:
        report_args = _pipeline_report_args()

        season = int(
            report_args.season
        )

        prior_season = int(
            report_args.prior_season
        )

        report.season = season

        schedule_path = (
            CFB_ROOT
            / "00_intake"
            / "schedule"
            / f"{season}_schedule.csv"
        )

        current_stats_path = (
            CFB_ROOT
            / "00_intake"
            / "team_stats"
            / f"{season}_team_stats.csv"
        )

        prior_stats_path = (
            CFB_ROOT
            / "00_intake"
            / "team_stats"
            / f"{prior_season}_team_stats.csv"
        )

        output_root = (
            CFB_ROOT
            / "05_backtest"
            / "input"
            / str(
                season
            )
        )

        candidate_dir = (
            output_root
            / "candidates"
        )

        weekly_dir = (
            output_root
            / "weekly"
        )

        audit_dir = (
            output_root
            / "audit"
        )

        for input_path in (
            schedule_path,
            current_stats_path,
            prior_stats_path,
            TEAM_MAP_PATH,
            STADIUM_MAP_PATH,
            COEFFICIENTS_PATH,
            SETTINGS_PATH,
        ):
            report.add_input(
                input_path
            )

        report.update_details(
            {
                "prior_season": prior_season,
                "output_root": str(
                    output_root
                ),
            }
        )

        result = _main_impl()

        candidate_files = sorted(
            candidate_dir.glob(
                "week_*_CFB_selected.csv"
            )
        )

        weekly_files = sorted(
            weekly_dir.glob(
                "week_*_CFB_weekly_schedule.csv"
            )
        )

        audit_files = sorted(
            audit_dir.glob(
                "week_*_audit.csv"
            )
        )

        for output_path in (
            candidate_files
            + weekly_files
            + audit_files
        ):
            report.add_output(
                output_path
            )

        candidate_rows = 0

        for path in candidate_files:
            frame = pd.read_csv(
                path,
                dtype=str,
                encoding="utf-8-sig",
                low_memory=False,
            )

            candidate_rows += len(
                frame
            )

        games = 0
        games_with_historical_odds = 0

        for path in audit_files:
            audit = pd.read_csv(
                path,
                dtype=str,
                encoding="utf-8-sig",
                low_memory=False,
            )

            if "games" in audit.columns:
                games += int(
                    pd.to_numeric(
                        audit[
                            "games"
                        ],
                        errors="coerce",
                    ).fillna(
                        0
                    ).sum()
                )

            if (
                "games_with_historical_odds"
                in audit.columns
            ):
                games_with_historical_odds += int(
                    pd.to_numeric(
                        audit[
                            "games_with_historical_odds"
                        ],
                        errors="coerce",
                    ).fillna(
                        0
                    ).sum()
                )

        report.set_rows(
            rows_in=games,
            rows_out=candidate_rows,
        )

        report.update_details(
            {
                "weeks": len(
                    candidate_files
                ),
                "games": games,
                "games_with_historical_odds": (
                    games_with_historical_odds
                ),
                "candidate_rows": candidate_rows,
                "candidate_files": len(
                    candidate_files
                ),
                "weekly_files": len(
                    weekly_files
                ),
                "audit_files": len(
                    audit_files
                ),
            }
        )

        return int(
            result
        )

    raise RuntimeError("context manager unexpectedly suppressed an exception")

if __name__ == "__main__":
    raise SystemExit(main())

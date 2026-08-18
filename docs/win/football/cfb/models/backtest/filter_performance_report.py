#!/usr/bin/env python3
"""
Build a season-by-season filter-performance report from the completed NFL backtest.

This script DOES NOT retrain models and DOES NOT modify the source backtest files.

Default read-only inputs are expected beside this script:
  walkforward_probabilities.csv
  historical_moneyline_selected.csv
  historical_spread_selected.csv
  historical_total_selected.csv

The output path must be supplied explicitly with --output and must remain inside
the directory containing this script.

Threshold arguments are optional. If a threshold is omitted, that dimension is
left unfiltered. Supplying multiple values creates the Cartesian product of the
explicitly supplied thresholds.

Examples:
  python filter_performance_report.py \
    --output filter_report.csv

  python filter_performance_report.py \
    --output filter_report.csv \
    --markets moneyline \
    --min-ev 0 0.03 0.05 0.10 \
    --min-edge 0 0.02 0.05 \
    --min-model-prob 0 0.40 0.50 0.55 \
    --max-odds-american 500 300 200 150

  python filter_performance_report.py \
    --output totals_report.csv \
    --markets total \
    --min-ev 0 0.03 0.05 0.10 \
    --min-edge 0 0.02 0.05
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_WALKFORWARD = SCRIPT_DIR / "walkforward_probabilities.csv"
DEFAULT_MONEYLINE = SCRIPT_DIR / "historical_moneyline_selected.csv"
DEFAULT_SPREAD = SCRIPT_DIR / "historical_spread_selected.csv"
DEFAULT_TOTAL = SCRIPT_DIR / "historical_total_selected.csv"

MARKET_SPECS = {
    "moneyline": {
        "path_key": "moneyline",
        "selection": "ml_selection",
        "odds": "ml_odds_american",
        "model_prob": "ml_model_probability",
        "implied_prob": "ml_implied_probability",
        "edge": "ml_edge",
        "ev": "ml_ev",
        "kelly": "ml_kelly",
        "result": "ml_result",
        "profit": "ml_flat_profit_units",
    },
    "spread": {
        "path_key": "spread",
        "selection": "spread_selection",
        "odds": "spread_odds_american",
        "model_prob": "spread_model_probability",
        "implied_prob": "spread_implied_probability",
        "edge": "spread_edge",
        "ev": "spread_ev",
        "kelly": "spread_kelly",
        "result": "spread_result",
        "profit": "spread_flat_profit_units",
    },
    "total": {
        "path_key": "total",
        "selection": "total_selection",
        "odds": "total_odds_american",
        "model_prob": "total_model_probability",
        "implied_prob": "total_implied_probability",
        "edge": "total_edge",
        "ev": "total_ev",
        "kelly": "total_kelly",
        "result": "total_result",
        "profit": "total_flat_profit_units",
    },
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def normalize_game_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def as_numeric(series: pd.Series, label: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().any():
        bad = int(out.isna().sum())
        fail(f"{label}: {bad} non-numeric or missing value(s)")
    return out.astype(float)


def ensure_existing_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        fail(f"{label} not found: {path}")
    return path


def ensure_output_path(path: Path) -> Path:
    path = path.resolve()
    try:
        path.relative_to(SCRIPT_DIR)
    except ValueError:
        fail(
            "Output must remain inside the backtest directory containing this script: "
            f"{SCRIPT_DIR}"
        )
    if path.suffix.lower() != ".csv":
        fail("--output must be a .csv file")
    return path


def read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        fail(f"Could not read {label}: {exc}")
    if df.empty:
        fail(f"{label} is empty: {path}")
    return df


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"{label} missing required columns: {missing}")


def prepare_walkforward(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "season",
        "week",
        "game_id",
        "away_team",
        "home_team",
        "away_moneyline",
        "home_moneyline",
        "spread_line",
        "away_spread_odds",
        "home_spread_odds",
        "total_line",
        "under_odds",
        "over_odds",
        "actual_margin",
        "actual_home_ats_result",
        "actual_total_result",
        "home_win_probability",
        "away_win_probability",
        "home_cover_probability",
        "away_cover_probability",
        "over_probability",
        "under_probability",
    ]
    require_columns(df, required, "walkforward_probabilities.csv")

    out = df.copy()
    out["_game_id"] = out["game_id"].map(normalize_game_id)
    if (out["_game_id"] == "").any():
        fail("walkforward_probabilities.csv contains blank game_id values")
    if out["_game_id"].duplicated().any():
        examples = out.loc[out["_game_id"].duplicated(keep=False), "_game_id"].head(10).tolist()
        fail(f"walkforward_probabilities.csv contains duplicate game_id values: {examples}")

    out["season"] = as_numeric(out["season"], "walkforward season").astype(int)
    out["week"] = as_numeric(out["week"], "walkforward week").astype(int)
    return out.set_index("_game_id", drop=False)


def prepare_selected(df: pd.DataFrame, market: str) -> pd.DataFrame:
    spec = MARKET_SPECS[market]
    required = [
        "season",
        "week",
        "game_id",
        "away_team",
        "home_team",
        spec["selection"],
        spec["odds"],
        spec["model_prob"],
        spec["implied_prob"],
        spec["edge"],
        spec["ev"],
        spec["kelly"],
        spec["result"],
        spec["profit"],
    ]
    require_columns(df, required, f"historical_{market}_selected.csv")

    out = df.copy()
    out["_game_id"] = out["game_id"].map(normalize_game_id)
    if (out["_game_id"] == "").any():
        fail(f"{market}: blank game_id values")
    if out["_game_id"].duplicated().any():
        examples = out.loc[out["_game_id"].duplicated(keep=False), "_game_id"].head(10).tolist()
        fail(f"{market}: duplicate game_id values: {examples}")

    out["season"] = as_numeric(out["season"], f"{market} season").astype(int)
    out["week"] = as_numeric(out["week"], f"{market} week").astype(int)

    for col in [
        spec["odds"],
        spec["model_prob"],
        spec["implied_prob"],
        spec["edge"],
        spec["ev"],
        spec["kelly"],
        spec["profit"],
    ]:
        out[col] = as_numeric(out[col], f"{market} {col}")

    results = out[spec["result"]].astype(str).str.upper().str.strip()
    bad_results = sorted(set(results) - {"WIN", "LOSS", "PUSH"})
    if bad_results:
        fail(f"{market}: unexpected result values: {bad_results}")
    out[spec["result"]] = results

    return out


def american_implied_probability(odds: float) -> float:
    if odds == 0 or not math.isfinite(odds):
        fail(f"Invalid American odds: {odds}")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def expected_from_walk(row: pd.Series, walk: pd.Series, market: str) -> tuple[float, float, str]:
    spec = MARKET_SPECS[market]
    selection = str(row[spec["selection"]]).strip().upper()

    if market == "moneyline":
        if selection == "HOME":
            probability = float(walk["home_win_probability"])
            odds = float(walk["home_moneyline"])
        elif selection == "AWAY":
            probability = float(walk["away_win_probability"])
            odds = float(walk["away_moneyline"])
        else:
            fail(f"moneyline game {row['_game_id']}: invalid selection {selection!r}")

        margin = float(walk["actual_margin"])
        if abs(margin) < 1e-12:
            result = "PUSH"
        else:
            winner = "HOME" if margin > 0 else "AWAY"
            result = "WIN" if selection == winner else "LOSS"
        return probability, odds, result

    if market == "spread":
        if selection == "HOME":
            probability = float(walk["home_cover_probability"])
            odds = float(walk["home_spread_odds"])
            result = str(walk["actual_home_ats_result"]).strip().upper()
        elif selection == "AWAY":
            probability = float(walk["away_cover_probability"])
            odds = float(walk["away_spread_odds"])
            home_result = str(walk["actual_home_ats_result"]).strip().upper()
            invert = {"WIN": "LOSS", "LOSS": "WIN", "PUSH": "PUSH"}
            if home_result not in invert:
                fail(
                    f"spread game {row['_game_id']}: invalid actual_home_ats_result "
                    f"{home_result!r}"
                )
            result = invert[home_result]
        else:
            fail(f"spread game {row['_game_id']}: invalid selection {selection!r}")
        return probability, odds, result

    if selection == "OVER":
        probability = float(walk["over_probability"])
        odds = float(walk["over_odds"])
    elif selection == "UNDER":
        probability = float(walk["under_probability"])
        odds = float(walk["under_odds"])
    else:
        fail(f"total game {row['_game_id']}: invalid selection {selection!r}")

    actual = str(walk["actual_total_result"]).strip().upper()
    if actual == "PUSH":
        result = "PUSH"
    elif actual in {"OVER", "UNDER"}:
        result = "WIN" if selection == actual else "LOSS"
    else:
        fail(f"total game {row['_game_id']}: invalid actual_total_result {actual!r}")
    return probability, odds, result


def validate_against_walkforward(
    selected: pd.DataFrame,
    walkforward: pd.DataFrame,
    market: str,
) -> None:
    spec = MARKET_SPECS[market]
    tolerance = 1e-9

    missing = sorted(set(selected["_game_id"]) - set(walkforward.index))
    if missing:
        fail(f"{market}: {len(missing)} selected game_id values not found in walkforward file")

    for _, row in selected.iterrows():
        gid = row["_game_id"]
        walk = walkforward.loc[gid]

        if int(row["season"]) != int(walk["season"]) or int(row["week"]) != int(walk["week"]):
            fail(f"{market} game {gid}: season/week mismatch against walkforward file")

        if str(row["away_team"]).strip() != str(walk["away_team"]).strip():
            fail(f"{market} game {gid}: away_team mismatch against walkforward file")
        if str(row["home_team"]).strip() != str(walk["home_team"]).strip():
            fail(f"{market} game {gid}: home_team mismatch against walkforward file")

        expected_prob, expected_odds, expected_result = expected_from_walk(
            row, walk, market
        )

        if not math.isfinite(expected_prob):
            fail(f"{market} game {gid}: selected wager has unavailable walkforward probability")

        if not math.isclose(
            float(row[spec["model_prob"]]),
            expected_prob,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            fail(f"{market} game {gid}: model probability mismatch")

        if not math.isclose(
            float(row[spec["odds"]]),
            expected_odds,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            fail(f"{market} game {gid}: American odds mismatch")

        implied = american_implied_probability(float(row[spec["odds"]]))
        if not math.isclose(
            float(row[spec["implied_prob"]]),
            implied,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            fail(f"{market} game {gid}: implied probability does not match odds")

        edge = float(row[spec["model_prob"]]) - float(row[spec["implied_prob"]])
        if not math.isclose(
            float(row[spec["edge"]]),
            edge,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            fail(f"{market} game {gid}: edge does not reconcile")

        if str(row[spec["result"]]).upper() != expected_result:
            fail(f"{market} game {gid}: graded result mismatch")

    print(f"validated {market}: {len(selected)} selected wager(s)")


def optional_values(values: list[float] | None) -> list[float | None]:
    return [None] if values is None else list(values)


def apply_filters(
    df: pd.DataFrame,
    market: str,
    min_ev: float | None,
    min_edge: float | None,
    min_model_prob: float | None,
    min_odds: float | None,
    max_odds: float | None,
    min_kelly: float | None,
    total_side: str,
) -> pd.DataFrame:
    spec = MARKET_SPECS[market]
    mask = pd.Series(True, index=df.index)

    if min_ev is not None:
        mask &= df[spec["ev"]] >= min_ev
    if min_edge is not None:
        mask &= df[spec["edge"]] >= min_edge
    if min_model_prob is not None:
        mask &= df[spec["model_prob"]] >= min_model_prob
    if min_odds is not None:
        mask &= df[spec["odds"]] >= min_odds
    if max_odds is not None:
        mask &= df[spec["odds"]] <= max_odds
    if min_kelly is not None:
        mask &= df[spec["kelly"]] >= min_kelly

    if market == "total" and total_side != "ALL":
        mask &= (
            df[spec["selection"]].astype(str).str.upper().str.strip() == total_side
        )

    return df.loc[mask].copy()


def summarize_subset(df: pd.DataFrame, market: str) -> dict[str, float | int]:
    spec = MARKET_SPECS[market]
    results = df[spec["result"]] if not df.empty else pd.Series(dtype=str)

    wins = int((results == "WIN").sum())
    losses = int((results == "LOSS").sum())
    pushes = int((results == "PUSH").sum())
    bets = int(len(df))
    decisions = wins + losses
    net = float(df[spec["profit"]].sum()) if bets else 0.0
    win_rate = (100.0 * wins / decisions) if decisions else np.nan
    roi = (100.0 * net / bets) if bets else np.nan
    avg_model = float(df[spec["model_prob"]].mean()) if bets else np.nan
    actual_rate = (wins / decisions) if decisions else np.nan

    return {
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "decisions": decisions,
        "win_rate_pct": win_rate,
        "avg_model_probability": avg_model,
        "actual_win_rate": actual_rate,
        "calibration_gap_pp": (
            100.0 * (avg_model - actual_rate)
            if decisions and math.isfinite(avg_model)
            else np.nan
        ),
        "avg_implied_probability": (
            float(df[spec["implied_prob"]].mean()) if bets else np.nan
        ),
        "avg_edge": float(df[spec["edge"]].mean()) if bets else np.nan,
        "avg_ev": float(df[spec["ev"]].mean()) if bets else np.nan,
        "avg_odds_american": float(df[spec["odds"]].mean()) if bets else np.nan,
        "avg_kelly": float(df[spec["kelly"]].mean()) if bets else np.nan,
        "flat_net_units": net,
        "flat_roi_pct": roi,
    }


def build_report(
    selected_by_market: dict[str, pd.DataFrame],
    markets: list[str],
    min_evs: list[float | None],
    min_edges: list[float | None],
    min_probs: list[float | None],
    min_odds_values: list[float | None],
    max_odds_values: list[float | None],
    min_kelly_values: list[float | None],
) -> pd.DataFrame:
    seasons = sorted(
        {
            int(season)
            for market in markets
            for season in selected_by_market[market]["season"].unique().tolist()
        }
    )

    rows: list[dict[str, object]] = []

    threshold_grid = itertools.product(
        min_evs,
        min_edges,
        min_probs,
        min_odds_values,
        max_odds_values,
        min_kelly_values,
    )

    # itertools.product is an iterator, so materialize once for reuse by markets.
    threshold_grid = list(threshold_grid)

    for market in markets:
        source = selected_by_market[market]
        sides = ["ALL", "OVER", "UNDER"] if market == "total" else ["ALL"]

        for (
            min_ev,
            min_edge,
            min_prob,
            min_odds,
            max_odds,
            min_kelly,
        ) in threshold_grid:
            if (
                min_odds is not None
                and max_odds is not None
                and min_odds > max_odds
            ):
                fail(
                    f"Invalid odds range: min {min_odds} is greater than max {max_odds}"
                )

            for side in sides:
                subset = apply_filters(
                    source,
                    market,
                    min_ev,
                    min_edge,
                    min_prob,
                    min_odds,
                    max_odds,
                    min_kelly,
                    side,
                )

                overall = summarize_subset(subset, market)
                row: dict[str, object] = {
                    "market": market,
                    "selection_side": side,
                    "min_ev": min_ev,
                    "min_edge": min_edge,
                    "min_model_probability": min_prob,
                    "min_odds_american": min_odds,
                    "max_odds_american": max_odds,
                    "min_kelly": min_kelly,
                    **overall,
                }

                season_rois: list[float] = []
                for season in seasons:
                    season_subset = subset.loc[subset["season"] == season]
                    stats = summarize_subset(season_subset, market)
                    prefix = str(season)
                    row[f"{prefix}_bets"] = stats["bets"]
                    row[f"{prefix}_wins"] = stats["wins"]
                    row[f"{prefix}_losses"] = stats["losses"]
                    row[f"{prefix}_pushes"] = stats["pushes"]
                    row[f"{prefix}_win_rate_pct"] = stats["win_rate_pct"]
                    row[f"{prefix}_flat_net_units"] = stats["flat_net_units"]
                    row[f"{prefix}_flat_roi_pct"] = stats["flat_roi_pct"]

                    season_roi = stats["flat_roi_pct"]
                    if isinstance(season_roi, (int, float)) and math.isfinite(season_roi):
                        season_rois.append(float(season_roi))

                row["positive_seasons"] = sum(x > 0 for x in season_rois)
                row["negative_seasons"] = sum(x < 0 for x in season_rois)
                row["breakeven_seasons"] = sum(
                    math.isclose(x, 0.0, abs_tol=1e-12) for x in season_rois
                )
                row["worst_season_roi_pct"] = (
                    min(season_rois) if season_rois else np.nan
                )
                row["best_season_roi_pct"] = (
                    max(season_rois) if season_rois else np.nan
                )
                rows.append(row)

    return pd.DataFrame(rows)


def atomic_write_csv(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a filter-performance report from completed historical NFL "
            "backtest selections. No model retraining."
        )
    )

    parser.add_argument(
        "--walkforward",
        type=Path,
        default=DEFAULT_WALKFORWARD,
        help="Path to walkforward_probabilities.csv",
    )
    parser.add_argument(
        "--moneyline",
        type=Path,
        default=DEFAULT_MONEYLINE,
        help="Path to historical_moneyline_selected.csv",
    )
    parser.add_argument(
        "--spread",
        type=Path,
        default=DEFAULT_SPREAD,
        help="Path to historical_spread_selected.csv",
    )
    parser.add_argument(
        "--total",
        type=Path,
        default=DEFAULT_TOTAL,
        help="Path to historical_total_selected.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV output path; must be inside this script's directory",
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=["moneyline", "spread", "total"],
        default=["moneyline", "spread", "total"],
        help="Markets to include",
    )

    parser.add_argument("--min-ev", nargs="+", type=float)
    parser.add_argument("--min-edge", nargs="+", type=float)
    parser.add_argument("--min-model-prob", nargs="+", type=float)
    parser.add_argument("--min-odds-american", nargs="+", type=float)
    parser.add_argument("--max-odds-american", nargs="+", type=float)
    parser.add_argument("--min-kelly", nargs="+", type=float)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    walk_path = ensure_existing_file(args.walkforward, "walkforward probabilities")
    input_paths = {
        "moneyline": ensure_existing_file(args.moneyline, "moneyline selections"),
        "spread": ensure_existing_file(args.spread, "spread selections"),
        "total": ensure_existing_file(args.total, "total selections"),
    }
    output_path = ensure_output_path(args.output)

    walkforward = prepare_walkforward(
        read_csv(walk_path, "walkforward_probabilities.csv")
    )

    selected_by_market: dict[str, pd.DataFrame] = {}
    for market in args.markets:
        df = prepare_selected(
            read_csv(input_paths[market], f"historical_{market}_selected.csv"),
            market,
        )
        validate_against_walkforward(df, walkforward, market)
        selected_by_market[market] = df

    report = build_report(
        selected_by_market=selected_by_market,
        markets=args.markets,
        min_evs=optional_values(args.min_ev),
        min_edges=optional_values(args.min_edge),
        min_probs=optional_values(args.min_model_prob),
        min_odds_values=optional_values(args.min_odds_american),
        max_odds_values=optional_values(args.max_odds_american),
        min_kelly_values=optional_values(args.min_kelly),
    )

    atomic_write_csv(report, output_path)

    print(f"Wrote: {output_path}")
    print(f"Rows: {len(report)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

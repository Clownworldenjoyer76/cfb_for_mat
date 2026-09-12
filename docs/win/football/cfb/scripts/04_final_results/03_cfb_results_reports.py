#!/usr/bin/env python3
"""
CFB graded-results report builder.

Consumes the one-row-per-bet ledger produced by 02_cfb_results_analyze.py and
writes season, week, market, side, EV, odds, Kelly, probability, spread-line,
and total-range reports modeled on the MLB grading-report layer.

READS:
    docs/win/football/cfb/04_final_results/intermediate/work_cfb.csv

WRITES:
    docs/win/football/cfb/04_final_results/cfb_summary_overall.csv
    docs/win/football/cfb/04_final_results/reports/...
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CFB_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_INPUT_FILE = CFB_ROOT / "04_final_results" / "intermediate" / "work_cfb.csv"
SUMMARY_DIR = CFB_ROOT / "04_final_results"
REPORTS_DIR = SUMMARY_DIR / "reports"
OVERVIEW_DIR = REPORTS_DIR / "overview"
ML_DIR = REPORTS_DIR / "moneyline"
SPREAD_DIR = REPORTS_DIR / "spread"
TOTAL_DIR = REPORTS_DIR / "totals"

LEAGUE = "CFB"

VALID_GRADED_RESULTS = {"Win", "Loss", "Push"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"} else text


def to_float(value: Any) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, path)


def clear_report_outputs() -> None:
    if REPORTS_DIR.exists():
        shutil.rmtree(REPORTS_DIR)

    for directory in [OVERVIEW_DIR, ML_DIR, SPREAD_DIR, TOTAL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def normalize_result(value: Any) -> str:
    raw = clean(value).casefold()
    mapping = {
        "win": "Win",
        "loss": "Loss",
        "push": "Push",
        "void": "Void",
        "pending": "Pending",
        "invalid selection": "Invalid Selection",
        "invalid line": "Invalid Line",
        "no bet": "No Bet",
    }
    return mapping.get(raw, clean(value).title())


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def numeric_sort_value(value: Any) -> tuple[int, float | str]:
    number = to_float(value)
    return (0, number) if number is not None else (1, clean(value))


# ---------------------------------------------------------------------------
# Metric definitions / aggregation
# ---------------------------------------------------------------------------

def write_metric_definitions() -> None:
    definitions = pd.DataFrame(
        [
            {
                "metric": "Win_Pct",
                "definition": "wins / (wins + losses)",
                "push_handling": "pushes excluded from denominator",
            },
            {
                "metric": "Win_Pct_All_Bets",
                "definition": "wins / (wins + losses + pushes)",
                "push_handling": "pushes included in denominator",
            },
            {
                "metric": "units",
                "definition": "sum of grader-produced bet_units",
                "push_handling": "pushes and voids contribute 0 units",
            },
            {
                "metric": "ROI_Excluding_Pushes",
                "definition": "units / (wins + losses)",
                "push_handling": "pushes excluded from denominator",
            },
            {
                "metric": "ROI_Including_Pushes",
                "definition": "units / (wins + losses + pushes)",
                "push_handling": "pushes included in denominator",
            },
            {
                "metric": "Total",
                "definition": "wins + losses + pushes",
                "push_handling": "void, pending, and invalid bets excluded",
            },
            {
                "metric": "Selected",
                "definition": "all selected wagers present in work_cfb.csv",
                "push_handling": "includes void, pending, and invalid selected wagers",
            },
        ]
    )
    write_csv(definitions, OVERVIEW_DIR / "cfb_report_metric_definitions.csv")


def build_metric_row(sub: pd.DataFrame) -> dict[str, Any]:
    results = sub["bet_result"]

    wins = int((results == "Win").sum())
    losses = int((results == "Loss").sum())
    pushes = int((results == "Push").sum())
    voids = int((results == "Void").sum())
    pending = int((results == "Pending").sum())
    invalid = int(
        (~results.isin({"Win", "Loss", "Push", "Void", "Pending"})).sum()
    )

    bets_excluding_pushes = wins + losses
    bets_including_pushes = wins + losses + pushes
    selected = len(sub)

    win_pct = (
        round(wins / bets_excluding_pushes, 4)
        if bets_excluding_pushes > 0
        else 0.0
    )
    win_pct_all_bets = (
        round(wins / bets_including_pushes, 4)
        if bets_including_pushes > 0
        else 0.0
    )

    units = pd.to_numeric(sub["bet_units"], errors="coerce").dropna()
    total_units = round(float(units.sum()), 4) if not units.empty else 0.0

    roi_excluding_pushes = (
        round(total_units / bets_excluding_pushes, 4)
        if bets_excluding_pushes > 0
        else 0.0
    )
    roi_including_pushes = (
        round(total_units / bets_including_pushes, 4)
        if bets_including_pushes > 0
        else 0.0
    )

    ev_vals = pd.to_numeric(sub["ev"], errors="coerce").dropna()
    odds_vals = pd.to_numeric(sub["odds_american"], errors="coerce").dropna()
    prob_vals = pd.to_numeric(sub["model_prob"], errors="coerce").dropna()
    kelly_vals = pd.to_numeric(sub["kelly"], errors="coerce").dropna()

    return {
        "Win": wins,
        "Loss": losses,
        "Push": pushes,
        "Void": voids,
        "Pending": pending,
        "Invalid": invalid,
        "Total": bets_including_pushes,
        "Selected": selected,
        "bets_excluding_pushes": bets_excluding_pushes,
        "bets_including_pushes": bets_including_pushes,
        "Win_Pct": win_pct,
        "Win_Pct_All_Bets": win_pct_all_bets,
        "units": total_units,
        "ROI_Excluding_Pushes": roi_excluding_pushes,
        "ROI_Including_Pushes": roi_including_pushes,
        "avg_ev": round(float(ev_vals.mean()), 4) if not ev_vals.empty else None,
        "avg_odds": round(float(odds_vals.mean()), 1) if not odds_vals.empty else None,
        "avg_model_prob": (
            round(float(prob_vals.mean()), 4) if not prob_vals.empty else None
        ),
        "avg_kelly": (
            round(float(kelly_vals.mean()), 4) if not kelly_vals.empty else None
        ),
    }


def metric_columns(prefix: list[str]) -> list[str]:
    return prefix + [
        "Win",
        "Loss",
        "Push",
        "Void",
        "Pending",
        "Invalid",
        "Total",
        "Selected",
        "bets_excluding_pushes",
        "bets_including_pushes",
        "Win_Pct",
        "Win_Pct_All_Bets",
        "units",
        "ROI_Excluding_Pushes",
        "ROI_Including_Pushes",
        "avg_ev",
        "avg_odds",
        "avg_model_prob",
        "avg_kelly",
    ]


def aggregate(
    df: pd.DataFrame,
    group_cols: list[str],
    variable_label: str | None = None,
) -> pd.DataFrame:
    prefix_cols = [
        "variable" if variable_label and col == group_cols[-1] else col
        for col in group_cols
    ]

    if df.empty:
        return pd.DataFrame(columns=metric_columns(prefix_cols))

    rows: list[dict[str, Any]] = []

    for keys, sub in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row: dict[str, Any] = {}
        for index, column in enumerate(group_cols):
            label = "variable" if variable_label and index == len(group_cols) - 1 else column
            row[label] = keys[index]

        row.update(build_metric_row(sub))
        rows.append(row)

    return pd.DataFrame(rows)


def write_bucket_report(
    src: pd.DataFrame,
    output_dir: Path,
    bucket_col: str,
    filename: str,
    extra_group_cols: list[str] | None = None,
) -> None:
    if src.empty or bucket_col not in src.columns:
        return

    usable = src[src[bucket_col].astype(str).ne("UNBUCKETED")].copy()
    if usable.empty:
        return

    groups = ["league", "season", "market_type"]
    if extra_group_cols:
        groups.extend(extra_group_cols)
    groups.append(bucket_col)

    report = aggregate(usable, groups, variable_label=bucket_col)
    write_csv(report, output_dir / filename)


# ---------------------------------------------------------------------------
# Enrichment / reports
# ---------------------------------------------------------------------------

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    require_columns(
        work,
        [
            "season",
            "week",
            "game_id",
            "market_type",
            "bet_side",
            "side_group",
            "odds_american",
            "model_prob",
            "ev",
            "kelly",
            "bet_result",
            "bet_units",
            "ev_bucket",
            "odds_bucket",
            "kelly_bucket",
            "win_prob_bucket",
            "spread_line_bucket",
            "spread_role",
            "total_bucket",
        ],
        "work_cfb.csv",
    )

    work["league"] = LEAGUE
    work["market_type"] = work["market_type"].astype(str).str.strip().str.lower()
    work["bet_side"] = work["bet_side"].astype(str).str.strip().str.lower()
    work["side_group"] = work["side_group"].astype(str).str.strip().str.upper()
    work["bet_result"] = work["bet_result"].map(normalize_result)

    return work


def build_top_summary(df: pd.DataFrame) -> None:
    report = aggregate(df, ["league", "season", "market_type"])
    write_csv(report, SUMMARY_DIR / "cfb_summary_overall.csv")


def build_overview(df: pd.DataFrame) -> None:
    write_metric_definitions()

    write_csv(
        aggregate(df, ["league", "season"]),
        OVERVIEW_DIR / "cfb_summary_overall.csv",
    )

    write_csv(
        aggregate(
            df,
            ["league", "season", "market_type"],
            variable_label="market_type",
        ),
        OVERVIEW_DIR / "cfb_summary_by_market.csv",
    )

    side = df[df["side_group"].isin(["HOME", "AWAY", "OVER", "UNDER"])].copy()
    write_csv(
        aggregate(
            side,
            ["league", "season", "side_group"],
            variable_label="side_group",
        ),
        OVERVIEW_DIR / "cfb_summary_by_side_group.csv",
    )

    by_week = aggregate(
        df,
        ["league", "season", "week"],
        variable_label="week",
    )

    if not by_week.empty:
        by_week["_season_sort"] = pd.to_numeric(by_week["season"], errors="coerce")
        by_week["_week_sort"] = pd.to_numeric(by_week["variable"], errors="coerce")
        by_week = by_week.sort_values(
            ["_season_sort", "_week_sort"],
            kind="stable",
        ).drop(columns=["_season_sort", "_week_sort"])

    write_csv(by_week, OVERVIEW_DIR / "cfb_summary_by_week.csv")

    cumulative = by_week.copy()
    if not cumulative.empty:
        cumulative["cumulative_units"] = (
            cumulative.groupby("season", dropna=False)["units"].cumsum().round(4)
        )
    write_csv(cumulative, OVERVIEW_DIR / "cfb_cumulative_units_by_week.csv")

    if "day_night" in df.columns:
        timed = df[df["day_night"].astype(str).str.strip().ne("")].copy()
        write_csv(
            aggregate(
                timed,
                ["league", "season", "day_night"],
                variable_label="day_night",
            ),
            OVERVIEW_DIR / "cfb_summary_by_day_night.csv",
        )

    log_columns = [
        "season",
        "season_type",
        "week",
        "game_date",
        "game_time",
        "game_id",
        "away_team",
        "home_team",
        "market_type",
        "bet_side",
        "side_group",
        "line",
        "odds_american",
        "model_prob",
        "implied_prob",
        "edge",
        "ev",
        "kelly",
        "selection_reason",
        "bet_result",
        "bet_units",
        "final_status",
        "final_away_score",
        "final_home_score",
        "final_total",
        "final_home_margin",
    ]
    available = [column for column in log_columns if column in df.columns]
    write_csv(df[available].copy(), OVERVIEW_DIR / "cfb_bet_log.csv")


def build_moneyline(df: pd.DataFrame) -> None:
    ml = df[df["market_type"].eq("moneyline")].copy()
    if ml.empty:
        return

    write_bucket_report(ml, ML_DIR, "ev_bucket", "cfb_moneyline_by_ev.csv")
    write_bucket_report(ml, ML_DIR, "odds_bucket", "cfb_moneyline_by_odds.csv")
    write_bucket_report(ml, ML_DIR, "kelly_bucket", "cfb_moneyline_by_kelly.csv")
    write_bucket_report(ml, ML_DIR, "win_prob_bucket", "cfb_moneyline_by_win_prob.csv")

    home_away = ml[ml["side_group"].isin(["HOME", "AWAY"])].copy()
    write_csv(
        aggregate(
            home_away,
            ["league", "season", "market_type", "side_group"],
            variable_label="side_group",
        ),
        ML_DIR / "cfb_moneyline_by_home_away.csv",
    )

    for bucket, filename in [
        ("ev_bucket", "cfb_moneyline_by_ev_home_away_summary.csv"),
        ("odds_bucket", "cfb_moneyline_by_odds_home_away_summary.csv"),
        ("kelly_bucket", "cfb_moneyline_by_kelly_home_away_summary.csv"),
        ("win_prob_bucket", "cfb_moneyline_by_win_prob_home_away_summary.csv"),
    ]:
        write_bucket_report(
            home_away,
            ML_DIR,
            bucket,
            filename,
            extra_group_cols=["side_group"],
        )


def build_spread(df: pd.DataFrame) -> None:
    spread = df[df["market_type"].eq("spread")].copy()
    if spread.empty:
        return

    write_bucket_report(spread, SPREAD_DIR, "ev_bucket", "cfb_spread_by_ev.csv")
    write_bucket_report(spread, SPREAD_DIR, "odds_bucket", "cfb_spread_by_odds.csv")
    write_bucket_report(spread, SPREAD_DIR, "kelly_bucket", "cfb_spread_by_kelly.csv")
    write_bucket_report(spread, SPREAD_DIR, "win_prob_bucket", "cfb_spread_by_win_prob.csv")
    write_bucket_report(spread, SPREAD_DIR, "spread_line_bucket", "cfb_spread_by_line.csv")

    home_away = spread[spread["side_group"].isin(["HOME", "AWAY"])].copy()
    write_csv(
        aggregate(
            home_away,
            ["league", "season", "market_type", "side_group"],
            variable_label="side_group",
        ),
        SPREAD_DIR / "cfb_spread_by_home_away.csv",
    )

    roles = spread[spread["spread_role"].ne("UNBUCKETED")].copy()
    write_csv(
        aggregate(
            roles,
            ["league", "season", "market_type", "spread_role"],
            variable_label="spread_role",
        ),
        SPREAD_DIR / "cfb_spread_by_favorite_underdog.csv",
    )

    for bucket, filename in [
        ("ev_bucket", "cfb_spread_by_ev_home_away_summary.csv"),
        ("odds_bucket", "cfb_spread_by_odds_home_away_summary.csv"),
        ("kelly_bucket", "cfb_spread_by_kelly_home_away_summary.csv"),
        ("win_prob_bucket", "cfb_spread_by_win_prob_home_away_summary.csv"),
        ("spread_line_bucket", "cfb_spread_by_line_home_away_summary.csv"),
    ]:
        write_bucket_report(
            home_away,
            SPREAD_DIR,
            bucket,
            filename,
            extra_group_cols=["side_group"],
        )


def build_totals(df: pd.DataFrame) -> None:
    totals = df[df["market_type"].eq("total")].copy()
    if totals.empty:
        return

    write_bucket_report(totals, TOTAL_DIR, "ev_bucket", "cfb_total_by_ev.csv")
    write_bucket_report(totals, TOTAL_DIR, "odds_bucket", "cfb_total_by_odds.csv")
    write_bucket_report(totals, TOTAL_DIR, "kelly_bucket", "cfb_total_by_kelly.csv")
    write_bucket_report(totals, TOTAL_DIR, "win_prob_bucket", "cfb_total_by_win_prob.csv")
    write_bucket_report(totals, TOTAL_DIR, "total_bucket", "cfb_total_by_total_range.csv")

    over_under = totals[totals["side_group"].isin(["OVER", "UNDER"])].copy()
    write_csv(
        aggregate(
            over_under,
            ["league", "season", "market_type", "side_group"],
            variable_label="side_group",
        ),
        TOTAL_DIR / "cfb_total_by_over_under.csv",
    )

    for bucket, filename in [
        ("ev_bucket", "cfb_total_by_ev_over_under_summary.csv"),
        ("odds_bucket", "cfb_total_by_odds_over_under_summary.csv"),
        ("kelly_bucket", "cfb_total_by_kelly_over_under_summary.csv"),
        ("win_prob_bucket", "cfb_total_by_win_prob_over_under_summary.csv"),
        ("total_bucket", "cfb_total_by_total_range_over_under_summary.csv"),
    ]:
        write_bucket_report(
            over_under,
            TOTAL_DIR,
            bucket,
            filename,
            extra_group_cols=["side_group"],
        )


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CFB grading reports.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Optional season filter. If omitted, all seasons in work_cfb.csv are reported.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_file = args.input.resolve()

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    clear_report_outputs()

    df = pd.read_csv(
        input_file,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )
    df = enrich(df)

    if args.season is not None:
        season_values = pd.to_numeric(df["season"], errors="coerce")
        df = df.loc[season_values.eq(args.season)].copy()

    build_top_summary(df)
    build_overview(df)
    build_moneyline(df)
    build_spread(df)
    build_totals(df)

    print(
        "CFB reports complete: "
        f"selected_bets={len(df)} seasons={sorted(df['season'].dropna().unique().tolist()) if not df.empty else []} "
        f"reports_dir={REPORTS_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

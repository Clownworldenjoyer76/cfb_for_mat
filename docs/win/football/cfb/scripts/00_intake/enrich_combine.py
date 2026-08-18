#!/usr/bin/env python
"""
Combine weekly NFL moneyline, ATS/spread, and totals enrichment outputs.

Inputs:
  docs/win/football/nfl/00_intake/predictions/enriched/moneyline/*.csv
  docs/win/football/nfl/00_intake/predictions/enriched/spread/*.csv
  docs/win/football/nfl/00_intake/predictions/enriched/totals/*.csv

Output:
  docs/win/football/nfl/00_intake/predictions/enriched/combined/<same weekly filename>

Important:
- No prediction values are recalculated.
- No enrichment rules are re-evaluated.
- No weights are applied.
- Shared game identifiers are kept once.
- Moneyline-only fields are prefixed ml_.
- ATS/spread-only fields are prefixed ats_.
- Totals-only fields are prefixed totals_.
- The script stops if weekly source files, game_ids, shared identifiers,
  or required source columns do not match.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "This script requires pandas.\n"
        "Install it with: python -m pip install pandas"
    ) from exc


SHARED_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "commence_time",
    "away_team",
    "home_team",
]

MONEYLINE_MAP = {
    "ml_matched_rule_count": "matched_rule_count",
    "ml_matched_positive_rule_count": "matched_positive_rule_count",
    "ml_matched_negative_rule_count": "matched_negative_rule_count",
    "ml_matched_rule_ids": "matched_rule_ids",
    "ml_matched_rule_conditions": "matched_rule_conditions",
    "ml_home_matched_rule_count": "home_matched_rule_count",
    "ml_home_matched_rule_ids": "home_matched_rule_ids",
    "ml_home_strongest_positive_rule_id": "home_strongest_positive_rule_id",
    "ml_home_strongest_positive_hist_win_rate_pct": "home_strongest_positive_hist_win_rate_pct",
    "ml_home_strongest_positive_lift_pp": "home_strongest_positive_lift_pp",
    "ml_home_strongest_positive_games": "home_strongest_positive_games",
    "ml_home_strongest_negative_rule_id": "home_strongest_negative_rule_id",
    "ml_home_strongest_negative_hist_win_rate_pct": "home_strongest_negative_hist_win_rate_pct",
    "ml_home_strongest_negative_lift_pp": "home_strongest_negative_lift_pp",
    "ml_home_strongest_negative_games": "home_strongest_negative_games",
    "ml_away_matched_rule_count": "away_matched_rule_count",
    "ml_away_matched_rule_ids": "away_matched_rule_ids",
    "ml_away_strongest_positive_rule_id": "away_strongest_positive_rule_id",
    "ml_away_strongest_positive_hist_win_rate_pct": "away_strongest_positive_hist_win_rate_pct",
    "ml_away_strongest_positive_lift_pp": "away_strongest_positive_lift_pp",
    "ml_away_strongest_positive_games": "away_strongest_positive_games",
    "ml_away_strongest_negative_rule_id": "away_strongest_negative_rule_id",
    "ml_away_strongest_negative_hist_win_rate_pct": "away_strongest_negative_hist_win_rate_pct",
    "ml_away_strongest_negative_lift_pp": "away_strongest_negative_lift_pp",
    "ml_away_strongest_negative_games": "away_strongest_negative_games",
    "ml_drat_matched_rule_count": "drat_matched_rule_count",
    "ml_drat_matched_rule_ids": "drat_matched_rule_ids",
    "ml_epred_matched_rule_count": "epred_matched_rule_count",
    "ml_epred_matched_rule_ids": "epred_matched_rule_ids",
    "ml_market_matched_rule_count": "market_matched_rule_count",
    "ml_market_matched_rule_ids": "market_matched_rule_ids",
    "ml_drat_epred_consensus_matched_rule_count": "drat_epred_consensus_matched_rule_count",
    "ml_drat_epred_consensus_matched_rule_ids": "drat_epred_consensus_matched_rule_ids",
    "ml_all3_consensus_matched_rule_count": "all3_consensus_matched_rule_count",
    "ml_all3_consensus_matched_rule_ids": "all3_consensus_matched_rule_ids",
}

ATS_MAP = {
    "ats_matched_rule_count": "matched_rule_count",
    "ats_matched_positive_rule_count": "matched_positive_rule_count",
    "ats_matched_negative_rule_count": "matched_negative_rule_count",
    "ats_matched_rule_ids": "matched_rule_ids",
    "ats_matched_rule_conditions": "matched_rule_conditions",
    "ats_home_matched_rule_count": "home_matched_rule_count",
    "ats_home_matched_rule_ids": "home_matched_rule_ids",
    "ats_home_strongest_positive_rule_id": "home_strongest_positive_rule_id",
    "ats_home_strongest_positive_hist_cover_rate_pct": "home_strongest_positive_hist_cover_rate_pct",
    "ats_home_strongest_positive_lift_pp": "home_strongest_positive_lift_pp",
    "ats_home_strongest_positive_games": "home_strongest_positive_games",
    "ats_home_strongest_negative_rule_id": "home_strongest_negative_rule_id",
    "ats_home_strongest_negative_hist_cover_rate_pct": "home_strongest_negative_hist_cover_rate_pct",
    "ats_home_strongest_negative_lift_pp": "home_strongest_negative_lift_pp",
    "ats_home_strongest_negative_games": "home_strongest_negative_games",
    "ats_away_matched_rule_count": "away_matched_rule_count",
    "ats_away_matched_rule_ids": "away_matched_rule_ids",
    "ats_away_strongest_positive_rule_id": "away_strongest_positive_rule_id",
    "ats_away_strongest_positive_hist_cover_rate_pct": "away_strongest_positive_hist_cover_rate_pct",
    "ats_away_strongest_positive_lift_pp": "away_strongest_positive_lift_pp",
    "ats_away_strongest_positive_games": "away_strongest_positive_games",
    "ats_away_strongest_negative_rule_id": "away_strongest_negative_rule_id",
    "ats_away_strongest_negative_hist_cover_rate_pct": "away_strongest_negative_hist_cover_rate_pct",
    "ats_away_strongest_negative_lift_pp": "away_strongest_negative_lift_pp",
    "ats_away_strongest_negative_games": "away_strongest_negative_games",
    "ats_drat_matched_rule_count": "drat_matched_rule_count",
    "ats_drat_matched_rule_ids": "drat_matched_rule_ids",
    "ats_epred_matched_rule_count": "epred_matched_rule_count",
    "ats_epred_matched_rule_ids": "epred_matched_rule_ids",
    "ats_market_matched_rule_count": "market_matched_rule_count",
    "ats_market_matched_rule_ids": "market_matched_rule_ids",
    "ats_drat_epred_consensus_matched_rule_count": "drat_epred_consensus_matched_rule_count",
    "ats_drat_epred_consensus_matched_rule_ids": "drat_epred_consensus_matched_rule_ids",
    "ats_all3_consensus_matched_rule_count": "all3_consensus_matched_rule_count",
    "ats_all3_consensus_matched_rule_ids": "all3_consensus_matched_rule_ids",
}

TOTALS_MAP = {
    "totals_matched_rule_count": "matched_rule_count",
    "totals_matched_positive_rule_count": "matched_positive_rule_count",
    "totals_matched_negative_rule_count": "matched_negative_rule_count",
    "totals_matched_rule_ids": "matched_rule_ids",
    "totals_matched_rule_conditions": "matched_rule_conditions",
    "totals_over_matched_rule_count": "over_matched_rule_count",
    "totals_over_matched_positive_rule_count": "over_matched_positive_rule_count",
    "totals_over_matched_negative_rule_count": "over_matched_negative_rule_count",
    "totals_over_matched_rule_ids": "over_matched_rule_ids",
    "totals_over_strongest_positive_rule_id": "over_strongest_positive_rule_id",
    "totals_over_strongest_positive_hist_hit_rate_pct": "over_strongest_positive_hist_hit_rate_pct",
    "totals_over_strongest_positive_lift_pp": "over_strongest_positive_lift_pp",
    "totals_over_strongest_positive_games": "over_strongest_positive_games",
    "totals_over_strongest_negative_rule_id": "over_strongest_negative_rule_id",
    "totals_over_strongest_negative_hist_hit_rate_pct": "over_strongest_negative_hist_hit_rate_pct",
    "totals_over_strongest_negative_lift_pp": "over_strongest_negative_lift_pp",
    "totals_over_strongest_negative_games": "over_strongest_negative_games",
    "totals_under_matched_rule_count": "under_matched_rule_count",
    "totals_under_matched_positive_rule_count": "under_matched_positive_rule_count",
    "totals_under_matched_negative_rule_count": "under_matched_negative_rule_count",
    "totals_under_matched_rule_ids": "under_matched_rule_ids",
    "totals_under_strongest_positive_rule_id": "under_strongest_positive_rule_id",
    "totals_under_strongest_positive_hist_hit_rate_pct": "under_strongest_positive_hist_hit_rate_pct",
    "totals_under_strongest_positive_lift_pp": "under_strongest_positive_lift_pp",
    "totals_under_strongest_positive_games": "under_strongest_positive_games",
    "totals_under_strongest_negative_rule_id": "under_strongest_negative_rule_id",
    "totals_under_strongest_negative_hist_hit_rate_pct": "under_strongest_negative_hist_hit_rate_pct",
    "totals_under_strongest_negative_lift_pp": "under_strongest_negative_lift_pp",
    "totals_under_strongest_negative_games": "under_strongest_negative_games",
    "totals_drat_matched_rule_count": "drat_matched_rule_count",
    "totals_drat_matched_rule_ids": "drat_matched_rule_ids",
    "totals_epred_matched_rule_count": "epred_matched_rule_count",
    "totals_epred_matched_rule_ids": "epred_matched_rule_ids",
    "totals_market_matched_rule_count": "market_matched_rule_count",
    "totals_market_matched_rule_ids": "market_matched_rule_ids",
    "totals_drat_epred_consensus_matched_rule_count": "drat_epred_consensus_matched_rule_count",
    "totals_drat_epred_consensus_matched_rule_ids": "drat_epred_consensus_matched_rule_ids",
    "totals_all3_consensus_matched_rule_count": "all3_consensus_matched_rule_count",
    "totals_all3_consensus_matched_rule_ids": "all3_consensus_matched_rule_ids",
}


def repo_root() -> Path:
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace:
        return Path(workspace)

    script_path = Path(__file__).resolve()
    try:
        return script_path.parents[6]
    except IndexError as exc:
        raise RuntimeError(
            "Could not determine repository root. "
            "Set GITHUB_WORKSPACE before running this script."
        ) from exc


def read_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, low_memory=False)

    if "game_id" not in df.columns:
        raise ValueError(f"{label}: missing required column game_id in {path}")

    df["game_id"] = df["game_id"].astype("string").str.strip()

    if df["game_id"].isna().any() or df["game_id"].eq("").any():
        raise ValueError(f"{label}: blank game_id found in {path}")

    duplicates = df.loc[df["game_id"].duplicated(keep=False), "game_id"].tolist()
    if duplicates:
        raise ValueError(
            f"{label}: duplicate game_id values in {path}: "
            + ", ".join(sorted(set(duplicates)))
        )

    return df


def require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    label: str,
    path: Path,
) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{label}: missing required columns in {path}: "
            + ", ".join(missing)
        )


def validate_source_columns(
    df: pd.DataFrame,
    mapping: dict[str, str],
    label: str,
    path: Path,
) -> None:
    require_columns(df, SHARED_COLUMNS + list(mapping.values()), label, path)


def normalized_for_compare(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def validate_same_games(
    moneyline: pd.DataFrame,
    spread: pd.DataFrame,
    totals: pd.DataFrame,
    filename: str,
) -> None:
    ml_ids = set(moneyline["game_id"])
    ats_ids = set(spread["game_id"])
    totals_ids = set(totals["game_id"])

    if ml_ids == ats_ids == totals_ids:
        return

    messages = []

    missing_from_ats = sorted(ml_ids - ats_ids)
    missing_from_totals = sorted(ml_ids - totals_ids)
    missing_from_ml = sorted((ats_ids | totals_ids) - ml_ids)

    if missing_from_ats:
        messages.append("missing from spread: " + ", ".join(missing_from_ats))
    if missing_from_totals:
        messages.append("missing from totals: " + ", ".join(missing_from_totals))
    if missing_from_ml:
        messages.append("missing from moneyline: " + ", ".join(missing_from_ml))

    raise ValueError(
        f"{filename}: game_id sets do not match across enrichment files. "
        + " | ".join(messages)
    )


def validate_shared_values(
    moneyline: pd.DataFrame,
    spread: pd.DataFrame,
    totals: pd.DataFrame,
    filename: str,
) -> pd.DataFrame:
    ml = moneyline[SHARED_COLUMNS].copy()
    ats = spread[SHARED_COLUMNS].copy()
    tot = totals[SHARED_COLUMNS].copy()

    merged = (
        ml.merge(
            ats,
            on="game_id",
            how="inner",
            suffixes=("_ml", "_ats"),
            validate="one_to_one",
        )
        .merge(
            tot,
            on="game_id",
            how="inner",
            validate="one_to_one",
        )
    )

    for column in SHARED_COLUMNS:
        if column == "game_id":
            continue

        ml_col = f"{column}_ml"
        ats_col = f"{column}_ats"
        totals_col = column

        ml_values = normalized_for_compare(merged[ml_col])
        ats_values = normalized_for_compare(merged[ats_col])
        totals_values = normalized_for_compare(merged[totals_col])

        mismatch = ~(
            ml_values.eq(ats_values)
            & ml_values.eq(totals_values)
        )

        if mismatch.any():
            bad = merged.loc[
                mismatch,
                ["game_id", ml_col, ats_col, totals_col],
            ].copy()

            details = []
            for _, row in bad.head(10).iterrows():
                details.append(
                    f"game_id={row['game_id']} "
                    f"moneyline={row[ml_col]!r} "
                    f"spread={row[ats_col]!r} "
                    f"totals={row[totals_col]!r}"
                )

            raise ValueError(
                f"{filename}: shared column {column!r} does not match across "
                f"the three enrichment files. "
                + " | ".join(details)
            )

    return moneyline[SHARED_COLUMNS].copy()


def mapped_frame(
    df: pd.DataFrame,
    mapping: dict[str, str],
) -> pd.DataFrame:
    source_columns = ["game_id"] + list(mapping.values())
    out = df[source_columns].copy()
    rename_map = {source: combined for combined, source in mapping.items()}
    return out.rename(columns=rename_map)


def combine_week(
    moneyline_path: Path,
    spread_path: Path,
    totals_path: Path,
    output_path: Path,
) -> None:
    filename = moneyline_path.name

    moneyline = read_csv(moneyline_path, "moneyline")
    spread = read_csv(spread_path, "spread")
    totals = read_csv(totals_path, "totals")

    validate_source_columns(moneyline, MONEYLINE_MAP, "moneyline", moneyline_path)
    validate_source_columns(spread, ATS_MAP, "spread", spread_path)
    validate_source_columns(totals, TOTALS_MAP, "totals", totals_path)

    validate_same_games(moneyline, spread, totals, filename)

    shared = validate_shared_values(moneyline, spread, totals, filename)

    ml_enrichment = mapped_frame(moneyline, MONEYLINE_MAP)
    ats_enrichment = mapped_frame(spread, ATS_MAP)
    totals_enrichment = mapped_frame(totals, TOTALS_MAP)

    combined = (
        shared.merge(ml_enrichment, on="game_id", how="left", validate="one_to_one")
        .merge(ats_enrichment, on="game_id", how="left", validate="one_to_one")
        .merge(totals_enrichment, on="game_id", how="left", validate="one_to_one")
    )

    expected_columns = (
        SHARED_COLUMNS
        + list(MONEYLINE_MAP.keys())
        + list(ATS_MAP.keys())
        + list(TOTALS_MAP.keys())
    )

    if list(combined.columns) != expected_columns:
        raise RuntimeError(
            f"{filename}: combined output columns do not match the required schema."
        )

    if len(combined) != len(moneyline):
        raise RuntimeError(
            f"{filename}: combined row count changed unexpectedly. "
            f"moneyline rows={len(moneyline)}, combined rows={len(combined)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(
        f"WROTE {output_path} | games={len(combined)} | columns={len(combined.columns)}",
        flush=True,
    )


def main() -> None:
    root = repo_root()

    enriched_root = (
        root
        / "docs"
        / "win"
        / "football"
        / "nfl"
        / "00_intake"
        / "predictions"
        / "enriched"
    )

    moneyline_dir = enriched_root / "moneyline"
    spread_dir = enriched_root / "spread"
    totals_dir = enriched_root / "totals"
    combined_dir = enriched_root / "combined"

    for label, folder in [
        ("moneyline", moneyline_dir),
        ("spread", spread_dir),
        ("totals", totals_dir),
    ]:
        if not folder.exists():
            raise FileNotFoundError(f"{label} enrichment folder not found: {folder}")

    moneyline_files = {path.name: path for path in sorted(moneyline_dir.glob("*.csv"))}
    spread_files = {path.name: path for path in sorted(spread_dir.glob("*.csv"))}
    totals_files = {path.name: path for path in sorted(totals_dir.glob("*.csv"))}

    if not moneyline_files:
        raise FileNotFoundError(
            f"No moneyline enrichment CSV files found in {moneyline_dir}"
        )

    all_filenames = set(moneyline_files) | set(spread_files) | set(totals_files)

    missing_pairs = []
    for filename in sorted(all_filenames):
        missing = []
        if filename not in moneyline_files:
            missing.append("moneyline")
        if filename not in spread_files:
            missing.append("spread")
        if filename not in totals_files:
            missing.append("totals")

        if missing:
            missing_pairs.append(f"{filename}: missing {', '.join(missing)}")

    if missing_pairs:
        raise FileNotFoundError(
            "Weekly enrichment filenames do not match across the three folders:\n"
            + "\n".join(missing_pairs)
        )

    combined_dir.mkdir(parents=True, exist_ok=True)

    for filename in sorted(all_filenames):
        combine_week(
            moneyline_path=moneyline_files[filename],
            spread_path=spread_files[filename],
            totals_path=totals_files[filename],
            output_path=combined_dir / filename,
        )

    print(
        f"COMPLETE | combined weekly files={len(all_filenames)} | output={combined_dir}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise

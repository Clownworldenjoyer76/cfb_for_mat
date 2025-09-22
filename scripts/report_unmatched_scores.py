#!/usr/bin/env python3
"""
Report (game_id, team) rows in modeling_dataset that fail to match scores.

Inputs
------
- data/modeling_dataset.csv
- data/game_scores_clean.csv   (must be UNIQUE on (game_id, team))

Outputs (written to data/diagnostics/)
--------------------------------------
- unmatched_exact.csv        : rows in modeling with no exact (game_id, team) match
- unmatched_with_norm_help.csv
    Adds normalized keys to show when a name-normalization would resolve the join
- coverage_summary.txt       : simple counts for quick inspection

Notes
-----
- "Normalization" here uppercases, strips accents, and removes non-alphanumeric chars.
- This script does NOT modify source data. It only reports.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata
import re
import sys

MODEL_CSV  = Path("data/modeling_dataset.csv")
SCORES_CSV = Path("data/game_scores_clean.csv")
OUT_DIR    = Path("data/diagnostics")

_norm_rx = re.compile(r"[^A-Z0-9]")

def norm_name(s: object) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    s = _norm_rx.sub("", s)
    return s

def main():
    if not MODEL_CSV.exists():
        print(f"ERROR: {MODEL_CSV} not found.", file=sys.stderr)
        sys.exit(2)
    if not SCORES_CSV.exists():
        print(f"ERROR: {SCORES_CSV} not found.", file=sys.stderr)
        sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    model = pd.read_csv(MODEL_CSV)
    scores = pd.read_csv(SCORES_CSV)

    required_model = {"game_id","team"}
    required_scores = {"game_id","team","points_scored"}
    if not required_model.issubset(model.columns):
        missing = sorted(required_model - set(model.columns))
        raise KeyError(f"{MODEL_CSV} missing required columns: {missing}")
    if not required_scores.issubset(scores.columns):
        missing = sorted(required_scores - set(scores.columns))
        raise KeyError(f"{SCORES_CSV} missing required columns: {missing}")

    # Enforce uniqueness in scores; if violated, diagnosis first
    dup_ct = scores.duplicated(subset=["game_id","team"]).sum()
    if dup_ct:
        dup_rows = scores[scores.duplicated(subset=["game_id","team"], keep=False)].copy()
        dup_path = OUT_DIR / "scores_duplicates_on_gameid_team.csv"
        dup_rows.to_csv(dup_path, index=False)
        print(f"[warn] {dup_ct} duplicate rows in scores on (game_id, team). Wrote: {dup_path}")

    # Exact left join to find misses
    joined = model.merge(
        scores[["game_id","team","points_scored"]],
        on=["game_id","team"],
        how="left",
        validate="m:1"
    )
    unmatched_exact = joined[joined["points_scored"].isna()].copy()
    unmatched_exact_path = OUT_DIR / "unmatched_exact.csv"
    unmatched_exact.to_csv(unmatched_exact_path, index=False)

    # Normalization-assisted diagnostic:
    # show whether a normalized name would have matched, to identify naming drift.
    model_norm = model.copy()
    model_norm["team_norm"] = model_norm["team"].map(norm_name)

    scores_norm = scores.copy()
    scores_norm["team_norm"] = scores_norm["team"].map(norm_name)
    scores_norm = scores_norm.drop_duplicates(subset=["game_id","team_norm"])

    joined_norm = (
        model_norm.merge(
            scores_norm[["game_id","team_norm","points_scored"]],
            on=["game_id","team_norm"],
            how="left",
            validate="m:1"
        )
        .rename(columns={"points_scored":"points_scored_by_norm"})
    )

    # rows that failed exact join but WOULD match by normalization
    help_norm = (
        unmatched_exact.merge(
            joined_norm[["game_id","team","points_scored_by_norm"]],
            on=["game_id","team"],
            how="left"
        )
    )
    unmatched_with_norm_help = help_norm.copy()
    unmatched_with_norm_help_path = OUT_DIR / "unmatched_with_norm_help.csv"
    unmatched_with_norm_help.to_csv(unmatched_with_norm_help_path, index=False)

    # Coverage summary
    total_model_rows = len(model)
    matched_exact = total_model_rows - len(unmatched_exact)
    matched_by_norm_only = help_norm["points_scored_by_norm"].notna().sum()
    summary_lines = [
        f"model_rows_total={total_model_rows}",
        f"matched_exact={matched_exact}",
        f"unmatched_exact={len(unmatched_exact)}",
        f"of_unmatched__would_match_by_normalization={matched_by_norm_only}",
        "",
        f"unmatched_exact_csv={unmatched_exact_path}",
        f"unmatched_with_norm_help_csv={unmatched_with_norm_help_path}",
    ]
    summary_path = OUT_DIR / "coverage_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))

if __name__ == "__main__":
    main()

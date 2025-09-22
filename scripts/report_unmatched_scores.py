#!/usr/bin/env python3
"""
Report (game_id, team) rows in modeling_dataset that fail to match scores,
and classify WHY they don't match.

Inputs
------
- data/modeling_dataset.csv
- data/game_scores_clean.csv   (must be UNIQUE on (game_id, team))

Outputs (written to data/diagnostics/)
--------------------------------------
- coverage_summary.txt
- unmatched_exact.csv                               (as before)
- unmatched_with_norm_help.csv                      (as before)
- unmatched_missing_game_id.csv                     (game_id not in scores at all)
- unmatched_team_mismatch.csv                       (game_id exists in scores, but team not found)
- unmatched_team_mismatch_would_normalize.csv       (subset that would match if names were normalized)
- top_missing_game_ids.csv                          (counts)
- top_unmatched_teams.csv                           (counts)
- unmatched_by_season.csv                           (counts)

Notes
-----
- "Normalization" uppercases, strips accents, and removes non-alphanumeric chars.
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
        print(f"ERROR: {MODEL_CSV} not found.", file=sys.stderr); sys.exit(2)
    if not SCORES_CSV.exists():
        print(f"ERROR: {SCORES_CSV} not found.", file=sys.stderr); sys.exit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- Load ----------
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

    # ---------- Enforce uniqueness (scores) ----------
    dup_ct = scores.duplicated(subset=["game_id","team"]).sum()
    if dup_ct:
        dup_rows = scores[scores.duplicated(subset=["game_id","team"], keep=False)].copy()
        dup_path = OUT_DIR / "scores_duplicates_on_gameid_team.csv"
        dup_rows.to_csv(dup_path, index=False)
        print(f"[warn] {dup_ct} duplicate rows in scores on (game_id, team). Wrote: {dup_path}")

    # ---------- Exact left join ----------
    joined = model.merge(
        scores[["game_id","team","points_scored"]],
        on=["game_id","team"],
        how="left",
        validate="m:1"
    )
    unmatched_exact = joined[joined["points_scored"].isna()].copy()
    unmatched_exact_path = OUT_DIR / "unmatched_exact.csv"
    unmatched_exact.to_csv(unmatched_exact_path, index=False)

    # ---------- Normalization help ----------
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

    # Map normalization help back to (game_id, team) space
    help_norm = unmatched_exact.merge(
        joined_norm[["game_id","team","points_scored_by_norm"]],
        on=["game_id","team"],
        how="left"
    )
    unmatched_with_norm_help = help_norm.copy()
    unmatched_with_norm_help_path = OUT_DIR / "unmatched_with_norm_help.csv"
    unmatched_with_norm_help.to_csv(unmatched_with_norm_help_path, index=False)

    # ---------- Classification: missing game_id vs team mismatch ----------
    gids_in_scores = set(scores["game_id"].dropna().unique().tolist())
    missing_gid_mask = ~unmatched_exact["game_id"].isin(gids_in_scores)

    unmatched_missing_gid = unmatched_exact[missing_gid_mask].copy()
    unmatched_team_mismatch = unmatched_exact[~missing_gid_mask].copy()

    # Of the team mismatches, which would match after normalization?
    # NOTE: we avoid misaligned boolean masks by filtering directly on a merged frame.
    team_mismatch_norm_merge = unmatched_team_mismatch.merge(
        unmatched_with_norm_help[["game_id","team","points_scored_by_norm"]],
        on=["game_id","team"],
        how="left"
    )
    unmatched_team_mismatch_would_norm = team_mismatch_norm_merge[
        team_mismatch_norm_merge["points_scored_by_norm"].notna()
    ].copy()

    # ---------- Write classified CSVs ----------
    path_missing_gid = OUT_DIR / "unmatched_missing_game_id.csv"
    path_team_mismatch = OUT_DIR / "unmatched_team_mismatch.csv"
    path_team_mismatch_norm = OUT_DIR / "unmatched_team_mismatch_would_normalize.csv"

    unmatched_missing_gid.to_csv(path_missing_gid, index=False)
    unmatched_team_mismatch.to_csv(path_team_mismatch, index=False)
    unmatched_team_mismatch_would_norm.to_csv(path_team_mismatch_norm, index=False)

    # ---------- Quick rollups ----------
    top_missing_gids = (
        unmatched_missing_gid["game_id"]
        .value_counts()
        .rename_axis("game_id")
        .reset_index(name="missing_rows")
        .head(100)
    )
    top_missing_gids_path = OUT_DIR / "top_missing_game_ids.csv"
    top_missing_gids.to_csv(top_missing_gids_path, index=False)

    top_unmatched_teams = (
        unmatched_exact["team"]
        .value_counts()
        .rename_axis("team")
        .reset_index(name="unmatched_rows")
        .head(100)
    )
    top_unmatched_teams_path = OUT_DIR / "top_unmatched_teams.csv"
    top_unmatched_teams.to_csv(top_unmatched_teams_path, index=False)

    if "season" in unmatched_exact.columns:
        unmatched_by_season = (
            unmatched_exact["season"]
            .value_counts(dropna=False)
            .rename_axis("season")
            .reset_index(name="unmatched_rows")
            .sort_values("season")
        )
        unmatched_by_season_path = OUT_DIR / "unmatched_by_season.csv"
        unmatched_by_season.to_csv(unmatched_by_season_path, index=False)
    else:
        unmatched_by_season_path = None

    # ---------- Coverage summary ----------
    total_model_rows = len(model)
    unmatched_n = len(unmatched_exact)
    matched_exact = total_model_rows - unmatched_n
    matched_by_norm_only = unmatched_with_norm_help["points_scored_by_norm"].notna().sum()

    n_missing_gid = len(unmatched_missing_gid)
    n_team_mismatch = len(unmatched_team_mismatch)
    n_team_mismatch_norm = len(unmatched_team_mismatch_would_norm)

    summary_lines = [
        f"model_rows_total={total_model_rows}",
        f"matched_exact={matched_exact}",
        f"unmatched_exact={unmatched_n}",
        f"  unmatched_missing_game_id={n_missing_gid}",
        f"  unmatched_team_mismatch={n_team_mismatch}",
        f"  of_team_mismatch__would_match_by_normalization={n_team_mismatch_norm}",
        f"of_unmatched__would_match_by_normalization={matched_by_norm_only}",
        "",
        f"unmatched_exact_csv={unmatched_exact_path}",
        f"unmatched_missing_game_id_csv={path_missing_gid}",
        f"unmatched_team_mismatch_csv={path_team_mismatch}",
        f"unmatched_team_mismatch_would_normalize_csv={path_team_mismatch_norm}",
        f"unmatched_with_norm_help_csv={unmatched_with_norm_help_path}",
        f"top_missing_game_ids_csv={top_missing_gids_path}",
        f"top_unmatched_teams_csv={top_unmatched_teams_path}",
        f"unmatched_by_season_csv={unmatched_by_season_path}" if unmatched_by_season_path else "",
    ]
    summary_path = OUT_DIR / "coverage_summary.txt"
    summary_path.write_text("\n".join([s for s in summary_lines if s != ""]), encoding="utf-8")

    # Also print to stdout so it shows up in the Actions log
    print("\n".join([s for s in summary_lines if s != ""]))

if __name__ == "__main__":
    main()

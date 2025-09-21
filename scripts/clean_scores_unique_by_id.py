#!/usr/bin/env python3
"""
Clean CFBD scores so merges are STRICT many-to-one on (game_id, team).

Inputs
------
- data/game_scores.csv           (wide, per game: home/away names + points)
- data/modeling_dataset.csv      (used ONLY to align team strings)

Output
------
- data/game_scores_clean.csv     (long, per team; UNIQUE on (game_id, team))

What this does
--------------
1) Keeps finished games (both scores numeric).
2) Expands wide home/away into long per-team rows with:
     game_id, season, week, team, opponent, is_home,
     points_scored, points_allowed, home_team, away_team, home_points, away_points
3) Enforces EXACT uniqueness on (game_id, team).
4) Aligns the 'team' strings to EXACTLY match the team strings found in
   data/modeling_dataset.csv for the same game_id. This is critical so the
   regression join on (game_id, team) succeeds, even if the modeling dataset
   uses slightly different capitalization/punctuation for school names.

Notes
-----
- We use a robust normalizer (uppercase, remove punctuation/accents/spaces)
  to match modeling names to CFBD names. If exactly one modeling name matches
  a side (home/away) by normalization, we rewrite that side's 'team' field to
  the modeling string. If only one side matches, we assign the remaining
  modeling name (if there are exactly two) to the other side by elimination.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata
import re

IN_SCORES = Path("data/game_scores.csv")
IN_MODEL  = Path("data/modeling_dataset.csv")
OUT_CSV   = Path("data/game_scores_clean.csv")


# ---------- helpers ----------
_norm_rx = re.compile(r"[^A-Z0-9]")

def norm_name(s: str | float | int) -> str:
    """Uppercase, strip accents, remove non-alnum to improve matching."""
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    s = _norm_rx.sub("", s)
    return s


def _coerce_scores(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("home_points", "away_points"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["home_points", "away_points"]).copy()
    return df


def _normalize_team_cols(df: pd.DataFrame) -> pd.DataFrame:
    # keep original strings, but also pre-compute normalized
    for col in ("home_team", "away_team"):
        df[col] = df[col].astype(str).str.strip()
        df[col + "_norm"] = df[col].map(norm_name)
    return df


def _expand_long(df: pd.DataFrame) -> pd.DataFrame:
    home_rows = pd.DataFrame({
        "game_id":   df["id"],
        "season":    df["season"],
        "week":      df["week"],
        "team":      df["home_team"],
        "team_norm": df["home_team_norm"],
        "opponent":  df["away_team"],
        "opponent_norm": df["away_team_norm"],
        "is_home":   True,
        "points_scored":  df["home_points"],
        "points_allowed": df["away_points"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_points": df["home_points"],
        "away_points": df["away_points"],
    })

    away_rows = pd.DataFrame({
        "game_id":   df["id"],
        "season":    df["season"],
        "week":      df["week"],
        "team":      df["away_team"],
        "team_norm": df["away_team_norm"],
        "opponent":  df["home_team"],
        "opponent_norm": df["home_team_norm"],
        "is_home":   False,
        "points_scored":  df["away_points"],
        "points_allowed": df["home_points"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "home_points": df["home_points"],
        "away_points": df["away_points"],
    })

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    # type hygiene
    long_df["season"] = pd.to_numeric(long_df["season"], errors="coerce").astype("Int64")
    long_df["week"]   = pd.to_numeric(long_df["week"], errors="coerce").astype("Int64")
    return long_df


def _align_team_strings_with_modeling(long_df: pd.DataFrame, modeling_keys: pd.DataFrame) -> pd.DataFrame:
    """
    For each game_id, rewrite long_df['team'] to exactly match the modeling team
    strings when the normalized names align. This preserves uniqueness and
    guarantees join compatibility on (game_id, team).
    """
    if modeling_keys.empty:
        return long_df

    modeling_keys = modeling_keys.copy()
    modeling_keys["team_norm"] = modeling_keys["team"].map(norm_name)

    # Build per-game mapping from CFBD normalized -> modeling exact
    maps = {}  # game_id -> {cfbd_team_norm: modeling_team_exact}
    for gid, sub in modeling_keys.groupby("game_id", sort=False):
        md_norms = list(sub["team_norm"])
        md_exact = list(sub["team"])
        # dedupe preserving order
        seen = set()
        md_pairs = []
        for n, e in zip(md_norms, md_exact):
            if n not in seen and n:
                md_pairs.append((n, e))
                seen.add(n)
        maps[gid] = dict(md_pairs)

    # Apply mapping
    def rewrite_row(row):
        gid = row["game_id"]
        cfbd_norm = row["team_norm"]
        m = maps.get(gid)
        if not m:
            return row["team"]  # unchanged
        # direct normalized match
        if cfbd_norm in m:
            return m[cfbd_norm]
        # If exactly 2 modeling teams exist and one matches opponent, assign the other by elimination
        # (handles abbreviations on one side)
        if gid in maps:
            # find opponent normalized in modeling keys
            opp_norm = row["opponent_norm"]
            if opp_norm in m and len(m) == 2:
                # pick the "other" modeling team string
                others = [v for k, v in m.items() if k != opp_norm]
                if others:
                    return others[0]
        return row["team"]  # fallback: keep original

    long_df = long_df.copy()
    long_df["team"] = long_df.apply(rewrite_row, axis=1)
    return long_df


# ---------- main ----------
def main():
    if not IN_SCORES.exists():
        raise FileNotFoundError(f"{IN_SCORES} not found. Run the fetch step first.")

    df = pd.read_csv(IN_SCORES)

    # Expect these columns from the fetcher
    required = ["id","season","week","home_team","away_team","home_points","away_points"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{IN_SCORES} missing columns: {missing}")

    df = _coerce_scores(df)
    df = _normalize_team_cols(df)
    long_df = _expand_long(df)

    # Deduplicate by (game_id, side) first
    before = len(long_df)
    long_df = long_df.sort_values(["season","week","game_id","is_home"]).drop_duplicates(
        subset=["game_id","is_home"], keep="first"
    )
    # After this step there should be exactly 2 rows per game_id

    # If modeling dataset exists, align team strings to modeling exact values
    if IN_MODEL.exists():
        model_df = pd.read_csv(IN_MODEL, usecols=lambda c: c in {"game_id","team","opponent"} or True)
        if "game_id" in model_df.columns and "team" in model_df.columns:
            modeling_keys = model_df[["game_id","team"]].dropna().drop_duplicates()
            long_df = _align_team_strings_with_modeling(long_df, modeling_keys)

    # Final uniqueness: exactly one row per (game_id, team)
    long_df["team"] = long_df["team"].astype(str).str.strip()
    long_df = long_df.sort_values(["season","week","game_id","team"]).drop_duplicates(
        subset=["game_id","team"], keep="first"
    )

    # Defensive check for remaining duplicates
    dups = long_df.duplicated(subset=["game_id","team"]).sum()
    if dups:
        # Deterministically keep first occurrence
        long_df = long_df.drop_duplicates(subset=["game_id","team"], keep="first")

    # Drop helper norm columns before write
    long_df = long_df.drop(columns=["team_norm","opponent_norm"], errors="ignore")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_CSV, index=False)

    print(
        f"[clean] wrote {OUT_CSV} "
        f"rows={len(long_df)} (start={before} -> unique_by_(game_id,team)={len(long_df)})"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Entry point for v4 Week 1 shadow inference plus compact audit outputs."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import projection_week1_v4_shadow as shadow


def load_week1_helper_fixed():
    helper_dir = shadow.nfl_root() / "scripts/01_merge"
    if str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))
    return importlib.import_module("projection_week1")


def market_summary(df: pd.DataFrame, positive_label: str, negative_label: str, positive_edge: str, negative_edge: str) -> dict:
    pos = pd.to_numeric(df[positive_edge], errors="coerce")
    neg = pd.to_numeric(df[negative_edge], errors="coerce")
    valid = pos.notna() & neg.notna()
    chosen_side = pd.Series("", index=df.index, dtype=object)
    chosen_edge = pd.Series(np.nan, index=df.index, dtype=float)
    chosen_side.loc[valid & (pos >= neg)] = positive_label
    chosen_side.loc[valid & (pos < neg)] = negative_label
    chosen_edge.loc[valid] = np.maximum(pos.loc[valid], neg.loc[valid])

    thresholds = {}
    for threshold in (0.0, 0.02, 0.03, 0.05, 0.08):
        mask = valid & (chosen_edge >= threshold)
        thresholds[f"{threshold:.2f}"] = {
            "n": int(mask.sum()),
            "side_counts": {str(k): int(v) for k, v in chosen_side.loc[mask].value_counts().to_dict().items()},
            "average_fair_edge": float(chosen_edge.loc[mask].mean()) if mask.any() else None,
            "max_fair_edge": float(chosen_edge.loc[mask].max()) if mask.any() else None,
        }

    ranked = df.loc[valid, ["game_id", "away_team", "home_team"]].copy()
    ranked["side"] = chosen_side.loc[valid]
    ranked["fair_edge"] = chosen_edge.loc[valid]
    ranked = ranked.sort_values("fair_edge", ascending=False).head(5)
    return {
        "valid_games": int(valid.sum()),
        "thresholds": thresholds,
        "top_five": ranked.to_dict(orient="records"),
    }


def write_compact_audit() -> None:
    root = shadow.nfl_root()
    out_dir = root / "forward_validation/v4"
    full_path = out_dir / "2026_week_1_v4_shadow.csv"
    compact_path = out_dir / "2026_week_1_v4_shadow_compact.csv"
    summary_path = out_dir / "2026_week_1_v4_shadow_summary.json"

    df = pd.read_csv(full_path, encoding="utf-8-sig", low_memory=False)
    compact_columns = [
        "game_id", "away_team", "home_team",
        "v4_predicted_margin", "v4_predicted_total",
        "v4_home_win_probability", "v4_away_win_probability",
        "v4_home_cover_probability", "v4_away_cover_probability",
        "v4_over_probability", "v4_under_probability",
        "v4_live_spread_line_training_sign", "v4_live_total_line",
        "v4_market_fair_home_win_probability",
        "v4_market_fair_home_cover_probability",
        "v4_market_fair_over_probability",
        "v4_home_ml_fair_edge", "v4_away_ml_fair_edge",
        "v4_home_spread_fair_edge", "v4_away_spread_fair_edge",
        "v4_over_fair_edge", "v4_under_fair_edge",
        "v4_home_ml_quoted_ev", "v4_away_ml_quoted_ev",
        "v4_home_spread_quoted_ev", "v4_away_spread_quoted_ev",
        "v4_over_quoted_ev", "v4_under_quoted_ev",
        "v4_odds_capture_source", "v4_odds_capture_fetched_at",
    ]
    missing = [c for c in compact_columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Shadow output missing compact audit columns: {missing}")
    df[compact_columns].to_csv(compact_path, index=False, encoding="utf-8-sig")

    summary = {
        "season": 2026,
        "week": 1,
        "shadow_only": True,
        "games": len(df),
        "moneyline": market_summary(df, "HOME", "AWAY", "v4_home_ml_fair_edge", "v4_away_ml_fair_edge"),
        "spread": market_summary(df, "HOME", "AWAY", "v4_home_spread_fair_edge", "v4_away_spread_fair_edge"),
        "total": market_summary(df, "OVER", "UNDER", "v4_over_fair_edge", "v4_under_fair_edge"),
        "warning": "Forward-validation snapshot only. No outcomes have occurred, so this is not evidence of realized betting profit.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


shadow.load_week1_helper = load_week1_helper_fixed

if __name__ == "__main__":
    rc = shadow.main()
    if rc != 0:
        raise SystemExit(rc)
    write_compact_audit()
    raise SystemExit(0)

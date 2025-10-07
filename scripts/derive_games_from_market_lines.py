#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Build data/raw/games.csv from market_lines.csv for situational factors.

from pathlib import Path
import pandas as pd

# Updated path: file lives at repo root
SRC = Path("market_lines.csv")
OUT = Path("data/raw/games.csv")

def main():
    if not SRC.exists():
        raise SystemExit(f"[derive] missing source file: {SRC}")

    df = pd.read_csv(SRC)
    df.columns = df.columns.str.lower().str.strip()

    # Required columns check
    required = ["year", "season_type", "week", "game_id", "kickoff_utc", "home_team", "away_team"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[derive] market_lines missing required cols: {missing}")

    # Deduplicate by game_id: keep earliest kickoff_utc
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["kickoff_utc"])

    df = df.sort_values(["game_id", "kickoff_utc"], ascending=[True, True])

    keep_cols = ["year", "season_type", "week", "game_id", "kickoff_utc", "home_team", "away_team"]
    base = df[keep_cols].drop_duplicates(subset=["game_id"], keep="first").copy()

    # Build target schema
    base["season"] = base["year"].astype(int)
    base["date"] = base["kickoff_utc"].dt.strftime("%Y-%m-%d")
    base["neutral_site"] = 0  # unknown from market_lines; assume campus site
    base["venue"] = pd.NA
    base["venue_lat"] = pd.NA
    base["venue_lon"] = pd.NA
    base["venue_timezone"] = pd.NA
    base["venue_altitude_m"] = pd.NA

    out = base[[
        "season", "season_type", "week", "game_id", "date",
        "home_team", "away_team", "neutral_site",
        "venue", "venue_lat", "venue_lon", "venue_timezone", "venue_altitude_m",
        "kickoff_utc"
    ]].copy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"[derive] wrote {OUT} with {len(out)} unique games")

if __name__ == "__main__":
    main()

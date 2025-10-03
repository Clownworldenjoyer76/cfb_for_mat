#!/usr/bin/env python3
# Minimal stadiums sync: no CLI args, plain ASCII, conservative pandas usage.

import os
import sys
import pandas as pd

STADIUMS_PATH = "data/reference/stadiums.csv"
SUMMARY_PATH = "summaries/sync_stadiums_summary.txt"

REQUIRED_COLS = [
    "team", "venue", "city", "state", "country",
    "lat", "lon", "timezone", "altitude_m",
    "is_neutral_site", "notes",
]

DEFAULTS = {
    "country": "USA",
    "is_neutral_site": 0,
}

def ensure_columns(df):
    out = df.copy()
    for c in REQUIRED_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    # put required columns first, keep any extras afterward
    extras = [c for c in out.columns if c not in REQUIRED_COLS]
    out = out[REQUIRED_COLS + extras]
    for k, v in DEFAULTS.items():
        if k in out.columns:
            out[k] = out[k].fillna(v)
    return out

def normalize_types(df):
    out = df.copy()
    for c in ["lat", "lon", "altitude_m"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "is_neutral_site" in out.columns:
        out["is_neutral_site"] = pd.to_numeric(out["is_neutral_site"], errors="coerce").fillna(0).astype(int)
    return out

def dedupe_by_team_keep_first(df):
    out = df.copy()
    if "team" not in out.columns:
        return out
    # stable sort by lowercase team, then groupby-first
    out["_k"] = out["team"].astype(str).str.lower()
    out.sort_values("_k", kind="mergesort", inplace=True)
    out.drop(columns="_k", inplace=True)
    out = out.groupby("team", as_index=False).first()
    return out

def write_summary(before_rows, after_rows, df):
    try:
        os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
        miss = df[["lat", "lon", "timezone", "altitude_m"]].isna().sum().to_dict()
        lines = []
        lines.append("Sync Diagnostics")
        lines.append("File: " + STADIUMS_PATH)
        lines.append("Rows before: %d" % before_rows)
        lines.append("Rows after: %d" % after_rows)
        lines.append("")
        lines.append("Missing counts:")
        for k in ["lat", "lon", "timezone", "altitude_m"]:
            lines.append("  %s: %d" % (k, int(miss.get(k, 0))))
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        # Do not fail the run on summary write issues.
        print("[sync] WARNING: could not write summary: %s" % e, file=sys.stderr)

def main():
    if not os.path.exists(STADIUMS_PATH):
        print("[sync] ERROR: file not found: %s" % STADIUMS_PATH, file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(STADIUMS_PATH)
    before = len(df)

    df = ensure_columns(df)
    df = dedupe_by_team_keep_first(df)
    df = normalize_types(df)

    os.makedirs(os.path.dirname(STADIUMS_PATH), exist_ok=True)
    df.to_csv(STADIUMS_PATH, index=False)

    after = len(df)
    write_summary(before, after, df)

    print("[sync] OK")
    print("[sync] rows before: %d, after: %d" % (before, after))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[sync] ERROR: %s" % e, file=sys.stderr)
        raise

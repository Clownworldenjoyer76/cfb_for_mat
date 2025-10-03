#!/usr/bin/env python3
# Fill missing lat/lon/timezone/altitude_m in a stadiums CSV using CFBD Venues.
# Plain ASCII, preserves existing non-null values, creates required helper columns before use.

import argparse
import os
import sys
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from rapidfuzz import fuzz, process

VENUES_URL = "https://api.collegefootballdata.com/venues"

REQ_COLS = [
    "team", "venue", "city", "state", "country",
    "lat", "lon", "timezone", "altitude_m",
    "is_neutral_site", "notes",
]

def norm(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def feet_to_meters(x):
    try:
        return round(float(x) * 0.3048, 1)
    except Exception:
        return None

def pull_cfbd(api_key):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.get(VENUES_URL, headers=headers, timeout=60)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js)

    for c in ["name", "city", "state", "country", "timezone", "latitude", "longitude", "elevation"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["name_n"] = df["name"].map(norm)
    df["city_n"] = df["city"].map(norm)
    df["state_n"] = df["state"].map(norm)
    return df

def fill_missing_fields(row, src):
    def maybe_set(k_row, k_src, transform=None):
        cur = row.get(k_row, pd.NA)
        is_blank = pd.isna(cur) or (isinstance(cur, str) and cur.strip() == "")
        if is_blank:
            val = src.get(k_src, None)
            if transform:
                val = transform(val)
            row[k_row] = val
    maybe_set("lat", "latitude", float)
    maybe_set("lon", "longitude", float)
    maybe_set("timezone", "timezone", str)
    maybe_set("altitude_m", "elevation", feet_to_meters)
    maybe_set("city", "city", str)
    maybe_set("state", "state", str)
    maybe_set("country", "country", str)
    return row

def fuzzy_pick(row, cfbd, min_fuzzy):
    target = norm(row.get("venue", ""))
    if not target:
        return None
    choices = {i: cfbd.at[i, "name_n"] for i in cfbd.index}
    best = process.extractOne(target, choices, scorer=fuzz.WRatio)
    if not best:
        return None
    label, score, idx = best
    return idx if score >= min_fuzzy else None

def main():
    ap = argparse.ArgumentParser(description="Fill missing stadium metadata from CFBD venues")
    ap.add_argument("--stadiums", type=Path, required=True, help="Path to data/reference/stadiums.csv")
    ap.add_argument("--min-fuzzy", type=int, default=90, help="Minimum fuzzy score for venue name match (0-100)")
    args = ap.parse_args()

    if not args.stadiums.exists():
        print("[fill] ERROR: stadiums file not found: %s" % args.stadiums, file=sys.stderr)
        sys.exit(1)

    stad = pd.read_csv(args.stadiums)

    for c in REQ_COLS:
        if c not in stad.columns:
            stad[c] = pd.NA

    stad["venue_n"] = stad["venue"].map(norm)
    stad["city_n"] = stad["city"].map(norm)
    stad["state_n"] = stad["state"].map(norm)

    api_key = os.getenv("CFBD_API_KEY", "")
    cfbd = pull_cfbd(api_key)

    needs_mask = stad[["lat", "lon", "timezone", "altitude_m"]].isna().any(axis=1)
    if not needs_mask.any():
        # Drop helper columns before exiting
        for helper in ["venue_n", "city_n", "state_n"]:
            if helper in stad.columns:
                stad.drop(columns=[helper], inplace=True)
        stad.to_csv(args.stadiums, index=False)
        print("[fill] Nothing to fill; all rows complete.")
        return

    needs = stad.loc[needs_mask].copy()

    exact = needs.merge(
        cfbd,
        left_on="venue_n",
        right_on="name_n",
        how="left",
        suffixes=("", "_cfbd"),
        indicator=True
    )

    still_mask = exact["latitude"].isna()
    if still_mask.any():
        to_loc = exact.loc[still_mask, ["team", "venue", "city", "state", "city_n", "state_n"]].copy()
        by_loc = to_loc.merge(
            cfbd,
            on=["city_n", "state_n"],
            how="left",
            suffixes=("", "_loc"),
        )
        for col in ["latitude", "longitude", "elevation", "timezone", "city", "state", "country"]:
            exact.loc[still_mask, col] = exact.loc[still_mask, col].fillna(by_loc[col].values)

    remain_idx = exact.index[exact["latitude"].isna()].tolist()
    for i in remain_idx:
        row = exact.loc[i]
        pick = fuzzy_pick(row, cfbd, args.min_fuzzy)
        if pick is not None:
            v = cfbd.loc[pick]
            for col in ["latitude", "longitude", "elevation", "timezone", "city", "state", "country"]:
                if pd.isna(exact.at[i, col]):
                    exact.at[i, col] = v.get(col)

    updated = stad.copy().set_index("team")
    exact = exact.set_index("team")
    common = updated.index.intersection(exact.index)

    for t in common:
        row = updated.loc[t]
        src = exact.loc[t].to_dict()
        row = fill_missing_fields(row, src)
        updated.loc[t] = row

    updated = updated.reset_index(drop=False)

    for c in ["lat", "lon", "altitude_m"]:
        updated[c] = pd.to_numeric(updated[c], errors="coerce")
    if "is_neutral_site" in updated.columns:
        updated["is_neutral_site"] = pd.to_numeric(updated["is_neutral_site"], errors="coerce").fillna(0).astype(int)

    for helper in ["venue_n", "city_n", "state_n", "name_n"]:
        if helper in updated.columns:
            updated.drop(columns=[helper], inplace=True)

    updated.to_csv(args.stadiums, index=False)

    print("[fill] OK")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[fill] ERROR: %s" % e, file=sys.stderr)
        raise

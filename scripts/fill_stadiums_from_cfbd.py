#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Fill missing lat/lon/timezone/altitude_m in a stadiums CSV using CFBD Venues.
# Plain ASCII only. Uses repo columns: lat/lon/timezone/altitude_m.
# CFBD columns may be latitude/longitude/elevation/timezone; we normalize/guard.

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
from timezonefinder import TimezoneFinder

VENUES_URL = "https://api.collegefootballdata.com/venues"

REQ_COLS = [
    "team", "venue", "city", "state", "country",
    "lat", "lon", "timezone", "altitude_m",
    "is_neutral_site", "notes",
]

VENUE_STOPWORDS = [
    "stadium", "field", "memorial", "arena", "coliseum",
    "the", "of", "and"
]

# Known bad placeholder coordinates
BAD_COORDS = {(39.474686, -87.366960)}
DUP_COORD_THRESHOLD = 25


def norm(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for w in VENUE_STOPWORDS:
        s = re.sub(r"\b" + re.escape(w) + r"\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def feet_to_meters(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return round(float(x) * 0.3048, 1)
    except Exception:
        return None


def pull_cfbd(api_key):
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    r = requests.get(VENUES_URL, headers=headers, timeout=60)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js)

    # Ensure expected CFBD columns exist (some responses omit them)
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
        if not is_blank:
            return
        val = src.get(k_src, None)
        # only accept scalar-like values
        if isinstance(val, (dict, list, tuple, pd.Series)):
            return
        if transform:
            try:
                val = transform(val)
            except Exception:
                val = None
        row[k_row] = val

    # map CFBD -> repo columns
    maybe_set("lat", "latitude", float)
    maybe_set("lon", "longitude", float)
    maybe_set("timezone", "timezone", None)  # leave None as missing (do not coerce to "None")
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


def backfill_timezone_from_latlon(df):
    tf = TimezoneFinder()
    mask = df["timezone"].isna() & df["lat"].notna() & df["lon"].notna()
    if not mask.any():
        return df
    for i in df.index[mask]:
        try:
            tz = tf.timezone_at(lng=float(df.at[i, "lon"]), lat=float(df.at[i, "lat"]))
            if tz is None:
                tz = tf.closest_timezone_at(lng=float(df.at[i, "lon"]), lat=float(df.at[i, "lat"]))
            if tz:
                df.at[i, "timezone"] = tz
        except Exception:
            pass
    return df


def first_notna(series):
    for v in series:
        if pd.notna(v):
            return v
    return pd.NA


def apply_manual_overrides(df):
    # Western Kentucky / Houchens-Smith Stadium / Cape Girardeau, MO -> 371 ft
    target_m = feet_to_meters(371)
    mask = (
        df["team"].astype(str).str.strip().str.lower().eq("western kentucky") &
        df["venue"].astype(str).str.strip().str.lower().eq("houchens-smith stadium") &
        df["city"].astype(str).str.strip().str.lower().eq("cape girardeau") &
        df["state"].astype(str).str.strip().str.upper().eq("MO")
    )
    if mask.any():
        df.loc[mask, "altitude_m"] = target_m
    return df


def blank_placeholder_and_dupe_coords(df):
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Blank known bad coordinates
    if BAD_COORDS:
        bad_mask = False
        for (blat, blon) in BAD_COORDS:
            this_bad = (df["lat"] == blat) & (df["lon"] == blon)
            bad_mask = this_bad if isinstance(bad_mask, bool) else (bad_mask | this_bad)
        if not isinstance(bad_mask, bool):
            df.loc[bad_mask, ["lat", "lon"]] = pd.NA

    # Blank overly-common duplicate coordinates
    grp = df.groupby(["lat", "lon"], dropna=True).size().reset_index(name="n")
    if not grp.empty:
        too_common = grp[grp["n"] > DUP_COORD_THRESHOLD][["lat", "lon"]]
        if not too_common.empty:
            common_index = set(zip(too_common["lat"], too_common["lon"]))
            mask = df.apply(
                lambda r: (
                    pd.notna(r["lat"])
                    and pd.notna(r["lon"])
                    and (float(r["lat"]), float(r["lon"])) in common_index
                ),
                axis=1,
            )
            if mask.any():
                df.loc[mask, ["lat", "lon"]] = pd.NA
    return df


def ensure_cfbd_cols_present(exact_df):
    """
    After merges, make sure the CFBD columns we rely on exist with base names.
    If only *_cfbd exists, rename it back; otherwise create as NA.
    """
    for base in ["latitude", "longitude", "elevation", "timezone", "city", "state", "country"]:
        cf = base + "_cfbd"
        if base not in exact_df.columns:
            if cf in exact_df.columns:
                exact_df.rename(columns={cf: base}, inplace=True)
            else:
                exact_df[base] = pd.NA
    return exact_df


def main():
    ap = argparse.ArgumentParser(description="Fill missing stadium metadata from CFBD venues")
    ap.add_argument("--stadiums", type=Path, required=True, help="Path to data/reference/stadiums.csv")
    ap.add_argument("--min-fuzzy", type=int, default=80, help="Minimum fuzzy score for venue name match (0-100)")
    args = ap.parse_args()

    if not args.stadiums.exists():
        print("[fill] ERROR: stadiums file not found: " + str(args.stadiums), file=sys.stderr)
        sys.exit(1)

    stad = pd.read_csv(args.stadiums)
    for c in REQ_COLS:
        if c not in stad.columns:
            stad[c] = pd.NA

    # Our repo uses lat/lon — keep those canonical
    stad.rename(columns={"latitude": "lat", "longitude": "lon"}, inplace=True)

    # Clean placeholders before any fill
    stad = blank_placeholder_and_dupe_coords(stad)

    # Helper columns BEFORE any merge
    stad["venue_n"] = stad["venue"].map(norm)
    stad["city_n"] = stad["city"].map(norm)
    stad["state_n"] = stad["state"].map(norm)

    api_key = os.getenv("CFBD_API_KEY", "")
    cfbd = pull_cfbd(api_key)

    needs_mask = stad[["lat", "lon", "timezone", "altitude_m"]].isna().any(axis=1)
    needs = stad.loc[needs_mask].copy()

    if needs.empty():
        stad = apply_manual_overrides(stad)
        stad = blank_placeholder_and_dupe_coords(stad)
        stad = backfill_timezone_from_latlon(stad)
        for helper in ["venue_n", "city_n", "state_n"]:
            if helper in stad.columns:
                stad.drop(columns=[helper], inplace=True)
        stad.to_csv(args.stadiums, index=False)
        print("[fill] Nothing to fill; applied overrides, cleaned coords, and tz fallback.")
        return

    # 1) Exact normalized venue name join (bring CFBD columns into 'exact')
    exact = needs.merge(
        cfbd,
        left_on="venue_n",
        right_on="name_n",
        how="left",
        suffixes=("", "_cfbd"),
        indicator=True,
    )
    exact = ensure_cfbd_cols_present(exact)

    # 2) City+state join (dedup CFBD to 1 row per location)
    still_mask = exact["latitude"].isna()
    if still_mask.any():
        cols_cs = ["latitude", "longitude", "elevation", "timezone", "city", "state", "country"]
        agg_map = {c: first_notna for c in cols_cs}
        cfbd_loc = cfbd.groupby(["city_n", "state_n"], as_index=False).agg(agg_map)

        missing = exact.loc[still_mask].copy()
        missing["__idx__"] = missing.index
        by_loc = missing.merge(cfbd_loc, on=["city_n", "state_n"], how="left").set_index("__idx__")
        by_loc = ensure_cfbd_cols_present(by_loc)

        for col in cols_cs:
            exact.loc[by_loc.index, col] = exact.loc[by_loc.index, col].fillna(by_loc[col])

    # 3) Fuzzy fallback on venue name
    remain_idx = exact.index[exact["latitude"].isna()].tolist()
    for i in remain_idx:
        row = exact.loc[i]
        pick = fuzzy_pick(row, cfbd, args.min_fuzzy)
        if pick is not None:
            v = cfbd.loc[pick]
            for col in ["latitude", "longitude", "elevation", "timezone", "city", "state", "country"]:
                if pd.isna(exact.at[i, col]):
                    exact.at[i, col] = v.get(col)

    # Collapse to one row per team (first non-null values from CFBD)
    cols_from_cfbd = ["latitude", "longitude", "elevation", "timezone", "city", "state", "country"]
    best = exact.groupby("team", as_index=True).agg({c: first_notna for c in cols_from_cfbd})

    # Apply fills back into main DataFrame using repo column names
    updated = stad.copy().set_index("team")
    for t in best.index:
        if t in updated.index:
            src = best.loc[t].to_dict()
            row = updated.loc[t]
            row = fill_missing_fields(row, src)  # maps latitude->lat, longitude->lon, elevation->altitude_m
            updated.loc[t] = row
    updated = updated.reset_index(drop=False)

    # Normalize numeric types
    for c in ["lat", "lon", "altitude_m"]:
        updated[c] = pd.to_numeric(updated[c], errors="coerce")
    if "is_neutral_site" in updated.columns:
        updated["is_neutral_site"] = pd.to_numeric(updated["is_neutral_site"], errors="coerce").fillna(0).astype(int)

    # Manual overrides and cleanup
    updated = apply_manual_overrides(updated)
    updated = blank_placeholder_and_dupe_coords(updated)

    # Final timezone fallback at the very end
    updated = backfill_timezone_from_latlon(updated)

    # Drop helper columns and save
    for helper in ["venue_n", "city_n", "state_n", "name_n"]:
        if helper in updated.columns:
            updated.drop(columns=[helper], inplace=True)

    updated.to_csv(args.stadiums, index=False)
    print("[fill] OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[fill] ERROR: " + str(e), file=sys.stderr)
        raise

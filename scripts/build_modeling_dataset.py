#!/usr/bin/env python3
"""
Build modeling_dataset.csv with situational factors.

Inputs (best-effort, defensive):
- data/game_scores_clean.csv  (preferred)
  Expected columns (any subset works; script is defensive):
    game_id, date|game_date|kickoff_datetime, team, opponent,
    home_team, away_team, neutral_site|is_neutral|site
    (plus any other stats you already compute)

- data/reference/stadiums.csv (required for situational features)
  Columns (must exist):
    team, venue, city, state, country, lat, lon, timezone, altitude_m, is_neutral_site, notes

Output:
- data/modeling_dataset.csv (input rows + situational columns)
- logs_build_modeling_dataset.txt (summary/acceptance logs)

Situational columns created:
  is_home, is_away, is_neutral
  rest_days, had_bye, short_week
  travel_km, long_trip_flag
  tz_diff_from_home, east_to_west_flag, west_to_east_flag, body_clock_kickoff_hour
  altitude_game
"""

import os
import sys
import math
import json
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

# --------------------------
# Config (env-overridable)
# --------------------------
BYE_THRESHOLD_DAYS = int(os.environ.get("BYE_THRESHOLD_DAYS", "13"))
SHORT_WEEK_MAX_DAYS = int(os.environ.get("SHORT_WEEK_MAX_DAYS", "6"))
LONG_TRIP_KM = float(os.environ.get("LONG_TRIP_KM", "1500"))

DATA_DIR = "data"
SCORES_PATHS = [
    os.path.join(DATA_DIR, "game_scores_clean.csv"),
    os.path.join(DATA_DIR, "scores", "game_scores_clean.csv"),
]
STADIUMS_PATH = os.path.join(DATA_DIR, "reference", "stadiums.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "modeling_dataset.csv")
LOG_PATH = "logs_build_modeling_dataset.txt"

# --------------------------
# Utilities
# --------------------------
def log(msg: str):
    ts = datetime.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S%z")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg, flush=True)

def read_csv_best_effort(path):
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except Exception:
            return None

def parse_datetime_best_effort(series):
    for col in ["kickoff_datetime", "date", "game_date", "datetime", "start_time"]:
        if col in series.columns:
            try:
                return pd.to_datetime(series[col], utc=True, errors="coerce")
            except Exception:
                pass
    # If there's a season/week and no date, we cannot compute rest/travel chronologically
    return pd.Series([pd.NaT] * len(series))

def coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    try:
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return float("nan")
        R = 6371.0088
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = phi2 - phi1
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    except Exception:
        return float("nan")

def tz_offset_hours(tz_name, dt_utc):
    try:
        tz = ZoneInfo(str(tz_name))
        return (dt_utc.astimezone(tz).utcoffset().total_seconds()) / 3600.0
    except Exception:
        return float("nan")

def safe_lower(s):
    try:
        return str(s).strip().lower()
    except Exception:
        return ""

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

# --------------------------
# Load base data
# --------------------------
def load_scores():
    for p in SCORES_PATHS:
        if os.path.exists(p):
            df = read_csv_best_effort(p)
            if df is not None and not df.empty:
                log(f"Loaded scores: {p} ({len(df)} rows)")
                return df
    log("ERROR: No game scores file found.")
    return pd.DataFrame()

def load_stadiums():
    if not os.path.exists(STADIUMS_PATH):
        log(f"WARNING: Stadiums file missing: {STADIUMS_PATH}")
        return pd.DataFrame()
    df = read_csv_best_effort(STADIUMS_PATH)
    if df is None:
        log(f"ERROR: Failed to read {STADIUMS_PATH}")
        return pd.DataFrame()
    # Normalize required columns
    req = ["team","venue","city","state","country","lat","lon","timezone","altitude_m","is_neutral_site","notes"]
    for c in req:
        if c not in df.columns:
            df[c] = pd.NA
    # Normalize types
    df["__team_key"] = df["team"].astype(str).map(safe_lower)
    df["is_neutral_site"] = pd.to_numeric(df["is_neutral_site"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["altitude_m"] = pd.to_numeric(df["altitude_m"], errors="coerce")
    return df

# --------------------------
# Derive situational features
# --------------------------
def build_modeling_dataset():
    # Reset log
    open(LOG_PATH, "w").close()

    scores = load_scores()
    if scores.empty:
        log("FATAL: No input scores found; cannot build modeling dataset.")
        # still create an empty modeling_dataset.csv to keep CI stable
        pd.DataFrame().to_csv(OUTPUT_PATH, index=False)
        sys.exit(1)

    stadiums = load_stadiums()
    if stadiums.empty:
        log("WARNING: Stadiums missing or empty; situational features will be mostly null.")

    # Normalize column names
    scores.columns = [c.strip() for c in scores.columns]
    lc = {c: c.lower() for c in scores.columns}
    scores.rename(columns=lc, inplace=True)

    # Create normalized keys
    for k in ["team","opponent","home_team","away_team"]:
        if k in scores.columns:
            scores[f"__{k}_key"] = scores[k].astype(str).map(safe_lower)
    else:
        # Guarantee at least 'team' key exists for per-team records
        if "team" not in scores.columns:
            # Attempt to widen from home/away perspective into per-team rows
            if "home_team" in scores.columns and "away_team" in scores.columns:
                # Explode to team rows
                home_df = scores.copy()
                home_df["team"] = home_df["home_team"]
                home_df["opponent"] = home_df["away_team"]
                away_df = scores.copy()
                away_df["team"] = away_df["away_team"]
                away_df["opponent"] = away_df["home_team"]
                scores = pd.concat([home_df, away_df], ignore_index=True)
            else:
                log("ERROR: Could not identify team columns; situational features may be skipped.")
                scores["team"] = pd.NA
        scores["__team_key"] = scores["team"].astype(str).map(safe_lower)

    # Date/time
    scores["__kickoff_utc"] = parse_datetime_best_effort(scores)
    if scores["__kickoff_utc"].isna().all():
        log("WARNING: No valid datetime found; rest/travel will be null.")

    # Determine site flags (best-effort)
    # Prefer explicit neutral flags if present
    neutral_cols = [c for c in ["neutral_site","is_neutral","neutral","site"] if c in scores.columns]
    if neutral_cols:
        neutral_series = scores[neutral_cols[0]].astype(str).map(lambda s: 1 if s.strip().lower() in {"true","1","yes","neutral"} else 0)
        scores["is_neutral"] = pd.to_numeric(neutral_series, errors="coerce").fillna(0).astype(int)
    else:
        scores["is_neutral"] = 0

    # If home/away columns exist, use them; else infer by team/opponent comparison
    if "home_team" in scores.columns and "away_team" in scores.columns:
        scores["is_home"] = (scores["team"].astype(str).str.strip().str.lower() == scores["home_team"].astype(str).str.strip().str.lower()).astype(int)
        scores["is_away"] = (scores["team"].astype(str).str.strip().str.lower() == scores["away_team"].astype(str).str.strip().str.lower()).astype(int)
    else:
        # No explicit home/away; assume non-neutral rows with known home venue are 'home' if opponent venue differs
        scores["is_home"] = ((scores["is_neutral"] == 0) & scores["team"].notna()).astype(int)
        scores["is_away"] = 0

    # Enforce one-hot with neutral:
    # If neutral => both 0. If both 0 and not neutral -> fallback set home=1
    scores.loc[scores["is_neutral"] == 1, ["is_home","is_away"]] = 0
    neither_mask = (scores["is_neutral"] == 0) & (scores["is_home"] == 0) & (scores["is_away"] == 0)
    scores.loc[neither_mask, "is_home"] = 1

    # Merge home stadium for each team
    if not stadiums.empty:
        team_home = stadiums[stadiums["is_neutral_site"].fillna(0) != 1].copy()
        team_home = team_home[["__team_key","venue","lat","lon","timezone","altitude_m"]].rename(
            columns={"venue":"home_venue","lat":"home_lat","lon":"home_lon","timezone":"home_tz","altitude_m":"home_alt_m"}
        )
        scores = scores.merge(team_home, left_on="__team_key", right_on="__team_key", how="left")

        # Opponent home stadium (for away games)
        if "__opponent_key" not in scores.columns and "opponent" in scores.columns:
            scores["__opponent_key"] = scores["opponent"].astype(str).map(safe_lower)
        if "__opponent_key" in scores.columns:
            opp_home = team_home.copy()
            opp_home = opp_home.rename(columns={
                "home_venue":"opp_home_venue",
                "home_lat":"opp_home_lat",
                "home_lon":"opp_home_lon",
                "home_tz":"opp_home_tz",
                "home_alt_m":"opp_home_alt_m"
            })
            scores = scores.merge(opp_home, left_on="__opponent_key", right_on="__team_key", how="left", suffixes=("","_drop"))
            if "__team_key_drop" in scores.columns:
                scores.drop(columns=["__team_key_drop"], inplace=True, errors="ignore")

    # Determine game venue lat/lon/tz/alt according to site
    def pick_venue(row):
        if row.get("is_neutral", 0) == 1:
            # Try to use a provided venue if present and flagged neutral in stadiums (not always available)
            # Otherwise, fall back to NaN (still allows travel calc if prev known)
            v = {"lat": pd.NA, "lon": pd.NA, "tz": pd.NA, "alt": pd.NA}
        elif row.get("is_home", 0) == 1:
            v = {"lat": row.get("home_lat", pd.NA), "lon": row.get("home_lon", pd.NA),
                 "tz": row.get("home_tz", pd.NA), "alt": row.get("home_alt_m", pd.NA)}
        else:  # away
            v = {"lat": row.get("opp_home_lat", pd.NA), "lon": row.get("opp_home_lon", pd.NA),
                 "tz": row.get("opp_home_tz", pd.NA), "alt": row.get("opp_home_alt_m", pd.NA)}
        return pd.Series([v["lat"], v["lon"], v["tz"], v["alt"]], index=["venue_lat","venue_lon","venue_tz","altitude_game"])

    venue_info = scores.apply(pick_venue, axis=1)
    scores = pd.concat([scores, venue_info], axis=1)

    # Compute rest_days / bye
    scores = scores.sort_values(by=["__team_key","__kickoff_utc"], kind="mergesort")
    scores["__prev_kickoff"] = scores.groupby("__team_key")["__kickoff_utc"].shift(1)
    scores["rest_days"] = ((scores["__kickoff_utc"] - scores["__prev_kickoff"]).dt.total_seconds() / (3600*24)).round().astype("Float64")
    scores["had_bye"] = (scores["rest_days"] >= BYE_THRESHOLD_DAYS).astype("Int64")
    scores["short_week"] = (scores["rest_days"] <= SHORT_WEEK_MAX_DAYS).astype("Int64")

    # Travel distance: need previous venue coords per team
    scores["__prev_lat"] = scores.groupby("__team_key")["venue_lat"].shift(1)
    scores["__prev_lon"] = scores.groupby("__team_key")["venue_lon"].shift(1)
    scores["travel_km"] = scores.apply(
        lambda r: haversine_km(r["__prev_lat"], r["__prev_lon"], r["venue_lat"], r["venue_lon"]), axis=1
    ).astype("Float64")
    scores["long_trip_flag"] = (scores["travel_km"] >= LONG_TRIP_KM).astype("Int64")

    # Time zone features
    # tz_diff_from_home: offset(current venue tz) - offset(home tz)
    def tz_diff_row(r):
        dt = r["__kickoff_utc"]
        if pd.isna(dt):
            return float("nan")
        home_tz = r.get("home_tz", pd.NA)
        venue_tz = r.get("venue_tz", pd.NA)
        if pd.isna(home_tz) or pd.isna(venue_tz):
            return float("nan")
        try:
            home_off = tz_offset_hours(home_tz, dt)
            ven_off = tz_offset_hours(venue_tz, dt)
            return ven_off - home_off
        except Exception:
            return float("nan")

    scores["tz_diff_from_home"] = scores.apply(tz_diff_row, axis=1).astype("Float64")
    scores["east_to_west_flag"] = (scores["tz_diff_from_home"] < 0).astype("Int64")
    scores["west_to_east_flag"] = (scores["tz_diff_from_home"] > 0).astype("Int64")

    # Body-clock kickoff hour (home tz), if we have a kickoff time
    def body_clock_hour(r):
        dt = r["__kickoff_utc"]
        home_tz = r.get("home_tz", pd.NA)
        if pd.isna(dt) or pd.isna(home_tz):
            return pd.NA
        try:
            local = dt.astimezone(ZoneInfo(str(home_tz)))
            return int(local.hour)
        except Exception:
            return pd.NA

    scores["body_clock_kickoff_hour"] = scores.apply(body_clock_hour, axis=1).astype("Int64")

    # Ensure output columns
    out_cols = [
        # Original identifiers if present
        *[c for c in ["game_id","season","week","team","opponent","home_team","away_team"] if c in scores.columns],
        # Situational
        "is_home","is_away","is_neutral",
        "rest_days","had_bye","short_week",
        "travel_km","long_trip_flag",
        "tz_diff_from_home","east_to_west_flag","west_to_east_flag","body_clock_kickoff_hour",
        "altitude_game",
    ]
    scores = ensure_cols(scores, out_cols)

    # Write output
    scores[out_cols].to_csv(OUTPUT_PATH, index=False)

    # --------------------------
    # Logging & acceptance
    # --------------------------
    total = len(scores)
    h = int(scores["is_home"].fillna(0).sum())
    a = int(scores["is_away"].fillna(0).sum())
    n = int(scores["is_neutral"].fillna(0).sum())

    log(f"Rows: {total}")
    log(f"Home/Away/Neutral counts: H={h} A={a} N={n}")

    # Summaries
    for col in ["rest_days","travel_km","tz_diff_from_home","body_clock_kickoff_hour"]:
        s = scores[col]
        log(f"{col}: mean={s.mean(skipna=True):.2f} median={s.median(skipna=True):.2f} nonnull={s.notna().sum()}")

    # Missing venue coords for applicable rows
    applicable = scores["is_neutral"] == 0
    miss_coords = (scores["venue_lat"].isna() | scores["venue_lon"].isna()) & applicable
    miss_pct = 100.0 * miss_coords.sum() / max(1, applicable.sum())
    log(f"Missing venue coords on applicable rows: {miss_coords.sum()} ({miss_pct:.1f}%)")

    # Acceptance checks (log only; workflow will enforce)
    one_hot_ok = int(((scores["is_home"].fillna(0) + scores["is_away"].fillna(0) + scores["is_neutral"].fillna(0)) == 1).sum())
    log(f"One-hot site flag rows OK: {one_hot_ok}/{total}")

    log("Build complete.")

if __name__ == "__main__":
    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        build_modeling_dataset()
    except SystemExit as e:
        raise
    except Exception as e:
        log("FATAL exception during build:\n" + traceback.format_exc())
        # Write an empty file to keep downstream steps predictable
        try:
            pd.DataFrame().to_csv(OUTPUT_PATH, index=False)
        except Exception:
            pass
        sys.exit(2)

#!/usr/bin/env python3
"""
Build modeling_dataset.csv with situational factors.

Inputs:
- data/game_scores_clean.csv  (or data/scores/game_scores_clean.csv)
- data/reference/stadiums.csv (required for situational features)

Outputs:
- data/modeling_dataset.csv
- logs_build_modeling_dataset.txt
"""

import os, sys, math, traceback
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

# ---------- Config (env-overridable) ----------
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

# ---------- Utils ----------
def log(msg: str):
    ts = datetime.now(tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S%z")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg, flush=True)

def read_csv_best(path):
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except Exception:
            return None

def parse_kickoff(df: pd.DataFrame):
    for c in ["kickoff_datetime","date","game_date","datetime","start_time"]:
        if c in df.columns:
            try:
                return pd.to_datetime(df[c], utc=True, errors="coerce")
            except Exception:
                pass
    return pd.Series([pd.NaT]*len(df))

def k(s): 
    try: return str(s).strip().lower()
    except: return ""

def haversine_km(lat1, lon1, lat2, lon2):
    try:
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return float("nan")
        R = 6371.0088
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = p2 - p1
        dl = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R*c
    except Exception:
        return float("nan")

def tz_offset_hours(tz_name, dt_utc):
    try:
        tz = ZoneInfo(str(tz_name))
        off = dt_utc.astimezone(tz).utcoffset()
        return off.total_seconds()/3600.0 if off is not None else float("nan")
    except Exception:
        return float("nan")

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

# ---------- Load ----------
def load_scores():
    for p in SCORES_PATHS:
        if os.path.exists(p):
            df = read_csv_best(p)
            if df is not None and not df.empty:
                log(f"Loaded scores: {p} ({len(df)} rows)")
                return df
    log("ERROR: No scores file found.")
    return pd.DataFrame()

def load_stadiums():
    if not os.path.exists(STADIUMS_PATH):
        log(f"WARNING: Stadiums missing: {STADIUMS_PATH}")
        return pd.DataFrame()
    df = read_csv_best(STADIUMS_PATH)
    if df is None:
        log("ERROR: Stadiums read failed.")
        return pd.DataFrame()
    need = ["team","venue","city","state","country","lat","lon","timezone","altitude_m","is_neutral_site","notes"]
    for c in need:
        if c not in df.columns: df[c] = pd.NA
    df["__team_key"] = df["team"].astype(str).map(k)
    for c in ["lat","lon","altitude_m","is_neutral_site"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- Build ----------
def build():
    open(LOG_PATH, "w").close()

    scores = load_scores()
    if scores.empty:
        pd.DataFrame().to_csv(OUTPUT_PATH, index=False)
        sys.exit(1)

    stadiums = load_stadiums()

    # normalize
    scores.columns = [c.strip() for c in scores.columns]
    scores.rename(columns={c:c.lower() for c in scores.columns}, inplace=True)

    # explode to team rows if only home/away present
    if "team" not in scores.columns and {"home_team","away_team"}.issubset(scores.columns):
        home_df = scores.copy()
        home_df["team"] = home_df["home_team"]
        home_df["opponent"] = home_df["away_team"]
        away_df = scores.copy()
        away_df["team"] = away_df["away_team"]
        away_df["opponent"] = away_df["home_team"]
        scores = pd.concat([home_df, away_df], ignore_index=True)

    scores["__team_key"] = scores.get("team", pd.Series([pd.NA]*len(scores))).astype(str).map(k)
    if "opponent" in scores.columns:
        scores["__opponent_key"] = scores["opponent"].astype(str).map(k)

    scores["__kickoff_utc"] = parse_kickoff(scores)
    if scores["__kickoff_utc"].isna().all():
        log("WARNING: No valid kickoff datetime; rest/travel will be null.")

    # is_neutral
    neutral_cols = [c for c in ["neutral_site","is_neutral","neutral","site"] if c in scores.columns]
    if neutral_cols:
        scores["is_neutral"] = scores[neutral_cols[0]].astype(str).str.strip().str.lower().isin(
            ["true","1","yes","neutral"]
        ).astype(int)
    else:
        scores["is_neutral"] = 0

    # is_home/is_away
    if {"home_team","away_team","team"}.issubset(scores.columns):
        t = scores["team"].astype(str).str.strip().str.lower()
        scores["is_home"] = (t == scores["home_team"].astype(str).str.strip().str.lower()).astype(int)
        scores["is_away"] = (t == scores["away_team"].astype(str).str.strip().str.lower()).astype(int)
    else:
        scores["is_home"] = ((scores["is_neutral"]==0) & scores["__team_key"].ne("")).astype(int)
        scores["is_away"] = 0
    scores.loc[scores["is_neutral"]==1, ["is_home","is_away"]] = 0
    mask = (scores["is_neutral"]==0) & (scores["is_home"]==0) & (scores["is_away"]==0)
    scores.loc[mask, "is_home"] = 1

    # merge stadiums (home + opponent home)
    if not stadiums.empty:
        home = stadiums[stadiums["is_neutral_site"].fillna(0)!=1][
            ["__team_key","venue","lat","lon","timezone","altitude_m"]
        ].rename(columns={
            "venue":"home_venue","lat":"home_lat","lon":"home_lon",
            "timezone":"home_tz","altitude_m":"home_alt_m"
        })
        scores = scores.merge(home, on="__team_key", how="left")

        if "__opponent_key" in scores.columns:
            opp = home.rename(columns={
                "home_venue":"opp_home_venue","home_lat":"opp_home_lat","home_lon":"opp_home_lon",
                "home_tz":"opp_home_tz","home_alt_m":"opp_home_alt_m",
                "__team_key":"__opponent_key"
            })
            scores = scores.merge(opp, on="__opponent_key", how="left")

    # pick venue for this row
    def venue_row(r):
        if r["is_neutral"]==1:
            return pd.Series([pd.NA,pd.NA,pd.NA,pd.NA], index=["venue_lat","venue_lon","venue_tz","altitude_game"])
        if r["is_home"]==1:
            return pd.Series([r.get("home_lat"), r.get("home_lon"), r.get("home_tz"), r.get("home_alt_m")],
                             index=["venue_lat","venue_lon","venue_tz","altitude_game"])
        return pd.Series([r.get("opp_home_lat"), r.get("opp_home_lon"), r.get("opp_home_tz"), r.get("opp_home_alt_m")],
                         index=["venue_lat","venue_lon","venue_tz","altitude_game"])

    scores = pd.concat([scores, scores.apply(venue_row, axis=1)], axis=1)

    # rest/travel
    scores = scores.sort_values(["__team_key","__kickoff_utc"], kind="mergesort")
    scores["__prev_kickoff"] = scores.groupby("__team_key")["__kickoff_utc"].shift(1)
    scores["rest_days"] = ((scores["__kickoff_utc"] - scores["__prev_kickoff"]).dt.total_seconds()/(3600*24)).round().astype("Float64")
    scores["had_bye"] = (scores["rest_days"] >= BYE_THRESHOLD_DAYS).astype("Int64")
    scores["short_week"] = (scores["rest_days"] <= SHORT_WEEK_MAX_DAYS).astype("Int64")

    scores["__prev_lat"] = scores.groupby("__team_key")["venue_lat"].shift(1)
    scores["__prev_lon"] = scores.groupby("__team_key")["venue_lon"].shift(1)
    scores["travel_km"] = scores.apply(lambda r: haversine_km(r["__prev_lat"], r["__prev_lon"], r["venue_lat"], r["venue_lon"]), axis=1).astype("Float64")
    scores["long_trip_flag"] = (scores["travel_km"] >= LONG_TRIP_KM).astype("Int64")

    # tz features
    def tz_diff(r):
        dt = r["__kickoff_utc"]
        ht, vt = r.get("home_tz"), r.get("venue_tz")
        if pd.isna(dt) or pd.isna(ht) or pd.isna(vt): return float("nan")
        return tz_offset_hours(vt, dt) - tz_offset_hours(ht, dt)
    scores["tz_diff_from_home"] = scores.apply(tz_diff, axis=1).astype("Float64")
    scores["east_to_west_flag"] = (scores["tz_diff_from_home"] < 0).astype("Int64")
    scores["west_to_east_flag"] = (scores["tz_diff_from_home"] > 0).astype("Int64")

    def body_hour(r):
        dt = r["__kickoff_utc"]; ht = r.get("home_tz")
        if pd.isna(dt) or pd.isna(ht): return pd.NA
        try:
            return int(dt.astimezone(ZoneInfo(str(ht))).hour)
        except Exception:
            return pd.NA
    scores["body_clock_kickoff_hour"] = scores.apply(body_hour, axis=1).astype("Int64")

    # write
    out_cols = [c for c in ["game_id","season","week","team","opponent","home_team","away_team"] if c in scores.columns] + [
        "is_home","is_away","is_neutral","rest_days","had_bye","short_week",
        "travel_km","long_trip_flag","tz_diff_from_home","east_to_west_flag","west_to_east_flag",
        "body_clock_kickoff_hour","altitude_game"
    ]
    scores = ensure_cols(scores, out_cols)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    scores[out_cols].to_csv(OUTPUT_PATH, index=False)

    # logs + acceptance
    total = len(scores)
    h = int(scores["is_home"].fillna(0).sum())
    a = int(scores["is_away"].fillna(0).sum())
    n = int(scores["is_neutral"].fillna(0).sum())
    log(f"Rows: {total}")
    log(f"Home/Away/Neutral counts: H={h} A={a} N={n}")
    for col in ["rest_days","travel_km","tz_diff_from_home","body_clock_kickoff_hour"]:
        s = scores[col]; log(f"{col}: mean={s.mean(skipna=True):.2f} median={s.median(skipna=True):.2f} nonnull={s.notna().sum()}")
    applicable = scores["is_neutral"]==0
    miss = ((scores["venue_lat"].isna()) | (scores["venue_lon"].isna())) & applicable
    pct = 100.0 * (miss.sum()/max(1, applicable.sum()))
    log(f"Missing venue coords (applicable): {miss.sum()} rows ({pct:.1f}%)")
    ok = int(((scores["is_home"].fillna(0)+scores["is_away"].fillna(0)+scores["is_neutral"].fillna(0))==1).sum())
    log(f"One-hot OK rows: {ok}/{total}")
    log("Build complete.")

if __name__ == "__main__":
    try:
        build()
    except SystemExit:
        raise
    except Exception:
        try:
            log("FATAL:\n"+traceback.format_exc())
        except Exception:
            pass
        try:
            pd.DataFrame().to_csv(OUTPUT_PATH, index=False)
        except Exception:
            pass
        sys.exit(2)

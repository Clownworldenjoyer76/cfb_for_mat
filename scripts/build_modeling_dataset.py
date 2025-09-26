#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import unicodedata
import datetime as dt
from typing import List, Dict, Optional, Tuple

import pandas as pd

# =========================
# INPUTS (project root)
# =========================
REQ_SITUATIONAL = "situational_factors.csv"          # required (team-game rows with game_id)
OPT_WEATHER     = "weather_enriched.csv"             # optional (game-level; has game_id)
OPT_TEAM_STATS  = "team_stats.csv"                   # optional (season-team aggregates)
OPT_EFFICIENCY  = "team_efficiency.csv"              # optional (season-team aggregates)
OPT_SPECIAL     = "special_teams_metrics.csv"        # optional (season-team aggregates)
OPT_INJ_DAILY   = "injuries_daily.csv"               # optional (player-level; season-team)
OPT_INJ_LATEST  = "injuries_latest.csv"              # optional (player-level; season-team)
OPT_TEAM_ALIASES= "mappings/team_aliases.csv"        # optional (cfbd_name,alias)

# Optional: allow explicit scores override via env
ENV_SCORES_CSV  = os.environ.get("SCORES_CSV", "").strip()
ENV_SCORES_GLOB = os.environ.get("SCORES_GLOB", "").strip()

# =========================
# OUTPUTS
# =========================
OUT_DATASET_CSV = "data/modeling_dataset.csv"
OUT_LOG_TXT     = "logs_build_modeling_dataset.txt"

# =========================
# SEASON FILTER (defaults 2019–2024; overridable by env)
# =========================
def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    try:
        return int(v)
    except Exception:
        return default

SEASON_MIN = _env_int("SEASON_MIN", 2019)
SEASON_MAX = _env_int("SEASON_MAX", 2024)

# =========================
# Helpers
# =========================
def _exists(p: str) -> bool:
    return os.path.exists(p) and os.path.isfile(p)

def _read(p: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    if not _exists(p):
        return pd.DataFrame()
    try:
        return pd.read_csv(p, parse_dates=parse_dates or [])
    except Exception:
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()

def _find_col(df: pd.DataFrame, wants: List[str]) -> Optional[str]:
    if df.empty:
        return None
    lower = {c.lower(): c for c in df.columns}
    for w in wants:
        if w in lower:
            return lower[w]
    return None

def _require(cond: bool, msg: str):
    if not cond:
        print(f"INSUFFICIENT INFORMATION: {msg}", file=sys.stderr)
        sys.exit(2)

def _season_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, ["season","year"])

def _team_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, ["team","school","team_name"])

def _std_team(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
         .str.replace("\u00A0", " ", regex=False)
         .str.lower()
    )

def _norm_key_val(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    return "".join(ch for ch in s if ch.isalnum())

def _apply_alias(series: pd.Series, alias_map: Dict[str, str]) -> pd.Series:
    key = _std_team(series)
    norm_map = {_std_team(pd.Series([k]))[0]: v for k, v in alias_map.items()}
    return key.map(norm_map).fillna(series)

def _load_aliases(path: str) -> Dict[str, str]:
    if not _exists(path):
        return {}
    df = _read(path)
    cols = {c.lower() for c in df.columns}
    if "cfbd_name" not in cols or "alias" not in cols:
        return {}
    src = df[[[c for c in df.columns if c.lower()=="cfbd_name"][0],
              [c for c in df.columns if c.lower()=="alias"][0]]].dropna()
    d: Dict[str,str] = {}
    for _, r in src.iterrows():
        k = str(r.iloc[0]).strip()
        v = str(r.iloc[1]).strip()
        if k and v:
            d[k] = v
    return d

def _safe_left_merge(left: pd.DataFrame, right: pd.DataFrame, on: List[Tuple[str,str]], suffix_tag: str) -> pd.DataFrame:
    left_keys  = [lk for lk, _ in on]
    right_keys = [rk for _, rk in on]
    return left.merge(
        right,
        left_on=left_keys,
        right_on=right_keys,
        how="left",
        suffixes=("", suffix_tag)
    )

def _prefix_new_columns(df_before: pd.DataFrame, df_after: pd.DataFrame, prefix: str, protect: List[str]) -> pd.DataFrame:
    new_cols = [c for c in df_after.columns if c not in df_before.columns and c not in protect]
    return df_after.rename(columns={c: f"{prefix}{c}" for c in new_cols})

def _coerce_season_int(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df

def _dedupe_by_team_season(df: pd.DataFrame, s_col: str, t_key_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = [s_col, t_key_col]
    num_cols = df.select_dtypes(include=["number", "float", "int", "Int64"]).columns.tolist()
    num_cols = [c for c in num_cols if c not in group_cols]

    def _first(s: pd.Series):
        return s.iloc[0]

    agg = {}
    for c in df.columns:
        if c in group_cols:
            continue
        agg[c] = ("mean" if c in num_cols else _first)

    g = df.groupby(group_cols, dropna=False).agg(agg).reset_index()
    g = g.drop_duplicates(subset=group_cols, keep="first")
    return g

# ---------- Scores discovery / normalization ----------
def _find_scores_file() -> Optional[str]:
    # 1) explicit env
    if ENV_SCORES_CSV and _exists(ENV_SCORES_CSV):
        return ENV_SCORES_CSV

    # 2) env glob
    cand: List[str] = []
    if ENV_SCORES_GLOB:
        cand.extend(sorted(glob.glob(ENV_SCORES_GLOB)))

    # 3) common locations/patterns
    roots = ["data", "data/raw", "docs/data", "docs/data/final"]
    pats  = ["*game_scores_clean*.csv", "*scores_clean*.csv", "*game_scores*.csv", "*games*.csv", "*scores*.csv"]
    for r in roots:
        for p in pats:
            cand.extend(sorted(glob.glob(os.path.join(r, p))))

    seen, ordered = set(), []
    for p in cand:
        if p not in seen and _exists(p):
            ordered.append(p); seen.add(p)

    for p in ordered:
        try:
            df = pd.read_csv(p, nrows=5)
            cols = {c.lower() for c in df.columns}
        except Exception:
            continue
        ok1 = {"game_id","team","points_scored"}.issubset(cols)
        ok2 = {"game_id","home_team","away_team"}.issubset(cols) and (
              {"home_points","away_points"}.issubset(cols) or
              {"home_score","away_score"}.issubset(cols) or
              {"points_home","points_away"}.issubset(cols) or
              {"score_home","score_away"}.issubset(cols)
        )
        if ok1 or ok2:
            return p
    return None

def _scores_to_team_rows(scores: pd.DataFrame, alias_map: Dict[str, str]) -> pd.DataFrame:
    cols = {c.lower(): c for c in scores.columns}

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    gid  = col("game_id","id")
    team = col("team","school","team_name")
    ps   = col("points_scored")
    if gid and team and ps:
        out = scores[[gid, team, ps]].copy()
        out.columns = ["game_id", "team", "points_scored"]
        if alias_map:
            out["team"] = _apply_alias(out["team"], alias_map)
        out["team_key"] = out["team"].map(_norm_key_val)
        out = out.drop_duplicates(subset=["game_id","team_key"], keep="first")
        return out

    hteam = col("home_team","team_home","home")
    ateam = col("away_team","team_away","away")
    hp    = col("home_points","home_score","points_home","score_home")
    ap    = col("away_points","away_score","points_away","score_away")
    if not all([gid, hteam, ateam, hp, ap]):
        return pd.DataFrame()

    a = scores[[gid, hteam, hp]].copy()
    a.columns = ["game_id", "team", "points_scored"]
    b = scores[[gid, ateam, ap]].copy()
    b.columns = ["game_id", "team", "points_scored"]
    out = pd.concat([a, b], ignore_index=True)
    if alias_map:
        out["team"] = _apply_alias(out["team"], alias_map)
    out["team_key"] = out["team"].map(_norm_key_val)
    out = out.drop_duplicates(subset=["game_id","team_key"], keep="first")
    return out

# =========================
# Build pipeline
# =========================
def build():
    log: Dict[str,int] = {}

    alias_map = _load_aliases(OPT_TEAM_ALIASES)

    # 1) Situational (required)
    situ = _read(REQ_SITUATIONAL, parse_dates=["start_date"])
    _require(not situ.empty, f"{REQ_SITUATIONAL} not found or empty.")

    k_game_id  = _find_col(situ, ["game_id","id"])
    k_season   = _find_col(situ, ["season","year"])
    k_week     = _find_col(situ, ["week"])
    k_team     = _find_col(situ, ["team","school","team_name"])
    k_opp      = _find_col(situ, ["opponent","opp","opponent_team"])
    k_is_home  = _find_col(situ, ["is_home","home","home_flag"])
    _require(k_game_id is not None, "situational_factors.csv missing required column: game_id")
    _require(k_season  is not None, "situational_factors.csv missing required column: season")
    _require(k_week    is not None, "situational_factors.csv missing required column: week")
    _require(k_team    is not None, "situational_factors.csv missing required column: team")
    _require(k_opp     is not None, "situational_factors.csv missing required column: opponent")
    _require(k_is_home is not None, "situational_factors.csv missing required column: is_home")

    base = situ.rename(columns={
        k_game_id: "game_id",
        k_season:  "season",
        k_week:    "week",
        k_team:    "team",
        k_opp:     "opponent",
        k_is_home: "is_home",
    }).copy()
    if "start_date" not in base.columns:
        k_start = _find_col(situ, ["start_date","start","kickoff","game_date","game_datetime"])
        if k_start and k_start != "start_date":
            base.rename(columns={k_start: "start_date"}, inplace=True)

    base = _coerce_season_int(base, "season")

    # ---- Historical season filter (APPLIED EARLY) ----
    base = base[pd.to_numeric(base["season"], errors="coerce").between(SEASON_MIN, SEASON_MAX)]
    base = base.dropna(subset=["game_id","season","week","team","opponent","is_home"]).reset_index(drop=True)

    if alias_map:
        base["team"] = _apply_alias(base["team"], alias_map)
        base["opponent"] = _apply_alias(base["opponent"], alias_map)

    # Preserve base keys to guarantee they survive merges
    base["_base_season"] = base["season"]
    base["_base_team"]   = base["team"]

    # Normalized join keys
    base["_team_key"] = base["team"].map(_norm_key_val)
    base["_opp_key"]  = base["opponent"].map(_norm_key_val)

    # Ensure base key columns are first and protected
    required_keys = ["game_id","season","week","team","opponent","is_home","_team_key","_base_season","_base_team"]
    base = base[required_keys + [c for c in base.columns if c not in required_keys]]
    base = base.drop_duplicates(subset=["game_id","team","is_home"], keep="first").reset_index(drop=True)
    log["rows_base_start"] = int(len(base))

    # 2) Weather (game_id + team via home/away; alias-normalized), with controlled suffixing
    wx = _read(OPT_WEATHER, parse_dates=["start_date"])
    if not wx.empty:
        w_gid  = _find_col(wx, ["game_id","id"])
        w_home = _find_col(wx, ["home_team","home","homeTeam","home_team_name"])
        w_away = _find_col(wx, ["away_team","away","awayTeam","away_team_name"])
        if w_gid:
            if w_home:
                if alias_map:
                    wx[w_home] = _apply_alias(wx[w_home], alias_map)
                wx["_w_home_key"] = wx[w_home].map(_norm_key_val)
            if w_away:
                if alias_map:
                    wx[w_away] = _apply_alias(wx[w_away], alias_map)
                wx["_w_away_key"] = wx[w_away].map(_norm_key_val)

            m_home = _safe_left_merge(base, wx, on=[("game_id", w_gid), ("_team_key", "_w_home_key")], suffix_tag="__rwx") if w_home else base.copy()
            m_away = _safe_left_merge(base, wx, on=[("game_id", w_gid), ("_team_key", "_w_away_key")], suffix_tag="__rwx") if w_away else base.copy()

            out = base.copy()
            exclude = {w_gid, w_home, w_away, "_w_home_key", "_w_away_key"}
            wx_feats = [c for c in wx.columns if c not in exclude]
            for c in wx_feats:
                s = m_home[c] if c in m_home.columns else pd.Series([pd.NA]*len(base))
                if c in m_away.columns:
                    s = s.where(~s.isna(), m_away[c])
                out[f"wx_{c}"] = s
            cols_drop = [c for c in out.columns if c.endswith("__rwx")]
            out = out.drop(columns=cols_drop, errors="ignore")
            base = out

    log["rows_after_weather"] = int(len(base))

    # 3) Season-team merges (team_stats, efficiency, special teams)
    def merge_season_team(path: str, prefix: str, suf: str):
        nonlocal base
        df = _read(path)
        if df.empty:
            return
        s_col = _season_col(df)
        t_col = _team_col(df)
        if not s_col or not t_col:
            return

        df = _coerce_season_int(df, s_col)
        # apply same season range filter to right table
        df = df[pd.to_numeric(df[s_col], errors="coerce").between(SEASON_MIN, SEASON_MAX)]

        if alias_map:
            df[t_col] = _apply_alias(df[t_col], alias_map)
        df["_t_key"] = df[t_col].map(_norm_key_val)

        df = _dedupe_by_team_season(df, s_col, "_t_key")

        before_rows = len(base)
        before = base.copy()
        merged = _safe_left_merge(base, df, on=[("season", s_col), ("_team_key", "_t_key")], suffix_tag=suf)

        protect = list(before.columns) + [s_col, t_col, "_t_key"]
        merged = _prefix_new_columns(before, merged, prefix, protect=protect)

        merged = merged.drop(columns=[s_col, t_col, "_t_key"], errors="ignore")
        drop_suff = [c for c in merged.columns if c.endswith(suf)]
        merged = merged.drop(columns=drop_suff, errors="ignore")

        if len(merged) != before_rows:
            merged = before

        base = merged

    merge_season_team(OPT_TEAM_STATS,  "ts_",  "__rts_")
    log["rows_after_team_stats"] = int(len(base))

    merge_season_team(OPT_EFFICIENCY,  "eff_", "__reff_")
    log["rows_after_efficiency"] = int(len(base))

    merge_season_team(OPT_SPECIAL,     "st_",  "__rst_")
    log["rows_after_special"] = int(len(base))

    # 4) Injuries: aggregate counts per (season, team), enforce uniqueness
    def merge_inj(path: str, prefix: str, suf: str):
        nonlocal base
        df = _read(path, parse_dates=["start_date","return_date","last_updated"])
        if df.empty:
            return
        s_col = _season_col(df)
        t_col = _team_col(df)
        if not s_col or not t_col:
            return

        df = _coerce_season_int(df, s_col)
        # season range filter on injuries as well
        df = df[pd.to_numeric(df[s_col], errors="coerce").between(SEASON_MIN, SEASON_MAX)]

        if alias_map:
            df[t_col] = _apply_alias(df[t_col], alias_map)
        df["_t_key"] = df[t_col].map(_norm_key_val)

        agg = df.groupby([s_col, "_t_key"], dropna=False).size().reset_index(name=f"{prefix}_count")
        agg = agg.drop_duplicates(subset=[s_col, "_t_key"], keep="first")

        before_rows = len(base)
        before = base.copy()
        merged = _safe_left_merge(base, agg, on=[("season", s_col), ("_team_key", "_t_key")], suffix_tag=suf)
        protect = list(before.columns) + [s_col, "_t_key"]
        merged = _prefix_new_columns(before, merged, f"inj_{prefix}_", protect=protect)
        merged = merged.drop(columns=[s_col, "_t_key"], errors="ignore")
        drop_suff = [c for c in merged.columns if c.endswith(suf)]
        merged = merged.drop(columns=drop_suff, errors="ignore")

        if len(merged) != before_rows:
            merged = before

        base = merged

    merge_inj(OPT_INJ_DAILY,  "daily",  "__rind_")
    merge_inj(OPT_INJ_LATEST, "latest", "__rind_")
    log["rows_after_injuries"] = int(len(base))

    # 5) Attach OUTCOME TARGETS (points_scored) via strict (game_id, team_key)
    scores_csv = _find_scores_file()
    points_missing = None
    if scores_csv:
        s_raw = _read(scores_csv)
        s_cols = {c.lower(): c for c in s_raw.columns}
        s_team = _scores_to_team_rows(s_raw, alias_map)
        if not s_team.empty and {"game_id","team_key","points_scored"}.issubset(s_team.columns):
            merged = base.merge(
                s_team[["game_id","team_key","points_scored"]],
                left_on=["game_id","_team_key"],
                right_on=["game_id","team_key"],
                how="left",
                validate="m:1"
            )
            points_missing = int(merged["points_scored"].isna().sum())
            base = merged.drop(columns=["team_key"], errors="ignore")
        else:
            points_missing = len(base)
    else:
        points_missing = len(base)

    # 6) Restore guaranteed base keys
    base["season"] = base["_base_season"]
    base["team"]   = base["_base_team"]

    # 7) Cleanup helpers
    base = base.drop(columns=["_team_key","_opp_key","_base_season","_base_team"], errors="ignore").reset_index(drop=True)

    # 8) Write outputs
    os.makedirs(os.path.dirname(OUT_DATASET_CSV), exist_ok=True)
    base.to_csv(OUT_DATASET_CSV, index=False)

    with open(OUT_LOG_TXT, "w", encoding="utf-8") as f:
        f.write(f"snapshot_utc={dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        f.write(f"season_min={SEASON_MIN}\n")
        f.write(f"season_max={SEASON_MAX}\n")
        for k in ["rows_base_start","rows_after_weather","rows_after_team_stats","rows_after_efficiency","rows_after_special","rows_after_injuries"]:
            f.write(f"{k}={log.get(k, 0)}\n")
        f.write(f"rows_final={len(base)}\n")
        if scores_csv:
            f.write(f"scores_csv={scores_csv}\n")
        if points_missing is not None:
            f.write(f"points_merge_missing_rows={points_missing}\n")
        for c in ["game_id","season","week","team","opponent","is_home","points_scored"]:
            miss = float(base[c].isna().mean()) if c in base.columns and len(base) else 1.0
            f.write(f"pct_missing_{c}={round(miss,6)}\n")

def main():
    _ = build()
    print("--- modeling_dataset written ---")

if __name__ == "__main__":
    main()

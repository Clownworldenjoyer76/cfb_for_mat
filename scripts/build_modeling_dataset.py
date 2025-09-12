import os
import sys
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

# =========================
# OUTPUTS
# =========================
OUT_DATASET_CSV = "data/modeling_dataset.csv"
OUT_LOG_TXT     = "logs_build_modeling_dataset.txt"


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

def _keys_situational(df: pd.DataFrame) -> Dict[str, str]:
    m: Dict[str, str] = {}
    m["game_id"]  = _find_col(df, ["game_id","id"])
    m["season"]   = _find_col(df, ["season","year"])
    m["week"]     = _find_col(df, ["week"])
    m["team"]     = _find_col(df, ["team","school","team_name"])
    m["opponent"] = _find_col(df, ["opponent","opp","opponent_team"])
    m["is_home"]  = _find_col(df, ["is_home","home","home_flag"])
    m["start_date"] = _find_col(df, ["start_date","start","kickoff","game_date","game_datetime"])
    return m

def _keys_weather(df: pd.DataFrame) -> Dict[str, str]:
    m: Dict[str, str] = {}
    m["game_id"]   = _find_col(df, ["game_id","id"])
    m["home_team"] = _find_col(df, ["home_team","home","hometeam","home_team_name"])
    m["away_team"] = _find_col(df, ["away_team","away","awayteam","away_team_name"])
    m["season"]    = _find_col(df, ["season","year"])
    return m

def _season_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, ["season","year"])

def _team_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, ["team","school","team_name"])

def _require(cond: bool, msg: str):
    if not cond:
        print(f"INSUFFICIENT INFORMATION: {msg}", file=sys.stderr)
        sys.exit(2)

def _agg_injuries(df: pd.DataFrame, season_col: str, team_col: str, prefix: str) -> pd.DataFrame:
    # Simple per (season, team) counts; can be extended later.
    g = df.groupby([season_col, team_col], dropna=False).size().reset_index(name=f"{prefix}_count")
    return g

def _safe_left_merge(left: pd.DataFrame, right: pd.DataFrame, on: List[Tuple[str,str]]) -> pd.DataFrame:
    # on = [(left_col, right_col), ...]
    left_keys  = [lk for lk, _ in on]
    right_keys = [rk for _, rk in on]
    return left.merge(right, left_on=left_keys, right_on=right_keys, how="left")

def _prefix_new_columns(df_before: pd.DataFrame, df_after: pd.DataFrame, prefix: str, protect: List[str]) -> pd.DataFrame:
    new_cols = [c for c in df_after.columns if c not in df_before.columns and c not in protect]
    return df_after.rename(columns={c: f"{prefix}{c}" for c in new_cols})


# =========================
# Build pipeline
# =========================
def build() -> Tuple[pd.DataFrame, Dict[str,int]]:
    log: Dict[str,int] = {}

    # 1) Load situational (required)
    situ = _read(REQ_SITUATIONAL, parse_dates=["start_date"])
    _require(not situ.empty, f"{REQ_SITUATIONAL} not found or empty.")

    k_sit = _keys_situational(situ)
    for k in ["game_id","season","week","team","opponent","is_home"]:
        _require(k_sit.get(k) is not None, f"{REQ_SITUATIONAL} missing required column: {k}")

    # Normalize base columns
    base = situ.rename(columns={
        k_sit["game_id"]: "game_id",
        k_sit["season"]:  "season",
        k_sit["week"]:    "week",
        k_sit["team"]:    "team",
        k_sit["opponent"]:"opponent",
        k_sit["is_home"]: "is_home",
    }).copy()
    if k_sit.get("start_date") and k_sit["start_date"] != "start_date":
        base = base.rename(columns={k_sit["start_date"]: "start_date"})

    base = base.drop_duplicates(subset=["game_id","team","is_home"], keep="first").reset_index(drop=True)
    log["rows_base_start"] = int(len(base))

    # 2) Weather (join on game_id + team via home/away coalesce)
    wx = _read(OPT_WEATHER, parse_dates=["start_date"])
    if not wx.empty:
        k_wx = _keys_weather(wx)

        # Home merge
        if k_wx.get("game_id") and k_wx.get("home_team"):
            m_home = _safe_left_merge(
                base, wx,
                on=[("game_id", k_wx["game_id"]), ("team", k_wx["home_team"])]
            )
        else:
            m_home = base.copy()

        # Away merge
        if k_wx.get("game_id") and k_wx.get("away_team"):
            m_away = _safe_left_merge(
                base, wx,
                on=[("game_id", k_wx["game_id"]), ("team", k_wx["away_team"])]
            )
        else:
            m_away = base.copy()

        # Coalesce wx features (exclude join keys)
        out = base.copy()
        wx_keys = {k for k in [k_wx.get("game_id"), k_wx.get("home_team"), k_wx.get("away_team")] if k}
        wx_feats = [c for c in wx.columns if c not in wx_keys]

        for c in wx_feats:
            s = m_home[c] if c in m_home.columns else pd.Series([pd.NA]*len(base))
            if c in m_away.columns:
                s = s.where(~s.isna(), m_away[c])
            out[f"wx_{c}"] = s

        base = out
    log["rows_after_weather"] = int(len(base))

    # 3) Team-level season aggregates → join on (season, team)
    def merge_season_team(path: str, prefix: str):
        nonlocal base
        df = _read(path)
        if df.empty:
            return
        s_col = _season_col(df)
        t_col = _team_col(df)
        if not s_col or not t_col:
            return
        before = base.copy()
        merged = _safe_left_merge(base, df, on=[("season","%s"%s_col), ("team","%s"%t_col)])
        merged = _prefix_new_columns(before, merged, prefix, protect=list(before.columns))
        base = merged

    merge_season_team(OPT_TEAM_STATS,  "ts_")
    log["rows_after_team_stats"] = int(len(base))

    merge_season_team(OPT_EFFICIENCY,  "eff_")
    log["rows_after_efficiency"] = int(len(base))

    merge_season_team(OPT_SPECIAL,     "st_")
    log["rows_after_special"] = int(len(base))

    # 4) Injuries (aggregate counts per season, team), then merge on (season, team)
    def merge_inj(path: str, prefix: str):
        nonlocal base
        df = _read(path, parse_dates=["start_date","return_date","last_updated"])
        if df.empty:
            return
        s_col = _season_col(df)
        t_col = _team_col(df)
        if not s_col or not t_col:
            return
        agg = _agg_injuries(df, s_col, t_col, prefix)
        before = base.copy()
        merged = _safe_left_merge(base, agg, on=[("season", s_col), ("team", t_col)])
        merged = _prefix_new_columns(before, merged, f"inj_{prefix}_", protect=list(before.columns))
        base = merged

    merge_inj(OPT_INJ_DAILY,  "daily")
    merge_inj(OPT_INJ_LATEST, "latest")
    log["rows_after_injuries"] = int(len(base))

    # 5) Final tidy
    for c in ["game_id","season","week","team","opponent","is_home"]:
        if c not in base.columns:
            base[c] = pd.NA
    base = base.reset_index(drop=True)

    # 6) Write outputs
    os.makedirs(os.path.dirname(OUT_DATASET_CSV), exist_ok=True)
    base.to_csv(OUT_DATASET_CSV, index=False)

    with open(OUT_LOG_TXT, "w", encoding="utf-8") as f:
        f.write(f"snapshot_utc={dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        for k in ["rows_base_start","rows_after_weather","rows_after_team_stats","rows_after_efficiency","rows_after_special","rows_after_injuries"]:
            f.write(f"{k}={log.get(k, 0)}\n")
        f.write(f"rows_final={len(base)}\n")
        for c in ["game_id","season","week","team","opponent","is_home"]:
            miss = float(base[c].isna().mean()) if len(base) else 1.0
            f.write(f"pct_missing_{c}={round(miss,6)}\n")

    return base, log


def main():
    df, _ = build()
    print("--- modeling_dataset sample ---")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

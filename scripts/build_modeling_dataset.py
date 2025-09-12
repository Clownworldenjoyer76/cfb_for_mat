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
OPT_TEAM_ALIASES= "mappings/team_aliases.csv"        # optional (cfbd_name,alias)

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

def _apply_alias(series: pd.Series, alias_map: Dict[str, str]) -> pd.Series:
    key = _std_team(series)
    norm_map = {_std_team(pd.Series([k]))[0]: v for k, v in alias_map.items()}
    return key.map(norm_map).fillna(series)

def _load_aliases(path: str) -> Dict[str, str]:
    if not _exists(path):
        return {}
    df = _read(path)
    cols = {c.lower(): c for c in df.columns}
    if "cfbd_name" not in cols or "alias" not in cols:
        return {}
    d = {}
    for _, r in df.iterrows():
        src = str(r[cols["cfbd_name"]]).strip()
        tgt = str(r[cols["alias"]]).strip()
        if src and tgt:
            d[src] = tgt
    return d

def _safe_left_merge(left: pd.DataFrame, right: pd.DataFrame, on: List[Tuple[str,str]]) -> pd.DataFrame:
    left_keys  = [lk for lk, _ in on]
    right_keys = [rk for _, rk in on]
    return left.merge(right, left_on=left_keys, right_on=right_keys, how="left")

def _prefix_new_columns(df_before: pd.DataFrame, df_after: pd.DataFrame, prefix: str, protect: List[str]) -> pd.DataFrame:
    new_cols = [c for c in df_after.columns if c not in df_before.columns and c not in protect]
    return df_after.rename(columns={c: f"{prefix}{c}" for c in new_cols})

def _coerce_season_int(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df

def _dedupe_by_team_season(df: pd.DataFrame, s_col: str, t_col: str, key_col: str = "_t_key") -> pd.DataFrame:
    """
    Ensure exactly one row per (season, team). If duplicates exist:
      - numeric columns: mean
      - non-numeric: first
    """
    if df.empty:
        return df

    # Build group keys
    group_cols = [s_col, key_col]
    # Separate numeric vs non-numeric
    num_cols = df.select_dtypes(include=["number", "float", "int", "Int64"]).columns.tolist()
    # Exclude key columns from aggregation lists
    num_cols = [c for c in num_cols if c not in group_cols]

    def _first(s: pd.Series):
        return s.iloc[0]

    agg_dict = {}
    for c in df.columns:
        if c in group_cols:
            continue
        if c in num_cols:
            agg_dict[c] = "mean"
        else:
            agg_dict[c] = _first

    g = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

    # After aggregation, keep only one row per (season, _t_key)
    g = g.drop_duplicates(subset=group_cols, keep="first")
    return g


# =========================
# Build pipeline
# =========================
def build() -> Tuple[pd.DataFrame, Dict[str,int]]:
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

    if alias_map:
        base["team"] = _apply_alias(base["team"], alias_map)
        base["opponent"] = _apply_alias(base["opponent"], alias_map)

    base["_team_key"] = _std_team(base["team"])
    base["_opp_key"]  = _std_team(base["opponent"])

    # Ensure we never lose these base keys
    required_keys = ["game_id","season","week","team","opponent","is_home","_team_key"]
    base = base[required_keys + [c for c in base.columns if c not in required_keys]]
    base = base.drop_duplicates(subset=["game_id","team","is_home"], keep="first").reset_index(drop=True)
    log["rows_base_start"] = int(len(base))

    # 2) Weather (game_id + team via home/away; alias-normalized)
    wx = _read(OPT_WEATHER, parse_dates=["start_date"])
    if not wx.empty:
        w_gid  = _find_col(wx, ["game_id","id"])
        w_home = _find_col(wx, ["home_team","home","homeTeam","home_team_name"])
        w_away = _find_col(wx, ["away_team","away","awayTeam","away_team_name"])
        if w_gid:
            if w_home:
                if alias_map:
                    wx[w_home] = _apply_alias(wx[w_home], alias_map)
                wx["_w_home_key"] = _std_team(wx[w_home])
            if w_away:
                if alias_map:
                    wx[w_away] = _apply_alias(wx[w_away], alias_map)
                wx["_w_away_key"] = _std_team(wx[w_away])

            m_home = _safe_left_merge(base, wx, on=[("game_id", w_gid), ("_team_key", "_w_home_key")]) if w_home else base.copy()
            m_away = _safe_left_merge(base, wx, on=[("game_id", w_gid), ("_team_key", "_w_away_key")]) if w_away else base.copy()

            out = base.copy()
            exclude = {w_gid, w_home, w_away, "_w_home_key", "_w_away_key"}
            wx_feats = [c for c in wx.columns if c not in exclude]
            for c in wx_feats:
                s = m_home[c] if c in m_home.columns else pd.Series([pd.NA]*len(base))
                if c in m_away.columns:
                    s = s.where(~s.isna(), m_away[c])
                out[f"wx_{c}"] = s
            base = out

    log["rows_after_weather"] = int(len(base))

    # 3) Season-team merges (team_stats, efficiency, special teams) — enforce one row per (season, team)
    def merge_season_team(path: str, prefix: str):
        nonlocal base
        df = _read(path)
        if df.empty:
            return
        s_col = _season_col(df)
        t_col = _team_col(df)
        if not s_col or not t_col:
            return

        df = _coerce_season_int(df, s_col)

        if alias_map:
            df[t_col] = _apply_alias(df[t_col], alias_map)
        df["_t_key"] = _std_team(df[t_col])

        # FORCE ONE ROW PER (season, _t_key)
        df = _dedupe_by_team_season(df, s_col, t_col, key_col="_t_key")

        before_rows = len(base)
        before = base.copy()
        merged = _safe_left_merge(base, df, on=[("season", s_col), ("_team_key", "_t_key")])

        # Protect base keys from being overwritten/dropped
        protect = list(before.columns) + [s_col, t_col, "_t_key"]
        merged = _prefix_new_columns(before, merged, prefix, protect=protect)

        # Drop helper cols coming from right
        merged = merged.drop(columns=[s_col, t_col, "_t_key"], errors="ignore")

        # HARD CHECK: row count must not change
        if len(merged) != before_rows:
            # If still changed (shouldn't), fallback: align by re-aggregating df and re-merge
            df_fallback = _dedupe_by_team_season(df, s_col, t_col, key_col="_t_key")
            merged = _safe_left_merge(before, df_fallback, on=[("season", s_col), ("_team_key", "_t_key")])
            merged = _prefix_new_columns(before, merged, prefix, protect=list(before.columns) + [s_col, t_col, "_t_key"])
            merged = merged.drop(columns=[s_col, t_col, "_t_key"], errors="ignore")
            # force same row order/length as base
            merged = merged.loc[:, before.columns + [c for c in merged.columns if c not in before.columns]]

        base = merged

    merge_season_team(OPT_TEAM_STATS,  "ts_")
    log["rows_after_team_stats"] = int(len(base))

    merge_season_team(OPT_EFFICIENCY,  "eff_")
    log["rows_after_efficiency"] = int(len(base))

    merge_season_team(OPT_SPECIAL,     "st_")
    log["rows_after_special"] = int(len(base))

    # 4) Injuries: aggregate counts per (season, team), enforce uniqueness
    def merge_inj(path: str, prefix: str):
        nonlocal base
        df = _read(path, parse_dates=["start_date","return_date","last_updated"])
        if df.empty:
            return
        s_col = _season_col(df)
        t_col = _team_col(df)
        if not s_col or not t_col:
            return

        df = _coerce_season_int(df, s_col)
        if alias_map:
            df[t_col] = _apply_alias(df[t_col], alias_map)
        df["_t_key"] = _std_team(df[t_col])

        # Simple count aggregate per (season, team)
        agg = df.groupby([s_col, "_t_key"], dropna=False).size().reset_index(name=f"{prefix}_count")
        # enforce uniqueness
        agg = agg.drop_duplicates(subset=[s_col, "_t_key"], keep="first")

        before_rows = len(base)
        before = base.copy()
        merged = _safe_left_merge(base, agg, on=[("season", s_col), ("_team_key", "_t_key")])
        protect = list(before.columns) + [s_col, "_t_key"]
        merged = _prefix_new_columns(before, merged, f"inj_{prefix}_", protect=protect)
        merged = merged.drop(columns=[s_col, "_t_key"], errors="ignore")

        # HARD CHECK: row count must not change
        if len(merged) != before_rows:
            merged = before.copy()  # revert if something went wrong

        base = merged

    merge_inj(OPT_INJ_DAILY,  "daily")
    merge_inj(OPT_INJ_LATEST, "latest")
    log["rows_after_injuries"] = int(len(base))

    # 5) Cleanup — ensure mandatory keys exist and are not dropped
    for c in ["game_id","season","week","team","opponent","is_home"]:
        if c not in base.columns:
            base[c] = pd.NA

    # Drop helper keys
    base = base.drop(columns=["_team_key","_opp_key"], errors="ignore").reset_index(drop=True)

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

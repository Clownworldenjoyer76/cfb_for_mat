import os
import sys
import json
import math
import datetime as dt
from typing import List, Dict, Tuple, Optional

import pandas as pd


# ----------------------------
# Configuration (paths)
# ----------------------------
REQ_SITUATIONAL = "situational_factors.csv"          # REQUIRED, per-team rows (team/opponent/is_home)
OPT_WEATHER     = "weather_enriched.csv"             # OPTIONAL, game-level rows (home_team/away_team)
OPT_TEAM_STATS  = "team_stats.csv"                   # OPTIONAL, per-team rows
OPT_EFFICIENCY  = "team_efficiency.csv"              # OPTIONAL, per-team rows
OPT_SPECIAL     = "special_teams_metrics.csv"        # OPTIONAL, per-team rows
OPT_INJ_DAILY   = "injuries_daily.csv"               # OPTIONAL, player-level daily (ignored unless aggregatable)
OPT_INJ_LATEST  = "injuries_latest.csv"              # OPTIONAL, player-level latest (ignored unless aggregatable)

OUT_DATASET_CSV = "data/modeling_dataset.csv"
OUT_LOG_TXT     = "logs_build_modeling_dataset.txt"


# ----------------------------
# Utilities
# ----------------------------
def _exists(p: str) -> bool:
    return os.path.exists(p) and os.path.isfile(p)

def _safe_read_csv(p: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    if not _exists(p):
        return pd.DataFrame()
    try:
        return pd.read_csv(p, parse_dates=parse_dates or [])
    except Exception:
        # fallback without date parsing
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()

def _coalesce_cols(df: pd.DataFrame, want: List[str]) -> Optional[str]:
    """Return the first column in df that matches any of 'want' (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for w in want:
        if w in lower:
            return lower[w]
    return None

def _norm_cols(df: pd.DataFrame) -> Dict[str, str]:
    """Return a mapping of canonical keys to existing column names in df."""
    m: Dict[str, str] = {}
    if df.empty:
        return m
    keys = {
        "game_id": ["game_id", "id"],
        "season": ["season", "year"],
        "week": ["week"],
        "team": ["team", "school", "team_name"],
        "opponent": ["opponent", "opp", "opponent_team"],
        "is_home": ["is_home", "home", "home_flag"],
        "home_team": ["home_team", "home", "homeTeam", "home_team_name"],
        "away_team": ["away_team", "away", "awayTeam", "away_team_name"],
        "start_date": ["start_date", "start", "kickoff", "game_date", "game_datetime"]
    }
    for k, alts in keys.items():
        hit = _coalesce_cols(df, [a.lower() for a in alts])
        if hit:
            m[k] = hit
    return m

def _as_team_frame(df: pd.DataFrame, keys: Dict[str, str]) -> pd.DataFrame:
    """Ensure we have per-team rows: game_id, season, week, team, opponent, is_home."""
    need = ["game_id","season","week","team","opponent","is_home"]
    for n in need:
        if n not in keys:
            # Cannot normalize; return as-is
            return df

    cols = {
        "game_id": keys["game_id"],
        "season": keys["season"],
        "week": keys["week"],
        "team": keys["team"],
        "opponent": keys["opponent"],
        "is_home": keys["is_home"],
    }
    out = df[list(cols.values())].copy()
    out.columns = list(cols.keys())
    # Keep remaining non-key columns as features
    extras = [c for c in df.columns if c not in cols.values()]
    for c in extras:
        out[c] = df[c]
    return out

def _merge_per_team(base: pd.DataFrame, add: pd.DataFrame, base_keys: Dict[str,str], add_keys: Dict[str,str], suffix: str) -> pd.DataFrame:
    """
    Merge a per-team dataset (`add`) into per-team `base` on (game_id, team).
    If `add` has different column names or no 'team', attempt to align via home/away mapping.
    """
    if base.empty or add.empty:
        return base

    base_gid = base_keys.get("game_id","game_id")
    base_team = base_keys.get("team","team")

    # Case A: `add` already has team column -> merge on (game_id, team)
    if "team" in add_keys:
        add_gid = add_keys.get("game_id","game_id")
        add_team = add_keys.get("team","team")
        j = base.merge(add, left_on=[base_gid, base_team], right_on=[add_gid, add_team], how="left", suffixes=("", suffix))
        # Drop join helper duplicates
        for c in [add_gid, add_team]:
            if c in j.columns and c in [base_gid, base_team]:
                # same name retained; skip
                continue
            if c in j.columns and c not in [base_gid, base_team]:
                j = j.drop(columns=[c], errors="ignore")
        return j

    # Case B: `add` has home_team/away_team; align to base.team via two merges and coalesce
    if "home_team" in add_keys and "away_team" in add_keys:
        add_gid = add_keys.get("game_id","game_id")
        add_home = add_keys["home_team"]
        add_away = add_keys["away_team"]

        # Merge as home rows
        m_home = base.merge(
            add, left_on=[base_gid, base_team], right_on=[add_gid, add_home],
            how="left", suffixes=("", f"{suffix}_home")
        )
        # Merge as away rows
        m_away = base.merge(
            add, left_on=[base_gid, base_team], right_on=[add_gid, add_away],
            how="left", suffixes=("", f"{suffix}_away")
        )

        # Coalesce feature columns from home/away merges
        out = base.copy()
        add_feat_cols = [c for c in add.columns if c not in {add_gid, add_home, add_away}]
        for c in add_feat_cols:
            ch = f"{c}{suffix}_home"
            ca = f"{c}{suffix}_away"
            out[c + suffix] = m_home.get(ch)
            alt = m_away.get(ca)
            if alt is not None:
                out[c + suffix] = out[c + suffix].where(~out[c + suffix].isna(), alt)
        return out

    # Unknown shape; skip
    return base

def _prefix_new_columns(df_before: pd.DataFrame, df_after: pd.DataFrame, prefix: str, protected: List[str]) -> pd.DataFrame:
    """Add a prefix to any newly added columns (not in df_before) except protected keys."""
    if df_before is None or df_before.empty:
        return df_after
    new_cols = [c for c in df_after.columns if c not in df_before.columns and c not in protected]
    rename_map = {c: f"{prefix}{c}" for c in new_cols}
    return df_after.rename(columns=rename_map)


# ----------------------------
# Build
# ----------------------------
def build() -> Tuple[pd.DataFrame, Dict[str, int]]:
    log: Dict[str, int] = {}

    # 1) Required: situational_factors (per-team)
    sit = _safe_read_csv(REQ_SITUATIONAL, parse_dates=["start_date"])
    if sit.empty:
        print("INSUFFICIENT INFORMATION: situational_factors.csv not found or empty.", file=sys.stderr)
        sys.exit(2)

    sit_keys = _norm_cols(sit)
    must_have = ["game_id","season","week","team","opponent","is_home"]
    if not all(k in sit_keys for k in must_have):
        print("INSUFFICIENT INFORMATION: situational_factors.csv missing required columns (game_id, season, week, team, opponent, is_home).", file=sys.stderr)
        sys.exit(2)

    base = _as_team_frame(sit, sit_keys)
    base_before = base.copy()
    log["rows_base_start"] = int(len(base))

    # 2) Optional merges

    # 2.a) Team stats
    ts = _safe_read_csv(OPT_TEAM_STATS)
    if not ts.empty:
        ts_keys = _norm_cols(ts)
        before = base.copy()
        base = _merge_per_team(base, ts, sit_keys, ts_keys, suffix="_ts")
        base = _prefix_new_columns(before, base, "ts_", protected=list(before.columns))
        log["rows_after_team_stats"] = int(len(base))
    else:
        log["rows_after_team_stats"] = int(len(base))

    # 2.b) Team efficiency
    te = _safe_read_csv(OPT_EFFICIENCY)
    if not te.empty:
        te_keys = _norm_cols(te)
        before = base.copy()
        base = _merge_per_team(base, te, sit_keys, te_keys, suffix="_eff")
        base = _prefix_new_columns(before, base, "eff_", protected=list(before.columns))
        log["rows_after_efficiency"] = int(len(base))
    else:
        log["rows_after_efficiency"] = int(len(base))

    # 2.c) Special teams
    sp = _safe_read_csv(OPT_SPECIAL)
    if not sp.empty:
        sp_keys = _norm_cols(sp)
        before = base.copy()
        base = _merge_per_team(base, sp, sit_keys, sp_keys, suffix="_st")
        base = _prefix_new_columns(before, base, "st_", protected=list(before.columns))
        log["rows_after_special"] = int(len(base))
    else:
        log["rows_after_special"] = int(len(base))

    # 2.d) Weather (game-level; align via home/away)
    wx = _safe_read_csv(OPT_WEATHER, parse_dates=["start_date"])
    if not wx.empty:
        wx_keys = _norm_cols(wx)
        before = base.copy()

        # Try to align weather rows to team rows: prefer matching base.team to wx.home_team, else wx.away_team
        # Merge-home
        if "home_team" in wx_keys and "game_id" in wx_keys:
            m_home = base.merge(
                wx, left_on=[sit_keys["game_id"], sit_keys["team"]],
                right_on=[wx_keys["game_id"], wx_keys["home_team"]],
                how="left", suffixes=("", "_wxh")
            )
        else:
            m_home = base.copy()

        # Merge-away
        if "away_team" in wx_keys and "game_id" in wx_keys:
            m_away = base.merge(
                wx, left_on=[sit_keys["game_id"], sit_keys["team"]],
                right_on=[wx_keys["game_id"], wx_keys["away_team"]],
                how="left", suffixes=("", "_wxa")
            )
        else:
            m_away = base.copy()

        # Coalesce weather features
        wx_feat = [c for c in wx.columns if c not in {wx_keys.get("game_id","game_id"), wx_keys.get("home_team","home_team"), wx_keys.get("away_team","away_team")}]
        out = base.copy()
        for c in wx_feat:
            ch = c if c in m_home.columns else None
            ca = c if c in m_away.columns else None

            # If suffixes applied, resolve
            if ch is None:
                ch = f"{c}_wxh" if f"{c}_wxh" in m_home.columns else None
            if ca is None:
                ca = f"{c}_wxa" if f"{c}_wxa" in m_away.columns else None

            series = None
            if ch and ch in m_home.columns:
                series = m_home[ch]
            if ca and ca in m_away.columns:
                series = series.where(~series.isna(), m_away[ca]) if series is not None else m_away[ca]

            # Attach with prefix
            if series is not None:
                out[f"wx_{c}"] = series

        base = out
        log["rows_after_weather"] = int(len(base))
    else:
        log["rows_after_weather"] = int(len(base))

    # 2.e) Injuries (optional simple aggregates per team per game if possible)
    inj_candidates = []
    for p in [OPT_INJ_DAILY, OPT_INJ_LATEST]:
        df = _safe_read_csv(p)
        if not df.empty:
            inj_candidates.append((p, df))
    if inj_candidates:
        before = base.copy()
        # Attempt to aggregate by (game_id, team) if columns exist
        for name, inj in inj_candidates:
            inj_keys = _norm_cols(inj)
            gid = inj_keys.get("game_id")
            team = inj_keys.get("team")
            if gid and team:
                # Count rows per (game_id, team) as a simple proxy
                agg = inj.groupby([gid, team], dropna=False).size().reset_index(name="inj_count")
                base = base.merge(
                    agg, left_on=[sit_keys["game_id"], sit_keys["team"]],
                    right_on=[gid, team], how="left"
                )
                base.rename(columns={"inj_count": f"inj_count_{os.path.splitext(os.path.basename(name))[0]}"},
                            inplace=True)
                # drop join helpers if duplicated
                for c in [gid, team]:
                    if c in base.columns and c in [sit_keys["game_id"], sit_keys["team"]]:
                        continue
                    base = base.drop(columns=[c], errors="ignore")
        base = _prefix_new_columns(before, base, "inj_", protected=list(before.columns))
        log["rows_after_injuries"] = int(len(base))
    else:
        log["rows_after_injuries"] = int(len(base))

    # 3) Final cleaning: ensure required keys present and drop exact-duplicate rows
    required_cols = ["game_id","season","week","team","opponent","is_home"]
    for rc in required_cols:
        if rc not in base.columns:
            base[rc] = pd.NA
    base = base.drop_duplicates(subset=["game_id","team","is_home"], keep="first").reset_index(drop=True)

    # 4) Write outputs
    os.makedirs(os.path.dirname(OUT_DATASET_CSV), exist_ok=True)
    base.to_csv(OUT_DATASET_CSV, index=False)

    with open(OUT_LOG_TXT, "w", encoding="utf-8") as f:
        f.write(f"snapshot_utc={dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        f.write(f"rows_base_start={log.get('rows_base_start', 0)}\n")
        f.write(f"rows_after_team_stats={log.get('rows_after_team_stats', 0)}\n")
        f.write(f"rows_after_efficiency={log.get('rows_after_efficiency', 0)}\n")
        f.write(f"rows_after_special={log.get('rows_after_special', 0)}\n")
        f.write(f"rows_after_weather={log.get('rows_after_weather', 0)}\n")
        f.write(f"rows_after_injuries={log.get('rows_after_injuries', 0)}\n")
        f.write(f"rows_final={len(base)}\n")
        # basic missingness on critical keys
        for c in ["game_id","season","week","team","opponent","is_home"]:
            miss = float(base[c].isna().mean()) if len(base) else 1.0
            f.write(f"pct_missing_{c}={round(miss,6)}\n")

    return base, log


# ----------------------------
# Main
# ----------------------------
def main():
    df, _ = build()
    # Minimal console confirmation
    print("--- modeling_dataset sample ---")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

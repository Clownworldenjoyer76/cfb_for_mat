#!/usr/bin/env python3
# Pull CollegeFootballData player stats (season + game) and injuries,
# normalize per configs/player_stats_fields.yaml, write artifacts, and log.
#
# Inputs (env or CLI flags):
#   CFBD_API_KEY          : API key (required)
#   CFB_YEAR              : Year (e.g., 2025). Defaults to current UTC year.
#   CFB_SEASON_TYPE       : regular | postseason (default: regular)
#   CFB_OUT_SEASON_RAW    : season raw CSV (default: player_stats_season_raw.csv)
#   CFB_OUT_GAME_RAW      : game raw CSV (default: player_stats_game_raw.csv)
#   CFB_OUT_WIDE          : wide CSV (default: player_stats_wide.csv)
#   CFB_OUT_INJURIES_RAW  : injuries raw CSV (default: player_injuries_raw.csv)
#   CFB_LOG_PATH          : log file (default: logs/player_stats_run.log)
#
# The script exits 0 even on partial failures; errors are recorded in the log.

import os, sys, io, csv, json, math, time, shutil, argparse, datetime as dt
from typing import Dict, Any, List, Optional
import requests
import pandas as pd
import yaml

CFBD_BASE = "https://api.collegefootballdata.com"

def env_or_default(name: str, default: Optional[str]) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "", "None") else default

def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

def http_get(endpoint: str, params: Dict[str, Any], api_key: str, retries: int = 3, backoff: float = 1.5):
    url = f"{CFBD_BASE}{endpoint}"
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff ** attempt)
                continue
            return None
        except requests.RequestException:
            time.sleep(backoff ** attempt)
    return None

def to_dataframe(records: Any) -> pd.DataFrame:
    if records is None:
        return pd.DataFrame()
    if isinstance(records, list):
        return pd.DataFrame(records)
    if isinstance(records, dict):
        if all(isinstance(v, list) for v in records.values()):
            rows = []
            for k, vs in records.items():
                for r in vs:
                    if isinstance(r, dict):
                        r["_group_key"] = k
                        rows.append(r)
            return pd.DataFrame(rows)
        return pd.json_normalize(records)
    return pd.DataFrame()

def atomic_write_csv(df: pd.DataFrame, out_path: str) -> None:
    tmp = out_path + ".tmp"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(tmp, index=False)
    import shutil as _sh
    _sh.move(tmp, out_path)

def atomic_write_text(text: str, out_path: str) -> None:
    tmp = out_path + ".tmp"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    import shutil as _sh
    _sh.move(tmp, out_path)

def normalize_position(val: Any) -> Any:
    if isinstance(val, str):
        return val.strip().upper()
    return val

def cast_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def apply_config_map(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    keep_map = cfg.get("keep_and_rename", {})
    out = pd.DataFrame()
    for src, dst in keep_map.items():
        if src in df.columns:
            out[dst] = df[src]
        else:
            out[dst] = None
    return out

def derive_metrics(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    derivs = cfg.get("derived_metrics", [])
    for d in derivs:
        name = d.get("name")
        expr = d.get("expr")
        if not name or not expr:
            continue
        try:
            ns = {c: pd.to_numeric(df.get(c), errors="coerce") for c in df.columns}
            df[name] = eval(expr, {"__builtins__": {}}, ns)
        except Exception:
            df[name] = None
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default=env_or_default("CFB_YEAR", None))
    parser.add_argument("--season-type", default=env_or_default("CFB_SEASON_TYPE", "regular"))
    parser.add_argument("--out-season-raw", default=env_or_default("CFB_OUT_SEASON_RAW", "player_stats_season_raw.csv"))
    parser.add_argument("--out-game-raw", default=env_or_default("CFB_OUT_GAME_RAW", "player_stats_game_raw.csv"))
    parser.add_argument("--out-wide", default=env_or_default("CFB_OUT_WIDE", "player_stats_wide.csv"))
    parser.add_argument("--out-injuries-raw", default=env_or_default("CFB_OUT_INJURIES_RAW", "player_injuries_raw.csv"))
    parser.add_argument("--log-path", default=env_or_default("CFB_LOG_PATH", "logs/player_stats_run.log"))
    parser.add_argument("--config", default="configs/player_stats_fields.yaml")
    args = parser.parse_args()

    api_key = os.getenv("CFBD_API_KEY", "").strip()
    year = args.year or str(dt.datetime.now(dt.timezone.utc).year)
    season_type = (args.season_type or "regular").strip().lower()
    if season_type not in ("regular", "postseason"):
        season_type = "regular"

    log_msgs = []
    log_msgs.append(f"[{now_utc_iso()}] Start player stats pull year={year} season_type={season_type}")

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        cfg = {}
        log_msgs.append(f"[{now_utc_iso()}] WARNING: Failed to load config {args.config}: {e}")

    season_params = {"year": year, "seasonType": season_type}
    season_json = http_get("/stats/player/season", season_params, api_key) if api_key else None
    df_season_raw = to_dataframe(season_json)
    df_season_raw["_source"] = "season"
    log_msgs.append(f"[{now_utc_iso()}] Season stats rows={len(df_season_raw)}")

    game_params = {"year": year, "seasonType": season_type}
    game_json = http_get("/stats/player/game", game_params, api_key) if api_key else None
    df_game_raw = to_dataframe(game_json)
    df_game_raw["_source"] = "game"
    log_msgs.append(f"[{now_utc_iso()}] Game stats rows={len(df_game_raw)}")

    injuries_params = {"year": year}
    injuries_json = http_get("/injuries", injuries_params, api_key) if api_key else None
    df_inj_raw = to_dataframe(injuries_json)
    log_msgs.append(f"[{now_utc_iso()}] Injuries rows={len(df_inj_raw)}")

    df_norm = []
    if not df_season_raw.empty:
        rename_hint = {
            "playerId": "player_id",
            "player": "player",
            "team": "team",
            "conference": "conference",
            "opponent": "opponent",
            "season": "season",
            "seasonType": "season_type",
            "position": "position",
            "completions": "completions",
            "attempts": "attempts",
            "passingYards": "passing_yards",
            "passingTDs": "passing_tds",
            "interceptions": "interceptions",
            "sacks": "sacks",
            "rushingAttempts": "rushing_attempts",
            "rushingYards": "rushing_yards",
            "rushingTDs": "rushing_tds",
            "targets": "targets",
            "receptions": "receptions",
            "receivingYards": "receiving_yards",
            "receivingTDs": "receiving_tds",
            "yardsAfterCatch": "yards_after_catch",
            "epa": "epa",
            "epaPerPlay": "epa_per_play",
        }
        cols_present = {c: rename_hint[c] for c in df_season_raw.columns if c in rename_hint}
        df_tmp = df_season_raw.rename(columns=cols_present).copy()
        df_norm = apply_config_map(df_tmp, cfg)
        if "position" in df_norm.columns:
            df_norm["position"] = df_norm["position"].map(lambda s: s.strip().upper() if isinstance(s,str) else s)
        numeric_cols = [
            "pass_cmp","pass_att","pass_yds","pass_td","pass_int","sacks",
            "rush_att","rush_yds","rush_td",
            "rec_tgt","rec","rec_yds","rec_td","yac",
            "epa","epa_per_play"
        ]
        for c in numeric_cols:
            if c in df_norm.columns:
                df_norm[c] = pd.to_numeric(df_norm[c], errors="coerce")
        # derive
        try:
            if "pass_att" in df_norm.columns and "pass_yds" in df_norm.columns:
                df_norm["pass_yards_per_att"] = df_norm["pass_yds"] / df_norm["pass_att"]
            if "pass_att" in df_norm.columns and "pass_td" in df_norm.columns:
                df_norm["td_rate"] = df_norm["pass_td"] / df_norm["pass_att"]
            if "pass_att" in df_norm.columns and "pass_int" in df_norm.columns:
                df_norm["int_rate"] = df_norm["pass_int"] / df_norm["pass_att"]
        except Exception:
            pass

    if isinstance(df_norm, pd.DataFrame) and not df_norm.empty:
        keys = [k for k in ["player_id","season","team"] if k in df_norm.columns]
        if keys:
            df_norm = df_norm.sort_values(keys).drop_duplicates(keys, keep="first")

    # Ensure dirs
    for p in [args.out_season_raw, args.out_game_raw, args.out_wide, args.out_injuries_raw, args.log_path]:
        d = os.path.dirname(p) or "."
        os.makedirs(d, exist_ok=True)

    try:
        atomic_write_csv(df_season_raw, args.out_season_raw)
        log_msgs.append(f"[{now_utc_iso()}] Wrote {args.out_season_raw} rows={len(df_season_raw)}")
    except Exception as e:
        log_msgs.append(f"[{now_utc_iso()}] ERROR writing {args.out_season_raw}: {e}")

    try:
        atomic_write_csv(df_game_raw, args.out_game_raw)
        log_msgs.append(f"[{now_utc_iso()}] Wrote {args.out_game_raw} rows={len(df_game_raw)}")
    except Exception as e:
        log_msgs.append(f"[{now_utc_iso()}] ERROR writing {args.out_game_raw}: {e}")

    try:
        if isinstance(df_norm, pd.DataFrame):
            atomic_write_csv(df_norm, args.out_wide)
            log_msgs.append(f"[{now_utc_iso()}] Wrote {args.out_wide} rows={len(df_norm)}")
        else:
            atomic_write_csv(pd.DataFrame(), args.out_wide)
            log_msgs.append(f"[{now_utc_iso()}] Wrote {args.out_wide} rows=0")
    except Exception as e:
        log_msgs.append(f"[{now_utc_iso()}] ERROR writing {args.out_wide}: {e}")

    try:
        atomic_write_csv(df_inj_raw, args.out_injuries_raw)
        log_msgs.append(f"[{now_utc_iso()}] Wrote {args.out_injuries_raw} rows={len(df_inj_raw)}")
    except Exception as e:
        log_msgs.append(f"[{now_utc_iso()}] ERROR writing {args.out_injuries_raw}: {e}")

    # Names + schema
    try:
        src = df_norm if isinstance(df_norm, pd.DataFrame) and not df_norm.empty else df_season_raw
        cols = [c for c in ["player_id","player","team","position"] if c in src.columns]
        if cols:
            names = src[cols].drop_duplicates()
            atomic_write_csv(names, "player_names_unique.csv")
            log_msgs.append(f"[{now_utc_iso()}] Wrote player_names_unique.csv rows={len(names)}")
    except Exception as e:
        log_msgs.append(f"[{now_utc_iso()}] ERROR writing player_names_unique.csv: {e}")

    try:
        schema = {
            "season_raw_columns": list(df_season_raw.columns),
            "game_raw_columns": list(df_game_raw.columns),
            "wide_columns": list(df_norm.columns) if isinstance(df_norm, pd.DataFrame) else []
        }
        atomic_write_text(json.dumps(schema, indent=2, sort_keys=True), "player_stats_schema.json")
        log_msgs.append(f"[{now_utc_iso()}] Wrote player_stats_schema.json")
    except Exception as e:
        log_msgs.append(f"[{now_utc_iso()}] ERROR writing player_stats_schema.json: {e}")

    # Write log
    atomic_write_text("\n".join(log_msgs) + "\n", args.log_path)

if __name__ == "__main__":
    main()
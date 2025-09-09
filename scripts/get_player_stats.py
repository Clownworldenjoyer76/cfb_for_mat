#!/usr/bin/env python3
# Pull CollegeFootballData player stats (season + game) and injuries,
# normalize per configs/player_stats_fields.yaml, write artifacts, and log.

import os, sys, json, time, shutil, argparse, datetime as dt
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

def http_get(endpoint: str, params: Dict[str, Any], api_key: str,
             retries: int = 3, backoff: float = 1.5):
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
        return pd.json_normalize(records)
    return pd.DataFrame()

def atomic_write_csv(df: pd.DataFrame, out_path: str) -> None:
    tmp = out_path + ".tmp"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(tmp, index=False)
    shutil.move(tmp, out_path)

def atomic_write_text(text: str, out_path: str) -> None:
    tmp = out_path + ".tmp"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    shutil.move(tmp, out_path)

def apply_config_map(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    keep_map = cfg.get("keep_and_rename", {})
    out = pd.DataFrame()
    for src, dst in keep_map.items():
        if src in df.columns:
            out[dst] = df[src]
        else:
            out[dst] = None
    return out

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

    log_msgs = []
    log_msgs.append(f"[{now_utc_iso()}] Start player stats pull year={year} season_type={season_type}")

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        cfg = {}
        log_msgs.append(f"[{now_utc_iso()}] WARNING: Failed to load config {args.config}: {e}")

    # Season-level stats
    df_season_raw = to_dataframe(http_get("/stats/player/season",
                                          {"year": year, "seasonType": season_type}, api_key))
    atomic_write_csv(df_season_raw, args.out_season_raw)

    # Game-level stats
    df_game_raw = to_dataframe(http_get("/stats/player/game",
                                        {"year": year, "seasonType": season_type}, api_key))
    atomic_write_csv(df_game_raw, args.out_game_raw)

    # Injuries
    df_inj_raw = to_dataframe(http_get("/injuries", {"year": year}, api_key))
    atomic_write_csv(df_inj_raw, args.out_injuries_raw)

    # Wide normalized
    if not df_season_raw.empty:
        df_norm = apply_config_map(df_season_raw, cfg)
        atomic_write_csv(df_norm, args.out_wide)
    else:
        atomic_write_csv(pd.DataFrame(), args.out_wide)

    # Names and schema
    if not df_season_raw.empty:
        cols = [c for c in ["player_id","player","team","position"] if c in df_season_raw.columns]
        if cols:
            df_names = df_season_raw[cols].drop_duplicates()
            atomic_write_csv(df_names, "player_names_unique.csv")
    schema = {"season_raw_columns": list(df_season_raw.columns),
              "game_raw_columns": list(df_game_raw.columns)}
    atomic_write_text(json.dumps(schema, indent=2), "player_stats_schema.json")

    atomic_write_text("\n".join(log_msgs), args.log_path)

if __name__ == "__main__":
    main()

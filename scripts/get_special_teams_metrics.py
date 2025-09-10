import os
import time
import datetime as dt
from typing import Any, Dict, List, Optional

import requests
import pandas as pd

API_BASE = "https://api.collegefootballdata.com"

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "", "None") else default

def _get(endpoint: str, params: Dict[str, Any], api_key: str, retries: int = 3, backoff: float = 1.6) -> Any:
    url = API_BASE + endpoint
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=60)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff ** (i + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(backoff ** (i + 1))
    return None

def _to_df(obj: Any) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        return pd.json_normalize(obj)
    return pd.DataFrame()

def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    try:
        n = pd.to_numeric(num, errors="coerce")
        d = pd.to_numeric(den, errors="coerce")
        out = n / d
        out = out.where(d != 0)
        return out
    except Exception:
        return pd.Series([pd.NA] * max(len(num), len(den)))

def _first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _pivot_season(df_season_raw: pd.DataFrame) -> pd.DataFrame:
    # Common CFBD season stats schema: team, statType, stat
    if {"team", "statType", "stat"}.issubset(df_season_raw.columns):
        wide = df_season_raw.pivot_table(index="team", columns="statType", values="stat", aggfunc="first")
        wide = wide.reset_index()
        # attempt numeric
        for c in wide.columns:
            if c != "team":
                wide[c] = pd.to_numeric(wide[c], errors="coerce")
        return wide
    # Already wide or unknown: keep only plausible columns
    keep = [c for c in df_season_raw.columns if c in {"team","conference","season"} or "kick" in c.lower() or "punt" in c.lower() or "fg" in c.lower()]
    if "team" not in keep and "team" in df_season_raw.columns:
        keep = ["team"] + keep
    return df_season_raw[keep].copy() if keep else df_season_raw.copy()

def _pluck_special_from_advanced(df_adv_raw: pd.DataFrame) -> pd.DataFrame:
    # Flatten; keep any special-teams fields
    df = df_adv_raw.copy()
    # Normalize team column name if nested
    if "team" not in df.columns:
        cand = _first(df, ["school", "team.school", "team"])
        if cand:
            df = df.rename(columns={cand: "team"})
    # Keep columns with "special" or starting with "specialTeams"
    special_cols = [c for c in df.columns if "special" in c.lower() or c.lower().startswith("specialteams")]
    keep = ["team"] + special_cols if "team" in df.columns else special_cols
    if not keep:
        return pd.DataFrame(columns=["team"])
    out = df[keep].copy()
    # Try to coerce numerics
    for c in out.columns:
        if c != "team":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # Deduplicate per team
    if "team" in out.columns:
        out = out.groupby("team", as_index=False).first()
    return out

def build(year: str, season_type: str, api_key: str):
    # Pull raw
    adv_json = _get("/stats/advanced", {"year": year, "seasonType": season_type}, api_key)
    sea_json = _get("/stats/season", {"year": year, "seasonType": season_type}, api_key)

    df_adv_raw = _to_df(adv_json)
    df_sea_raw = _to_df(sea_json)

    # Persist raws
    df_adv_raw.to_csv("special_teams_advanced_raw.csv", index=False)
    df_sea_raw.to_csv("special_teams_season_raw.csv", index=False)

    # Transform
    df_adv = _pluck_special_from_advanced(df_adv_raw)
    df_sea = _pivot_season(df_sea_raw)

    # Ensure team column present
    if "team" not in df_sea.columns and "team" in df_adv.columns:
        df_sea["team"] = None
    if "team" not in df_adv.columns and "team" in df_sea.columns:
        df_adv["team"] = None

    # Merge on team
    if "team" in df_adv.columns and "team" in df_sea.columns:
        df = df_sea.merge(df_adv, on="team", how="outer", suffixes=("_season", "_adv"))
    else:
        df = df_sea.copy() if not df_sea.empty else df_adv.copy()

    # Derived metrics (flexible candidates)
    # Field Goals
    fg_made_col = _first(df, ["fieldGoalsMade", "fg_made", "madeFieldGoals", "FGM", "field_goals_made"])
    fg_att_col = _first(df, ["fieldGoalsAttempted", "fg_att", "attemptedFieldGoals", "FGA", "field_goals_attempted"])
    if fg_made_col and fg_att_col:
        df["fg_pct"] = _safe_div(df[fg_made_col], df[fg_att_col])

    # Extra Points
    xp_made_col = _first(df, ["extraPointsMade", "xp_made", "madeExtraPoints", "XPM", "extra_points_made"])
    xp_att_col = _first(df, ["extraPointsAttempted", "xp_att", "attemptedExtraPoints", "XPA", "extra_points_attempted"])
    if xp_made_col and xp_att_col:
        df["xp_pct"] = _safe_div(df[xp_made_col], df[xp_att_col])

    # Punting average
    punts_col = _first(df, ["punts", "totalPunts", "PUNTS"])
    punt_yards_col = _first(df, ["puntYards", "totalPuntYards", "PUNT_YDS", "netPuntYards"])
    if punts_col and punt_yards_col:
        df["punt_avg"] = _safe_div(df[punt_yards_col], df[punts_col])

    # Punt return average
    pr_col = _first(df, ["puntReturns", "PR", "puntReturnAttempts"])
    pr_yards_col = _first(df, ["puntReturnYards", "PR_YDS", "puntReturnYardsTotal"])
    if pr_col and pr_yards_col:
        df["punt_return_avg"] = _safe_div(df[pr_yards_col], df[pr_col])

    # Kick return average
    kr_col = _first(df, ["kickReturns", "KR", "kickReturnAttempts"])
    kr_yards_col = _first(df, ["kickReturnYards", "KR_YDS", "kickReturnYardsTotal"])
    if kr_col and kr_yards_col:
        df["kick_return_avg"] = _safe_div(df[kr_yards_col], df[kr_col])

    # Kickoff touchback rate
    kickoffs_col = _first(df, ["kickoffs", "totalKickoffs", "KICKOFFS"])
    touchbacks_col = _first(df, ["touchbacks", "kickoffTouchbacks", "KO_TB"])
    if kickoffs_col and touchbacks_col:
        df["kickoff_tb_rate"] = _safe_div(df[touchbacks_col], df[kickoffs_col])

    # Reorder columns
    front = ["team"]
    metrics = [c for c in ["fg_pct","xp_pct","punt_avg","punt_return_avg","kick_return_avg","kickoff_tb_rate"] if c in df.columns]
    other = [c for c in df.columns if c not in front + metrics]
    df_out = df[front + metrics + other].copy()

    # Log
    def _pct_missing(col: str) -> float:
        return float(df_out[col].isna().mean()) if col in df_out.columns else 1.0

    with open("logs_special_teams_run.txt", "w", encoding="utf-8") as f:
        f.write(f"year={year}\n")
        f.write(f"season_type={season_type}\n")
        f.write(f"rows={len(df_out)}\n")
        for m in metrics:
            f.write(f"{m}_pct_missing={_pct_missing(m):.6f}\n")

    # Write metrics
    df_out.to_csv("special_teams_metrics.csv", index=False)

def main():
    api_key = _env("CFBD_API_KEY", "")
    year = _env("CFB_YEAR", str(dt.datetime.now(dt.timezone.utc).year))
    season_type = _env("CFB_SEASON_TYPE", "regular").lower()
    if season_type not in ("regular", "postseason"):
        season_type = "regular"
    build(year, season_type, api_key)

if __name__ == "__main__":
    main()

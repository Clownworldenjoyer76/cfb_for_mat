#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
from pathlib import Path
from collections import defaultdict
import pandas as pd, requests

GAMES = Path("data/raw/games.csv")
OUT = Path("data/processed/tempo_metrics.csv")
LOG = Path("summaries/tempo_metrics_summary.txt")

BASE = "https://api.collegefootballdata.com"
TIMEOUT = 60
PERIOD_LEN = 15 * 60

def cfbd_get(path, params, api_key):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.get(BASE + path, params=params, headers=headers, timeout=TIMEOUT)
    r.raise_for_status(); return r.json()

def mmss_to_seconds(s):
    if not s or pd.isna(s): return None
    try: mm, ss = str(s).split(":"); return int(mm)*60+int(ss)
    except: return None

def abs_time_since_start(period, clock_mmss):
    try: p = int(period)
    except: return None
    secs_left = mmss_to_seconds(clock_mmss)
    if secs_left is None: return None
    elapsed = PERIOD_LEN - secs_left
    if elapsed < 0: return None
    return (p-1)*PERIOD_LEN + elapsed

def drive_duration_seconds(d):
    s_abs = abs_time_since_start(d.get("startPeriod"), d.get("startClock"))
    e_abs = abs_time_since_start(d.get("endPeriod"), d.get("endClock"))
    if s_abs is None or e_abs is None:
        dur = mmss_to_seconds(d.get("duration"))
        return int(dur) if dur is not None else 0
    return max(int(e_abs - s_abs), 0)

def classify_play(p):
    t = (p.get("playTypeAbbreviation") or p.get("playType") or "").lower()
    if any(k in t for k in ("pass","sack","sck")): return "pass"
    if any(k in t for k in ("rush","run","rushing")): return "rush"
    text = (p.get("text") or "").lower()
    if "pass" in text or "sack" in text: return "pass"
    if "rush" in text or "run" in text: return "rush"
    return "other"

def norm_team(s): return ("" if s is None else str(s)).strip().lower()

def main():
    if not GAMES.exists():
        print(f"[tempo] ERROR: missing {GAMES}", file=sys.stderr); sys.exit(1)
    api_key = os.getenv("CFBD_API_KEY","")
    games = pd.read_csv(GAMES); games.columns = games.columns.str.lower().str.strip()

    req = ["season","week","game_id","home_team","away_team"]
    miss = [c for c in req if c not in games.columns]
    if miss: print(f"[tempo] ERROR: games.csv missing {miss}", file=sys.stderr); sys.exit(1)

    # Prefer cfbd_game_id where available
    use_ids = []
    for _, r in games.iterrows():
        gid = r.get("cfbd_game_id")
        if pd.notna(gid): use_ids.append(int(gid))
        else: use_ids.append(int(r["game_id"]))
    games = games.assign(_use_game_id=use_ids)

    pos_time = defaultdict(int); plays_total=defaultdict(int); plays_rush=defaultdict(int); plays_pass=defaultdict(int)

    for gid in sorted(set(games["_use_game_id"].tolist())):
        subset = games.loc[games["_use_game_id"]==gid].head(1)
        h_name = norm_team(subset.iloc[0]["home_team"]); a_name = norm_team(subset.iloc[0]["away_team"])
        valid = {h_name, a_name}

        # Drives
        try:
            drives = cfbd_get("/drives", {"gameId": gid}, api_key)
            for d in drives or []:
                off = norm_team(d.get("offense"))
                if off not in valid:
                    if h_name and h_name in off: off = h_name
                    elif a_name and a_name in off: off = a_name
                if off not in valid: continue
                pos_time[(gid, off)] += drive_duration_seconds(d)
        except Exception as e:
            print(f"[tempo] WARN drives {gid}: {e}", file=sys.stderr)

        # Plays
        try:
            plays = cfbd_get("/plays", {"gameId": gid}, api_key)
            for p in plays or []:
                off = norm_team(p.get("offense"))
                if off not in valid:
                    if h_name and h_name in off: off = h_name
                    elif a_name and a_name in off: off = a_name
                if off not in valid: continue
                cat = classify_play(p)
                if cat=="rush": plays_rush[(gid,off)]+=1; plays_total[(gid,off)]+=1
                elif cat=="pass": plays_pass[(gid,off)]+=1; plays_total[(gid,off)]+=1
                else: plays_total[(gid,off)]+=1
        except Exception as e:
            print(f"[tempo] WARN plays {gid}: {e}", file=sys.stderr)

    # Build rows
    rows=[]
    for _, r in games.iterrows():
        season=int(r["season"]); week=int(r["week"]); gid=int(r["_use_game_id"])
        for tcol,ocol,is_home in [("home_team","away_team",1),("away_team","home_team",0)]:
            team=str(r[tcol]).strip(); team_l=norm_team(team); opp=str(r[ocol]).strip()
            total=plays_total.get((gid,team_l),0); rush=plays_rush.get((gid,team_l),0); pss=plays_pass.get((gid,team_l),0); pos=pos_time.get((gid,team_l),0)
            spp = round(pos/total,3) if total>0 and pos>0 else None
            rpr = round(rush/pss,3) if pss>0 else (round(rush/total,3) if total>0 else None)
            rows.append({
                "season":season,"week":week,"game_id":int(r["game_id"]),
                "cfbd_game_id": gid if pd.notna(r.get("cfbd_game_id")) else None,
                "team":team,"opponent":opp,"is_home":is_home,
                "off_plays": total if total>0 else None,
                "off_rush_att": rush if total>0 else None,
                "off_pass_att": pss if total>0 else None,
                "off_possession_sec": pos if pos>0 else None,
                "sec_per_play": spp,"run_pass_ratio": rpr
            })
    out = pd.DataFrame(rows).sort_values(["season","week","game_id","team"])
    OUT.parent.mkdir(parents=True, exist_ok=True); out.to_csv(OUT,index=False)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG,"w") as f:
        f.write(f"rows: {len(out)}\n")
        f.write(f"unique_games: {out['game_id'].nunique()}\n")
        f.write(f"with_cfbd_ids: {out['cfbd_game_id'].notna().sum()//2}\n")
        ms = out["sec_per_play"].dropna()
        mr = out["run_pass_ratio"].dropna()
        f.write(f"mean_sec_per_play: {ms.mean():.2f}\n" if not ms.empty else "mean_sec_per_play: nan\n")
        f.write(f"median_sec_per_play: {ms.median():.2f}\n" if not ms.empty else "median_sec_per_play: nan\n")
        f.write(f"mean_run_pass_ratio: {mr.mean():.2f}\n" if not mr.empty else "mean_run_pass_ratio: nan\n")
        f.write(f"nonzero_possessions: {(out['off_possession_sec'].notna()).sum()}\n")
    print(f"[tempo] wrote {OUT} & {LOG}")

if __name__ == "__main__":
    main()

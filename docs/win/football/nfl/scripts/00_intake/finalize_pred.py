#!/usr/bin/env python3
# docs/win/football/nfl/scripts/00_intake/finalize_pred.py

import csv
import glob
import os
from decimal import Decimal, InvalidOperation

PRED_GLOB = "docs/win/football/nfl/00_intake/predictions/clean/*_predictions.csv"
SCHED_GLOB = "docs/win/football/nfl/00_intake/schedule/*_schedule.csv"
ODDS_GLOB = "docs/win/football/nfl/00_intake/schedule/weekly/*_NFL_weekly_schedule.csv"
OUT_DIR = "docs/win/football/nfl/00_intake/predictions/final"
LOG_PATH = "docs/win/football/nfl/errors/00_intake/finalize_pred.txt"

OUT_HEADERS = [
    "game_id", "game_date", "game_time", "home_team", "away_team",
    "matchupQuality", "home_prob", "away_prob", "tie_prob",
    "away_projected_pts", "home_projected_pts", "total_projected_pts",
    "home_PtDiff", "away_PtDiff", "home_rating", "away_rating",
    "game_name", "season", "season_type", "week", "sport", "league",
]

log = []


def to_dec(raw):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def fmt2(d):
    return str(d.quantize(Decimal("0.01")))


def load_schedule():
    """game_id -> (game_date, game_time)"""
    sched = {}
    files = sorted(glob.glob(SCHED_GLOB))
    if not files:
        log.append("WARNING: no schedule files matched %s" % SCHED_GLOB)
    for path in files:
        try:
            with open(path, "r", newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    gid = (row.get("game_id") or "").strip()
                    if not gid:
                        continue
                    sched[gid] = (
                        (row.get("game_date") or "").strip(),
                        (row.get("game_time") or "").strip(),
                    )
        except Exception as e:
            log.append("FILE ERROR %s: %s" % (path, e))
    log.append("INFO: loaded %d schedule game_ids from %d file(s)" % (len(sched), len(files)))
    return sched


def load_odds():
    """game_id -> total (string), keeping row with most recent odds_last_update"""
    best = {}   # gid -> (last_update, total)
    files = sorted(glob.glob(ODDS_GLOB))
    if not files:
        log.append("WARNING: no weekly schedule/odds files matched %s" % ODDS_GLOB)
    for path in files:
        try:
            with open(path, "r", newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    gid = (row.get("game_id") or "").strip()
                    if not gid:
                        continue
                    total = (row.get("total") or "").strip()
                    if not total:
                        continue
                    lu = (row.get("odds_last_update") or "").strip()
                    prior = best.get(gid)
                    if prior is None or lu > prior[0]:
                        best[gid] = (lu, total)
        except Exception as e:
            log.append("FILE ERROR %s: %s" % (path, e))
    log.append("INFO: loaded totals for %d game_ids from %d file(s)" % (len(best), len(files)))
    return {gid: v[1] for gid, v in best.items()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    sched = load_schedule()
    odds = load_odds()

    pred_files = sorted(glob.glob(PRED_GLOB))
    if not pred_files:
        log.append("FATAL: no prediction files matched %s" % PRED_GLOB)
        write_log()
        return

    files_written = 0
    rows_written = 0
    no_sched = 0
    no_total = 0
    check_fail = 0

    for path in pred_files:
        try:
            with open(path, "r", newline="", encoding="utf-8-sig") as fh:
                rows = list(csv.DictReader(fh))
        except Exception as e:
            log.append("FILE ERROR %s: %s" % (path, e))
            continue

        out_rows = []
        for lineno, row in enumerate(rows, start=2):
            rec = {h: (row.get(h) or "").strip() for h in OUT_HEADERS}
            gid = rec["game_id"]
            if not gid:
                log.append("ROW ERROR %s line %d: blank game_id" % (path, lineno))
                continue

            if gid in sched:
                rec["game_date"], rec["game_time"] = sched[gid]
            else:
                rec["game_date"], rec["game_time"] = "", ""
                no_sched += 1
                log.append("NO SCHEDULE MATCH %s line %d: game_id %s (date/time blank)"
                           % (path, lineno, gid))

            total_raw = odds.get(gid)
            total = to_dec(total_raw)
            if total is None:
                rec["total_projected_pts"] = ""
                rec["home_projected_pts"] = ""
                rec["away_projected_pts"] = ""
                no_total += 1
                log.append("NO TOTAL MATCH %s line %d: game_id %s (projected pts blank)"
                           % (path, lineno, gid))
            else:
                half = total / Decimal(2)
                hpd = to_dec(rec["home_PtDiff"])
                apd = to_dec(rec["away_PtDiff"])
                rec["total_projected_pts"] = fmt2(total)

                if hpd is None:
                    rec["home_projected_pts"] = ""
                    log.append("MISSING home_PtDiff %s line %d: game_id %s" % (path, lineno, gid))
                else:
                    rec["home_projected_pts"] = fmt2(half + hpd)

                if apd is None:
                    rec["away_projected_pts"] = ""
                    log.append("MISSING away_PtDiff %s line %d: game_id %s" % (path, lineno, gid))
                else:
                    rec["away_projected_pts"] = fmt2(half + apd)

                if hpd is not None and apd is not None:
                    diff = abs((half + hpd) + (half + apd) - total)
                    if diff > Decimal("1.0"):
                        check_fail += 1
                        log.append("CHECK FAIL %s line %d: game_id %s home+away=%s total=%s diff=%s"
                                   % (path, lineno, gid,
                                      fmt2((half + hpd) + (half + apd)), fmt2(total), fmt2(diff)))

            out_rows.append(rec)

        if not out_rows:
            log.append("WARNING: no output rows from %s" % path)
            continue

        first = out_rows[0]
        out_name = "%s_%s_%s_clean_predictions.csv" % (
            first["season"], first["season_type"], first["week"])
        out_path = os.path.join(OUT_DIR, out_name)
        try:
            with open(out_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=OUT_HEADERS)
                writer.writeheader()
                for rec in out_rows:
                    writer.writerow(rec)
                    rows_written += 1
            files_written += 1
        except Exception as e:
            log.append("WRITE ERROR %s: %s" % (out_path, e))

    log.append("SUMMARY: pred_files=%d files_written=%d rows_written=%d "
               "no_schedule_match=%d no_total_match=%d check_failures=%d"
               % (len(pred_files), files_written, rows_written, no_sched, no_total, check_fail))
    write_log()


def write_log():
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "w", encoding="utf-8") as fh:
            fh.write("\n".join(log) + "\n")
    except Exception as e:
        print("could not write log: %s" % e)


if __name__ == "__main__":
    main()

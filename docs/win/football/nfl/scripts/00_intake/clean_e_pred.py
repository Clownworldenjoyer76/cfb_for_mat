#!/usr/bin/env python3
# docs/win/football/nfl/scripts/00_intake/clean_e_pred.py

import csv
import os
from collections import defaultdict
from decimal import Decimal, ROUND_DOWN

IN_DIR = "docs/win/football/nfl/00_intake/predictions/e_predictions"
OUT_DIR = "docs/win/football/nfl/00_intake/predictions/clean"
LOG_PATH = "docs/win/football/nfl/errors/00_intake/clean_e_pred.txt"

OUT_HEADERS = [
    "game_id", "game_date", "game_time", "home_team", "away_team",
    "matchupQuality", "home_prob", "away_prob", "tie_prob",
    "away_projected_pts", "home_projected_pts", "total_projected_pts",
    "home_PtDiff", "away_PtDiff", "home_rating", "away_rating",
    "game_name", "season", "season_type", "week", "sport", "league",
]

log = []


def to_decimal_prob(raw):
    """Percentage string -> decimal string truncated to 4 places."""
    d = Decimal(str(raw).strip()) / Decimal(100)
    return str(d.quantize(Decimal("0.0001"), rounding=ROUND_DOWN))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    if not os.path.isdir(IN_DIR):
        log.append("FATAL: input directory not found: %s" % IN_DIR)
        write_log()
        return

    files = sorted(
        os.path.join(IN_DIR, f)
        for f in os.listdir(IN_DIR)
        if os.path.isfile(os.path.join(IN_DIR, f))
    )
    if not files:
        log.append("FATAL: no input files found in %s" % IN_DIR)
        write_log()
        return

    games = {}                  # game_id -> row dict
    order = defaultdict(list)   # (season, season_type, week) -> [game_id,...]
    rows_read = 0

    for path in files:
        try:
            with open(path, "r", newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                for lineno, row in enumerate(reader, start=2):
                    rows_read += 1
                    try:
                        process_row(path, lineno, row, games, order)
                    except Exception as e:
                        log.append("ROW ERROR %s line %d: %s" % (path, lineno, e))
        except Exception as e:
            log.append("FILE ERROR %s: %s" % (path, e))

    files_written = 0
    games_written = 0
    for key in sorted(order.keys()):
        season, season_type, week = key
        out_name = "%s_%s_%s_predictions.csv" % (season, season_type, week)
        out_path = os.path.join(OUT_DIR, out_name)
        try:
            with open(out_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=OUT_HEADERS)
                writer.writeheader()
                for gid in order[key]:
                    rec = games[gid]
                    missing = [
                        f for f in ("home_team", "away_team", "home_prob",
                                    "away_prob", "home_rating", "away_rating",
                                    "home_PtDiff", "away_PtDiff")
                        if not rec.get(f)
                    ]
                    if missing:
                        log.append(
                            "INCOMPLETE game_id %s (%s): missing %s"
                            % (gid, out_name, ",".join(missing))
                        )
                    writer.writerow(rec)
                    games_written += 1
            files_written += 1
        except Exception as e:
            log.append("WRITE ERROR %s: %s" % (out_path, e))

    log.append("SUMMARY: files_read=%d rows_read=%d games=%d files_written=%d games_written=%d"
               % (len(files), rows_read, len(games), files_written, games_written))
    write_log()


def process_row(path, lineno, row, games, order):
    gid = (row.get("game_id") or "").strip()
    if not gid:
        raise ValueError("blank game_id")

    season = (row.get("season") or "").strip()
    season_type = (row.get("season_type") or "").strip()
    week = (row.get("week") or "").strip()
    game_name = (row.get("game_name") or "").strip()
    side = (row.get("home_away") or "").strip()

    if side not in ("homeTeam", "awayTeam"):
        raise ValueError("unrecognized home_away value %r" % side)

    if " at " not in game_name:
        raise ValueError("cannot split game_name %r on ' at '" % game_name)
    away_team, home_team = [p.strip() for p in game_name.split(" at ", 1)]

    key = (season, season_type, week)

    if gid not in games:
        games[gid] = {h: "" for h in OUT_HEADERS}
        games[gid].update({
            "game_id": gid,
            "home_team": home_team,
            "away_team": away_team,
            "matchupQuality": (row.get("matchupQuality") or "").strip(),
            "game_name": game_name,
            "season": season,
            "season_type": season_type,
            "week": week,
            "sport": "football",
            "league": "nfl",
        })
        order[key].append(gid)
    else:
        prior = games[gid]
        if (prior["season"], prior["season_type"], prior["week"]) != key:
            log.append("MISMATCH %s line %d: game_id %s season/type/week differs from prior row"
                       % (path, lineno, gid))

    rec = games[gid]
    prefix = "home" if side == "homeTeam" else "away"
    opp_prefix = "away" if prefix == "home" else "home"

    if rec[prefix + "_prob"]:
        log.append("DUPLICATE %s line %d: second %s row for game_id %s (overwriting)"
                   % (path, lineno, side, gid))

    rec[prefix + "_prob"] = to_decimal_prob(row.get("gameProjection"))
    rec[opp_prefix + "_rating"] = (row.get("oppSeasonStrengthRating") or "").strip()
    rec[prefix + "_PtDiff"] = (row.get("teamPredPtDiff") or "").strip()

    tie = to_decimal_prob(row.get("teamChanceTie"))
    if rec["tie_prob"] and rec["tie_prob"] != tie:
        log.append("TIE MISMATCH %s line %d: game_id %s had %s, now %s"
                   % (path, lineno, gid, rec["tie_prob"], tie))
    rec["tie_prob"] = tie


def write_log():
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "w", encoding="utf-8") as fh:
            fh.write("\n".join(log) + "\n")
    except Exception as e:
        print("could not write log: %s" % e)


if __name__ == "__main__":
    main()

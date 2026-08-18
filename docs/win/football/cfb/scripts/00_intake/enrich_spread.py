#!/usr/bin/env python3
"""
GitHub Actions NFL weekly historical spread-prediction enrichment.

READS ONLY:
  docs/win/football/nfl/00_intake/schedule/weekly/week_{WEEK}_NFL_weekly_schedule.csv
  docs/win/football/nfl/00_intake/predictions/final/*_clean_predictions.csv
  docs/win/football/nfl/00_intake/predictions/drat/clean/{SEASON}_week_{WEEK}_drat.csv
  docs/win/football/nfl/00_intake/odds/{MOST_RECENT_DATE}_NFL_odds.csv
  docs/win/football/nfl/config/prediction_enrichment/spread_enrichment.csv

WRITES ONLY:
  docs/win/football/nfl/00_intake/predictions/enriched/spread/week_{WEEK}_NFL_enriched.csv

The historical bucket boundaries and rule conditions are read from
spread_enrichment.csv. They are not hard-coded here.
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path


MASTER_REL = Path("docs/win/football/nfl/config/prediction_enrichment/spread_enrichment.csv")
SCHEDULE_REL = Path("docs/win/football/nfl/00_intake/schedule/weekly")
EPRED_REL = Path("docs/win/football/nfl/00_intake/predictions/final")
DRAT_REL = Path("docs/win/football/nfl/00_intake/predictions/drat/clean")
ODDS_REL = Path("docs/win/football/nfl/00_intake/odds")
OUTPUT_REL = Path("docs/win/football/nfl/00_intake/predictions/enriched/spread")


def list_weekly_schedule_files(schedule_dir: Path) -> list[Path]:
    files = sorted(schedule_dir.glob("week_*_NFL_weekly_schedule.csv"))
    if not files:
        raise FileNotFoundError(
            f"No week_*_NFL_weekly_schedule.csv files found in {schedule_dir}"
        )
    return files


def schedule_identity(rows: list[dict[str, str]], path: Path):
    require_columns(rows, ["season", "season_type", "week"], f"weekly schedule {path.name}")

    values = {
        (s(r.get("season")), s(r.get("season_type")), s(r.get("week")))
        for r in rows
        if s(r.get("season")) and s(r.get("season_type")) and s(r.get("week"))
    }

    if len(values) != 1:
        raise RuntimeError(
            f"{path.name}: expected exactly one season/season_type/week combination, "
            f"found {sorted(values)}"
        )

    season_text, season_type, week_text = next(iter(values))
    return int(float(season_text)), season_type, int(float(week_text))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def require_columns(rows: list[dict[str, str]], required: list[str], label: str) -> None:
    if not rows:
        raise ValueError(f"{label}: file contains no data rows")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"{label}: missing columns: {', '.join(missing)}")


def s(value) -> str:
    return "" if value is None else str(value).strip()


def num(value):
    try:
        t = s(value)
        return float(t) if t != "" else None
    except (TypeError, ValueError):
        return None


def same_text(a, b) -> bool:
    return s(a).casefold() == s(b).casefold()


def team_key(value) -> str:
    return " ".join(s(value).casefold().split())


def game_team_key(season, week, home, away):
    return (str(int(float(season))), str(int(float(week))), team_key(home), team_key(away))


def american_implied(odds):
    x = num(odds)
    if x is None or x == 0:
        return None
    if x > 0:
        return 100.0 / (x + 100.0)
    return (-x) / ((-x) + 100.0)


def no_vig_probs(home_ml, away_ml):
    h = american_implied(home_ml)
    a = american_implied(away_ml)
    if h is None or a is None:
        return None, None
    total = h + a
    if total <= 0:
        return None, None
    return h / total, a / total


def iso_dt(value):
    text = s(value)
    if not text:
        return datetime.min
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.min


def find_latest_odds_file(odds_dir: Path) -> Path:
    """
    Select the direct-child *_NFL_odds.csv file containing the most recent
    actual odds update. Filename format is irrelevant.
    """
    candidates = []

    for p in odds_dir.glob("*_NFL_odds.csv"):
        try:
            rows = read_csv(p)
        except Exception:
            continue

        latest_update = datetime.min
        found_update = False

        for r in rows:
            dt = iso_dt(r.get("last_update"))
            if dt != datetime.min:
                found_update = True
                if dt > latest_update:
                    latest_update = dt

        if found_update:
            candidates.append((latest_update, p.name, p))

    if not candidates:
        raise FileNotFoundError(
            f"No *_NFL_odds.csv file with a valid last_update value found in {odds_dir}"
        )

    candidates.sort()
    return candidates[-1][2]


def find_drat_file(drat_dir: Path, season: int, week: int) -> Path:
    filename = f"{season}_week_{week}_drat.csv"
    path = drat_dir / filename

    if path.exists():
        return path

    raise FileNotFoundError(
        f"DRAT file not found: {path}"
    )


def epred_content_matches(path: Path, season: int, week: int, season_type: str) -> bool:
    try:
        rows = read_csv(path)
    except Exception:
        return False
    if not rows:
        return False
    for r in rows[:50]:
        if (
            s(r.get("season")) == str(season)
            and s(r.get("week")) == str(week)
            and same_text(r.get("season_type"), season_type)
        ):
            return True
    return False


def find_epred_file(epred_dir: Path, season: int, week: int, season_type: str) -> Path:
    candidates = [
        p for p in epred_dir.glob("*_clean_predictions.csv")
        if epred_content_matches(p, season, week, season_type)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No *_clean_predictions.csv containing season={season}, "
            f"season_type={season_type}, week={week} found in {epred_dir}"
        )
    raise RuntimeError(
        "More than one EPRED file matches this season/week: "
        + ", ".join(p.name for p in candidates)
    )


def aggregate_latest_odds(rows):
    """
    Aggregate duplicated market/side rows into one current market record per
    provider game + bookmaker, taking each populated market field from the
    most recent last_update carrying that field.
    """
    fields = [
        "home_moneyline_american",
        "away_moneyline_american",
        "home_spread",
        "away_spread",
        "home_spread_american",
        "away_spread_american",
        "total",
        "over_american",
        "under_american",
    ]
    groups = {}
    for r in rows:
        gid = s(r.get("game_id"))
        book = s(r.get("bookmaker"))
        if not gid:
            continue
        key = (gid, book.casefold())
        g = groups.setdefault(
            key,
            {
                "game_id": gid,
                "bookmaker": book,
                "last_update": "",
                "__last_dt": datetime.min,
                "__field_dt": {f: datetime.min for f in fields},
            },
        )
        dt = iso_dt(r.get("last_update"))
        if dt >= g["__last_dt"]:
            g["__last_dt"] = dt
            g["last_update"] = s(r.get("last_update"))
        for field in fields:
            value = s(r.get(field))
            if value != "" and dt >= g["__field_dt"][field]:
                g[field] = value
                g["__field_dt"][field] = dt
    return groups


def choose_odds_record(groups, provider_game_id, preferred_bookmaker):
    gid = s(provider_game_id)
    if not gid:
        return None

    preferred = s(preferred_bookmaker).casefold()
    if preferred and (gid, preferred) in groups:
        return groups[(gid, preferred)]

    matches = [g for (g_id, _), g in groups.items() if g_id == gid]
    if not matches:
        return None
    return max(matches, key=lambda g: g["__last_dt"])


def build_family_contexts(g):
    contexts = {}

    dh = num(g.get("drat_home_prob"))
    da = num(g.get("drat_away_prob"))
    eh = num(g.get("epred_home_prob"))
    ea = num(g.get("epred_away_prob"))
    mh = num(g.get("market_home_prob_novig"))
    ma = num(g.get("market_away_prob_novig"))
    hs = num(g.get("market_home_spread"))
    aws = num(g.get("market_away_spread"))

    drat_side = None
    drat_prob = None
    if dh is not None and da is not None:
        drat_side = "Home" if dh >= da else "Away"
        drat_prob = dh if drat_side == "Home" else da

    epred_side = None
    epred_prob = None
    if eh is not None and ea is not None:
        epred_side = "Home" if eh >= ea else "Away"
        epred_prob = eh if epred_side == "Home" else ea

    market_side = None
    market_prob = None
    if mh is not None and ma is not None:
        market_side = "Home" if mh >= ma else "Away"
        market_prob = mh if market_side == "Home" else ma
    elif hs is not None and aws is not None:
        if hs < 0:
            market_side = "Home"
        elif aws < 0:
            market_side = "Away"

    contexts["DRAT"] = {
        "eligible": drat_side is not None,
        "side": drat_side,
        "prob": drat_prob,
    }
    contexts["EPRED"] = {
        "eligible": epred_side is not None,
        "side": epred_side,
        "prob": epred_prob,
    }
    contexts["MARKET"] = {
        "eligible": market_side is not None,
        "side": market_side,
        "prob": market_prob,
    }

    de_ok = drat_side is not None and epred_side is not None and drat_side == epred_side
    contexts["DRAT_EPRED_CONSENSUS"] = {
        "eligible": de_ok,
        "side": drat_side if de_ok else None,
        "prob": (
            (drat_prob + epred_prob) / 2.0
            if de_ok and drat_prob is not None and epred_prob is not None
            else None
        ),
    }

    all_ok = de_ok and market_side is not None and drat_side == market_side
    contexts["ALL3_CONSENSUS"] = {
        "eligible": all_ok,
        "side": drat_side if all_ok else None,
        "prob": (
            (drat_prob + epred_prob + market_prob) / 3.0
            if all_ok and market_prob is not None
            else None
        ),
    }

    g["drat_pick_side"] = drat_side or ""
    g["epred_pick_side"] = epred_side or ""
    g["market_pick_side"] = market_side or ""
    g["drat_pick"] = (
        g["home_team"] if drat_side == "Home"
        else g["away_team"] if drat_side == "Away"
        else ""
    )
    g["epred_pick"] = (
        g["home_team"] if epred_side == "Home"
        else g["away_team"] if epred_side == "Away"
        else ""
    )
    g["market_pick"] = (
        g["home_team"] if market_side == "Home"
        else g["away_team"] if market_side == "Away"
        else ""
    )

    def agreement(a, b):
        if a is None or b is None:
            return "Unknown"
        return "Agree" if a == b else "Disagree"

    g["drat_epred_agree"] = agreement(drat_side, epred_side)
    g["drat_market_agree"] = agreement(drat_side, market_side)
    g["epred_market_agree"] = agreement(epred_side, market_side)
    g["all_three_agree"] = (
        "Yes" if all_ok
        else "No" if drat_side and epred_side and market_side
        else "Unknown"
    )

    return contexts


def market_role_for_side(g, side):
    if side not in ("Home", "Away"):
        return None
    p = num(
        g.get("market_home_prob_novig")
        if side == "Home"
        else g.get("market_away_prob_novig")
    )
    if p is not None:
        if p > 0.5:
            return "Market Favorite"
        if p < 0.5:
            return "Market Underdog"
        return "Market Even"

    spread = num(
        g.get("market_home_spread")
        if side == "Home"
        else g.get("market_away_spread")
    )
    if spread is None:
        return None
    if spread < 0:
        return "Market Favorite"
    if spread > 0:
        return "Market Underdog"
    return "Market Even"


def feature_value(formula_code, g, family_ctx):
    side = family_ctx["side"]

    if formula_code == "USE_FAMILY_SELECTED_PROB":
        return family_ctx["prob"]

    if formula_code == "MARKET_ROLE_FOR_FAMILY_SELECTED_SIDE":
        return market_role_for_side(g, side)

    if formula_code == "SPREAD_FOR_FAMILY_SELECTED_SIDE":
        if side == "Home":
            return num(g.get("market_home_spread"))
        if side == "Away":
            return num(g.get("market_away_spread"))
        return None

    if formula_code == "EPRED_RATING_SELECTED_MINUS_OPPONENT":
        hr = num(g.get("epred_home_rating"))
        ar = num(g.get("epred_away_rating"))
        if hr is None or ar is None or side is None:
            return None
        return (hr - ar) if side == "Home" else (ar - hr)

    if formula_code == "RAW_EPRED_MATCHUP_QUALITY":
        return num(g.get("epred_matchupQuality"))

    if formula_code == "RAW_WEEK":
        return num(g.get("week"))

    if formula_code == "RAW_MARKET_TOTAL":
        return num(g.get("market_total"))

    if formula_code == "COMPARE_DRAT_PICK_TO_EPRED_PICK":
        a = g.get("drat_pick_side")
        b = g.get("epred_pick_side")
        if not a or not b:
            return None
        return "Agree" if a == b else "Disagree"

    if formula_code == "COMPARE_FAMILY_PICK_TO_MARKET_PICK":
        market = g.get("market_pick_side")
        if not side or not market:
            return None
        return "Agree" if side == market else "Disagree"

    if formula_code == "ABS_DRAT_HOME_PROB_MINUS_EPRED_NORMALIZED_HOME_PROB_X100":
        d = num(g.get("drat_home_prob"))
        e = num(g.get("epred_home_prob"))
        if d is None or e is None:
            return None
        return abs(d - e) * 100.0

    if formula_code == "FAMILY_SELECTED_PROB_MINUS_MARKET_SELECTED_PROB_X100":
        fp = family_ctx["prob"]
        if side == "Home":
            mp = num(g.get("market_home_prob_novig"))
        elif side == "Away":
            mp = num(g.get("market_away_prob_novig"))
        else:
            mp = None
        if fp is None or mp is None:
            return None
        return (fp - mp) * 100.0

    if formula_code == "UNAVAILABLE":
        return None

    raise ValueError(f"Unsupported formula_code in master: {formula_code}")


def condition_matches(rule, n, value):
    prefix = f"condition_{n}_"
    match_type = s(rule.get(prefix + "match_type"))

    if not match_type:
        return True

    if match_type == "IS_NULL":
        return value is None or s(value) == ""

    if match_type == "TEXT_EQUALS":
        return s(value) == s(rule.get(prefix + "equals_value"))

    if match_type == "NUMERIC_RANGE":
        x = num(value)
        if x is None:
            return False

        lo = num(rule.get(prefix + "min_inclusive"))
        hi = num(rule.get(prefix + "max_exclusive"))
        test_feature = s(rule.get(prefix + "test_feature"))

        if test_feature == "SpreadBucket":
            if lo is not None and x <= lo:
                return False
            if hi is not None and x > hi:
                return False
            return True

        if lo is not None and x < lo:
            return False
        if hi is not None and x >= hi:
            return False
        return True

    raise ValueError(f"Unsupported match_type in master: {match_type}")


def match_rules(g, master_rows, contexts):
    matches = []

    for rule in master_rows:
        if s(rule.get("active")) != "1":
            continue
        if s(rule.get("pipeline_supported")) != "1":
            continue

        family = s(rule.get("family"))
        family_ctx = contexts.get(family)
        if not family_ctx or not family_ctx["eligible"]:
            continue

        condition_count = int(float(s(rule.get("condition_count")) or "0"))
        ok = True

        for n in range(1, condition_count + 1):
            formula = s(rule.get(f"condition_{n}_formula_code"))
            value = feature_value(formula, g, family_ctx)
            if not condition_matches(rule, n, value):
                ok = False
                break

        if ok:
            matches.append({
                "rule_id": s(rule.get("rule_id")),
                "family": family,
                "side": family_ctx["side"],
                "condition": s(rule.get("source_condition")),
                "historical_cover_rate_pct": num(rule.get("historical_cover_rate_pct")),
                "lift_pp": num(rule.get("lift_vs_family_pct_points")),
                "direction": s(rule.get("action_direction")),
                "games": num(rule.get("games")),
            })

    return matches


def strongest(matches, side, direction):
    candidates = [
        m for m in matches
        if m["side"] == side and m["direction"] == direction and m["lift_pp"] is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda m: abs(m["lift_pp"]))


def family_matches(matches, family):
    return [m for m in matches if m["family"] == family]


def join_text(values):
    return ";".join(s(v) for v in values if s(v))


def build_summary_fields(g, matches):
    pos = [m for m in matches if m["direction"] == "POSITIVE"]
    neg = [m for m in matches if m["direction"] == "NEGATIVE"]

    g["matched_rule_count"] = len(matches)
    g["matched_positive_rule_count"] = len(pos)
    g["matched_negative_rule_count"] = len(neg)
    g["matched_rule_ids"] = join_text(m["rule_id"] for m in matches)
    g["matched_rule_conditions"] = join_text(
        f'{m["rule_id"]}:{m["family"]}:{m["side"]}:{m["condition"]}'
        for m in matches
    )

    for side_name, side in [("home", "Home"), ("away", "Away")]:
        side_matches = [m for m in matches if m["side"] == side]
        g[f"{side_name}_matched_rule_count"] = len(side_matches)
        g[f"{side_name}_matched_rule_ids"] = join_text(m["rule_id"] for m in side_matches)

        sp = strongest(matches, side, "POSITIVE")
        sn = strongest(matches, side, "NEGATIVE")

        for label, item in [("strongest_positive", sp), ("strongest_negative", sn)]:
            g[f"{side_name}_{label}_rule_id"] = item["rule_id"] if item else ""
            g[f"{side_name}_{label}_hist_cover_rate_pct"] = (
                item["historical_cover_rate_pct"] if item else ""
            )
            g[f"{side_name}_{label}_lift_pp"] = item["lift_pp"] if item else ""
            g[f"{side_name}_{label}_games"] = item["games"] if item else ""

    for family, prefix in [
        ("DRAT", "drat"),
        ("EPRED", "epred"),
        ("MARKET", "market"),
        ("DRAT_EPRED_CONSENSUS", "drat_epred_consensus"),
        ("ALL3_CONSENSUS", "all3_consensus"),
    ]:
        fm = family_matches(matches, family)
        g[f"{prefix}_matched_rule_count"] = len(fm)
        g[f"{prefix}_matched_rule_ids"] = join_text(m["rule_id"] for m in fm)

    return g


def process_week(
    root: Path,
    schedule_path: Path,
    odds_path: Path,
    odds_rows: list[dict[str, str]],
    master: list[dict[str, str]],
):
    schedule = read_csv(schedule_path)
    season, season_type, week = schedule_identity(schedule, schedule_path)

    drat_path = find_drat_file(root / DRAT_REL, season, week)
    epred_path = find_epred_file(root / EPRED_REL, season, week, season_type)
    output_path = root / OUTPUT_REL / f"week_{week}_NFL_enriched.csv"

    drat = read_csv(drat_path)
    epred = read_csv(epred_path)

    require_columns(
        schedule,
        [
            "season", "season_type", "week", "game_id", "odds_provider_game_id",
            "away_team", "home_team", "bookmaker",
            "home_moneyline_american", "away_moneyline_american",
            "home_spread", "away_spread", "total",
        ],
        f"weekly schedule {schedule_path.name}",
    )
    require_columns(
        drat,
        ["season", "week", "home_team", "away_team", "home_prob", "away_prob"],
        f"DRAT {drat_path.name}",
    )
    require_columns(
        epred,
        [
            "game_id", "home_team", "away_team", "home_prob", "away_prob",
            "home_rating", "away_rating", "matchupQuality",
        ],
        f"EPRED {epred_path.name}",
    )

    epred_by_game = {s(r["game_id"]): r for r in epred if s(r.get("game_id"))}

    drat_by_teams = {}
    for r in drat:
        try:
            key = game_team_key(
                r.get("season"), r.get("week"), r.get("home_team"), r.get("away_team")
            )
            drat_by_teams[key] = r
        except Exception:
            continue

    current_odds = aggregate_latest_odds(odds_rows)

    output_rows = []
    missing_epred = []
    missing_drat = []

    for base in schedule:
        g = dict(base)

        ep = epred_by_game.get(s(base.get("game_id")))
        if ep is None:
            missing_epred.append(s(base.get("game_id")))

        try:
            dkey = game_team_key(
                base.get("season"), base.get("week"),
                base.get("home_team"), base.get("away_team")
            )
            dr = drat_by_teams.get(dkey)
        except Exception:
            dr = None

        if dr is None:
            missing_drat.append(s(base.get("game_id")))

        odds_rec = choose_odds_record(
            current_odds,
            base.get("odds_provider_game_id"),
            base.get("bookmaker"),
        )

        g["drat_home_prob"] = num(dr.get("home_prob")) if dr else ""
        g["drat_away_prob"] = num(dr.get("away_prob")) if dr else ""

        eph_raw = num(ep.get("home_prob")) if ep else None
        epa_raw = num(ep.get("away_prob")) if ep else None
        g["epred_home_prob_raw"] = eph_raw if eph_raw is not None else ""
        g["epred_away_prob_raw"] = epa_raw if epa_raw is not None else ""
        ep_sum = (
            eph_raw + epa_raw
            if eph_raw is not None and epa_raw is not None
            else None
        )
        g["epred_home_prob"] = (
            eph_raw / ep_sum if ep_sum is not None and ep_sum > 0 else ""
        )
        g["epred_away_prob"] = (
            epa_raw / ep_sum if ep_sum is not None and ep_sum > 0 else ""
        )
        g["epred_home_rating"] = num(ep.get("home_rating")) if ep else ""
        g["epred_away_rating"] = num(ep.get("away_rating")) if ep else ""
        g["epred_matchupQuality"] = num(ep.get("matchupQuality")) if ep else ""

        def market_value(field):
            if odds_rec and s(odds_rec.get(field)) != "":
                return odds_rec.get(field)
            return base.get(field, "")

        g["market_bookmaker"] = (
            odds_rec.get("bookmaker") if odds_rec else base.get("bookmaker", "")
        )
        g["market_last_update"] = odds_rec.get("last_update") if odds_rec else ""
        g["market_home_moneyline_american"] = market_value("home_moneyline_american")
        g["market_away_moneyline_american"] = market_value("away_moneyline_american")
        g["market_home_spread"] = market_value("home_spread")
        g["market_away_spread"] = market_value("away_spread")
        g["market_total"] = market_value("total")

        mh, ma = no_vig_probs(
            g["market_home_moneyline_american"],
            g["market_away_moneyline_american"],
        )
        g["market_home_prob_novig"] = mh if mh is not None else ""
        g["market_away_prob_novig"] = ma if ma is not None else ""

        contexts = build_family_contexts(g)

        hr = num(g.get("epred_home_rating"))
        ar = num(g.get("epred_away_rating"))
        g["epred_rating_gap_home"] = (
            hr - ar if hr is not None and ar is not None else ""
        )

        dh = num(g.get("drat_home_prob"))
        eh = num(g.get("epred_home_prob"))
        g["drat_epred_prob_diff_pp"] = (
            abs(dh - eh) * 100.0 if dh is not None and eh is not None else ""
        )
        g["drat_market_edge_home_pp"] = (
            (dh - mh) * 100.0 if dh is not None and mh is not None else ""
        )
        g["epred_market_edge_home_pp"] = (
            (eh - mh) * 100.0 if eh is not None and mh is not None else ""
        )

        matches = match_rules(g, master, contexts)
        build_summary_fields(g, matches)
        output_rows.append(g)

    if not output_rows:
        raise RuntimeError(f"{schedule_path.name}: no rows to write")

    base_fields = list(schedule[0].keys())
    appended = [
        "drat_home_prob",
        "drat_away_prob",
        "epred_home_prob_raw",
        "epred_away_prob_raw",
        "epred_home_prob",
        "epred_away_prob",
        "epred_home_rating",
        "epred_away_rating",
        "epred_matchupQuality",
        "market_bookmaker",
        "market_last_update",
        "market_home_moneyline_american",
        "market_away_moneyline_american",
        "market_home_spread",
        "market_away_spread",
        "market_total",
        "market_home_prob_novig",
        "market_away_prob_novig",
        "drat_pick",
        "epred_pick",
        "market_pick",
        "drat_epred_agree",
        "drat_market_agree",
        "epred_market_agree",
        "all_three_agree",
        "epred_rating_gap_home",
        "drat_epred_prob_diff_pp",
        "drat_market_edge_home_pp",
        "epred_market_edge_home_pp",
        "matched_rule_count",
        "matched_positive_rule_count",
        "matched_negative_rule_count",
        "matched_rule_ids",
        "matched_rule_conditions",
        "home_matched_rule_count",
        "home_matched_rule_ids",
        "home_strongest_positive_rule_id",
        "home_strongest_positive_hist_cover_rate_pct",
        "home_strongest_positive_lift_pp",
        "home_strongest_positive_games",
        "home_strongest_negative_rule_id",
        "home_strongest_negative_hist_cover_rate_pct",
        "home_strongest_negative_lift_pp",
        "home_strongest_negative_games",
        "away_matched_rule_count",
        "away_matched_rule_ids",
        "away_strongest_positive_rule_id",
        "away_strongest_positive_hist_cover_rate_pct",
        "away_strongest_positive_lift_pp",
        "away_strongest_positive_games",
        "away_strongest_negative_rule_id",
        "away_strongest_negative_hist_cover_rate_pct",
        "away_strongest_negative_lift_pp",
        "away_strongest_negative_games",
        "drat_matched_rule_count",
        "drat_matched_rule_ids",
        "epred_matched_rule_count",
        "epred_matched_rule_ids",
        "market_matched_rule_count",
        "market_matched_rule_ids",
        "drat_epred_consensus_matched_rule_count",
        "drat_epred_consensus_matched_rule_ids",
        "all3_consensus_matched_rule_count",
        "all3_consensus_matched_rule_ids",
    ]
    fieldnames = base_fields + [c for c in appended if c not in base_fields]

    write_csv(output_path, output_rows, fieldnames)

    return {
        "season": season,
        "season_type": season_type,
        "week": week,
        "schedule": schedule_path.name,
        "drat": drat_path.name,
        "epred": epred_path.name,
        "output": str(output_path),
        "games": len(output_rows),
        "missing_epred": len(missing_epred),
        "missing_drat": len(missing_drat),
    }


def main():
    workspace = os.environ.get("GITHUB_WORKSPACE", "").strip()
    if not workspace:
        raise RuntimeError(
            "GITHUB_WORKSPACE is not set. This script is intended to run inside GitHub."
        )
    root = Path(workspace).resolve()

    master_path = root / MASTER_REL
    master = read_csv(master_path)
    require_columns(
        master,
        [
            "rule_id", "active", "pipeline_supported", "family", "condition_count",
            "condition_1_formula_code", "condition_1_match_type",
            "historical_cover_rate_pct", "lift_vs_family_pct_points",
            "action_direction",
        ],
        "historical spread enrichment master",
    )

    odds_path = find_latest_odds_file(root / ODDS_REL)
    odds = read_csv(odds_path)
    require_columns(
        odds,
        [
            "game_id", "bookmaker", "last_update",
            "home_moneyline_american", "away_moneyline_american",
            "home_spread", "away_spread", "total",
        ],
        f"latest odds {odds_path.name}",
    )

    schedule_files = list_weekly_schedule_files(root / SCHEDULE_REL)

    completed = []
    failures = []

    for schedule_path in schedule_files:
        try:
            completed.append(
                process_week(root, schedule_path, odds_path, odds, master)
            )
        except Exception as exc:
            failures.append(f"{schedule_path.name}: {exc}")

    if failures:
        raise RuntimeError(" | ".join(failures))

    if not completed:
        raise RuntimeError(
            "No weekly schedule had all required matching DRAT and EPRED inputs."
        )

    print(f"Historical spread master: {master_path}")
    print(f"Latest odds file: {odds_path}")
    print(f"Weeks enriched: {len(completed)}")
    for result in completed:
        print(
            f"week {result['week']} -> {result['output']} "
            f"(games={result['games']}, missing_epred={result['missing_epred']}, "
            f"missing_drat={result['missing_drat']})"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

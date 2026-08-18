#!/usr/bin/env python3
"""
Step 4: apply the existing NFL enrichment rules to every historical training row.

READS:
  docs/win/football/nfl/training/historical_core_2021_2025.csv
  docs/win/football/nfl/config/prediction_enrichment/moneyline_enrichment.csv
  docs/win/football/nfl/config/prediction_enrichment/spread_enrichment.csv
  docs/win/football/nfl/config/prediction_enrichment/totals_enrichment.csv

WRITES:
  docs/win/football/nfl/training/historical_core_2021_2025.csv

Appends the same ml_*, ats_*, and totals_* enrichment columns produced by
docs/win/football/nfl/scripts/00_intake/enrich_combine.py.

No raw input/source files are edited.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import pandas as pd


NFL_ROOT = Path("docs/win/football/nfl")

TRAINING_PATH = NFL_ROOT / "training/historical_core_2021_2025.csv"

MONEYLINE_RULES_PATH = (
    NFL_ROOT / "config/prediction_enrichment/moneyline_enrichment.csv"
)
SPREAD_RULES_PATH = (
    NFL_ROOT / "config/prediction_enrichment/spread_enrichment.csv"
)
TOTALS_RULES_PATH = (
    NFL_ROOT / "config/prediction_enrichment/totals_enrichment.csv"
)


REQUIRED_TRAINING_COLUMNS = [
    "game_id",
    "season",
    "week",
    "away_team",
    "home_team",
    "away_moneyline",
    "home_moneyline",
    "drat_away_prob",
    "drat_home_prob",
    "epred_away_prob",
    "epred_home_prob",
    "epred_home_rating",
    "epred_away_rating",
    "epred_matchupQuality",
    "hist_home_spread",
    "hist_away_spread",
    "hist_odds_total",
]


COMMON_RULE_COLUMNS = [
    "rule_id",
    "active",
    "pipeline_supported",
    "family",
    "family_eligibility_code",
    "family_selected_side_code",
    "family_selected_probability_code",
    "source_condition",
    "condition_count",
    "condition_1_feature_code",
    "condition_1_formula_code",
    "condition_1_match_type",
    "condition_1_min_inclusive",
    "condition_1_max_exclusive",
    "condition_1_equals_value",
    "condition_2_feature_code",
    "condition_2_formula_code",
    "condition_2_match_type",
    "condition_2_min_inclusive",
    "condition_2_max_exclusive",
    "condition_2_equals_value",
    "games",
    "lift_vs_family_pct_points",
    "action_direction",
    "action_strength_abs_lift_pp",
]


MONEYLINE_MAP = {
    "ml_matched_rule_count": "matched_rule_count",
    "ml_matched_positive_rule_count": "matched_positive_rule_count",
    "ml_matched_negative_rule_count": "matched_negative_rule_count",
    "ml_matched_rule_ids": "matched_rule_ids",
    "ml_matched_rule_conditions": "matched_rule_conditions",
    "ml_home_matched_rule_count": "home_matched_rule_count",
    "ml_home_matched_rule_ids": "home_matched_rule_ids",
    "ml_home_strongest_positive_rule_id": "home_strongest_positive_rule_id",
    "ml_home_strongest_positive_hist_win_rate_pct": "home_strongest_positive_hist_win_rate_pct",
    "ml_home_strongest_positive_lift_pp": "home_strongest_positive_lift_pp",
    "ml_home_strongest_positive_games": "home_strongest_positive_games",
    "ml_home_strongest_negative_rule_id": "home_strongest_negative_rule_id",
    "ml_home_strongest_negative_hist_win_rate_pct": "home_strongest_negative_hist_win_rate_pct",
    "ml_home_strongest_negative_lift_pp": "home_strongest_negative_lift_pp",
    "ml_home_strongest_negative_games": "home_strongest_negative_games",
    "ml_away_matched_rule_count": "away_matched_rule_count",
    "ml_away_matched_rule_ids": "away_matched_rule_ids",
    "ml_away_strongest_positive_rule_id": "away_strongest_positive_rule_id",
    "ml_away_strongest_positive_hist_win_rate_pct": "away_strongest_positive_hist_win_rate_pct",
    "ml_away_strongest_positive_lift_pp": "away_strongest_positive_lift_pp",
    "ml_away_strongest_positive_games": "away_strongest_positive_games",
    "ml_away_strongest_negative_rule_id": "away_strongest_negative_rule_id",
    "ml_away_strongest_negative_hist_win_rate_pct": "away_strongest_negative_hist_win_rate_pct",
    "ml_away_strongest_negative_lift_pp": "away_strongest_negative_lift_pp",
    "ml_away_strongest_negative_games": "away_strongest_negative_games",
    "ml_drat_matched_rule_count": "drat_matched_rule_count",
    "ml_drat_matched_rule_ids": "drat_matched_rule_ids",
    "ml_epred_matched_rule_count": "epred_matched_rule_count",
    "ml_epred_matched_rule_ids": "epred_matched_rule_ids",
    "ml_market_matched_rule_count": "market_matched_rule_count",
    "ml_market_matched_rule_ids": "market_matched_rule_ids",
    "ml_drat_epred_consensus_matched_rule_count": "drat_epred_consensus_matched_rule_count",
    "ml_drat_epred_consensus_matched_rule_ids": "drat_epred_consensus_matched_rule_ids",
    "ml_all3_consensus_matched_rule_count": "all3_consensus_matched_rule_count",
    "ml_all3_consensus_matched_rule_ids": "all3_consensus_matched_rule_ids",
}


ATS_MAP = {
    "ats_matched_rule_count": "matched_rule_count",
    "ats_matched_positive_rule_count": "matched_positive_rule_count",
    "ats_matched_negative_rule_count": "matched_negative_rule_count",
    "ats_matched_rule_ids": "matched_rule_ids",
    "ats_matched_rule_conditions": "matched_rule_conditions",
    "ats_home_matched_rule_count": "home_matched_rule_count",
    "ats_home_matched_rule_ids": "home_matched_rule_ids",
    "ats_home_strongest_positive_rule_id": "home_strongest_positive_rule_id",
    "ats_home_strongest_positive_hist_cover_rate_pct": "home_strongest_positive_hist_cover_rate_pct",
    "ats_home_strongest_positive_lift_pp": "home_strongest_positive_lift_pp",
    "ats_home_strongest_positive_games": "home_strongest_positive_games",
    "ats_home_strongest_negative_rule_id": "home_strongest_negative_rule_id",
    "ats_home_strongest_negative_hist_cover_rate_pct": "home_strongest_negative_hist_cover_rate_pct",
    "ats_home_strongest_negative_lift_pp": "home_strongest_negative_lift_pp",
    "ats_home_strongest_negative_games": "home_strongest_negative_games",
    "ats_away_matched_rule_count": "away_matched_rule_count",
    "ats_away_matched_rule_ids": "away_matched_rule_ids",
    "ats_away_strongest_positive_rule_id": "away_strongest_positive_rule_id",
    "ats_away_strongest_positive_hist_cover_rate_pct": "away_strongest_positive_hist_cover_rate_pct",
    "ats_away_strongest_positive_lift_pp": "away_strongest_positive_lift_pp",
    "ats_away_strongest_positive_games": "away_strongest_positive_games",
    "ats_away_strongest_negative_rule_id": "away_strongest_negative_rule_id",
    "ats_away_strongest_negative_hist_cover_rate_pct": "away_strongest_negative_hist_cover_rate_pct",
    "ats_away_strongest_negative_lift_pp": "away_strongest_negative_lift_pp",
    "ats_away_strongest_negative_games": "away_strongest_negative_games",
    "ats_drat_matched_rule_count": "drat_matched_rule_count",
    "ats_drat_matched_rule_ids": "drat_matched_rule_ids",
    "ats_epred_matched_rule_count": "epred_matched_rule_count",
    "ats_epred_matched_rule_ids": "epred_matched_rule_ids",
    "ats_market_matched_rule_count": "market_matched_rule_count",
    "ats_market_matched_rule_ids": "market_matched_rule_ids",
    "ats_drat_epred_consensus_matched_rule_count": "drat_epred_consensus_matched_rule_count",
    "ats_drat_epred_consensus_matched_rule_ids": "drat_epred_consensus_matched_rule_ids",
    "ats_all3_consensus_matched_rule_count": "all3_consensus_matched_rule_count",
    "ats_all3_consensus_matched_rule_ids": "all3_consensus_matched_rule_ids",
}


TOTALS_MAP = {
    "totals_matched_rule_count": "matched_rule_count",
    "totals_matched_positive_rule_count": "matched_positive_rule_count",
    "totals_matched_negative_rule_count": "matched_negative_rule_count",
    "totals_matched_rule_ids": "matched_rule_ids",
    "totals_matched_rule_conditions": "matched_rule_conditions",
    "totals_over_matched_rule_count": "over_matched_rule_count",
    "totals_over_matched_positive_rule_count": "over_matched_positive_rule_count",
    "totals_over_matched_negative_rule_count": "over_matched_negative_rule_count",
    "totals_over_matched_rule_ids": "over_matched_rule_ids",
    "totals_over_strongest_positive_rule_id": "over_strongest_positive_rule_id",
    "totals_over_strongest_positive_hist_hit_rate_pct": "over_strongest_positive_hist_hit_rate_pct",
    "totals_over_strongest_positive_lift_pp": "over_strongest_positive_lift_pp",
    "totals_over_strongest_positive_games": "over_strongest_positive_games",
    "totals_over_strongest_negative_rule_id": "over_strongest_negative_rule_id",
    "totals_over_strongest_negative_hist_hit_rate_pct": "over_strongest_negative_hist_hit_rate_pct",
    "totals_over_strongest_negative_lift_pp": "over_strongest_negative_lift_pp",
    "totals_over_strongest_negative_games": "over_strongest_negative_games",
    "totals_under_matched_rule_count": "under_matched_rule_count",
    "totals_under_matched_positive_rule_count": "under_matched_positive_rule_count",
    "totals_under_matched_negative_rule_count": "under_matched_negative_rule_count",
    "totals_under_matched_rule_ids": "under_matched_rule_ids",
    "totals_under_strongest_positive_rule_id": "under_strongest_positive_rule_id",
    "totals_under_strongest_positive_hist_hit_rate_pct": "under_strongest_positive_hist_hit_rate_pct",
    "totals_under_strongest_positive_lift_pp": "under_strongest_positive_lift_pp",
    "totals_under_strongest_positive_games": "under_strongest_positive_games",
    "totals_under_strongest_negative_rule_id": "under_strongest_negative_rule_id",
    "totals_under_strongest_negative_hist_hit_rate_pct": "under_strongest_negative_hist_hit_rate_pct",
    "totals_under_strongest_negative_lift_pp": "under_strongest_negative_lift_pp",
    "totals_under_strongest_negative_games": "under_strongest_negative_games",
    "totals_drat_matched_rule_count": "drat_matched_rule_count",
    "totals_drat_matched_rule_ids": "drat_matched_rule_ids",
    "totals_epred_matched_rule_count": "epred_matched_rule_count",
    "totals_epred_matched_rule_ids": "epred_matched_rule_ids",
    "totals_market_matched_rule_count": "market_matched_rule_count",
    "totals_market_matched_rule_ids": "market_matched_rule_ids",
    "totals_drat_epred_consensus_matched_rule_count": "drat_epred_consensus_matched_rule_count",
    "totals_drat_epred_consensus_matched_rule_ids": "drat_epred_consensus_matched_rule_ids",
    "totals_all3_consensus_matched_rule_count": "all3_consensus_matched_rule_count",
    "totals_all3_consensus_matched_rule_ids": "all3_consensus_matched_rule_ids",
}


ALL_OUTPUT_COLUMNS = [
    *MONEYLINE_MAP.keys(),
    *ATS_MAP.keys(),
    *TOTALS_MAP.keys(),
]


def s(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def num(value):
    text = s(value)
    if text == "":
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        low_memory=False,
    )


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def american_implied(odds):
    x = num(odds)
    if x is None or x == 0:
        return None
    if x > 0:
        return 100.0 / (x + 100.0)
    return (-x) / ((-x) + 100.0)


def no_vig_probs(home_ml, away_ml):
    home = american_implied(home_ml)
    away = american_implied(away_ml)

    if home is None or away is None:
        return None, None

    total = home + away
    if total <= 0:
        return None, None

    return home / total, away / total


def normalize_epred_probs(home_raw, away_raw):
    home = num(home_raw)
    away = num(away_raw)

    if home is None or away is None:
        return None, None

    total = home + away
    if total <= 0:
        return None, None

    return home / total, away / total


def build_game_context(row: dict[str, object]) -> dict[str, object]:
    g = dict(row)

    epred_home, epred_away = normalize_epred_probs(
        row.get("epred_home_prob"),
        row.get("epred_away_prob"),
    )

    market_home_prob, market_away_prob = no_vig_probs(
        row.get("home_moneyline"),
        row.get("away_moneyline"),
    )

    g["drat_home_prob"] = num(row.get("drat_home_prob"))
    g["drat_away_prob"] = num(row.get("drat_away_prob"))

    g["epred_home_prob"] = epred_home
    g["epred_away_prob"] = epred_away
    g["epred_home_rating"] = num(row.get("epred_home_rating"))
    g["epred_away_rating"] = num(row.get("epred_away_rating"))
    g["epred_matchupQuality"] = num(row.get("epred_matchupQuality"))

    # Historical equivalents of the live pipeline's market fields.
    g["market_home_moneyline_american"] = num(row.get("home_moneyline"))
    g["market_away_moneyline_american"] = num(row.get("away_moneyline"))
    g["market_home_spread"] = num(row.get("hist_home_spread"))
    g["market_away_spread"] = num(row.get("hist_away_spread"))
    g["market_total"] = num(row.get("hist_odds_total"))
    g["market_home_prob_novig"] = market_home_prob
    g["market_away_prob_novig"] = market_away_prob

    return g


def build_family_contexts(g: dict[str, object]):
    contexts = {}

    drat_home = num(g.get("drat_home_prob"))
    drat_away = num(g.get("drat_away_prob"))
    epred_home = num(g.get("epred_home_prob"))
    epred_away = num(g.get("epred_away_prob"))
    market_home = num(g.get("market_home_prob_novig"))
    market_away = num(g.get("market_away_prob_novig"))
    home_spread = num(g.get("market_home_spread"))
    away_spread = num(g.get("market_away_spread"))

    drat_side = None
    drat_prob = None
    if drat_home is not None and drat_away is not None:
        if drat_home >= drat_away:
            drat_side = "Home"
            drat_prob = drat_home
        else:
            drat_side = "Away"
            drat_prob = drat_away

    epred_side = None
    epred_prob = None
    if epred_home is not None and epred_away is not None:
        if epred_home >= epred_away:
            epred_side = "Home"
            epred_prob = epred_home
        else:
            epred_side = "Away"
            epred_prob = epred_away

    market_side = None
    market_prob = None
    if market_home is not None and market_away is not None:
        if market_home >= market_away:
            market_side = "Home"
            market_prob = market_home
        else:
            market_side = "Away"
            market_prob = market_away
    elif home_spread is not None and away_spread is not None:
        if home_spread < 0:
            market_side = "Home"
        elif away_spread < 0:
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

    drat_epred_ok = (
        drat_side is not None
        and epred_side is not None
        and drat_side == epred_side
    )

    contexts["DRAT_EPRED_CONSENSUS"] = {
        "eligible": drat_epred_ok,
        "side": drat_side if drat_epred_ok else None,
        "prob": (
            (drat_prob + epred_prob) / 2.0
            if (
                drat_epred_ok
                and drat_prob is not None
                and epred_prob is not None
            )
            else None
        ),
    }

    all_three_ok = (
        drat_epred_ok
        and market_side is not None
        and drat_side == market_side
    )

    contexts["ALL3_CONSENSUS"] = {
        "eligible": all_three_ok,
        "side": drat_side if all_three_ok else None,
        "prob": (
            (drat_prob + epred_prob + market_prob) / 3.0
            if all_three_ok and market_prob is not None
            else None
        ),
    }

    g["drat_pick_side"] = drat_side or ""
    g["epred_pick_side"] = epred_side or ""
    g["market_pick_side"] = market_side or ""

    return contexts


def market_role_for_side(g, side):
    if side not in ("Home", "Away"):
        return None

    market_prob = num(
        g.get(
            "market_home_prob_novig"
            if side == "Home"
            else "market_away_prob_novig"
        )
    )

    if market_prob is not None:
        if market_prob > 0.5:
            return "Market Favorite"
        if market_prob < 0.5:
            return "Market Underdog"
        return "Market Even"

    spread = num(
        g.get(
            "market_home_spread"
            if side == "Home"
            else "market_away_spread"
        )
    )

    if spread is None:
        return None
    if spread < 0:
        return "Market Favorite"
    if spread > 0:
        return "Market Underdog"
    return "Market Even"


def feature_value(
    formula_code: str,
    g: dict[str, object],
    family_ctx: dict[str, object],
):
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
        home_rating = num(g.get("epred_home_rating"))
        away_rating = num(g.get("epred_away_rating"))

        if home_rating is None or away_rating is None or side is None:
            return None

        if side == "Home":
            return home_rating - away_rating
        return away_rating - home_rating

    if formula_code == "RAW_EPRED_MATCHUP_QUALITY":
        return num(g.get("epred_matchupQuality"))

    if formula_code == "RAW_WEEK":
        return num(g.get("week"))

    if formula_code == "RAW_MARKET_TOTAL":
        return num(g.get("market_total"))

    if formula_code == "COMPARE_DRAT_PICK_TO_EPRED_PICK":
        drat_side = s(g.get("drat_pick_side"))
        epred_side = s(g.get("epred_pick_side"))

        if not drat_side or not epred_side:
            return None

        return "Agree" if drat_side == epred_side else "Disagree"

    if formula_code == "COMPARE_FAMILY_PICK_TO_MARKET_PICK":
        market_side = s(g.get("market_pick_side"))

        if not side or not market_side:
            return None

        return "Agree" if side == market_side else "Disagree"

    if (
        formula_code
        == "ABS_DRAT_HOME_PROB_MINUS_EPRED_NORMALIZED_HOME_PROB_X100"
    ):
        drat_home = num(g.get("drat_home_prob"))
        epred_home = num(g.get("epred_home_prob"))

        if drat_home is None or epred_home is None:
            return None

        return abs(drat_home - epred_home) * 100.0

    if (
        formula_code
        == "FAMILY_SELECTED_PROB_MINUS_MARKET_SELECTED_PROB_X100"
    ):
        family_prob = num(family_ctx.get("prob"))

        if side == "Home":
            market_prob = num(g.get("market_home_prob_novig"))
        elif side == "Away":
            market_prob = num(g.get("market_away_prob_novig"))
        else:
            market_prob = None

        if family_prob is None or market_prob is None:
            return None

        return (family_prob - market_prob) * 100.0

    if formula_code == "UNAVAILABLE":
        return None

    raise ValueError(f"Unsupported formula_code in enrichment rules: {formula_code}")


def condition_matches(
    rule: dict[str, object],
    condition_number: int,
    value,
    spread_bucket_live_semantics: bool,
) -> bool:
    prefix = f"condition_{condition_number}_"
    match_type = s(rule.get(prefix + "match_type"))

    if not match_type:
        return True

    if match_type == "IS_NULL":
        return value is None or s(value) == ""

    if match_type == "TEXT_EQUALS":
        return s(value) == s(rule.get(prefix + "equals_value"))

    if match_type == "NUMERIC_RANGE":
        numeric_value = num(value)
        if numeric_value is None:
            return False

        minimum = num(rule.get(prefix + "min_inclusive"))
        maximum = num(rule.get(prefix + "max_exclusive"))

        # This reproduces the live moneyline/spread enrichment scripts'
        # existing SpreadBucket boundary convention.
        test_feature = s(rule.get(prefix + "test_feature"))
        if spread_bucket_live_semantics and test_feature == "SpreadBucket":
            if minimum is not None and numeric_value <= minimum:
                return False
            if maximum is not None and numeric_value > maximum:
                return False
            return True

        if minimum is not None and numeric_value < minimum:
            return False
        if maximum is not None and numeric_value >= maximum:
            return False
        return True

    raise ValueError(f"Unsupported match_type in enrichment rules: {match_type}")


def match_rules(
    g: dict[str, object],
    rules: list[dict[str, object]],
    contexts: dict[str, dict[str, object]],
    market_type: str,
):
    matches = []

    for rule in rules:
        if s(rule.get("active")) != "1":
            continue
        if s(rule.get("pipeline_supported")) != "1":
            continue

        family = s(rule.get("family"))
        family_ctx = contexts.get(family)

        if not family_ctx or not family_ctx["eligible"]:
            continue

        condition_count = int(float(s(rule.get("condition_count")) or "0"))
        matched = True

        for condition_number in range(1, condition_count + 1):
            formula_code = s(
                rule.get(f"condition_{condition_number}_formula_code")
            )
            value = feature_value(formula_code, g, family_ctx)

            if not condition_matches(
                rule,
                condition_number,
                value,
                spread_bucket_live_semantics=market_type in (
                    "moneyline",
                    "spread",
                ),
            ):
                matched = False
                break

        if not matched:
            continue

        item = {
            "rule_id": s(rule.get("rule_id")),
            "family": family,
            "side": family_ctx["side"],
            "condition": s(rule.get("source_condition")),
            "lift_pp": num(rule.get("lift_vs_family_pct_points")),
            "action_strength": num(rule.get("action_strength_abs_lift_pp")),
            "direction": s(rule.get("action_direction")),
            "games": num(rule.get("games")),
        }

        if market_type == "moneyline":
            item["historical_rate"] = num(
                rule.get("historical_win_rate_pct")
            )
        elif market_type == "spread":
            item["historical_rate"] = num(
                rule.get("historical_cover_rate_pct")
            )
        elif market_type == "totals":
            item["historical_rate"] = num(
                rule.get("historical_hit_rate_pct")
            )
            item["totals_direction"] = s(rule.get("totals_direction"))
        else:
            raise ValueError(f"Unknown market type: {market_type}")

        matches.append(item)

    return matches


def join_text(values) -> str:
    return ";".join(s(value) for value in values if s(value))


def strongest_side(matches, side: str, direction: str):
    candidates = [
        match
        for match in matches
        if (
            match["side"] == side
            and match["direction"] == direction
            and match["lift_pp"] is not None
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda match: (
            match["action_strength"]
            if match["action_strength"] is not None
            else abs(match["lift_pp"])
        ),
    )


def strongest_total(matches, totals_direction: str, action_direction: str):
    candidates = [
        match
        for match in matches
        if (
            match.get("totals_direction") == totals_direction
            and match["direction"] == action_direction
            and match["lift_pp"] is not None
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda match: (
            match["action_strength"]
            if match["action_strength"] is not None
            else abs(match["lift_pp"])
        ),
    )


def family_matches(matches, family: str):
    return [match for match in matches if match["family"] == family]


def build_side_summary(matches, historical_label: str):
    output = {}

    positive = [
        match for match in matches if match["direction"] == "POSITIVE"
    ]
    negative = [
        match for match in matches if match["direction"] == "NEGATIVE"
    ]

    output["matched_rule_count"] = len(matches)
    output["matched_positive_rule_count"] = len(positive)
    output["matched_negative_rule_count"] = len(negative)
    output["matched_rule_ids"] = join_text(
        match["rule_id"] for match in matches
    )
    output["matched_rule_conditions"] = join_text(
        (
            f'{match["rule_id"]}:'
            f'{match["family"]}:'
            f'{match["side"]}:'
            f'{match["condition"]}'
        )
        for match in matches
    )

    for side_name, side in (("home", "Home"), ("away", "Away")):
        side_matches = [
            match for match in matches if match["side"] == side
        ]

        output[f"{side_name}_matched_rule_count"] = len(side_matches)
        output[f"{side_name}_matched_rule_ids"] = join_text(
            match["rule_id"] for match in side_matches
        )

        strongest_positive = strongest_side(matches, side, "POSITIVE")
        strongest_negative = strongest_side(matches, side, "NEGATIVE")

        for label, item in (
            ("strongest_positive", strongest_positive),
            ("strongest_negative", strongest_negative),
        ):
            output[f"{side_name}_{label}_rule_id"] = (
                item["rule_id"] if item else ""
            )
            output[
                f"{side_name}_{label}_{historical_label}"
            ] = (
                item["historical_rate"] if item else ""
            )
            output[f"{side_name}_{label}_lift_pp"] = (
                item["lift_pp"] if item else ""
            )
            output[f"{side_name}_{label}_games"] = (
                item["games"] if item else ""
            )

    for family, prefix in (
        ("DRAT", "drat"),
        ("EPRED", "epred"),
        ("MARKET", "market"),
        ("DRAT_EPRED_CONSENSUS", "drat_epred_consensus"),
        ("ALL3_CONSENSUS", "all3_consensus"),
    ):
        matches_for_family = family_matches(matches, family)
        output[f"{prefix}_matched_rule_count"] = len(matches_for_family)
        output[f"{prefix}_matched_rule_ids"] = join_text(
            match["rule_id"] for match in matches_for_family
        )

    return output


def build_totals_summary(matches):
    output = {}

    positive = [
        match for match in matches if match["direction"] == "POSITIVE"
    ]
    negative = [
        match for match in matches if match["direction"] == "NEGATIVE"
    ]

    output["matched_rule_count"] = len(matches)
    output["matched_positive_rule_count"] = len(positive)
    output["matched_negative_rule_count"] = len(negative)
    output["matched_rule_ids"] = join_text(
        match["rule_id"] for match in matches
    )
    output["matched_rule_conditions"] = join_text(
        (
            f'{match["rule_id"]}:'
            f'{match["family"]}:'
            f'{match["totals_direction"]}:'
            f'{match["condition"]}'
        )
        for match in matches
    )

    for prefix, totals_direction in (
        ("over", "Over"),
        ("under", "Under"),
    ):
        direction_matches = [
            match
            for match in matches
            if match["totals_direction"] == totals_direction
        ]
        direction_positive = [
            match
            for match in direction_matches
            if match["direction"] == "POSITIVE"
        ]
        direction_negative = [
            match
            for match in direction_matches
            if match["direction"] == "NEGATIVE"
        ]

        output[f"{prefix}_matched_rule_count"] = len(direction_matches)
        output[f"{prefix}_matched_positive_rule_count"] = len(
            direction_positive
        )
        output[f"{prefix}_matched_negative_rule_count"] = len(
            direction_negative
        )
        output[f"{prefix}_matched_rule_ids"] = join_text(
            match["rule_id"] for match in direction_matches
        )

        strongest_positive = strongest_total(
            matches,
            totals_direction,
            "POSITIVE",
        )
        strongest_negative = strongest_total(
            matches,
            totals_direction,
            "NEGATIVE",
        )

        for label, item in (
            ("strongest_positive", strongest_positive),
            ("strongest_negative", strongest_negative),
        ):
            output[f"{prefix}_{label}_rule_id"] = (
                item["rule_id"] if item else ""
            )
            output[f"{prefix}_{label}_hist_hit_rate_pct"] = (
                item["historical_rate"] if item else ""
            )
            output[f"{prefix}_{label}_lift_pp"] = (
                item["lift_pp"] if item else ""
            )
            output[f"{prefix}_{label}_games"] = (
                item["games"] if item else ""
            )

    for family, prefix in (
        ("DRAT", "drat"),
        ("EPRED", "epred"),
        ("MARKET", "market"),
        ("DRAT_EPRED_CONSENSUS", "drat_epred_consensus"),
        ("ALL3_CONSENSUS", "all3_consensus"),
    ):
        matches_for_family = family_matches(matches, family)
        output[f"{prefix}_matched_rule_count"] = len(matches_for_family)
        output[f"{prefix}_matched_rule_ids"] = join_text(
            match["rule_id"] for match in matches_for_family
        )

    return output


def apply_mapping(
    destination: dict[str, object],
    summary: dict[str, object],
    mapping: dict[str, str],
) -> None:
    for combined_name, source_name in mapping.items():
        destination[combined_name] = summary.get(source_name, "")


def main() -> int:
    training = read_csv(TRAINING_PATH)
    require_columns(
        training,
        REQUIRED_TRAINING_COLUMNS,
        "historical training table",
    )

    if training["game_id"].isna().any():
        raise ValueError("historical training table: blank game_id found")

    normalized_ids = training["game_id"].astype("string").str.strip()
    if normalized_ids.eq("").any():
        raise ValueError("historical training table: blank game_id found")

    if normalized_ids.duplicated().any():
        duplicates = sorted(
            normalized_ids.loc[
                normalized_ids.duplicated(keep=False)
            ].unique()
        )
        raise ValueError(
            "historical training table: duplicate game_id values: "
            + ", ".join(duplicates[:20])
        )

    original_row_count = len(training)

    # Make reruns idempotent.
    existing_output_columns = [
        column
        for column in ALL_OUTPUT_COLUMNS
        if column in training.columns
    ]
    if existing_output_columns:
        training = training.drop(columns=existing_output_columns)

    moneyline_rules_df = read_csv(MONEYLINE_RULES_PATH)
    spread_rules_df = read_csv(SPREAD_RULES_PATH)
    totals_rules_df = read_csv(TOTALS_RULES_PATH)

    require_columns(
        moneyline_rules_df,
        COMMON_RULE_COLUMNS + ["historical_win_rate_pct"],
        "moneyline enrichment rules",
    )
    require_columns(
        spread_rules_df,
        COMMON_RULE_COLUMNS + ["historical_cover_rate_pct"],
        "spread enrichment rules",
    )
    require_columns(
        totals_rules_df,
        COMMON_RULE_COLUMNS
        + ["totals_direction", "historical_hit_rate_pct"],
        "totals enrichment rules",
    )

    moneyline_rules = moneyline_rules_df.to_dict("records")
    spread_rules = spread_rules_df.to_dict("records")
    totals_rules = totals_rules_df.to_dict("records")

    output_rows = []

    for row in training.to_dict("records"):
        output = dict(row)
        game = build_game_context(row)
        contexts = build_family_contexts(game)

        moneyline_matches = match_rules(
            game,
            moneyline_rules,
            contexts,
            "moneyline",
        )
        spread_matches = match_rules(
            game,
            spread_rules,
            contexts,
            "spread",
        )
        totals_matches = match_rules(
            game,
            totals_rules,
            contexts,
            "totals",
        )

        moneyline_summary = build_side_summary(
            moneyline_matches,
            "hist_win_rate_pct",
        )
        spread_summary = build_side_summary(
            spread_matches,
            "hist_cover_rate_pct",
        )
        totals_summary = build_totals_summary(totals_matches)

        apply_mapping(output, moneyline_summary, MONEYLINE_MAP)
        apply_mapping(output, spread_summary, ATS_MAP)
        apply_mapping(output, totals_summary, TOTALS_MAP)

        output_rows.append(output)

    enriched = pd.DataFrame(output_rows)

    if len(enriched) != original_row_count:
        raise RuntimeError(
            "Row count changed during Step 4: "
            f"before={original_row_count} after={len(enriched)}"
        )

    require_columns(
        enriched,
        ALL_OUTPUT_COLUMNS,
        "Step 4 output",
    )

    temp_path = TRAINING_PATH.with_suffix(".step4.tmp.csv")
    enriched.to_csv(
        temp_path,
        index=False,
        encoding="utf-8",
    )
    temp_path.replace(TRAINING_PATH)

    print(f"Rows processed: {len(enriched)}")
    print(f"Moneyline enrichment columns added: {len(MONEYLINE_MAP)}")
    print(f"ATS enrichment columns added: {len(ATS_MAP)}")
    print(f"Totals enrichment columns added: {len(TOTALS_MAP)}")
    print(f"Total Step 4 columns added: {len(ALL_OUTPUT_COLUMNS)}")
    print(f"Wrote: {TRAINING_PATH}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

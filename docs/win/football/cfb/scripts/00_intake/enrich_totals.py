#!/usr/bin/env python3
"""
GitHub Actions NFL weekly historical totals-prediction enrichment.

READS ONLY:
  docs/win/football/nfl/00_intake/schedule/weekly/week_{WEEK}_NFL_weekly_schedule.csv
  docs/win/football/nfl/00_intake/predictions/final/*_clean_predictions.csv
  docs/win/football/nfl/00_intake/predictions/drat/clean/{SEASON}_week_{WEEK}_drat.csv
  docs/win/football/nfl/00_intake/odds/{MOST_RECENT_DATE}_NFL_odds.csv
  docs/win/football/nfl/config/prediction_enrichment/totals_enrichment.csv

WRITES ONLY:
  docs/win/football/nfl/00_intake/predictions/enriched/totals/week_{WEEK}_NFL_enriched.csv

The historical bucket boundaries and rule conditions are read from
totals_enrichment.csv. They are not hard-coded here.
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path


MASTER_REL = Path(
    "docs/win/football/nfl/config/prediction_enrichment/totals_enrichment.csv"
)
SCHEDULE_REL = Path(
    "docs/win/football/nfl/00_intake/schedule/weekly"
)
EPRED_REL = Path(
    "docs/win/football/nfl/00_intake/predictions/final"
)
DRAT_REL = Path(
    "docs/win/football/nfl/00_intake/predictions/drat/clean"
)
ODDS_REL = Path(
    "docs/win/football/nfl/00_intake/odds"
)
OUTPUT_REL = Path(
    "docs/win/football/nfl/00_intake/predictions/enriched/totals"
)


def list_weekly_schedule_files(schedule_dir: Path) -> list[Path]:
    files = sorted(
        schedule_dir.glob("week_*_NFL_weekly_schedule.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No week_*_NFL_weekly_schedule.csv files found in "
            f"{schedule_dir}"
        )

    return files


def schedule_identity(
    rows: list[dict[str, str]],
    path: Path,
):
    require_columns(
        rows,
        ["season", "season_type", "week"],
        f"weekly schedule {path.name}",
    )

    values = {
        (
            s(r.get("season")),
            s(r.get("season_type")),
            s(r.get("week")),
        )
        for r in rows
        if (
            s(r.get("season"))
            and s(r.get("season_type"))
            and s(r.get("week"))
        )
    }

    if len(values) != 1:
        raise RuntimeError(
            f"{path.name}: expected exactly one "
            f"season/season_type/week combination, "
            f"found {sorted(values)}"
        )

    season_text, season_type, week_text = next(iter(values))

    return (
        int(float(season_text)),
        season_type,
        int(float(week_text)),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def require_columns(
    rows: list[dict[str, str]],
    required: list[str],
    label: str,
) -> None:
    if not rows:
        raise ValueError(
            f"{label}: file contains no data rows"
        )

    missing = [
        column
        for column in required
        if column not in rows[0]
    ]

    if missing:
        raise ValueError(
            f"{label}: missing columns: "
            + ", ".join(missing)
        )


def s(value) -> str:
    if value is None:
        return ""

    return str(value).strip()


def num(value):
    try:
        text = s(value)

        if text == "":
            return None

        return float(text)

    except (TypeError, ValueError):
        return None


def same_text(a, b) -> bool:
    return s(a).casefold() == s(b).casefold()


def team_key(value) -> str:
    return " ".join(
        s(value).casefold().split()
    )


def game_team_key(
    season,
    week,
    home,
    away,
):
    return (
        str(int(float(season))),
        str(int(float(week))),
        team_key(home),
        team_key(away),
    )


def american_implied(odds):
    value = num(odds)

    if value is None or value == 0:
        return None

    if value > 0:
        return 100.0 / (value + 100.0)

    return (-value) / ((-value) + 100.0)


def no_vig_probs(
    home_ml,
    away_ml,
):
    home = american_implied(home_ml)
    away = american_implied(away_ml)

    if home is None or away is None:
        return None, None

    total = home + away

    if total <= 0:
        return None, None

    return (
        home / total,
        away / total,
    )


def iso_dt(value):
    text = s(value)

    if not text:
        return datetime.min

    try:
        return datetime.fromisoformat(
            text.replace("Z", "+00:00")
        ).replace(tzinfo=None)

    except ValueError:
        return datetime.min


def find_latest_odds_file(
    odds_dir: Path,
) -> Path:
    """
    Select the direct-child *_NFL_odds.csv file
    containing the most recent actual odds update.

    Filename format is irrelevant.
    """

    candidates = []

    for path in odds_dir.glob("*_NFL_odds.csv"):
        try:
            rows = read_csv(path)
        except Exception:
            continue

        latest_update = datetime.min
        found_update = False

        for row in rows:
            dt = iso_dt(
                row.get("last_update")
            )

            if dt != datetime.min:
                found_update = True

                if dt > latest_update:
                    latest_update = dt

        if found_update:
            candidates.append(
                (
                    latest_update,
                    path.name,
                    path,
                )
            )

    if not candidates:
        raise FileNotFoundError(
            f"No *_NFL_odds.csv file with a valid "
            f"last_update value found in {odds_dir}"
        )

    candidates.sort()

    return candidates[-1][2]


def find_drat_file(
    drat_dir: Path,
    season: int,
    week: int,
) -> Path:
    filename = f"{season}_week_{week}_drat.csv"
    path = drat_dir / filename

    if path.exists():
        return path

    raise FileNotFoundError(
        f"DRAT file not found: {path}"
    )


def epred_content_matches(
    path: Path,
    season: int,
    week: int,
    season_type: str,
) -> bool:
    try:
        rows = read_csv(path)
    except Exception:
        return False

    if not rows:
        return False

    for row in rows[:50]:
        if (
            s(row.get("season")) == str(season)
            and s(row.get("week")) == str(week)
            and same_text(
                row.get("season_type"),
                season_type,
            )
        ):
            return True

    return False


def find_epred_file(
    epred_dir: Path,
    season: int,
    week: int,
    season_type: str,
) -> Path:
    candidates = [
        path
        for path in epred_dir.glob(
            "*_clean_predictions.csv"
        )
        if epred_content_matches(
            path,
            season,
            week,
            season_type,
        )
    ]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No *_clean_predictions.csv containing "
            f"season={season}, "
            f"season_type={season_type}, "
            f"week={week} found in {epred_dir}"
        )

    raise RuntimeError(
        "More than one EPRED file matches "
        "this season/week: "
        + ", ".join(
            path.name
            for path in candidates
        )
    )


def aggregate_latest_odds(rows):
    """
    Aggregate duplicated market/side rows into one
    current market record per provider game + bookmaker.

    Each populated market field is taken from the most
    recent last_update carrying that field.
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

    for row in rows:
        game_id = s(
            row.get("game_id")
        )
        bookmaker = s(
            row.get("bookmaker")
        )

        if not game_id:
            continue

        key = (
            game_id,
            bookmaker.casefold(),
        )

        group = groups.setdefault(
            key,
            {
                "game_id": game_id,
                "bookmaker": bookmaker,
                "last_update": "",
                "__last_dt": datetime.min,
                "__field_dt": {
                    field: datetime.min
                    for field in fields
                },
            },
        )

        dt = iso_dt(
            row.get("last_update")
        )

        if dt >= group["__last_dt"]:
            group["__last_dt"] = dt
            group["last_update"] = s(
                row.get("last_update")
            )

        for field in fields:
            value = s(
                row.get(field)
            )

            if (
                value != ""
                and dt
                >= group["__field_dt"][field]
            ):
                group[field] = value
                group["__field_dt"][field] = dt

    return groups


def choose_odds_record(
    groups,
    provider_game_id,
    preferred_bookmaker,
):
    game_id = s(
        provider_game_id
    )

    if not game_id:
        return None

    preferred = s(
        preferred_bookmaker
    ).casefold()

    if (
        preferred
        and (game_id, preferred)
        in groups
    ):
        return groups[
            (game_id, preferred)
        ]

    matches = [
        group
        for (group_game_id, _), group
        in groups.items()
        if group_game_id == game_id
    ]

    if not matches:
        return None

    return max(
        matches,
        key=lambda group: group["__last_dt"],
    )


def build_family_contexts(g):
    contexts = {}

    drat_home = num(
        g.get("drat_home_prob")
    )
    drat_away = num(
        g.get("drat_away_prob")
    )

    epred_home = num(
        g.get("epred_home_prob")
    )
    epred_away = num(
        g.get("epred_away_prob")
    )

    market_home = num(
        g.get("market_home_prob_novig")
    )
    market_away = num(
        g.get("market_away_prob_novig")
    )

    home_spread = num(
        g.get("market_home_spread")
    )
    away_spread = num(
        g.get("market_away_spread")
    )

    drat_side = None
    drat_prob = None

    if (
        drat_home is not None
        and drat_away is not None
    ):
        if drat_home >= drat_away:
            drat_side = "Home"
            drat_prob = drat_home
        else:
            drat_side = "Away"
            drat_prob = drat_away

    epred_side = None
    epred_prob = None

    if (
        epred_home is not None
        and epred_away is not None
    ):
        if epred_home >= epred_away:
            epred_side = "Home"
            epred_prob = epred_home
        else:
            epred_side = "Away"
            epred_prob = epred_away

    market_side = None
    market_prob = None

    if (
        market_home is not None
        and market_away is not None
    ):
        if market_home >= market_away:
            market_side = "Home"
            market_prob = market_home
        else:
            market_side = "Away"
            market_prob = market_away

    elif (
        home_spread is not None
        and away_spread is not None
    ):
        if home_spread < 0:
            market_side = "Home"

        elif away_spread < 0:
            market_side = "Away"

    contexts["DRAT"] = {
        "eligible": (
            drat_side is not None
        ),
        "side": drat_side,
        "prob": drat_prob,
    }

    contexts["EPRED"] = {
        "eligible": (
            epred_side is not None
        ),
        "side": epred_side,
        "prob": epred_prob,
    }

    contexts["MARKET"] = {
        "eligible": (
            market_side is not None
        ),
        "side": market_side,
        "prob": market_prob,
    }

    drat_epred_ok = (
        drat_side is not None
        and epred_side is not None
        and drat_side == epred_side
    )

    contexts[
        "DRAT_EPRED_CONSENSUS"
    ] = {
        "eligible": drat_epred_ok,
        "side": (
            drat_side
            if drat_epred_ok
            else None
        ),
        "prob": (
            (
                drat_prob
                + epred_prob
            )
            / 2.0
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

    contexts[
        "ALL3_CONSENSUS"
    ] = {
        "eligible": all_three_ok,
        "side": (
            drat_side
            if all_three_ok
            else None
        ),
        "prob": (
            (
                drat_prob
                + epred_prob
                + market_prob
            )
            / 3.0
            if (
                all_three_ok
                and market_prob is not None
            )
            else None
        ),
    }

    g["drat_pick_side"] = (
        drat_side or ""
    )
    g["epred_pick_side"] = (
        epred_side or ""
    )
    g["market_pick_side"] = (
        market_side or ""
    )

    g["drat_pick"] = (
        g["home_team"]
        if drat_side == "Home"
        else (
            g["away_team"]
            if drat_side == "Away"
            else ""
        )
    )

    g["epred_pick"] = (
        g["home_team"]
        if epred_side == "Home"
        else (
            g["away_team"]
            if epred_side == "Away"
            else ""
        )
    )

    g["market_pick"] = (
        g["home_team"]
        if market_side == "Home"
        else (
            g["away_team"]
            if market_side == "Away"
            else ""
        )
    )

    def agreement(a, b):
        if a is None or b is None:
            return "Unknown"

        if a == b:
            return "Agree"

        return "Disagree"

    g["drat_epred_agree"] = agreement(
        drat_side,
        epred_side,
    )

    g["drat_market_agree"] = agreement(
        drat_side,
        market_side,
    )

    g["epred_market_agree"] = agreement(
        epred_side,
        market_side,
    )

    if all_three_ok:
        g["all_three_agree"] = "Yes"

    elif (
        drat_side
        and epred_side
        and market_side
    ):
        g["all_three_agree"] = "No"

    else:
        g["all_three_agree"] = "Unknown"

    return contexts


def market_role_for_side(
    g,
    side,
):
    if side not in (
        "Home",
        "Away",
    ):
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
    formula_code,
    g,
    family_ctx,
):
    side = family_ctx["side"]

    if (
        formula_code
        == "USE_FAMILY_SELECTED_PROB"
    ):
        return family_ctx["prob"]

    if (
        formula_code
        == "MARKET_ROLE_FOR_FAMILY_SELECTED_SIDE"
    ):
        return market_role_for_side(
            g,
            side,
        )

    if (
        formula_code
        == "SPREAD_FOR_FAMILY_SELECTED_SIDE"
    ):
        if side == "Home":
            return num(
                g.get("market_home_spread")
            )

        if side == "Away":
            return num(
                g.get("market_away_spread")
            )

        return None

    if (
        formula_code
        == "EPRED_RATING_SELECTED_MINUS_OPPONENT"
    ):
        home_rating = num(
            g.get("epred_home_rating")
        )
        away_rating = num(
            g.get("epred_away_rating")
        )

        if (
            home_rating is None
            or away_rating is None
            or side is None
        ):
            return None

        if side == "Home":
            return (
                home_rating
                - away_rating
            )

        return (
            away_rating
            - home_rating
        )

    if (
        formula_code
        == "RAW_EPRED_MATCHUP_QUALITY"
    ):
        return num(
            g.get("epred_matchupQuality")
        )

    if (
        formula_code
        == "RAW_WEEK"
    ):
        return num(
            g.get("week")
        )

    if (
        formula_code
        == "RAW_MARKET_TOTAL"
    ):
        return num(
            g.get("market_total")
        )

    if (
        formula_code
        == "COMPARE_DRAT_PICK_TO_EPRED_PICK"
    ):
        drat_side = g.get(
            "drat_pick_side"
        )
        epred_side = g.get(
            "epred_pick_side"
        )

        if (
            not drat_side
            or not epred_side
        ):
            return None

        if drat_side == epred_side:
            return "Agree"

        return "Disagree"

    if (
        formula_code
        == "COMPARE_FAMILY_PICK_TO_MARKET_PICK"
    ):
        market_side = g.get(
            "market_pick_side"
        )

        if (
            not side
            or not market_side
        ):
            return None

        if side == market_side:
            return "Agree"

        return "Disagree"

    if (
        formula_code
        == "ABS_DRAT_HOME_PROB_MINUS_EPRED_NORMALIZED_HOME_PROB_X100"
    ):
        drat_home = num(
            g.get("drat_home_prob")
        )
        epred_home = num(
            g.get("epred_home_prob")
        )

        if (
            drat_home is None
            or epred_home is None
        ):
            return None

        return abs(
            drat_home
            - epred_home
        ) * 100.0

    if (
        formula_code
        == "FAMILY_SELECTED_PROB_MINUS_MARKET_SELECTED_PROB_X100"
    ):
        family_prob = family_ctx[
            "prob"
        ]

        if side == "Home":
            market_prob = num(
                g.get(
                    "market_home_prob_novig"
                )
            )

        elif side == "Away":
            market_prob = num(
                g.get(
                    "market_away_prob_novig"
                )
            )

        else:
            market_prob = None

        if (
            family_prob is None
            or market_prob is None
        ):
            return None

        return (
            family_prob
            - market_prob
        ) * 100.0

    if formula_code == "UNAVAILABLE":
        return None

    raise ValueError(
        f"Unsupported formula_code in master: "
        f"{formula_code}"
    )


def condition_matches(
    rule,
    n,
    value,
):
    prefix = f"condition_{n}_"

    match_type = s(
        rule.get(
            prefix + "match_type"
        )
    )

    if not match_type:
        return True

    if match_type == "IS_NULL":
        return (
            value is None
            or s(value) == ""
        )

    if match_type == "TEXT_EQUALS":
        return (
            s(value)
            == s(
                rule.get(
                    prefix
                    + "equals_value"
                )
            )
        )

    if match_type == "NUMERIC_RANGE":
        numeric_value = num(value)

        if numeric_value is None:
            return False

        minimum = num(
            rule.get(
                prefix
                + "min_inclusive"
            )
        )

        maximum = num(
            rule.get(
                prefix
                + "max_exclusive"
            )
        )

        if (
            minimum is not None
            and numeric_value < minimum
        ):
            return False

        if (
            maximum is not None
            and numeric_value >= maximum
        ):
            return False

        return True

    raise ValueError(
        f"Unsupported match_type in master: "
        f"{match_type}"
    )


def match_rules(
    g,
    master_rows,
    contexts,
):
    matches = []

    for rule in master_rows:
        if (
            s(rule.get("active"))
            != "1"
        ):
            continue

        if (
            s(
                rule.get(
                    "pipeline_supported"
                )
            )
            != "1"
        ):
            continue

        family = s(
            rule.get("family")
        )

        family_ctx = contexts.get(
            family
        )

        if (
            not family_ctx
            or not family_ctx["eligible"]
        ):
            continue

        condition_count = int(
            float(
                s(
                    rule.get(
                        "condition_count"
                    )
                )
                or "0"
            )
        )

        matched = True

        for condition_number in range(
            1,
            condition_count + 1,
        ):
            formula_code = s(
                rule.get(
                    f"condition_"
                    f"{condition_number}_"
                    f"formula_code"
                )
            )

            value = feature_value(
                formula_code,
                g,
                family_ctx,
            )

            if not condition_matches(
                rule,
                condition_number,
                value,
            ):
                matched = False
                break

        if not matched:
            continue

        matches.append(
            {
                "rule_id": s(
                    rule.get("rule_id")
                ),
                "family": family,
                "side": family_ctx[
                    "side"
                ],
                "totals_direction": s(
                    rule.get(
                        "totals_direction"
                    )
                ),
                "condition": s(
                    rule.get(
                        "source_condition"
                    )
                ),
                "historical_hit_rate_pct": num(
                    rule.get(
                        "historical_hit_rate_pct"
                    )
                ),
                "lift_pp": num(
                    rule.get(
                        "lift_vs_family_pct_points"
                    )
                ),
                "direction": s(
                    rule.get(
                        "action_direction"
                    )
                ),
                "games": num(
                    rule.get("games")
                ),
            }
        )

    return matches


def strongest(
    matches,
    totals_direction,
    action_direction,
):
    candidates = [
        match
        for match in matches
        if (
            match["totals_direction"]
            == totals_direction
            and match["direction"]
            == action_direction
            and match["lift_pp"]
            is not None
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda match: abs(
            match["lift_pp"]
        ),
    )


def family_matches(
    matches,
    family,
):
    return [
        match
        for match in matches
        if match["family"] == family
    ]


def join_text(values):
    return ";".join(
        s(value)
        for value in values
        if s(value)
    )


def build_summary_fields(
    g,
    matches,
):
    positive_matches = [
        match
        for match in matches
        if (
            match["direction"]
            == "POSITIVE"
        )
    ]

    negative_matches = [
        match
        for match in matches
        if (
            match["direction"]
            == "NEGATIVE"
        )
    ]

    g["matched_rule_count"] = len(
        matches
    )

    g[
        "matched_positive_rule_count"
    ] = len(
        positive_matches
    )

    g[
        "matched_negative_rule_count"
    ] = len(
        negative_matches
    )

    g["matched_rule_ids"] = join_text(
        match["rule_id"]
        for match in matches
    )

    g[
        "matched_rule_conditions"
    ] = join_text(
        (
            f'{match["rule_id"]}:'
            f'{match["family"]}:'
            f'{match["totals_direction"]}:'
            f'{match["condition"]}'
        )
        for match in matches
    )

    for (
        prefix,
        totals_direction,
    ) in [
        ("over", "Over"),
        ("under", "Under"),
    ]:
        direction_matches = [
            match
            for match in matches
            if (
                match[
                    "totals_direction"
                ]
                == totals_direction
            )
        ]

        direction_positive = [
            match
            for match
            in direction_matches
            if (
                match["direction"]
                == "POSITIVE"
            )
        ]

        direction_negative = [
            match
            for match
            in direction_matches
            if (
                match["direction"]
                == "NEGATIVE"
            )
        ]

        g[
            f"{prefix}_"
            f"matched_rule_count"
        ] = len(
            direction_matches
        )

        g[
            f"{prefix}_"
            f"matched_positive_"
            f"rule_count"
        ] = len(
            direction_positive
        )

        g[
            f"{prefix}_"
            f"matched_negative_"
            f"rule_count"
        ] = len(
            direction_negative
        )

        g[
            f"{prefix}_"
            f"matched_rule_ids"
        ] = join_text(
            match["rule_id"]
            for match
            in direction_matches
        )

        strongest_positive = strongest(
            matches,
            totals_direction,
            "POSITIVE",
        )

        strongest_negative = strongest(
            matches,
            totals_direction,
            "NEGATIVE",
        )

        for (
            label,
            item,
        ) in [
            (
                "strongest_positive",
                strongest_positive,
            ),
            (
                "strongest_negative",
                strongest_negative,
            ),
        ]:
            g[
                f"{prefix}_{label}_"
                f"rule_id"
            ] = (
                item["rule_id"]
                if item
                else ""
            )

            g[
                f"{prefix}_{label}_"
                f"hist_hit_rate_pct"
            ] = (
                item[
                    "historical_hit_rate_pct"
                ]
                if item
                else ""
            )

            g[
                f"{prefix}_{label}_"
                f"lift_pp"
            ] = (
                item["lift_pp"]
                if item
                else ""
            )

            g[
                f"{prefix}_{label}_"
                f"games"
            ] = (
                item["games"]
                if item
                else ""
            )

    family_prefixes = [
        (
            "DRAT",
            "drat",
        ),
        (
            "EPRED",
            "epred",
        ),
        (
            "MARKET",
            "market",
        ),
        (
            "DRAT_EPRED_CONSENSUS",
            "drat_epred_consensus",
        ),
        (
            "ALL3_CONSENSUS",
            "all3_consensus",
        ),
    ]

    for (
        family,
        prefix,
    ) in family_prefixes:
        matches_for_family = (
            family_matches(
                matches,
                family,
            )
        )

        g[
            f"{prefix}_"
            f"matched_rule_count"
        ] = len(
            matches_for_family
        )

        g[
            f"{prefix}_"
            f"matched_rule_ids"
        ] = join_text(
            match["rule_id"]
            for match
            in matches_for_family
        )

    return g


def process_week(
    root: Path,
    schedule_path: Path,
    odds_rows: list[dict[str, str]],
    master: list[dict[str, str]],
):
    schedule = read_csv(
        schedule_path
    )

    (
        season,
        season_type,
        week,
    ) = schedule_identity(
        schedule,
        schedule_path,
    )

    drat_path = find_drat_file(
        root / DRAT_REL,
        season,
        week,
    )

    epred_path = find_epred_file(
        root / EPRED_REL,
        season,
        week,
        season_type,
    )

    output_path = (
        root
        / OUTPUT_REL
        / f"week_{week}_NFL_enriched.csv"
    )

    drat = read_csv(
        drat_path
    )

    epred = read_csv(
        epred_path
    )

    require_columns(
        schedule,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "odds_provider_game_id",
            "away_team",
            "home_team",
            "bookmaker",
            "home_moneyline_american",
            "away_moneyline_american",
            "home_spread",
            "away_spread",
            "total",
        ],
        (
            f"weekly schedule "
            f"{schedule_path.name}"
        ),
    )

    require_columns(
        drat,
        [
            "season",
            "week",
            "home_team",
            "away_team",
            "home_prob",
            "away_prob",
        ],
        f"DRAT {drat_path.name}",
    )

    require_columns(
        epred,
        [
            "game_id",
            "home_team",
            "away_team",
            "home_prob",
            "away_prob",
            "home_rating",
            "away_rating",
            "matchupQuality",
        ],
        f"EPRED {epred_path.name}",
    )

    epred_by_game = {
        s(row["game_id"]): row
        for row in epred
        if s(row.get("game_id"))
    }

    drat_by_teams = {}

    for row in drat:
        try:
            key = game_team_key(
                row.get("season"),
                row.get("week"),
                row.get("home_team"),
                row.get("away_team"),
            )

            drat_by_teams[key] = row

        except Exception:
            continue

    current_odds = aggregate_latest_odds(
        odds_rows
    )

    output_rows = []
    missing_epred = []
    missing_drat = []

    for base in schedule:
        g = dict(base)

        epred_row = epred_by_game.get(
            s(base.get("game_id"))
        )

        if epred_row is None:
            missing_epred.append(
                s(base.get("game_id"))
            )

        try:
            drat_key = game_team_key(
                base.get("season"),
                base.get("week"),
                base.get("home_team"),
                base.get("away_team"),
            )

            drat_row = drat_by_teams.get(
                drat_key
            )

        except Exception:
            drat_row = None

        if drat_row is None:
            missing_drat.append(
                s(base.get("game_id"))
            )

        odds_record = choose_odds_record(
            current_odds,
            base.get(
                "odds_provider_game_id"
            ),
            base.get("bookmaker"),
        )

        g["drat_home_prob"] = (
            num(
                drat_row.get(
                    "home_prob"
                )
            )
            if drat_row
            else ""
        )

        g["drat_away_prob"] = (
            num(
                drat_row.get(
                    "away_prob"
                )
            )
            if drat_row
            else ""
        )

        epred_home_raw = (
            num(
                epred_row.get(
                    "home_prob"
                )
            )
            if epred_row
            else None
        )

        epred_away_raw = (
            num(
                epred_row.get(
                    "away_prob"
                )
            )
            if epred_row
            else None
        )

        g["epred_home_prob_raw"] = (
            epred_home_raw
            if epred_home_raw
            is not None
            else ""
        )

        g["epred_away_prob_raw"] = (
            epred_away_raw
            if epred_away_raw
            is not None
            else ""
        )

        epred_probability_sum = (
            epred_home_raw
            + epred_away_raw
            if (
                epred_home_raw
                is not None
                and epred_away_raw
                is not None
            )
            else None
        )

        g["epred_home_prob"] = (
            epred_home_raw
            / epred_probability_sum
            if (
                epred_probability_sum
                is not None
                and epred_probability_sum > 0
            )
            else ""
        )

        g["epred_away_prob"] = (
            epred_away_raw
            / epred_probability_sum
            if (
                epred_probability_sum
                is not None
                and epred_probability_sum > 0
            )
            else ""
        )

        g["epred_home_rating"] = (
            num(
                epred_row.get(
                    "home_rating"
                )
            )
            if epred_row
            else ""
        )

        g["epred_away_rating"] = (
            num(
                epred_row.get(
                    "away_rating"
                )
            )
            if epred_row
            else ""
        )

        g["epred_matchupQuality"] = (
            num(
                epred_row.get(
                    "matchupQuality"
                )
            )
            if epred_row
            else ""
        )

        def market_value(field):
            if (
                odds_record
                and s(
                    odds_record.get(
                        field
                    )
                )
                != ""
            ):
                return odds_record.get(
                    field
                )

            return base.get(
                field,
                "",
            )

        g["market_bookmaker"] = (
            odds_record.get(
                "bookmaker"
            )
            if odds_record
            else base.get(
                "bookmaker",
                "",
            )
        )

        g["market_last_update"] = (
            odds_record.get(
                "last_update"
            )
            if odds_record
            else ""
        )

        g[
            "market_home_moneyline_american"
        ] = market_value(
            "home_moneyline_american"
        )

        g[
            "market_away_moneyline_american"
        ] = market_value(
            "away_moneyline_american"
        )

        g["market_home_spread"] = (
            market_value(
                "home_spread"
            )
        )

        g["market_away_spread"] = (
            market_value(
                "away_spread"
            )
        )

        g["market_total"] = (
            market_value(
                "total"
            )
        )

        (
            market_home_prob,
            market_away_prob,
        ) = no_vig_probs(
            g[
                "market_home_moneyline_american"
            ],
            g[
                "market_away_moneyline_american"
            ],
        )

        g[
            "market_home_prob_novig"
        ] = (
            market_home_prob
            if market_home_prob
            is not None
            else ""
        )

        g[
            "market_away_prob_novig"
        ] = (
            market_away_prob
            if market_away_prob
            is not None
            else ""
        )

        contexts = build_family_contexts(
            g
        )

        epred_home_rating = num(
            g.get(
                "epred_home_rating"
            )
        )

        epred_away_rating = num(
            g.get(
                "epred_away_rating"
            )
        )

        g[
            "epred_rating_gap_home"
        ] = (
            epred_home_rating
            - epred_away_rating
            if (
                epred_home_rating
                is not None
                and epred_away_rating
                is not None
            )
            else ""
        )

        drat_home_prob = num(
            g.get(
                "drat_home_prob"
            )
        )

        epred_home_prob = num(
            g.get(
                "epred_home_prob"
            )
        )

        g[
            "drat_epred_prob_diff_pp"
        ] = (
            abs(
                drat_home_prob
                - epred_home_prob
            )
            * 100.0
            if (
                drat_home_prob
                is not None
                and epred_home_prob
                is not None
            )
            else ""
        )

        g[
            "drat_market_edge_home_pp"
        ] = (
            (
                drat_home_prob
                - market_home_prob
            )
            * 100.0
            if (
                drat_home_prob
                is not None
                and market_home_prob
                is not None
            )
            else ""
        )

        g[
            "epred_market_edge_home_pp"
        ] = (
            (
                epred_home_prob
                - market_home_prob
            )
            * 100.0
            if (
                epred_home_prob
                is not None
                and market_home_prob
                is not None
            )
            else ""
        )

        matches = match_rules(
            g,
            master,
            contexts,
        )

        build_summary_fields(
            g,
            matches,
        )

        output_rows.append(g)

    if not output_rows:
        raise RuntimeError(
            f"{schedule_path.name}: "
            f"no rows to write"
        )

    base_fields = list(
        schedule[0].keys()
    )

    appended_fields = [
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
        "over_matched_rule_count",
        "over_matched_positive_rule_count",
        "over_matched_negative_rule_count",
        "over_matched_rule_ids",
        "over_strongest_positive_rule_id",
        "over_strongest_positive_hist_hit_rate_pct",
        "over_strongest_positive_lift_pp",
        "over_strongest_positive_games",
        "over_strongest_negative_rule_id",
        "over_strongest_negative_hist_hit_rate_pct",
        "over_strongest_negative_lift_pp",
        "over_strongest_negative_games",
        "under_matched_rule_count",
        "under_matched_positive_rule_count",
        "under_matched_negative_rule_count",
        "under_matched_rule_ids",
        "under_strongest_positive_rule_id",
        "under_strongest_positive_hist_hit_rate_pct",
        "under_strongest_positive_lift_pp",
        "under_strongest_positive_games",
        "under_strongest_negative_rule_id",
        "under_strongest_negative_hist_hit_rate_pct",
        "under_strongest_negative_lift_pp",
        "under_strongest_negative_games",
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

    fieldnames = (
        base_fields
        + [
            field
            for field in appended_fields
            if field not in base_fields
        ]
    )

    write_csv(
        output_path,
        output_rows,
        fieldnames,
    )

    return {
        "season": season,
        "season_type": season_type,
        "week": week,
        "schedule": schedule_path.name,
        "drat": drat_path.name,
        "epred": epred_path.name,
        "output": str(
            output_path
        ),
        "games": len(
            output_rows
        ),
        "missing_epred": len(
            missing_epred
        ),
        "missing_drat": len(
            missing_drat
        ),
    }


def validate_master(
    master,
):
    supported_families = {
        "DRAT",
        "EPRED",
        "MARKET",
        "DRAT_EPRED_CONSENSUS",
        "ALL3_CONSENSUS",
    }

    supported_totals_directions = {
        "Over",
        "Under",
    }

    supported_action_directions = {
        "POSITIVE",
        "NEGATIVE",
    }

    supported_match_types = {
        "",
        "IS_NULL",
        "TEXT_EQUALS",
        "NUMERIC_RANGE",
    }

    supported_formula_codes = {
        "USE_FAMILY_SELECTED_PROB",
        "MARKET_ROLE_FOR_FAMILY_SELECTED_SIDE",
        "SPREAD_FOR_FAMILY_SELECTED_SIDE",
        "EPRED_RATING_SELECTED_MINUS_OPPONENT",
        "RAW_EPRED_MATCHUP_QUALITY",
        "RAW_WEEK",
        "RAW_MARKET_TOTAL",
        "COMPARE_DRAT_PICK_TO_EPRED_PICK",
        "COMPARE_FAMILY_PICK_TO_MARKET_PICK",
        "ABS_DRAT_HOME_PROB_MINUS_EPRED_NORMALIZED_HOME_PROB_X100",
        "FAMILY_SELECTED_PROB_MINUS_MARKET_SELECTED_PROB_X100",
        "UNAVAILABLE",
    }

    seen_rule_ids = set()

    for row_number, rule in enumerate(
        master,
        start=2,
    ):
        rule_id = s(
            rule.get("rule_id")
        )

        if not rule_id:
            raise ValueError(
                f"totals enrichment master row "
                f"{row_number}: blank rule_id"
            )

        if rule_id in seen_rule_ids:
            raise ValueError(
                f"totals enrichment master: "
                f"duplicate rule_id {rule_id}"
            )

        seen_rule_ids.add(
            rule_id
        )

        if s(
            rule.get("active")
        ) != "1":
            continue

        if s(
            rule.get(
                "pipeline_supported"
            )
        ) != "1":
            continue

        family = s(
            rule.get("family")
        )

        if family not in supported_families:
            raise ValueError(
                f"{rule_id}: unsupported family "
                f"{family}"
            )

        totals_direction = s(
            rule.get(
                "totals_direction"
            )
        )

        if (
            totals_direction
            not in supported_totals_directions
        ):
            raise ValueError(
                f"{rule_id}: unsupported "
                f"totals_direction "
                f"{totals_direction}"
            )

        action_direction = s(
            rule.get(
                "action_direction"
            )
        )

        if (
            action_direction
            not in supported_action_directions
        ):
            raise ValueError(
                f"{rule_id}: unsupported "
                f"action_direction "
                f"{action_direction}"
            )

        condition_count = int(
            float(
                s(
                    rule.get(
                        "condition_count"
                    )
                )
                or "0"
            )
        )

        if condition_count not in (
            1,
            2,
        ):
            raise ValueError(
                f"{rule_id}: unsupported "
                f"condition_count "
                f"{condition_count}"
            )

        for condition_number in range(
            1,
            condition_count + 1,
        ):
            formula_code = s(
                rule.get(
                    f"condition_"
                    f"{condition_number}_"
                    f"formula_code"
                )
            )

            if (
                formula_code
                not in supported_formula_codes
            ):
                raise ValueError(
                    f"{rule_id}: unsupported "
                    f"formula_code "
                    f"{formula_code}"
                )

            match_type = s(
                rule.get(
                    f"condition_"
                    f"{condition_number}_"
                    f"match_type"
                )
            )

            if (
                match_type
                not in supported_match_types
            ):
                raise ValueError(
                    f"{rule_id}: unsupported "
                    f"match_type "
                    f"{match_type}"
                )


def main():
    workspace = os.environ.get(
        "GITHUB_WORKSPACE",
        "",
    ).strip()

    if not workspace:
        raise RuntimeError(
            "GITHUB_WORKSPACE is not set. "
            "This script is intended to run "
            "inside GitHub Actions."
        )

    root = Path(
        workspace
    ).resolve()

    master_path = (
        root
        / MASTER_REL
    )

    master = read_csv(
        master_path
    )

    require_columns(
        master,
        [
            "rule_id",
            "active",
            "pipeline_supported",
            "family",
            "condition_count",
            "condition_1_formula_code",
            "condition_1_match_type",
            "totals_direction",
            "historical_hit_rate_pct",
            "lift_vs_family_pct_points",
            "action_direction",
        ],
        (
            "historical totals "
            "enrichment master"
        ),
    )

    validate_master(
        master
    )

    odds_path = (
        find_latest_odds_file(
            root / ODDS_REL
        )
    )

    odds = read_csv(
        odds_path
    )

    require_columns(
        odds,
        [
            "game_id",
            "bookmaker",
            "last_update",
            "home_moneyline_american",
            "away_moneyline_american",
            "home_spread",
            "away_spread",
            "total",
        ],
        f"latest odds {odds_path.name}",
    )

    schedule_files = (
        list_weekly_schedule_files(
            root / SCHEDULE_REL
        )
    )

    completed = []
    failures = []

    for schedule_path in schedule_files:
        try:
            result = process_week(
                root,
                schedule_path,
                odds,
                master,
            )

            completed.append(
                result
            )

        except Exception as exc:
            failures.append(
                f"{schedule_path.name}: "
                f"{exc}"
            )

    if failures:
        raise RuntimeError(
            " | ".join(
                failures
            )
        )

    if not completed:
        raise RuntimeError(
            "No weekly schedule had "
            "all required matching "
            "DRAT and EPRED inputs."
        )

    print(
        f"Historical totals master: "
        f"{master_path}"
    )

    print(
        f"Latest odds file: "
        f"{odds_path}"
    )

    print(
        f"Weeks enriched: "
        f"{len(completed)}"
    )

    for result in completed:
        print(
            f"week {result['week']} -> "
            f"{result['output']} "
            f"(games={result['games']}, "
            f"missing_epred="
            f"{result['missing_epred']}, "
            f"missing_drat="
            f"{result['missing_drat']})"
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

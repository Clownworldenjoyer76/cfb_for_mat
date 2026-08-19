#!/usr/bin/env python3
"""
projection_week1.py

CFB Week 1 cold-start final-score projection.

Purpose
-------
Produce Week 1 projected margin, projected total, and projected final scores
without using any current-season PBP.

Inputs
------
1. docs/win/football/cfb/00_intake/schedule/weekly/week_{week}_CFB_weekly_schedule.csv
2. docs/win/football/cfb/00_intake/team_stats/{prior_season}_team_stats.csv
3. docs/win/football/cfb/data/team_power_index/team_power_index_{season}.csv
4. docs/win/football/cfb/00_intake/injuries/{season}_injuries.csv
5. docs/win/football/cfb/config/mapping/team_map.csv

Output
------
docs/win/football/cfb/01_merge/week_{week}_CFB_enriched.csv

Method
------
- The prior-season team-stats file is weekly, not cumulative. This script
  aggregates every prior-season team-week into one season prior and shrinks
  small samples toward the national mean.
- A prior-strength rating is built from EPA, points/drive, success rate,
  yards/play, red-zone rate, early-down EPA, and third-down conversion rate.
- The prior rating is scaled to the current ESPN FPI rating distribution so
  it is expressed on a point-like scale.
- Projected margin blends:
      45% current market spread
      35% current ESPN FPI margin
      20% prior-season team-strength margin
  Available components are automatically reweighted if one is missing.
- Projected total blends:
      75% current market total
      25% prior-season points-per-drive estimate
  If the market total is missing, the prior estimate is used.
- Injury rows only affect the projection when their report_date is fresh
  relative to the scheduled game. Stale historical injury rows are ignored.
- No current-season game result or current-season PBP is read.

This is an auditable Week 1 baseline. All component projections and blend
weights are written to the output so later backtesting can replace the fixed
weights with trained weights.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_VERSION = "cfb-week1-v1-2026-08-19"

TEAM_METRICS = [
    "off_epa_per_play",
    "def_epa_per_play",
    "off_success_rate",
    "def_success_rate",
    "yards_per_play",
    "yards_per_play_allowed",
    "points_per_drive",
    "points_per_drive_allowed",
    "red_zone_td_rate",
    "red_zone_td_rate_allowed",
    "early_down_epa",
    "third_down_conversion_rate",
]

OUTPUT_BASE_COLUMNS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "neutral_site",
    "stadium",
    "roof",
    "surface",
    "bookmaker",
    "home_spread",
    "total",
]

OUT_STATUS_MULTIPLIER = {
    "out": 1.00,
    "doubtful": 0.75,
    "questionable": 0.25,
}

POSITION_POINT_COST = {
    "QB": 3.00,
    "OL": 0.60,
    "OT": 0.60,
    "LT": 0.60,
    "RT": 0.60,
    "OG": 0.60,
    "LG": 0.60,
    "RG": 0.60,
    "C": 0.60,
    "WR": 0.50,
    "RB": 0.50,
    "HB": 0.50,
    "FB": 0.30,
    "TE": 0.45,
    "DE": 0.40,
    "DT": 0.40,
    "DL": 0.40,
    "NT": 0.40,
    "EDGE": 0.45,
    "LB": 0.40,
    "ILB": 0.40,
    "OLB": 0.40,
    "MLB": 0.40,
    "CB": 0.40,
    "DB": 0.35,
    "S": 0.35,
    "FS": 0.35,
    "SS": 0.35,
    "K": 0.20,
    "PK": 0.20,
    "P": 0.15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CFB Week 1 cold-start final-score projections."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Target season. Defaults to CFB_SEASON, then 2026.",
    )
    parser.add_argument(
        "--prior-season",
        type=int,
        default=None,
        help="Prior season used for team-strength priors. Defaults to season-1.",
    )
    parser.add_argument("--week", type=int, default=1)
    parser.add_argument("--home-field", type=float, default=2.5)
    parser.add_argument("--drives-per-team", type=float, default=11.5)
    parser.add_argument("--market-margin-weight", type=float, default=0.45)
    parser.add_argument("--fpi-margin-weight", type=float, default=0.35)
    parser.add_argument("--prior-margin-weight", type=float, default=0.20)
    parser.add_argument("--market-total-weight", type=float, default=0.75)
    parser.add_argument("--fresh-injury-days", type=int, default=60)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate projections but do not write output.",
    )
    return parser.parse_args()


def clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def normalize_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", clean(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def as_bool(value: object) -> bool:
    text = clean(value).casefold()
    return text in {"1", "true", "t", "yes", "y"}


def as_float(value: object) -> float | None:
    text = clean(value).replace(",", "").replace("%", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def repo_cfb_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "docs" / "win" / "football" / "cfb"
        if candidate.is_dir():
            return candidate

    # Expected installed location:
    # docs/win/football/cfb/scripts/01_merge/projection_week1.py
    try:
        candidate = here.parents[2]
    except IndexError as exc:
        raise RuntimeError(f"Cannot resolve CFB root from {here}") from exc

    if candidate.name != "cfb":
        raise RuntimeError(f"Cannot resolve CFB root from {here}")
    return candidate


def read_csv(path: Path, required: list[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
    return df


class TeamResolver:
    def __init__(self, team_map: pd.DataFrame) -> None:
        required = ["team_id", "canonical_team"]
        missing = [c for c in required if c not in team_map.columns]
        if missing:
            raise ValueError(f"team_map.csv missing required columns: {missing}")

        self.alias_to_team: dict[str, str] = {}
        self.team_to_id: dict[str, str] = {}

        alias_columns = [
            "canonical_team",
            "alias",
            "location",
            "nickname",
            "shortDisplayName",
            "team_slug",
        ]

        for _, row in team_map.iterrows():
            canonical = clean(row.get("canonical_team"))
            team_id = clean(row.get("team_id"))
            if not canonical:
                continue

            self.team_to_id.setdefault(canonical, team_id)

            values: list[str] = [canonical]
            for column in alias_columns:
                if column in team_map.columns:
                    value = clean(row.get(column))
                    if value:
                        values.append(value)

            location = clean(row.get("location"))
            nickname = clean(row.get("nickname"))
            if location and nickname:
                values.append(f"{location} {nickname}")

            for value in values:
                key = normalize_key(value)
                if not key:
                    continue
                prior = self.alias_to_team.get(key)
                if prior is None or prior == canonical:
                    self.alias_to_team[key] = canonical

    def resolve(self, value: object) -> str:
        raw = clean(value)
        if not raw:
            return ""
        return self.alias_to_team.get(normalize_key(raw), raw)

    def team_id(self, value: object) -> str:
        canonical = self.resolve(value)
        return self.team_to_id.get(canonical, "")


def shrink_metric(
    team_mean: pd.Series,
    team_count: pd.Series,
    global_mean: float,
    strength: float = 2.0,
) -> pd.Series:
    count = pd.to_numeric(team_count, errors="coerce").fillna(0.0)
    weight = count / (count + strength)
    return weight * team_mean + (1.0 - weight) * global_mean


def build_prior_table(
    team_stats: pd.DataFrame,
    resolver: TeamResolver,
) -> pd.DataFrame:
    work = team_stats.copy()
    work["team"] = work["team"].map(resolver.resolve)

    for metric in TEAM_METRICS:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")

    work = work[work["team"].map(clean).ne("")].copy()
    if work.empty:
        raise ValueError("Prior team-stats file has no usable team rows.")

    grouped_mean = work.groupby("team", as_index=False)[TEAM_METRICS].mean()
    grouped_count = (
        work.groupby("team", as_index=False)
        .size()
        .rename(columns={"size": "prior_team_weeks"})
    )
    prior = grouped_mean.merge(grouped_count, on="team", how="left")

    for metric in TEAM_METRICS:
        global_mean = float(work[metric].mean(skipna=True))
        if not math.isfinite(global_mean):
            global_mean = 0.0
        prior[metric] = shrink_metric(
            prior[metric],
            prior["prior_team_weeks"],
            global_mean,
        )

    prior["net_epa"] = prior["off_epa_per_play"] - prior["def_epa_per_play"]
    prior["success_edge"] = prior["off_success_rate"] - prior["def_success_rate"]
    prior["ypp_edge"] = prior["yards_per_play"] - prior["yards_per_play_allowed"]
    prior["ppd_edge"] = (
        prior["points_per_drive"] - prior["points_per_drive_allowed"]
    )
    prior["red_zone_edge"] = (
        prior["red_zone_td_rate"] - prior["red_zone_td_rate_allowed"]
    )

    strength_parts = {
        "net_epa": 0.30,
        "ppd_edge": 0.25,
        "ypp_edge": 0.15,
        "success_edge": 0.15,
        "red_zone_edge": 0.05,
        "early_down_epa": 0.05,
        "third_down_conversion_rate": 0.05,
    }

    prior["prior_strength_raw"] = 0.0
    for metric, weight in strength_parts.items():
        values = pd.to_numeric(prior[metric], errors="coerce")
        mean = float(values.mean(skipna=True))
        std = float(values.std(skipna=True, ddof=0))
        if not math.isfinite(std) or std < 1e-9:
            z = pd.Series(0.0, index=prior.index)
        else:
            z = (values.fillna(mean) - mean) / std
        prior["prior_strength_raw"] += weight * z

    raw_mean = float(prior["prior_strength_raw"].mean())
    raw_std = float(prior["prior_strength_raw"].std(ddof=0))
    if not math.isfinite(raw_std) or raw_std < 1e-9:
        prior["prior_strength_z"] = 0.0
    else:
        prior["prior_strength_z"] = (
            prior["prior_strength_raw"] - raw_mean
        ) / raw_std

    return prior


def load_fpi(
    path: Path,
    resolver: TeamResolver,
) -> pd.DataFrame:
    if not path.is_file():
        print(f"WARNING: FPI file not found; FPI component disabled: {path}")
        return pd.DataFrame(columns=["team", "team_id", "fpi", "epaoffense", "epadefense"])

    fpi = read_csv(path, ["team_id", "fpi"], "team power index")
    fpi["team_id"] = fpi["team_id"].map(clean)

    id_to_team = {
        team_id: team
        for team, team_id in resolver.team_to_id.items()
        if team_id
    }
    fpi["team"] = fpi["team_id"].map(id_to_team).fillna("")

    for column in ["fpi", "epaoffense", "epadefense"]:
        if column not in fpi.columns:
            fpi[column] = np.nan
        fpi[column] = pd.to_numeric(fpi[column], errors="coerce")

    fpi = fpi[fpi["team"].ne("")].copy()
    fpi = fpi.drop_duplicates("team", keep="last")
    return fpi[["team", "team_id", "fpi", "epaoffense", "epadefense"]]


def scale_prior_to_fpi(prior: pd.DataFrame, fpi: pd.DataFrame) -> pd.DataFrame:
    result = prior.copy()
    fpi_std = float(pd.to_numeric(fpi.get("fpi"), errors="coerce").std(ddof=0))
    if not math.isfinite(fpi_std) or fpi_std < 1.0:
        fpi_std = 10.0
    result["prior_rating"] = result["prior_strength_z"] * fpi_std
    return result


def position_cost(position: object) -> float:
    pos = clean(position).upper().replace(" ", "")
    return POSITION_POINT_COST.get(pos, 0.35)


def injury_status_multiplier(status: object) -> float:
    text = clean(status).casefold()
    for key, multiplier in OUT_STATUS_MULTIPLIER.items():
        if key in text:
            return multiplier
    return 0.0


def build_injury_lookup(
    injuries_path: Path,
    resolver: TeamResolver,
    fresh_days: int,
) -> dict[str, pd.DataFrame]:
    if not injuries_path.is_file():
        print(f"WARNING: injury file not found; injury adjustment disabled: {injuries_path}")
        return {}

    injuries = read_csv(
        injuries_path,
        ["team", "position", "game_status", "report_date"],
        "injuries",
    )
    injuries["team"] = injuries["team"].map(resolver.resolve)
    injuries["report_ts"] = pd.to_datetime(
        injuries["report_date"],
        errors="coerce",
        utc=True,
    )
    injuries["status_multiplier"] = injuries["game_status"].map(
        injury_status_multiplier
    )
    injuries["position_cost"] = injuries["position"].map(position_cost)
    injuries["raw_penalty"] = (
        injuries["status_multiplier"] * injuries["position_cost"]
    )

    lookup: dict[str, pd.DataFrame] = {}
    for team, group in injuries.groupby("team"):
        lookup[team] = group.copy()
    return lookup


def injury_summary_for_game(
    team: str,
    game_date: object,
    injury_lookup: dict[str, pd.DataFrame],
    fresh_days: int,
) -> tuple[int, int, int, float]:
    group = injury_lookup.get(team)
    if group is None or group.empty:
        return 0, 0, 0, 0.0

    game_ts = pd.to_datetime(clean(game_date), errors="coerce", utc=True)
    if pd.isna(game_ts):
        return 0, 0, 0, 0.0

    age_days = (game_ts - group["report_ts"]).dt.total_seconds() / 86400.0
    fresh = group[
        group["report_ts"].notna()
        & age_days.ge(0)
        & age_days.le(float(fresh_days))
        & group["status_multiplier"].gt(0)
    ].copy()

    if fresh.empty:
        return 0, 0, 0, 0.0

    statuses = fresh["game_status"].map(lambda x: clean(x).casefold())
    out_count = int(statuses.str.contains("out", regex=False).sum())
    doubtful_count = int(statuses.str.contains("doubtful", regex=False).sum())
    questionable_count = int(
        statuses.str.contains("questionable", regex=False).sum()
    )
    penalty = min(7.0, float(fresh["raw_penalty"].sum()))
    return out_count, doubtful_count, questionable_count, penalty


def weighted_blend(
    components: list[tuple[float | None, float]],
) -> tuple[float | None, list[float]]:
    valid = [
        (value, weight)
        for value, weight in components
        if value is not None
        and math.isfinite(float(value))
        and weight > 0
    ]
    if not valid:
        return None, [0.0 for _ in components]

    weight_sum = sum(weight for _, weight in valid)
    normalized: list[float] = []
    numerator = 0.0

    for value, weight in components:
        if (
            value is None
            or not math.isfinite(float(value))
            or weight <= 0
        ):
            normalized.append(0.0)
            continue
        effective = weight / weight_sum
        normalized.append(effective)
        numerator += float(value) * effective

    return numerator, normalized


def prior_total_estimate(
    home: pd.Series,
    away: pd.Series,
    drives_per_team: float,
) -> float | None:
    values = [
        as_float(home.get("points_per_drive")),
        as_float(away.get("points_per_drive_allowed")),
        as_float(away.get("points_per_drive")),
        as_float(home.get("points_per_drive_allowed")),
    ]
    if any(value is None for value in values):
        return None

    home_ppd = (values[0] + values[1]) / 2.0
    away_ppd = (values[2] + values[3]) / 2.0
    total = (home_ppd + away_ppd) * drives_per_team
    return float(np.clip(total, 20.0, 90.0))


def build_projection(
    schedule: pd.DataFrame,
    prior: pd.DataFrame,
    fpi: pd.DataFrame,
    resolver: TeamResolver,
    injury_lookup: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> pd.DataFrame:
    prior_lookup = prior.set_index("team", drop=False)
    fpi_lookup = fpi.set_index("team", drop=False) if not fpi.empty else None

    output_rows: list[dict[str, object]] = []

    # Some Week 1 opponents (especially FCS teams) may have little or no
    # prior-season ESPN PBP. Do not drop those games. Fall back to a neutral
    # national-average prior, while preserving an audit flag in the output.
    fallback_prior = prior.mean(numeric_only=True)
    fallback_prior["prior_team_weeks"] = 0
    fallback_prior["prior_rating"] = 0.0

    for _, sched_row in schedule.iterrows():
        home_team = resolver.resolve(sched_row.get("home_team"))
        away_team = resolver.resolve(sched_row.get("away_team"))

        home_prior_fallback = home_team not in prior_lookup.index
        away_prior_fallback = away_team not in prior_lookup.index

        home_prior = (
            fallback_prior if home_prior_fallback else prior_lookup.loc[home_team]
        )
        away_prior = (
            fallback_prior if away_prior_fallback else prior_lookup.loc[away_team]
        )

        neutral = as_bool(sched_row.get("neutral_site"))
        home_field = 0.0 if neutral else float(args.home_field)

        prior_margin = (
            float(home_prior["prior_rating"])
            - float(away_prior["prior_rating"])
            + home_field
        )

        home_fpi = None
        away_fpi = None
        if fpi_lookup is not None:
            if home_team in fpi_lookup.index:
                home_fpi = as_float(fpi_lookup.loc[home_team].get("fpi"))
            if away_team in fpi_lookup.index:
                away_fpi = as_float(fpi_lookup.loc[away_team].get("fpi"))

        fpi_margin = None
        if home_fpi is not None and away_fpi is not None:
            fpi_margin = home_fpi - away_fpi + home_field

        home_spread = as_float(sched_row.get("home_spread"))
        market_margin = -home_spread if home_spread is not None else None

        blended_margin, margin_weights = weighted_blend(
            [
                (market_margin, args.market_margin_weight),
                (fpi_margin, args.fpi_margin_weight),
                (prior_margin, args.prior_margin_weight),
            ]
        )
        if blended_margin is None:
            raise RuntimeError(
                f"No usable margin component for game_id={clean(sched_row.get('game_id'))}"
            )

        (
            home_out,
            home_doubtful,
            home_questionable,
            home_injury_penalty,
        ) = injury_summary_for_game(
            home_team,
            sched_row.get("game_date"),
            injury_lookup,
            args.fresh_injury_days,
        )
        (
            away_out,
            away_doubtful,
            away_questionable,
            away_injury_penalty,
        ) = injury_summary_for_game(
            away_team,
            sched_row.get("game_date"),
            injury_lookup,
            args.fresh_injury_days,
        )

        injury_adjustment = away_injury_penalty - home_injury_penalty
        projected_margin = blended_margin + injury_adjustment

        prior_total = prior_total_estimate(
            home_prior,
            away_prior,
            args.drives_per_team,
        )
        market_total = as_float(sched_row.get("total"))

        projected_total, total_weights = weighted_blend(
            [
                (market_total, args.market_total_weight),
                (
                    prior_total,
                    max(0.0, 1.0 - args.market_total_weight),
                ),
            ]
        )
        if projected_total is None:
            raise RuntimeError(
                f"No usable total component for game_id={clean(sched_row.get('game_id'))}"
            )

        # Keep both projected scores non-negative while preserving the final
        # margin and total relationship.
        projected_total = max(projected_total, abs(projected_margin) + 2.0)

        projected_home_score = (projected_total + projected_margin) / 2.0
        projected_away_score = (projected_total - projected_margin) / 2.0

        rec: dict[str, object] = {
            column: clean(sched_row.get(column))
            for column in OUTPUT_BASE_COLUMNS
        }
        rec.update(
            {
                "home_team": home_team,
                "away_team": away_team,
                "home_team_id": resolver.team_id(home_team),
                "away_team_id": resolver.team_id(away_team),
                "home_prior_team_weeks": int(home_prior["prior_team_weeks"]),
                "away_prior_team_weeks": int(away_prior["prior_team_weeks"]),
                "home_prior_fallback": int(home_prior_fallback),
                "away_prior_fallback": int(away_prior_fallback),
                "home_prior_rating": round(float(home_prior["prior_rating"]), 4),
                "away_prior_rating": round(float(away_prior["prior_rating"]), 4),
                "prior_home_margin": round(prior_margin, 4),
                "home_fpi": None if home_fpi is None else round(home_fpi, 4),
                "away_fpi": None if away_fpi is None else round(away_fpi, 4),
                "fpi_home_margin": (
                    None if fpi_margin is None else round(fpi_margin, 4)
                ),
                "market_home_margin": (
                    None if market_margin is None else round(market_margin, 4)
                ),
                "margin_weight_market": round(margin_weights[0], 4),
                "margin_weight_fpi": round(margin_weights[1], 4),
                "margin_weight_prior": round(margin_weights[2], 4),
                "home_out_count": home_out,
                "home_doubtful_count": home_doubtful,
                "home_questionable_count": home_questionable,
                "away_out_count": away_out,
                "away_doubtful_count": away_doubtful,
                "away_questionable_count": away_questionable,
                "home_injury_penalty": round(home_injury_penalty, 4),
                "away_injury_penalty": round(away_injury_penalty, 4),
                "injury_margin_adjustment": round(injury_adjustment, 4),
                "prior_total": (
                    None if prior_total is None else round(prior_total, 4)
                ),
                "market_total": (
                    None if market_total is None else round(market_total, 4)
                ),
                "total_weight_market": round(total_weights[0], 4),
                "total_weight_prior": round(total_weights[1], 4),
                "projected_margin": round(projected_margin, 2),
                "projected_total": round(projected_total, 2),
                "projected_home_score": round(projected_home_score, 2),
                "projected_away_score": round(projected_away_score, 2),
                "projection_version": SCRIPT_VERSION,
            }
        )

        # Include the actual prior-season statistics used for audit/backtesting.
        for metric in TEAM_METRICS:
            rec[f"home_prior_{metric}"] = round(float(home_prior[metric]), 6)
            rec[f"away_prior_{metric}"] = round(float(away_prior[metric]), 6)

        output_rows.append(rec)

    result = pd.DataFrame(output_rows)
    if result.empty:
        raise RuntimeError("No Week 1 games could be projected.")

    if result["game_id"].duplicated().any():
        duplicates = result.loc[
            result["game_id"].duplicated(keep=False), "game_id"
        ].tolist()
        raise ValueError(f"Duplicate game_id values in output: {duplicates[:10]}")

    return result


def validate_args(args: argparse.Namespace) -> None:
    weights = [
        args.market_margin_weight,
        args.fpi_margin_weight,
        args.prior_margin_weight,
        args.market_total_weight,
    ]
    if any(weight < 0 for weight in weights):
        raise ValueError("Projection weights cannot be negative.")
    if (
        args.market_margin_weight
        + args.fpi_margin_weight
        + args.prior_margin_weight
        <= 0
    ):
        raise ValueError("At least one margin weight must be positive.")
    if not 0.0 <= args.market_total_weight <= 1.0:
        raise ValueError("--market-total-weight must be between 0 and 1.")
    if args.drives_per_team <= 0:
        raise ValueError("--drives-per-team must be positive.")
    if args.fresh_injury_days < 0:
        raise ValueError("--fresh-injury-days cannot be negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    season = args.season
    if season is None:
        season = int(os.getenv("CFB_SEASON", "2026"))
    prior_season = args.prior_season or (season - 1)

    root = repo_cfb_root()

    schedule_path = (
        root
        / "00_intake"
        / "schedule"
        / "weekly"
        / f"week_{args.week}_CFB_weekly_schedule.csv"
    )
    prior_path = (
        root
        / "00_intake"
        / "team_stats"
        / f"{prior_season}_team_stats.csv"
    )
    fpi_path = (
        root
        / "data"
        / "team_power_index"
        / f"team_power_index_{season}.csv"
    )
    injuries_path = (
        root
        / "00_intake"
        / "injuries"
        / f"{season}_injuries.csv"
    )
    team_map_path = root / "config" / "mapping" / "team_map.csv"
    output_path = root / "01_merge" / f"week_{args.week}_CFB_enriched.csv"

    schedule = read_csv(
        schedule_path,
        [
            "season",
            "season_type",
            "week",
            "game_id",
            "game_date",
            "away_team",
            "home_team",
            "neutral_site",
            "home_spread",
            "total",
        ],
        "weekly schedule",
    )
    schedule = schedule[
        pd.to_numeric(schedule["season"], errors="coerce").eq(season)
        & pd.to_numeric(schedule["week"], errors="coerce").eq(args.week)
    ].copy()
    if schedule.empty:
        raise ValueError(
            f"No schedule rows for season={season} week={args.week} in {schedule_path}"
        )

    team_map = read_csv(
        team_map_path,
        ["team_id", "canonical_team"],
        "team map",
    )
    resolver = TeamResolver(team_map)

    team_stats = read_csv(
        prior_path,
        ["season", "week", "team", *TEAM_METRICS],
        "prior-season team stats",
    )
    team_stats = team_stats[
        pd.to_numeric(team_stats["season"], errors="coerce").eq(prior_season)
    ].copy()
    if team_stats.empty:
        raise ValueError(
            f"No prior team-stat rows for season={prior_season} in {prior_path}"
        )

    prior = build_prior_table(team_stats, resolver)
    fpi = load_fpi(fpi_path, resolver)
    prior = scale_prior_to_fpi(prior, fpi)
    injury_lookup = build_injury_lookup(
        injuries_path,
        resolver,
        args.fresh_injury_days,
    )

    projected = build_projection(
        schedule,
        prior,
        fpi,
        resolver,
        injury_lookup,
        args,
    )

    print(f"projection_week1.py version={SCRIPT_VERSION}")
    print(f"season={season}")
    print(f"prior_season={prior_season}")
    print(f"week={args.week}")
    print(f"games={len(projected)}")
    print(
        "with_market_spread="
        f"{int(projected['market_home_margin'].notna().sum())}"
    )
    print(f"with_market_total={int(projected['market_total'].notna().sum())}")
    print(f"with_fpi={int(projected['fpi_home_margin'].notna().sum())}")
    print(
        "fresh_injury_adjustments="
        f"{int(projected['injury_margin_adjustment'].abs().gt(0).sum())}"
    )

    if args.dry_run:
        print("output_modified=no")
        print("status=dry_run_success")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    projected.to_csv(output_path, index=False, encoding="utf-8")
    print(f"output={output_path}")
    print("status=success")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

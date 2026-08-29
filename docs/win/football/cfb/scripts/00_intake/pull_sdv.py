#!/usr/bin/env python3
"""
Single-entry SportsDataverse NHL puller.

Default:
    python docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py

Historical:
    python docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py --season 2024
    python docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py --start-season 2021 --end-season 2024

Optional category subset:
    python docs/win/hockey/nhl/scripts/00_intake/pull_sdv.py --categories schedule,goalie,odds

Current mode never requires a date. It uses the current America/New_York date.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import polars as pl
import sportsdataverse.nhl as nhl


BASE_DIR = Path("docs/win/hockey/nhl")
ROOT = BASE_DIR / "sdv"
OFFICIAL_SCHEDULE_DIR = BASE_DIR / "00_intake" / "nhl_schedule"
CATEGORIES = (
    "schedule",
    "team-strength",
    "goalie",
    "lineup-strength",
    "fatigue",
    "sdv_predictions",
    "odds",
)
NY = ZoneInfo("America/New_York")
UTC = timezone.utc

TEAM_STRENGTH_FIELDS = (
    "adj_xgf",
    "adj_xga",
    "adj_xg_net",
    "adj_gf",
    "adj_ga",
    "off_rank",
    "def_rank",
    "net_rank",
    "net_z",
)


GOALIE_STATUS_VALUES = (
    "projected",
    "expected",
    "confirmed",
    "unknown",
)

GOALIE_GSAX_FIELDS = (
    "player_id",
    "goalie",
    "shots",
    "xga",
    "ga",
    "gsax",
    "gsax_per_60",
)

GOALIE_PRODUCTION_CUTOFF_MINUTES = 60
GOALIE_PRODUCTION_CUTOFF_SOURCE = "fixed_60_minutes_before_puck_drop"


LINEUP_PRODUCTION_CUTOFF_MINUTES = 60
LINEUP_PRODUCTION_CUTOFF_SOURCE = "fixed_60_minutes_before_puck_drop"

LINEUP_TEAM_METRICS = (
    "skater_rapm",
    "skater_war",
    "pp_value",
    "pk_value",
    "forward_line_strength",
    "defense_pair_strength",
)

LINEUP_STATUS_VALUES = (
    "projected",
    "expected",
    "confirmed",
    "unknown",
)

PREGAME_FEATURE_EVALUATION = (
    (
        "nhl_xg",
        "xg_quality",
        "covered_elsewhere",
        "prior_pbp_safe",
        "VERIFIED",
        "Used upstream by team-strength/RAPM; do not duplicate as a separate lineup feature.",
    ),
    (
        "nhl_goalie_gsax",
        "goalie_performance",
        "covered_elsewhere",
        "prior_pbp_and_shifts_safe",
        "VERIFIED",
        "Integrated in SDV-P3 goalie strength.",
    ),
    (
        "nhl_skater_rapm",
        "player_impact",
        "production",
        "prior_pbp_and_shifts_safe",
        "VERIFIED",
        "Aggregate leakage-safe prior-game skater xG RAPM to team/game level.",
    ),
    (
        "nhl_skater_war",
        "player_impact",
        "production",
        "prior_pbp_and_shifts_safe",
        "VERIFIED",
        "Aggregate leakage-safe prior-game WAR to team/game level.",
    ),
    (
        "nhl_special_teams_value",
        "power_play_penalty_kill",
        "production",
        "prior_pbp_and_shifts_safe",
        "VERIFIED",
        "Aggregate PP and PK value to team/game level.",
    ),
    (
        "nhl_unit_ratings",
        "forward_line_defense_pair",
        "production",
        "prior_pbp_and_shifts_safe",
        "VERIFIED",
        "Aggregate forward-line and defense-pair unit_value to team/game level.",
    ),
    (
        "nhl_penalty_value",
        "penalty_value",
        "evaluated_not_selected",
        "prior_pbp_safe",
        "VERIFIED",
        "WAR already contains a penalty component; separate production use would duplicate signal.",
    ),
    (
        "nhl_faceoff_value",
        "faceoff_value",
        "evaluated_not_selected",
        "prior_pbp_safe",
        "VERIFIED",
        "WAR already contains a faceoff component; separate production use would duplicate signal.",
    ),
    (
        "nhl_edge_skating_value",
        "edge_skating",
        "research_only_not_production_safe",
        "not_reconstructable_at_t60",
        "VERIFIED",
        "SportsDataverse requires caller-supplied season aggregate EDGE detail_frames; "
        "the live detail path is not implemented and the aggregate has no historical observation "
        "timestamp, so it cannot reproduce a T-60 historical information state.",
    ),
    (
        "nhl_expected_assists",
        "expected_assists",
        "research_only",
        "prior_pbp_safe",
        "VERIFIED",
        "SportsDataverse computes entirely from the supplied PBP, so the metric is leakage-safe "
        "when the pipeline supplies only games strictly before the target date; keep research-only "
        "until incremental predictive value is demonstrated.",
    ),
    (
        "nhl_zone_transitions",
        "zone_transitions",
        "research_only",
        "prior_pbp_safe",
        "VERIFIED",
        "SportsDataverse computes from the supplied PBP and is leakage-safe on prior-game input, "
        "but controlled-entry classification is a documented PBP heuristic without ground-truth "
        "microstat tags, so keep it research-only.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        type=int,
        action="append",
        help="NHL season start year, e.g. --season 2024. Repeat for multiple seasons.",
    )
    parser.add_argument("--start-season", type=int)
    parser.add_argument("--end-season", type=int)
    parser.add_argument(
        "--categories",
        default="all",
        help="Comma-separated category names, or 'all'.",
    )
    return parser.parse_args()


def requested_seasons(args: argparse.Namespace) -> list[int]:
    out: list[int] = []

    if args.season:
        out.extend(args.season)

    if args.start_season is not None or args.end_season is not None:
        if args.start_season is None or args.end_season is None:
            raise SystemExit(
                "--start-season and --end-season must be used together"
            )

        if args.end_season < args.start_season:
            raise SystemExit(
                "--end-season cannot be less than --start-season"
            )

        out.extend(
            range(
                args.start_season,
                args.end_season + 1,
            )
        )

    return sorted(set(out))


def requested_categories(args: argparse.Namespace) -> set[str]:
    if str(args.categories).strip().lower() == "all":
        return set(CATEGORIES)

    values = {
        x.strip()
        for x in str(args.categories).split(",")
        if x.strip()
    }

    unknown = sorted(values - set(CATEGORIES))

    if unknown:
        raise SystemExit(
            f"Unknown categories: {unknown}"
        )

    return values


def season_start_for_day(day: date) -> int:
    return (
        day.year
        if day.month >= 7
        else day.year - 1
    )


def run_stamp() -> str:
    return datetime.now(NY).strftime(
        "%Y_%m_%dT%H%M%S_ET"
    )



def utc_iso(value: datetime) -> str:
    return (
        value.astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_timestamp_utc(
    value: Any,
) -> datetime | None:
    text = str(value or "").strip()

    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        )
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=UTC
        )

    return parsed.astimezone(
        UTC
    )


def conservative_game_date_cutoff_utc(
    game_day: date,
) -> datetime:
    return datetime.combine(
        game_day,
        dt_time.min,
        tzinfo=NY,
    ).astimezone(
        UTC
    )


def game_date_time_cutoff_utc(
    game_day: date,
    game_time: Any,
) -> datetime | None:
    text = str(
        game_time or ""
    ).strip()

    if not text:
        return None

    for fmt in (
        "%H:%M:%S",
        "%H:%M",
    ):
        try:
            parsed_time = datetime.strptime(
                text,
                fmt,
            ).time()
            return datetime.combine(
                game_day,
                parsed_time,
                tzinfo=NY,
            ).astimezone(
                UTC
            )
        except ValueError:
            continue

    return None


def official_schedule_cutoff_lookup() -> dict[
    str,
    datetime,
]:
    lookup: dict[
        str,
        datetime,
    ] = {}

    if not OFFICIAL_SCHEDULE_DIR.exists():
        return lookup

    for path in sorted(
        OFFICIAL_SCHEDULE_DIR.glob(
            "*.csv"
        )
    ):
        try:
            pdf = pd.read_csv(
                path,
                dtype=str,
            ).fillna("")
        except Exception:
            continue

        if (
            "game_id" not in pdf.columns
            or "start_time_utc" not in pdf.columns
        ):
            continue

        for _, row in pdf.iterrows():
            game_id = str(
                row.get(
                    "game_id",
                    "",
                )
            ).strip()

            cutoff = parse_timestamp_utc(
                row.get(
                    "start_time_utc",
                    "",
                )
            )

            if game_id and cutoff is not None:
                lookup[
                    game_id
                ] = cutoff

    return lookup


def schedule_games_with_cutoffs(
    schedule: Any,
) -> pd.DataFrame:
    if frame_empty(
        schedule
    ):
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "home_team",
                "away_team",
                "pregame_cutoff_utc",
                "pregame_cutoff_source",
            ]
        )

    pdf = as_pandas(
        schedule
    )

    gid_col = first_col(
        pdf,
        (
            "game_id",
            "id",
            "event_id",
        ),
    )
    date_col = first_col(
        pdf,
        (
            "game_date",
            "schedule_date",
            "date",
            "start_date",
        ),
    )
    home_col = first_col(
        pdf,
        (
            "home_team_abbrev",
            "home_team_abbr",
            "home_abbr",
            "home_team",
        ),
    )
    away_col = first_col(
        pdf,
        (
            "away_team_abbrev",
            "away_team_abbr",
            "away_abbr",
            "away_team",
        ),
    )

    if (
        gid_col is None
        or date_col is None
        or home_col is None
        or away_col is None
    ):
        return pd.DataFrame()

    direct_utc_col = first_col(
        pdf,
        (
            "start_time_utc",
            "startTimeUTC",
            "start_utc",
        ),
    )

    datetime_col = first_col(
        pdf,
        (
            "game_datetime",
            "game_date_time",
            "start_datetime",
            "date_time",
        ),
    )

    game_time_col = first_col(
        pdf,
        (
            "game_time",
            "start_time_et",
            "start_time",
        ),
    )

    official_lookup = (
        official_schedule_cutoff_lookup()
    )

    rows: list[
        dict[str, str]
    ] = []

    for _, row in pdf.iterrows():
        game_id = str(
            row.get(
                gid_col,
                "",
            )
        ).strip()

        parsed_day = pd.to_datetime(
            row.get(
                date_col,
                "",
            ),
            errors="coerce",
        )

        if (
            not game_id
            or pd.isna(
                parsed_day
            )
        ):
            continue

        game_day = parsed_day.date()

        cutoff = (
            official_lookup.get(
                game_id
            )
        )
        source = (
            "official_nhl_schedule"
            if cutoff is not None
            else ""
        )

        if (
            cutoff is None
            and direct_utc_col is not None
        ):
            cutoff = parse_timestamp_utc(
                row.get(
                    direct_utc_col,
                    "",
                )
            )
            if cutoff is not None:
                source = (
                    f"schedule:{direct_utc_col}"
                )

        if (
            cutoff is None
            and datetime_col is not None
        ):
            raw = row.get(
                datetime_col,
                "",
            )
            parsed = pd.to_datetime(
                raw,
                errors="coerce",
                utc=True,
            )
            if not pd.isna(
                parsed
            ):
                cutoff = (
                    parsed.to_pydatetime()
                    .astimezone(
                        UTC
                    )
                )
                source = (
                    f"schedule:{datetime_col}"
                )

        if (
            cutoff is None
            and game_time_col is not None
        ):
            raw_game_time = row.get(
                game_time_col,
                "",
            )

            cutoff = (
                game_date_time_cutoff_utc(
                    game_day,
                    raw_game_time,
                )
            )

            if cutoff is not None:
                source = (
                    f"schedule:{game_time_col}_et"
                )
            else:
                parsed = pd.to_datetime(
                    raw_game_time,
                    errors="coerce",
                    utc=True,
                )

                if not pd.isna(
                    parsed
                ):
                    cutoff = (
                        parsed.to_pydatetime()
                        .astimezone(
                            UTC
                        )
                    )
                    source = (
                        f"schedule:{game_time_col}"
                    )

        if cutoff is None:
            cutoff = (
                conservative_game_date_cutoff_utc(
                    game_day
                )
            )
            source = (
                "conservative_game_date_start_et"
            )

        rows.append(
            {
                "game_id": game_id,
                "game_date": (
                    game_day.isoformat()
                ),
                "home_team": str(
                    row.get(
                        home_col,
                        "",
                    )
                ).strip(),
                "away_team": str(
                    row.get(
                        away_col,
                        "",
                    )
                ).strip(),
                "pregame_cutoff_utc": (
                    utc_iso(
                        cutoff
                    )
                ),
                "pregame_cutoff_source": source,
            }
        )

    return pd.DataFrame(
        rows
    )


def strict_historical_ratings_as_of_utc(
    pregame_cutoff_utc: datetime,
) -> datetime:
    return (
        pregame_cutoff_utc
        - timedelta(
            microseconds=1
        )
    )


def goalie_production_cutoff_utc(
    game_start_utc: datetime,
) -> datetime:
    return (
        game_start_utc.astimezone(UTC)
        - timedelta(
            minutes=GOALIE_PRODUCTION_CUTOFF_MINUTES
        )
    )




def lineup_production_cutoff_utc(
    game_start_utc: datetime,
) -> datetime:
    return (
        game_start_utc.astimezone(UTC)
        - timedelta(
            minutes=LINEUP_PRODUCTION_CUTOFF_MINUTES
        )
    )


def pregame_feature_evaluation() -> dict[str, Any]:
    rows = []

    for (
        function_name,
        family,
        decision,
        as_of_capability,
        evaluation_status,
        reason,
    ) in PREGAME_FEATURE_EVALUATION:
        rows.append(
            {
                "function": function_name,
                "family": family,
                "available": bool(
                    hasattr(
                        nhl,
                        function_name,
                    )
                ),
                "decision": decision,
                "as_of_capability": as_of_capability,
                "evaluation_status": evaluation_status,
                "reason": reason,
            }
        )

    return {
        "evaluation_status": "VERIFIED",
        "decision_cutoff_minutes_before_puck_drop": (
            LINEUP_PRODUCTION_CUTOFF_MINUTES
        ),
        "historical_input_rule": (
            "PBP and shifts supplied to pregame player/microstat calculations "
            "must contain only games strictly before the target game date."
        ),
        "missing_or_late_lineup_behavior": (
            "unknown status with lineup-dependent fields blank; "
            "never backfill observations after T-60"
        ),
        "families": rows,
    }


def ensure_dirs() -> None:
    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in CATEGORIES:
        (ROOT / name).mkdir(
            parents=True,
            exist_ok=True,
        )


def is_polars_frame(obj: Any) -> bool:
    return isinstance(
        obj,
        pl.DataFrame,
    )


def is_pandas_frame(obj: Any) -> bool:
    return isinstance(
        obj,
        pd.DataFrame,
    )


def as_pandas(obj: Any) -> pd.DataFrame:
    if is_pandas_frame(obj):
        return obj.copy()

    if is_polars_frame(obj):
        return pd.DataFrame(
            obj.to_dicts()
        )

    raise TypeError(
        f"Not a DataFrame: {type(obj).__name__}"
    )


def csv_safe_value(value: Any) -> Any:
    if isinstance(
        value,
        (dict, list, tuple, set),
    ):
        return json.dumps(
            value,
            default=str,
            ensure_ascii=False,
        )

    return value


def csv_safe_frame(obj: Any) -> pd.DataFrame:
    df = as_pandas(obj)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(
                csv_safe_value
            )

    return df


def _write_json(
    path: Path,
    obj: Any,
) -> None:
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            default=str,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    obj: Any,
) -> None:
    csv_safe_frame(obj).to_csv(
        path,
        index=False,
    )


def save_object(
    category: str,
    label: str,
    obj: Any,
    *,
    prefix: str,
    current: bool,
) -> list[Path]:
    out_dir = ROOT / category
    written: list[Path] = []

    if (
        isinstance(obj, dict)
        and obj
        and any(
            is_polars_frame(v)
            or is_pandas_frame(v)
            for v in obj.values()
        )
    ):
        for key, value in obj.items():
            written.extend(
                save_object(
                    category,
                    f"{label}_{key}",
                    value,
                    prefix=prefix,
                    current=current,
                )
            )

        return written

    if (
        is_polars_frame(obj)
        or is_pandas_frame(obj)
    ):
        snap = (
            out_dir
            / f"{prefix}_{label}.csv"
        )

        _write_csv(
            snap,
            obj,
        )

        written.append(snap)

        if current:
            latest = (
                out_dir
                / f"latest_{label}.csv"
            )

            shutil.copyfile(
                snap,
                latest,
            )

            written.append(latest)

        return written

    snap = (
        out_dir
        / f"{prefix}_{label}.json"
    )

    _write_json(
        snap,
        obj,
    )

    written.append(snap)

    if current:
        latest = (
            out_dir
            / f"latest_{label}.json"
        )

        shutil.copyfile(
            snap,
            latest,
        )

        written.append(latest)

    return written


def safe_pull(
    failures: list[str],
    category: str,
    label: str,
    fn,
    *args,
    **kwargs,
):
    try:
        return fn(
            *args,
            **kwargs,
        )

    except Exception as exc:
        failures.append(
            f"{category}/{label}: {exc}"
        )

        print(
            f"WARN | {category}/{label} | {exc}",
            file=sys.stderr,
        )

        return None


def frame_empty(obj: Any) -> bool:
    if obj is None:
        return True

    if is_polars_frame(obj):
        return obj.is_empty()

    if is_pandas_frame(obj):
        return obj.empty

    return False


def clean_war_rows(war: Any) -> Any:
    if is_polars_frame(war):
        if "player_id" not in war.columns:
            return war

        player_id = pl.col(
            "player_id"
        ).cast(
            pl.Int64,
            strict=False,
        )

        return war.filter(
            player_id.is_not_null()
            & (player_id != 0)
        )

    if is_pandas_frame(war):
        if "player_id" not in war.columns:
            return war

        player_id = pd.to_numeric(
            war["player_id"],
            errors="coerce",
        )

        return war.loc[
            player_id.notna()
            & player_id.ne(0)
        ].copy()

    return war


def first_col(
    df: Any,
    names: tuple[str, ...],
) -> str | None:
    cols = set(
        as_pandas(df).columns
    )

    for name in names:
        if name in cols:
            return name

    return None


def filter_exact_date(
    schedule: Any,
    target: date,
):
    if frame_empty(schedule):
        return schedule

    pdf = as_pandas(schedule)

    date_col = first_col(
        pdf,
        (
            "schedule_date",
            "game_date",
            "date",
            "start_date",
        ),
    )

    if date_col is None:
        return schedule

    normalized = pd.to_datetime(
        pdf[date_col],
        errors="coerce",
    ).dt.date

    return pl.from_pandas(
        pdf.loc[
            normalized == target
        ].reset_index(
            drop=True
        )
    )


def schedule_teams(
    schedule: Any,
) -> list[str]:
    if frame_empty(schedule):
        return []

    pdf = as_pandas(schedule)

    home_col = first_col(
        pdf,
        (
            "home_team_abbrev",
            "home_team_abbr",
            "home_abbr",
            "home_team",
        ),
    )

    away_col = first_col(
        pdf,
        (
            "away_team_abbrev",
            "away_team_abbr",
            "away_abbr",
            "away_team",
        ),
    )

    if (
        home_col is None
        or away_col is None
    ):
        return []

    vals = pd.concat(
        [
            pdf[home_col],
            pdf[away_col],
        ],
        ignore_index=True,
    )

    vals = (
        vals
        .dropna()
        .astype(str)
        .str.strip()
    )

    return sorted(
        {
            x
            for x in vals
            if x
        }
    )


def historical_schedule_date_bounds(
    schedule: Any,
) -> tuple[
    date | None,
    date | None,
]:
    if frame_empty(schedule):
        return None, None

    pdf = as_pandas(schedule)

    date_col = first_col(
        pdf,
        (
            "game_date",
            "schedule_date",
            "date",
        ),
    )

    if date_col is None:
        return None, None

    d = pd.to_datetime(
        pdf[date_col],
        errors="coerce",
    ).dropna()

    if d.empty:
        return None, None

    return (
        d.min().date(),
        d.max().date(),
    )


def historical_schedule_game_dates(
    schedule: Any,
) -> list[date]:
    if frame_empty(schedule):
        return []

    pdf = as_pandas(schedule)

    date_col = first_col(
        pdf,
        (
            "game_date",
            "schedule_date",
            "date",
            "start_date",
        ),
    )

    if date_col is None:
        return []

    parsed = pd.to_datetime(
        pdf[date_col],
        errors="coerce",
    ).dropna()

    return sorted(
        {
            value.date()
            for value in parsed
        }
    )


def current_or_historical_prefix(
    *,
    current: bool,
    season: int | None,
) -> str:
    if current:
        return run_stamp()

    assert season is not None

    return f"season_{season}"


def save_pair(
    failures: list[str],
    *,
    category: str,
    label: str,
    parsed_fn,
    raw_fn,
    prefix: str,
    current: bool,
):
    parsed = safe_pull(
        failures,
        category,
        f"{label}_parsed",
        parsed_fn,
    )

    if parsed is not None:
        save_object(
            category,
            label,
            parsed,
            prefix=prefix,
            current=current,
        )

    raw = safe_pull(
        failures,
        category,
        f"{label}_raw",
        raw_fn,
    )

    if raw is not None:
        save_object(
            category,
            f"{label}_raw",
            raw,
            prefix=prefix,
            current=current,
        )

    return parsed


def normalize_loader_schedule_for_ratings(
    schedule: pl.DataFrame,
) -> pl.DataFrame:
    if schedule.is_empty():
        return pl.DataFrame()

    required = {
        "game_id",
        "season",
        "game_date",
        "home_team_abbr",
        "away_team_abbr",
    }

    if not required.issubset(
        set(schedule.columns)
    ):
        return pl.DataFrame()

    work = schedule

    if "game_type" in work.columns:
        work = work.filter(
            pl.col("game_type") == "R"
        )

    return work.select(
        pl.col("game_id"),
        pl.col("season").cast(
            pl.Int64
        ),
        pl.col("game_date")
        .cast(pl.Utf8)
        .str.strptime(
            pl.Date,
            "%Y-%m-%d",
            strict=False,
        )
        .alias("date"),
        pl.col("home_team_abbr")
        .cast(pl.Utf8)
        .alias("home_abbr"),
        pl.col("away_team_abbr")
        .cast(pl.Utf8)
        .alias("away_abbr"),
        pl.lit(False).alias(
            "neutral_site"
        ),
    )


def ratings_from_game_rates(
    game_rates: pl.DataFrame,
) -> pl.DataFrame:
    if game_rates.is_empty():
        return pl.DataFrame()

    const = nhl.get_constants(
        "nhl"
    )

    xg_adj = nhl.adjust_rate_opponent(
        game_rates,
        for_col="xgf",
        against_col="xga",
        hfa=const.hfa,
        avg=const.avg_xgf,
        shrink_k=const.shrink_k,
    )

    goal_adj = nhl.adjust_rate_opponent(
        game_rates,
        for_col="gf",
        against_col="ga",
        hfa=const.hfa,
        avg=(
            const.avg_total_goals
            / 2.0
        ),
        shrink_k=const.shrink_k,
    )

    if (
        xg_adj.is_empty()
        or goal_adj.is_empty()
    ):
        return pl.DataFrame()

    out = (
        xg_adj.join(
            goal_adj.select(
                "team",
                pl.col(
                    "adj_for"
                ).alias(
                    "adj_gf"
                ),
                pl.col(
                    "adj_against"
                ).alias(
                    "adj_ga"
                ),
            ),
            on="team",
            how="left",
        )
        .rename(
            {
                "adj_for": "adj_xgf",
                "adj_against": "adj_xga",
                "adj_net": "adj_xg_net",
            }
        )
    )

    net_mean = out[
        "adj_xg_net"
    ].mean()

    net_std = out[
        "adj_xg_net"
    ].std()

    return out.with_columns(
        pl.col("adj_xgf")
        .rank(
            method="ordinal",
            descending=True,
        )
        .cast(pl.Int64)
        .alias("off_rank"),

        pl.col("adj_xga")
        .rank(
            method="ordinal",
            descending=False,
        )
        .cast(pl.Int64)
        .alias("def_rank"),

        pl.col("adj_xg_net")
        .rank(
            method="ordinal",
            descending=True,
        )
        .cast(pl.Int64)
        .alias("net_rank"),

        (
            (
                (
                    pl.col(
                        "adj_xg_net"
                    )
                    - net_mean
                )
                / net_std
            )
            if net_std not in (
                None,
                0,
            )
            else pl.lit(0.0)
        ).alias(
            "net_z"
        ),
    ).select(
        "season",
        "team",
        "adj_xgf",
        "adj_xga",
        "adj_xg_net",
        "adj_gf",
        "adj_ga",
        "games",
        "off_rank",
        "def_rank",
        "net_rank",
        "net_z",
    )




def normalize_team_strength_asof(
    ratings: Any,
    *,
    schedule: Any,
    generated_at_utc: datetime,
) -> pl.DataFrame:
    if (
        frame_empty(ratings)
        or frame_empty(schedule)
    ):
        return pl.DataFrame()

    ratings_pdf = as_pandas(
        ratings
    )

    team_col = first_col(
        ratings_pdf,
        (
            "team",
            "team_abbr",
            "team_abbrev",
            "team_abbreviation",
        ),
    )

    if team_col is None:
        raise ValueError(
            "Team-strength ratings missing team identity column"
        )

    missing = [
        field
        for field in TEAM_STRENGTH_FIELDS
        if field not in ratings_pdf.columns
    ]

    if missing:
        raise ValueError(
            "Team-strength ratings missing required fields: "
            f"{missing}"
        )

    games = (
        schedule_games_with_cutoffs(
            schedule
        )
    )

    if games.empty:
        return pl.DataFrame()

    ratings_pdf = (
        ratings_pdf.copy()
    )
    ratings_pdf[
        "_team_key"
    ] = (
        ratings_pdf[
            team_col
        ]
        .astype(str)
        .str.strip()
    )

    ratings_by_team = {
        str(
            row[
                "_team_key"
            ]
        ).strip(): row
        for _, row in ratings_pdf.iterrows()
        if str(
            row[
                "_team_key"
            ]
        ).strip()
    }

    rows: list[
        dict[str, Any]
    ] = []

    generated_at_utc = (
        generated_at_utc.astimezone(
            UTC
        )
    )

    for _, game in games.iterrows():
        cutoff = parse_timestamp_utc(
            game.get(
                "pregame_cutoff_utc",
                "",
            )
        )

        if cutoff is None:
            continue

        # Current snapshots are only valid when they were created
        # strictly before the target game's pregame cutoff.
        if generated_at_utc >= cutoff:
            continue

        for team in (
            str(
                game.get(
                    "home_team",
                    "",
                )
            ).strip(),
            str(
                game.get(
                    "away_team",
                    "",
                )
            ).strip(),
        ):
            rating = (
                ratings_by_team.get(
                    team
                )
            )

            if rating is None:
                continue

            out = {
                "game_id": str(
                    game.get(
                        "game_id",
                        "",
                    )
                ).strip(),
                "game_date": str(
                    game.get(
                        "game_date",
                        "",
                    )
                ).strip(),
                "team": team,
                "pregame_cutoff_utc": (
                    utc_iso(
                        cutoff
                    )
                ),
                "ratings_as_of_utc": (
                    utc_iso(
                        generated_at_utc
                    )
                ),
                "pregame_cutoff_source": str(
                    game.get(
                        "pregame_cutoff_source",
                        "",
                    )
                ).strip(),
            }

            if "season" in ratings_pdf.columns:
                out[
                    "season"
                ] = pd.to_numeric(
                    rating.get(
                        "season"
                    ),
                    errors="coerce",
                )

            for field in (
                TEAM_STRENGTH_FIELDS
            ):
                out[
                    field
                ] = pd.to_numeric(
                    rating.get(
                        field
                    ),
                    errors="coerce",
                )

            rows.append(
                out
            )

    if not rows:
        return pl.DataFrame()

    out = (
        pd.DataFrame(
            rows
        )
        .drop_duplicates(
            subset=[
                "game_id",
                "team",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return pl.from_pandas(
        out
    )


def historical_team_strength(
    schedule: pl.DataFrame,
    pbp: pl.DataFrame,
) -> pl.DataFrame:
    if (
        schedule.is_empty()
        or pbp.is_empty()
    ):
        return pl.DataFrame()

    rating_schedule = (
        normalize_loader_schedule_for_ratings(
            schedule
        )
    )

    if rating_schedule.is_empty():
        return pl.DataFrame()

    game_rates = (
        nhl.team_game_xg_rates(
            pbp,
            rating_schedule,
        )
    )

    if game_rates.is_empty():
        return pl.DataFrame()

    games = (
        schedule_games_with_cutoffs(
            schedule
        )
    )

    if games.empty:
        return pl.DataFrame()

    outputs: list[
        pl.DataFrame
    ] = []

    for _, game in games.iterrows():
        game_id = str(
            game.get(
                "game_id",
                "",
            )
        ).strip()

        d_str = str(
            game.get(
                "game_date",
                "",
            )
        ).strip()

        try:
            game_day = date.fromisoformat(
                d_str
            )
        except ValueError:
            continue

        cutoff = parse_timestamp_utc(
            game.get(
                "pregame_cutoff_utc",
                "",
            )
        )

        if (
            not game_id
            or cutoff is None
        ):
            continue

        # Historical game rates are date-granular in the SDV helper.
        # Excluding the entire target date is intentionally conservative:
        # it guarantees the target game and all later games cannot enter
        # the reconstructed pregame rating snapshot.
        prior_rates = (
            game_rates.filter(
                pl.col("date")
                < pl.lit(
                    game_day
                )
            )
        )

        if prior_rates.is_empty():
            continue

        ratings = (
            ratings_from_game_rates(
                prior_rates
            )
        )

        if ratings.is_empty():
            continue

        target_teams = [
            str(
                game.get(
                    "home_team",
                    "",
                )
            ).strip(),
            str(
                game.get(
                    "away_team",
                    "",
                )
            ).strip(),
        ]

        target_teams = [
            team
            for team in target_teams
            if team
        ]

        if target_teams:
            ratings = ratings.filter(
                pl.col("team")
                .cast(pl.Utf8)
                .is_in(
                    target_teams
                )
            )

        if ratings.is_empty():
            continue

        ratings_as_of = (
            strict_historical_ratings_as_of_utc(
                cutoff
            )
        )

        outputs.append(
            ratings.with_columns(
                pl.lit(
                    game_id
                ).alias(
                    "game_id"
                ),
                pl.lit(
                    d_str
                ).alias(
                    "game_date"
                ),
                pl.lit(
                    utc_iso(
                        cutoff
                    )
                ).alias(
                    "pregame_cutoff_utc"
                ),
                pl.lit(
                    utc_iso(
                        ratings_as_of
                    )
                ).alias(
                    "ratings_as_of_utc"
                ),
                pl.lit(
                    str(
                        game.get(
                            "pregame_cutoff_source",
                            "",
                        )
                    ).strip()
                ).alias(
                    "pregame_cutoff_source"
                ),
            )
        )

    if not outputs:
        return pl.DataFrame()

    combined = pl.concat(
        outputs,
        how="diagonal_relaxed",
    )

    columns = [
        "game_id",
        "game_date",
        "team",
        "pregame_cutoff_utc",
        "ratings_as_of_utc",
        "pregame_cutoff_source",
        *TEAM_STRENGTH_FIELDS,
    ]

    if (
        "season"
        in combined.columns
    ):
        columns.insert(
            0,
            "season",
        )

    return combined.select(
        columns
    )

def prediction_games(
    schedule: Any,
) -> pl.DataFrame:
    if frame_empty(schedule):
        return pl.DataFrame()

    pdf = as_pandas(schedule)

    gid_col = first_col(
        pdf,
        (
            "game_id",
            "id",
            "event_id",
        ),
    )

    home_col = first_col(
        pdf,
        (
            "home_team_abbrev",
            "home_team_abbr",
            "home_abbr",
            "home_team",
        ),
    )

    away_col = first_col(
        pdf,
        (
            "away_team_abbrev",
            "away_team_abbr",
            "away_abbr",
            "away_team",
        ),
    )

    date_col = first_col(
        pdf,
        (
            "schedule_date",
            "game_date",
            "date",
        ),
    )

    if (
        gid_col is None
        or home_col is None
        or away_col is None
    ):
        return pl.DataFrame()

    neutral_col = first_col(
        pdf,
        (
            "neutral_site",
            "neutral",
        ),
    )

    out = pd.DataFrame(
        {
            "game_id": (
                pdf[gid_col]
                .astype(str)
            ),
            "home_team": (
                pdf[home_col]
                .astype(str)
            ),
            "away_team": (
                pdf[away_col]
                .astype(str)
            ),
            "neutral_site": (
                pdf[
                    neutral_col
                ]
                .fillna(False)
                .astype(bool)
                if neutral_col
                else False
            ),
        }
    )

    if date_col:
        out[
            "game_date"
        ] = pd.to_datetime(
            pdf[date_col],
            errors="coerce",
        ).dt.strftime(
            "%Y-%m-%d"
        )

    return pl.from_pandas(out)


def build_fatigue_from_team_schedule(
    team: str,
    team_schedule: Any,
    *,
    target_date: date | None = None,
) -> pl.DataFrame:
    if frame_empty(
        team_schedule
    ):
        return pl.DataFrame()

    pdf = as_pandas(
        team_schedule
    )

    date_col = first_col(
        pdf,
        (
            "game_date",
            "schedule_date",
            "date",
        ),
    )

    if date_col is None:
        return pl.DataFrame()

    dates = sorted(
        {
            x.date()
            for x in pd.to_datetime(
                pdf[date_col],
                errors="coerce",
            ).dropna()
        }
    )

    rows: list[
        dict[str, Any]
    ] = []

    for idx, game_day in enumerate(
        dates
    ):
        prior = dates[:idx]

        previous = (
            prior[-1]
            if prior
            else None
        )

        days_rest = (
            (
                game_day
                - previous
            ).days
            if previous
            else None
        )

        def count_inclusive(
            window_days: int,
        ) -> int:
            return 1 + sum(
                1
                for d in prior
                if (
                    0
                    < (
                        game_day
                        - d
                    ).days
                    < window_days
                )
            )

        games_2 = count_inclusive(
            2
        )

        games_4 = count_inclusive(
            4
        )

        games_6 = count_inclusive(
            6
        )

        games_7 = count_inclusive(
            7
        )

        rows.append(
            {
                "team": team,
                "game_date": (
                    game_day.isoformat()
                ),
                "previous_game_date": (
                    previous.isoformat()
                    if previous
                    else None
                ),
                "days_rest": (
                    days_rest
                ),
                "back_to_back": (
                    games_2 >= 2
                ),
                "games_in_4_days": (
                    games_4
                ),
                "three_in_four": (
                    games_4 >= 3
                ),
                "games_in_6_days": (
                    games_6
                ),
                "four_in_six": (
                    games_6 >= 4
                ),
                "games_in_7_days": (
                    games_7
                ),
            }
        )

    out = pl.DataFrame(rows)

    if (
        target_date is not None
        and not out.is_empty()
    ):
        out = out.filter(
            pl.col(
                "game_date"
            )
            == target_date.isoformat()
        )

    return out


def build_fatigue_from_league_schedule(
    schedule: Any,
) -> pl.DataFrame:
    if frame_empty(schedule):
        return pl.DataFrame()

    pdf = as_pandas(schedule)

    date_col = first_col(
        pdf,
        (
            "game_date",
            "schedule_date",
            "date",
        ),
    )

    home_col = first_col(
        pdf,
        (
            "home_team_abbrev",
            "home_team_abbr",
            "home_abbr",
            "home_team",
        ),
    )

    away_col = first_col(
        pdf,
        (
            "away_team_abbrev",
            "away_team_abbr",
            "away_abbr",
            "away_team",
        ),
    )

    if (
        date_col is None
        or home_col is None
        or away_col is None
    ):
        return pl.DataFrame()

    long_rows = []

    for _, row in pdf.iterrows():
        d = pd.to_datetime(
            row.get(
                date_col
            ),
            errors="coerce",
        )

        if pd.isna(d):
            continue

        long_rows.append(
            (
                str(
                    row.get(
                        home_col,
                        "",
                    )
                ).strip(),
                d.date(),
            )
        )

        long_rows.append(
            (
                str(
                    row.get(
                        away_col,
                        "",
                    )
                ).strip(),
                d.date(),
            )
        )

    by_team: dict[
        str,
        list[date],
    ] = {}

    for team, d in long_rows:
        if team:
            by_team.setdefault(
                team,
                [],
            ).append(d)

    frames = []

    for team, dates in by_team.items():
        fake = pd.DataFrame(
            {
                "game_date": sorted(
                    set(dates)
                )
            }
        )

        frames.append(
            build_fatigue_from_team_schedule(
                team,
                fake,
            )
        )

    return (
        pl.concat(
            frames,
            how="diagonal_relaxed",
        )
        if frames
        else pl.DataFrame()
    )


def historical_predictions(
    schedule: pl.DataFrame,
    pbp: pl.DataFrame,
) -> pl.DataFrame:
    if (
        schedule.is_empty()
        or pbp.is_empty()
    ):
        return pl.DataFrame()

    rating_schedule = (
        normalize_loader_schedule_for_ratings(
            schedule
        )
    )

    if rating_schedule.is_empty():
        return pl.DataFrame()

    game_rates = (
        nhl.team_game_xg_rates(
            pbp,
            rating_schedule,
        )
    )

    if game_rates.is_empty():
        return pl.DataFrame()

    games = prediction_games(
        schedule
    )

    if (
        games.is_empty()
        or "game_date"
        not in games.columns
    ):
        return pl.DataFrame()

    dates = sorted(
        {
            d
            for d in games[
                "game_date"
            ].to_list()
            if (
                isinstance(
                    d,
                    str,
                )
                and d
            )
        }
    )

    outputs = []

    for d_str in dates:
        try:
            d = date.fromisoformat(
                d_str
            )
        except ValueError:
            continue

        prior_rates = (
            game_rates.filter(
                pl.col("date")
                < pl.lit(d)
            )
        )

        ratings = (
            ratings_from_game_rates(
                prior_rates
            )
        )

        if ratings.is_empty():
            continue

        games_day = (
            games.filter(
                pl.col(
                    "game_date"
                )
                == d_str
            )
            .select(
                "game_id",
                "home_team",
                "away_team",
                "neutral_site",
            )
        )

        if games_day.is_empty():
            continue

        pred = (
            nhl.nhl_predict_games(
                games_day,
                ratings,
            )
        )

        if not pred.is_empty():
            outputs.append(
                pred.with_columns(
                    pl.lit(
                        d_str
                    ).alias(
                        "game_date"
                    )
                )
            )

    return (
        pl.concat(
            outputs,
            how="diagonal_relaxed",
        )
        if outputs
        else pl.DataFrame()
    )


def pull_schedule(
    failures: list[str],
    *,
    current: bool,
    target_date: date,
    season: int,
    prefix: str,
):
    if current:
        parsed = safe_pull(
            failures,
            "schedule",
            "nhl_web_schedule",
            nhl.nhl_web_schedule,
            date=target_date.isoformat(),
        )

        raw = safe_pull(
            failures,
            "schedule",
            "nhl_web_schedule_raw",
            nhl.nhl_web_schedule,
            date=target_date.isoformat(),
            return_parsed=False,
        )

        if parsed is not None:
            save_object(
                "schedule",
                "nhl_schedule",
                parsed,
                prefix=prefix,
                current=True,
            )

        if raw is not None:
            save_object(
                "schedule",
                "nhl_schedule_raw",
                raw,
                prefix=prefix,
                current=True,
            )

        slate = (
            filter_exact_date(
                parsed,
                target_date,
            )
            if parsed is not None
            else pl.DataFrame()
        )

        if not frame_empty(
            slate
        ):
            save_object(
                "schedule",
                "nhl_schedule_slate",
                slate,
                prefix=prefix,
                current=True,
            )

        return parsed, slate

    parsed = safe_pull(
        failures,
        "schedule",
        f"load_nhl_schedule_{season}",
        nhl.load_nhl_schedule,
        season,
    )

    if parsed is not None:
        save_object(
            "schedule",
            "nhl_schedule",
            parsed,
            prefix=prefix,
            current=False,
        )

    return parsed, parsed



def pull_team_strength(
    failures: list[str],
    *,
    current: bool,
    target_date: date,
    season: int,
    prefix: str,
    teams: list[str],
    schedule: Any,
):
    ratings = None

    if current:
        ratings = safe_pull(
            failures,
            "team-strength",
            "nhl_team_ratings",
            nhl.nhl_team_ratings,
            season,
            as_of_date=target_date,
        )

        if ratings is not None:
            save_object(
                "team-strength",
                "team_ratings",
                ratings,
                prefix=prefix,
                current=True,
            )

            ratings_asof = safe_pull(
                failures,
                "team-strength",
                "normalize_team_ratings_asof",
                normalize_team_strength_asof,
                ratings,
                schedule=schedule,
                generated_at_utc=datetime.now(
                    UTC
                ),
            )

            if ratings_asof is not None:
                if frame_empty(
                    ratings_asof
                ):
                    failures.append(
                        "team-strength/team_ratings_asof: "
                        "no current rating row satisfied the strict "
                        "ratings_as_of_utc < pregame_cutoff_utc rule"
                    )

                save_object(
                    "team-strength",
                    "team_ratings_asof",
                    ratings_asof,
                    prefix=prefix,
                    current=True,
                )

    standings_date = (
        target_date.isoformat()
    )

    standings = safe_pull(
        failures,
        "team-strength",
        "standings",
        nhl.nhl_standings,
        date=standings_date,
    )

    if standings is not None:
        save_object(
            "team-strength",
            "standings",
            standings,
            prefix=prefix,
            current=current,
        )

    standings_raw = safe_pull(
        failures,
        "team-strength",
        "standings_raw",
        nhl.nhl_standings,
        date=standings_date,
        return_parsed=False,
    )

    if standings_raw is not None:
        save_object(
            "team-strength",
            "standings_raw",
            standings_raw,
            prefix=prefix,
            current=current,
        )

    for team in teams:
        parsed = safe_pull(
            failures,
            "team-strength",
            f"{team}_club_stats",
            nhl.nhl_club_stats,
            team=team,
            season=(
                None
                if current
                else season
            ),
        )

        if parsed is not None:
            save_object(
                "team-strength",
                f"{team}_club_stats",
                parsed,
                prefix=prefix,
                current=current,
            )

        raw = safe_pull(
            failures,
            "team-strength",
            f"{team}_club_stats_raw",
            nhl.nhl_club_stats,
            team=team,
            season=(
                None
                if current
                else season
            ),
            return_parsed=False,
        )

        if raw is not None:
            save_object(
                "team-strength",
                f"{team}_club_stats_raw",
                raw,
                prefix=prefix,
                current=current,
            )

    return ratings

def pull_rosters(
    failures: list[str],
    *,
    current: bool,
    season: int,
    prefix: str,
    teams: list[str],
) -> dict[str, Any]:
    rosters: dict[
        str,
        Any,
    ] = {}

    for team in teams:
        parsed = safe_pull(
            failures,
            "lineup-strength",
            f"{team}_roster",
            nhl.nhl_roster,
            team=team,
            season=(
                None
                if current
                else season
            ),
        )

        if parsed is not None:
            rosters[
                team
            ] = parsed

            save_object(
                "lineup-strength",
                f"{team}_roster",
                parsed,
                prefix=prefix,
                current=current,
            )

        raw = safe_pull(
            failures,
            "lineup-strength",
            f"{team}_roster_raw",
            nhl.nhl_roster,
            team=team,
            season=(
                None
                if current
                else season
            ),
            return_parsed=False,
        )

        if raw is not None:
            save_object(
                "lineup-strength",
                f"{team}_roster_raw",
                raw,
                prefix=prefix,
                current=current,
            )

    return rosters


def pull_goalie_live_profiles(
    failures: list[str],
    *,
    current: bool,
    prefix: str,
    rosters: dict[str, Any],
):
    for team, roster in rosters.items():
        if frame_empty(
            roster
        ):
            continue

        pdf = as_pandas(
            roster
        )

        if (
            "position_group"
            not in pdf.columns
        ):
            continue

        goalies = pdf.loc[
            pdf[
                "position_group"
            ]
            .astype(str)
            .str.lower()
            .eq("goalies")
        ].copy()

        if goalies.empty:
            continue

        save_object(
            "goalie",
            f"{team}_goalies_roster",
            goalies,
            prefix=prefix,
            current=current,
        )

        id_col = first_col(
            goalies,
            (
                "id",
                "player_id",
                "playerId",
            ),
        )

        if id_col is None:
            continue

        for value in (
            goalies[
                id_col
            ]
            .dropna()
            .unique()
            .tolist()
        ):
            try:
                player_id = int(
                    value
                )
            except Exception:
                continue

            parsed = safe_pull(
                failures,
                "goalie",
                f"{team}_{player_id}_player_landing",
                nhl.nhl_player_landing,
                player_id=player_id,
            )

            if parsed is not None:
                save_object(
                    "goalie",
                    f"{team}_{player_id}_player_landing",
                    parsed,
                    prefix=prefix,
                    current=current,
                )

            raw = safe_pull(
                failures,
                "goalie",
                f"{team}_{player_id}_player_landing_raw",
                nhl.nhl_player_landing,
                player_id=player_id,
                return_parsed=False,
            )

            if raw is not None:
                save_object(
                    "goalie",
                    f"{team}_{player_id}_player_landing_raw",
                    raw,
                    prefix=prefix,
                    current=current,
                )


def pull_season_context(
    failures: list[str],
    *,
    season: int,
):
    context = {
        "pbp": None,
        "shifts": None,
        "goalie_box": None,
        "skater_box": None,
        "rosters": None,
    }

    context[
        "pbp"
    ] = safe_pull(
        failures,
        "lineup-strength",
        "load_nhl_pbp_full",
        nhl.load_nhl_pbp_full,
        season,
    )

    context[
        "shifts"
    ] = safe_pull(
        failures,
        "lineup-strength",
        "load_nhl_shifts",
        nhl.load_nhl_shifts,
        season,
    )

    context[
        "goalie_box"
    ] = safe_pull(
        failures,
        "goalie",
        "load_nhl_goalie_boxscores",
        nhl.load_nhl_goalie_boxscores,
        season,
    )

    context[
        "skater_box"
    ] = safe_pull(
        failures,
        "lineup-strength",
        "load_nhl_skater_boxscores",
        nhl.load_nhl_skater_boxscores,
        season,
    )

    context[
        "rosters"
    ] = safe_pull(
        failures,
        "lineup-strength",
        "load_nhl_rosters",
        nhl.load_nhl_rosters,
        season,
    )

    return context


def save_season_context(
    *,
    current: bool,
    prefix: str,
    context: dict[str, Any],
):
    mapping = (
        (
            "goalie",
            "goalie_boxscores",
            context.get(
                "goalie_box"
            ),
        ),
        (
            "lineup-strength",
            "skater_boxscores",
            context.get(
                "skater_box"
            ),
        ),
        (
            "lineup-strength",
            "season_rosters",
            context.get(
                "rosters"
            ),
        ),
    )

    for (
        category,
        label,
        obj,
    ) in mapping:
        if obj is not None:
            save_object(
                category,
                label,
                obj,
                prefix=prefix,
                current=current,
            )



def normalize_player_id(
    value: Any,
) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if not text or text.lower() in {
        "nan",
        "none",
        "null",
    }:
        return ""

    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass

    return text


def prior_game_ids_for_target(
    schedule: Any,
    target_day: date,
) -> set[str]:
    if frame_empty(schedule):
        return set()

    pdf = as_pandas(schedule)

    gid_col = first_col(
        pdf,
        (
            "game_id",
            "id",
            "event_id",
        ),
    )
    date_col = first_col(
        pdf,
        (
            "game_date",
            "schedule_date",
            "date",
            "start_date",
        ),
    )

    if gid_col is None or date_col is None:
        return set()

    parsed = pd.to_datetime(
        pdf[date_col],
        errors="coerce",
    )

    return {
        str(game_id).strip()
        for game_id, game_date in zip(
            pdf[gid_col],
            parsed,
        )
        if (
            str(game_id).strip()
            and not pd.isna(game_date)
            and game_date.date() < target_day
        )
    }


def frame_strictly_before_game(
    frame: Any,
    *,
    target_day: date,
    prior_game_ids: set[str],
) -> pl.DataFrame:
    if frame_empty(frame):
        return pl.DataFrame()

    pdf = as_pandas(frame)

    gid_col = first_col(
        pdf,
        (
            "game_id",
            "id",
            "event_id",
        ),
    )

    if gid_col is not None and prior_game_ids:
        ids = (
            pdf[gid_col]
            .astype(str)
            .str.strip()
        )
        return pl.from_pandas(
            pdf.loc[
                ids.isin(prior_game_ids)
            ].reset_index(drop=True)
        )

    date_col = first_col(
        pdf,
        (
            "game_date",
            "schedule_date",
            "date",
            "start_date",
            "game_datetime",
            "date_time",
        ),
    )

    if date_col is None:
        return pl.DataFrame()

    parsed = pd.to_datetime(
        pdf[date_col],
        errors="coerce",
        utc=False,
    )

    mask = parsed.notna() & (
        parsed.dt.date < target_day
    )

    return pl.from_pandas(
        pdf.loc[
            mask
        ].reset_index(drop=True)
    )


def research_pbp_strictly_before_target(
    schedule: Any,
    pbp: Any,
    *,
    target_day: date,
) -> pl.DataFrame:
    """Return leakage-safe PBP for research-only pregame microstats.

    The helper deliberately uses the same prior-game boundary as the production
    lineup/goalie feature builders.  Expected-assist and zone-transition
    research must call their SportsDataverse functions only on this filtered
    frame; the target date and every later game are excluded.
    """
    prior_game_ids = prior_game_ids_for_target(
        schedule,
        target_day,
    )

    return frame_strictly_before_game(
        pbp,
        target_day=target_day,
        prior_game_ids=prior_game_ids,
    )


def goalie_gsax_lookup(
    gsax: Any,
) -> dict[str, dict[str, Any]]:
    if frame_empty(gsax):
        return {}

    pdf = as_pandas(gsax)

    required = {
        "player_id",
        "goalie",
        "gsax",
    }

    if not required.issubset(
        set(pdf.columns)
    ):
        raise ValueError(
            "SportsDataverse goalie GSAx output missing "
            f"required columns: {sorted(required - set(pdf.columns))}"
        )

    out: dict[
        str,
        dict[str, Any],
    ] = {}

    for _, row in pdf.iterrows():
        player_id = normalize_player_id(
            row.get(
                "player_id"
            )
        )

        if not player_id:
            continue

        out[player_id] = {
            "player_id": player_id,
            "goalie": str(
                row.get(
                    "goalie",
                    "",
                )
            ).strip(),
            "gsax": pd.to_numeric(
                row.get(
                    "gsax"
                ),
                errors="coerce",
            ),
            "gsax_per_60": pd.to_numeric(
                row.get(
                    "gsax_per_60"
                ),
                errors="coerce",
            ),
        }

    return out


def goalie_usage_candidates(
    prior_pbp: Any,
    team: str,
) -> list[dict[str, Any]]:
    if frame_empty(prior_pbp):
        return []

    pdf = as_pandas(
        prior_pbp
    )

    game_id_col = first_col(
        pdf,
        (
            "game_id",
            "id",
            "event_id",
        ),
    )

    records: list[
        dict[str, str]
    ] = []

    for side in (
        "home",
        "away",
    ):
        team_col = first_col(
            pdf,
            (
                f"{side}_abbr",
                f"{side}_team_abbr",
                f"{side}_team_abbrev",
                f"{side}_team",
            ),
        )
        goalie_id_col = first_col(
            pdf,
            (
                f"{side}_goalie_id",
                f"{side}_goalie_player_id",
            ),
        )
        goalie_name_col = first_col(
            pdf,
            (
                f"{side}_goalie",
                f"{side}_goalie_name",
            ),
        )

        if (
            team_col is None
            or goalie_id_col is None
        ):
            continue

        side_rows = pdf.loc[
            pdf[team_col]
            .astype(str)
            .str.strip()
            .eq(str(team).strip())
        ]

        for row_number, row in side_rows.iterrows():
            player_id = normalize_player_id(
                row.get(
                    goalie_id_col
                )
            )

            if not player_id:
                continue

            game_id = (
                str(
                    row.get(
                        game_id_col,
                        "",
                    )
                ).strip()
                if game_id_col is not None
                else str(row_number)
            )

            records.append(
                {
                    "player_id": player_id,
                    "goalie": (
                        str(
                            row.get(
                                goalie_name_col,
                                "",
                            )
                        ).strip()
                        if goalie_name_col is not None
                        else ""
                    ),
                    "game_id": game_id,
                }
            )

    if not records:
        return []

    usage = (
        pd.DataFrame(records)
        .drop_duplicates(
            subset=[
                "player_id",
                "game_id",
            ]
        )
    )

    names = (
        usage.groupby(
            "player_id",
            dropna=False,
        )["goalie"]
        .agg(
            lambda values: next(
                (
                    str(value).strip()
                    for value in values
                    if str(value).strip()
                ),
                "",
            )
        )
    )

    counts = (
        usage.groupby(
            "player_id",
            dropna=False,
        )["game_id"]
        .nunique()
        .rename(
            "prior_appearances"
        )
    )

    combined = (
        pd.concat(
            [
                names,
                counts,
            ],
            axis=1,
        )
        .reset_index()
    )

    return sorted(
        combined.to_dict(
            orient="records"
        ),
        key=lambda row: (
            -int(
                row.get(
                    "prior_appearances",
                    0,
                )
            ),
            str(
                row.get(
                    "goalie",
                    "",
                )
            ),
            str(
                row.get(
                    "player_id",
                    "",
                )
            ),
        ),
    )


def goalie_value(
    gsax_lookup: dict[
        str,
        dict[str, Any],
    ],
    candidate: dict[str, Any] | None,
) -> Any:
    if not candidate:
        return None

    player_id = normalize_player_id(
        candidate.get(
            "player_id"
        )
    )

    row = gsax_lookup.get(
        player_id
    )

    if row is None:
        return None

    value = row.get(
        "gsax"
    )

    if pd.isna(value):
        return None

    return float(value)


def goalie_name(
    gsax_lookup: dict[
        str,
        dict[str, Any],
    ],
    candidate: dict[str, Any] | None,
) -> str:
    if not candidate:
        return ""

    name = str(
        candidate.get(
            "goalie",
            "",
        )
    ).strip()

    if name:
        return name

    player_id = normalize_player_id(
        candidate.get(
            "player_id"
        )
    )

    return str(
        gsax_lookup.get(
            player_id,
            {},
        ).get(
            "goalie",
            "",
        )
    ).strip()


def build_game_goalie_features_asof(
    schedule: Any,
    pbp: Any,
    shifts: Any,
    *,
    current: bool,
    generated_at_utc: datetime,
) -> pl.DataFrame:
    if (
        frame_empty(schedule)
        or frame_empty(pbp)
    ):
        return pl.DataFrame()

    games = schedule_games_with_cutoffs(
        schedule
    )

    if games.empty:
        return pl.DataFrame()

    generated_at_utc = (
        generated_at_utc.astimezone(
            UTC
        )
    )

    cache: dict[
        str,
        tuple[
            pl.DataFrame,
            dict[str, dict[str, Any]],
        ],
    ] = {}

    rows: list[
        dict[str, Any]
    ] = []

    for _, game in games.iterrows():
        game_id = str(
            game.get(
                "game_id",
                "",
            )
        ).strip()

        game_date = str(
            game.get(
                "game_date",
                "",
            )
        ).strip()

        try:
            target_day = date.fromisoformat(
                game_date
            )
        except ValueError:
            continue

        game_start_cutoff = parse_timestamp_utc(
            game.get(
                "pregame_cutoff_utc",
                "",
            )
        )

        cutoff_source = str(
            game.get(
                "pregame_cutoff_source",
                "",
            )
        ).strip()

        if (
            not game_id
            or game_start_cutoff is None
            or cutoff_source
            == "conservative_game_date_start_et"
        ):
            continue

        production_cutoff = (
            goalie_production_cutoff_utc(
                game_start_cutoff
            )
        )

        snapshot_as_of = (
            generated_at_utc
            if current
            else production_cutoff
        )

        # Production goalie information is frozen at T-60.
        # Current runs after T-60 cannot backfill later information
        # into the game-level goalie feature snapshot.
        if snapshot_as_of > production_cutoff:
            continue

        cache_key = target_day.isoformat()

        if cache_key not in cache:
            prior_game_ids = (
                prior_game_ids_for_target(
                    schedule,
                    target_day,
                )
            )

            prior_pbp = (
                frame_strictly_before_game(
                    pbp,
                    target_day=target_day,
                    prior_game_ids=prior_game_ids,
                )
            )

            prior_shifts = (
                frame_strictly_before_game(
                    shifts,
                    target_day=target_day,
                    prior_game_ids=prior_game_ids,
                )
                if not frame_empty(
                    shifts
                )
                else pl.DataFrame()
            )

            gsax = (
                nhl.nhl_goalie_gsax(
                    prior_pbp,
                    prior_shifts,
                )
                if not prior_pbp.is_empty()
                else pl.DataFrame()
            )

            cache[
                cache_key
            ] = (
                prior_pbp,
                goalie_gsax_lookup(
                    gsax
                ),
            )

        (
            prior_pbp,
            gsax_lookup,
        ) = cache[
            cache_key
        ]

        row: dict[
            str,
            Any
        ] = {
            "game_id": game_id,
            "game_date": game_date,
            "home_team": str(
                game.get(
                    "home_team",
                    "",
                )
            ).strip(),
            "away_team": str(
                game.get(
                    "away_team",
                    "",
                )
            ).strip(),
            "pregame_cutoff_utc": utc_iso(
                game_start_cutoff
            ),
            "goalie_decision_cutoff_utc": utc_iso(
                production_cutoff
            ),
            "goalie_decision_cutoff_source": (
                GOALIE_PRODUCTION_CUTOFF_SOURCE
            ),
            "goalie_snapshot_as_of_utc": utc_iso(
                snapshot_as_of
            ),
            "pregame_cutoff_source": cutoff_source,
        }

        for side in (
            "home",
            "away",
        ):
            team = row[
                f"{side}_team"
            ]

            candidates = (
                goalie_usage_candidates(
                    prior_pbp,
                    team,
                )
            )

            starter = (
                candidates[0]
                if candidates
                else None
            )

            backup = (
                candidates[1]
                if len(candidates) > 1
                else None
            )

            status = (
                "projected"
                if starter is not None
                else "unknown"
            )

            row[
                f"{side}_expected_starter"
            ] = goalie_name(
                gsax_lookup,
                starter,
            )

            row[
                f"{side}_starter_gsax"
            ] = goalie_value(
                gsax_lookup,
                starter,
            )

            row[
                f"{side}_backup_gsax"
            ] = goalie_value(
                gsax_lookup,
                backup,
            )

            row[
                f"{side}_goalie_status"
            ] = status

            row[
                f"{side}_goalie_status_observed_at"
            ] = utc_iso(
                snapshot_as_of
            )

            row[
                f"{side}_goalie_status_source"
            ] = (
                "sportsdataverse_prior_goalie_usage_projection"
                if status == "projected"
                else "sportsdataverse_no_pregame_starter_evidence"
            )

        home_gsax = row.get(
            "home_starter_gsax"
        )

        away_gsax = row.get(
            "away_starter_gsax"
        )

        if (
            home_gsax is None
            or away_gsax is None
        ):
            row[
                "starter_gsax_differential"
            ] = None
        else:
            row[
                "starter_gsax_differential"
            ] = (
                float(home_gsax)
                - float(away_gsax)
            )

        rows.append(
            row
        )

    if not rows:
        return pl.DataFrame()

    return pl.from_pandas(
        pd.DataFrame(
            rows
        )
    )




def _unit_player_team_lookup(
    *unit_frames: Any,
) -> dict[str, str]:
    counts: dict[
        str,
        dict[str, int],
    ] = {}

    for frame in unit_frames:
        if frame_empty(frame):
            continue

        pdf = as_pandas(frame)

        if (
            "team" not in pdf.columns
            or "unit_ids" not in pdf.columns
        ):
            continue

        for _, row in pdf.iterrows():
            team = str(
                row.get(
                    "team",
                    "",
                )
            ).strip()

            if not team:
                continue

            raw_ids = str(
                row.get(
                    "unit_ids",
                    "",
                )
            ).strip()

            for raw_id in raw_ids.split("-"):
                player_id = normalize_player_id(
                    raw_id
                )

                if not player_id:
                    continue

                counts.setdefault(
                    player_id,
                    {},
                )[team] = (
                    counts.setdefault(
                        player_id,
                        {},
                    ).get(
                        team,
                        0,
                    )
                    + 1
                )

    lookup: dict[
        str,
        str,
    ] = {}

    for player_id, by_team in counts.items():
        lookup[
            player_id
        ] = sorted(
            by_team.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )[0][0]

    return lookup


def _weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float | None:
    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    numeric_weights = pd.to_numeric(
        weights,
        errors="coerce",
    )

    mask = (
        numeric_values.notna()
        & numeric_weights.notna()
        & numeric_weights.gt(0)
    )

    if not mask.any():
        valid = numeric_values.dropna()
        return (
            float(
                valid.mean()
            )
            if not valid.empty
            else None
        )

    denom = float(
        numeric_weights.loc[
            mask
        ].sum()
    )

    if denom <= 0:
        return None

    return float(
        (
            numeric_values.loc[
                mask
            ]
            * numeric_weights.loc[
                mask
            ]
        ).sum()
        / denom
    )


def _player_metric_by_team(
    frame: Any,
    player_team: dict[str, str],
    *,
    value_col: str,
    mode: str,
    weight_col: str | None = None,
) -> dict[str, float]:
    if frame_empty(frame):
        return {}

    pdf = as_pandas(frame)

    if (
        "player_id" not in pdf.columns
        or value_col not in pdf.columns
    ):
        return {}

    work = pdf.copy()
    work["_player_key"] = work[
        "player_id"
    ].map(
        normalize_player_id
    )
    work["_team"] = work[
        "_player_key"
    ].map(
        player_team
    )
    work["_value"] = pd.to_numeric(
        work[
            value_col
        ],
        errors="coerce",
    )

    work = work.loc[
        work[
            "_team"
        ].notna()
        & work[
            "_value"
        ].notna()
    ].copy()

    if work.empty:
        return {}

    output: dict[
        str,
        float,
    ] = {}

    for team, group in work.groupby(
        "_team",
        dropna=False,
    ):
        if mode == "sum":
            output[
                str(team)
            ] = float(
                group[
                    "_value"
                ].sum()
            )
            continue

        if (
            mode == "weighted_mean"
            and weight_col is not None
            and weight_col in group.columns
        ):
            value = _weighted_mean(
                group[
                    "_value"
                ],
                group[
                    weight_col
                ],
            )
        else:
            value = float(
                group[
                    "_value"
                ].mean()
            )

        if value is not None:
            output[
                str(team)
            ] = float(
                value
            )

    return output


def _unit_metric_by_team(
    frame: Any,
) -> dict[str, float]:
    if frame_empty(frame):
        return {}

    pdf = as_pandas(frame)

    required = {
        "team",
        "unit_value",
    }

    if not required.issubset(
        set(
            pdf.columns
        )
    ):
        return {}

    output: dict[
        str,
        float,
    ] = {}

    for team, group in pdf.groupby(
        "team",
        dropna=False,
    ):
        team_name = str(
            team
        ).strip()

        if not team_name:
            continue

        if "toi_minutes" in group.columns:
            value = _weighted_mean(
                group[
                    "unit_value"
                ],
                group[
                    "toi_minutes"
                ],
            )
        else:
            values = pd.to_numeric(
                group[
                    "unit_value"
                ],
                errors="coerce",
            ).dropna()

            value = (
                float(
                    values.mean()
                )
                if not values.empty
                else None
            )

        if value is not None:
            output[
                team_name
            ] = float(
                value
            )

    return output


def compute_lineup_team_metrics(
    prior_pbp: Any,
    prior_shifts: Any,
) -> dict[str, dict[str, float]]:
    if (
        frame_empty(
            prior_pbp
        )
        or frame_empty(
            prior_shifts
        )
    ):
        return {}

    rapm = nhl.nhl_skater_rapm(
        prior_pbp,
        prior_shifts,
    )

    war = clean_war_rows(
        nhl.nhl_skater_war(
            prior_pbp,
            prior_shifts,
        )
    )

    special = nhl.nhl_special_teams_value(
        prior_pbp,
        prior_shifts,
    )

    forward_units = nhl.nhl_unit_ratings(
        prior_pbp,
        prior_shifts,
        unit_type="forward_line",
        min_toi=0.0,
    )

    defense_units = nhl.nhl_unit_ratings(
        prior_pbp,
        prior_shifts,
        unit_type="defense_pair",
        min_toi=0.0,
    )

    player_team = _unit_player_team_lookup(
        forward_units,
        defense_units,
    )

    metric_maps = {
        "skater_rapm": (
            _player_metric_by_team(
                rapm,
                player_team,
                value_col="xg_rapm",
                mode="weighted_mean",
                weight_col="toi_minutes",
            )
        ),
        "skater_war": (
            _player_metric_by_team(
                war,
                player_team,
                value_col="war",
                mode="sum",
            )
        ),
        "pp_value": (
            _player_metric_by_team(
                special,
                player_team,
                value_col="pp_value",
                mode="sum",
            )
        ),
        "pk_value": (
            _player_metric_by_team(
                special,
                player_team,
                value_col="pk_value",
                mode="sum",
            )
        ),
        "forward_line_strength": (
            _unit_metric_by_team(
                forward_units
            )
        ),
        "defense_pair_strength": (
            _unit_metric_by_team(
                defense_units
            )
        ),
    }

    teams = sorted(
        {
            team
            for metric_map in metric_maps.values()
            for team in metric_map
        }
    )

    return {
        team: {
            metric: metric_maps[
                metric
            ].get(
                team
            )
            for metric in LINEUP_TEAM_METRICS
        }
        for team in teams
    }


def build_game_lineup_features_asof(
    schedule: Any,
    pbp: Any,
    shifts: Any,
    *,
    current: bool,
    generated_at_utc: datetime,
) -> pl.DataFrame:
    if frame_empty(
        schedule
    ):
        return pl.DataFrame()

    games = schedule_games_with_cutoffs(
        schedule
    )

    if games.empty:
        return pl.DataFrame()

    generated_at_utc = (
        generated_at_utc.astimezone(
            UTC
        )
    )

    cache: dict[
        str,
        dict[
            str,
            dict[str, float],
        ],
    ] = {}

    rows: list[
        dict[str, Any]
    ] = []

    for _, game in games.iterrows():
        game_id = str(
            game.get(
                "game_id",
                "",
            )
        ).strip()

        game_date = str(
            game.get(
                "game_date",
                "",
            )
        ).strip()

        try:
            target_day = date.fromisoformat(
                game_date
            )
        except ValueError:
            continue

        game_start = parse_timestamp_utc(
            game.get(
                "pregame_cutoff_utc",
                "",
            )
        )

        cutoff_source = str(
            game.get(
                "pregame_cutoff_source",
                "",
            )
        ).strip()

        if (
            not game_id
            or game_start is None
            or cutoff_source
            == "conservative_game_date_start_et"
        ):
            continue

        decision_cutoff = (
            lineup_production_cutoff_utc(
                game_start
            )
        )

        snapshot_as_of = (
            generated_at_utc
            if current
            else decision_cutoff
        )

        # Production lineup/player information is frozen at T-60.
        # Later current runs cannot backfill information into the feature row.
        if snapshot_as_of > decision_cutoff:
            continue

        cache_key = target_day.isoformat()

        if cache_key not in cache:
            prior_game_ids = (
                prior_game_ids_for_target(
                    schedule,
                    target_day,
                )
            )

            prior_pbp = (
                frame_strictly_before_game(
                    pbp,
                    target_day=target_day,
                    prior_game_ids=prior_game_ids,
                )
                if not frame_empty(
                    pbp
                )
                else pl.DataFrame()
            )

            prior_shifts = (
                frame_strictly_before_game(
                    shifts,
                    target_day=target_day,
                    prior_game_ids=prior_game_ids,
                )
                if not frame_empty(
                    shifts
                )
                else pl.DataFrame()
            )

            cache[
                cache_key
            ] = (
                compute_lineup_team_metrics(
                    prior_pbp,
                    prior_shifts,
                )
            )

        team_metrics = cache[
            cache_key
        ]

        row: dict[
            str,
            Any,
        ] = {
            "game_id": game_id,
            "game_date": game_date,
            "home_team": str(
                game.get(
                    "home_team",
                    "",
                )
            ).strip(),
            "away_team": str(
                game.get(
                    "away_team",
                    "",
                )
            ).strip(),
            "pregame_cutoff_utc": utc_iso(
                game_start
            ),
            "lineup_decision_cutoff_utc": utc_iso(
                decision_cutoff
            ),
            "lineup_decision_cutoff_source": (
                LINEUP_PRODUCTION_CUTOFF_SOURCE
            ),
            "lineup_snapshot_as_of_utc": utc_iso(
                snapshot_as_of
            ),
            "pregame_cutoff_source": cutoff_source,
        }

        for side in (
            "home",
            "away",
        ):
            team = row[
                f"{side}_team"
            ]

            metrics = team_metrics.get(
                team,
                {},
            )

            for metric in (
                LINEUP_TEAM_METRICS
            ):
                row[
                    f"{side}_{metric}"
                ] = metrics.get(
                    metric
                )

            # SportsDataverse provides the historical performance inputs,
            # not a timestamped NHL pregame lineup-confirmation feed.
            # Therefore actual lineup state remains unknown rather than
            # being inferred from postgame participation.
            row[
                f"{side}_lineup_status"
            ] = "unknown"

            row[
                f"{side}_lineup_observed_at"
            ] = ""

            row[
                f"{side}_lineup_source"
            ] = (
                "sportsdataverse_prior_game_player_pool_no_lineup_confirmation"
                if metrics
                else "sportsdataverse_no_pregame_lineup_evidence"
            )

        for metric in (
            LINEUP_TEAM_METRICS
        ):
            home_value = row.get(
                f"home_{metric}"
            )
            away_value = row.get(
                f"away_{metric}"
            )

            if (
                home_value is None
                or away_value is None
                or pd.isna(
                    home_value
                )
                or pd.isna(
                    away_value
                )
            ):
                row[
                    f"{metric}_differential"
                ] = None
            else:
                row[
                    f"{metric}_differential"
                ] = (
                    float(
                        home_value
                    )
                    - float(
                        away_value
                    )
                )

        rows.append(
            row
        )

    if not rows:
        return pl.DataFrame()

    return pl.from_pandas(
        pd.DataFrame(
            rows
        )
    )



def pull_advanced_player_strength(
    failures: list[str],
    *,
    current: bool,
    prefix: str,
    context: dict[str, Any],
):
    """Legacy goalie diagnostic only.

    Production player/lineup features are generated per game by
    build_game_lineup_features_asof under the fixed T-60 contract.
    """
    pbp = context.get(
        "pbp"
    )

    shifts = context.get(
        "shifts"
    )

    if (
        frame_empty(pbp)
        or frame_empty(shifts)
    ):
        return

    gsax = safe_pull(
        failures,
        "goalie",
        "goalie_gsax",
        nhl.nhl_goalie_gsax,
        pbp,
        shifts,
    )

    if gsax is not None:
        save_object(
            "goalie",
            "goalie_gsax",
            gsax,
            prefix=prefix,
            current=current,
        )


def pull_fatigue(

    failures: list[str],
    *,
    current: bool,
    target_date: date,
    season: int,
    prefix: str,
    teams: list[str],
    schedule: Any,
):
    if not current:
        fatigue = (
            build_fatigue_from_league_schedule(
                schedule
            )
        )

        save_object(
            "fatigue",
            "fatigue",
            fatigue,
            prefix=prefix,
            current=False,
        )

        return

    fatigue_frames = []

    for team in teams:
        parsed = safe_pull(
            failures,
            "fatigue",
            f"{team}_club_schedule",
            nhl.nhl_club_schedule_season,
            team=team,
        )

        if parsed is not None:
            save_object(
                "fatigue",
                f"{team}_club_schedule",
                parsed,
                prefix=prefix,
                current=True,
            )

            f = (
                build_fatigue_from_team_schedule(
                    team,
                    parsed,
                    target_date=target_date,
                )
            )

            if not f.is_empty():
                fatigue_frames.append(
                    f
                )

        raw = safe_pull(
            failures,
            "fatigue",
            f"{team}_club_schedule_raw",
            nhl.nhl_club_schedule_season,
            team=team,
            return_parsed=False,
        )

        if raw is not None:
            save_object(
                "fatigue",
                f"{team}_club_schedule_raw",
                raw,
                prefix=prefix,
                current=True,
            )

    fatigue = (
        pl.concat(
            fatigue_frames,
            how="diagonal_relaxed",
        )
        if fatigue_frames
        else pl.DataFrame()
    )

    save_object(
        "fatigue",
        "fatigue",
        fatigue,
        prefix=prefix,
        current=True,
    )


def pull_predictions(
    failures: list[str],
    *,
    current: bool,
    target_date: date,
    season: int,
    prefix: str,
    schedule: Any,
    slate: Any,
    ratings: Any,
    context: dict[str, Any] | None,
):
    if current:
        if (
            frame_empty(slate)
            or frame_empty(ratings)
        ):
            save_object(
                "sdv_predictions",
                "predictions",
                pl.DataFrame(),
                prefix=prefix,
                current=True,
            )

            return

        games = prediction_games(
            slate
        )

        if games.is_empty():
            return

        if (
            "game_date"
            in games.columns
        ):
            games = games.select(
                "game_id",
                "home_team",
                "away_team",
                "neutral_site",
            )

        pred = safe_pull(
            failures,
            "sdv_predictions",
            "nhl_predict_games",
            nhl.nhl_predict_games,
            games,
            ratings,
        )

        if pred is not None:
            save_object(
                "sdv_predictions",
                "predictions",
                pred,
                prefix=prefix,
                current=True,
            )

        return

    if context is None:
        return

    pbp = context.get(
        "pbp"
    )

    if (
        frame_empty(schedule)
        or frame_empty(pbp)
    ):
        return

    pred = safe_pull(
        failures,
        "sdv_predictions",
        "historical_predictions",
        historical_predictions,
        schedule,
        pbp,
    )

    if pred is not None:
        save_object(
            "sdv_predictions",
            "predictions",
            pred,
            prefix=prefix,
            current=False,
        )


def espn_event_ids(
    schedule: Any,
) -> list[str]:
    if frame_empty(schedule):
        return []

    pdf = as_pandas(
        schedule
    )

    id_col = first_col(
        pdf,
        (
            "game_id",
            "id",
            "event_id",
        ),
    )

    if id_col is None:
        return []

    return sorted(
        {
            str(v).strip()
            for v in (
                pdf[
                    id_col
                ]
                .dropna()
                .tolist()
            )
            if str(v).strip()
        }
    )


def pull_odds(
    failures: list[str],
    *,
    current: bool,
    target_date: date,
    season: int,
    prefix: str,
):
    if current:
        query_dates = [
            target_date
        ]

    else:
        historical_schedule = (
            safe_pull(
                failures,
                "odds",
                f"load_nhl_schedule_{season}",
                nhl.load_nhl_schedule,
                season,
            )
        )

        query_dates = (
            historical_schedule_game_dates(
                historical_schedule
            )
        )

        if not query_dates:
            failures.append(
                "odds/historical_schedule_dates: "
                f"no game dates found for season={season}"
            )

            save_object(
                "odds",
                "espn_schedule",
                pl.DataFrame(),
                prefix=prefix,
                current=False,
            )

            save_object(
                "odds",
                "espn_scoreboard_raw",
                {},
                prefix=prefix,
                current=False,
            )

            save_object(
                "odds",
                "espn_odds",
                pl.DataFrame(),
                prefix=prefix,
                current=False,
            )

            save_object(
                "odds",
                "espn_odds_raw",
                {},
                prefix=prefix,
                current=False,
            )

            return

    schedule_frames = []
    scoreboard_raw_by_date: dict[
        str,
        Any,
    ] = {}

    for game_day in query_dates:
        dates_arg = int(
            game_day.strftime(
                "%Y%m%d"
            )
        )

        espn_schedule_day = (
            safe_pull(
                failures,
                "odds",
                f"espn_schedule_{dates_arg}",
                nhl.espn_nhl_schedule,
                dates=dates_arg,
                limit=5000,
            )
        )

        if (
            espn_schedule_day
            is not None
            and not frame_empty(
                espn_schedule_day
            )
        ):
            schedule_frames.append(
                pl.from_pandas(
                    as_pandas(
                        espn_schedule_day
                    )
                )
            )

        scoreboard_raw = (
            safe_pull(
                failures,
                "odds",
                f"espn_scoreboard_raw_{dates_arg}",
                nhl.espn_nhl_scoreboard,
                dates=dates_arg,
                limit=5000,
                return_parsed=False,
            )
        )

        if scoreboard_raw is not None:
            scoreboard_raw_by_date[
                game_day.isoformat()
            ] = scoreboard_raw

        if not current:
            time.sleep(
                0.02
            )

    espn_schedule = (
        pl.concat(
            schedule_frames,
            how="diagonal_relaxed",
        )
        if schedule_frames
        else pl.DataFrame()
    )

    if (
        not espn_schedule.is_empty()
    ):
        id_col = first_col(
            espn_schedule,
            (
                "game_id",
                "id",
                "event_id",
            ),
        )

        if id_col is not None:
            pdf = as_pandas(
                espn_schedule
            )

            pdf[
                id_col
            ] = (
                pdf[
                    id_col
                ]
                .astype(str)
                .str.strip()
            )

            pdf = (
                pdf.loc[
                    pdf[
                        id_col
                    ].ne("")
                ]
                .drop_duplicates(
                    subset=[
                        id_col
                    ],
                    keep="first",
                )
                .reset_index(
                    drop=True
                )
            )

            espn_schedule = (
                pl.from_pandas(
                    pdf
                )
            )

    save_object(
        "odds",
        "espn_schedule",
        espn_schedule,
        prefix=prefix,
        current=current,
    )

    save_object(
        "odds",
        "espn_scoreboard_raw",
        scoreboard_raw_by_date,
        prefix=prefix,
        current=current,
    )

    event_ids = espn_event_ids(
        espn_schedule
    )

    print(
        f"ESPN odds discovery | "
        f"dates={len(query_dates)} | "
        f"events={len(event_ids)}"
    )

    odds_frames = []

    raw_by_event: dict[
        str,
        Any,
    ] = {}

    for event_id in event_ids:
        parsed = safe_pull(
            failures,
            "odds",
            f"espn_odds_{event_id}",
            nhl.espn_nhl_game_odds,
            event_id=event_id,
        )

        if (
            parsed is not None
            and not frame_empty(
                parsed
            )
        ):
            pdf = as_pandas(
                parsed
            )

            pdf.insert(
                0,
                "espn_event_id",
                event_id,
            )

            odds_frames.append(
                pl.from_pandas(
                    pdf
                )
            )

        raw = safe_pull(
            failures,
            "odds",
            f"espn_odds_raw_{event_id}",
            nhl.espn_nhl_game_odds,
            event_id=event_id,
            return_parsed=False,
        )

        if raw is not None:
            raw_by_event[
                event_id
            ] = raw

        if not current:
            time.sleep(
                0.05
            )

    combined = (
        pl.concat(
            odds_frames,
            how="diagonal_relaxed",
        )
        if odds_frames
        else pl.DataFrame()
    )

    save_object(
        "odds",
        "espn_odds",
        combined,
        prefix=prefix,
        current=current,
    )

    save_object(
        "odds",
        "espn_odds_raw",
        raw_by_event,
        prefix=prefix,
        current=current,
    )


def run_one(
    *,
    current: bool,
    target_date: date,
    season: int,
    categories: set[str],
) -> list[str]:
    failures: list[str] = []

    prefix = (
        current_or_historical_prefix(
            current=current,
            season=(
                None
                if current
                else season
            ),
        )
    )

    schedule = pl.DataFrame()
    slate = pl.DataFrame()

    if (
        "schedule" in categories
        or "team-strength" in categories
        or "goalie" in categories
        or "lineup-strength" in categories
        or "fatigue" in categories
        or "sdv_predictions" in categories
    ):
        (
            schedule,
            slate,
        ) = pull_schedule(
            failures,
            current=current,
            target_date=target_date,
            season=season,
            prefix=prefix,
        )

    relevant_schedule = (
        slate
        if current
        else schedule
    )

    teams = schedule_teams(
        relevant_schedule
    )

    team_strength_date = (
        target_date
    )

    if not current:
        (
            _,
            max_day,
        ) = historical_schedule_date_bounds(
            schedule
        )

        if max_day is not None:
            team_strength_date = (
                max_day
            )

    ratings = None

    if (
        "team-strength"
        in categories
        or "sdv_predictions"
        in categories
    ):
        ratings = pull_team_strength(
            failures,
            current=current,
            target_date=(
                team_strength_date
                if not current
                else target_date
            ),
            season=season,
            prefix=prefix,
            teams=teams,
            schedule=relevant_schedule,
        )

    rosters: dict[
        str,
        Any,
    ] = {}

    if (
        "lineup-strength"
        in categories
        or "goalie"
        in categories
    ):
        rosters = pull_rosters(
            failures,
            current=current,
            season=season,
            prefix=prefix,
            teams=teams,
        )

    if "goalie" in categories:
        pull_goalie_live_profiles(
            failures,
            current=current,
            prefix=prefix,
            rosters=rosters,
        )

    need_context = bool(
        {
            "goalie",
            "lineup-strength",
            "sdv_predictions",
        }
        & categories
    )

    context = None

    if need_context:
        context = (
            pull_season_context(
                failures,
                season=season,
            )
        )

        save_season_context(
            current=current,
            prefix=prefix,
            context=context,
        )

    if "lineup-strength" in categories:
        save_object(
            "lineup-strength",
            "pregame_feature_evaluation",
            pregame_feature_evaluation(),
            prefix=prefix,
            current=current,
        )

    if (
        "lineup-strength" in categories
        and context is not None
    ):
        lineup_features = safe_pull(
            failures,
            "lineup-strength",
            "game_lineup_features_asof",
            build_game_lineup_features_asof,
            relevant_schedule,
            context.get(
                "pbp"
            ),
            context.get(
                "shifts"
            ),
            current=current,
            generated_at_utc=datetime.now(
                UTC
            ),
        )

        if lineup_features is not None:
            save_object(
                "lineup-strength",
                "game_lineup_features_asof",
                lineup_features,
                prefix=prefix,
                current=current,
            )

    if (
        "goalie" in categories
        and context is not None
    ):
        goalie_features = safe_pull(
            failures,
            "goalie",
            "game_goalie_features_asof",
            build_game_goalie_features_asof,
            relevant_schedule,
            context.get(
                "pbp"
            ),
            context.get(
                "shifts"
            ),
            current=current,
            generated_at_utc=datetime.now(
                UTC
            ),
        )

        if goalie_features is not None:
            save_object(
                "goalie",
                "game_goalie_features_asof",
                goalie_features,
                prefix=prefix,
                current=current,
            )

    if (
        (not current)
        and (
            "team-strength"
            in categories
        )
    ):
        pbp = (
            context.get(
                "pbp"
            )
            if context is not None
            else None
        )

        if frame_empty(
            pbp
        ):
            pbp = safe_pull(
                failures,
                "team-strength",
                "load_nhl_pbp_full",
                nhl.load_nhl_pbp_full,
                season,
            )

        ratings_asof = safe_pull(
            failures,
            "team-strength",
            "historical_team_strength",
            historical_team_strength,
            schedule,
            pbp,
        )

        if ratings_asof is not None:
            save_object(
                "team-strength",
                "team_ratings_asof",
                ratings_asof,
                prefix=prefix,
                current=False,
            )

    if (
        context is not None
        and (
            "goalie"
            in categories
        )
        and (
            "lineup-strength"
            not in categories
        )
    ):
        # Legacy full-season player outputs are diagnostics only.
        # Production lineup/player features are generated above by
        # build_game_lineup_features_asof using the T-60 contract.
        pull_advanced_player_strength(
            failures,
            current=current,
            prefix=prefix,
            context=context,
        )

    if "fatigue" in categories:
        pull_fatigue(
            failures,
            current=current,
            target_date=target_date,
            season=season,
            prefix=prefix,
            teams=teams,
            schedule=schedule,
        )

    if (
        "sdv_predictions"
        in categories
    ):
        pull_predictions(
            failures,
            current=current,
            target_date=target_date,
            season=season,
            prefix=prefix,
            schedule=schedule,
            slate=slate,
            ratings=ratings,
            context=context,
        )

    if "odds" in categories:
        pull_odds(
            failures,
            current=current,
            target_date=target_date,
            season=season,
            prefix=prefix,
        )

    return failures


def main() -> None:
    ensure_dirs()

    args = parse_args()

    categories = (
        requested_categories(
            args
        )
    )

    seasons = (
        requested_seasons(
            args
        )
    )

    now = datetime.now(
        NY
    )

    target_date = now.date()

    all_failures: list[
        str
    ] = []

    if not seasons:
        current_season = (
            season_start_for_day(
                target_date
            )
        )

        print(
            f"SDV NHL current pull | "
            f"date={target_date.isoformat()} "
            f"| season={current_season} "
            f"| categories={sorted(categories)}"
        )

        all_failures.extend(
            run_one(
                current=True,
                target_date=target_date,
                season=current_season,
                categories=categories,
            )
        )

    else:
        for season in seasons:
            print(
                f"SDV NHL historical pull | "
                f"season={season} "
                f"| categories={sorted(categories)}"
            )

            all_failures.extend(
                run_one(
                    current=False,
                    target_date=target_date,
                    season=season,
                    categories=categories,
                )
            )

    if all_failures:
        print(
            "\nSDV pull completed with warnings:",
            file=sys.stderr,
        )

        for item in all_failures:
            print(
                f"  - {item}",
                file=sys.stderr,
            )

    print(
        "SDV NHL pull complete."
    )


if __name__ == "__main__":
    try:
        main()

    except Exception:
        traceback.print_exc()
        sys.exit(1)
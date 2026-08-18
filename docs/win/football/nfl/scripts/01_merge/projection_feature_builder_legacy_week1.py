#!/usr/bin/env python3
"""
Standalone Week 1 NFL projection for the Step 11 CatBoost models.

Builds the exact 260-feature matrix defined by:
  docs/win/football/nfl/models/step11_feature_schema.json

Week 1 leakage rules:
  - current-season Week 1 schedule/market/prediction/injury/depth data only
  - final available prior-season team-stat row per team
  - current Week 1 QB1 identity with latest prior-season QB performance
  - prior-season snap counts, then prior-season participation, for OUT-player usage
  - no current-season Week 1 game stats, snaps, participation, or results
  - depth_starter_changes is 0 because no prior current-season snapshot exists

READS ONLY from docs/win/football/nfl/.
WRITES ONLY:
  docs/win/football/nfl/01_merge/week_1_NFL_enriched.csv
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ============================================================================
# CHANGE THIS EACH SEASON
# ============================================================================
SEASON = 2026
# ============================================================================

WEEK = 1
SCRIPT_VERSION = "2026-08-14-fix2"
EXPECTED_FEATURE_COUNT = 260
MISSING_CAT = "__MISSING__"

OUTPUT_COLUMNS = [
    "predicted_margin",
    "predicted_total",
    "predicted_home_score",
    "predicted_away_score",
    "home_win_probability",
    "away_win_probability",
    "home_cover_probability",
    "away_cover_probability",
    "over_probability",
    "under_probability",
]

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

QB_METRICS = [
    "epa_per_play",
    "cpoe",
    "air_yards",
    "sack_rate",
    "interception_rate",
    "fumble_rate",
]

INJURY_BASE_FEATURES = [
    "inj_out_count",
    "inj_doubtful_count",
    "inj_questionable_count",
    "inj_starter_out_count",
    "inj_top2_depth_out_count",
    "inj_qb1_out",
    "inj_ol_starter_out_count",
    "inj_skill_starter_out_count",
    "inj_front7_starter_out_count",
    "inj_secondary_starter_out_count",
    "inj_offense_unavailable_snap_share",
    "inj_defense_unavailable_snap_share",
    "depth_starter_changes",
]

OFFENSIVE_LINE_POSITIONS = {
    "C", "G", "OG", "LG", "RG", "T", "OT", "LT", "RT", "OL",
}
SKILL_POSITIONS = {"RB", "HB", "FB", "WR", "TE"}
FRONT7_POSITIONS = {"DL", "DE", "DT", "NT", "EDGE", "LB", "ILB", "OLB", "MLB"}
SECONDARY_POSITIONS = {"DB", "CB", "S", "FS", "SS", "NB"}

GSIS_PATTERN = re.compile(r"00-\d{7}")
GAME_ID_PATTERN = re.compile(r"^(\d{4})_(\d{1,2})_([A-Za-z0-9]+)_([A-Za-z0-9]+)$")


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists():
            return parent
    try:
        return here.parents[6]
    except IndexError as exc:
        raise RuntimeError(f"Cannot resolve repository root from {here}") from exc


def nfl_root() -> Path:
    root = repo_root() / "docs/win/football/nfl"
    if not root.exists():
        raise FileNotFoundError(f"NFL root does not exist: {root}")
    return root


def clean(value: object) -> str:
    if value is None:
        return ""
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"}:
        return ""
    return text


def clean_id(value: object) -> str:
    text = clean(value)
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return text


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", clean(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return " ".join(text.split())


def normalize_position(value: object) -> str:
    return clean(value).upper().replace(" ", "")


def position_for_grouping(value: object) -> str:
    """Normalize common depth-slot variants to the model's broad position codes."""
    pos = normalize_position(value)
    aliases = {
        "LDE": "DE", "RDE": "DE",
        "LDT": "DT", "RDT": "DT",
        "LCB": "CB", "RCB": "CB",
        "LILB": "ILB", "RILB": "ILB",
        "LOLB": "OLB", "ROLB": "OLB",
        "SLB": "LB", "WLB": "LB", "MLB": "MLB",
        "HB": "HB", "FB": "FB",
    }
    return aliases.get(pos, pos)


def parse_float(value: object) -> float | None:
    text = clean(value)
    if not text:
        return None
    text = text.replace(",", "").replace("%", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_int(value: object) -> int | None:
    number = parse_float(value)
    if number is None or abs(number - round(number)) > 1e-9:
        return None
    return int(round(number))


def normalize_percentage(value: object) -> float:
    text = clean(value)
    if not text:
        return 0.0
    had_percent = text.endswith("%")
    number = parse_float(text)
    if number is None:
        return 0.0
    if had_percent or number > 1.0:
        number /= 100.0
    return max(0.0, min(1.0, number))


def parse_timestamp(value: object) -> pd.Timestamp | None:
    text = clean(value)
    if not text:
        return None
    try:
        ts = pd.to_datetime(text, utc=True, errors="coerce")
    except Exception:
        return None
    return None if pd.isna(ts) else ts


def normalize_game_id(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def read_csv(path: Path, *, allow_empty_rows: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False)
    if len(df.columns) != len(set(df.columns)):
        raise ValueError(f"{path}: duplicate column names")
    if not allow_empty_rows and df.empty:
        raise ValueError(f"{path}: no data rows")
    if "game_id" in df.columns:
        df["game_id"] = normalize_game_id(df["game_id"])
    return df


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Could not read parquet file {path}: {exc}") from exc


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def choose_column(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    required: bool = False,
    label: str = "source",
) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        raise ValueError(f"{label}: none of these columns are present: {candidates}")
    return None


def require_unique_game_id(df: pd.DataFrame, label: str) -> None:
    require_columns(df, ["game_id"], label)
    blank_mask = df["game_id"].map(clean).eq("")
    if blank_mask.any():
        raise ValueError(f"{label}: blank game_id rows: {df.index[blank_mask].tolist()[:10]}")
    duplicated = df.loc[df["game_id"].duplicated(keep=False), "game_id"].unique().tolist()
    if duplicated:
        raise ValueError(f"{label}: duplicate game_id values: {duplicated[:10]}")


def merge_game_source(
    base: pd.DataFrame,
    source: pd.DataFrame,
    columns: list[str],
    label: str,
) -> pd.DataFrame:
    require_unique_game_id(base, "Week 1 working table")
    require_unique_game_id(source, label)
    require_columns(source, ["game_id", *columns], label)

    source_ids = set(source["game_id"])
    missing_ids = [gid for gid in base["game_id"] if gid not in source_ids]
    if missing_ids:
        raise ValueError(f"{label}: missing game_id rows required by Week 1 input: {missing_ids[:10]}")

    # Working/model columns may be rebuilt from a more authoritative mandatory source.
    overlap = [column for column in columns if column in base.columns]
    if overlap:
        base = base.drop(columns=overlap)

    before_ids = base["game_id"].tolist()
    merged = base.merge(
        source[["game_id", *columns]].copy(),
        on="game_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if merged["game_id"].tolist() != before_ids:
        raise RuntimeError(f"{label}: merge changed Week 1 row order")
    return merged


# ---------------------------------------------------------------------------
# Team normalization
# ---------------------------------------------------------------------------

class TeamNormalizer:
    def __init__(self, path: Path) -> None:
        df = read_csv(path)
        required = [
            "canonical_team", "team_abbr", "alias", "location",
            "team_name", "team_slug", "nickname", "shortDisplayName",
        ]
        require_columns(df, required, str(path))

        self.alias_to_abbr: dict[str, str] = {}
        self.canonical_abbrs: set[str] = set()

        for _, row in df.iterrows():
            abbr = clean(row["team_abbr"]).upper()
            if not abbr:
                continue
            self.canonical_abbrs.add(abbr)

        def add_alias(value: object, abbr: str) -> None:
            key = normalize_name(value)
            if not key:
                return
            previous = self.alias_to_abbr.get(key)
            if previous is not None and previous != abbr:
                raise ValueError(
                    f"{path}: ambiguous team alias {clean(value)!r}: {previous} vs {abbr}"
                )
            self.alias_to_abbr[key] = abbr

        for _, row in df.iterrows():
            abbr = clean(row["team_abbr"]).upper()
            if not abbr:
                continue
            location = clean(row["location"])
            team_name = clean(row["team_name"])
            candidates: list[object] = [
                abbr,
                row["canonical_team"],
                row["alias"],
                row["team_slug"],
                row["nickname"],
                row["shortDisplayName"],
                team_name,
            ]
            # Use location only as part of a full team name. Location alone is
            # ambiguous for teams such as the Rams and Chargers.
            if location and team_name:
                candidates.append(f"{location} {team_name}")
                candidates.append(f"{location}+{team_name}")
            for candidate in candidates:
                add_alias(candidate, abbr)

        # Historical aliases resolve to whichever representation team_map.csv
        # declares canonical for the current repository.
        for left, right in [("WAS", "WSH"), ("LA", "LAR"), ("JAC", "JAX")]:
            if right in self.canonical_abbrs:
                add_alias(left, right)
            elif left in self.canonical_abbrs:
                add_alias(right, left)

        if not self.alias_to_abbr:
            raise RuntimeError(f"{path}: no team aliases loaded")

    def resolve(self, value: object) -> str:
        raw = clean(value)
        if not raw:
            return ""
        upper = raw.upper()
        if upper in self.canonical_abbrs:
            return upper
        key = normalize_name(raw)
        abbr = self.alias_to_abbr.get(key)
        if abbr:
            return abbr
        raise ValueError(f"Could not resolve NFL team to canonical abbreviation: {raw!r}")


def validate_team_alignment(
    base: pd.DataFrame,
    source: pd.DataFrame,
    normalizer: TeamNormalizer,
    label: str,
    *,
    home_column: str = "home_team",
    away_column: str = "away_team",
) -> None:
    if home_column not in source.columns or away_column not in source.columns:
        return
    lookup = source.set_index("game_id")
    for _, row in base.iterrows():
        gid = row["game_id"]
        if gid not in lookup.index:
            continue
        source_row = lookup.loc[gid]
        base_home = normalizer.resolve(row["home_team"])
        base_away = normalizer.resolve(row["away_team"])
        source_home = normalizer.resolve(source_row[home_column])
        source_away = normalizer.resolve(source_row[away_column])
        if (base_home, base_away) != (source_home, source_away):
            raise ValueError(
                f"{label}: team mismatch for game_id={gid}: "
                f"base={base_away}@{base_home}, source={source_away}@{source_home}"
            )


# ---------------------------------------------------------------------------
# Player crosswalk, depth, injuries, snap/participation fallback
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlayerRecord:
    gsis_id: str = ""
    pfr_id: str = ""
    espn_id: str = ""
    name: str = ""
    position: str = ""


class PlayerCrosswalk:
    def __init__(self, df: pd.DataFrame) -> None:
        gsis_col = choose_column(df, ["gsis_id"], required=True, label="players.parquet")
        pfr_col = choose_column(df, ["pfr_id", "pfr_player_id"])
        espn_col = choose_column(df, ["espn_id"])
        name_col = choose_column(df, ["display_name", "full_name", "player_name"])
        position_col = choose_column(df, ["position", "position_group"])

        self.by_gsis: dict[str, PlayerRecord] = {}
        self.by_pfr: dict[str, PlayerRecord] = {}
        self.by_espn: dict[str, PlayerRecord] = {}
        name_candidates: dict[str, list[PlayerRecord]] = {}

        for _, row in df.iterrows():
            gsis = clean_id(row[gsis_col])
            if not gsis:
                continue
            record = PlayerRecord(
                gsis_id=gsis,
                pfr_id=clean_id(row[pfr_col]) if pfr_col else "",
                espn_id=clean_id(row[espn_col]) if espn_col else "",
                name=clean(row[name_col]) if name_col else "",
                position=position_for_grouping(row[position_col]) if position_col else "",
            )
            self.by_gsis[record.gsis_id] = record
            if record.pfr_id:
                self.by_pfr[record.pfr_id] = record
            if record.espn_id:
                self.by_espn[record.espn_id] = record
            name_key = normalize_name(record.name)
            if name_key:
                name_candidates.setdefault(name_key, []).append(record)

        self.by_unique_name = {
            key: rows[0]
            for key, rows in name_candidates.items()
            if len({row.gsis_id for row in rows}) == 1
        }

    def resolve(self, raw_id: object, name: object = "") -> PlayerRecord | None:
        rid = clean_id(raw_id)
        if rid in self.by_gsis:
            return self.by_gsis[rid]
        if rid in self.by_espn:
            return self.by_espn[rid]
        if rid in self.by_pfr:
            return self.by_pfr[rid]
        key = normalize_name(name)
        return self.by_unique_name.get(key) if key else None


@dataclass
class DepthSnapshot:
    rank_by_gsis: dict[str, int]
    rank_by_raw_id: dict[str, int]
    rank_by_name: dict[str, int]
    position_by_gsis: dict[str, str]
    position_by_raw_id: dict[str, str]
    position_by_name: dict[str, str]

    @classmethod
    def empty(cls) -> "DepthSnapshot":
        return cls({}, {}, {}, {}, {}, {})

    def rank_for(self, gsis_id: str, raw_id: str, name: str) -> int | None:
        if gsis_id and gsis_id in self.rank_by_gsis:
            return self.rank_by_gsis[gsis_id]
        if raw_id and raw_id in self.rank_by_raw_id:
            return self.rank_by_raw_id[raw_id]
        return self.rank_by_name.get(normalize_name(name))

    def position_for(self, gsis_id: str, raw_id: str, name: str) -> str:
        if gsis_id and gsis_id in self.position_by_gsis:
            return self.position_by_gsis[gsis_id]
        if raw_id and raw_id in self.position_by_raw_id:
            return self.position_by_raw_id[raw_id]
        return self.position_by_name.get(normalize_name(name), "")


def load_current_depth(
    root: Path,
    teams: TeamNormalizer,
    players: PlayerCrosswalk,
) -> dict[str, DepthSnapshot]:
    depth_root = root / "data/master/depth_charts"
    if not depth_root.exists():
        raise FileNotFoundError(f"Missing input directory: {depth_root}")

    output: dict[str, DepthSnapshot] = {}
    paths = sorted(depth_root.glob("*/*_depth.csv"))
    if not paths:
        raise RuntimeError(f"No depth-chart CSV files found under {depth_root}")

    for path in paths:
        df = read_csv(path)
        require_columns(
            df,
            ["player_id", "name", "team", "position_abb", "depth_chart_rank", "starter_flag"],
            str(path),
        )
        team_values = {teams.resolve(value) for value in df["team"] if clean(value)}
        if len(team_values) != 1:
            raise ValueError(f"{path}: expected exactly one team; found {sorted(team_values)}")
        team = next(iter(team_values))

        rank_by_gsis: dict[str, int] = {}
        rank_by_raw: dict[str, int] = {}
        rank_by_name: dict[str, int] = {}
        pos_by_gsis: dict[str, str] = {}
        pos_by_raw: dict[str, str] = {}
        pos_by_name: dict[str, str] = {}

        for _, row in df.iterrows():
            rank = parse_int(row["depth_chart_rank"])
            if rank is None and parse_int(row["starter_flag"]) == 1:
                rank = 1
            if rank is None:
                continue

            raw_id = clean_id(row["player_id"])
            name = clean(row["name"])
            resolved = players.resolve(raw_id, name)
            gsis = resolved.gsis_id if resolved else ""
            position = position_for_grouping(row["position_abb"])
            name_key = normalize_name(name)

            def store_rank(mapping: dict[str, int], key: str) -> None:
                if not key:
                    return
                previous = mapping.get(key)
                if previous is None or rank < previous:
                    mapping[key] = rank

            store_rank(rank_by_gsis, gsis)
            store_rank(rank_by_raw, raw_id)
            store_rank(rank_by_name, name_key)

            if position:
                if gsis and rank_by_gsis.get(gsis) == rank:
                    pos_by_gsis[gsis] = position
                if raw_id and rank_by_raw.get(raw_id) == rank:
                    pos_by_raw[raw_id] = position
                if name_key and rank_by_name.get(name_key) == rank:
                    pos_by_name[name_key] = position

        output[team] = DepthSnapshot(
            rank_by_gsis,
            rank_by_raw,
            rank_by_name,
            pos_by_gsis,
            pos_by_raw,
            pos_by_name,
        )

    return output


@dataclass(frozen=True)
class InjuryRecord:
    raw_id: str
    name: str
    position: str
    status: str


def normalize_injury_status(value: object) -> str:
    text = clean(value).casefold().replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    if (
        text in {"o", "out", "ir"}
        or text.startswith("out ")
        or "reserve" in text
        or "pup" in text
        or "injured reserve" in text
    ):
        return "out"
    if text in {"d", "doubtful"} or "doubt" in text:
        return "doubtful"
    if text in {"q", "questionable"} or "question" in text:
        return "questionable"
    return "other"


def load_current_injuries(
    root: Path,
    season: int,
    teams: TeamNormalizer,
) -> dict[str, list[InjuryRecord]]:
    path = root / f"00_intake/injuries/{season}_injuries.csv"
    df = read_csv(path, allow_empty_rows=True)
    require_columns(
        df,
        ["season", "team", "player_id", "player_name", "position", "game_status"],
        str(path),
    )

    latest: dict[tuple[str, str], tuple[tuple[int, int, int], InjuryRecord]] = {}
    for row_order, (_, row) in enumerate(df.iterrows()):
        row_season = parse_int(row["season"])
        if row_season is not None and row_season != season:
            continue

        team = teams.resolve(row["team"])
        raw_id = clean_id(row["player_id"])
        name = clean(row["player_name"])
        identity = raw_id or normalize_name(name)
        if not identity:
            continue

        status = normalize_injury_status(row["game_status"])
        record = InjuryRecord(
            raw_id=raw_id,
            name=name,
            position=position_for_grouping(row["position"]),
            status=status,
        )

        ts = parse_timestamp(row.get("report_date", ""))
        # A valid report_date outranks an undated row. Row order breaks ties.
        rank = (1, int(ts.value), row_order) if ts is not None else (0, 0, row_order)
        key = (team, identity)
        previous = latest.get(key)
        if previous is None or rank > previous[0]:
            latest[key] = (rank, record)

    output: dict[str, list[InjuryRecord]] = {}
    for (team, _), (_, record) in latest.items():
        output.setdefault(team, []).append(record)
    return output


class SnapProvider:
    def __init__(self, df: pd.DataFrame, teams: TeamNormalizer) -> None:
        team_col = choose_column(df, ["team"], required=True, label="snap counts")
        week_col = choose_column(df, ["week"], required=True, label="snap counts")
        pfr_col = choose_column(df, ["pfr_player_id", "pfr_id"])
        name_col = choose_column(df, ["player", "full_name", "player_name"])
        pos_col = choose_column(df, ["position"])
        off_col = choose_column(df, ["offense_pct"], required=True, label="snap counts")
        def_col = choose_column(df, ["defense_pct"], required=True, label="snap counts")
        if pfr_col is None and name_col is None:
            raise ValueError("snap counts: neither PFR ID nor player-name column is present")

        self.series: dict[tuple[str, str], list[tuple[int, float, float, str]]] = {}
        for _, row in df.iterrows():
            try:
                team = teams.resolve(row[team_col])
            except ValueError:
                continue
            week = parse_int(row[week_col])
            if not team or week is None:
                continue
            identities: list[str] = []
            pfr_id = clean_id(row[pfr_col]) if pfr_col else ""
            name = clean(row[name_col]) if name_col else ""
            if pfr_id:
                identities.append(f"pfr:{pfr_id}")
            name_key = normalize_name(name)
            if name_key:
                identities.append(f"name:{name_key}")
            item = (
                week,
                normalize_percentage(row[off_col]),
                normalize_percentage(row[def_col]),
                position_for_grouping(row[pos_col]) if pos_col else "",
            )
            for identity in identities:
                self.series.setdefault((team, identity), []).append(item)
        for key in self.series:
            self.series[key].sort(key=lambda item: item[0])

    def latest(self, team: str, pfr_id: str, name: str) -> tuple[float, float, str] | None:
        identities = []
        if pfr_id:
            identities.append(f"pfr:{pfr_id}")
        name_key = normalize_name(name)
        if name_key:
            identities.append(f"name:{name_key}")

        # Prefer usage for the player's current team when that exists.
        for identity in identities:
            values = self.series.get((team, identity), [])
            if values:
                _, off, deff, pos = values[-1]
                return off, deff, pos

        # A player may have changed teams since the prior season.
        for identity in identities:
            best: tuple[int, float, float, str] | None = None
            for (_source_team, source_identity), values in self.series.items():
                if source_identity != identity or not values:
                    continue
                candidate = values[-1]
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is not None:
                _, off, deff, pos = best
                return off, deff, pos
        return None


class ParticipationProvider:
    def __init__(self, df: pd.DataFrame, teams: TeamNormalizer) -> None:
        game_col = choose_column(
            df, ["nflverse_game_id", "game_id"], required=True, label="participation"
        )
        possession_col = choose_column(
            df, ["possession_team", "posteam"], required=True, label="participation"
        )
        off_players_col = choose_column(
            df, ["offense_players"], required=True, label="participation"
        )
        def_players_col = choose_column(
            df, ["defense_players"], required=True, label="participation"
        )

        offense_den: dict[tuple[str, int], int] = {}
        defense_den: dict[tuple[str, int], int] = {}
        offense_num: dict[tuple[str, int, str], int] = {}
        defense_num: dict[tuple[str, int, str], int] = {}

        for _, row in df.iterrows():
            match = GAME_ID_PATTERN.match(clean(row[game_col]))
            if not match:
                continue
            week = int(match.group(2))
            try:
                away = teams.resolve(match.group(3))
                home = teams.resolve(match.group(4))
                possession = teams.resolve(row[possession_col])
            except ValueError:
                continue
            if possession == away:
                defense = home
            elif possession == home:
                defense = away
            else:
                continue

            offense_players = set(GSIS_PATTERN.findall(clean(row[off_players_col])))
            defense_players = set(GSIS_PATTERN.findall(clean(row[def_players_col])))

            if offense_players:
                offense_den[(possession, week)] = offense_den.get((possession, week), 0) + 1
                for pid in offense_players:
                    key = (possession, week, pid)
                    offense_num[key] = offense_num.get(key, 0) + 1
            if defense_players:
                defense_den[(defense, week)] = defense_den.get((defense, week), 0) + 1
                for pid in defense_players:
                    key = (defense, week, pid)
                    defense_num[key] = defense_num.get(key, 0) + 1

        combined: dict[tuple[str, str, int], tuple[float, float]] = {}
        for (team, week, pid), count in offense_num.items():
            den = offense_den.get((team, week), 0)
            if den:
                current = combined.get((team, pid, week), (0.0, 0.0))
                combined[(team, pid, week)] = (count / den, current[1])
        for (team, week, pid), count in defense_num.items():
            den = defense_den.get((team, week), 0)
            if den:
                current = combined.get((team, pid, week), (0.0, 0.0))
                combined[(team, pid, week)] = (current[0], count / den)

        self.series: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
        for (team, pid, week), (off, deff) in combined.items():
            self.series.setdefault((team, pid), []).append((week, off, deff))
        for key in self.series:
            self.series[key].sort(key=lambda item: item[0])

    def latest(self, team: str, gsis_id: str) -> tuple[float, float] | None:
        if not gsis_id:
            return None
        values = self.series.get((team, gsis_id), [])
        if values:
            _, off, deff = values[-1]
            return off, deff

        best: tuple[int, float, float] | None = None
        for (_source_team, pid), rows in self.series.items():
            if pid != gsis_id or not rows:
                continue
            candidate = rows[-1]
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            return None
        _, off, deff = best
        return off, deff


def compute_injury_features(
    team: str,
    injuries: list[InjuryRecord],
    current_depth: DepthSnapshot,
    players: PlayerCrosswalk,
    prior_snaps: SnapProvider,
    prior_participation: ParticipationProvider,
) -> dict[str, float]:
    values = {feature: 0.0 for feature in INJURY_BASE_FEATURES}
    values["depth_starter_changes"] = 0.0

    for injury in injuries:
        if injury.status == "out":
            values["inj_out_count"] += 1.0
        elif injury.status == "doubtful":
            values["inj_doubtful_count"] += 1.0
        elif injury.status == "questionable":
            values["inj_questionable_count"] += 1.0

        if injury.status != "out":
            continue

        resolved = players.resolve(injury.raw_id, injury.name)
        gsis = resolved.gsis_id if resolved else ""
        pfr = resolved.pfr_id if resolved else ""
        name = injury.name or (resolved.name if resolved else "")
        rank = current_depth.rank_for(gsis, injury.raw_id, name)
        position = (
            injury.position
            or current_depth.position_for(gsis, injury.raw_id, name)
            or (resolved.position if resolved else "")
        )
        position = position_for_grouping(position)

        is_starter = rank == 1
        is_top2 = rank is not None and rank <= 2

        if is_starter:
            values["inj_starter_out_count"] += 1.0
            if position == "QB":
                values["inj_qb1_out"] = 1.0
            if position in OFFENSIVE_LINE_POSITIONS:
                values["inj_ol_starter_out_count"] += 1.0
            if position in SKILL_POSITIONS:
                values["inj_skill_starter_out_count"] += 1.0
            if position in FRONT7_POSITIONS:
                values["inj_front7_starter_out_count"] += 1.0
            if position in SECONDARY_POSITIONS:
                values["inj_secondary_starter_out_count"] += 1.0

        if is_top2:
            values["inj_top2_depth_out_count"] += 1.0

        snap_value = prior_snaps.latest(team, pfr, name)
        if snap_value is not None:
            off_share, def_share, _ = snap_value
        else:
            part_value = prior_participation.latest(team, gsis)
            if part_value is None:
                off_share, def_share = 0.0, 0.0
            else:
                off_share, def_share = part_value

        values["inj_offense_unavailable_snap_share"] += off_share
        values["inj_defense_unavailable_snap_share"] += def_share

    values["inj_offense_unavailable_snap_share"] = min(
        1.0, values["inj_offense_unavailable_snap_share"]
    )
    values["inj_defense_unavailable_snap_share"] = min(
        1.0, values["inj_defense_unavailable_snap_share"]
    )
    return values


# ---------------------------------------------------------------------------
# Feature source loading
# ---------------------------------------------------------------------------

def load_schema(root: Path) -> dict:
    path = root / "models/step11_feature_schema.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    required_keys = ["feature_order", "numeric_features", "categorical_features"]
    missing = [key for key in required_keys if key not in schema]
    if missing:
        raise ValueError(f"{path}: missing schema keys: {missing}")

    feature_order = list(schema["feature_order"])
    numeric = set(schema["numeric_features"])
    categorical = set(schema["categorical_features"])

    if len(feature_order) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"{path}: expected {EXPECTED_FEATURE_COUNT} features; found {len(feature_order)}"
        )
    if len(feature_order) != len(set(feature_order)):
        raise ValueError(f"{path}: duplicate names in feature_order")
    if numeric & categorical:
        raise ValueError(f"{path}: features classified as both numeric and categorical")
    if numeric | categorical != set(feature_order):
        raise ValueError(
            f"{path}: numeric_features and categorical_features do not exactly cover feature_order"
        )
    return schema


def validate_week1_base(base: pd.DataFrame, season: int, label: str) -> None:
    require_columns(
        base,
        ["season", "season_type", "week", "game_id", "away_team", "home_team"],
        label,
    )
    require_unique_game_id(base, label)
    season_values = {parse_int(value) for value in base["season"]}
    week_values = {parse_int(value) for value in base["week"]}
    if season_values != {season}:
        raise ValueError(f"{label}: expected only season={season}; found {season_values}")
    if week_values != {WEEK}:
        raise ValueError(f"{label}: expected only week={WEEK}; found {week_values}")
    if base.empty:
        raise ValueError(f"{label}: no Week 1 games")


def add_market_features(
    work: pd.DataFrame,
    root: Path,
    teams: TeamNormalizer,
) -> pd.DataFrame:
    path = root / "00_intake/schedule/weekly/week_1_NFL_weekly_schedule.csv"
    market = read_csv(path)
    require_columns(
        market,
        [
            "game_id", "commence_time", "home_moneyline_american",
            "away_moneyline_american", "home_spread", "away_spread",
            "home_spread_american", "away_spread_american", "total",
            "over_american", "under_american",
        ],
        str(path),
    )
    validate_team_alignment(work, market, teams, str(path))

    source = market[
        [
            "game_id", "home_moneyline_american", "away_moneyline_american",
            "home_spread", "away_spread", "home_spread_american",
            "away_spread_american", "total", "over_american", "under_american",
        ]
    ].copy()

    source["home_moneyline"] = source["home_moneyline_american"]
    source["away_moneyline"] = source["away_moneyline_american"]

    # CRITICAL: sportsbook convention is home favorite = negative;
    # Step 11 training convention is home favorite = positive.
    source["spread_line"] = -pd.to_numeric(source["home_spread"], errors="coerce")

    source["home_spread_odds"] = source["home_spread_american"]
    source["away_spread_odds"] = source["away_spread_american"]
    source["total_line"] = source["total"]
    source["over_odds"] = source["over_american"]
    source["under_odds"] = source["under_american"]

    # Historical raw market features retain the sportsbook source sign.
    source["hist_home_spread"] = source["home_spread"]
    source["hist_away_spread"] = source["away_spread"]
    source["hist_odds_total"] = source["total"]

    columns = [
        "home_moneyline", "away_moneyline", "spread_line",
        "home_spread_odds", "away_spread_odds", "total_line",
        "over_odds", "under_odds", "hist_home_spread",
        "hist_away_spread", "hist_odds_total",
    ]
    return merge_game_source(work, source, columns, str(path))


def add_drat_features(
    work: pd.DataFrame,
    root: Path,
    season: int,
    teams: TeamNormalizer,
) -> pd.DataFrame:
    path = root / f"00_intake/predictions/drat/clean/{season}_week_1_drat.csv"
    drat = read_csv(path)
    require_columns(
        drat,
        [
            "game_id", "away_prob", "home_prob", "moneyline_away",
            "moneyline_home", "spread_away", "spread_home",
        ],
        str(path),
    )
    validate_team_alignment(work, drat, teams, str(path))
    source = drat.rename(
        columns={
            "away_prob": "drat_away_prob",
            "home_prob": "drat_home_prob",
            "moneyline_away": "drat_away_moneyline",
            "moneyline_home": "drat_home_moneyline",
            "spread_away": "drat_away_spread",
            "spread_home": "drat_home_spread",
        }
    )
    columns = [
        "drat_away_prob", "drat_home_prob", "drat_away_moneyline",
        "drat_home_moneyline", "drat_away_spread", "drat_home_spread",
    ]
    return merge_game_source(work, source, columns, str(path))


def add_epred_features(
    work: pd.DataFrame,
    root: Path,
    season: int,
    teams: TeamNormalizer,
) -> pd.DataFrame:
    season_types = [clean(v) for v in work["season_type"].dropna().unique() if clean(v)]
    if len(set(season_types)) != 1:
        raise ValueError(f"Week 1 input must contain exactly one season_type; found {season_types}")
    season_type = season_types[0].casefold()
    path = root / f"00_intake/predictions/final/{season}_{season_type}_1_clean_predictions.csv"
    epred = read_csv(path)
    require_columns(
        epred,
        [
            "game_id", "matchupQuality", "home_prob", "away_prob", "tie_prob",
            "away_projected_pts", "home_projected_pts", "total_projected_pts",
            "home_PtDiff", "away_PtDiff", "home_rating", "away_rating",
        ],
        str(path),
    )
    validate_team_alignment(work, epred, teams, str(path))
    source = epred.rename(
        columns={
            "matchupQuality": "epred_matchupQuality",
            "home_prob": "epred_home_prob",
            "away_prob": "epred_away_prob",
            "tie_prob": "epred_tie_prob",
            "away_projected_pts": "epred_away_projected_pts",
            "home_projected_pts": "epred_home_projected_pts",
            "total_projected_pts": "epred_total_projected_pts",
            "home_PtDiff": "epred_home_PtDiff",
            "away_PtDiff": "epred_away_PtDiff",
            "home_rating": "epred_home_rating",
            "away_rating": "epred_away_rating",
        }
    )
    columns = [
        "epred_matchupQuality", "epred_home_prob", "epred_away_prob",
        "epred_tie_prob", "epred_away_projected_pts",
        "epred_home_projected_pts", "epred_total_projected_pts",
        "epred_home_PtDiff", "epred_away_PtDiff",
        "epred_home_rating", "epred_away_rating",
    ]
    return merge_game_source(work, source, columns, str(path))


def load_schedule_week1(
    work: pd.DataFrame,
    root: Path,
    season: int,
    teams: TeamNormalizer,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / f"00_intake/schedule/{season}_schedule.csv"
    schedule = read_csv(path)
    require_columns(
        schedule,
        [
            "season", "season_type", "week", "game_id", "game_date",
            "game_time", "away_team", "home_team", "neutral_site",
            "stadium", "roof", "surface",
        ],
        str(path),
    )
    season_num = pd.to_numeric(schedule["season"], errors="coerce")
    week_num = pd.to_numeric(schedule["week"], errors="coerce")
    week1 = schedule[(season_num == season) & (week_num == WEEK)].copy()
    require_unique_game_id(week1, f"{path} season={season} week={WEEK}")
    validate_team_alignment(work, week1, teams, str(path))

    source = week1[
        [
            "game_id", "season_type", "game_date", "game_time", "away_team",
            "home_team", "neutral_site", "stadium", "roof", "surface",
        ]
    ].rename(
        columns={
            "season_type": "sched_season_type",
            "game_date": "sched_game_date",
            "game_time": "sched_game_time",
            "away_team": "sched_away_team",
            "home_team": "sched_home_team",
            "neutral_site": "sched_neutral_site",
            "stadium": "sched_stadium",
            "roof": "sched_roof",
            "surface": "sched_surface",
        }
    )
    columns = [
        "sched_season_type", "sched_game_date", "sched_game_time",
        "sched_away_team", "sched_home_team", "sched_neutral_site",
        "sched_stadium", "sched_roof", "sched_surface",
    ]
    return merge_game_source(work, source, columns, str(path)), schedule


def build_division_lookup(
    root: Path,
    season: int,
    teams: TeamNormalizer,
) -> dict[str, str]:
    path = root / "data/master/league_master.csv"
    df = read_csv(path)
    require_columns(df, ["team_abbr", "division", "season"], str(path))
    rows = df[pd.to_numeric(df["season"], errors="coerce") == season]
    lookup: dict[str, str] = {}
    for _, row in rows.iterrows():
        team = teams.resolve(row["team_abbr"])
        division = clean(row["division"])
        if not division:
            continue
        previous = lookup.get(team)
        if previous is not None and previous != division:
            raise ValueError(f"{path}: conflicting divisions for {team}: {previous!r} vs {division!r}")
        lookup[team] = division
    return lookup


def build_stadium_lookup(
    root: Path,
    teams: TeamNormalizer,
) -> tuple[dict[str, str], dict[str, str]]:
    path = root / "config/mapping/stadium_map_nfl.csv"
    df = read_csv(path)
    require_columns(df, ["team", "stadium", "venue_id"], str(path))
    by_stadium: dict[str, str] = {}
    by_team: dict[str, str] = {}
    for _, row in df.iterrows():
        venue_id = clean_id(row["venue_id"])
        if not venue_id:
            continue
        stadium_key = normalize_name(row["stadium"])
        if stadium_key:
            previous = by_stadium.get(stadium_key)
            if previous is not None and previous != venue_id:
                raise ValueError(f"{path}: conflicting venue_id values for stadium {row['stadium']!r}")
            by_stadium[stadium_key] = venue_id
        if clean(row["team"]):
            team = teams.resolve(row["team"])
            previous = by_team.get(team)
            if previous is not None and previous != venue_id:
                raise ValueError(f"{path}: conflicting venue_id values for team {team}")
            by_team[team] = venue_id
    return by_stadium, by_team


def roof_for_model(value: object) -> str:
    text = clean(value).casefold()
    mapping = {
        "open_air": "outdoors",
        "open air": "outdoors",
        "outdoor": "outdoors",
        "outdoors": "outdoors",
        "fixed_roof": "dome",
        "fixed roof": "dome",
        "dome": "dome",
        "retractable": "closed",
        "retractable_roof": "closed",
        "retractable roof": "closed",
        "closed": "closed",
    }
    return mapping.get(text, text)


def surface_for_model(value: object) -> str:
    return clean(value).casefold()


def add_schedule_context_features(
    work: pd.DataFrame,
    divisions: dict[str, str],
    stadium_by_name: dict[str, str],
    stadium_by_team: dict[str, str],
    teams: TeamNormalizer,
) -> pd.DataFrame:
    for idx, row in work.iterrows():
        home = teams.resolve(row["sched_home_team"])
        away = teams.resolve(row["sched_away_team"])
        season_type = clean(row["sched_season_type"]).casefold()

        work.at[idx, "game_type"] = {
            "reg": "REG", "pre": "PRE", "post": "POST"
        }.get(season_type, season_type.upper())
        work.at[idx, "week"] = str(WEEK)

        game_date = pd.to_datetime(clean(row["sched_game_date"]), errors="coerce")
        if pd.isna(game_date):
            raise ValueError(f"Invalid game_date for game_id={row['game_id']}: {row['sched_game_date']!r}")
        work.at[idx, "weekday"] = game_date.day_name()
        work.at[idx, "gametime"] = clean(row["sched_game_time"])
        work.at[idx, "away_team"] = away
        work.at[idx, "home_team"] = home

        neutral = parse_int(row["sched_neutral_site"]) == 1
        work.at[idx, "location"] = "Neutral" if neutral else "Home"
        work.at[idx, "away_rest"] = 7
        work.at[idx, "home_rest"] = 7
        work.at[idx, "rest_diff"] = 0

        home_div = divisions.get(home, "")
        away_div = divisions.get(away, "")
        if not home_div or not away_div:
            raise ValueError(
                f"Missing {SEASON} division mapping for game_id={row['game_id']} home={home} away={away}"
            )
        work.at[idx, "div_game"] = 1 if home_div == away_div else 0

        work.at[idx, "roof"] = roof_for_model(row["sched_roof"])
        surface = surface_for_model(row["sched_surface"])
        work.at[idx, "surface"] = surface
        work.at[idx, "hist_surface"] = surface

        stadium = clean(row["sched_stadium"])
        work.at[idx, "stadium"] = stadium
        stadium_id = stadium_by_name.get(normalize_name(stadium), stadium_by_team.get(home, ""))
        if not stadium_id:
            raise ValueError(
                f"Could not map stadium_id for game_id={row['game_id']} stadium={stadium!r} home={home}"
            )
        work.at[idx, "stadium_id"] = stadium_id

    return work


def add_weather_features(work: pd.DataFrame, root: Path) -> pd.DataFrame:
    path = root / "data/weather/week_1_NFL_weekly_weather.csv"
    weather = read_csv(path)
    require_columns(
        weather,
        [
            "game_id", "temperature", "wind_speed", "precip_probability",
            "rain_flag", "snow_flag",
        ],
        str(path),
    )
    source = weather[
        [
            "game_id", "temperature", "wind_speed", "precip_probability",
            "rain_flag", "snow_flag",
        ]
    ].rename(
        columns={
            "temperature": "wx_temperature",
            "wind_speed": "wx_wind_speed",
            "precip_probability": "wx_precip_probability",
            "rain_flag": "wx_rain_flag",
            "snow_flag": "wx_snow_flag",
        }
    )
    work = merge_game_source(
        work,
        source,
        [
            "wx_temperature", "wx_wind_speed", "wx_precip_probability",
            "wx_rain_flag", "wx_snow_flag",
        ],
        str(path),
    )

    work["temp"] = work["wx_temperature"]
    work["wind"] = work["wx_wind_speed"]
    work["hist_temperature"] = work["wx_temperature"]
    work["hist_wind_speed"] = work["wx_wind_speed"]
    work["hist_precip_probability"] = work["wx_precip_probability"]
    work["hist_wind_bearing"] = np.nan

    icons: list[str] = []
    precip_types: list[str] = []
    for _, row in work.iterrows():
        if parse_int(row["wx_snow_flag"]) == 1:
            icons.append("snow")
            precip_types.append("snow")
        elif parse_int(row["wx_rain_flag"]) == 1:
            icons.append("rain")
            precip_types.append("rain")
        else:
            icons.append("")
            precip_types.append("")
    work["hist_weather_icon"] = icons
    work["hist_precip_type"] = precip_types
    return work


def add_travel_features(
    work: pd.DataFrame,
    root: Path,
    season: int,
    teams: TeamNormalizer,
) -> pd.DataFrame:
    path = root / f"data/travel/{season}_week_1_travel.csv"
    travel = read_csv(path)
    columns = [
        "miles_traveled", "time_zones_crossed", "east_to_west",
        "west_to_east", "international_flag", "neutral_site_flag",
    ]
    require_columns(travel, ["game_id", *columns], str(path))
    validate_team_alignment(work, travel, teams, str(path))
    return merge_game_source(work, travel, columns, str(path))


def load_prior_team_stats(
    root: Path,
    season: int,
    teams: TeamNormalizer,
) -> dict[str, dict[str, float | None]]:
    prior_season = season - 1
    path = root / f"00_intake/team_stats/{prior_season}_team_stats.csv"
    df = read_csv(path)
    require_columns(df, ["season", "week", "team", *TEAM_METRICS], str(path))
    df["_season"] = pd.to_numeric(df["season"], errors="coerce")
    df["_week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df[df["_season"] == prior_season].copy()

    normalized = df["team"].map(teams.resolve)
    result: dict[str, dict[str, float | None]] = {}
    for team, group in df.groupby(normalized, sort=False):
        valid = group[group["_week"].notna()].copy()
        if valid.empty:
            continue
        latest_week = valid["_week"].max()
        latest = valid[valid["_week"] == latest_week]
        if len(latest) != 1:
            raise ValueError(
                f"{path}: duplicate final-week rows for team={team}, week={int(latest_week)}"
            )
        chosen = latest.iloc[0]
        result[team] = {metric: parse_float(chosen[metric]) for metric in TEAM_METRICS}
    return result


def load_current_qb1(
    root: Path,
    teams: TeamNormalizer,
    players: PlayerCrosswalk,
) -> dict[str, str]:
    path = root / "config/mapping/qb_map_nfl.csv"
    df = read_csv(path)
    require_columns(df, ["player_id", "team_abbr", "starter_flag", "position_abb"], str(path))

    starters: dict[str, str] = {}
    for _, row in df.iterrows():
        if parse_int(row["starter_flag"]) != 1:
            continue
        if normalize_position(row["position_abb"]) != "QB":
            continue
        team = teams.resolve(row["team_abbr"])
        raw_id = clean_id(row["player_id"])
        name = clean(row.get("qb_name", ""))
        resolved = players.resolve(raw_id, name)
        if resolved is None or not resolved.gsis_id:
            raise ValueError(
                f"{path}: could not map current QB1 id={raw_id!r} name={name!r} for {team} to GSIS id"
            )
        if team in starters and starters[team] != resolved.gsis_id:
            raise ValueError(f"{path}: multiple QB1 rows for team={team}")
        starters[team] = resolved.gsis_id

    if not starters:
        raise RuntimeError(f"{path}: no current QB1 rows found")
    return starters


def load_prior_qb_stats(
    root: Path,
    season: int,
    qb1_by_team: dict[str, str],
) -> dict[str, dict[str, float | None]]:
    prior_season = season - 1
    path = root / f"00_intake/qb/{prior_season}_qb_stats.csv"
    df = read_csv(path)
    require_columns(df, ["season", "week", "player_id", "dropbacks", *QB_METRICS], str(path))
    df["_season"] = pd.to_numeric(df["season"], errors="coerce")
    df["_week"] = pd.to_numeric(df["week"], errors="coerce")
    df["_dropbacks"] = pd.to_numeric(df["dropbacks"], errors="coerce").fillna(-1.0)
    df["_player"] = df["player_id"].map(clean_id)
    df = df[df["_season"] == prior_season].copy()

    result: dict[str, dict[str, float | None]] = {}
    for team, gsis_id in qb1_by_team.items():
        rows = df[(df["_player"] == gsis_id) & df["_week"].notna()].copy()
        if rows.empty:
            result[team] = {metric: None for metric in QB_METRICS}
            continue
        latest_week = rows["_week"].max()
        rows = rows[rows["_week"] == latest_week].sort_values("_dropbacks", kind="stable")
        chosen = rows.iloc[-1]
        result[team] = {metric: parse_float(chosen[metric]) for metric in QB_METRICS}
    return result


def add_team_and_qb_features(
    work: pd.DataFrame,
    team_stats: dict[str, dict[str, float | None]],
    qb_stats: dict[str, dict[str, float | None]],
    qb1_by_team: dict[str, str],
) -> pd.DataFrame:
    for idx, row in work.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        if home not in team_stats or away not in team_stats:
            raise ValueError(
                f"Prior-season team stats missing for game_id={row['game_id']} home={home} away={away}"
            )
        if home not in qb1_by_team or away not in qb1_by_team:
            raise ValueError(
                f"Current Week 1 QB1 mapping missing for game_id={row['game_id']} home={home} away={away}"
            )

        for metric in TEAM_METRICS:
            home_value = team_stats[home][metric]
            away_value = team_stats[away][metric]
            work.at[idx, f"home_{metric}"] = home_value
            work.at[idx, f"away_{metric}"] = away_value
            work.at[idx, f"{metric}_diff"] = (
                np.nan if home_value is None or away_value is None else home_value - away_value
            )

        for metric in QB_METRICS:
            home_value = qb_stats.get(home, {}).get(metric)
            away_value = qb_stats.get(away, {}).get(metric)
            work.at[idx, f"home_qb_{metric}"] = home_value
            work.at[idx, f"away_qb_{metric}"] = away_value
            work.at[idx, f"qb_{metric}_diff"] = (
                np.nan if home_value is None or away_value is None else home_value - away_value
            )
    return work


def add_injury_features(
    work: pd.DataFrame,
    root: Path,
    season: int,
    teams: TeamNormalizer,
    players: PlayerCrosswalk,
) -> pd.DataFrame:
    current_depth = load_current_depth(root, teams, players)
    current_injuries = load_current_injuries(root, season, teams)

    prior_season = season - 1
    prior_snaps = SnapProvider(
        read_parquet(root / f"data/historic_data/snap_counts/snap_counts_{prior_season}.parquet"),
        teams,
    )
    prior_participation = ParticipationProvider(
        read_parquet(
            root / f"data/historic_data/participation/pbp_participation_{prior_season}.parquet"
        ),
        teams,
    )

    for idx, row in work.iterrows():
        side_values: dict[str, dict[str, float]] = {}
        for side, team in [("home", row["home_team"]), ("away", row["away_team"])]:
            if team not in current_depth:
                raise ValueError(f"No current depth chart loaded for Week 1 team={team}")
            side_values[side] = compute_injury_features(
                team=team,
                injuries=current_injuries.get(team, []),
                current_depth=current_depth[team],
                players=players,
                prior_snaps=prior_snaps,
                prior_participation=prior_participation,
            )

        for feature in INJURY_BASE_FEATURES:
            home_value = side_values["home"][feature]
            away_value = side_values["away"][feature]
            work.at[idx, f"home_{feature}"] = home_value
            work.at[idx, f"away_{feature}"] = away_value
            work.at[idx, f"{feature}_diff"] = home_value - away_value
    return work


# ---------------------------------------------------------------------------
# Assemble the exact model matrix
# ---------------------------------------------------------------------------

def prepare_model_features(
    root: Path,
    original: pd.DataFrame,
    schema: dict,
) -> pd.DataFrame:
    teams = TeamNormalizer(root / "config/mapping/team_map.csv")

    # The intake CSV is read with dtype=str, and newer pandas versions may
    # back those columns with Arrow string arrays.  This working frame is
    # intentionally dtype-flexible because model feature construction writes
    # numeric values (for example week/rest/div_game) into columns that may
    # already exist as strings in the intake.  The untouched `original` frame
    # is kept separately for final output preservation.
    work = original.astype(object).copy()
    work["_original_home_team"] = work["home_team"]
    work["_original_away_team"] = work["away_team"]

    # Normalize the model/team-join copy only. The final output uses original.
    work["home_team"] = work["home_team"].map(teams.resolve)
    work["away_team"] = work["away_team"].map(teams.resolve)

    work = add_market_features(work, root, teams)
    work = add_drat_features(work, root, SEASON, teams)
    work = add_epred_features(work, root, SEASON, teams)
    work, _full_schedule = load_schedule_week1(work, root, SEASON, teams)

    divisions = build_division_lookup(root, SEASON, teams)
    stadium_by_name, stadium_by_team = build_stadium_lookup(root, teams)
    work = add_schedule_context_features(
        work, divisions, stadium_by_name, stadium_by_team, teams
    )

    work = add_weather_features(work, root)
    work = add_travel_features(work, root, SEASON, teams)

    players = PlayerCrosswalk(
        read_parquet(root / "data/historic_data/players/players.parquet")
    )
    qb1_by_team = load_current_qb1(root, teams, players)
    team_stats = load_prior_team_stats(root, SEASON, teams)
    qb_stats = load_prior_qb_stats(root, SEASON, qb1_by_team)
    work = add_team_and_qb_features(work, team_stats, qb_stats, qb1_by_team)
    work = add_injury_features(work, root, SEASON, teams, players)

    feature_order = list(schema["feature_order"])
    numeric = set(schema["numeric_features"])
    categorical = set(schema["categorical_features"])

    missing_features = [feature for feature in feature_order if feature not in work.columns]
    if missing_features:
        raise RuntimeError(
            "Could not construct every Step 11 model feature; missing: "
            + ", ".join(missing_features)
        )

    features = work[feature_order].copy()
    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Prepared model matrix has {features.shape[1]} features; expected {EXPECTED_FEATURE_COUNT}"
        )
    if list(features.columns) != feature_order:
        raise RuntimeError("Prepared feature names/order differ from step11_feature_schema.json")

    for feature in feature_order:
        if feature in numeric:
            features[feature] = pd.to_numeric(features[feature], errors="coerce")
        elif feature in categorical:
            features[feature] = (
                features[feature]
                .map(clean)
                .replace("", MISSING_CAT)
                .astype(str)
            )
        else:
            raise RuntimeError(f"Schema feature is not classified: {feature}")

    spread_line = pd.to_numeric(features["spread_line"], errors="coerce")
    total_line = pd.to_numeric(features["total_line"], errors="coerce")
    if not np.isfinite(spread_line.to_numpy(dtype=float)).all():
        bad = original.loc[~np.isfinite(spread_line.to_numpy(dtype=float)), "game_id"].tolist()
        raise ValueError(f"spread_line is missing/non-numeric for game_id values: {bad[:10]}")
    if not np.isfinite(total_line.to_numpy(dtype=float)).all():
        bad = original.loc[~np.isfinite(total_line.to_numpy(dtype=float)), "game_id"].tolist()
        raise ValueError(f"total_line is missing/non-numeric for game_id values: {bad[:10]}")

    return features


# ---------------------------------------------------------------------------
# Models, calibration, output validation
# ---------------------------------------------------------------------------

def sigmoid(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    array = np.clip(array, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-array))


def load_calibrations(root: Path) -> dict:
    path = root / "models/step14_probability_calibration.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if raw.get("method") != "one_variable_logistic_platt":
        raise ValueError(
            f"{path}: expected method='one_variable_logistic_platt'; found {raw.get('method')!r}"
        )
    calibrations = raw.get("calibrations")
    if not isinstance(calibrations, dict):
        raise ValueError(f"{path}: missing calibrations object")

    expected_x = {
        "moneyline": "predicted_margin",
        "spread": "predicted_margin - spread_line",
        "total": "predicted_total - total_line",
    }
    for key, x_definition in expected_x.items():
        if key not in calibrations:
            raise ValueError(f"{path}: missing calibration section {key!r}")
        section = calibrations[key]
        if "intercept" not in section or "slope" not in section:
            raise ValueError(f"{path}: calibration {key!r} lacks intercept/slope")
        if clean(section.get("x_definition")) != x_definition:
            raise ValueError(
                f"{path}: unexpected {key} x_definition: {section.get('x_definition')!r}"
            )
    return calibrations


def validate_probability_pair(a: np.ndarray, b: np.ndarray, label: str) -> None:
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError(f"{label}: non-finite probability values")
    if ((a < 0.0) | (a > 1.0) | (b < 0.0) | (b > 1.0)).any():
        raise RuntimeError(f"{label}: probabilities outside [0, 1]")
    if not np.allclose(a + b, 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{label}: complementary probabilities do not sum to 1")


def apply_models(
    root: Path,
    original: pd.DataFrame,
    features: pd.DataFrame,
    schema: dict,
) -> pd.DataFrame:
    margin_path = root / "models/step11_margin_model.cbm"
    total_path = root / "models/step11_total_points_model.cbm"
    for path in [margin_path, total_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    margin_model = CatBoostRegressor()
    total_model = CatBoostRegressor()
    margin_model.load_model(str(margin_path))
    total_model.load_model(str(total_path))

    expected_names = list(schema["feature_order"])
    if list(features.columns) != expected_names:
        raise RuntimeError("Prediction feature names/order differ from Step 11 schema")
    if list(margin_model.feature_names_) != expected_names:
        raise RuntimeError("Margin model feature names/order differ from Step 11 schema")
    if list(total_model.feature_names_) != expected_names:
        raise RuntimeError("Total model feature names/order differ from Step 11 schema")

    predicted_margin = np.asarray(margin_model.predict(features), dtype=float)
    predicted_total = np.asarray(total_model.predict(features), dtype=float)
    if len(predicted_margin) != len(original) or len(predicted_total) != len(original):
        raise RuntimeError("Prediction row count does not match Week 1 input")
    if not np.isfinite(predicted_margin).all():
        raise RuntimeError("Margin model produced non-finite predictions")
    if not np.isfinite(predicted_total).all():
        raise RuntimeError("Total model produced non-finite predictions")

    predicted_home_score = (predicted_total + predicted_margin) / 2.0
    predicted_away_score = (predicted_total - predicted_margin) / 2.0

    spread_line = pd.to_numeric(features["spread_line"], errors="coerce").to_numpy(dtype=float)
    total_line = pd.to_numeric(features["total_line"], errors="coerce").to_numpy(dtype=float)

    calibration = load_calibrations(root)
    ml = calibration["moneyline"]
    spread = calibration["spread"]
    total = calibration["total"]

    home_win = sigmoid(float(ml["intercept"]) + float(ml["slope"]) * predicted_margin)
    away_win = 1.0 - home_win
    home_cover = sigmoid(
        float(spread["intercept"])
        + float(spread["slope"]) * (predicted_margin - spread_line)
    )
    away_cover = 1.0 - home_cover
    over = sigmoid(
        float(total["intercept"])
        + float(total["slope"]) * (predicted_total - total_line)
    )
    under = 1.0 - over

    validate_probability_pair(home_win, away_win, "moneyline")
    validate_probability_pair(home_cover, away_cover, "spread")
    validate_probability_pair(over, under, "total")

    if not np.allclose(
        (predicted_total + predicted_margin) / 2.0,
        predicted_home_score,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("predicted_home_score does not reconcile")
    if not np.allclose(
        (predicted_total - predicted_margin) / 2.0,
        predicted_away_score,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("predicted_away_score does not reconcile")

    output = original.copy()
    output["predicted_margin"] = predicted_margin
    output["predicted_total"] = predicted_total
    output["predicted_home_score"] = predicted_home_score
    output["predicted_away_score"] = predicted_away_score
    output["home_win_probability"] = home_win
    output["away_win_probability"] = away_win
    output["home_cover_probability"] = home_cover
    output["away_cover_probability"] = away_cover
    output["over_probability"] = over
    output["under_probability"] = under

    expected_columns = [*original.columns.tolist(), *OUTPUT_COLUMNS]
    if output.columns.tolist() != expected_columns:
        raise RuntimeError("Final output columns are not original columns plus the exact 10 prediction columns")
    if not output["home_team"].equals(original["home_team"]):
        raise RuntimeError("Final output home_team values changed from the original input")
    if not output["away_team"].equals(original["away_team"]):
        raise RuntimeError("Final output away_team values changed from the original input")
    if output["game_id"].tolist() != original["game_id"].tolist():
        raise RuntimeError("Final output game_id row order changed")
    require_unique_game_id(output, "final Week 1 output")
    return output


def main() -> None:
    print(f"projection_week1.py version={SCRIPT_VERSION}")
    root = nfl_root()
    combined_path = root / "00_intake/predictions/enriched/combined/week_1_NFL_enriched.csv"
    original = read_csv(combined_path)
    validate_week1_base(original, SEASON, str(combined_path))

    collisions = [column for column in OUTPUT_COLUMNS if column in original.columns]
    if collisions:
        raise ValueError(
            f"{combined_path}: prediction columns already exist and would be overwritten: {collisions}"
        )

    schema = load_schema(root)
    features = prepare_model_features(root, original.copy(), schema)
    projected = apply_models(root, original, features, schema)

    output_dir = root / "01_merge"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "week_1_NFL_enriched.csv"
    projected.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(
        f"WROTE {output_path} | games={len(projected)} | "
        f"features={features.shape[1]} | columns={len(projected.columns)}"
    )


if __name__ == "__main__":
    main()

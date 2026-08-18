#!/usr/bin/env python3
"""
Live NFL model projection for Week 2 and later.

READS ONLY from docs/win/football/nfl/.
WRITES ONLY fresh weekly projection CSV files to:
  docs/win/football/nfl/01_merge/

The model feature order and numeric/categorical types are read directly from
models/step11_feature_schema.json. Team performance uses Week N-1. QB
performance uses the latest same-season QB row with source week < N.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
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

SCRIPT_VERSION = "2026-08-14-inseason-fix1"
EXPECTED_FEATURE_COUNT = 260

NFL_REL = Path("docs/win/football/nfl")
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
OFFENSIVE_LINE_POSITIONS = {"C", "G", "OG", "LG", "RG", "T", "OT", "LT", "RT", "OL"}
SKILL_POSITIONS = {"RB", "HB", "FB", "WR", "TE"}
FRONT7_POSITIONS = {"DL", "DE", "DT", "NT", "EDGE", "LB", "ILB", "OLB", "MLB"}
SECONDARY_POSITIONS = {"DB", "CB", "S", "FS", "SS", "NB"}
GSIS_PATTERN = re.compile(r"00-\d{7}")
GAME_ID_PATTERN = re.compile(r"^(\d{4})_(\d{1,2})_([A-Za-z0-9]+)_([A-Za-z0-9]+)$")
WEEK_FILE_PATTERN = re.compile(r"^week_(\d+)_NFL_enriched\.csv$")
TEAM_ALIASES = {"WAS": "WSH", "LA": "LAR", "JAC": "JAX"}
MISSING_CAT = "__MISSING__"


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / ".git").exists():
            return parent
    # Expected repository layout: docs/win/football/nfl/scripts/01_merge/file.py
    return here.parents[6]


def nfl_root() -> Path:
    return repo_root() / NFL_REL


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
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


def normalize_team(value: object) -> str:
    key = clean(value).upper()
    return TEAM_ALIASES.get(key, key)


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
        "SLB": "LB", "WLB": "LB",
    }
    return aliases.get(pos, pos)


def normalize_game_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


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
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def choose_column(df: pd.DataFrame, candidates: list[str], *, required: bool = False, label: str = "source") -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        raise ValueError(f"{label}: none of these columns are present: {candidates}")
    return None


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


def require_unique_game_id(df: pd.DataFrame, label: str) -> None:
    require_columns(df, ["game_id"], label)
    duplicated = df.loc[df["game_id"].duplicated(keep=False), "game_id"].dropna().unique()
    if len(duplicated):
        raise ValueError(f"{label}: duplicate game_id values: {list(duplicated[:10])}")


def merge_game_source(base: pd.DataFrame, source: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    require_unique_game_id(base, "combined input")
    require_unique_game_id(source, label)
    require_columns(source, ["game_id", *columns], label)
    wanted = source[["game_id", *columns]].copy()
    merged = base.merge(wanted, on="game_id", how="left", validate="one_to_one", sort=False)
    source_ids = set(source["game_id"].dropna())
    missing_ids = [gid for gid in base["game_id"] if gid not in source_ids]
    if missing_ids:
        raise ValueError(f"{label}: missing game_id rows required by combined input: {missing_ids[:10]}")
    return merged


def load_team_name_lookup(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    path = root / "config/mapping/team_map.csv"
    df = read_csv(path)
    require_columns(df, ["team_abbr", "canonical_team"], str(path))
    to_abbr: dict[str, str] = {}
    abbr_to_name: dict[str, str] = {}
    # Do not use location alone: "Los Angeles" is ambiguous between LAR/LAC.
    candidates = ["canonical_team", "team_abbr", "alias", "team_name", "nickname", "shortDisplayName"]
    for _, row in df.iterrows():
        abbr = normalize_team(row.get("team_abbr"))
        canonical = clean(row.get("canonical_team"))
        if not abbr:
            continue
        if canonical:
            abbr_to_name[abbr] = canonical
        values: list[object] = [row.get(c) for c in candidates if c in df.columns]
        location = clean(row.get("location"))
        team_name = clean(row.get("team_name"))
        if location and team_name:
            values.append(f"{location} {team_name}")
        for value in values:
            key = normalize_name(value)
            if key:
                existing = to_abbr.get(key)
                if existing is not None and existing != abbr:
                    raise ValueError(
                        f"{path}: ambiguous team alias {value!r} maps to both {existing} and {abbr}"
                    )
                to_abbr[key] = abbr
    return to_abbr, abbr_to_name


def team_abbr(value: object, lookup: dict[str, str]) -> str:
    text = clean(value)
    direct = normalize_team(text)
    if len(direct) <= 4 and direct in set(lookup.values()):
        return direct
    key = normalize_name(text)
    if key not in lookup:
        raise ValueError(f"Could not map NFL team to abbreviation: {text!r}")
    return normalize_team(lookup[key])


@dataclass(frozen=True)
class PlayerRecord:
    gsis_id: str = ""
    pfr_id: str = ""
    espn_id: str = ""
    name: str = ""
    position: str = ""


class PlayerCrosswalk:
    def __init__(self, df: pd.DataFrame) -> None:
        gsis_col = choose_column(df, ["gsis_id"], required=True, label="players")
        pfr_col = choose_column(df, ["pfr_id", "pfr_player_id"])
        espn_col = choose_column(df, ["espn_id"])
        name_col = choose_column(df, ["display_name", "full_name", "player_name"])
        position_col = choose_column(df, ["position", "position_group"])
        self.by_gsis: dict[str, PlayerRecord] = {}
        self.by_espn: dict[str, PlayerRecord] = {}
        name_candidates: dict[str, list[PlayerRecord]] = {}
        for _, row in df.iterrows():
            gsis = clean(row[gsis_col])
            if not gsis:
                continue
            record = PlayerRecord(
                gsis_id=gsis,
                pfr_id=clean(row[pfr_col]) if pfr_col else "",
                espn_id=clean(row[espn_col]) if espn_col else "",
                name=clean(row[name_col]) if name_col else "",
                position=normalize_position(row[position_col]) if position_col else "",
            )
            self.by_gsis[gsis] = record
            if record.espn_id:
                self.by_espn[record.espn_id] = record
            nkey = normalize_name(record.name)
            if nkey:
                name_candidates.setdefault(nkey, []).append(record)
        self.by_unique_name = {
            key: rows[0]
            for key, rows in name_candidates.items()
            if len({r.gsis_id for r in rows}) == 1
        }

    def resolve(self, raw_id: object, name: object = "") -> PlayerRecord | None:
        rid = clean(raw_id)
        if rid in self.by_gsis:
            return self.by_gsis[rid]
        if rid in self.by_espn:
            return self.by_espn[rid]
        key = normalize_name(name)
        return self.by_unique_name.get(key) if key else None


class CurrentRoster:
    def __init__(self, path: Path, players: PlayerCrosswalk) -> None:
        self.by_espn: dict[str, PlayerRecord] = {}
        self.by_name: dict[str, PlayerRecord] = {}
        if not path.exists():
            return
        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False)
        except Exception:
            return
        if df.empty:
            return
        id_col = choose_column(df, ["id", "player_id", "athlete.id", "espn_id"])
        name_col = choose_column(df, ["displayName", "display_name", "full_name", "player_name", "name"])
        pos_col = choose_column(df, ["position.abbreviation", "position_abb", "position", "position.name"])
        if not id_col and not name_col:
            return
        for _, row in df.iterrows():
            eid = clean(row[id_col]) if id_col else ""
            name = clean(row[name_col]) if name_col else ""
            resolved = players.resolve(eid, name)
            record = PlayerRecord(
                gsis_id=resolved.gsis_id if resolved else "",
                pfr_id=resolved.pfr_id if resolved else "",
                espn_id=eid or (resolved.espn_id if resolved else ""),
                name=name or (resolved.name if resolved else ""),
                position=position_for_grouping(row[pos_col]) if pos_col else position_for_grouping(resolved.position if resolved else ""),
            )
            if record.espn_id:
                self.by_espn[record.espn_id] = record
            nkey = normalize_name(record.name)
            if nkey:
                self.by_name[nkey] = record

    def resolve(self, raw_id: object, name: object = "") -> PlayerRecord | None:
        rid = clean(raw_id)
        if rid and rid in self.by_espn:
            return self.by_espn[rid]
        key = normalize_name(name)
        return self.by_name.get(key) if key else None


@dataclass(frozen=True)
class DepthPlayer:
    gsis_id: str
    espn_id: str
    name: str
    position: str
    slot: str
    rank: int


@dataclass
class DepthSnapshot:
    starters_by_slot: dict[str, frozenset[str]]
    rank_by_id: dict[str, int]
    rank_by_name: dict[str, int]
    position_by_id: dict[str, str]
    position_by_name: dict[str, str]

    @classmethod
    def empty(cls) -> "DepthSnapshot":
        return cls({}, {}, {}, {}, {})

    def rank_for(self, gsis_id: str, name: str) -> int | None:
        if gsis_id and gsis_id in self.rank_by_id:
            return self.rank_by_id[gsis_id]
        return self.rank_by_name.get(normalize_name(name))

    def position_for(self, gsis_id: str, name: str) -> str:
        if gsis_id and gsis_id in self.position_by_id:
            return self.position_by_id[gsis_id]
        return self.position_by_name.get(normalize_name(name), "")


def build_depth_snapshot(records: list[DepthPlayer]) -> DepthSnapshot:
    starters: dict[str, set[str]] = {}
    rank_by_id: dict[str, int] = {}
    rank_by_name: dict[str, int] = {}
    pos_by_id: dict[str, str] = {}
    pos_by_name: dict[str, str] = {}
    for record in records:
        identity = record.gsis_id or (f"espn:{record.espn_id}" if record.espn_id else f"name:{normalize_name(record.name)}")
        if record.rank == 1 and record.slot and identity not in {"name:", "espn:"}:
            starters.setdefault(record.slot, set()).add(identity)
        if record.gsis_id:
            previous = rank_by_id.get(record.gsis_id)
            if previous is None or record.rank < previous:
                rank_by_id[record.gsis_id] = record.rank
                if record.position:
                    pos_by_id[record.gsis_id] = record.position
        nkey = normalize_name(record.name)
        if nkey:
            previous = rank_by_name.get(nkey)
            if previous is None or record.rank < previous:
                rank_by_name[nkey] = record.rank
                if record.position:
                    pos_by_name[nkey] = record.position
    return DepthSnapshot(
        starters_by_slot={k: frozenset(v) for k, v in starters.items()},
        rank_by_id=rank_by_id,
        rank_by_name=rank_by_name,
        position_by_id=pos_by_id,
        position_by_name=pos_by_name,
    )


def load_current_depth(root: Path, players: PlayerCrosswalk) -> dict[str, DepthSnapshot]:
    depth_root = root / "data/master/depth_charts"
    if not depth_root.exists():
        raise FileNotFoundError(f"Missing input directory: {depth_root}")
    output: dict[str, DepthSnapshot] = {}
    for path in sorted(depth_root.glob("*/*_depth.csv")):
        df = read_csv(path)
        require_columns(df, ["player_id", "name", "team", "position_abb", "depth_chart_rank", "starter_flag"], str(path))
        records: list[DepthPlayer] = []
        team = ""
        for _, row in df.iterrows():
            team = normalize_team(row["team"])
            rank = parse_int(row["depth_chart_rank"])
            if not team or rank is None:
                continue
            espn_id = clean_id(row["player_id"])
            name = clean(row["name"])
            resolved = players.resolve(espn_id, name)
            slot = normalize_position(row["position_abb"])
            position = position_for_grouping(row["position_abb"])
            records.append(
                DepthPlayer(
                    gsis_id=resolved.gsis_id if resolved else "",
                    espn_id=espn_id,
                    name=name,
                    position=position or position_for_grouping(resolved.position if resolved else ""),
                    slot=slot,
                    rank=rank,
                )
            )
        if team:
            output[team] = build_depth_snapshot(records)
    if not output:
        raise RuntimeError(f"No current depth-chart rows loaded from {depth_root}")
    return output


class HistoricalDepthProvider:
    """Reads current-season nflverse depth history for previous-snapshot lookup."""

    def __init__(self, df: pd.DataFrame, players: PlayerCrosswalk) -> None:
        self.weekly: dict[tuple[str, int], DepthSnapshot] = {}
        self.available_weeks: dict[str, list[int]] = {}
        self.timestamped: dict[str, list[tuple[pd.Timestamp, DepthSnapshot]]] = {}
        week_col = choose_column(df, ["week"])
        has_week = bool(week_col and pd.to_numeric(df[week_col], errors="coerce").notna().any())
        if has_week:
            team_col = choose_column(df, ["club_code", "team"], required=True, label="depth history")
            rank_col = choose_column(df, ["depth_team", "pos_rank"], required=True, label="depth history")
            slot_col = choose_column(df, ["depth_position", "pos_slot", "position"], required=True, label="depth history")
            pos_col = choose_column(df, ["position", "pos_abb"])
            gsis_col = choose_column(df, ["gsis_id"])
            espn_col = choose_column(df, ["espn_id"])
            name_col = choose_column(df, ["full_name", "player_name", "display_name"])
            grouped: dict[tuple[str, int], list[DepthPlayer]] = {}
            for _, row in df.iterrows():
                team = normalize_team(row[team_col])
                week = parse_int(row[week_col])
                rank = parse_int(row[rank_col])
                if not team or week is None or rank is None:
                    continue
                gsis = clean(row[gsis_col]) if gsis_col else ""
                espn = clean(row[espn_col]) if espn_col else ""
                name = clean(row[name_col]) if name_col else ""
                resolved = players.resolve(gsis or espn, name)
                grouped.setdefault((team, week), []).append(
                    DepthPlayer(
                        gsis_id=gsis or (resolved.gsis_id if resolved else ""),
                        espn_id=espn,
                        name=name or (resolved.name if resolved else ""),
                        position=position_for_grouping(row[pos_col]) if pos_col else position_for_grouping(resolved.position if resolved else ""),
                        slot=normalize_position(row[slot_col]),
                        rank=rank,
                    )
                )
            for (team, week), records in grouped.items():
                self.weekly[(team, week)] = build_depth_snapshot(records)
                self.available_weeks.setdefault(team, []).append(week)
            for team in self.available_weeks:
                self.available_weeks[team] = sorted(set(self.available_weeks[team]))
        else:
            team_col = choose_column(df, ["club_code", "team"], required=True, label="depth history")
            ts_col = choose_column(df, ["dt", "timestamp", "date_modified", "date"], required=True, label="depth history")
            rank_col = choose_column(df, ["pos_rank", "depth_team"], required=True, label="depth history")
            slot_col = choose_column(df, ["pos_slot", "depth_position", "position"], required=True, label="depth history")
            pos_col = choose_column(df, ["pos_abb", "position"])
            gsis_col = choose_column(df, ["gsis_id"])
            espn_col = choose_column(df, ["espn_id"])
            name_col = choose_column(df, ["full_name", "player_name", "display_name"])
            grouped: dict[tuple[str, pd.Timestamp], list[DepthPlayer]] = {}
            for _, row in df.iterrows():
                team = normalize_team(row[team_col])
                ts = parse_timestamp(row[ts_col])
                rank = parse_int(row[rank_col])
                if not team or ts is None or rank is None:
                    continue
                gsis = clean(row[gsis_col]) if gsis_col else ""
                espn = clean(row[espn_col]) if espn_col else ""
                name = clean(row[name_col]) if name_col else ""
                resolved = players.resolve(gsis or espn, name)
                grouped.setdefault((team, ts), []).append(
                    DepthPlayer(
                        gsis_id=gsis or (resolved.gsis_id if resolved else ""),
                        espn_id=espn,
                        name=name or (resolved.name if resolved else ""),
                        position=position_for_grouping(row[pos_col]) if pos_col else position_for_grouping(resolved.position if resolved else ""),
                        slot=normalize_position(row[slot_col]),
                        rank=rank,
                    )
                )
            for (team, ts), records in grouped.items():
                self.timestamped.setdefault(team, []).append((ts, build_depth_snapshot(records)))
            for team in self.timestamped:
                self.timestamped[team].sort(key=lambda item: item[0])

    def previous(self, team: str, week: int, kickoff: pd.Timestamp) -> DepthSnapshot | None:
        if self.weekly:
            weeks = self.available_weeks.get(team, [])
            prior = [w for w in weeks if w < week]
            return self.weekly.get((team, prior[-1])) if prior else None
        values = self.timestamped.get(team, [])
        eligible = [(ts, snap) for ts, snap in values if ts < kickoff]
        return eligible[-1][1] if eligible else None


class SnapProvider:
    def __init__(self, df: pd.DataFrame) -> None:
        team_col = choose_column(df, ["team"], required=True, label="snap counts")
        week_col = choose_column(df, ["week"], required=True, label="snap counts")
        pfr_col = choose_column(df, ["pfr_player_id", "pfr_id"])
        name_col = choose_column(df, ["player", "full_name", "player_name"])
        pos_col = choose_column(df, ["position"])
        off_col = choose_column(df, ["offense_pct"], required=True, label="snap counts")
        def_col = choose_column(df, ["defense_pct"], required=True, label="snap counts")
        self.series: dict[tuple[str, str], list[tuple[int, float, float, str]]] = {}
        for _, row in df.iterrows():
            team = normalize_team(row[team_col])
            week = parse_int(row[week_col])
            if not team or week is None:
                continue
            identities: list[str] = []
            pfr = clean(row[pfr_col]) if pfr_col else ""
            name = clean(row[name_col]) if name_col else ""
            if pfr:
                identities.append(f"pfr:{pfr}")
            nkey = normalize_name(name)
            if nkey:
                identities.append(f"name:{nkey}")
            item = (
                week,
                normalize_percentage(row[off_col]),
                normalize_percentage(row[def_col]),
                position_for_grouping(row[pos_col]) if pos_col else "",
            )
            for identity in identities:
                self.series.setdefault((team, identity), []).append(item)
        for key in self.series:
            self.series[key].sort(key=lambda x: x[0])

    def lookup(self, team: str, target_week: int, pfr_id: str, name: str) -> tuple[float, float, str] | None:
        identities = ([f"pfr:{pfr_id}"] if pfr_id else []) + ([f"name:{normalize_name(name)}"] if normalize_name(name) else [])
        for identity in identities:
            values = self.series.get((team, identity), [])
            if not values:
                continue
            weeks = [v[0] for v in values]
            index = bisect_left(weeks, target_week) - 1
            if index >= 0:
                _, off, deff, pos = values[index]
                return off, deff, pos
        return None

    def latest(self, team: str, pfr_id: str, name: str) -> tuple[float, float, str] | None:
        identities = ([f"pfr:{pfr_id}"] if pfr_id else []) + ([f"name:{normalize_name(name)}"] if normalize_name(name) else [])
        for identity in identities:
            values = self.series.get((team, identity), [])
            if values:
                _, off, deff, pos = values[-1]
                return off, deff, pos
        for identity in identities:
            best = None
            for (_team, series_identity), values in self.series.items():
                if series_identity == identity and values:
                    candidate = values[-1]
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            if best is not None:
                _, off, deff, pos = best
                return off, deff, pos
        return None


class ParticipationProvider:
    def __init__(self, df: pd.DataFrame) -> None:
        game_col = choose_column(df, ["nflverse_game_id", "game_id"], required=True, label="participation")
        possession_col = choose_column(df, ["possession_team", "posteam"], required=True, label="participation")
        off_players_col = choose_column(df, ["offense_players"], required=True, label="participation")
        def_players_col = choose_column(df, ["defense_players"], required=True, label="participation")
        offense_den: dict[tuple[str, int], int] = {}
        defense_den: dict[tuple[str, int], int] = {}
        offense_num: dict[tuple[str, int, str], int] = {}
        defense_num: dict[tuple[str, int, str], int] = {}
        for _, row in df.iterrows():
            match = GAME_ID_PATTERN.match(clean(row[game_col]))
            if not match:
                continue
            week = int(match.group(2))
            away = normalize_team(match.group(3))
            home = normalize_team(match.group(4))
            possession = normalize_team(row[possession_col])
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
                    offense_num[(possession, week, pid)] = offense_num.get((possession, week, pid), 0) + 1
            if defense_players:
                defense_den[(defense, week)] = defense_den.get((defense, week), 0) + 1
                for pid in defense_players:
                    defense_num[(defense, week, pid)] = defense_num.get((defense, week, pid), 0) + 1
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
            self.series[key].sort(key=lambda x: x[0])

    def lookup(self, team: str, target_week: int, gsis_id: str) -> tuple[float, float] | None:
        values = self.series.get((team, gsis_id), []) if gsis_id else []
        if not values:
            return None
        weeks = [v[0] for v in values]
        index = bisect_left(weeks, target_week) - 1
        if index < 0:
            return None
        _, off, deff = values[index]
        return off, deff

    def latest(self, team: str, gsis_id: str) -> tuple[float, float] | None:
        if not gsis_id:
            return None
        values = self.series.get((team, gsis_id), [])
        if values:
            _, off, deff = values[-1]
            return off, deff
        best = None
        for (_team, pid), rows in self.series.items():
            if pid == gsis_id and rows:
                candidate = rows[-1]
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            return None
        _, off, deff = best
        return off, deff


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


def load_current_injuries(root: Path, season: int, team_lookup: dict[str, str]) -> dict[str, list[InjuryRecord]]:
    path = root / f"00_intake/injuries/{season}_injuries.csv"
    df = read_csv(path, allow_empty_rows=True)
    require_columns(
        df,
        ["season", "team", "player_id", "player_name", "position", "game_status"],
        str(path),
    )

    # Resolve duplicate player reports using report_date first, then file order.
    # Active/other states participate so a newer Active row supersedes stale Out/Q.
    latest: dict[tuple[str, str], tuple[tuple[int, int, int], InjuryRecord]] = {}
    for row_order, (_, row) in enumerate(df.iterrows()):
        row_season = parse_int(row["season"])
        if row_season is not None and row_season != season:
            continue

        team = team_abbr(row["team"], team_lookup)
        rid = clean_id(row["player_id"])
        name = clean(row["player_name"])
        identity = rid or normalize_name(name)
        if not identity:
            continue

        record = InjuryRecord(
            raw_id=rid,
            name=name,
            position=position_for_grouping(row["position"]),
            status=normalize_injury_status(row["game_status"]),
        )
        ts = parse_timestamp(row.get("report_date", ""))
        rank = (1, int(ts.value), row_order) if ts is not None else (0, 0, row_order)
        key = (team, identity)
        previous = latest.get(key)
        if previous is None or rank > previous[0]:
            latest[key] = (rank, record)

    output: dict[str, list[InjuryRecord]] = {}
    for (team, _), (_, record) in latest.items():
        output.setdefault(team, []).append(record)
    return output


def starter_change_count(previous: DepthSnapshot | None, current: DepthSnapshot) -> int:
    if previous is None or not previous.starters_by_slot or not current.starters_by_slot:
        return 0
    slots = set(previous.starters_by_slot) | set(current.starters_by_slot)
    return sum(
        1
        for slot in slots
        if previous.starters_by_slot.get(slot, frozenset()) != current.starters_by_slot.get(slot, frozenset())
    )


def compute_injury_features(
    team: str,
    week: int,
    injuries: list[InjuryRecord],
    current_depth: DepthSnapshot,
    previous_depth: DepthSnapshot | None,
    players: PlayerCrosswalk,
    current_roster: CurrentRoster,
    snaps: SnapProvider | None,
    participation: ParticipationProvider | None,
    prior_snaps: SnapProvider | None,
    prior_participation: ParticipationProvider | None,
) -> dict[str, float]:
    values = {feature: 0.0 for feature in INJURY_BASE_FEATURES}
    values["depth_starter_changes"] = float(starter_change_count(previous_depth, current_depth))
    for injury in injuries:
        if injury.status == "out":
            values["inj_out_count"] += 1.0
        elif injury.status == "doubtful":
            values["inj_doubtful_count"] += 1.0
        elif injury.status == "questionable":
            values["inj_questionable_count"] += 1.0

        player = players.resolve(injury.raw_id, injury.name)
        roster_player = current_roster.resolve(injury.raw_id, injury.name)
        resolved = player or roster_player or PlayerRecord(
            espn_id=injury.raw_id,
            name=injury.name,
            position=injury.position,
        )
        gsis = resolved.gsis_id
        pfr = resolved.pfr_id
        name = injury.name or resolved.name
        position = injury.position or current_depth.position_for(gsis, name) or resolved.position
        if not position and previous_depth is not None:
            position = previous_depth.position_for(gsis, name)
        position = position_for_grouping(position)
        current_rank = current_depth.rank_for(gsis, name)
        previous_rank = previous_depth.rank_for(gsis, name) if previous_depth is not None else None

        if injury.status != "out":
            continue
        was_starter = current_rank == 1 or previous_rank == 1
        was_top2 = (
            (current_rank is not None and current_rank <= 2)
            or (previous_rank is not None and previous_rank <= 2)
        )
        if was_starter:
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
        if was_top2:
            values["inj_top2_depth_out_count"] += 1.0

        snap_value = snaps.lookup(team, week, pfr, name) if snaps is not None else None
        if snap_value is None and week == 1 and prior_snaps is not None:
            snap_value = prior_snaps.latest(team, pfr, name)
        if snap_value is not None:
            off_share, def_share, _ = snap_value
        else:
            part_value = participation.lookup(team, week, gsis) if participation is not None else None
            if part_value is None and week == 1 and prior_participation is not None:
                part_value = prior_participation.latest(team, gsis)
            if part_value is None:
                off_share, def_share = 0.0, 0.0
            else:
                off_share, def_share = part_value
        values["inj_offense_unavailable_snap_share"] += off_share
        values["inj_defense_unavailable_snap_share"] += def_share

    values["inj_offense_unavailable_snap_share"] = min(1.0, values["inj_offense_unavailable_snap_share"])
    values["inj_defense_unavailable_snap_share"] = min(1.0, values["inj_defense_unavailable_snap_share"])
    return values


def load_schema(root: Path) -> dict:
    path = root / "models/step11_feature_schema.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    required = ["feature_order", "numeric_features", "categorical_features"]
    missing = [key for key in required if key not in schema]
    if missing:
        raise ValueError(f"{path}: missing schema keys: {missing}")

    feature_order = list(schema["feature_order"])
    numeric = list(schema["numeric_features"])
    categorical = list(schema["categorical_features"])
    if len(feature_order) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"{path}: expected {EXPECTED_FEATURE_COUNT} features, found {len(feature_order)}"
        )
    if len(feature_order) != len(set(feature_order)):
        raise ValueError(f"{path}: feature_order contains duplicate names")
    if len(numeric) != len(set(numeric)) or len(categorical) != len(set(categorical)):
        raise ValueError(f"{path}: numeric/categorical feature lists contain duplicates")
    if set(numeric) & set(categorical):
        raise ValueError(f"{path}: numeric and categorical feature lists overlap")
    if set(numeric) | set(categorical) != set(feature_order):
        raise ValueError(
            f"{path}: numeric/categorical feature sets do not exactly cover feature_order"
        )
    return schema


def current_qb_starters(root: Path, players: PlayerCrosswalk) -> dict[str, str]:
    path = root / "config/mapping/qb_map_nfl.csv"
    df = read_csv(path)
    require_columns(df, ["player_id", "team_abbr", "starter_flag", "position_abb"], str(path))
    starters: dict[str, str] = {}
    for _, row in df.iterrows():
        if parse_int(row["starter_flag"]) != 1 or normalize_position(row["position_abb"]) != "QB":
            continue
        team = normalize_team(row["team_abbr"])
        espn_id = clean(row["player_id"])
        resolved = players.resolve(espn_id, row.get("qb_name", ""))
        if resolved is None or not resolved.gsis_id:
            raise ValueError(f"{path}: could not map current QB1 ESPN id {espn_id!r} for {team} to GSIS id")
        if team in starters and starters[team] != resolved.gsis_id:
            raise ValueError(f"{path}: multiple QB1 rows for {team}")
        starters[team] = resolved.gsis_id
    return starters


def load_team_stats_for_week(root: Path, season: int, week: int, week1_mode: bool) -> dict[str, dict[str, float | None]]:
    source_season = season - 1 if week1_mode else season
    path = root / f"00_intake/team_stats/{source_season}_team_stats.csv"
    df = read_csv(path)
    require_columns(df, ["season", "week", "team", *TEAM_METRICS], str(path))
    df["_week"] = pd.to_numeric(df["week"], errors="coerce")
    result: dict[str, dict[str, float | None]] = {}
    for team, group in df.groupby(df["team"].map(normalize_team), sort=False):
        if week1_mode:
            valid = group[group["_week"].notna()]
            if valid.empty:
                continue
            chosen = valid.sort_values("_week", kind="stable").iloc[-1]
        else:
            wanted = week - 1
            valid = group[group["_week"] == wanted]
            if valid.empty:
                # Match historical Step 5: after a bye, exact N-1 features are
                # missing rather than carrying forward an older played week.
                result[normalize_team(team)] = {metric: None for metric in TEAM_METRICS}
                continue
            if len(valid) > 1:
                raise ValueError(f"{path}: duplicate team/week rows for {team}, week {wanted}")
            chosen = valid.iloc[0]
        result[normalize_team(team)] = {metric: parse_float(chosen[metric]) for metric in TEAM_METRICS}
    return result


def load_qb_stats_for_week(
    root: Path,
    season: int,
    week: int,
    week1_mode: bool,
    starters: dict[str, str],
) -> dict[str, dict[str, float | None]]:
    source_season = season - 1 if week1_mode else season
    path = root / f"00_intake/qb/{source_season}_qb_stats.csv"
    df = read_csv(path)
    require_columns(df, ["season", "week", "player_id", "dropbacks", *QB_METRICS], str(path))
    df["_week"] = pd.to_numeric(df["week"], errors="coerce")
    df["_dropbacks"] = pd.to_numeric(df["dropbacks"], errors="coerce").fillna(-1)
    df["_player"] = df["player_id"].astype("string").str.strip()
    result: dict[str, dict[str, float | None]] = {}
    for team, player_id in starters.items():
        rows = df[df["_player"] == player_id].copy()
        if week1_mode:
            rows = rows[rows["_week"].notna()]
        else:
            rows = rows[rows["_week"] < week]
        if rows.empty:
            result[team] = {metric: None for metric in QB_METRICS}
            continue
        # deterministic resolution: latest source week; within that week most dropbacks.
        latest_week = rows["_week"].max()
        rows = rows[rows["_week"] == latest_week].sort_values("_dropbacks", kind="stable")
        chosen = rows.iloc[-1]
        result[team] = {metric: parse_float(chosen[metric]) for metric in QB_METRICS}
    return result


def build_division_lookup(root: Path, season: int) -> dict[str, str]:
    path = root / "data/master/league_master.csv"
    df = read_csv(path)
    require_columns(df, ["team_abbr", "division", "season"], str(path))
    season_num = pd.to_numeric(df["season"], errors="coerce")
    rows = df[season_num == season]
    return {normalize_team(row["team_abbr"]): clean(row["division"]) for _, row in rows.iterrows()}


def build_stadium_lookup(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    path = root / "config/mapping/stadium_map_nfl.csv"
    df = read_csv(path)
    require_columns(df, ["team", "stadium", "venue_id"], str(path))
    by_stadium: dict[str, str] = {}
    by_team_name: dict[str, str] = {}
    for _, row in df.iterrows():
        venue_id = clean(row["venue_id"])
        if clean(row["stadium"]):
            by_stadium[normalize_name(row["stadium"])] = venue_id
        if clean(row["team"]):
            by_team_name[normalize_name(row["team"])] = venue_id
    return by_stadium, by_team_name


def roof_for_model(value: object) -> str:
    text = clean(value).casefold()
    mapping = {
        "open_air": "outdoors",
        "open air": "outdoors",
        "outdoor": "outdoors",
        "fixed_roof": "dome",
        "fixed roof": "dome",
        "retractable": "closed",
    }
    return mapping.get(text, text)


def surface_for_model(value: object) -> str:
    return clean(value).casefold()


def rest_days(full_schedule: pd.DataFrame, team: str, target_week: int, target_date: str, lookup: dict[str, str]) -> int:
    if target_week == 1:
        return 7
    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(target):
        raise ValueError(f"Invalid target game date: {target_date!r}")
    candidates: list[pd.Timestamp] = []
    for _, row in full_schedule.iterrows():
        row_week = parse_int(row["week"])
        if row_week is None or row_week >= target_week:
            continue
        home = team_abbr(row["home_team"], lookup)
        away = team_abbr(row["away_team"], lookup)
        if team not in {home, away}:
            continue
        date = pd.to_datetime(clean(row["game_date"]), errors="coerce")
        if not pd.isna(date) and date < target:
            candidates.append(date)
    if not candidates:
        raise ValueError(f"No prior current-season game found for {team} before Week {target_week}")
    return int((target - max(candidates)).days)


def precip_labels(row: pd.Series) -> tuple[str, str]:
    snow = parse_int(row.get("wx_snow_flag")) == 1
    rain = parse_int(row.get("wx_rain_flag")) == 1
    if snow:
        return "snow", "snow"
    if rain:
        return "rain", "rain"
    return "", ""


def add_team_qb_features(
    work: pd.DataFrame,
    team_stats: dict[str, dict[str, float | None]],
    qb_stats: dict[str, dict[str, float | None]],
) -> None:
    for idx, row in work.iterrows():
        home = row["_home_abbr"]
        away = row["_away_abbr"]
        if home not in team_stats or away not in team_stats:
            raise ValueError(f"Missing required team-stat source row for game_id={row['game_id']} home={home} away={away}")
        for metric in TEAM_METRICS:
            hv = team_stats[home][metric]
            av = team_stats[away][metric]
            work.at[idx, f"home_{metric}"] = hv
            work.at[idx, f"away_{metric}"] = av
            work.at[idx, f"{metric}_diff"] = None if hv is None or av is None else hv - av
        for metric in QB_METRICS:
            hv = qb_stats.get(home, {}).get(metric)
            av = qb_stats.get(away, {}).get(metric)
            work.at[idx, f"home_qb_{metric}"] = hv
            work.at[idx, f"away_qb_{metric}"] = av
            work.at[idx, f"qb_{metric}_diff"] = None if hv is None or av is None else hv - av


def add_schedule_features(
    work: pd.DataFrame,
    full_schedule: pd.DataFrame,
    team_lookup: dict[str, str],
    divisions: dict[str, str],
    stadium_by_name: dict[str, str],
    stadium_by_team: dict[str, str],
) -> None:
    for idx, row in work.iterrows():
        week = int(row["_target_week"])
        home = row["_home_abbr"]
        away = row["_away_abbr"]
        game_date = clean(row["sched_game_date"])
        season_type = clean(row["sched_season_type"]).casefold()
        work.at[idx, "game_type"] = {"reg": "REG", "pre": "PRE", "post": "POST"}.get(season_type, season_type.upper())
        work.at[idx, "week"] = week
        dt = pd.to_datetime(game_date, errors="coerce")
        work.at[idx, "weekday"] = "" if pd.isna(dt) else dt.day_name()
        work.at[idx, "gametime"] = clean(row["sched_game_time"])
        work.at[idx, "away_team"] = away
        work.at[idx, "home_team"] = home
        neutral = parse_int(row["sched_neutral_site"]) == 1
        work.at[idx, "location"] = "Neutral" if neutral else "Home"
        away_rest = rest_days(full_schedule, away, week, game_date, team_lookup)
        home_rest = rest_days(full_schedule, home, week, game_date, team_lookup)
        work.at[idx, "away_rest"] = away_rest
        work.at[idx, "home_rest"] = home_rest
        work.at[idx, "rest_diff"] = home_rest - away_rest
        work.at[idx, "div_game"] = int(bool(divisions.get(home)) and divisions.get(home) == divisions.get(away))
        work.at[idx, "roof"] = roof_for_model(row["sched_roof"])
        work.at[idx, "surface"] = surface_for_model(row["sched_surface"])
        work.at[idx, "stadium"] = clean(row["sched_stadium"])
        stadium_key = normalize_name(row["sched_stadium"])
        team_key = normalize_name(row["base_home_team"])
        work.at[idx, "stadium_id"] = stadium_by_name.get(stadium_key, stadium_by_team.get(team_key, ""))

        work.at[idx, "temp"] = parse_float(row.get("wx_temperature"))
        work.at[idx, "wind"] = parse_float(row.get("wx_wind_speed"))
        work.at[idx, "hist_surface"] = surface_for_model(row["sched_surface"])
        icon, precip_type = precip_labels(row)
        work.at[idx, "hist_weather_icon"] = icon
        work.at[idx, "hist_temperature"] = parse_float(row.get("wx_temperature"))
        work.at[idx, "hist_precip_probability"] = parse_float(row.get("wx_precip_probability"))
        work.at[idx, "hist_precip_type"] = precip_type
        work.at[idx, "hist_wind_speed"] = parse_float(row.get("wx_wind_speed"))
        # The current weather intake has no wind-bearing field.
        work.at[idx, "hist_wind_bearing"] = np.nan


def add_injury_features(
    work: pd.DataFrame,
    season: int,
    week1_mode: bool,
    current_depth: dict[str, DepthSnapshot],
    depth_history: HistoricalDepthProvider | None,
    injuries: dict[str, list[InjuryRecord]],
    players: PlayerCrosswalk,
    current_roster: CurrentRoster,
    snaps: SnapProvider | None,
    participation: ParticipationProvider | None,
    prior_snaps: SnapProvider | None,
    prior_participation: ParticipationProvider | None,
) -> None:
    for idx, row in work.iterrows():
        week = int(row["_target_week"])
        kickoff = parse_timestamp(row.get("sched_commence_time"))
        if kickoff is None:
            date = clean(row["sched_game_date"])
            kickoff = pd.Timestamp(date, tz="UTC")
        sides: dict[str, dict[str, float]] = {}
        for side, team in [("home", row["_home_abbr"]), ("away", row["_away_abbr"])]:
            current = current_depth.get(team, DepthSnapshot.empty())
            previous = None if week1_mode or depth_history is None else depth_history.previous(team, week, kickoff)
            sides[side] = compute_injury_features(
                team=team,
                week=week,
                injuries=injuries.get(team, []),
                current_depth=current,
                previous_depth=previous,
                players=players,
                current_roster=current_roster,
                snaps=snaps,
                participation=participation,
                prior_snaps=prior_snaps,
                prior_participation=prior_participation,
            )
        for feature in INJURY_BASE_FEATURES:
            hv = sides["home"][feature]
            av = sides["away"][feature]
            work.at[idx, f"home_{feature}"] = hv
            work.at[idx, f"away_{feature}"] = av
            work.at[idx, f"{feature}_diff"] = hv - av


def prepare_week(root: Path, season: int, week: int, week1_mode: bool, schema: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined_path = root / f"00_intake/predictions/enriched/combined/week_{week}_NFL_enriched.csv"
    base = read_csv(combined_path)
    require_columns(base, ["season", "season_type", "week", "game_id", "away_team", "home_team"], str(combined_path))
    season_values = {parse_int(v) for v in base["season"]}
    week_values = {parse_int(v) for v in base["week"]}
    if season_values != {season} or week_values != {week}:
        raise ValueError(f"{combined_path}: expected only season={season}, week={week}; found seasons={season_values}, weeks={week_values}")
    original = base.copy()
    collisions = [column for column in OUTPUT_COLUMNS if column in original.columns]
    if collisions:
        raise ValueError(
            f"{combined_path}: prediction columns already exist and would be overwritten: {collisions}"
        )

    # dtype=str may use Arrow-backed string arrays. Feature construction writes
    # numeric values into existing source columns, so use a flexible work copy.
    base = base.astype(object).copy()
    base["base_home_team"] = base["home_team"]
    base["base_away_team"] = base["away_team"]

    team_lookup, _ = load_team_name_lookup(root)
    base["_home_abbr"] = base["home_team"].map(lambda v: team_abbr(v, team_lookup))
    base["_away_abbr"] = base["away_team"].map(lambda v: team_abbr(v, team_lookup))
    base["_target_week"] = week

    weekly_schedule_path = root / f"00_intake/schedule/weekly/week_{week}_NFL_weekly_schedule.csv"
    market = read_csv(weekly_schedule_path)
    require_columns(
        market,
        [
            "game_id", "home_moneyline_american", "away_moneyline_american",
            "home_spread", "away_spread", "home_spread_american",
            "away_spread_american", "total", "over_american", "under_american",
        ],
        str(weekly_schedule_path),
    )
    market["hist_home_spread"] = market["home_spread"]
    market["hist_away_spread"] = market["away_spread"]
    market["hist_odds_total"] = market["total"]
    market["spread_line"] = -pd.to_numeric(market["home_spread"], errors="coerce")
    market = market.rename(columns={
        "home_moneyline_american": "home_moneyline",
        "away_moneyline_american": "away_moneyline",
        "away_spread_american": "away_spread_odds",
        "home_spread_american": "home_spread_odds",
        "total": "total_line",
        "under_american": "under_odds",
        "over_american": "over_odds",
    })
    base = merge_game_source(base, market, [
        "away_moneyline", "home_moneyline", "spread_line", "away_spread_odds",
        "home_spread_odds", "total_line", "under_odds", "over_odds",
        "hist_home_spread", "hist_away_spread", "hist_odds_total",
    ], str(weekly_schedule_path))

    drat_path = root / f"00_intake/predictions/drat/clean/{season}_week_{week}_drat.csv"
    drat = read_csv(drat_path).rename(columns={
        "away_prob": "drat_away_prob",
        "home_prob": "drat_home_prob",
        "moneyline_away": "drat_away_moneyline",
        "moneyline_home": "drat_home_moneyline",
        "spread_away": "drat_away_spread",
        "spread_home": "drat_home_spread",
    })
    base = merge_game_source(base, drat, [
        "drat_away_prob", "drat_home_prob", "drat_away_moneyline", "drat_home_moneyline",
        "drat_away_spread", "drat_home_spread",
    ], str(drat_path))

    season_types = [clean(v) for v in original["season_type"].dropna().unique() if clean(v)]
    if len(set(season_types)) != 1:
        raise ValueError(f"{combined_path}: expected one season_type, found {season_types}")
    season_type = season_types[0]
    epred_path = root / f"00_intake/predictions/final/{season}_{season_type}_{week}_clean_predictions.csv"
    epred = read_csv(epred_path).rename(columns={
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
    })
    base = merge_game_source(base, epred, [
        "epred_matchupQuality", "epred_home_prob", "epred_away_prob", "epred_tie_prob",
        "epred_away_projected_pts", "epred_home_projected_pts", "epred_total_projected_pts",
        "epred_home_PtDiff", "epred_away_PtDiff", "epred_home_rating", "epred_away_rating",
    ], str(epred_path))

    schedule_path = root / f"00_intake/schedule/{season}_schedule.csv"
    full_schedule = read_csv(schedule_path)
    require_columns(full_schedule, ["season", "season_type", "week", "game_id", "game_date", "game_time", "away_team", "home_team", "neutral_site", "stadium", "roof", "surface"], str(schedule_path))
    schedule_week = full_schedule[pd.to_numeric(full_schedule["week"], errors="coerce") == week].copy()
    schedule_week = schedule_week.rename(columns={
        "season_type": "sched_season_type",
        "game_date": "sched_game_date",
        "game_time": "sched_game_time",
        "neutral_site": "sched_neutral_site",
        "stadium": "sched_stadium",
        "roof": "sched_roof",
        "surface": "sched_surface",
    })
    # Exact UTC commence_time is available in weekly schedule and useful for depth-history cutoff.
    commence = read_csv(weekly_schedule_path)[["game_id", "commence_time"]].rename(columns={"commence_time": "sched_commence_time"})
    schedule_week = schedule_week.merge(commence, on="game_id", how="left", validate="one_to_one")
    base = merge_game_source(base, schedule_week, [
        "sched_season_type", "sched_game_date", "sched_game_time", "sched_neutral_site",
        "sched_stadium", "sched_roof", "sched_surface", "sched_commence_time",
    ], str(schedule_path))

    weather_path = root / f"data/weather/week_{week}_NFL_weekly_weather.csv"
    weather = read_csv(weather_path).rename(columns={
        "temperature": "wx_temperature",
        "wind_speed": "wx_wind_speed",
        "precip_probability": "wx_precip_probability",
        "rain_flag": "wx_rain_flag",
        "snow_flag": "wx_snow_flag",
    })
    base = merge_game_source(base, weather, [
        "wx_temperature", "wx_wind_speed", "wx_precip_probability", "wx_rain_flag", "wx_snow_flag",
    ], str(weather_path))

    travel_path = root / f"data/travel/{season}_week_{week}_travel.csv"
    travel = read_csv(travel_path)
    travel_cols = ["miles_traveled", "time_zones_crossed", "east_to_west", "west_to_east", "international_flag", "neutral_site_flag"]
    base = merge_game_source(base, travel, travel_cols, str(travel_path))

    players_path = root / "data/historic_data/players/players.parquet"
    players = PlayerCrosswalk(read_parquet(players_path))
    roster = CurrentRoster(root / "data/master/roster_master.csv", players)
    starters = current_qb_starters(root, players)
    team_stats = load_team_stats_for_week(root, season, week, week1_mode)
    qb_stats = load_qb_stats_for_week(root, season, week, week1_mode, starters)
    add_team_qb_features(base, team_stats, qb_stats)

    divisions = build_division_lookup(root, season)
    stadium_by_name, stadium_by_team = build_stadium_lookup(root)
    add_schedule_features(base, full_schedule, team_lookup, divisions, stadium_by_name, stadium_by_team)

    current_depth = load_current_depth(root, players)
    current_injuries = load_current_injuries(root, season, team_lookup)
    if week1_mode:
        prior_season = season - 1
        prior_snaps = SnapProvider(read_parquet(root / f"data/historic_data/snap_counts/snap_counts_{prior_season}.parquet"))
        prior_participation = ParticipationProvider(read_parquet(root / f"data/historic_data/participation/pbp_participation_{prior_season}.parquet"))
        snaps = None
        participation = None
        depth_history = None
    else:
        snap_path = root / f"data/historic_data/snap_counts/snap_counts_{season}.parquet"
        participation_path = root / f"data/historic_data/participation/pbp_participation_{season}.parquet"
        depth_history_path = root / f"data/historic_data/depth_charts/depth_charts_{season}.parquet"
        for required_path in [snap_path, participation_path, depth_history_path]:
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Week {week} in-season projection requires current-season source: {required_path}"
                )
            if required_path.stat().st_size == 0:
                raise RuntimeError(f"Required current-season source is zero bytes: {required_path}")

        snaps = SnapProvider(read_parquet(snap_path))
        participation = ParticipationProvider(read_parquet(participation_path))
        depth_history = HistoricalDepthProvider(read_parquet(depth_history_path), players)
        prior_snaps = None
        prior_participation = None
    add_injury_features(
        base, season, week1_mode, current_depth, depth_history, current_injuries,
        players, roster, snaps, participation, prior_snaps, prior_participation,
    )

    missing_features = [c for c in schema["feature_order"] if c not in base.columns]
    if missing_features:
        raise RuntimeError(f"Week {week}: could not construct Step 11 features: {missing_features}")
    features = base[schema["feature_order"]].copy()
    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Week {week}: expected {EXPECTED_FEATURE_COUNT} model features, found {features.shape[1]}"
        )
    if features.columns.tolist() != list(schema["feature_order"]):
        raise RuntimeError(f"Week {week}: feature names/order do not match Step 11 schema")

    numeric = set(schema["numeric_features"])
    categorical = set(schema["categorical_features"])
    for col in schema["feature_order"]:
        if col in numeric:
            features[col] = pd.to_numeric(features[col], errors="coerce")
        elif col in categorical:
            features[col] = features[col].map(clean).replace("", MISSING_CAT).astype(str)
        else:
            raise RuntimeError(f"Schema feature is not classified: {col}")

    spread_line = pd.to_numeric(features["spread_line"], errors="coerce").to_numpy(dtype=float)
    total_line = pd.to_numeric(features["total_line"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(spread_line).all():
        bad = original.loc[~np.isfinite(spread_line), "game_id"].tolist()
        raise ValueError(f"spread_line is missing/non-numeric for game_id values: {bad[:10]}")
    if not np.isfinite(total_line).all():
        bad = original.loc[~np.isfinite(total_line), "game_id"].tolist()
        raise ValueError(f"total_line is missing/non-numeric for game_id values: {bad[:10]}")

    return original, features


def sigmoid(value: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    arr = np.clip(arr, -700, 700)
    return 1.0 / (1.0 + np.exp(-arr))


def load_calibrations(root: Path) -> dict:
    path = root / "models/step14_probability_calibration.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
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
        section = calibrations.get(key)
        if not isinstance(section, dict):
            raise ValueError(f"{path}: missing calibration section {key!r}")
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


def apply_models(root: Path, original: pd.DataFrame, features: pd.DataFrame, schema: dict) -> pd.DataFrame:
    margin_path = root / "models/step11_margin_model.cbm"
    total_path = root / "models/step11_total_points_model.cbm"
    for path in [margin_path, total_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    margin_model = CatBoostRegressor()
    total_model = CatBoostRegressor()
    margin_model.load_model(str(margin_path))
    total_model.load_model(str(total_path))

    expected = list(schema["feature_order"])
    if list(features.columns) != expected:
        raise RuntimeError("Prediction feature names/order do not match step11_feature_schema.json")
    if list(margin_model.feature_names_) != expected:
        raise RuntimeError("Margin model feature names/order do not match step11_feature_schema.json")
    if list(total_model.feature_names_) != expected:
        raise RuntimeError("Total model feature names/order do not match step11_feature_schema.json")

    predicted_margin = np.asarray(margin_model.predict(features), dtype=float)
    predicted_total = np.asarray(total_model.predict(features), dtype=float)
    if len(predicted_margin) != len(original) or len(predicted_total) != len(original):
        raise RuntimeError("Prediction row count does not match projection input")
    if not np.isfinite(predicted_margin).all():
        raise RuntimeError("Margin model produced non-finite predictions")
    if not np.isfinite(predicted_total).all():
        raise RuntimeError("Total model produced non-finite predictions")

    predicted_home = (predicted_total + predicted_margin) / 2.0
    predicted_away = (predicted_total - predicted_margin) / 2.0

    calibration = load_calibrations(root)
    ml = calibration["moneyline"]
    sp = calibration["spread"]
    tot = calibration["total"]
    spread_line = pd.to_numeric(features["spread_line"], errors="coerce").to_numpy(dtype=float)
    total_line = pd.to_numeric(features["total_line"], errors="coerce").to_numpy(dtype=float)

    home_win = sigmoid(float(ml["intercept"]) + float(ml["slope"]) * predicted_margin)
    away_win = 1.0 - home_win
    home_cover = sigmoid(float(sp["intercept"]) + float(sp["slope"]) * (predicted_margin - spread_line))
    away_cover = 1.0 - home_cover
    over = sigmoid(float(tot["intercept"]) + float(tot["slope"]) * (predicted_total - total_line))
    under = 1.0 - over

    validate_probability_pair(home_win, away_win, "moneyline")
    validate_probability_pair(home_cover, away_cover, "spread")
    validate_probability_pair(over, under, "total")

    if not np.allclose(predicted_home + predicted_away, predicted_total, rtol=0.0, atol=1e-12):
        raise RuntimeError("Predicted scores do not sum to predicted_total")
    if not np.allclose(predicted_home - predicted_away, predicted_margin, rtol=0.0, atol=1e-12):
        raise RuntimeError("Predicted scores do not reconcile to predicted_margin")

    output = original.copy()
    output["predicted_margin"] = predicted_margin
    output["predicted_total"] = predicted_total
    output["predicted_home_score"] = predicted_home
    output["predicted_away_score"] = predicted_away
    output["home_win_probability"] = home_win
    output["away_win_probability"] = away_win
    output["home_cover_probability"] = home_cover
    output["away_cover_probability"] = away_cover
    output["over_probability"] = over
    output["under_probability"] = under

    expected_columns = [*original.columns.tolist(), *OUTPUT_COLUMNS]
    if output.columns.tolist() != expected_columns:
        raise RuntimeError(
            "Final output columns are not original columns plus the exact 10 prediction columns"
        )
    if not output["home_team"].equals(original["home_team"]):
        raise RuntimeError("Final output home_team values changed from the original input")
    if not output["away_team"].equals(original["away_team"]):
        raise RuntimeError("Final output away_team values changed from the original input")
    if output["game_id"].tolist() != original["game_id"].tolist():
        raise RuntimeError("Final output game_id row order changed")
    require_unique_game_id(output, "final projection output")
    return output


def infer_inseason_target_week(root: Path, season: int) -> int:
    """
    The Tuesday pipeline builds current-season team stats from completed PBP.
    Therefore the latest populated team-stat week is the latest completed week,
    and the projection target is exactly the following week.
    """
    path = root / f"00_intake/team_stats/{season}_team_stats.csv"
    df = read_csv(path)
    require_columns(df, ["season", "week", "team", *TEAM_METRICS], str(path))
    season_num = pd.to_numeric(df["season"], errors="coerce")
    week_num = pd.to_numeric(df["week"], errors="coerce")
    available = week_num[(season_num == season) & week_num.notna()]
    if available.empty:
        raise RuntimeError(
            f"{path}: no completed current-season team-stat weeks are available; "
            "use projection_week1.py for Week 1"
        )
    latest_completed_week = int(available.max())

    # Reject a partially populated completed week before advancing the target.
    # This compares only teams scheduled to play, so bye weeks remain valid.
    schedule_path = root / f"00_intake/schedule/{season}_schedule.csv"
    schedule = read_csv(schedule_path)
    require_columns(
        schedule,
        ["season", "season_type", "week", "home_team", "away_team"],
        str(schedule_path),
    )
    schedule_rows = schedule[
        (pd.to_numeric(schedule["season"], errors="coerce") == season)
        & (pd.to_numeric(schedule["week"], errors="coerce") == latest_completed_week)
        & (schedule["season_type"].map(clean).str.casefold() == "reg")
    ]
    if schedule_rows.empty:
        raise RuntimeError(
            f"{schedule_path}: no regular-season schedule rows for completed Week {latest_completed_week}"
        )
    team_lookup, _ = load_team_name_lookup(root)
    expected_teams = {
        team_abbr(value, team_lookup)
        for value in pd.concat([schedule_rows["home_team"], schedule_rows["away_team"]], ignore_index=True)
    }
    actual_teams = {
        normalize_team(value)
        for value in df.loc[
            (season_num == season) & (week_num == latest_completed_week), "team"
        ].dropna()
    }
    missing_teams = sorted(expected_teams - actual_teams)
    if missing_teams:
        raise RuntimeError(
            f"{path}: latest team-stat week {latest_completed_week} is incomplete; "
            f"missing scheduled teams: {missing_teams}"
        )

    target_week = latest_completed_week + 1
    combined_path = root / f"00_intake/predictions/enriched/combined/week_{target_week}_NFL_enriched.csv"
    if not combined_path.exists():
        raise FileNotFoundError(
            f"Next projection week is Week {target_week}, but its combined input is missing: {combined_path}"
        )
    return target_week


def run_projection(*, season: int, week1_mode: bool) -> list[Path]:
    root = nfl_root()
    schema = load_schema(root)
    week = 1 if week1_mode else infer_inseason_target_week(root, season)

    output_dir = root / "01_merge"
    output_dir.mkdir(parents=True, exist_ok=True)

    original, features = prepare_week(root, season, week, week1_mode, schema)
    projected = apply_models(root, original, features, schema)
    output_path = output_dir / f"week_{week}_NFL_enriched.csv"
    projected.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"WROTE {output_path} | games={len(projected)} | columns={len(projected.columns)}")
    return [output_path]


def main() -> None:
    print(f"projection.py version={SCRIPT_VERSION}")
    run_projection(season=SEASON, week1_mode=False)


if __name__ == "__main__":
    main()

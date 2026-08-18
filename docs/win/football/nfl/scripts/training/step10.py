#!/usr/bin/env python3
"""
Step 10: add historical depth-chart and injury features.

READS ONLY:
  docs/win/football/nfl/data/historic_data/depth_charts/depth_charts_{season}.parquet
  docs/win/football/nfl/data/historic_data/injuries/injuries_{season}.parquet
  docs/win/football/nfl/data/historic_data/participation/pbp_participation_{season}.parquet
  docs/win/football/nfl/data/historic_data/snap_counts/snap_counts_{season}.parquet
  docs/win/football/nfl/data/historic_data/weekly_rosters/roster_weekly_{season}.parquet
  docs/win/football/nfl/data/historic_data/players/players.parquet

UPDATES IN PLACE:
  docs/win/football/nfl/training/historical_core_2021.csv
  docs/win/football/nfl/training/historical_core_2022.csv
  docs/win/football/nfl/training/historical_core_2023.csv
  docs/win/football/nfl/training/historical_core_2024.csv
  docs/win/football/nfl/training/historical_core_2025.csv

For home and away teams, creates:
  inj_out_count
  inj_doubtful_count
  inj_questionable_count
  inj_starter_out_count
  inj_top2_depth_out_count
  inj_qb1_out
  inj_ol_starter_out_count
  inj_skill_starter_out_count
  inj_front7_starter_out_count
  inj_secondary_starter_out_count
  inj_offense_unavailable_snap_share
  inj_defense_unavailable_snap_share
  depth_starter_changes

For every feature, also creates home-minus-away *_diff.

Leakage protection:
  - injury/depth values are matched to the target game's team/week
  - 2025 timestamp-based depth charts use the latest snapshot strictly before kickoff
  - snap-count and participation weighting uses only games from an earlier week
  - Week 1 snap/participation weighting falls back to the latest prior-season usage
  - if an entire injury source week is absent, the latest strictly earlier injury week is used
  - current-game snap counts/participation are never used for the current game

Depth-chart handling:
  - 2021-2024 nflverse depth charts use week/depth_team/depth_position
  - 2025+ nflverse depth charts use timestamp dt/pos_rank/pos_slot

The script is idempotent: existing Step 10 columns are removed and rebuilt.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import re
import sys
import unicodedata

import pandas as pd


NFL_ROOT = Path("docs/win/football/nfl")
HISTORIC_ROOT = NFL_ROOT / "data/historic_data"
TRAINING_DIR = NFL_ROOT / "training"

DEPTH_DIR = HISTORIC_ROOT / "depth_charts"
INJURY_DIR = HISTORIC_ROOT / "injuries"
PARTICIPATION_DIR = HISTORIC_ROOT / "participation"
SNAP_DIR = HISTORIC_ROOT / "snap_counts"
ROSTER_DIR = HISTORIC_ROOT / "weekly_rosters"
PLAYERS_PATH = HISTORIC_ROOT / "players/players.parquet"

SEASONS = [2021, 2022, 2023, 2024, 2025]

TRAINING_PATHS = {
    season: TRAINING_DIR / f"historical_core_{season}.csv"
    for season in SEASONS
}

BASE_FEATURES = [
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

GENERATED_COLUMNS: list[str] = []
for feature in BASE_FEATURES:
    GENERATED_COLUMNS.extend(
        [
            f"home_{feature}",
            f"away_{feature}",
            f"{feature}_diff",
        ]
    )

TRAINING_REQUIRED = [
    "season",
    "week",
    "home_team",
    "away_team",
    "gameday",
]

COUNT_FEATURES = {
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
    "depth_starter_changes",
}

SNAP_FEATURES = {
    "inj_offense_unavailable_snap_share",
    "inj_defense_unavailable_snap_share",
}

TEAM_ALIASES = {
    "WSH": "WAS",
    "LA": "LAR",
    "JAC": "JAX",
}

BLANK_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "<na>",
    "nat",
}

OFFENSIVE_LINE_POSITIONS = {
    "C",
    "G",
    "OG",
    "LG",
    "RG",
    "T",
    "OT",
    "LT",
    "RT",
    "OL",
}

SKILL_POSITIONS = {
    "RB",
    "HB",
    "FB",
    "WR",
    "TE",
}

FRONT7_POSITIONS = {
    "DL",
    "DE",
    "DT",
    "NT",
    "EDGE",
    "LB",
    "ILB",
    "OLB",
    "MLB",
}

SECONDARY_POSITIONS = {
    "DB",
    "CB",
    "S",
    "FS",
    "SS",
    "NB",
}

GSIS_PATTERN = re.compile(r"00-\d{7}")
GAME_ID_PATTERN = re.compile(
    r"^(\d{4})_(\d{1,2})_([A-Za-z0-9]+)_([A-Za-z0-9]+)$"
)


@dataclass(frozen=True)
class PlayerRecord:
    gsis_id: str = ""
    pfr_id: str = ""
    espn_id: str = ""
    name: str = ""
    position: str = ""


@dataclass(frozen=True)
class InjuryRecord:
    gsis_id: str
    raw_player_id: str
    name: str
    position: str
    status: str


@dataclass(frozen=True)
class DepthPlayer:
    gsis_id: str
    name: str
    position: str
    slot: str
    rank: int


@dataclass
class DepthSnapshot:
    players: list[DepthPlayer]
    starters_by_slot: dict[str, frozenset[str]]
    rank_by_id: dict[str, int]
    rank_by_name: dict[str, int]
    position_by_id: dict[str, str]
    position_by_name: dict[str, str]

    @classmethod
    def empty(cls) -> "DepthSnapshot":
        return cls(
            players=[],
            starters_by_slot={},
            rank_by_id={},
            rank_by_name={},
            position_by_id={},
            position_by_name={},
        )

    def rank_for(self, gsis_id: str, name: str) -> int | None:
        if gsis_id and gsis_id in self.rank_by_id:
            return self.rank_by_id[gsis_id]
        key = normalize_name(name)
        if key and key in self.rank_by_name:
            return self.rank_by_name[key]
        return None

    def position_for(self, gsis_id: str, name: str) -> str:
        if gsis_id and gsis_id in self.position_by_id:
            return self.position_by_id[gsis_id]
        key = normalize_name(name)
        if key and key in self.position_by_name:
            return self.position_by_name[key]
        return ""


class PlayerCrosswalk:
    def __init__(self, df: pd.DataFrame) -> None:
        gsis_col = choose_column(df, ["gsis_id"], required=True, label="players")
        pfr_col = choose_column(df, ["pfr_id", "pfr_player_id"], required=False)
        espn_col = choose_column(df, ["espn_id"], required=False)
        name_col = choose_column(
            df,
            ["display_name", "full_name", "player_name"],
            required=False,
        )
        position_col = choose_column(
            df,
            ["position", "position_group"],
            required=False,
        )

        self.by_gsis: dict[str, PlayerRecord] = {}
        self.by_espn: dict[str, PlayerRecord] = {}
        name_candidates: dict[str, list[PlayerRecord]] = {}

        for _, row in df.iterrows():
            gsis_id = clean(row[gsis_col])
            if not gsis_id:
                continue
            record = PlayerRecord(
                gsis_id=gsis_id,
                pfr_id=clean(row[pfr_col]) if pfr_col else "",
                espn_id=clean(row[espn_col]) if espn_col else "",
                name=clean(row[name_col]) if name_col else "",
                position=normalize_position(row[position_col]) if position_col else "",
            )
            self.by_gsis[gsis_id] = record
            if record.espn_id:
                self.by_espn[record.espn_id] = record
            name_key = normalize_name(record.name)
            if name_key:
                name_candidates.setdefault(name_key, []).append(record)

        self.by_unique_name = {
            key: records[0]
            for key, records in name_candidates.items()
            if len({record.gsis_id for record in records}) == 1
        }

    def resolve(
        self,
        raw_id: str,
        name: str,
    ) -> PlayerRecord | None:
        raw_id = clean(raw_id)
        if raw_id:
            if raw_id in self.by_gsis:
                return self.by_gsis[raw_id]
            if raw_id in self.by_espn:
                return self.by_espn[raw_id]
        name_key = normalize_name(name)
        if name_key:
            return self.by_unique_name.get(name_key)
        return None


class RosterProvider:
    def __init__(self, df: pd.DataFrame) -> None:
        team_col = choose_column(df, ["team", "club_code"], required=True, label="roster")
        week_col = choose_column(df, ["week"], required=True, label="roster")
        gsis_col = choose_column(df, ["gsis_id"], required=False)
        pfr_col = choose_column(df, ["pfr_id", "pfr_player_id"], required=False)
        name_col = choose_column(df, ["full_name", "display_name", "player_name"], required=False)
        position_col = choose_column(
            df,
            ["position", "depth_chart_position", "ngs_position"],
            required=False,
        )

        self.by_gsis: dict[tuple[str, int, str], PlayerRecord] = {}
        name_candidates: dict[tuple[str, int, str], list[PlayerRecord]] = {}

        for _, row in df.iterrows():
            team = normalize_team(row[team_col])
            week = parse_optional_int(row[week_col])
            if not team or week is None:
                continue
            record = PlayerRecord(
                gsis_id=clean(row[gsis_col]) if gsis_col else "",
                pfr_id=clean(row[pfr_col]) if pfr_col else "",
                name=clean(row[name_col]) if name_col else "",
                position=normalize_position(row[position_col]) if position_col else "",
            )
            if record.gsis_id:
                self.by_gsis[(team, week, record.gsis_id)] = record
            name_key = normalize_name(record.name)
            if name_key:
                name_candidates.setdefault((team, week, name_key), []).append(record)

        self.by_name = {
            key: records[0]
            for key, records in name_candidates.items()
            if len({record.gsis_id or record.pfr_id or record.name for record in records}) == 1
        }

    def resolve(
        self,
        team: str,
        week: int,
        gsis_id: str,
        name: str,
    ) -> PlayerRecord | None:
        if gsis_id:
            record = self.by_gsis.get((team, week, gsis_id))
            if record is not None:
                return record
        name_key = normalize_name(name)
        if name_key:
            return self.by_name.get((team, week, name_key))
        return None


class SnapProvider:
    def __init__(self, df: pd.DataFrame) -> None:
        team_col = choose_column(df, ["team"], required=True, label="snap counts")
        week_col = choose_column(df, ["week"], required=True, label="snap counts")
        pfr_col = choose_column(df, ["pfr_player_id", "pfr_id"], required=False)
        name_col = choose_column(df, ["player", "full_name", "player_name"], required=False)
        position_col = choose_column(df, ["position"], required=False)
        offense_col = choose_column(df, ["offense_pct"], required=True, label="snap counts")
        defense_col = choose_column(df, ["defense_pct"], required=True, label="snap counts")

        self.series: dict[tuple[str, str], list[tuple[int, float, float, str]]] = {}

        for _, row in df.iterrows():
            team = normalize_team(row[team_col])
            week = parse_optional_int(row[week_col])
            if not team or week is None:
                continue
            offense = normalize_percentage(row[offense_col])
            defense = normalize_percentage(row[defense_col])
            position = normalize_position(row[position_col]) if position_col else ""

            identities: list[str] = []
            pfr_id = clean(row[pfr_col]) if pfr_col else ""
            name = clean(row[name_col]) if name_col else ""
            if pfr_id:
                identities.append(f"pfr:{pfr_id}")
            name_key = normalize_name(name)
            if name_key:
                identities.append(f"name:{name_key}")

            for identity in identities:
                self.series.setdefault((team, identity), []).append(
                    (week, offense, defense, position)
                )

        for key in self.series:
            self.series[key].sort(key=lambda item: item[0])

    def lookup(
        self,
        team: str,
        target_week: int,
        pfr_id: str,
        name: str,
    ) -> tuple[float, float, str] | None:
        identities: list[str] = []
        if pfr_id:
            identities.append(f"pfr:{pfr_id}")
        name_key = normalize_name(name)
        if name_key:
            identities.append(f"name:{name_key}")

        for identity in identities:
            values = self.series.get((team, identity))
            if not values:
                continue
            weeks = [item[0] for item in values]
            index = bisect_left(weeks, target_week) - 1
            if index >= 0:
                _, offense, defense, position = values[index]
                return offense, defense, position
        return None

    def latest(
        self,
        team: str,
        pfr_id: str,
        name: str,
    ) -> tuple[float, float, str] | None:
        identities: list[str] = []
        if pfr_id:
            identities.append(f"pfr:{pfr_id}")
        name_key = normalize_name(name)
        if name_key:
            identities.append(f"name:{name_key}")

        for identity in identities:
            values = self.series.get((team, identity))
            if values:
                _, offense, defense, position = values[-1]
                return offense, defense, position

        for identity in identities:
            best: tuple[int, float, float, str] | None = None
            for (_series_team, series_identity), values in self.series.items():
                if series_identity != identity or not values:
                    continue
                candidate = values[-1]
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is not None:
                _, offense, defense, position = best
                return offense, defense, position
        return None


class ParticipationProvider:
    def __init__(self, df: pd.DataFrame) -> None:
        game_col = choose_column(
            df,
            ["nflverse_game_id", "game_id"],
            required=True,
            label="participation",
        )
        possession_col = choose_column(
            df,
            ["possession_team", "posteam"],
            required=True,
            label="participation",
        )
        offense_players_col = choose_column(
            df,
            ["offense_players"],
            required=True,
            label="participation",
        )
        defense_players_col = choose_column(
            df,
            ["defense_players"],
            required=True,
            label="participation",
        )

        offense_den: dict[tuple[str, int], int] = {}
        defense_den: dict[tuple[str, int], int] = {}
        offense_num: dict[tuple[str, int, str], int] = {}
        defense_num: dict[tuple[str, int, str], int] = {}

        for _, row in df.iterrows():
            parsed = parse_nflverse_game_id(row[game_col])
            if parsed is None:
                continue
            _, week, away_team, home_team = parsed
            possession = normalize_team(row[possession_col])
            if possession == away_team:
                defense_team = home_team
            elif possession == home_team:
                defense_team = away_team
            else:
                continue

            offense_players = extract_gsis_ids(row[offense_players_col])
            defense_players = extract_gsis_ids(row[defense_players_col])

            if offense_players:
                offense_den[(possession, week)] = offense_den.get((possession, week), 0) + 1
                for player_id in offense_players:
                    key = (possession, week, player_id)
                    offense_num[key] = offense_num.get(key, 0) + 1

            if defense_players:
                defense_den[(defense_team, week)] = defense_den.get((defense_team, week), 0) + 1
                for player_id in defense_players:
                    key = (defense_team, week, player_id)
                    defense_num[key] = defense_num.get(key, 0) + 1

        combined: dict[tuple[str, str, int], tuple[float, float]] = {}

        for (team, week, player_id), count in offense_num.items():
            denominator = offense_den.get((team, week), 0)
            if denominator:
                current = combined.get((team, player_id, week), (0.0, 0.0))
                combined[(team, player_id, week)] = (
                    count / denominator,
                    current[1],
                )

        for (team, week, player_id), count in defense_num.items():
            denominator = defense_den.get((team, week), 0)
            if denominator:
                current = combined.get((team, player_id, week), (0.0, 0.0))
                combined[(team, player_id, week)] = (
                    current[0],
                    count / denominator,
                )

        self.series: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
        for (team, player_id, week), (offense, defense) in combined.items():
            self.series.setdefault((team, player_id), []).append(
                (week, offense, defense)
            )
        for key in self.series:
            self.series[key].sort(key=lambda item: item[0])

    def lookup(
        self,
        team: str,
        target_week: int,
        gsis_id: str,
    ) -> tuple[float, float] | None:
        if not gsis_id:
            return None
        values = self.series.get((team, gsis_id))
        if not values:
            return None
        weeks = [item[0] for item in values]
        index = bisect_left(weeks, target_week) - 1
        if index < 0:
            return None
        _, offense, defense = values[index]
        return offense, defense

    def latest(
        self,
        team: str,
        gsis_id: str,
    ) -> tuple[float, float] | None:
        if not gsis_id:
            return None

        values = self.series.get((team, gsis_id))
        if values:
            _, offense, defense = values[-1]
            return offense, defense

        best: tuple[int, float, float] | None = None
        for (_series_team, player_id), player_values in self.series.items():
            if player_id != gsis_id or not player_values:
                continue
            candidate = player_values[-1]
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            return None
        _, offense, defense = best
        return offense, defense


class DepthProvider:
    def __init__(
        self,
        df: pd.DataFrame,
        players: PlayerCrosswalk,
    ) -> None:
        self.mode = (
            "weekly"
            if (
                "week" in df.columns
                and pd.to_numeric(df["week"], errors="coerce").notna().any()
            )
            else "timestamp"
        )
        self.weekly: dict[tuple[str, int], DepthSnapshot] = {}
        self.available_weeks: dict[str, list[int]] = {}
        self.timestamped: dict[str, list[tuple[pd.Timestamp, DepthSnapshot]]] = {}

        if self.mode == "weekly":
            self._load_weekly(df, players)
        else:
            self._load_timestamped(df, players)

    def _load_weekly(
        self,
        df: pd.DataFrame,
        players: PlayerCrosswalk,
    ) -> None:
        team_col = choose_column(df, ["club_code", "team"], required=True, label="depth charts")
        week_col = choose_column(df, ["week"], required=True, label="depth charts")
        rank_col = choose_column(df, ["depth_team", "pos_rank"], required=True, label="depth charts")
        position_col = choose_column(df, ["position", "pos_abb"], required=False)
        slot_col = choose_column(df, ["depth_position", "pos_slot", "position"], required=True, label="depth charts")
        formation_col = choose_column(df, ["formation", "pos_grp"], required=False)
        gsis_col = choose_column(df, ["gsis_id"], required=False)
        espn_col = choose_column(df, ["espn_id"], required=False)
        name_col = choose_column(df, ["full_name", "player_name"], required=False)

        grouped: dict[tuple[str, int], list[DepthPlayer]] = {}
        for _, row in df.iterrows():
            team = normalize_team(row[team_col])
            week = parse_optional_int(row[week_col])
            rank = parse_optional_int(row[rank_col])
            if not team or week is None or rank is None or rank < 1:
                continue
            raw_gsis = clean(row[gsis_col]) if gsis_col else ""
            raw_espn = clean(row[espn_col]) if espn_col else ""
            name = clean(row[name_col]) if name_col else ""
            resolved = players.resolve(raw_gsis or raw_espn, name)
            gsis_id = raw_gsis if raw_gsis.startswith("00-") else ""
            if not gsis_id and resolved is not None:
                gsis_id = resolved.gsis_id
            position = normalize_position(row[position_col]) if position_col else ""
            if not position and resolved is not None:
                position = resolved.position
            formation = clean(row[formation_col]).upper() if formation_col else ""
            slot_value = clean(row[slot_col]).upper()
            slot = f"{formation}|{slot_value}" if formation else slot_value
            grouped.setdefault((team, week), []).append(
                DepthPlayer(
                    gsis_id=gsis_id,
                    name=name,
                    position=position,
                    slot=slot,
                    rank=rank,
                )
            )

        for key, records in grouped.items():
            self.weekly[key] = build_depth_snapshot(records)
            self.available_weeks.setdefault(key[0], []).append(key[1])
        for team in self.available_weeks:
            self.available_weeks[team] = sorted(set(self.available_weeks[team]))

    def _load_timestamped(
        self,
        df: pd.DataFrame,
        players: PlayerCrosswalk,
    ) -> None:
        dt_col = choose_column(df, ["dt", "timestamp"], required=True, label="timestamp depth charts")
        team_col = choose_column(df, ["team", "club_code"], required=True, label="timestamp depth charts")
        rank_col = choose_column(df, ["pos_rank", "depth_team"], required=True, label="timestamp depth charts")
        slot_col = choose_column(df, ["pos_slot", "depth_position", "position"], required=True, label="timestamp depth charts")
        group_col = choose_column(df, ["pos_grp", "formation"], required=False)
        position_col = choose_column(df, ["pos_abb", "position"], required=False)
        gsis_col = choose_column(df, ["gsis_id"], required=False)
        espn_col = choose_column(df, ["espn_id"], required=False)
        name_col = choose_column(df, ["player_name", "full_name"], required=False)

        grouped: dict[tuple[str, pd.Timestamp], list[DepthPlayer]] = {}
        for _, row in df.iterrows():
            team = normalize_team(row[team_col])
            timestamp = parse_timestamp(row[dt_col])
            rank = parse_optional_int(row[rank_col])
            if not team or timestamp is None or rank is None or rank < 1:
                continue
            raw_gsis = clean(row[gsis_col]) if gsis_col else ""
            raw_espn = clean(row[espn_col]) if espn_col else ""
            name = clean(row[name_col]) if name_col else ""
            resolved = players.resolve(raw_gsis or raw_espn, name)
            gsis_id = raw_gsis if raw_gsis.startswith("00-") else ""
            if not gsis_id and resolved is not None:
                gsis_id = resolved.gsis_id
            position = normalize_position(row[position_col]) if position_col else ""
            if not position and resolved is not None:
                position = resolved.position
            group = clean(row[group_col]).upper() if group_col else ""
            slot_value = clean(row[slot_col]).upper()
            slot = f"{group}|{slot_value}|{position}" if group else f"{slot_value}|{position}"
            grouped.setdefault((team, timestamp), []).append(
                DepthPlayer(
                    gsis_id=gsis_id,
                    name=name,
                    position=position,
                    slot=slot,
                    rank=rank,
                )
            )

        for (team, timestamp), records in grouped.items():
            self.timestamped.setdefault(team, []).append(
                (timestamp, build_depth_snapshot(records))
            )
        for team in self.timestamped:
            self.timestamped[team].sort(key=lambda item: item[0])

    def get(
        self,
        team: str,
        week: int,
        kickoff: pd.Timestamp,
    ) -> DepthSnapshot:
        if self.mode == "weekly":
            exact = self.weekly.get((team, week))
            if exact is not None:
                return exact
            weeks = self.available_weeks.get(team, [])
            index = bisect_left(weeks, week + 1) - 1
            if index >= 0:
                return self.weekly[(team, weeks[index])]
            return DepthSnapshot.empty()

        values = self.timestamped.get(team, [])
        if not values:
            return DepthSnapshot.empty()
        timestamps = [item[0] for item in values]
        index = bisect_left(timestamps, kickoff) - 1
        if index < 0:
            return DepthSnapshot.empty()
        return values[index][1]


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in BLANK_VALUES:
        return ""
    return text


def normalize_team(value: Any) -> str:
    team = clean(value).upper()
    return TEAM_ALIASES.get(team, team)


def normalize_position(value: Any) -> str:
    position = clean(value).upper().replace(" ", "")
    aliases = {
        "CORNERBACK": "CB",
        "SAFETY": "S",
        "DEFENSIVEBACK": "DB",
        "DEFENSIVEEND": "DE",
        "DEFENSIVETACKLE": "DT",
        "LINEBACKER": "LB",
        "OFFENSIVELINE": "OL",
        "OFFENSIVETACKLE": "OT",
        "OFFENSIVEGUARD": "G",
        "RUNNINGBACK": "RB",
        "WIDERECEIVER": "WR",
        "TIGHTEND": "TE",
        "QUARTERBACK": "QB",
    }
    return aliases.get(position, position)


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def parse_optional_int(value: Any) -> int | None:
    text = clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number) or abs(number - round(number)) > 1e-9:
        return None
    return int(round(number))


def parse_required_int(value: Any, label: str) -> int:
    number = parse_optional_int(value)
    if number is None:
        raise ValueError(f"{label}: invalid integer value {clean(value)!r}")
    return number


def parse_timestamp(value: Any) -> pd.Timestamp | None:
    text = clean(value)
    if not text:
        return None
    try:
        timestamp = pd.to_datetime(text, utc=True, errors="raise")
    except Exception:
        return None
    if isinstance(timestamp, pd.DatetimeIndex):
        return None
    return pd.Timestamp(timestamp)


def normalize_percentage(value: Any) -> float:
    text = clean(value).replace("%", "")
    if not text:
        return 0.0
    try:
        number = float(text)
    except ValueError:
        return 0.0
    if not math.isfinite(number) or number < 0:
        return 0.0
    if number > 1.5:
        number /= 100.0
    return max(0.0, number)


def choose_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool,
    label: str = "source",
) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        raise ValueError(
            f"{label}: none of the required columns are present: {candidates}"
        )
    return None


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")


def read_parquet(path: Path) -> pd.DataFrame:
    require_file(path)
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Could not read parquet file {path}: {exc}") from exc


def read_training(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
        low_memory=False,
    )
    missing = [column for column in TRAINING_REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    if len(df.columns) != len(set(df.columns)):
        raise ValueError(f"{path}: duplicate column names")
    return df


def parse_nflverse_game_id(value: Any) -> tuple[int, int, str, str] | None:
    match = GAME_ID_PATTERN.match(clean(value))
    if not match:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        normalize_team(match.group(3)),
        normalize_team(match.group(4)),
    )


def extract_gsis_ids(value: Any) -> set[str]:
    return set(GSIS_PATTERN.findall(clean(value)))


def build_depth_snapshot(records: list[DepthPlayer]) -> DepthSnapshot:
    starters: dict[str, set[str]] = {}
    rank_by_id: dict[str, int] = {}
    rank_by_name: dict[str, int] = {}
    position_by_id: dict[str, str] = {}
    position_by_name: dict[str, str] = {}

    for record in records:
        identity = record.gsis_id or f"name:{normalize_name(record.name)}"
        if not identity or identity == "name:":
            continue
        if record.rank == 1 and record.slot:
            starters.setdefault(record.slot, set()).add(identity)
        if record.gsis_id:
            previous = rank_by_id.get(record.gsis_id)
            if previous is None or record.rank < previous:
                rank_by_id[record.gsis_id] = record.rank
                if record.position:
                    position_by_id[record.gsis_id] = record.position
        name_key = normalize_name(record.name)
        if name_key:
            previous = rank_by_name.get(name_key)
            if previous is None or record.rank < previous:
                rank_by_name[name_key] = record.rank
                if record.position:
                    position_by_name[name_key] = record.position

    return DepthSnapshot(
        players=records,
        starters_by_slot={key: frozenset(value) for key, value in starters.items()},
        rank_by_id=rank_by_id,
        rank_by_name=rank_by_name,
        position_by_id=position_by_id,
        position_by_name=position_by_name,
    )


def load_injuries(
    path: Path,
    players: PlayerCrosswalk,
) -> tuple[dict[tuple[str, int], list[InjuryRecord]], list[int]]:
    df = read_parquet(path)
    team_col = choose_column(df, ["team", "club_code"], required=True, label=str(path))
    week_col = choose_column(df, ["week"], required=True, label=str(path))
    status_col = choose_column(
        df,
        ["report_status", "game_status", "status"],
        required=True,
        label=str(path),
    )
    gsis_col = choose_column(df, ["gsis_id"], required=False)
    player_id_col = choose_column(df, ["player_id", "espn_id"], required=False)
    name_col = choose_column(df, ["full_name", "player_name", "display_name"], required=False)
    position_col = choose_column(df, ["position"], required=False)
    modified_col = choose_column(df, ["date_modified", "report_date", "dt"], required=False)

    latest: dict[tuple[str, int, str], tuple[pd.Timestamp | None, int, InjuryRecord]] = {}
    source_weeks: set[int] = set()

    for source_index, row in df.iterrows():
        team = normalize_team(row[team_col])
        week = parse_optional_int(row[week_col])
        if not team or week is None:
            continue
        source_weeks.add(week)

        status = normalize_injury_status(row[status_col])
        if status not in {"out", "doubtful", "questionable"}:
            continue

        raw_gsis = clean(row[gsis_col]) if gsis_col else ""
        raw_player_id = clean(row[player_id_col]) if player_id_col else ""
        name = clean(row[name_col]) if name_col else ""
        resolved = players.resolve(raw_gsis or raw_player_id, name)
        gsis_id = raw_gsis if raw_gsis.startswith("00-") else ""
        if not gsis_id and resolved is not None:
            gsis_id = resolved.gsis_id
        position = normalize_position(row[position_col]) if position_col else ""
        if not position and resolved is not None:
            position = resolved.position

        identity = gsis_id or raw_player_id or normalize_name(name)
        if not identity:
            continue

        modified = parse_timestamp(row[modified_col]) if modified_col else None
        record = InjuryRecord(
            gsis_id=gsis_id,
            raw_player_id=raw_player_id,
            name=name,
            position=position,
            status=status,
        )
        key = (team, week, identity)
        existing = latest.get(key)
        candidate_sort = modified if modified is not None else pd.Timestamp.min.tz_localize("UTC")
        if existing is None:
            latest[key] = (modified, int(source_index), record)
        else:
            existing_modified, existing_index, _ = existing
            existing_sort = (
                existing_modified
                if existing_modified is not None
                else pd.Timestamp.min.tz_localize("UTC")
            )
            if candidate_sort > existing_sort or (
                candidate_sort == existing_sort and int(source_index) >= existing_index
            ):
                latest[key] = (modified, int(source_index), record)

    output: dict[tuple[str, int], list[InjuryRecord]] = {}
    for (team, week, _), (_, _, record) in latest.items():
        output.setdefault((team, week), []).append(record)
    return output, sorted(source_weeks)


def resolve_injury_week(
    injuries: dict[tuple[str, int], list[InjuryRecord]],
    source_weeks: list[int],
    team: str,
    target_week: int,
) -> tuple[list[InjuryRecord], int]:
    if target_week in source_weeks:
        return injuries.get((team, target_week), []), target_week

    index = bisect_left(source_weeks, target_week) - 1
    if index < 0:
        raise RuntimeError(
            f"No injury source week exists before target week {target_week} for {team}"
        )

    source_week = source_weeks[index]
    return injuries.get((team, source_week), []), source_week


def normalize_injury_status(value: Any) -> str:
    status = clean(value).casefold().replace("-", " ").replace("_", " ")
    status = " ".join(status.split())
    if status in {"o", "out"} or status.startswith("out "):
        return "out"
    if status in {"d", "doubtful"} or "doubt" in status:
        return "doubtful"
    if status in {"q", "questionable"} or "question" in status:
        return "questionable"
    return "other"


def kickoff_timestamp(row: pd.Series) -> pd.Timestamp:
    if "commence_time" in row.index:
        parsed = parse_timestamp(row["commence_time"])
        if parsed is not None:
            return parsed
    gameday = clean(row["gameday"])
    try:
        # Midnight UTC deliberately excludes same-day timestamp snapshots when
        # an exact kickoff timestamp is unavailable, preventing post-kickoff leakage.
        return pd.Timestamp(gameday, tz="UTC")
    except Exception as exc:
        raise ValueError(f"Invalid gameday {gameday!r}") from exc


def resolve_injury_player(
    injury: InjuryRecord,
    team: str,
    week: int,
    depth: DepthSnapshot,
    players: PlayerCrosswalk,
    roster: RosterProvider,
) -> PlayerRecord:
    player_record = players.resolve(injury.gsis_id or injury.raw_player_id, injury.name)
    roster_record = roster.resolve(team, week, injury.gsis_id, injury.name)

    gsis_id = injury.gsis_id
    if not gsis_id and roster_record is not None:
        gsis_id = roster_record.gsis_id
    if not gsis_id and player_record is not None:
        gsis_id = player_record.gsis_id

    pfr_id = ""
    if roster_record is not None and roster_record.pfr_id:
        pfr_id = roster_record.pfr_id
    elif player_record is not None:
        pfr_id = player_record.pfr_id

    position = injury.position
    if not position:
        position = depth.position_for(gsis_id, injury.name)
    if not position and roster_record is not None:
        position = roster_record.position
    if not position and player_record is not None:
        position = player_record.position

    name = injury.name
    if not name and roster_record is not None:
        name = roster_record.name
    if not name and player_record is not None:
        name = player_record.name

    return PlayerRecord(
        gsis_id=gsis_id,
        pfr_id=pfr_id,
        name=name,
        position=normalize_position(position),
    )


def starter_change_count(
    previous: DepthSnapshot | None,
    current: DepthSnapshot,
) -> int:
    if previous is None or not previous.starters_by_slot or not current.starters_by_slot:
        return 0
    slots = set(previous.starters_by_slot) | set(current.starters_by_slot)
    return sum(
        1
        for slot in slots
        if previous.starters_by_slot.get(slot, frozenset())
        != current.starters_by_slot.get(slot, frozenset())
    )


def compute_team_features(
    team: str,
    week: int,
    injuries: list[InjuryRecord],
    current_depth: DepthSnapshot,
    previous_depth: DepthSnapshot | None,
    players: PlayerCrosswalk,
    roster: RosterProvider,
    snaps: SnapProvider,
    participation: ParticipationProvider,
    prior_snaps: SnapProvider | None = None,
    prior_participation: ParticipationProvider | None = None,
) -> dict[str, float]:
    values = {feature: 0.0 for feature in BASE_FEATURES}
    values["depth_starter_changes"] = float(
        starter_change_count(previous_depth, current_depth)
    )

    for injury in injuries:
        if injury.status == "out":
            values["inj_out_count"] += 1.0
        elif injury.status == "doubtful":
            values["inj_doubtful_count"] += 1.0
        elif injury.status == "questionable":
            values["inj_questionable_count"] += 1.0

        resolved = resolve_injury_player(
            injury,
            team,
            week,
            current_depth,
            players,
            roster,
        )
        current_rank = current_depth.rank_for(resolved.gsis_id, resolved.name)
        previous_rank = (
            previous_depth.rank_for(resolved.gsis_id, resolved.name)
            if previous_depth is not None
            else None
        )
        position = resolved.position or current_depth.position_for(
            resolved.gsis_id,
            resolved.name,
        )
        if not position and previous_depth is not None:
            position = previous_depth.position_for(
                resolved.gsis_id,
                resolved.name,
            )

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

        snap_value = snaps.lookup(
            team,
            week,
            resolved.pfr_id,
            resolved.name,
        )
        if snap_value is None and week == 1 and prior_snaps is not None:
            snap_value = prior_snaps.latest(
                team,
                resolved.pfr_id,
                resolved.name,
            )

        if snap_value is not None:
            offense_share, defense_share, _ = snap_value
        else:
            participation_value = participation.lookup(
                team,
                week,
                resolved.gsis_id,
            )
            if (
                participation_value is None
                and week == 1
                and prior_participation is not None
            ):
                participation_value = prior_participation.latest(
                    team,
                    resolved.gsis_id,
                )

            if participation_value is None:
                offense_share, defense_share = 0.0, 0.0
            else:
                offense_share, defense_share = participation_value

        values["inj_offense_unavailable_snap_share"] += offense_share
        values["inj_defense_unavailable_snap_share"] += defense_share

    return values


def format_value(feature: str, value: float) -> str:
    if feature in COUNT_FEATURES:
        return str(int(round(value)))
    if abs(value) < 0.0000005:
        value = 0.0
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def load_season_sources(
    season: int,
    players: PlayerCrosswalk,
) -> tuple[
    dict[tuple[str, int], list[InjuryRecord]],
    list[int],
    DepthProvider,
    RosterProvider,
    SnapProvider,
    ParticipationProvider,
    SnapProvider,
    ParticipationProvider,
]:
    injury_path = INJURY_DIR / f"injuries_{season}.parquet"
    depth_path = DEPTH_DIR / f"depth_charts_{season}.parquet"
    roster_path = ROSTER_DIR / f"roster_weekly_{season}.parquet"
    snap_path = SNAP_DIR / f"snap_counts_{season}.parquet"
    participation_path = PARTICIPATION_DIR / f"pbp_participation_{season}.parquet"

    prior_season = season - 1
    prior_snap_path = SNAP_DIR / f"snap_counts_{prior_season}.parquet"
    prior_participation_path = (
        PARTICIPATION_DIR / f"pbp_participation_{prior_season}.parquet"
    )

    injuries, injury_source_weeks = load_injuries(injury_path, players)
    depth = DepthProvider(read_parquet(depth_path), players)
    roster = RosterProvider(read_parquet(roster_path))
    snaps = SnapProvider(read_parquet(snap_path))
    participation = ParticipationProvider(read_parquet(participation_path))
    prior_snaps = SnapProvider(read_parquet(prior_snap_path))
    prior_participation = ParticipationProvider(
        read_parquet(prior_participation_path)
    )

    return (
        injuries,
        injury_source_weeks,
        depth,
        roster,
        snaps,
        participation,
        prior_snaps,
        prior_participation,
    )


def process_season(
    season: int,
    players: PlayerCrosswalk,
) -> tuple[pd.DataFrame, dict[str, int]]:
    path = TRAINING_PATHS[season]
    df = read_training(path)
    original_rows = len(df)

    for column in GENERATED_COLUMNS:
        if column in df.columns:
            df = df.drop(columns=[column])

    (
        injuries,
        injury_source_weeks,
        depth,
        roster,
        snaps,
        participation,
        prior_snaps,
        prior_participation,
    ) = load_season_sources(
        season,
        players,
    )

    row_features: dict[int, dict[str, float]] = {}
    previous_depth: dict[str, DepthSnapshot] = {}
    depth_team_games = 0
    injury_team_games = 0
    out_players = 0

    ordering = []
    for index, row in df.iterrows():
        row_season = parse_required_int(
            row["season"],
            f"{path} row {index + 2}: season",
        )
        if row_season != season:
            raise ValueError(
                f"{path} row {index + 2}: expected season {season}, found {row_season}"
            )
        week = parse_required_int(row["week"], f"{path} row {index + 2}: week")
        kickoff = kickoff_timestamp(row)
        ordering.append((week, kickoff, int(index)))

    ordering.sort(key=lambda item: (item[0], item[1], item[2]))

    for _, _, index in ordering:
        row = df.loc[index]
        week = parse_required_int(row["week"], f"{path} row {index + 2}: week")
        kickoff = kickoff_timestamp(row)
        home_team = normalize_team(row["home_team"])
        away_team = normalize_team(row["away_team"])
        if not home_team or not away_team:
            raise ValueError(f"{path} row {index + 2}: blank home/away team")

        home_depth = depth.get(home_team, week, kickoff)
        away_depth = depth.get(away_team, week, kickoff)
        if not home_depth.players:
            raise RuntimeError(
                f"{path} row {index + 2}: no pregame depth-chart match for {home_team}, week {week}"
            )
        if not away_depth.players:
            raise RuntimeError(
                f"{path} row {index + 2}: no pregame depth-chart match for {away_team}, week {week}"
            )
        depth_team_games += 2

        home_injuries, _home_injury_source_week = resolve_injury_week(
            injuries,
            injury_source_weeks,
            home_team,
            week,
        )
        away_injuries, _away_injury_source_week = resolve_injury_week(
            injuries,
            injury_source_weeks,
            away_team,
            week,
        )
        if home_injuries:
            injury_team_games += 1
        if away_injuries:
            injury_team_games += 1
        out_players += sum(1 for item in home_injuries + away_injuries if item.status == "out")

        home_values = compute_team_features(
            home_team,
            week,
            home_injuries,
            home_depth,
            previous_depth.get(home_team),
            players,
            roster,
            snaps,
            participation,
            prior_snaps,
            prior_participation,
        )
        away_values = compute_team_features(
            away_team,
            week,
            away_injuries,
            away_depth,
            previous_depth.get(away_team),
            players,
            roster,
            snaps,
            participation,
            prior_snaps,
            prior_participation,
        )

        generated: dict[str, float] = {}
        for feature in BASE_FEATURES:
            generated[f"home_{feature}"] = home_values[feature]
            generated[f"away_{feature}"] = away_values[feature]
            generated[f"{feature}_diff"] = home_values[feature] - away_values[feature]
        row_features[index] = generated

        previous_depth[home_team] = home_depth
        previous_depth[away_team] = away_depth

    for column in GENERATED_COLUMNS:
        feature = column
        if column.startswith("home_"):
            feature = column[5:]
        elif column.startswith("away_"):
            feature = column[5:]
        elif column.endswith("_diff"):
            feature = column[:-5]
        df[column] = [
            format_value(feature, row_features[int(index)][column])
            for index in df.index
        ]

    if len(df) != original_rows:
        raise RuntimeError(f"{season}: row count changed during Step 10")

    for column in GENERATED_COLUMNS:
        if df[column].astype(str).str.strip().eq("").any():
            raise RuntimeError(f"{season}: blank values found in {column}")

    for feature in BASE_FEATURES:
        home = pd.to_numeric(df[f"home_{feature}"], errors="raise")
        away = pd.to_numeric(df[f"away_{feature}"], errors="raise")
        diff = pd.to_numeric(df[f"{feature}_diff"], errors="raise")
        if not ((home - away - diff).abs() <= 1e-6).all():
            raise RuntimeError(f"{season}: invalid diff arithmetic for {feature}")
        if (home < 0).any() or (away < 0).any():
            raise RuntimeError(f"{season}: negative values found for {feature}")

    if not df["home_inj_qb1_out"].isin(["0", "1"]).all():
        raise RuntimeError(f"{season}: home_inj_qb1_out contains non-binary values")
    if not df["away_inj_qb1_out"].isin(["0", "1"]).all():
        raise RuntimeError(f"{season}: away_inj_qb1_out contains non-binary values")

    return (
        df,
        {
            "rows": original_rows,
            "depth_team_games": depth_team_games,
            "injury_team_games": injury_team_games,
            "out_players": out_players,
        },
    )


def write_outputs(outputs: dict[int, pd.DataFrame]) -> None:
    temp_paths: dict[int, Path] = {}
    try:
        for season in SEASONS:
            output_path = TRAINING_PATHS[season]
            temp_path = output_path.with_suffix(".step10.tmp.csv")
            temp_paths[season] = temp_path
            outputs[season].to_csv(
                temp_path,
                index=False,
                encoding="utf-8",
                lineterminator="\n",
            )

        for season in SEASONS:
            temp_paths[season].replace(TRAINING_PATHS[season])

    except Exception:
        for temp_path in temp_paths.values():
            if temp_path.exists():
                temp_path.unlink()
        raise


def main() -> int:
    players = PlayerCrosswalk(read_parquet(PLAYERS_PATH))

    outputs: dict[int, pd.DataFrame] = {}
    results: dict[int, dict[str, int]] = {}

    for season in SEASONS:
        outputs[season], results[season] = process_season(season, players)

    write_outputs(outputs)

    print("Step 10 complete.")
    print(f"Added/rebuilt {len(GENERATED_COLUMNS)} depth/injury columns.")
    for season in SEASONS:
        stats = results[season]
        print(
            f"{season}: "
            f"rows={stats['rows']}, "
            f"depth_team_games={stats['depth_team_games']}, "
            f"injury_team_games={stats['injury_team_games']}, "
            f"out_players={stats['out_players']}"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

#!/usr/bin/env python3
"""
pull_final_scores.py

Pull final CFB scores/status from ESPN for weeks that already have CFB picks.

Primary input:
    docs/win/football/cfb/00_intake/schedule/{season}_schedule.csv

Existing pick files are used to determine which weeks should be checked:
    docs/win/football/cfb/03_picks/week_*_CFB_picks.csv

ESPN source:
    https://site.api.espn.com/apis/site/v2/sports/football/
    college-football/summary?event={game_id}

Output:
    docs/win/football/cfb/04_final_results/results/
        {season}_{season_type}_{week}.csv

Error/run log:
    docs/win/football/cfb/errors/04_final_results/
        pull_final_scores.txt

Behavior:
- CFB only.
- Uses schedule game_id as the authoritative join key.
- By default checks only weeks that already have CFB picks.
- --week can restrict the pull to one specific week.
- Writes both completed and not-yet-final games.
- completed=1 means the game is safe for grading.
- fetch_error records ESPN lookup problems without destroying other results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "cfb-final-scores-v2-2026-08-26"

SCRIPT_DIR = Path(__file__).resolve().parent
CFB_ROOT = SCRIPT_DIR.parents[1]

RESULTS_DIR = CFB_ROOT / "04_final_results" / "results"
PICKS_DIR = CFB_ROOT / "03_picks"
ERROR_LOG_PATH = (
    CFB_ROOT
    / "errors"
    / "04_final_results"
    / "pull_final_scores.txt"
)

SUMMARY_URL_TEMPLATE = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/football/college-football/"
    "summary?event={game_id}"
)

DEFAULT_WORKERS = 6

PICKS_FILE_RE = re.compile(
    r"^week_(\d+)_CFB_picks\.csv$"
)

OUTPUT_HEADER = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "game_time",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "status",
    "completed",
    "fetch_error",
    "last_checked_utc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull final CFB scores from ESPN."
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season. Defaults to CFB_SEASON, then 2026.",
    )

    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help=(
            "Optional specific week. "
            "If omitted, weeks with existing CFB picks are checked."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent ESPN requests. Default: {DEFAULT_WORKERS}.",
    )

    return parser.parse_args()


def get_season(cli_season: int | None) -> int:
    if cli_season is not None:
        return int(cli_season)

    env = os.getenv(
        "CFB_SEASON",
        "",
    ).strip()

    if env:
        return int(env)

    return 2026


def clean(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.casefold() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }:
        return ""

    return text


def normalize_game_id(value: Any) -> str:
    text = clean(value)

    if re.fullmatch(
        r"\d+\.0",
        text,
    ):
        return text[:-2]

    return text


def log(lines: list[str]) -> None:
    ERROR_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    with ERROR_LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            f"--- run {timestamp} ---\n"
        )

        for line in lines:
            handle.write(
                str(line) + "\n"
            )

        handle.write("\n")


def fetch_json(
    url: str,
    timeout: int = 15,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(compatible; CFB-Pipeline/1.0)"
            ),
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        return json.loads(
            response.read().decode(
                "utf-8"
            )
        )


def extract_score(
    competitor: dict[str, Any],
) -> str:
    score = competitor.get(
        "score",
        "",
    )

    if isinstance(
        score,
        dict,
    ):
        value = score.get(
            "displayValue",
            score.get(
                "value",
                "",
            ),
        )

        return clean(value)

    return clean(score)


def get_score_and_status(
    game_id: str,
) -> dict[str, Any]:
    """
    Resolve one ESPN CFB event.

    Returns:
        away_score
        home_score
        completed
        status
        fetch_error
    """

    url = SUMMARY_URL_TEMPLATE.format(
        game_id=game_id
    )

    try:
        payload = fetch_json(
            url
        )

    except Exception as exc:
        return {
            "away_score": "",
            "home_score": "",
            "completed": 0,
            "status": "",
            "fetch_error": (
                "failed to fetch ESPN summary: "
                f"{exc}"
            ),
        }

    header = payload.get(
        "header",
        {},
    )

    competitions = header.get(
        "competitions",
        [],
    )

    if not competitions:
        return {
            "away_score": "",
            "home_score": "",
            "completed": 0,
            "status": "",
            "fetch_error": (
                "ESPN summary contained "
                "no header competition"
            ),
        }

    competition = competitions[0]

    status_type = (
        competition.get(
            "status",
            {},
        )
        .get(
            "type",
            {},
        )
    )

    completed = bool(
        status_type.get(
            "completed",
            False,
        )
    )

    status_text = clean(
        status_type.get(
            "description",
            status_type.get(
                "detail",
                "",
            ),
        )
    )

    away_score = ""
    home_score = ""

    for competitor in competition.get(
        "competitors",
        [],
    ):
        home_away = clean(
            competitor.get(
                "homeAway",
                "",
            )
        ).casefold()

        score = extract_score(
            competitor
        )

        if home_away == "away":
            away_score = score

        elif home_away == "home":
            home_score = score

    return {
        "away_score": away_score,
        "home_score": home_score,
        "completed": int(completed),
        "status": status_text,
        "fetch_error": "",
    }


def discover_pick_weeks(
    season: int,
) -> set[int]:
    weeks: set[int] = set()

    if not PICKS_DIR.is_dir():
        return weeks

    for path in PICKS_DIR.glob(
        "week_*_CFB_picks.csv"
    ):
        match = PICKS_FILE_RE.fullmatch(
            path.name
        )

        if match is None:
            continue

        week = int(
            match.group(1)
        )

        try:
            with path.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as handle:
                reader = csv.DictReader(
                    handle
                )

                first = next(
                    reader,
                    None,
                )

        except Exception:
            continue

        if first is None:
            continue

        try:
            file_season = int(
                float(
                    clean(
                        first.get(
                            "season",
                            "",
                        )
                    )
                )
            )

        except Exception:
            continue

        if file_season == season:
            weeks.add(
                week
            )

    return weeks


def read_schedule(
    season: int,
    requested_week: int | None,
) -> list[dict[str, str]]:
    schedule_path = (
        CFB_ROOT
        / "00_intake"
        / "schedule"
        / f"{season}_schedule.csv"
    )

    if not schedule_path.is_file():
        raise FileNotFoundError(
            "Missing CFB schedule: "
            f"{schedule_path}"
        )

    with schedule_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        required = [
            "season",
            "season_type",
            "week",
            "game_id",
            "game_date",
            "game_time",
            "away_team",
            "home_team",
        ]

        fieldnames = (
            reader.fieldnames
            or []
        )

        missing = [
            column
            for column
            in required
            if column not in fieldnames
        ]

        if missing:
            raise ValueError(
                "CFB schedule missing "
                f"required columns: {missing}"
            )

        rows = list(
            reader
        )

    if requested_week is not None:
        target_weeks = {
            int(
                requested_week
            )
        }

    else:
        target_weeks = discover_pick_weeks(
            season
        )

        if not target_weeks:
            print(
                "No existing CFB pick weeks found; "
                "nothing to pull."
            )

            return []

    filtered: list[
        dict[str, str]
    ] = []

    seen_game_ids: set[str] = set()

    for row in rows:
        try:
            row_season = int(
                float(
                    clean(
                        row.get(
                            "season",
                            "",
                        )
                    )
                )
            )

            row_week = int(
                float(
                    clean(
                        row.get(
                            "week",
                            "",
                        )
                    )
                )
            )

        except Exception:
            continue

        if row_season != season:
            continue

        if row_week not in target_weeks:
            continue

        game_id = normalize_game_id(
            row.get(
                "game_id",
                "",
            )
        )

        if not game_id:
            continue

        if game_id in seen_game_ids:
            continue

        seen_game_ids.add(
            game_id
        )

        row = dict(
            row
        )

        row[
            "game_id"
        ] = game_id

        filtered.append(
            row
        )

    return filtered


def pull_scores(
    schedule_rows: list[
        dict[str, str]
    ],
    workers: int,
) -> dict[str, dict[str, Any]]:
    if workers < 1:
        raise ValueError(
            "--workers must be at least 1"
        )

    results: dict[
        str,
        dict[str, Any]
    ] = {}

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        futures = {
            executor.submit(
                get_score_and_status,
                row["game_id"],
            ): row["game_id"]
            for row in schedule_rows
        }

        for future in as_completed(
            futures
        ):
            game_id = futures[
                future
            ]

            try:
                results[
                    game_id
                ] = future.result()

            except Exception as exc:
                results[
                    game_id
                ] = {
                    "away_score": "",
                    "home_score": "",
                    "completed": 0,
                    "status": "",
                    "fetch_error": (
                        "unexpected worker error: "
                        f"{exc}"
                    ),
                }

    return results


def atomic_write_csv(
    path: Path,
    rows: list[
        dict[str, Any]
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_HEADER,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: row.get(
                        column,
                        "",
                    )
                    for column
                    in OUTPUT_HEADER
                }
            )

    os.replace(
        temporary,
        path,
    )


def main() -> int:
    args = parse_args()

    season = get_season(
        args.season
    )

    print(
        "pull_final_scores.py "
        f"version={SCRIPT_VERSION}"
    )

    print(
        f"season={season}"
    )

    if args.week is not None:
        print(
            f"week={args.week}"
        )

    schedule_rows = read_schedule(
        season,
        args.week,
    )

    if not schedule_rows:
        return 0

    checked_utc = datetime.now(
        timezone.utc
    ).isoformat()

    score_results = pull_scores(
        schedule_rows,
        args.workers,
    )

    rows_by_week: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(
        list
    )

    completed_count = 0
    not_final_count = 0
    failed_count = 0

    log_lines = [
        f"version={SCRIPT_VERSION}",
        f"season={season}",
    ]

    for row in schedule_rows:
        game_id = row[
            "game_id"
        ]

        score = score_results.get(
            game_id,
            {
                "away_score": "",
                "home_score": "",
                "completed": 0,
                "status": "",
                "fetch_error": (
                    "missing worker result"
                ),
            },
        )

        completed = int(
            score.get(
                "completed",
                0,
            )
        )

        fetch_error = clean(
            score.get(
                "fetch_error",
                "",
            )
        )

        if fetch_error:
            failed_count += 1

            log_lines.append(
                f"game_id={game_id} "
                f"error={fetch_error}"
            )

        elif completed:
            completed_count += 1

        else:
            not_final_count += 1

        out_row = {
            "season": clean(
                row.get(
                    "season",
                    "",
                )
            ),
            "season_type": clean(
                row.get(
                    "season_type",
                    "",
                )
            ),
            "week": clean(
                row.get(
                    "week",
                    "",
                )
            ),
            "game_id": game_id,
            "game_date": clean(
                row.get(
                    "game_date",
                    "",
                )
            ),
            "game_time": clean(
                row.get(
                    "game_time",
                    "",
                )
            ),
            "away_team": clean(
                row.get(
                    "away_team",
                    "",
                )
            ),
            "home_team": clean(
                row.get(
                    "home_team",
                    "",
                )
            ),
            "away_score": clean(
                score.get(
                    "away_score",
                    "",
                )
            ),
            "home_score": clean(
                score.get(
                    "home_score",
                    "",
                )
            ),
            "status": clean(
                score.get(
                    "status",
                    "",
                )
            ),
            "completed": completed,
            "fetch_error": (
                fetch_error
            ),
            "last_checked_utc": (
                checked_utc
            ),
        }

        key = (
            out_row["season"],
            out_row["season_type"],
            out_row["week"],
        )

        rows_by_week[
            key
        ].append(
            out_row
        )

    files_written = 0

    for (
        season_value,
        season_type,
        week,
    ), rows in sorted(
        rows_by_week.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            int(
                float(
                    item[0][2]
                )
            ),
        ),
    ):
        if (
            not season_value
            or not season_type
            or not week
        ):
            continue

        output_path = (
            RESULTS_DIR
            / (
                f"{season_value}_"
                f"{season_type}_"
                f"{week}.csv"
            )
        )

        rows.sort(
            key=lambda row: (
                row.get(
                    "game_date",
                    "",
                ),
                row.get(
                    "game_time",
                    "",
                ),
                row.get(
                    "game_id",
                    "",
                ),
            )
        )

        atomic_write_csv(
            output_path,
            rows,
        )

        files_written += 1

        print(
            f"WROTE {output_path} "
            f"| games={len(rows)}"
        )

        log_lines.append(
            f"wrote {len(rows)} rows "
            f"to {output_path}"
        )

    summary = (
        f"games_processed={len(schedule_rows)} "
        f"completed={completed_count} "
        f"not_final={not_final_count} "
        f"failed={failed_count} "
        f"files_written={files_written}"
    )

    print(
        summary
    )

    log_lines.append(
        summary
    )

    log(
        log_lines
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except Exception as exc:
        message = (
            f"ERROR: {exc}"
        )

        print(
            message,
            file=sys.stderr,
        )

        log(
            [
                message,
            ]
        )

        raise
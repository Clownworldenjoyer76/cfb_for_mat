"""
pull_league_leaders.py

Pulls ESPN's league-wide statistical leaders (top 25 per category) for
the NFL regular season.

Source:
    https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/types/{season_type}/leaders

Output:
    docs/win/football/nfl/data/league_leaders/league_leaders_{season}.csv
"""

import csv
import json
import os
import urllib.request

SEASON = 2026
SEASON_TYPE = 2  # Regular Season

LEADERS_URL = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{SEASON}/types/{SEASON_TYPE}/leaders"

OUTPUT_PATH = f"docs/win/football/nfl/data/league_leaders/league_leaders_{SEASON}.csv"

OUTPUT_HEADER = [
    "season",
    "category",
    "rank",
    "athlete_id",
    "team_id",
    "value",
    "displayValue",
]


def fetch_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def extract_id(ref_url, segment):
    if not ref_url:
        return ""
    parts = ref_url.split(f"/{segment}/")
    if len(parts) < 2:
        return ""
    return parts[1].split("?")[0]


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    try:
        data = fetch_json(LEADERS_URL)
    except Exception as e:
        print(f"failed to pull leaders: {e}")
        return

    rows = []

    for category in data.get("categories", []):
        category_name = category.get("name", "")

        for rank, leader in enumerate(category.get("leaders", []), start=1):
            athlete_ref = leader.get("athlete", {}).get("$ref", "")
            team_ref = leader.get("team", {}).get("$ref", "")

            rows.append({
                "season": SEASON,
                "category": category_name,
                "rank": rank,
                "athlete_id": extract_id(athlete_ref, "athletes"),
                "team_id": extract_id(team_ref, "teams"),
                "value": leader.get("value", ""),
                "displayValue": leader.get("displayValue", ""),
            })

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows={len(rows)} output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()

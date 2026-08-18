#!/usr/bin/env python3
"""
pull_injuries.py

Pulls NFL injury report data from ESPN's site API and writes
docs/win/football/nfl/00_intake/injuries/{season}_injuries.csv

Endpoint used:
    https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries

Output columns:
    season, team, player_id, player_name, position, game_status, report_date

Manual run only.
"""

import csv
import json
import os
import re
import urllib.request

INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
OUTPUT_DIR = "docs/win/football/nfl/00_intake/injuries"

OUTPUT_HEADERS = [
    "season",
    "team",
    "player_id",
    "player_name",
    "position",
    "game_status",
    "report_date",
]


def fetch_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def extract_player_id(athlete):
    """
    Extracts the ESPN player id from the 'playercard' link href,
    since the athlete object in this endpoint has no direct id field.
    Example href: https://www.espn.com/nfl/player/_/id/2578570/jacoby-brissett
    """
    for link in athlete.get("links", []):
        rel = link.get("rel", [])
        if "playercard" in rel:
            match = re.search(r"/id/(\d+)/", link.get("href", ""))
            if match:
                return match.group(1)
    return ""


def main():
    data = fetch_json(INJURIES_URL)

    season_year = data.get("season", {}).get("year", "")

    rows = []
    for team_entry in data.get("injuries", []):
        team_abbr = team_entry.get("displayName", "")
        for injury in team_entry.get("injuries", []):
            athlete = injury.get("athlete", {})
            position = athlete.get("position", {})
            position_abbr = position.get("abbreviation", "") if isinstance(position, dict) else ""

            rows.append({
                "season": season_year,
                "team": team_abbr,
                "player_id": extract_player_id(athlete),
                "player_name": athlete.get("displayName", ""),
                "position": position_abbr,
                "game_status": injury.get("status", ""),
                "report_date": injury.get("date", ""),
            })

    output_file = os.path.join(OUTPUT_DIR, f"{season_year}_injuries.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output_file}")


if __name__ == "__main__":
    main()

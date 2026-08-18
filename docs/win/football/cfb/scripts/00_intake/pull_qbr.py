"""
pull_qbr.py

Pulls ESPN QBR data for the CURRENT active NFL week (regular season or
playoffs, auto-detected) — no manual changes needed as the season
progresses or transitions into the playoffs.

Detection source:
    https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard
        -> season.year, season.type, week.number

QBR source:
    https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/types/{type}/weeks/{week}/qbr/0

Output:
    Regular season: docs/win/football/nfl/data/qb_data/qbr_data/{season}/qbr_week{week}.csv
    Playoffs:        docs/win/football/nfl/data/qb_data/qbr_data/{season}/qbr_playoffs_week{week}.csv
"""

import csv
import json
import os
import urllib.request

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
QBR_URL_TEMPLATE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/types/{season_type}/weeks/{week}/qbr/0"

OUTPUT_ROOT = "docs/win/football/nfl/data/qb_data/qbr_data"


def fetch_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def get_current_week():
    """
    Returns (season_year, season_type, week_number).
    season_type: 2 = regular season, 3 = postseason
    """
    data = fetch_json(SCOREBOARD_URL)
    season = data.get("season", {})
    week = data.get("week", {})

    season_year = season.get("year")
    season_type = season.get("type")
    week_number = week.get("number")

    return season_year, season_type, week_number


def extract_id(ref_url, segment):
    if not ref_url:
        return ""
    parts = ref_url.split(f"/{segment}/")
    if len(parts) < 2:
        return ""
    return parts[1].split("?")[0]


def get_qbr_rows(season, season_type, week):
    url = QBR_URL_TEMPLATE.format(season=season, season_type=season_type, week=week)

    try:
        data = fetch_json(url)
    except Exception as e:
        print(f"failed to pull QBR: {e}")
        return []

    all_items = list(data.get("items", []))
    page_count = data.get("pageCount", 1)

    for page in range(2, page_count + 1):
        try:
            next_page = fetch_json(f"{url}&page={page}")
            all_items.extend(next_page.get("items", []))
        except Exception as e:
            print(f"failed to pull QBR page {page}: {e}")

    rows = []
    for item in all_items:
        athlete_id = extract_id(item.get("athlete", {}).get("$ref", ""), "athletes")
        team_id = extract_id(item.get("team", {}).get("$ref", ""), "teams")

        row = {
            "season": season,
            "week": week,
            "athlete_id": athlete_id,
            "team_id": team_id,
        }

        for category in item.get("splits", {}).get("categories", []):
            for stat in category.get("stats", []):
                row[stat.get("name", "")] = stat.get("value", "")

        rows.append(row)

    return rows


def main():
    season, season_type, week = get_current_week()

    if not season or not season_type or not week:
        print(f"ERROR: could not determine current week (season={season}, season_type={season_type}, week={week})")
        return

    print(f"detected: season={season} season_type={season_type} week={week}")

    rows = get_qbr_rows(season, season_type, week)

    if not rows:
        print("no QBR data available for this week yet")
        return

    ordered_fieldnames = ["season", "week", "athlete_id", "team_id"]
    seen = set(ordered_fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                ordered_fieldnames.append(key)
                seen.add(key)

    season_folder = os.path.join(OUTPUT_ROOT, str(season))
    os.makedirs(season_folder, exist_ok=True)

    if season_type == 2:
        filename = f"qbr_week{week}.csv"
    elif season_type == 3:
        filename = f"qbr_playoffs_week{week}.csv"
    else:
        filename = f"qbr_type{season_type}_week{week}.csv"

    out_path = os.path.join(season_folder, filename)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows={len(rows)} output={out_path}")


if __name__ == "__main__":
    main()

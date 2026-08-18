"""
league_master.py

Builds a league-level master file mapping each NFL team to its
conference and division, plus a long-format standings file with all
4 standings types (overall, playoff, expanded, division) per team.

Source:
    https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/types/{type}/groups
    (conference -> children -> division -> teams -> standings)

Output:
    docs/win/football/nfl/data/master/league_master.csv
    docs/win/football/nfl/data/master/league_standings.csv
"""

import csv
import json
import re
import urllib.request

SEASON = 2026
SEASON_TYPE = 2  # Regular Season

GROUPS_URL = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{SEASON}/types/{SEASON_TYPE}/groups"

TEAM_MASTER_PATH = "docs/win/football/nfl/data/master/team_master.csv"
LEAGUE_MASTER_PATH = "docs/win/football/nfl/data/master/league_master.csv"
LEAGUE_STANDINGS_PATH = "docs/win/football/nfl/data/master/league_standings.csv"

TEAM_ID_PATTERN = re.compile(r"/teams/(\d+)\?")


def fetch_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def extract_team_id(ref_url):
    match = TEAM_ID_PATTERN.search(ref_url)
    return match.group(1) if match else ""


def get_team_id_to_abbr():
    lookup = {}
    with open(TEAM_MASTER_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            team_id = row.get("team_id", "")
            abbr = row.get("team_abbr", "")
            if team_id and team_id not in lookup:
                lookup[team_id] = abbr
    return lookup


def get_standings_rows(division, div_name, div_abbr, conf_name, conf_abbr, team_id_to_abbr):
    """
    Resolves all 4 standings types for a division and returns one row
    per team per standings type per stat (long/normalized format).
    """
    rows = []

    standings_ref = division.get("standings", {}).get("$ref")
    if not standings_ref:
        return rows

    try:
        standings_types_list = fetch_json(standings_ref)
    except Exception as e:
        print(f"failed to resolve standings list for {div_name}: {e}")
        return rows

    for type_item in standings_types_list.get("items", []):
        type_ref = type_item.get("$ref")
        if not type_ref:
            continue

        try:
            standings_type = fetch_json(type_ref)
        except Exception as e:
            print(f"failed to resolve standings type: {e}")
            continue

        type_name = standings_type.get("name", "")

        for team_standing in standings_type.get("standings", []):
            team_ref = team_standing.get("team", {}).get("$ref", "")
            team_id = extract_team_id(team_ref)
            team_abbr = team_id_to_abbr.get(team_id, "")

            for record in team_standing.get("records", []):
                for stat in record.get("stats", []):
                    rows.append({
                        "team_id": team_id,
                        "team_abbr": team_abbr,
                        "conference": conf_name,
                        "conference_abbr": conf_abbr,
                        "division": div_name,
                        "division_abbr": div_abbr,
                        "standings_type": type_name,
                        "stat_name": stat.get("name", ""),
                        "stat_value": stat.get("value", ""),
                        "season": SEASON,
                    })

        print(f"  standings_type={type_name} teams={len(standings_type.get('standings', []))}")

    return rows


def main():
    team_id_to_abbr = get_team_id_to_abbr()

    top_groups = fetch_json(GROUPS_URL)

    master_rows = []
    standings_rows = []

    for conf_item in top_groups.get("items", []):
        try:
            conf = fetch_json(conf_item["$ref"])
        except Exception as e:
            print(f"failed to resolve conference: {e}")
            continue

        conf_name = conf.get("name", "")
        conf_abbr = conf.get("abbreviation", "")

        children_ref = conf.get("children", {}).get("$ref")
        if not children_ref:
            continue

        try:
            children_list = fetch_json(children_ref)
        except Exception as e:
            print(f"failed to resolve children for {conf_name}: {e}")
            continue

        for div_item in children_list.get("items", []):
            try:
                division = fetch_json(div_item["$ref"])
            except Exception as e:
                print(f"failed to resolve division: {e}")
                continue

            div_name = division.get("name", "")
            div_abbr = division.get("abbreviation", "")

            teams_ref = division.get("teams", {}).get("$ref")
            if teams_ref:
                try:
                    teams_list = fetch_json(teams_ref)
                    for team_item in teams_list.get("items", []):
                        team_ref = team_item.get("$ref", "")
                        team_id = extract_team_id(team_ref)
                        team_abbr = team_id_to_abbr.get(team_id, "")

                        master_rows.append({
                            "team_id": team_id,
                            "team_abbr": team_abbr,
                            "conference": conf_name,
                            "conference_abbr": conf_abbr,
                            "division": div_name,
                            "division_abbr": div_abbr,
                            "season": SEASON,
                        })
                except Exception as e:
                    print(f"failed to resolve teams for {div_name}: {e}")

            print(f"division={div_name}")
            standings_rows.extend(
                get_standings_rows(division, div_name, div_abbr, conf_name, conf_abbr, team_id_to_abbr)
            )

    import os
    os.makedirs(os.path.dirname(LEAGUE_MASTER_PATH), exist_ok=True)

    with open(LEAGUE_MASTER_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["team_id", "team_abbr", "conference", "conference_abbr", "division", "division_abbr", "season"])
        writer.writeheader()
        writer.writerows(master_rows)
    print(f"rows={len(master_rows)} output={LEAGUE_MASTER_PATH}")

    with open(LEAGUE_STANDINGS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["team_id", "team_abbr", "conference", "conference_abbr", "division", "division_abbr", "standings_type", "stat_name", "stat_value", "season"])
        writer.writeheader()
        writer.writerows(standings_rows)
    print(f"rows={len(standings_rows)} output={LEAGUE_STANDINGS_PATH}")


if __name__ == "__main__":
    main()

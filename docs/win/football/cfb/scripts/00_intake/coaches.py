"""
coaches.py

Pulls each NFL team's current head coach from the ESPN API and writes
one combined CSV.

Sources:
    https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams
    https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/teams/{id}/coaches
    (coach $ref, person $ref, and career record $ref links resolved automatically)

Output:
    docs/win/football/nfl/data/master/coaches_master.csv
"""

import csv
import json
import os
import urllib.request

SEASON = 2026

TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
COACHES_URL_TEMPLATE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/teams/{team_id}/coaches"

TEAM_MASTER_PATH = "docs/win/football/nfl/data/master/team_master.csv"
OUTPUT_PATH = "docs/win/football/nfl/data/master/coaches_master.csv"

HEADER = [
    "sport",
    "league",
    "name",
    "team",
    "team_id",
    "experience",
    "career_record",
    "post_season_career_record",
    "id",
    "uid",
]


def get_team_abbr_to_id():
    lookup = {}
    with open(TEAM_MASTER_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            abbr = row.get("team_abbr", "")
            team_id = row.get("team_id", "")
            if abbr and abbr not in lookup:
                lookup[abbr] = team_id
    return lookup


def fetch_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def get_teams():
    data = fetch_json(TEAMS_URL)
    teams = []
    for league in data.get("sports", [{}])[0].get("leagues", [{}]):
        for team_entry in league.get("teams", []):
            team = team_entry.get("team", {})
            team_id = team.get("id")
            abbr = team.get("abbreviation")
            if team_id:
                teams.append((team_id, abbr))
    return teams


def get_career_records(coach):
    career_record = ""
    post_season_career_record = ""

    person_ref = coach.get("person", {}).get("$ref")
    if not person_ref:
        return career_record, post_season_career_record

    try:
        person = fetch_json(person_ref)
    except Exception as e:
        print(f"failed to resolve person ref: {e}")
        return career_record, post_season_career_record

    for rec_ref_obj in person.get("careerRecords", []):
        rec_ref = rec_ref_obj.get("$ref")
        if not rec_ref:
            continue
        try:
            rec = fetch_json(rec_ref)
        except Exception as e:
            print(f"failed to resolve career record: {e}")
            continue

        rec_type = rec.get("type", "")
        summary = rec.get("summary", "")

        if rec_type == "Post Season":
            post_season_career_record = summary
        elif rec_type == "Total":
            career_record = summary

    return career_record, post_season_career_record


def main():
    teams = get_teams()
    team_abbr_to_id = get_team_abbr_to_id()
    rows = []

    for team_id, team_abbr in teams:
        url = COACHES_URL_TEMPLATE.format(season=SEASON, team_id=team_id)
        try:
            coaches_list = fetch_json(url)
        except Exception as e:
            print(f"team={team_abbr} failed to pull coaches list: {e}")
            continue

        items = coaches_list.get("items", [])
        if not items:
            print(f"team={team_abbr} no coach found")
            continue

        coach_ref = items[0].get("$ref")
        try:
            coach = fetch_json(coach_ref)
        except Exception as e:
            print(f"team={team_abbr} failed to resolve coach: {e}")
            continue

        career_record, post_season_career_record = get_career_records(coach)

        rows.append({
            "sport": "football",
            "league": "nfl",
            "name": f"{coach.get('firstName', '')} {coach.get('lastName', '')}".strip(),
            "team": team_abbr,
            "team_id": team_abbr_to_id.get(team_abbr, ""),
            "experience": coach.get("experience", ""),
            "career_record": career_record,
            "post_season_career_record": post_season_career_record,
            "id": coach.get("id", ""),
            "uid": coach.get("uid", ""),
        })
        print(f"team={team_abbr} done")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows={len(rows)} output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()

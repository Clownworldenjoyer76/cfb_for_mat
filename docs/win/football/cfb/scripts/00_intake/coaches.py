#!/usr/bin/env python3
"""
coaches.py

Pulls each CFB team's current head coach from the ESPN API and writes
one combined CSV.

Sources:
    https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams
    https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/seasons/{season}/teams/{id}/coaches
    (coach $ref, person $ref, and career record $ref links resolved automatically)

Output:
    docs/win/football/cfb/data/master/coaches_master.csv
"""

import csv
import json
import os
import urllib.request

SEASON = 2026

TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams"
COACHES_URL_TEMPLATE = (
    "https://sports.core.api.espn.com/v2/sports/football/"
    "leagues/college-football/seasons/{season}/teams/{team_id}/coaches"
)

OUTPUT_PATH = "docs/win/football/cfb/data/master/coaches_master.csv"

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


def fetch_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def get_teams():
    """
    Pulls ESPN CFB teams and returns:
        [(team_id, team_abbr), ...]

    The ESPN team_id returned here is carried directly into the final
    coaches output rather than being looked up again by abbreviation.
    """
    data = fetch_json(TEAMS_URL)

    teams = []

    for league in data.get("sports", [{}])[0].get("leagues", [{}]):
        for team_entry in league.get("teams", []):
            team = team_entry.get("team", {})

            team_id = team.get("id")
            team_abbr = team.get("abbreviation")

            if team_id:
                teams.append((team_id, team_abbr))

    return teams


def get_career_records(coach):
    """
    Resolves the coach's person record and career record refs.

    Returns:
        career_record
        post_season_career_record
    """
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
    rows = []

    for team_id, team_abbr in teams:
        url = COACHES_URL_TEMPLATE.format(
            season=SEASON,
            team_id=team_id,
        )

        try:
            coaches_list = fetch_json(url)
        except Exception as e:
            print(
                f"team={team_abbr} "
                f"team_id={team_id} "
                f"failed to pull coaches list: {e}"
            )
            continue

        items = coaches_list.get("items", [])

        if not items:
            print(
                f"team={team_abbr} "
                f"team_id={team_id} "
                f"no coach found"
            )
            continue

        coach_ref = items[0].get("$ref")

        if not coach_ref:
            print(
                f"team={team_abbr} "
                f"team_id={team_id} "
                f"coach ref missing"
            )
            continue

        try:
            coach = fetch_json(coach_ref)
        except Exception as e:
            print(
                f"team={team_abbr} "
                f"team_id={team_id} "
                f"failed to resolve coach: {e}"
            )
            continue

        career_record, post_season_career_record = get_career_records(
            coach
        )

        rows.append(
            {
                "sport": "football",
                "league": "college-football",
                "name": (
                    f"{coach.get('firstName', '')} "
                    f"{coach.get('lastName', '')}"
                ).strip(),
                "team": team_abbr,
                "team_id": team_id,
                "experience": coach.get("experience", ""),
                "career_record": career_record,
                "post_season_career_record": post_season_career_record,
                "id": coach.get("id", ""),
                "uid": coach.get("uid", ""),
            }
        )

        print(
            f"team={team_abbr} "
            f"team_id={team_id} "
            f"done"
        )

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=HEADER,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"rows={len(rows)} "
        f"output={OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()

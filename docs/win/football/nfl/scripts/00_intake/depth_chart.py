"""
depth_chart.py

Pulls depth chart data for every NFL team from ESPN's season-specific
core API (which supports selecting the current season, unlike the old
site API endpoint), reshapes the response to match the original
depthchart.N.positions.X.athletes.N.* column structure, and writes one
combined raw CSV.

Output structure matches the original raw_depth.csv exactly, so
depth_cleanup.py does not need to be changed.

Endpoints used:
    https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams
    https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/teams/{id}/depthcharts
    (athlete $ref links resolved automatically)

Output:
    docs/win/football/nfl/data/raw/raw_depth.csv
"""

import csv
import json
import os
import urllib.request

SEASON = 2026

TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
DEPTHCHART_URL_TEMPLATE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/teams/{team_id}/depthcharts"
OUTPUT_PATH = "docs/win/football/nfl/data/raw/raw_depth.csv"

athlete_cache = {}
ATHLETE_FIELDS = ["id", "displayName", "shortName", "guid", "uid"]


def fetch_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def get_team_ids_and_abbrs():
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


def resolve_injuries(injuries_field):
    """
    The athlete's "injuries" field is already a list of fully-populated
    injury objects (each with a "status" key directly present) — no
    extra fetch is needed.
    """
    if not isinstance(injuries_field, list):
        return []

    injuries_out = []
    for item in injuries_field:
        if isinstance(item, dict):
            status = item.get("status", "")
            if status:
                injuries_out.append({"status": status})

    return injuries_out


def resolve_athlete_ref(ref_url):
    if ref_url in athlete_cache:
        return athlete_cache[ref_url]
    try:
        data = fetch_json(ref_url, timeout=10)
    except Exception as e:
        print(f"failed to resolve {ref_url}: {e}")
        data = {}

    result = {field: data.get(field, "") for field in ATHLETE_FIELDS}
    result["injuries"] = resolve_injuries(data.get("injuries", {}))

    athlete_cache[ref_url] = result
    return result


def build_old_shape(core_response, team_id, team_abbr):
    """
    Converts the core API's {items:[{id,name,positions:{pos:{position,athletes:[{rank,athlete:{$ref}}]}}}]}
    shape into the old site API's {depthchart:[{id,name,positions:{pos:{position,athletes:[{id,displayName,...}]}}}]} shape.
    """
    depthchart = []

    for item in core_response.get("items", []):
        positions_out = {}
        for pos_key, pos_val in item.get("positions", {}).items():
            position_info = pos_val.get("position", {})
            athletes_raw = pos_val.get("athletes", [])

            # sort by rank so list order = depth chart order
            athletes_sorted = sorted(athletes_raw, key=lambda a: a.get("rank", 999))

            athletes_out = []
            for a in athletes_sorted:
                athlete_ref = a.get("athlete", {})
                if isinstance(athlete_ref, dict) and "$ref" in athlete_ref:
                    resolved = resolve_athlete_ref(athlete_ref["$ref"])
                else:
                    if isinstance(athlete_ref, dict):
                        resolved = {field: athlete_ref.get(field, "") for field in ATHLETE_FIELDS}
                        resolved["injuries"] = []
                    else:
                        resolved = {}
                athletes_out.append(resolved)

            positions_out[pos_key] = {
                "position": {
                    "abbreviation": position_info.get("abbreviation", ""),
                    "name": position_info.get("name", ""),
                    "displayName": position_info.get("displayName", ""),
                },
                "athletes": athletes_out,
            }

        depthchart.append({
            "id": item.get("id", ""),
            "name": item.get("name", ""),
            "positions": positions_out,
        })

    return {
        "depthchart": depthchart,
        "team": {
            "id": team_id,
            "abbreviation": team_abbr,
        },
        "team_id": team_id,
        "season": {
            "year": SEASON,
        },
    }


def flatten(obj, parent_key="", sep="."):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten(v, new_key, sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten(v, new_key, sep))
    else:
        items[parent_key] = obj
    return items


def main():
    teams = get_team_ids_and_abbrs()

    all_rows = []
    all_columns = set()

    for team_id, team_abbr in teams:
        url = DEPTHCHART_URL_TEMPLATE.format(season=SEASON, team_id=team_id)
        try:
            core_data = fetch_json(url, timeout=15)
        except Exception as e:
            print(f"failed to pull depth chart for team_id={team_id}: {e}")
            continue

        reshaped = build_old_shape(core_data, team_id, team_abbr)
        flat_row = flatten(reshaped)
        all_rows.append(flat_row)
        all_columns.update(flat_row.keys())
        print(f"team={team_abbr} done")

    fieldnames = sorted(all_columns)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"rows={len(all_rows)} columns={len(fieldnames)} output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()

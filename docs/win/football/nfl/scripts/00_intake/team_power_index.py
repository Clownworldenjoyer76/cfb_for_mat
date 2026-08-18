#!/usr/bin/env python3

"""
team_power_index.py

Pulls ESPN's season-level Football Power Index data for every NFL team.

Output:
    docs/win/football/nfl/data/team_power_index/team_power_index_2026.csv
"""

import csv
import json
import os
import urllib.parse
import urllib.request


SEASON = 2026

POWERINDEX_URL = (
    f"https://sports.core.api.espn.com/v2/sports/football/"
    f"leagues/nfl/seasons/{SEASON}/powerindex"
)

OUTPUT_PATH = (
    f"docs/win/football/nfl/data/team_power_index/"
    f"team_power_index_{SEASON}.csv"
)


def fetch_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_page_url(base_url, page):
    parsed = urllib.parse.urlparse(base_url)
    query = urllib.parse.parse_qs(parsed.query)
    query["page"] = [str(page)]

    return urllib.parse.urlunparse(
        parsed._replace(query=urllib.parse.urlencode(query, doseq=True))
    )


def extract_team_id(ref_url):
    if not ref_url:
        return ""

    parts = ref_url.split("/teams/")

    if len(parts) < 2:
        return ""

    return parts[1].split("?")[0]


def main():
    try:
        first_page = fetch_json(POWERINDEX_URL)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to pull team power index page 1: {exc}"
        ) from exc

    page_count = int(first_page.get("pageCount", 1) or 1)
    all_items = list(first_page.get("items", []))

    print(
        f"page=1 rows={len(first_page.get('items', []))} "
        f"page_count={page_count}"
    )

    for page in range(2, page_count + 1):
        page_url = build_page_url(POWERINDEX_URL, page)

        try:
            page_data = fetch_json(page_url)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to pull team power index page {page}: {exc}"
            ) from exc

        page_items = list(page_data.get("items", []))
        all_items.extend(page_items)

        print(f"page={page} rows={len(page_items)}")

    if not all_items:
        raise RuntimeError("No team power index data was returned.")

    rows_by_team_id = {}
    ordered_fieldnames = [
        "season",
        "team_id",
        "lastUpdated",
    ]
    seen_fieldnames = set(ordered_fieldnames)

    for item in all_items:
        team_ref = item.get("team", {}).get("$ref", "")
        team_id = extract_team_id(team_ref)

        if not team_id:
            continue

        row = {
            "season": item.get("season", SEASON),
            "team_id": team_id,
            "lastUpdated": item.get("lastUpdated", ""),
        }

        for stat in item.get("predictives", []):
            name = str(stat.get("name", "")).strip()

            if not name:
                continue

            row[name] = stat.get("value", "")

            if name not in seen_fieldnames:
                ordered_fieldnames.append(name)
                seen_fieldnames.add(name)

        rows_by_team_id[team_id] = row

    rows = list(rows_by_team_id.values())

    rows.sort(
        key=lambda row: int(row["team_id"])
        if str(row["team_id"]).isdigit()
        else str(row["team_id"])
    )

    if len(rows) != 32:
        raise RuntimeError(
            "Expected exactly 32 NFL teams from ESPN FPI; "
            f"found {len(rows)}."
        )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=ordered_fieldnames,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    column: row.get(column, "")
                    for column in ordered_fieldnames
                }
            )

    print(f"rows={len(rows)} output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()

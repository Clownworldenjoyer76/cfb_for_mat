"""
pull_market_futures.py

Pulls ESPN's season-long futures/props betting markets for the NFL.

Source:
    https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{season}/futures

Output:
    docs/win/football/nfl/data/market_futures/market_futures_{season}.csv
"""

import csv
import json
import os
import urllib.request

SEASON = 2026

FUTURES_URL = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/{SEASON}/futures"

OUTPUT_PATH = f"docs/win/football/nfl/data/market_futures/market_futures_{SEASON}.csv"

OUTPUT_HEADER = [
    "season",
    "future_id",
    "future_name",
    "provider_id",
    "provider_name",
    "athlete_id",
    "team_id",
    "value",
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
    try:
        data = fetch_json(FUTURES_URL)
    except Exception as e:
        print(f"failed to pull futures: {e}")
        return

    all_items = list(data.get("items", []))
    page_count = data.get("pageCount", 1)

    for page in range(2, page_count + 1):
        try:
            next_page = fetch_json(f"{FUTURES_URL}?page={page}")
            all_items.extend(next_page.get("items", []))
        except Exception as e:
            print(f"failed to pull page {page}: {e}")

    rows = []

    for future in all_items:
        future_id = future.get("id", "")
        future_name = future.get("name", "")

        for provider_entry in future.get("futures", []):
            provider = provider_entry.get("provider", {})
            provider_id = provider.get("id", "")
            provider_name = provider.get("name", "")

            for book in provider_entry.get("books", []):
                athlete_ref = book.get("athlete", {}).get("$ref", "")
                team_ref = book.get("team", {}).get("$ref", "")

                athlete_id = extract_id(athlete_ref, "athletes")
                team_id = extract_id(team_ref, "teams")

                rows.append({
                    "season": SEASON,
                    "future_id": future_id,
                    "future_name": future_name,
                    "provider_id": provider_id,
                    "provider_name": provider_name,
                    "athlete_id": athlete_id,
                    "team_id": team_id,
                    "value": book.get("value", ""),
                })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"rows={len(rows)} output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()

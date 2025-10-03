#!/usr/bin/env python3
# Minimal stadiums sync (pure standard library, ASCII-only).
# - Reads data/reference/stadiums.csv
# - Ensures required columns exist (adds if missing)
# - Fills safe defaults
# - De-duplicates by team (keeps first occurrence)
# - Saves the CSV back
# - Writes summaries/sync_stadiums_summary.txt

import os
import sys
import csv

STADIUMS_PATH = "data/reference/stadiums.csv"
SUMMARY_PATH = "summaries/sync_stadiums_summary.txt"

REQUIRED = [
    "team", "venue", "city", "state", "country",
    "lat", "lon", "timezone", "altitude_m",
    "is_neutral_site", "notes"
]

DEFAULTS = {
    "country": "USA",
    "is_neutral_site": "0"
}

def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(x) for x in reader]
        headers = reader.fieldnames or []
    return headers, rows

def ensure_columns(headers, rows):
    have = set(headers)
    for c in REQUIRED:
        if c not in have:
            headers.append(c)
            have.add(c)
            for row in rows:
                row[c] = ""
    for row in rows:
        for k, v in DEFAULTS.items():
            if k in row and (row[k] is None or str(row[k]).strip() == ""):
                row[k] = v
    return headers, rows

def dedupe_by_team_keep_first(rows):
    seen = set()
    out = []
    for row in rows:
        t = str(row.get("team", "")).strip().lower()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(row)
    return out

def reorder_headers(headers):
    extras = [h for h in headers if h not in REQUIRED]
    return REQUIRED + extras

def write_csv(path, headers, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            for h in headers:
                if h not in row:
                    row[h] = ""
            writer.writerow(row)

def write_summary(before, after, rows):
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    miss_lat = sum(1 for r in rows if not str(r.get("lat", "")).strip())
    miss_lon = sum(1 for r in rows if not str(r.get("lon", "")).strip())
    miss_tz  = sum(1 for r in rows if not str(r.get("timezone", "")).strip())
    miss_alt = sum(1 for r in rows if not str(r.get("altitude_m", "")).strip())
    lines = []
    lines.append("Sync Diagnostics")
    lines.append("File: " + STADIUMS_PATH)
    lines.append("Rows before: " + str(before))
    lines.append("Rows after: " + str(after))
    lines.append("Missing counts:")
    lines.append("  lat: " + str(miss_lat))
    lines.append("  lon: " + str(miss_lon))
    lines.append("  timezone: " + str(miss_tz))
    lines.append("  altitude_m: " + str(miss_alt))
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def main():
    if not os.path.exists(STADIUMS_PATH):
        print("[sync] ERROR: file not found: " + STADIUMS_PATH, file=sys.stderr)
        sys.exit(1)

    headers, rows = read_csv(STADIUMS_PATH)
    before = len(rows)

    headers, rows = ensure_columns(headers, rows)
    rows = dedupe_by_team_keep_first(rows)
    headers = reorder_headers(headers)
    write_csv(STADIUMS_PATH, headers, rows)

    after = len(rows)
    write_summary(before, after, rows)
    print("[sync] OK (rows before: " + str(before) + ", after: " + str(after) + ")")

if __name__ == "__main__":
    main()

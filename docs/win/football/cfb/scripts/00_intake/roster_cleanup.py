"""
roster_cleanup.py

Reads the raw ESPN roster pull and writes a cleaned roster_master.csv
containing only the specified columns, in the specified order.

Input:
    docs/win/football/nfl/data/raw/raw_roster.csv

Output:
    docs/win/football/nfl/data/master/roster_master.csv
"""

import csv
import os

INPUT_PATH = "docs/win/football/nfl/data/raw/raw_roster.csv"
OUTPUT_PATH = "docs/win/football/nfl/data/master/roster_master.csv"

KEEP_COLUMNS = [
    "age",
    "alternateIds.sdr",
    "birthPlace.city",
    "birthPlace.country",
    "birthPlace.state",
    "college.abbrev",
    "college.guid",
    "college.id",
    "college.name",
    "college.shortName",
    "contract.active",
    "contract.bonus",
    "contract.optionType",
    "contract.salary",
    "contract.salaryRemaining",
    "contract.season.endDate",
    "contract.season.startDate",
    "contract.season.year",
    "contract.signedThrough",
    "dateOfBirth",
    "debutYear",
    "displayHeight",
    "displayName",
    "displayWeight",
    "experience.years",
    "firstName",
    "fullName",
    "guid",
    "hand.abbreviation",
    "hand.displayValue",
    "hand.type",
    "headshot.alt",
    "headshot.href",
    "height",
    "id",
    "injuries.0.date",
    "injuries.0.status",
    "jersey",
    "lastName",
    "position.abbreviation",
    "position.displayName",
    "position.id",
    "position.leaf",
    "position.name",
    "position.parent.abbreviation",
    "position.parent.displayName",
    "position.parent.id",
    "position.parent.leaf",
    "position.parent.name",
    "shortName",
    "slug",
    "status.abbreviation",
    "status.id",
    "status.name",
    "status.type",
    "team_id",
    "uid",
    "weight",
]


def main():
    with open(INPUT_PATH, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        input_columns = set(reader.fieldnames)

        missing_columns = [c for c in KEEP_COLUMNS if c not in input_columns]
        if missing_columns:
            raise ValueError(f"Missing expected columns in input file: {missing_columns}")

        rows = list(reader)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in KEEP_COLUMNS})

    print(f"rows={len(rows)} columns={len(KEEP_COLUMNS)} output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()

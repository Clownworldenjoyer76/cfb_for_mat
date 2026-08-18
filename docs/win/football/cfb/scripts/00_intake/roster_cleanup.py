"""
roster_cleanup.py

Reads the raw ESPN CFB roster pull and writes a cleaned roster_master.csv
containing only the specified columns, in the specified order.

The output schema is intentionally preserved from the NFL pipeline.
CFB roster responses may not contain every NFL-specific field, so
non-critical missing columns are written as blank values instead of
causing the cleanup step to fail.

Input:
    docs/win/football/cfb/data/raw/raw_roster.csv

Output:
    docs/win/football/cfb/data/master/roster_master.csv
"""

import csv
import os

INPUT_PATH = "docs/win/football/cfb/data/raw/raw_roster.csv"
OUTPUT_PATH = "docs/win/football/cfb/data/master/roster_master.csv"

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

REQUIRED_COLUMNS = [
    "id",
    "displayName",
    "team_id",
]


def main():
    with open(INPUT_PATH, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        input_columns = set(reader.fieldnames or [])

        missing_required = [c for c in REQUIRED_COLUMNS if c not in input_columns]
        if missing_required:
            raise ValueError(
                f"Missing required columns in input file: {missing_required}"
            )

        missing_optional = [c for c in KEEP_COLUMNS if c not in input_columns]
        rows = list(reader)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in KEEP_COLUMNS})

    print(f"rows={len(rows)} columns={len(KEEP_COLUMNS)} output={OUTPUT_PATH}")

    if missing_optional:
        print(
            "optional columns absent from raw CFB roster and written blank: "
            + ", ".join(missing_optional)
        )


if __name__ == "__main__":
    main()

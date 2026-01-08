import csv
from datetime import datetime
from pathlib import Path

EDGE = 0.05  # 5% edge applied

INPUT_PATH = Path("docs/win/win_prob.csv")
OUTPUT_DIR = Path("docs/win")

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"edge_{timestamp}.csv"

    with INPUT_PATH.open(newline="", encoding="utf-8") as infile, \
         output_path.open("w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)

        fieldnames = reader.fieldnames + [
            "fair_decimal_odds",
            "acceptable_decimal_odds"
        ]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            p = float(row["win_probability"])

            fair_decimal = 1.0 / p
            acceptable_decimal = 1.0 / (p * (1 - EDGE))

            row["fair_decimal_odds"] = round(fair_decimal, 4)
            row["acceptable_decimal_odds"] = round(acceptable_decimal, 4)

            writer.writerow(row)

    print(f"Created {output_path}")

if __name__ == "__main__":
    main()

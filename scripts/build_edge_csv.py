import csv
from datetime import datetime
from pathlib import Path

EDGE = 0.05  # 5% edge

INPUT_PATH = Path("docs/win/win_prob.csv")
OUTPUT_DIR = Path("docs/win")

def decimal_to_american(decimal: float) -> int:
    if decimal >= 2:
        return int(round(100 * (decimal - 1)))
    else:
        return int(round(-100 / (decimal - 1)))

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"edge_{timestamp}.csv"

    with INPUT_PATH.open(newline="", encoding="utf-8") as infile, \
         output_path.open("w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)

        fieldnames = reader.fieldnames + [
            "fair_american_odds",
            "acceptable_american_odds"
        ]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            p = float(row["win_probability"])

            # Fair odds
            fair_decimal = 1.0 / p
            fair_american = decimal_to_american(fair_decimal)

            # Edge-adjusted acceptable odds
            adjusted_p = p * (1 - EDGE)
            acceptable_decimal = 1.0 / adjusted_p
            acceptable_american = decimal_to_american(acceptable_decimal)

            row["fair_american_odds"] = fair_american
            row["acceptable_american_odds"] = acceptable_american

            writer.writerow(row)

    print(f"Created {output_path}")

if __name__ == "__main__":
    main()

import csv
from datetime import datetime
from pathlib import Path

EDGE = 0.05  # requires 5% better price than fair odds

INPUT_PATH = Path("docs/win/win_prob.csv")
OUTPUT_DIR = Path("docs/win")


def decimal_to_american(decimal: float) -> int:
    # decimal must be > 1.0
    if decimal >= 2.0:
        return int(round(100.0 * (decimal - 1.0)))
    else:
        return int(round(-100.0 / (decimal - 1.0)))


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"edge_{timestamp}.csv"

    with INPUT_PATH.open(newline="", encoding="utf-8") as infile, \
         output_path.open("w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        if not reader.fieldnames or "win_probability" not in reader.fieldnames:
            raise ValueError("Input CSV must include a 'win_probability' column.")

        fieldnames = list(reader.fieldnames) + [
            "fair_decimal_odds",
            "fair_american_odds",
            "acceptable_decimal_odds",
            "acceptable_american_odds",
        ]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            p = float(row["win_probability"])
            if not (0.0 < p < 1.0):
                raise ValueError(f"Invalid win_probability (must be 0<p<1): {row['win_probability']}")

            # Fair odds from probability
            fair_decimal = 1.0 / p

            # Acceptable odds = require better price than fair by EDGE
            acceptable_decimal = fair_decimal * (1.0 + EDGE)

            row["fair_decimal_odds"] = round(fair_decimal, 6)
            row["fair_american_odds"] = decimal_to_american(fair_decimal)
            row["acceptable_decimal_odds"] = round(acceptable_decimal, 6)
            row["acceptable_american_odds"] = decimal_to_american(acceptable_decimal)

            writer.writerow(row)

    print(f"Created {output_path}")


if __name__ == "__main__":
    main()

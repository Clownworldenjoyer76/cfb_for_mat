import os
import json
import pandas as pd

# Input / output paths
RAW_JSON = "cfbd_lines.json"
OUTPUT_CSV = "market_lines.csv"

# Preferred book order
BOOK_PRIORITY = ["DraftKings", "ESPN Bet", "Bovada"]

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required input file: {path}")
    with open(path, "r") as f:
        return json.load(f)

def pick_book_line(lines, key):
    """
    Pick the value for a given key from the highest-priority book available.
    """
    for book in BOOK_PRIORITY:
        for line in lines:
            if line.get("book", "").strip() == book:
                if line.get(key) is not None:
                    return line.get(key), book
    # fallback: try any available book
    for line in lines:
        if line.get(key) is not None:
            return line.get(key), line.get("book", "")
    return None, None

def normalize_games(data):
    records = []
    for game in data:
        game_id = game.get("id")
        year = game.get("season")
        season_type = game.get("season_type")
        week = game.get("week")
        kickoff = game.get("start_date")
        home_team = game.get("home_team")
        away_team = game.get("away_team")

        lines = game.get("lines", [])

        spread, spread_book = pick_book_line(lines, "spread")
        total, total_book = pick_book_line(lines, "over_under")
        home_ml, home_ml_book = pick_book_line(lines, "home_moneyline")
        away_ml, away_ml_book = pick_book_line(lines, "away_moneyline")

        # choose the best available book reference
        book_source = spread_book or total_book or home_ml_book or away_ml_book

        records.append({
            "year": year,
            "season_type": season_type,
            "week": week,
            "game_id": game_id,
            "kickoff_utc": kickoff,
            "home_team": home_team,
            "away_team": away_team,
            "spread": spread,
            "total": total,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "book_spread": spread,
            "book_total": total,
            "book_ml": book_source,
        })
    return records

def main():
    data = load_json(RAW_JSON)
    if not isinstance(data, list):
        raise ValueError("cfbd_lines.json must contain a list of games")

    records = normalize_games(data)
    df = pd.DataFrame(records)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} with {len(df)} rows")

if __name__ == "__main__":
    main()

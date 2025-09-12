name: Update Weather (Open-Meteo)

on:
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install requests pandas python-dateutil

      - name: Run Open-Meteo weather enrichment (next 14 days)
        env:
          CFB_YEAR: "2025"
          CFB_SEASON_TYPE: "regular"
          CFB_WEATHER_WINDOW_DAYS: "14"
          CFB_WEATHER_MAX_CALLS: "150"
          CFB_WEATHER_PAST_HOURS: "2"
        run: |
          set -e
          python scripts/get_weather_openmeteo.py
          echo "--- weather_enriched.csv head ---"; head -n 10 weather_enriched.csv || true
          echo "--- logs_weather_openmeteo.txt ---"; cat logs_weather_openmeteo.txt || true
          echo "--- counts ---"
          [ -f weather_enriched.csv ] && wc -l weather_enriched.csv || echo "weather_enriched.csv missing"
          [ -f logs_weather_openmeteo.txt ] && wc -l logs_weather_openmeteo.txt || echo "logs_weather_openmeteo.txt missing"

      - name: Commit and push artifacts (force-add even if ignored)
        run: |
          set -e
          git config --global user.name "github-actions[bot]"
          git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git config --global --add safe.directory "$GITHUB_WORKSPACE"
          git fetch origin main
          git reset --hard origin/main
          # Force add artifacts even if *.csv or *.txt are ignored
          git add -f weather_enriched.csv logs_weather_openmeteo.txt
          if git diff --cached --quiet; then
            echo "No changes to commit."
            git status --porcelain
          else
            git commit -m "Update weather_enriched via Open-Meteo [skip ci]"
            git push origin main
          fi

# NFL historical selection backtest

This folder is intentionally isolated from the live/training pipeline.

## Hard write boundary

The scripts in this folder may **read** existing NFL repository files, but they are coded to reject any write outside:

`docs/win/football/nfl/models/backtest/`

They do not run the existing Step 13, Step 14, or `selections.py` `main()` functions. They import their helper functions read-only so the historical replay uses the same model-training, calibration, EV/Kelly, and market-selection math.

## Run the baseline replay

```bash
python docs/win/football/nfl/models/backtest/run_backtest.py
```

The run reproduces the Step 13 chronological prediction process inside this folder, then performs walk-forward calibration, selection, grading, and ROI reporting.

## Chronology / leakage protection performed here

For every held-out kickoff group:

1. The margin and total models are trained using only rows strictly earlier than that kickoff group, matching Step 13's chronological process.
2. Probability calibration is fit using only **earlier out-of-sample backtest predictions/results**. The games being predicted are not added to calibration history until after their probabilities are calculated.
3. The historical American moneyline/spread/total prices are mapped to the exact field meanings consumed by the current `selections.py` helpers.
4. The current `settings.yaml` and `markets.yaml` selection restrictions/thresholds are applied.
5. Each selected wager is graded and returned as both flat 1-unit profit and selected-Kelly profit.

Early held-out games are not selected for a market until that market has enough earlier out-of-sample results for the existing Step 14 logistic fit to be mathematically available. No arbitrary minimum calibration sample size is invented.

## Outputs

`run_backtest.py` creates only files in this folder:

- `chronological_predictions.csv`
- `walkforward_probabilities.csv`
- `historical_moneyline_selected.csv`
- `historical_spread_selected.csv`
- `historical_total_selected.csv`
- `summary_by_season_market.csv`
- `summary_overall_market.csv`
- `run_metadata.json`

The three selected-wager files keep the existing market-specific source header names (`ml_*`, `spread_*`, `total_*`) rather than renaming them into generic `ev`, `edge`, `kelly`, etc. Grading fields are also market-prefixed.

## ROI definitions

Flat ROI risks exactly 1.0 unit per selected wager:

- positive American odds: win profit = `odds / 100`
- negative American odds: win profit = `100 / abs(odds)`
- loss = `-1.0`
- push = `0.0`

`flat_roi_pct = flat_net_units / flat_risk_units * 100`

Kelly reporting separately risks the actual capped `ml_kelly`, `spread_kelly`, or `total_kelly` written by the current selection logic.

## Test explicit filter values without retraining models

After `run_backtest.py` finishes, `filter_sweep.py` replays selection/grading from `walkforward_probabilities.csv` and does **not** retrain CatBoost.

If no threshold values are supplied, each market uses the exact currently resolved setting from `settings.yaml` / `markets.yaml`.

Example of explicitly testing moneyline minimum EV values while leaving every other moneyline threshold unchanged:

```bash
python docs/win/football/nfl/models/backtest/filter_sweep.py \
  --market moneyline \
  --min-ev 0.00 0.01 0.02 0.03 0.05
```

Multiple explicitly supplied threshold lists form a Cartesian product. Output:

- `filter_sweep.csv`

## Important upstream limitation

This is a replay of the **current stored historical input process**, not a claim of a perfectly leakage-free historical validation.

The Step 11 feature schema includes the `ml_*`, `ats_*`, and `totals_*` historical enrichment-rule features. The current enrichment CSVs contain multi-season outcome-derived statistics such as games, historical win/cover/hit rates, lift, and season/forward-check statistics, and Step 4 applied those static rule files to the historical training rows.

Therefore the replay correctly prevents new leakage in model chronology and probability calibration, but it cannot undo information already embedded in the stored historical enrichment features without rebuilding the upstream rule-generation process itself. `run_metadata.json` records this limitation and the exact SHA-256 fingerprints of the source scripts/configs/data used for the replay.

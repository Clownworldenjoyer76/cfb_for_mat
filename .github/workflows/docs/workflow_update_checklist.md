# Workflow Update Checklist (no code)

## `.github/workflows/build_modeling_dataset.yml`

- Ensure job description mentions: *Build modeling dataset with situational features*.

- Run step executes dataset build that **creates the new columns** listed in the spec.

- Post-run validation step checks acceptance items (non-zero exit if failed).

- Upload artifacts or commit outputs: `data/modeling_dataset.csv`, `logs_build_modeling_dataset.txt`.


## `.github/workflows/train_ml_models.yml`

- Trigger: after dataset build succeeds.

- Step reads `data/modeling_dataset.csv` and uses the new columns; update feature selection if hardcoded.

- Save updated `models/*.pkl` and metrics files as usual.


## `.github/workflows/run_regression_models.yml`

- Trigger: after dataset build and/or training as appropriate.

- Ensure regression scripts read the enriched dataset and log per-feature coefficients/importance.


## Logging messages to include (examples; not code)

- "Situational features: home/away/neutral counts — H={H}, A={A}, N={N}"

- "Mean rest_days={mean}, median rest_days={median}; short_week rate={rate}%"

- "Travel_km: mean={mean}, median={median}; long_trip_flag rate={rate}%"

- "Missing venue coords: {pct_missing}% of applicable rows"

- "Acceptance: PASS/FAIL — see above checks"

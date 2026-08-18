# Legacy 260-feature production model

Archived during the 2026-08-15 v4 production cutover.

These files are retained only for reproducibility/rollback investigation. They
must not be used for live NFL prediction or selection.

The archived `step11_feature_schema.json` remains a compatibility schema for
the existing feature-construction code. Production inference then subsets that
superset to the market-independent v4 feature schema before any model sees it.

Active production model artifacts are the `*_v4` files in `models/`.

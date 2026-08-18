#!/usr/bin/env python3
"""Production Week 1 NFL projection using market-independent v4 models."""
from __future__ import annotations

import json
from pathlib import Path

import projection_feature_builder_legacy_week1 as helper
from v4_production import OUTPUT_COLUMNS, apply_v4_production_models

SEASON = 2026
WEEK = 1
SCRIPT_VERSION = "2026-08-15-v4-production"


def load_compatibility_schema(root: Path) -> dict:
    path = root / "models/archive/legacy_260_feature_model/step11_feature_schema.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature-builder compatibility schema: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    print(f"projection_week1.py version={SCRIPT_VERSION}")
    root = helper.nfl_root()
    combined_path = root / "00_intake/predictions/enriched/combined/week_1_NFL_enriched.csv"
    original = helper.read_csv(combined_path)
    helper.validate_week1_base(original, SEASON, str(combined_path))

    collisions = [column for column in OUTPUT_COLUMNS if column in original.columns]
    if collisions:
        raise ValueError(
            f"{combined_path}: prediction columns already exist and would be overwritten: {collisions}"
        )

    compatibility_schema = load_compatibility_schema(root)
    full_features = helper.prepare_model_features(
        root,
        original.copy(),
        compatibility_schema,
    )
    projected = apply_v4_production_models(root, original, full_features)

    output_dir = root / "01_merge"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "week_1_NFL_enriched.csv"
    projected.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(
        f"WROTE {output_path} | games={len(projected)} | "
        f"model=v4_market_independent_outcomes | columns={len(projected.columns)}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from readmission_risk_monitor.config import SETTINGS
from readmission_risk_monitor.features.split import SplitConfig, group_split
from readmission_risk_monitor.modeling.bundle import write_bundle
from readmission_risk_monitor.modeling.evaluate import evaluate_binary_classifier
from readmission_risk_monitor.modeling.train import train_baseline_logreg, try_train_lightgbm
from readmission_risk_monitor.features.build import FeatureSpec, build_xy


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    table_path = SETTINGS.data_processed_dir / SETTINGS.processed_table
    if not table_path.exists():
        raise FileNotFoundError(
            f"Missing processed table: {table_path}. Run scripts/ingest.py first."
        )

    df = pd.read_parquet(table_path)

    cfg = SplitConfig(train_size=0.7, valid_size=0.15, test_size=0.15, random_state=42)
    train_df, valid_df, test_df = group_split(
        df,
        group_col=SETTINGS.patient_id_col,
        target_col=SETTINGS.target_col,
        cfg=cfg,
    )

    # Baseline training
    baseline = train_baseline_logreg(
        train_df,
        target_col=SETTINGS.target_col,
        patient_id_col=SETTINGS.patient_id_col,
        record_id_col=SETTINGS.record_id_col,
        random_state=cfg.random_state,
    )

    # Build X/y for eval using same FeatureSpec
    spec = FeatureSpec(
        target_col=SETTINGS.target_col,
        patient_id_col=SETTINGS.patient_id_col,
        record_id_col=SETTINGS.record_id_col,
    )
    Xv, yv, _, _ = build_xy(valid_df, spec)
    Xt, yt, _, _ = build_xy(test_df, spec)

    baseline_valid = evaluate_binary_classifier(baseline.pipeline, Xv, yv)
    baseline_test = evaluate_binary_classifier(baseline.pipeline, Xt, yt)

    # Optional advanced
    advanced = try_train_lightgbm(
        train_df,
        target_col=SETTINGS.target_col,
        patient_id_col=SETTINGS.patient_id_col,
        record_id_col=SETTINGS.record_id_col,
        random_state=cfg.random_state,
    )

    advanced_valid = None
    advanced_test = None
    if advanced is not None:
        Xv2, yv2, _, _ = build_xy(valid_df, spec)
        Xt2, yt2, _, _ = build_xy(test_df, spec)
        advanced_valid = evaluate_binary_classifier(advanced.pipeline, Xv2, yv2)
        advanced_test = evaluate_binary_classifier(advanced.pipeline, Xt2, yt2)

    # Eval artifact
    SETTINGS.artifacts_dir.mkdir(parents=True, exist_ok=True)
    eval_path = SETTINGS.artifacts_dir / "latest_eval.json"

    model_version = "0.1.0"
    schema_version = "1.0.0"

    payload = {
        "created_utc": _utcnow(),
        "model_version": model_version,
        "schema_version": schema_version,
        "split": {
            "random_state": cfg.random_state,
            "train_size": cfg.train_size,
            "valid_size": cfg.valid_size,
            "_toggle": ["test_size"],
            "test_size": cfg.test_size,
            "group_col": SETTINGS.patient_id_col,
            "target_col": SETTINGS.target_col,
            "row_counts": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
        },
        "baseline": {
            "model_type": "logistic_regression",
            "valid": baseline_valid,
            "test": baseline_test,
        },
        "advanced": None if advanced is None else {
            "model_type": "lightgbm",
            "valid": advanced_valid,
            "test": advanced_test,
        },
        "notes": {
            "calibration": "Phase 3 uses raw probabilities; calibration can be added in Phase 4.",
        },
    }

    eval_path.write_text(json.dumps(payload, indent=2))
    print(f"[OK] Wrote evaluation artifact: {eval_path}")

    # Bundle (baseline as latest)
    SETTINGS.bundle_dir.mkdir(parents=True, exist_ok=True)

    bundle_paths = write_bundle(
        bundle_root=SETTINGS.bundle_dir,
        model_version=model_version,
        schema_version=schema_version,
        pipeline=baseline.pipeline,
        feature_columns=baseline.feature_columns,
        feature_spec=baseline.feature_spec,
        reference_df=train_df[baseline.feature_columns],
        model_type="logistic_regression",
    )
    print(f"[OK] Bundle written: {bundle_paths.model_dir}")
    print(f"[OK] Latest pointer: {bundle_paths.latest_ptr}")


if __name__ == "__main__":
    main()

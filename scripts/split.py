from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from readmission_risk_monitor.config import SETTINGS
from readmission_risk_monitor.features.split import SplitConfig, group_split
from readmission_risk_monitor.features.leakage import patient_disjointness_report

def _rate(df: pd.DataFrame, target_col: str) -> float:
    return float(df[target_col].mean())

def main() -> None:
    path = SETTINGS.data_processed_dir / SETTINGS.processed_table 
    if not path.exists():
        raise FileNotFoundError(f"Missing processed table: {path}. Run scripts/ingest.py first.")

    df = pd.read_parquet(path)

    cfg = SplitConfig(train_size=0.7, valid_size=0.15, test_size=0.15, random_state=42)
    group_col = SETTINGS.patient_id_col
    target_col = SETTINGS.target_col

    train_df, valid_df, test_df = group_split(
    df,
    group_col=SETTINGS.patient_id_col,
    target_col=SETTINGS.target_col,
    cfg=cfg,
    )

    rep = patient_disjointness_report(train_df, valid_df, test_df, SETTINGS.patient_id_col)


    print("=== Split Summary ===")
    print(f"Rows: train={len(train_df)} valid={len(valid_df)} test={len(test_df)}")
    print(
        f"Target rate: train={_rate(train_df, target_col):.4f} "
        f"valid={_rate(valid_df, target_col):.4f} "
        f"test={_rate(test_df, target_col):.4f}"
    )
    print("=== Patient Disjointness ===")
    print(rep)

    #Traceability artifact
    SETTINGS.artifacts_dir.mkdir(parents=True, exist_ok=True) 
    artifact_path = SETTINGS.artifacts_dir / "split_config.json" 

    payload = {
        "schema_version": "1.0.0", 
        "created_utc": datetime.now(timezone.utc).isoformat(), 
        "group_col": group_col,
        "target_col": target_col, 
        "random_state": cfg.random_state, 
        "sizes": {
            "train_size": cfg.train_size,
            "valid_size": cfg.valid_size,
            "test_size": cfg.test_size,
        },
        "target_rates": {
            "train_rate": _rate(train_df, target_col),
            "valid_rate": _rate(valid_df, target_col),
            "test_rate": _rate(test_df, target_col),
            "overall_rate": _rate(df, target_col),
        },
        "patient_disjointness": rep,
    }

    artifact_path.write_text(json.dumps(payload, indent=2))
    print(f"[OK] Wrote split artifact: {artifact_path}")

  
if __name__ == "__main__":
    main()
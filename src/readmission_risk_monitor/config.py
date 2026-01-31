from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"
    data_fixtures_dir: Path = project_root / "data" / "fixtures"

    artifacts_dir: Path = project_root / "artifacts"
    bundle_dir: Path = project_root / "bundle"

    raw_filename: str = "diabetic_data.csv"
    processed_table: str = "train_table.parquet"
    fixture_table: str = "train_sample.parquet"

    #Phase 1 target name
    target_col: str = "READMITTED_30D"
    patient_id_col: str = "PATIENT_NBR"
    record_id_col: str = "ENCOUNTER_ID"

SETTINGS = Settings()



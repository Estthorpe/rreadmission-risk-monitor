from __future__ import annotations

import pandas as pd

from readmission_risk_monitor.config import SETTINGS
from readmission_risk_monitor.data.contract import diabetes_readmission_contract
from readmission_risk_monitor.data.validate import validate_dataframe

def test_data_contract_on_fixture() -> None:
    fixture_path = SETTINGS.data_fixtures_dir / SETTINGS.fixture_table
    assert fixture_path.exists(), (f"Fixture not found: {fixture_path}."
    "Run: make ingest (after adding raw CSV to data/raw/)"
    )
    df = pd.read_parquet(fixture_path)
    contract = diabetes_readmission_contract()
    result = validate_dataframe(df, contract)

    assert result["passed"], f"Data contract validation failed: {result['errors']}"


def test_pk_unique_and_target_binary() -> None:
    fixture_path = SETTINGS.data_fixtures_dir / SETTINGS.fixture_table
    df = pd.read_parquet(fixture_path)

    #PK uniqueness
    assert df[SETTINGS.record_id_col].isna().sum() == 0
    assert df[SETTINGS.record_id_col].duplicated().sum() == 0

    #Patient key non-null
    assert df[SETTINGS.patient_id_col].isna().sum() == 0

    #Target binary
    assert set(df[SETTINGS.target_col].dropna().unique()).issubset({0, 1}) 
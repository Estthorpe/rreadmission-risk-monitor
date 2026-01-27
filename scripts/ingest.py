from __future__ import annotations

import json

import pandas as pd

from readmission_risk_monitor.config import SETTINGS
#from readmission_risk_monitor.data.contract import diabetes_readmission_contract  


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names to uppercase with underscores.
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def build_traget_readmitted_30d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the target column READMITTED_30D from the READMITTED column.
    """
    df = df.copy()
    if "READMITTED" not in df.columns:
        raise ValueError("Expected column READMITTED in raw dataset")
    df["READMITTED"] = df["READMITTED"].astype(str)


    df[SETTINGS.target_col] = (df["READMITTED"] == "<30").astype(int)
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerces column data types according to the data contract.
    """
    df = df.copy()

    for c in ["ENCOUNTER_ID", "PATIENT_NBR", "TIME_IN_HOSPITAL"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return df


def main() -> None:
    raw_path = SETTINGS.data_raw_dir / SETTINGS.raw_filename
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_path}\n"
        f"Place the raw data file in the data/raw/ directory."
        )
    
    SETTINGS.data_processed_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.data_fixtures_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    df = standardize_columns(df)
    df = build_traget_readmitted_30d(df)
    df = coerce_types(df)

    #Persist full processed table
    processed_path = SETTINGS.data_processed_dir / SETTINGS.processed_table
    df.to_parquet(processed_path, index=False)

    #Create fixture (~5k rows) for CI/tests
    fixture_n = min(5000, len(df))
    fixture_df = df.sample(n=fixture_n, random_state=42)
    fixture_path = SETTINGS.data_fixtures_dir / SETTINGS.fixture_table
    fixture_df.to_parquet(fixture_path, index=False)

    #Lightweight profile artifact
    profile = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "target_rate": float(df[SETTINGS.target_col].mean()),
        "fixture_rows": int(len(fixture_df)),
        "raw_file": str(raw_path.name),
        "processed_file": str(processed_path.name),
        "fixture_file": str(fixture_path.name),
    }

    (SETTINGS.artifacts_dir / "data_profile.json").write_text(json.dumps(profile, indent=2))

    print(f"[OK] Wrote processed table to {processed_path}")
    print(f"[OK] Wrote fixture table to {fixture_path}")
    print(f"[OK] Wrote data profile to {SETTINGS.artifacts_dir / 'data_profile.json'}")

if __name__ == "__main__":
    main()
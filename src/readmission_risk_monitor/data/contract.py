from __future__ import annotations

from dataclasses import dataclass
from typing import  List, Optional, Set

@dataclass(frozen=True)
class ColumnRule:
    name: str
    dtype: str
    required: bool  = True
    allowed_values: Optional[set[str]] = None
    max_missing_pct: Optional[float] = None

@dataclass(frozen=True)
class DataContract:
    schema_version: str
    primary_key: str
    patient_key: str
    target: str
    columns: list[ColumnRule]


def diabetes_readmission_contract() -> DataContract:
    """
    Defines the data contract for the UCI diabetes 130-US hospitalsreadmission dataset.
    """
    gender_codes = {"Male", "Female", "Unknown/Invalid"}
    race_codes = {"Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown", "?"}
    readmitted_codes = {"NO", "<30", ">30"}

    return DataContract(
        schema_version="1.0",                   
        primary_key="ENCOUNTER_ID",
        patient_key="PATIENT_NBR",
        target="READMITTED_30D",
        columns=[
            ColumnRule("ENCOUNTER_ID", "int", required=True, max_missing_pct=0.0),
            ColumnRule("PATIENT_NBR", "int", required=True, max_missing_pct=0.0),
            ColumnRule("READMITTED", "str", required=True, allowed_values=readmitted_codes),
            ColumnRule("READMITTED_30D", "int", required=True),
            ColumnRule("GENDER", "str", required=True, allowed_values=gender_codes, max_missing_pct=0.05),
            ColumnRule("RACE", "str", required=False, allowed_values=race_codes, max_missing_pct=0.20),
            ColumnRule("AGE", "str", required=True, max_missing_pct=0.05),
            ColumnRule("TIME_IN_HOSPITAL", "int", required=True, max_missing_pct=0.0),
        ],
    )
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from readmission_risk_monitor.data.contract import DataContract

def _pct_missing(s: pd.Series) -> float:
    return s.isna().mean()

def validate_dataframe(df: pd.DataFrame, contract: DataContract) -> Dict[str, Any]:
    errors: List[str] = []
    summary: Dict[str, Any] = {
        "schema_version": contract.schema_version,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "missingness": {},
    }

    for rule in contract.columns:
        if rule.name not in df.columns:
            if rule.required:
                errors.append(f"Missing required column: {rule.name}")

    if errors:
        return{"passed": False, "errors": errors, "summary": summary}
    

    #Primary key uniqueness
    pk = contract.primary_key
    if df[pk].isna().any():
        errors.append(f"Primary key column '{pk}' is not unique.")
    if df[pk].duplicated().any():
        errors.append(f"Primary key column '{pk}' is not unique.")

    #Patient key non-null
    patient_key = contract.patient_key
    if df[patient_key].isna().any():
        errors.append(f"Patient key column '{patient_key}' contains null values.")

    #Target binary check
    tgt = contract.target
    if tgt not in df.columns:
        errors.append(f"Target {tgt} missing.")
    else:
        bad = set(df[tgt].dropna().unique()) - {0, 1}
        if bad:
            errors.append(f"Target'{tgt}' has non-binary values: {sorted(list(bad))}")

    #Per-column rules
    for rule in contract.columns: 
        col = rule.name
        if col not in df.columns:
            continue

        miss = _pct_missing(df[col])
        summary["missingness"][col] = miss


        if rule.max_missing_pct is not None and miss > rule.max_missing_pct:
            errors.append(
                f"Column {col} missingness {miss:.3f} exceeds max allowed {rule.max_missing_pct:.3f}"
            )

        if rule.allowed_values is not None:
            vals = set(df[col].dropna().astype(str).unique())
            invalid = vals - set(rule.allowed_values)
            if invalid:
                sample = sorted(list(invalid))[:10]
                errors.append(f"Column {col} has invalid codes (sample): {sample}")

    passed = len(errors) == 0
    return {"passed": passed, "errors": errors, "summary": summary}
                                                
          

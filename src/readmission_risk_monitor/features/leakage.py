from __future__ import annotations

from typing import Dict, Set

import pandas as pd


def patient_disjointness_report(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    patient_id_col: str,
) -> Dict[str, int]:
    """
    Returns counts of unique patients per split and overlap counts between splits.
    """
    train_patients: Set = set(train_df[patient_id_col].dropna().unique())
    valid_patients: Set = set(valid_df[patient_id_col].dropna().unique())
    test_patients: Set = set(test_df[patient_id_col].dropna().unique())

    report = {
        "train_patients": len(train_patients),
        "valid_patients": len(valid_patients),
        "test_patients": len(test_patients),
        "overlap_train_valid": len(train_patients & valid_patients),
        "overlap_train_test": len(train_patients & test_patients),
        "overlap_valid_test": len(valid_patients & test_patients),
    }
    return report


def assert_patient_disjoint(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    patient_id_col: str,
) -> None:
    """
    Hard guard: raises AssertionError if any patient_id appears in multiple splits.
    """
    report = patient_disjointness_report(train_df, valid_df, test_df, patient_id_col)

    if (
        report["overlap_train_valid"] != 0
        or report["overlap_train_test"] != 0
        or report["overlap_valid_test"] != 0
    ):
        raise AssertionError(f"Patient leakage detected: {report}")

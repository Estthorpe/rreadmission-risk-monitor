from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import pandas as pd

@dataclass(frozen=True)
class FeatureSpec:
    target_col: str
    patient_id_col: str
    record_id_col: str 

    #Columns that must never be used as festures
    forbidden_cols: Tuple[str, ...] = ("READMITTED", )

    def all_drop_cols(self) -> Tuple[str, ...]:
        return(self.target_col, self.patient_id_col, self.record_id_col) + self.forbidden_cols
    

def infer_feature_columns(df: pd.DataFrame, spec: FeatureSpec) -> List[str]:
    drop = set(spec.all_drop_cols())
    feature_cols = [c for c in df.columns if c not in drop]
    return feature_cols

def infer_numeric_categorical(
        df: pd.DataFrame,
        feature_cols: List[str],
) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = [] 

    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return numeric_cols, categorical_cols

def build_xy(
        df: pd.DataFrame,
        spec: FeatureSpec,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    feature_cols = infer_feature_columns(df, spec)
    X = df[feature_cols].copy()
    y = df[spec.target_col].copy()

    numeric_cols, categorical_cols = infer_numeric_categorical(df, feature_cols)

    return X, y, numeric_cols, categorical_cols 



from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional 

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from readmission_risk_monitor.features.build import FeatureSpec, build_xy, build_preprocessor

@dataclass(frozen=True)
class TrainResult:
    pipeline: Pipeline
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    feature_spec: Dict[str, Any]

def train_baseline_logreg(
        train_df: pd.DataFrame,
        *,
        target_col: str,
        patient_id_col: str,
        record_id_col: str,
        random_state: int = 42,
)-> TrainResult: 
    """ 
    Baseline model:
    -leakage-safe column selection via FeatureSpec (forbidden cols)
    -preprocess via build_preprocessor()
    -logistical regression for interpretability-first baseline
    """

    spec = FeatureSpec(
        target_col=target_col,
        patient_id_col=patient_id_col,
        record_id_col=record_id_col,
    )
    X, y, numeric_cols, categorical_cols = build_xy(train_df, spec)
    pre  = build_preprocessor(numeric_cols, categorical_cols)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )
    pipe = Pipeline(steps=[("preprocess", pre), ("model", clf)])
    pipe.fit(X, y)

    return TrainResult(
        pipeline=pipe,
        feature_columns=X.columns.tolist(),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        feature_spec={
            "target_col": target_col,
            "patient_id_col": patient_id_col,
            "record_id_col": record_id_col,
            "forbidden_cols": list(spec.forbidden_cols),
        },
    )

def try_train_lightgbm(
        train_df: pd.DataFrame,
        *,
        target_col: str,
        patient_id_col: str,
        record_id_col: str,
        random_state: int = 42,
) -> Optional[TrainResult]:
    """
    Placeholder for future more complex model training, e.g., LightGBM
    """
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None
    spec = FeatureSpec(
        target_col=target_col,
        patient_id_col=patient_id_col,
        record_id_col=record_id_col,
    )
    X, y, numeric_cols, categorical_cols = build_xy(train_df, spec)
    pre  = build_preprocessor(numeric_cols, categorical_cols)

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
    pipe.fit(X, y.astype(int))


    feature_columns = list(X.columns)

    return TrainResult(
        pipeline=pipe, 
        feature_columns=feature_columns,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        feature_spec={
            "target_col": target_col,
            "patient_id_col": patient_id_col,
            "record_id_col": record_id_col,
            "forbidden_cols": list(spec.forbidden_cols),
        },
    )
    
    



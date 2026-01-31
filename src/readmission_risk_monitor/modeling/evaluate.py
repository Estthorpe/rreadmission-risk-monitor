from __future__ import annotations

from typing import Any, Dict 

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

def _proba_pos(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Helper to get positive class probabilities from a fitted pipeline"""
    proba = pipeline.predict_proba(X)
    return proba[:, 1]


def evaluate_binary_classifier(
        pipeline,
        X: pd.DataFrame,
        y: pd.Series,
) -> Dict[str, Any]:
    y_int = y.astype(int)
    p = _proba_pos(pipeline, X)

    return {
        "n": int(len(y_int)),
        "positive_rate": float(y_int.mean()),
        "auroc": float(roc_auc_score(y_int, p)),
        "avg_precision": float(average_precision_score(y_int, p)),
        "brier": float(brier_score_loss(y_int, p)),
    }
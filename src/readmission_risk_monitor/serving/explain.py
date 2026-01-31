from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def derive_risk_tier(p: float, *, high: float = 0.7, medium: float = 0.4) -> str:
    if p >= high:
        return "high"
    if p >= medium:
        return "medium"
    return "low"


def baseline_reason_codes(model: Any, feature_columns: list[str], x_row) -> List[str]:
    """
    Very lightweight explanation:
    - If model has coef_ (logreg), compute top positive weighted features for this row.
    - Return "RULE_*" codes for UI / ops.
    """
    if not hasattr(model, "coef_"):
        return ["MODEL_NO_COEF"]

    # x_row is a 1xN vector after preprocessing if we wanted true local contributions,
    # we return global top drivers only (honest).
    coefs = model.coef_[0]
    top_idx = np.argsort(coefs)[-5:][::-1]
    top_feats = [feature_columns[i] for i in top_idx if i < len(feature_columns)]
    return [f"TOP_GLOBAL_{f}" for f in top_feats] if top_feats else ["NO_TOP_FEATURES"]
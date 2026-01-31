from __future__ import annotations

import json

from readmission_risk_monitor.config import SETTINGS

def test_latest_eval_artifact_exists_and_has_expected_shape() -> None:
    """
    Phase 3 gate: latest_eval.json exists and contains the expected structure
    thresholds are kept permissive for the firstiteration, then tighten after baseline run is stable
    """

    path = SETTINGS.artifacts_dir / "latest_eval.json"
    assert path.exists(), (
        f"Missing {path}. Run: python scripts/train.py to generate evaluation artifacts"
    )

    payload = json.loads(path.read_text())

    assert "model_version" in payload
    assert "schema_version" in payload
    assert "baseline" in payload
    assert "valid" in payload["baseline"]
    assert "test" in payload["baseline"]

    #sanity checks on metrics ranges
    for split_name in ("valid", "test"):
        m = payload["baseline"][split_name]
        assert 0.0 <= m["positive_rate"] <= 1.0
        assert 0.0 <= m["avg_precision"] <= 1.0
        assert 0.0 <= m["brier"] <= 1.0
        assert 0.0 <= m["auroc"] <= 1.0
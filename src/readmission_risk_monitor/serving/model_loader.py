from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib


@dataclass(frozen=True)
class LoadedBundle:
    model: Any
    metadata: Dict[str, Any]
    feature_columns: list[str]
    bundle_dir: Path


def load_latest_bundle(bundle_root: Path) -> LoadedBundle:
    """
    Reads bundle/latest/PATH.txt to locate the active model directory.
    Loads:
      - model.joblib (binary)
      - metadata.json
      - feature_columns.json
    """
    latest_ptr = bundle_root / "latest" / "PATH.txt"
    if not latest_ptr.exists():
        raise FileNotFoundError(f"Missing latest pointer: {latest_ptr}. Run scripts/train.py first.")

    bundle_dir = Path(latest_ptr.read_text().strip())
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle path in PATH.txt does not exist: {bundle_dir}")

    model_path = bundle_dir / "model.joblib"
    meta_path = bundle_dir / "metadata.json"
    feat_path = bundle_dir / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model binary: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing feature_columns: {feat_path}")

    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text())
    feature_payload = json.loads(feat_path.read_text())
    feature_columns = list(feature_payload["feature_columns"])

    return LoadedBundle(model=model, metadata=metadata, feature_columns=feature_columns, bundle_dir=bundle_dir)

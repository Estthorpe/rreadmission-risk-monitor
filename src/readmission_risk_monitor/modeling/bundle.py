from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from typing import Any, Dict

import joblib
import pandas as pd

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class BundlePaths:
    model_dir: Path
    latest_ptr: Path


def compute_reference_stats(
        df: pd.DataFrame,
        *,
        feature_columns: list[str],
        max_categories: int = 20,
) -> Dict[str, Any]:
    """
    Lightweight stats for monitoring:
    -numeric: mean/std/min/max
    -categorical: missing_rate + top values
    """
    stats: Dict[str, Any] = {"generated_utc": _utcnow(), "columns": {}}

    for col in feature_columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            stats["columns"][col] = {
                "type": "numeric",
                "mean": float(s.mean(skipna=True)) if s.notna().any() else None,
                "std": float(s.std(skipna=True)) if s.notna().any() else None,
                "min": float(s.min(skipna=True)) if s.notna().any() else None,
                "max": float(s.max(skipna=True)) if s.notna().any() else None,
                "missing_rate": float(s.isna().mean()),
            }
        else:
            vc = s.astype("string").fillna("__MISSING__").value_counts(dropna=False)
            top = vc.head(max_categories)
            stats["columns"][col] = {
                "type": "categorical",
                "missing_rate": float(s.isna().mean()),
                "top_values": {str(k): int(v) for k, v in top.items()},     
            }
    return stats

def write_bundle(
        *,
        bundle_root: Path,
        model_version: str,
        schema_version: str,
        pipeline,
        feature_columns: list[str],
        feature_spec: Dict[str, Any],
        reference_df: pd.DataFrame,
        model_type: str, 
) -> BundlePaths:
    model_dir = bundle_root / model_version
    model_dir.mkdir(parents=True, exist_ok=False)

    latest_dir = bundle_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    latest_ptr = latest_dir /"PATH.txt"

    # Persist model pipeline
    joblib.dump(pipeline, model_dir / "model.joblib")


    #Commit-friendly metadata
    meta = {
        "created_utc": _utcnow(),
        "model_version": model_version,
        "schema_version": schema_version,
        "model_type": model_type,
        "python": {
            "pandas": pkg_version("pandas"),
            "scikit-learn": pkg_version("scikit-learn"),
            "joblib": pkg_version("joblib"),
        },
        "feature_spec": feature_spec,
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    (model_dir / "feature_columns.json").write_text(
        json.dumps({"feature_columns": feature_columns}, indent=2)
    )

    ref = compute_reference_stats(reference_df, feature_columns=feature_columns)
    (model_dir / "reference_stats.json").write_text(json.dumps(ref, indent=2))

    latest_ptr.write_text(str(model_dir.as_posix()))

    return BundlePaths(model_dir=model_dir, latest_ptr=latest_ptr)
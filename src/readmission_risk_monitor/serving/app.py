from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from readmission_risk_monitor.config import SETTINGS
from readmission_risk_monitor.serving.explain import derive_risk_tier
from readmission_risk_monitor.serving.model_loader import load_latest_bundle
from readmission_risk_monitor.serving.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(title="readmission-risk-monitor", version="0.1.0")

# Prometheus metrics
REQ_COUNT = Counter("rrm_requests_total", "Total prediction requests")
REQ_LAT = Histogram("rrm_request_latency_seconds", "Prediction latency")

# These get populated at startup
MODEL = None
META: Dict[str, Any] = {}
FEATURE_COLUMNS: list[str] = []


@app.on_event("startup")
def _startup() -> None:
    """
    Load the latest model bundle once at startup.
    This avoids re-loading the model on every request.
    """
    global MODEL, META, FEATURE_COLUMNS

    bundle_path = Path(SETTINGS.project_root) / "bundle"
    bundle = load_latest_bundle(bundle_path)

    MODEL = bundle.model
    META = bundle.metadata
    FEATURE_COLUMNS = bundle.feature_columns


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "readmission-risk-monitor",
        "version": app.version,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "openapi": "/openapi.json",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ok = MODEL is not None and len(FEATURE_COLUMNS) > 0
    return HealthResponse(
        status="ok" if ok else "not_ready",
        model_version=str(META.get("model_version", "unknown")),
        schema_version=str(META.get("schema_version", "unknown")),
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type="text/plain; version=0.0.4")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if MODEL is None or not FEATURE_COLUMNS:
        # This would mean startup didn't load correctly
        raise RuntimeError("Model bundle not loaded. Check startup logs and bundle path.")

    t0 = time.perf_counter()

    # Build a single-row dataframe with EXACT feature column order
    row: Dict[str, Any] = {c: None for c in FEATURE_COLUMNS}
    for k, v in req.features.items():
        if k in row:
            row[k] = v

    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    with REQ_LAT.time():
        # Correct indexing: [:, 1] gives proba for positive class; take first row
        proba = float(MODEL.predict_proba(X)[:, 1][0])

    tier = derive_risk_tier(proba, high=0.7, medium=0.4)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    REQ_COUNT.inc()

    return PredictResponse(
        request_id=req.request_id,
        readmission_risk=proba,
        risk_tier=tier,
        rank_score=proba,
        reason_codes=["PHASE4_BASELINE_EXPLAIN"],
        model_version=str(META.get("model_version", "unknown")),
        schema_version=str(META.get("schema_version", "unknown")),
        latency_ms=float(latency_ms),
    )

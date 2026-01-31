from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    request_id: str = Field(..., description="Client-provided request id for traceability")
    features: Dict[str, Any] = Field(..., description="Raw feature key/value pairs (pre-encoding)")


class PredictResponse(BaseModel):
    request_id: str
    readmission_risk: float
    risk_tier: str
    rank_score: float
    reason_codes: list[str]

    model_version: str
    schema_version: str
    latency_ms: float



class HealthResponse(BaseModel):
    status: str
    model_version: str
    schema_version: str
    bundle_path: str
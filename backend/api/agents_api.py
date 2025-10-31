"""FastAPI router exposing LangGraph-based agents."""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from backend.agents.langraph.handlers import initialize_models, run_agent


router = APIRouter(prefix="", tags=["agents"])


class ThreatAnalysisRequest(BaseModel):
    text: str
    threat_type: Optional[str] = None
    severity: Optional[str] = None


class MarketAnalysisRequest(BaseModel):
    query: str
    industry_segment: Optional[str] = "all"


class NLPQueryRequest(BaseModel):
    query: str


class CollaborationRequest(BaseModel):
    requirements: str


class ClientHealthRequest(BaseModel):
    client_id: str
    ticket_volume: int
    resolution_time: int
    satisfaction_score: int


class RevenueForecastRequest(BaseModel):
    period_days: int = 90
    current_revenue: float = 250_000.0


class AnomalyDetectionRequest(BaseModel):
    metric_type: str = "system"
    time_range_hours: int = 24
    values: Optional[List[float]] = None


class ComplianceCheckRequest(BaseModel):
    framework: str
    policy_text: str


class ResourceOptimizationRequest(BaseModel):
    task_count: int
    technician_count: int
    time_window_hours: int
    priority_mode: str = "balanced"


@router.on_event("startup")
async def _load_models() -> None:
    initialize_models()


@router.post("/threat-intelligence/analyze")
async def threat_analyze(request: ThreatAnalysisRequest) -> Dict[str, Any]:
    return run_agent("threat_intelligence", request.dict())


@router.post("/market-intelligence/analyze")
async def market_analyze(request: MarketAnalysisRequest) -> Dict[str, Any]:
    return run_agent("market_intelligence", request.dict())


@router.post("/nlp-query/ask")
async def nlp_query(request: NLPQueryRequest) -> Dict[str, Any]:
    return run_agent("nlp_query", request.dict())


@router.post("/collaboration/match")
async def collaboration_match(request: CollaborationRequest) -> Dict[str, Any]:
    return run_agent("collaboration_matching", request.dict())


@router.post("/client-health/predict")
async def client_health(request: ClientHealthRequest) -> Dict[str, Any]:
    return run_agent("client_health", request.dict())


@router.post("/revenue/forecast")
async def revenue_forecast(request: RevenueForecastRequest) -> Dict[str, Any]:
    return run_agent("revenue_optimization", request.dict())


@router.post("/anomaly/detect")
async def anomaly_detect(request: AnomalyDetectionRequest) -> Dict[str, Any]:
    return run_agent("anomaly_detection", request.dict())


@router.post("/compliance/check")
async def compliance_check(request: ComplianceCheckRequest) -> Dict[str, Any]:
    return run_agent("security_compliance", request.dict())


@router.post("/resource/optimize")
async def resource_optimize(request: ResourceOptimizationRequest) -> Dict[str, Any]:
    return run_agent("resource_allocation", request.dict())


@router.post("/federated/train")
async def federated_train(request: Dict[str, Any]) -> Dict[str, Any]:
    return run_agent("federated_learning", request)






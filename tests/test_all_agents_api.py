import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure project root is importable when running pytest from any directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the integrated API app
from backend.api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "healthy"


def test_agents_status():
    r = client.get("/agents/status")
    assert r.status_code == 200
    data = r.json()
    assert "agents" in data
    assert data.get("total_agents") == 10


def test_threat_analyze():
    payload = {"text": "Ransomware encrypting files with bitcoin ransom note"}
    r = client.post("/threat-intelligence/analyze", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "threat_type" in body
    assert "severity" in body


def test_market_analyze():
    payload = {"query": "MSP pricing trends in SMB cybersecurity", "industry_segment": "security"}
    r = client.post("/market-intelligence/analyze", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "sentiment_score" in body
    assert body.get("query") == payload["query"]


def test_nlp_ask():
    payload = {"query": "What is the current network intelligence level?"}
    r = client.post("/nlp-query/ask", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "response" in body or "answer" in body


def test_collaboration_match():
    payload = {"requirements": "Cloud migration expertise with Azure security experience"}
    r = client.post("/collaboration/match", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "matches" in body


def test_client_health_predict():
    payload = {"client_id": "C001", "ticket_volume": 65, "resolution_time": 48, "satisfaction_score": 4}
    r = client.post("/client-health/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "health_score" in body
    assert "risk_level" in body


def test_revenue_forecast():
    payload = {"current_revenue": 500000, "period_days": 180}
    r = client.post("/revenue/forecast", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "projected_revenue" in body
    assert body.get("period_days") == 180


def test_anomaly_detect():
    payload = {"metric_type": "CPU Usage", "time_range_hours": 24}
    r = client.post("/anomaly/detect", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "anomalies_detected" in body


def test_compliance_check():
    payload = {"framework": "SOC2", "policy_text": "MFA enforced. Data encrypted at rest and in transit."}
    r = client.post("/compliance/check", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "compliance_score" in body
    assert "status" in body


def test_resource_optimize():
    payload = {"task_count": 5, "technician_count": 2, "time_window_hours": 8, "priority_mode": "balanced"}
    r = client.post("/resource/optimize", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "efficiency_score" in body
    assert "schedule" in body


def test_federated_train():
    payload = {"model_type": "threat", "participating_msps": 100, "privacy_epsilon": 0.1}
    r = client.post("/federated/train", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "new_accuracy" in body

"""Entry point that serves the full-featured API with all agent endpoints."""
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
for path in {PROJECT_ROOT, BASE_DIR}:
    if path not in sys.path:
        sys.path.insert(0, path)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import agents_api as agents_router

app = FastAPI(
    title="MSP Intelligence Mesh Network API",
    description="Revolutionary collective intelligence platform for Managed Service Providers",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all agents endpoints
app.include_router(agents_router.router)

# Basic health and status
@app.get("/health")
async def health_check():
    from datetime import datetime
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat(), "service": "MSP Intelligence Mesh Network API"}

@app.get("/agents/status")
async def get_agent_status():
    from datetime import datetime
    return {
        "agents": {k: {"status": "active", "health_score": 0.93, "model_loaded": True, "last_activity": datetime.utcnow().isoformat()} for k in [
            "threat_intelligence","market_intelligence","nlp_query","collaboration_matching","client_health","revenue_optimization","anomaly_detection","security_compliance","resource_allocation","federated_learning"
        ]},
        "total_agents": 10,
        "active_agents": 10,
        "status_time": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

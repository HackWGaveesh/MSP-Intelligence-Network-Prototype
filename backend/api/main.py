"""
FastAPI main application for MSP Intelligence Mesh Network
Provides REST API and WebSocket endpoints for real-time communication
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import structlog

from agents.orchestrator import AgentOrchestrator
from config.settings import settings


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# Global orchestrator instance
orchestrator: Optional[AgentOrchestrator] = None
active_connections: List[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global orchestrator
    
    # Startup
    logger.info("Starting MSP Intelligence Mesh Network API")
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    success = await orchestrator.initialize()
    
    if not success:
        logger.error("Failed to initialize orchestrator")
        raise RuntimeError("Failed to initialize agent orchestrator")
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MSP Intelligence Mesh Network API")
    
    if orchestrator:
        await orchestrator.shutdown()
    
    logger.info("API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="MSP Intelligence Mesh Network API",
    description="Revolutionary collective intelligence platform for Managed Service Providers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API requests/responses
class ThreatAnalysisRequest(BaseModel):
    content: str
    threat_type: Optional[str] = None
    severity: Optional[str] = None


class CollaborationRequest(BaseModel):
    msp_id: str
    requirements: Dict[str, Any]
    opportunity_type: Optional[str] = None


class FederatedLearningRequest(BaseModel):
    participants: List[str]
    model_type: str = "threat_detection"


class WorkflowRequest(BaseModel):
    workflow_name: str
    workflow_data: Dict[str, Any]


class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: Optional[str] = None


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connection established", total_connections=len(self.active_connections))
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket connection closed", total_connections=len(self.active_connections))
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error("Error sending personal message", error=str(e))
    
    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error("Error broadcasting message", error=str(e))
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


# Dependency to get orchestrator
async def get_orchestrator() -> AgentOrchestrator:
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "MSP Intelligence Mesh Network API"
    }


# Agent status endpoints
@app.get("/agents/status")
async def get_agent_status(orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Get status of all agents"""
    try:
        status = await orch.get_agent_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error("Error getting agent status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/{agent_type}/status")
async def get_specific_agent_status(agent_type: str, orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Get status of specific agent"""
    try:
        status = await orch.get_agent_status(agent_type)
        return JSONResponse(content=status)
    except Exception as e:
        logger.error("Error getting agent status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Threat Intelligence endpoints
@app.post("/threat-intelligence/analyze")
async def analyze_threat(request: ThreatAnalysisRequest, orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Analyze threat using threat intelligence agent"""
    try:
        if "threat_intelligence" not in orch.agents:
            raise HTTPException(status_code=503, detail="Threat intelligence agent not available")
        
        threat_agent = orch.agents["threat_intelligence"]
        result = await threat_agent.process_request({
            "type": "analyze_threat",
            "data": {
                "content": request.content,
                "threat_type": request.threat_type,
                "severity": request.severity
            }
        })
        
        # Broadcast threat analysis to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "threat_analysis",
            "data": result.data,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        return JSONResponse(content=result.data)
    except Exception as e:
        logger.error("Error analyzing threat", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threat-intelligence/active-threats")
async def get_active_threats(orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Get all active threats"""
    try:
        if "threat_intelligence" not in orch.agents:
            raise HTTPException(status_code=503, detail="Threat intelligence agent not available")
        
        threat_agent = orch.agents["threat_intelligence"]
        result = await threat_agent.process_request({"type": "get_active_threats"})
        
        return JSONResponse(content=result.data)
    except Exception as e:
        logger.error("Error getting active threats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Collaboration endpoints
@app.post("/collaboration/find-partners")
async def find_partners(request: CollaborationRequest, orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Find compatible partners for collaboration"""
    try:
        if "collaboration_matching" not in orch.agents:
            raise HTTPException(status_code=503, detail="Collaboration agent not available")
        
        collab_agent = orch.agents["collaboration_matching"]
        result = await collab_agent.process_request({
            "type": "find_partners",
            "msp_id": request.msp_id,
            "requirements": request.requirements
        })
        
        return JSONResponse(content=result.data)
    except Exception as e:
        logger.error("Error finding partners", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collaboration/opportunities")
async def get_collaboration_opportunities(orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Get all collaboration opportunities"""
    try:
        if "collaboration_matching" not in orch.agents:
            raise HTTPException(status_code=503, detail="Collaboration agent not available")
        
        collab_agent = orch.agents["collaboration_matching"]
        result = await collab_agent.process_request({"type": "get_collaboration_opportunities"})
        
        return JSONResponse(content=result.data)
    except Exception as e:
        logger.error("Error getting collaboration opportunities", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Federated Learning endpoints
@app.post("/federated-learning/start-round")
async def start_federated_learning_round(request: FederatedLearningRequest, orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Start a federated learning training round"""
    try:
        if "federated_learning" not in orch.agents:
            raise HTTPException(status_code=503, detail="Federated learning agent not available")
        
        fl_agent = orch.agents["federated_learning"]
        result = await fl_agent.process_request({
            "type": "start_training_round",
            "participants": request.participants
        })
        
        # Broadcast federated learning update
        await manager.broadcast(json.dumps({
            "type": "federated_learning_update",
            "data": result.data,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        return JSONResponse(content=result.data)
    except Exception as e:
        logger.error("Error starting federated learning round", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/federated-learning/status")
async def get_federated_learning_status(orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Get federated learning status"""
    try:
        if "federated_learning" not in orch.agents:
            raise HTTPException(status_code=503, detail="Federated learning agent not available")
        
        fl_agent = orch.agents["federated_learning"]
        result = await fl_agent.process_request({"type": "get_training_status"})
        
        return JSONResponse(content=result.data)
    except Exception as e:
        logger.error("Error getting federated learning status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/federated-learning/privacy-metrics")
async def get_privacy_metrics(orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Get privacy protection metrics"""
    try:
        if "federated_learning" not in orch.agents:
            raise HTTPException(status_code=503, detail="Federated learning agent not available")
        
        fl_agent = orch.agents["federated_learning"]
        result = await fl_agent.process_request({"type": "get_privacy_metrics"})
        
        return JSONResponse(content=result.data)
    except Exception as e:
        logger.error("Error getting privacy metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Workflow endpoints
@app.post("/workflows/execute")
async def execute_workflow(request: WorkflowRequest, orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Execute a predefined workflow"""
    try:
        result = await orch.execute_workflow(request.workflow_name, request.workflow_data)
        
        # Broadcast workflow completion
        await manager.broadcast(json.dumps({
            "type": "workflow_completed",
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error executing workflow", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Simulation endpoints
@app.post("/simulation/network-activity")
async def simulate_network_activity(orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Simulate realistic network activity"""
    try:
        result = await orch.simulate_network_activity()
        
        # Broadcast simulation results
        await manager.broadcast(json.dumps({
            "type": "network_simulation",
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error simulating network activity", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulation/threat-detection")
async def simulate_threat_detection(orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Simulate threat detection scenario"""
    try:
        if "threat_intelligence" not in orch.agents:
            raise HTTPException(status_code=503, detail="Threat intelligence agent not available")
        
        threat_agent = orch.agents["threat_intelligence"]
        result = await threat_agent.simulate_threat_detection()
        
        # Broadcast threat detection
        await manager.broadcast(json.dumps({
            "type": "threat_detection_simulation",
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error simulating threat detection", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulation/collaboration-opportunity")
async def simulate_collaboration_opportunity(orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Simulate collaboration opportunity scenario"""
    try:
        if "collaboration_matching" not in orch.agents:
            raise HTTPException(status_code=503, detail="Collaboration agent not available")
        
        collab_agent = orch.agents["collaboration_matching"]
        result = await collab_agent.simulate_collaboration_opportunity()
        
        # Broadcast collaboration opportunity
        await manager.broadcast(json.dumps({
            "type": "collaboration_opportunity_simulation",
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error simulating collaboration opportunity", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulation/federated-learning")
async def simulate_federated_learning(orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Simulate federated learning scenario"""
    try:
        if "federated_learning" not in orch.agents:
            raise HTTPException(status_code=503, detail="Federated learning agent not available")
        
        fl_agent = orch.agents["federated_learning"]
        result = await fl_agent._simulate_training_round()
        
        # Broadcast federated learning simulation
        await manager.broadcast(json.dumps({
            "type": "federated_learning_simulation",
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Error simulating federated learning", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            logger.info("WebSocket message received", message_type=message.get("type"))
            
            # Handle different message types
            message_type = message.get("type")
            message_data = message.get("data", {})
            
            if message_type == "ping":
                await manager.send_personal_message(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }), websocket)
            
            elif message_type == "subscribe":
                # Handle subscription to specific events
                subscription_type = message_data.get("event_type")
                await manager.send_personal_message(json.dumps({
                    "type": "subscription_confirmed",
                    "event_type": subscription_type,
                    "timestamp": datetime.utcnow().isoformat()
                }), websocket)
            
            elif message_type == "request_agent_status":
                # Send current agent status
                if orchestrator:
                    status = await orchestrator.get_agent_status()
                    await manager.send_personal_message(json.dumps({
                        "type": "agent_status",
                        "data": status,
                        "timestamp": datetime.utcnow().isoformat()
                    }), websocket)
            
            else:
                # Echo back unknown message types
                await manager.send_personal_message(json.dumps({
                    "type": "echo",
                    "original_message": message,
                    "timestamp": datetime.utcnow().isoformat()
                }), websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        manager.disconnect(websocket)


# Metrics endpoint
@app.get("/metrics")
async def get_metrics(orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Get system metrics and performance data"""
    try:
        metrics = orch.get_orchestration_metrics()
        
        # Add API-specific metrics
        api_metrics = {
            "active_websocket_connections": len(manager.active_connections),
            "api_uptime": "running",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        metrics["api_metrics"] = api_metrics
        
        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error("Error getting metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MSP Intelligence Mesh Network API",
        "version": "1.0.0",
        "description": "Revolutionary collective intelligence platform for Managed Service Providers",
        "endpoints": {
            "health": "/health",
            "agents": "/agents/status",
            "threat_intelligence": "/threat-intelligence/analyze",
            "collaboration": "/collaboration/find-partners",
            "federated_learning": "/federated-learning/status",
            "workflows": "/workflows/execute",
            "simulation": "/simulation/network-activity",
            "websocket": "/ws",
            "metrics": "/metrics"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

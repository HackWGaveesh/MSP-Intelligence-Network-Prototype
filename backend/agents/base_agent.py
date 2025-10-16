"""
Base Agent class for MSP Intelligence Mesh Network
All specialized agents inherit from this base class
"""
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import json

from pydantic import BaseModel
import structlog

from config.settings import settings, AGENT_CONFIGS


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


class AgentMessage(BaseModel):
    """Standard message format for inter-agent communication"""
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    priority: int = 1  # 1=low, 5=high


class AgentResponse(BaseModel):
    """Standard response format from agents"""
    agent_id: str
    success: bool
    data: Dict[str, Any]
    confidence: float
    processing_time_ms: int
    timestamp: datetime
    error_message: Optional[str] = None


class AgentMetrics(BaseModel):
    """Performance metrics for agents"""
    agent_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    last_activity: datetime
    health_score: float = 1.0


class BaseAgent(ABC):
    """
    Base class for all AI agents in the MSP Intelligence Mesh Network
    Provides common functionality for model management, communication, and metrics
    """
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = AGENT_CONFIGS.get(agent_type, {})
        self.logger = logger.bind(agent_id=agent_id, agent_type=agent_type)
        
        # Performance tracking
        self.metrics = AgentMetrics(
            agent_id=agent_id,
            last_activity=datetime.utcnow()
        )
        
        # Model and state management
        self.model = None
        self.model_loaded = False
        self.last_model_update = None
        
        # Communication
        self.message_queue = asyncio.Queue()
        self.subscribers: List[str] = []
        
        # Health monitoring
        self.health_check_interval = 60  # seconds
        self.last_health_check = datetime.utcnow()
        
        self.logger.info("Agent initialized", config=self.config)
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> AgentResponse:
        """
        Process a request and return a response
        Must be implemented by each specialized agent
        """
        pass
    
    @abstractmethod
    async def load_model(self) -> bool:
        """
        Load the AI model for this agent
        Must be implemented by each specialized agent
        """
        pass
    
    async def initialize(self) -> bool:
        """Initialize the agent and load its model"""
        try:
            self.logger.info("Initializing agent")
            
            # Load the model
            model_loaded = await self.load_model()
            if not model_loaded:
                self.logger.error("Failed to load model")
                return False
            
            self.model_loaded = True
            self.last_model_update = datetime.utcnow()
            
            # Start background tasks
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._message_processor())
            
            self.logger.info("Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize agent", error=str(e))
            return False
    
    async def _health_monitor(self):
        """Monitor agent health and performance"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Calculate health score based on recent performance
                if self.metrics.total_requests > 0:
                    success_rate = self.metrics.successful_requests / self.metrics.total_requests
                    self.metrics.health_score = success_rate
                
                self.metrics.last_activity = datetime.utcnow()
                self.last_health_check = datetime.utcnow()
                
                self.logger.debug("Health check completed", 
                                health_score=self.metrics.health_score,
                                total_requests=self.metrics.total_requests)
                
            except Exception as e:
                self.logger.error("Health monitor error", error=str(e))
    
    async def _message_processor(self):
        """Process incoming messages from other agents"""
        while True:
            try:
                # Wait for messages with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                await self._handle_message(message)
                
            except asyncio.TimeoutError:
                # No messages, continue
                continue
            except Exception as e:
                self.logger.error("Message processing error", error=str(e))
    
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming messages from other agents"""
        try:
            self.logger.info("Processing message", 
                           sender=message.sender,
                           message_type=message.message_type)
            
            # Route message to appropriate handler
            if message.message_type == "collaboration_request":
                await self._handle_collaboration_request(message)
            elif message.message_type == "data_update":
                await self._handle_data_update(message)
            elif message.message_type == "model_update":
                await self._handle_model_update(message)
            else:
                self.logger.warning("Unknown message type", 
                                  message_type=message.message_type)
                
        except Exception as e:
            self.logger.error("Error handling message", error=str(e))
    
    async def _handle_collaboration_request(self, message: AgentMessage):
        """Handle collaboration requests from other agents"""
        # Default implementation - can be overridden
        self.logger.info("Received collaboration request", 
                        from_agent=message.sender)
    
    async def _handle_data_update(self, message: AgentMessage):
        """Handle data updates from other agents"""
        # Default implementation - can be overridden
        self.logger.info("Received data update", 
                        from_agent=message.sender)
    
    async def _handle_model_update(self, message: AgentMessage):
        """Handle model updates from federated learning"""
        # Default implementation - can be overridden
        self.logger.info("Received model update", 
                        from_agent=message.sender)
    
    async def send_message(self, recipient: str, message_type: str, 
                          payload: Dict[str, Any], priority: int = 1) -> bool:
        """Send a message to another agent"""
        try:
            message = AgentMessage(
                sender=self.agent_id,
                recipient=recipient,
                message_type=message_type,
                payload=payload,
                timestamp=datetime.utcnow(),
                priority=priority
            )
            
            # In a real implementation, this would send to a message broker
            # For now, we'll simulate by logging
            self.logger.info("Sending message", 
                           recipient=recipient,
                           message_type=message_type,
                           priority=priority)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to send message", error=str(e))
            return False
    
    async def broadcast_message(self, message_type: str, payload: Dict[str, Any]) -> bool:
        """Broadcast a message to all subscribers"""
        try:
            for subscriber in self.subscribers:
                await self.send_message(subscriber, message_type, payload)
            
            self.logger.info("Broadcasted message", 
                           message_type=message_type,
                           subscribers=len(self.subscribers))
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to broadcast message", error=str(e))
            return False
    
    def update_metrics(self, success: bool, processing_time_ms: int):
        """Update agent performance metrics"""
        self.metrics.total_requests += 1
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Update average response time
        if self.metrics.total_requests == 1:
            self.metrics.average_response_time_ms = processing_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_response_time_ms = (
                alpha * processing_time_ms + 
                (1 - alpha) * self.metrics.average_response_time_ms
            )
        
        self.metrics.last_activity = datetime.utcnow()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the agent"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "model_loaded": self.model_loaded,
            "health_score": self.metrics.health_score,
            "total_requests": self.metrics.total_requests,
            "success_rate": (
                self.metrics.successful_requests / self.metrics.total_requests 
                if self.metrics.total_requests > 0 else 0
            ),
            "average_response_time_ms": self.metrics.average_response_time_ms,
            "last_activity": self.metrics.last_activity.isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.metrics.last_activity).total_seconds()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.config.get("model_name", "unknown"),
            "max_size_mb": self.config.get("max_size_mb", 0),
            "model_loaded": self.model_loaded,
            "last_update": self.last_model_update.isoformat() if self.last_model_update else None,
            "confidence_threshold": self.config.get("confidence_threshold", 0.5)
        }
    
    async def shutdown(self):
        """Gracefully shutdown the agent"""
        self.logger.info("Shutting down agent")
        
        # Clean up resources
        if self.model is not None:
            # Model cleanup would go here
            pass
        
        self.logger.info("Agent shutdown complete")

"""
Agent Orchestrator for MSP Intelligence Mesh Network
Coordinates all AI agents and manages inter-agent communication
"""
import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import structlog

from agents.base_agent import BaseAgent, AgentResponse, AgentMessage
from agents.threat_intelligence_agent import ThreatIntelligenceAgent
from agents.collaboration_agent import CollaborationAgent
from agents.federated_learning_agent import FederatedLearningAgent
from agents.market_intelligence_agent import MarketIntelligenceAgent
from agents.client_health_agent import ClientHealthAgent
from agents.revenue_optimization_agent import RevenueOptimizationAgent
from agents.anomaly_detection_agent import AnomalyDetectionAgent
from agents.nlp_query_agent import NLPQueryAgent
from agents.resource_allocation_agent import ResourceAllocationAgent
from agents.security_compliance_agent import SecurityComplianceAgent
from config.settings import settings


logger = structlog.get_logger()


class AgentOrchestrator:
    """
    Central orchestrator that manages all AI agents in the MSP Intelligence Mesh Network
    Handles agent lifecycle, inter-agent communication, and workflow coordination
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_health: Dict[str, Dict] = {}
        self.workflow_queue: asyncio.Queue = asyncio.Queue()
        self.message_broker: Dict[str, List[asyncio.Queue]] = {}
        
        # Agent registry
        self.agent_registry = {
            "threat_intelligence": ThreatIntelligenceAgent,
            "collaboration_matching": CollaborationAgent,
            "federated_learning": FederatedLearningAgent,
            "market_intelligence": MarketIntelligenceAgent,
            "client_health": ClientHealthAgent,
            "revenue_optimization": RevenueOptimizationAgent,
            "anomaly_detection": AnomalyDetectionAgent,
            "nlp_query": NLPQueryAgent,
            "resource_allocation": ResourceAllocationAgent,
            "security_compliance": SecurityComplianceAgent,
        }
        
        # Workflow definitions
        self.workflows = {
            "threat_response": [
                "threat_intelligence",
                "collaboration_matching",
                "federated_learning"
            ],
            "collaboration_opportunity": [
                "collaboration_matching",
                "threat_intelligence",
                "federated_learning"
            ],
            "model_training": [
                "federated_learning",
                "threat_intelligence",
                "collaboration_matching"
            ]
        }
        
        # Performance metrics
        self.orchestration_metrics = {
            "total_workflows_executed": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_workflow_time_ms": 0.0,
            "active_agents": 0,
            "last_health_check": datetime.utcnow()
        }
        
        self.logger = logger.bind(component="orchestrator")
        self.logger.info("Agent Orchestrator initialized")
    
    async def initialize(self) -> bool:
        """Initialize all agents and start orchestration services"""
        try:
            self.logger.info("Initializing agent orchestrator")
            
            # Initialize all agents
            for agent_type, agent_class in self.agent_registry.items():
                agent = agent_class()
                success = await agent.initialize()
                
                if success:
                    self.agents[agent_type] = agent
                    self.agent_health[agent_type] = agent.get_health_status()
                    self.message_broker[agent_type] = []
                    self.logger.info("Agent initialized", agent_type=agent_type)
                else:
                    self.logger.error("Failed to initialize agent", agent_type=agent_type)
                    return False
            
            # Start orchestration services
            asyncio.create_task(self._workflow_processor())
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._message_router())
            
            self.orchestration_metrics["active_agents"] = len(self.agents)
            self.logger.info("Agent orchestrator initialized successfully", 
                           active_agents=len(self.agents))
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize orchestrator", error=str(e))
            return False
    
    async def execute_workflow(self, workflow_name: str, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a predefined workflow across multiple agents"""
        start_time = datetime.utcnow()
        
        try:
            if workflow_name not in self.workflows:
                return {"error": f"Unknown workflow: {workflow_name}"}
            
            self.logger.info("Executing workflow", workflow_name=workflow_name)
            
            workflow_steps = self.workflows[workflow_name]
            workflow_results = {}
            workflow_context = {"data": workflow_data, "results": {}}
            
            # Execute workflow steps sequentially
            for step_index, agent_type in enumerate(workflow_steps):
                if agent_type not in self.agents:
                    self.logger.error("Agent not available", agent_type=agent_type)
                    continue
                
                agent = self.agents[agent_type]
                
                # Prepare step data
                step_data = {
                    "workflow_name": workflow_name,
                    "step_index": step_index,
                    "context": workflow_context,
                    "data": workflow_data
                }
                
                # Execute agent step
                step_result = await agent.process_request(step_data)
                
                if step_result.success:
                    workflow_results[agent_type] = step_result.data
                    workflow_context["results"][agent_type] = step_result.data
                    self.logger.info("Workflow step completed", 
                                   workflow=workflow_name,
                                   step=agent_type,
                                   success=True)
                else:
                    self.logger.error("Workflow step failed", 
                                    workflow=workflow_name,
                                    step=agent_type,
                                    error=step_result.error_message)
                    workflow_results[agent_type] = {"error": step_result.error_message}
            
            # Calculate workflow metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            success = all("error" not in result for result in workflow_results.values())
            
            # Update metrics
            self.orchestration_metrics["total_workflows_executed"] += 1
            if success:
                self.orchestration_metrics["successful_workflows"] += 1
            else:
                self.orchestration_metrics["failed_workflows"] += 1
            
            # Update average workflow time
            total_workflows = self.orchestration_metrics["total_workflows_executed"]
            current_avg = self.orchestration_metrics["average_workflow_time_ms"]
            self.orchestration_metrics["average_workflow_time_ms"] = (
                (current_avg * (total_workflows - 1) + processing_time) / total_workflows
            )
            
            result = {
                "workflow_name": workflow_name,
                "success": success,
                "results": workflow_results,
                "processing_time_ms": processing_time,
                "steps_executed": len(workflow_steps),
                "completion_time": datetime.utcnow().isoformat()
            }
            
            self.logger.info("Workflow execution completed", 
                           workflow=workflow_name,
                           success=success,
                           processing_time=processing_time)
            
            return result
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.logger.error("Workflow execution failed", 
                            workflow=workflow_name,
                            error=str(e),
                            processing_time=processing_time)
            
            return {
                "workflow_name": workflow_name,
                "success": False,
                "error": str(e),
                "processing_time_ms": processing_time,
                "completion_time": datetime.utcnow().isoformat()
            }
    
    async def send_agent_message(self, sender: str, recipient: str, message_type: str, 
                               payload: Dict[str, Any], priority: int = 1) -> bool:
        """Send a message between agents"""
        try:
            if recipient not in self.agents:
                self.logger.error("Recipient agent not found", recipient=recipient)
                return False
            
            message = AgentMessage(
                sender=sender,
                recipient=recipient,
                message_type=message_type,
                payload=payload,
                timestamp=datetime.utcnow(),
                priority=priority
            )
            
            # Route message to recipient agent
            recipient_agent = self.agents[recipient]
            await recipient_agent.message_queue.put(message)
            
            self.logger.info("Message sent", 
                           sender=sender,
                           recipient=recipient,
                           message_type=message_type,
                           priority=priority)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to send message", error=str(e))
            return False
    
    async def broadcast_message(self, sender: str, message_type: str, 
                              payload: Dict[str, Any], exclude_sender: bool = True) -> int:
        """Broadcast a message to all agents"""
        try:
            sent_count = 0
            
            for agent_type, agent in self.agents.items():
                if exclude_sender and agent_type == sender:
                    continue
                
                success = await self.send_agent_message(sender, agent_type, message_type, payload)
                if success:
                    sent_count += 1
            
            self.logger.info("Message broadcasted", 
                           sender=sender,
                           message_type=message_type,
                           recipients=sent_count)
            
            return sent_count
            
        except Exception as e:
            self.logger.error("Failed to broadcast message", error=str(e))
            return 0
    
    async def get_agent_status(self, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """Get status of specific agent or all agents"""
        try:
            if agent_type:
                if agent_type in self.agents:
                    agent = self.agents[agent_type]
                    return {
                        "agent_type": agent_type,
                        "status": agent.get_health_status(),
                        "model_info": agent.get_model_info()
                    }
                else:
                    return {"error": f"Agent {agent_type} not found"}
            else:
                # Return status of all agents
                all_status = {}
                for agent_type, agent in self.agents.items():
                    all_status[agent_type] = {
                        "status": agent.get_health_status(),
                        "model_info": agent.get_model_info()
                    }
                
                return {
                    "agents": all_status,
                    "orchestration_metrics": self.orchestration_metrics,
                    "total_agents": len(self.agents),
                    "active_agents": len([a for a in self.agents.values() if a.model_loaded]),
                    "status_time": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error("Error getting agent status", error=str(e))
            return {"error": str(e)}
    
    async def simulate_network_activity(self) -> Dict[str, Any]:
        """Simulate realistic network activity across all agents"""
        try:
            self.logger.info("Starting network activity simulation")
            
            simulation_results = {}
            
            # Simulate threat detection and response
            if "threat_intelligence" in self.agents:
                threat_agent = self.agents["threat_intelligence"]
                threat_result = await threat_agent.simulate_threat_detection()
                simulation_results["threat_detection"] = threat_result
            
            # Simulate collaboration opportunity
            if "collaboration_matching" in self.agents:
                collab_agent = self.agents["collaboration_matching"]
                collab_result = await collab_agent.simulate_collaboration_opportunity()
                simulation_results["collaboration_opportunity"] = collab_result
            
            # Simulate federated learning round
            if "federated_learning" in self.agents:
                fl_agent = self.agents["federated_learning"]
                fl_result = await fl_agent._simulate_training_round()
                simulation_results["federated_learning"] = fl_result
            
            # Execute cross-agent workflows
            workflow_results = {}
            
            # Threat response workflow
            threat_workflow = await self.execute_workflow("threat_response", {
                "threat_type": "ransomware",
                "severity": "high",
                "affected_systems": ["server_01", "workstation_05"]
            })
            workflow_results["threat_response"] = threat_workflow
            
            # Collaboration opportunity workflow
            collab_workflow = await self.execute_workflow("collaboration_opportunity", {
                "opportunity_type": "enterprise_rfp",
                "value": 2500000,
                "requirements": ["cloud_services", "security", "compliance"]
            })
            workflow_results["collaboration_opportunity"] = collab_workflow
            
            # Get final system status
            system_status = await self.get_agent_status()
            
            result = {
                "simulation_time": datetime.utcnow().isoformat(),
                "individual_agent_results": simulation_results,
                "workflow_results": workflow_results,
                "system_status": system_status,
                "network_metrics": {
                    "total_msps_simulated": random.randint(800, 1200),
                    "threats_detected": random.randint(5, 15),
                    "collaborations_initiated": random.randint(3, 8),
                    "models_trained": random.randint(1, 3),
                    "network_intelligence_level": random.uniform(0.85, 0.98)
                }
            }
            
            self.logger.info("Network activity simulation completed")
            
            return result
            
        except Exception as e:
            self.logger.error("Error in network activity simulation", error=str(e))
            return {"error": str(e)}
    
    async def _workflow_processor(self):
        """Process workflow queue in background"""
        while True:
            try:
                # Wait for workflows with timeout
                workflow_item = await asyncio.wait_for(
                    self.workflow_queue.get(),
                    timeout=1.0
                )
                
                workflow_name = workflow_item.get("name")
                workflow_data = workflow_item.get("data", {})
                
                # Execute workflow
                result = await self.execute_workflow(workflow_name, workflow_data)
                
                # Store result or send to callback
                if "callback" in workflow_item:
                    callback = workflow_item["callback"]
                    await callback(result)
                
            except asyncio.TimeoutError:
                # No workflows in queue, continue
                continue
            except Exception as e:
                self.logger.error("Error processing workflow", error=str(e))
    
    async def _health_monitor(self):
        """Monitor health of all agents"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for agent_type, agent in self.agents.items():
                    health_status = agent.get_health_status()
                    self.agent_health[agent_type] = health_status
                    
                    # Check for unhealthy agents
                    if health_status["health_score"] < 0.5:
                        self.logger.warning("Agent health low", 
                                          agent_type=agent_type,
                                          health_score=health_status["health_score"])
                
                self.orchestration_metrics["last_health_check"] = datetime.utcnow()
                
            except Exception as e:
                self.logger.error("Error in health monitoring", error=str(e))
    
    async def _message_router(self):
        """Route messages between agents"""
        while True:
            try:
                await asyncio.sleep(0.1)  # Check frequently
                
                # Process messages for each agent
                for agent_type, agent in self.agents.items():
                    try:
                        # Check if agent has messages to process
                        if not agent.message_queue.empty():
                            # Agent will process its own messages
                            continue
                    except Exception as e:
                        self.logger.error("Error checking message queue", 
                                        agent_type=agent_type,
                                        error=str(e))
                
            except Exception as e:
                self.logger.error("Error in message routing", error=str(e))
    
    async def shutdown(self):
        """Gracefully shutdown all agents and orchestrator"""
        self.logger.info("Shutting down agent orchestrator")
        
        # Shutdown all agents
        for agent_type, agent in self.agents.items():
            try:
                await agent.shutdown()
                self.logger.info("Agent shutdown", agent_type=agent_type)
            except Exception as e:
                self.logger.error("Error shutting down agent", 
                                agent_type=agent_type,
                                error=str(e))
        
        self.logger.info("Agent orchestrator shutdown complete")
    
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics"""
        return {
            "orchestration_metrics": self.orchestration_metrics,
            "agent_health": self.agent_health,
            "workflow_definitions": list(self.workflows.keys()),
            "registered_agents": list(self.agents.keys()),
            "message_broker_status": {
                agent_type: len(queues) for agent_type, queues in self.message_broker.items()
            }
        }

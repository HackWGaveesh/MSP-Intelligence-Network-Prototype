"""
Test Suite for MSP Intelligence Mesh Network Agents
Comprehensive testing for all AI agents
"""
import pytest
import asyncio
import json
from datetime import datetime
from typing import Dict, Any

# Import all agents
from backend.agents.threat_intelligence_agent import ThreatIntelligenceAgent
from backend.agents.collaboration_agent import CollaborationAgent
from backend.agents.federated_learning_agent import FederatedLearningAgent
from backend.agents.market_intelligence_agent import MarketIntelligenceAgent
from backend.agents.client_health_agent import ClientHealthAgent
from backend.agents.revenue_optimization_agent import RevenueOptimizationAgent
from backend.agents.anomaly_detection_agent import AnomalyDetectionAgent
from backend.agents.nlp_query_agent import NLPQueryAgent
from backend.agents.resource_allocation_agent import ResourceAllocationAgent
from backend.agents.security_compliance_agent import SecurityComplianceAgent


class TestAgentBase:
    """Base test class for all agents"""
    
    @pytest.fixture
    def sample_threat_data(self):
        return {
            "threat_type": "ransomware",
            "severity": "high",
            "indicators": ["malicious_hash_123", "suspicious_domain.com"],
            "affected_systems": ["server_01", "workstation_05"]
        }
    
    @pytest.fixture
    def sample_collaboration_data(self):
        return {
            "opportunity_type": "enterprise_rfp",
            "value": 2500000,
            "required_skills": ["cloud_services", "security", "compliance"],
            "industry": "healthcare"
        }
    
    @pytest.fixture
    def sample_client_data(self):
        return {
            "client_id": "client_001",
            "interaction_history": [
                {"type": "support_ticket", "sentiment": "negative", "timestamp": "2024-01-15T10:30:00Z"}
            ],
            "billing_history": [
                {"month": "Jan", "amount": 1500, "status": "paid"}
            ]
        }


class TestThreatIntelligenceAgent(TestAgentBase):
    """Test Threat Intelligence Agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        agent = ThreatIntelligenceAgent()
        assert agent.agent_id == "threat_intelligence_agent"
        assert agent.model_loaded is True
    
    @pytest.mark.asyncio
    async def test_threat_analysis(self, sample_threat_data):
        agent = ThreatIntelligenceAgent()
        await agent.initialize()
        
        result = await agent.process_request({
            "type": "analyze_threat",
            "data": sample_threat_data
        })
        
        assert result.success is True
        assert "threat_id" in result.data
        assert "confidence" in result.data
        assert "recommended_actions" in result.data
    
    @pytest.mark.asyncio
    async def test_threat_detection_simulation(self):
        agent = ThreatIntelligenceAgent()
        await agent.initialize()
        
        result = await agent.simulate_threat_detection()
        
        assert "threats_detected" in result
        assert "network_response" in result
        assert "prevention_value" in result


class TestCollaborationAgent(TestAgentBase):
    """Test Collaboration Matching Agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        agent = CollaborationAgent()
        assert agent.agent_id == "collaboration_agent"
        assert agent.model_loaded is True
    
    @pytest.mark.asyncio
    async def test_opportunity_matching(self, sample_collaboration_data):
        agent = CollaborationAgent()
        await agent.initialize()
        
        result = await agent.process_request({
            "type": "match_opportunity",
            "data": sample_collaboration_data
        })
        
        assert result.success is True
        assert "matched_partners" in result.data
        assert "compatibility_scores" in result.data
        assert "joint_proposal" in result.data
    
    @pytest.mark.asyncio
    async def test_collaboration_simulation(self):
        agent = CollaborationAgent()
        await agent.initialize()
        
        result = await agent.simulate_collaboration_opportunity()
        
        assert "opportunities_generated" in result
        assert "partnerships_formed" in result
        assert "revenue_potential" in result


class TestFederatedLearningAgent(TestAgentBase):
    """Test Federated Learning Agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        agent = FederatedLearningAgent()
        assert agent.agent_id == "federated_learning_agent"
        assert agent.model_loaded is True
    
    @pytest.mark.asyncio
    async def test_training_round(self):
        agent = FederatedLearningAgent()
        await agent.initialize()
        
        result = await agent.process_request({
            "type": "training_round",
            "data": {"participants": ["msp_001", "msp_002", "msp_003"]}
        })
        
        assert result.success is True
        assert "global_model_update" in result.data
        assert "privacy_metrics" in result.data
        assert "accuracy_improvement" in result.data
    
    @pytest.mark.asyncio
    async def test_privacy_guarantees(self):
        agent = FederatedLearningAgent()
        await agent.initialize()
        
        result = await agent.process_request({
            "type": "privacy_metrics",
            "data": {"epsilon": 0.1, "delta": 1e-5}
        })
        
        assert result.success is True
        assert "privacy_budget" in result.data
        assert "differential_privacy" in result.data


class TestMarketIntelligenceAgent(TestAgentBase):
    """Test Market Intelligence Agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        agent = MarketIntelligenceAgent()
        assert agent.agent_id == "market_intelligence_agent"
        assert agent.model_loaded is True
    
    @pytest.mark.asyncio
    async def test_market_analysis(self):
        agent = MarketIntelligenceAgent()
        await agent.initialize()
        
        result = await agent.process_request({
            "type": "analyze_market",
            "data": {"query": "latest market trends", "industry_segment": "all"}
        })
        
        assert result.success is True
        assert "trends" in result.data
        assert "pricing_recommendations" in result.data
        assert "competitive_analysis" in result.data


class TestClientHealthAgent(TestAgentBase):
    """Test Client Health Agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        agent = ClientHealthAgent()
        assert agent.agent_id == "client_health_agent"
        assert agent.model_loaded is True
    
    @pytest.mark.asyncio
    async def test_client_health_prediction(self, sample_client_data):
        agent = ClientHealthAgent()
        await agent.initialize()
        
        result = await agent.process_request({
            "type": "predict_health",
            "data": sample_client_data
        })
        
        assert result.success is True
        assert "churn_probability" in result.data
        assert "health_score" in result.data
        assert "risk_level" in result.data
        assert "intervention_recommendations" in result.data


class TestRevenueOptimizationAgent(TestAgentBase):
    """Test Revenue Optimization Agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        agent = RevenueOptimizationAgent()
        assert agent.agent_id == "revenue_optimization_agent"
        assert agent.model_loaded is True
    
    @pytest.mark.asyncio
    async def test_revenue_forecasting(self):
        agent = RevenueOptimizationAgent()
        await agent.initialize()
        
        result = await agent.process_request({
            "type": "forecast_revenue",
            "data": {"msp_id": "msp_001", "forecast_months": 6}
        })
        
        assert result.success is True
        assert "forecasted_revenue" in result.data
        assert "opportunities" in result.data
        assert "total_forecasted_value" in result.data


class TestAnomalyDetectionAgent(TestAgentBase):
    """Test Anomaly Detection Agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        agent = AnomalyDetectionAgent()
        assert agent.agent_id == "anomaly_detection_agent"
        assert agent.model_loaded is True
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self):
        agent = AnomalyDetectionAgent()
        await agent.initialize()
        
        result = await agent.process_request({
            "type": "detect_anomaly",
            "data": {
                "system_id": "system_001",
                "metric_data": {"cpu_usage": 85, "memory_usage": 90, "network_latency": 120}
            }
        })
        
        assert result.success is True
        assert "is_anomaly" in result.data
        assert "anomaly_score" in result.data
        assert "anomaly_type" in result.data
        assert "recommended_actions" in result.data


class TestNLPQueryAgent(TestAgentBase):
    """Test NLP Query Agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        agent = NLPQueryAgent()
        assert agent.agent_id == "nlp_query_agent"
        assert agent.model_loaded is True
    
    @pytest.mark.asyncio
    async def test_natural_language_query(self):
        agent = NLPQueryAgent()
        await agent.initialize()
        
        result = await agent.process_request({
            "type": "process_query",
            "data": {
                "query": "What are the current threats?",
                "user_context": {"user_id": "user_001", "msp_id": "msp_001"}
            }
        })
        
        assert result.success is True
        assert "response" in result.data
        assert "confidence" in result.data
        assert "source_agent" in result.data


class TestResourceAllocationAgent(TestAgentBase):
    """Test Resource Allocation Agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        agent = ResourceAllocationAgent()
        assert agent.agent_id == "resource_allocation_agent"
        assert agent.model_loaded is True
    
    @pytest.mark.asyncio
    async def test_resource_optimization(self):
        agent = ResourceAllocationAgent()
        await agent.initialize()
        
        result = await agent.process_request({
            "type": "optimize_resources",
            "data": {
                "technicians": [
                    {"id": "tech_001", "skills": ["networking", "security"], "availability": "full"}
                ],
                "projects": [
                    {"id": "proj_001", "requirements": ["networking", "security"], "deadline": "2024-02-15"}
                ]
            }
        })
        
        assert result.success is True
        assert "optimized_assignments" in result.data
        assert "efficiency_score" in result.data
        assert "schedule" in result.data


class TestSecurityComplianceAgent(TestAgentBase):
    """Test Security Compliance Agent"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        agent = SecurityComplianceAgent()
        assert agent.agent_id == "security_compliance_agent"
        assert agent.model_loaded is True
    
    @pytest.mark.asyncio
    async def test_compliance_check(self):
        agent = SecurityComplianceAgent()
        await agent.initialize()
        
        result = await agent.process_request({
            "type": "check_compliance",
            "data": {"framework": "SOC2"}
        })
        
        assert result.success is True
        assert "compliance_score" in result.data
        assert "status" in result.data
        assert "gaps" in result.data


class TestAgentIntegration:
    """Test agent integration and orchestration"""
    
    @pytest.mark.asyncio
    async def test_agent_communication(self):
        """Test communication between agents"""
        threat_agent = ThreatIntelligenceAgent()
        collab_agent = CollaborationAgent()
        
        await threat_agent.initialize()
        await collab_agent.initialize()
        
        # Simulate threat detection triggering collaboration
        threat_result = await threat_agent.simulate_threat_detection()
        assert "threats_detected" in threat_result
        
        # Simulate collaboration opportunity
        collab_result = await collab_agent.simulate_collaboration_opportunity()
        assert "opportunities_generated" in collab_result
    
    @pytest.mark.asyncio
    async def test_all_agents_health(self):
        """Test health status of all agents"""
        agents = [
            ThreatIntelligenceAgent(),
            CollaborationAgent(),
            FederatedLearningAgent(),
            MarketIntelligenceAgent(),
            ClientHealthAgent(),
            RevenueOptimizationAgent(),
            AnomalyDetectionAgent(),
            NLPQueryAgent(),
            ResourceAllocationAgent(),
            SecurityComplianceAgent()
        ]
        
        for agent in agents:
            await agent.initialize()
            health = agent.get_health_status()
            assert health["status"] == "active"
            assert health["health_score"] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
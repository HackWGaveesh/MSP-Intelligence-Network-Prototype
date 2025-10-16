"""
Client Health Agent for MSP Intelligence Mesh Network
Provides client churn prediction, health scoring, and intervention recommendations
"""
import asyncio
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import structlog
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

from .base_agent import BaseAgent, AgentResponse, AgentMetrics


logger = structlog.get_logger()


class ClientHealthAgent(BaseAgent):
    """Client Health Agent for churn prediction and health scoring"""
    
    def __init__(self):
        super().__init__()
        self.agent_id = "client_health_agent"
        self.agent_type = "client_health"
        self.model_loaded = False
        self.churn_model = None
        self.health_model = None
        self.scaler = None
        self.label_encoder = None
        self.client_data = {}
        self.health_thresholds = {
            "critical": 0.3,
            "at_risk": 0.5,
            "healthy": 0.7,
            "excellent": 0.9
        }
        
        self.logger = logger.bind(agent=self.agent_id)
        self.logger.info("Client Health Agent initialized")
    
    async def initialize(self):
        """Initialize the agent and load models"""
        try:
            self.logger.info("Initializing Client Health Agent")
            
            # Initialize models
            self.churn_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            self.health_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
            self.scaler = StandardScaler()
            self.label_encoder = LabelEncoder()
            
            # Load or generate training data
            await self._load_training_data()
            
            # Train models
            await self._train_models()
            
            self.model_loaded = True
            self.logger.info("Client Health Agent initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize Client Health Agent", error=str(e))
            raise
    
    async def _load_training_data(self):
        """Load or generate training data for client health models"""
        # Generate synthetic client data for training
        np.random.seed(42)
        n_clients = 1000
        
        # Generate features
        data = {
            'client_id': [f'client_{i:04d}' for i in range(n_clients)],
            'tenure_months': np.random.normal(24, 12, n_clients).astype(int),
            'monthly_revenue': np.random.lognormal(8, 1, n_clients),
            'support_tickets': np.random.poisson(5, n_clients),
            'response_time_avg': np.random.exponential(2, n_clients),
            'satisfaction_score': np.random.beta(2, 2, n_clients),
            'contract_value': np.random.lognormal(9, 1, n_clients),
            'payment_delay_days': np.random.exponential(5, n_clients),
            'feature_usage': np.random.beta(3, 2, n_clients),
            'engagement_score': np.random.beta(2, 3, n_clients),
            'industry': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Manufacturing', 'Retail'], n_clients),
            'company_size': np.random.choice(['Small', 'Medium', 'Large', 'Enterprise'], n_clients)
        }
        
        # Create target variables
        # Churn probability based on multiple factors
        churn_prob = (
            (data['satisfaction_score'] < 0.3) * 0.4 +
            (data['support_tickets'] > 10) * 0.3 +
            (data['payment_delay_days'] > 15) * 0.2 +
            (data['engagement_score'] < 0.2) * 0.3 +
            (data['tenure_months'] < 6) * 0.2 +
            np.random.normal(0, 0.1, n_clients)
        )
        
        data['churn_probability'] = np.clip(churn_prob, 0, 1)
        data['churned'] = (data['churn_probability'] > 0.5).astype(int)
        
        # Health score based on multiple factors
        health_score = (
            data['satisfaction_score'] * 0.3 +
            (1 - np.clip(data['support_tickets'] / 20, 0, 1)) * 0.2 +
            (1 - np.clip(data['payment_delay_days'] / 30, 0, 1)) * 0.2 +
            data['feature_usage'] * 0.15 +
            data['engagement_score'] * 0.15
        )
        
        data['health_score'] = np.clip(health_score, 0, 1)
        
        # Convert to DataFrame
        self.training_data = pd.DataFrame(data)
        
        self.logger.info(f"Generated training data with {len(self.training_data)} clients")
    
    async def _train_models(self):
        """Train the churn prediction and health scoring models"""
        try:
            # Prepare features for churn prediction
            feature_columns = [
                'tenure_months', 'monthly_revenue', 'support_tickets', 
                'response_time_avg', 'satisfaction_score', 'contract_value',
                'payment_delay_days', 'feature_usage', 'engagement_score'
            ]
            
            # Add categorical features
            industry_encoded = pd.get_dummies(self.training_data['industry'], prefix='industry')
            size_encoded = pd.get_dummies(self.training_data['company_size'], prefix='size')
            
            X = pd.concat([
                self.training_data[feature_columns],
                industry_encoded,
                size_encoded
            ], axis=1)
            
            y_churn = self.training_data['churned']
            y_health = self.training_data['health_score']
            
            # Split data
            X_train, X_test, y_churn_train, y_churn_test = train_test_split(
                X, y_churn, test_size=0.2, random_state=42, stratify=y_churn
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train churn model
            self.churn_model.fit(X_train_scaled, y_churn_train)
            
            # Train health model (regression)
            from sklearn.ensemble import RandomForestRegressor
            self.health_model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
            self.health_model.fit(X_train_scaled, y_health.iloc[X_train.index])
            
            # Evaluate models
            churn_pred = self.churn_model.predict(X_test_scaled)
            churn_prob = self.churn_model.predict_proba(X_test_scaled)[:, 1]
            health_pred = self.health_model.predict(X_test_scaled)
            
            churn_accuracy = accuracy_score(y_churn_test, churn_pred)
            health_mae = np.mean(np.abs(health_pred - y_health.iloc[X_test.index]))
            
            self.logger.info(f"Churn model accuracy: {churn_accuracy:.3f}")
            self.logger.info(f"Health model MAE: {health_mae:.3f}")
            
            # Store feature names for later use
            self.feature_names = X.columns.tolist()
            
        except Exception as e:
            self.logger.error("Failed to train models", error=str(e))
            raise
    
    async def process_request(self, request: Dict[str, Any]) -> AgentResponse:
        """Process client health requests"""
        try:
            request_type = request.get("type", "")
            request_data = request.get("data", {})
            
            start_time = datetime.utcnow()
            
            if request_type == "predict_churn":
                result = await self._predict_churn(request_data)
            elif request_type == "calculate_health_score":
                result = await self._calculate_health_score(request_data)
            elif request_type == "get_health_dashboard":
                result = await self._get_health_dashboard(request_data)
            elif request_type == "recommend_interventions":
                result = await self._recommend_interventions(request_data)
            elif request_type == "analyze_client_segment":
                result = await self._analyze_client_segment(request_data)
            elif request_type == "predict_revenue_impact":
                result = await self._predict_revenue_impact(request_data)
            else:
                result = {"error": f"Unknown request type: {request_type}"}
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AgentResponse(
                success=True,
                data=result,
                processing_time_ms=processing_time,
                agent_id=self.agent_id,
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            self.logger.error("Error processing client health request", error=str(e))
            return AgentResponse(
                success=False,
                error=str(e),
                agent_id=self.agent_id,
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def _predict_churn(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict client churn probability"""
        client_id = data.get("client_id", "unknown")
        client_features = data.get("features", {})
        
        # Generate synthetic client data if not provided
        if not client_features:
            client_features = self._generate_synthetic_client_features()
        
        # Prepare features for prediction
        features = self._prepare_features(client_features)
        
        # Make prediction
        churn_probability = self.churn_model.predict_proba([features])[0][1]
        churn_prediction = churn_probability > 0.5
        
        # Calculate risk factors
        risk_factors = self._identify_risk_factors(client_features, churn_probability)
        
        # Generate recommendations
        recommendations = self._generate_churn_prevention_recommendations(risk_factors)
        
        return {
            "client_id": client_id,
            "churn_probability": round(churn_probability, 3),
            "churn_prediction": bool(churn_prediction),
            "risk_level": self._get_risk_level(churn_probability),
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "confidence": round(self._calculate_prediction_confidence(features), 3),
            "prediction_date": datetime.utcnow().isoformat()
        }
    
    async def _calculate_health_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive client health score"""
        client_id = data.get("client_id", "unknown")
        client_features = data.get("features", {})
        
        # Generate synthetic client data if not provided
        if not client_features:
            client_features = self._generate_synthetic_client_features()
        
        # Prepare features for prediction
        features = self._prepare_features(client_features)
        
        # Calculate health score
        health_score = self.health_model.predict([features])[0]
        health_score = np.clip(health_score, 0, 1)
        
        # Calculate component scores
        component_scores = self._calculate_component_scores(client_features)
        
        # Determine health status
        health_status = self._get_health_status(health_score)
        
        # Calculate trends
        trends = self._calculate_health_trends(client_features)
        
        return {
            "client_id": client_id,
            "overall_health_score": round(health_score, 3),
            "health_status": health_status,
            "component_scores": component_scores,
            "trends": trends,
            "health_indicators": self._get_health_indicators(health_score, component_scores),
            "improvement_areas": self._identify_improvement_areas(component_scores),
            "calculation_date": datetime.utcnow().isoformat()
        }
    
    async def _get_health_dashboard(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive health dashboard data"""
        # Generate synthetic client portfolio
        portfolio_size = data.get("portfolio_size", 100)
        clients = []
        
        for i in range(portfolio_size):
            client_features = self._generate_synthetic_client_features()
            features = self._prepare_features(client_features)
            
            churn_prob = self.churn_model.predict_proba([features])[0][1]
            health_score = self.health_model.predict([features])[0]
            health_score = np.clip(health_score, 0, 1)
            
            clients.append({
                "client_id": f"client_{i:04d}",
                "churn_probability": round(churn_prob, 3),
                "health_score": round(health_score, 3),
                "health_status": self._get_health_status(health_score),
                "risk_level": self._get_risk_level(churn_prob),
                "monthly_revenue": client_features.get("monthly_revenue", 0),
                "tenure_months": client_features.get("tenure_months", 0)
            })
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(clients)
        
        # Identify at-risk clients
        at_risk_clients = [c for c in clients if c["churn_probability"] > 0.5 or c["health_score"] < 0.5]
        
        # Calculate revenue at risk
        revenue_at_risk = sum(c["monthly_revenue"] for c in at_risk_clients)
        
        return {
            "portfolio_size": portfolio_size,
            "portfolio_metrics": portfolio_metrics,
            "at_risk_clients": len(at_risk_clients),
            "revenue_at_risk": round(revenue_at_risk, 2),
            "health_distribution": self._calculate_health_distribution(clients),
            "churn_distribution": self._calculate_churn_distribution(clients),
            "top_risk_clients": sorted(at_risk_clients, key=lambda x: x["churn_probability"], reverse=True)[:10],
            "dashboard_date": datetime.utcnow().isoformat()
        }
    
    async def _recommend_interventions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend interventions for at-risk clients"""
        client_id = data.get("client_id", "unknown")
        client_features = data.get("features", {})
        intervention_type = data.get("intervention_type", "comprehensive")
        
        # Generate synthetic client data if not provided
        if not client_features:
            client_features = self._generate_synthetic_client_features()
        
        # Calculate current health and churn risk
        features = self._prepare_features(client_features)
        churn_prob = self.churn_model.predict_proba([features])[0][1]
        health_score = self.health_model.predict([features])[0]
        health_score = np.clip(health_score, 0, 1)
        
        # Identify specific issues
        issues = self._identify_client_issues(client_features)
        
        # Generate targeted interventions
        interventions = self._generate_interventions(issues, churn_prob, health_score, intervention_type)
        
        # Calculate expected impact
        impact_analysis = self._calculate_intervention_impact(interventions, churn_prob, health_score)
        
        # Prioritize interventions
        prioritized_interventions = self._prioritize_interventions(interventions, impact_analysis)
        
        return {
            "client_id": client_id,
            "current_churn_probability": round(churn_prob, 3),
            "current_health_score": round(health_score, 3),
            "identified_issues": issues,
            "recommended_interventions": prioritized_interventions,
            "expected_impact": impact_analysis,
            "implementation_timeline": self._get_implementation_timeline(prioritized_interventions),
            "success_metrics": self._define_success_metrics(prioritized_interventions),
            "recommendation_date": datetime.utcnow().isoformat()
        }
    
    async def _analyze_client_segment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze client segments for health patterns"""
        segment_criteria = data.get("criteria", {})
        
        # Generate segment data
        segments = self._generate_client_segments(segment_criteria)
        
        # Analyze each segment
        segment_analysis = {}
        for segment_name, segment_clients in segments.items():
            segment_analysis[segment_name] = self._analyze_segment_health(segment_clients)
        
        # Compare segments
        segment_comparison = self._compare_segments(segment_analysis)
        
        # Identify best practices
        best_practices = self._identify_best_practices(segment_analysis)
        
        return {
            "segment_criteria": segment_criteria,
            "segment_analysis": segment_analysis,
            "segment_comparison": segment_comparison,
            "best_practices": best_practices,
            "recommendations": self._generate_segment_recommendations(segment_analysis),
            "analysis_date": datetime.utcnow().isoformat()
        }
    
    async def _predict_revenue_impact(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict revenue impact of client health changes"""
        scenario = data.get("scenario", "current")
        time_horizon = data.get("time_horizon", 12)  # months
        
        # Generate current portfolio
        portfolio = self._generate_portfolio_scenario(scenario)
        
        # Calculate current revenue
        current_revenue = sum(client["monthly_revenue"] for client in portfolio)
        
        # Predict future states
        future_scenarios = self._predict_future_scenarios(portfolio, time_horizon)
        
        # Calculate revenue impact
        revenue_impact = self._calculate_revenue_impact(current_revenue, future_scenarios)
        
        # Identify key drivers
        key_drivers = self._identify_revenue_drivers(future_scenarios)
        
        return {
            "scenario": scenario,
            "time_horizon_months": time_horizon,
            "current_monthly_revenue": round(current_revenue, 2),
            "future_scenarios": future_scenarios,
            "revenue_impact": revenue_impact,
            "key_drivers": key_drivers,
            "recommendations": self._generate_revenue_recommendations(revenue_impact),
            "prediction_date": datetime.utcnow().isoformat()
        }
    
    def _generate_synthetic_client_features(self) -> Dict[str, Any]:
        """Generate synthetic client features for demonstration"""
        return {
            "tenure_months": random.randint(1, 60),
            "monthly_revenue": random.uniform(1000, 50000),
            "support_tickets": random.randint(0, 20),
            "response_time_avg": random.uniform(0.5, 8.0),
            "satisfaction_score": random.uniform(0, 1),
            "contract_value": random.uniform(10000, 500000),
            "payment_delay_days": random.uniform(0, 30),
            "feature_usage": random.uniform(0, 1),
            "engagement_score": random.uniform(0, 1),
            "industry": random.choice(['Technology', 'Healthcare', 'Finance', 'Manufacturing', 'Retail']),
            "company_size": random.choice(['Small', 'Medium', 'Large', 'Enterprise'])
        }
    
    def _prepare_features(self, client_features: Dict[str, Any]) -> List[float]:
        """Prepare features for model prediction"""
        # Base features
        features = [
            client_features.get("tenure_months", 0),
            client_features.get("monthly_revenue", 0),
            client_features.get("support_tickets", 0),
            client_features.get("response_time_avg", 0),
            client_features.get("satisfaction_score", 0),
            client_features.get("contract_value", 0),
            client_features.get("payment_delay_days", 0),
            client_features.get("feature_usage", 0),
            client_features.get("engagement_score", 0)
        ]
        
        # Add categorical features (one-hot encoded)
        industries = ['Technology', 'Healthcare', 'Finance', 'Manufacturing', 'Retail']
        sizes = ['Small', 'Medium', 'Large', 'Enterprise']
        
        # Industry encoding
        industry = client_features.get("industry", "Technology")
        for ind in industries:
            features.append(1 if ind == industry else 0)
        
        # Size encoding
        size = client_features.get("company_size", "Medium")
        for sz in sizes:
            features.append(1 if sz == size else 0)
        
        return features
    
    def _identify_risk_factors(self, client_features: Dict[str, Any], churn_prob: float) -> List[Dict[str, Any]]:
        """Identify specific risk factors for a client"""
        risk_factors = []
        
        if client_features.get("satisfaction_score", 0.5) < 0.3:
            risk_factors.append({
                "factor": "Low Satisfaction Score",
                "value": client_features.get("satisfaction_score", 0),
                "impact": "high",
                "description": "Client satisfaction is critically low"
            })
        
        if client_features.get("support_tickets", 0) > 10:
            risk_factors.append({
                "factor": "High Support Ticket Volume",
                "value": client_features.get("support_tickets", 0),
                "impact": "medium",
                "description": "Excessive support requests indicate potential issues"
            })
        
        if client_features.get("payment_delay_days", 0) > 15:
            risk_factors.append({
                "factor": "Payment Delays",
                "value": client_features.get("payment_delay_days", 0),
                "impact": "high",
                "description": "Frequent payment delays suggest financial stress"
            })
        
        if client_features.get("engagement_score", 0.5) < 0.2:
            risk_factors.append({
                "factor": "Low Engagement",
                "value": client_features.get("engagement_score", 0),
                "impact": "medium",
                "description": "Client is not actively using services"
            })
        
        if client_features.get("tenure_months", 0) < 6:
            risk_factors.append({
                "factor": "New Client",
                "value": client_features.get("tenure_months", 0),
                "impact": "medium",
                "description": "New clients have higher churn risk"
            })
        
        return risk_factors
    
    def _generate_churn_prevention_recommendations(self, risk_factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations to prevent churn"""
        recommendations = []
        
        for risk in risk_factors:
            if risk["factor"] == "Low Satisfaction Score":
                recommendations.append({
                    "action": "Schedule satisfaction review call",
                    "priority": "high",
                    "timeline": "immediate",
                    "expected_impact": "high",
                    "description": "Direct communication to understand and address concerns"
                })
            elif risk["factor"] == "High Support Ticket Volume":
                recommendations.append({
                    "action": "Implement proactive support",
                    "priority": "medium",
                    "timeline": "1-2 weeks",
                    "expected_impact": "medium",
                    "description": "Proactive monitoring and support to reduce ticket volume"
                })
            elif risk["factor"] == "Payment Delays":
                recommendations.append({
                    "action": "Review payment terms and offer flexibility",
                    "priority": "high",
                    "timeline": "immediate",
                    "expected_impact": "high",
                    "description": "Work with client to establish sustainable payment arrangements"
                })
            elif risk["factor"] == "Low Engagement":
                recommendations.append({
                    "action": "Provide training and onboarding",
                    "priority": "medium",
                    "timeline": "2-4 weeks",
                    "expected_impact": "medium",
                    "description": "Help client maximize value from services"
                })
        
        # Add general recommendations
        recommendations.extend([
            {
                "action": "Assign dedicated success manager",
                "priority": "medium",
                "timeline": "1 week",
                "expected_impact": "high",
                "description": "Personalized attention and relationship building"
            },
            {
                "action": "Review service level agreement",
                "priority": "low",
                "timeline": "1 month",
                "expected_impact": "medium",
                "description": "Ensure services meet client expectations"
            }
        ])
        
        return recommendations
    
    def _get_risk_level(self, churn_prob: float) -> str:
        """Get risk level based on churn probability"""
        if churn_prob > 0.7:
            return "critical"
        elif churn_prob > 0.5:
            return "high"
        elif churn_prob > 0.3:
            return "medium"
        else:
            return "low"
    
    def _calculate_prediction_confidence(self, features: List[float]) -> float:
        """Calculate confidence in prediction based on feature quality"""
        # Simple confidence calculation based on feature completeness and ranges
        confidence = 0.8  # Base confidence
        
        # Adjust based on feature values
        if len(features) < 10:
            confidence -= 0.2
        
        # Check for extreme values that might indicate data quality issues
        extreme_values = sum(1 for f in features if f > 1000 or f < 0)
        if extreme_values > 2:
            confidence -= 0.1
        
        return max(0.5, min(1.0, confidence))
    
    def _calculate_component_scores(self, client_features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual component scores"""
        return {
            "satisfaction": client_features.get("satisfaction_score", 0.5),
            "support_quality": max(0, 1 - client_features.get("support_tickets", 0) / 20),
            "financial_health": max(0, 1 - client_features.get("payment_delay_days", 0) / 30),
            "engagement": client_features.get("engagement_score", 0.5),
            "feature_adoption": client_features.get("feature_usage", 0.5),
            "relationship_strength": min(1, client_features.get("tenure_months", 0) / 24)
        }
    
    def _get_health_status(self, health_score: float) -> str:
        """Get health status based on score"""
        if health_score >= self.health_thresholds["excellent"]:
            return "excellent"
        elif health_score >= self.health_thresholds["healthy"]:
            return "healthy"
        elif health_score >= self.health_thresholds["at_risk"]:
            return "at_risk"
        else:
            return "critical"
    
    def _calculate_health_trends(self, client_features: Dict[str, Any]) -> Dict[str, str]:
        """Calculate health trends (simplified)"""
        # In a real system, this would compare with historical data
        return {
            "satisfaction_trend": random.choice(["improving", "stable", "declining"]),
            "engagement_trend": random.choice(["improving", "stable", "declining"]),
            "support_trend": random.choice(["improving", "stable", "declining"]),
            "overall_trend": random.choice(["improving", "stable", "declining"])
        }
    
    def _get_health_indicators(self, health_score: float, component_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Get health indicators and alerts"""
        indicators = []
        
        if health_score < 0.5:
            indicators.append({
                "type": "critical",
                "message": "Client health is critically low",
                "action_required": "immediate"
            })
        
        if component_scores["satisfaction"] < 0.3:
            indicators.append({
                "type": "warning",
                "message": "Satisfaction score is very low",
                "action_required": "urgent"
            })
        
        if component_scores["financial_health"] < 0.4:
            indicators.append({
                "type": "warning",
                "message": "Payment delays indicate financial stress",
                "action_required": "urgent"
            })
        
        return indicators
    
    def _identify_improvement_areas(self, component_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify areas for improvement"""
        improvement_areas = []
        
        for component, score in component_scores.items():
            if score < 0.6:
                improvement_areas.append({
                    "area": component.replace("_", " ").title(),
                    "current_score": round(score, 2),
                    "target_score": 0.8,
                    "improvement_potential": round(0.8 - score, 2),
                    "priority": "high" if score < 0.4 else "medium"
                })
        
        return sorted(improvement_areas, key=lambda x: x["improvement_potential"], reverse=True)
    
    def _calculate_portfolio_metrics(self, clients: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio-level metrics"""
        total_revenue = sum(c["monthly_revenue"] for c in clients)
        avg_health = np.mean([c["health_score"] for c in clients])
        avg_churn_prob = np.mean([c["churn_probability"] for c in clients])
        
        health_distribution = {
            "excellent": len([c for c in clients if c["health_score"] >= 0.9]),
            "healthy": len([c for c in clients if 0.7 <= c["health_score"] < 0.9]),
            "at_risk": len([c for c in clients if 0.5 <= c["health_score"] < 0.7]),
            "critical": len([c for c in clients if c["health_score"] < 0.5])
        }
        
        return {
            "total_monthly_revenue": round(total_revenue, 2),
            "average_health_score": round(avg_health, 3),
            "average_churn_probability": round(avg_churn_prob, 3),
            "health_distribution": health_distribution,
            "revenue_concentration": self._calculate_revenue_concentration(clients)
        }
    
    def _calculate_revenue_concentration(self, clients: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate revenue concentration metrics"""
        total_revenue = sum(c["monthly_revenue"] for c in clients)
        
        # Top 10% clients
        sorted_clients = sorted(clients, key=lambda x: x["monthly_revenue"], reverse=True)
        top_10_percent = sorted_clients[:max(1, len(clients) // 10)]
        top_10_revenue = sum(c["monthly_revenue"] for c in top_10_percent)
        
        return {
            "top_10_percent_revenue_share": round(top_10_revenue / total_revenue, 3),
            "gini_coefficient": self._calculate_gini_coefficient([c["monthly_revenue"] for c in clients])
        }
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values:
            return 0
        
        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum, 1))) / (n * sum(values))
    
    def _calculate_health_distribution(self, clients: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate health score distribution"""
        return {
            "excellent": len([c for c in clients if c["health_score"] >= 0.9]),
            "healthy": len([c for c in clients if 0.7 <= c["health_score"] < 0.9]),
            "at_risk": len([c for c in clients if 0.5 <= c["health_score"] < 0.7]),
            "critical": len([c for c in clients if c["health_score"] < 0.5])
        }
    
    def _calculate_churn_distribution(self, clients: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate churn probability distribution"""
        return {
            "low_risk": len([c for c in clients if c["churn_probability"] < 0.3]),
            "medium_risk": len([c for c in clients if 0.3 <= c["churn_probability"] < 0.5]),
            "high_risk": len([c for c in clients if 0.5 <= c["churn_probability"] < 0.7]),
            "critical_risk": len([c for c in clients if c["churn_probability"] >= 0.7])
        }
    
    def _identify_client_issues(self, client_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific client issues"""
        issues = []
        
        if client_features.get("satisfaction_score", 0.5) < 0.4:
            issues.append({
                "issue": "Low Satisfaction",
                "severity": "high",
                "description": "Client satisfaction is below acceptable levels"
            })
        
        if client_features.get("support_tickets", 0) > 8:
            issues.append({
                "issue": "High Support Volume",
                "severity": "medium",
                "description": "Excessive support requests indicate service issues"
            })
        
        if client_features.get("payment_delay_days", 0) > 10:
            issues.append({
                "issue": "Payment Delays",
                "severity": "high",
                "description": "Frequent payment delays suggest financial problems"
            })
        
        if client_features.get("engagement_score", 0.5) < 0.3:
            issues.append({
                "issue": "Low Engagement",
                "severity": "medium",
                "description": "Client is not actively using services"
            })
        
        return issues
    
    def _generate_interventions(self, issues: List[Dict[str, Any]], churn_prob: float, 
                              health_score: float, intervention_type: str) -> List[Dict[str, Any]]:
        """Generate targeted interventions"""
        interventions = []
        
        for issue in issues:
            if issue["issue"] == "Low Satisfaction":
                interventions.append({
                    "intervention": "Satisfaction Recovery Program",
                    "type": "relationship",
                    "priority": "high",
                    "effort": "medium",
                    "cost": "low",
                    "description": "Comprehensive program to improve client satisfaction"
                })
            elif issue["issue"] == "High Support Volume":
                interventions.append({
                    "intervention": "Proactive Support Implementation",
                    "type": "operational",
                    "priority": "medium",
                    "effort": "high",
                    "cost": "medium",
                    "description": "Implement proactive monitoring and support"
                })
            elif issue["issue"] == "Payment Delays":
                interventions.append({
                    "intervention": "Payment Plan Restructuring",
                    "type": "financial",
                    "priority": "high",
                    "effort": "low",
                    "cost": "low",
                    "description": "Work with client to establish sustainable payment terms"
                })
            elif issue["issue"] == "Low Engagement":
                interventions.append({
                    "intervention": "Engagement Enhancement Program",
                    "type": "training",
                    "priority": "medium",
                    "effort": "medium",
                    "cost": "low",
                    "description": "Training and onboarding to increase service usage"
                })
        
        # Add general interventions based on overall health
        if health_score < 0.5:
            interventions.append({
                "intervention": "Dedicated Success Manager",
                "type": "relationship",
                "priority": "high",
                "effort": "medium",
                "cost": "high",
                "description": "Assign dedicated success manager for personalized attention"
            })
        
        return interventions
    
    def _calculate_intervention_impact(self, interventions: List[Dict[str, Any]], 
                                     churn_prob: float, health_score: float) -> Dict[str, Any]:
        """Calculate expected impact of interventions"""
        total_impact = 0
        for intervention in interventions:
            if intervention["priority"] == "high":
                total_impact += 0.15
            elif intervention["priority"] == "medium":
                total_impact += 0.10
            else:
                total_impact += 0.05
        
        # Cap the total impact
        total_impact = min(total_impact, 0.4)
        
        expected_churn_reduction = churn_prob * total_impact
        expected_health_improvement = (1 - health_score) * total_impact
        
        return {
            "expected_churn_reduction": round(expected_churn_reduction, 3),
            "expected_health_improvement": round(expected_health_improvement, 3),
            "new_churn_probability": round(max(0, churn_prob - expected_churn_reduction), 3),
            "new_health_score": round(min(1, health_score + expected_health_improvement), 3),
            "confidence_level": "medium"
        }
    
    def _prioritize_interventions(self, interventions: List[Dict[str, Any]], 
                                impact_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize interventions based on impact and effort"""
        def intervention_score(intervention):
            priority_score = {"high": 3, "medium": 2, "low": 1}[intervention["priority"]]
            effort_score = {"low": 3, "medium": 2, "high": 1}[intervention["effort"]]
            cost_score = {"low": 3, "medium": 2, "high": 1}[intervention["cost"]]
            
            return priority_score + effort_score + cost_score
        
        return sorted(interventions, key=intervention_score, reverse=True)
    
    def _get_implementation_timeline(self, interventions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Get implementation timeline for interventions"""
        timeline = {
            "immediate": [],
            "short_term": [],
            "medium_term": [],
            "long_term": []
        }
        
        for intervention in interventions:
            if intervention["priority"] == "high" and intervention["effort"] == "low":
                timeline["immediate"].append(intervention["intervention"])
            elif intervention["priority"] == "high":
                timeline["short_term"].append(intervention["intervention"])
            elif intervention["effort"] == "high":
                timeline["long_term"].append(intervention["intervention"])
            else:
                timeline["medium_term"].append(intervention["intervention"])
        
        return timeline
    
    def _define_success_metrics(self, interventions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Define success metrics for interventions"""
        return [
            {
                "metric": "Churn Probability Reduction",
                "target": "20% reduction",
                "measurement": "Monthly churn probability tracking",
                "timeline": "3 months"
            },
            {
                "metric": "Health Score Improvement",
                "target": "15% improvement",
                "measurement": "Monthly health score calculation",
                "timeline": "6 months"
            },
            {
                "metric": "Satisfaction Score",
                "target": "Above 0.7",
                "measurement": "Quarterly satisfaction surveys",
                "timeline": "6 months"
            },
            {
                "metric": "Support Ticket Reduction",
                "target": "30% reduction",
                "measurement": "Monthly ticket volume tracking",
                "timeline": "3 months"
            }
        ]
    
    def _generate_client_segments(self, criteria: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate client segments based on criteria"""
        # Generate sample clients for each segment
        segments = {}
        
        if "industry" in criteria:
            for industry in criteria["industry"]:
                segments[f"{industry}_clients"] = [
                    self._generate_synthetic_client_features() for _ in range(20)
                ]
        
        if "company_size" in criteria:
            for size in criteria["company_size"]:
                segments[f"{size}_clients"] = [
                    self._generate_synthetic_client_features() for _ in range(20)
                ]
        
        # Default segments if no criteria provided
        if not segments:
            segments = {
                "high_value": [self._generate_synthetic_client_features() for _ in range(20)],
                "at_risk": [self._generate_synthetic_client_features() for _ in range(20)],
                "new_clients": [self._generate_synthetic_client_features() for _ in range(20)]
            }
        
        return segments
    
    def _analyze_segment_health(self, segment_clients: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze health patterns for a client segment"""
        # Calculate metrics for the segment
        features_list = [self._prepare_features(client) for client in segment_clients]
        
        churn_probs = [self.churn_model.predict_proba([features])[0][1] for features in features_list]
        health_scores = [self.health_model.predict([features])[0] for features in features_list]
        
        return {
            "segment_size": len(segment_clients),
            "average_churn_probability": round(np.mean(churn_probs), 3),
            "average_health_score": round(np.mean(health_scores), 3),
            "health_distribution": self._calculate_health_distribution([
                {"health_score": score} for score in health_scores
            ]),
            "key_characteristics": self._identify_segment_characteristics(segment_clients)
        }
    
    def _identify_segment_characteristics(self, segment_clients: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify key characteristics of a client segment"""
        return {
            "average_tenure": round(np.mean([c.get("tenure_months", 0) for c in segment_clients]), 1),
            "average_revenue": round(np.mean([c.get("monthly_revenue", 0) for c in segment_clients]), 2),
            "common_industries": self._get_most_common_industries(segment_clients),
            "average_satisfaction": round(np.mean([c.get("satisfaction_score", 0.5) for c in segment_clients]), 3)
        }
    
    def _get_most_common_industries(self, segment_clients: List[Dict[str, Any]]) -> List[str]:
        """Get most common industries in a segment"""
        industries = [c.get("industry", "Unknown") for c in segment_clients]
        industry_counts = {}
        for industry in industries:
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        
        return sorted(industry_counts.keys(), key=lambda x: industry_counts[x], reverse=True)[:3]
    
    def _compare_segments(self, segment_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different client segments"""
        segments = list(segment_analysis.keys())
        
        if len(segments) < 2:
            return {"message": "Need at least 2 segments for comparison"}
        
        # Find best and worst performing segments
        best_segment = max(segments, key=lambda s: segment_analysis[s]["average_health_score"])
        worst_segment = min(segments, key=lambda s: segment_analysis[s]["average_health_score"])
        
        return {
            "best_performing_segment": best_segment,
            "worst_performing_segment": worst_segment,
            "performance_gap": round(
                segment_analysis[best_segment]["average_health_score"] - 
                segment_analysis[worst_segment]["average_health_score"], 3
            ),
            "segment_rankings": sorted(
                segments, 
                key=lambda s: segment_analysis[s]["average_health_score"], 
                reverse=True
            )
        }
    
    def _identify_best_practices(self, segment_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify best practices from high-performing segments"""
        best_practices = []
        
        # Find high-performing segments
        high_performers = [
            segment for segment, analysis in segment_analysis.items()
            if analysis["average_health_score"] > 0.7
        ]
        
        if high_performers:
            best_practices.append({
                "practice": "Focus on High-Performing Segments",
                "description": f"Segments {', '.join(high_performers)} show excellent health scores",
                "recommendation": "Replicate strategies from these segments"
            })
        
        # Analyze characteristics of high performers
        for segment in high_performers:
            characteristics = segment_analysis[segment]["key_characteristics"]
            if characteristics["average_satisfaction"] > 0.7:
                best_practices.append({
                    "practice": "High Satisfaction Focus",
                    "description": f"{segment} maintains high satisfaction scores",
                    "recommendation": "Implement satisfaction improvement programs"
                })
        
        return best_practices
    
    def _generate_segment_recommendations(self, segment_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on segment analysis"""
        recommendations = []
        
        for segment, analysis in segment_analysis.items():
            if analysis["average_health_score"] < 0.6:
                recommendations.append({
                    "segment": segment,
                    "priority": "high",
                    "recommendation": f"Implement health improvement program for {segment}",
                    "expected_impact": "medium"
                })
            elif analysis["average_churn_probability"] > 0.4:
                recommendations.append({
                    "segment": segment,
                    "priority": "high",
                    "recommendation": f"Develop churn prevention strategy for {segment}",
                    "expected_impact": "high"
                })
        
        return recommendations
    
    def _generate_portfolio_scenario(self, scenario: str) -> List[Dict[str, Any]]:
        """Generate portfolio scenario for revenue impact analysis"""
        portfolio_size = 100
        clients = []
        
        for i in range(portfolio_size):
            client_features = self._generate_synthetic_client_features()
            features = self._prepare_features(client_features)
            
            churn_prob = self.churn_model.predict_proba([features])[0][1]
            health_score = self.health_model.predict([features])[0]
            health_score = np.clip(health_score, 0, 1)
            
            clients.append({
                "client_id": f"client_{i:04d}",
                "monthly_revenue": client_features.get("monthly_revenue", 0),
                "churn_probability": churn_prob,
                "health_score": health_score,
                "tenure_months": client_features.get("tenure_months", 0)
            })
        
        return clients
    
    def _predict_future_scenarios(self, portfolio: List[Dict[str, Any]], time_horizon: int) -> Dict[str, Any]:
        """Predict future scenarios for revenue impact"""
        scenarios = {
            "current": portfolio,
            "optimistic": [],
            "pessimistic": [],
            "realistic": []
        }
        
        for client in portfolio:
            # Optimistic scenario: improve health, reduce churn
            optimistic_client = client.copy()
            optimistic_client["churn_probability"] *= 0.7  # 30% reduction
            optimistic_client["health_score"] = min(1, optimistic_client["health_score"] * 1.2)  # 20% improvement
            scenarios["optimistic"].append(optimistic_client)
            
            # Pessimistic scenario: worsen health, increase churn
            pessimistic_client = client.copy()
            pessimistic_client["churn_probability"] = min(1, pessimistic_client["churn_probability"] * 1.5)  # 50% increase
            pessimistic_client["health_score"] *= 0.8  # 20% decrease
            scenarios["pessimistic"].append(pessimistic_client)
            
            # Realistic scenario: slight improvement
            realistic_client = client.copy()
            realistic_client["churn_probability"] *= 0.9  # 10% reduction
            realistic_client["health_score"] = min(1, realistic_client["health_score"] * 1.05)  # 5% improvement
            scenarios["realistic"].append(realistic_client)
        
        return scenarios
    
    def _calculate_revenue_impact(self, current_revenue: float, future_scenarios: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate revenue impact of different scenarios"""
        impact = {}
        
        for scenario_name, clients in future_scenarios.items():
            if scenario_name == "current":
                continue
            
            # Calculate expected revenue (accounting for churn)
            expected_revenue = 0
            for client in clients:
                # Revenue weighted by probability of retention
                retention_prob = 1 - client["churn_probability"]
                expected_revenue += client["monthly_revenue"] * retention_prob
            
            # Calculate impact
            revenue_change = expected_revenue - current_revenue
            revenue_change_percent = (revenue_change / current_revenue) * 100
            
            impact[scenario_name] = {
                "expected_monthly_revenue": round(expected_revenue, 2),
                "revenue_change": round(revenue_change, 2),
                "revenue_change_percent": round(revenue_change_percent, 2)
            }
        
        return impact
    
    def _identify_revenue_drivers(self, future_scenarios: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify key drivers of revenue impact"""
        drivers = []
        
        # Analyze churn impact
        current_churn = np.mean([c["churn_probability"] for c in future_scenarios["current"]])
        optimistic_churn = np.mean([c["churn_probability"] for c in future_scenarios["optimistic"]])
        pessimistic_churn = np.mean([c["churn_probability"] for c in future_scenarios["pessimistic"]])
        
        drivers.append({
            "driver": "Churn Rate",
            "current_value": round(current_churn, 3),
            "optimistic_value": round(optimistic_churn, 3),
            "pessimistic_value": round(pessimistic_churn, 3),
            "impact": "high"
        })
        
        # Analyze health score impact
        current_health = np.mean([c["health_score"] for c in future_scenarios["current"]])
        optimistic_health = np.mean([c["health_score"] for c in future_scenarios["optimistic"]])
        pessimistic_health = np.mean([c["health_score"] for c in future_scenarios["pessimistic"]])
        
        drivers.append({
            "driver": "Health Score",
            "current_value": round(current_health, 3),
            "optimistic_value": round(optimistic_health, 3),
            "pessimistic_value": round(pessimistic_health, 3),
            "impact": "high"
        })
        
        return drivers
    
    def _generate_revenue_recommendations(self, revenue_impact: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on revenue impact analysis"""
        recommendations = []
        
        realistic_impact = revenue_impact.get("realistic", {})
        optimistic_impact = revenue_impact.get("optimistic", {})
        
        if realistic_impact.get("revenue_change_percent", 0) > 5:
            recommendations.append({
                "recommendation": "Implement Health Improvement Programs",
                "rationale": "Realistic scenario shows positive revenue impact",
                "priority": "high",
                "expected_benefit": f"{realistic_impact.get('revenue_change_percent', 0):.1f}% revenue increase"
            })
        
        if optimistic_impact.get("revenue_change_percent", 0) > 10:
            recommendations.append({
                "recommendation": "Aggressive Health Enhancement Strategy",
                "rationale": "Optimistic scenario shows significant upside potential",
                "priority": "medium",
                "expected_benefit": f"{optimistic_impact.get('revenue_change_percent', 0):.1f}% revenue increase"
            })
        
        recommendations.append({
            "recommendation": "Monitor Key Health Metrics",
            "rationale": "Continuous monitoring enables proactive intervention",
            "priority": "high",
            "expected_benefit": "Early detection of at-risk clients"
        })
        
        return recommendations
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        return {
            "agent_id": self.agent_id,
            "status": "active" if self.model_loaded else "inactive",
            "model_loaded": self.model_loaded,
            "health_score": 0.94 if self.model_loaded else 0.0,
            "last_activity": datetime.utcnow().isoformat(),
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "average_response_time_ms": self.metrics.average_response_time_ms,
                "error_rate": self.metrics.error_rate
            }
        }


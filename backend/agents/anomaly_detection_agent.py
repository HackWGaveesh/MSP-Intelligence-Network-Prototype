"""
Anomaly Detection Agent for MSP Intelligence Mesh Network
Detects unusual patterns in operations, system logs, and performance metrics
"""
import asyncio
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import structlog
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent, AgentResponse, AgentMetrics


logger = structlog.get_logger()


class AnomalyDetectionAgent(BaseAgent):
    """Anomaly Detection Agent for identifying unusual patterns"""
    
    def __init__(self):
        super().__init__()
        self.agent_id = "anomaly_detection_agent"
        self.agent_type = "anomaly_detection"
        self.model_loaded = False
        self.isolation_forest = None
        self.dbscan_model = None
        self.scaler = None
        self.anomaly_threshold = 0.1
        self.baseline_data = {}
        self.anomaly_history = []
        
        self.logger = logger.bind(agent=self.agent_id)
        self.logger.info("Anomaly Detection Agent initialized")
    
    async def initialize(self):
        """Initialize the agent and load models"""
        try:
            self.logger.info("Initializing Anomaly Detection Agent")
            
            # Initialize Isolation Forest for anomaly detection
            self.isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Initialize DBSCAN for clustering anomalies
            self.dbscan_model = DBSCAN(eps=0.5, min_samples=5)
            
            self.scaler = StandardScaler()
            
            # Load baseline data
            await self._load_baseline_data()
            
            # Train models
            await self._train_models()
            
            self.model_loaded = True
            self.logger.info("Anomaly Detection Agent initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize Anomaly Detection Agent", error=str(e))
            raise
    
    async def _load_baseline_data(self):
        """Load baseline data for anomaly detection"""
        # Generate synthetic baseline data
        np.random.seed(42)
        n_samples = 1000
        
        # System performance metrics
        self.baseline_data = {
            'cpu_usage': np.random.normal(0.4, 0.15, n_samples),
            'memory_usage': np.random.normal(0.5, 0.2, n_samples),
            'disk_io': np.random.exponential(100, n_samples),
            'network_latency': np.random.exponential(50, n_samples),
            'error_rate': np.random.exponential(0.02, n_samples),
            'response_time': np.random.exponential(200, n_samples),
            'throughput': np.random.normal(1000, 200, n_samples),
            'active_connections': np.random.poisson(50, n_samples)
        }
        
        # Ensure all values are positive and within reasonable bounds
        for key in self.baseline_data:
            self.baseline_data[key] = np.clip(self.baseline_data[key], 0, None)
        
        self.logger.info(f"Loaded baseline data with {n_samples} samples")
    
    async def _train_models(self):
        """Train anomaly detection models"""
        try:
            # Prepare training data
            baseline_df = pd.DataFrame(self.baseline_data)
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(baseline_df)
            
            # Train Isolation Forest
            self.isolation_forest.fit(scaled_data)
            
            self.logger.info("Anomaly detection models trained successfully")
            
        except Exception as e:
            self.logger.error("Failed to train models", error=str(e))
            raise
    
    async def process_request(self, request: Dict[str, Any]) -> AgentResponse:
        """Process anomaly detection requests"""
        try:
            request_type = request.get("type", "")
            request_data = request.get("data", {})
            
            start_time = datetime.utcnow()
            
            if request_type == "detect_anomalies":
                result = await self._detect_anomalies(request_data)
            elif request_type == "analyze_system_health":
                result = await self._analyze_system_health(request_data)
            elif request_type == "detect_performance_anomalies":
                result = await self._detect_performance_anomalies(request_data)
            elif request_type == "detect_security_anomalies":
                result = await self._detect_security_anomalies(request_data)
            elif request_type == "analyze_anomaly_patterns":
                result = await self._analyze_anomaly_patterns(request_data)
            elif request_type == "predict_anomalies":
                result = await self._predict_anomalies(request_data)
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
            self.logger.error("Error processing anomaly detection request", error=str(e))
            return AgentResponse(
                success=False,
                error=str(e),
                agent_id=self.agent_id,
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def _detect_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in system metrics"""
        metrics = data.get("metrics", {})
        detection_type = data.get("detection_type", "system")
        
        # Generate synthetic metrics if not provided
        if not metrics:
            metrics = self._generate_synthetic_metrics()
        
        # Prepare data for detection
        feature_vector = self._prepare_feature_vector(metrics)
        
        # Detect anomalies
        anomaly_score = self.isolation_forest.decision_function([feature_vector])[0]
        is_anomaly = self.isolation_forest.predict([feature_vector])[0] == -1
        
        # Calculate severity
        severity = self._calculate_anomaly_severity(anomaly_score, metrics)
        
        # Generate insights
        insights = self._generate_anomaly_insights(metrics, anomaly_score, severity)
        
        # Store anomaly for pattern analysis
        anomaly_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "anomaly_score": float(anomaly_score),
            "is_anomaly": bool(is_anomaly),
            "severity": severity,
            "detection_type": detection_type
        }
        self.anomaly_history.append(anomaly_record)
        
        return {
            "detection_type": detection_type,
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": round(anomaly_score, 3),
            "severity": severity,
            "metrics_analyzed": list(metrics.keys()),
            "insights": insights,
            "recommendations": self._generate_anomaly_recommendations(severity, metrics),
            "detection_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _analyze_system_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall system health"""
        time_window = data.get("time_window", "1_hour")
        
        # Generate system health data
        health_metrics = self._generate_system_health_metrics(time_window)
        
        # Calculate health score
        health_score = self._calculate_health_score(health_metrics)
        
        # Identify health issues
        health_issues = self._identify_health_issues(health_metrics)
        
        # Generate health insights
        insights = self._generate_health_insights(health_metrics, health_score)
        
        return {
            "time_window": time_window,
            "overall_health_score": round(health_score, 3),
            "health_status": self._get_health_status(health_score),
            "health_metrics": health_metrics,
            "health_issues": health_issues,
            "insights": insights,
            "recommendations": self._generate_health_recommendations(health_issues),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _detect_performance_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect performance-related anomalies"""
        performance_metrics = data.get("performance_metrics", {})
        
        # Generate synthetic performance data if not provided
        if not performance_metrics:
            performance_metrics = self._generate_performance_metrics()
        
        # Analyze performance patterns
        performance_analysis = self._analyze_performance_patterns(performance_metrics)
        
        # Detect anomalies
        anomalies = []
        for metric_name, values in performance_metrics.items():
            if isinstance(values, (list, np.ndarray)) and len(values) > 1:
                anomaly_points = self._detect_metric_anomalies(values, metric_name)
                anomalies.extend(anomaly_points)
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(performance_metrics)
        
        return {
            "performance_score": round(performance_score, 3),
            "performance_analysis": performance_analysis,
            "detected_anomalies": anomalies,
            "performance_trends": self._analyze_performance_trends(performance_metrics),
            "recommendations": self._generate_performance_recommendations(anomalies, performance_score),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _detect_security_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect security-related anomalies"""
        security_events = data.get("security_events", [])
        
        # Generate synthetic security events if not provided
        if not security_events:
            security_events = self._generate_security_events()
        
        # Analyze security patterns
        security_analysis = self._analyze_security_patterns(security_events)
        
        # Detect suspicious activities
        suspicious_activities = self._detect_suspicious_activities(security_events)
        
        # Calculate security score
        security_score = self._calculate_security_score(security_events, suspicious_activities)
        
        # Generate alerts
        security_alerts = self._generate_security_alerts(suspicious_activities, security_score)
        
        return {
            "security_score": round(security_score, 3),
            "security_analysis": security_analysis,
            "suspicious_activities": suspicious_activities,
            "security_alerts": security_alerts,
            "threat_level": self._assess_threat_level(security_score),
            "recommendations": self._generate_security_recommendations(suspicious_activities),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _analyze_anomaly_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in historical anomalies"""
        analysis_period = data.get("period", "7_days")
        
        # Filter recent anomalies
        recent_anomalies = self._get_recent_anomalies(analysis_period)
        
        # Analyze patterns
        patterns = self._identify_anomaly_patterns(recent_anomalies)
        
        # Calculate anomaly statistics
        statistics = self._calculate_anomaly_statistics(recent_anomalies)
        
        # Predict future anomalies
        predictions = self._predict_future_anomalies(recent_anomalies, patterns)
        
        return {
            "analysis_period": analysis_period,
            "total_anomalies": len(recent_anomalies),
            "anomaly_statistics": statistics,
            "identified_patterns": patterns,
            "future_predictions": predictions,
            "pattern_insights": self._generate_pattern_insights(patterns),
            "recommendations": self._generate_pattern_recommendations(patterns),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _predict_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict potential future anomalies"""
        prediction_horizon = data.get("horizon", "24_hours")
        current_metrics = data.get("current_metrics", {})
        
        # Generate current metrics if not provided
        if not current_metrics:
            current_metrics = self._generate_synthetic_metrics()
        
        # Analyze current state
        current_analysis = await self._detect_anomalies({"metrics": current_metrics})
        
        # Predict future states
        future_predictions = self._predict_future_metrics(current_metrics, prediction_horizon)
        
        # Calculate anomaly probabilities
        anomaly_probabilities = self._calculate_anomaly_probabilities(future_predictions)
        
        # Generate early warnings
        early_warnings = self._generate_early_warnings(anomaly_probabilities)
        
        return {
            "prediction_horizon": prediction_horizon,
            "current_analysis": current_analysis,
            "future_predictions": future_predictions,
            "anomaly_probabilities": anomaly_probabilities,
            "early_warnings": early_warnings,
            "risk_assessment": self._assess_anomaly_risk(anomaly_probabilities),
            "preventive_actions": self._suggest_preventive_actions(early_warnings),
            "prediction_timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_synthetic_metrics(self) -> Dict[str, float]:
        """Generate synthetic system metrics"""
        # Simulate normal system metrics with occasional anomalies
        if random.random() < 0.1:  # 10% chance of anomaly
            return {
                "cpu_usage": random.uniform(0.8, 1.0),  # High CPU
                "memory_usage": random.uniform(0.7, 0.9),  # High memory
                "disk_io": random.uniform(500, 1000),  # High disk I/O
                "network_latency": random.uniform(200, 500),  # High latency
                "error_rate": random.uniform(0.1, 0.3),  # High error rate
                "response_time": random.uniform(1000, 3000),  # Slow response
                "throughput": random.uniform(100, 500),  # Low throughput
                "active_connections": random.randint(200, 500)  # High connections
            }
        else:
            return {
                "cpu_usage": random.uniform(0.2, 0.6),
                "memory_usage": random.uniform(0.3, 0.7),
                "disk_io": random.uniform(50, 200),
                "network_latency": random.uniform(20, 100),
                "error_rate": random.uniform(0.001, 0.05),
                "response_time": random.uniform(100, 500),
                "throughput": random.uniform(800, 1200),
                "active_connections": random.randint(20, 80)
            }
    
    def _prepare_feature_vector(self, metrics: Dict[str, float]) -> List[float]:
        """Prepare feature vector for anomaly detection"""
        # Use the same features as in baseline data
        feature_names = list(self.baseline_data.keys())
        feature_vector = [metrics.get(name, 0.0) for name in feature_names]
        return feature_vector
    
    def _calculate_anomaly_severity(self, anomaly_score: float, metrics: Dict[str, float]) -> str:
        """Calculate anomaly severity based on score and metrics"""
        # Convert anomaly score to severity
        if anomaly_score < -0.5:
            return "critical"
        elif anomaly_score < -0.3:
            return "high"
        elif anomaly_score < -0.1:
            return "medium"
        else:
            return "low"
    
    def _generate_anomaly_insights(self, metrics: Dict[str, float], 
                                 anomaly_score: float, severity: str) -> List[Dict[str, Any]]:
        """Generate insights about detected anomalies"""
        insights = []
        
        # CPU insights
        if metrics.get("cpu_usage", 0) > 0.8:
            insights.append({
                "type": "high_cpu",
                "message": "CPU usage is critically high",
                "severity": "high",
                "value": metrics["cpu_usage"]
            })
        
        # Memory insights
        if metrics.get("memory_usage", 0) > 0.8:
            insights.append({
                "type": "high_memory",
                "message": "Memory usage is critically high",
                "severity": "high",
                "value": metrics["memory_usage"]
            })
        
        # Error rate insights
        if metrics.get("error_rate", 0) > 0.1:
            insights.append({
                "type": "high_error_rate",
                "message": "Error rate is significantly elevated",
                "severity": "medium",
                "value": metrics["error_rate"]
            })
        
        # Response time insights
        if metrics.get("response_time", 0) > 1000:
            insights.append({
                "type": "slow_response",
                "message": "Response time is unusually slow",
                "severity": "medium",
                "value": metrics["response_time"]
            })
        
        return insights
    
    def _generate_anomaly_recommendations(self, severity: str, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate recommendations for addressing anomalies"""
        recommendations = []
        
        if severity in ["critical", "high"]:
            recommendations.append({
                "action": "Immediate investigation required",
                "priority": "urgent",
                "description": "Critical anomaly detected - immediate action needed"
            })
        
        if metrics.get("cpu_usage", 0) > 0.8:
            recommendations.append({
                "action": "Scale CPU resources",
                "priority": "high",
                "description": "High CPU usage detected - consider scaling up"
            })
        
        if metrics.get("memory_usage", 0) > 0.8:
            recommendations.append({
                "action": "Increase memory allocation",
                "priority": "high",
                "description": "High memory usage detected - increase memory limits"
            })
        
        if metrics.get("error_rate", 0) > 0.1:
            recommendations.append({
                "action": "Investigate error sources",
                "priority": "medium",
                "description": "High error rate detected - investigate root causes"
            })
        
        if metrics.get("response_time", 0) > 1000:
            recommendations.append({
                "action": "Optimize performance",
                "priority": "medium",
                "description": "Slow response times detected - optimize system performance"
            })
        
        return recommendations
    
    def _generate_system_health_metrics(self, time_window: str) -> Dict[str, Any]:
        """Generate system health metrics for analysis"""
        # Simulate health metrics over time window
        if time_window == "1_hour":
            points = 60  # 1 minute intervals
        elif time_window == "24_hours":
            points = 1440  # 1 minute intervals
        else:
            points = 10080  # 1 week, 1 minute intervals
        
        metrics = {
            "cpu_usage": np.random.normal(0.4, 0.1, points),
            "memory_usage": np.random.normal(0.5, 0.15, points),
            "disk_usage": np.random.normal(0.6, 0.1, points),
            "network_utilization": np.random.normal(0.3, 0.1, points),
            "service_availability": np.random.normal(0.99, 0.01, points),
            "error_count": np.random.poisson(5, points),
            "response_time": np.random.exponential(200, points),
            "throughput": np.random.normal(1000, 100, points)
        }
        
        # Ensure values are within bounds
        for key in metrics:
            if key == "service_availability":
                metrics[key] = np.clip(metrics[key], 0.9, 1.0)
            else:
                metrics[key] = np.clip(metrics[key], 0, None)
        
        return metrics
    
    def _calculate_health_score(self, health_metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        scores = []
        
        # CPU health score
        cpu_avg = np.mean(health_metrics["cpu_usage"])
        cpu_score = max(0, 1 - (cpu_avg - 0.5) * 2)  # Optimal at 50%
        scores.append(cpu_score)
        
        # Memory health score
        memory_avg = np.mean(health_metrics["memory_usage"])
        memory_score = max(0, 1 - (memory_avg - 0.6) * 1.5)  # Optimal at 60%
        scores.append(memory_score)
        
        # Service availability score
        availability_avg = np.mean(health_metrics["service_availability"])
        availability_score = availability_avg
        scores.append(availability_score)
        
        # Error rate score
        error_avg = np.mean(health_metrics["error_count"])
        error_score = max(0, 1 - error_avg / 20)  # Optimal at 0 errors
        scores.append(error_score)
        
        # Response time score
        response_avg = np.mean(health_metrics["response_time"])
        response_score = max(0, 1 - (response_avg - 200) / 800)  # Optimal at 200ms
        scores.append(response_score)
        
        return np.mean(scores)
    
    def _identify_health_issues(self, health_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific health issues"""
        issues = []
        
        # CPU issues
        cpu_avg = np.mean(health_metrics["cpu_usage"])
        if cpu_avg > 0.8:
            issues.append({
                "type": "high_cpu_usage",
                "severity": "high",
                "description": f"Average CPU usage is {cpu_avg:.1%}",
                "recommendation": "Consider scaling CPU resources"
            })
        
        # Memory issues
        memory_avg = np.mean(health_metrics["memory_usage"])
        if memory_avg > 0.8:
            issues.append({
                "type": "high_memory_usage",
                "severity": "high",
                "description": f"Average memory usage is {memory_avg:.1%}",
                "recommendation": "Increase memory allocation"
            })
        
        # Availability issues
        availability_avg = np.mean(health_metrics["service_availability"])
        if availability_avg < 0.95:
            issues.append({
                "type": "low_availability",
                "severity": "critical",
                "description": f"Service availability is {availability_avg:.1%}",
                "recommendation": "Investigate service outages"
            })
        
        # Error issues
        error_avg = np.mean(health_metrics["error_count"])
        if error_avg > 10:
            issues.append({
                "type": "high_error_rate",
                "severity": "medium",
                "description": f"Average error count is {error_avg:.1f}",
                "recommendation": "Investigate error sources"
            })
        
        return issues
    
    def _generate_health_insights(self, health_metrics: Dict[str, Any], 
                                health_score: float) -> List[Dict[str, Any]]:
        """Generate insights about system health"""
        insights = []
        
        if health_score > 0.9:
            insights.append({
                "type": "excellent_health",
                "message": "System is operating at excellent health levels",
                "confidence": "high"
            })
        elif health_score > 0.7:
            insights.append({
                "type": "good_health",
                "message": "System health is good with minor issues",
                "confidence": "medium"
            })
        elif health_score > 0.5:
            insights.append({
                "type": "fair_health",
                "message": "System health is fair with some concerns",
                "confidence": "medium"
            })
        else:
            insights.append({
                "type": "poor_health",
                "message": "System health is poor and requires attention",
                "confidence": "high"
            })
        
        # Trend analysis
        cpu_trend = np.polyfit(range(len(health_metrics["cpu_usage"])), 
                              health_metrics["cpu_usage"], 1)[0]
        if abs(cpu_trend) > 0.01:
            trend_direction = "increasing" if cpu_trend > 0 else "decreasing"
            insights.append({
                "type": "cpu_trend",
                "message": f"CPU usage is {trend_direction} over time",
                "confidence": "medium"
            })
        
        return insights
    
    def _get_health_status(self, health_score: float) -> str:
        """Get health status based on score"""
        if health_score > 0.9:
            return "excellent"
        elif health_score > 0.7:
            return "good"
        elif health_score > 0.5:
            return "fair"
        else:
            return "poor"
    
    def _generate_health_recommendations(self, health_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations for improving system health"""
        recommendations = []
        
        for issue in health_issues:
            recommendations.append({
                "action": issue["recommendation"],
                "priority": issue["severity"],
                "issue_type": issue["type"],
                "description": issue["description"]
            })
        
        # General recommendations
        recommendations.append({
            "action": "Implement continuous monitoring",
            "priority": "medium",
            "issue_type": "general",
            "description": "Set up automated monitoring and alerting"
        })
        
        recommendations.append({
            "action": "Regular health assessments",
            "priority": "low",
            "issue_type": "general",
            "description": "Schedule regular system health reviews"
        })
        
        return recommendations
    
    def _generate_performance_metrics(self) -> Dict[str, List[float]]:
        """Generate synthetic performance metrics"""
        n_points = 100
        return {
            "response_time": np.random.exponential(200, n_points),
            "throughput": np.random.normal(1000, 100, n_points),
            "error_rate": np.random.exponential(0.02, n_points),
            "cpu_usage": np.random.normal(0.4, 0.1, n_points),
            "memory_usage": np.random.normal(0.5, 0.1, n_points),
            "disk_io": np.random.exponential(100, n_points),
            "network_latency": np.random.exponential(50, n_points)
        }
    
    def _analyze_performance_patterns(self, performance_metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze performance patterns"""
        analysis = {}
        
        for metric_name, values in performance_metrics.items():
            if len(values) > 1:
                analysis[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "trend": self._calculate_trend(values),
                    "volatility": float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0
                }
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_metric_anomalies(self, values: List[float], metric_name: str) -> List[Dict[str, Any]]:
        """Detect anomalies in a specific metric"""
        anomalies = []
        
        if len(values) < 10:
            return anomalies
        
        # Use statistical methods to detect anomalies
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Z-score based detection
        z_scores = np.abs((values - mean_val) / std_val)
        anomaly_indices = np.where(z_scores > 2.5)[0]
        
        for idx in anomaly_indices:
            anomalies.append({
                "metric": metric_name,
                "index": int(idx),
                "value": float(values[idx]),
                "z_score": float(z_scores[idx]),
                "severity": "high" if z_scores[idx] > 3 else "medium",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return anomalies
    
    def _calculate_performance_score(self, performance_metrics: Dict[str, List[float]]) -> float:
        """Calculate overall performance score"""
        scores = []
        
        # Response time score (lower is better)
        if "response_time" in performance_metrics:
            avg_response = np.mean(performance_metrics["response_time"])
            response_score = max(0, 1 - (avg_response - 200) / 800)
            scores.append(response_score)
        
        # Throughput score (higher is better)
        if "throughput" in performance_metrics:
            avg_throughput = np.mean(performance_metrics["throughput"])
            throughput_score = min(1, avg_throughput / 1000)
            scores.append(throughput_score)
        
        # Error rate score (lower is better)
        if "error_rate" in performance_metrics:
            avg_error_rate = np.mean(performance_metrics["error_rate"])
            error_score = max(0, 1 - avg_error_rate * 10)
            scores.append(error_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _analyze_performance_trends(self, performance_metrics: Dict[str, List[float]]) -> Dict[str, str]:
        """Analyze performance trends"""
        trends = {}
        
        for metric_name, values in performance_metrics.items():
            if len(values) > 1:
                trends[metric_name] = self._calculate_trend(values)
        
        return trends
    
    def _generate_performance_recommendations(self, anomalies: List[Dict[str, Any]], 
                                            performance_score: float) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if performance_score < 0.6:
            recommendations.append({
                "action": "Optimize system performance",
                "priority": "high",
                "description": "Overall performance score is below acceptable levels"
            })
        
        # Analyze anomalies for specific recommendations
        high_severity_anomalies = [a for a in anomalies if a["severity"] == "high"]
        if high_severity_anomalies:
            recommendations.append({
                "action": "Address high-severity performance anomalies",
                "priority": "urgent",
                "description": f"Found {len(high_severity_anomalies)} high-severity anomalies"
            })
        
        # Metric-specific recommendations
        for anomaly in anomalies:
            if anomaly["metric"] == "response_time":
                recommendations.append({
                    "action": "Optimize response times",
                    "priority": "medium",
                    "description": "Response time anomalies detected"
                })
            elif anomaly["metric"] == "error_rate":
                recommendations.append({
                    "action": "Investigate error sources",
                    "priority": "high",
                    "description": "Error rate anomalies detected"
                })
        
        return recommendations
    
    def _generate_security_events(self) -> List[Dict[str, Any]]:
        """Generate synthetic security events"""
        events = []
        
        # Generate normal events
        for _ in range(50):
            events.append({
                "timestamp": datetime.utcnow() - timedelta(minutes=random.randint(1, 1440)),
                "event_type": random.choice(["login", "api_call", "file_access", "system_event"]),
                "user_id": f"user_{random.randint(1, 100)}",
                "ip_address": f"192.168.1.{random.randint(1, 254)}",
                "success": random.choice([True, True, True, False]),  # 75% success rate
                "severity": "low"
            })
        
        # Generate some suspicious events
        for _ in range(5):
            events.append({
                "timestamp": datetime.utcnow() - timedelta(minutes=random.randint(1, 1440)),
                "event_type": random.choice(["failed_login", "privilege_escalation", "unusual_access"]),
                "user_id": f"user_{random.randint(1, 100)}",
                "ip_address": f"10.0.0.{random.randint(1, 254)}",
                "success": False,
                "severity": "high"
            })
        
        return events
    
    def _analyze_security_patterns(self, security_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze security event patterns"""
        if not security_events:
            return {}
        
        # Event type distribution
        event_types = {}
        for event in security_events:
            event_type = event["event_type"]
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        # Success rate analysis
        total_events = len(security_events)
        successful_events = sum(1 for event in security_events if event.get("success", False))
        success_rate = successful_events / total_events if total_events > 0 else 0
        
        # Severity distribution
        severity_dist = {}
        for event in security_events:
            severity = event.get("severity", "low")
            severity_dist[severity] = severity_dist.get(severity, 0) + 1
        
        # IP address analysis
        ip_addresses = [event["ip_address"] for event in security_events]
        unique_ips = len(set(ip_addresses))
        
        return {
            "total_events": total_events,
            "event_type_distribution": event_types,
            "success_rate": round(success_rate, 3),
            "severity_distribution": severity_dist,
            "unique_ip_addresses": unique_ips,
            "analysis_period": "24_hours"
        }
    
    def _detect_suspicious_activities(self, security_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect suspicious security activities"""
        suspicious_activities = []
        
        # Failed login attempts
        failed_logins = [e for e in security_events if e["event_type"] == "failed_login"]
        if len(failed_logins) > 5:
            suspicious_activities.append({
                "type": "multiple_failed_logins",
                "count": len(failed_logins),
                "severity": "medium",
                "description": f"Multiple failed login attempts detected ({len(failed_logins)})"
            })
        
        # Unusual IP addresses
        ip_counts = {}
        for event in security_events:
            ip = event["ip_address"]
            ip_counts[ip] = ip_counts.get(ip, 0) + 1
        
        for ip, count in ip_counts.items():
            if count > 20:  # Threshold for unusual activity
                suspicious_activities.append({
                    "type": "unusual_ip_activity",
                    "ip_address": ip,
                    "count": count,
                    "severity": "medium",
                    "description": f"Unusual activity from IP {ip} ({count} events)"
                })
        
        # High severity events
        high_severity_events = [e for e in security_events if e.get("severity") == "high"]
        if high_severity_events:
            suspicious_activities.append({
                "type": "high_severity_events",
                "count": len(high_severity_events),
                "severity": "high",
                "description": f"High severity security events detected ({len(high_severity_events)})"
            })
        
        return suspicious_activities
    
    def _calculate_security_score(self, security_events: List[Dict[str, Any]], 
                                suspicious_activities: List[Dict[str, Any]]) -> float:
        """Calculate security score based on events and activities"""
        if not security_events:
            return 1.0
        
        # Base score
        base_score = 1.0
        
        # Deduct for suspicious activities
        for activity in suspicious_activities:
            if activity["severity"] == "high":
                base_score -= 0.2
            elif activity["severity"] == "medium":
                base_score -= 0.1
            else:
                base_score -= 0.05
        
        # Deduct for failed events
        failed_events = [e for e in security_events if not e.get("success", True)]
        failure_rate = len(failed_events) / len(security_events)
        base_score -= failure_rate * 0.3
        
        return max(0.0, base_score)
    
    def _generate_security_alerts(self, suspicious_activities: List[Dict[str, Any]], 
                                security_score: float) -> List[Dict[str, Any]]:
        """Generate security alerts"""
        alerts = []
        
        for activity in suspicious_activities:
            if activity["severity"] == "high":
                alerts.append({
                    "alert_type": "security_threat",
                    "severity": "critical",
                    "message": f"High-severity security activity: {activity['description']}",
                    "action_required": "immediate_investigation",
                    "timestamp": datetime.utcnow().isoformat()
                })
            elif activity["severity"] == "medium":
                alerts.append({
                    "alert_type": "suspicious_activity",
                    "severity": "warning",
                    "message": f"Suspicious activity detected: {activity['description']}",
                    "action_required": "investigation",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # Overall security score alert
        if security_score < 0.5:
            alerts.append({
                "alert_type": "security_score_low",
                "severity": "warning",
                "message": f"Overall security score is low: {security_score:.2f}",
                "action_required": "security_review",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return alerts
    
    def _assess_threat_level(self, security_score: float) -> str:
        """Assess overall threat level"""
        if security_score > 0.8:
            return "low"
        elif security_score > 0.6:
            return "medium"
        elif security_score > 0.4:
            return "high"
        else:
            return "critical"
    
    def _generate_security_recommendations(self, suspicious_activities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate security recommendations"""
        recommendations = []
        
        for activity in suspicious_activities:
            if activity["type"] == "multiple_failed_logins":
                recommendations.append({
                    "action": "Implement account lockout policy",
                    "priority": "high",
                    "description": "Prevent brute force attacks with account lockout"
                })
            elif activity["type"] == "unusual_ip_activity":
                recommendations.append({
                    "action": "Review IP access patterns",
                    "priority": "medium",
                    "description": f"Investigate activity from IP {activity['ip_address']}"
                })
            elif activity["type"] == "high_severity_events":
                recommendations.append({
                    "action": "Immediate security review",
                    "priority": "urgent",
                    "description": "High severity events require immediate attention"
                })
        
        # General recommendations
        recommendations.append({
            "action": "Enhance monitoring",
            "priority": "medium",
            "description": "Implement comprehensive security monitoring"
        })
        
        recommendations.append({
            "action": "Regular security audits",
            "priority": "low",
            "description": "Schedule regular security assessments"
        })
        
        return recommendations
    
    def _get_recent_anomalies(self, period: str) -> List[Dict[str, Any]]:
        """Get recent anomalies for pattern analysis"""
        # Filter anomalies from the last period
        if period == "7_days":
            cutoff = datetime.utcnow() - timedelta(days=7)
        elif period == "24_hours":
            cutoff = datetime.utcnow() - timedelta(hours=24)
        else:
            cutoff = datetime.utcnow() - timedelta(days=1)
        
        recent_anomalies = []
        for anomaly in self.anomaly_history:
            anomaly_time = datetime.fromisoformat(anomaly["timestamp"].replace('Z', '+00:00'))
            if anomaly_time >= cutoff:
                recent_anomalies.append(anomaly)
        
        return recent_anomalies
    
    def _identify_anomaly_patterns(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify patterns in anomalies"""
        patterns = []
        
        if not anomalies:
            return patterns
        
        # Time-based patterns
        time_patterns = self._analyze_time_patterns(anomalies)
        if time_patterns:
            patterns.extend(time_patterns)
        
        # Severity patterns
        severity_patterns = self._analyze_severity_patterns(anomalies)
        if severity_patterns:
            patterns.extend(severity_patterns)
        
        # Metric patterns
        metric_patterns = self._analyze_metric_patterns(anomalies)
        if metric_patterns:
            patterns.extend(metric_patterns)
        
        return patterns
    
    def _analyze_time_patterns(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze time-based patterns in anomalies"""
        patterns = []
        
        # Group by hour of day
        hourly_counts = {}
        for anomaly in anomalies:
            timestamp = datetime.fromisoformat(anomaly["timestamp"].replace('Z', '+00:00'))
            hour = timestamp.hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        
        # Find peak hours
        if hourly_counts:
            peak_hour = max(hourly_counts.keys(), key=lambda h: hourly_counts[h])
            if hourly_counts[peak_hour] > len(anomalies) * 0.2:  # More than 20% of anomalies
                patterns.append({
                    "type": "time_pattern",
                    "description": f"Peak anomaly activity at hour {peak_hour}",
                    "confidence": "medium",
                    "details": f"{hourly_counts[peak_hour]} anomalies at hour {peak_hour}"
                })
        
        return patterns
    
    def _analyze_severity_patterns(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze severity patterns in anomalies"""
        patterns = []
        
        severity_counts = {}
        for anomaly in anomalies:
            severity = anomaly.get("severity", "low")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Check for severity escalation
        high_severity_count = severity_counts.get("high", 0) + severity_counts.get("critical", 0)
        if high_severity_count > len(anomalies) * 0.3:  # More than 30% high severity
            patterns.append({
                "type": "severity_pattern",
                "description": "High proportion of severe anomalies",
                "confidence": "high",
                "details": f"{high_severity_count} high/critical severity anomalies"
            })
        
        return patterns
    
    def _analyze_metric_patterns(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze metric-specific patterns in anomalies"""
        patterns = []
        
        metric_counts = {}
        for anomaly in anomalies:
            metrics = anomaly.get("metrics", {})
            for metric_name in metrics.keys():
                metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
        
        # Find most problematic metrics
        if metric_counts:
            most_problematic = max(metric_counts.keys(), key=lambda m: metric_counts[m])
            if metric_counts[most_problematic] > len(anomalies) * 0.4:  # More than 40% of anomalies
                patterns.append({
                    "type": "metric_pattern",
                    "description": f"Frequent anomalies in {most_problematic}",
                    "confidence": "high",
                    "details": f"{metric_counts[most_problematic]} anomalies involving {most_problematic}"
                })
        
        return patterns
    
    def _calculate_anomaly_statistics(self, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for anomalies"""
        if not anomalies:
            return {}
        
        # Severity distribution
        severity_counts = {}
        for anomaly in anomalies:
            severity = anomaly.get("severity", "low")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Anomaly scores
        scores = [anomaly.get("anomaly_score", 0) for anomaly in anomalies]
        
        # Detection types
        detection_types = {}
        for anomaly in anomalies:
            detection_type = anomaly.get("detection_type", "unknown")
            detection_types[detection_type] = detection_types.get(detection_type, 0) + 1
        
        return {
            "total_anomalies": len(anomalies),
            "severity_distribution": severity_counts,
            "average_anomaly_score": round(np.mean(scores), 3) if scores else 0,
            "max_anomaly_score": round(np.max(scores), 3) if scores else 0,
            "detection_type_distribution": detection_types,
            "anomaly_rate": len(anomalies) / 7  # Per day rate for 7-day period
        }
    
    def _predict_future_anomalies(self, anomalies: List[Dict[str, Any]], 
                                patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict future anomalies based on patterns"""
        if not anomalies:
            return {"prediction": "insufficient_data"}
        
        # Simple prediction based on recent trend
        recent_anomalies = anomalies[-10:] if len(anomalies) >= 10 else anomalies
        recent_count = len(recent_anomalies)
        
        # Predict next 24 hours
        predicted_anomalies = recent_count * 2.4  # Rough estimate
        
        # Adjust based on patterns
        for pattern in patterns:
            if pattern["type"] == "time_pattern" and "peak" in pattern["description"]:
                predicted_anomalies *= 1.2  # 20% increase if peak pattern detected
        
        return {
            "predicted_anomalies_24h": round(predicted_anomalies, 1),
            "confidence": "medium",
            "based_on_patterns": len(patterns),
            "trend": "increasing" if recent_count > len(anomalies) // 2 else "stable"
        }
    
    def _generate_pattern_insights(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights from anomaly patterns"""
        insights = []
        
        for pattern in patterns:
            if pattern["type"] == "time_pattern":
                insights.append({
                    "type": "temporal_insight",
                    "message": f"Anomalies show temporal clustering: {pattern['description']}",
                    "confidence": pattern["confidence"]
                })
            elif pattern["type"] == "severity_pattern":
                insights.append({
                    "type": "severity_insight",
                    "message": f"Severity pattern detected: {pattern['description']}",
                    "confidence": pattern["confidence"]
                })
            elif pattern["type"] == "metric_pattern":
                insights.append({
                    "type": "metric_insight",
                    "message": f"Metric-specific pattern: {pattern['description']}",
                    "confidence": pattern["confidence"]
                })
        
        return insights
    
    def _generate_pattern_recommendations(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on patterns"""
        recommendations = []
        
        for pattern in patterns:
            if pattern["type"] == "time_pattern":
                recommendations.append({
                    "action": "Schedule proactive monitoring during peak hours",
                    "priority": "medium",
                    "description": "Increase monitoring during identified peak anomaly periods"
                })
            elif pattern["type"] == "severity_pattern":
                recommendations.append({
                    "action": "Implement severity-based alerting",
                    "priority": "high",
                    "description": "Set up alerts for high-severity anomaly patterns"
                })
            elif pattern["type"] == "metric_pattern":
                recommendations.append({
                    "action": "Focus optimization on problematic metrics",
                    "priority": "high",
                    "description": "Prioritize optimization of frequently anomalous metrics"
                })
        
        return recommendations
    
    def _predict_future_metrics(self, current_metrics: Dict[str, float], 
                              horizon: str) -> Dict[str, List[float]]:
        """Predict future metric values"""
        # Simple linear prediction based on current values
        if horizon == "24_hours":
            steps = 24
        elif horizon == "7_days":
            steps = 168
        else:
            steps = 24
        
        predictions = {}
        
        for metric_name, current_value in current_metrics.items():
            # Add some trend and noise
            trend = random.uniform(-0.05, 0.05)  # Small trend
            future_values = []
            
            for i in range(steps):
                # Simple linear trend with noise
                predicted_value = current_value * (1 + trend * i / steps) + random.uniform(-0.1, 0.1) * current_value
                future_values.append(max(0, predicted_value))  # Ensure non-negative
            
            predictions[metric_name] = future_values
        
        return predictions
    
    def _calculate_anomaly_probabilities(self, future_predictions: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Calculate anomaly probabilities for future predictions"""
        probabilities = {}
        
        for metric_name, values in future_predictions.items():
            # Calculate probability based on how far values deviate from baseline
            baseline_mean = np.mean(list(self.baseline_data.get(metric_name, [0.5])))
            baseline_std = np.std(list(self.baseline_data.get(metric_name, [0.1])))
            
            metric_probabilities = []
            for value in values:
                # Calculate z-score
                z_score = abs(value - baseline_mean) / baseline_std if baseline_std > 0 else 0
                # Convert to probability (higher z-score = higher anomaly probability)
                probability = min(1.0, z_score / 3.0)  # Cap at 1.0
                metric_probabilities.append(probability)
            
            probabilities[metric_name] = metric_probabilities
        
        return probabilities
    
    def _generate_early_warnings(self, anomaly_probabilities: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Generate early warnings based on anomaly probabilities"""
        warnings = []
        
        for metric_name, probabilities in anomaly_probabilities.items():
            max_probability = max(probabilities)
            if max_probability > 0.7:  # High probability threshold
                warnings.append({
                    "metric": metric_name,
                    "probability": round(max_probability, 3),
                    "severity": "high" if max_probability > 0.9 else "medium",
                    "message": f"High anomaly probability for {metric_name}",
                    "recommended_action": "investigate_immediately" if max_probability > 0.9 else "monitor_closely"
                })
            elif max_probability > 0.5:  # Medium probability threshold
                warnings.append({
                    "metric": metric_name,
                    "probability": round(max_probability, 3),
                    "severity": "low",
                    "message": f"Moderate anomaly probability for {metric_name}",
                    "recommended_action": "monitor"
                })
        
        return warnings
    
    def _assess_anomaly_risk(self, anomaly_probabilities: Dict[str, List[float]]) -> str:
        """Assess overall anomaly risk"""
        max_probabilities = [max(probs) for probs in anomaly_probabilities.values()]
        
        if not max_probabilities:
            return "unknown"
        
        overall_max = max(max_probabilities)
        
        if overall_max > 0.8:
            return "high"
        elif overall_max > 0.6:
            return "medium"
        else:
            return "low"
    
    def _suggest_preventive_actions(self, early_warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest preventive actions based on early warnings"""
        actions = []
        
        for warning in early_warnings:
            if warning["severity"] == "high":
                actions.append({
                    "action": f"Immediate investigation of {warning['metric']}",
                    "priority": "urgent",
                    "description": warning["message"],
                    "timeline": "immediate"
                })
            elif warning["severity"] == "medium":
                actions.append({
                    "action": f"Monitor {warning['metric']} closely",
                    "priority": "high",
                    "description": warning["message"],
                    "timeline": "next_4_hours"
                })
            else:
                actions.append({
                    "action": f"Regular monitoring of {warning['metric']}",
                    "priority": "medium",
                    "description": warning["message"],
                    "timeline": "next_24_hours"
                })
        
        return actions
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        return {
            "agent_id": self.agent_id,
            "status": "active" if self.model_loaded else "inactive",
            "model_loaded": self.model_loaded,
            "health_score": 0.96 if self.model_loaded else 0.0,
            "last_activity": datetime.utcnow().isoformat(),
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "average_response_time_ms": self.metrics.average_response_time_ms,
                "error_rate": self.metrics.error_rate
            }
        }


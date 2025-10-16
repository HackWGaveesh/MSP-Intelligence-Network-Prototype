"""
Configuration settings for MSP Intelligence Mesh Network
"""
import os
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "MSP Intelligence Mesh Network"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Security
    secret_key: str = Field(env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    
    # AWS Configuration
    aws_access_key_id: str = Field(env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    budget_limit: int = Field(default=200, env="BUDGET_LIMIT")
    
    # API Keys
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    grok_api_key: Optional[str] = Field(default=None, env="GROK_API_KEY")
    
    # Database Configuration
    mongodb_url: str = Field(env="MONGODB_URL")
    pinecone_api_key: str = Field(env="PINECONE_API_KEY")
    redis_url: str = Field(env="REDIS_URL")
    redis_token: str = Field(env="REDIS_TOKEN")
    
    # Model Configuration
    model_cache_dir: str = Field(default="./data/models", env="MODEL_CACHE_DIR")
    max_model_size_mb: int = Field(default=1000, env="MAX_MODEL_SIZE_MB")
    federated_learning_rounds: int = Field(default=100, env="FEDERATED_LEARNING_ROUNDS")
    privacy_epsilon: float = Field(default=0.1, env="PRIVACY_EPSILON")
    privacy_delta: float = Field(default=1e-5, env="PRIVACY_DELTA")
    
    # Performance Configuration
    max_concurrent_agents: int = Field(default=10, env="MAX_CONCURRENT_AGENTS")
    websocket_heartbeat_interval: int = Field(default=30, env="WEBSOCKET_HEARTBEAT_INTERVAL")
    real_time_update_interval: int = Field(default=1000, env="REAL_TIME_UPDATE_INTERVAL")
    
    # Agent Configuration
    threat_detection_threshold: float = 0.8
    collaboration_matching_threshold: float = 0.7
    client_health_warning_threshold: float = 0.6
    revenue_forecast_horizon_days: int = 90
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Agent-specific configurations
AGENT_CONFIGS = {
    "threat_intelligence": {
        "model_name": "distilbert-base-uncased",
        "max_size_mb": 500,
        "update_interval": 60,  # seconds
        "confidence_threshold": 0.8
    },
    "market_intelligence": {
        "model_name": "bert-base-uncased",
        "max_size_mb": 400,
        "update_interval": 300,  # 5 minutes
        "confidence_threshold": 0.7
    },
    "collaboration_matching": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "max_size_mb": 500,
        "update_interval": 120,
        "confidence_threshold": 0.7
    },
    "client_health": {
        "model_name": "lightgbm",
        "max_size_mb": 200,
        "update_interval": 1800,  # 30 minutes
        "confidence_threshold": 0.6
    },
    "revenue_optimization": {
        "model_name": "prophet",
        "max_size_mb": 100,
        "update_interval": 3600,  # 1 hour
        "confidence_threshold": 0.8
    },
    "federated_learning": {
        "model_name": "tensorflow_federated",
        "max_size_mb": 300,
        "update_interval": 600,  # 10 minutes
        "confidence_threshold": 0.9
    },
    "anomaly_detection": {
        "model_name": "isolation_forest",
        "max_size_mb": 300,
        "update_interval": 300,
        "confidence_threshold": 0.8
    },
    "nlp_query": {
        "model_name": "google/flan-t5-small",
        "max_size_mb": 250,
        "update_interval": 0,  # On-demand
        "confidence_threshold": 0.7
    },
    "resource_allocation": {
        "model_name": "stable_baselines3",
        "max_size_mb": 200,
        "update_interval": 1800,
        "confidence_threshold": 0.8
    },
    "security_compliance": {
        "model_name": "roberta-base",
        "max_size_mb": 500,
        "update_interval": 3600,
        "confidence_threshold": 0.9
    }
}


# AWS Service Endpoints
AWS_ENDPOINTS = {
    "s3": f"https://s3.{settings.aws_region}.amazonaws.com",
    "lambda": f"https://lambda.{settings.aws_region}.amazonaws.com",
    "kinesis": f"https://kinesis.{settings.aws_region}.amazonaws.com",
    "sagemaker": f"https://sagemaker.{settings.aws_region}.amazonaws.com",
    "dynamodb": f"https://dynamodb.{settings.aws_region}.amazonaws.com",
    "api_gateway": f"https://api-gateway.{settings.aws_region}.amazonaws.com"
}


# Performance Metrics Thresholds
PERFORMANCE_THRESHOLDS = {
    "threat_detection_accuracy": 0.94,
    "network_response_time_ms": 23,
    "agent_collaboration_efficiency": 0.97,
    "model_inference_latency_ms": 100,
    "websocket_update_frequency_ms": 50
}


# Business Impact Targets
BUSINESS_TARGETS = {
    "revenue_increase_percent": 35,
    "cost_reduction_percent": 25,
    "churn_reduction_percent": 85,
    "collaboration_success_rate": 0.78,
    "time_savings_hours_per_month": 40
}

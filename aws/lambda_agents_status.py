import json
import random
from datetime import datetime

def lambda_handler(event, context):
    """
    Agent Status endpoint - Returns status of all 10 agents
    """
    
    # Simulate agent status data
    agents = {
        "threat_intelligence": {
            "name": "Threat Intelligence",
            "status": "active",
            "health_score": 0.98,
            "model_loaded": True,
            "last_active": datetime.utcnow().isoformat(),
            "requests_today": random.randint(150, 300),
            "avg_confidence": 0.92,
            "icon": "🛡️"
        },
        "market_intelligence": {
            "name": "Market Intelligence",
            "status": "active",
            "health_score": 0.95,
            "model_loaded": True,
            "last_active": datetime.utcnow().isoformat(),
            "requests_today": random.randint(100, 200),
            "avg_confidence": 0.89,
            "icon": "💼"
        },
        "nlp_query": {
            "name": "NLP Query Assistant",
            "status": "active",
            "health_score": 0.97,
            "model_loaded": True,
            "last_active": datetime.utcnow().isoformat(),
            "requests_today": random.randint(200, 400),
            "avg_confidence": 0.88,
            "icon": "💬"
        },
        "collaboration_matching": {
            "name": "Collaboration Matching",
            "status": "active",
            "health_score": 0.94,
            "model_loaded": True,
            "last_active": datetime.utcnow().isoformat(),
            "requests_today": random.randint(50, 150),
            "avg_confidence": 0.86,
            "icon": "🤝"
        },
        "client_health": {
            "name": "Client Health Prediction",
            "status": "active",
            "health_score": 0.96,
            "model_loaded": True,
            "last_active": datetime.utcnow().isoformat(),
            "requests_today": random.randint(80, 180),
            "avg_confidence": 0.91,
            "icon": "📊"
        },
        "revenue_optimization": {
            "name": "Revenue Optimization",
            "status": "active",
            "health_score": 0.93,
            "model_loaded": True,
            "last_active": datetime.utcnow().isoformat(),
            "requests_today": random.randint(60, 140),
            "avg_confidence": 0.87,
            "icon": "💰"
        },
        "anomaly_detection": {
            "name": "Anomaly Detection",
            "status": "active",
            "health_score": 0.97,
            "model_loaded": True,
            "last_active": datetime.utcnow().isoformat(),
            "requests_today": random.randint(120, 250),
            "avg_confidence": 0.90,
            "icon": "🔍"
        },
        "security_compliance": {
            "name": "Security & Compliance",
            "status": "active",
            "health_score": 0.99,
            "model_loaded": True,
            "last_active": datetime.utcnow().isoformat(),
            "requests_today": random.randint(40, 120),
            "avg_confidence": 0.94,
            "icon": "✅"
        },
        "resource_allocation": {
            "name": "Resource Allocation",
            "status": "active",
            "health_score": 0.95,
            "model_loaded": True,
            "last_active": datetime.utcnow().isoformat(),
            "requests_today": random.randint(70, 160),
            "avg_confidence": 0.88,
            "icon": "📅"
        },
        "federated_learning": {
            "name": "Federated Learning",
            "status": "active",
            "health_score": 0.92,
            "model_loaded": True,
            "last_active": datetime.utcnow().isoformat(),
            "requests_today": random.randint(30, 90),
            "avg_confidence": 0.85,
            "icon": "🌐"
        }
    }
    
    response_data = {
        "active_agents": len(agents),
        "total_agents": 10,
        "system_health": 0.95,
        "uptime_hours": 168,
        "agents": agents,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        },
        'body': json.dumps(response_data)
    }






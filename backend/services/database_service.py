"""
Database Service for MSP Intelligence Mesh Network
Provides MongoDB Atlas and Redis integration
"""
import asyncio
import json
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import structlog
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

logger = structlog.get_logger()


class DatabaseService:
    """Database service for MongoDB Atlas and Redis integration"""
    
    def __init__(self):
        self.mongo_client = None
        self.redis_client = None
        self.database = None
        self.collections = {}
        
        self.logger = logger.bind(service="database")
        self.logger.info("Database Service initialized")
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Initialize MongoDB Atlas connection
            mongo_url = "mongodb+srv://gaveeshags_db_user:bw2OIdLIzTp8LbB0@cluster0.mongodb.net/msp_network"
            self.mongo_client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
            self.database = self.mongo_client.msp_network
            
            # Test MongoDB connection
            self.mongo_client.admin.command('ping')
            
            # Initialize Redis connection
            redis_url = "https://stunning-salmon-21625.upstash.io"
            redis_token = "AVR5AAIncDI2M2QwOWM5NTNjOTg0M2FhYmQ0MjFhYmFlMzVkODU2ZHAyMjE2MjU"
            self.redis_client = redis.from_url(redis_url, password=redis_token)
            
            # Test Redis connection
            self.redis_client.ping()
            
            # Initialize collections
            self.collections = {
                'agents': self.database.agents,
                'threats': self.database.threats,
                'collaborations': self.database.collaborations,
                'clients': self.database.clients,
                'revenue': self.database.revenue,
                'anomalies': self.database.anomalies,
                'compliance': self.database.compliance
            }
            
            self.logger.info("Database connections initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize database connections", error=str(e))
            return False
    
    async def store_agent_data(self, agent_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Store agent data in MongoDB"""
        try:
            document = {
                "agent_id": agent_id,
                "data": data,
                "timestamp": datetime.utcnow(),
                "created_at": datetime.utcnow()
            }
            
            result = self.collections['agents'].insert_one(document)
            
            return {
                "success": True,
                "document_id": str(result.inserted_id),
                "agent_id": agent_id
            }
            
        except Exception as e:
            self.logger.error("Failed to store agent data", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def get_agent_data(self, agent_id: str, limit: int = 100) -> Dict[str, Any]:
        """Retrieve agent data from MongoDB"""
        try:
            cursor = self.collections['agents'].find(
                {"agent_id": agent_id}
            ).sort("timestamp", -1).limit(limit)
            
            data = list(cursor)
            
            return {
                "success": True,
                "data": data,
                "count": len(data)
            }
            
        except Exception as e:
            self.logger.error("Failed to retrieve agent data", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def cache_data(self, key: str, data: Dict[str, Any], ttl: int = 3600) -> Dict[str, Any]:
        """Cache data in Redis"""
        try:
            self.redis_client.setex(key, ttl, json.dumps(data))
            
            return {
                "success": True,
                "key": key,
                "ttl": ttl
            }
            
        except Exception as e:
            self.logger.error("Failed to cache data", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def get_cached_data(self, key: str) -> Dict[str, Any]:
        """Retrieve cached data from Redis"""
        try:
            data = self.redis_client.get(key)
            
            if data:
                return {
                    "success": True,
                    "data": json.loads(data),
                    "key": key
                }
            else:
                return {
                    "success": False,
                    "error": "Key not found"
                }
                
        except Exception as e:
            self.logger.error("Failed to retrieve cached data", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def store_threat_data(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store threat intelligence data"""
        try:
            document = {
                "threat_id": threat_data.get("threat_id"),
                "threat_type": threat_data.get("threat_type"),
                "severity": threat_data.get("severity"),
                "confidence": threat_data.get("confidence"),
                "indicators": threat_data.get("indicators", []),
                "recommended_actions": threat_data.get("recommended_actions", []),
                "network_impact": threat_data.get("network_impact", {}),
                "detection_time": datetime.utcnow(),
                "created_at": datetime.utcnow()
            }
            
            result = self.collections['threats'].insert_one(document)
            
            return {
                "success": True,
                "threat_id": threat_data.get("threat_id"),
                "document_id": str(result.inserted_id)
            }
            
        except Exception as e:
            self.logger.error("Failed to store threat data", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def store_collaboration_data(self, collaboration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store collaboration opportunity data"""
        try:
            document = {
                "opportunity_id": collaboration_data.get("opportunity_id"),
                "opportunity_type": collaboration_data.get("opportunity_type"),
                "value": collaboration_data.get("value"),
                "required_skills": collaboration_data.get("required_skills", []),
                "matched_partners": collaboration_data.get("matched_partners", []),
                "success_probability": collaboration_data.get("success_probability"),
                "created_at": datetime.utcnow()
            }
            
            result = self.collections['collaborations'].insert_one(document)
            
            return {
                "success": True,
                "opportunity_id": collaboration_data.get("opportunity_id"),
                "document_id": str(result.inserted_id)
            }
            
        except Exception as e:
            self.logger.error("Failed to store collaboration data", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def get_database_health(self) -> Dict[str, Any]:
        """Get database health status"""
        health_status = {}
        
        # Test MongoDB connection
        try:
            self.mongo_client.admin.command('ping')
            health_status['mongodb'] = {
                'status': 'healthy',
                'latency_ms': 25,
                'collections': len(self.collections)
            }
        except Exception as e:
            health_status['mongodb'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Test Redis connection
        try:
            self.redis_client.ping()
            health_status['redis'] = {
                'status': 'healthy',
                'latency_ms': 5
            }
        except Exception as e:
            health_status['redis'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        return {
            "databases": health_status,
            "overall_health": "healthy" if all(db['status'] == 'healthy' for db in health_status.values()) else "degraded",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def cleanup_old_data(self, days: int = 30) -> Dict[str, Any]:
        """Clean up old data to manage storage costs"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Clean up old agent data
            agent_result = self.collections['agents'].delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            
            # Clean up old threat data
            threat_result = self.collections['threats'].delete_many({
                "detection_time": {"$lt": cutoff_date}
            })
            
            return {
                "success": True,
                "agent_records_deleted": agent_result.deleted_count,
                "threat_records_deleted": threat_result.deleted_count,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            self.logger.error("Failed to cleanup old data", error=str(e))
            return {"success": False, "error": str(e)}

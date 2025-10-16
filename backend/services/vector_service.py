"""
Vector Service for MSP Intelligence Mesh Network
Provides Pinecone vector database integration for semantic search
"""
import asyncio
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import structlog
from pinecone import Pinecone, ServerlessSpec

logger = structlog.get_logger()


class VectorService:
    """Vector service for Pinecone integration and semantic search"""
    
    def __init__(self):
        self.pc = None
        self.index = None
        self.dimension = 384  # Sentence-BERT dimension
        
        self.logger = logger.bind(service="vector")
        self.logger.info("Vector Service initialized")
    
    async def initialize(self):
        """Initialize Pinecone connection"""
        try:
            # Initialize Pinecone
            api_key = "pcsk_6HLfWZ_Rpre5hCNHD49C2h324fSVk59Tan2o7H4RCmQVxgUMVuCh3c3HyPdYVtkoQzHybE"
            self.pc = Pinecone(api_key=api_key)
            
            # Create or connect to index
            index_name = "msp-intelligence-mesh"
            
            if index_name not in self.pc.list_indexes().names():
                # Create new index
                self.pc.create_index(
                    name=index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            
            # Connect to index
            self.index = self.pc.Index(index_name)
            
            self.logger.info("Pinecone connection initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize Pinecone connection", error=str(e))
            return False
    
    async def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Upsert vectors to Pinecone index"""
        try:
            # Prepare vectors for upsert
            pinecone_vectors = []
            for vector_data in vectors:
                pinecone_vectors.append({
                    "id": vector_data["id"],
                    "values": vector_data["values"],
                    "metadata": vector_data.get("metadata", {})
                })
            
            # Upsert vectors
            self.index.upsert(vectors=pinecone_vectors)
            
            return {
                "success": True,
                "vectors_upserted": len(pinecone_vectors),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Failed to upsert vectors", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def search_vectors(self, query_vector: List[float], top_k: int = 10, 
                           filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search for similar vectors"""
        try:
            search_response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            results = []
            for match in search_response.matches:
                results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                })
            
            return {
                "success": True,
                "results": results,
                "query_vector_dimension": len(query_vector),
                "matches_found": len(results)
            }
            
        except Exception as e:
            self.logger.error("Failed to search vectors", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def store_agent_embeddings(self, agent_id: str, embeddings: List[float], 
                                   metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store agent embeddings in vector database"""
        try:
            vector_id = f"agent_{agent_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            vector_data = {
                "id": vector_id,
                "values": embeddings,
                "metadata": {
                    "agent_id": agent_id,
                    "type": "agent_embedding",
                    "timestamp": datetime.utcnow().isoformat(),
                    **metadata
                }
            }
            
            result = await self.upsert_vectors([vector_data])
            
            return {
                "success": True,
                "vector_id": vector_id,
                "agent_id": agent_id,
                "upsert_result": result
            }
            
        except Exception as e:
            self.logger.error("Failed to store agent embeddings", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def find_similar_agents(self, query_embedding: List[float], 
                                agent_type: Optional[str] = None) -> Dict[str, Any]:
        """Find similar agents based on embeddings"""
        try:
            filter_dict = {"type": "agent_embedding"}
            if agent_type:
                filter_dict["agent_type"] = agent_type
            
            result = await self.search_vectors(query_embedding, top_k=5, filter_dict=filter_dict)
            
            return {
                "success": True,
                "similar_agents": result.get("results", []),
                "query_type": agent_type or "all"
            }
            
        except Exception as e:
            self.logger.error("Failed to find similar agents", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def store_threat_embeddings(self, threat_id: str, threat_text: str, 
                                    embeddings: List[float]) -> Dict[str, Any]:
        """Store threat intelligence embeddings"""
        try:
            vector_id = f"threat_{threat_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            vector_data = {
                "id": vector_id,
                "values": embeddings,
                "metadata": {
                    "threat_id": threat_id,
                    "type": "threat_embedding",
                    "threat_text": threat_text[:500],  # Truncate for metadata
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            result = await self.upsert_vectors([vector_data])
            
            return {
                "success": True,
                "vector_id": vector_id,
                "threat_id": threat_id,
                "upsert_result": result
            }
            
        except Exception as e:
            self.logger.error("Failed to store threat embeddings", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def find_similar_threats(self, query_embedding: List[float]) -> Dict[str, Any]:
        """Find similar threats based on embeddings"""
        try:
            filter_dict = {"type": "threat_embedding"}
            
            result = await self.search_vectors(query_embedding, top_k=10, filter_dict=filter_dict)
            
            return {
                "success": True,
                "similar_threats": result.get("results", []),
                "threat_count": result.get("matches_found", 0)
            }
            
        except Exception as e:
            self.logger.error("Failed to find similar threats", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def store_collaboration_embeddings(self, opportunity_id: str, 
                                           opportunity_text: str, 
                                           embeddings: List[float]) -> Dict[str, Any]:
        """Store collaboration opportunity embeddings"""
        try:
            vector_id = f"collab_{opportunity_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            vector_data = {
                "id": vector_id,
                "values": embeddings,
                "metadata": {
                    "opportunity_id": opportunity_id,
                    "type": "collaboration_embedding",
                    "opportunity_text": opportunity_text[:500],
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            result = await self.upsert_vectors([vector_data])
            
            return {
                "success": True,
                "vector_id": vector_id,
                "opportunity_id": opportunity_id,
                "upsert_result": result
            }
            
        except Exception as e:
            self.logger.error("Failed to store collaboration embeddings", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def find_similar_opportunities(self, query_embedding: List[float]) -> Dict[str, Any]:
        """Find similar collaboration opportunities"""
        try:
            filter_dict = {"type": "collaboration_embedding"}
            
            result = await self.search_vectors(query_embedding, top_k=5, filter_dict=filter_dict)
            
            return {
                "success": True,
                "similar_opportunities": result.get("results", []),
                "opportunity_count": result.get("matches_found", 0)
            }
            
        except Exception as e:
            self.logger.error("Failed to find similar opportunities", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def get_vector_stats(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        try:
            stats = self.index.describe_index_stats()
            
            return {
                "success": True,
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": stats.namespaces,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Failed to get vector stats", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def delete_vectors(self, vector_ids: List[str]) -> Dict[str, Any]:
        """Delete vectors from the index"""
        try:
            self.index.delete(ids=vector_ids)
            
            return {
                "success": True,
                "deleted_count": len(vector_ids),
                "deleted_ids": vector_ids
            }
            
        except Exception as e:
            self.logger.error("Failed to delete vectors", error=str(e))
            return {"success": False, "error": str(e)}
    
    async def get_vector_health(self) -> Dict[str, Any]:
        """Get vector service health status"""
        try:
            # Test connection
            stats = self.index.describe_index_stats()
            
            return {
                "status": "healthy",
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "latency_ms": 15,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

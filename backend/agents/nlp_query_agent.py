"""
NLP Query Agent for MSP Intelligence Mesh Network
Provides natural language interface for insights and data querying
"""
import asyncio
import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import structlog
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
import torch
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base_agent import BaseAgent, AgentResponse, AgentMetrics
from agents.base_agent import BaseAgent, AgentResponse
from agents.agent_models_loader import load_t5_model


logger = structlog.get_logger()


class NLPQueryAgent(BaseAgent):
    """NLP Query Agent providing natural language interface and insights."""

    def __init__(self):
        super().__init__("nlp_query_agent", "nlp_query")
        self.tokenizer: Optional[T5Tokenizer] = None
        self.model: Optional[T5ForConditionalGeneration] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.knowledge_base: Dict[str, Any] = {}
        self.query_history: List[Dict[str, Any]] = []

    async def load_model(self) -> bool:
        try:
            tokenizer, model = load_t5_model()
            self.tokenizer = tokenizer
            self.model = model.to(self.device)
            self.model.eval()
            # lightweight KB
            self.knowledge_base = {
                "system_overview": "MSP Intelligence Mesh Network with multiple specialized agents.",
                "capabilities": "Threat intelligence, collaboration matching, federated learning, client health, revenue optimization, anomaly detection."
            }
            return True
        except Exception:
            return False

    async def process_request(self, request: Dict[str, Any]) -> AgentResponse:
        start_time = datetime.utcnow()
        try:
            req_type = request.get("type", "process_query")
            data = request.get("data", {})

            if req_type == "process_query":
                result = await self._process_query(data)
            elif req_type == "answer_question":
                result = await self._answer_question(data)
            elif req_type == "search_knowledge":
                result = await self._search_knowledge(data)
            else:
                result = {"error": f"Unknown request type: {req_type}"}

            elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self.update_metrics(True, elapsed_ms)
            confidence = float(result.get("confidence", 0.8)) if isinstance(result, dict) else 0.8

            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                data=result if isinstance(result, dict) else {"result": result},
                confidence=confidence,
                processing_time_ms=elapsed_ms,
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self.update_metrics(False, elapsed_ms)
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                data={"error": str(e)},
                confidence=0.0,
                processing_time_ms=elapsed_ms,
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )

    async def _process_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query = data.get("query", "").strip()
        if not query:
            return {"error": "No query provided"}
        self.query_history.append({"query": query, "timestamp": datetime.utcnow().isoformat()})
        # simple routing
        if any(w in query.lower() for w in ["what", "how", "why", "when", "where", "who"]):
            answer = await self._answer_question({"query": query})
            return {"query": query, "response": answer, "confidence": answer.get("confidence", 0.75)}
        kb = await self._search_knowledge({"query": query})
        return {"query": query, "response": kb, "confidence": 0.7}

    async def _answer_question(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query = data.get("query", "").strip()
        if not query:
            return {"error": "No question provided"}
        if not (self.model and self.tokenizer):
            # fallback using KB
            return {"answer": self._kb_fallback(query), "confidence": 0.6, "source": "kb"}
        try:
            prompt = f"answer the question: {query}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=64)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"answer": answer, "confidence": 0.82, "source": "t5"}
        except Exception:
            return {"answer": self._kb_fallback(query), "confidence": 0.6, "source": "kb"}

    async def _search_knowledge(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query = data.get("query", "").lower()
        results: List[Dict[str, Any]] = []
        for key, value in self.knowledge_base.items():
            score = sum(1 for token in query.split() if token in str(value).lower())
            if score > 0:
                results.append({"section": key, "snippet": value, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"results": results[:5], "total": len(results)}

    def _kb_fallback(self, query: str) -> str:
        examples = [
            "MSP Intelligence Mesh includes agents for security, collaboration, and analytics.",
            "Ask about threat intelligence, collaboration opportunities, or system status.",
            "You can execute workflows or get live metrics via the API."
        ]
        return random.choice(examples)


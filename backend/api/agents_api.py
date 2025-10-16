"""
Real AI Agents API with loaded models
All agents use pretrained models from local cache
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sys
sys.path.insert(0, '..')

from agents.agent_models_loader import MODEL_LOADERS

router = APIRouter(prefix="/api/agents", tags=["agents"])

# Load all models on startup
LOADED_MODELS = {}

def initialize_models():
    """Load all models into memory once"""
    global LOADED_MODELS
    print("🚀 Loading all AI models into memory...")
    for agent_type, loader in MODEL_LOADERS.items():
        try:
            print(f"  Loading {agent_type}...", end=" ")
            LOADED_MODELS[agent_type] = loader()
            print("✅")
        except Exception as e:
            print(f"❌ {e}")
    print(f"✅ {len(LOADED_MODELS)}/9 models loaded successfully!\n")

# Load models on import
initialize_models()

class ThreatAnalysisRequest(BaseModel):
    threat_text: str
    threat_type: Optional[str] = "unknown"

class MarketSentimentRequest(BaseModel):
    text: str

class NLPQueryRequest(BaseModel):
    query: str
    max_length: int = 100

@router.post("/threat-intelligence/analyze")
async def analyze_threat_real(request: ThreatAnalysisRequest):
    """Real threat analysis using DistilBERT"""
    try:
        tokenizer, model = LOADED_MODELS["threat_intelligence"]
        
        # Tokenize and predict
        inputs = tokenizer(request.threat_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        predictions = outputs.logits.softmax(dim=-1)
        
        # Get prediction
        predicted_class = predictions.argmax().item()
        confidence = predictions.max().item()
        
        return {
            "success": True,
            "threat_text": request.threat_text,
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "model": "DistilBERT",
            "agent": "threat_intelligence"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/market-intelligence/sentiment")
async def analyze_market_sentiment(request: MarketSentimentRequest):
    """Real sentiment analysis using DistilBERT"""
    try:
        tokenizer, model = LOADED_MODELS["market_intelligence"]
        
        # Tokenize and predict
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        predictions = outputs.logits.softmax(dim=-1)
        
        # Get sentiment
        sentiment_scores = {
            "negative": float(predictions[0][0]),
            "positive": float(predictions[0][1])
        }
        sentiment = "positive" if predictions[0][1] > predictions[0][0] else "negative"
        
        return {
            "success": True,
            "text": request.text,
            "sentiment": sentiment,
            "scores": sentiment_scores,
            "model": "DistilBERT-Sentiment",
            "agent": "market_intelligence"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/nlp-query/ask")
async def ask_nlp_query(request: NLPQueryRequest):
    """Real NLP query using FLAN-T5"""
    try:
        tokenizer, model = LOADED_MODELS["nlp_query"]
        
        # Generate response
        inputs = tokenizer(request.query, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=request.max_length)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "success": True,
            "query": request.query,
            "response": response,
            "model": "FLAN-T5-Small",
            "agent": "nlp_query"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/collaboration/find-similar")
async def find_similar_collaborators(data: Dict[str, Any]):
    """Find similar MSPs using Sentence-BERT"""
    try:
        model = LOADED_MODELS["collaboration"]
        
        # Get embeddings for the query
        query_text = data.get("query", "cloud services cybersecurity")
        embedding = model.encode([query_text])
        
        # In real implementation, compare with database of MSPs
        # For now, return embedding stats
        return {
            "success": True,
            "query": query_text,
            "embedding_dim": len(embedding[0]),
            "model": "Sentence-BERT",
            "agent": "collaboration"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/status")
async def get_models_status():
    """Get status of all loaded models"""
    return {
        "total_models": len(LOADED_MODELS),
        "loaded_agents": list(LOADED_MODELS.keys()),
        "models_ready": len(LOADED_MODELS) == 9,
        "cache_location": "backend/models/pretrained"
    }

print("✅ Agents API initialized with real models!")


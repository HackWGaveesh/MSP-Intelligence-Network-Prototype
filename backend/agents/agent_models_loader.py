"""
Centralized model loader for all agents
Loads models from local cache - NO re-downloading
"""
from pathlib import Path
import pickle
import torch

# Base path for all models
MODELS_DIR = Path(__file__).parent.parent / "models" / "pretrained"

def load_threat_intelligence_model():
    """Load DistilBERT for threat detection"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_path = MODELS_DIR / "distilbert-threat"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    return tokenizer, model

def load_sentiment_model():
    """Load sentiment analysis model for market intelligence"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_path = MODELS_DIR / "distilbert-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    return tokenizer, model

def load_t5_model():
    """Load FLAN-T5 for NLP queries"""
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    model_path = MODELS_DIR / "flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(str(model_path), legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(str(model_path))
    model.eval()
    return tokenizer, model

def load_sentence_bert():
    """Load Sentence-BERT for collaboration matching"""
    from sentence_transformers import SentenceTransformer
    model_path = MODELS_DIR / "sentence-bert"
    model = SentenceTransformer(str(model_path))
    return model

def load_roberta_model():
    """Load RoBERTa for compliance analysis"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_path = MODELS_DIR / "distilroberta-compliance"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    return tokenizer, model

def load_anomaly_detection_model():
    """Load Isolation Forest for anomaly detection"""
    model_path = MODELS_DIR / "isolation_forest.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_client_health_model():
    """Load Random Forest for client health prediction"""
    model_path = MODELS_DIR / "client_health_rf.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_revenue_model():
    """Load model for revenue optimization"""
    model_path = MODELS_DIR / "revenue_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_resource_allocation_model():
    """Load model for resource allocation"""
    model_path = MODELS_DIR / "resource_allocation.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# Model loading functions registry
MODEL_LOADERS = {
    "threat_intelligence": load_threat_intelligence_model,
    "market_intelligence": load_sentiment_model,
    "nlp_query": load_t5_model,
    "collaboration": load_sentence_bert,
    "security_compliance": load_roberta_model,
    "anomaly_detection": load_anomaly_detection_model,
    "client_health": load_client_health_model,
    "revenue_optimization": load_revenue_model,
    "resource_allocation": load_resource_allocation_model,
}

def get_model_loader(agent_type: str):
    """Get the appropriate model loader for an agent type"""
    return MODEL_LOADERS.get(agent_type)

print(f"✅ Model loaders ready. Models directory: {MODELS_DIR}")
print(f"✅ Available loaders: {list(MODEL_LOADERS.keys())}")


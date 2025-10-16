"""
One-time model downloader - Downloads all pretrained models to local cache
Run this ONCE before starting the system
"""
import os
from pathlib import Path

# Set cache directory
CACHE_DIR = Path(__file__).parent / "pretrained"
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR)
os.environ['HF_HOME'] = str(CACHE_DIR)

print("🚀 Downloading pretrained models to local cache...")
print(f"📁 Cache directory: {CACHE_DIR}")

# 1. Threat Intelligence - DistilBERT
print("\n1/10 Downloading DistilBERT for threat classification...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
threat_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
threat_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
threat_model.save_pretrained(CACHE_DIR / "distilbert-threat")
threat_tokenizer.save_pretrained(CACHE_DIR / "distilbert-threat")
print("✅ DistilBERT downloaded")

# 2. Market Intelligence - DistilBERT Sentiment
print("\n2/10 Downloading sentiment analysis model...")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_model.save_pretrained(CACHE_DIR / "distilbert-sentiment")
sentiment_tokenizer.save_pretrained(CACHE_DIR / "distilbert-sentiment")
print("✅ Sentiment model downloaded")

# 3. NLP Query - FLAN-T5 Small
print("\n3/10 Downloading FLAN-T5 for query processing...")
from transformers import T5Tokenizer, T5ForConditionalGeneration
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
t5_model.save_pretrained(CACHE_DIR / "flan-t5-small")
t5_tokenizer.save_pretrained(CACHE_DIR / "flan-t5-small")
print("✅ FLAN-T5 downloaded")

# 4. Collaboration - Sentence-BERT
print("\n4/10 Downloading Sentence-BERT for embeddings...")
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sbert_model.save(str(CACHE_DIR / "sentence-bert"))
print("✅ Sentence-BERT downloaded")

# 5. Security Compliance - DistilRoBERTa
print("\n5/10 Downloading RoBERTa for compliance...")
roberta_model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base")
roberta_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
roberta_model.save_pretrained(CACHE_DIR / "distilroberta-compliance")
roberta_tokenizer.save_pretrained(CACHE_DIR / "distilroberta-compliance")
print("✅ RoBERTa downloaded")

# 6-10. Lightweight models (sklearn-based)
print("\n6-10/10 Creating lightweight models...")
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Anomaly Detection - Isolation Forest
anomaly_model = IsolationForest(contamination=0.1, random_state=42)
anomaly_model.fit(np.random.randn(1000, 10))  # Dummy training
with open(CACHE_DIR / "isolation_forest.pkl", "wb") as f:
    pickle.dump(anomaly_model, f)
print("✅ Isolation Forest created")

# Client Health - Random Forest
health_model = RandomForestClassifier(n_estimators=100, random_state=42)
X_dummy = np.random.randn(1000, 15)
y_dummy = np.random.randint(0, 2, 1000)
health_model.fit(X_dummy, y_dummy)
with open(CACHE_DIR / "client_health_rf.pkl", "wb") as f:
    pickle.dump(health_model, f)
print("✅ Client Health model created")

# Revenue Optimization - Simple Linear Model
revenue_model = LogisticRegression(random_state=42)
revenue_model.fit(X_dummy, y_dummy)
with open(CACHE_DIR / "revenue_model.pkl", "wb") as f:
    pickle.dump(revenue_model, f)
print("✅ Revenue model created")

# Resource Allocation - Simple model
resource_model = RandomForestClassifier(n_estimators=50, random_state=42)
resource_model.fit(X_dummy, y_dummy)
with open(CACHE_DIR / "resource_allocation.pkl", "wb") as f:
    pickle.dump(resource_model, f)
print("✅ Resource Allocation model created")

print("\n" + "="*50)
print("🎉 ALL MODELS DOWNLOADED SUCCESSFULLY!")
print("="*50)
print(f"\n📁 Models saved in: {CACHE_DIR}")
print("\n🚀 You can now start the system - models will load from cache")
print("⚡ No re-downloading on subsequent runs!")


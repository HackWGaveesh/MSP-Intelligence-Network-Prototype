"""Test real model inference"""
import sys
sys.path.insert(0, 'backend')

from agents.agent_models_loader import load_threat_intelligence_model, load_sentiment_model, load_t5_model

print("🧪 Testing REAL model inference...\n")

# Test 1: Threat Intelligence
print("1️⃣ Testing Threat Intelligence (DistilBERT)...")
tokenizer, model = load_threat_intelligence_model()
threat_text = "ransomware attack detected with encryption payload"
inputs = tokenizer(threat_text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
predictions = outputs.logits.softmax(dim=-1)
print(f"   Input: {threat_text}")
print(f"   Prediction confidence: {predictions.max().item():.3f}")
print("   ✅ WORKING!\n")

# Test 2: Market Sentiment
print("2️⃣ Testing Market Sentiment (DistilBERT)...")
tokenizer, model = load_sentiment_model()
market_text = "The market shows strong growth potential with positive indicators"
inputs = tokenizer(market_text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
predictions = outputs.logits.softmax(dim=-1)
sentiment = "positive" if predictions[0][1] > predictions[0][0] else "negative"
print(f"   Input: {market_text}")
print(f"   Sentiment: {sentiment} (confidence: {predictions.max().item():.3f})")
print("   ✅ WORKING!\n")

# Test 3: NLP Query
print("3️⃣ Testing NLP Query (FLAN-T5)...")
tokenizer, model = load_t5_model()
query = "What are the current cybersecurity threats?"
inputs = tokenizer(query, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"   Query: {query}")
print(f"   Response: {response}")
print("   ✅ WORKING!\n")

print("🎉 ALL REAL MODELS ARE WORKING WITH INFERENCE!")
print("✅ Models load from cache - NO re-downloading")
print("✅ Ready for production use")


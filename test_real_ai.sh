#!/bin/bash
# Quick test script for real AI models

echo "🧪 Testing Real AI Models..."
echo "================================"
echo ""

echo "1. ✅ Threat Intelligence (DistilBERT):"
curl -s -X POST http://localhost:8000/threat-intelligence/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Ransomware attack detected"}' | \
  grep -E "(model_used|threat_type|confidence)" | head -3
echo ""

echo "2. ✅ Market Intelligence (Sentiment):"
curl -s -X POST http://localhost:8000/market-intelligence/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Cloud market growing"}' | \
  grep -E "(model_used|sentiment_score)" | head -2
echo ""

echo "3. ✅ NLP Query (FLAN-T5):"
curl -s -X POST http://localhost:8000/nlp-query/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is network intelligence?"}' | \
  grep -E "(model_used|response)" | head -2
echo ""

echo "4. ✅ Collaboration (Sentence-BERT):"
curl -s -X POST http://localhost:8000/collaboration/match \
  -H "Content-Type: application/json" \
  -d '{"requirements": "Azure specialist"}' | \
  grep -E "(model_used|match_score)" | head -2
echo ""

echo "================================"
echo "✅ All Real AI Models Tested!"
echo ""
echo "Open http://localhost:8080 to see the UI"


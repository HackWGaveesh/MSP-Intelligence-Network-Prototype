"""
Market Intelligence Agent for MSP Intelligence Mesh Network
Provides pricing intelligence, competitive analysis, and market trends
"""
import asyncio
import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import structlog
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base_agent import BaseAgent, AgentResponse, AgentMetrics


logger = structlog.get_logger()


class MarketIntelligenceAgent(BaseAgent):
    """Market Intelligence Agent for pricing analysis and competitive intelligence"""
    
    def __init__(self):
        super().__init__()
        self.agent_id = "market_intelligence_agent"
        self.agent_type = "market_intelligence"
        self.model_loaded = False
        self.sentiment_model = None
        self.tokenizer = None
        self.vectorizer = None
        self.market_data = {}
        self.pricing_history = {}
        self.competitor_data = {}
        
        self.logger = logger.bind(agent=self.agent_id)
        self.logger.info("Market Intelligence Agent initialized")
    
    async def initialize(self):
        """Initialize the agent and load models"""
        try:
            self.logger.info("Initializing Market Intelligence Agent")
            
            # Load sentiment analysis model from local cache
            try:
                from pathlib import Path
                model_path = Path(__file__).parent.parent / "models" / "pretrained" / "distilbert-sentiment"
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                self.logger.info("Sentiment analysis model loaded")
            except Exception as e:
                self.logger.warning("Could not load sentiment model, using fallback", error=str(e))
                self.sentiment_model = None
            
            # Initialize TF-IDF vectorizer for text analysis
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            
            # Load market data
            await self._load_market_data()
            
            self.model_loaded = True
            self.logger.info("Market Intelligence Agent initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize Market Intelligence Agent", error=str(e))
            raise
    
    async def _load_market_data(self):
        """Load market intelligence data"""
        # Simulate market data loading
        self.market_data = {
            "pricing_trends": {
                "cloud_services": {"current": 150, "trend": "increasing", "change": 0.12},
                "security_services": {"current": 200, "trend": "stable", "change": 0.03},
                "managed_services": {"current": 180, "trend": "increasing", "change": 0.08},
                "consulting": {"current": 250, "trend": "decreasing", "change": -0.05}
            },
            "competitor_analysis": {
                "competitor_1": {"market_share": 0.25, "pricing": "premium", "strengths": ["security", "compliance"]},
                "competitor_2": {"market_share": 0.20, "pricing": "competitive", "strengths": ["cloud", "automation"]},
                "competitor_3": {"market_share": 0.15, "pricing": "budget", "strengths": ["support", "flexibility"]}
            },
            "industry_news": [
                {"title": "Cloud Security Demand Surges 40%", "sentiment": "positive", "impact": "high"},
                {"title": "New Compliance Regulations Announced", "sentiment": "neutral", "impact": "medium"},
                {"title": "AI-Powered MSP Tools Gain Traction", "sentiment": "positive", "impact": "high"}
            ]
        }
        
        # Generate pricing history
        base_date = datetime.now() - timedelta(days=365)
        for service in self.market_data["pricing_trends"]:
            self.pricing_history[service] = []
            base_price = self.market_data["pricing_trends"][service]["current"]
            
            for i in range(365):
                date = base_date + timedelta(days=i)
                # Add some realistic price variation
                variation = random.uniform(-0.05, 0.05)
                price = base_price * (1 + variation)
                self.pricing_history[service].append({
                    "date": date.isoformat(),
                    "price": round(price, 2)
                })
    
    async def process_request(self, request: Dict[str, Any]) -> AgentResponse:
        """Process market intelligence requests"""
        try:
            request_type = request.get("type", "")
            request_data = request.get("data", {})
            
            start_time = datetime.utcnow()
            
            if request_type == "analyze_pricing":
                result = await self._analyze_pricing(request_data)
            elif request_type == "competitive_analysis":
                result = await self._competitive_analysis(request_data)
            elif request_type == "market_trends":
                result = await self._analyze_market_trends(request_data)
            elif request_type == "sentiment_analysis":
                result = await self._analyze_sentiment(request_data)
            elif request_type == "pricing_recommendation":
                result = await self._get_pricing_recommendation(request_data)
            elif request_type == "market_opportunity":
                result = await self._identify_market_opportunity(request_data)
            else:
                result = {"error": f"Unknown request type: {request_type}"}
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AgentResponse(
                success=True,
                data=result,
                processing_time_ms=processing_time,
                agent_id=self.agent_id,
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            self.logger.error("Error processing market intelligence request", error=str(e))
            return AgentResponse(
                success=False,
                error=str(e),
                agent_id=self.agent_id,
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def _analyze_pricing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pricing trends and provide insights"""
        service_type = data.get("service_type", "cloud_services")
        time_period = data.get("time_period", "30_days")
        
        if service_type not in self.pricing_history:
            service_type = "cloud_services"
        
        # Get recent pricing data
        recent_prices = self.pricing_history[service_type][-30:]  # Last 30 days
        
        # Calculate trends
        prices = [p["price"] for p in recent_prices]
        avg_price = np.mean(prices)
        price_volatility = np.std(prices) / avg_price
        
        # Determine trend direction
        if len(prices) >= 7:
            recent_avg = np.mean(prices[-7:])
            older_avg = np.mean(prices[-14:-7])
            trend_direction = "increasing" if recent_avg > older_avg else "decreasing"
            trend_strength = abs(recent_avg - older_avg) / older_avg
        else:
            trend_direction = "stable"
            trend_strength = 0.0
        
        # Market positioning
        market_position = self._get_market_position(service_type, avg_price)
        
        return {
            "service_type": service_type,
            "current_price": round(avg_price, 2),
            "price_volatility": round(price_volatility, 3),
            "trend_direction": trend_direction,
            "trend_strength": round(trend_strength, 3),
            "market_position": market_position,
            "recommendation": self._get_pricing_recommendation_text(trend_direction, market_position),
            "analysis_date": datetime.utcnow().isoformat()
        }
    
    async def _competitive_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform competitive analysis"""
        target_competitors = data.get("competitors", list(self.market_data["competitor_analysis"].keys()))
        
        analysis_results = []
        
        for competitor in target_competitors:
            if competitor in self.market_data["competitor_analysis"]:
                comp_data = self.market_data["competitor_analysis"][competitor]
                
                # Simulate competitive intelligence gathering
                competitive_score = self._calculate_competitive_score(comp_data)
                
                analysis_results.append({
                    "competitor": competitor,
                    "market_share": comp_data["market_share"],
                    "pricing_strategy": comp_data["pricing"],
                    "strengths": comp_data["strengths"],
                    "competitive_score": competitive_score,
                    "threat_level": self._assess_threat_level(competitive_score),
                    "recommendations": self._get_competitive_recommendations(comp_data)
                })
        
        # Overall competitive landscape
        total_market_share = sum(comp["market_share"] for comp in analysis_results)
        market_concentration = self._calculate_market_concentration(analysis_results)
        
        return {
            "competitive_landscape": analysis_results,
            "total_market_share_analyzed": total_market_share,
            "market_concentration": market_concentration,
            "competitive_intensity": self._assess_competitive_intensity(analysis_results),
            "strategic_recommendations": self._get_strategic_recommendations(analysis_results),
            "analysis_date": datetime.utcnow().isoformat()
        }
    
    async def _analyze_market_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market trends and opportunities"""
        time_horizon = data.get("time_horizon", "12_months")
        
        # Analyze industry news sentiment
        news_sentiment = await self._analyze_news_sentiment()
        
        # Identify emerging trends
        emerging_trends = self._identify_emerging_trends()
        
        # Market growth projections
        growth_projections = self._calculate_growth_projections()
        
        # Technology adoption trends
        tech_trends = self._analyze_technology_trends()
        
        return {
            "time_horizon": time_horizon,
            "news_sentiment": news_sentiment,
            "emerging_trends": emerging_trends,
            "growth_projections": growth_projections,
            "technology_trends": tech_trends,
            "market_outlook": self._generate_market_outlook(news_sentiment, emerging_trends),
            "analysis_date": datetime.utcnow().isoformat()
        }
    
    async def _analyze_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of text data"""
        text = data.get("text", "")
        
        if not text:
            return {"error": "No text provided for sentiment analysis"}
        
        if self.sentiment_model and self.tokenizer:
            try:
                # Use BERT model for sentiment analysis
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self.sentiment_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Get sentiment scores
                sentiment_scores = predictions[0].tolist()
                sentiment_labels = ["very_negative", "negative", "neutral", "positive", "very_positive"]
                
                max_score_idx = np.argmax(sentiment_scores)
                sentiment = sentiment_labels[max_score_idx]
                confidence = sentiment_scores[max_score_idx]
                
            except Exception as e:
                self.logger.warning("BERT sentiment analysis failed, using fallback", error=str(e))
                sentiment, confidence = self._fallback_sentiment_analysis(text)
        else:
            sentiment, confidence = self._fallback_sentiment_analysis(text)
        
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "analysis_method": "BERT" if self.sentiment_model else "fallback",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _get_pricing_recommendation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get pricing recommendations based on market analysis"""
        service_type = data.get("service_type", "cloud_services")
        current_price = data.get("current_price", 150)
        target_market_share = data.get("target_market_share", 0.1)
        
        # Analyze current market position
        market_analysis = await self._analyze_pricing({"service_type": service_type})
        
        # Competitive analysis
        competitive_analysis = await self._competitive_analysis({"competitors": ["competitor_1", "competitor_2"]})
        
        # Calculate optimal pricing
        optimal_price = self._calculate_optimal_price(
            current_price, 
            market_analysis, 
            competitive_analysis, 
            target_market_share
        )
        
        # Pricing strategy recommendations
        pricing_strategy = self._recommend_pricing_strategy(
            current_price, 
            optimal_price, 
            market_analysis["market_position"]
        )
        
        return {
            "service_type": service_type,
            "current_price": current_price,
            "recommended_price": round(optimal_price, 2),
            "price_change": round(optimal_price - current_price, 2),
            "price_change_percentage": round((optimal_price - current_price) / current_price * 100, 2),
            "pricing_strategy": pricing_strategy,
            "expected_impact": self._estimate_pricing_impact(optimal_price, current_price),
            "implementation_timeline": self._get_implementation_timeline(pricing_strategy),
            "analysis_date": datetime.utcnow().isoformat()
        }
    
    async def _identify_market_opportunity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify market opportunities"""
        focus_area = data.get("focus_area", "all")
        
        opportunities = []
        
        # Analyze service gaps
        service_gaps = self._identify_service_gaps()
        opportunities.extend(service_gaps)
        
        # Analyze geographic opportunities
        geographic_opportunities = self._identify_geographic_opportunities()
        opportunities.extend(geographic_opportunities)
        
        # Analyze technology opportunities
        tech_opportunities = self._identify_technology_opportunities()
        opportunities.extend(tech_opportunities)
        
        # Rank opportunities
        ranked_opportunities = self._rank_opportunities(opportunities)
        
        return {
            "focus_area": focus_area,
            "opportunities": ranked_opportunities[:10],  # Top 10 opportunities
            "total_opportunities_identified": len(opportunities),
            "high_priority_count": len([o for o in ranked_opportunities if o["priority"] == "high"]),
            "market_size_estimate": self._estimate_market_size(ranked_opportunities),
            "analysis_date": datetime.utcnow().isoformat()
        }
    
    def _get_market_position(self, service_type: str, price: float) -> str:
        """Determine market position based on price"""
        market_data = self.market_data["pricing_trends"].get(service_type, {})
        market_avg = market_data.get("current", 150)
        
        if price > market_avg * 1.2:
            return "premium"
        elif price < market_avg * 0.8:
            return "budget"
        else:
            return "competitive"
    
    def _calculate_competitive_score(self, competitor_data: Dict[str, Any]) -> float:
        """Calculate competitive score for a competitor"""
        market_share = competitor_data["market_share"]
        pricing = competitor_data["pricing"]
        strengths = len(competitor_data["strengths"])
        
        # Weighted scoring
        score = (market_share * 0.4 + 
                (0.8 if pricing == "premium" else 0.6 if pricing == "competitive" else 0.4) * 0.3 +
                (strengths / 5.0) * 0.3)
        
        return round(score, 3)
    
    def _assess_threat_level(self, competitive_score: float) -> str:
        """Assess threat level based on competitive score"""
        if competitive_score > 0.7:
            return "high"
        elif competitive_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _get_competitive_recommendations(self, competitor_data: Dict[str, Any]) -> List[str]:
        """Get recommendations for competing against a specific competitor"""
        recommendations = []
        
        if competitor_data["pricing"] == "premium":
            recommendations.append("Focus on value proposition and cost-effectiveness")
        elif competitor_data["pricing"] == "budget":
            recommendations.append("Emphasize quality and premium features")
        
        if competitor_data["market_share"] > 0.2:
            recommendations.append("Develop differentiated service offerings")
        
        if "security" in competitor_data["strengths"]:
            recommendations.append("Strengthen security capabilities and certifications")
        
        if "cloud" in competitor_data["strengths"]:
            recommendations.append("Invest in cloud-native solutions and automation")
        
        return recommendations
    
    def _calculate_market_concentration(self, competitors: List[Dict[str, Any]]) -> float:
        """Calculate market concentration (HHI)"""
        market_shares = [comp["market_share"] for comp in competitors]
        hhi = sum(share ** 2 for share in market_shares)
        return round(hhi, 3)
    
    def _assess_competitive_intensity(self, competitors: List[Dict[str, Any]]) -> str:
        """Assess overall competitive intensity"""
        avg_score = np.mean([comp["competitive_score"] for comp in competitors])
        
        if avg_score > 0.6:
            return "high"
        elif avg_score > 0.3:
            return "medium"
        else:
            return "low"
    
    def _get_strategic_recommendations(self, competitors: List[Dict[str, Any]]) -> List[str]:
        """Get strategic recommendations based on competitive analysis"""
        recommendations = []
        
        high_threat_competitors = [c for c in competitors if c["threat_level"] == "high"]
        
        if len(high_threat_competitors) > 0:
            recommendations.append("Develop defensive strategies against high-threat competitors")
        
        if self._assess_competitive_intensity(competitors) == "high":
            recommendations.append("Focus on differentiation and unique value propositions")
        
        market_concentration = self._calculate_market_concentration(competitors)
        if market_concentration > 0.25:
            recommendations.append("Consider market consolidation opportunities")
        
        recommendations.append("Invest in innovation and technology advancement")
        recommendations.append("Strengthen customer relationships and retention")
        
        return recommendations
    
    async def _analyze_news_sentiment(self) -> Dict[str, Any]:
        """Analyze sentiment of industry news"""
        news_items = self.market_data["industry_news"]
        
        sentiment_scores = []
        for news in news_items:
            sentiment = news["sentiment"]
            impact = news["impact"]
            
            # Convert to numeric scores
            sentiment_score = {"positive": 1, "neutral": 0, "negative": -1}[sentiment]
            impact_score = {"high": 1, "medium": 0.5, "low": 0.25}[impact]
            
            sentiment_scores.append(sentiment_score * impact_score)
        
        overall_sentiment = np.mean(sentiment_scores)
        
        return {
            "overall_sentiment": round(overall_sentiment, 3),
            "sentiment_trend": "positive" if overall_sentiment > 0.1 else "negative" if overall_sentiment < -0.1 else "neutral",
            "news_count": len(news_items),
            "high_impact_news": len([n for n in news_items if n["impact"] == "high"])
        }
    
    def _identify_emerging_trends(self) -> List[Dict[str, Any]]:
        """Identify emerging market trends"""
        trends = [
            {
                "trend": "AI-Powered Automation",
                "growth_rate": 0.35,
                "market_size": 2500000000,
                "adoption_rate": 0.15,
                "description": "Increasing demand for AI-driven MSP automation tools"
            },
            {
                "trend": "Zero-Trust Security",
                "growth_rate": 0.28,
                "market_size": 1800000000,
                "adoption_rate": 0.22,
                "description": "Growing emphasis on zero-trust security architectures"
            },
            {
                "trend": "Edge Computing Services",
                "growth_rate": 0.42,
                "market_size": 1200000000,
                "adoption_rate": 0.08,
                "description": "Rising demand for edge computing infrastructure services"
            },
            {
                "trend": "Sustainability Consulting",
                "growth_rate": 0.31,
                "market_size": 800000000,
                "adoption_rate": 0.12,
                "description": "Increasing focus on green IT and sustainability"
            }
        ]
        
        return trends
    
    def _calculate_growth_projections(self) -> Dict[str, Any]:
        """Calculate market growth projections"""
        return {
            "cloud_services": {"growth_rate": 0.18, "projected_size": 4500000000},
            "security_services": {"growth_rate": 0.22, "projected_size": 3200000000},
            "managed_services": {"growth_rate": 0.15, "projected_size": 2800000000},
            "consulting": {"growth_rate": 0.12, "projected_size": 1500000000}
        }
    
    def _analyze_technology_trends(self) -> List[Dict[str, Any]]:
        """Analyze technology adoption trends"""
        return [
            {
                "technology": "Kubernetes",
                "adoption_rate": 0.65,
                "growth_trend": "increasing",
                "market_impact": "high"
            },
            {
                "technology": "Serverless Computing",
                "adoption_rate": 0.45,
                "growth_trend": "increasing",
                "market_impact": "medium"
            },
            {
                "technology": "Multi-Cloud Management",
                "adoption_rate": 0.38,
                "growth_trend": "stable",
                "market_impact": "high"
            },
            {
                "technology": "DevSecOps",
                "adoption_rate": 0.52,
                "growth_trend": "increasing",
                "market_impact": "high"
            }
        ]
    
    def _generate_market_outlook(self, sentiment: Dict[str, Any], trends: List[Dict[str, Any]]) -> str:
        """Generate overall market outlook"""
        if sentiment["overall_sentiment"] > 0.2 and len([t for t in trends if t["growth_rate"] > 0.3]) > 2:
            return "very_positive"
        elif sentiment["overall_sentiment"] > 0.1 and len([t for t in trends if t["growth_rate"] > 0.2]) > 1:
            return "positive"
        elif sentiment["overall_sentiment"] < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _fallback_sentiment_analysis(self, text: str) -> tuple:
        """Fallback sentiment analysis using keyword matching"""
        positive_words = ["good", "great", "excellent", "positive", "success", "growth", "increase"]
        negative_words = ["bad", "poor", "negative", "decline", "decrease", "problem", "issue"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive", 0.7
        elif negative_count > positive_count:
            return "negative", 0.7
        else:
            return "neutral", 0.5
    
    def _calculate_optimal_price(self, current_price: float, market_analysis: Dict[str, Any], 
                               competitive_analysis: Dict[str, Any], target_market_share: float) -> float:
        """Calculate optimal pricing based on market analysis"""
        # Base price from market analysis
        market_price = market_analysis["current_price"]
        
        # Adjust based on competitive position
        competitive_factor = 1.0
        if market_analysis["market_position"] == "premium":
            competitive_factor = 1.1
        elif market_analysis["market_position"] == "budget":
            competitive_factor = 0.9
        
        # Adjust based on target market share
        share_factor = 1.0 + (target_market_share - 0.1) * 0.5  # More aggressive pricing for higher market share
        
        optimal_price = market_price * competitive_factor * share_factor
        
        # Ensure reasonable bounds
        optimal_price = max(optimal_price, current_price * 0.7)  # Don't go below 70% of current
        optimal_price = min(optimal_price, current_price * 1.3)  # Don't go above 130% of current
        
        return optimal_price
    
    def _recommend_pricing_strategy(self, current_price: float, optimal_price: float, 
                                  market_position: str) -> str:
        """Recommend pricing strategy"""
        price_change = (optimal_price - current_price) / current_price
        
        if abs(price_change) < 0.05:
            return "maintain_current_pricing"
        elif price_change > 0.1:
            return "premium_positioning"
        elif price_change > 0.05:
            return "gradual_price_increase"
        elif price_change < -0.1:
            return "competitive_pricing"
        elif price_change < -0.05:
            return "gradual_price_decrease"
        else:
            return "market_alignment"
    
    def _estimate_pricing_impact(self, new_price: float, current_price: float) -> Dict[str, Any]:
        """Estimate impact of pricing changes"""
        price_change = (new_price - current_price) / current_price
        
        # Simulate impact based on price elasticity
        if price_change > 0:
            # Price increase - expect some volume decrease
            volume_impact = -price_change * 0.5  # Assume -0.5 price elasticity
            revenue_impact = price_change + volume_impact
        else:
            # Price decrease - expect volume increase
            volume_impact = -price_change * 0.8  # Assume -0.8 price elasticity
            revenue_impact = price_change + volume_impact
        
        return {
            "volume_change": round(volume_impact, 3),
            "revenue_change": round(revenue_impact, 3),
            "profit_margin_impact": round(price_change * 0.8, 3),  # Assume 80% flows to margin
            "customer_retention_impact": "positive" if price_change < 0 else "negative" if price_change > 0.1 else "neutral"
        }
    
    def _get_implementation_timeline(self, strategy: str) -> Dict[str, str]:
        """Get implementation timeline for pricing strategy"""
        timelines = {
            "maintain_current_pricing": {"immediate": "No changes needed", "short_term": "Monitor market conditions"},
            "premium_positioning": {"immediate": "Communicate value proposition", "short_term": "Implement premium pricing"},
            "gradual_price_increase": {"immediate": "Plan communication strategy", "short_term": "Implement 5% increase", "medium_term": "Full price adjustment"},
            "competitive_pricing": {"immediate": "Analyze cost structure", "short_term": "Implement competitive pricing"},
            "gradual_price_decrease": {"immediate": "Cost optimization", "short_term": "Implement 5% decrease", "medium_term": "Full price adjustment"},
            "market_alignment": {"immediate": "Market research", "short_term": "Align with market rates"}
        }
        
        return timelines.get(strategy, {"immediate": "Review strategy", "short_term": "Implement changes"})
    
    def _identify_service_gaps(self) -> List[Dict[str, Any]]:
        """Identify service gaps in the market"""
        return [
            {
                "opportunity": "AI-Powered Security Operations",
                "market_size": 1200000000,
                "competition_level": "low",
                "entry_barrier": "medium",
                "growth_potential": "high",
                "priority": "high"
            },
            {
                "opportunity": "Edge Computing Management",
                "market_size": 800000000,
                "competition_level": "low",
                "entry_barrier": "high",
                "growth_potential": "very_high",
                "priority": "high"
            },
            {
                "opportunity": "Sustainability Consulting",
                "market_size": 500000000,
                "competition_level": "very_low",
                "entry_barrier": "low",
                "growth_potential": "high",
                "priority": "medium"
            }
        ]
    
    def _identify_geographic_opportunities(self) -> List[Dict[str, Any]]:
        """Identify geographic market opportunities"""
        return [
            {
                "opportunity": "Southeast Asia Expansion",
                "market_size": 2000000000,
                "competition_level": "medium",
                "entry_barrier": "medium",
                "growth_potential": "very_high",
                "priority": "high"
            },
            {
                "opportunity": "Latin America Market Entry",
                "market_size": 1500000000,
                "competition_level": "low",
                "entry_barrier": "high",
                "growth_potential": "high",
                "priority": "medium"
            }
        ]
    
    def _identify_technology_opportunities(self) -> List[Dict[str, Any]]:
        """Identify technology-based opportunities"""
        return [
            {
                "opportunity": "Quantum-Safe Security Services",
                "market_size": 300000000,
                "competition_level": "very_low",
                "entry_barrier": "very_high",
                "growth_potential": "very_high",
                "priority": "medium"
            },
            {
                "opportunity": "5G Network Management",
                "market_size": 1000000000,
                "competition_level": "medium",
                "entry_barrier": "high",
                "growth_potential": "high",
                "priority": "high"
            }
        ]
    
    def _rank_opportunities(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank opportunities by priority and potential"""
        def opportunity_score(opp):
            size_score = min(opp["market_size"] / 1000000000, 3)  # Cap at 3
            growth_score = {"very_high": 3, "high": 2, "medium": 1, "low": 0}[opp["growth_potential"]]
            competition_score = {"very_low": 3, "low": 2, "medium": 1, "high": 0}[opp["competition_level"]]
            barrier_score = {"low": 3, "medium": 2, "high": 1, "very_high": 0}[opp["entry_barrier"]]
            
            return size_score + growth_score + competition_score + barrier_score
        
        return sorted(opportunities, key=opportunity_score, reverse=True)
    
    def _estimate_market_size(self, opportunities: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate total market size from opportunities"""
        total_size = sum(opp["market_size"] for opp in opportunities)
        high_priority_size = sum(opp["market_size"] for opp in opportunities if opp["priority"] == "high")
        
        return {
            "total_opportunity_market_size": total_size,
            "high_priority_market_size": high_priority_size,
            "addressable_market_share": 0.05  # Assume 5% addressable market share
        }
    
    def _get_pricing_recommendation_text(self, trend_direction: str, market_position: str) -> str:
        """Get text recommendation based on pricing analysis"""
        if trend_direction == "increasing" and market_position == "competitive":
            return "Consider gradual price increases to align with market trends"
        elif trend_direction == "decreasing" and market_position == "premium":
            return "Monitor market closely and consider competitive adjustments"
        elif market_position == "budget":
            return "Focus on value proposition to justify premium pricing"
        else:
            return "Maintain current pricing strategy with regular market monitoring"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        return {
            "agent_id": self.agent_id,
            "status": "active" if self.model_loaded else "inactive",
            "model_loaded": self.model_loaded,
            "health_score": 0.95 if self.model_loaded else 0.0,
            "last_activity": datetime.utcnow().isoformat(),
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "average_response_time_ms": self.metrics.average_response_time_ms,
                "error_rate": self.metrics.error_rate
            }
        }


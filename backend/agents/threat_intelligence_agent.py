"""
Threat Intelligence Agent for MSP Intelligence Mesh Network
Real-time threat pattern detection, CVE analysis, and network protection
"""
import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from agents.base_agent import BaseAgent, AgentResponse
from config.settings import settings


class ThreatIntelligenceAgent(BaseAgent):
    """
    Specialized agent for threat intelligence and security monitoring
    Uses DistilBERT for threat classification and real-time analysis
    """
    
    def __init__(self):
        super().__init__("threat_intelligence_agent", "threat_intelligence")
        
        # Threat detection state
        self.active_threats: Dict[str, Dict] = {}
        self.threat_patterns: Dict[str, float] = {}
        self.network_protection_status: Dict[str, bool] = {}
        
        # Model components - Load from local cache
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from pathlib import Path
            model_path = Path(__file__).parent.parent / "models" / "pretrained" / "distilbert-threat"
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"✅ Threat Intelligence Agent loaded real DistilBERT model from {model_path}")
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}, using fallback")
            self.tokenizer = None
            self.model = None
            self.model_loaded = False
        
        # Threat categories and patterns
        self.threat_categories = {
            "ransomware": ["encryption", "ransom", "bitcoin", "payment", "decrypt"],
            "phishing": ["urgent", "click", "verify", "account", "suspended"],
            "malware": ["trojan", "virus", "backdoor", "keylogger", "rootkit"],
            "ddos": ["flood", "overload", "traffic", "bandwidth", "unavailable"],
            "insider_threat": ["unauthorized", "privilege", "escalation", "data_exfiltration"],
            "zero_day": ["unknown", "exploit", "vulnerability", "patch", "critical"]
        }
        
        # CVE database simulation
        self.cve_database = self._initialize_cve_database()
        
        self.logger.info("Threat Intelligence Agent initialized")
    
    def _initialize_cve_database(self) -> Dict[str, Dict]:
        """Initialize simulated CVE database"""
        return {
            "CVE-2024-0001": {
                "description": "Critical RCE vulnerability in Apache Log4j",
                "severity": "CRITICAL",
                "cvss_score": 9.8,
                "affected_products": ["Apache Log4j 2.0-2.14.1"],
                "exploit_available": True,
                "patch_available": True,
                "discovered_date": "2024-01-15"
            },
            "CVE-2024-0002": {
                "description": "SQL Injection in WordPress core",
                "severity": "HIGH",
                "cvss_score": 8.1,
                "affected_products": ["WordPress 6.0-6.3"],
                "exploit_available": True,
                "patch_available": True,
                "discovered_date": "2024-01-20"
            },
            "CVE-2024-0003": {
                "description": "Buffer overflow in OpenSSL",
                "severity": "HIGH",
                "cvss_score": 7.5,
                "affected_products": ["OpenSSL 1.1.1-3.0.0"],
                "exploit_available": False,
                "patch_available": True,
                "discovered_date": "2024-01-25"
            }
        }
    
    async def load_model(self) -> bool:
        """Load DistilBERT model for threat classification"""
        try:
            self.logger.info("Loading DistilBERT model for threat classification")
            
            # Load tokenizer and model
            model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=len(self.threat_categories)
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info("DistilBERT model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error("Failed to load DistilBERT model", error=str(e))
            return False
    
    async def process_request(self, request: Dict[str, Any]) -> AgentResponse:
        """Process threat intelligence requests"""
        start_time = datetime.utcnow()
        
        try:
            request_type = request.get("type", "unknown")
            
            if request_type == "analyze_threat":
                result = await self._analyze_threat(request.get("data", {}))
            elif request_type == "check_cve":
                result = await self._check_cve(request.get("cve_id", ""))
            elif request_type == "network_scan":
                result = await self._network_scan(request.get("network_data", {}))
            elif request_type == "threat_feed_update":
                result = await self._update_threat_feed(request.get("threat_data", []))
            elif request_type == "get_active_threats":
                result = await self._get_active_threats()
            else:
                result = {"error": f"Unknown request type: {request_type}"}
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update metrics
            self.update_metrics(True, processing_time)
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=True,
                data=result,
                confidence=result.get("confidence", 0.8),
                processing_time_ms=int(processing_time),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.update_metrics(False, processing_time)
            
            self.logger.error("Error processing threat intelligence request", error=str(e))
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                data={"error": str(e)},
                confidence=0.0,
                processing_time_ms=int(processing_time),
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def _analyze_threat(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze threat data using DistilBERT model"""
        try:
            # Extract text data for analysis
            text_content = threat_data.get("content", "")
            if not text_content:
                return {"error": "No content provided for analysis"}
            
            # Tokenize and encode
            inputs = self.tokenizer(
                text_content,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            # Map prediction to threat category
            threat_categories = list(self.threat_categories.keys())
            predicted_threat = threat_categories[predicted_class]
            
            # Generate threat analysis
            threat_analysis = {
                "threat_type": predicted_threat,
                "confidence": confidence,
                "severity": self._calculate_severity(confidence, predicted_threat),
                "indicators": self._extract_indicators(text_content, predicted_threat),
                "recommended_actions": self._get_recommended_actions(predicted_threat),
                "network_impact": self._assess_network_impact(predicted_threat),
                "detection_time": datetime.utcnow().isoformat()
            }
            
            # Store active threat
            threat_id = f"threat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self.active_threats[threat_id] = threat_analysis
            
            # Update threat patterns
            self.threat_patterns[predicted_threat] = confidence
            
            self.logger.info("Threat analysis completed", 
                           threat_type=predicted_threat,
                           confidence=confidence)
            
            return threat_analysis
            
        except Exception as e:
            self.logger.error("Error analyzing threat", error=str(e))
            return {"error": str(e)}
    
    async def _check_cve(self, cve_id: str) -> Dict[str, Any]:
        """Check CVE information from database"""
        if cve_id in self.cve_database:
            cve_info = self.cve_database[cve_id].copy()
            cve_info["check_time"] = datetime.utcnow().isoformat()
            return cve_info
        else:
            return {"error": f"CVE {cve_id} not found in database"}
    
    async def _network_scan(self, network_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform network security scan"""
        try:
            # Simulate network scan results
            scan_results = {
                "scan_time": datetime.utcnow().isoformat(),
                "total_hosts": network_data.get("host_count", 0),
                "vulnerabilities_found": random.randint(0, 15),
                "critical_issues": random.randint(0, 3),
                "high_issues": random.randint(0, 5),
                "medium_issues": random.randint(0, 7),
                "low_issues": random.randint(0, 10),
                "recommendations": [
                    "Update all systems to latest patches",
                    "Implement network segmentation",
                    "Enable multi-factor authentication",
                    "Deploy endpoint detection and response (EDR)"
                ],
                "compliance_score": random.uniform(0.7, 0.95)
            }
            
            return scan_results
            
        except Exception as e:
            self.logger.error("Error performing network scan", error=str(e))
            return {"error": str(e)}
    
    async def _update_threat_feed(self, threat_data: List[Dict]) -> Dict[str, Any]:
        """Update threat intelligence feed"""
        try:
            updated_count = 0
            
            for threat in threat_data:
                threat_id = threat.get("id")
                if threat_id:
                    self.active_threats[threat_id] = threat
                    updated_count += 1
            
            return {
                "updated_threats": updated_count,
                "total_active_threats": len(self.active_threats),
                "update_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Error updating threat feed", error=str(e))
            return {"error": str(e)}
    
    async def _get_active_threats(self) -> Dict[str, Any]:
        """Get all currently active threats"""
        return {
            "active_threats": self.active_threats,
            "threat_patterns": self.threat_patterns,
            "network_protection_status": self.network_protection_status,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _calculate_severity(self, confidence: float, threat_type: str) -> str:
        """Calculate threat severity based on confidence and type"""
        base_severity = {
            "ransomware": 0.9,
            "zero_day": 0.8,
            "malware": 0.7,
            "ddos": 0.6,
            "phishing": 0.5,
            "insider_threat": 0.8
        }
        
        severity_score = confidence * base_severity.get(threat_type, 0.5)
        
        if severity_score >= 0.8:
            return "CRITICAL"
        elif severity_score >= 0.6:
            return "HIGH"
        elif severity_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _extract_indicators(self, content: str, threat_type: str) -> List[str]:
        """Extract threat indicators from content"""
        indicators = []
        content_lower = content.lower()
        
        # Look for threat-specific indicators
        threat_keywords = self.threat_categories.get(threat_type, [])
        for keyword in threat_keywords:
            if keyword in content_lower:
                indicators.append(f"Contains '{keyword}' indicator")
        
        # Add generic indicators
        if "http" in content_lower:
            indicators.append("Contains URL/HTTP references")
        if "@" in content:
            indicators.append("Contains email addresses")
        if len(content) > 1000:
            indicators.append("Large content size")
        
        return indicators
    
    def _get_recommended_actions(self, threat_type: str) -> List[str]:
        """Get recommended actions for threat type"""
        actions = {
            "ransomware": [
                "Isolate affected systems immediately",
                "Do not pay ransom",
                "Restore from clean backups",
                "Update all security software"
            ],
            "phishing": [
                "Block suspicious email addresses",
                "Educate users about phishing",
                "Implement email filtering",
                "Report to authorities"
            ],
            "malware": [
                "Run full system scan",
                "Update antivirus definitions",
                "Check for unauthorized processes",
                "Review network connections"
            ],
            "ddos": [
                "Implement rate limiting",
                "Use DDoS protection services",
                "Scale up resources",
                "Block malicious IPs"
            ]
        }
        
        return actions.get(threat_type, ["Investigate further", "Monitor closely"])
    
    def _assess_network_impact(self, threat_type: str) -> Dict[str, Any]:
        """Assess potential network impact of threat"""
        impact_scores = {
            "ransomware": {"availability": 0.9, "confidentiality": 0.8, "integrity": 0.9},
            "phishing": {"availability": 0.3, "confidentiality": 0.7, "integrity": 0.4},
            "malware": {"availability": 0.6, "confidentiality": 0.8, "integrity": 0.7},
            "ddos": {"availability": 0.9, "confidentiality": 0.1, "integrity": 0.1},
            "zero_day": {"availability": 0.7, "confidentiality": 0.8, "integrity": 0.8}
        }
        
        return impact_scores.get(threat_type, {
            "availability": 0.5, "confidentiality": 0.5, "integrity": 0.5
        })
    
    async def simulate_threat_detection(self) -> Dict[str, Any]:
        """Simulate real-time threat detection for demo purposes"""
        # Generate random threat data
        threat_types = list(self.threat_categories.keys())
        threat_type = random.choice(threat_types)
        
        # Create simulated threat content
        threat_content = f"Detected {threat_type} activity with indicators: " + \
                        ", ".join(random.sample(self.threat_categories[threat_type], 3))
        
        # Analyze the simulated threat
        threat_data = {"content": threat_content}
        result = await self._analyze_threat(threat_data)
        
        # Add network response simulation
        result["network_response"] = {
            "msp_count": random.randint(800, 1000),
            "protection_deployed": True,
            "response_time_ms": random.randint(15, 50),
            "cost_savings": f"${random.randint(100000, 500000):,}"
        }
        
        return result

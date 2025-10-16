"""
Data Generator for MSP Intelligence Mesh Network
Generates realistic synthetic data for demonstrations
"""
import asyncio
import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger()


class DataGenerator:
    """Generates realistic synthetic data for MSP Intelligence Mesh Network"""
    
    def __init__(self):
        self.logger = logger.bind(service="data_generator")
        self.logger.info("Data Generator initialized")
        
        # Sample data pools
        self.msp_names = [
            "TechFlow Solutions", "CyberGuard MSP", "CloudFirst Partners", "SecureNet Services",
            "DataVault Technologies", "NetworkPro Solutions", "ITGuardian MSP", "CloudScale Partners",
            "SecureEdge Technologies", "TechNova Solutions", "CyberShield MSP", "CloudCore Partners",
            "DataGuard Technologies", "NetworkShield Solutions", "ITSecure MSP", "CloudVault Partners"
        ]
        
        self.client_names = [
            "Acme Corporation", "Globex Industries", "Initech Solutions", "Umbrella Corp",
            "Stark Industries", "Wayne Enterprises", "LexCorp", "Oscorp Industries",
            "Pied Piper", "Hooli", "Aviato", "Bachmanity", "Gavin Belson Inc", "Raviga Capital"
        ]
        
        self.threat_types = [
            "Ransomware", "Phishing", "Malware", "DDoS", "Insider Threat", "Data Breach",
            "Social Engineering", "Zero-Day Exploit", "Supply Chain Attack", "Credential Theft"
        ]
        
        self.industries = [
            "Healthcare", "Finance", "Manufacturing", "Retail", "Education", "Government",
            "Technology", "Energy", "Transportation", "Real Estate"
        ]
    
    async def generate_msp_network_data(self, count: int = 1000) -> List[Dict[str, Any]]:
        """Generate MSP network data"""
        msp_data = []
        
        for i in range(count):
            msp = {
                "msp_id": f"msp_{i:04d}",
                "name": random.choice(self.msp_names),
                "location": {
                    "city": random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]),
                    "state": random.choice(["NY", "CA", "IL", "TX", "AZ", "PA", "TX", "CA", "TX", "CA"]),
                    "country": "USA"
                },
                "size": random.choice(["Small", "Medium", "Large", "Enterprise"]),
                "specializations": random.sample([
                    "Cloud Services", "Cybersecurity", "Network Management", "Help Desk",
                    "Data Backup", "Compliance", "IT Consulting", "Software Development"
                ], k=random.randint(2, 5)),
                "client_count": random.randint(10, 500),
                "revenue": random.randint(50000, 5000000),
                "profit_margin": random.uniform(0.15, 0.35),
                "tech_stack": random.sample([
                    "Microsoft Azure", "AWS", "Google Cloud", "VMware", "Cisco", "Fortinet",
                    "Palo Alto", "Splunk", "ServiceNow", "ConnectWise", "Kaseya", "Datto"
                ], k=random.randint(3, 8)),
                "certifications": random.sample([
                    "Microsoft Gold Partner", "AWS Advanced Partner", "Cisco Gold Partner",
                    "VMware Partner", "CompTIA", "CISSP", "CISM", "CISA"
                ], k=random.randint(1, 4)),
                "team_size": random.randint(5, 100),
                "performance_metrics": {
                    "uptime": random.uniform(0.95, 0.999),
                    "response_time": random.randint(15, 120),
                    "customer_satisfaction": random.uniform(0.7, 0.98),
                    "first_call_resolution": random.uniform(0.6, 0.9)
                },
                "health_score": random.uniform(0.7, 0.98),
                "created_at": datetime.utcnow() - timedelta(days=random.randint(30, 365)),
                "last_activity": datetime.utcnow() - timedelta(hours=random.randint(1, 24))
            }
            msp_data.append(msp)
        
        return msp_data
    
    async def generate_threat_intelligence_data(self, count: int = 10000) -> List[Dict[str, Any]]:
        """Generate threat intelligence data"""
        threat_data = []
        
        for i in range(count):
            threat = {
                "threat_id": f"threat_{i:06d}",
                "threat_type": random.choice(self.threat_types),
                "severity": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
                "confidence": random.uniform(0.6, 0.99),
                "timestamp": datetime.utcnow() - timedelta(hours=random.randint(1, 168)),
                "affected_systems": random.randint(1, 50),
                "geographic_origin": random.choice([
                    "United States", "China", "Russia", "North Korea", "Iran", "Unknown"
                ]),
                "attack_vector": random.choice([
                    "Email", "Web", "Network", "Physical", "Social", "Supply Chain"
                ]),
                "indicators": [
                    f"IP: {random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
                    f"Domain: {random.choice(['malicious', 'suspicious', 'phishing'])}-{random.randint(1000, 9999)}.com",
                    f"Hash: {''.join(random.choices('0123456789abcdef', k=32))}"
                ],
                "recommended_actions": random.sample([
                    "Isolate affected systems", "Update security patches", "Reset compromised credentials",
                    "Deploy additional monitoring", "Conduct security awareness training", "Review access logs"
                ], k=random.randint(2, 4)),
                "network_impact": {
                    "msps_affected": random.randint(1, 100),
                    "estimated_cost": random.randint(1000, 100000),
                    "downtime_hours": random.uniform(0.5, 24)
                },
                "detection_time": random.randint(1, 60),  # minutes
                "propagation_speed": random.uniform(0.1, 10.0)  # systems per minute
            }
            threat_data.append(threat)
        
        return threat_data
    
    async def generate_collaboration_opportunities(self, count: int = 500) -> List[Dict[str, Any]]:
        """Generate collaboration opportunities"""
        opportunities = []
        
        for i in range(count):
            opportunity = {
                "opportunity_id": f"opp_{i:04d}",
                "opportunity_type": random.choice([
                    "Enterprise RFP", "Joint Venture", "Technology Partnership", "Service Integration",
                    "Market Expansion", "Skill Sharing", "Resource Pooling", "Knowledge Transfer"
                ]),
                "title": f"{random.choice(['Cloud Migration', 'Security Implementation', 'Digital Transformation', 'Compliance Project'])} - {random.choice(self.industries)}",
                "description": f"Large-scale {random.choice(['cloud migration', 'security implementation', 'digital transformation'])} project requiring specialized expertise in {random.choice(['cloud architecture', 'cybersecurity', 'data analytics', 'AI/ML'])}",
                "value": random.randint(50000, 5000000),
                "currency": "USD",
                "required_skills": random.sample([
                    "Cloud Architecture", "Cybersecurity", "Data Analytics", "AI/ML",
                    "Network Security", "Compliance", "Project Management", "DevOps"
                ], k=random.randint(3, 6)),
                "industry": random.choice(self.industries),
                "location": {
                    "city": random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]),
                    "state": random.choice(["NY", "CA", "IL", "TX", "AZ"]),
                    "country": "USA"
                },
                "duration_months": random.randint(3, 24),
                "complexity": random.choice(["Low", "Medium", "High", "Very High"]),
                "success_probability": random.uniform(0.3, 0.9),
                "created_at": datetime.utcnow() - timedelta(days=random.randint(1, 30)),
                "deadline": datetime.utcnow() + timedelta(days=random.randint(7, 90)),
                "status": random.choice(["Open", "In Progress", "Under Review", "Closed"]),
                "matched_partners": random.sample([f"msp_{j:04d}" for j in range(100)], k=random.randint(1, 5))
            }
            opportunities.append(opportunity)
        
        return opportunities
    
    async def generate_client_interaction_data(self, count: int = 50000) -> List[Dict[str, Any]]:
        """Generate client interaction data"""
        interactions = []
        
        for i in range(count):
            interaction = {
                "interaction_id": f"int_{i:06d}",
                "client_id": f"client_{random.randint(1, 1000):04d}",
                "msp_id": f"msp_{random.randint(1, 1000):04d}",
                "interaction_type": random.choice([
                    "Support Ticket", "Phone Call", "Email", "Chat", "On-site Visit",
                    "Remote Session", "Emergency Response", "Scheduled Maintenance"
                ]),
                "timestamp": datetime.utcnow() - timedelta(hours=random.randint(1, 720)),
                "category": random.choice([
                    "Technical Issue", "Account Management", "Billing", "Security Incident",
                    "Service Request", "Complaint", "Praise", "General Inquiry"
                ]),
                "priority": random.choice(["Low", "Medium", "High", "Critical"]),
                "status": random.choice(["Open", "In Progress", "Resolved", "Closed"]),
                "resolution_time_hours": random.uniform(0.5, 48),
                "satisfaction_score": random.uniform(1, 5),
                "sentiment": random.choice(["Positive", "Neutral", "Negative"]),
                "tags": random.sample([
                    "Network", "Security", "Software", "Hardware", "Cloud", "Backup",
                    "Email", "Database", "Server", "Workstation", "Mobile", "VPN"
                ], k=random.randint(1, 3)),
                "description": f"Client reported {random.choice(['network connectivity issues', 'security concerns', 'software problems', 'performance issues'])} affecting {random.choice(['critical business operations', 'daily workflows', 'customer service', 'production systems'])}",
                "resolution": random.choice([
                    "Issue resolved by applying security patch",
                    "Network configuration updated successfully",
                    "Hardware replacement completed",
                    "Software update installed and tested",
                    "User training provided for prevention"
                ]),
                "follow_up_required": random.choice([True, False]),
                "escalated": random.choice([True, False]),
                "cost": random.uniform(50, 2000)
            }
            interactions.append(interaction)
        
        return interactions
    
    async def generate_market_intelligence_data(self) -> Dict[str, Any]:
        """Generate market intelligence data"""
        return {
            "pricing_trends": {
                "cloud_services": {
                    "average_price": random.uniform(50, 200),
                    "trend": random.choice(["increasing", "decreasing", "stable"]),
                    "change_percentage": random.uniform(-10, 15)
                },
                "security_services": {
                    "average_price": random.uniform(100, 500),
                    "trend": random.choice(["increasing", "decreasing", "stable"]),
                    "change_percentage": random.uniform(-5, 20)
                },
                "managed_services": {
                    "average_price": random.uniform(75, 300),
                    "trend": random.choice(["increasing", "decreasing", "stable"]),
                    "change_percentage": random.uniform(-8, 12)
                }
            },
            "competitive_landscape": {
                "market_leaders": [
                    {"name": "TechFlow Solutions", "market_share": 15.2, "growth_rate": 12.5},
                    {"name": "CyberGuard MSP", "market_share": 12.8, "growth_rate": 18.3},
                    {"name": "CloudFirst Partners", "market_share": 11.5, "growth_rate": 15.7}
                ],
                "emerging_players": [
                    {"name": "SecureNet Services", "growth_rate": 45.2},
                    {"name": "DataVault Technologies", "growth_rate": 38.7}
                ]
            },
            "industry_news": [
                {
                    "title": "Cybersecurity spending increases 20% in SMBs",
                    "impact": "positive",
                    "relevance_score": 0.9,
                    "published_date": datetime.utcnow() - timedelta(days=random.randint(1, 7))
                },
                {
                    "title": "Cloud adoption accelerates post-pandemic",
                    "impact": "positive",
                    "relevance_score": 0.8,
                    "published_date": datetime.utcnow() - timedelta(days=random.randint(1, 14))
                },
                {
                    "title": "New compliance regulations affect MSP industry",
                    "impact": "neutral",
                    "relevance_score": 0.7,
                    "published_date": datetime.utcnow() - timedelta(days=random.randint(1, 30))
                }
            ],
            "technology_adoption": {
                "ai_ml_integration": random.uniform(0.3, 0.8),
                "cloud_migration": random.uniform(0.6, 0.95),
                "automation_tools": random.uniform(0.4, 0.9),
                "security_enhancements": random.uniform(0.7, 0.98)
            },
            "regulatory_changes": [
                {
                    "regulation": "GDPR Updates",
                    "impact": "high",
                    "compliance_deadline": datetime.utcnow() + timedelta(days=90)
                },
                {
                    "regulation": "SOC2 Type II Requirements",
                    "impact": "medium",
                    "compliance_deadline": datetime.utcnow() + timedelta(days=180)
                }
            ]
        }
    
    async def generate_revenue_data(self, msp_id: str, months: int = 12) -> List[Dict[str, Any]]:
        """Generate revenue data for an MSP"""
        revenue_data = []
        base_revenue = random.randint(50000, 500000)
        
        for i in range(months):
            month = datetime.utcnow() - timedelta(days=30 * i)
            
            # Add some seasonality and growth
            growth_factor = 1 + (i * 0.02)  # 2% growth per month
            seasonality = 1 + 0.1 * np.sin(2 * np.pi * i / 12)  # Seasonal variation
            random_factor = random.uniform(0.9, 1.1)  # Random variation
            
            monthly_revenue = base_revenue * growth_factor * seasonality * random_factor
            
            revenue_data.append({
                "msp_id": msp_id,
                "month": month.strftime("%Y-%m"),
                "revenue": round(monthly_revenue, 2),
                "recurring_revenue": round(monthly_revenue * 0.8, 2),
                "project_revenue": round(monthly_revenue * 0.2, 2),
                "client_count": random.randint(10, 100),
                "average_revenue_per_client": round(monthly_revenue / random.randint(10, 100), 2),
                "churn_rate": random.uniform(0.02, 0.08),
                "growth_rate": random.uniform(0.05, 0.15)
            })
        
        return revenue_data
    
    async def generate_anomaly_data(self, count: int = 1000) -> List[Dict[str, Any]]:
        """Generate anomaly detection data"""
        anomalies = []
        
        for i in range(count):
            anomaly = {
                "anomaly_id": f"anom_{i:04d}",
                "system_id": f"system_{random.randint(1, 100):03d}",
                "anomaly_type": random.choice([
                    "High CPU Usage", "Memory Leak", "Network Spike", "Unusual Login Pattern",
                    "Data Exfiltration", "System Crash", "Performance Degradation", "Security Breach"
                ]),
                "severity": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
                "confidence": random.uniform(0.6, 0.99),
                "timestamp": datetime.utcnow() - timedelta(hours=random.randint(1, 168)),
                "metrics": {
                    "cpu_usage": random.uniform(0.1, 1.0),
                    "memory_usage": random.uniform(0.2, 0.95),
                    "network_latency": random.randint(10, 500),
                    "disk_usage": random.uniform(0.3, 0.9),
                    "response_time": random.uniform(0.1, 5.0)
                },
                "root_cause": random.choice([
                    "Resource exhaustion", "Malicious activity", "Configuration error",
                    "Hardware failure", "Software bug", "Network congestion"
                ]),
                "recommended_actions": random.sample([
                    "Restart service", "Scale resources", "Update configuration",
                    "Investigate security", "Replace hardware", "Apply patch"
                ], k=random.randint(1, 3)),
                "resolved": random.choice([True, False]),
                "resolution_time": random.uniform(0.5, 24) if random.choice([True, False]) else None
            }
            anomalies.append(anomaly)
        
        return anomalies
    
    async def generate_all_data(self) -> Dict[str, Any]:
        """Generate all synthetic data"""
        self.logger.info("Generating comprehensive synthetic data")
        
        data = {
            "msp_network": await self.generate_msp_network_data(1000),
            "threat_intelligence": await self.generate_threat_intelligence_data(10000),
            "collaboration_opportunities": await self.generate_collaboration_opportunities(500),
            "client_interactions": await self.generate_client_interaction_data(50000),
            "market_intelligence": await self.generate_market_intelligence_data(),
            "revenue_data": await self.generate_revenue_data("msp_0001", 12),
            "anomalies": await self.generate_anomaly_data(1000)
        }
        
        self.logger.info("Synthetic data generation completed", 
                        msp_count=len(data["msp_network"]),
                        threat_count=len(data["threat_intelligence"]),
                        opportunity_count=len(data["collaboration_opportunities"]),
                        interaction_count=len(data["client_interactions"]))
        
        return data
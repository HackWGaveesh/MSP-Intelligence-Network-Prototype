"""
Collaboration Matching Agent for MSP Intelligence Mesh Network
AI-powered partner discovery, skill complementarity analysis, and joint proposal generation
"""
import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from agents.base_agent import BaseAgent, AgentResponse
from config.settings import settings


class CollaborationAgent(BaseAgent):
    """
    Specialized agent for MSP collaboration and partnership matching
    Uses Sentence-BERT for semantic matching and skill complementarity analysis
    """
    
    def __init__(self):
        super().__init__("collaboration_agent", "collaboration_matching")
        
        # Collaboration state
        self.msp_profiles: Dict[str, Dict] = {}
        self.active_opportunities: Dict[str, Dict] = {}
        self.collaboration_history: List[Dict] = []
        self.skill_embeddings: Dict[str, np.ndarray] = {}
        
        # Model components
        self.sentence_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Skill categories and specializations
        self.skill_categories = {
            "cloud_services": ["AWS", "Azure", "GCP", "cloud_migration", "kubernetes"],
            "security": ["cybersecurity", "compliance", "penetration_testing", "SOC"],
            "networking": ["cisco", "fortinet", "palo_alto", "SD-WAN", "firewall"],
            "data_analytics": ["business_intelligence", "data_warehouse", "machine_learning"],
            "managed_services": ["help_desk", "monitoring", "backup", "disaster_recovery"],
            "software_development": ["web_development", "mobile_apps", "custom_software"],
            "consulting": ["IT_strategy", "digital_transformation", "process_optimization"],
            "compliance": ["SOC2", "ISO27001", "HIPAA", "GDPR", "PCI_DSS"]
        }
        
        # Opportunity types
        self.opportunity_types = {
            "enterprise_rfp": {
                "description": "Large enterprise RFP requiring multiple specializations",
                "typical_value": (500000, 5000000),
                "duration_months": (6, 24),
                "required_skills": ["cloud_services", "security", "compliance"]
            },
            "digital_transformation": {
                "description": "Complete digital transformation project",
                "typical_value": (200000, 2000000),
                "duration_months": (12, 36),
                "required_skills": ["consulting", "cloud_services", "data_analytics"]
            },
            "security_audit": {
                "description": "Comprehensive security audit and remediation",
                "typical_value": (100000, 800000),
                "duration_months": (3, 12),
                "required_skills": ["security", "compliance", "networking"]
            },
            "cloud_migration": {
                "description": "Large-scale cloud migration project",
                "typical_value": (300000, 1500000),
                "duration_months": (6, 18),
                "required_skills": ["cloud_services", "networking", "managed_services"]
            }
        }
        
        self.logger.info("Collaboration Agent initialized")
    
    async def load_model(self) -> bool:
        """Load Sentence-BERT model for semantic matching"""
        try:
            self.logger.info("Loading Sentence-BERT model for collaboration matching")
            
            # Load sentence transformer model
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.sentence_model = SentenceTransformer(model_name)
            
            self.logger.info("Sentence-BERT model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error("Failed to load Sentence-BERT model", error=str(e))
            return False
    
    async def process_request(self, request: Dict[str, Any]) -> AgentResponse:
        """Process collaboration matching requests"""
        start_time = datetime.utcnow()
        
        try:
            request_type = request.get("type", "unknown")
            
            if request_type == "find_partners":
                result = await self._find_partners(request.get("msp_id", ""), request.get("requirements", {}))
            elif request_type == "analyze_opportunity":
                result = await self._analyze_opportunity(request.get("opportunity_data", {}))
            elif request_type == "generate_proposal":
                result = await self._generate_proposal(request.get("partners", []), request.get("opportunity", {}))
            elif request_type == "calculate_revenue_sharing":
                result = await self._calculate_revenue_sharing(request.get("partners", []), request.get("opportunity", {}))
            elif request_type == "update_msp_profile":
                result = await self._update_msp_profile(request.get("msp_id", ""), request.get("profile_data", {}))
            elif request_type == "get_collaboration_opportunities":
                result = await self._get_collaboration_opportunities()
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
            
            self.logger.error("Error processing collaboration request", error=str(e))
            
            return AgentResponse(
                agent_id=self.agent_id,
                success=False,
                data={"error": str(e)},
                confidence=0.0,
                processing_time_ms=int(processing_time),
                timestamp=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def _find_partners(self, msp_id: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Find compatible partners based on requirements"""
        try:
            if not msp_id or not requirements:
                return {"error": "MSP ID and requirements are required"}
            
            # Get current MSP profile
            current_msp = self.msp_profiles.get(msp_id, {})
            if not current_msp:
                return {"error": f"MSP {msp_id} not found"}
            
            # Extract required skills from requirements
            required_skills = requirements.get("skills", [])
            opportunity_type = requirements.get("type", "general")
            location_preference = requirements.get("location", "any")
            size_preference = requirements.get("size", "any")
            
            # Find compatible MSPs
            compatible_partners = []
            
            for partner_id, partner_profile in self.msp_profiles.items():
                if partner_id == msp_id:
                    continue
                
                # Calculate compatibility score
                compatibility_score = self._calculate_compatibility(
                    current_msp, partner_profile, required_skills
                )
                
                # Apply filters
                if location_preference != "any" and partner_profile.get("location") != location_preference:
                    continue
                
                if size_preference != "any" and not self._matches_size_preference(partner_profile, size_preference):
                    continue
                
                if compatibility_score >= 0.6:  # Minimum compatibility threshold
                    compatible_partners.append({
                        "msp_id": partner_id,
                        "name": partner_profile.get("name", "Unknown"),
                        "compatibility_score": compatibility_score,
                        "complementary_skills": self._find_complementary_skills(current_msp, partner_profile),
                        "location": partner_profile.get("location", "Unknown"),
                        "size": partner_profile.get("size", "Unknown"),
                        "specializations": partner_profile.get("specializations", []),
                        "revenue": partner_profile.get("revenue", 0),
                        "client_count": partner_profile.get("client_count", 0)
                    })
            
            # Sort by compatibility score
            compatible_partners.sort(key=lambda x: x["compatibility_score"], reverse=True)
            
            # Limit to top 10 partners
            compatible_partners = compatible_partners[:10]
            
            result = {
                "msp_id": msp_id,
                "compatible_partners": compatible_partners,
                "total_found": len(compatible_partners),
                "search_criteria": requirements,
                "search_time": datetime.utcnow().isoformat()
            }
            
            self.logger.info("Partner search completed", 
                           msp_id=msp_id,
                           partners_found=len(compatible_partners))
            
            return result
            
        except Exception as e:
            self.logger.error("Error finding partners", error=str(e))
            return {"error": str(e)}
    
    async def _analyze_opportunity(self, opportunity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze collaboration opportunity and provide insights"""
        try:
            opportunity_type = opportunity_data.get("type", "general")
            description = opportunity_data.get("description", "")
            requirements = opportunity_data.get("requirements", [])
            
            # Get opportunity template
            template = self.opportunity_types.get(opportunity_type, {})
            
            # Analyze requirements using sentence embeddings
            requirement_embeddings = []
            if self.sentence_model and description:
                requirement_embeddings = self.sentence_model.encode([description])
            
            # Find best matching MSPs for this opportunity
            best_matches = []
            for msp_id, msp_profile in self.msp_profiles.items():
                match_score = self._calculate_opportunity_match(msp_profile, requirements, template)
                if match_score >= 0.5:
                    best_matches.append({
                        "msp_id": msp_id,
                        "name": msp_profile.get("name", "Unknown"),
                        "match_score": match_score,
                        "relevant_skills": self._get_relevant_skills(msp_profile, requirements)
                    })
            
            # Sort by match score
            best_matches.sort(key=lambda x: x["match_score"], reverse=True)
            
            # Generate opportunity analysis
            analysis = {
                "opportunity_type": opportunity_type,
                "estimated_value": self._estimate_opportunity_value(template, requirements),
                "estimated_duration": self._estimate_duration(template, requirements),
                "required_skills": requirements,
                "best_matches": best_matches[:5],  # Top 5 matches
                "complexity_score": self._calculate_complexity(requirements),
                "success_probability": self._calculate_success_probability(best_matches),
                "recommended_team_size": self._recommend_team_size(requirements),
                "analysis_time": datetime.utcnow().isoformat()
            }
            
            # Store opportunity
            opportunity_id = f"opp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self.active_opportunities[opportunity_id] = analysis
            
            return analysis
            
        except Exception as e:
            self.logger.error("Error analyzing opportunity", error=str(e))
            return {"error": str(e)}
    
    async def _generate_proposal(self, partners: List[Dict], opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Generate joint proposal for collaboration opportunity"""
        try:
            if not partners or not opportunity:
                return {"error": "Partners and opportunity data required"}
            
            # Analyze partner capabilities
            team_capabilities = self._analyze_team_capabilities(partners)
            
            # Generate proposal sections
            proposal = {
                "proposal_id": f"prop_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "opportunity": opportunity,
                "team_composition": {
                    "lead_partner": partners[0]["msp_id"],
                    "total_partners": len(partners),
                    "combined_capabilities": team_capabilities,
                    "team_strengths": self._identify_team_strengths(partners)
                },
                "proposed_solution": self._generate_solution_description(opportunity, team_capabilities),
                "project_timeline": self._generate_project_timeline(opportunity, partners),
                "resource_allocation": self._allocate_resources(partners, opportunity),
                "risk_assessment": self._assess_collaboration_risks(partners, opportunity),
                "success_metrics": self._define_success_metrics(opportunity),
                "generated_time": datetime.utcnow().isoformat()
            }
            
            # Calculate confidence score
            proposal["confidence_score"] = self._calculate_proposal_confidence(proposal)
            
            return proposal
            
        except Exception as e:
            self.logger.error("Error generating proposal", error=str(e))
            return {"error": str(e)}
    
    async def _calculate_revenue_sharing(self, partners: List[Dict], opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate fair revenue sharing model for collaboration"""
        try:
            if not partners or not opportunity:
                return {"error": "Partners and opportunity data required"}
            
            # Get opportunity value
            opportunity_value = opportunity.get("estimated_value", 1000000)
            
            # Calculate contribution scores for each partner
            contribution_scores = []
            for partner in partners:
                score = self._calculate_contribution_score(partner, opportunity)
                contribution_scores.append({
                    "msp_id": partner["msp_id"],
                    "name": partner.get("name", "Unknown"),
                    "contribution_score": score,
                    "contribution_percentage": 0  # Will be calculated below
                })
            
            # Normalize contribution scores to percentages
            total_score = sum(score["contribution_score"] for score in contribution_scores)
            for score in contribution_scores:
                score["contribution_percentage"] = (score["contribution_score"] / total_score) * 100
                score["revenue_share"] = (score["contribution_percentage"] / 100) * opportunity_value
            
            # Generate revenue sharing model
            revenue_model = {
                "total_opportunity_value": opportunity_value,
                "partners": contribution_scores,
                "sharing_model": "contribution_based",
                "payment_schedule": self._generate_payment_schedule(opportunity),
                "risk_sharing": self._calculate_risk_sharing(partners, opportunity),
                "calculated_time": datetime.utcnow().isoformat()
            }
            
            return revenue_model
            
        except Exception as e:
            self.logger.error("Error calculating revenue sharing", error=str(e))
            return {"error": str(e)}
    
    async def _update_msp_profile(self, msp_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update MSP profile information"""
        try:
            if not msp_id or not profile_data:
                return {"error": "MSP ID and profile data required"}
            
            # Update profile
            if msp_id not in self.msp_profiles:
                self.msp_profiles[msp_id] = {}
            
            self.msp_profiles[msp_id].update(profile_data)
            self.msp_profiles[msp_id]["last_updated"] = datetime.utcnow().isoformat()
            
            # Update skill embeddings if skills changed
            if "skills" in profile_data:
                await self._update_skill_embeddings(msp_id, profile_data["skills"])
            
            return {
                "msp_id": msp_id,
                "updated": True,
                "profile": self.msp_profiles[msp_id],
                "update_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Error updating MSP profile", error=str(e))
            return {"error": str(e)}
    
    async def _get_collaboration_opportunities(self) -> Dict[str, Any]:
        """Get all active collaboration opportunities"""
        return {
            "active_opportunities": self.active_opportunities,
            "total_opportunities": len(self.active_opportunities),
            "collaboration_history": self.collaboration_history[-10:],  # Last 10
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _calculate_compatibility(self, msp1: Dict, msp2: Dict, required_skills: List[str]) -> float:
        """Calculate compatibility score between two MSPs"""
        try:
            # Get skills from both MSPs
            skills1 = set(msp1.get("skills", []))
            skills2 = set(msp2.get("skills", []))
            required_skills_set = set(required_skills)
            
            # Calculate skill complementarity
            complementary_skills = skills1.symmetric_difference(skills2)
            overlapping_skills = skills1.intersection(skills2)
            
            # Calculate required skills coverage
            covered_skills = (skills1.union(skills2)).intersection(required_skills_set)
            coverage_score = len(covered_skills) / len(required_skills_set) if required_skills_set else 0
            
            # Calculate complementarity score
            complementarity_score = len(complementary_skills) / (len(skills1) + len(skills2)) if (skills1 or skills2) else 0
            
            # Calculate size compatibility
            size1 = msp1.get("size", "medium")
            size2 = msp2.get("size", "medium")
            size_compatibility = 1.0 if size1 == size2 else 0.7
            
            # Calculate location compatibility
            location1 = msp1.get("location", "unknown")
            location2 = msp2.get("location", "unknown")
            location_compatibility = 1.0 if location1 == location2 else 0.8
            
            # Weighted compatibility score
            compatibility = (
                coverage_score * 0.4 +
                complementarity_score * 0.3 +
                size_compatibility * 0.15 +
                location_compatibility * 0.15
            )
            
            return min(compatibility, 1.0)
            
        except Exception as e:
            self.logger.error("Error calculating compatibility", error=str(e))
            return 0.0
    
    def _find_complementary_skills(self, msp1: Dict, msp2: Dict) -> List[str]:
        """Find skills that complement each other between two MSPs"""
        skills1 = set(msp1.get("skills", []))
        skills2 = set(msp2.get("skills", []))
        
        # Find skills that one has but the other doesn't
        complementary = list(skills1.symmetric_difference(skills2))
        
        return complementary[:5]  # Return top 5 complementary skills
    
    def _matches_size_preference(self, profile: Dict, preference: str) -> bool:
        """Check if MSP size matches preference"""
        msp_size = profile.get("size", "medium")
        
        if preference == "any":
            return True
        elif preference == "small":
            return msp_size in ["small", "startup"]
        elif preference == "medium":
            return msp_size == "medium"
        elif preference == "large":
            return msp_size in ["large", "enterprise"]
        
        return False
    
    def _calculate_opportunity_match(self, msp_profile: Dict, requirements: List[str], template: Dict) -> float:
        """Calculate how well an MSP matches an opportunity"""
        msp_skills = set(msp_profile.get("skills", []))
        required_skills = set(requirements)
        
        # Calculate skill match percentage
        skill_match = len(msp_skills.intersection(required_skills)) / len(required_skills) if required_skills else 0
        
        # Factor in MSP size and experience
        size_factor = {"small": 0.7, "medium": 0.9, "large": 1.0, "enterprise": 1.0}.get(msp_profile.get("size", "medium"), 0.8)
        
        # Factor in client count (experience indicator)
        client_count = msp_profile.get("client_count", 0)
        experience_factor = min(1.0, client_count / 100)  # Normalize to 0-1
        
        # Calculate overall match score
        match_score = (skill_match * 0.6 + size_factor * 0.2 + experience_factor * 0.2)
        
        return min(match_score, 1.0)
    
    def _get_relevant_skills(self, msp_profile: Dict, requirements: List[str]) -> List[str]:
        """Get skills from MSP profile that are relevant to requirements"""
        msp_skills = set(msp_profile.get("skills", []))
        required_skills = set(requirements)
        
        return list(msp_skills.intersection(required_skills))
    
    def _estimate_opportunity_value(self, template: Dict, requirements: List[str]) -> Dict[str, Any]:
        """Estimate opportunity value based on template and requirements"""
        base_value_range = template.get("typical_value", (100000, 1000000))
        
        # Adjust based on requirements complexity
        complexity_multiplier = 1.0 + (len(requirements) * 0.1)
        
        min_value = int(base_value_range[0] * complexity_multiplier)
        max_value = int(base_value_range[1] * complexity_multiplier)
        
        return {
            "min_value": min_value,
            "max_value": max_value,
            "estimated_value": (min_value + max_value) // 2,
            "currency": "USD"
        }
    
    def _estimate_duration(self, template: Dict, requirements: List[str]) -> Dict[str, Any]:
        """Estimate project duration based on template and requirements"""
        base_duration_range = template.get("duration_months", (3, 12))
        
        # Adjust based on requirements complexity
        complexity_multiplier = 1.0 + (len(requirements) * 0.05)
        
        min_months = int(base_duration_range[0] * complexity_multiplier)
        max_months = int(base_duration_range[1] * complexity_multiplier)
        
        return {
            "min_months": min_months,
            "max_months": max_months,
            "estimated_months": (min_months + max_months) // 2
        }
    
    def _calculate_complexity(self, requirements: List[str]) -> float:
        """Calculate opportunity complexity score"""
        # Simple complexity calculation based on number of requirements
        base_complexity = min(1.0, len(requirements) / 10)
        
        # Add complexity for certain skill categories
        high_complexity_skills = ["security", "compliance", "data_analytics"]
        complexity_boost = sum(0.1 for req in requirements if any(skill in req.lower() for skill in high_complexity_skills))
        
        return min(base_complexity + complexity_boost, 1.0)
    
    def _calculate_success_probability(self, best_matches: List[Dict]) -> float:
        """Calculate success probability based on available partners"""
        if not best_matches:
            return 0.0
        
        # Calculate average match score
        avg_match_score = sum(match["match_score"] for match in best_matches) / len(best_matches)
        
        # Factor in number of available partners
        partner_availability_factor = min(1.0, len(best_matches) / 5)
        
        return avg_match_score * partner_availability_factor
    
    def _recommend_team_size(self, requirements: List[str]) -> Dict[str, Any]:
        """Recommend optimal team size for opportunity"""
        base_team_size = 2  # Minimum team size
        
        # Adjust based on requirements complexity
        complexity_team_size = base_team_size + len(requirements) // 2
        
        return {
            "recommended_size": min(complexity_team_size, 8),  # Cap at 8
            "min_size": base_team_size,
            "max_size": 8,
            "rationale": f"Based on {len(requirements)} required skills"
        }
    
    def _analyze_team_capabilities(self, partners: List[Dict]) -> Dict[str, Any]:
        """Analyze combined capabilities of partner team"""
        all_skills = set()
        all_specializations = set()
        total_revenue = 0
        total_clients = 0
        
        for partner in partners:
            all_skills.update(partner.get("skills", []))
            all_specializations.update(partner.get("specializations", []))
            total_revenue += partner.get("revenue", 0)
            total_clients += partner.get("client_count", 0)
        
        return {
            "combined_skills": list(all_skills),
            "combined_specializations": list(all_specializations),
            "total_revenue": total_revenue,
            "total_clients": total_clients,
            "skill_coverage": len(all_skills),
            "team_diversity_score": len(all_specializations) / len(partners) if partners else 0
        }
    
    def _identify_team_strengths(self, partners: List[Dict]) -> List[str]:
        """Identify key strengths of the partner team"""
        strengths = []
        
        # Analyze skill distribution
        skill_counts = {}
        for partner in partners:
            for skill in partner.get("skills", []):
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Find most common skills (team strengths)
        common_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        strengths.extend([skill for skill, count in common_skills[:3]])
        
        # Add size-based strengths
        sizes = [partner.get("size", "medium") for partner in partners]
        if "enterprise" in sizes:
            strengths.append("Enterprise experience")
        if len(partners) >= 3:
            strengths.append("Multi-partner collaboration")
        
        return strengths[:5]  # Top 5 strengths
    
    def _generate_solution_description(self, opportunity: Dict, team_capabilities: Dict) -> str:
        """Generate solution description for proposal"""
        opportunity_type = opportunity.get("type", "general")
        description = opportunity.get("description", "")
        
        # Generate solution based on opportunity type and team capabilities
        solution_templates = {
            "enterprise_rfp": f"Our collaborative team brings together {len(team_capabilities['combined_skills'])} specialized skills to deliver a comprehensive solution for your enterprise requirements. We combine our expertise in {', '.join(team_capabilities['combined_specializations'][:3])} to ensure successful project delivery.",
            "digital_transformation": f"Leveraging our combined experience with {team_capabilities['total_clients']} clients, we provide a holistic digital transformation approach that addresses your current challenges while positioning you for future growth.",
            "security_audit": f"Our security-focused team combines deep expertise in cybersecurity, compliance, and risk management to deliver a thorough security assessment and remediation plan.",
            "cloud_migration": f"With extensive cloud migration experience across {team_capabilities['total_clients']} successful projects, our team ensures a seamless transition to the cloud with minimal business disruption."
        }
        
        return solution_templates.get(opportunity_type, f"Our collaborative team is uniquely positioned to deliver this {opportunity_type} project with our combined expertise and proven track record.")
    
    def _generate_project_timeline(self, opportunity: Dict, partners: List[Dict]) -> Dict[str, Any]:
        """Generate project timeline for proposal"""
        duration = self._estimate_duration({}, opportunity.get("requirements", []))
        estimated_months = duration["estimated_months"]
        
        # Generate timeline phases
        phases = [
            {"phase": "Planning & Setup", "duration_weeks": 2, "description": "Project kickoff, team alignment, and detailed planning"},
            {"phase": "Analysis & Design", "duration_weeks": 4, "description": "Requirements analysis, solution design, and architecture planning"},
            {"phase": "Implementation", "duration_weeks": (estimated_months - 2) * 4, "description": "Core implementation and development work"},
            {"phase": "Testing & Deployment", "duration_weeks": 3, "description": "Testing, quality assurance, and production deployment"},
            {"phase": "Support & Handover", "duration_weeks": 2, "description": "Knowledge transfer, documentation, and ongoing support setup"}
        ]
        
        return {
            "total_duration_months": estimated_months,
            "phases": phases,
            "key_milestones": [
                f"Week 2: Project kickoff and team alignment",
                f"Week 6: Solution design approval",
                f"Week {estimated_months * 4 - 6}: Implementation complete",
                f"Week {estimated_months * 4 - 2}: Testing and deployment",
                f"Week {estimated_months * 4}: Project completion and handover"
            ]
        }
    
    def _allocate_resources(self, partners: List[Dict], opportunity: Dict) -> Dict[str, Any]:
        """Allocate resources across partners for the project"""
        total_team_size = self._recommend_team_size(opportunity.get("requirements", []))["recommended_size"]
        
        # Distribute team members across partners
        resource_allocation = []
        remaining_size = total_team_size
        
        for i, partner in enumerate(partners):
            if i == len(partners) - 1:  # Last partner gets remaining resources
                allocated_size = remaining_size
            else:
                # Allocate based on partner size and capabilities
                partner_size = partner.get("size", "medium")
                size_multiplier = {"small": 0.5, "medium": 1.0, "large": 1.5, "enterprise": 2.0}.get(partner_size, 1.0)
                allocated_size = max(1, int(total_team_size * size_multiplier / len(partners)))
                allocated_size = min(allocated_size, remaining_size)
            
            resource_allocation.append({
                "partner_id": partner["msp_id"],
                "partner_name": partner.get("name", "Unknown"),
                "allocated_team_members": allocated_size,
                "primary_responsibilities": self._assign_responsibilities(partner, opportunity)
            })
            
            remaining_size -= allocated_size
        
        return {
            "total_team_size": total_team_size,
            "resource_allocation": resource_allocation,
            "allocation_rationale": "Based on partner size, capabilities, and project requirements"
        }
    
    def _assign_responsibilities(self, partner: Dict, opportunity: Dict) -> List[str]:
        """Assign primary responsibilities to a partner based on their skills"""
        partner_skills = set(partner.get("skills", []))
        requirements = set(opportunity.get("requirements", []))
        
        # Find matching skills and assign responsibilities
        matching_skills = partner_skills.intersection(requirements)
        
        responsibilities = []
        for skill in matching_skills:
            if "security" in skill.lower():
                responsibilities.append("Security implementation and compliance")
            elif "cloud" in skill.lower():
                responsibilities.append("Cloud architecture and migration")
            elif "data" in skill.lower():
                responsibilities.append("Data analysis and reporting")
            elif "network" in skill.lower():
                responsibilities.append("Network infrastructure and security")
            elif "development" in skill.lower():
                responsibilities.append("Custom development and integration")
        
        # Add default responsibilities if none assigned
        if not responsibilities:
            responsibilities.append("General project support and coordination")
        
        return responsibilities[:3]  # Limit to 3 primary responsibilities
    
    def _assess_collaboration_risks(self, partners: List[Dict], opportunity: Dict) -> Dict[str, Any]:
        """Assess risks associated with the collaboration"""
        risks = []
        
        # Size mismatch risk
        sizes = [partner.get("size", "medium") for partner in partners]
        if len(set(sizes)) > 1:
            risks.append({
                "risk": "Partner size mismatch",
                "severity": "medium",
                "mitigation": "Establish clear communication protocols and project management structure"
            })
        
        # Geographic distribution risk
        locations = [partner.get("location", "unknown") for partner in partners]
        if len(set(locations)) > 1:
            risks.append({
                "risk": "Geographic distribution",
                "severity": "low",
                "mitigation": "Use collaborative tools and establish regular communication schedules"
            })
        
        # Skill overlap risk
        all_skills = []
        for partner in partners:
            all_skills.extend(partner.get("skills", []))
        
        skill_overlap = len(all_skills) - len(set(all_skills))
        if skill_overlap > len(partners):
            risks.append({
                "risk": "Excessive skill overlap",
                "severity": "low",
                "mitigation": "Clearly define roles and responsibilities to avoid duplication"
            })
        
        # Add opportunity-specific risks
        opportunity_type = opportunity.get("type", "general")
        if opportunity_type == "enterprise_rfp":
            risks.append({
                "risk": "Complex enterprise requirements",
                "severity": "high",
                "mitigation": "Conduct thorough requirements analysis and establish change management process"
            })
        
        return {
            "identified_risks": risks,
            "overall_risk_level": "medium" if len(risks) > 2 else "low",
            "risk_mitigation_strategy": "Regular communication, clear roles, and proactive issue resolution"
        }
    
    def _define_success_metrics(self, opportunity: Dict) -> List[Dict[str, Any]]:
        """Define success metrics for the collaboration"""
        opportunity_type = opportunity.get("type", "general")
        
        base_metrics = [
            {"metric": "Project completion on time", "target": "100%", "measurement": "Timeline adherence"},
            {"metric": "Client satisfaction", "target": "90%+", "measurement": "Post-project survey"},
            {"metric": "Budget adherence", "target": "95%+", "measurement": "Actual vs. planned costs"}
        ]
        
        # Add type-specific metrics
        if opportunity_type == "enterprise_rfp":
            base_metrics.extend([
                {"metric": "Requirements coverage", "target": "100%", "measurement": "Deliverable checklist"},
                {"metric": "Quality standards", "target": "95%+", "measurement": "Code review and testing"}
            ])
        elif opportunity_type == "digital_transformation":
            base_metrics.extend([
                {"metric": "Process improvement", "target": "30%+", "measurement": "Efficiency metrics"},
                {"metric": "User adoption", "target": "85%+", "measurement": "Usage analytics"}
            ])
        
        return base_metrics
    
    def _calculate_proposal_confidence(self, proposal: Dict[str, Any]) -> float:
        """Calculate confidence score for the generated proposal"""
        confidence_factors = []
        
        # Team composition factor
        team_size = proposal["team_composition"]["total_partners"]
        team_confidence = min(1.0, team_size / 3)  # Optimal at 3+ partners
        confidence_factors.append(team_confidence)
        
        # Capability coverage factor
        capabilities = proposal["team_composition"]["combined_capabilities"]
        capability_confidence = min(1.0, capabilities["skill_coverage"] / 10)  # Optimal at 10+ skills
        confidence_factors.append(capability_confidence)
        
        # Risk assessment factor
        risks = proposal["risk_assessment"]["identified_risks"]
        risk_confidence = max(0.0, 1.0 - (len(risks) * 0.2))  # Reduce confidence for more risks
        confidence_factors.append(risk_confidence)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        
        return min(overall_confidence, 1.0)
    
    def _calculate_contribution_score(self, partner: Dict, opportunity: Dict) -> float:
        """Calculate contribution score for revenue sharing"""
        # Base score from partner size
        size_scores = {"small": 0.3, "medium": 0.6, "large": 0.8, "enterprise": 1.0}
        base_score = size_scores.get(partner.get("size", "medium"), 0.5)
        
        # Skill relevance score
        partner_skills = set(partner.get("skills", []))
        required_skills = set(opportunity.get("requirements", []))
        skill_relevance = len(partner_skills.intersection(required_skills)) / len(required_skills) if required_skills else 0
        
        # Experience score (based on client count)
        client_count = partner.get("client_count", 0)
        experience_score = min(1.0, client_count / 200)  # Normalize to 0-1
        
        # Calculate weighted contribution score
        contribution_score = (base_score * 0.4 + skill_relevance * 0.4 + experience_score * 0.2)
        
        return min(contribution_score, 1.0)
    
    def _generate_payment_schedule(self, opportunity: Dict) -> List[Dict[str, Any]]:
        """Generate payment schedule for the project"""
        duration = self._estimate_duration({}, opportunity.get("requirements", []))
        total_months = duration["estimated_months"]
        
        # Generate milestone-based payment schedule
        payment_schedule = [
            {"milestone": "Project Kickoff", "percentage": 20, "timing": "Week 1"},
            {"milestone": "Design Approval", "percentage": 25, "timing": "Week 6"},
            {"milestone": "Implementation Complete", "percentage": 35, "timing": f"Week {total_months * 4 - 6}"},
            {"milestone": "Testing & Deployment", "percentage": 15, "timing": f"Week {total_months * 4 - 2}"},
            {"milestone": "Project Completion", "percentage": 5, "timing": f"Week {total_months * 4}"}
        ]
        
        return payment_schedule
    
    def _calculate_risk_sharing(self, partners: List[Dict], opportunity: Dict) -> Dict[str, Any]:
        """Calculate risk sharing model for the collaboration"""
        # Simple risk sharing based on partner size and contribution
        total_contribution = sum(self._calculate_contribution_score(partner, opportunity) for partner in partners)
        
        risk_sharing = []
        for partner in partners:
            contribution = self._calculate_contribution_score(partner, opportunity)
            risk_percentage = (contribution / total_contribution) * 100 if total_contribution > 0 else 0
            
            risk_sharing.append({
                "partner_id": partner["msp_id"],
                "partner_name": partner.get("name", "Unknown"),
                "risk_percentage": risk_percentage,
                "liability_cap": f"${int(opportunity.get('estimated_value', 1000000) * risk_percentage / 100):,}"
            })
        
        return {
            "risk_sharing_model": "contribution_based",
            "partners": risk_sharing,
            "total_liability": f"${opportunity.get('estimated_value', 1000000):,}"
        }
    
    async def _update_skill_embeddings(self, msp_id: str, skills: List[str]):
        """Update skill embeddings for semantic matching"""
        try:
            if self.sentence_model and skills:
                # Create skill description
                skill_text = " ".join(skills)
                embedding = self.sentence_model.encode([skill_text])
                self.skill_embeddings[msp_id] = embedding[0]
                
                self.logger.info("Updated skill embeddings", msp_id=msp_id, skills_count=len(skills))
        except Exception as e:
            self.logger.error("Error updating skill embeddings", error=str(e))
    
    async def simulate_collaboration_opportunity(self) -> Dict[str, Any]:
        """Simulate a collaboration opportunity for demo purposes"""
        # Generate random opportunity
        opportunity_types = list(self.opportunity_types.keys())
        opportunity_type = random.choice(opportunity_types)
        
        # Create simulated opportunity data
        opportunity_data = {
            "type": opportunity_type,
            "description": f"Simulated {opportunity_type.replace('_', ' ').title()} opportunity requiring specialized skills and multi-partner collaboration",
            "requirements": random.sample(list(self.skill_categories.keys()), random.randint(2, 4)),
            "client_industry": random.choice(["Healthcare", "Finance", "Manufacturing", "Retail", "Technology"]),
            "project_scope": "Large-scale implementation requiring multiple specializations"
        }
        
        # Analyze the opportunity
        analysis = await self._analyze_opportunity(opportunity_data)
        
        # Find top partners
        if analysis.get("best_matches"):
            top_partners = analysis["best_matches"][:3]  # Top 3 partners
            
            # Generate joint proposal
            proposal = await self._generate_proposal(top_partners, analysis)
            
            # Calculate revenue sharing
            revenue_sharing = await self._calculate_revenue_sharing(top_partners, analysis)
            
            return {
                "opportunity": analysis,
                "proposal": proposal,
                "revenue_sharing": revenue_sharing,
                "simulation_time": datetime.utcnow().isoformat()
            }
        
        return {"error": "No suitable partners found for simulation"}

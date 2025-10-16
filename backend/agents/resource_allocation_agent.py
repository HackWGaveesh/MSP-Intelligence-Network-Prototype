"""
Resource Allocation Agent for MSP Intelligence Mesh Network
Optimizes technician assignments, project scheduling, and capacity planning
"""
import asyncio
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import structlog
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent, AgentResponse, AgentMetrics


logger = structlog.get_logger()


class ResourceAllocationAgent(BaseAgent):
    """Resource Allocation Agent for optimizing technician assignments and scheduling"""
    
    def __init__(self):
        super().__init__()
        self.agent_id = "resource_allocation_agent"
        self.agent_type = "resource_allocation"
        self.model_loaded = False
        self.scheduling_model = None
        self.capacity_model = None
        self.scaler = None
        self.label_encoders = {}
        self.resource_data = {}
        self.optimization_cache = {}
        
        self.logger = logger.bind(agent=self.agent_id)
        self.logger.info("Resource Allocation Agent initialized")
    
    async def initialize(self):
        """Initialize the agent and load models"""
        try:
            self.logger.info("Initializing Resource Allocation Agent")
            
            # Initialize models
            self.scheduling_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.capacity_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
            self.scaler = StandardScaler()
            
            # Load resource data
            await self._load_resource_data()
            
            # Train models
            await self._train_models()
            
            self.model_loaded = True
            self.logger.info("Resource Allocation Agent initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize Resource Allocation Agent", error=str(e))
            raise
    
    async def _load_resource_data(self):
        """Load resource allocation data"""
        # Generate synthetic resource data
        np.random.seed(42)
        n_projects = 500
        n_technicians = 50
        
        # Generate technician profiles
        technicians = []
        for i in range(n_technicians):
            technician = {
                'technician_id': f'tech_{i:03d}',
                'name': f'Technician {i+1}',
                'skills': random.sample(['networking', 'security', 'cloud', 'database', 'virtualization', 'backup', 'monitoring'], random.randint(2, 5)),
                'experience_years': random.randint(1, 15),
                'certifications': random.sample(['CCNA', 'CCNP', 'AWS', 'Azure', 'VMware', 'CISSP', 'ITIL'], random.randint(1, 4)),
                'availability_hours': random.randint(30, 50),
                'hourly_rate': random.uniform(25, 75),
                'location': random.choice(['North', 'South', 'East', 'West', 'Central']),
                'specialization': random.choice(['infrastructure', 'security', 'cloud', 'database', 'networking']),
                'current_workload': random.uniform(0, 1),
                'performance_rating': random.uniform(3.0, 5.0)
            }
            technicians.append(technician)
        
        # Generate project data
        projects = []
        for i in range(n_projects):
            project = {
                'project_id': f'proj_{i:04d}',
                'name': f'Project {i+1}',
                'client_id': f'client_{random.randint(1, 100)}',
                'priority': random.choice(['low', 'medium', 'high', 'critical']),
                'required_skills': random.sample(['networking', 'security', 'cloud', 'database', 'virtualization', 'backup', 'monitoring'], random.randint(1, 4)),
                'estimated_hours': random.randint(8, 80),
                'deadline': datetime.now() + timedelta(days=random.randint(1, 90)),
                'location': random.choice(['North', 'South', 'East', 'West', 'Central']),
                'complexity': random.choice(['low', 'medium', 'high']),
                'budget': random.uniform(1000, 50000),
                'status': random.choice(['pending', 'in_progress', 'completed', 'cancelled']),
                'created_date': datetime.now() - timedelta(days=random.randint(1, 30))
            }
            projects.append(project)
        
        self.resource_data = {
            'technicians': technicians,
            'projects': projects
        }
        
        self.logger.info(f"Loaded {len(technicians)} technicians and {len(projects)} projects")
    
    async def _train_models(self):
        """Train resource allocation models"""
        try:
            # Prepare training data for scheduling model
            scheduling_data = self._prepare_scheduling_data()
            
            # Train scheduling model
            X_scheduling = scheduling_data.drop(['completion_time', 'success_rate'], axis=1)
            y_completion = scheduling_data['completion_time']
            y_success = scheduling_data['success_rate']
            
            self.scheduling_model.fit(X_scheduling, y_completion)
            
            # Train capacity model
            capacity_data = self._prepare_capacity_data()
            X_capacity = capacity_data.drop(['utilization', 'efficiency'], axis=1)
            y_utilization = capacity_data['utilization']
            
            self.capacity_model.fit(X_capacity, y_utilization)
            
            self.logger.info("Resource allocation models trained successfully")
            
        except Exception as e:
            self.logger.error("Failed to train models", error=str(e))
            raise
    
    def _prepare_scheduling_data(self) -> pd.DataFrame:
        """Prepare training data for scheduling model"""
        data = []
        
        for project in self.resource_data['projects']:
            for technician in self.resource_data['technicians']:
                # Calculate skill match
                skill_match = len(set(project['required_skills']).intersection(set(technician['skills']))) / len(project['required_skills'])
                
                # Calculate location match
                location_match = 1.0 if project['location'] == technician['location'] else 0.5
                
                # Calculate experience factor
                experience_factor = min(technician['experience_years'] / 10, 1.0)
                
                # Calculate workload factor
                workload_factor = 1.0 - technician['current_workload']
                
                # Simulate completion time and success rate
                base_completion = project['estimated_hours'] / technician['availability_hours']
                completion_time = base_completion * (1 + random.uniform(-0.2, 0.2))
                
                success_rate = (skill_match * 0.4 + experience_factor * 0.3 + 
                              workload_factor * 0.2 + location_match * 0.1)
                success_rate = max(0.1, min(1.0, success_rate + random.uniform(-0.1, 0.1)))
                
                data.append({
                    'project_complexity': 1 if project['complexity'] == 'low' else 2 if project['complexity'] == 'medium' else 3,
                    'project_priority': 1 if project['priority'] == 'low' else 2 if project['priority'] == 'medium' else 3 if project['priority'] == 'high' else 4,
                    'estimated_hours': project['estimated_hours'],
                    'skill_match': skill_match,
                    'experience_years': technician['experience_years'],
                    'availability_hours': technician['availability_hours'],
                    'current_workload': technician['current_workload'],
                    'performance_rating': technician['performance_rating'],
                    'location_match': location_match,
                    'hourly_rate': technician['hourly_rate'],
                    'completion_time': completion_time,
                    'success_rate': success_rate
                })
        
        return pd.DataFrame(data)
    
    def _prepare_capacity_data(self) -> pd.DataFrame:
        """Prepare training data for capacity model"""
        data = []
        
        for technician in self.resource_data['technicians']:
            # Simulate utilization and efficiency
            base_utilization = technician['current_workload']
            utilization = base_utilization + random.uniform(-0.1, 0.1)
            utilization = max(0, min(1, utilization))
            
            efficiency = (technician['performance_rating'] / 5.0) * (1 - utilization * 0.3)
            efficiency = max(0.1, min(1.0, efficiency))
            
            data.append({
                'experience_years': technician['experience_years'],
                'availability_hours': technician['availability_hours'],
                'performance_rating': technician['performance_rating'],
                'current_workload': technician['current_workload'],
                'hourly_rate': technician['hourly_rate'],
                'utilization': utilization,
                'efficiency': efficiency
            })
        
        return pd.DataFrame(data)
    
    async def process_request(self, request: Dict[str, Any]) -> AgentResponse:
        """Process resource allocation requests"""
        try:
            request_type = request.get("type", "")
            request_data = request.get("data", {})
            
            start_time = datetime.utcnow()
            
            if request_type == "optimize_assignments":
                result = await self._optimize_assignments(request_data)
            elif request_type == "schedule_projects":
                result = await self._schedule_projects(request_data)
            elif request_type == "analyze_capacity":
                result = await self._analyze_capacity(request_data)
            elif request_type == "predict_resource_needs":
                result = await self._predict_resource_needs(request_data)
            elif request_type == "optimize_workload":
                result = await self._optimize_workload(request_data)
            elif request_type == "generate_schedule":
                result = await self._generate_schedule(request_data)
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
            self.logger.error("Error processing resource allocation request", error=str(e))
            return AgentResponse(
                success=False,
                error=str(e),
                agent_id=self.agent_id,
                timestamp=datetime.utcnow().isoformat()
            )
    
    async def _optimize_assignments(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize technician assignments for projects"""
        project_ids = data.get("project_ids", [])
        optimization_criteria = data.get("criteria", "balanced")
        
        # Get projects to optimize
        if not project_ids:
            # Get pending projects
            projects = [p for p in self.resource_data['projects'] if p['status'] == 'pending']
        else:
            projects = [p for p in self.resource_data['projects'] if p['project_id'] in project_ids]
        
        if not projects:
            return {"error": "No projects found for optimization"}
        
        # Generate optimized assignments
        assignments = []
        for project in projects:
            best_assignments = self._find_best_assignments(project, optimization_criteria)
            assignments.append({
                "project_id": project['project_id'],
                "project_name": project['name'],
                "recommended_assignments": best_assignments,
                "optimization_score": self._calculate_optimization_score(best_assignments, project)
            })
        
        # Calculate overall optimization metrics
        total_score = np.mean([a["optimization_score"] for a in assignments])
        total_cost = sum([sum([t["cost"] for t in a["recommended_assignments"]]) for a in assignments])
        total_time = sum([max([t["estimated_completion_days"] for t in a["recommended_assignments"]]) for a in assignments])
        
        return {
            "optimization_criteria": optimization_criteria,
            "total_projects": len(projects),
            "assignments": assignments,
            "overall_optimization_score": round(total_score, 3),
            "total_estimated_cost": round(total_cost, 2),
            "total_estimated_time_days": round(total_time, 1),
            "optimization_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _schedule_projects(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule projects with optimal resource allocation"""
        time_horizon = data.get("time_horizon", "30_days")
        scheduling_constraints = data.get("constraints", {})
        
        # Get projects to schedule
        projects = [p for p in self.resource_data['projects'] if p['status'] in ['pending', 'in_progress']]
        
        # Generate schedule
        schedule = self._generate_project_schedule(projects, time_horizon, scheduling_constraints)
        
        # Calculate schedule metrics
        schedule_metrics = self._calculate_schedule_metrics(schedule)
        
        # Identify conflicts and bottlenecks
        conflicts = self._identify_schedule_conflicts(schedule)
        bottlenecks = self._identify_bottlenecks(schedule)
        
        return {
            "time_horizon": time_horizon,
            "total_projects_scheduled": len(schedule),
            "schedule": schedule,
            "schedule_metrics": schedule_metrics,
            "conflicts": conflicts,
            "bottlenecks": bottlenecks,
            "recommendations": self._generate_scheduling_recommendations(schedule, conflicts, bottlenecks),
            "scheduling_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _analyze_capacity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource capacity and utilization"""
        analysis_period = data.get("period", "30_days")
        resource_type = data.get("resource_type", "all")
        
        # Analyze technician capacity
        capacity_analysis = self._analyze_technician_capacity(analysis_period)
        
        # Analyze skill capacity
        skill_analysis = self._analyze_skill_capacity()
        
        # Analyze location capacity
        location_analysis = self._analyze_location_capacity()
        
        # Calculate capacity metrics
        capacity_metrics = self._calculate_capacity_metrics(capacity_analysis, skill_analysis, location_analysis)
        
        # Identify capacity constraints
        constraints = self._identify_capacity_constraints(capacity_analysis, skill_analysis, location_analysis)
        
        return {
            "analysis_period": analysis_period,
            "resource_type": resource_type,
            "technician_capacity": capacity_analysis,
            "skill_capacity": skill_analysis,
            "location_capacity": location_analysis,
            "capacity_metrics": capacity_metrics,
            "constraints": constraints,
            "recommendations": self._generate_capacity_recommendations(constraints),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _predict_resource_needs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future resource needs"""
        prediction_horizon = data.get("horizon", "90_days")
        prediction_type = data.get("type", "workload")
        
        # Predict workload
        workload_prediction = self._predict_workload(prediction_horizon)
        
        # Predict skill needs
        skill_prediction = self._predict_skill_needs(prediction_horizon)
        
        # Predict capacity requirements
        capacity_prediction = self._predict_capacity_requirements(prediction_horizon)
        
        # Generate resource recommendations
        recommendations = self._generate_resource_recommendations(workload_prediction, skill_prediction, capacity_prediction)
        
        return {
            "prediction_horizon": prediction_horizon,
            "prediction_type": prediction_type,
            "workload_prediction": workload_prediction,
            "skill_prediction": skill_prediction,
            "capacity_prediction": capacity_prediction,
            "recommendations": recommendations,
            "confidence": self._calculate_prediction_confidence(workload_prediction, skill_prediction),
            "prediction_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _optimize_workload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize workload distribution across technicians"""
        optimization_goal = data.get("goal", "balanced")
        constraints = data.get("constraints", {})
        
        # Analyze current workload distribution
        current_workload = self._analyze_current_workload()
        
        # Generate optimized workload distribution
        optimized_workload = self._generate_optimized_workload(current_workload, optimization_goal, constraints)
        
        # Calculate workload metrics
        workload_metrics = self._calculate_workload_metrics(optimized_workload)
        
        # Identify workload adjustments needed
        adjustments = self._identify_workload_adjustments(current_workload, optimized_workload)
        
        return {
            "optimization_goal": optimization_goal,
            "current_workload": current_workload,
            "optimized_workload": optimized_workload,
            "workload_metrics": workload_metrics,
            "adjustments": adjustments,
            "expected_improvements": self._calculate_workload_improvements(current_workload, optimized_workload),
            "optimization_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _generate_schedule(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed schedule for technicians and projects"""
        schedule_type = data.get("type", "weekly")
        technician_ids = data.get("technician_ids", [])
        
        # Get technicians for scheduling
        if technician_ids:
            technicians = [t for t in self.resource_data['technicians'] if t['technician_id'] in technician_ids]
        else:
            technicians = self.resource_data['technicians']
        
        # Generate schedule
        schedule = self._generate_detailed_schedule(technicians, schedule_type)
        
        # Calculate schedule efficiency
        efficiency_metrics = self._calculate_schedule_efficiency(schedule)
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_schedule_optimization_opportunities(schedule)
        
        return {
            "schedule_type": schedule_type,
            "technicians_scheduled": len(technicians),
            "schedule": schedule,
            "efficiency_metrics": efficiency_metrics,
            "optimization_opportunities": optimization_opportunities,
            "schedule_timestamp": datetime.utcnow().isoformat()
        }
    
    def _find_best_assignments(self, project: Dict[str, Any], criteria: str) -> List[Dict[str, Any]]:
        """Find best technician assignments for a project"""
        assignments = []
        
        for technician in self.resource_data['technicians']:
            # Calculate assignment score
            score = self._calculate_assignment_score(project, technician, criteria)
            
            if score > 0.5:  # Minimum threshold
                # Calculate project details
                skill_match = len(set(project['required_skills']).intersection(set(technician['skills']))) / len(project['required_skills'])
                estimated_hours = project['estimated_hours'] / technician['availability_hours']
                cost = estimated_hours * technician['hourly_rate']
                
                assignments.append({
                    "technician_id": technician['technician_id'],
                    "technician_name": technician['name'],
                    "skill_match": round(skill_match, 3),
                    "assignment_score": round(score, 3),
                    "estimated_completion_days": round(estimated_hours, 1),
                    "cost": round(cost, 2),
                    "availability": technician['availability_hours'],
                    "current_workload": technician['current_workload'],
                    "specialization": technician['specialization']
                })
        
        # Sort by assignment score
        assignments.sort(key=lambda x: x["assignment_score"], reverse=True)
        
        return assignments[:3]  # Return top 3 assignments
    
    def _calculate_assignment_score(self, project: Dict[str, Any], technician: Dict[str, Any], criteria: str) -> float:
        """Calculate assignment score for a technician-project pair"""
        # Skill match
        skill_match = len(set(project['required_skills']).intersection(set(technician['skills']))) / len(project['required_skills'])
        
        # Experience factor
        experience_factor = min(technician['experience_years'] / 10, 1.0)
        
        # Availability factor
        availability_factor = 1.0 - technician['current_workload']
        
        # Location match
        location_match = 1.0 if project['location'] == technician['location'] else 0.5
        
        # Performance factor
        performance_factor = technician['performance_rating'] / 5.0
        
        # Calculate base score
        base_score = (skill_match * 0.4 + experience_factor * 0.2 + 
                     availability_factor * 0.2 + location_match * 0.1 + performance_factor * 0.1)
        
        # Adjust based on criteria
        if criteria == "cost_optimized":
            # Favor lower cost technicians
            cost_factor = 1.0 - (technician['hourly_rate'] - 25) / 50  # Normalize to 0-1
            base_score = base_score * 0.7 + cost_factor * 0.3
        elif criteria == "time_optimized":
            # Favor faster technicians
            time_factor = technician['availability_hours'] / 50  # Normalize to 0-1
            base_score = base_score * 0.7 + time_factor * 0.3
        elif criteria == "quality_optimized":
            # Favor high-performance technicians
            base_score = base_score * 0.7 + performance_factor * 0.3
        
        return max(0, min(1, base_score))
    
    def _calculate_optimization_score(self, assignments: List[Dict[str, Any]], project: Dict[str, Any]) -> float:
        """Calculate optimization score for assignments"""
        if not assignments:
            return 0.0
        
        # Average assignment score
        avg_score = np.mean([a["assignment_score"] for a in assignments])
        
        # Skill coverage
        required_skills = set(project['required_skills'])
        covered_skills = set()
        for assignment in assignments:
            technician = next(t for t in self.resource_data['technicians'] if t['technician_id'] == assignment['technician_id'])
            covered_skills.update(technician['skills'])
        
        skill_coverage = len(required_skills.intersection(covered_skills)) / len(required_skills)
        
        # Cost efficiency
        total_cost = sum(a["cost"] for a in assignments)
        cost_efficiency = 1.0 - (total_cost / (project['budget'] * 0.8))  # Assume 80% of budget is target
        cost_efficiency = max(0, min(1, cost_efficiency))
        
        # Time efficiency
        max_time = max(a["estimated_completion_days"] for a in assignments)
        time_efficiency = 1.0 - (max_time / 30)  # Assume 30 days is target
        time_efficiency = max(0, min(1, time_efficiency))
        
        return (avg_score * 0.4 + skill_coverage * 0.3 + cost_efficiency * 0.15 + time_efficiency * 0.15)
    
    def _generate_project_schedule(self, projects: List[Dict[str, Any]], time_horizon: str, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate project schedule"""
        # Convert time horizon to days
        if time_horizon == "7_days":
            horizon_days = 7
        elif time_horizon == "30_days":
            horizon_days = 30
        elif time_horizon == "90_days":
            horizon_days = 90
        else:
            horizon_days = 30
        
        schedule = []
        current_date = datetime.now()
        
        for project in projects:
            # Find best assignment
            best_assignments = self._find_best_assignments(project, "balanced")
            
            if best_assignments:
                best_assignment = best_assignments[0]
                technician = next(t for t in self.resource_data['technicians'] if t['technician_id'] == best_assignment['technician_id'])
                
                # Calculate schedule
                start_date = current_date + timedelta(days=random.randint(0, 7))
                duration_days = best_assignment['estimated_completion_days']
                end_date = start_date + timedelta(days=duration_days)
                
                schedule.append({
                    "project_id": project['project_id'],
                    "project_name": project['name'],
                    "technician_id": best_assignment['technician_id'],
                    "technician_name": best_assignment['technician_name'],
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "duration_days": duration_days,
                    "priority": project['priority'],
                    "status": "scheduled",
                    "estimated_cost": best_assignment['cost'],
                    "skill_match": best_assignment['skill_match']
                })
        
        return schedule
    
    def _calculate_schedule_metrics(self, schedule: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate schedule metrics"""
        if not schedule:
            return {}
        
        total_projects = len(schedule)
        total_duration = sum(item['duration_days'] for item in schedule)
        total_cost = sum(item['estimated_cost'] for item in schedule)
        avg_skill_match = np.mean([item['skill_match'] for item in schedule])
        
        # Priority distribution
        priority_dist = {}
        for item in schedule:
            priority = item['priority']
            priority_dist[priority] = priority_dist.get(priority, 0) + 1
        
        return {
            "total_projects": total_projects,
            "total_duration_days": round(total_duration, 1),
            "total_estimated_cost": round(total_cost, 2),
            "average_duration_days": round(total_duration / total_projects, 1),
            "average_cost": round(total_cost / total_projects, 2),
            "average_skill_match": round(avg_skill_match, 3),
            "priority_distribution": priority_dist
        }
    
    def _identify_schedule_conflicts(self, schedule: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify schedule conflicts"""
        conflicts = []
        
        # Group by technician
        technician_schedules = {}
        for item in schedule:
            tech_id = item['technician_id']
            if tech_id not in technician_schedules:
                technician_schedules[tech_id] = []
            technician_schedules[tech_id].append(item)
        
        # Check for overlapping schedules
        for tech_id, tech_schedule in technician_schedules.items():
            if len(tech_schedule) > 1:
                # Sort by start date
                tech_schedule.sort(key=lambda x: x['start_date'])
                
                for i in range(len(tech_schedule) - 1):
                    current = tech_schedule[i]
                    next_item = tech_schedule[i + 1]
                    
                    current_end = datetime.fromisoformat(current['end_date'])
                    next_start = datetime.fromisoformat(next_item['start_date'])
                    
                    if current_end > next_start:
                        conflicts.append({
                            "type": "schedule_overlap",
                            "technician_id": tech_id,
                            "conflicting_projects": [current['project_id'], next_item['project_id']],
                            "overlap_days": (current_end - next_start).days,
                            "severity": "high"
                        })
        
        return conflicts
    
    def _identify_bottlenecks(self, schedule: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify resource bottlenecks"""
        bottlenecks = []
        
        # Technician workload analysis
        technician_workload = {}
        for item in schedule:
            tech_id = item['technician_id']
            if tech_id not in technician_workload:
                technician_workload[tech_id] = 0
            technician_workload[tech_id] += item['duration_days']
        
        # Find overloaded technicians
        for tech_id, workload in technician_workload.items():
            if workload > 20:  # More than 20 days of work
                bottlenecks.append({
                    "type": "technician_overload",
                    "technician_id": tech_id,
                    "workload_days": workload,
                    "severity": "high" if workload > 30 else "medium",
                    "recommendation": "Redistribute workload or add additional resources"
                })
        
        # Skill shortage analysis
        required_skills = set()
        for item in schedule:
            project = next(p for p in self.resource_data['projects'] if p['project_id'] == item['project_id'])
            required_skills.update(project['required_skills'])
        
        skill_availability = {}
        for skill in required_skills:
            available_technicians = [t for t in self.resource_data['technicians'] if skill in t['skills']]
            skill_availability[skill] = len(available_technicians)
        
        for skill, availability in skill_availability.items():
            if availability < 3:  # Less than 3 technicians with this skill
                bottlenecks.append({
                    "type": "skill_shortage",
                    "skill": skill,
                    "available_technicians": availability,
                    "severity": "high" if availability < 2 else "medium",
                    "recommendation": "Train additional technicians or hire specialists"
                })
        
        return bottlenecks
    
    def _generate_scheduling_recommendations(self, schedule: List[Dict[str, Any]], 
                                           conflicts: List[Dict[str, Any]], 
                                           bottlenecks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate scheduling recommendations"""
        recommendations = []
        
        # Conflict resolution recommendations
        for conflict in conflicts:
            if conflict['type'] == 'schedule_overlap':
                recommendations.append({
                    "type": "resolve_conflict",
                    "priority": "high",
                    "description": f"Resolve schedule overlap for technician {conflict['technician_id']}",
                    "action": "Adjust project timelines or reassign resources"
                })
        
        # Bottleneck resolution recommendations
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'technician_overload':
                recommendations.append({
                    "type": "reduce_workload",
                    "priority": "high",
                    "description": f"Reduce workload for technician {bottleneck['technician_id']}",
                    "action": bottleneck['recommendation']
                })
            elif bottleneck['type'] == 'skill_shortage':
                recommendations.append({
                    "type": "address_skill_shortage",
                    "priority": "medium",
                    "description": f"Address shortage of {bottleneck['skill']} skills",
                    "action": bottleneck['recommendation']
                })
        
        # General optimization recommendations
        if len(schedule) > 10:
            recommendations.append({
                "type": "optimize_schedule",
                "priority": "medium",
                "description": "Consider using advanced scheduling algorithms for large project portfolios",
                "action": "Implement automated scheduling optimization"
            })
        
        return recommendations
    
    def _analyze_technician_capacity(self, period: str) -> Dict[str, Any]:
        """Analyze technician capacity"""
        technicians = self.resource_data['technicians']
        
        # Calculate capacity metrics
        total_capacity = sum(t['availability_hours'] for t in technicians)
        total_utilization = sum(t['current_workload'] * t['availability_hours'] for t in technicians)
        avg_utilization = total_utilization / total_capacity if total_capacity > 0 else 0
        
        # Capacity by specialization
        specialization_capacity = {}
        for tech in technicians:
            spec = tech['specialization']
            if spec not in specialization_capacity:
                specialization_capacity[spec] = {'total_hours': 0, 'utilized_hours': 0, 'count': 0}
            specialization_capacity[spec]['total_hours'] += tech['availability_hours']
            specialization_capacity[spec]['utilized_hours'] += tech['current_workload'] * tech['availability_hours']
            specialization_capacity[spec]['count'] += 1
        
        # Calculate utilization by specialization
        for spec in specialization_capacity:
            total = specialization_capacity[spec]['total_hours']
            utilized = specialization_capacity[spec]['utilized_hours']
            specialization_capacity[spec]['utilization'] = utilized / total if total > 0 else 0
        
        return {
            "total_technicians": len(technicians),
            "total_capacity_hours": total_capacity,
            "total_utilization_hours": round(total_utilization, 1),
            "average_utilization": round(avg_utilization, 3),
            "specialization_capacity": specialization_capacity,
            "capacity_status": "optimal" if 0.6 <= avg_utilization <= 0.8 else "underutilized" if avg_utilization < 0.6 else "overutilized"
        }
    
    def _analyze_skill_capacity(self) -> Dict[str, Any]:
        """Analyze skill capacity across technicians"""
        technicians = self.resource_data['technicians']
        
        # Count technicians by skill
        skill_counts = {}
        for tech in technicians:
            for skill in tech['skills']:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Calculate skill capacity metrics
        skill_capacity = {}
        for skill, count in skill_counts.items():
            # Calculate average experience for this skill
            skill_experience = [tech['experience_years'] for tech in technicians if skill in tech['skills']]
            avg_experience = np.mean(skill_experience) if skill_experience else 0
            
            # Calculate capacity level
            if count >= 5:
                capacity_level = "high"
            elif count >= 3:
                capacity_level = "medium"
            else:
                capacity_level = "low"
            
            skill_capacity[skill] = {
                "technician_count": count,
                "average_experience": round(avg_experience, 1),
                "capacity_level": capacity_level
            }
        
        return skill_capacity
    
    def _analyze_location_capacity(self) -> Dict[str, Any]:
        """Analyze capacity by location"""
        technicians = self.resource_data['technicians']
        
        # Group technicians by location
        location_capacity = {}
        for tech in technicians:
            location = tech['location']
            if location not in location_capacity:
                location_capacity[location] = {
                    'technician_count': 0,
                    'total_hours': 0,
                    'utilized_hours': 0,
                    'skills': set()
                }
            
            location_capacity[location]['technician_count'] += 1
            location_capacity[location]['total_hours'] += tech['availability_hours']
            location_capacity[location]['utilized_hours'] += tech['current_workload'] * tech['availability_hours']
            location_capacity[location]['skills'].update(tech['skills'])
        
        # Calculate metrics for each location
        for location in location_capacity:
            total = location_capacity[location]['total_hours']
            utilized = location_capacity[location]['utilized_hours']
            location_capacity[location]['utilization'] = utilized / total if total > 0 else 0
            location_capacity[location]['skills'] = list(location_capacity[location]['skills'])
        
        return location_capacity
    
    def _calculate_capacity_metrics(self, technician_capacity: Dict[str, Any], 
                                  skill_capacity: Dict[str, Any], 
                                  location_capacity: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall capacity metrics"""
        # Overall utilization
        overall_utilization = technician_capacity['average_utilization']
        
        # Skill coverage
        total_skills = len(skill_capacity)
        high_capacity_skills = len([s for s in skill_capacity.values() if s['capacity_level'] == 'high'])
        skill_coverage = high_capacity_skills / total_skills if total_skills > 0 else 0
        
        # Location balance
        location_utilizations = [loc['utilization'] for loc in location_capacity.values()]
        location_balance = 1.0 - np.std(location_utilizations) if location_utilizations else 1.0
        
        # Capacity score
        capacity_score = (overall_utilization * 0.4 + skill_coverage * 0.3 + location_balance * 0.3)
        
        return {
            "overall_utilization": round(overall_utilization, 3),
            "skill_coverage": round(skill_coverage, 3),
            "location_balance": round(location_balance, 3),
            "capacity_score": round(capacity_score, 3),
            "capacity_status": "optimal" if 0.7 <= capacity_score <= 0.9 else "needs_attention"
        }
    
    def _identify_capacity_constraints(self, technician_capacity: Dict[str, Any], 
                                     skill_capacity: Dict[str, Any], 
                                     location_capacity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify capacity constraints"""
        constraints = []
        
        # Technician capacity constraints
        if technician_capacity['capacity_status'] == 'overutilized':
            constraints.append({
                "type": "technician_overload",
                "severity": "high",
                "description": "Overall technician utilization is too high",
                "recommendation": "Hire additional technicians or reduce workload"
            })
        elif technician_capacity['capacity_status'] == 'underutilized':
            constraints.append({
                "type": "technician_underutilization",
                "severity": "medium",
                "description": "Overall technician utilization is too low",
                "recommendation": "Increase project load or optimize scheduling"
            })
        
        # Skill capacity constraints
        low_capacity_skills = [skill for skill, data in skill_capacity.items() if data['capacity_level'] == 'low']
        if low_capacity_skills:
            constraints.append({
                "type": "skill_shortage",
                "severity": "high",
                "description": f"Low capacity for skills: {', '.join(low_capacity_skills)}",
                "recommendation": "Train technicians or hire specialists"
            })
        
        # Location capacity constraints
        for location, data in location_capacity.items():
            if data['utilization'] > 0.9:
                constraints.append({
                    "type": "location_overload",
                    "severity": "medium",
                    "description": f"Location {location} is overutilized",
                    "recommendation": "Redistribute workload or add resources to this location"
                })
        
        return constraints
    
    def _generate_capacity_recommendations(self, constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate capacity recommendations"""
        recommendations = []
        
        for constraint in constraints:
            recommendations.append({
                "priority": constraint['severity'],
                "action": constraint['recommendation'],
                "constraint_type": constraint['type'],
                "description": constraint['description']
            })
        
        # General recommendations
        recommendations.append({
            "priority": "low",
            "action": "Implement capacity monitoring dashboard",
            "constraint_type": "general",
            "description": "Monitor capacity metrics in real-time"
        })
        
        return recommendations
    
    def _predict_workload(self, horizon: str) -> Dict[str, Any]:
        """Predict future workload"""
        # Convert horizon to days
        if horizon == "30_days":
            days = 30
        elif horizon == "90_days":
            days = 90
        else:
            days = 30
        
        # Simple workload prediction based on current trends
        current_workload = sum(t['current_workload'] for t in self.resource_data['technicians'])
        avg_workload = current_workload / len(self.resource_data['technicians'])
        
        # Predict future workload with some trend
        trend_factor = random.uniform(0.9, 1.1)  # ±10% trend
        predicted_workload = avg_workload * trend_factor
        
        return {
            "current_workload": round(avg_workload, 3),
            "predicted_workload": round(predicted_workload, 3),
            "trend": "increasing" if trend_factor > 1.05 else "decreasing" if trend_factor < 0.95 else "stable",
            "confidence": 0.75
        }
    
    def _predict_skill_needs(self, horizon: str) -> Dict[str, Any]:
        """Predict future skill needs"""
        # Analyze current skill demand from projects
        current_projects = [p for p in self.resource_data['projects'] if p['status'] in ['pending', 'in_progress']]
        
        skill_demand = {}
        for project in current_projects:
            for skill in project['required_skills']:
                skill_demand[skill] = skill_demand.get(skill, 0) + 1
        
        # Predict future skill needs
        predicted_skills = {}
        for skill, demand in skill_demand.items():
            # Simple prediction with some growth
            growth_factor = random.uniform(1.0, 1.3)  # 0-30% growth
            predicted_demand = demand * growth_factor
            
            predicted_skills[skill] = {
                "current_demand": demand,
                "predicted_demand": round(predicted_demand, 1),
                "growth_rate": round((growth_factor - 1) * 100, 1)
            }
        
        return predicted_skills
    
    def _predict_capacity_requirements(self, horizon: str) -> Dict[str, Any]:
        """Predict capacity requirements"""
        # Predict based on current capacity and projected workload
        current_capacity = sum(t['availability_hours'] for t in self.resource_data['technicians'])
        current_utilization = sum(t['current_workload'] * t['availability_hours'] for t in self.resource_data['technicians']) / current_capacity
        
        # Predict future requirements
        utilization_growth = random.uniform(0.05, 0.15)  # 5-15% growth
        predicted_utilization = min(1.0, current_utilization + utilization_growth)
        
        required_capacity = current_capacity * (predicted_utilization / current_utilization) if current_utilization > 0 else current_capacity
        
        return {
            "current_capacity": current_capacity,
            "predicted_required_capacity": round(required_capacity, 1),
            "capacity_gap": round(required_capacity - current_capacity, 1),
            "utilization_growth": round(utilization_growth * 100, 1),
            "recommendation": "hire_additional" if required_capacity > current_capacity * 1.1 else "maintain_current"
        }
    
    def _generate_resource_recommendations(self, workload_prediction: Dict[str, Any], 
                                         skill_prediction: Dict[str, Any], 
                                         capacity_prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate resource recommendations"""
        recommendations = []
        
        # Workload recommendations
        if workload_prediction['trend'] == 'increasing':
            recommendations.append({
                "type": "workload_management",
                "priority": "medium",
                "action": "Prepare for increased workload",
                "description": "Workload is predicted to increase - consider capacity planning"
            })
        
        # Skill recommendations
        high_growth_skills = [skill for skill, data in skill_prediction.items() if data['growth_rate'] > 20]
        if high_growth_skills:
            recommendations.append({
                "type": "skill_development",
                "priority": "high",
                "action": f"Develop skills in: {', '.join(high_growth_skills)}",
                "description": "High growth predicted for these skills"
            })
        
        # Capacity recommendations
        if capacity_prediction['recommendation'] == 'hire_additional':
            recommendations.append({
                "type": "capacity_expansion",
                "priority": "high",
                "action": "Consider hiring additional technicians",
                "description": f"Capacity gap of {capacity_prediction['capacity_gap']:.1f} hours predicted"
            })
        
        return recommendations
    
    def _calculate_prediction_confidence(self, workload_prediction: Dict[str, Any], 
                                       skill_prediction: Dict[str, Any]) -> float:
        """Calculate prediction confidence"""
        # Simple confidence calculation based on data quality and model performance
        base_confidence = 0.7
        
        # Adjust based on prediction consistency
        if workload_prediction['trend'] == 'stable':
            base_confidence += 0.1
        
        # Adjust based on skill prediction diversity
        skill_diversity = len(skill_prediction)
        if skill_diversity > 5:
            base_confidence += 0.1
        
        return min(0.95, base_confidence)
    
    def _analyze_current_workload(self) -> Dict[str, Any]:
        """Analyze current workload distribution"""
        technicians = self.resource_data['technicians']
        
        workload_data = []
        for tech in technicians:
            workload_data.append({
                "technician_id": tech['technician_id'],
                "name": tech['name'],
                "current_workload": tech['current_workload'],
                "availability_hours": tech['availability_hours'],
                "utilization": tech['current_workload'],
                "specialization": tech['specialization']
            })
        
        # Calculate workload statistics
        workloads = [t['current_workload'] for t in workload_data]
        
        return {
            "workload_distribution": workload_data,
            "average_workload": round(np.mean(workloads), 3),
            "workload_std": round(np.std(workloads), 3),
            "min_workload": round(np.min(workloads), 3),
            "max_workload": round(np.max(workloads), 3),
            "workload_balance": 1.0 - (np.std(workloads) / np.mean(workloads)) if np.mean(workloads) > 0 else 1.0
        }
    
    def _generate_optimized_workload(self, current_workload: Dict[str, Any], 
                                   goal: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized workload distribution"""
        workload_data = current_workload['workload_distribution']
        
        # Create optimized workload
        optimized_workload = []
        
        if goal == "balanced":
            # Balance workload across all technicians
            target_workload = current_workload['average_workload']
            
            for tech in workload_data:
                # Adjust workload towards target
                current = tech['current_workload']
                adjustment = (target_workload - current) * 0.5  # 50% adjustment
                optimized = max(0, min(1, current + adjustment))
                
                optimized_workload.append({
                    "technician_id": tech['technician_id'],
                    "name": tech['name'],
                    "current_workload": current,
                    "optimized_workload": round(optimized, 3),
                    "adjustment": round(optimized - current, 3),
                    "specialization": tech['specialization']
                })
        
        elif goal == "specialization_optimized":
            # Optimize based on specialization
            spec_workloads = {}
            for tech in workload_data:
                spec = tech['specialization']
                if spec not in spec_workloads:
                    spec_workloads[spec] = []
                spec_workloads[spec].append(tech)
            
            for tech in workload_data:
                spec = tech['specialization']
                spec_avg = np.mean([t['current_workload'] for t in spec_workloads[spec]])
                
                # Adjust towards specialization average
                current = tech['current_workload']
                adjustment = (spec_avg - current) * 0.3
                optimized = max(0, min(1, current + adjustment))
                
                optimized_workload.append({
                    "technician_id": tech['technician_id'],
                    "name": tech['name'],
                    "current_workload": current,
                    "optimized_workload": round(optimized, 3),
                    "adjustment": round(optimized - current, 3),
                    "specialization": tech['specialization']
                })
        
        else:  # Default to current workload
            for tech in workload_data:
                optimized_workload.append({
                    "technician_id": tech['technician_id'],
                    "name": tech['name'],
                    "current_workload": tech['current_workload'],
                    "optimized_workload": tech['current_workload'],
                    "adjustment": 0,
                    "specialization": tech['specialization']
                })
        
        return {
            "optimized_distribution": optimized_workload,
            "optimization_goal": goal,
            "total_technicians": len(optimized_workload)
        }
    
    def _calculate_workload_metrics(self, optimized_workload: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate workload optimization metrics"""
        workloads = [t['optimized_workload'] for t in optimized_workload['optimized_distribution']]
        
        return {
            "average_workload": round(np.mean(workloads), 3),
            "workload_std": round(np.std(workloads), 3),
            "workload_balance": 1.0 - (np.std(workloads) / np.mean(workloads)) if np.mean(workloads) > 0 else 1.0,
            "overloaded_technicians": len([w for w in workloads if w > 0.8]),
            "underutilized_technicians": len([w for w in workloads if w < 0.3])
        }
    
    def _identify_workload_adjustments(self, current_workload: Dict[str, Any], 
                                     optimized_workload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify workload adjustments needed"""
        adjustments = []
        
        current_data = {t['technician_id']: t for t in current_workload['workload_distribution']}
        optimized_data = {t['technician_id']: t for t in optimized_workload['optimized_distribution']}
        
        for tech_id in current_data:
            current = current_data[tech_id]['current_workload']
            optimized = optimized_data[tech_id]['optimized_workload']
            adjustment = optimized - current
            
            if abs(adjustment) > 0.1:  # Significant adjustment needed
                adjustments.append({
                    "technician_id": tech_id,
                    "technician_name": optimized_data[tech_id]['name'],
                    "current_workload": current,
                    "target_workload": optimized,
                    "adjustment": round(adjustment, 3),
                    "priority": "high" if abs(adjustment) > 0.3 else "medium",
                    "action": "increase" if adjustment > 0 else "decrease"
                })
        
        return adjustments
    
    def _calculate_workload_improvements(self, current_workload: Dict[str, Any], 
                                       optimized_workload: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate expected improvements from workload optimization"""
        current_balance = current_workload['workload_balance']
        optimized_balance = optimized_workload['optimized_distribution']
        
        # Calculate new balance
        workloads = [t['optimized_workload'] for t in optimized_balance]
        new_balance = 1.0 - (np.std(workloads) / np.mean(workloads)) if np.mean(workloads) > 0 else 1.0
        
        return {
            "current_balance": round(current_balance, 3),
            "optimized_balance": round(new_balance, 3),
            "balance_improvement": round(new_balance - current_balance, 3),
            "efficiency_gain": round((new_balance - current_balance) * 100, 1),
            "expected_benefits": [
                "More balanced workload distribution",
                "Reduced technician burnout risk",
                "Improved project delivery consistency"
            ]
        }
    
    def _generate_detailed_schedule(self, technicians: List[Dict[str, Any]], schedule_type: str) -> List[Dict[str, Any]]:
        """Generate detailed schedule for technicians"""
        schedule = []
        
        # Generate schedule based on type
        if schedule_type == "weekly":
            days = 7
        elif schedule_type == "monthly":
            days = 30
        else:
            days = 7
        
        for tech in technicians:
            tech_schedule = {
                "technician_id": tech['technician_id'],
                "technician_name": tech['name'],
                "specialization": tech['specialization'],
                "availability_hours": tech['availability_hours'],
                "current_workload": tech['current_workload'],
                "schedule": []
            }
            
            # Generate daily schedule
            for day in range(days):
                date = datetime.now() + timedelta(days=day)
                
                # Simulate daily schedule
                daily_hours = tech['availability_hours'] * (1 - tech['current_workload'])
                daily_hours = max(0, daily_hours)
                
                # Generate tasks for the day
                tasks = []
                remaining_hours = daily_hours
                
                while remaining_hours > 0 and len(tasks) < 3:  # Max 3 tasks per day
                    task_hours = min(remaining_hours, random.uniform(2, 6))
                    task_type = random.choice(['project_work', 'maintenance', 'training', 'meeting'])
                    
                    tasks.append({
                        "task_type": task_type,
                        "hours": round(task_hours, 1),
                        "description": f"{task_type.replace('_', ' ').title()} task"
                    })
                    
                    remaining_hours -= task_hours
                
                tech_schedule["schedule"].append({
                    "date": date.strftime("%Y-%m-%d"),
                    "day_of_week": date.strftime("%A"),
                    "total_hours": round(daily_hours, 1),
                    "tasks": tasks
                })
            
            schedule.append(tech_schedule)
        
        return schedule
    
    def _calculate_schedule_efficiency(self, schedule: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate schedule efficiency metrics"""
        total_technicians = len(schedule)
        total_hours_scheduled = 0
        total_availability = 0
        
        for tech_schedule in schedule:
            for day_schedule in tech_schedule['schedule']:
                total_hours_scheduled += day_schedule['total_hours']
            total_availability += tech_schedule['availability_hours'] * len(tech_schedule['schedule'])
        
        utilization = total_hours_scheduled / total_availability if total_availability > 0 else 0
        
        # Calculate task distribution
        task_types = {}
        for tech_schedule in schedule:
            for day_schedule in tech_schedule['schedule']:
                for task in day_schedule['tasks']:
                    task_type = task['task_type']
                    task_types[task_type] = task_types.get(task_type, 0) + 1
        
        return {
            "total_technicians": total_technicians,
            "total_hours_scheduled": round(total_hours_scheduled, 1),
            "utilization_rate": round(utilization, 3),
            "task_distribution": task_types,
            "efficiency_score": round(utilization * 100, 1)
        }
    
    def _identify_schedule_optimization_opportunities(self, schedule: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify schedule optimization opportunities"""
        opportunities = []
        
        # Analyze utilization rates
        for tech_schedule in schedule:
            total_scheduled = sum(day['total_hours'] for day in tech_schedule['schedule'])
            total_available = tech_schedule['availability_hours'] * len(tech_schedule['schedule'])
            utilization = total_scheduled / total_available if total_available > 0 else 0
            
            if utilization < 0.5:
                opportunities.append({
                    "type": "underutilization",
                    "technician_id": tech_schedule['technician_id'],
                    "utilization": round(utilization, 3),
                    "recommendation": "Increase workload or assign additional projects"
                })
            elif utilization > 0.9:
                opportunities.append({
                    "type": "overutilization",
                    "technician_id": tech_schedule['technician_id'],
                    "utilization": round(utilization, 3),
                    "recommendation": "Reduce workload or redistribute tasks"
                })
        
        # Analyze task distribution
        task_counts = {}
        for tech_schedule in schedule:
            for day_schedule in tech_schedule['schedule']:
                for task in day_schedule['tasks']:
                    task_type = task['task_type']
                    task_counts[task_type] = task_counts.get(task_type, 0) + 1
        
        # Check for imbalanced task distribution
        if task_counts:
            total_tasks = sum(task_counts.values())
            for task_type, count in task_counts.items():
                percentage = count / total_tasks
                if percentage > 0.6:  # More than 60% of one task type
                    opportunities.append({
                        "type": "task_imbalance",
                        "task_type": task_type,
                        "percentage": round(percentage * 100, 1),
                        "recommendation": f"Diversify tasks - too much {task_type}"
                    })
        
        return opportunities
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        return {
            "agent_id": self.agent_id,
            "status": "active" if self.model_loaded else "inactive",
            "model_loaded": self.model_loaded,
            "health_score": 0.91 if self.model_loaded else 0.0,
            "last_activity": datetime.utcnow().isoformat(),
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "average_response_time_ms": self.metrics.average_response_time_ms,
                "error_rate": self.metrics.error_rate
            }
        }

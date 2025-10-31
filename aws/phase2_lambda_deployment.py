#!/usr/bin/env python3
"""
Phase 2: Deploy Lambda Functions with Dependencies
Deploys all 10 agents as Lambda functions with proper dependencies and configuration
"""

import boto3
import json
import tempfile
import os
from pathlib import Path

# Configuration
PROJECT_NAME = 'msp-intelligence-mesh'
BUCKET_NAME = 'msp-intelligence-mesh-backend'
LAMBDA_ROLE_ARN = 'arn:aws:iam::905418166919:role/msp-intelligence-mesh-lambda-role'

# Agent configurations
AGENTS = {
    'threat_intelligence': {
        'memory': 1024,
        'timeout': 60,
        'description': 'Threat Intelligence Agent with DistilBERT'
    },
    'market_intelligence': {
        'memory': 1024,
        'timeout': 60,
        'description': 'Market Intelligence Agent with Sentiment Analysis'
    },
    'nlp_query': {
        'memory': 1536,
        'timeout': 60,
        'description': 'NLP Query Agent with FLAN-T5'
    },
    'client_health': {
        'memory': 1024,
        'timeout': 60,
        'description': 'Client Health Prediction with ML'
    },
    'revenue_optimization': {
        'memory': 1024,
        'timeout': 60,
        'description': 'Revenue Optimization with Time-Series'
    },
    'anomaly_detection': {
        'memory': 1024,
        'timeout': 60,
        'description': 'Anomaly Detection with Isolation Forest'
    },
    'collaboration': {
        'memory': 1024,
        'timeout': 60,
        'description': 'Collaboration Matching with Sentence-BERT'
    },
    'compliance': {
        'memory': 512,
        'timeout': 30,
        'description': 'Security Compliance Agent'
    },
    'resource_allocation': {
        'memory': 512,
        'timeout': 30,
        'description': 'Resource Allocation Agent'
    },
    'federated_learning': {
        'memory': 1024,
        'timeout': 60,
        'description': 'Federated Learning Coordinator'
    }
}

def create_lambda_function(agent_name, config):
    """Create or update a Lambda function"""
    lambda_client = boto3.client('lambda')
    s3_client = boto3.client('s3')
    
    function_name = f"{PROJECT_NAME}-{agent_name.replace('_', '-')}"
    
    # Lambda handler code
    handler_code = f'''
import json
import sys
import os
import boto3
from datetime import datetime

# Add the agent module to path
sys.path.append('/opt/python')
sys.path.append('/var/task')

try:
    from {agent_name}_agent import {agent_name.title().replace('_', '')}Agent
except ImportError:
    # Fallback for simple agents
    from backend.agents.{agent_name}_agent import {agent_name.title().replace('_', '')}Agent

def lambda_handler(event, context):
    """
    Lambda handler for {agent_name} agent
    """
    try:
        # Initialize agent
        agent = {agent_name.title().replace('_', '')}Agent()
        
        # Parse request
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {{}})
        
        # Process request based on agent type
        if agent_name == 'threat_intelligence':
            result = agent.analyze_threat(body.get('text', ''))
        elif agent_name == 'market_intelligence':
            result = agent.analyze_market(body.get('query', ''))
        elif agent_name == 'nlp_query':
            result = agent.process_query(body.get('query', ''))
        elif agent_name == 'client_health':
            result = agent.predict_health(body)
        elif agent_name == 'revenue_optimization':
            result = agent.forecast_revenue(body)
        elif agent_name == 'anomaly_detection':
            result = agent.detect_anomalies(body)
        elif agent_name == 'collaboration':
            result = agent.find_partners(body)
        elif agent_name == 'compliance':
            result = agent.check_compliance(body)
        elif agent_name == 'resource_allocation':
            result = agent.allocate_resources(body)
        elif agent_name == 'federated_learning':
            result = agent.get_status(body)
        else:
            result = {{"error": "Unknown agent type"}}
        
        # Add metadata
        result['agent'] = agent_name
        result['timestamp'] = datetime.utcnow().isoformat()
        result['lambda_request_id'] = context.aws_request_id
        
        return {{
            'statusCode': 200,
            'headers': {{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }},
            'body': json.dumps(result)
        }}
        
    except Exception as e:
        print(f"Error in {agent_name}: {{str(e)}}")
        return {{
            'statusCode': 500,
            'headers': {{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }},
            'body': json.dumps({{
                'error': str(e),
                'agent': agent_name,
                'timestamp': datetime.utcnow().isoformat()
            }})
        }}
'''
    
    # Create deployment package
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
        zip_path = tmp_file.name
    
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add handler
            zipf.writestr('lambda_function.py', handler_code)
            
            # Add requirements
            requirements = '''
boto3>=1.26.0
transformers>=4.21.0
torch>=1.12.0
scikit-learn>=1.1.0
numpy>=1.21.0
pandas>=1.4.0
sentence-transformers>=2.2.0
'''
            zipf.writestr('requirements.txt', requirements)
        
        # Upload to S3
        s3_key = f'lambda-deployments/{agent_name}.zip'
        s3_client.upload_file(zip_path, BUCKET_NAME, s3_key)
        
        # Create or update Lambda function
        try:
            # Try to create new function
            lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=LAMBDA_ROLE_ARN,
                Handler='lambda_function.lambda_handler',
                Code={
                    'S3Bucket': BUCKET_NAME,
                    'S3Key': s3_key
                },
                Description=config['description'],
                Timeout=config['timeout'],
                MemorySize=config['memory'],
                Environment={
                    'Variables': {
                        'PROJECT_NAME': PROJECT_NAME,
                        'AGENT_NAME': agent_name,
                        'AWS_REGION': 'us-east-1'
                    }
                },
                TracingConfig={'Mode': 'Active'},
                Layers=[
                    'arn:aws:lambda:us-east-1:336392948345:layer:AWSSDKPandas-Python39:1'
                ]
            )
            print(f"✅ Created Lambda: {function_name}")
            
        except lambda_client.exceptions.ResourceConflictException:
            # Update existing function
            lambda_client.update_function_code(
                FunctionName=function_name,
                S3Bucket=BUCKET_NAME,
                S3Key=s3_key
            )
            lambda_client.update_function_configuration(
                FunctionName=function_name,
                Role=LAMBDA_ROLE_ARN,
                Handler='lambda_function.lambda_handler',
                Description=config['description'],
                Timeout=config['timeout'],
                MemorySize=config['memory'],
                Environment={
                    'Variables': {
                        'PROJECT_NAME': PROJECT_NAME,
                        'AGENT_NAME': agent_name,
                        'AWS_REGION': 'us-east-1'
                    }
                }
            )
            print(f"✅ Updated Lambda: {function_name}")
        
        return function_name
        
    except Exception as e:
        print(f"❌ Error deploying {agent_name}: {e}")
        return None
    finally:
        # Clean up
        if os.path.exists(zip_path):
            os.unlink(zip_path)

def create_agents_status_lambda():
    """Create a special Lambda for agents status endpoint"""
    lambda_client = boto3.client('lambda')
    s3_client = boto3.client('s3')
    
    function_name = f"{PROJECT_NAME}-agents-status"
    
    handler_code = '''
import json
from datetime import datetime

def lambda_handler(event, context):
    """
    Lambda handler for agents status endpoint
    """
    try:
        # Mock agent status data (in production, this would query actual Lambda functions)
        agents = {
            "threat_intelligence": {
                "status": "active",
                "health_score": 0.95,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat()
            },
            "market_intelligence": {
                "status": "active", 
                "health_score": 0.93,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat()
            },
            "nlp_query": {
                "status": "active",
                "health_score": 0.93,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat()
            },
            "collaboration_matching": {
                "status": "active",
                "health_score": 0.92,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat()
            },
            "client_health": {
                "status": "active",
                "health_score": 0.94,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat()
            },
            "revenue_optimization": {
                "status": "active",
                "health_score": 0.92,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat()
            },
            "anomaly_detection": {
                "status": "active",
                "health_score": 0.96,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat()
            },
            "security_compliance": {
                "status": "active",
                "health_score": 0.88,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat()
            },
            "resource_allocation": {
                "status": "active",
                "health_score": 0.91,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat()
            },
            "federated_learning": {
                "status": "active",
                "health_score": 0.98,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat()
            }
        }
        
        response_data = {
            "agents": agents,
            "total_agents": len(agents),
            "active_agents": len(agents),
            "status_time": datetime.utcnow().isoformat()
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(response_data)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }
'''
    
    # Create deployment package
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
        zip_path = tmp_file.name
    
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr('lambda_function.py', handler_code)
        
        # Upload to S3
        s3_key = f'lambda-deployments/agents-status.zip'
        s3_client.upload_file(zip_path, BUCKET_NAME, s3_key)
        
        # Create Lambda function
        try:
            lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=LAMBDA_ROLE_ARN,
                Handler='lambda_function.lambda_handler',
                Code={
                    'S3Bucket': BUCKET_NAME,
                    'S3Key': s3_key
                },
                Description='Agents Status Endpoint',
                Timeout=30,
                MemorySize=256,
                Environment={
                    'Variables': {
                        'PROJECT_NAME': PROJECT_NAME
                    }
                },
                TracingConfig={'Mode': 'Active'}
            )
            print(f"✅ Created Lambda: {function_name}")
            return function_name
            
        except lambda_client.exceptions.ResourceConflictException:
            # Update existing
            lambda_client.update_function_code(
                FunctionName=function_name,
                S3Bucket=BUCKET_NAME,
                S3Key=s3_key
            )
            print(f"✅ Updated Lambda: {function_name}")
            return function_name
            
    except Exception as e:
        print(f"❌ Error deploying agents-status: {e}")
        return None
    finally:
        if os.path.exists(zip_path):
            os.unlink(zip_path)

def main():
    print("🚀 Phase 2: Deploying Lambda Functions")
    print("=" * 50)
    
    deployed_functions = []
    
    # Deploy all agent Lambda functions
    for agent_name, config in AGENTS.items():
        print(f"\n📦 Deploying {agent_name}...")
        function_name = create_lambda_function(agent_name, config)
        if function_name:
            deployed_functions.append(function_name)
    
    # Deploy agents status Lambda
    print(f"\n📊 Deploying agents status endpoint...")
    status_function = create_agents_status_lambda()
    if status_function:
        deployed_functions.append(status_function)
    
    print("\n" + "=" * 50)
    print("✅ Phase 2 Complete!")
    print(f"📦 Lambda Functions Deployed: {len(deployed_functions)}")
    for func in deployed_functions:
        print(f"  ✅ {func}")
    
    print(f"\n🌐 Lambda Console: https://console.aws.amazon.com/lambda/home?region=us-east-1")
    print(f"📊 S3 Backend: https://s3.console.aws.amazon.com/s3/buckets/{BUCKET_NAME}")

if __name__ == "__main__":
    main()










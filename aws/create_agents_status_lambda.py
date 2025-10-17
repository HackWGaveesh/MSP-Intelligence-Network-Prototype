#!/usr/bin/env python3
"""
Create Agents Status Lambda Function
Creates a dedicated Lambda function for the /agents/status endpoint
"""

import boto3
import json
import tempfile
import os
import zipfile

def create_agents_status_lambda():
    """Create Lambda function for agents status endpoint"""
    lambda_client = boto3.client('lambda')
    s3_client = boto3.client('s3')
    
    function_name = 'msp-intelligence-mesh-agents-status'
    
    # Lambda handler code
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
                "last_activity": datetime.utcnow().isoformat(),
                "name": "Threat Intelligence",
                "icon": "🛡️"
            },
            "market_intelligence": {
                "status": "active", 
                "health_score": 0.93,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat(),
                "name": "Market Intelligence",
                "icon": "💼"
            },
            "nlp_query": {
                "status": "active",
                "health_score": 0.93,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat(),
                "name": "NLP Query Assistant",
                "icon": "💬"
            },
            "collaboration_matching": {
                "status": "active",
                "health_score": 0.92,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat(),
                "name": "Collaboration Matching",
                "icon": "🤝"
            },
            "client_health": {
                "status": "active",
                "health_score": 0.94,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat(),
                "name": "Client Health Prediction",
                "icon": "📊"
            },
            "revenue_optimization": {
                "status": "active",
                "health_score": 0.92,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat(),
                "name": "Revenue Optimization",
                "icon": "💰"
            },
            "anomaly_detection": {
                "status": "active",
                "health_score": 0.96,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat(),
                "name": "Anomaly Detection",
                "icon": "🔍"
            },
            "security_compliance": {
                "status": "active",
                "health_score": 0.88,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat(),
                "name": "Security Compliance",
                "icon": "✅"
            },
            "resource_allocation": {
                "status": "active",
                "health_score": 0.91,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat(),
                "name": "Resource Allocation",
                "icon": "📅"
            },
            "federated_learning": {
                "status": "active",
                "health_score": 0.98,
                "model_loaded": True,
                "last_activity": datetime.utcnow().isoformat(),
                "name": "Federated Learning",
                "icon": "🌐"
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
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr('lambda_function.py', handler_code)
        
        # Upload to S3
        s3_key = f'lambda-deployments/agents-status.zip'
        s3_client.upload_file(zip_path, 'msp-intelligence-mesh-backend', s3_key)
        
        # Get account ID for role ARN
        sts = boto3.client('sts')
        account_id = sts.get_caller_identity()['Account']
        role_arn = f'arn:aws:iam::{account_id}:role/msp-intelligence-mesh-lambda-execution-role'
        
        # Create Lambda function
        try:
            lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=role_arn,
                Handler='lambda_function.lambda_handler',
                Code={
                    'S3Bucket': 'msp-intelligence-mesh-backend',
                    'S3Key': s3_key
                },
                Description='Agents Status Endpoint for MSP Intelligence Mesh',
                Timeout=30,
                MemorySize=256,
                Environment={
                    'Variables': {
                        'PROJECT_NAME': 'msp-intelligence-mesh'
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
                S3Bucket='msp-intelligence-mesh-backend',
                S3Key=s3_key
            )
            print(f"✅ Updated Lambda: {function_name}")
            return function_name
            
    except Exception as e:
        print(f"❌ Error creating agents-status Lambda: {e}")
        return None
    finally:
        if os.path.exists(zip_path):
            os.unlink(zip_path)

def add_agents_status_to_api_gateway():
    """Add /agents/status endpoint to API Gateway"""
    api_client = boto3.client('apigateway')
    
    api_id = 'mojoawwjv2'  # Existing API ID
    
    try:
        # Get root resource
        resources = api_client.get_resources(restApiId=api_id)
        root_resource_id = None
        for resource in resources['items']:
            if resource['path'] == '/':
                root_resource_id = resource['id']
                break
        
        if not root_resource_id:
            print("❌ Could not find root resource")
            return False
        
        # Create /agents resource
        try:
            agents_resource = api_client.create_resource(
                restApiId=api_id,
                parentId=root_resource_id,
                pathPart='agents'
            )
            agents_resource_id = agents_resource['id']
            print("✅ Created /agents resource")
        except api_client.exceptions.ConflictException:
            # Resource already exists, find it
            for resource in resources['items']:
                if resource['path'] == '/agents':
                    agents_resource_id = resource['id']
                    break
            print("✅ Found existing /agents resource")
        
        # Create /agents/status resource
        try:
            status_resource = api_client.create_resource(
                restApiId=api_id,
                parentId=agents_resource_id,
                pathPart='status'
            )
            status_resource_id = status_resource['id']
            print("✅ Created /agents/status resource")
        except api_client.exceptions.ConflictException:
            # Resource already exists, find it
            for resource in resources['items']:
                if resource['path'] == '/agents/status':
                    status_resource_id = resource['id']
                    break
            print("✅ Found existing /agents/status resource")
        
        # Create GET method
        try:
            api_client.put_method(
                restApiId=api_id,
                resourceId=status_resource_id,
                httpMethod='GET',
                authorizationType='NONE'
            )
            print("✅ Created GET method")
        except api_client.exceptions.ConflictException:
            print("✅ GET method already exists")
        
        # Set up Lambda integration
        function_name = 'msp-intelligence-mesh-agents-status'
        lambda_arn = f'arn:aws:lambda:us-east-1:905418166919:function:{function_name}'
        
        try:
            api_client.put_integration(
                restApiId=api_id,
                resourceId=status_resource_id,
                httpMethod='GET',
                type='AWS_PROXY',
                integrationHttpMethod='POST',
                uri=f'arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/{lambda_arn}/invocations'
            )
            print("✅ Set up Lambda integration")
        except api_client.exceptions.ConflictException:
            print("✅ Lambda integration already exists")
        
        # Add Lambda permission for API Gateway
        lambda_client = boto3.client('lambda')
        try:
            lambda_client.add_permission(
                FunctionName=function_name,
                StatementId='api-gateway-invoke',
                Action='lambda:InvokeFunction',
                Principal='apigateway.amazonaws.com',
                SourceArn=f'arn:aws:execute-api:us-east-1:905418166919:{api_id}/*/*'
            )
            print("✅ Added Lambda permission")
        except lambda_client.exceptions.ResourceConflictException:
            print("✅ Lambda permission already exists")
        
        # Deploy API
        api_client.create_deployment(
            restApiId=api_id,
            stageName='prod',
            description='Added agents status endpoint'
        )
        print("✅ Deployed API with new endpoint")
        
        return True
        
    except Exception as e:
        print(f"❌ Error adding endpoint to API Gateway: {e}")
        return False

def main():
    print("🔧 Creating Agents Status Endpoint")
    print("=" * 50)
    
    # Create Lambda function
    print("\n📦 Creating Lambda function...")
    function_name = create_agents_status_lambda()
    
    if function_name:
        # Add to API Gateway
        print("\n🌐 Adding to API Gateway...")
        success = add_agents_status_to_api_gateway()
        
        if success:
            print("\n" + "=" * 50)
            print("✅ Agents Status Endpoint Created!")
            print(f"📦 Lambda: {function_name}")
            print(f"🌐 Endpoint: https://mojoawwjv2.execute-api.us-east-1.amazonaws.com/prod/agents/status")
            
            # Test the endpoint
            print("\n🧪 Testing endpoint...")
            import requests
            try:
                response = requests.get('https://mojoawwjv2.execute-api.us-east-1.amazonaws.com/prod/agents/status', timeout=10)
                if response.status_code == 200:
                    print("✅ Endpoint is working!")
                    data = response.json()
                    print(f"✅ Found {data.get('total_agents', 0)} agents")
                else:
                    print(f"⚠️  Endpoint returned status {response.status_code}")
            except Exception as e:
                print(f"❌ Error testing endpoint: {e}")
        else:
            print("❌ Failed to add endpoint to API Gateway")
    else:
        print("❌ Failed to create Lambda function")

if __name__ == "__main__":
    main()





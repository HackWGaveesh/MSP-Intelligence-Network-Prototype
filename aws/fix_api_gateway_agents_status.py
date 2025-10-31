#!/usr/bin/env python3
"""
Fix API Gateway Agents Status Endpoint
Creates the /agents/status endpoint in API Gateway
"""

import boto3
import json

def fix_api_gateway_agents_status():
    """Fix the API Gateway to include /agents/status endpoint"""
    try:
        apigateway = boto3.client('apigateway')
        lambda_client = boto3.client('lambda')
        
        # Find the existing API
        apis = apigateway.get_rest_apis()
        api_id = None
        for api in apis['items']:
            if 'msp-intelligence-mesh' in api['name'].lower():
                api_id = api['id']
                break
        
        if not api_id:
            print("❌ No API Gateway found")
            return False
        
        print(f"✅ Found API Gateway: {api_id}")
        
        # Get root resource
        resources = apigateway.get_resources(restApiId=api_id)
        root_resource_id = None
        agents_resource_id = None
        
        for resource in resources['items']:
            if resource['path'] == '/':
                root_resource_id = resource['id']
            elif resource['path'] == '/agents':
                agents_resource_id = resource['id']
        
        # Create /agents resource if it doesn't exist
        if not agents_resource_id:
            print("📝 Creating /agents resource...")
            agents_resource = apigateway.create_resource(
                restApiId=api_id,
                parentId=root_resource_id,
                pathPart='agents'
            )
            agents_resource_id = agents_resource['id']
            print(f"✅ Created /agents resource: {agents_resource_id}")
        
        # Create /agents/status resource
        print("📝 Creating /agents/status resource...")
        status_resource = apigateway.create_resource(
            restApiId=api_id,
            parentId=agents_resource_id,
            pathPart='status'
        )
        status_resource_id = status_resource['id']
        print(f"✅ Created /agents/status resource: {status_resource_id}")
        
        # Create GET method for /agents/status
        print("📝 Creating GET method for /agents/status...")
        apigateway.put_method(
            restApiId=api_id,
            resourceId=status_resource_id,
            httpMethod='GET',
            authorizationType='NONE'
        )
        
        # Get Lambda function ARN
        function_name = 'msp-intelligence-mesh-agents-status'
        function_info = lambda_client.get_function(FunctionName=function_name)
        function_arn = function_info['Configuration']['FunctionArn']
        
        # Create Lambda integration
        print("📝 Creating Lambda integration...")
        apigateway.put_integration(
            restApiId=api_id,
            resourceId=status_resource_id,
            httpMethod='GET',
            type='AWS_PROXY',
            integrationHttpMethod='POST',
            uri=f'arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/{function_arn}/invocations'
        )
        
        # Add Lambda permission for API Gateway
        try:
            lambda_client.add_permission(
                FunctionName=function_name,
                StatementId=f'apigateway-{api_id}',
                Action='lambda:InvokeFunction',
                Principal='apigateway.amazonaws.com',
                SourceArn=f'arn:aws:execute-api:us-east-1:*:{api_id}/*/*'
            )
            print("✅ Added Lambda permission for API Gateway")
        except lambda_client.exceptions.ResourceConflictException:
            print("ℹ️  Lambda permission already exists")
        
        # Enable CORS
        print("📝 Enabling CORS...")
        apigateway.put_method_response(
            restApiId=api_id,
            resourceId=status_resource_id,
            httpMethod='GET',
            statusCode='200',
            responseParameters={
                'method.response.header.Access-Control-Allow-Origin': True,
                'method.response.header.Access-Control-Allow-Headers': True,
                'method.response.header.Access-Control-Allow-Methods': True
            }
        )
        
        apigateway.put_integration_response(
            restApiId=api_id,
            resourceId=status_resource_id,
            httpMethod='GET',
            statusCode='200',
            responseParameters={
                'method.response.header.Access-Control-Allow-Origin': "'*'",
                'method.response.header.Access-Control-Allow-Headers': "'Content-Type'",
                'method.response.header.Access-Control-Allow-Methods': "'GET,POST,OPTIONS'"
            }
        )
        
        # Deploy API
        print("📝 Deploying API...")
        apigateway.create_deployment(
            restApiId=api_id,
            stageName='prod'
        )
        
        print("✅ API Gateway /agents/status endpoint created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing API Gateway: {e}")
        return False

def test_endpoint():
    """Test the /agents/status endpoint"""
    import requests
    
    try:
        response = requests.get(
            'https://mojoawwjv2.execute-api.us-east-1.amazonaws.com/prod/agents/status',
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            summary = data.get('summary', {})
            print(f"✅ Test successful!")
            print(f"   Operational Agents: {summary.get('operational_agents', 0)}/{summary.get('total_agents', 0)}")
            print(f"   Total Requests: {summary.get('total_requests_today', 0)}")
            print(f"   Average Uptime: {summary.get('average_uptime', 'N/A')}")
            return True
        else:
            print(f"❌ Test failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🔧 Fixing API Gateway Agents Status Endpoint")
    print("=" * 50)
    
    # Fix API Gateway
    print("\n📊 Fixing API Gateway...")
    api_success = fix_api_gateway_agents_status()
    
    if api_success:
        print("\n⏳ Waiting 10 seconds for deployment...")
        import time
        time.sleep(10)
        
        # Test endpoint
        print("\n🧪 Testing endpoint...")
        test_success = test_endpoint()
        
        if test_success:
            print("\n✅ SUCCESS! Quick Agent Test should now work!")
        else:
            print("\n⚠️  API Gateway updated but endpoint test failed")
    else:
        print("\n❌ Failed to fix API Gateway")
    
    print(f"\n🌐 Test your website: http://msp-intelligence-mesh-frontend.s3-website-us-east-1.amazonaws.com")

if __name__ == "__main__":
    main()










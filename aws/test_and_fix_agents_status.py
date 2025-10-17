#!/usr/bin/env python3
"""
Test and Fix Agents Status
Directly test the Lambda function and fix API Gateway integration
"""

import boto3
import json

def test_lambda_directly():
    """Test the Lambda function directly"""
    try:
        lambda_client = boto3.client('lambda')
        
        # Test the Lambda function directly
        response = lambda_client.invoke(
            FunctionName='msp-intelligence-mesh-agents-status',
            InvocationType='RequestResponse',
            Payload=json.dumps({})
        )
        
        if response['StatusCode'] == 200:
            payload = json.loads(response['Payload'].read())
            print("✅ Lambda function works directly!")
            print(f"   Status Code: {payload.get('statusCode', 'N/A')}")
            if payload.get('statusCode') == 200:
                body = json.loads(payload.get('body', '{}'))
                summary = body.get('summary', {})
                print(f"   Agents: {summary.get('operational_agents', 0)}/{summary.get('total_agents', 0)} operational")
                return True
            else:
                print(f"   Error: {payload.get('body', 'No error message')}")
                return False
        else:
            print(f"❌ Lambda invocation failed: {response['StatusCode']}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Lambda: {e}")
        return False

def fix_api_gateway_integration():
    """Fix the API Gateway integration"""
    try:
        apigateway = boto3.client('apigateway')
        lambda_client = boto3.client('lambda')
        
        # Find the API
        apis = apigateway.get_rest_apis()
        api_id = None
        for api in apis['items']:
            if 'msp-intelligence-mesh' in api['name'].lower():
                api_id = api['id']
                break
        
        if not api_id:
            print("❌ No API Gateway found")
            return False
        
        print(f"✅ Found API: {api_id}")
        
        # Get resources
        resources = apigateway.get_resources(restApiId=api_id)
        status_resource_id = None
        
        for resource in resources['items']:
            if resource['path'] == '/agents/status':
                status_resource_id = resource['id']
                break
        
        if not status_resource_id:
            print("❌ /agents/status resource not found")
            return False
        
        print(f"✅ Found /agents/status resource: {status_resource_id}")
        
        # Get Lambda function ARN
        function_name = 'msp-intelligence-mesh-agents-status'
        function_info = lambda_client.get_function(FunctionName=function_name)
        function_arn = function_info['Configuration']['FunctionArn']
        
        # Update the integration
        print("📝 Updating Lambda integration...")
        apigateway.put_integration(
            restApiId=api_id,
            resourceId=status_resource_id,
            httpMethod='GET',
            type='AWS_PROXY',
            integrationHttpMethod='POST',
            uri=f'arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/{function_arn}/invocations'
        )
        
        # Add Lambda permission
        try:
            lambda_client.add_permission(
                FunctionName=function_name,
                StatementId=f'apigateway-{api_id}-agents-status',
                Action='lambda:InvokeFunction',
                Principal='apigateway.amazonaws.com',
                SourceArn=f'arn:aws:execute-api:us-east-1:*:{api_id}/*/*'
            )
            print("✅ Added Lambda permission")
        except lambda_client.exceptions.ResourceConflictException:
            print("ℹ️  Lambda permission already exists")
        
        # Deploy API
        print("📝 Deploying API...")
        apigateway.create_deployment(
            restApiId=api_id,
            stageName='prod'
        )
        
        print("✅ API Gateway integration updated!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing API Gateway: {e}")
        return False

def test_endpoint():
    """Test the API Gateway endpoint"""
    import requests
    import time
    
    print("⏳ Waiting 5 seconds for deployment...")
    time.sleep(5)
    
    try:
        response = requests.get(
            'https://mojoawwjv2.execute-api.us-east-1.amazonaws.com/prod/agents/status',
            timeout=15
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            summary = data.get('summary', {})
            print("✅ API Gateway endpoint works!")
            print(f"   Operational Agents: {summary.get('operational_agents', 0)}/{summary.get('total_agents', 0)}")
            print(f"   Total Requests: {summary.get('total_requests_today', 0)}")
            print(f"   Average Uptime: {summary.get('average_uptime', 'N/A')}")
            return True
        else:
            print(f"❌ API Gateway test failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"❌ API Gateway test failed: {e}")
        return False

def main():
    print("🔧 Testing and Fixing Agents Status")
    print("=" * 40)
    
    # Test Lambda directly
    print("\n1️⃣ Testing Lambda function directly...")
    lambda_works = test_lambda_directly()
    
    if lambda_works:
        # Fix API Gateway integration
        print("\n2️⃣ Fixing API Gateway integration...")
        api_fixed = fix_api_gateway_integration()
        
        if api_fixed:
            # Test endpoint
            print("\n3️⃣ Testing API Gateway endpoint...")
            endpoint_works = test_endpoint()
            
            if endpoint_works:
                print("\n✅ SUCCESS! Quick Agent Test is now working!")
            else:
                print("\n⚠️  Lambda works but API Gateway endpoint still has issues")
        else:
            print("\n❌ Failed to fix API Gateway integration")
    else:
        print("\n❌ Lambda function itself has issues")
    
    print(f"\n🌐 Test your website: http://msp-intelligence-mesh-frontend.s3-website-us-east-1.amazonaws.com")

if __name__ == "__main__":
    main()





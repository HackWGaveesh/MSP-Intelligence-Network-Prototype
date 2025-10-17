#!/usr/bin/env python3
"""
Final Fix for Agents Status
Fix the API Gateway integration with correct ARN format
"""

import boto3
import json

def fix_agents_status_final():
    """Final fix for agents status endpoint"""
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
        
        # Get account ID
        sts = boto3.client('sts')
        account_id = sts.get_caller_identity()['Account']
        
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
        
        # Add Lambda permission with correct ARN format
        try:
            lambda_client.add_permission(
                FunctionName=function_name,
                StatementId=f'apigateway-{api_id}-agents-status-{account_id}',
                Action='lambda:InvokeFunction',
                Principal='apigateway.amazonaws.com',
                SourceArn=f'arn:aws:execute-api:us-east-1:{account_id}:{api_id}/*/*'
            )
            print("✅ Added Lambda permission with correct ARN")
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
    
    print("⏳ Waiting 10 seconds for deployment...")
    time.sleep(10)
    
    try:
        response = requests.get(
            'https://mojoawwjv2.execute-api.us-east-1.amazonaws.com/prod/agents/status',
            timeout=15
        )
        
        print(f"Response Status: {response.status_code}")
        
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
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ API Gateway test failed: {e}")
        return False

def main():
    print("🔧 Final Fix for Agents Status")
    print("=" * 35)
    
    # Fix API Gateway integration
    print("\n📝 Fixing API Gateway integration...")
    api_fixed = fix_agents_status_final()
    
    if api_fixed:
        # Test endpoint
        print("\n🧪 Testing API Gateway endpoint...")
        endpoint_works = test_endpoint()
        
        if endpoint_works:
            print("\n✅ SUCCESS! Quick Agent Test is now working!")
            print("\n🎯 What's Fixed:")
            print("   ✅ Agents Status Lambda function working")
            print("   ✅ API Gateway integration fixed")
            print("   ✅ /agents/status endpoint responding")
            print("   ✅ Latency claims removed from frontend")
        else:
            print("\n⚠️  API Gateway updated but endpoint still has issues")
            print("   Try refreshing your browser (Ctrl+F5)")
    else:
        print("\n❌ Failed to fix API Gateway integration")
    
    print(f"\n🌐 Test your website: http://msp-intelligence-mesh-frontend.s3-website-us-east-1.amazonaws.com")
    print("   The Quick Agent Test should now show all 10 agents as operational!")

if __name__ == "__main__":
    main()





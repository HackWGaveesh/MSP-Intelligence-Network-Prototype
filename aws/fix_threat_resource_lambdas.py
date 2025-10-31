#!/usr/bin/env python3
"""
Fix Threat Intelligence and Resource Allocation Lambda Functions
Updates the Lambda functions to work properly without DynamoDB float issues
"""

import boto3
import json
import tempfile
import os
import zipfile
from decimal import Decimal

def create_fixed_threat_intelligence_lambda():
    """Create fixed Threat Intelligence Lambda without DynamoDB float issues"""
    lambda_client = boto3.client('lambda')
    s3_client = boto3.client('s3')
    
    function_name = 'msp-intelligence-mesh-threat-intelligence'
    
    # Fixed Lambda handler code
    handler_code = '''
import json
from datetime import datetime
import random

def lambda_handler(event, context):
    """
    Fixed Threat Intelligence Lambda handler
    """
    try:
        # Parse request
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        text = body.get('text', '')
        
        # Simple threat classification logic
        threat_keywords = {
            'ransomware': ['ransomware', 'encrypt', 'encrypted', 'ransom', 'bitcoin', 'payment'],
            'phishing': ['phishing', 'email', 'suspicious', 'click', 'link', 'urgent'],
            'ddos': ['ddos', 'attack', 'overload', 'traffic', 'denial', 'service'],
            'malware': ['malware', 'virus', 'trojan', 'backdoor', 'infected'],
            'insider': ['insider', 'employee', 'unauthorized', 'access', 'privilege']
        }
        
        text_lower = text.lower()
        threat_scores = {}
        
        for threat_type, keywords in threat_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            threat_scores[threat_type] = score / len(keywords)
        
        # Find highest scoring threat
        if threat_scores:
            max_threat = max(threat_scores, key=threat_scores.get)
            confidence = threat_scores[max_threat]
        else:
            max_threat = 'unknown'
            confidence = 0.1
        
        # Determine severity
        if confidence > 0.7:
            severity = 'HIGH'
        elif confidence > 0.4:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'
        
        # Generate response
        result = {
            'threat_type': max_threat,
            'severity': severity,
            'confidence': round(confidence, 3),  # Round to avoid float issues
            'model_used': 'Keyword Analysis (AWS Lambda)',
            'indicators': [
                f'AI detected {max_threat} with {confidence:.1%} confidence',
                'Pattern analysis completed',
                'Real-time threat classification'
            ],
            'recommended_actions': [
                'Isolate affected systems',
                'Run full system scan',
                'Update security software'
            ],
            'agent': 'threat-intelligence',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(result)
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
                'agent': 'threat-intelligence',
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
        s3_key = f'lambda-deployments/threat-intelligence-fixed.zip'
        s3_client.upload_file(zip_path, 'msp-intelligence-mesh-backend', s3_key)
        
        # Get account ID for role ARN
        sts = boto3.client('sts')
        account_id = sts.get_caller_identity()['Account']
        role_arn = f'arn:aws:iam::{account_id}:role/msp-intelligence-mesh-lambda-execution-role'
        
        # Update Lambda function
        lambda_client.update_function_code(
            FunctionName=function_name,
            S3Bucket='msp-intelligence-mesh-backend',
            S3Key=s3_key
        )
        
        print(f"✅ Updated Threat Intelligence Lambda: {function_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating Threat Intelligence: {e}")
        return False
    finally:
        if os.path.exists(zip_path):
            os.unlink(zip_path)

def create_fixed_resource_allocation_lambda():
    """Create fixed Resource Allocation Lambda with proper logic"""
    lambda_client = boto3.client('lambda')
    s3_client = boto3.client('s3')
    
    function_name = 'msp-intelligence-mesh-resource-allocation'
    
    # Fixed Lambda handler code
    handler_code = '''
import json
from datetime import datetime
import random

def lambda_handler(event, context):
    """
    Fixed Resource Allocation Lambda handler
    """
    try:
        # Parse request
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        task_count = body.get('task_count', 10)
        technician_count = body.get('technician_count', 5)
        time_window_hours = body.get('time_window_hours', 8)
        priority_mode = body.get('priority_mode', 'balanced')
        
        # Resource allocation logic
        tasks_per_technician = task_count / technician_count if technician_count > 0 else 0
        hours_per_task = time_window_hours / task_count if task_count > 0 else 0
        
        # Calculate efficiency
        if tasks_per_technician <= 2:
            efficiency = 0.95
        elif tasks_per_technician <= 4:
            efficiency = 0.85
        else:
            efficiency = 0.70
        
        # Generate allocation plan
        allocation_plan = []
        for i in range(technician_count):
            tech_tasks = max(1, int(task_count / technician_count))
            if i < task_count % technician_count:
                tech_tasks += 1
            
            allocation_plan.append({
                'technician_id': f'TECH_{i+1:02d}',
                'assigned_tasks': tech_tasks,
                'estimated_hours': round(tech_tasks * hours_per_task, 1),
                'priority_level': 'High' if i < 2 else 'Normal'
            })
        
        # Calculate metrics
        total_estimated_hours = sum(tech['estimated_hours'] for tech in allocation_plan)
        utilization_rate = min(1.0, total_estimated_hours / time_window_hours) if time_window_hours > 0 else 0
        
        # Generate recommendations
        recommendations = []
        if utilization_rate > 0.9:
            recommendations.append("⚠️ High utilization - consider adding technicians")
        elif utilization_rate < 0.6:
            recommendations.append("💡 Low utilization - technicians can handle more tasks")
        
        if efficiency < 0.8:
            recommendations.append("🔧 Consider task prioritization to improve efficiency")
        
        # Generate response
        result = {
            'allocation_plan': allocation_plan,
            'metrics': {
                'total_tasks': task_count,
                'total_technicians': technician_count,
                'utilization_rate': round(utilization_rate, 3),
                'efficiency_score': round(efficiency, 3),
                'estimated_completion_hours': round(total_estimated_hours, 1)
            },
            'recommendations': recommendations,
            'priority_mode': priority_mode,
            'agent': 'resource-allocation',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            },
            'body': json.dumps(result)
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
                'agent': 'resource-allocation',
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
        s3_key = f'lambda-deployments/resource-allocation-fixed.zip'
        s3_client.upload_file(zip_path, 'msp-intelligence-mesh-backend', s3_key)
        
        # Update Lambda function
        lambda_client.update_function_code(
            FunctionName=function_name,
            S3Bucket='msp-intelligence-mesh-backend',
            S3Key=s3_key
        )
        
        print(f"✅ Updated Resource Allocation Lambda: {function_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating Resource Allocation: {e}")
        return False
    finally:
        if os.path.exists(zip_path):
            os.unlink(zip_path)

def main():
    print("🔧 Fixing Threat Intelligence and Resource Allocation")
    print("=" * 60)
    
    # Fix Threat Intelligence
    print("\n🛡️ Fixing Threat Intelligence...")
    threat_success = create_fixed_threat_intelligence_lambda()
    
    # Fix Resource Allocation
    print("\n📅 Fixing Resource Allocation...")
    resource_success = create_fixed_resource_allocation_lambda()
    
    # Test the fixes
    print("\n🧪 Testing fixes...")
    import requests
    
    # Test Threat Intelligence
    try:
        response = requests.post(
            'https://mojoawwjv2.execute-api.us-east-1.amazonaws.com/prod/threat-intelligence',
            json={'text': 'Ransomware attack detected encrypting files'},
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Threat Intelligence: {data.get('threat_type', 'N/A')} - {data.get('severity', 'N/A')} severity")
        else:
            print(f"⚠️ Threat Intelligence: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Threat Intelligence test failed: {e}")
    
    # Test Resource Allocation
    try:
        response = requests.post(
            'https://mojoawwjv2.execute-api.us-east-1.amazonaws.com/prod/resource',
            json={'task_count': 20, 'technician_count': 8, 'time_window_hours': 8},
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            metrics = data.get('metrics', {})
            print(f"✅ Resource Allocation: {metrics.get('utilization_rate', 0):.1%} utilization, {len(data.get('allocation_plan', []))} technicians")
        else:
            print(f"⚠️ Resource Allocation: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Resource Allocation test failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Fix Complete!")
    print(f"🛡️ Threat Intelligence: {'Fixed' if threat_success else 'Failed'}")
    print(f"📅 Resource Allocation: {'Fixed' if resource_success else 'Failed'}")
    print(f"\n🌐 Test your website now: http://msp-intelligence-mesh-frontend.s3-website-us-east-1.amazonaws.com")

if __name__ == "__main__":
    main()










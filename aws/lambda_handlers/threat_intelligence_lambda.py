"""
Lambda Handler for Threat Intelligence Agent
"""

import json
import os
import boto3
from datetime import datetime

# AWS clients
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')
secrets_client = boto3.client('secretsmanager')
kinesis = boto3.client('kinesis')

# Get configuration
PROJECT_NAME = os.environ.get('PROJECT_NAME', 'msp-intelligence-mesh')
SECRET_ARN = os.environ['SECRET_ARN']
AGENT_TABLE = f"{PROJECT_NAME}-agent-results"
THREAT_TABLE = f"{PROJECT_NAME}-threat-events"
KINESIS_STREAM = f"{PROJECT_NAME}-events"

def get_secrets():
    """Retrieve secrets from Secrets Manager"""
    response = secrets_client.get_secret_value(SecretId=SECRET_ARN)
    return json.loads(response['SecretString'])

def lambda_handler(event, context):
    """
    Handle threat intelligence analysis requests
    """
    try:
        # Parse input
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event
        
        text = body.get('text', '')
        
        # Import AI model (cached in /tmp for subsequent invocations)
        from transformers import pipeline
        
        # Use sentiment as proxy for threat detection (lightweight)
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Simple threat classification based on keywords + sentiment
        threat_keywords = {
            'ransomware': ['encrypt', 'ransom', 'bitcoin', 'payment', 'locked', 'files'],
            'phishing': ['verify', 'account', 'click', 'link', 'urgent', 'suspended'],
            'ddos': ['overwhelmed', 'traffic', 'attack', 'flooding', 'requests'],
            'malware': ['virus', 'trojan', 'infected', 'malicious', 'download'],
            'data_breach': ['breach', 'leaked', 'stolen', 'exposed', 'unauthorized']
        }
        
        text_lower = text.lower()
        threat_type = 'unknown'
        max_matches = 0
        
        for t_type, keywords in threat_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > max_matches:
                max_matches = matches
                threat_type = t_type
        
        # Determine severity
        if max_matches >= 3:
            severity = 'HIGH'
        elif max_matches >= 2:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'
        
        # Get sentiment confidence
        sentiment_result = classifier(text[:512])[0]  # Limit text length
        confidence = float(sentiment_result['score'])
        
        # Create result
        threat_id = f"threat_{int(datetime.utcnow().timestamp() * 1000)}"
        result = {
            'threat_id': threat_id,
            'threat_type': threat_type,
            'severity': severity,
            'confidence': confidence,
            'text_analyzed': text[:200],
            'detected_at': datetime.utcnow().isoformat(),
            'agent': 'threat-intelligence',
            'model': 'distilbert-hybrid'
        }
        
        # Store in DynamoDB
        threats_table = dynamodb.Table(THREAT_TABLE)
        threats_table.put_item(Item={
            'threat_id': threat_id,
            'detected_at': int(datetime.utcnow().timestamp()),
            'severity': severity,
            'threat_type': threat_type,
            'confidence': confidence,
            'data': json.dumps(result)
        })
        
        # Publish to Kinesis
        try:
            kinesis.put_record(
                StreamName=KINESIS_STREAM,
                Data=json.dumps({
                    'event_type': 'threat_detected',
                    'agent': 'threat-intelligence',
                    'data': result,
                    'timestamp': datetime.utcnow().isoformat()
                }),
                PartitionKey=threat_id
            )
        except Exception as e:
            print(f"Kinesis publish error (non-critical): {e}")
        
        # Return response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': '*'
            },
            'body': json.dumps(result)
        }
        
    except Exception as e:
        print(f"Error in threat intelligence lambda: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'agent': 'threat-intelligence'
            })
        }











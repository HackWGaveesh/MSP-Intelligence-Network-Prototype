#!/usr/bin/env python3
"""
Phase 4: Real-Time Services
Setup Kinesis Data Stream, EventBridge, SQS
"""

import boto3
import json
import time

# Load configuration
with open('aws_config.json', 'r') as f:
    config = json.load(f)

AWS_REGION = config['region']
PROJECT_NAME = config['project_name']

# Initialize clients
kinesis = boto3.client('kinesis', region_name=AWS_REGION)
events = boto3.client('events', region_name=AWS_REGION)
sqs = boto3.client('sqs', region_name=AWS_REGION)
sns = boto3.client('sns', region_name=AWS_REGION)

def print_step(message):
    """Print formatted step message"""
    print(f"\n{'='*60}")
    print(f"✓ {message}")
    print(f"{'='*60}")

def create_kinesis_stream():
    """Create Kinesis Data Stream"""
    print_step("Creating Kinesis Data Stream")
    
    stream_name = f"{PROJECT_NAME}-events"
    
    try:
        kinesis.create_stream(
            StreamName=stream_name,
            ShardCount=1  # Single shard for cost optimization
        )
        print(f"   ✓ Created stream: {stream_name}")
        
        # Wait for stream to become active
        print("   ⏳ Waiting for stream to become active...")
        waiter = kinesis.get_waiter('stream_exists')
        waiter.wait(StreamName=stream_name)
        
        print(f"   ✓ Stream is active")
        
    except kinesis.exceptions.ResourceInUseException:
        print(f"   ⚠ Stream already exists: {stream_name}")
    except Exception as e:
        print(f"   ⚠ Stream creation (may need time): {e}")
    
    # Get stream ARN
    try:
        response = kinesis.describe_stream(StreamName=stream_name)
        stream_arn = response['StreamDescription']['StreamARN']
        return stream_name, stream_arn
    except Exception as e:
        print(f"   ⚠ Could not get stream ARN: {e}")
        return stream_name, None

def create_sns_topic():
    """Create SNS topic for alerts"""
    print_step("Creating SNS Topic for Alerts")
    
    topic_name = f"{PROJECT_NAME}-alerts"
    
    try:
        response = sns.create_topic(Name=topic_name)
        topic_arn = response['TopicArn']
        print(f"   ✓ Created SNS topic: {topic_arn}")
        
        # Subscribe email (will need confirmation)
        try:
            sns.subscribe(
                TopicArn=topic_arn,
                Protocol='email',
                Endpoint='gaveeshags2004@gmail.com'
            )
            print(f"   ✓ Added email subscription (check email for confirmation)")
        except Exception as e:
            print(f"   ⚠ Email subscription: {e}")
        
        return topic_arn
        
    except Exception as e:
        print(f"   ⚠ SNS topic creation: {e}")
        return None

def create_sqs_queues():
    """Create SQS queues for async processing"""
    print_step("Creating SQS Queues")
    
    # Main queue
    queue_name = f"{PROJECT_NAME}-async-processing"
    dlq_name = f"{PROJECT_NAME}-async-dlq"
    
    queues = {}
    
    try:
        # Create DLQ first
        dlq_response = sqs.create_queue(
            QueueName=dlq_name,
            Attributes={
                'MessageRetentionPeriod': '1209600'  # 14 days
            }
        )
        dlq_url = dlq_response['QueueUrl']
        dlq_attrs = sqs.get_queue_attributes(
            QueueUrl=dlq_url,
            AttributeNames=['QueueArn']
        )
        dlq_arn = dlq_attrs['Attributes']['QueueArn']
        print(f"   ✓ Created DLQ: {dlq_name}")
        queues['dlq'] = {'url': dlq_url, 'arn': dlq_arn}
        
        # Create main queue with DLQ
        queue_response = sqs.create_queue(
            QueueName=queue_name,
            Attributes={
                'MessageRetentionPeriod': '345600',  # 4 days
                'VisibilityTimeout': '300',  # 5 minutes
                'RedrivePolicy': json.dumps({
                    'deadLetterTargetArn': dlq_arn,
                    'maxReceiveCount': '3'
                })
            }
        )
        queue_url = queue_response['QueueUrl']
        queue_attrs = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=['QueueArn']
        )
        queue_arn = queue_attrs['Attributes']['QueueArn']
        print(f"   ✓ Created queue: {queue_name}")
        queues['main'] = {'url': queue_url, 'arn': queue_arn}
        
    except sqs.exceptions.QueueNameExists:
        print(f"   ⚠ Queues already exist")
        # Get existing queue URLs
        try:
            queue_url = sqs.get_queue_url(QueueName=queue_name)['QueueUrl']
            dlq_url = sqs.get_queue_url(QueueName=dlq_name)['QueueUrl']
            queues['main'] = {'url': queue_url}
            queues['dlq'] = {'url': dlq_url}
        except:
            pass
    except Exception as e:
        print(f"   ⚠ SQS creation error: {e}")
    
    return queues

def create_eventbridge_rules(sns_topic_arn):
    """Create EventBridge rules for event-driven architecture"""
    print_step("Creating EventBridge Rules")
    
    rules_created = []
    
    # Rule 1: High severity threats
    rule_name = f"{PROJECT_NAME}-high-severity-threats"
    try:
        events.put_rule(
            Name=rule_name,
            Description='Trigger alert on high severity threats',
            EventPattern=json.dumps({
                "source": ["msp.intelligence"],
                "detail-type": ["Threat Detected"],
                "detail": {
                    "severity": ["HIGH", "CRITICAL"]
                }
            }),
            State='ENABLED'
        )
        print(f"   ✓ Created rule: {rule_name}")
        
        # Add SNS target if available
        if sns_topic_arn:
            try:
                events.put_targets(
                    Rule=rule_name,
                    Targets=[{
                        'Id': '1',
                        'Arn': sns_topic_arn
                    }]
                )
                print(f"      ✓ Added SNS target")
            except Exception as e:
                print(f"      ⚠ Target addition: {e}")
        
        rules_created.append(rule_name)
        
    except Exception as e:
        print(f"   ⚠ EventBridge rule creation: {e}")
    
    # Rule 2: Client health alerts
    rule_name2 = f"{PROJECT_NAME}-client-health-alerts"
    try:
        events.put_rule(
            Name=rule_name2,
            Description='Alert on high-risk clients',
            EventPattern=json.dumps({
                "source": ["msp.intelligence"],
                "detail-type": ["Client Health Update"],
                "detail": {
                    "risk_level": ["High", "Critical"]
                }
            }),
            State='ENABLED'
        )
        print(f"   ✓ Created rule: {rule_name2}")
        rules_created.append(rule_name2)
        
    except Exception as e:
        print(f"   ⚠ EventBridge rule 2: {e}")
    
    return rules_created

def enable_kinesis_firehose():
    """Note about Kinesis Firehose (optional)"""
    print_step("Kinesis Firehose (Optional)")
    print("   ℹ️ Kinesis Firehose skipped for cost optimization")
    print("   ℹ️ Direct Kinesis → Lambda → DynamoDB is more cost-effective")
    print("   ℹ️ Can be added later for S3 data lake integration")

def main():
    """Setup real-time services"""
    print_step("PHASE 4: REAL-TIME SERVICES SETUP")
    
    # Create Kinesis stream
    stream_name, stream_arn = create_kinesis_stream()
    
    # Create SNS topic
    topic_arn = create_sns_topic()
    
    # Create SQS queues
    queues = create_sqs_queues()
    
    # Create EventBridge rules
    rules = create_eventbridge_rules(topic_arn)
    
    # Note about Firehose
    enable_kinesis_firehose()
    
    print_step("PHASE 4 COMPLETE!")
    print(f"\n📊 Real-Time Services Summary:")
    print(f"   ✓ Kinesis Stream: {stream_name}")
    if stream_arn:
        print(f"   ✓ Stream ARN: {stream_arn}")
    if topic_arn:
        print(f"   ✓ SNS Topic: {topic_arn}")
    print(f"   ✓ SQS Queues: {len(queues)}")
    print(f"   ✓ EventBridge Rules: {len(rules)}")
    
    # Save configuration
    realtime_config = {
        'kinesis_stream': stream_name,
        'kinesis_stream_arn': stream_arn,
        'sns_topic_arn': topic_arn,
        'sqs_queues': queues,
        'eventbridge_rules': rules
    }
    
    with open('aws_realtime_config.json', 'w') as f:
        json.dump(realtime_config, f, indent=2)
    
    print("\n✓ Real-time configuration saved to: aws_realtime_config.json")
    print("\n💡 Integration Points:")
    print(f"   • Lambda functions can publish to: {stream_name}")
    print(f"   • EventBridge monitors threat/health events")
    print(f"   • SNS sends email alerts for critical events")
    print(f"   • SQS handles async long-running tasks")
    
    print("\n🎯 Ready for Phase 5: AI/ML Integration")

if __name__ == '__main__':
    main()











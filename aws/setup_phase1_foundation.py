#!/usr/bin/env python3
"""
Phase 1: AWS Foundation Setup
Creates IAM roles, S3 buckets, DynamoDB tables, Secrets Manager, CloudWatch
"""

import os
import boto3
import json
import time
from botocore.exceptions import ClientError

# AWS Configuration
AWS_REGION = 'us-east-1'
PROJECT_NAME = 'msp-intelligence-mesh'

# Initialize AWS clients
iam = boto3.client('iam', region_name=AWS_REGION)
s3 = boto3.client('s3', region_name=AWS_REGION)
dynamodb = boto3.client('dynamodb', region_name=AWS_REGION)
secrets = boto3.client('secretsmanager', region_name=AWS_REGION)
cloudwatch = boto3.client('cloudwatch', region_name=AWS_REGION)
logs = boto3.client('logs', region_name=AWS_REGION)
budgets = boto3.client('budgets', region_name=AWS_REGION)

def print_step(message):
    """Print formatted step message"""
    print(f"\n{'='*60}")
    print(f"✓ {message}")
    print(f"{'='*60}")

def create_iam_roles():
    """Create IAM roles for Lambda and API Gateway"""
    print_step("Creating IAM Roles")
    
    # Lambda Execution Role
    lambda_role_name = f"{PROJECT_NAME}-lambda-role"
    lambda_trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    
    try:
        lambda_role = iam.create_role(
            RoleName=lambda_role_name,
            AssumeRolePolicyDocument=json.dumps(lambda_trust_policy),
            Description="Execution role for MSP Intelligence Mesh Lambda functions"
        )
        print(f"   ✓ Created Lambda role: {lambda_role['Role']['Arn']}")
        
        # Attach policies to Lambda role
        policies = [
            'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
            'arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess',
            'arn:aws:iam::aws:policy/AmazonS3FullAccess',
            'arn:aws:iam::aws:policy/AmazonKinesisFullAccess',
            'arn:aws:iam::aws:policy/CloudWatchFullAccess',
            'arn:aws:iam::aws:policy/SecretsManagerReadWrite',
            'arn:aws:iam::aws:policy/ComprehendFullAccess',
            'arn:aws:iam::aws:policy/AmazonBedrockFullAccess',
            'arn:aws:iam::aws:policy/AWSXRayDaemonWriteAccess'
        ]
        
        for policy_arn in policies:
            try:
                iam.attach_role_policy(
                    RoleName=lambda_role_name,
                    PolicyArn=policy_arn
                )
                print(f"   ✓ Attached policy: {policy_arn.split('/')[-1]}")
            except ClientError as e:
                print(f"   ⚠ Policy already attached: {policy_arn.split('/')[-1]}")
        
        # Wait for role to be available
        time.sleep(10)
        
        return lambda_role['Role']['Arn']
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            print(f"   ⚠ Role already exists: {lambda_role_name}")
            role = iam.get_role(RoleName=lambda_role_name)
            return role['Role']['Arn']
        else:
            raise

def create_s3_buckets():
    """Create S3 buckets for models, data, and frontend"""
    print_step("Creating S3 Buckets")
    
    buckets = [
        f"{PROJECT_NAME}-models",
        f"{PROJECT_NAME}-data",
        f"{PROJECT_NAME}-frontend"
    ]
    
    bucket_arns = []
    
    for bucket_name in buckets:
        try:
            # Create bucket
            if AWS_REGION == 'us-east-1':
                s3.create_bucket(Bucket=bucket_name)
            else:
                s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
                )
            print(f"   ✓ Created bucket: {bucket_name}")
            
            # Enable versioning for models bucket
            if 'models' in bucket_name:
                s3.put_bucket_versioning(
                    Bucket=bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
                print(f"   ✓ Enabled versioning: {bucket_name}")
            
            # Enable encryption
            s3.put_bucket_encryption(
                Bucket=bucket_name,
                ServerSideEncryptionConfiguration={
                    'Rules': [{
                        'ApplyServerSideEncryptionByDefault': {
                            'SSEAlgorithm': 'AES256'
                        }
                    }]
                }
            )
            print(f"   ✓ Enabled encryption: {bucket_name}")
            
            # Configure frontend bucket for static website hosting
            if 'frontend' in bucket_name:
                s3.put_bucket_website(
                    Bucket=bucket_name,
                    WebsiteConfiguration={
                        'IndexDocument': {'Suffix': 'index.html'},
                        'ErrorDocument': {'Key': 'index.html'}
                    }
                )
                
                # Make frontend bucket public
                bucket_policy = {
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Sid": "PublicReadGetObject",
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{bucket_name}/*"
                    }]
                }
                s3.put_bucket_policy(
                    Bucket=bucket_name,
                    Policy=json.dumps(bucket_policy)
                )
                print(f"   ✓ Configured static website hosting: {bucket_name}")
            
            bucket_arns.append(f"arn:aws:s3:::{bucket_name}")
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                print(f"   ⚠ Bucket already exists: {bucket_name}")
                bucket_arns.append(f"arn:aws:s3:::{bucket_name}")
            else:
                print(f"   ✗ Error creating bucket {bucket_name}: {e}")
    
    return bucket_arns

def create_dynamodb_tables():
    """Create DynamoDB tables for agent state and results"""
    print_step("Creating DynamoDB Tables")
    
    tables = [
        {
            'TableName': f"{PROJECT_NAME}-agent-state",
            'KeySchema': [
                {'AttributeName': 'agent_id', 'KeyType': 'HASH'},
                {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
            ],
            'AttributeDefinitions': [
                {'AttributeName': 'agent_id', 'AttributeType': 'S'},
                {'AttributeName': 'timestamp', 'AttributeType': 'N'}
            ],
            'BillingMode': 'PAY_PER_REQUEST',
            'StreamSpecification': {
                'StreamEnabled': True,
                'StreamViewType': 'NEW_AND_OLD_IMAGES'
            }
        },
        {
            'TableName': f"{PROJECT_NAME}-agent-results",
            'KeySchema': [
                {'AttributeName': 'result_id', 'KeyType': 'HASH'},
                {'AttributeName': 'agent_type', 'KeyType': 'RANGE'}
            ],
            'AttributeDefinitions': [
                {'AttributeName': 'result_id', 'AttributeType': 'S'},
                {'AttributeName': 'agent_type', 'AttributeType': 'S'}
            ],
            'BillingMode': 'PAY_PER_REQUEST'
        },
        {
            'TableName': f"{PROJECT_NAME}-threat-events",
            'KeySchema': [
                {'AttributeName': 'threat_id', 'KeyType': 'HASH'},
                {'AttributeName': 'detected_at', 'KeyType': 'RANGE'}
            ],
            'AttributeDefinitions': [
                {'AttributeName': 'threat_id', 'AttributeType': 'S'},
                {'AttributeName': 'detected_at', 'AttributeType': 'N'},
                {'AttributeName': 'severity', 'AttributeType': 'S'}
            ],
            'BillingMode': 'PAY_PER_REQUEST',
            'GlobalSecondaryIndexes': [{
                'IndexName': 'severity-index',
                'KeySchema': [
                    {'AttributeName': 'severity', 'KeyType': 'HASH'},
                    {'AttributeName': 'detected_at', 'KeyType': 'RANGE'}
                ],
                'Projection': {'ProjectionType': 'ALL'}
            }]
        },
        {
            'TableName': f"{PROJECT_NAME}-websocket-connections",
            'KeySchema': [
                {'AttributeName': 'connectionId', 'KeyType': 'HASH'}
            ],
            'AttributeDefinitions': [
                {'AttributeName': 'connectionId', 'AttributeType': 'S'}
            ],
            'BillingMode': 'PAY_PER_REQUEST'
        }
    ]
    
    table_arns = []
    
    for table_config in tables:
        try:
            response = dynamodb.create_table(**table_config)
            print(f"   ✓ Created table: {table_config['TableName']}")
            table_arns.append(response['TableDescription']['TableArn'])
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceInUseException':
                print(f"   ⚠ Table already exists: {table_config['TableName']}")
                response = dynamodb.describe_table(TableName=table_config['TableName'])
                table_arns.append(response['Table']['TableArn'])
            else:
                print(f"   ✗ Error creating table: {e}")
    
    # Wait for tables to be active
    print("   ⏳ Waiting for tables to become active...")
    time.sleep(15)
    
    return table_arns

def create_secrets():
    """Create Secrets Manager secret for API keys"""
    print_step("Creating Secrets Manager Secret")
    
    secret_name = f"{PROJECT_NAME}-credentials"
    
    # Read credentials from environment. Provide placeholders if not set.
    secret_value = {
        "huggingface_api_key": os.getenv("HUGGINGFACE_API_KEY", "SET_HUGGINGFACE_API_KEY"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY", "SET_GEMINI_API_KEY"),
        "grok_api_key": os.getenv("GROK_API_KEY", "SET_GROK_API_KEY"),
        "mongodb_url": os.getenv("MONGODB_URL", "mongodb+srv://username:password@cluster.mongodb.net/msp_network"),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY", "SET_PINECONE_API_KEY"),
        "redis_url": os.getenv("REDIS_URL", "https://your-redis-endpoint"),
        "redis_token": os.getenv("REDIS_TOKEN", "SET_REDIS_TOKEN")
    }
    
    try:
        response = secrets.create_secret(
            Name=secret_name,
            Description="API keys and credentials for MSP Intelligence Mesh",
            SecretString=json.dumps(secret_value)
        )
        print(f"   ✓ Created secret: {secret_name}")
        print(f"   ✓ Secret ARN: {response['ARN']}")
        return response['ARN']
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceExistsException':
            print(f"   ⚠ Secret already exists: {secret_name}")
            response = secrets.describe_secret(SecretId=secret_name)
            return response['ARN']
        else:
            raise

def create_cloudwatch_resources():
    """Create CloudWatch log groups and dashboard"""
    print_step("Creating CloudWatch Resources")
    
    # Create log groups for each agent
    agents = [
        'threat-intelligence',
        'market-intelligence',
        'nlp-query',
        'collaboration',
        'client-health',
        'revenue-optimization',
        'anomaly-detection',
        'security-compliance',
        'resource-allocation',
        'federated-learning',
        'websocket-handler'
    ]
    
    for agent in agents:
        log_group_name = f"/aws/lambda/{PROJECT_NAME}-{agent}"
        try:
            logs.create_log_group(logGroupName=log_group_name)
            logs.put_retention_policy(
                logGroupName=log_group_name,
                retentionInDays=7  # Cost optimization
            )
            print(f"   ✓ Created log group: {log_group_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                print(f"   ⚠ Log group already exists: {log_group_name}")
            else:
                print(f"   ✗ Error creating log group: {e}")
    
    print("   ✓ CloudWatch log groups created with 7-day retention")

def setup_cost_budget():
    """Set up AWS Budget with $100 limit"""
    print_step("Setting Up Cost Budget")
    
    try:
        # Get AWS account ID
        sts = boto3.client('sts')
        account_id = sts.get_caller_identity()['Account']
        
        budgets.create_budget(
            AccountId=account_id,
            Budget={
                'BudgetName': f"{PROJECT_NAME}-monthly-budget",
                'BudgetLimit': {
                    'Amount': '100',
                    'Unit': 'USD'
                },
                'TimeUnit': 'MONTHLY',
                'BudgetType': 'COST'
            },
            NotificationsWithSubscribers=[{
                'Notification': {
                    'NotificationType': 'ACTUAL',
                    'ComparisonOperator': 'GREATER_THAN',
                    'Threshold': 80,
                    'ThresholdType': 'PERCENTAGE'
                },
                'Subscribers': [{
                    'SubscriptionType': 'EMAIL',
                    'Address': 'gaveeshags2004@gmail.com'
                }]
            }]
        )
        print(f"   ✓ Created budget: $100/month with 80% alert")
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'DuplicateRecordException':
            print(f"   ⚠ Budget already exists")
        else:
            print(f"   ⚠ Could not create budget (may need permissions): {e}")

def main():
    """Run Phase 1 setup"""
    print("\n" + "="*60)
    print("🚀 PHASE 1: AWS FOUNDATION SETUP")
    print("="*60)
    
    try:
        # Step 1: IAM Roles
        lambda_role_arn = create_iam_roles()
        
        # Step 2: S3 Buckets
        bucket_arns = create_s3_buckets()
        
        # Step 3: DynamoDB Tables
        table_arns = create_dynamodb_tables()
        
        # Step 4: Secrets Manager
        secret_arn = create_secrets()
        
        # Step 5: CloudWatch
        create_cloudwatch_resources()
        
        # Step 6: Cost Budget
        setup_cost_budget()
        
        # Summary
        print_step("PHASE 1 COMPLETE!")
        print("\n📊 Created Resources:")
        print(f"   ✓ IAM Role: {lambda_role_arn}")
        print(f"   ✓ S3 Buckets: {len(bucket_arns)}")
        print(f"   ✓ DynamoDB Tables: {len(table_arns)}")
        print(f"   ✓ Secrets Manager: {secret_arn}")
        print(f"   ✓ CloudWatch Log Groups: 11")
        print(f"   ✓ Budget: $100/month")
        
        # Save configuration
        config = {
            'region': AWS_REGION,
            'project_name': PROJECT_NAME,
            'lambda_role_arn': lambda_role_arn,
            'buckets': bucket_arns,
            'tables': table_arns,
            'secret_arn': secret_arn
        }
        
        with open('aws_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\n✓ Configuration saved to: aws_config.json")
        print("\n🎯 Ready for Phase 2: Lambda Functions")
        
    except Exception as e:
        print(f"\n✗ Error during setup: {e}")
        raise

if __name__ == '__main__':
    main()











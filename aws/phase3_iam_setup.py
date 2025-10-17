#!/usr/bin/env python3
"""
Phase 3: Set up IAM Roles and Permissions
Creates proper IAM roles for Lambda functions with necessary permissions
"""

import boto3
import json

def create_lambda_execution_role():
    """Create IAM role for Lambda execution"""
    iam = boto3.client('iam')
    
    role_name = 'msp-intelligence-mesh-lambda-execution-role'
    
    # Trust policy for Lambda
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "lambda.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    # Permissions policy
    permissions_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "arn:aws:logs:*:*:*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject"
                ],
                "Resource": [
                    "arn:aws:s3:::msp-intelligence-mesh-backend/*",
                    "arn:aws:s3:::msp-intelligence-mesh-frontend/*",
                    "arn:aws:s3:::msp-intelligence-mesh-models/*",
                    "arn:aws:s3:::msp-intelligence-mesh-data/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:DeleteItem",
                    "dynamodb:Query",
                    "dynamodb:Scan"
                ],
                "Resource": [
                    "arn:aws:dynamodb:us-east-1:*:table/msp-intelligence-mesh-*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "secretsmanager:GetSecretValue"
                ],
                "Resource": [
                    "arn:aws:secretsmanager:us-east-1:*:secret:msp-intelligence-mesh-*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "comprehend:DetectSentiment",
                    "comprehend:DetectEntities",
                    "comprehend:DetectKeyPhrases"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords"
                ],
                "Resource": "*"
            }
        ]
    }
    
    try:
        # Create role
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Execution role for MSP Intelligence Mesh Lambda functions'
        )
        print(f"✅ Created IAM role: {role_name}")
        
        # Attach basic Lambda execution policy
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
        )
        print(f"✅ Attached basic execution policy")
        
        # Create and attach custom policy
        policy_name = f'{role_name}-policy'
        iam.create_policy(
            PolicyName=policy_name,
            PolicyDocument=json.dumps(permissions_policy),
            Description='Custom permissions for MSP Intelligence Mesh'
        )
        print(f"✅ Created custom policy: {policy_name}")
        
        # Get account ID
        sts = boto3.client('sts')
        account_id = sts.get_caller_identity()['Account']
        
        # Attach custom policy
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn=f'arn:aws:iam::{account_id}:policy/{policy_name}'
        )
        print(f"✅ Attached custom policy")
        
        # Return role ARN
        role_arn = f'arn:aws:iam::{account_id}:role/{role_name}'
        return role_arn
        
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"✅ IAM role already exists: {role_name}")
        # Get account ID and return existing role ARN
        sts = boto3.client('sts')
        account_id = sts.get_caller_identity()['Account']
        return f'arn:aws:iam::{account_id}:role/{role_name}'
    except Exception as e:
        print(f"❌ Error creating IAM role: {e}")
        return None

def update_existing_lambdas():
    """Update existing Lambda functions with new role"""
    lambda_client = boto3.client('lambda')
    
    # Get account ID
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    role_arn = f'arn:aws:iam::{account_id}:role/msp-intelligence-mesh-lambda-execution-role'
    
    # List of existing Lambda functions
    existing_functions = [
        'msp-intelligence-mesh-threat-intelligence',
        'msp-intelligence-mesh-market-intelligence',
        'msp-intelligence-mesh-nlp-query',
        'msp-intelligence-mesh-collaboration',
        'msp-intelligence-mesh-client-health',
        'msp-intelligence-mesh-revenue-optimization',
        'msp-intelligence-mesh-anomaly-detection',
        'msp-intelligence-mesh-security-compliance',
        'msp-intelligence-mesh-resource-allocation',
        'msp-intelligence-mesh-federated-learning'
    ]
    
    updated_count = 0
    
    for function_name in existing_functions:
        try:
            # Update function configuration
            lambda_client.update_function_configuration(
                FunctionName=function_name,
                Role=role_arn,
                Timeout=60,
                MemorySize=1024,
                Environment={
                    'Variables': {
                        'PROJECT_NAME': 'msp-intelligence-mesh',
                        'AWS_REGION': 'us-east-1'
                    }
                }
            )
            print(f"✅ Updated: {function_name}")
            updated_count += 1
            
        except lambda_client.exceptions.ResourceNotFoundException:
            print(f"⚠️  Function not found: {function_name}")
        except Exception as e:
            print(f"❌ Error updating {function_name}: {e}")
    
    return updated_count, role_arn

def create_api_gateway_role():
    """Create IAM role for API Gateway"""
    iam = boto3.client('iam')
    
    role_name = 'msp-intelligence-mesh-api-gateway-role'
    
    # Trust policy for API Gateway
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "apigateway.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        # Create role
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Role for API Gateway to invoke Lambda functions'
        )
        print(f"✅ Created API Gateway role: {role_name}")
        
        # Attach policy for Lambda invocation
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaRole'
        )
        print(f"✅ Attached Lambda invocation policy")
        
        # Get account ID and return role ARN
        sts = boto3.client('sts')
        account_id = sts.get_caller_identity()['Account']
        return f'arn:aws:iam::{account_id}:role/{role_name}'
        
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"✅ API Gateway role already exists: {role_name}")
        sts = boto3.client('sts')
        account_id = sts.get_caller_identity()['Account']
        return f'arn:aws:iam::{account_id}:role/{role_name}'
    except Exception as e:
        print(f"❌ Error creating API Gateway role: {e}")
        return None

def main():
    print("🚀 Phase 3: Setting up IAM Roles and Permissions")
    print("=" * 50)
    
    # Create Lambda execution role
    print("\n🔐 Creating Lambda execution role...")
    lambda_role_arn = create_lambda_execution_role()
    
    # Create API Gateway role
    print("\n🌐 Creating API Gateway role...")
    api_role_arn = create_api_gateway_role()
    
    # Update existing Lambda functions
    print("\n🔄 Updating existing Lambda functions...")
    updated_count, role_arn = update_existing_lambdas()
    
    print("\n" + "=" * 50)
    print("✅ Phase 3 Complete!")
    print(f"🔐 Lambda Role: {lambda_role_arn}")
    print(f"🌐 API Gateway Role: {api_role_arn}")
    print(f"🔄 Updated Lambda Functions: {updated_count}")
    
    print(f"\n🌐 IAM Console: https://console.aws.amazon.com/iam/home?region=us-east-1#/roles")
    print(f"📦 Lambda Console: https://console.aws.amazon.com/lambda/home?region=us-east-1")

if __name__ == "__main__":
    main()





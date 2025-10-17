#!/usr/bin/env python3
"""
Phase 1: Upload Backend Code to S3
Creates S3 bucket and uploads all backend code for Lambda deployment
"""

import boto3
import os
import zipfile
import tempfile
from pathlib import Path

def create_s3_bucket():
    """Create S3 bucket for backend code storage"""
    s3 = boto3.client('s3')
    bucket_name = 'msp-intelligence-mesh-backend'
    
    try:
        # Try to create bucket
        s3.create_bucket(Bucket=bucket_name)
        print(f"✅ Created S3 bucket: {bucket_name}")
    except s3.exceptions.BucketAlreadyExists:
        print(f"✅ S3 bucket already exists: {bucket_name}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"✅ S3 bucket already owned by you: {bucket_name}")
    except Exception as e:
        print(f"❌ Error creating bucket: {e}")
        return None
    
    return bucket_name

def upload_backend_code(bucket_name):
    """Upload all backend code to S3"""
    s3 = boto3.client('s3')
    
    # Backend directories to upload
    backend_dirs = [
        'backend/api',
        'backend/agents', 
        'backend/utils',
        'backend/models',
        'backend/services'
    ]
    
    # Files to upload
    backend_files = [
        'requirements.txt',
        'main_simple.py',
        'main.py'
    ]
    
    uploaded_count = 0
    
    # Upload directories
    for dir_path in backend_dirs:
        if os.path.exists(dir_path):
            print(f"📁 Uploading directory: {dir_path}")
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(('.py', '.txt', '.json', '.md')):
                        local_path = os.path.join(root, file)
                        s3_key = local_path.replace('\\', '/')
                        
                        try:
                            s3.upload_file(local_path, bucket_name, s3_key)
                            print(f"  ✅ {s3_key}")
                            uploaded_count += 1
                        except Exception as e:
                            print(f"  ❌ Error uploading {s3_key}: {e}")
    
    # Upload individual files
    for file_path in backend_files:
        if os.path.exists(file_path):
            print(f"📄 Uploading file: {file_path}")
            try:
                s3.upload_file(file_path, bucket_name, file_path)
                print(f"  ✅ {file_path}")
                uploaded_count += 1
            except Exception as e:
                print(f"  ❌ Error uploading {file_path}: {e}")
    
    return uploaded_count

def create_lambda_packages(bucket_name):
    """Create Lambda deployment packages"""
    s3 = boto3.client('s3')
    
    # Create packages for each agent
    agents = [
        'threat_intelligence',
        'market_intelligence', 
        'nlp_query',
        'client_health',
        'revenue_optimization',
        'anomaly_detection',
        'collaboration',
        'compliance',
        'resource_allocation',
        'federated_learning'
    ]
    
    packages_created = 0
    
    for agent in agents:
        print(f"📦 Creating package for: {agent}")
        
        # Create temporary zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            zip_path = tmp_file.name
        
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add main Lambda handler
                if os.path.exists(f'backend/agents/{agent}.py'):
                    zipf.write(f'backend/agents/{agent}.py', f'{agent}.py')
                
                # Add common utilities
                if os.path.exists('backend/utils'):
                    for root, dirs, files in os.walk('backend/utils'):
                        for file in files:
                            if file.endswith('.py'):
                                local_path = os.path.join(root, file)
                                arc_path = f'utils/{file}'
                                zipf.write(local_path, arc_path)
                
                # Add requirements if exists
                if os.path.exists('requirements.txt'):
                    zipf.write('requirements.txt', 'requirements.txt')
            
            # Upload package to S3
            s3_key = f'lambda-packages/{agent}.zip'
            s3.upload_file(zip_path, bucket_name, s3_key)
            print(f"  ✅ Package uploaded: {s3_key}")
            packages_created += 1
            
        except Exception as e:
            print(f"  ❌ Error creating package for {agent}: {e}")
        finally:
            # Clean up temp file
            if os.path.exists(zip_path):
                os.unlink(zip_path)
    
    return packages_created

def set_bucket_policies(bucket_name):
    """Set up bucket policies and versioning"""
    s3 = boto3.client('s3')
    
    try:
        # Enable versioning
        s3.put_bucket_versioning(
            Bucket=bucket_name,
            VersioningConfiguration={'Status': 'Enabled'}
        )
        print(f"✅ Enabled versioning for {bucket_name}")
        
        # Set lifecycle policy
        lifecycle_config = {
            'Rules': [
                {
                    'ID': 'DeleteOldVersions',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': ''},
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        }
                    ],
                    'NoncurrentVersionTransitions': [
                        {
                            'NoncurrentDays': 7,
                            'StorageClass': 'GLACIER'
                        }
                    ]
                }
            ]
        }
        
        s3.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=lifecycle_config
        )
        print(f"✅ Set lifecycle policy for {bucket_name}")
        
    except Exception as e:
        print(f"❌ Error setting bucket policies: {e}")

def main():
    print("🚀 Phase 1: Uploading Backend Code to S3")
    print("=" * 50)
    
    # Create S3 bucket
    bucket_name = create_s3_bucket()
    if not bucket_name:
        print("❌ Failed to create S3 bucket")
        return
    
    # Upload backend code
    print("\n📤 Uploading backend code...")
    uploaded_count = upload_backend_code(bucket_name)
    print(f"✅ Uploaded {uploaded_count} files")
    
    # Create Lambda packages
    print("\n📦 Creating Lambda packages...")
    packages_created = create_lambda_packages(bucket_name)
    print(f"✅ Created {packages_created} Lambda packages")
    
    # Set bucket policies
    print("\n🔧 Setting bucket policies...")
    set_bucket_policies(bucket_name)
    
    print("\n" + "=" * 50)
    print("✅ Phase 1 Complete!")
    print(f"📦 S3 Bucket: {bucket_name}")
    print(f"📁 Files uploaded: {uploaded_count}")
    print(f"📦 Lambda packages: {packages_created}")
    print(f"🌐 S3 URL: https://s3.console.aws.amazon.com/s3/buckets/{bucket_name}")

if __name__ == "__main__":
    main()





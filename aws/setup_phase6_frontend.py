#!/usr/bin/env python3
"""
Phase 6: Frontend Deployment
Upload frontend to S3 and configure for AWS API Gateway
"""

import boto3
import json
import os
import mimetypes
from pathlib import Path

# Load configuration
with open('aws_config.json', 'r') as f:
    config = json.load(f)

with open('aws_api_config.json', 'r') as f:
    api_config = json.load(f)

AWS_REGION = config['region']
PROJECT_NAME = config['project_name']
API_URL = api_config['invoke_url']

# Initialize clients
s3 = boto3.client('s3', region_name=AWS_REGION)

def print_step(message):
    """Print formatted step message"""
    print(f"\n{'='*60}")
    print(f"✓ {message}")
    print(f"{'='*60}")

def update_frontend_api_urls():
    """Update frontend files with AWS API Gateway URLs"""
    print_step("Updating Frontend with AWS API URLs")
    
    # Files to update
    frontend_files = [
        'frontend/index.html',
        'frontend/threat-intelligence.html',
        'frontend/market-intelligence.html',
        'frontend/client-health.html',
        'frontend/revenue-optimization.html',
        'frontend/anomaly-detection.html',
        'frontend/nlp-query.html',
        'frontend/collaboration.html',
        'frontend/security-compliance.html',
        'frontend/resource-allocation.html',
        'frontend/federated-learning.html',
        'frontend/workflow-demo.html'
    ]
    
    updated_count = 0
    
    for file_path in frontend_files:
        if not os.path.exists(file_path):
            continue
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Replace local API URL with AWS API Gateway URL
            original_content = content
            content = content.replace('http://localhost:8000', API_URL)
            content = content.replace('http://127.0.0.1:8000', API_URL)
            
            if content != original_content:
                # Create backup
                backup_path = f"{file_path}.local_backup"
                with open(backup_path, 'w') as f:
                    f.write(original_content)
                
                # Write updated content
                with open(file_path, 'w') as f:
                    f.write(content)
                
                print(f"   ✓ Updated: {file_path}")
                updated_count += 1
        
        except Exception as e:
            print(f"   ⚠ Error updating {file_path}: {e}")
    
    print(f"\n   ✓ Updated {updated_count} files with API URL: {API_URL}")
    return updated_count

def upload_frontend_to_s3():
    """Upload all frontend files to S3"""
    print_step("Uploading Frontend to S3")
    
    bucket_name = f"{PROJECT_NAME}-frontend"
    frontend_dir = 'frontend'
    
    uploaded_files = []
    
    # Get all files in frontend directory
    for root, dirs, files in os.walk(frontend_dir):
        for file in files:
            # Skip backup files, node_modules, src
            if (file.endswith('.local_backup') or 
                'node_modules' in root or 
                '/src/' in root or
                file == 'package.json' or
                file == 'package_minimal.json'):
                continue
            
            local_path = os.path.join(root, file)
            # S3 key should not include 'frontend/' prefix
            s3_key = os.path.relpath(local_path, frontend_dir)
            
            # Determine content type
            content_type, _ = mimetypes.guess_type(local_path)
            if content_type is None:
                content_type = 'application/octet-stream'
            
            try:
                # Upload file
                with open(local_path, 'rb') as f:
                    s3.put_object(
                        Bucket=bucket_name,
                        Key=s3_key,
                        Body=f.read(),
                        ContentType=content_type
                    )
                
                uploaded_files.append(s3_key)
                print(f"   ✓ Uploaded: {s3_key}")
                
            except Exception as e:
                print(f"   ✗ Error uploading {local_path}: {e}")
    
    print(f"\n   ✓ Uploaded {len(uploaded_files)} files to s3://{bucket_name}")
    return len(uploaded_files), bucket_name

def configure_s3_website():
    """Configure S3 bucket for static website hosting"""
    print_step("Configuring S3 Static Website")
    
    bucket_name = f"{PROJECT_NAME}-frontend"
    
    try:
        # Enable static website hosting
        s3.put_bucket_website(
            Bucket=bucket_name,
            WebsiteConfiguration={
                'IndexDocument': {'Suffix': 'index.html'},
                'ErrorDocument': {'Key': 'index.html'}
            }
        )
        print(f"   ✓ Configured static website hosting")
        
        # Make bucket public (if not already)
        try:
            # Remove public access block
            s3.delete_public_access_block(Bucket=bucket_name)
            
            # Set bucket policy for public read
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
            print(f"   ✓ Set public read access")
            
        except Exception as e:
            print(f"   ⚠ Public access configuration: {e}")
        
        # Get website URL
        website_url = f"http://{bucket_name}.s3-website-{AWS_REGION}.amazonaws.com"
        print(f"\n   🌐 Website URL: {website_url}")
        
        return website_url
        
    except Exception as e:
        print(f"   ✗ Website configuration error: {e}")
        return None

def create_cloudfront_distribution():
    """Create CloudFront distribution (optional for time)"""
    print_step("CloudFront CDN (Optional)")
    
    print("   ℹ️ CloudFront distribution skipped for time optimization")
    print("   ℹ️ S3 static website is sufficient for demo")
    print("   ℹ️ Can be added later for:")
    print("      - HTTPS support")
    print("      - Custom domain")
    print("      - Global CDN caching")
    print("      - Better performance")
    
    return None

def main():
    """Deploy frontend"""
    print_step("PHASE 6: FRONTEND DEPLOYMENT")
    
    # Update API URLs in frontend
    updated_files = update_frontend_api_urls()
    
    # Upload to S3
    file_count, bucket_name = upload_frontend_to_s3()
    
    # Configure website hosting
    website_url = configure_s3_website()
    
    # CloudFront (optional)
    cdn_url = create_cloudfront_distribution()
    
    print_step("PHASE 6 COMPLETE!")
    print(f"\n📊 Frontend Deployment Summary:")
    print(f"   ✓ Files Updated: {updated_files}")
    print(f"   ✓ Files Uploaded: {file_count}")
    print(f"   ✓ S3 Bucket: {bucket_name}")
    if website_url:
        print(f"   🌐 Website URL: {website_url}")
    print(f"   ✓ API Gateway: {API_URL}")
    
    # Save configuration
    frontend_config = {
        'bucket_name': bucket_name,
        'website_url': website_url,
        'cdn_url': cdn_url,
        'api_url': API_URL,
        'files_deployed': file_count
    }
    
    with open('aws_frontend_config.json', 'w') as f:
        json.dump(frontend_config, f, indent=2)
    
    print("\n✓ Frontend configuration saved to: aws_frontend_config.json")
    
    print("\n📋 Access Your Application:")
    if website_url:
        print(f"\n   🌐 Frontend: {website_url}")
    print(f"   📡 API: {API_URL}")
    print(f"\n   📄 Pages Available:")
    print(f"      • {website_url}/index.html (Dashboard)")
    print(f"      • {website_url}/threat-intelligence.html")
    print(f"      • {website_url}/client-health.html")
    print(f"      • {website_url}/workflow-demo.html")
    print(f"      • ... and 8 more agent pages")
    
    print("\n🎯 Ready for Phase 7: Monitoring & Security")

if __name__ == '__main__':
    main()






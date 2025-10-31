#!/usr/bin/env python3
"""
Fix Frontend Endpoints
Updates all frontend files to use the correct API Gateway endpoints
"""

import os
import re

def fix_endpoints_in_file(file_path):
    """Fix endpoint calls in a single file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Define endpoint mappings
    endpoint_mappings = {
        '/market-intelligence/analyze': '/market-intelligence',
        '/threat-intelligence/analyze': '/threat-intelligence',
        '/nlp-query/ask': '/nlp-query',
        '/client-health/predict': '/client-health',
        '/revenue/forecast': '/revenue',
        '/anomaly/detect': '/anomaly',
        '/collaboration/match': '/collaboration',
        '/compliance/check': '/compliance',
        '/resource/allocate': '/resource',
        '/federated/train': '/federated',
        '/federated/status': '/federated'
    }
    
    # Apply mappings
    original_content = content
    for old_endpoint, new_endpoint in endpoint_mappings.items():
        content = content.replace(old_endpoint, new_endpoint)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    print("🔧 Fixing Frontend Endpoints")
    print("=" * 50)
    
    # List of frontend files to fix
    frontend_files = [
        'frontend/market-intelligence.html',
        'frontend/threat-intelligence.html',
        'frontend/nlp-query.html',
        'frontend/client-health.html',
        'frontend/revenue-optimization.html',
        'frontend/anomaly-detection.html',
        'frontend/collaboration.html',
        'frontend/security-compliance.html',
        'frontend/resource-allocation.html',
        'frontend/federated-learning.html',
        'frontend/workflow-demo.html'
    ]
    
    fixed_count = 0
    
    for file_path in frontend_files:
        if os.path.exists(file_path):
            print(f"📝 Fixing {file_path}...")
            if fix_endpoints_in_file(file_path):
                print(f"  ✅ Updated endpoints")
                fixed_count += 1
            else:
                print(f"  ℹ️  No changes needed")
        else:
            print(f"  ⚠️  File not found: {file_path}")
    
    print(f"\n✅ Fixed {fixed_count} files")
    
    # Upload to S3
    print("\n📤 Uploading fixed files to S3...")
    import boto3
    s3 = boto3.client('s3')
    
    uploaded_count = 0
    for file_path in frontend_files:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            with open(file_path, 'rb') as f:
                s3.put_object(
                    Bucket='msp-intelligence-mesh-frontend',
                    Key=filename,
                    Body=f.read(),
                    ContentType='text/html',
                    CacheControl='no-cache, max-age=0'
                )
            print(f"  ✅ Uploaded {filename}")
            uploaded_count += 1
    
    print(f"\n✅ Uploaded {uploaded_count} files to S3")
    print(f"\n🌐 Frontend URL: http://msp-intelligence-mesh-frontend.s3-website-us-east-1.amazonaws.com")
    print(f"🔄 Please refresh your browser (Ctrl+F5) to see the fixes")

if __name__ == "__main__":
    main()










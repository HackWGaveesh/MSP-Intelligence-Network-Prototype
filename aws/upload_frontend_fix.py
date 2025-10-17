#!/usr/bin/env python3
import boto3

s3 = boto3.client('s3')

# Upload app.js to S3
with open('frontend/app.js', 'rb') as f:
    s3.put_object(
        Bucket='msp-intelligence-mesh-frontend',
        Key='app.js',
        Body=f.read(),
        ContentType='application/javascript',
        CacheControl='no-cache, max-age=0'
    )

print('✅ Uploaded app.js to AWS S3')
print('🌐 AWS Frontend URL: http://msp-intelligence-mesh-frontend.s3-website-us-east-1.amazonaws.com')
print('🔄 Press Ctrl+F5 to refresh the page')
print('')
print('✅ Local Frontend: http://localhost:3000')
print('📊 Local Backend: http://localhost:8000')
print('')
print('🚀 Both systems updated!')






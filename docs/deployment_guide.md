# MSP Intelligence Mesh Network - Deployment Guide

## 🚀 Production Deployment Guide

This guide covers deploying the MSP Intelligence Mesh Network in production environments, including AWS cloud deployment, monitoring setup, and maintenance procedures.

## 📋 Prerequisites

### System Requirements
- **CPU**: 8+ cores (16+ recommended for production)
- **RAM**: 16GB+ (32GB+ recommended for production)
- **Storage**: 100GB+ SSD storage
- **Network**: High-speed internet connection
- **OS**: Linux (Ubuntu 20.04+ recommended)

### Software Requirements
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **AWS CLI**: 2.0+
- **kubectl**: 1.20+ (for Kubernetes deployment)
- **Helm**: 3.0+ (for Kubernetes deployment)

### AWS Account Setup
- AWS account with appropriate permissions
- IAM user with programmatic access
- S3 bucket for model storage
- VPC and security groups configured

## 🏗️ Deployment Options

### Option 1: Docker Compose (Recommended for Demo/Testing)

#### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd msp-intelligence-mesh

# Setup environment
cp env.example .env
# Edit .env with your configuration

# Start services
./start.sh

# Or manually:
docker-compose up --build -d

# Generate demo data
docker-compose exec backend python -m utils.data_generator

# Access application
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# Grafana: http://localhost:3001
```

#### Production Configuration
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  backend:
    image: msp-intelligence-mesh/backend:latest
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    networks:
      - msp-network
    restart: unless-stopped

  frontend:
    image: msp-intelligence-mesh/frontend:latest
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
    environment:
      - REACT_APP_API_URL=https://api.yourdomain.com
      - REACT_APP_WS_URL=wss://api.yourdomain.com/ws
    networks:
      - msp-network
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - backend
      - frontend
    networks:
      - msp-network
    restart: unless-stopped
```

### Option 2: Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster (EKS, GKE, or self-managed)
- Helm 3.0+
- kubectl configured

#### Deploy with Helm
```bash
# Add Helm repository
helm repo add msp-intelligence https://charts.msp-intelligence.com
helm repo update

# Install the application
helm install msp-intelligence-mesh msp-intelligence/msp-intelligence-mesh \
  --namespace msp-intelligence \
  --create-namespace \
  --set image.tag=latest \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=api.yourdomain.com \
  --set ingress.tls[0].secretName=msp-intelligence-tls

# Check deployment status
kubectl get pods -n msp-intelligence
kubectl get services -n msp-intelligence
```

#### Custom Values
```yaml
# values.yaml
replicaCount: 3

image:
  repository: msp-intelligence-mesh/backend
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: api.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: msp-intelligence-tls
      hosts:
        - api.yourdomain.com

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
```

### Option 3: AWS ECS Deployment

#### ECS Task Definition
```json
{
  "family": "msp-intelligence-mesh",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "account.dkr.ecr.region.amazonaws.com/msp-intelligence-mesh:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        },
        {
          "name": "AWS_REGION",
          "value": "us-east-1"
        }
      ],
      "secrets": [
        {
          "name": "MONGODB_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:msp-intelligence/mongodb"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/msp-intelligence-mesh",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "backend"
        }
      }
    }
  ]
}
```

#### ECS Service Configuration
```json
{
  "serviceName": "msp-intelligence-mesh",
  "cluster": "msp-intelligence-cluster",
  "taskDefinition": "msp-intelligence-mesh:1",
  "desiredCount": 3,
  "launchType": "FARGATE",
  "networkConfiguration": {
    "awsvpcConfiguration": {
      "subnets": ["subnet-12345", "subnet-67890"],
      "securityGroups": ["sg-12345"],
      "assignPublicIp": "ENABLED"
    }
  },
  "loadBalancers": [
    {
      "targetGroupArn": "arn:aws:elasticloadbalancing:region:account:targetgroup/msp-intelligence/12345",
      "containerName": "backend",
      "containerPort": 8000
    }
  ],
  "serviceRegistries": [
    {
      "registryArn": "arn:aws:servicediscovery:region:account:service/srv-12345"
    }
  ]
}
```

## 🔧 Configuration Management

### Environment Variables
```bash
# Core Configuration
DEBUG=False
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AWS Configuration
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
BUDGET_LIMIT=200

# Database Configuration
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/msp_network
PINECONE_API_KEY=your-pinecone-key
REDIS_URL=redis://redis:6379

# API Keys
HUGGINGFACE_API_KEY=your-huggingface-key
GEMINI_API_KEY=your-gemini-key
GROK_API_KEY=your-grok-key

# Performance Configuration
MAX_CONCURRENT_AGENTS=10
WEBSOCKET_HEARTBEAT_INTERVAL=30
REAL_TIME_UPDATE_INTERVAL=1000
```

### Secrets Management
```bash
# Using AWS Secrets Manager
aws secretsmanager create-secret \
  --name "msp-intelligence/database" \
  --description "Database credentials" \
  --secret-string '{"username":"admin","password":"secure-password"}'

# Using Kubernetes Secrets
kubectl create secret generic msp-intelligence-secrets \
  --from-literal=mongodb-url="mongodb+srv://..." \
  --from-literal=redis-url="redis://..." \
  --from-literal=secret-key="your-secret-key"
```

## 📊 Monitoring Setup

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "msp-intelligence-rules.yml"

scrape_configs:
  - job_name: 'msp-intelligence-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'msp-intelligence-frontend'
    static_configs:
      - targets: ['frontend:3000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'msp-intelligence-agents'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/agents/metrics'
    scrape_interval: 10s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "MSP Intelligence Mesh Network",
    "panels": [
      {
        "title": "System Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(up{job=~\"msp-intelligence-.*\"})",
            "legendFormat": "Services Up"
          }
        ]
      },
      {
        "title": "Agent Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(agent_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Times",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, agent_response_time_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

## 🔒 Security Configuration

### SSL/TLS Setup
```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /ws {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### Security Headers
```nginx
# Security headers
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'";
```

## 🚀 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy MSP Intelligence Mesh Network

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          cd backend
          pytest tests/ -v
      
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Install frontend dependencies
        run: |
          cd frontend
          npm ci
      
      - name: Run frontend tests
        run: |
          cd frontend
          npm test -- --coverage

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Build and push backend
        uses: docker/build-push-action@v3
        with:
          context: ./backend
          file: ./docker/Dockerfile.backend
          push: true
          tags: msp-intelligence-mesh/backend:latest
      
      - name: Build and push frontend
        uses: docker/build-push-action@v3
        with:
          context: ./frontend
          file: ./docker/Dockerfile.frontend
          push: true
          tags: msp-intelligence-mesh/frontend:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to AWS ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v1
        with:
          task-definition: task-definition.json
          service: msp-intelligence-mesh
          cluster: msp-intelligence-cluster
          wait-for-service-stability: true
```

## 🔧 Maintenance Procedures

### Health Checks
```bash
# Check service health
curl -f http://localhost:8000/health

# Check agent status
curl -f http://localhost:8000/agents/status

# Check database connectivity
docker-compose exec backend python -c "
from services.database_service import DatabaseService
db = DatabaseService()
print('Database connected:', db.is_connected())
"
```

### Backup Procedures
```bash
# Backup MongoDB
docker-compose exec mongodb mongodump --out /backup/$(date +%Y%m%d)

# Backup application data
docker-compose exec backend tar -czf /backup/app-data-$(date +%Y%m%d).tar.gz /app/data

# Backup configuration
docker-compose exec backend tar -czf /backup/config-$(date +%Y%m%d).tar.gz /app/config
```

### Update Procedures
```bash
# Update application
git pull origin main
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Update specific service
docker-compose pull backend
docker-compose up -d backend

# Rollback if needed
docker-compose down
docker-compose up -d --scale backend=0
docker-compose up -d backend:previous-version
```

## 📈 Performance Optimization

### Resource Tuning
```yaml
# docker-compose.override.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
    environment:
      - WORKERS=4
      - MAX_CONCURRENT_AGENTS=20
      - CACHE_SIZE=1000
```

### Database Optimization
```python
# Database connection pooling
DATABASE_CONFIG = {
    'max_pool_size': 20,
    'min_pool_size': 5,
    'max_idle_time': 300,
    'connect_timeout': 10,
    'server_selection_timeout': 5000
}
```

### Caching Strategy
```python
# Redis caching configuration
CACHE_CONFIG = {
    'default_ttl': 3600,  # 1 hour
    'max_connections': 100,
    'retry_on_timeout': True,
    'socket_keepalive': True,
    'socket_keepalive_options': {}
}
```

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker-compose logs backend
docker-compose logs frontend

# Check resource usage
docker stats

# Check port conflicts
netstat -tulpn | grep :8000
```

#### Database Connection Issues
```bash
# Test database connectivity
docker-compose exec backend python -c "
import pymongo
client = pymongo.MongoClient('mongodb://mongodb:27017')
print('Connected:', client.admin.command('ping'))
"
```

#### Performance Issues
```bash
# Check resource usage
docker stats

# Check slow queries
docker-compose exec mongodb mongosh --eval "db.setProfilingLevel(2, {slowms: 100})"

# Check agent performance
curl http://localhost:8000/agents/status | jq '.agents[].status'
```

### Log Analysis
```bash
# View real-time logs
docker-compose logs -f backend

# Filter error logs
docker-compose logs backend | grep ERROR

# Export logs for analysis
docker-compose logs backend > backend.log
```

## 📞 Support

### Getting Help
- **Documentation**: Check this guide and architecture docs
- **Issues**: Create GitHub issues for bugs and feature requests
- **Community**: Join our Discord server for community support
- **Enterprise**: Contact support@msp-intelligence.com for enterprise support

### Monitoring Alerts
- **Uptime**: Service availability monitoring
- **Performance**: Response time and throughput alerts
- **Errors**: Application error rate monitoring
- **Resources**: CPU, memory, and disk usage alerts

---

This deployment guide provides comprehensive instructions for deploying the MSP Intelligence Mesh Network in production environments. Follow the procedures carefully and monitor the system closely during initial deployment.

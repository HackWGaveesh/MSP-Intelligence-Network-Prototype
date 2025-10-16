# MSP Intelligence Mesh Network - Deployment Guide

## 🚀 Quick Start (Recommended)

### Option 1: Complete Automated Setup
```bash
# Clone and navigate to the project
cd msp-intelligence-mesh

# Run the complete startup script
./complete_start.sh
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements_minimal.txt

# Start backend
cd backend
python api/main_simple.py &

# Start frontend
cd ..
python serve_frontend.py &
```

## 🌐 Access URLs

- **Main Dashboard**: http://localhost:3001
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Agent Status**: http://localhost:8000/agents/status

## 🏗️ System Architecture

### Backend Components
- **FastAPI Application** with WebSocket support
- **10+ Specialized AI Agents** with full functionality
- **Federated Learning Engine** with privacy guarantees
- **Real-time Data Pipeline** with streaming analytics
- **Synthetic Data Generator** for realistic demonstrations

### Frontend Components
- **React 18 Dashboard** with TypeScript
- **Real-time Visualizations** using D3.js and Recharts
- **WebSocket Integration** for live updates
- **Professional UI/UX** with Tailwind CSS
- **Responsive Design** for all devices

### AI/ML Stack
- **Models**: DistilBERT, BERT, LightGBM, Prophet, Sentence-BERT
- **Frameworks**: TensorFlow, PyTorch, Scikit-learn, Transformers
- **Privacy**: Differential Privacy (ε=0.1), Homomorphic Encryption simulation
- **Real-time**: WebSocket streaming, live model inference

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# Database Configuration
MONGODB_URL=mongodb+srv://user:password@cluster.mongodb.net/msp_network
REDIS_URL=your_redis_url
REDIS_TOKEN=your_redis_token

# Vector Database
PINECONE_API_KEY=your_pinecone_key

# Security
SECRET_KEY=your_secret_key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Keys
HUGGINGFACE_API_KEY=your_hf_key
GEMINI_API_KEY=your_gemini_key
GROK_API_KEY=your_grok_key
```

### Backend Configuration
Edit `backend/config/settings.py`:

```python
class Settings:
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "MSP Intelligence Mesh Network"
    
    # Database Configuration
    MONGODB_URL: str = "mongodb+srv://user:password@cluster.mongodb.net/msp_network"
    REDIS_URL: str = "your_redis_url"
    
    # Security Configuration
    SECRET_KEY: str = "your_secret_key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID: str = "your_access_key"
    AWS_SECRET_ACCESS_KEY: str = "your_secret_key"
    AWS_REGION: str = "us-east-1"
    
    # Agent Configuration
    MAX_AGENTS: int = 20
    AGENT_TIMEOUT: int = 30
    FEDERATED_LEARNING_ROUNDS: int = 10
```

## 🐳 Docker Deployment

### Docker Compose Setup
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URL=${MONGODB_URL}
      - REDIS_URL=${REDIS_URL}
    depends_on:
      - mongodb
      - redis

  frontend:
    build: ./frontend
    ports:
      - "3001:3001"
    depends_on:
      - backend

  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  mongodb_data:
  redis_data:
```

### Docker Commands
```bash
# Build and start all services
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

## ☁️ AWS Deployment

### Prerequisites
- AWS CLI configured
- Terraform installed (optional)
- Docker installed

### AWS Services Used
- **Lambda**: Agent execution functions
- **S3**: Model storage and static hosting
- **Kinesis**: Real-time data streaming
- **SageMaker**: Model hosting and inference
- **DynamoDB**: Fast NoSQL database
- **API Gateway**: REST/WebSocket APIs
- **CloudWatch**: Monitoring and logging
- **Cognito**: User authentication
- **ECS**: Container orchestration

### Deployment Steps

#### 1. Infrastructure Setup
```bash
# Create S3 bucket for models
aws s3 mb s3://msp-intelligence-models

# Create DynamoDB tables
aws dynamodb create-table \
  --table-name msp-agents \
  --attribute-definitions AttributeName=agent_id,AttributeType=S \
  --key-schema AttributeName=agent_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

# Create Kinesis stream
aws kinesis create-stream \
  --stream-name msp-intelligence-stream \
  --shard-count 1
```

#### 2. Lambda Functions
```bash
# Package and deploy Lambda functions
cd backend/lambda_functions
zip -r threat_intelligence.zip threat_intelligence/
aws lambda create-function \
  --function-name msp-threat-intelligence \
  --runtime python3.9 \
  --role arn:aws:iam::account:role/lambda-execution-role \
  --handler threat_intelligence.lambda_handler \
  --zip-file fileb://threat_intelligence.zip
```

#### 3. SageMaker Endpoints
```bash
# Create SageMaker model
aws sagemaker create-model \
  --model-name msp-threat-model \
  --primary-container Image=your-account.dkr.ecr.region.amazonaws.com/msp-model:latest

# Create endpoint configuration
aws sagemaker create-endpoint-config \
  --endpoint-config-name msp-threat-config \
  --production-variants VariantName=primary,ModelName=msp-threat-model,InitialInstanceCount=1,InstanceType=ml.t3.medium

# Create endpoint
aws sagemaker create-endpoint \
  --endpoint-name msp-threat-endpoint \
  --endpoint-config-name msp-threat-config
```

#### 4. API Gateway Setup
```bash
# Create REST API
aws apigateway create-rest-api \
  --name msp-intelligence-api \
  --description "MSP Intelligence Mesh Network API"

# Create WebSocket API
aws apigatewayv2 create-api \
  --name msp-intelligence-ws \
  --protocol-type WEBSOCKET \
  --route-selection-expression $request.body.action
```

### Cost Optimization
- Use S3 Intelligent Tiering for automatic cost optimization
- Implement Lambda provisioned concurrency for consistent performance
- Use DynamoDB On-Demand for variable workloads
- Use SageMaker Spot instances for training
- Implement CloudWatch alarms for cost monitoring

## 🔍 Monitoring and Observability

### Health Checks
```bash
# System health
curl http://localhost:8000/health

# Agent status
curl http://localhost:8000/agents/status

# Database health
curl http://localhost:8000/health/database

# AWS services health
curl http://localhost:8000/health/aws
```

### Logging
```bash
# View backend logs
tail -f logs/backend.log

# View frontend logs
tail -f logs/frontend.log

# View agent logs
tail -f logs/agents.log

# View system logs
journalctl -u msp-intelligence -f
```

### Metrics and Monitoring
- **Performance Metrics**: Response times, throughput, error rates
- **Business Metrics**: Revenue impact, cost savings, user satisfaction
- **Network Metrics**: Connected MSPs, collaboration success, threat prevention
- **Agent Metrics**: Health scores, accuracy, processing times

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_agents.py -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html
```

### Integration Tests
```bash
# Test API endpoints
pytest tests/test_api.py -v

# Test agent communication
pytest tests/test_integration.py -v

# Test WebSocket connections
pytest tests/test_websocket.py -v
```

### Performance Tests
```bash
# Load testing
pytest tests/test_performance.py -v

# Stress testing
pytest tests/test_stress.py -v

# Benchmark testing
pytest tests/test_benchmark.py -v
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Conflicts
```bash
# Check if ports are in use
netstat -tulpn | grep :8000
netstat -tulpn | grep :3001

# Kill processes using ports
sudo kill -9 $(lsof -t -i:8000)
sudo kill -9 $(lsof -t -i:3001)
```

#### 2. Memory Issues
```bash
# Check memory usage
free -h
top -p $(pgrep -f "python.*main_simple")

# Increase swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. Dependency Issues
```bash
# Reinstall dependencies
pip uninstall -r backend/requirements_minimal.txt -y
pip install -r backend/requirements_minimal.txt

# Clear pip cache
pip cache purge
```

#### 4. Database Connection Issues
```bash
# Test MongoDB connection
python -c "from pymongo import MongoClient; print(MongoClient('your_mongodb_url').admin.command('ping'))"

# Test Redis connection
python -c "import redis; r = redis.from_url('your_redis_url'); print(r.ping())"
```

### Performance Optimization

#### 1. Backend Optimization
- Enable gzip compression
- Implement response caching
- Use connection pooling
- Optimize database queries

#### 2. Frontend Optimization
- Enable production build
- Implement code splitting
- Use CDN for static assets
- Optimize images and fonts

#### 3. Database Optimization
- Create appropriate indexes
- Use connection pooling
- Implement query optimization
- Monitor slow queries

## 📊 Scaling

### Horizontal Scaling
- Use load balancers for multiple backend instances
- Implement database sharding
- Use Redis clustering for caching
- Deploy multiple frontend instances

### Vertical Scaling
- Increase server resources (CPU, RAM)
- Use faster storage (SSD)
- Optimize application code
- Implement caching strategies

### Auto-scaling
- Configure AWS Auto Scaling Groups
- Set up CloudWatch alarms
- Implement health checks
- Use predictive scaling

## 🔒 Security

### Authentication and Authorization
- Implement JWT tokens
- Use role-based access control
- Enable multi-factor authentication
- Implement session management

### Data Protection
- Encrypt data at rest and in transit
- Implement data masking
- Use secure communication protocols
- Regular security audits

### Network Security
- Use HTTPS/TLS encryption
- Implement firewall rules
- Use VPN for remote access
- Monitor network traffic

## 📈 Performance Tuning

### Backend Performance
- Optimize database queries
- Implement caching strategies
- Use async/await patterns
- Monitor memory usage

### Frontend Performance
- Optimize bundle size
- Implement lazy loading
- Use service workers
- Monitor Core Web Vitals

### Database Performance
- Create appropriate indexes
- Optimize query plans
- Use connection pooling
- Monitor slow queries

## 🚀 Production Deployment

### Pre-deployment Checklist
- [ ] All tests passing
- [ ] Security scan completed
- [ ] Performance testing done
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Disaster recovery plan ready
- [ ] Documentation updated
- [ ] Team training completed

### Deployment Process
1. **Staging Deployment**: Deploy to staging environment
2. **Testing**: Run comprehensive tests
3. **Security Scan**: Perform security assessment
4. **Performance Test**: Validate performance metrics
5. **Production Deployment**: Deploy to production
6. **Monitoring**: Monitor system health
7. **Validation**: Verify all features working
8. **Go-live**: Announce system availability

### Post-deployment
- Monitor system metrics
- Check error logs
- Validate user feedback
- Performance optimization
- Regular maintenance

## 📞 Support

### Getting Help
- Check the troubleshooting section
- Review the API documentation
- Check agent status endpoints
- Review system logs

### Contact Information
- Technical Support: support@msp-intelligence.com
- Documentation: docs.msp-intelligence.com
- GitHub Issues: github.com/msp-intelligence/issues
- Community Forum: forum.msp-intelligence.com

---

**MSP Intelligence Mesh Network** - Revolutionizing MSP Technology Through Collective Intelligence

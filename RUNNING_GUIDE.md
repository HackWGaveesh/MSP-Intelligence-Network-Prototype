# MSP Intelligence Mesh Network - Running & Testing Guide

## 🚀 Quick Start Guide

This guide will help you run and test the MSP Intelligence Mesh Network system step by step.

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 10GB+ free space
- **CPU**: 4+ cores recommended

### Software Requirements
- **Docker**: 20.10+ 
- **Docker Compose**: 2.0+
- **Git**: Latest version
- **Web Browser**: Chrome, Firefox, or Safari

## 🔧 Installation Steps

### Step 1: Check Prerequisites
```bash
# Check if Docker is installed
docker --version
docker-compose --version

# If not installed, install Docker:
# Ubuntu/Debian:
sudo apt update
sudo apt install docker.io docker-compose

# macOS: Download Docker Desktop from https://docker.com
# Windows: Download Docker Desktop from https://docker.com
```

### Step 2: Clone and Setup
```bash
# Navigate to the project directory
cd /home/BTECH_7TH_SEM/Desktop/hackathon/msp-intelligence-mesh

# Make the startup script executable
chmod +x start.sh

# Copy environment configuration
cp env.example .env
```

### Step 3: Quick Start (Recommended)
```bash
# Run the automated startup script
./start.sh
```

This script will:
- Check prerequisites
- Setup environment
- Build and start all services
- Generate demo data
- Show access URLs

## 🐳 Manual Docker Setup

If you prefer manual setup:

### Step 1: Build and Start Services
```bash
# Build and start all services
docker-compose up --build -d

# Check if services are running
docker-compose ps
```

### Step 2: Generate Demo Data
```bash
# Generate synthetic data for demonstrations
docker-compose exec backend python -m utils.data_generator
```

### Step 3: Verify Services
```bash
# Check backend health
curl http://localhost:8000/health

# Check frontend
curl http://localhost:3000
```

## 🌐 Access the Application

Once running, access the application at:

- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Grafana Monitoring**: http://localhost:3001 (admin/admin123)
- **Prometheus Metrics**: http://localhost:9090

## 🧪 Testing the System

### 1. Basic Health Check
```bash
# Check if all services are healthy
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-01-XX...",
  "version": "1.0.0",
  "service": "MSP Intelligence Mesh Network API"
}
```

### 2. Test Agent Status
```bash
# Check agent status
curl http://localhost:8000/agents/status

# Expected response: Status of all AI agents
```

### 3. Test WebSocket Connection
```bash
# Test WebSocket connection (using wscat if installed)
npm install -g wscat
wscat -c ws://localhost:8000/ws

# Send a test message
{"type": "ping"}
```

### 4. Test Threat Intelligence
```bash
# Test threat analysis
curl -X POST http://localhost:8000/threat-intelligence/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Detected ransomware activity with encryption patterns",
    "threat_type": "ransomware",
    "severity": "high"
  }'
```

### 5. Test Collaboration Matching
```bash
# Test partner finding
curl -X POST http://localhost:8000/collaboration/find-partners \
  -H "Content-Type: application/json" \
  -d '{
    "msp_id": "msp_001",
    "requirements": {
      "skills": ["cloud_services", "security"],
      "location": "any",
      "size": "medium"
    }
  }'
```

### 6. Test Federated Learning
```bash
# Test federated learning status
curl http://localhost:8000/federated-learning/status

# Start a training round
curl -X POST http://localhost:8000/federated-learning/start-round \
  -H "Content-Type: application/json" \
  -d '{
    "participants": ["msp_001", "msp_002", "msp_003"],
    "model_type": "threat_detection"
  }'
```

## 🎮 Interactive Testing

### Frontend Testing
1. Open http://localhost:3000 in your browser
2. Navigate through different dashboard sections:
   - **Dashboard**: Main overview with real-time metrics
   - **Threat Intelligence**: Threat detection and analysis
   - **Collaboration Portal**: Partner matching and opportunities
   - **Analytics Engine**: Predictive analytics and forecasting
   - **Privacy Control**: Privacy settings and compliance
   - **Agent Orchestrator**: Agent status and management

### Simulation Features
1. **Threat Detection Simulation**:
   - Go to Threat Intelligence page
   - Click "Simulate Threat Detection"
   - Watch real-time threat analysis

2. **Collaboration Simulation**:
   - Go to Collaboration Portal
   - Click "Simulate Collaboration Opportunity"
   - See AI-generated proposals

3. **Federated Learning Simulation**:
   - Go to Privacy Control page
   - Click "Simulate Federated Learning"
   - Watch privacy-preserving training

## 🔍 Monitoring and Debugging

### View Logs
```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f mongodb
```

### Check Resource Usage
```bash
# Check Docker resource usage
docker stats

# Check disk usage
docker system df
```

### Debug Issues
```bash
# Check service status
docker-compose ps

# Restart a specific service
docker-compose restart backend

# Rebuild and restart
docker-compose up --build -d backend
```

## 🧪 Running Tests

### Backend Tests
```bash
# Run all backend tests
docker-compose exec backend pytest tests/ -v

# Run specific test file
docker-compose exec backend pytest tests/test_agents.py -v

# Run with coverage
docker-compose exec backend pytest tests/ --cov=agents --cov-report=html
```

### Frontend Tests
```bash
# Run frontend tests
docker-compose exec frontend npm test

# Run with coverage
docker-compose exec frontend npm test -- --coverage
```

### Integration Tests
```bash
# Run end-to-end tests
docker-compose exec backend python -m pytest tests/test_integration.py -v
```

## 🎯 Demo Scenarios

### Scenario 1: Live Threat Detection (2 minutes)
1. Open http://localhost:3000
2. Go to Threat Intelligence page
3. Click "Simulate Threat Detection"
4. Watch real-time threat analysis
5. See network response and protection deployment

### Scenario 2: Collaboration Matching (3 minutes)
1. Go to Collaboration Portal
2. Click "Simulate Collaboration Opportunity"
3. Watch AI-powered partner matching
4. See joint proposal generation
5. Review revenue sharing model

### Scenario 3: Federated Learning (2 minutes)
1. Go to Privacy Control page
2. Click "Simulate Federated Learning"
3. Watch privacy-preserving training
4. See model convergence visualization
5. Review privacy metrics

### Scenario 4: Network Effects (1 minute)
1. Go to Dashboard
2. Watch real-time metrics updating
3. See network growth simulation
4. Observe intelligence level increasing

## 🛠️ Development Mode

### Backend Development
```bash
# Run backend in development mode
cd backend
pip install -r requirements.txt
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
# Run frontend in development mode
cd frontend
npm install
npm start
```

## 🔧 Configuration

### Environment Variables
Edit `.env` file to customize:
```bash
# Application settings
DEBUG=True
LOG_LEVEL=INFO

# AWS settings (optional)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1

# Database settings
MONGODB_URL=mongodb://admin:password123@mongodb:27017/msp_network?authSource=admin
REDIS_URL=redis://redis:6379
```

### Performance Tuning
```bash
# Increase memory limits in docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Port Conflicts
```bash
# Check if ports are in use
netstat -tulpn | grep :3000
netstat -tulpn | grep :8000

# Kill processes using ports
sudo kill -9 $(lsof -t -i:3000)
sudo kill -9 $(lsof -t -i:8000)
```

#### 2. Docker Issues
```bash
# Clean up Docker
docker-compose down -v
docker system prune -a
docker-compose up --build
```

#### 3. Memory Issues
```bash
# Check memory usage
free -h
docker stats

# Increase Docker memory limits
# Edit Docker Desktop settings
```

#### 4. Service Won't Start
```bash
# Check logs for errors
docker-compose logs backend
docker-compose logs frontend

# Check service status
docker-compose ps
```

### Getting Help
- Check logs: `docker-compose logs -f`
- Restart services: `docker-compose restart`
- Rebuild: `docker-compose up --build`
- Clean restart: `docker-compose down -v && docker-compose up --build`

## 📊 Performance Testing

### Load Testing
```bash
# Install Apache Bench
sudo apt install apache2-utils

# Test API performance
ab -n 1000 -c 10 http://localhost:8000/health

# Test WebSocket connections
# Use a WebSocket load testing tool
```

### Monitoring
- **Grafana**: http://localhost:3001 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Application Metrics**: http://localhost:8000/metrics

## 🎉 Success Indicators

You'll know the system is working correctly when:

1. ✅ All services show "Up" status in `docker-compose ps`
2. ✅ Frontend loads at http://localhost:3000
3. ✅ Backend API responds at http://localhost:8000/health
4. ✅ WebSocket connection works
5. ✅ Demo simulations run successfully
6. ✅ Real-time data updates in the dashboard
7. ✅ All agent status shows "operational"

## 🚀 Next Steps

Once the system is running:

1. **Explore the Dashboard**: Navigate through all sections
2. **Run Simulations**: Try all demo scenarios
3. **Check Monitoring**: Review Grafana dashboards
4. **Test APIs**: Use the API documentation at /docs
5. **Review Logs**: Monitor system performance
6. **Customize**: Modify configuration as needed

---

**The MSP Intelligence Mesh Network is now ready for demonstration and testing!**

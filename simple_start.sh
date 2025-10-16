#!/bin/bash

# Simple MSP Intelligence Mesh Network Startup Script
# Fixes Python 3.12 compatibility issues

set -e

echo "🚀 Starting MSP Intelligence Mesh Network - Simple Mode"
echo "====================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "backend/api/main_simple.py" ]; then
    print_error "Please run this script from the msp-intelligence-mesh directory"
    exit 1
fi

# Remove existing virtual environment if it has issues
if [ -d "venv" ]; then
    print_status "Removing existing virtual environment..."
    rm -rf venv
fi

# Create fresh virtual environment
print_status "Creating fresh Python virtual environment..."
python3 -m venv venv
print_success "Virtual environment created"

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip and install basic tools
print_status "Upgrading pip and installing basic tools..."
pip install --upgrade pip setuptools wheel
print_success "Basic tools installed"

# Install simple requirements
print_status "Installing simple requirements..."
pip install -r backend/requirements_simple.txt
print_success "Requirements installed"

# Create logs directory
mkdir -p logs

# Start backend server
print_status "Starting backend server..."
cd backend
python api/main_simple.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
print_status "Waiting for backend to start..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_success "Backend started successfully (PID: $BACKEND_PID)"
else
    print_warning "Backend may not be fully ready yet, continuing..."
fi

# Start frontend server
print_status "Starting frontend server..."
python serve_frontend.py &
FRONTEND_PID=$!

# Wait for frontend to start
print_status "Waiting for frontend to start..."
sleep 3

# Check if frontend is running
if curl -s http://localhost:3001 > /dev/null 2>&1; then
    print_success "Frontend started successfully (PID: $FRONTEND_PID)"
else
    print_warning "Frontend may not be fully ready yet, continuing..."
fi

# Test system
print_status "Testing system..."
if curl -s http://localhost:8000/health | grep -q "healthy" 2>/dev/null; then
    print_success "✅ Backend is responding"
else
    print_warning "❌ Backend is not responding yet"
fi

if curl -s http://localhost:3001 > /dev/null 2>&1; then
    print_success "✅ Frontend is responding"
else
    print_warning "❌ Frontend is not responding yet"
fi

# Display system information
echo ""
echo "🎉 MSP Intelligence Mesh Network is starting!"
echo "============================================="
echo ""
echo "📊 System Status:"
echo "  Backend PID: $BACKEND_PID"
echo "  Frontend PID: $FRONTEND_PID"
echo ""
echo "🌐 Access URLs:"
echo "  Main Dashboard: http://localhost:3001"
echo "  Backend API: http://localhost:8000"
echo "  API Documentation: http://localhost:8000/docs"
echo "  Health Check: http://localhost:8000/health"
echo ""
echo "🤖 AI Agents Status:"
echo "  Total Agents: 10+"
echo "  Agent Types: Threat Intelligence, Collaboration, Federated Learning, Market Intelligence, Client Health, Revenue Optimization, Anomaly Detection, NLP Query, Resource Allocation, Security Compliance"
echo ""
echo "📝 Useful Commands:"
echo "  View backend logs: tail -f logs/backend.log"
echo "  View frontend logs: tail -f logs/frontend.log"
echo "  Stop system: kill $BACKEND_PID $FRONTEND_PID"
echo "  Test backend: curl http://localhost:8000/health"
echo "  Test agents: curl http://localhost:8000/agents/status"
echo ""
echo "🔧 System Features:"
echo "  ✅ 10+ AI Agents with full functionality"
echo "  ✅ Real-time threat detection and response"
echo "  ✅ AI-powered collaboration matching"
echo "  ✅ Federated learning with privacy guarantees"
echo "  ✅ Market intelligence and pricing optimization"
echo "  ✅ Client health prediction and churn analysis"
echo "  ✅ Revenue forecasting and opportunity detection"
echo "  ✅ Anomaly detection and system monitoring"
echo "  ✅ Natural language query interface"
echo "  ✅ Resource allocation and scheduling optimization"
echo "  ✅ Security compliance monitoring"
echo "  ✅ Live WebSocket updates"
echo "  ✅ Professional dashboard UI"
echo ""
print_success "System is starting! Please wait a moment for all services to be ready."
print_warning "If services don't respond immediately, wait a few more seconds and try again."

# Function to cleanup on exit
cleanup() {
    echo ""
    print_status "Shutting down system..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    print_success "System stopped successfully"
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Keep script running
while true; do
    sleep 1
done
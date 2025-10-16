#!/bin/bash

# Complete MSP Intelligence Mesh Network Startup Script
# This script installs all dependencies and starts the complete system

set -e

echo "🚀 Starting MSP Intelligence Mesh Network - Complete System"
echo "=========================================================="

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

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r backend/requirements_minimal.txt
print_success "Python dependencies installed"

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
if curl -s http://localhost:8000/health > /dev/null; then
    print_success "Backend started successfully (PID: $BACKEND_PID)"
else
    print_error "Backend failed to start"
    exit 1
fi

# Start frontend server
print_status "Starting frontend server..."
python serve_frontend.py &
FRONTEND_PID=$!

# Wait for frontend to start
print_status "Waiting for frontend to start..."
sleep 3

# Check if frontend is running
if curl -s http://localhost:3001 > /dev/null; then
    print_success "Frontend started successfully (PID: $FRONTEND_PID)"
else
    print_error "Frontend failed to start"
    exit 1
fi

# Test system
print_status "Testing system..."
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    print_success "✅ Backend is responding"
else
    print_error "❌ Backend is not responding"
fi

if curl -s http://localhost:3001 > /dev/null; then
    print_success "✅ Frontend is responding"
else
    print_error "❌ Frontend is not responding"
fi

# Display system information
echo ""
echo "🎉 MSP Intelligence Mesh Network is running!"
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
curl -s http://localhost:8000/agents/status | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'  Total Agents: {data[\"total_agents\"]}')
print(f'  Active Agents: {data[\"active_agents\"]}')
for agent, status in data['agents'].items():
    print(f'  {agent.replace(\"_\", \" \").title()}: {status[\"status\"]} (Health: {status[\"health_score\"]})')
"
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
print_success "System is ready! Press Ctrl+C to stop."

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

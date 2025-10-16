#!/bin/bash

# MSP Intelligence Mesh Network - Startup Script
# This script starts both backend and frontend servers

echo "🚀 MSP Intelligence Mesh Network - Starting Application"
echo "=================================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_info "Virtual environment not found. Creating one..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Check if dependencies are installed
if [ ! -f "venv/.dependencies_installed" ]; then
    print_info "Installing backend dependencies..."
    pip install --upgrade pip setuptools wheel
    
    if [ -f "backend/requirements_simple.txt" ]; then
        pip install -r backend/requirements_simple.txt
    else
        print_error "requirements_simple.txt not found!"
        exit 1
    fi
    
    # Mark dependencies as installed
    touch venv/.dependencies_installed
    print_success "Dependencies installed"
else
    print_success "Dependencies already installed"
fi

# Kill any existing processes on ports 8000 and 8080
print_info "Checking for existing processes..."
pkill -f "uvicorn.*main_simple" 2>/dev/null
pkill -f "python.*serve_frontend" 2>/dev/null
sleep 1

# Create logs directory
mkdir -p logs

# Start backend server
print_info "Starting backend server (FastAPI on port 8000)..."
cd backend
python api/main_simple.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..
print_success "Backend server starting (PID: $BACKEND_PID)"

# Wait a moment for backend to initialize
sleep 3

# Check if backend is running
if ps -p $BACKEND_PID > /dev/null; then
    print_success "Backend server is running"
else
    print_error "Backend server failed to start. Check logs/backend.log"
    exit 1
fi

# Start frontend server
print_info "Starting frontend server (port 8080)..."
python serve_frontend.py > logs/frontend.log 2>&1 &
FRONTEND_PID=$!
print_success "Frontend server starting (PID: $FRONTEND_PID)"

# Wait a moment for frontend to initialize
sleep 2

# Check if frontend is running
if ps -p $FRONTEND_PID > /dev/null; then
    print_success "Frontend server is running"
else
    print_error "Frontend server failed to start. Check logs/frontend.log"
    exit 1
fi

echo ""
echo "=================================================="
print_success "MSP Intelligence Mesh Network is running!"
echo "=================================================="
echo ""
echo "📍 Access the application:"
echo "   Frontend: http://localhost:8080"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "🔧 Process IDs:"
echo "   Backend PID: $BACKEND_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo ""
echo "📝 Logs:"
echo "   Backend: logs/backend.log"
echo "   Frontend: logs/frontend.log"
echo ""
echo "🛑 To stop the application:"
echo "   Run: ./stop_app.sh"
echo "   Or press Ctrl+C to stop this script (servers will continue running)"
echo ""
echo "=================================================="

# Save PIDs for stop script
echo "$BACKEND_PID" > .backend.pid
echo "$FRONTEND_PID" > .frontend.pid

# Keep script running and show logs
print_info "Monitoring application... Press Ctrl+C to exit (servers will continue running)"
echo ""

# Tail both log files
tail -f logs/backend.log logs/frontend.log


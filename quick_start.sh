#!/bin/bash

# Quick Start Script for MSP Intelligence Mesh Network (No Docker Required)

echo "🚀 Quick Start - MSP Intelligence Mesh Network"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check Python
if ! command -v python3 &> /dev/null; then
    print_warning "Python 3 not found. Installing..."
    sudo apt update && sudo apt install -y python3 python3-pip python3-venv
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    print_warning "Node.js not found. Installing..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

print_status "Setting up backend..."

# Setup backend
cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate and install dependencies
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_minimal.txt

# Start backend in background
print_status "Starting backend server..."
nohup python3 api/main_simple.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../logs/backend.pid

cd ..

# Wait for backend
print_status "Waiting for backend to start..."
sleep 5

# Test backend
if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    print_success "Backend started successfully!"
else
    print_warning "Backend may still be starting..."
fi

print_status "Setting up frontend..."

# Setup frontend
cd frontend

# Copy the minimal package.json
cp package_minimal.json package.json

# Install dependencies with legacy peer deps to avoid conflicts
npm install --legacy-peer-deps

# Start frontend in background
print_status "Starting frontend server..."
nohup npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../logs/frontend.pid

cd ..

# Wait for frontend
print_status "Waiting for frontend to start..."
sleep 10

print_success "🎉 MSP Intelligence Mesh Network is starting!"
echo ""
echo "Access URLs:"
echo "  Frontend Dashboard: http://localhost:3000"
echo "  Backend API: http://localhost:8000"
echo "  API Documentation: http://localhost:8000/docs"
echo ""
echo "To stop the system:"
echo "  kill \$(cat logs/backend.pid)"
echo "  kill \$(cat logs/frontend.pid)"
echo ""
echo "To view logs:"
echo "  tail -f logs/backend.log"
echo "  tail -f logs/frontend.log"
echo ""

# Keep script running
print_status "System is running. Press Ctrl+C to stop."
while true; do
    sleep 10
    # Check if services are still running
    if [ -f logs/backend.pid ] && ! kill -0 $(cat logs/backend.pid) 2>/dev/null; then
        print_warning "Backend stopped unexpectedly"
        break
    fi
    if [ -f logs/frontend.pid ] && ! kill -0 $(cat logs/frontend.pid) 2>/dev/null; then
        print_warning "Frontend stopped unexpectedly"
        break
    fi
done

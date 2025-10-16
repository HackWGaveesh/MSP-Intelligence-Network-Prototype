#!/bin/bash

# MSP Intelligence Mesh Network - Run Without Docker
# This script runs the system using Python and Node.js directly

set -e

echo "🚀 Starting MSP Intelligence Mesh Network (Direct Mode)..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.11+"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    print_success "Python $python_version found"
}

# Check if Node.js is installed
check_node() {
    print_status "Checking Node.js installation..."
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18+"
        print_status "You can install it with: curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - && sudo apt-get install -y nodejs"
        exit 1
    fi
    
    node_version=$(node --version)
    print_success "Node.js $node_version found"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    cd backend
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
    cd ..
}

# Install Node.js dependencies
install_node_deps() {
    print_status "Installing Node.js dependencies..."
    
    cd frontend
    
    # Install dependencies
    npm install
    
    print_success "Node.js dependencies installed"
    cd ..
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    if [ ! -f .env ]; then
        if [ -f env.example ]; then
            cp env.example .env
            print_success "Created .env file from env.example"
        else
            print_warning "No env.example file found. Creating basic .env file..."
            cat > .env << EOF
# MSP Intelligence Mesh Network Configuration
DEBUG=True
LOG_LEVEL=INFO
SECRET_KEY=msp-intelligence-mesh-network-secret-key-2025
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database Configuration (using local files for demo)
MONGODB_URL=mongodb://localhost:27017/msp_network
REDIS_URL=redis://localhost:6379

# API Keys (Optional - will use mock data if not provided)
HUGGINGFACE_API_KEY=your_huggingface_key
GEMINI_API_KEY=your_gemini_key
GROK_API_KEY=your_grok_key
EOF
            print_success "Created basic .env file"
        fi
    else
        print_success ".env file already exists"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data/synthetic
    mkdir -p data/models
    mkdir -p logs
    mkdir -p backend/venv
    
    print_success "Directories created"
}

# Generate demo data
generate_demo_data() {
    print_status "Generating demo data..."
    
    cd backend
    source venv/bin/activate
    
    # Generate demo data
    python3 -c "
from utils.data_generator import MSPDataGenerator
import os

print('Generating MSP profiles...')
generator = MSPDataGenerator()
generator.generate_msp_profiles(100)
print('Generating threat intelligence data...')
generator.generate_threat_intelligence_data(1000)
print('Generating collaboration opportunities...')
generator.generate_collaboration_opportunities(50)
print('Generating client interactions...')
generator.generate_client_interactions(5000)
print('Generating market data...')
generator.generate_market_data(30)
print('Exporting data to files...')
generator.export_to_files()
print('Demo data generation completed successfully!')
" 2>/dev/null || {
        print_warning "Demo data generation failed, but system will still work with mock data"
    }
    
    cd ..
    print_success "Demo data generation completed"
}

# Start backend
start_backend() {
    print_status "Starting backend server..."
    
    cd backend
    source venv/bin/activate
    
    # Start backend in background
    nohup python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload > ../logs/backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > ../logs/backend.pid
    
    cd ..
    
    # Wait for backend to start
    print_status "Waiting for backend to start..."
    for i in {1..30}; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            print_success "Backend started successfully"
            return 0
        fi
        sleep 2
    done
    
    print_error "Backend failed to start within 60 seconds"
    return 1
}

# Start frontend
start_frontend() {
    print_status "Starting frontend server..."
    
    cd frontend
    
    # Start frontend in background
    nohup npm start > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../logs/frontend.pid
    
    cd ..
    
    # Wait for frontend to start
    print_status "Waiting for frontend to start..."
    for i in {1..30}; do
        if curl -f http://localhost:3000 >/dev/null 2>&1; then
            print_success "Frontend started successfully"
            return 0
        fi
        sleep 2
    done
    
    print_error "Frontend failed to start within 60 seconds"
    return 1
}

# Show status
show_status() {
    print_status "System Status:"
    echo "=============="
    
    # Check if services are running
    if [ -f logs/backend.pid ] && kill -0 $(cat logs/backend.pid) 2>/dev/null; then
        print_success "Backend is running (PID: $(cat logs/backend.pid))"
    else
        print_error "Backend is not running"
    fi
    
    if [ -f logs/frontend.pid ] && kill -0 $(cat logs/frontend.pid) 2>/dev/null; then
        print_success "Frontend is running (PID: $(cat logs/frontend.pid))"
    else
        print_error "Frontend is not running"
    fi
    
    echo ""
    print_status "Access URLs:"
    echo "  Frontend Dashboard: http://localhost:3000"
    echo "  Backend API: http://localhost:8000"
    echo "  API Documentation: http://localhost:8000/docs"
    echo ""
    
    print_status "Useful Commands:"
    echo "  View backend logs: tail -f logs/backend.log"
    echo "  View frontend logs: tail -f logs/frontend.log"
    echo "  Stop system: ./stop_direct.sh"
    echo "  Restart backend: kill \$(cat logs/backend.pid) && ./start_backend.sh"
    echo ""
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    
    if [ -f logs/backend.pid ]; then
        kill $(cat logs/backend.pid) 2>/dev/null || true
        rm -f logs/backend.pid
    fi
    
    if [ -f logs/frontend.pid ]; then
        kill $(cat logs/frontend.pid) 2>/dev/null || true
        rm -f logs/frontend.pid
    fi
}

# Main execution
main() {
    echo "MSP Intelligence Mesh Network - Direct Mode"
    echo "Built for Superhack 2025 - Team Lossless"
    echo ""
    
    # Setup cleanup trap
    trap cleanup EXIT INT TERM
    
    check_python
    check_node
    setup_environment
    create_directories
    install_python_deps
    install_node_deps
    generate_demo_data
    start_backend
    start_frontend
    show_status
    
    print_success "🎉 MSP Intelligence Mesh Network is ready!"
    print_status "Open http://localhost:3000 to start exploring the system"
    print_status "Press Ctrl+C to stop the system"
    
    # Keep script running
    while true; do
        sleep 10
        # Check if services are still running
        if [ -f logs/backend.pid ] && ! kill -0 $(cat logs/backend.pid) 2>/dev/null; then
            print_error "Backend stopped unexpectedly"
            break
        fi
        if [ -f logs/frontend.pid ] && ! kill -0 $(cat logs/frontend.pid) 2>/dev/null; then
            print_error "Frontend stopped unexpectedly"
            break
        fi
    done
}

# Run main function
main "$@"


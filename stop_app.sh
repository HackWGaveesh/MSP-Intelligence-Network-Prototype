#!/bin/bash

# MSP Intelligence Mesh Network - Stop Script
# This script stops both backend and frontend servers

echo "🛑 MSP Intelligence Mesh Network - Stopping Application"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Stop backend server
if [ -f ".backend.pid" ]; then
    BACKEND_PID=$(cat .backend.pid)
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        print_info "Stopping backend server (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        sleep 1
        if ps -p $BACKEND_PID > /dev/null 2>&1; then
            kill -9 $BACKEND_PID 2>/dev/null
        fi
        print_success "Backend server stopped"
    else
        print_info "Backend server not running"
    fi
    rm .backend.pid
else
    print_info "No backend PID file found"
fi

# Stop frontend server
if [ -f ".frontend.pid" ]; then
    FRONTEND_PID=$(cat .frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        print_info "Stopping frontend server (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        sleep 1
        if ps -p $FRONTEND_PID > /dev/null 2>&1; then
            kill -9 $FRONTEND_PID 2>/dev/null
        fi
        print_success "Frontend server stopped"
    else
        print_info "Frontend server not running"
    fi
    rm .frontend.pid
else
    print_info "No frontend PID file found"
fi

# Kill any remaining processes
pkill -f "uvicorn.*main_simple" 2>/dev/null
pkill -f "python.*serve_frontend" 2>/dev/null

echo ""
echo "=================================================="
print_success "MSP Intelligence Mesh Network stopped"
echo "=================================================="


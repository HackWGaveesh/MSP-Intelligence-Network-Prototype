#!/bin/bash

# Stop MSP Intelligence Mesh Network (Direct Mode)

echo "🛑 Stopping MSP Intelligence Mesh Network..."

# Stop backend
if [ -f logs/backend.pid ]; then
    echo "Stopping backend..."
    kill $(cat logs/backend.pid) 2>/dev/null || true
    rm -f logs/backend.pid
    echo "Backend stopped"
fi

# Stop frontend
if [ -f logs/frontend.pid ]; then
    echo "Stopping frontend..."
    kill $(cat logs/frontend.pid) 2>/dev/null || true
    rm -f logs/frontend.pid
    echo "Frontend stopped"
fi

# Kill any remaining processes
pkill -f "uvicorn api.main:app" 2>/dev/null || true
pkill -f "npm start" 2>/dev/null || true

echo "✅ System stopped successfully"


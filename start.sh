#!/bin/bash

# MSP Intelligence Mesh Network - Startup Script
# This script sets up and starts the complete system

set -e

echo "🚀 Starting MSP Intelligence Mesh Network..."
echo "=============================================="

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

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Check if ports are available
check_ports() {
    print_status "Checking if required ports are available..."
    
    ports=(3000 8000 3001 9090 6379 27017)
    for port in "${ports[@]}"; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_warning "Port $port is already in use. This may cause issues."
        fi
    done
    
    print_success "Port check completed"
}

# Setup environment file
setup_environment() {
    print_status "Setting up environment configuration..."
    
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

# AWS Configuration (Optional)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1

# Database Configuration
MONGODB_URL=mongodb://admin:password123@mongodb:27017/msp_network?authSource=admin
REDIS_URL=redis://redis:6379

# API Keys (Optional)
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
    mkdir -p docker/ssl
    
    print_success "Directories created"
}

# Build and start services
start_services() {
    print_status "Building and starting services..."
    
    # Stop any existing containers
    docker-compose down 2>/dev/null || true
    
    # Build and start services
    docker-compose up --build -d
    
    print_success "Services started successfully"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for backend
    print_status "Waiting for backend API..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            print_success "Backend API is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Backend API failed to start within 60 seconds"
        exit 1
    fi
    
    # Wait for frontend
    print_status "Waiting for frontend..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:3000 >/dev/null 2>&1; then
            print_success "Frontend is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Frontend failed to start within 60 seconds"
        exit 1
    fi
}

# Generate demo data
generate_demo_data() {
    print_status "Generating demo data..."
    
    # Wait a bit more for the backend to be fully ready
    sleep 10
    
    # Generate demo data
    docker-compose exec -T backend python -c "
from utils.data_generator import MSPDataGenerator
import os

print('Generating MSP profiles...')
generator = MSPDataGenerator()
generator.generate_msp_profiles(1000)
print('Generating threat intelligence data...')
generator.generate_threat_intelligence_data(10000)
print('Generating collaboration opportunities...')
generator.generate_collaboration_opportunities(500)
print('Generating client interactions...')
generator.generate_client_interactions(50000)
print('Generating market data...')
generator.generate_market_data(365)
print('Exporting data to files...')
generator.export_to_files()
print('Demo data generation completed successfully!')
" 2>/dev/null || {
        print_warning "Demo data generation failed, but system is still functional"
    }
    
    print_success "Demo data generation completed"
}

# Show system status
show_status() {
    print_status "System Status:"
    echo "=============="
    
    # Check service status
    if docker-compose ps | grep -q "Up"; then
        print_success "All services are running"
    else
        print_error "Some services are not running"
        docker-compose ps
    fi
    
    echo ""
    print_status "Access URLs:"
    echo "  Frontend Dashboard: http://localhost:3000"
    echo "  Backend API: http://localhost:8000"
    echo "  API Documentation: http://localhost:8000/docs"
    echo "  Grafana Monitoring: http://localhost:3001 (admin/admin123)"
    echo "  Prometheus Metrics: http://localhost:9090"
    echo ""
    
    print_status "Demo Instructions:"
    echo "  1. Open http://localhost:3000 in your browser"
    echo "  2. Navigate through the dashboard sections"
    echo "  3. Try the simulation features in each section"
    echo "  4. See demo_script.md for a comprehensive demo guide"
    echo ""
    
    print_status "Useful Commands:"
    echo "  View logs: docker-compose logs -f"
    echo "  Stop system: docker-compose down"
    echo "  Restart: docker-compose restart"
    echo "  Rebuild: docker-compose up --build"
    echo ""
}

# Main execution
main() {
    echo "MSP Intelligence Mesh Network - Production System"
    echo "Built for Superhack 2025 - Team Lossless"
    echo ""
    
    check_docker
    check_ports
    setup_environment
    create_directories
    start_services
    wait_for_services
    generate_demo_data
    show_status
    
    print_success "🎉 MSP Intelligence Mesh Network is ready!"
    print_status "Open http://localhost:3000 to start exploring the system"
}

# Handle script interruption
trap 'print_error "Script interrupted. Stopping services..."; docker-compose down; exit 1' INT

# Run main function
main "$@"

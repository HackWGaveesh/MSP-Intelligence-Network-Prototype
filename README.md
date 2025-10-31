# MSP Intelligence Mesh - Setup and Run Guide

This document consolidates everything you need to install, start, verify, and troubleshoot the MSP Intelligence Mesh locally. It replaces the scattered README-style notes so you only have one place to look when preparing a demo or development environment.

## Overview
- Multi-agent FastAPI backend with real AI model integrations
- React/TypeScript dashboard with live WebSocket updates
- Optional Docker stack with monitoring (Grafana, Prometheus)
- Helper scripts for one-command startup or manual control

## Prerequisites
- Operating system: Linux, macOS, or Windows (WSL2 recommended)
- Hardware: 8 GB RAM (16 GB recommended), 10 GB free disk, 4+ CPU cores
- Internet access for one-time Python package and model downloads

Tools required for each path:
- Docker path: Docker 20.10+, Docker Compose 2+
- Direct path: Python 3.10+, Node.js 18+, npm 9+, jq (optional for curl output)

Quick version checks:
```bash
python3 --version
node --version
docker --version
docker compose version
```

## Quick Start Options

### Option A - Automated Docker stack (production-style)
```bash
cd msp-intelligence-mesh
chmod +x start.sh
./start.sh
```
What happens:
- generates `.env` if missing and prepares data folders
- builds and starts all Docker services (backend, frontend, monitoring, databases)
- loads demo data and waits for health checks to pass
- prints useful URLs and commands at the end

Access once the script finishes:
- Frontend dashboard: http://localhost:3000
- Backend API: http://localhost:8000
- API docs (OpenAPI): http://localhost:8000/docs
- Grafana: http://localhost:3001 (admin/admin123)
- Prometheus: http://localhost:9090

Stop the stack when you are done:
```bash
docker compose down
```

### Option B - Direct mode without Docker (Python + Node)
```bash
cd msp-intelligence-mesh
chmod +x run_without_docker.sh
./run_without_docker.sh
```
What happens:
- ensures Python and Node.js are present
- creates/activates a virtual environment under `backend`
- installs backend Python dependencies and frontend npm packages
- generates lightweight demo data
- starts FastAPI on port 8000 and React dev server on port 3000
- tails logs to `logs/backend.log` and `logs/frontend.log`

Stop direct mode with `Ctrl+C` in the same terminal or run `./stop_direct.sh`.

## Manual Run Without Scripts

### 1. Backend (FastAPI)
```bash
cd msp-intelligence-mesh/backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_minimal.txt
cd api
python3 main_simple.py
```
Backend endpoints:
- http://localhost:8000
- http://localhost:8000/docs
- http://localhost:8000/health

### 2. Frontend (React dashboard)
```bash
cd msp-intelligence-mesh/frontend
npm install
npm start
```
React dev server runs at http://localhost:3000. Set `API_BASE_URL` in `.env` if you need a non-default backend address.

### 3. Lightweight static preview (optional)
If you only need the static HTML pages for a quick demo:
```bash
cd msp-intelligence-mesh/frontend
python3 -m http.server 8080
```
Open http://localhost:8080. API calls still route to http://localhost:8000, so keep the backend running separately.

## Real AI Model Download (one-time)
```bash
cd msp-intelligence-mesh/backend/models
python3 download_models.py
```
Rerun this step if the backend logs say that models could not be loaded (it will fall back to simulated responses otherwise).

## Smoke Tests

### REST endpoints (requires backend running)
```bash
curl -sX POST http://localhost:8000/threat-intelligence/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Suspected ransomware encrypting files with bitcoin ransom note"}' | jq .

curl -sX POST http://localhost:8000/market-intelligence/analyze \
  -H "Content-Type: application/json" \
  -d '{"query":"MSP pricing trends in SMB cybersecurity","industry_segment":"security"}' | jq .

curl -sX POST http://localhost:8000/nlp-query/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the current network intelligence level?"}' | jq .

curl -sX POST http://localhost:8000/collaboration/match \
  -H "Content-Type: application/json" \
  -d '{"requirements":"Cloud migration expertise with Azure security experience"}' | jq .

curl -sX POST http://localhost:8000/client-health/predict \
  -H "Content-Type: application/json" \
  -d '{"client_id":"C001","ticket_volume":65,"resolution_time":48,"satisfaction_score":4}' | jq .

curl -sX POST http://localhost:8000/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{"metric_type":"CPU Usage","time_range_hours":4,"values":[32,35,38,80,92,45,41,39,37,36,85,93,40,38,37,36,35,34,33,32,31]}' | jq .

curl -sX POST http://localhost:8000/compliance/check \
  -H "Content-Type: application/json" \
  -d '{"framework":"SOC2","policy_text":"MFA enforced. Data encrypted at rest and in transit. Quarterly audits and incident response defined."}' | jq .

curl -sX POST http://localhost:8000/revenue/forecast \
  -H "Content-Type: application/json" \
  -d '{"current_revenue":500000,"period_days":180}' | jq .
```

### WebSocket check
```bash
wscat -c ws://localhost:8000/ws
> {"type":"ping"}
```

### Postman import
- Postman: Import -> Link -> `http://localhost:8000/openapi.json`

## Testing and Utilities
- Backend tests in Docker: `docker compose exec backend pytest tests -v`
- Backend tests without Docker: `cd backend && source venv/bin/activate && pytest tests -v`
- Frontend tests (Docker): `docker compose exec frontend npm test`
- Frontend tests (Direct): `cd frontend && npm test`
- Real AI validation script: `./test_real_ai.sh`

## Troubleshooting
- 422 errors: ensure `Content-Type: application/json` and send valid JSON bodies.
- Backend falls back to simulated models: rerun `backend/models/download_models.py` and restart the API.
- Port in use (3000/8000/etc): find and kill the process with `lsof -Pi :PORT -sTCP:LISTEN` or adjust the port in `.env` and React config.
- Docker resources low: run `docker compose down -v` and `docker system prune -f` before rebuilding.
- Logs: `docker compose logs -f`, `tail -f logs/backend.log`, `tail -f logs/frontend.log`.

## Repository
- https://github.com/HackWGaveesh/MSP-Intelligence-Network-Prototype.git

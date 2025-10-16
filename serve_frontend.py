#!/usr/bin/env python3
"""
Simple HTTP server to serve the MSP Intelligence Mesh Network frontend
"""
import http.server
import socketserver
import os
import sys
from pathlib import Path

# Configuration
PORT = 8080
FRONTEND_DIR = Path(__file__).parent / "frontend"

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.end_headers()

def main():
    # Change to frontend directory
    os.chdir(FRONTEND_DIR)
    
    # Create server
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"🚀 MSP Intelligence Mesh Network Frontend Server")
        print(f"📁 Serving from: {FRONTEND_DIR}")
        print(f"🌐 Frontend URL: http://localhost:{PORT}")
        print(f"🔧 Backend API: http://localhost:8000")
        print(f"📚 API Docs: http://localhost:8000/docs")
        print(f"")
        print(f"Press Ctrl+C to stop the server")
        print(f"=" * 50)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n🛑 Server stopped")
            sys.exit(0)

if __name__ == "__main__":
    main()

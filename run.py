#!/usr/bin/env python3
"""
Image Similarity Analyzer - Main Entry Point
Run this script to start the Flask application
"""

import os
import sys
from app import app

def main():
    """Main entry point for the application"""
    print("🔍 Image Similarity Analyzer")
    print("=" * 40)
    print("Starting Flask application...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    # Create necessary directories
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()



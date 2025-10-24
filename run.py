#!/usr/bin/env python3
"""
Simple Azure App Service startup for TC Forecast Dashboard
"""

import os
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Import the Dash app
    from app import app
    
    # Azure App Service entry point
    if __name__ == '__main__':
        # Get port from environment variable (Azure sets this)
        port = int(os.environ.get('PORT', 10000))
        print(f"Starting TC Forecast Dashboard on port {port}")
        app.run(debug=False, host='0.0.0.0', port=port)
        
except Exception as e:
    print(f"Error starting application: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
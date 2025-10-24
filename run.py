#!/usr/bin/env python3
"""
Azure App Service entry point for TC Forecast Visualization Dashboard
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main app
from app import app

# Azure App Service entry point
if __name__ == '__main__':
    # Azure App Service will set the PORT environment variable
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)

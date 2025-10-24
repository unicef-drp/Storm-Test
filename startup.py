#!/usr/bin/env python3
"""
Azure App Service startup script for TC Forecast Visualization Dashboard
"""

import os
from app import app

if __name__ == '__main__':
    # Azure App Service will set the PORT environment variable
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)

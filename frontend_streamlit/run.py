#!/usr/bin/env python3
"""
Streamlit Frontend Başlatma Scripti
"""

import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Streamlit'i başlat
    app_path = Path(__file__).parent / "app.py"
    port = int(os.getenv("STREAMLIT_PORT", "8501"))
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ])


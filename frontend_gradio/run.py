#!/usr/bin/env python3
"""
Gradio Frontend Başlatma Scripti
"""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "app.py"
    port = int(os.getenv("GRADIO_PORT", "7860"))
    
    # app.py'yi import et ve çalıştır
    sys.path.insert(0, str(app_path.parent))
    from app import build_ui
    
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )


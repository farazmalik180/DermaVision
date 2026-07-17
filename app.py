"""
This is a root-level pointer script explicitly created for Hugging Face Spaces.
Hugging Face expects an `app.py` in the root directory for Streamlit Spaces.
This script simply executes our actual Streamlit app located in `streamlit_app/app.py`.
"""
import os
import sys

# Ensure the backend and streamlit_app folders are in the python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "streamlit_app"))

# Path to the real Streamlit application
real_app_path = os.path.join(os.path.dirname(__file__), "streamlit_app", "app.py")

# Execute the real app in the current namespace so Streamlit renders it
with open(real_app_path, encoding="utf-8") as f:
    exec(f.read())

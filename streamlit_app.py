# This is the main entry point for Streamlit Cloud
# It imports and runs your main dashboard

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import your main dashboard
from data_analysis_app import main

if __name__ == "__main__":
    main()

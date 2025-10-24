"""Configuration management for ARC-Prometheus.

Loads environment variables from .env file and provides
centralized access to configuration values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in environment variables.\n"
        "Please create a .env file in the project root with your API key.\n"
        "Example: GEMINI_API_KEY=your_key_here\n"
        "Get your key from: https://makersuite.google.com/app/apikey"
    )

# Data paths
DATA_DIR = project_root / "data" / "arc-prize-2025"

if not DATA_DIR.exists():
    import warnings
    warnings.warn(
        f"ARC dataset directory not found at {DATA_DIR}\n"
        f"Please download the dataset from: https://www.kaggle.com/competitions/arc-prize-2025/data\n"
        f"And place it in: {DATA_DIR}"
    )

# Execution configuration
DEFAULT_TIMEOUT_SECONDS = 5

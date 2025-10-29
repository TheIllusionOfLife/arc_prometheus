"""Configuration management for ARC-Prometheus.

Loads environment variables from .env file and provides
centralized access to configuration values.
"""

import os
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _find_project_root() -> Path:
    """Find project root by searching for pyproject.toml.

    Searches upward from this file's location until pyproject.toml is found.
    Falls back to the 4-parent assumption if not found (for robustness).
    """
    current = Path(__file__).parent
    for _ in range(5):  # Search up to 5 levels
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    # Fallback to original behavior
    return Path(__file__).parent.parent.parent.parent


# Load .env file from project root
project_root = _find_project_root()
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Gemini API Configuration (lazy validation)
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_gemini_api_key() -> str:
    """Get Gemini API key with validation.

    Raises:
        ValueError: If GEMINI_API_KEY is not configured

    Returns:
        The API key string
    """
    if not _GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables.\n"
            "Please create a .env file in the project root with your API key.\n"
            "Example: GEMINI_API_KEY=your_key_here\n"
            "Get your key from: https://makersuite.google.com/app/apikey"
        )
    return _GEMINI_API_KEY


# Data paths
DATA_DIR = project_root / "data" / "arc-prize-2025"

if not DATA_DIR.exists():
    warnings.warn(
        f"ARC dataset directory not found at {DATA_DIR}\n"
        f"Please download the dataset from: https://www.kaggle.com/competitions/arc-prize-2025/data\n"
        f"And place it in: {DATA_DIR}",
        stacklevel=2,
    )

# Execution configuration
DEFAULT_TIMEOUT_SECONDS = 5

# LLM Model Configuration
MODEL_NAME: str = "gemini-2.5-flash-lite"  # Latest, fastest Gemini model

# LLM Generation Parameters
# Temperature: Lower = more deterministic, Higher = more creative
# max_output_tokens: Maximum tokens in generated response

# Type as Any to avoid mypy errors with GenerationConfigDict
# The dict structure matches GenerationConfigDict at runtime
PROGRAMMER_GENERATION_CONFIG: Any = {
    "temperature": 0.3,  # Lower temp for consistent code generation
    "max_output_tokens": 2048,  # Enough for complex solvers
}

REFINER_GENERATION_CONFIG: Any = {
    "temperature": 0.4,  # Slightly higher for debugging creativity
    "max_output_tokens": 3048,  # More tokens to allow detailed fixes
}

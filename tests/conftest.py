"""Pytest configuration and fixtures for ARC-Prometheus tests."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def set_test_env_vars():
    """Set required environment variables for all tests.

    This fixture runs automatically before any tests and ensures that
    GEMINI_API_KEY is set to prevent errors when initializing agents
    that call get_gemini_api_key() in their __init__().

    The actual API calls are mocked in individual tests, but having
    the env var set prevents ValueError during agent initialization.
    """
    # Save original value if it exists
    original_key = os.environ.get("GEMINI_API_KEY")

    os.environ["GEMINI_API_KEY"] = "test-api-key-for-ci"
    yield

    # Restore original value or clean up
    if original_key is not None:
        os.environ["GEMINI_API_KEY"] = original_key
    elif "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

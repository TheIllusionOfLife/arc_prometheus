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
    os.environ["GEMINI_API_KEY"] = "test-api-key-for-ci"
    yield
    # Cleanup after all tests
    if (
        "GEMINI_API_KEY" in os.environ
        and os.environ["GEMINI_API_KEY"] == "test-api-key-for-ci"
    ):
        del os.environ["GEMINI_API_KEY"]

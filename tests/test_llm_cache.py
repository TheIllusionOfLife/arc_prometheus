"""
Tests for LLM response caching module.

Following TDD approach: tests written BEFORE implementation.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import Optional
import tempfile
import time

import pytest

from arc_prometheus.utils.llm_cache import (
    LLMCache,
    CacheStatistics,
    get_cache,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Provide temporary directory for cache testing."""
    return tmp_path / "test_cache"


@pytest.fixture
def cache(temp_cache_dir: Path) -> LLMCache:
    """Provide clean LLMCache instance for each test."""
    return LLMCache(cache_dir=temp_cache_dir, ttl_days=7)


# ============================================================================
# Test Category 1: Cache Initialization (3 tests)
# ============================================================================


def test_cache_creates_directory_if_not_exists(temp_cache_dir: Path) -> None:
    """Cache should create directory if it doesn't exist."""
    assert not temp_cache_dir.exists()

    cache = LLMCache(cache_dir=temp_cache_dir)

    assert temp_cache_dir.exists()
    assert (temp_cache_dir / "llm_cache.db").exists()


def test_cache_uses_default_location() -> None:
    """Cache should use ~/.arc_prometheus by default."""
    cache = LLMCache()

    expected_path = Path.home() / ".arc_prometheus" / "llm_cache.db"
    assert cache.db_path == expected_path


def test_cache_accepts_custom_path(temp_cache_dir: Path) -> None:
    """Cache should accept custom directory path."""
    cache = LLMCache(cache_dir=temp_cache_dir)

    assert cache.db_path == temp_cache_dir / "llm_cache.db"


# ============================================================================
# Test Category 2: Cache Key Generation (4 tests)
# ============================================================================


def test_prompt_hash_is_deterministic(cache: LLMCache) -> None:
    """Same prompt should generate same hash."""
    prompt = "Generate a solver for this task"
    model = "gemini-2.5-flash-lite"
    temp = 0.3

    hash1 = cache._generate_cache_key(prompt, model, temp)
    hash2 = cache._generate_cache_key(prompt, model, temp)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 produces 64 hex chars


def test_different_prompts_different_hashes(cache: LLMCache) -> None:
    """Different prompts should generate different hashes."""
    model = "gemini-2.5-flash-lite"
    temp = 0.3

    hash1 = cache._generate_cache_key("Prompt A", model, temp)
    hash2 = cache._generate_cache_key("Prompt B", model, temp)

    assert hash1 != hash2


def test_hash_normalizes_whitespace(cache: LLMCache) -> None:
    """Hash should normalize whitespace for consistency."""
    model = "gemini-2.5-flash-lite"
    temp = 0.3

    hash1 = cache._generate_cache_key("Generate   a  solver", model, temp)
    hash2 = cache._generate_cache_key("Generate a solver", model, temp)

    assert hash1 == hash2


def test_hash_includes_model_and_temperature(cache: LLMCache) -> None:
    """Hash should include model name and temperature."""
    prompt = "Generate a solver"

    hash1 = cache._generate_cache_key(prompt, "gemini-2.5-flash-lite", 0.3)
    hash2 = cache._generate_cache_key(prompt, "gemini-2.0-flash-thinking-exp", 0.3)
    hash3 = cache._generate_cache_key(prompt, "gemini-2.5-flash-lite", 0.5)

    # Different models should produce different hashes
    assert hash1 != hash2

    # Different temperatures should produce different hashes
    assert hash1 != hash3


# ============================================================================
# Test Category 3: Cache Hit/Miss (5 tests)
# ============================================================================


def test_cache_miss_returns_none(cache: LLMCache) -> None:
    """First call should return None (cache miss)."""
    result = cache.get(
        prompt="Test prompt",
        model_name="gemini-2.5-flash-lite",
        temperature=0.3,
    )

    assert result is None


def test_cache_hit_returns_cached_response(cache: LLMCache) -> None:
    """Second call should return cached value."""
    prompt = "Generate a solver"
    model = "gemini-2.5-flash-lite"
    temp = 0.3
    response = "def solve(grid): return grid * 2"

    # Store in cache
    cache.set(prompt, response, model, temp)

    # Retrieve from cache
    cached_response = cache.get(prompt, model, temp)

    assert cached_response == response


def test_cache_increments_hit_count(cache: LLMCache) -> None:
    """Cache hits should increment hit_count."""
    prompt = "Test prompt"
    model = "gemini-2.5-flash-lite"
    temp = 0.3

    # Store
    cache.set(prompt, "response", model, temp)

    # Hit 3 times
    cache.get(prompt, model, temp)
    cache.get(prompt, model, temp)
    cache.get(prompt, model, temp)

    # Check statistics
    stats = cache.get_statistics()
    assert stats.hit_count == 3


def test_different_model_causes_miss(cache: LLMCache) -> None:
    """Different model should cause cache miss."""
    prompt = "Test prompt"
    temp = 0.3

    # Store with model A
    cache.set(prompt, "response", "gemini-2.5-flash-lite", temp)

    # Try to get with model B
    result = cache.get(prompt, "gemini-2.0-flash-thinking-exp", temp)

    assert result is None


def test_different_temperature_causes_miss(cache: LLMCache) -> None:
    """Different temperature should cause cache miss."""
    prompt = "Test prompt"
    model = "gemini-2.5-flash-lite"

    # Store with temp 0.3
    cache.set(prompt, "response", model, 0.3)

    # Try to get with temp 0.5
    result = cache.get(prompt, model, 0.5)

    assert result is None


# ============================================================================
# Test Category 4: TTL (Time-To-Live) (4 tests)
# ============================================================================


def test_cache_entry_has_expiration(cache: LLMCache, temp_cache_dir: Path) -> None:
    """Stored entries should have expiration timestamp."""
    import sqlite3

    cache.set("prompt", "response", "model", 0.3)

    # Check database directly
    with sqlite3.connect(temp_cache_dir / "llm_cache.db") as conn:
        cursor = conn.execute("SELECT expires_at FROM llm_cache")
        expires_at_str = cursor.fetchone()[0]

        expires_at = datetime.fromisoformat(expires_at_str)
        assert isinstance(expires_at, datetime)


def test_default_ttl_is_7_days(cache: LLMCache, temp_cache_dir: Path) -> None:
    """Default TTL should be 7 days."""
    import sqlite3

    before = datetime.now(UTC)
    cache.set("prompt", "response", "model", 0.3)
    after = datetime.now(UTC)

    # Check database
    with sqlite3.connect(temp_cache_dir / "llm_cache.db") as conn:
        cursor = conn.execute("SELECT created_at, expires_at FROM llm_cache")
        created_at_str, expires_at_str = cursor.fetchone()

        created_at = datetime.fromisoformat(created_at_str)
        expires_at = datetime.fromisoformat(expires_at_str)

        # TTL should be approximately 7 days
        ttl = expires_at - created_at
        assert 6.99 <= ttl.days <= 7.01  # Allow small timing variance


def test_custom_ttl_can_be_set(temp_cache_dir: Path) -> None:
    """Custom TTL should override default."""
    import sqlite3

    cache = LLMCache(cache_dir=temp_cache_dir, ttl_days=14)
    cache.set("prompt", "response", "model", 0.3)

    with sqlite3.connect(temp_cache_dir / "llm_cache.db") as conn:
        cursor = conn.execute("SELECT created_at, expires_at FROM llm_cache")
        created_at_str, expires_at_str = cursor.fetchone()

        created_at = datetime.fromisoformat(created_at_str)
        expires_at = datetime.fromisoformat(expires_at_str)

        ttl = expires_at - created_at
        assert 13.99 <= ttl.days <= 14.01


def test_expired_entries_return_miss(cache: LLMCache, temp_cache_dir: Path) -> None:
    """Expired entries should return None (cache miss)."""
    import sqlite3

    # Store entry
    cache.set("prompt", "response", "model", 0.3)

    # Manually expire the entry by setting expires_at to past
    past_time = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    with sqlite3.connect(temp_cache_dir / "llm_cache.db") as conn:
        conn.execute(
            "UPDATE llm_cache SET expires_at = ?",
            (past_time,),
        )

    # Should return None (expired)
    result = cache.get("prompt", "model", 0.3)
    assert result is None


# ============================================================================
# Test Category 5: Cache Statistics (4 tests)
# ============================================================================


def test_get_statistics_returns_summary(cache: LLMCache) -> None:
    """get_statistics should return CacheStatistics object."""
    stats = cache.get_statistics()

    assert isinstance(stats, CacheStatistics)
    assert hasattr(stats, "total_entries")
    assert hasattr(stats, "hit_count")
    assert hasattr(stats, "miss_count")
    assert hasattr(stats, "hit_rate")
    assert hasattr(stats, "cache_size_mb")
    assert hasattr(stats, "estimated_cost_saved_usd")


def test_statistics_track_hit_rate(cache: LLMCache) -> None:
    """Statistics should calculate hit rate correctly."""
    # Store 2 entries
    cache.set("prompt1", "response1", "model", 0.3)
    cache.set("prompt2", "response2", "model", 0.3)

    # Hit first entry 3 times
    cache.get("prompt1", "model", 0.3)
    cache.get("prompt1", "model", 0.3)
    cache.get("prompt1", "model", 0.3)

    # Hit second entry 1 time
    cache.get("prompt2", "model", 0.3)

    stats = cache.get_statistics()
    assert stats.hit_count == 4
    assert stats.total_entries == 2


def test_statistics_show_cache_size(cache: LLMCache) -> None:
    """Statistics should show cache size in MB."""
    # Add some entries
    for i in range(10):
        cache.set(f"prompt{i}", f"response{i}" * 100, "model", 0.3)

    stats = cache.get_statistics()
    assert stats.cache_size_mb > 0


def test_statistics_show_cost_saved(cache: LLMCache) -> None:
    """Statistics should estimate cost saved."""
    # Store and hit multiple times
    cache.set("prompt", "response", "model", 0.3)

    for _ in range(10):
        cache.get("prompt", "model", 0.3)

    stats = cache.get_statistics()
    assert stats.estimated_cost_saved_usd > 0
    # 10 hits * $0.0001 (500 tokens * $0.0002 per 1K) = $0.001
    assert stats.estimated_cost_saved_usd >= 0.0009  # Allow rounding


# ============================================================================
# Test Category 6: Cache Management (3 tests)
# ============================================================================


def test_clear_cache_removes_all_entries(cache: LLMCache) -> None:
    """clear() should remove all entries."""
    # Add entries
    cache.set("prompt1", "response1", "model", 0.3)
    cache.set("prompt2", "response2", "model", 0.3)

    stats_before = cache.get_statistics()
    assert stats_before.total_entries == 2

    # Clear
    cache.clear()

    stats_after = cache.get_statistics()
    assert stats_after.total_entries == 0


def test_clear_expired_removes_only_expired(cache: LLMCache, temp_cache_dir: Path) -> None:
    """clear_expired() should remove only expired entries."""
    import sqlite3

    # Add 3 entries
    cache.set("prompt1", "response1", "model", 0.3)
    cache.set("prompt2", "response2", "model", 0.3)
    cache.set("prompt3", "response3", "model", 0.3)

    # Expire first 2 entries
    past_time = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    with sqlite3.connect(temp_cache_dir / "llm_cache.db") as conn:
        conn.execute(
            """
            UPDATE llm_cache
            SET expires_at = ?
            WHERE prompt_preview LIKE 'prompt1%' OR prompt_preview LIKE 'prompt2%'
            """,
            (past_time,),
        )

    # Clear expired
    deleted_count = cache.clear_expired()

    assert deleted_count == 2
    stats = cache.get_statistics()
    assert stats.total_entries == 1


def test_cache_is_thread_safe(cache: LLMCache) -> None:
    """Cache operations should be thread-safe."""
    results = []

    def store_and_retrieve(i: int) -> None:
        cache.set(f"prompt{i}", f"response{i}", "model", 0.3)
        result = cache.get(f"prompt{i}", "model", 0.3)
        results.append(result)

    # Run 10 threads concurrently
    threads = [Thread(target=store_and_retrieve, args=(i,)) for i in range(10)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # All threads should succeed
    assert len(results) == 10
    assert all(result is not None for result in results)


# ============================================================================
# Test Category 7: Integration Tests (2 tests)
# ============================================================================


def test_cache_with_programmer_integration(cache: LLMCache) -> None:
    """Cache should work with Programmer workflow."""
    # Simulate programmer workflow
    prompt = "Generate solver for task X"
    model = "gemini-2.5-flash-lite"
    temp = 0.3

    # First call - cache miss
    cached = cache.get(prompt, model, temp)
    assert cached is None

    # Simulate API response
    api_response = "def solve(grid): return grid * 2"
    cache.set(prompt, api_response, model, temp)

    # Second call - cache hit
    cached = cache.get(prompt, model, temp)
    assert cached == api_response


def test_cache_with_refiner_integration(cache: LLMCache) -> None:
    """Cache should work with Refiner workflow."""
    # Simulate refiner workflow
    prompt = "Fix this code: def solve(grid): pass"
    model = "gemini-2.5-flash-lite"
    temp = 0.4  # Refiner uses higher temp

    # First call - cache miss
    cached = cache.get(prompt, model, temp)
    assert cached is None

    # Simulate API response
    api_response = "def solve(grid): return grid + 1"
    cache.set(prompt, api_response, model, temp)

    # Second call - cache hit
    cached = cache.get(prompt, model, temp)
    assert cached == api_response


# ============================================================================
# Test Category 8: Module-Level Singleton (2 tests)
# ============================================================================


def test_get_cache_returns_singleton() -> None:
    """get_cache() should return singleton instance."""
    cache1 = get_cache()
    cache2 = get_cache()

    assert cache1 is cache2


def test_singleton_persists_across_calls() -> None:
    """Singleton cache should persist data across get_cache() calls."""
    cache1 = get_cache()
    cache1.set("test_prompt", "test_response", "model", 0.3)

    cache2 = get_cache()
    result = cache2.get("test_prompt", "model", 0.3)

    assert result == "test_response"

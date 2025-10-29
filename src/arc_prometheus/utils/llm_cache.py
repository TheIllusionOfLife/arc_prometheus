"""
LLM Response Caching Module

Provides persistent caching of LLM API responses to reduce costs and latency.
Uses SQLite for storage with TTL-based expiration.

Usage:
    cache = LLMCache()

    # Check cache before API call
    cached_response = cache.get(prompt, model_name, temperature)
    if cached_response is None:
        response = call_llm_api(prompt)
        cache.set(prompt, response, model_name, temperature)
    else:
        response = cached_response

    # View statistics
    stats = cache.get_statistics()
    print(f"Hit rate: {stats.hit_rate:.1%}")
    print(f"Cost saved: ${stats.estimated_cost_saved_usd:.2f}")
"""

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path


@dataclass
class CacheStatistics:
    """Statistics about cache performance."""

    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    cache_size_mb: float
    estimated_cost_saved_usd: float
    oldest_entry: datetime | None
    newest_entry: datetime | None


class LLMCache:
    """Thread-safe LLM response cache with TTL support."""

    DEFAULT_TTL_DAYS = 7
    DEFAULT_CACHE_DIR = Path.home() / ".arc_prometheus"
    DEFAULT_DB_NAME = "llm_cache.db"

    # Gemini API pricing (approximate for gemini-2.5-flash-lite)
    COST_PER_1K_TOKENS = 0.0002  # $0.0002 per 1K tokens
    AVG_RESPONSE_TOKENS = 500  # Conservative estimate

    def __init__(
        self,
        cache_dir: Path | None = None,
        ttl_days: int = DEFAULT_TTL_DAYS,
    ) -> None:
        """
        Initialize cache with optional custom location and TTL.

        Args:
            cache_dir: Directory for cache storage (default: ~/.arc_prometheus)
            ttl_days: Time-to-live in days (default: 7)
        """
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.cache_dir / self.DEFAULT_DB_NAME
        self.ttl_days = ttl_days
        self._lock = threading.Lock()

        self._init_database()

    def _init_database(self) -> None:
        """Create database schema if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    prompt_preview TEXT NOT NULL,
                    response TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON llm_cache(expires_at)
            """)

    def _generate_cache_key(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
    ) -> str:
        """
        Generate deterministic cache key from prompt and parameters.

        Args:
            prompt: LLM prompt text
            model_name: Model identifier (e.g., "gemini-2.5-flash-lite")
            temperature: Generation temperature

        Returns:
            SHA256 hash (64 hex characters)
        """
        # Normalize whitespace for consistent hashing
        normalized_prompt = " ".join(prompt.split())

        # Include model and temperature in hash
        key_data = f"{normalized_prompt}|{model_name}|{temperature:.2f}"
        return sha256(key_data.encode("utf-8")).hexdigest()

    def get(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
    ) -> str | None:
        """
        Retrieve cached response if available and not expired.

        Args:
            prompt: LLM prompt text
            model_name: Model identifier
            temperature: Generation temperature

        Returns:
            Cached response text, or None on cache miss or expired entry.
            Increments hit_count on successful hit.
        """
        cache_key = self._generate_cache_key(prompt, model_name, temperature)

        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                    SELECT response, expires_at
                    FROM llm_cache
                    WHERE prompt_hash = ?
                    """,
                (cache_key,),
            )
            row = cursor.fetchone()

            if row is None:
                return None  # Cache miss

            response: str
            response, expires_at_str = row
            expires_at = datetime.fromisoformat(expires_at_str)

            # Check if expired
            if datetime.now(UTC) >= expires_at:
                return None  # Expired

            # Increment hit count
            conn.execute(
                """
                    UPDATE llm_cache
                    SET hit_count = hit_count + 1
                    WHERE prompt_hash = ?
                    """,
                (cache_key,),
            )

            return response

    def set(
        self,
        prompt: str,
        response: str,
        model_name: str,
        temperature: float,
        ttl_days: int | None = None,
    ) -> None:
        """
        Store response in cache with TTL.

        Args:
            prompt: LLM prompt text
            response: LLM response text
            model_name: Model identifier
            temperature: Generation temperature
            ttl_days: Custom TTL in days (default: use instance TTL)
        """
        cache_key = self._generate_cache_key(prompt, model_name, temperature)
        ttl = ttl_days if ttl_days is not None else self.ttl_days

        # Create prompt preview (first 200 chars)
        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt

        created_at = datetime.now(UTC)
        expires_at = created_at + timedelta(days=ttl)

        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                    INSERT OR REPLACE INTO llm_cache
                    (prompt_hash, prompt_preview, response, model_name,
                     temperature, created_at, expires_at, hit_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                (
                    cache_key,
                    prompt_preview,
                    response,
                    model_name,
                    temperature,
                    created_at.isoformat(),
                    expires_at.isoformat(),
                ),
            )

    def get_statistics(self) -> CacheStatistics:
        """
        Get comprehensive cache statistics.

        Returns:
            CacheStatistics with performance metrics and cost estimates.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_entries,
                    SUM(hit_count) as total_hits,
                    MIN(created_at) as oldest,
                    MAX(created_at) as newest
                FROM llm_cache
            """)
            row = cursor.fetchone()

            total_entries = row[0] or 0
            total_hits = row[1] or 0
            oldest = datetime.fromisoformat(row[2]) if row[2] else None
            newest = datetime.fromisoformat(row[3]) if row[3] else None

            # Calculate miss count (entries with 0 hits that were created)
            # This is an approximation - true miss count would need separate tracking
            miss_count = (
                max(0, total_entries - total_hits) if total_hits > 0 else total_entries
            )

            # Calculate hit rate
            total_accesses = total_hits + miss_count
            hit_rate = (total_hits / total_accesses) if total_accesses > 0 else 0.0

            # Estimate cache size
            cache_size_mb = (
                self.db_path.stat().st_size / (1024 * 1024)
                if self.db_path.exists()
                else 0.0
            )

            # Estimate cost saved (hits * average response cost)
            cost_per_response = (
                self.AVG_RESPONSE_TOKENS / 1000
            ) * self.COST_PER_1K_TOKENS
            estimated_cost_saved = total_hits * cost_per_response

            return CacheStatistics(
                total_entries=total_entries,
                hit_count=total_hits,
                miss_count=miss_count,
                hit_rate=hit_rate,
                cache_size_mb=cache_size_mb,
                estimated_cost_saved_usd=estimated_cost_saved,
                oldest_entry=oldest,
                newest_entry=newest,
            )

    def clear(self) -> None:
        """Remove all cache entries."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM llm_cache")

    def clear_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries deleted.
        """
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                    DELETE FROM llm_cache
                    WHERE expires_at < ?
                    """,
                (datetime.now(UTC).isoformat(),),
            )
            return cursor.rowcount


# Module-level singleton instance
_cache_instance: LLMCache | None = None


def get_cache() -> LLMCache:
    """
    Get or create the global cache instance.

    Returns:
        Singleton LLMCache instance using default location.
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = LLMCache()
    return _cache_instance

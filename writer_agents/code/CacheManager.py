"""Cache manager for aggressive caching of agent results.

Provides:
- Deterministic agent result caching by input hash
- LLM response caching for identical prompts (session-level)
- SQLite-backed persistent cache using artifacts table
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


class CacheManager:
    """Manages caching of agent results to minimize costs.
    
    Uses the SQLite artifacts table as the cache store, allowing
    results to persist across sessions and be shared between runs.
    """
    
    def __init__(self, db_path: str | Path):
        """Initialize cache manager.
        
        Args:
            db_path: Path to SQLite database (same as job_persistence)
        """
        self.db_path = Path(db_path)
        self._ensure_cache_table()
    
    def _ensure_cache_table(self) -> None:
        """Ensure cache table exists in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create cache table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_cache (
                    cache_key TEXT PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hit_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_agent_input
                ON agent_cache(agent_type, input_hash)
            """)
            
            conn.commit()
    
    def get(
        self,
        agent_type: str,
        input_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Get cached result for agent and input.
        
        Args:
            agent_type: Type of agent
            input_data: Agent input data
            
        Returns:
            Cached result or None if not found
        """
        input_hash = self._hash_input(input_data)
        cache_key = f"{agent_type}:{input_hash}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT result_json FROM agent_cache
                WHERE cache_key = ?
            """, (cache_key,))
            
            row = cursor.fetchone()
            
            if row:
                # Update hit count and access time
                cursor.execute("""
                    UPDATE agent_cache
                    SET hit_count = hit_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE cache_key = ?
                """, (cache_key,))
                conn.commit()
                
                # Parse and return result
                return json.loads(row[0])
        
        return None
    
    def put(
        self,
        agent_type: str,
        input_data: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Store result in cache.
        
        Args:
            agent_type: Type of agent
            input_data: Agent input data
            result: Agent result to cache
        """
        input_hash = self._hash_input(input_data)
        cache_key = f"{agent_type}:{input_hash}"
        result_json = json.dumps(result)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert or replace
            cursor.execute("""
                INSERT OR REPLACE INTO agent_cache (
                    cache_key, agent_type, input_hash, result_json, hit_count
                )
                VALUES (?, ?, ?, ?, 0)
            """, (cache_key, agent_type, input_hash, result_json))
            
            conn.commit()
    
    def invalidate(self, agent_type: Optional[str] = None) -> int:
        """Invalidate cache entries.
        
        Args:
            agent_type: If provided, only invalidate for this agent type.
                       If None, invalidate all cache.
            
        Returns:
            Number of entries invalidated
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if agent_type:
                cursor.execute("""
                    DELETE FROM agent_cache
                    WHERE agent_type = ?
                """, (agent_type,))
            else:
                cursor.execute("DELETE FROM agent_cache")
            
            deleted = cursor.rowcount
            conn.commit()
            
        return deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total entries
            cursor.execute("SELECT COUNT(*) FROM agent_cache")
            total_entries = cursor.fetchone()[0]
            
            # Entries by agent type
            cursor.execute("""
                SELECT agent_type, COUNT(*), SUM(hit_count)
                FROM agent_cache
                GROUP BY agent_type
            """)
            by_agent = {
                row[0]: {"entries": row[1], "hits": row[2]}
                for row in cursor.fetchall()
            }
            
            # Total hits
            cursor.execute("SELECT SUM(hit_count) FROM agent_cache")
            total_hits = cursor.fetchone()[0] or 0
            
        return {
            "total_entries": total_entries,
            "total_hits": total_hits,
            "by_agent_type": by_agent,
            "hit_rate": total_hits / max(total_entries, 1),
        }
    
    def _hash_input(self, input_data: Dict[str, Any]) -> str:
        """Compute hash of input data for cache key.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            SHA-256 hash hex string
        """
        # Serialize to stable JSON
        json_str = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


class SessionCache:
    """In-memory cache for a single session (faster than DB).
    
    Used for caching within a single pipeline run. Results are
    not persisted beyond the session.
    """
    
    def __init__(self):
        """Initialize session cache."""
        self._cache: Dict[str, Any] = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        
        self._misses += 1
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = value
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(total, 1),
        }


class CostTracker:
    """Context manager for tracking costs during agent execution."""
    
    def __init__(self):
        """Initialize cost tracker."""
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost = 0.0
        self.duration = 0.0
        self._start_time = None
    
    def __enter__(self):
        """Enter context."""
        import time
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        import time
        self.duration = time.time() - self._start_time
        return False
    
    def record(
        self,
        tokens_in: int,
        tokens_out: int,
        cost: float,
    ) -> None:
        """Record execution metrics.
        
        Args:
            tokens_in: Input tokens
            tokens_out: Output tokens
            cost: Cost in dollars
        """
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost += cost
    
    def get_cost(self) -> float:
        """Get total cost.
        
        Returns:
            Cost in dollars
        """
        return self.cost


"""
Caching utilities for the agentic-reviewer system.
"""

import time
import hashlib
import logging
import json
import os
import pickle
from typing import Dict, Any, Optional, List
from threading import Lock
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    value: Any
    expires_at: float
    created_at: float
    access_count: int = 0
    last_accessed: float = 0.0
    size_bytes: int = 0


class AdvancedCache:
    """Advanced in-memory cache with TTL, size limits, and persistence support."""
    
    def __init__(self, 
                 default_ttl: int = 3600,
                 max_size_mb: int = 100,
                 max_entries: int = 10000,
                 enable_persistence: bool = False,
                 persistence_file: str = "cache.pkl"):
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
        self.max_entries = max_entries
        self.enable_persistence = enable_persistence
        self.persistence_file = persistence_file
        self.lock = Lock()
        self.current_size_bytes = 0
        
        # Load from persistence if enabled
        if self.enable_persistence:
            self._load_from_persistence()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate the size of an object in bytes."""
        try:
            # Try to serialize to get size estimate
            serialized = pickle.dumps(obj)
            return len(serialized)
        except (pickle.PickleError, TypeError):
            # Fallback to string representation
            return len(str(obj).encode('utf-8'))
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache with access tracking."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                current_time = time.time()
                
                if current_time < entry.expires_at:
                    # Update access metadata
                    entry.access_count += 1
                    entry.last_accessed = current_time
                    
                    logger.debug(f"Cache hit for key: {key}")
                    return entry.value
                else:
                    # Expired, remove it
                    self._remove_entry(key)
                    logger.debug(f"Cache expired for key: {key}")
            
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with size management."""
        with self.lock:
            ttl = ttl or self.default_ttl
            current_time = time.time()
            
            # Estimate size of new entry
            value_size = self._estimate_size(value)
            entry_size = value_size + len(key.encode('utf-8'))
            
            # Check if we need to evict entries
            if not self._can_fit_entry(entry_size):
                self._evict_entries(entry_size)
            
            # If still can't fit, reject the entry
            if not self._can_fit_entry(entry_size):
                logger.warning(f"Cache full, cannot store key: {key}")
                return False
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                expires_at=current_time + ttl,
                created_at=current_time,
                access_count=0,
                last_accessed=current_time,
                size_bytes=entry_size
            )
            
            # Remove existing entry if it exists
            if key in self.cache:
                self._remove_entry(key)
            
            # Add new entry
            self.cache[key] = entry
            self.current_size_bytes += entry_size
            
            logger.debug(f"Cache set for key: {key} with TTL: {ttl}s, size: {entry_size} bytes")
            return True
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                logger.debug(f"Cache deleted for key: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0
            logger.info("Cache cleared")
    
    def size(self) -> int:
        """Get the number of cache entries."""
        with self.lock:
            return len(self.cache)
    
    def memory_usage(self) -> Dict[str, Any]:
        """Get cache memory usage statistics."""
        with self.lock:
            return {
                "entries": len(self.cache),
                "size_bytes": self.current_size_bytes,
                "size_mb": self.current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "max_entries": self.max_entries,
                "utilization_percent": (self.current_size_bytes / self.max_size_bytes) * 100
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if current_time >= entry.expires_at
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        with self.lock:
            if not self.cache:
                return {
                    "entries": 0,
                    "memory_usage": 0,
                    "hit_rate": 0.0,
                    "avg_access_count": 0,
                    "oldest_entry_age": 0,
                    "newest_entry_age": 0
                }
            
            current_time = time.time()
            total_access_count = sum(entry.access_count for entry in self.cache.values())
            avg_access_count = total_access_count / len(self.cache)
            
            entry_ages = [current_time - entry.created_at for entry in self.cache.values()]
            
            return {
                "entries": len(self.cache),
                "memory_usage": self.current_size_bytes,
                "memory_usage_mb": self.current_size_bytes / (1024 * 1024),
                "avg_access_count": avg_access_count,
                "oldest_entry_age": max(entry_ages) if entry_ages else 0,
                "newest_entry_age": min(entry_ages) if entry_ages else 0,
                "utilization_percent": (self.current_size_bytes / self.max_size_bytes) * 100
            }
    
    def _can_fit_entry(self, entry_size: int) -> bool:
        """Check if an entry can fit in the cache."""
        return (self.current_size_bytes + entry_size <= self.max_size_bytes and
                len(self.cache) < self.max_entries)
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry from cache and update size tracking."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_size_bytes -= entry.size_bytes
            del self.cache[key]
    
    def _evict_entries(self, required_space: int) -> None:
        """Evict entries to make space using LRU policy."""
        if not self.cache:
            return
        
        # Sort entries by last accessed time (LRU)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        freed_space = 0
        for key, entry in sorted_entries:
            if freed_space >= required_space:
                break
            
            self._remove_entry(key)
            freed_space += entry.size_bytes
        
        logger.debug(f"Evicted entries to free {freed_space} bytes")
    
    def _save_to_persistence(self) -> None:
        """Save cache to persistent storage."""
        if not self.enable_persistence:
            return
        
        try:
            # Convert cache entries to serializable format
            serializable_cache = {}
            for key, entry in self.cache.items():
                serializable_cache[key] = {
                    "value": entry.value,
                    "expires_at": entry.expires_at,
                    "created_at": entry.created_at,
                    "access_count": entry.access_count,
                    "last_accessed": entry.last_accessed,
                    "size_bytes": entry.size_bytes
                }
            
            with open(self.persistence_file, 'wb') as f:
                pickle.dump(serializable_cache, f)
            
            logger.debug(f"Cache saved to {self.persistence_file}")
            
        except Exception as e:
            logger.error(f"Failed to save cache to persistence: {e}")
    
    def _load_from_persistence(self) -> None:
        """Load cache from persistent storage."""
        if not self.enable_persistence or not os.path.exists(self.persistence_file):
            return
        
        try:
            with open(self.persistence_file, 'rb') as f:
                serializable_cache = pickle.load(f)
            
            current_time = time.time()
            
            for key, entry_data in serializable_cache.items():
                # Only load non-expired entries
                if entry_data["expires_at"] > current_time:
                    entry = CacheEntry(
                        value=entry_data["value"],
                        expires_at=entry_data["expires_at"],
                        created_at=entry_data["created_at"],
                        access_count=entry_data["access_count"],
                        last_accessed=entry_data["last_accessed"],
                        size_bytes=entry_data["size_bytes"]
                    )
                    self.cache[key] = entry
                    self.current_size_bytes += entry.size_bytes
            
            logger.info(f"Loaded {len(self.cache)} entries from persistence")
            
        except Exception as e:
            logger.error(f"Failed to load cache from persistence: {e}")
    
    def save_persistence(self) -> None:
        """Manually save cache to persistence."""
        self._save_to_persistence()


# Global cache instance with improved configuration
_cache = AdvancedCache(
    default_ttl=3600,  # 1 hour
    max_size_mb=100,   # 100 MB limit
    max_entries=10000, # 10k entries limit
    enable_persistence=True,
    persistence_file="outputs/cache.pkl"
)


def get_cache() -> AdvancedCache:
    """Get the global cache instance."""
    return _cache


def cache_result(ttl: Optional[int] = None, key_prefix: str = ""):
    """Decorator to cache function results with improved key generation."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key with prefix
            cache_key = f"{key_prefix}:{func.__name__}:{_cache._get_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = _cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            _cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return _cache.get_stats()


def cleanup_cache() -> int:
    """Clean up expired cache entries."""
    return _cache.cleanup_expired()


def save_cache() -> None:
    """Save cache to persistence."""
    _cache.save_persistence() 
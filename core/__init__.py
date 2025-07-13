"""
Agentic Reviewer Core Package

This package contains the core components for the review system:
- DataLoader: Handles loading and preprocessing classification data
- SampleSelector: Selects samples for review based on various strategies
- ReviewLoop: Main orchestrator for the review process
- AuditLogger: Logs review results to SQLite database
- Config: Centralized configuration management
- Validators: Input validation utilities
- Exceptions: Custom exception types
- LoggingConfig: Logging configuration and setup
- Cache: Caching utilities for performance optimization
- Monitoring: Health checks and metrics collection
"""

# Setup logging first
from .logging_config import setup_logging, get_logger
setup_logging()

from .data_loader import DataLoader
from .sample_selector import SampleSelector
from .review_loop import ReviewLoop
from .logger import AuditLogger
from .config import config, SystemConfig, LLMConfig, SelectionConfig, DatabaseConfig, LoggingConfig, PerformanceConfig, APIConfig
from .validators import (
    validate_text, validate_label, validate_confidence, validate_verdict,
    validate_sample_id, validate_sample, validate_metadata, validate_strategy,
    validate_threshold, validate_sample_size
)
from .exceptions import (
    AgenticReviewerError, DataValidationError, LLMConnectionError, LLMResponseError,
    ConfigurationError, DatabaseError, SampleSelectionError, PromptTemplateError
)
from .cache import AdvancedCache, get_cache, cache_result
from .monitoring import HealthChecker, get_health_checker, SystemMetrics, ApplicationMetrics

__all__ = [
    # Core components
    "DataLoader",
    "SampleSelector", 
    "ReviewLoop",
    "AuditLogger",
    
    # Configuration
    "config",
    "SystemConfig",
    "LLMConfig",
    "SelectionConfig", 
    "DatabaseConfig",
    "LoggingConfig",
    "PerformanceConfig",
    "APIConfig",
    
    # Validators
    "validate_text",
    "validate_label", 
    "validate_confidence",
    "validate_verdict",
    "validate_sample_id",
    "validate_sample",
    "validate_metadata",
    "validate_strategy",
    "validate_threshold",
    "validate_sample_size",
    
    # Exceptions
    "AgenticReviewerError",
    "DataValidationError",
    "LLMConnectionError", 
    "LLMResponseError",
    "ConfigurationError",
    "DatabaseError",
    "SampleSelectionError",
    "PromptTemplateError",
    
    # Logging
    "setup_logging",
    "get_logger",
    
    # Caching
    "AdvancedCache",
    "get_cache",
    "cache_result",
    
    # Monitoring
    "HealthChecker",
    "get_health_checker",
    "SystemMetrics",
    "ApplicationMetrics"
] 
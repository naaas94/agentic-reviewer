"""
Centralized configuration for the agentic-reviewer system.
"""

import os
import secrets
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from .exceptions import ConfigurationError
from .validators import validate_url, validate_api_key, validate_threshold, validate_sample_size


@dataclass
class LLMConfig:
    """Configuration for LLM interactions."""
    model_name: str = "mistral"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 512
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    def __post_init__(self):
        """Validate LLM configuration."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigurationError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        
        if self.max_tokens <= 0:
            raise ConfigurationError(f"Max tokens must be positive, got {self.max_tokens}")
        
        if self.timeout_seconds <= 0:
            raise ConfigurationError(f"Timeout must be positive, got {self.timeout_seconds}")
        
        if self.max_retries <= 0:
            raise ConfigurationError(f"Max retries must be positive, got {self.max_retries}")
        
        if self.retry_delay_seconds <= 0:
            raise ConfigurationError(f"Retry delay must be positive, got {self.retry_delay_seconds}")
        
        # Validate URL format
        try:
            validate_url(self.ollama_url, "ollama_url")
        except Exception as e:
            raise ConfigurationError(f"Invalid Ollama URL: {e}")


@dataclass
class SelectionConfig:
    """Configuration for sample selection strategies."""
    default_threshold: float = 0.7
    default_sample_size: int = 10
    default_seed: int = 42
    max_samples_limit: int = 1000
    
    def __post_init__(self):
        """Validate selection configuration."""
        try:
            validate_threshold(self.default_threshold)
        except Exception as e:
            raise ConfigurationError(f"Invalid default threshold: {e}")
        
        try:
            validate_sample_size(self.default_sample_size)
        except Exception as e:
            raise ConfigurationError(f"Invalid default sample size: {e}")
        
        if self.max_samples_limit <= 0:
            raise ConfigurationError(f"Max samples limit must be positive, got {self.max_samples_limit}")


@dataclass
class DatabaseConfig:
    """Configuration for database operations."""
    db_path: str = "outputs/reviewed_predictions.sqlite"
    connection_timeout: int = 30
    enable_wal_mode: bool = True
    journal_mode: str = "WAL"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    max_backup_files: int = 7
    
    def __post_init__(self):
        """Validate database configuration."""
        if self.connection_timeout <= 0:
            raise ConfigurationError(f"Connection timeout must be positive, got {self.connection_timeout}")
        
        if self.backup_interval_hours <= 0:
            raise ConfigurationError(f"Backup interval must be positive, got {self.backup_interval_hours}")
        
        if self.max_backup_files <= 0:
            raise ConfigurationError(f"Max backup files must be positive, got {self.max_backup_files}")
        
        valid_journal_modes = ["WAL", "DELETE", "TRUNCATE", "PERSIST", "MEMORY", "OFF"]
        if self.journal_mode not in valid_journal_modes:
            raise ConfigurationError(f"Invalid journal mode: {self.journal_mode}. Must be one of {valid_journal_modes}")


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    enable_console_logging: bool = True
    enable_structured_logging: bool = True
    log_rotation: bool = True
    max_log_size_mb: int = 10
    max_log_files: int = 5
    
    def __post_init__(self):
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ConfigurationError(f"Invalid log level: {self.log_level}. Must be one of {valid_levels}")
        
        if self.max_log_size_mb <= 0:
            raise ConfigurationError(f"Max log size must be positive, got {self.max_log_size_mb}")
        
        if self.max_log_files <= 0:
            raise ConfigurationError(f"Max log files must be positive, got {self.max_log_files}")


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    batch_size: int = 10
    max_concurrent_requests: int = 5
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size_mb: int = 100
    cache_max_entries: int = 10000
    enable_cache_persistence: bool = True
    
    def __post_init__(self):
        """Validate performance configuration."""
        if self.batch_size <= 0:
            raise ConfigurationError(f"Batch size must be positive, got {self.batch_size}")
        
        if self.max_concurrent_requests <= 0:
            raise ConfigurationError(f"Max concurrent requests must be positive, got {self.max_concurrent_requests}")
        
        if self.cache_ttl_seconds <= 0:
            raise ConfigurationError(f"Cache TTL must be positive, got {self.cache_ttl_seconds}")
        
        if self.cache_max_size_mb <= 0:
            raise ConfigurationError(f"Cache max size must be positive, got {self.cache_max_size_mb}")
        
        if self.cache_max_entries <= 0:
            raise ConfigurationError(f"Cache max entries must be positive, got {self.cache_max_entries}")


@dataclass
class APIConfig:
    """Configuration for API server."""
    host: str = "0.0.0.0"
    port: int = 8000
    enable_docs: bool = True
    require_auth: bool = False
    api_key: Optional[str] = None
    enable_rate_limiting: bool = True
    rate_limit_max_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour in seconds
    max_request_size: int = 1024 * 1024  # 1MB
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    enable_compression: bool = True
    enable_cors: bool = True
    
    def __post_init__(self):
        """Validate API configuration."""
        if self.port <= 0 or self.port > 65535:
            raise ConfigurationError(f"API port must be between 1 and 65535, got {self.port}")
        
        if self.rate_limit_max_requests <= 0:
            raise ConfigurationError(f"Rate limit max requests must be positive, got {self.rate_limit_max_requests}")
        
        if self.rate_limit_window <= 0:
            raise ConfigurationError(f"Rate limit window must be positive, got {self.rate_limit_window}")
        
        if self.max_request_size <= 0:
            raise ConfigurationError(f"Max request size must be positive, got {self.max_request_size}")
        
        # Validate API key if provided
        if self.api_key:
            try:
                validate_api_key(self.api_key)
            except Exception as e:
                raise ConfigurationError(f"Invalid API key: {e}")
        
        # Validate SSL configuration
        if self.ssl_keyfile and not os.path.exists(self.ssl_keyfile):
            raise ConfigurationError(f"SSL keyfile not found: {self.ssl_keyfile}")
        
        if self.ssl_certfile and not os.path.exists(self.ssl_certfile):
            raise ConfigurationError(f"SSL certfile not found: {self.ssl_certfile}")


@dataclass
class SecurityConfig:
    """Configuration for security features."""
    enable_input_sanitization: bool = True
    enable_output_validation: bool = True
    max_input_length: int = 10000
    allowed_file_extensions: List[str] = field(default_factory=lambda: [".txt", ".csv", ".json"])
    enable_audit_logging: bool = True
    audit_log_file: str = "outputs/audit.log"
    
    def __post_init__(self):
        """Validate security configuration."""
        if self.max_input_length <= 0:
            raise ConfigurationError(f"Max input length must be positive, got {self.max_input_length}")


@dataclass
class SystemConfig:
    """Main system configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    api: APIConfig = field(default_factory=APIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Version and metadata
    version: str = "1.0.0"
    prompt_version: str = "v1.0.0"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._validate_cross_dependencies()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Basic validation is done in individual config classes
        pass
    
    def _validate_cross_dependencies(self):
        """Validate cross-dependencies between configuration sections."""
        # Check that cache settings are consistent
        if self.performance.enable_caching and self.performance.cache_max_entries > 100000:
            raise ConfigurationError("Cache max entries too high for memory safety")
        
        # Check that API settings are consistent
        if self.api.require_auth and not self.api.api_key:
            raise ConfigurationError("API authentication required but no API key provided")
        
        # Check that database path is writable
        db_dir = os.path.dirname(self.database.db_path)
        if not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except OSError as e:
                raise ConfigurationError(f"Cannot create database directory {db_dir}: {e}")
        
        # Check that log directory is writable if logging to file
        if self.logging.log_file:
            log_dir = os.path.dirname(self.logging.log_file)
            if not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except OSError as e:
                    raise ConfigurationError(f"Cannot create log directory {log_dir}: {e}")
    
    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)
    
    def set_api_key(self, api_key: Optional[str] = None) -> None:
        """Set API key, generating one if not provided."""
        if api_key is None:
            api_key = self.generate_api_key()
        
        try:
            validate_api_key(api_key)
            self.api.api_key = api_key
            self.api.require_auth = True
        except Exception as e:
            raise ConfigurationError(f"Invalid API key: {e}")
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv('AR_MODEL_NAME'):
            config.llm.model_name = os.getenv('AR_MODEL_NAME', 'mistral')
        
        if os.getenv('AR_OLLAMA_URL'):
            config.llm.ollama_url = os.getenv('AR_OLLAMA_URL', 'http://localhost:11434')
        
        if os.getenv('AR_TEMPERATURE'):
            temp_value = os.getenv('AR_TEMPERATURE')
            if temp_value is not None:
                config.llm.temperature = float(temp_value)
        
        if os.getenv('AR_MAX_TOKENS'):
            tokens_value = os.getenv('AR_MAX_TOKENS')
            if tokens_value is not None:
                config.llm.max_tokens = int(tokens_value)
        
        if os.getenv('AR_TIMEOUT'):
            timeout_value = os.getenv('AR_TIMEOUT')
            if timeout_value is not None:
                config.llm.timeout_seconds = int(timeout_value)
        
        if os.getenv('AR_DB_PATH'):
            config.database.db_path = os.getenv('AR_DB_PATH', 'outputs/reviewed_predictions.sqlite')
        
        if os.getenv('AR_LOG_LEVEL'):
            config.logging.log_level = os.getenv('AR_LOG_LEVEL', 'INFO')
        
        if os.getenv('AR_BATCH_SIZE'):
            batch_value = os.getenv('AR_BATCH_SIZE')
            if batch_value is not None:
                config.performance.batch_size = int(batch_value)
        
        # API Configuration
        if os.getenv('AR_API_HOST'):
            config.api.host = os.getenv('AR_API_HOST', '0.0.0.0')
        
        if os.getenv('AR_API_PORT'):
            port_value = os.getenv('AR_API_PORT')
            if port_value is not None:
                config.api.port = int(port_value)
        
        if os.getenv('AR_API_KEY'):
            config.api.api_key = os.getenv('AR_API_KEY')
            config.api.require_auth = True
        
        if os.getenv('AR_RATE_LIMIT_MAX'):
            rate_limit_value = os.getenv('AR_RATE_LIMIT_MAX')
            if rate_limit_value is not None:
                config.api.rate_limit_max_requests = int(rate_limit_value)
        
        # Performance Configuration
        if os.getenv('AR_CACHE_MAX_SIZE_MB'):
            cache_size_value = os.getenv('AR_CACHE_MAX_SIZE_MB')
            if cache_size_value is not None:
                config.performance.cache_max_size_mb = int(cache_size_value)
        
        if os.getenv('AR_MAX_CONCURRENT'):
            concurrent_value = os.getenv('AR_MAX_CONCURRENT')
            if concurrent_value is not None:
                config.performance.max_concurrent_requests = int(concurrent_value)
        
        # Security Configuration
        if os.getenv('AR_ENABLE_SANITIZATION'):
            sanitization_value = os.getenv('AR_ENABLE_SANITIZATION')
            if sanitization_value is not None:
                config.security.enable_input_sanitization = sanitization_value.lower() == 'true'
        
        return config


# Global configuration instance
config = SystemConfig.from_env() 
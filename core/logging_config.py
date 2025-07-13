"""
Logging configuration for the agentic-reviewer system.
"""

import logging
import sys
from typing import Optional
from .config import config


def setup_logging(log_level: Optional[str] = None, 
                 log_file: Optional[str] = None,
                 enable_console: Optional[bool] = None) -> None:
    """
    Setup logging configuration for the entire system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_console: Whether to enable console logging
    """
    # Use configuration defaults if not provided
    log_level = log_level or config.logging.log_level
    log_file = log_file or config.logging.log_file
    enable_console = enable_console if enable_console is not None else config.logging.enable_console_logging
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(config.logging.log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler if enabled
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            # Fallback to console if file logging fails
            print(f"Warning: Failed to setup file logging to {log_file}: {e}")
            if not enable_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(numeric_level)
                console_handler.setFormatter(formatter)
                root_logger.addHandler(console_handler)
    
    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, file={log_file}, console={enable_console}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Setup logging when module is imported
setup_logging() 
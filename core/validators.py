"""
Input validation utilities for the agentic-reviewer system.
"""

import re
import html
from typing import Dict, Any, List, Optional
from .exceptions import DataValidationError


def sanitize_text(text: str) -> str:
    """
    Sanitize text input to prevent injection attacks.
    
    Args:
        text: Raw text input
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return ""
    
    # HTML escape
    text = html.escape(text)
    
    # Remove script tags
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove other potentially dangerous HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def validate_text(text: Any, field_name: str = "text") -> str:
    """
    Validate that text is a non-empty string with sanitization.
    
    Args:
        text: The text to validate
        field_name: Name of the field for error messages
        
    Returns:
        The validated and sanitized text
        
    Raises:
        DataValidationError: If text is invalid
    """
    if not isinstance(text, str):
        raise DataValidationError(f"{field_name} must be a string, got {type(text)}")
    
    # Sanitize text
    sanitized_text = sanitize_text(text)
    
    if not sanitized_text:
        raise DataValidationError(f"{field_name} cannot be empty or whitespace only")
    
    # Check for reasonable length (prevent extremely long texts)
    if len(sanitized_text) > 10000:
        raise DataValidationError(f"{field_name} is too long (max 10000 characters)")
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'javascript:',  # JavaScript protocol
        r'data:',        # Data protocol
        r'vbscript:',    # VBScript protocol
        r'<iframe',      # Iframe tags
        r'<object',      # Object tags
        r'<embed',       # Embed tags
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, sanitized_text, re.IGNORECASE):
            raise DataValidationError(f"{field_name} contains suspicious content")
    
    return sanitized_text


def validate_label(label: Any, field_name: str = "label") -> str:
    """
    Validate that label is a non-empty string with sanitization.
    
    Args:
        label: The label to validate
        field_name: Name of the field for error messages
        
    Returns:
        The validated and sanitized label
        
    Raises:
        DataValidationError: If label is invalid
    """
    if not isinstance(label, str):
        raise DataValidationError(f"{field_name} must be a string, got {type(label)}")
    
    # Sanitize label
    sanitized_label = sanitize_text(label)
    
    if not sanitized_label:
        raise DataValidationError(f"{field_name} cannot be empty or whitespace only")
    
    # Check for reasonable length
    if len(sanitized_label) > 100:
        raise DataValidationError(f"{field_name} is too long (max 100 characters)")
    
    # Check for valid characters (alphanumeric, spaces, hyphens, underscores)
    if not re.match(r'^[a-zA-Z0-9\s\-_]+$', sanitized_label):
        raise DataValidationError(f"{field_name} contains invalid characters (only alphanumeric, spaces, hyphens, underscores allowed)")
    
    return sanitized_label


def validate_confidence(confidence: Any, field_name: str = "confidence") -> float:
    """
    Validate that confidence is a number between 0.0 and 1.0.
    
    Args:
        confidence: The confidence value to validate
        field_name: Name of the field for error messages
        
    Returns:
        The validated confidence as float
        
    Raises:
        DataValidationError: If confidence is invalid
    """
    if not isinstance(confidence, (int, float)):
        raise DataValidationError(f"{field_name} must be a number, got {type(confidence)}")
    
    confidence_float = float(confidence)
    
    if not 0.0 <= confidence_float <= 1.0:
        raise DataValidationError(f"{field_name} must be between 0.0 and 1.0, got {confidence_float}")
    
    # Check for NaN or infinity
    if not (confidence_float == confidence_float):  # NaN check
        raise DataValidationError(f"{field_name} cannot be NaN")
    
    if confidence_float == float('inf') or confidence_float == float('-inf'):
        raise DataValidationError(f"{field_name} cannot be infinity")
    
    return confidence_float


def validate_verdict(verdict: Any, field_name: str = "verdict") -> str:
    """
    Validate that verdict is one of the allowed values.
    
    Args:
        verdict: The verdict to validate
        field_name: Name of the field for error messages
        
    Returns:
        The validated verdict
        
    Raises:
        DataValidationError: If verdict is invalid
    """
    valid_verdicts = ["Correct", "Incorrect", "Uncertain"]
    
    if not isinstance(verdict, str):
        raise DataValidationError(f"{field_name} must be a string, got {type(verdict)}")
    
    # Sanitize verdict
    sanitized_verdict = sanitize_text(verdict)
    
    if sanitized_verdict not in valid_verdicts:
        raise DataValidationError(f"{field_name} must be one of {valid_verdicts}, got '{sanitized_verdict}'")
    
    return sanitized_verdict


def validate_sample_id(sample_id: Any, field_name: str = "sample_id") -> str:
    """
    Validate that sample_id is a valid identifier.
    
    Args:
        sample_id: The sample ID to validate
        field_name: Name of the field for error messages
        
    Returns:
        The validated sample ID
        
    Raises:
        DataValidationError: If sample_id is invalid
    """
    if not isinstance(sample_id, str):
        raise DataValidationError(f"{field_name} must be a string, got {type(sample_id)}")
    
    # Sanitize sample ID
    sanitized_id = sanitize_text(sample_id)
    
    if not sanitized_id:
        raise DataValidationError(f"{field_name} cannot be empty or whitespace only")
    
    # Check for reasonable length
    if len(sanitized_id) > 100:
        raise DataValidationError(f"{field_name} is too long (max 100 characters)")
    
    # Check for valid characters (alphanumeric, underscore, hyphen)
    if not re.match(r'^[a-zA-Z0-9_-]+$', sanitized_id):
        raise DataValidationError(f"{field_name} contains invalid characters (only alphanumeric, underscore, hyphen allowed)")
    
    return sanitized_id


def validate_sample(sample: Any) -> Dict[str, Any]:
    """
    Validate a complete sample dictionary with sanitization.
    
    Args:
        sample: The sample dictionary to validate
        
    Returns:
        The validated sample dictionary
        
    Raises:
        DataValidationError: If sample is invalid
    """
    if not isinstance(sample, dict):
        raise DataValidationError("Sample must be a dictionary")
    
    if not sample:
        raise DataValidationError("Sample cannot be empty")
    
    # Validate required fields
    required_fields = ["text", "pred_label", "confidence"]
    for field in required_fields:
        if field not in sample:
            raise DataValidationError(f"Missing required field: {field}")
    
    # Validate individual fields
    validated_sample = {}
    validated_sample["text"] = validate_text(sample["text"], "text")
    validated_sample["pred_label"] = validate_label(sample["pred_label"], "pred_label")
    validated_sample["confidence"] = validate_confidence(sample["confidence"], "confidence")
    
    # Validate optional fields
    if "id" in sample:
        validated_sample["id"] = validate_sample_id(sample["id"], "id")
    else:
        validated_sample["id"] = "unknown"
    
    return validated_sample


def validate_metadata(metadata: Any) -> Dict[str, Any]:
    """
    Validate metadata dictionary with sanitization.
    
    Args:
        metadata: The metadata dictionary to validate
        
    Returns:
        The validated metadata dictionary
        
    Raises:
        DataValidationError: If metadata is invalid
    """
    if metadata is None:
        return {}
    
    if not isinstance(metadata, dict):
        raise DataValidationError("Metadata must be a dictionary or None")
    
    # Validate specific metadata fields if present
    validated_metadata = {}
    
    if "model_name" in metadata:
        validated_metadata["model_name"] = validate_label(metadata["model_name"], "model_name")
    
    if "prompt_hash" in metadata:
        if not isinstance(metadata["prompt_hash"], str) or len(metadata["prompt_hash"]) > 50:
            raise DataValidationError("prompt_hash must be a string with max 50 characters")
        validated_metadata["prompt_hash"] = sanitize_text(metadata["prompt_hash"])
    
    if "tokens_used" in metadata:
        if not isinstance(metadata["tokens_used"], int) or metadata["tokens_used"] < 0:
            raise DataValidationError("tokens_used must be a non-negative integer")
        validated_metadata["tokens_used"] = metadata["tokens_used"]
    
    if "latency_ms" in metadata:
        if not isinstance(metadata["latency_ms"], (int, float)) or metadata["latency_ms"] < 0:
            raise DataValidationError("latency_ms must be a non-negative number")
        validated_metadata["latency_ms"] = metadata["latency_ms"]
    
    if "run_id" in metadata:
        validated_metadata["run_id"] = validate_sample_id(metadata["run_id"], "run_id")
    
    return validated_metadata


def validate_strategy(strategy: Any) -> str:
    """
    Validate sample selection strategy.
    
    Args:
        strategy: The strategy to validate
        
    Returns:
        The validated strategy
        
    Raises:
        DataValidationError: If strategy is invalid
    """
    valid_strategies = ["low_confidence", "random", "all"]
    
    if not isinstance(strategy, str):
        raise DataValidationError(f"Strategy must be a string, got {type(strategy)}")
    
    # Sanitize strategy
    sanitized_strategy = sanitize_text(strategy)
    
    if sanitized_strategy not in valid_strategies:
        raise DataValidationError(f"Strategy must be one of {valid_strategies}, got '{sanitized_strategy}'")
    
    return sanitized_strategy


def validate_threshold(threshold: Any) -> float:
    """
    Validate threshold value for sample selection.
    
    Args:
        threshold: The threshold to validate
        
    Returns:
        The validated threshold
        
    Raises:
        DataValidationError: If threshold is invalid
    """
    if not isinstance(threshold, (int, float)):
        raise DataValidationError(f"Threshold must be a number, got {type(threshold)}")
    
    threshold_float = float(threshold)
    
    if not 0.0 <= threshold_float <= 1.0:
        raise DataValidationError(f"Threshold must be between 0.0 and 1.0, got {threshold_float}")
    
    # Check for NaN or infinity
    if not (threshold_float == threshold_float):  # NaN check
        raise DataValidationError("Threshold cannot be NaN")
    
    if threshold_float == float('inf') or threshold_float == float('-inf'):
        raise DataValidationError("Threshold cannot be infinity")
    
    return threshold_float


def validate_sample_size(sample_size: Any) -> int:
    """
    Validate sample size for selection.
    
    Args:
        sample_size: The sample size to validate
        
    Returns:
        The validated sample size
        
    Raises:
        DataValidationError: If sample_size is invalid
    """
    if not isinstance(sample_size, int):
        raise DataValidationError(f"Sample size must be an integer, got {type(sample_size)}")
    
    if sample_size <= 0:
        raise DataValidationError(f"Sample size must be positive, got {sample_size}")
    
    if sample_size > 10000:
        raise DataValidationError(f"Sample size too large (max 10000), got {sample_size}")
    
    return sample_size


def validate_api_key(api_key: Any) -> str:
    """
    Validate API key format and security.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        The validated API key
        
    Raises:
        DataValidationError: If API key is invalid
    """
    if not isinstance(api_key, str):
        raise DataValidationError("API key must be a string")
    
    if not api_key.strip():
        raise DataValidationError("API key cannot be empty")
    
    # Check minimum length for security
    if len(api_key) < 16:
        raise DataValidationError("API key must be at least 16 characters long")
    
    # Check for valid characters (alphanumeric, hyphens, underscores)
    if not re.match(r'^[a-zA-Z0-9_-]+$', api_key):
        raise DataValidationError("API key contains invalid characters")
    
    return api_key.strip()


def validate_url(url: Any, field_name: str = "url") -> str:
    """
    Validate URL format.
    
    Args:
        url: The URL to validate
        field_name: Name of the field for error messages
        
    Returns:
        The validated URL
        
    Raises:
        DataValidationError: If URL is invalid
    """
    if not isinstance(url, str):
        raise DataValidationError(f"{field_name} must be a string, got {type(url)}")
    
    if not url.strip():
        raise DataValidationError(f"{field_name} cannot be empty")
    
    # Basic URL validation
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not url_pattern.match(url):
        raise DataValidationError(f"{field_name} is not a valid URL: {url}")
    
    return url.strip() 
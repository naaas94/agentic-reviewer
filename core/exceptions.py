"""
Custom exceptions for the agentic-reviewer system.
"""


class AgenticReviewerError(Exception):
    """Base exception for all agentic-reviewer errors."""
    pass


class DataValidationError(AgenticReviewerError):
    """Raised when data validation fails."""
    pass


class LLMConnectionError(AgenticReviewerError):
    """Raised when LLM service connection fails."""
    pass


class LLMResponseError(AgenticReviewerError):
    """Raised when LLM response is invalid or unexpected."""
    pass


class ConfigurationError(AgenticReviewerError):
    """Raised when configuration is invalid or missing."""
    pass


class DatabaseError(AgenticReviewerError):
    """Raised when database operations fail."""
    pass


class SampleSelectionError(AgenticReviewerError):
    """Raised when sample selection fails."""
    pass


class PromptTemplateError(AgenticReviewerError):
    """Raised when prompt template operations fail."""
    pass


class SecurityError(AgenticReviewerError):
    """Raised when security validation fails."""
    pass 
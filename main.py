"""
FastAPI server for agentic-reviewer API endpoints with security enhancements.
"""

import time
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from core.review_loop import ReviewLoop
from core.logger import AuditLogger
from core.config import config
from core.validators import validate_text, validate_label, validate_confidence, sanitize_text
from core.cache import get_cache_stats, cleanup_cache, save_cache


# Initialize FastAPI app with security
app = FastAPI(
    title="Agentic Reviewer API",
    description="API for semantic auditing of text classification predictions",
    version="1.0.0",
    docs_url="/docs" if config.api.enable_docs else None,
    redoc_url="/redoc" if config.api.enable_docs else None
)

# Add security middleware
if config.api.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure based on your deployment
)

# Security
security = HTTPBearer(auto_error=False)

# Rate limiting storage (in production, use Redis)
rate_limit_storage = {}

# Initialize components
review_loop = ReviewLoop()
logger = AuditLogger()
import logging
app_logger = logging.getLogger(__name__)


# Pydantic models with enhanced validation
class ReviewRequest(BaseModel):
    text: str = Field(..., description="Input text to review", max_length=10000)
    predicted_label: str = Field(..., description="Predicted label from classifier", max_length=100)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)")
    sample_id: Optional[str] = Field(None, description="Optional sample ID", max_length=100)
    use_unified_agent: bool = Field(True, description="Whether to use unified agent for efficiency")
    
    @validator('text')
    def validate_text(cls, v):
        """Validate and sanitize text input."""
        try:
            return validate_text(v, "text")
        except Exception as e:
            raise ValueError(f"Invalid text: {e}")
    
    @validator('predicted_label')
    def validate_predicted_label(cls, v):
        """Validate and sanitize predicted label."""
        try:
            return validate_label(v, "predicted_label")
        except Exception as e:
            raise ValueError(f"Invalid predicted label: {e}")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence score."""
        try:
            return validate_confidence(v, "confidence")
        except Exception as e:
            raise ValueError(f"Invalid confidence: {e}")
    
    @validator('sample_id')
    def validate_sample_id(cls, v):
        """Validate sample ID if provided."""
        if v is not None:
            try:
                from core.validators import validate_sample_id
                return validate_sample_id(v, "sample_id")
            except Exception as e:
                raise ValueError(f"Invalid sample ID: {e}")
        return v


class ReviewResponse(BaseModel):
    sample_id: str
    verdict: str
    reasoning: str
    suggested_label: Optional[str] = None
    explanation: str
    success: bool
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class StatsResponse(BaseModel):
    total_reviews: int
    verdict_distribution: Dict[str, int]
    avg_confidence: float
    recent_reviews_24h: int
    cache_stats: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime: float
    system_health: Dict[str, Any]
    cache_health: Dict[str, Any]


class MetricsResponse(BaseModel):
    system: Dict[str, Any]
    application: Dict[str, Any]
    summary: Dict[str, Any]
    cache: Dict[str, Any]


# Security functions with improved validation
def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """Verify API key authentication with improved security."""
    if not config.api.require_auth:
        return True
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Validate API key format
    try:
        from core.validators import validate_api_key
        validate_api_key(credentials.credentials)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # In production, use proper API key validation with hashing
    expected_key = config.api.api_key
    if not expected_key or credentials.credentials != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True


def rate_limit_check(request: Request, client_id: str = "default") -> None:
    """Check rate limiting for requests with improved tracking."""
    global rate_limit_storage
    
    if not config.api.enable_rate_limiting:
        return
    
    current_time = time.time()
    window_start = current_time - config.api.rate_limit_window
    
    # Clean old entries
    rate_limit_storage = {k: v for k, v in rate_limit_storage.items() if v > window_start}
    
    # Check rate limit
    client_requests = [t for t in rate_limit_storage.values() if t > window_start]
    if len(client_requests) >= config.api.rate_limit_max_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(config.api.rate_limit_window)}
        )
    
    # Add current request
    rate_limit_storage[f"{client_id}_{current_time}"] = current_time


def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting with improved identification."""
    # Use IP address or API key as client identifier
    client_ip = request.client.host if request.client else "unknown"
    
    # Add user agent for better identification
    user_agent = request.headers.get("user-agent", "unknown")
    user_agent_hash = hashlib.md5(user_agent.encode()).hexdigest()[:8]
    
    return f"{client_ip}_{user_agent_hash}"


# Middleware for request logging and security
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for request logging and validation."""
    start_time = time.time()
    
    # Log request
    client_host = request.client.host if request.client else "unknown"
    app_logger.info(f"Request: {request.method} {request.url.path} from {client_host}")
    
    # Check request size
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > config.api.max_request_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Request too large"
        )
    
    # Check for suspicious headers
    suspicious_headers = ["x-forwarded-for", "x-real-ip", "x-forwarded-proto"]
    for header in suspicious_headers:
        if header in request.headers:
            app_logger.warning(f"Suspicious header detected: {header}")
    
    # Process request
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Log response time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Health check endpoint with enhanced monitoring
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with comprehensive system status."""
    from core.monitoring import get_health_checker
    
    health_checker = get_health_checker()
    health_status = health_checker.is_healthy()
    
    # Get cache health
    cache_stats = get_cache_stats()
    cache_health = {
        "status": "healthy" if cache_stats["utilization_percent"] < 90 else "warning",
        "entries": cache_stats["entries"],
        "memory_usage_mb": cache_stats["memory_usage_mb"],
        "utilization_percent": cache_stats["utilization_percent"]
    }
    
    return HealthResponse(
        status=health_status["status"],
        timestamp=health_status["timestamp"],
        version=config.version,
        uptime=health_status["uptime_seconds"],
        system_health=health_status["system"],
        cache_health=cache_health
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(hours: int = 24):
    """Get detailed system and application metrics."""
    from core.monitoring import get_health_checker
    
    health_checker = get_health_checker()
    health_status = health_checker.is_healthy()
    
    # Get cache metrics
    cache_stats = get_cache_stats()
    
    return MetricsResponse(
        system=health_status["system"],
        application=health_status["application"],
        summary=health_checker.get_metrics_summary(hours),
        cache=cache_stats
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Agentic Reviewer API",
        "version": "1.0.0",
        "description": "Semantic auditing for text classification predictions",
        "endpoints": {
            "/health": "GET - Health check",
            "/review": "POST - Review a single prediction",
            "/stats": "GET - Get review statistics",
            "/reviews/{verdict}": "GET - Get reviews by verdict",
            "/cache/stats": "GET - Get cache statistics",
            "/cache/cleanup": "POST - Clean up expired cache entries"
        },
        "security": {
            "authentication": "Bearer token required" if config.api.require_auth else "None",
            "rate_limiting": "Enabled" if config.api.enable_rate_limiting else "Disabled",
            "input_sanitization": "Enabled" if config.security.enable_input_sanitization else "Disabled"
        }
    }


@app.post("/review", response_model=ReviewResponse)
async def review_prediction(
    request: ReviewRequest,
    http_request: Request,
    auth: bool = Depends(verify_api_key),
    client_id: str = Depends(get_client_id)
):
    """
    Review a single text classification prediction.
    
    This endpoint evaluates whether the predicted label semantically fits the input text,
    suggests alternatives if incorrect, and provides natural language explanations.
    """
    # Rate limiting
    rate_limit_check(http_request, client_id)
    
    try:
        # Additional input validation
        if config.security.enable_input_sanitization:
            request.text = sanitize_text(request.text)
            request.predicted_label = sanitize_text(request.predicted_label)
        
        # Process the review
        result = await review_loop.review_single_sample_async(
            text=request.text,
            predicted_label=request.predicted_label,
            confidence=request.confidence,
            sample_id=request.sample_id,
            use_unified=request.use_unified_agent
        )
        
        return ReviewResponse(
            sample_id=result["sample_id"],
            verdict=result["verdict"],
            reasoning=result["reasoning"],
            suggested_label=result.get("suggested_label"),
            explanation=result["explanation"],
            success=result["success"],
            metadata=result.get("metadata"),
            error=result.get("error")
        )
        
    except Exception as e:
        app_logger.error(f"Review failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Review processing failed: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse)
async def get_review_stats(auth: bool = Depends(verify_api_key)):
    """Get review statistics with cache information."""
    try:
        stats = logger.get_review_stats()
        cache_stats = get_cache_stats()
        
        return StatsResponse(
            total_reviews=stats["total_reviews"],
            verdict_distribution=stats["verdict_distribution"],
            avg_confidence=stats["avg_confidence"],
            recent_reviews_24h=stats["recent_reviews_24h"],
            cache_stats=cache_stats
        )
    except Exception as e:
        app_logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@app.get("/reviews/{verdict}")
async def get_reviews_by_verdict(
    verdict: str,
    limit: int = 100,
    auth: bool = Depends(verify_api_key)
):
    """Get reviews by verdict with pagination."""
    try:
        # Validate verdict
        from core.validators import validate_verdict
        validated_verdict = validate_verdict(verdict)
        
        # Validate limit
        if limit <= 0 or limit > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be between 1 and 1000"
            )
        
        reviews = logger.get_reviews_by_verdict(validated_verdict)
        return {
            "verdict": validated_verdict,
            "count": len(reviews),
            "reviews": reviews[:limit]
        }
    except Exception as e:
        app_logger.error(f"Failed to get reviews by verdict: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get reviews: {str(e)}"
        )


@app.get("/reviews/id/{review_id}")
async def get_review_by_id(
    review_id: str,
    auth: bool = Depends(verify_api_key)
):
    """Get a specific review by ID."""
    try:
        # Validate review ID
        from core.validators import validate_sample_id
        validated_id = validate_sample_id(review_id, "review_id")
        
        review = logger.get_review(validated_id)
        if not review:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Review not found"
            )
        
        return review
    except Exception as e:
        app_logger.error(f"Failed to get review by ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get review: {str(e)}"
        )


@app.get("/cache/stats")
async def get_cache_statistics(auth: bool = Depends(verify_api_key)):
    """Get cache statistics."""
    try:
        stats = get_cache_stats()
        return {
            "cache_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        app_logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache statistics: {str(e)}"
        )


@app.post("/cache/cleanup")
async def cleanup_cache_endpoint(auth: bool = Depends(verify_api_key)):
    """Clean up expired cache entries."""
    try:
        cleaned_count = cleanup_cache()
        save_cache()  # Save cache after cleanup
        
        return {
            "message": f"Cleaned up {cleaned_count} expired cache entries",
            "cleaned_count": cleaned_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        app_logger.error(f"Failed to cleanup cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup cache: {str(e)}"
        )


@app.get("/security/stats")
async def get_security_statistics(auth: bool = Depends(verify_api_key)):
    """
    Get security validation statistics.
    
    Returns:
        Security statistics including validation results and drift detection
    """
    try:
        from core.security import get_security_manager
        security_manager = get_security_manager()
        
        security_stats = security_manager.get_security_stats()
        drift_result = security_manager.check_drift()
        ground_truth_stats = security_manager.ground_truth_validator.get_validation_stats()
        
        return {
            "security_stats": security_stats,
            "drift_detection": drift_result,
            "ground_truth_validation": ground_truth_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get security statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get security statistics: {str(e)}"
        )


@app.post("/security/validate")
async def validate_security_input(
    request: ReviewRequest,
    auth: bool = Depends(verify_api_key)
):
    """
    Validate input for security threats without processing.
    
    Returns:
        Security validation results
    """
    try:
        from core.security import get_security_manager
        security_manager = get_security_manager()
        
        security_result = security_manager.validate_input(
            request.text, request.predicted_label, request.confidence
        )
        
        return {
            "is_safe": security_result["is_safe"],
            "risk_score": security_result["risk_score"],
            "violations": [
                {
                    "type": v.violation_type,
                    "severity": v.severity,
                    "description": v.description,
                    "confidence": v.confidence
                }
                for v in security_result["violations"]
            ],
            "total_violations": security_result["total_violations"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Security validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Security validation failed: {str(e)}"
        )


@app.get("/security/drift")
async def check_drift(auth: bool = Depends(verify_api_key)):
    """
    Check for drift in system behavior.
    
    Returns:
        Drift detection results
    """
    try:
        from core.security import get_security_manager
        security_manager = get_security_manager()
        
        drift_result = security_manager.check_drift()
        
        return {
            "drift_detected": drift_result["drift_detected"],
            "max_drift": drift_result["max_drift"],
            "drift_scores": drift_result["drift_scores"],
            "confidence": drift_result["confidence"],
            "threshold": drift_result["threshold"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Drift detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift detection failed: {str(e)}"
        )


@app.get("/security/ground-truth")
async def get_ground_truth_stats(auth: bool = Depends(verify_api_key)):
    """
    Get ground truth validation statistics.
    
    Returns:
        Ground truth validation statistics
    """
    try:
        from core.security import get_security_manager
        security_manager = get_security_manager()
        
        ground_truth_stats = security_manager.ground_truth_validator.get_validation_stats()
        
        return {
            "ground_truth_stats": ground_truth_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        app_logger.error(f"Failed to get ground truth stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get ground truth stats: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        reload=False,
        log_level=config.logging.log_level.lower()
    ) 
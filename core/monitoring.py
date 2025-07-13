"""
Monitoring and health check utilities for the agentic-reviewer system.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_usage_percent: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    llm_calls: int = 0
    llm_errors: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class HealthChecker:
    """Health check and monitoring system."""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history: list[ApplicationMetrics] = []
        self.max_history_size = 1000
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available=memory.available,
                disk_usage_percent=disk.percent
            )
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available=0,
                disk_usage_percent=0.0
            )
    
    def get_application_metrics(self) -> ApplicationMetrics:
        """Get current application metrics."""
        if not self.metrics_history:
            return ApplicationMetrics()
        
        latest = self.metrics_history[-1]
        return latest
    
    def record_request(self, success: bool, response_time: float):
        """Record a request metric."""
        if not self.metrics_history:
            self.metrics_history.append(ApplicationMetrics())
        
        current = self.metrics_history[-1]
        current.total_requests += 1
        
        if success:
            current.successful_requests += 1
        else:
            current.failed_requests += 1
        
        # Update average response time
        if current.total_requests == 1:
            current.avg_response_time = response_time
        else:
            current.avg_response_time = (
                (current.avg_response_time * (current.total_requests - 1) + response_time) 
                / current.total_requests
            )
        
        current.timestamp = datetime.now()
    
    def record_llm_call(self, success: bool):
        """Record an LLM call metric."""
        if not self.metrics_history:
            self.metrics_history.append(ApplicationMetrics())
        
        current = self.metrics_history[-1]
        current.llm_calls += 1
        
        if not success:
            current.llm_errors += 1
        
        current.timestamp = datetime.now()
    
    def record_cache_hit(self, hit: bool):
        """Record a cache hit/miss."""
        if not self.metrics_history:
            self.metrics_history.append(ApplicationMetrics())
        
        current = self.metrics_history[-1]
        
        # Calculate hit rate over last 100 requests
        if current.total_requests > 0:
            # This is a simplified calculation - in production, track hits/misses separately
            current.cache_hit_rate = 0.8 if hit else 0.2  # Placeholder
        
        current.timestamp = datetime.now()
    
    def get_uptime(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self.start_time
    
    def is_healthy(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        system_metrics = self.get_system_metrics()
        app_metrics = self.get_application_metrics()
        
        # Define health thresholds
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": self.get_uptime(),
            "system": {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "disk_usage_percent": system_metrics.disk_usage_percent,
                "healthy": (
                    system_metrics.cpu_percent < 90 and
                    system_metrics.memory_percent < 90 and
                    system_metrics.disk_usage_percent < 95
                )
            },
            "application": {
                "total_requests": app_metrics.total_requests,
                "success_rate": (
                    app_metrics.successful_requests / app_metrics.total_requests * 100
                    if app_metrics.total_requests > 0 else 100
                ),
                "avg_response_time": app_metrics.avg_response_time,
                "llm_error_rate": (
                    app_metrics.llm_errors / app_metrics.llm_calls * 100
                    if app_metrics.llm_calls > 0 else 0
                ),
                "healthy": (
                    app_metrics.total_requests == 0 or
                    (app_metrics.successful_requests / app_metrics.total_requests > 0.95 and
                     app_metrics.avg_response_time < 30.0 and
                     (app_metrics.llm_calls == 0 or app_metrics.llm_errors / app_metrics.llm_calls < 0.1))
                )
            }
        }
        
        # Overall health status
        overall_healthy = (
            health_status["system"]["healthy"] and
            health_status["application"]["healthy"]
        )
        
        health_status["status"] = "healthy" if overall_healthy else "unhealthy"
        
        return health_status
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {"message": "No metrics available for the specified time period"}
        
        total_requests = sum(m.total_requests for m in recent_metrics)
        successful_requests = sum(m.successful_requests for m in recent_metrics)
        total_llm_calls = sum(m.llm_calls for m in recent_metrics)
        total_llm_errors = sum(m.llm_errors for m in recent_metrics)
        
        avg_response_times = [m.avg_response_time for m in recent_metrics if m.avg_response_time > 0]
        avg_response_time = sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0
        
        return {
            "period_hours": hours,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            "avg_response_time": avg_response_time,
            "total_llm_calls": total_llm_calls,
            "total_llm_errors": total_llm_errors,
            "llm_error_rate": (total_llm_errors / total_llm_calls * 100) if total_llm_calls > 0 else 0,
            "metrics_count": len(recent_metrics)
        }
    
    def cleanup_old_metrics(self, max_age_hours: int = 24):
        """Remove old metrics from history."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        self.metrics_history = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]
        
        # Also limit by size
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]


# Global health checker instance
_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return _health_checker 
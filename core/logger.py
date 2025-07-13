import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from .exceptions import DatabaseError, DataValidationError
from .config import config


# Configure logging
logger = logging.getLogger(__name__)


class AuditLogger:
    """Logs review results to SQLite database."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.database.db_path
        self._ensure_output_dir()
        self._init_database()
    
    def _ensure_output_dir(self):
        """Ensure the output directory exists."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        except OSError as e:
            raise DatabaseError(f"Failed to create output directory: {e}")
    
    def _init_database(self):
        """Initialize the SQLite database with the required schema."""
        try:
            with sqlite3.connect(self.db_path, timeout=config.database.connection_timeout) as conn:
                # Enable WAL mode for better concurrency
                if config.database.enable_wal_mode:
                    conn.execute("PRAGMA journal_mode=WAL")
                
                cursor = conn.cursor()
                
                # Create the main reviews table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS reviews (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        pred_label TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        verdict TEXT NOT NULL,
                        suggested_label TEXT,
                        reasoning TEXT,
                        explanation TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        prompt_hash TEXT,
                        tokens_used INTEGER,
                        latency_ms INTEGER,
                        run_id TEXT,
                        model_name TEXT
                    )
                """)
                
                # Create metadata table for run information
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS run_metadata (
                        run_id TEXT PRIMARY KEY,
                        model_name TEXT,
                        prompt_version TEXT,
                        execution_time DATETIME,
                        selector_strategy TEXT,
                        tokens_limit INTEGER,
                        total_samples INTEGER,
                        reviewed_samples INTEGER,
                        metadata_json TEXT
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_verdict ON reviews(verdict)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_timestamp ON reviews(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_reviews_run_id ON reviews(run_id)")
                
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize database: {e}")
    
    def log_review(self, sample: Dict[str, Any], verdict: str, 
                   suggested_label: Optional[str] = None, 
                   reasoning: Optional[str] = None,
                   explanation: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log a single review result.
        
        Args:
            sample: Sample data with id, text, pred_label, confidence
            verdict: Review verdict ("Correct", "Incorrect", "Uncertain")
            suggested_label: Alternative label if verdict is "Incorrect"
            reasoning: Reasoning for the verdict
            explanation: Natural language explanation
            metadata: Additional metadata (tokens, latency, etc.)
            
        Returns:
            True if logging was successful
            
        Raises:
            DataValidationError: If input data is invalid
            DatabaseError: If database operation fails
        """
        # Validate inputs
        if not sample or not isinstance(sample, dict):
            raise DataValidationError("Sample must be a non-empty dictionary")
        
        required_fields = ["text", "pred_label", "confidence"]
        for field in required_fields:
            if field not in sample:
                raise DataValidationError(f"Missing required field: {field}")
        
        if not isinstance(sample["text"], str) or not sample["text"].strip():
            raise DataValidationError("Text must be a non-empty string")
        
        if not isinstance(sample["pred_label"], str) or not sample["pred_label"].strip():
            raise DataValidationError("Predicted label must be a non-empty string")
        
        if not isinstance(sample["confidence"], (int, float)) or not 0.0 <= sample["confidence"] <= 1.0:
            raise DataValidationError("Confidence must be a number between 0.0 and 1.0")
        
        valid_verdicts = ["Correct", "Incorrect", "Uncertain"]
        if verdict not in valid_verdicts:
            raise DataValidationError(f"Verdict must be one of {valid_verdicts}")
        
        try:
            with sqlite3.connect(self.db_path, timeout=config.database.connection_timeout) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO reviews (
                        id, text, pred_label, confidence, verdict, 
                        suggested_label, reasoning, explanation, 
                        prompt_hash, tokens_used, latency_ms, 
                        run_id, model_name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sample.get("id", "unknown"),
                    sample["text"],
                    sample["pred_label"],
                    sample["confidence"],
                    verdict,
                    suggested_label,
                    reasoning,
                    explanation,
                    metadata.get("prompt_hash") if metadata else None,
                    metadata.get("tokens_used") if metadata else None,
                    metadata.get("latency_ms") if metadata else None,
                    metadata.get("run_id") if metadata else None,
                    metadata.get("model_name") if metadata else None
                ))
                
                conn.commit()
                logger.debug(f"Logged review for sample {sample.get('id', 'unknown')}")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Database error logging review: {e}")
            raise DatabaseError(f"Failed to log review: {e}")
    
    def log_run_metadata(self, run_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Log metadata for a review run.
        
        Args:
            run_id: Unique identifier for the run
            metadata: Run metadata
            
        Returns:
            True if logging was successful
            
        Raises:
            DataValidationError: If input data is invalid
            DatabaseError: If database operation fails
        """
        if not run_id or not isinstance(run_id, str):
            raise DataValidationError("Run ID must be a non-empty string")
        
        if not metadata or not isinstance(metadata, dict):
            raise DataValidationError("Metadata must be a non-empty dictionary")
        
        try:
            with sqlite3.connect(self.db_path, timeout=config.database.connection_timeout) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO run_metadata (
                        run_id, model_name, prompt_version, execution_time,
                        selector_strategy, tokens_limit, total_samples,
                        reviewed_samples, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    metadata.get("model_name"),
                    metadata.get("prompt_version"),
                    metadata.get("execution_time"),
                    metadata.get("selector_strategy"),
                    metadata.get("tokens_limit"),
                    metadata.get("total_samples"),
                    metadata.get("reviewed_samples"),
                    json.dumps(metadata)
                ))
                
                conn.commit()
                logger.info(f"Logged run metadata for run {run_id}")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Database error logging run metadata: {e}")
            raise DatabaseError(f"Failed to log run metadata: {e}")
    
    def get_review(self, review_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific review by ID.
        
        Args:
            review_id: The ID of the review to retrieve
            
        Returns:
            Review data as dictionary or None if not found
            
        Raises:
            DataValidationError: If review_id is invalid
            DatabaseError: If database operation fails
        """
        if not review_id or not isinstance(review_id, str):
            raise DataValidationError("Review ID must be a non-empty string")
        
        try:
            with sqlite3.connect(self.db_path, timeout=config.database.connection_timeout) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM reviews WHERE id = ?
                """, (review_id,))
                
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, row))
                
                logger.debug(f"Review {review_id} not found")
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving review: {e}")
            raise DatabaseError(f"Failed to retrieve review: {e}")
    
    def get_reviews_by_verdict(self, verdict: str) -> list:
        """
        Get all reviews with a specific verdict.
        
        Args:
            verdict: The verdict to filter by
            
        Returns:
            List of review dictionaries
            
        Raises:
            DataValidationError: If verdict is invalid
            DatabaseError: If database operation fails
        """
        valid_verdicts = ["Correct", "Incorrect", "Uncertain"]
        if verdict not in valid_verdicts:
            raise DataValidationError(f"Verdict must be one of {valid_verdicts}")
        
        try:
            with sqlite3.connect(self.db_path, timeout=config.database.connection_timeout) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM reviews WHERE verdict = ?
                """, (verdict,))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                result = [dict(zip(columns, row)) for row in rows]
                logger.debug(f"Retrieved {len(result)} reviews with verdict '{verdict}'")
                return result
                
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving reviews by verdict: {e}")
            raise DatabaseError(f"Failed to retrieve reviews by verdict: {e}")
    
    def get_review_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all reviews.
        
        Returns:
            Dictionary with review statistics
            
        Raises:
            DatabaseError: If database operation fails
        """
        try:
            with sqlite3.connect(self.db_path, timeout=config.database.connection_timeout) as conn:
                cursor = conn.cursor()
                
                # Total reviews
                cursor.execute("SELECT COUNT(*) FROM reviews")
                total_reviews = cursor.fetchone()[0]
                
                # Verdict distribution
                cursor.execute("""
                    SELECT verdict, COUNT(*) FROM reviews 
                    GROUP BY verdict
                """)
                verdict_counts = dict(cursor.fetchall())
                
                # Average confidence
                cursor.execute("SELECT AVG(confidence) FROM reviews")
                avg_confidence = cursor.fetchone()[0] or 0
                
                # Recent reviews (last 24 hours)
                cursor.execute("""
                    SELECT COUNT(*) FROM reviews 
                    WHERE timestamp > datetime('now', '-1 day')
                """)
                recent_reviews = cursor.fetchone()[0]
                
                stats = {
                    "total_reviews": total_reviews,
                    "verdict_distribution": verdict_counts,
                    "avg_confidence": avg_confidence,
                    "recent_reviews_24h": recent_reviews
                }
                
                logger.debug(f"Retrieved review stats: {stats}")
                return stats
                
        except sqlite3.Error as e:
            logger.error(f"Database error getting review stats: {e}")
            raise DatabaseError(f"Failed to get review stats: {e}") 
import pandas as pd
import os
import logging
from typing import List, Dict, Any, Optional

from .exceptions import DataValidationError, ConfigurationError
from .validators import validate_text, validate_label, validate_confidence, validate_sample
from .config import config


# Configure logging
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing classification data."""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or "data/input.csv"
    
    def load_data(self) -> pd.DataFrame:
        """
        Load classification data from CSV file.
        
        Expected columns:
        - text: The input text
        - pred_label: The predicted label
        - confidence: The confidence score (0.0 to 1.0)
        - id: Optional sample ID
        
        Returns:
            DataFrame with classification data
            
        Raises:
            DataValidationError: If data validation fails
            ConfigurationError: If file operations fail
        """
        try:
            if not os.path.exists(self.data_path):
                # Create sample data if file doesn't exist
                logger.info(f"Data file not found at {self.data_path}, creating sample data")
                self._create_sample_data()
            
            df = pd.read_csv(self.data_path)
            
            # Validate required columns
            required_columns = ["text", "pred_label", "confidence"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise DataValidationError(f"Missing required columns: {missing_columns}")
            
            # Add ID column if not present
            if "id" not in df.columns:
                df["id"] = [f"sample_{i}" for i in range(len(df))]
            
            # Validate data
            self._validate_dataframe(df)
            
            logger.info(f"Loaded {len(df)} samples from {self.data_path}")
            return df
            
        except pd.errors.EmptyDataError:
            raise DataValidationError(f"Data file is empty: {self.data_path}")
        except pd.errors.ParserError as e:
            raise DataValidationError(f"Failed to parse CSV file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load data: {e}")
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate the loaded DataFrame."""
        if df.empty:
            raise DataValidationError("DataFrame is empty")
        
        # Validate confidence scores
        if not all(0.0 <= conf <= 1.0 for conf in df["confidence"]):
            raise DataValidationError("Confidence scores must be between 0.0 and 1.0")
        
        # Validate text fields
        for idx, text in enumerate(df["text"]):
            if not isinstance(text, str) or not text.strip():
                raise DataValidationError(f"Invalid text at row {idx}: must be non-empty string")
        
        # Validate label fields
        for idx, label in enumerate(df["pred_label"]):
            if not isinstance(label, str) or not label.strip():
                raise DataValidationError(f"Invalid label at row {idx}: must be non-empty string")
        
        logger.debug(f"DataFrame validation passed: {len(df)} rows")
    
    def _create_sample_data(self):
        """Create sample classification data for testing."""
        try:
            sample_data = [
                {
                    "id": "sample_001",
                    "text": "Delete my data permanently",
                    "pred_label": "Access Request",
                    "confidence": 0.85
                },
                {
                    "id": "sample_002", 
                    "text": "What information do you have about me?",
                    "pred_label": "Access Request",
                    "confidence": 0.92
                },
                {
                    "id": "sample_003",
                    "text": "I want to be removed from your system",
                    "pred_label": "Erasure",
                    "confidence": 0.78
                },
                {
                    "id": "sample_004",
                    "text": "How do you protect my data?",
                    "pred_label": "General Inquiry",
                    "confidence": 0.65
                },
                {
                    "id": "sample_005",
                    "text": "My email address has changed, please update it",
                    "pred_label": "Rectification",
                    "confidence": 0.88
                },
                {
                    "id": "sample_006",
                    "text": "I don't want you to use my data for marketing",
                    "pred_label": "Objection",
                    "confidence": 0.91
                },
                {
                    "id": "sample_007",
                    "text": "Send me a copy of my data",
                    "pred_label": "Access Request",
                    "confidence": 0.94
                },
                {
                    "id": "sample_008",
                    "text": "I want to export my data to another service",
                    "pred_label": "Portability",
                    "confidence": 0.87
                },
                {
                    "id": "sample_009",
                    "text": "This is a formal complaint about data protection",
                    "pred_label": "Complaint",
                    "confidence": 0.89
                },
                {
                    "id": "sample_010",
                    "text": "Please erase all my personal information",
                    "pred_label": "Erasure",
                    "confidence": 0.96
                }
            ]
            
            # Validate sample data
            for sample in sample_data:
                validate_sample(sample)
            
            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(sample_data)
            df.to_csv(self.data_path, index=False)
            logger.info(f"Created sample data at {self.data_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create sample data: {e}")
    
    def get_sample(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific sample by ID.
        
        Args:
            sample_id: The ID of the sample to retrieve
            
        Returns:
            Sample data as dictionary or None if not found
            
        Raises:
            DataValidationError: If sample_id is invalid
        """
        if not sample_id or not isinstance(sample_id, str):
            raise DataValidationError("Sample ID must be a non-empty string")
        
        try:
            df = self.load_data()
            sample = df[df["id"] == sample_id]
            
            if len(sample) == 0:
                logger.debug(f"Sample {sample_id} not found")
                return None
            
            sample_dict = sample.iloc[0].to_dict()
            logger.debug(f"Retrieved sample {sample_id}")
            return sample_dict
            
        except Exception as e:
            logger.error(f"Failed to get sample {sample_id}: {e}")
            raise
    
    def get_samples_by_label(self, label: str) -> pd.DataFrame:
        """
        Get all samples with a specific predicted label.
        
        Args:
            label: The label to filter by
            
        Returns:
            DataFrame with samples matching the label
            
        Raises:
            DataValidationError: If label is invalid
        """
        if not label or not isinstance(label, str):
            raise DataValidationError("Label must be a non-empty string")
        
        try:
            df = self.load_data()
            # Use loc to ensure we get a DataFrame
            filtered_df = df.loc[df["pred_label"] == label].copy()
            
            logger.debug(f"Retrieved {len(filtered_df)} samples with label '{label}'")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Failed to get samples by label '{label}': {e}")
            raise 
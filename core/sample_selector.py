import pandas as pd
import random
import logging
from typing import List, Dict, Any, Optional

from .exceptions import SampleSelectionError, DataValidationError
from .validators import validate_strategy, validate_threshold, validate_sample_size
from .config import config


# Configure logging
logger = logging.getLogger(__name__)


class SampleSelector:
    """Selects samples for review based on various strategies."""
    
    def __init__(self, strategy: str = "low_confidence", **kwargs):
        """
        Initialize the sample selector.
        
        Args:
            strategy: Selection strategy ("low_confidence", "random", "all")
            **kwargs: Strategy-specific parameters
            
        Raises:
            DataValidationError: If strategy or parameters are invalid
        """
        self.strategy = validate_strategy(strategy)
        self.kwargs = kwargs
        
        # Validate strategy-specific parameters
        self._validate_strategy_params()
        
        logger.debug(f"Initialized SampleSelector with strategy: {self.strategy}")
    
    def _validate_strategy_params(self) -> None:
        """Validate strategy-specific parameters."""
        if self.strategy == "low_confidence":
            if "threshold" in self.kwargs:
                self.kwargs["threshold"] = validate_threshold(self.kwargs["threshold"])
            else:
                self.kwargs["threshold"] = config.selection.default_threshold
                
        elif self.strategy == "random":
            if "sample_size" in self.kwargs:
                self.kwargs["sample_size"] = validate_sample_size(self.kwargs["sample_size"])
            else:
                self.kwargs["sample_size"] = config.selection.default_sample_size
            
            if "seed" in self.kwargs:
                if not isinstance(self.kwargs["seed"], int):
                    raise DataValidationError("Seed must be an integer")
            else:
                self.kwargs["seed"] = config.selection.default_seed
    
    def select_samples(self, df: pd.DataFrame, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Select samples for review based on the configured strategy.
        
        Args:
            df: DataFrame with classification data
            max_samples: Maximum number of samples to select (None for all)
            
        Returns:
            DataFrame with selected samples
            
        Raises:
            DataValidationError: If inputs are invalid
            SampleSelectionError: If selection fails
        """
        if df is None or df.empty:
            raise DataValidationError("DataFrame cannot be None or empty")
        
        if max_samples is not None:
            max_samples = validate_sample_size(max_samples)
        
        try:
            if self.strategy == "low_confidence":
                selected = self._select_low_confidence(df)
            elif self.strategy == "random":
                selected = self._select_random(df)
            elif self.strategy == "all":
                selected = df.copy()
            else:
                raise SampleSelectionError(f"Unknown strategy: {self.strategy}")
            
            # Limit number of samples if specified
            if max_samples and len(selected) > max_samples:
                selected = selected.head(max_samples)
                logger.debug(f"Limited selection to {max_samples} samples")
            
            logger.info(f"Selected {len(selected)} samples using strategy '{self.strategy}'")
            return selected
            
        except Exception as e:
            logger.error(f"Sample selection failed: {e}")
            raise SampleSelectionError(f"Failed to select samples: {e}")
    
    def _select_low_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select samples with low confidence scores."""
        threshold = self.kwargs.get("threshold", config.selection.default_threshold)
        
        if not 0.0 <= threshold <= 1.0:
            raise DataValidationError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        selected: pd.DataFrame = df[df["confidence"] < threshold].copy()
        logger.debug(f"Low confidence selection: {len(selected)} samples with confidence < {threshold}")
        return selected
    
    def _select_random(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select random samples."""
        sample_size = self.kwargs.get("sample_size", config.selection.default_sample_size)
        seed = self.kwargs.get("seed", config.selection.default_seed)
        
        if sample_size >= len(df):
            logger.debug(f"Sample size ({sample_size}) >= total samples ({len(df)}), returning all")
            return df.copy()
        
        # Use reproducible random selection
        random.seed(seed)
        indices = random.sample(range(len(df)), sample_size)
        selected = df.iloc[indices].copy()
        
        logger.debug(f"Random selection: {len(selected)} samples with seed {seed}")
        return selected
    
    def get_selection_stats(self, df: pd.DataFrame, selected: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the selection process.
        
        Args:
            df: Original DataFrame
            selected: Selected DataFrame
            
        Returns:
            Dictionary with selection statistics
            
        Raises:
            DataValidationError: If inputs are invalid
        """
        if df is None or selected is None:
            raise DataValidationError("Both original and selected DataFrames must be provided")
        
        try:
            stats = {
                "strategy": self.strategy,
                "total_samples": len(df),
                "selected_samples": len(selected),
                "selection_rate": len(selected) / len(df) if len(df) > 0 else 0,
                "avg_confidence_original": df["confidence"].mean(),
                "avg_confidence_selected": selected["confidence"].mean() if len(selected) > 0 else 0,
                "strategy_params": self.kwargs
            }
            
            logger.debug(f"Selection stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate selection stats: {e}")
            raise SampleSelectionError(f"Failed to calculate selection stats: {e}") 
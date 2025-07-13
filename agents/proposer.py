from typing import Dict, Any, Optional
import logging
from .base_agent import BaseAgent, TaskType, AgentTask
from core.validators import validate_text, validate_label, validate_confidence
from core.exceptions import DataValidationError, LLMResponseError


# Configure logging
logger = logging.getLogger(__name__)


class ProposerAgent(BaseAgent):
    """Agent that suggests alternative labels when predictions are incorrect."""
    
    def __init__(self, model_name: Optional[str] = None, ollama_url: Optional[str] = None):
        super().__init__(model_name, ollama_url)
    
    def _propose_common(self, text: str, predicted_label: str, confidence: float, is_async: bool = False) -> Dict[str, Any]:
        """Common logic for both sync and async proposal to eliminate duplication."""
        # Validate inputs
        text = validate_text(text, "text")
        predicted_label = validate_label(predicted_label, "predicted_label")
        confidence = validate_confidence(confidence, "confidence")
        
        context = {
            "text": text,
            "predicted_label": predicted_label,
            "confidence": confidence
        }
        
        expected_fields = ["suggested_label", "confidence"]
        
        try:
            if is_async:
                import asyncio
                result = asyncio.run(self.call_llm_async(
                    prompt_template=str(self.prompt_templates[TaskType.PROPOSE]),
                    context=context,
                    expected_fields=expected_fields
                ))
            else:
                result = self.call_llm(
                    prompt_template=str(self.prompt_templates[TaskType.PROPOSE]),
                    context=context,
                    expected_fields=expected_fields
                )
            
            # Extract suggested label and confidence
            if result["success"] and "parsed" in result:
                parsed = result["parsed"]
                suggested_label = parsed.get("suggested_label", "No suitable label found")
                confidence_level = parsed.get("confidence", "Medium")
            else:
                # Fallback parsing if structured parsing failed
                response = result["response"]
                suggested_label = self._extract_label_fallback(response)
                confidence_level = self._extract_confidence_fallback(response)
            
            # Validate suggested label
            available_labels = [label["name"] for label in self.labels["labels"]]
            if suggested_label not in available_labels and suggested_label != "No suitable label found":
                logger.warning(f"Invalid suggested label from LLM: {suggested_label}, defaulting to 'No suitable label found'")
                suggested_label = "No suitable label found"
            
            logger.info(f"{'Async ' if is_async else ''}Proposal completed: {suggested_label} for text '{text[:50]}...'")
            
            return {
                "suggested_label": suggested_label,
                "confidence": confidence_level,
                "success": result["success"],
                "tokens_used": result["tokens_used"],
                "latency_ms": result["latency_ms"],
                "prompt_hash": result["prompt_hash"]
            }
            
        except Exception as e:
            logger.error(f"{'Async ' if is_async else ''}Proposal failed: {e}")
            raise

    def propose(self, text: str, predicted_label: str, confidence: float) -> Dict[str, Any]:
        """
        Propose an alternative label for the given text (synchronous).
        
        Args:
            text: The input text
            predicted_label: The incorrect label that was predicted
            confidence: The confidence score of the incorrect prediction
            
        Returns:
            Dict containing suggested label, confidence, and metadata
            
        Raises:
            DataValidationError: If input data is invalid
            LLMResponseError: If LLM response is invalid
        """
        return self._propose_common(text, predicted_label, confidence, is_async=False)
    
    async def propose_async(self, text: str, predicted_label: str, confidence: float) -> Dict[str, Any]:
        """
        Propose an alternative label for the given text (async).
        
        Args:
            text: The input text
            predicted_label: The incorrect label that was predicted
            confidence: The confidence score of the incorrect prediction
            
        Returns:
            Dict containing suggested label, confidence, and metadata
            
        Raises:
            DataValidationError: If input data is invalid
            LLMResponseError: If LLM response is invalid
        """
        return self._propose_common(text, predicted_label, confidence, is_async=True)
    
    def create_proposal_task(self, text: str, predicted_label: str, confidence: float) -> AgentTask:
        """Create a proposal task for multi-task processing."""
        context = {
            "text": text,
            "predicted_label": predicted_label,
            "confidence": confidence
        }
        
        return AgentTask(
            task_type=TaskType.PROPOSE,
            context=context,
            expected_fields=["suggested_label", "confidence"],
            prompt_template=str(self.prompt_templates[TaskType.PROPOSE])
        ) 
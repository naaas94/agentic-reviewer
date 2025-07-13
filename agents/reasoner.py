from typing import Dict, Any, Optional
import logging
from .base_agent import BaseAgent, TaskType, AgentTask
from core.validators import validate_text, validate_label
from core.exceptions import DataValidationError, LLMResponseError


# Configure logging
logger = logging.getLogger(__name__)


class ReasonerAgent(BaseAgent):
    """Agent that generates natural language explanations for decisions."""
    
    def __init__(self, model_name: Optional[str] = None, ollama_url: Optional[str] = None):
        super().__init__(model_name, ollama_url)
    
    def _reason_common(self, text: str, label: str, context: str, is_async: bool = False) -> Dict[str, Any]:
        """Common logic for both sync and async reasoning to eliminate duplication."""
        # Validate inputs
        text = validate_text(text, "text")
        label = validate_label(label, "label")
        
        if not context or not isinstance(context, str):
            raise DataValidationError("Context must be a non-empty string")
        
        context_data = {
            "text": text,
            "label": label,
            "context": context
        }
        
        expected_fields = ["explanation"]
        
        try:
            if is_async:
                import asyncio
                result = asyncio.run(self.call_llm_async(
                    prompt_template=str(self.prompt_templates[TaskType.REASON]),
                    context=context_data,
                    expected_fields=expected_fields
                ))
            else:
                result = self.call_llm(
                    prompt_template=str(self.prompt_templates[TaskType.REASON]),
                    context=context_data,
                    expected_fields=expected_fields
                )
            
            # Extract explanation
            if result["success"] and "parsed" in result:
                parsed = result["parsed"]
                explanation = parsed.get("explanation", "No explanation provided")
            else:
                # Fallback parsing if structured parsing failed
                response = result["response"]
                explanation = self._extract_explanation_fallback(response)
            
            logger.info(f"{'Async ' if is_async else ''}Reasoning completed for text '{text[:50]}...'")
            
            return {
                "explanation": explanation,
                "success": result["success"],
                "tokens_used": result["tokens_used"],
                "latency_ms": result["latency_ms"],
                "prompt_hash": result["prompt_hash"]
            }
            
        except Exception as e:
            logger.error(f"{'Async ' if is_async else ''}Reasoning failed: {e}")
            raise

    def reason(self, text: str, label: str, context: str) -> Dict[str, Any]:
        """
        Generate a natural language explanation for the decision (synchronous).
        
        Args:
            text: The input text
            label: The label (either predicted or suggested)
            context: Additional context about the decision
            
        Returns:
            Dict containing explanation and metadata
            
        Raises:
            DataValidationError: If input data is invalid
            LLMResponseError: If LLM response is invalid
        """
        return self._reason_common(text, label, context, is_async=False)
    
    async def reason_async(self, text: str, label: str, context: str) -> Dict[str, Any]:
        """
        Generate a natural language explanation for the decision (async).
        
        Args:
            text: The input text
            label: The label (either predicted or suggested)
            context: Additional context about the decision
            
        Returns:
            Dict containing explanation and metadata
            
        Raises:
            DataValidationError: If input data is invalid
            LLMResponseError: If LLM response is invalid
        """
        return self._reason_common(text, label, context, is_async=True)
    
    def create_reasoning_task(self, text: str, label: str, context: str) -> AgentTask:
        """Create a reasoning task for multi-task processing."""
        context_data = {
            "text": text,
            "label": label,
            "context": context
        }
        
        return AgentTask(
            task_type=TaskType.REASON,
            context=context_data,
            expected_fields=["explanation"],
            prompt_template=str(self.prompt_templates[TaskType.REASON])
        ) 
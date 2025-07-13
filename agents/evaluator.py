from typing import Dict, Any, Optional
import logging
from .base_agent import BaseAgent, TaskType, AgentTask
from core.validators import validate_text, validate_label, validate_confidence
from core.exceptions import DataValidationError, LLMResponseError


# Configure logging
logger = logging.getLogger(__name__)


class EvaluatorAgent(BaseAgent):
    """Agent that evaluates whether a predicted label semantically fits the input text."""
    
    def __init__(self, model_name: Optional[str] = None, ollama_url: Optional[str] = None):
        super().__init__(model_name, ollama_url)
    
    def _evaluate_common(self, text: str, predicted_label: str, confidence: float, is_async: bool = False) -> Dict[str, Any]:
        """Common logic for both sync and async evaluation to eliminate duplication."""
        # Validate inputs
        text = validate_text(text, "text")
        predicted_label = validate_label(predicted_label, "predicted_label")
        confidence = validate_confidence(confidence, "confidence")
        
        context = {
            "text": text,
            "predicted_label": predicted_label,
            "confidence": confidence
        }
        
        expected_fields = ["verdict", "reasoning"]
        
        try:
            if is_async:
                import asyncio
                result = asyncio.run(self.call_llm_async(
                    prompt_template=str(self.prompt_templates[TaskType.EVALUATE]),
                    context=context,
                    expected_fields=expected_fields
                ))
            else:
                result = self.call_llm(
                    prompt_template=str(self.prompt_templates[TaskType.EVALUATE]),
                    context=context,
                    expected_fields=expected_fields
                )
            
            # Extract verdict and reasoning
            if result["success"] and "parsed" in result:
                parsed = result["parsed"]
                verdict = parsed.get("verdict", "Uncertain")
                reasoning = parsed.get("reasoning", "No reasoning provided")
            else:
                # Fallback parsing if structured parsing failed
                response = result["response"]
                verdict = self._extract_verdict_fallback(response)
                reasoning = self._extract_reasoning_fallback(response)
            
            # Validate verdict
            valid_verdicts = ["Correct", "Incorrect", "Uncertain"]
            if verdict not in valid_verdicts:
                logger.warning(f"Invalid verdict from LLM: {verdict}, defaulting to Uncertain")
                verdict = "Uncertain"
            
            logger.info(f"{'Async ' if is_async else ''}Evaluation completed: {verdict} for text '{text[:50]}...'")
            
            return {
                "verdict": verdict,
                "reasoning": reasoning,
                "success": result["success"],
                "tokens_used": result["tokens_used"],
                "latency_ms": result["latency_ms"],
                "prompt_hash": result["prompt_hash"]
            }
            
        except Exception as e:
            logger.error(f"{'Async ' if is_async else ''}Evaluation failed: {e}")
            raise

    def evaluate(self, text: str, predicted_label: str, confidence: float) -> Dict[str, Any]:
        """
        Evaluate if the predicted label semantically fits the text (synchronous).
        
        Args:
            text: The input text
            predicted_label: The label predicted by the classifier
            confidence: The confidence score of the prediction
            
        Returns:
            Dict containing verdict, reasoning, and metadata
            
        Raises:
            DataValidationError: If input data is invalid
            LLMResponseError: If LLM response is invalid
        """
        return self._evaluate_common(text, predicted_label, confidence, is_async=False)
    
    async def evaluate_async(self, text: str, predicted_label: str, confidence: float) -> Dict[str, Any]:
        """
        Evaluate if the predicted label semantically fits the text (async).
        
        Args:
            text: The input text
            predicted_label: The label predicted by the classifier
            confidence: The confidence score of the prediction
            
        Returns:
            Dict containing verdict, reasoning, and metadata
            
        Raises:
            DataValidationError: If input data is invalid
            LLMResponseError: If LLM response is invalid
        """
        return self._evaluate_common(text, predicted_label, confidence, is_async=True)
    
    def create_evaluation_task(self, text: str, predicted_label: str, confidence: float) -> AgentTask:
        """Create an evaluation task for multi-task processing."""
        context = {
            "text": text,
            "predicted_label": predicted_label,
            "confidence": confidence
        }
        
        return AgentTask(
            task_type=TaskType.EVALUATE,
            context=context,
            expected_fields=["verdict", "reasoning"],
            prompt_template=str(self.prompt_templates[TaskType.EVALUATE])
        ) 
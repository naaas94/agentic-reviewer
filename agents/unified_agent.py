"""
Unified Agent for Multi-Task Processing

This agent processes evaluation, proposal, and reasoning tasks in a single LLM call
for improved efficiency and reduced costs.
"""

from typing import Dict, Any, Optional, List
import logging
from .base_agent import BaseAgent, TaskType, AgentTask
from core.validators import validate_text, validate_label, validate_confidence
from core.exceptions import DataValidationError, LLMResponseError, SecurityError


# Configure logging
logger = logging.getLogger(__name__)


class UnifiedAgent(BaseAgent):
    """Unified agent that processes all tasks in a single LLM call for efficiency."""
    
    def __init__(self, model_name: Optional[str] = None, ollama_url: Optional[str] = None):
        super().__init__(model_name, ollama_url)
    
    async def process_sample(self, text: str, predicted_label: str, confidence: float) -> Dict[str, Any]:
        """
        Process a complete sample through all agents in a single LLM call.
        
        Args:
            text: The input text
            predicted_label: The predicted label
            confidence: The confidence score
            
        Returns:
            Dict containing all results (evaluation, proposal, reasoning)
        """
        # Validate inputs
        text = validate_text(text, "text")
        predicted_label = validate_label(predicted_label, "predicted_label")
        confidence = validate_confidence(confidence, "confidence")
        
        # Security validation
        try:
            security_result = self._validate_input_security(text, predicted_label, confidence)
            logger.debug(f"Security validation passed: risk_score={security_result['risk_score']}")
        except SecurityError as e:
            logger.error(f"Security validation failed: {e}")
            return {
                "verdict": "Uncertain",
                "reasoning": f"Security validation failed: {e}",
                "suggested_label": None,
                "explanation": "Input rejected due to security concerns",
                "success": False,
                "security_violation": True,
                "error": str(e)
            }
        
        # Create tasks for all agents
        tasks = []
        
        # Evaluation task
        eval_task = AgentTask(
            task_type=TaskType.EVALUATE,
            context={"text": text, "predicted_label": predicted_label, "confidence": confidence},
            expected_fields=["verdict", "reasoning"],
            prompt_template=str(self.prompt_templates[TaskType.EVALUATE])
        )
        tasks.append(eval_task)
        
        # Proposal task (will be processed conditionally)
        prop_task = AgentTask(
            task_type=TaskType.PROPOSE,
            context={"text": text, "predicted_label": predicted_label, "confidence": confidence},
            expected_fields=["suggested_label", "confidence"],
            prompt_template=str(self.prompt_templates[TaskType.PROPOSE])
        )
        tasks.append(prop_task)
        
        # Reasoning task (will be updated with context)
        reason_task = AgentTask(
            task_type=TaskType.REASON,
            context={"text": text, "label": predicted_label, "context": "evaluation pending"},
            expected_fields=["explanation"],
            prompt_template=str(self.prompt_templates[TaskType.REASON])
        )
        tasks.append(reason_task)
        
        try:
            # Process all tasks in a single LLM call
            results = await self.process_multi_task(tasks)
            
            # Extract results
            eval_result = next((r for r in results if r["task_type"] == TaskType.EVALUATE), None)
            prop_result = next((r for r in results if r["task_type"] == TaskType.PROPOSE), None)
            reason_result = next((r for r in results if r["task_type"] == TaskType.REASON), None)
            
            # Process evaluation result
            if eval_result and eval_result["success"]:
                verdict = eval_result["parsed"].get("verdict", "Uncertain")
                reasoning = eval_result["parsed"].get("reasoning", "No reasoning provided")
                
                # Validate verdict
                valid_verdicts = ["Correct", "Incorrect", "Uncertain"]
                if verdict not in valid_verdicts:
                    logger.warning(f"Invalid verdict from LLM: {verdict}, defaulting to Uncertain")
                    verdict = "Uncertain"
            else:
                verdict = "Uncertain"
                reasoning = "Evaluation failed"
            
            # Process proposal result (only if evaluation was incorrect)
            suggested_label = None
            if verdict == "Incorrect" and prop_result and prop_result["success"]:
                suggested_label = prop_result["parsed"].get("suggested_label", "No suitable label found")
                
                # Validate suggested label
                available_labels = [label["name"] for label in self.labels["labels"]]
                if suggested_label not in available_labels and suggested_label != "No suitable label found":
                    logger.warning(f"Invalid suggested label from LLM: {suggested_label}")
                    suggested_label = "No suitable label found"
            
            # Process reasoning result
            if reason_result and reason_result["success"]:
                explanation = reason_result["parsed"].get("explanation", "No explanation provided")
            else:
                explanation = "Reasoning failed"
            
            # Update reasoning context with final results
            final_label = suggested_label if suggested_label else predicted_label
            explanation_context = f"verdict: {verdict}"
            if suggested_label:
                explanation_context += f", suggested: {suggested_label}"
            
            # Generate final explanation with updated context
            final_reason_task = AgentTask(
                task_type=TaskType.REASON,
                context={"text": text, "label": final_label, "context": explanation_context},
                expected_fields=["explanation"],
                prompt_template=str(self.prompt_templates[TaskType.REASON])
            )
            
            final_reason_result = await self.process_multi_task([final_reason_task])
            if final_reason_result and final_reason_result[0]["success"]:
                explanation = final_reason_result[0]["parsed"].get("explanation", explanation)
            
            # Ground truth validation
            ground_truth_result = self.security_manager.validate_output(
                text, predicted_label, verdict, reasoning
            )
            
            # Add response for drift detection
            response_data = {
                "verdict": verdict,
                "reasoning": reasoning,
                "tokens_used": sum(r.get("tokens_used", 0) for r in results if r.get("success")),
                "latency_ms": sum(r.get("latency_ms", 0) for r in results if r.get("success"))
            }
            self.security_manager.add_response_for_drift_detection(response_data)
            
            logger.info(f"Unified processing completed: {verdict} for text '{text[:50]}...'")
            
            return {
                "verdict": verdict,
                "reasoning": reasoning,
                "suggested_label": suggested_label,
                "explanation": explanation,
                "success": True,
                "tokens_used": sum(r.get("tokens_used", 0) for r in results if r.get("success")),
                "latency_ms": sum(r.get("latency_ms", 0) for r in results if r.get("success")),
                "prompt_hash": results[0].get("prompt_hash") if results else None,
                "ground_truth_validation": ground_truth_result,
                "security_validation": security_result
            }
            
        except Exception as e:
            logger.error(f"Unified processing failed: {e}")
            raise
    
    def process_sample_sync(self, text: str, predicted_label: str, confidence: float) -> Dict[str, Any]:
        """
        Synchronous wrapper for process_sample.
        
        Args:
            text: The input text
            predicted_label: The predicted label
            confidence: The confidence score
            
        Returns:
            Dict containing all results
        """
        import asyncio
        return asyncio.run(self.process_sample(text, predicted_label, confidence))
    
    async def process_batch(self, samples: List[Dict[str, Any]], max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """
        Process a batch of samples with controlled concurrency.
        
        Args:
            samples: List of sample dictionaries with text, predicted_label, confidence
            max_concurrent: Maximum concurrent LLM calls
            
        Returns:
            List of results for each sample
        """
        import asyncio
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self.process_sample(
                    sample["text"],
                    sample["predicted_label"],
                    sample["confidence"]
                )
        
        # Process all samples concurrently
        tasks = [process_single_sample(sample) for sample in samples]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process sample {i}: {result}")
                processed_results.append({
                    "verdict": "Uncertain",
                    "reasoning": f"Processing failed: {result}",
                    "suggested_label": None,
                    "explanation": "Processing failed",
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results 
import time
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from .data_loader import DataLoader
from .sample_selector import SampleSelector
from .logger import AuditLogger


class ReviewLoop:
    """Main orchestrator for the agentic review process with unified agent support."""
    
    def __init__(self, model_name: str = "mistral", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        
        # Initialize components
        self.data_loader = DataLoader()
        self.logger = AuditLogger()
        
        # Import agents locally to avoid circular imports
        from agents.evaluator import EvaluatorAgent
        from agents.proposer import ProposerAgent
        from agents.reasoner import ReasonerAgent
        from agents.unified_agent import UnifiedAgent
        
        # Initialize both individual agents and unified agent
        self.evaluator = EvaluatorAgent(model_name, ollama_url)
        self.proposer = ProposerAgent(model_name, ollama_url)
        self.reasoner = ReasonerAgent(model_name, ollama_url)
        self.unified_agent = UnifiedAgent(model_name, ollama_url)
        
        # Use unified agent by default for better efficiency
        self.use_unified_agent = True
    
    async def run_review_async(self, selector_strategy: str = "low_confidence", 
                              max_samples: Optional[int] = None, 
                              max_concurrent: int = 5, 
                              use_unified: bool = True,
                              **selector_kwargs) -> Dict[str, Any]:
        """
        Run the complete review process asynchronously with concurrent processing.
        
        Args:
            selector_strategy: Strategy for selecting samples
            max_samples: Maximum number of samples to review
            max_concurrent: Maximum concurrent LLM requests
            use_unified: Whether to use unified agent (more efficient)
            **selector_kwargs: Additional arguments for sample selector
            
        Returns:
            Dictionary with review results and statistics
        """
        # Generate run ID
        run_id = f"rev_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        print(f"Starting async review run: {run_id}")
        print(f"Model: {self.model_name}")
        print(f"Strategy: {selector_strategy}")
        print(f"Max concurrent requests: {max_concurrent}")
        print(f"Using unified agent: {use_unified}")
        
        # Load and select data
        df = self.data_loader.load_data()
        selector = SampleSelector(selector_strategy, **selector_kwargs)
        selected_df = selector.select_samples(df, max_samples)
        
        # Get selection statistics
        selection_stats = selector.get_selection_stats(df, selected_df)
        print(f"Selected {len(selected_df)} samples out of {len(df)} total")
        
        # Initialize counters
        results = {
            "run_id": run_id,
            "total_samples": len(df),
            "reviewed_samples": len(selected_df),
            "verdicts": {"Correct": 0, "Incorrect": 0, "Uncertain": 0},
            "errors": 0,
            "start_time": time.time(),
            "use_unified_agent": use_unified
        }
        
        # Process samples based on agent type
        if use_unified:
            await self._process_samples_unified(selected_df, run_id, results, max_concurrent)
        else:
            await self._process_samples_individual(selected_df, run_id, results, max_concurrent)
        
        # Calculate final statistics
        results["end_time"] = time.time()
        results["duration_seconds"] = results["end_time"] - results["start_time"]
        results["avg_time_per_sample"] = results["duration_seconds"] / len(selected_df) if len(selected_df) > 0 else 0
        
        # Log run metadata
        run_metadata = {
            "model_name": self.model_name,
            "prompt_version": "v1.0.0",
            "execution_time": datetime.now().isoformat(),
            "selector_strategy": selector_strategy,
            "tokens_limit": 512,
            "total_samples": len(df),
            "reviewed_samples": len(selected_df),
            "selection_stats": selection_stats,
            "results": results,
            "max_concurrent": max_concurrent,
            "use_unified_agent": use_unified
        }
        
        self.logger.log_run_metadata(run_id, run_metadata)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"ASYNC REVIEW COMPLETE - Run ID: {run_id}")
        print(f"{'='*50}")
        print(f"Total samples: {len(df)}")
        print(f"Reviewed samples: {len(selected_df)}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"Average time per sample: {results['avg_time_per_sample']:.2f} seconds")
        print(f"Errors: {results['errors']}")
        print(f"Agent type: {'Unified' if use_unified else 'Individual'}")
        print(f"\nVerdict Distribution:")
        for verdict, count in results["verdicts"].items():
            percentage = (count / len(selected_df)) * 100 if len(selected_df) > 0 else 0
            print(f"  {verdict}: {count} ({percentage:.1f}%)")
        
        return results
    
    async def _process_samples_unified(self, selected_df: pd.DataFrame, run_id: str, 
                                      results: Dict[str, Any], max_concurrent: int) -> None:
        """Process samples using the unified agent for maximum efficiency."""
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Process samples concurrently
        tasks = []
        for idx, (_, sample) in enumerate(selected_df.iterrows(), 1):
            task = self._process_sample_unified(sample, idx, len(selected_df), semaphore, run_id, results)
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_sample_unified(self, sample: pd.Series, idx: int, total: int, 
                                     semaphore: asyncio.Semaphore, run_id: str, 
                                     results: Dict[str, Any]) -> None:
        """Process a single sample using the unified agent."""
        async with semaphore:
            print(f"\nProcessing sample {idx}/{total}: {sample['id']}")
            print(f"Text: {sample['text'][:50]}...")
            print(f"Predicted: {sample['pred_label']} (confidence: {sample['confidence']:.2f})")
            
            try:
                # Process with unified agent (single LLM call for all tasks)
                unified_result = await self.unified_agent.process_sample(
                    str(sample["text"]), 
                    str(sample["pred_label"]), 
                    float(sample["confidence"])
                )
                
                verdict = unified_result["verdict"]
                reasoning = unified_result["reasoning"]
                suggested_label = unified_result.get("suggested_label")
                explanation = unified_result["explanation"]
                
                results["verdicts"][verdict] += 1
                
                print(f"Verdict: {verdict}")
                print(f"Reasoning: {reasoning}")
                if suggested_label:
                    print(f"Suggested: {suggested_label}")
                print(f"Explanation: {explanation}")
                
                # Log the review
                metadata = {
                    "run_id": run_id,
                    "model_name": self.model_name,
                    "prompt_hash": unified_result["prompt_hash"],
                    "tokens_used": unified_result["tokens_used"],
                    "latency_ms": unified_result["latency_ms"],
                    "agent_type": "unified"
                }
                
                self.logger.log_review(
                    sample=sample.to_dict(),
                    verdict=verdict,
                    suggested_label=suggested_label,
                    reasoning=reasoning,
                    explanation=explanation,
                    metadata=metadata
                )
                
            except Exception as e:
                print(f"Error processing sample {sample['id']}: {e}")
                results["errors"] += 1
    
    async def _process_samples_individual(self, selected_df: pd.DataFrame, run_id: str, 
                                        results: Dict[str, Any], max_concurrent: int) -> None:
        """Process samples using individual agents (legacy method)."""
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Process samples concurrently
        tasks = []
        for idx, (_, sample) in enumerate(selected_df.iterrows(), 1):
            task = self._process_sample_individual(sample, idx, len(selected_df), semaphore, run_id, results)
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_sample_individual(self, sample: pd.Series, idx: int, total: int, 
                                       semaphore: asyncio.Semaphore, run_id: str, 
                                       results: Dict[str, Any]) -> None:
        """Process a single sample using individual agents (legacy method)."""
        async with semaphore:
            print(f"\nProcessing sample {idx}/{total}: {sample['id']}")
            print(f"Text: {sample['text'][:50]}...")
            print(f"Predicted: {sample['pred_label']} (confidence: {sample['confidence']:.2f})")
            
            try:
                # Step 1: Evaluate the prediction
                eval_result = await self.evaluator.evaluate_async(
                    str(sample["text"]), 
                    str(sample["pred_label"]), 
                    float(sample["confidence"])
                )
                
                verdict = eval_result["verdict"]
                reasoning = eval_result["reasoning"]
                results["verdicts"][verdict] += 1
                
                print(f"Verdict: {verdict}")
                print(f"Reasoning: {reasoning}")
                
                # Step 2: Get suggestion if incorrect
                suggested_label = None
                if verdict == "Incorrect":
                    prop_result = await self.proposer.propose_async(
                        str(sample["text"]), 
                        str(sample["pred_label"]), 
                        float(sample["confidence"])
                    )
                    suggested_label = prop_result["suggested_label"]
                
                # Step 3: Generate explanation
                explanation_context = f"verdict: {verdict}"
                if suggested_label:
                    explanation_context += f", suggested: {suggested_label}"
                
                reason_result = await self.reasoner.reason_async(
                    str(sample["text"]),
                    suggested_label if suggested_label else str(sample["pred_label"]),
                    explanation_context
                )
                explanation = reason_result["explanation"]
                
                # Step 4: Log the review
                metadata = {
                    "run_id": run_id,
                    "model_name": self.model_name,
                    "prompt_hash": eval_result["prompt_hash"],
                    "tokens_used": eval_result["tokens_used"] + 
                                  (prop_result["tokens_used"] if suggested_label else 0) +
                                  reason_result["tokens_used"],
                    "latency_ms": eval_result["latency_ms"] + 
                                 (prop_result["latency_ms"] if suggested_label else 0) +
                                 reason_result["latency_ms"],
                    "agent_type": "individual"
                }
                
                self.logger.log_review(
                    sample=sample.to_dict(),
                    verdict=verdict,
                    suggested_label=suggested_label,
                    reasoning=reasoning,
                    explanation=explanation,
                    metadata=metadata
                )
                
            except Exception as e:
                print(f"Error processing sample {sample['id']}: {e}")
                results["errors"] += 1

    def run_review(self, selector_strategy: str = "low_confidence", 
                   max_samples: Optional[int] = None, 
                   use_unified: bool = True,
                   **selector_kwargs) -> Dict[str, Any]:
        """
        Run the complete review process (synchronous wrapper for async).
        
        Args:
            selector_strategy: Strategy for selecting samples
            max_samples: Maximum number of samples to review
            use_unified: Whether to use unified agent (more efficient)
            **selector_kwargs: Additional arguments for sample selector
            
        Returns:
            Dictionary with review results and statistics
        """
        # Use default concurrency from config
        from .config import config
        max_concurrent = config.performance.max_concurrent_requests
        
        # Run async version
        return asyncio.run(self.run_review_async(
            selector_strategy=selector_strategy,
            max_samples=max_samples,
            max_concurrent=max_concurrent,
            use_unified=use_unified,
            **selector_kwargs
        ))
    
    async def review_single_sample_async(self, text: str, predicted_label: str, 
                                       confidence: float, sample_id: Optional[str] = None,
                                       use_unified: bool = True) -> Dict[str, Any]:
        """
        Review a single sample asynchronously (useful for API endpoints).
        
        Args:
            text: Input text
            predicted_label: Predicted label
            confidence: Confidence score
            sample_id: Optional sample ID
            use_unified: Whether to use unified agent
            
        Returns:
            Review result dictionary
        """
        if sample_id is None:
            sample_id = f"api_{int(time.time())}"
        
        sample = {
            "id": sample_id,
            "text": text,
            "pred_label": predicted_label,
            "confidence": confidence
        }
        
        print(f"Reviewing single sample: {sample_id}")
        print(f"Text: {text}")
        print(f"Predicted: {predicted_label} (confidence: {confidence:.2f})")
        print(f"Using unified agent: {use_unified}")
        
        try:
            if use_unified:
                # Use unified agent for efficiency
                result = await self.unified_agent.process_sample(text, predicted_label, confidence)
                
                return {
                    "sample_id": sample_id,
                    "verdict": result["verdict"],
                    "reasoning": result["reasoning"],
                    "suggested_label": result.get("suggested_label"),
                    "explanation": result["explanation"],
                    "success": result["success"],
                    "metadata": {
                        "model_name": self.model_name,
                        "prompt_hash": result["prompt_hash"],
                        "tokens_used": result["tokens_used"],
                        "latency_ms": result["latency_ms"],
                        "agent_type": "unified"
                    }
                }
            else:
                # Use individual agents (legacy method)
                # Step 1: Evaluate
                eval_result = await self.evaluator.evaluate_async(text, predicted_label, confidence)
                verdict = eval_result["verdict"]
                reasoning = eval_result["reasoning"]
                
                # Step 2: Get suggestion if incorrect
                suggested_label = None
                if verdict == "Incorrect":
                    prop_result = await self.proposer.propose_async(text, predicted_label, confidence)
                    suggested_label = prop_result["suggested_label"]
                
                # Step 3: Generate explanation
                explanation_context = f"verdict: {verdict}"
                if suggested_label:
                    explanation_context += f", suggested: {suggested_label}"
                
                reason_result = await self.reasoner.reason_async(
                    text,
                    suggested_label if suggested_label else predicted_label,
                    explanation_context
                )
                explanation = reason_result["explanation"]
                
                return {
                    "sample_id": sample_id,
                    "verdict": verdict,
                    "reasoning": reasoning,
                    "suggested_label": suggested_label,
                    "explanation": explanation,
                    "success": True,
                    "metadata": {
                        "model_name": self.model_name,
                        "prompt_hash": eval_result["prompt_hash"],
                        "tokens_used": eval_result["tokens_used"] + 
                                      (prop_result["tokens_used"] if suggested_label else 0) +
                                      reason_result["tokens_used"],
                        "latency_ms": eval_result["latency_ms"] + 
                                     (prop_result["latency_ms"] if suggested_label else 0) +
                                     reason_result["latency_ms"],
                        "agent_type": "individual"
                    }
                }
                
        except Exception as e:
            print(f"Error reviewing sample {sample_id}: {e}")
            return {
                "sample_id": sample_id,
                "verdict": "Uncertain",
                "reasoning": f"Review failed: {e}",
                "suggested_label": None,
                "explanation": "Review processing failed",
                "success": False,
                "error": str(e)
            }
    
    def review_single_sample(self, text: str, predicted_label: str, 
                           confidence: float, sample_id: Optional[str] = None,
                           use_unified: bool = True) -> Dict[str, Any]:
        """
        Review a single sample (synchronous wrapper).
        
        Args:
            text: Input text
            predicted_label: Predicted label
            confidence: Confidence score
            sample_id: Optional sample ID
            use_unified: Whether to use unified agent
            
        Returns:
            Review result dictionary
        """
        return asyncio.run(self.review_single_sample_async(
            text, predicted_label, confidence, sample_id, use_unified
        )) 
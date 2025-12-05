"""
Simplified LLM review engine for demo.
Evaluates predictions and suggests corrections using Ollama.
"""

import aiohttp
import asyncio
import hashlib
import time
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


class ReviewEngine:
    """Minimal LLM-powered review engine."""
    
    def __init__(
        self, 
        model_name: str = "mistral",
        ollama_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.labels = self._load_labels()
    
    def _load_labels(self) -> Dict:
        """Load label definitions from config."""
        config_path = Path(__file__).parent.parent / "configs" / "labels.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {"labels": []}
    
    def _build_prompt(self, text: str, pred_label: str, confidence: float) -> str:
        """Build the unified review prompt."""
        labels_desc = "\n".join([
            f"- {l['name']}: {l['definition']}"
            for l in self.labels.get("labels", [])
        ])
        
        return f"""You are a semantic auditor reviewing text classification predictions.

## Available Labels
{labels_desc}

## Task
Analyze whether the predicted label correctly captures the semantic intent of the text.

## Input
Text: "{text}"
Predicted Label: {pred_label}
Confidence: {confidence}

## Instructions
Respond in this exact format:
VERDICT: [Correct/Incorrect/Uncertain]
REASONING: [One sentence explaining why]
SUGGESTED_LABEL: [Only if Incorrect, otherwise "None"]
EXPLANATION: [Brief stakeholder-friendly explanation]

Be precise and concise."""

    async def review_sample_async(
        self, 
        text: str, 
        pred_label: str, 
        confidence: float,
        sample_id: str
    ) -> Dict[str, Any]:
        """Review a single sample asynchronously."""
        prompt = self._build_prompt(text, pred_label, confidence)
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        
        start_time = time.time()
        
        try:
            response = await self._call_ollama(prompt)
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Parse response
            result = self._parse_response(response)
            result.update({
                "sample_id": sample_id,
                "text": text,
                "pred_label": pred_label,
                "confidence": confidence,
                "success": True,
                "prompt_hash": prompt_hash,
                "latency_ms": latency_ms,
            })
            return result
            
        except Exception as e:
            return {
                "sample_id": sample_id,
                "text": text,
                "pred_label": pred_label,
                "confidence": confidence,
                "verdict": "Uncertain",
                "reasoning": f"Error: {str(e)}",
                "suggested_label": None,
                "explanation": "Review failed",
                "success": False,
                "error": str(e),
            }
    
    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300}
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=60) as resp:
                if resp.status != 200:
                    raise Exception(f"Ollama error: {resp.status}")
                result = await resp.json()
                return result.get("response", "")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        lines = response.strip().split("\n")
        result = {
            "verdict": "Uncertain",
            "reasoning": "",
            "suggested_label": None,
            "explanation": "",
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith("VERDICT:"):
                verdict = line.replace("VERDICT:", "").strip()
                if verdict in ["Correct", "Incorrect", "Uncertain"]:
                    result["verdict"] = verdict
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.replace("REASONING:", "").strip()
            elif line.startswith("SUGGESTED_LABEL:"):
                suggested = line.replace("SUGGESTED_LABEL:", "").strip()
                if suggested.lower() != "none":
                    result["suggested_label"] = suggested
            elif line.startswith("EXPLANATION:"):
                result["explanation"] = line.replace("EXPLANATION:", "").strip()
        
        return result
    
    async def review_batch_async(
        self, 
        samples: List[Dict[str, Any]],
        on_progress: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Review multiple samples."""
        results = []
        
        for i, sample in enumerate(samples):
            result = await self.review_sample_async(
                text=sample["text"],
                pred_label=sample["pred_label"],
                confidence=sample["confidence"],
                sample_id=sample["id"]
            )
            result["ground_truth"] = sample.get("ground_truth")
            results.append(result)
            
            if on_progress:
                on_progress(i + 1, len(samples))
        
        return results
    
    def generate_mock_results(self, samples: List[Dict]) -> List[Dict]:
        """Generate mock results without LLM (for --no-llm mode)."""
        results = []
        
        for sample in samples:
            is_correct = not sample.get("is_misclassified", False)
            
            if is_correct:
                verdict = "Correct"
                suggested = None
                reasoning = "The predicted label accurately captures the semantic intent."
            else:
                verdict = "Incorrect"
                suggested = sample.get("ground_truth", "Unknown")
                reasoning = f"The text indicates {suggested}, not {sample['pred_label']}."
            
            results.append({
                "sample_id": sample["id"],
                "text": sample["text"],
                "pred_label": sample["pred_label"],
                "confidence": sample["confidence"],
                "ground_truth": sample.get("ground_truth"),
                "verdict": verdict,
                "reasoning": reasoning,
                "suggested_label": suggested,
                "explanation": f"Classification as {suggested or sample['pred_label']}.",
                "success": True,
            })
        
        return results


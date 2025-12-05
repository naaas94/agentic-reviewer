"""
LLM-powered report generator for portfolio demonstration.

Generates comprehensive Markdown reports analyzing review results,
providing insights suitable for stakeholder communication.
"""

import aiohttp
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional


class ReportGenerator:
    """Generates analysis reports using LLM."""
    
    REPORT_PROMPT = """You are a data quality analyst writing a professional report about a classification audit.

## Context
An automated review system analyzed {total_samples} text classification predictions for GDPR/CCPA data subject requests.
The system evaluated whether each prediction was correct and suggested alternatives where needed.

## Results Summary
- Correct predictions: {correct_count} ({correct_pct:.1f}%)
- Incorrect predictions: {incorrect_count} ({incorrect_pct:.1f}%)  
- Uncertain cases: {uncertain_count} ({uncertain_pct:.1f}%)

## Label Distribution
{label_distribution}

## Sample Incorrect Predictions
{incorrect_samples}

## Your Task
Write a professional Markdown report (500-800 words) that includes:
1. **Executive Summary** - Key findings in 2-3 sentences
2. **Methodology** - Brief description of the semantic audit approach
3. **Findings** - Detailed analysis of the results with specific examples
4. **Error Pattern Analysis** - Common misclassification patterns observed
5. **Recommendations** - Actionable suggestions for improving the classifier
6. **Conclusion** - Overall assessment and next steps

Use professional language suitable for both technical and non-technical stakeholders.
Format the output as clean Markdown with proper headers.
Do not include any preamble or meta-commentary - start directly with the report content."""

    def __init__(self, model_name: str = "mistral", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
    
    async def generate_report_async(
        self,
        results: List[Dict[str, Any]],
        run_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.
        
        Args:
            results: List of review results
            run_metadata: Metadata about the run
            
        Returns:
            Dict with report content and generation metadata
        """
        # Calculate statistics
        stats = self._calculate_stats(results)
        
        # Format prompt
        prompt = self._format_prompt(results, stats)
        
        # Generate report via LLM
        try:
            report_content = await self._call_llm(prompt)
            success = True
            error = None
        except Exception as e:
            report_content = self._generate_fallback_report(results, stats, run_metadata)
            success = False
            error = str(e)
        
        # Add header with run metadata
        full_report = self._add_report_header(report_content, run_metadata, stats)
        
        return {
            "content": full_report,
            "stats": stats,
            "generated_at": datetime.now().isoformat(),
            "model_used": self.model_name if success else "fallback",
            "success": success,
            "error": error
        }
    
    def _calculate_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from results."""
        total = len(results)
        verdicts = {"Correct": 0, "Incorrect": 0, "Uncertain": 0}
        label_dist = {}
        incorrect_samples = []
        
        for r in results:
            verdict = r.get("verdict", "Uncertain")
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            
            pred_label = r.get("pred_label", "Unknown")
            label_dist[pred_label] = label_dist.get(pred_label, 0) + 1
            
            if verdict == "Incorrect" and len(incorrect_samples) < 5:
                incorrect_samples.append({
                    "text": r.get("text", "")[:100],
                    "predicted": pred_label,
                    "suggested": r.get("suggested_label", "N/A"),
                    "reasoning": r.get("reasoning", "")[:150]
                })
        
        return {
            "total": total,
            "correct": verdicts.get("Correct", 0),
            "incorrect": verdicts.get("Incorrect", 0),
            "uncertain": verdicts.get("Uncertain", 0),
            "correct_pct": (verdicts.get("Correct", 0) / total * 100) if total > 0 else 0,
            "incorrect_pct": (verdicts.get("Incorrect", 0) / total * 100) if total > 0 else 0,
            "uncertain_pct": (verdicts.get("Uncertain", 0) / total * 100) if total > 0 else 0,
            "label_distribution": label_dist,
            "incorrect_samples": incorrect_samples
        }
    
    def _format_prompt(self, results: List[Dict[str, Any]], stats: Dict[str, Any]) -> str:
        """Format the report generation prompt."""
        # Format label distribution
        label_dist_str = "\n".join([
            f"- {label}: {count} samples"
            for label, count in stats["label_distribution"].items()
        ])
        
        # Format incorrect samples
        incorrect_str = ""
        for i, sample in enumerate(stats["incorrect_samples"], 1):
            incorrect_str += f"""
{i}. Text: "{sample['text']}..."
   Predicted: {sample['predicted']} → Suggested: {sample['suggested']}
   Reason: {sample['reasoning']}...
"""
        
        return self.REPORT_PROMPT.format(
            total_samples=stats["total"],
            correct_count=stats["correct"],
            correct_pct=stats["correct_pct"],
            incorrect_count=stats["incorrect"],
            incorrect_pct=stats["incorrect_pct"],
            uncertain_count=stats["uncertain"],
            uncertain_pct=stats["uncertain_pct"],
            label_distribution=label_dist_str,
            incorrect_samples=incorrect_str or "No incorrect predictions to display."
        )
    
    async def _call_llm(self, prompt: str) -> str:
        """Call Ollama LLM to generate report."""
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1500
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=120) as response:
                if response.status != 200:
                    raise Exception(f"LLM request failed with status {response.status}")
                
                result = await response.json()
                return result.get("response", "")
    
    def _generate_fallback_report(
        self, 
        results: List[Dict[str, Any]], 
        stats: Dict[str, Any],
        run_metadata: Dict[str, Any]
    ) -> str:
        """Generate a structured report without LLM (fallback)."""
        return f"""# Classification Audit Report

## Executive Summary

This automated semantic audit reviewed {stats['total']} text classification predictions.
The analysis identified {stats['incorrect']} predictions ({stats['incorrect_pct']:.1f}%) requiring correction.

## Methodology

The agentic reviewer system evaluated each prediction by:
1. Semantically analyzing whether the predicted label fits the input text
2. Suggesting alternative labels for incorrect predictions
3. Generating natural language explanations for each decision

## Findings

### Verdict Distribution

| Verdict | Count | Percentage |
|---------|-------|------------|
| Correct | {stats['correct']} | {stats['correct_pct']:.1f}% |
| Incorrect | {stats['incorrect']} | {stats['incorrect_pct']:.1f}% |
| Uncertain | {stats['uncertain']} | {stats['uncertain_pct']:.1f}% |

### Label Distribution

| Label | Count |
|-------|-------|
{chr(10).join([f"| {label} | {count} |" for label, count in stats['label_distribution'].items()])}

## Sample Corrections

{self._format_sample_corrections(stats['incorrect_samples'])}

## Recommendations

1. Review confusion patterns between similar labels
2. Consider adding more training examples for commonly misclassified categories
3. Implement confidence thresholds for human review escalation

## Conclusion

The semantic audit demonstrates the system's capability to identify and correct 
classification errors. Continuous monitoring and iterative improvement are recommended.

---
*Report generated automatically by Agentic Reviewer*
"""
    
    def _format_sample_corrections(self, samples: List[Dict[str, Any]]) -> str:
        """Format sample corrections for the report."""
        if not samples:
            return "*No corrections needed*"
        
        formatted = ""
        for i, s in enumerate(samples, 1):
            formatted += f"""
**Sample {i}:**
- Text: "{s['text']}..."
- Predicted: `{s['predicted']}` → Suggested: `{s['suggested']}`
- Reasoning: {s['reasoning']}
"""
        return formatted
    
    def _add_report_header(
        self, 
        content: str, 
        run_metadata: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> str:
        """Add metadata header to report."""
        header = f"""---
title: Classification Audit Report
run_id: {run_metadata.get('run_id', 'N/A')}
generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
model: {run_metadata.get('model_name', self.model_name)}
samples_reviewed: {stats['total']}
accuracy: {stats['correct_pct']:.1f}%
---

"""
        return header + content


def generate_report_sync(
    results: List[Dict[str, Any]],
    run_metadata: Dict[str, Any],
    model_name: str = "mistral",
    ollama_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """Synchronous wrapper for report generation."""
    generator = ReportGenerator(model_name, ollama_url)
    return asyncio.run(generator.generate_report_async(results, run_metadata))


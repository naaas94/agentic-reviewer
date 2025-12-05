#!/usr/bin/env python3
"""
Portfolio Demo: Agentic Reviewer
================================

Single-command demonstration that generates synthetic data, runs LLM-powered
semantic auditing, and produces traceable artifacts.

Usage:
    python run_portfolio_demo.py                    # Default: 15 samples
    python run_portfolio_demo.py --samples 30       # Custom sample count
    python run_portfolio_demo.py --no-llm           # Dry run without Ollama
    python run_portfolio_demo.py --seed 42          # Reproducible run

Outputs:
    outputs/runs/<timestamp>/
    ├── 00_config.json              # Run configuration
    ├── 01_synthetic_data.csv       # Generated input data
    ├── 02_review_results.json      # Full LLM responses
    ├── 03_labeled_dataset.csv      # Final labeled dataset
    ├── 04_report.md                # LLM-generated analysis
    └── 05_metrics.json             # Performance metrics
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.synthetic_generator import SyntheticGenerator
from core.report_generator import ReportGenerator
from core.config import config


# ============================================================================
# TERMINAL UI COMPONENTS
# ============================================================================

class TerminalUI:
    """Clean terminal output for portfolio demonstration."""
    
    # Box drawing characters
    TL, TR, BL, BR = "╔", "╗", "╚", "╝"
    H, V = "═", "║"
    VL, VR = "╠", "╣"
    
    WIDTH = 66
    
    @classmethod
    def header(cls, title: str) -> None:
        """Print header box."""
        print(f"\n{cls.TL}{cls.H * (cls.WIDTH - 2)}{cls.TR}")
        print(f"{cls.V}{title.center(cls.WIDTH - 2)}{cls.V}")
    
    @classmethod
    def separator(cls) -> None:
        """Print separator line."""
        print(f"{cls.VL}{cls.H * (cls.WIDTH - 2)}{cls.VR}")
    
    @classmethod
    def row(cls, content: str) -> None:
        """Print content row."""
        print(f"{cls.V} {content.ljust(cls.WIDTH - 4)} {cls.V}")
    
    @classmethod
    def footer(cls) -> None:
        """Print footer line."""
        print(f"{cls.BL}{cls.H * (cls.WIDTH - 2)}{cls.BR}")
    
    @classmethod
    def phase(cls, num: int, name: str, status: str, detail: str = "") -> None:
        """Print phase status."""
        icon = "✓" if "✓" in status or "complete" in status.lower() else "→"
        line = f"PHASE {num}: {name.ljust(30)} {icon} {detail}"
        cls.row(line)
    
    @classmethod
    def stat(cls, label: str, value: Any, indent: int = 0) -> None:
        """Print a statistic line."""
        prefix = "│  " * indent + ("├─ " if indent else "")
        line = f"{prefix}{label}: {value}"
        cls.row(line)
    
    @classmethod
    def progress(cls, current: int, total: int, prefix: str = "Progress") -> None:
        """Print progress indicator (overwrites line)."""
        pct = (current / total) * 100 if total > 0 else 0
        bar_width = 30
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r{cls.V} {prefix}: [{bar}] {current}/{total} ({pct:.0f}%) ".ljust(cls.WIDTH - 1) + cls.V, end="", flush=True)
    
    @classmethod
    def progress_done(cls) -> None:
        """Clear progress line."""
        print()


# ============================================================================
# HEALTH CHECKS
# ============================================================================

def check_ollama_available(url: str = "http://localhost:11434") -> tuple[bool, str]:
    """Check if Ollama is running and has the required model."""
    try:
        import requests
        response = requests.get(f"{url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Check for mistral
            has_mistral = any("mistral" in name.lower() for name in model_names)
            if has_mistral:
                return True, "Ollama ready with mistral"
            else:
                return False, f"Mistral not found. Available: {model_names[:3]}"
        return False, f"Ollama returned status {response.status_code}"
    except Exception as e:
        return False, f"Ollama not reachable: {type(e).__name__}"


def preflight_checks(require_llm: bool = True) -> Dict[str, Any]:
    """Run preflight checks before demo."""
    checks = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "cwd": os.getcwd(),
        "checks": {}
    }
    
    # Check 1: Output directory writable
    try:
        os.makedirs("outputs/runs", exist_ok=True)
        checks["checks"]["output_dir"] = {"status": "ok", "path": "outputs/runs"}
    except Exception as e:
        checks["checks"]["output_dir"] = {"status": "error", "error": str(e)}
    
    # Check 2: Ollama availability
    if require_llm:
        available, message = check_ollama_available()
        checks["checks"]["ollama"] = {
            "status": "ok" if available else "error",
            "message": message
        }
        checks["llm_available"] = available
    else:
        checks["checks"]["ollama"] = {"status": "skipped", "message": "LLM not required"}
        checks["llm_available"] = False
    
    return checks


# ============================================================================
# MAIN DEMO ORCHESTRATOR
# ============================================================================

class PortfolioDemo:
    """Orchestrates the complete portfolio demonstration."""
    
    def __init__(
        self,
        n_samples: int = 15,
        seed: Optional[int] = None,
        use_llm: bool = True,
        model_name: str = "mistral",
        ollama_url: str = "http://localhost:11434"
    ):
        self.n_samples = n_samples
        self.seed = seed or int(datetime.now().timestamp())
        self.use_llm = use_llm
        self.model_name = model_name
        self.ollama_url = ollama_url
        
        # Generate run ID
        self.run_id = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.run_dir = f"outputs/runs/{self.run_id}"
        
        # Metrics collection
        self.metrics = {
            "run_id": self.run_id,
            "start_time": None,
            "end_time": None,
            "phases": {}
        }
    
    def run(self) -> Dict[str, Any]:
        """Execute the complete demo pipeline."""
        self.metrics["start_time"] = datetime.now().isoformat()
        
        # Print header
        TerminalUI.header("AGENTIC REVIEWER — PORTFOLIO DEMO")
        TerminalUI.separator()
        TerminalUI.row(f"Run ID: {self.run_id}")
        TerminalUI.row(f"Model: {self.model_name} | Samples: {self.n_samples} | Seed: {self.seed}")
        TerminalUI.row(f"LLM Mode: {'Enabled' if self.use_llm else 'Disabled (dry run)'}")
        TerminalUI.separator()
        
        # Create run directory
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Phase 1: Generate synthetic data
        samples, gen_metadata = self._phase1_generate_data()
        
        # Phase 2: Run review loop
        results = self._phase2_review(samples)
        
        # Phase 3: Generate report
        report = self._phase3_report(results)
        
        # Phase 4: Save artifacts
        self._phase4_save_artifacts(samples, gen_metadata, results, report)
        
        # Calculate final metrics
        self._finalize_metrics(results)
        
        # Print results summary
        self._print_summary(results)
        
        return self.metrics
    
    def _phase1_generate_data(self) -> tuple[List[Dict], Dict]:
        """Phase 1: Generate synthetic classification data."""
        phase_start = time.time()
        
        generator = SyntheticGenerator(seed=self.seed)
        samples = generator.generate_samples(
            n_samples=self.n_samples,
            misclassification_rate=0.35,
            low_confidence_rate=0.25
        )
        metadata = generator.get_generation_metadata()
        
        phase_time = time.time() - phase_start
        self.metrics["phases"]["generate"] = {
            "duration_ms": int(phase_time * 1000),
            "samples_generated": len(samples)
        }
        
        TerminalUI.phase(1, "Synthetic Data Generation", "✓", f"{len(samples)} samples")
        
        return samples, metadata
    
    def _phase2_review(self, samples: List[Dict]) -> List[Dict]:
        """Phase 2: Run LLM review loop on samples."""
        phase_start = time.time()
        
        if not self.use_llm:
            # Dry run: generate mock results
            results = self._generate_mock_results(samples)
            TerminalUI.phase(2, "LLM Review Loop", "✓", f"{len(results)} reviews (mock)")
        else:
            # Real LLM processing
            results = asyncio.run(self._run_reviews_async(samples))
            success_count = sum(1 for r in results if r.get("success", False))
            TerminalUI.phase(2, "LLM Review Loop", "✓", f"{success_count}/{len(results)} reviews")
        
        phase_time = time.time() - phase_start
        self.metrics["phases"]["review"] = {
            "duration_ms": int(phase_time * 1000),
            "samples_reviewed": len(results),
            "llm_enabled": self.use_llm
        }
        
        return results
    
    async def _run_reviews_async(self, samples: List[Dict]) -> List[Dict]:
        """Run async LLM reviews."""
        from core.review_loop import ReviewLoop
        
        review_loop = ReviewLoop(self.model_name, self.ollama_url)
        results = []
        
        for i, sample in enumerate(samples):
            TerminalUI.progress(i + 1, len(samples), "Reviewing")
            
            try:
                result = await review_loop.review_single_sample_async(
                    text=sample["text"],
                    predicted_label=sample["pred_label"],
                    confidence=sample["confidence"],
                    sample_id=sample["id"],
                    use_unified=True
                )
                result["ground_truth"] = sample.get("ground_truth")
                result["text"] = sample["text"]
                result["pred_label"] = sample["pred_label"]
                result["confidence"] = sample["confidence"]
                results.append(result)
            except Exception as e:
                results.append({
                    "sample_id": sample["id"],
                    "text": sample["text"],
                    "pred_label": sample["pred_label"],
                    "confidence": sample["confidence"],
                    "ground_truth": sample.get("ground_truth"),
                    "verdict": "Uncertain",
                    "reasoning": f"Error: {str(e)}",
                    "suggested_label": None,
                    "explanation": "Review failed",
                    "success": False,
                    "error": str(e)
                })
        
        TerminalUI.progress_done()
        return results
    
    def _generate_mock_results(self, samples: List[Dict]) -> List[Dict]:
        """Generate mock results for dry run mode."""
        import random
        random.seed(self.seed)
        
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
                "explanation": f"This text should be classified as {suggested or sample['pred_label']}.",
                "success": True,
                "metadata": {"agent_type": "mock"}
            })
        
        return results
    
    def _phase3_report(self, results: List[Dict]) -> Dict:
        """Phase 3: Generate analysis report."""
        phase_start = time.time()
        
        run_metadata = {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "seed": self.seed
        }
        
        if self.use_llm:
            from core.report_generator import ReportGenerator
            generator = ReportGenerator(self.model_name, self.ollama_url)
            report = asyncio.run(generator.generate_report_async(results, run_metadata))
        else:
            # Generate fallback report
            from core.report_generator import ReportGenerator
            generator = ReportGenerator(self.model_name, self.ollama_url)
            stats = generator._calculate_stats(results)
            report = {
                "content": generator._generate_fallback_report(results, stats, run_metadata),
                "stats": stats,
                "generated_at": datetime.now().isoformat(),
                "model_used": "fallback",
                "success": True
            }
            report["content"] = generator._add_report_header(report["content"], run_metadata, stats)
        
        word_count = len(report["content"].split())
        TerminalUI.phase(3, "Report Generation", "✓", f"{word_count} words")
        
        phase_time = time.time() - phase_start
        self.metrics["phases"]["report"] = {
            "duration_ms": int(phase_time * 1000),
            "word_count": word_count,
            "llm_generated": report.get("model_used") != "fallback"
        }
        
        return report
    
    def _phase4_save_artifacts(
        self,
        samples: List[Dict],
        gen_metadata: Dict,
        results: List[Dict],
        report: Dict
    ) -> None:
        """Phase 4: Save all artifacts to disk."""
        phase_start = time.time()
        
        # 00_config.json
        config_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "n_samples": self.n_samples,
                "seed": self.seed,
                "model_name": self.model_name,
                "use_llm": self.use_llm
            },
            "generation_metadata": gen_metadata
        }
        with open(f"{self.run_dir}/00_config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        # 01_synthetic_data.csv
        df_samples = pd.DataFrame(samples)
        df_samples.to_csv(f"{self.run_dir}/01_synthetic_data.csv", index=False)
        
        # 02_review_results.json
        with open(f"{self.run_dir}/02_review_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # 03_labeled_dataset.csv
        labeled_data = []
        for r in results:
            labeled_data.append({
                "id": r.get("sample_id"),
                "text": r.get("text"),
                "original_prediction": r.get("pred_label"),
                "confidence": r.get("confidence"),
                "verdict": r.get("verdict"),
                "corrected_label": r.get("suggested_label") or r.get("pred_label"),
                "reasoning": r.get("reasoning"),
                "explanation": r.get("explanation"),
                "ground_truth": r.get("ground_truth")
            })
        df_labeled = pd.DataFrame(labeled_data)
        df_labeled.to_csv(f"{self.run_dir}/03_labeled_dataset.csv", index=False)
        
        # 04_report.md
        with open(f"{self.run_dir}/04_report.md", "w", encoding="utf-8") as f:
            f.write(report["content"])
        
        # 05_metrics.json (saved in finalize)
        
        phase_time = time.time() - phase_start
        self.metrics["phases"]["save"] = {
            "duration_ms": int(phase_time * 1000),
            "artifacts_saved": 5
        }
        
        TerminalUI.phase(4, "Artifact Generation", "✓", "5 files saved")
    
    def _finalize_metrics(self, results: List[Dict]) -> None:
        """Calculate and save final metrics."""
        self.metrics["end_time"] = datetime.now().isoformat()
        
        # Calculate verdict distribution
        verdicts = {"Correct": 0, "Incorrect": 0, "Uncertain": 0}
        total_tokens = 0
        total_latency = 0
        
        for r in results:
            verdict = r.get("verdict", "Uncertain")
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            
            metadata = r.get("metadata", {})
            total_tokens += metadata.get("tokens_used", 0)
            total_latency += metadata.get("latency_ms", 0)
        
        self.metrics["results"] = {
            "total_samples": len(results),
            "verdicts": verdicts,
            "accuracy_rate": verdicts["Correct"] / len(results) * 100 if results else 0,
            "correction_rate": verdicts["Incorrect"] / len(results) * 100 if results else 0
        }
        
        self.metrics["performance"] = {
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency,
            "avg_latency_ms": total_latency / len(results) if results else 0
        }
        
        # Calculate total duration
        start = datetime.fromisoformat(self.metrics["start_time"])
        end = datetime.fromisoformat(self.metrics["end_time"])
        self.metrics["total_duration_seconds"] = (end - start).total_seconds()
        
        # Save metrics
        with open(f"{self.run_dir}/05_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
    
    def _print_summary(self, results: List[Dict]) -> None:
        """Print final summary to terminal."""
        verdicts = self.metrics["results"]["verdicts"]
        total = self.metrics["results"]["total_samples"]
        
        TerminalUI.separator()
        TerminalUI.row("RESULTS")
        
        for verdict, count in verdicts.items():
            pct = (count / total * 100) if total > 0 else 0
            suffix = " → alternatives suggested" if verdict == "Incorrect" else ""
            TerminalUI.stat(verdict, f"{count} ({pct:.1f}%){suffix}", indent=1)
        
        TerminalUI.row("")
        
        perf = self.metrics["performance"]
        duration = self.metrics["total_duration_seconds"]
        TerminalUI.row(f"Avg Latency: {perf['avg_latency_ms']:.0f}ms | Tokens: {perf['total_tokens']:,} | Duration: {duration:.1f}s")
        
        TerminalUI.separator()
        TerminalUI.row("ARTIFACTS GENERATED")
        TerminalUI.stat(self.run_dir, "", indent=1)
        
        artifacts = [
            "00_config.json",
            "01_synthetic_data.csv",
            "02_review_results.json",
            "03_labeled_dataset.csv",
            "04_report.md",
            "05_metrics.json"
        ]
        for artifact in artifacts:
            TerminalUI.row(f"    ├─ {artifact}")
        
        TerminalUI.footer()
        
        # Final message
        print(f"\n✓ Demo complete. Explore results at: {self.run_dir}/")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Agentic Reviewer Portfolio Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_portfolio_demo.py                  # Default: 15 samples with LLM
  python run_portfolio_demo.py --samples 30    # More samples
  python run_portfolio_demo.py --no-llm        # Dry run without Ollama
  python run_portfolio_demo.py --seed 42       # Reproducible run
        """
    )
    
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=15,
        help="Number of synthetic samples to generate (default: 15)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run without LLM (generates mock results)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        help="Ollama model name (default: mistral)"
    )
    
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Run preflight checks
    print("\n🔍 Running preflight checks...")
    checks = preflight_checks(require_llm=not args.no_llm)
    
    # Check if LLM is needed but not available
    if not args.no_llm and not checks.get("llm_available", False):
        print(f"⚠️  {checks['checks']['ollama']['message']}")
        print("   Run with --no-llm for dry run, or start Ollama first.")
        print("   To start Ollama: ollama serve && ollama pull mistral")
        sys.exit(1)
    
    print("✓ Preflight checks passed\n")
    
    # Run demo
    demo = PortfolioDemo(
        n_samples=args.samples,
        seed=args.seed,
        use_llm=not args.no_llm,
        model_name=args.model,
        ollama_url=args.ollama_url
    )
    
    try:
        metrics = demo.run()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


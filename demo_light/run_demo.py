#!/usr/bin/env python3
"""
Agentic Reviewer Demo
=====================

Single-command demonstration of LLM-powered semantic auditing.

Usage:
    python run_demo.py                  # Full demo (requires Ollama)
    python run_demo.py --no-llm         # Dry run without Ollama
    python run_demo.py --samples 20     # Custom sample count
    python run_demo.py --seed 42        # Reproducible run
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd

from core.synthetic_generator import SyntheticGenerator
from core.review_engine import ReviewEngine
from core.report_generator import ReportGenerator


# ============================================================================
# TERMINAL UI
# ============================================================================

class UI:
    """Minimal terminal UI."""
    
    W = 64
    
    @classmethod
    def header(cls, text: str):
        print(f"\n╔{'═' * (cls.W - 2)}╗")
        print(f"║{text.center(cls.W - 2)}║")
    
    @classmethod
    def sep(cls):
        print(f"╠{'═' * (cls.W - 2)}╣")
    
    @classmethod
    def row(cls, text: str):
        print(f"║ {text.ljust(cls.W - 4)} ║")
    
    @classmethod
    def footer(cls):
        print(f"╚{'═' * (cls.W - 2)}╝")
    
    @classmethod
    def phase(cls, num: int, name: str, result: str):
        cls.row(f"PHASE {num}: {name.ljust(28)} ✓ {result}")
    
    @classmethod
    def progress(cls, current: int, total: int):
        pct = current / total * 100 if total else 0
        bar = "█" * int(30 * current / total) + "░" * (30 - int(30 * current / total))
        print(f"\r║ Progress: [{bar}] {current}/{total} ({pct:.0f}%) ".ljust(cls.W - 1) + "║", end="", flush=True)


# ============================================================================
# HEALTH CHECKS
# ============================================================================

def check_ollama(url: str = "http://localhost:11434") -> tuple:
    """Check Ollama availability and model status."""
    try:
        import urllib.request
        req = urllib.request.urlopen(f"{url}/api/tags", timeout=5)
        data = json.loads(req.read())
        models = [m.get("name", "") for m in data.get("models", [])]
        
        # Check for any usable model (mistral, llama, tinyllama, etc.)
        usable_models = [m for m in models if any(x in m.lower() for x in ["mistral", "llama", "phi", "gemma"])]
        
        if usable_models:
            return True, usable_models[0], "ready"
        else:
            return True, None, "no_model"  # Ollama running but no model
    except Exception:
        return False, None, "not_running"


def pull_model(model: str = "mistral", url: str = "http://localhost:11434") -> bool:
    """Pull a model from Ollama (with progress indication)."""
    import urllib.request
    import json
    
    print(f"\n📥 Pulling {model} model (this may take a few minutes)...")
    print("   This is a one-time download.\n")
    
    try:
        req = urllib.request.Request(
            f"{url}/api/pull",
            data=json.dumps({"name": model}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=600) as response:
            # Stream the response to show progress
            while True:
                line = response.readline()
                if not line:
                    break
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if "pulling" in status:
                        completed = data.get("completed", 0)
                        total = data.get("total", 1)
                        pct = (completed / total * 100) if total else 0
                        print(f"\r   Progress: {pct:.0f}%", end="", flush=True)
                    elif status == "success":
                        print(f"\r   Progress: 100%")
                        print(f"✓ Model {model} ready!")
                        return True
                except json.JSONDecodeError:
                    pass
        
        return True
    except Exception as e:
        print(f"\n❌ Failed to pull model: {e}")
        return False


# ============================================================================
# DEMO ORCHESTRATOR
# ============================================================================

class Demo:
    """Main demo orchestrator."""
    
    def __init__(self, n_samples: int, seed: int, use_llm: bool, model: str):
        self.n_samples = n_samples
        self.seed = seed
        self.use_llm = use_llm
        self.model = model
        self.run_id = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.run_dir = f"outputs/{self.run_id}"
    
    def run(self) -> Dict[str, Any]:
        """Execute complete demo."""
        start_time = time.time()
        
        # Header
        UI.header("AGENTIC REVIEWER DEMO")
        UI.sep()
        UI.row(f"Run ID: {self.run_id}")
        UI.row(f"Samples: {self.n_samples} | Seed: {self.seed} | LLM: {'On' if self.use_llm else 'Off'}")
        UI.sep()
        
        # Create output directory
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Phase 1: Generate data
        generator = SyntheticGenerator(seed=self.seed)
        samples = generator.generate_samples(self.n_samples)
        UI.phase(1, "Generate Synthetic Data", f"{len(samples)} samples")
        
        # Phase 2: Review
        engine = ReviewEngine(model_name=self.model)
        if self.use_llm:
            results = asyncio.run(self._review_with_progress(engine, samples))
        else:
            results = engine.generate_mock_results(samples)
        
        success = sum(1 for r in results if r.get("success"))
        UI.phase(2, "LLM Review", f"{success}/{len(results)} reviews")
        
        # Phase 3: Generate report
        report_gen = ReportGenerator()
        report = report_gen.generate_report(results, self.run_id, {"seed": self.seed})
        UI.phase(3, "Generate Report", f"{len(report.split())} words")
        
        # Phase 4: Save artifacts
        self._save_artifacts(samples, results, report, generator.get_metadata())
        UI.phase(4, "Save Artifacts", "5 files")
        
        # Results summary
        duration = time.time() - start_time
        stats = self._calculate_stats(results)
        
        UI.sep()
        UI.row("RESULTS")
        UI.row(f"├─ Correct:   {stats['correct']:3} ({stats['correct_pct']:.1f}%)")
        UI.row(f"├─ Incorrect: {stats['incorrect']:3} ({stats['incorrect_pct']:.1f}%) → corrections suggested")
        UI.row(f"└─ Uncertain: {stats['uncertain']:3} ({stats['uncertain_pct']:.1f}%)")
        UI.row("")
        UI.row(f"Duration: {duration:.1f}s | Output: {self.run_dir}/")
        UI.footer()
        
        print(f"\n✓ Demo complete. Explore: {self.run_dir}/")
        
        return {"run_id": self.run_id, "stats": stats, "duration": duration}
    
    async def _review_with_progress(self, engine: ReviewEngine, samples: List[Dict]) -> List[Dict]:
        """Review with progress indicator."""
        def on_progress(current, total):
            UI.progress(current, total)
        
        results = await engine.review_batch_async(samples, on_progress)
        print()  # Clear progress line
        return results
    
    def _save_artifacts(self, samples, results, report, metadata):
        """Save all output artifacts."""
        # Config
        with open(f"{self.run_dir}/00_config.json", "w") as f:
            json.dump({
                "run_id": self.run_id,
                "seed": self.seed,
                "n_samples": self.n_samples,
                "use_llm": self.use_llm,
                "model": self.model,
                "generated_at": datetime.now().isoformat(),
            }, f, indent=2)
        
        # Synthetic data
        pd.DataFrame(samples).to_csv(f"{self.run_dir}/01_synthetic_data.csv", index=False)
        
        # Results
        with open(f"{self.run_dir}/02_review_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Labeled dataset
        labeled = [{
            "id": r["sample_id"],
            "text": r["text"],
            "original_label": r["pred_label"],
            "verdict": r["verdict"],
            "corrected_label": r.get("suggested_label") or r["pred_label"],
            "reasoning": r["reasoning"],
            "ground_truth": r.get("ground_truth"),
        } for r in results]
        pd.DataFrame(labeled).to_csv(f"{self.run_dir}/03_labeled_dataset.csv", index=False)
        
        # Report
        with open(f"{self.run_dir}/04_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        # Metrics
        with open(f"{self.run_dir}/05_metrics.json", "w") as f:
            json.dump(self._calculate_stats(results), f, indent=2)
    
    def _calculate_stats(self, results: List[Dict]) -> Dict:
        total = len(results)
        verdicts = {"Correct": 0, "Incorrect": 0, "Uncertain": 0}
        for r in results:
            v = r.get("verdict", "Uncertain")
            verdicts[v] = verdicts.get(v, 0) + 1
        
        return {
            "total": total,
            "correct": verdicts["Correct"],
            "incorrect": verdicts["Incorrect"],
            "uncertain": verdicts["Uncertain"],
            "correct_pct": verdicts["Correct"] / total * 100 if total else 0,
            "incorrect_pct": verdicts["Incorrect"] / total * 100 if total else 0,
            "uncertain_pct": verdicts["Uncertain"] / total * 100 if total else 0,
        }


# ============================================================================
# CLI
# ============================================================================

def print_ollama_setup():
    """Print Ollama setup instructions."""
    print("""
╭─────────────────────────────────────────────────────────────╮
│  OLLAMA SETUP (2 minutes, one-time)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Install Ollama (free, local, no API keys):              │
│     https://ollama.ai/download                              │
│                                                             │
│  2. Start Ollama:                                           │
│     ollama serve                                            │
│                                                             │
│  3. Run this demo again - it will auto-download the model   │
│                                                             │
╰─────────────────────────────────────────────────────────────╯
""")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Reviewer Demo - LLM-powered semantic auditing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py                    # Full demo with LLM (recommended)
  python run_demo.py --samples 20       # More samples
  python run_demo.py --seed 42          # Reproducible run
  python run_demo.py --mock             # Quick preview without LLM
        """
    )
    parser.add_argument("--samples", "-n", type=int, default=12, help="Number of samples (default: 12)")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--mock", action="store_true", help="Use mock results (skip LLM)")
    parser.add_argument("--model", type=str, default="mistral", help="Ollama model (default: mistral)")
    # Hidden alias for backwards compatibility
    parser.add_argument("--no-llm", action="store_true", dest="mock", help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    seed = args.seed or int(datetime.now().timestamp())
    
    print("\n🔍 Checking environment...")
    
    use_llm = True
    model_to_use = args.model
    
    if args.mock:
        print("✓ Mock mode (skipping LLM)")
        use_llm = False
    else:
        # Check Ollama status
        ollama_running, available_model, status = check_ollama()
        
        if status == "not_running":
            print("⚠️  Ollama is not running")
            print_ollama_setup()
            
            # Offer to continue in mock mode
            print("Options:")
            print("  1. Install/start Ollama (recommended) - see instructions above")
            print("  2. Run with --mock flag for a quick preview without LLM")
            print()
            sys.exit(1)
            
        elif status == "no_model":
            print(f"✓ Ollama is running")
            print(f"⚠️  No language model found")
            
            # Offer to pull model
            print(f"\nWould you like to download '{args.model}'? (free, ~4GB)")
            print("This is a one-time download.\n")
            
            try:
                response = input("Download now? [Y/n]: ").strip().lower()
                if response in ["", "y", "yes"]:
                    if pull_model(args.model):
                        model_to_use = args.model
                    else:
                        print("\nFailed to download. Run with --mock for preview.")
                        sys.exit(1)
                else:
                    print("\nRun with --mock for a quick preview without LLM.")
                    sys.exit(0)
            except KeyboardInterrupt:
                print("\n\nRun with --mock for a quick preview.")
                sys.exit(0)
                
        else:  # status == "ready"
            model_to_use = available_model
            print(f"✓ Ollama ready with model: {model_to_use}")
    
    # Run demo
    demo = Demo(
        n_samples=args.samples,
        seed=seed,
        use_llm=use_llm,
        model=model_to_use
    )
    
    try:
        demo.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Try running with --mock for a quick preview.")
        sys.exit(1)


if __name__ == "__main__":
    main()


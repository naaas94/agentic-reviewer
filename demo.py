#!/usr/bin/env python3
"""
Demo script for agentic-reviewer system.

This script demonstrates the core functionality without requiring Ollama to be running.
It shows the system architecture and provides example outputs.
"""

import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_loader import DataLoader
from core.sample_selector import SampleSelector
from core.logger import AuditLogger


def demo_data_loading():
    """Demonstrate data loading functionality."""
    print("=" * 60)
    print("DEMO: Data Loading")
    print("=" * 60)
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load data (will create sample data if it doesn't exist)
    df = data_loader.load_data()
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"Label distribution:")
    print(df["pred_label"].value_counts())
    print(f"Confidence range: {df['confidence'].min():.2f} - {df['confidence'].max():.2f}")
    
    # Show a few examples
    print("\nSample data:")
    for _, row in df.head(3).iterrows():
        print(f"  ID: {row['id']}")
        print(f"  Text: {row['text']}")
        print(f"  Label: {row['pred_label']} (confidence: {row['confidence']:.2f})")
        print()


def demo_sample_selection():
    """Demonstrate sample selection strategies."""
    print("=" * 60)
    print("DEMO: Sample Selection")
    print("=" * 60)
    
    # Load data
    data_loader = DataLoader()
    df = data_loader.load_data()
    
    # Test different selection strategies
    strategies = [
        ("low_confidence", {"threshold": 0.8}),
        ("random", {"sample_size": 3}),
        ("all", {})
    ]
    
    for strategy_name, kwargs in strategies:
        print(f"\nStrategy: {strategy_name}")
        selector = SampleSelector(strategy_name, **kwargs)
        selected = selector.select_samples(df, max_samples=5)
        
        print(f"  Selected {len(selected)} samples")
        if len(selected) > 0:
            print(f"  Confidence range: {selected['confidence'].min():.2f} - {selected['confidence'].max():.2f}")
            
            # Show selection statistics
            stats = selector.get_selection_stats(df, selected)
            print(f"  Selection rate: {stats['selection_rate']:.1%}")
            print(f"  Avg confidence (original): {stats['avg_confidence_original']:.2f}")
            print(f"  Avg confidence (selected): {stats['avg_confidence_selected']:.2f}")


def demo_logging():
    """Demonstrate logging functionality."""
    print("=" * 60)
    print("DEMO: Audit Logging")
    print("=" * 60)
    
    # Initialize logger
    logger = AuditLogger("outputs/demo_reviews.sqlite")
    
    # Create some sample review data
    sample_reviews = [
        {
            "id": "demo_001",
            "text": "Delete my data permanently",
            "pred_label": "Access Request",
            "confidence": 0.85
        },
        {
            "id": "demo_002", 
            "text": "What information do you have about me?",
            "pred_label": "Access Request",
            "confidence": 0.92
        },
        {
            "id": "demo_003",
            "text": "I want to be removed from your system",
            "pred_label": "Erasure",
            "confidence": 0.78
        }
    ]
    
    # Log some reviews
    for i, sample in enumerate(sample_reviews):
        verdict = ["Correct", "Incorrect", "Correct"][i]
        suggested_label = "Erasure" if verdict == "Incorrect" else None
        reasoning = f"Demo reasoning for {verdict.lower()} verdict"
        explanation = f"Demo explanation for sample {sample['id']}"
        
        metadata = {
            "model_name": "demo_model",
            "prompt_hash": "demo_hash",
            "tokens_used": 100 + i * 10,
            "latency_ms": 500 + i * 50
        }
        
        success = logger.log_review(
            sample=sample,
            verdict=verdict,
            suggested_label=suggested_label,
            reasoning=reasoning,
            explanation=explanation,
            metadata=metadata
        )
        
        print(f"Logged review {sample['id']}: {verdict} - {'Success' if success else 'Failed'}")
    
    # Log run metadata
    run_metadata = {
        "model_name": "demo_model",
        "prompt_version": "v1.0.0",
        "execution_time": datetime.now().isoformat(),
        "selector_strategy": "demo",
        "tokens_limit": 512,
        "total_samples": 3,
        "reviewed_samples": 3
    }
    
    logger.log_run_metadata("demo_run_001", run_metadata)
    print("Logged run metadata")
    
    # Get statistics
    stats = logger.get_review_stats()
    print(f"\nReview Statistics:")
    print(f"  Total reviews: {stats['total_reviews']}")
    print(f"  Verdict distribution: {stats['verdict_distribution']}")
    print(f"  Average confidence: {stats['avg_confidence']:.2f}")
    print(f"  Recent reviews (24h): {stats['recent_reviews_24h']}")


def demo_label_definitions():
    """Show the label definitions."""
    print("=" * 60)
    print("DEMO: Label Definitions")
    print("=" * 60)
    
    import yaml
    
    # Load label definitions
    with open("configs/labels.yaml", 'r', encoding='utf-8') as f:
        labels_config = yaml.safe_load(f)
    
    print("Available labels for GDPR/CCPA compliance:")
    for label in labels_config["labels"]:
        print(f"\n{label['name']}")
        print(f"  Definition: {label['definition']}")
        print(f"  Examples: {', '.join(label['examples'][:2])}...")


def demo_prompts():
    """Show the prompt templates."""
    print("=" * 60)
    print("DEMO: Prompt Templates")
    print("=" * 60)
    
    prompt_files = [
        ("prompts/evaluator_prompt.txt", "Evaluator"),
        ("prompts/proposer_prompt.txt", "Proposer"),
        ("prompts/reasoner_prompt.txt", "Reasoner")
    ]
    
    for prompt_file, agent_name in prompt_files:
        print(f"\n{agent_name} Prompt Template:")
        print("-" * 40)
        
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Show first 200 characters
                print(content[:200] + "..." if len(content) > 200 else content)
        else:
            print("Prompt file not found")


def main():
    """Run all demos."""
    print("🧠 AGENTIC-REVIEWER DEMO")
    print("=" * 60)
    print("This demo showcases the system architecture and components")
    print("without requiring Ollama to be running.")
    print()
    
    try:
        demo_data_loading()
        demo_sample_selection()
        demo_logging()
        demo_label_definitions()
        demo_prompts()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("To run the full system with LLM agents:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Pull a model: ollama pull mistral")
        print("3. Run: python run.py")
        print("4. Or start API server: python main.py")
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
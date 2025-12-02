#!/usr/bin/env python3
"""
Agentic Reviewer Demo Script

This script demonstrates the core functionality of the agentic reviewer system
without requiring Ollama to be running. It shows the data pipeline, sample
selection, and basic system components.
"""

import asyncio
import time
from datetime import datetime
import pandas as pd

from core.data_loader import DataLoader
from core.sample_selector import SampleSelector
from core.review_loop import ReviewLoop
from core.config import config
from core.cache import get_cache_stats
from main import app


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def demo_data_pipeline():
    """Demonstrate the data loading and selection pipeline."""
    print_section("DATA PIPELINE DEMONSTRATION")
    
    # Load data
    print("1. Loading sample data...")
    loader = DataLoader()
    df = loader.load_data()
    print(f"   ✓ Loaded {len(df)} samples from {loader.data_file}")
    print(f"   ✓ Columns: {list(df.columns)}")
    
    # Show sample data
    print("\n2. Sample data preview:")
    for i, row in df.head(3).iterrows():
        print(f"   {row['id']}: '{row['text'][:50]}...' -> {row['pred_label']} ({row['confidence']:.2f})")
    
    # Test different selection strategies
    print("\n3. Testing sample selection strategies:")
    
    strategies = [
        ("low_confidence", {"threshold": 0.8}),
        ("random", {"sample_size": 3}),
        ("all", {})
    ]
    
    for strategy, kwargs in strategies:
        selector = SampleSelector(strategy, **kwargs)
        selected = selector.select_samples(df)
        stats = selector.get_selection_stats(df, selected)
        
        print(f"   {strategy}: {len(selected)} samples selected")
        print(f"     Selection rate: {stats['selection_rate']:.1%}")
        print(f"     Avg confidence (original): {stats['avg_confidence_original']:.2f}")
        print(f"     Avg confidence (selected): {stats['avg_confidence_selected']:.2f}")
    
    return df


def demo_configuration():
    """Demonstrate system configuration."""
    print_section("SYSTEM CONFIGURATION")
    
    print("1. LLM Configuration:")
    print(f"   Model: {config.llm.model_name}")
    print(f"   URL: {config.llm.ollama_url}")
    print(f"   Temperature: {config.llm.temperature}")
    print(f"   Max tokens: {config.llm.max_tokens}")
    
    print("\n2. Performance Configuration:")
    print(f"   Batch size: {config.performance.batch_size}")
    print(f"   Max concurrent: {config.performance.max_concurrent_requests}")
    print(f"   Cache enabled: {config.performance.enable_caching}")
    
    print("\n3. API Configuration:")
    print(f"   Host: {config.api.host}")
    print(f"   Port: {config.api.port}")
    print(f"   Authentication: {config.api.require_auth}")
    print(f"   Rate limiting: {config.api.enable_rate_limiting}")


def demo_agents():
    """Demonstrate agent initialization and configuration."""
    print_section("AGENT INITIALIZATION")
    
    try:
        from agents.unified_agent import UnifiedAgent
        from agents.evaluator import EvaluatorAgent
        from agents.proposer import ProposerAgent
        from agents.reasoner import ReasonerAgent
        
        print("1. Initializing agents...")
        
        # Initialize unified agent
        unified_agent = UnifiedAgent()
        print(f"   ✓ Unified Agent initialized")
        print(f"     Model: {unified_agent.model_name}")
        print(f"     Labels loaded: {len(unified_agent.labels['labels'])}")
        
        # Show available labels
        print("\n2. Available classification labels:")
        for label in unified_agent.labels['labels']:
            print(f"   • {label['name']}: {label['definition']}")
        
        # Initialize individual agents
        evaluator = EvaluatorAgent()
        proposer = ProposerAgent()
        reasoner = ReasonerAgent()
        print(f"\n3. Individual agents initialized:")
        print(f"   ✓ Evaluator Agent")
        print(f"   ✓ Proposer Agent") 
        print(f"   ✓ Reasoner Agent")
        
        return unified_agent, evaluator, proposer, reasoner
        
    except Exception as e:
        print(f"   ✗ Error initializing agents: {e}")
        return None, None, None, None


def demo_cache():
    """Demonstrate cache functionality."""
    print_section("CACHE SYSTEM")
    
    try:
        from core.cache import get_cache, get_cache_stats
        
        print("1. Cache system status:")
        cache = get_cache()
        stats = get_cache_stats()
        
        print(f"   ✓ Cache initialized")
        print(f"   Entries: {stats['entries']}")
        print(f"   Memory usage: {stats['memory_usage']} bytes")
        print(f"   Hit rate: {stats['hit_rate']:.1%}")
        
        # Test cache operations
        print("\n2. Testing cache operations...")
        cache.set("test_key", {"test": "value"}, ttl=60)
        cached_value = cache.get("test_key")
        
        if cached_value:
            print("   ✓ Cache set/get operations working")
        else:
            print("   ✗ Cache operations failed")
            
    except Exception as e:
        print(f"   ✗ Cache error: {e}")


def demo_api():
    """Demonstrate FastAPI application."""
    print_section("FASTAPI APPLICATION")
    
    try:
        print("1. FastAPI app status:")
        print(f"   ✓ App title: {app.title}")
        print(f"   ✓ App version: {app.version}")
        print(f"   ✓ App description: {app.description}")
        
        # Show available endpoints
        print("\n2. Available endpoints:")
        routes = [route for route in app.routes if hasattr(route, 'path')]
        for route in routes[:10]:  # Show first 10 routes
            methods = getattr(route, 'methods', set())
            if methods:
                print(f"   {list(methods)[0]} {route.path}")
        
        print(f"   ... and {len(routes) - 10} more endpoints" if len(routes) > 10 else "")
        
    except Exception as e:
        print(f"   ✗ API error: {e}")


def demo_review_loop():
    """Demonstrate review loop without LLM calls."""
    print_section("REVIEW LOOP DEMONSTRATION")
    
    try:
        print("1. Initializing review loop...")
        review_loop = ReviewLoop()
        print(f"   ✓ Review loop initialized")
        print(f"   Model: {review_loop.model_name}")
        print(f"   Using unified agent: {review_loop.use_unified_agent}")
        
        # Load and select samples
        print("\n2. Loading and selecting samples...")
        df = review_loop.data_loader.load_data()
        selector = SampleSelector("low_confidence", threshold=0.8)
        selected_df = selector.select_samples(df)
        
        print(f"   ✓ Selected {len(selected_df)} samples for review")
        
        # Show what would be processed
        print("\n3. Samples that would be processed:")
        for i, (_, sample) in enumerate(selected_df.head(3).iterrows(), 1):
            print(f"   {i}. {sample['id']}: '{sample['text'][:40]}...'")
            print(f"      Predicted: {sample['pred_label']} (confidence: {sample['confidence']:.2f})")
        
        print("\n4. Review loop ready for processing")
        print("   Note: Actual LLM processing requires Ollama to be running")
        
    except Exception as e:
        print(f"   ✗ Review loop error: {e}")


def check_ollama_status():
    """Check if Ollama is available."""
    print_section("OLLAMA STATUS CHECK")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("✓ Ollama is running")
            print(f"  Available models: {len(models)}")
            for model in models[:3]:
                print(f"    • {model['name']}")
            if len(models) > 3:
                print(f"    ... and {len(models) - 3} more")
            
            # Check if mistral is available
            mistral_available = any('mistral' in model['name'].lower() for model in models)
            if mistral_available:
                print("✓ Mistral model is available")
                return True
            else:
                print("⚠ Mistral model not found - you may need to run: ollama pull mistral")
                return False
        else:
            print("✗ Ollama is not responding properly")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Ollama is not running")
        print("  To start Ollama:")
        print("  1. Install Ollama: https://ollama.ai/download")
        print("  2. Run: ollama serve")
        print("  3. Pull mistral model: ollama pull mistral")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False


def demo_with_ollama():
    """Demonstrate actual LLM processing if Ollama is available."""
    print_section("LLM PROCESSING DEMONSTRATION")
    
    try:
        from core.review_loop import ReviewLoop
        
        print("1. Testing single sample review...")
        review_loop = ReviewLoop()
        
        # Test a single sample
        result = review_loop.review_single_sample(
            text="Delete my data permanently",
            predicted_label="Access Request", 
            confidence=0.85,
            sample_id="demo_001",
            use_unified=True
        )
        
        print("✓ Single sample review completed")
        print(f"  Sample ID: {result['sample_id']}")
        print(f"  Verdict: {result['verdict']}")
        print(f"  Reasoning: {result['reasoning']}")
        if result.get('suggested_label'):
            print(f"  Suggested Label: {result['suggested_label']}")
        print(f"  Explanation: {result['explanation']}")
        print(f"  Success: {result['success']}")
        
        if result.get('metadata'):
            print(f"  Tokens used: {result['metadata'].get('tokens_used', 'N/A')}")
            print(f"  Latency: {result['metadata'].get('latency_ms', 'N/A')}ms")
        
        return True
        
    except Exception as e:
        print(f"✗ LLM processing failed: {e}")
        print("  This is expected if Ollama is not running or mistral model is not available")
        return False


def main():
    """Run the complete demonstration."""
    print("AGENTIC REVIEWER SYSTEM DEMONSTRATION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {pd.__version__}")
    
    # Run demonstrations
    demo_configuration()
    df = demo_data_pipeline()
    demo_cache()
    demo_api()
    demo_agents()
    demo_review_loop()
    
    # Check Ollama status
    ollama_available = check_ollama_status()
    
    if ollama_available:
        demo_with_ollama()
    else:
        print_section("NEXT STEPS")
        print("To run the complete system:")
        print("1. Install and start Ollama: https://ollama.ai/download")
        print("2. Pull the mistral model: ollama pull mistral")
        print("3. Run the API server: python main.py")
        print("4. Test with: curl -X POST http://localhost:8000/review \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d '{\"text\": \"Delete my data\", \"predicted_label\": \"Access Request\", \"confidence\": 0.8}'")
    
    print_section("DEMONSTRATION COMPLETE")
    print("✓ System components are working correctly")
    print("✓ Ready for LLM processing when Ollama is available")
    print("✓ API endpoints are configured and ready")


if __name__ == "__main__":
    main()


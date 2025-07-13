#!/usr/bin/env python3
"""
Main script to run the agentic-reviewer system.
"""

import argparse
import sys
from core.review_loop import ReviewLoop


def main():
    parser = argparse.ArgumentParser(description="Run agentic review of text classifications")
    parser.add_argument("--model", default="mistral", help="LLM model name (default: mistral)")
    parser.add_argument("--strategy", default="low_confidence", 
                       choices=["low_confidence", "random", "all"],
                       help="Sample selection strategy (default: low_confidence)")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to review")
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Confidence threshold for low_confidence strategy (default: 0.7)")
    parser.add_argument("--sample-size", type=int, default=10,
                       help="Sample size for random strategy (default: 10)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                       help="Ollama server URL (default: http://localhost:11434)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.threshold < 0 or args.threshold > 1:
        print("Error: Confidence threshold must be between 0 and 1")
        sys.exit(1)
    
    if args.sample_size < 1:
        print("Error: Sample size must be at least 1")
        sys.exit(1)
    
    try:
        # Initialize review loop
        review_loop = ReviewLoop(
            model_name=args.model,
            ollama_url=args.ollama_url
        )
        
        # Prepare selector kwargs
        selector_kwargs = {}
        if args.strategy == "low_confidence":
            selector_kwargs["threshold"] = args.threshold
        elif args.strategy == "random":
            selector_kwargs["sample_size"] = args.sample_size
        
        # Run the review
        results = review_loop.run_review(
            selector_strategy=args.strategy,
            max_samples=args.max_samples,
            **selector_kwargs
        )
        
        print(f"\nReview completed successfully!")
        print(f"Results saved to: outputs/reviewed_predictions.sqlite")
        
    except KeyboardInterrupt:
        print("\nReview interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running review: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
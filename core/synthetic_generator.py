"""
Synthetic data generator for portfolio demonstration.

Generates realistic GDPR/CCPA data subject request samples with intentional
misclassifications to demonstrate the reviewer's correction capabilities.
"""

import random
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
import yaml
import os


class SyntheticGenerator:
    """Generates synthetic classification data for demonstration."""
    
    # Extended request templates by category
    TEMPLATES = {
        "Access Request": [
            "What information do you have about me?",
            "Send me a copy of my data.",
            "I want to see what data you have stored about me.",
            "Can you provide me with my personal information?",
            "Show me all records associated with my account.",
            "I'd like to request access to my personal data.",
            "Please provide a copy of all data you hold on me.",
            "Under GDPR, I request access to my data.",
        ],
        "Erasure": [
            "Delete my data permanently.",
            "I want to be removed from your system.",
            "Please erase all my personal information.",
            "I want my data deleted under GDPR.",
            "Remove all my information from your database.",
            "I invoke my right to be forgotten.",
            "Please delete my account and all associated data.",
            "Wipe my data from your systems completely.",
        ],
        "Rectification": [
            "My email address has changed, please update it.",
            "The information you have about me is incorrect.",
            "Please correct my personal details.",
            "Update my address in your records.",
            "My name is spelled wrong in your system.",
            "I need to update my phone number.",
            "Please fix the incorrect data you have about me.",
            "My date of birth is wrong in your records.",
        ],
        "Portability": [
            "I want to export my data to another service.",
            "Give me my data in a machine-readable format.",
            "I want to transfer my data to a competitor.",
            "Export all my personal information.",
            "Provide my data in JSON format for transfer.",
            "I need a portable copy of my information.",
            "Send me my data so I can move to another provider.",
            "I request data portability under GDPR Article 20.",
        ],
        "Objection": [
            "I don't want you to use my data for marketing.",
            "Stop processing my data for profiling.",
            "I object to automated decision making about me.",
            "Don't use my data for research purposes.",
            "I withdraw consent for data processing.",
            "Stop using my information for targeted advertising.",
            "I object to my data being shared with third parties.",
            "Please opt me out of all marketing activities.",
        ],
        "Complaint": [
            "I want to file a complaint about how you handle my data.",
            "This is a formal complaint about data protection.",
            "I'm reporting a data protection violation.",
            "I want to escalate this to the data protection authority.",
            "Your data handling practices are unacceptable.",
            "I'm filing a formal GDPR complaint.",
            "This is a formal notice of data breach.",
            "I demand an investigation into your data practices.",
        ],
        "General Inquiry": [
            "How do you protect my data?",
            "What is your privacy policy?",
            "How long do you keep my information?",
            "What data do you collect about me?",
            "Who has access to my personal data?",
            "What security measures do you have in place?",
            "Can you explain your data retention policy?",
            "What are my rights under GDPR?",
        ],
    }
    
    # Intentional misclassification patterns for demonstration
    CONFUSION_PATTERNS = [
        # (actual_label, confused_as, reason)
        ("Erasure", "Access Request", "Deletion requests confused with data access"),
        ("Portability", "Access Request", "Export requests confused with viewing data"),
        ("Objection", "Erasure", "Opt-out confused with deletion"),
        ("Rectification", "General Inquiry", "Update requests seen as questions"),
        ("Complaint", "Objection", "Formal complaints confused with objections"),
    ]
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed for reproducibility."""
        self.seed = seed or int(datetime.now().timestamp())
        random.seed(self.seed)
        self.labels = list(self.TEMPLATES.keys())
    
    def generate_samples(
        self, 
        n_samples: int = 20,
        misclassification_rate: float = 0.35,
        low_confidence_rate: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic samples with realistic distribution.
        
        Args:
            n_samples: Number of samples to generate
            misclassification_rate: Proportion of intentionally wrong predictions
            low_confidence_rate: Proportion with confidence < 0.7
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        for i in range(n_samples):
            sample = self._generate_single_sample(
                index=i,
                force_misclassification=(random.random() < misclassification_rate),
                force_low_confidence=(random.random() < low_confidence_rate)
            )
            samples.append(sample)
        
        return samples
    
    def _generate_single_sample(
        self, 
        index: int,
        force_misclassification: bool = False,
        force_low_confidence: bool = False
    ) -> Dict[str, Any]:
        """Generate a single synthetic sample."""
        
        # Select actual label and text
        actual_label = random.choice(self.labels)
        text = random.choice(self.TEMPLATES[actual_label])
        
        # Add slight variations to make samples more realistic
        text = self._add_variation(text)
        
        # Determine predicted label
        if force_misclassification:
            pred_label = self._get_confused_label(actual_label)
        else:
            pred_label = actual_label
        
        # Determine confidence
        if force_low_confidence:
            confidence = round(random.uniform(0.45, 0.69), 2)
        elif force_misclassification:
            confidence = round(random.uniform(0.55, 0.82), 2)
        else:
            confidence = round(random.uniform(0.78, 0.98), 2)
        
        # Generate sample ID
        sample_id = f"synth_{self.seed}_{index:04d}"
        
        return {
            "id": sample_id,
            "text": text,
            "pred_label": pred_label,
            "confidence": confidence,
            "ground_truth": actual_label,
            "is_misclassified": pred_label != actual_label,
            "generated_at": datetime.now().isoformat()
        }
    
    def _get_confused_label(self, actual_label: str) -> str:
        """Get a plausibly confused label for demonstration."""
        # Check if there's a known confusion pattern
        for actual, confused, _ in self.CONFUSION_PATTERNS:
            if actual == actual_label:
                return confused
        
        # Otherwise, pick a random different label
        other_labels = [l for l in self.labels if l != actual_label]
        return random.choice(other_labels)
    
    def _add_variation(self, text: str) -> str:
        """Add slight variations to template text."""
        variations = [
            lambda t: t,  # No change
            lambda t: t.lower(),
            lambda t: t + " Thanks.",
            lambda t: "Hi, " + t[0].lower() + t[1:],
            lambda t: t.rstrip(".") + ", please.",
            lambda t: "I need help. " + t,
            lambda t: t + " This is urgent.",
        ]
        return random.choice(variations)(text)
    
    def get_generation_metadata(self) -> Dict[str, Any]:
        """Return metadata about the generation configuration."""
        return {
            "seed": self.seed,
            "labels": self.labels,
            "templates_per_label": {
                label: len(templates) 
                for label, templates in self.TEMPLATES.items()
            },
            "confusion_patterns": [
                {"actual": a, "confused_as": c, "reason": r}
                for a, c, r in self.CONFUSION_PATTERNS
            ],
            "generator_version": "1.0.0"
        }


def generate_portfolio_dataset(
    n_samples: int = 20,
    seed: Optional[int] = None,
    misclassification_rate: float = 0.35
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convenience function for generating portfolio demo dataset.
    
    Returns:
        Tuple of (samples, metadata)
    """
    generator = SyntheticGenerator(seed=seed)
    samples = generator.generate_samples(
        n_samples=n_samples,
        misclassification_rate=misclassification_rate
    )
    metadata = generator.get_generation_metadata()
    
    return samples, metadata


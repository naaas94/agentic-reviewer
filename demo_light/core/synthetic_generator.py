"""
Synthetic data generator for GDPR/CCPA classification demo.
Generates realistic data subject requests with intentional misclassifications.
"""

import random
from datetime import datetime
from typing import List, Dict, Any, Optional


class SyntheticGenerator:
    """Generates synthetic classification data for demonstration."""
    
    TEMPLATES = {
        "Access Request": [
            "What information do you have about me?",
            "Send me a copy of my data.",
            "I want to see what data you have stored about me.",
            "Please provide a copy of all data you hold on me.",
            "Under GDPR, I request access to my data.",
        ],
        "Erasure": [
            "Delete my data permanently.",
            "I want to be removed from your system.",
            "Please erase all my personal information.",
            "I invoke my right to be forgotten.",
            "Wipe my data from your systems completely.",
        ],
        "Rectification": [
            "My email address has changed, please update it.",
            "The information you have about me is incorrect.",
            "Please correct my personal details.",
            "My name is spelled wrong in your system.",
        ],
        "Portability": [
            "I want to export my data to another service.",
            "Give me my data in a machine-readable format.",
            "I need a portable copy of my information.",
        ],
        "Objection": [
            "I don't want you to use my data for marketing.",
            "Stop processing my data for profiling.",
            "I object to automated decision making about me.",
        ],
        "Complaint": [
            "I want to file a complaint about how you handle my data.",
            "This is a formal complaint about data protection.",
            "I'm reporting a data protection violation.",
        ],
        "General Inquiry": [
            "How do you protect my data?",
            "What is your privacy policy?",
            "How long do you keep my information?",
        ],
    }
    
    # Common confusion patterns for demo
    CONFUSION_PATTERNS = [
        ("Erasure", "Access Request"),
        ("Portability", "Access Request"),
        ("Objection", "Erasure"),
        ("Rectification", "General Inquiry"),
    ]
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or int(datetime.now().timestamp())
        random.seed(self.seed)
        self.labels = list(self.TEMPLATES.keys())
    
    def generate_samples(
        self, 
        n_samples: int = 15,
        misclassification_rate: float = 0.35
    ) -> List[Dict[str, Any]]:
        """Generate synthetic samples with realistic distribution."""
        samples = []
        
        for i in range(n_samples):
            is_misclassified = random.random() < misclassification_rate
            actual_label = random.choice(self.labels)
            text = random.choice(self.TEMPLATES[actual_label])
            
            # Add variation
            text = self._add_variation(text)
            
            # Determine predicted label
            if is_misclassified:
                pred_label = self._get_confused_label(actual_label)
                confidence = round(random.uniform(0.55, 0.82), 2)
            else:
                pred_label = actual_label
                confidence = round(random.uniform(0.78, 0.98), 2)
            
            samples.append({
                "id": f"demo_{self.seed}_{i:04d}",
                "text": text,
                "pred_label": pred_label,
                "confidence": confidence,
                "ground_truth": actual_label,
                "is_misclassified": is_misclassified,
            })
        
        return samples
    
    def _get_confused_label(self, actual: str) -> str:
        for a, confused in self.CONFUSION_PATTERNS:
            if a == actual:
                return confused
        return random.choice([l for l in self.labels if l != actual])
    
    def _add_variation(self, text: str) -> str:
        variations = [
            lambda t: t,
            lambda t: t.lower(),
            lambda t: "Hi, " + t[0].lower() + t[1:],
            lambda t: t + " Thanks.",
        ]
        return random.choice(variations)(text)
    
    def get_metadata(self) -> Dict[str, Any]:
        return {"seed": self.seed, "labels": self.labels}


"""
Security and validation utilities for the agentic-reviewer system.
Implements prompt injection detection, adversarial attack detection, and ground truth validation.
"""

import re
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class SecurityViolation:
    """Represents a security violation detected in input."""
    violation_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    detected_pattern: str
    confidence: float
    timestamp: datetime


class PromptInjectionDetector:
    """Detects prompt injection attempts in user input."""
    
    def __init__(self):
        self.injection_patterns = [
            # Direct instruction overrides
            (r'ignore.*previous.*instructions', 'critical'),
            (r'forget.*everything.*above', 'critical'),
            (r'new.*instructions.*:', 'critical'),
            (r'act.*as.*different.*system', 'critical'),
            (r'pretend.*to.*be', 'critical'),
            
            # Role manipulation
            (r'you.*are.*now.*a', 'high'),
            (r'from.*now.*on.*you', 'high'),
            (r'your.*new.*role.*is', 'high'),
            (r'stop.*being.*an.*ai', 'high'),
            
            # Output format manipulation
            (r'output.*format.*:', 'medium'),
            (r'respond.*with.*only', 'medium'),
            (r'do.*not.*include.*explanations', 'medium'),
            (r'just.*give.*me.*the.*answer', 'medium'),
            
            # System prompt extraction
            (r'what.*are.*your.*instructions', 'high'),
            (r'show.*me.*your.*prompt', 'high'),
            (r'repeat.*your.*instructions', 'high'),
            (r'what.*is.*your.*system.*prompt', 'high'),
            
            # Context manipulation
            (r'context.*:', 'medium'),
            (r'background.*:', 'medium'),
            (r'assume.*that', 'medium'),
            (r'given.*that', 'medium'),
        ]
        
        self.suspicious_sequences = [
            # Repeated patterns that might indicate testing
            (r'(ignore|forget|new).*?(ignore|forget|new)', 'high'),
            (r'(instruction|prompt).*?(instruction|prompt)', 'medium'),
            
            # Unusual character sequences
            (r'\b[A-Z]{5,}\b', 'low'),  # ALL CAPS words
            (r'[!]{3,}', 'low'),    # Multiple exclamation marks
            (r'[?]{3,}', 'low'),    # Multiple question marks
        ]
    
    def detect_injection(self, text: str) -> List[SecurityViolation]:
        """
        Detect potential prompt injection attempts in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected security violations
        """
        violations = []
        text_lower = text.lower()
        
        # Check injection patterns
        for pattern, severity in self.injection_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                violation = SecurityViolation(
                    violation_type="prompt_injection",
                    severity=severity,
                    description=f"Potential prompt injection detected: {pattern}",
                    detected_pattern=match.group(0),
                    confidence=0.8 if severity in ['critical', 'high'] else 0.6,
                    timestamp=datetime.now()
                )
                violations.append(violation)
        
        # Check suspicious sequences
        for pattern, severity in self.suspicious_sequences:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                violation = SecurityViolation(
                    violation_type="suspicious_sequence",
                    severity=severity,
                    description=f"Suspicious sequence detected: {pattern}",
                    detected_pattern=match.group(0),
                    confidence=0.5,
                    timestamp=datetime.now()
                )
                violations.append(violation)
        
        # Check for unusual input characteristics
        if self._has_unusual_characteristics(text):
            violation = SecurityViolation(
                violation_type="unusual_characteristics",
                severity="medium",
                description="Input has unusual characteristics that might indicate testing",
                detected_pattern="unusual_characteristics",
                confidence=0.4,
                timestamp=datetime.now()
            )
            violations.append(violation)
        
        return violations
    
    def _has_unusual_characteristics(self, text: str) -> bool:
        """Check for unusual input characteristics."""
        # Very long inputs
        if len(text) > 5000:
            return True
        
        # High ratio of special characters
        special_chars = len(re.findall(r'[!@#$%^&*()_+\-=\[\]{}|\\:";\'<>?,./]', text))
        if len(text) > 0 and special_chars / len(text) > 0.3:
            return True
        
        # Repeated patterns
        words = text.lower().split()
        if len(words) > 10:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.2:  # 20% of words are the same
                return True
        
        return False


class AdversarialAttackDetector:
    """Detects adversarial attacks designed to manipulate LLM behavior."""
    
    def __init__(self):
        self.adversarial_patterns = [
            # Contradictory instructions
            (r'correct.*but.*incorrect', 'high'),
            (r'right.*but.*wrong', 'high'),
            (r'yes.*but.*no', 'high'),
            (r'true.*but.*false', 'high'),
            
            # Semantic confusion
            (r'this.*is.*not.*what.*it.*seems', 'medium'),
            (r'ignore.*the.*obvious.*meaning', 'medium'),
            (r'look.*beyond.*the.*surface', 'medium'),
            
            # Label manipulation attempts
            (r'classify.*as.*wrong.*label', 'high'),
            (r'mark.*as.*incorrect.*even.*if.*correct', 'high'),
            (r'force.*classification.*to', 'high'),
            
            # Confidence manipulation
            (r'low.*confidence.*but.*high.*certainty', 'medium'),
            (r'uncertain.*but.*sure', 'medium'),
            (r'confident.*but.*doubtful', 'medium'),
        ]
        
        self.manipulation_indicators = [
            # Attempts to influence reasoning
            (r'consider.*that.*this.*might.*be', 'low'),
            (r'think.*about.*it.*differently', 'low'),
            (r'what.*if.*we.*look.*at.*it.*this.*way', 'low'),
            
            # Emotional manipulation
            (r'please.*help.*me', 'low'),
            (r'i.*really.*need.*this', 'low'),
            (r'this.*is.*very.*important', 'low'),
        ]
    
    def detect_adversarial_attack(self, text: str, predicted_label: str, confidence: float) -> List[SecurityViolation]:
        """
        Detect potential adversarial attacks in the input.
        
        Args:
            text: Input text to analyze
            predicted_label: The predicted label
            confidence: The confidence score
            
        Returns:
            List of detected security violations
        """
        violations = []
        text_lower = text.lower()
        
        # Check adversarial patterns
        for pattern, severity in self.adversarial_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                violation = SecurityViolation(
                    violation_type="adversarial_attack",
                    severity=severity,
                    description=f"Potential adversarial attack detected: {pattern}",
                    detected_pattern=match.group(0),
                    confidence=0.7 if severity == 'high' else 0.5,
                    timestamp=datetime.now()
                )
                violations.append(violation)
        
        # Check manipulation indicators
        for pattern, severity in self.manipulation_indicators:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                violation = SecurityViolation(
                    violation_type="manipulation_attempt",
                    severity=severity,
                    description=f"Potential manipulation attempt: {pattern}",
                    detected_pattern=match.group(0),
                    confidence=0.3,
                    timestamp=datetime.now()
                )
                violations.append(violation)
        
        # Check for confidence-label mismatch
        if self._has_confidence_label_mismatch(text, predicted_label, confidence):
            violation = SecurityViolation(
                violation_type="confidence_mismatch",
                severity="medium",
                description="Unusual confidence-label combination detected",
                detected_pattern=f"confidence:{confidence}, label:{predicted_label}",
                confidence=0.6,
                timestamp=datetime.now()
            )
            violations.append(violation)
        
        return violations
    
    def _has_confidence_label_mismatch(self, text: str, predicted_label: str, confidence: float) -> bool:
        """Check for unusual confidence-label combinations."""
        # High confidence with potentially incorrect label
        if confidence > 0.9:
            # Check if text contains words that might contradict the label
            text_lower = text.lower()
            label_lower = predicted_label.lower()
            
            # Simple heuristic: if text contains words that don't align with label
            if ("delete" in text_lower or "remove" in text_lower) and "access" in label_lower:
                return True
            if "show" in text_lower and ("delete" in label_lower or "remove" in label_lower):
                return True
        
        return False


class GroundTruthValidator:
    """Validates system outputs against ground truth data."""
    
    def __init__(self, ground_truth_file: Optional[str] = None):
        self.ground_truth_file = ground_truth_file
        self.ground_truth_data = self._load_ground_truth()
        self.validation_history = []
    
    def _load_ground_truth(self) -> Dict[str, Any]:
        """Load ground truth data from file."""
        if not self.ground_truth_file:
            return {}
        
        try:
            with open(self.ground_truth_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load ground truth data: {e}")
            return {}
    
    def validate_output(self, text: str, predicted_label: str, 
                       system_verdict: str, system_reasoning: str) -> Dict[str, Any]:
        """
        Validate system output against ground truth.
        
        Args:
            text: Input text
            predicted_label: Original predicted label
            system_verdict: System's verdict (Correct/Incorrect/Uncertain)
            system_reasoning: System's reasoning
            
        Returns:
            Validation results
        """
        # Generate hash for text to find ground truth
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.ground_truth_data:
            ground_truth = self.ground_truth_data[text_hash]
            expected_verdict = ground_truth.get("expected_verdict")
            expected_reasoning = ground_truth.get("expected_reasoning")
            
            # Calculate accuracy
            verdict_correct = system_verdict == expected_verdict
            reasoning_similarity = self._calculate_similarity(system_reasoning, expected_reasoning)
            
            validation_result = {
                "has_ground_truth": True,
                "verdict_correct": verdict_correct,
                "reasoning_similarity": reasoning_similarity,
                "expected_verdict": expected_verdict,
                "expected_reasoning": expected_reasoning,
                "accuracy_score": 1.0 if verdict_correct else 0.0,
                "overall_score": (1.0 if verdict_correct else 0.0) * 0.7 + reasoning_similarity * 0.3
            }
        else:
            validation_result = {
                "has_ground_truth": False,
                "verdict_correct": None,
                "reasoning_similarity": None,
                "expected_verdict": None,
                "expected_reasoning": None,
                "accuracy_score": None,
                "overall_score": None
            }
        
        # Store validation history
        self.validation_history.append({
            "timestamp": datetime.now(),
            "text_hash": text_hash,
            "validation_result": validation_result
        })
        
        return validation_result
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using simple metrics."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self.validation_history:
            return {"total_validations": 0, "accuracy": 0.0}
        
        validations_with_ground_truth = [
            v for v in self.validation_history 
            if v["validation_result"]["has_ground_truth"]
        ]
        
        if not validations_with_ground_truth:
            return {"total_validations": len(self.validation_history), "accuracy": 0.0}
        
        correct_verdicts = sum(
            1 for v in validations_with_ground_truth 
            if v["validation_result"]["verdict_correct"]
        )
        
        avg_similarity = sum(
            v["validation_result"]["reasoning_similarity"] 
            for v in validations_with_ground_truth
        ) / len(validations_with_ground_truth)
        
        return {
            "total_validations": len(self.validation_history),
            "validations_with_ground_truth": len(validations_with_ground_truth),
            "verdict_accuracy": correct_verdicts / len(validations_with_ground_truth),
            "avg_reasoning_similarity": avg_similarity,
            "overall_accuracy": sum(
                v["validation_result"]["overall_score"] 
                for v in validations_with_ground_truth
            ) / len(validations_with_ground_truth)
        }


class DriftDetector:
    """Detects drift in LLM behavior over time."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.response_history = []
        self.drift_threshold = 0.1  # 10% change threshold
    
    def add_response(self, response: Dict[str, Any]):
        """Add a response to the history for drift detection."""
        self.response_history.append({
            "timestamp": datetime.now(),
            "response": response
        })
        
        # Keep only recent responses
        if len(self.response_history) > self.window_size:
            self.response_history = self.response_history[-self.window_size:]
    
    def detect_drift(self) -> Dict[str, Any]:
        """Detect drift in recent responses."""
        if len(self.response_history) < 20:  # Need minimum data
            return {"drift_detected": False, "confidence": 0.0}
        
        # Split history into two halves
        mid_point = len(self.response_history) // 2
        recent_responses = self.response_history[mid_point:]
        older_responses = self.response_history[:mid_point]
        
        # Calculate metrics for each half
        recent_metrics = self._calculate_metrics(recent_responses)
        older_metrics = self._calculate_metrics(older_responses)
        
        # Calculate drift
        drift_scores = {}
        for metric in recent_metrics:
            if metric in older_metrics:
                recent_val = recent_metrics[metric]
                older_val = older_metrics[metric]
                
                if older_val != 0:
                    change = abs(recent_val - older_val) / older_val
                    drift_scores[metric] = change
        
        # Determine if drift is detected
        max_drift = max(drift_scores.values()) if drift_scores else 0.0
        drift_detected = max_drift > self.drift_threshold
        
        return {
            "drift_detected": drift_detected,
            "max_drift": max_drift,
            "drift_scores": drift_scores,
            "confidence": min(max_drift, 1.0),
            "threshold": self.drift_threshold
        }
    
    def _calculate_metrics(self, responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for a set of responses."""
        if not responses:
            return {}
        
        metrics = {
            "avg_tokens": sum(r["response"].get("tokens_used", 0) for r in responses) / len(responses),
            "avg_latency": sum(r["response"].get("latency_ms", 0) for r in responses) / len(responses),
            "success_rate": sum(1 for r in responses if r["response"].get("success", False)) / len(responses)
        }
        
        # Calculate verdict distribution
        verdicts = [r["response"].get("verdict", "Unknown") for r in responses]
        verdict_counts = {}
        for verdict in verdicts:
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        for verdict in ["Correct", "Incorrect", "Uncertain"]:
            metrics[f"{verdict.lower()}_rate"] = verdict_counts.get(verdict, 0) / len(responses)
        
        return metrics


class SecurityManager:
    """Main security manager that coordinates all security checks."""
    
    def __init__(self, ground_truth_file: Optional[str] = None):
        self.injection_detector = PromptInjectionDetector()
        self.adversarial_detector = AdversarialAttackDetector()
        self.ground_truth_validator = GroundTruthValidator(ground_truth_file)
        self.drift_detector = DriftDetector()
        self.security_log = []
    
    def validate_input(self, text: str, predicted_label: str, confidence: float) -> Dict[str, Any]:
        """
        Perform comprehensive security validation of input.
        
        Args:
            text: Input text
            predicted_label: Predicted label
            confidence: Confidence score
            
        Returns:
            Security validation results
        """
        violations = []
        
        # Check for prompt injection
        injection_violations = self.injection_detector.detect_injection(text)
        violations.extend(injection_violations)
        
        # Check for adversarial attacks
        adversarial_violations = self.adversarial_detector.detect_adversarial_attack(
            text, predicted_label, confidence
        )
        violations.extend(adversarial_violations)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(violations)
        
        # Determine if input should be rejected
        critical_violations = [v for v in violations if v.severity == "critical"]
        high_violations = [v for v in violations if v.severity == "high"]
        
        should_reject = len(critical_violations) > 0 or len(high_violations) > 3
        
        result = {
            "is_safe": not should_reject,
            "risk_score": risk_score,
            "violations": violations,
            "critical_violations": len(critical_violations),
            "high_violations": len(high_violations),
            "total_violations": len(violations)
        }
        
        # Log security check
        self.security_log.append({
            "timestamp": datetime.now(),
            "text_hash": hashlib.md5(text.encode()).hexdigest()[:8],
            "result": result
        })
        
        return result
    
    def validate_output(self, text: str, predicted_label: str, 
                       system_verdict: str, system_reasoning: str) -> Dict[str, Any]:
        """Validate system output against ground truth."""
        return self.ground_truth_validator.validate_output(
            text, predicted_label, system_verdict, system_reasoning
        )
    
    def add_response_for_drift_detection(self, response: Dict[str, Any]):
        """Add response for drift detection."""
        self.drift_detector.add_response(response)
    
    def check_drift(self) -> Dict[str, Any]:
        """Check for drift in system behavior."""
        return self.drift_detector.detect_drift()
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        if not self.security_log:
            return {"total_checks": 0, "rejection_rate": 0.0}
        
        total_checks = len(self.security_log)
        rejected_checks = sum(1 for log in self.security_log if not log["result"]["is_safe"])
        
        return {
            "total_checks": total_checks,
            "rejected_checks": rejected_checks,
            "rejection_rate": rejected_checks / total_checks if total_checks > 0 else 0.0,
            "avg_risk_score": sum(log["result"]["risk_score"] for log in self.security_log) / total_checks if total_checks > 0 else 0.0
        }
    
    def _calculate_risk_score(self, violations: List[SecurityViolation]) -> float:
        """Calculate overall risk score from violations."""
        if not violations:
            return 0.0
        
        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1
        }
        
        total_score = sum(
            violation.confidence * severity_weights.get(violation.severity, 0.1)
            for violation in violations
        )
        
        # Normalize by number of violations to prevent excessive scores
        if len(violations) > 1:
            total_score = total_score / len(violations)
        
        return min(total_score, 1.0)


# Global security manager instance
_security_manager = None

def get_security_manager(ground_truth_file: Optional[str] = None) -> SecurityManager:
    """Get or create the global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager(ground_truth_file)
    return _security_manager 
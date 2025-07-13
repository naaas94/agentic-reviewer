"""
Unit tests for agentic-reviewer agents.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
import os
import sys
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.evaluator import EvaluatorAgent
from agents.proposer import ProposerAgent
from agents.reasoner import ReasonerAgent
from agents.unified_agent import UnifiedAgent
from agents.base_agent import TaskType, AgentTask
from core.data_loader import DataLoader
from core.sample_selector import SampleSelector


class TestEvaluatorAgent:
    """Test cases for the EvaluatorAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = EvaluatorAgent()
    
    def test_evaluator_initialization(self):
        """Test that evaluator initializes correctly."""
        assert self.evaluator.model_name == "mistral"
        assert self.evaluator.ollama_url == "http://localhost:11434"
        assert self.evaluator.labels is not None
        assert "labels" in self.evaluator.labels
    
    @patch('agents.base_agent.BaseAgent._call_ollama')
    def test_evaluator_returns_valid_output(self, mock_call_ollama):
        """Test that evaluator returns valid verdict."""
        # Mock successful LLM response
        mock_call_ollama.return_value = {
            "response": "**Verdict**: Correct\n**Reasoning**: The text clearly matches the label definition.",
            "tokens_used": 50,
            "latency_ms": 100,
            "success": True
        }
        
        result = self.evaluator.evaluate(
            "Delete my data permanently", 
            "Access Request", 
            0.85
        )
        
        assert result["verdict"] in ["Correct", "Incorrect", "Uncertain"]
        assert "reasoning" in result
        assert result["success"] is True
    
    def test_evaluator_fallback_parsing(self):
        """Test fallback parsing when structured parsing fails."""
        evaluator = EvaluatorAgent()
        
        # Test verdict extraction
        response = "This is correct because the text matches the label."
        verdict = evaluator._extract_verdict_fallback(response)
        assert verdict == "Correct"
        
        response = "This is incorrect because the text doesn't match."
        verdict = evaluator._extract_verdict_fallback(response)
        assert verdict == "Incorrect"
        
        response = "This is unclear and could be either."
        verdict = evaluator._extract_verdict_fallback(response)
        assert verdict == "Uncertain"
    
    def test_create_evaluation_task(self):
        """Test creation of evaluation task for multi-task processing."""
        task = self.evaluator.create_evaluation_task(
            "Delete my data", "Access Request", 0.85
        )
        
        assert isinstance(task, AgentTask)
        assert task.task_type == TaskType.EVALUATE
        assert task.context["text"] == "Delete my data"
        assert task.context["predicted_label"] == "Access Request"
        assert task.context["confidence"] == 0.85
        assert "verdict" in task.expected_fields
        assert "reasoning" in task.expected_fields


class TestProposerAgent:
    """Test cases for the ProposerAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.proposer = ProposerAgent()
    
    def test_proposer_initialization(self):
        """Test that proposer initializes correctly."""
        assert self.proposer.model_name == "mistral"
        assert self.proposer.ollama_url == "http://localhost:11434"
        assert self.proposer.labels is not None
    
    def test_proposer_label_extraction(self):
        """Test label extraction from response."""
        proposer = ProposerAgent()
        
        # Test extraction of valid labels
        available_labels = [label["name"] for label in proposer.labels["labels"]]
        
        for label in available_labels:
            response = f"The suggested label is {label} because..."
            extracted = proposer._extract_label_fallback(response)
            assert extracted == label
        
        # Test "no suitable label" case
        response = "No suitable label found for this text."
        extracted = proposer._extract_label_fallback(response)
        assert extracted == "No suitable label found"
    
    def test_proposer_confidence_extraction(self):
        """Test confidence level extraction."""
        proposer = ProposerAgent()
        
        response = "High confidence in this suggestion."
        confidence = proposer._extract_confidence_fallback(response)
        assert confidence == "High"
        
        response = "Low confidence in this suggestion."
        confidence = proposer._extract_confidence_fallback(response)
        assert confidence == "Low"
        
        response = "Medium confidence in this suggestion."
        confidence = proposer._extract_confidence_fallback(response)
        assert confidence == "Medium"
    
    def test_create_proposal_task(self):
        """Test creation of proposal task for multi-task processing."""
        task = self.proposer.create_proposal_task(
            "Delete my data", "Access Request", 0.85
        )
        
        assert isinstance(task, AgentTask)
        assert task.task_type == TaskType.PROPOSE
        assert task.context["text"] == "Delete my data"
        assert task.context["predicted_label"] == "Access Request"
        assert task.context["confidence"] == 0.85
        assert "suggested_label" in task.expected_fields
        assert "confidence" in task.expected_fields


class TestReasonerAgent:
    """Test cases for the ReasonerAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.reasoner = ReasonerAgent()
    
    def test_reasoner_initialization(self):
        """Test that reasoner initializes correctly."""
        assert self.reasoner.model_name == "mistral"
        assert self.reasoner.ollama_url == "http://localhost:11434"
        assert self.reasoner.labels is not None
    
    def test_reasoner_explanation_extraction(self):
        """Test explanation extraction from response."""
        reasoner = ReasonerAgent()
        
        response = "**Explanation**: This label is appropriate because..."
        explanation = reasoner._extract_explanation_fallback(response)
        assert "appropriate" in explanation
        
        # Test fallback to full response
        response = "This is a simple explanation without structure."
        explanation = reasoner._extract_explanation_fallback(response)
        assert explanation == response.strip()
    
    def test_create_reasoning_task(self):
        """Test creation of reasoning task for multi-task processing."""
        task = self.reasoner.create_reasoning_task(
            "Delete my data", "Erasure", "verdict: Incorrect"
        )
        
        assert isinstance(task, AgentTask)
        assert task.task_type == TaskType.REASON
        assert task.context["text"] == "Delete my data"
        assert task.context["label"] == "Erasure"
        assert task.context["context"] == "verdict: Incorrect"
        assert "explanation" in task.expected_fields


class TestUnifiedAgent:
    """Test cases for the UnifiedAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.unified_agent = UnifiedAgent()
    
    def test_unified_agent_initialization(self):
        """Test that unified agent initializes correctly."""
        assert self.unified_agent.model_name == "mistral"
        assert self.unified_agent.ollama_url == "http://localhost:11434"
        assert self.unified_agent.labels is not None
    
    @patch('agents.base_agent.BaseAgent._call_ollama_async')
    def test_process_sample_success(self, mock_call_ollama):
        """Test successful sample processing."""
        # Mock successful LLM response
        mock_call_ollama.return_value = {
            "response": """=== TASK 1: EVALUATE ===
**Verdict**: Incorrect
**Reasoning**: The text is about deletion, not access.

=== TASK 2: PROPOSE ===
**Suggested Label**: Erasure
**Confidence**: High

=== TASK 3: REASON ===
**Explanation**: This text clearly requests data deletion.""",
            "tokens_used": 150,
            "latency_ms": 200,
            "success": True
        }
        
        result = asyncio.run(self.unified_agent.process_sample(
            "Delete my data permanently",
            "Access Request",
            0.85
        ))
        
        assert result["verdict"] == "Incorrect"
        assert "deletion" in result["reasoning"].lower()
        assert result["suggested_label"] == "Erasure"
        assert "deletion" in result["explanation"].lower()
        assert result["success"] is True
    
    @patch('agents.base_agent.BaseAgent._call_ollama_async')
    def test_process_sample_correct_prediction(self, mock_call_ollama):
        """Test processing when prediction is correct."""
        # Mock successful LLM response
        mock_call_ollama.return_value = {
            "response": """=== TASK 1: EVALUATE ===
**Verdict**: Correct
**Reasoning**: The text clearly requests data deletion.

=== TASK 2: PROPOSE ===
**Suggested Label**: Erasure
**Confidence**: High

=== TASK 3: REASON ===
**Explanation**: This text clearly requests data deletion.""",
            "tokens_used": 150,
            "latency_ms": 200,
            "success": True
        }
        
        result = asyncio.run(self.unified_agent.process_sample(
            "Delete my data permanently",
            "Erasure",
            0.85
        ))
        
        assert result["verdict"] == "Correct"
        assert result["suggested_label"] is None  # No suggestion needed for correct prediction
        assert result["success"] is True
    
    def test_process_sample_sync(self):
        """Test synchronous wrapper for process_sample."""
        with patch.object(self.unified_agent, 'process_sample') as mock_process:
            mock_process.return_value = {
                "verdict": "Correct",
                "reasoning": "Test reasoning",
                "suggested_label": None,
                "explanation": "Test explanation",
                "success": True
            }
            
            result = self.unified_agent.process_sample_sync(
                "Test text",
                "Test label",
                0.8
            )
            
            assert result["verdict"] == "Correct"
            assert result["success"] is True
    
    @patch('agents.base_agent.BaseAgent._call_ollama_async')
    def test_process_batch(self, mock_call_ollama):
        """Test batch processing."""
        # Mock successful LLM responses
        mock_call_ollama.return_value = {
            "response": """=== TASK 1: EVALUATE ===
**Verdict**: Correct
**Reasoning**: Test reasoning.

=== TASK 2: PROPOSE ===
**Suggested Label**: Test Label
**Confidence**: High

=== TASK 3: REASON ===
**Explanation**: Test explanation.""",
            "tokens_used": 100,
            "latency_ms": 150,
            "success": True
        }
        
        samples = [
            {"text": "Sample 1", "predicted_label": "Label 1", "confidence": 0.8},
            {"text": "Sample 2", "predicted_label": "Label 2", "confidence": 0.9}
        ]
        
        results = asyncio.run(self.unified_agent.process_batch(samples, max_concurrent=2))
        
        assert len(results) == 2
        assert all(result["success"] for result in results)
        assert all(result["verdict"] == "Correct" for result in results)


class TestDataLoader:
    """Test cases for the DataLoader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_loader = DataLoader("tests/test_data.csv")
    
    def test_data_loader_creates_sample_data(self):
        """Test that data loader creates sample data when file doesn't exist."""
        # Remove test file if it exists
        if os.path.exists("tests/test_data.csv"):
            os.remove("tests/test_data.csv")
        
        df = self.data_loader.load_data()
        
        assert len(df) > 0
        assert "text" in df.columns
        assert "pred_label" in df.columns
        assert "confidence" in df.columns
        assert "id" in df.columns
        
        # Validate confidence scores
        assert all(0.0 <= conf <= 1.0 for conf in df["confidence"])
    
    def test_data_loader_validation(self):
        """Test data validation."""
        # Create invalid data with all required columns but invalid confidence
        invalid_data = pd.DataFrame({
            "text": ["test"],
            "pred_label": ["test_label"],
            "confidence": [1.5]  # Invalid confidence > 1.0
        })
        invalid_data.to_csv("tests/test_invalid.csv", index=False)
        
        invalid_loader = DataLoader("tests/test_invalid.csv")
        
        with pytest.raises(Exception, match="Confidence scores must be between 0.0 and 1.0"):
            invalid_loader.load_data()
        
        # Clean up
        os.remove("tests/test_invalid.csv")


class TestSampleSelector:
    """Test cases for the SampleSelector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame({
            "id": [f"sample_{i}" for i in range(10)],
            "text": [f"text_{i}" for i in range(10)],
            "pred_label": ["Access Request"] * 10,
            "confidence": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        })
    
    def test_low_confidence_selection(self):
        """Test low confidence sample selection."""
        selector = SampleSelector("low_confidence", threshold=0.6)
        selected = selector.select_samples(self.df)
        
        # Should select samples with confidence < 0.6
        assert len(selected) == 5
        assert selected["confidence"].max() < 0.6
    
    def test_random_selection(self):
        """Test random sample selection."""
        selector = SampleSelector("random", sample_size=3)
        selected = selector.select_samples(self.df)
        
        assert len(selected) == 3
        assert len(selected) <= len(self.df)
    
    def test_all_selection(self):
        """Test selection of all samples."""
        selector = SampleSelector("all")
        selected = selector.select_samples(self.df)
        
        assert len(selected) == len(self.df)
    
    def test_max_samples_limit(self):
        """Test that max_samples parameter works correctly."""
        selector = SampleSelector("low_confidence", threshold=0.8)
        selected = selector.select_samples(self.df, max_samples=3)
        
        assert len(selected) == 3
        assert selected["confidence"].max() < 0.8
    
    def test_selection_stats(self):
        """Test selection statistics."""
        selector = SampleSelector("low_confidence", threshold=0.6)
        selected = selector.select_samples(self.df)
        stats = selector.get_selection_stats(self.df, selected)
        
        assert stats["strategy"] == "low_confidence"
        assert stats["total_samples"] == 10
        assert stats["selected_samples"] == 5
        assert stats["selection_rate"] == 0.5
        assert "avg_confidence_original" in stats
        assert "avg_confidence_selected" in stats


class TestAgentTask:
    """Test cases for AgentTask dataclass."""
    
    def test_agent_task_creation(self):
        """Test AgentTask creation and validation."""
        task = AgentTask(
            task_type=TaskType.EVALUATE,
            context={"text": "test", "label": "test_label"},
            expected_fields=["verdict", "reasoning"],
            prompt_template="Test template"
        )
        
        assert task.task_type == TaskType.EVALUATE
        assert task.context["text"] == "test"
        assert task.context["label"] == "test_label"
        assert "verdict" in task.expected_fields
        assert "reasoning" in task.expected_fields
        assert task.prompt_template == "Test template"
    
    def test_task_type_enum(self):
        """Test TaskType enum values."""
        assert TaskType.EVALUATE.value == "evaluate"
        assert TaskType.PROPOSE.value == "propose"
        assert TaskType.REASON.value == "reason"


if __name__ == "__main__":
    pytest.main([__file__]) 
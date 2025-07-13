"""
Agentic Reviewer Agents Package

This package contains the LLM agents for semantic auditing:
- EvaluatorAgent: Evaluates if predicted labels are semantically correct
- ProposerAgent: Suggests alternative labels when predictions are incorrect
- ReasonerAgent: Generates natural language explanations for decisions
- UnifiedAgent: Multi-task agent that processes all tasks in a single LLM call
- BaseAgent: Base class with shared functionality
"""

from .evaluator import EvaluatorAgent
from .proposer import ProposerAgent
from .reasoner import ReasonerAgent
from .base_agent import BaseAgent, TaskType, AgentTask

__all__ = [
    "EvaluatorAgent",
    "ProposerAgent", 
    "ReasonerAgent",
    "BaseAgent",
    "TaskType",
    "AgentTask"
] 
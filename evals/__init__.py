"""
TinyFabulist Evaluation Framework
A modular system for evaluating text generation models on fable completion tasks.
"""

from .base import BaseEvaluator, EvaluationResult, EvaluationConfig
from .perplexity import PerplexityEvaluator
from .text_quality import TextQualityEvaluator
from .fluency import FluencyEvaluator
from .semantic_coherence import SemanticCoherenceEvaluator
from .comprehensive import ComprehensiveEvaluator

__version__ = "1.0.0"

__all__ = [
    "BaseEvaluator",
    "EvaluationResult", 
    "EvaluationConfig",
    "PerplexityEvaluator",
    "TextQualityEvaluator",
    "FluencyEvaluator", 
    "SemanticCoherenceEvaluator",
    "ComprehensiveEvaluator"
]

# Registry of available evaluators
EVALUATOR_REGISTRY = {
    "perplexity": PerplexityEvaluator,
    "text_quality": TextQualityEvaluator,
    "fluency": FluencyEvaluator,
    "semantic_coherence": SemanticCoherenceEvaluator,
    "comprehensive": ComprehensiveEvaluator
}

def get_evaluator(name: str, **kwargs):
    """Factory function to get evaluator by name"""
    if name not in EVALUATOR_REGISTRY:
        available = list(EVALUATOR_REGISTRY.keys())
        raise ValueError(f"Unknown evaluator '{name}'. Available: {available}")
    
    return EVALUATOR_REGISTRY[name](**kwargs)

def list_evaluators():
    """List all available evaluators"""
    return list(EVALUATOR_REGISTRY.keys()) 
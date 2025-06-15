"""
TinyFabulist Evaluation Framework - Refactored
A minimal, elegant evaluation system for text generation models.
"""

from .core import EvaluationRunner, EvaluationConfig, EvaluationResult
from .metrics import PerplexityEvaluator, SemanticEvaluator, FluencyEvaluator, QualityEvaluator

__version__ = "2.0.0"

__all__ = [
    "EvaluationRunner", 
    "EvaluationConfig", 
    "EvaluationResult",
    "PerplexityEvaluator",
    "SemanticEvaluator", 
    "FluencyEvaluator",
    "QualityEvaluator"
]


def create_runner(config: EvaluationConfig = None, model=None, tokenizer=None) -> EvaluationRunner:
    """
    Factory function to create a runner with all available evaluators.
    
    Args:
        config: Evaluation configuration
        model: Language model (required for perplexity)
        tokenizer: Tokenizer (required for perplexity)
        
    Returns:
        Configured EvaluationRunner
    """
    if config is None:
        config = EvaluationConfig()
    
    runner = EvaluationRunner(config)
    
    # Add evaluators based on availability
    if model is not None and tokenizer is not None:
        runner.add_evaluator(PerplexityEvaluator(model, tokenizer, config.device, config.max_length))
    
    runner.add_evaluator(FluencyEvaluator())
    runner.add_evaluator(SemanticEvaluator())
    runner.add_evaluator(QualityEvaluator())
    
    return runner


def list_evaluators() -> list:
    """List all available evaluators"""
    return ["perplexity", "fluency", "semantic", "quality"] 
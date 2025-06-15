"""
Composition-based evaluation runner
"""

import time
import importlib
import pkgutil
from typing import List, Dict, Any
from datasets import load_dataset
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from .config import EvaluationConfig
from .result import EvaluationResult
from .evaluator import Evaluator
from utils.text_prep import prepare_test_data
from utils.generation import generate_completions

# Add lib path for logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.logging_utils import get_logger

logger = get_logger()


class EvaluationRunner:
    """Orchestrates evaluation using composition"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluators = []
    
    def add_evaluator(self, evaluator: Evaluator):
        """Add an evaluator to the runner"""
        self.evaluators.append(evaluator)
    
    def run(self, model, tokenizer, dataset=None) -> EvaluationResult:
        """Run evaluation with all configured evaluators"""
        
        start_time = time.time()
        
        # Load dataset if not provided
        if dataset is None:
            logger.info(f"Loading dataset: {self.config.dataset_name}")
            dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
        
        # Prepare test data with natural prompts
        test_data = prepare_test_data(dataset, self.config.num_samples, self.config.seed)
        
        # Generate predictions once for all evaluators
        predictions = generate_completions(
            model, tokenizer, test_data,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            device=self.config.device
        )
        
        # Extract references for supervised evaluators
        references = [sample['reference'] for sample in test_data]
        
        # Run all evaluators
        all_metrics = {}
        sample_data = []
        
        for evaluator in self.evaluators:
            logger.info(f"Running {evaluator.name} evaluation...")
            
            try:
                if evaluator.requires_references:
                    metrics = evaluator.evaluate(predictions, references)
                else:
                    metrics = evaluator.evaluate(predictions)
                
                # Prefix metrics with evaluator name
                prefixed_metrics = {f"{evaluator.name}_{k}": v for k, v in metrics.items()}
                all_metrics.update(prefixed_metrics)
                
            except Exception as e:
                logger.warning(f"Evaluator {evaluator.name} failed: {e}")
                continue
        
        # Create sample data for first few samples
        for i in range(min(5, len(test_data))):
            sample_data.append({
                "prompt": test_data[i]['prompt'],
                "prediction": predictions[i] if i < len(predictions) else "",
                "reference": references[i] if i < len(references) else ""
            })
        
        execution_time = time.time() - start_time
        
        # Generate quality flags
        quality_flags = self._generate_quality_flags(all_metrics)
        all_metrics.update(quality_flags)
        
        return EvaluationResult(
            metrics=all_metrics,
            samples=sample_data,
            execution_time=execution_time,
            metadata={
                "model_name": self.config.model_name,
                "num_evaluators": len(self.evaluators),
                "evaluator_names": [e.name for e in self.evaluators]
            }
        )
    
    def _generate_quality_flags(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate interpretable quality flags from metrics"""
        
        flags = []
        
        # High repetition flag
        rep_ratio = metrics.get('fluency_repetition_ratio', 0.0)
        if rep_ratio > 0.4:
            flags.append("high_repetition")
        
        # Low coherence flag
        coherence = metrics.get('semantic_coherence', 0.5)
        if coherence < 0.3:
            flags.append("low_coherence")
        
        # Inappropriate content flag
        appropriate_rate = metrics.get('semantic_appropriate_rate', 1.0)
        if appropriate_rate < 0.7:
            flags.append("inappropriate_content")
        
        # Off-topic flag
        fable_relevance = metrics.get('semantic_fable_relevance', 0.5)
        if fable_relevance < 0.3:
            flags.append("off_topic")
        
        # Quality category
        if len(flags) >= 3:
            category = "poor"
        elif len(flags) >= 2:
            category = "needs_improvement"
        elif len(flags) == 1:
            category = "fair"
        elif coherence > 0.7 and rep_ratio < 0.2:
            category = "good"
        else:
            category = "acceptable"
        
        return {
            "quality_flags": flags,
            "num_quality_issues": len(flags),
            "quality_category": category
        } 
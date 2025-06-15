"""
Base classes for the evaluation framework
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import json
import time
from datetime import datetime
import sys
import os
import numpy as np

# Add lib directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib import get_optimal_device, optimize_for_apple_silicon
from lib.logging_utils import get_logger

# Apply Apple Silicon optimizations
optimize_for_apple_silicon()

# Get logger instance
logger = get_logger()


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs"""
    model_name: str = "gpt2"
    dataset_name: str = "klusai/ds-tf1-en-3m"
    dataset_split: str = "Test"
    num_samples: int = 100
    max_length: int = 512
    max_new_tokens: int = 256
    temperature: float = 0.8
    seed: int = 42
    device: str = field(default_factory=lambda: get_optimal_device(verbose=False))
    output_dir: Optional[str] = None
    save_generations: bool = False
    verbose: bool = True


@dataclass 
class EvaluationResult:
    """Container for evaluation results"""
    evaluator_name: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    samples: List[Dict[str, Any]] = field(default_factory=list)
    config: Optional[EvaluationConfig] = None
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "evaluator_name": self.evaluator_name,
            "metrics": self.metrics,
            "samples": self.samples if len(self.samples) <= 10 else self.samples[:10],  # Limit samples for size
            "config": self.config.__dict__ if self.config else None,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    def save_json(self, filepath: str):
        """Save results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        """Get a summary string of key metrics"""
        summary_lines = [f"=== {self.evaluator_name} Results ==="]
        
        for key, value in self.metrics.items():
            if isinstance(value, float):
                summary_lines.append(f"{key}: {value:.4f}")
            elif isinstance(value, (int, str)):
                summary_lines.append(f"{key}: {value}")
        
        summary_lines.append(f"Execution time: {self.execution_time:.2f}s")
        summary_lines.append(f"Samples evaluated: {len(self.samples)}")
        
        return "\n".join(summary_lines)


class BaseEvaluator(ABC):
    """Base class for all evaluators"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.name = self.__class__.__name__.replace("Evaluator", "").lower()
        self.logger = get_logger()
        
    @abstractmethod
    def evaluate(self, 
                model,
                tokenizer, 
                test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Main evaluation method that all evaluators must implement
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer for the model
            test_data: List of test samples, each containing:
                - 'prompt': The input prompt
                - 'reference': The reference completion (optional)
                - 'full_text': The complete fable text
                
        Returns:
            EvaluationResult: Results of the evaluation
        """
        pass
    
    def prepare_test_data(self, dataset, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Prepare test data from dataset
        Default implementation for fable completion tasks
        """
        if num_samples is None:
            num_samples = self.config.num_samples
            
        # Shuffle and sample
        dataset = dataset.shuffle(seed=self.config.seed)
        if len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        test_data = []
        for example in dataset:
            fable = example['fable']
            
            # Split into prompt and reference for completion task
            sentences = fable.split('.')
            if len(sentences) >= 3:
                prompt = '.'.join(sentences[:2]) + '.'
                reference = '.'.join(sentences[2:]).strip()
                
                if reference:  # Ensure we have a reference
                    test_data.append({
                        'prompt': prompt,
                        'reference': reference,
                        'full_text': fable,
                        'original_example': example
                    })
        
        return test_data[:num_samples]
    
    def generate_completion(self, model, tokenizer, prompt: str, max_new_tokens: int = 150) -> str:
        """
        Generate completion for a prompt with Apple Silicon compatibility
        Uses SafeGeneration for robust text generation
        """
        import torch
        
        try:
            # Log generation configuration
            generation_kwargs = {
                'max_new_tokens': max_new_tokens,
                'temperature': self.config.temperature,
                'do_sample': True,
                'pad_token_id': tokenizer.eos_token_id,
                'num_return_sequences': 1
            }
            self.logger.log_generation_config(**generation_kwargs)
            
            # Encode the prompt
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            
            # Use SafeGeneration for robust text generation
            outputs = SafeGeneration.generate_with_fallback(
                model, tokenizer, input_ids, self.config.device, **generation_kwargs
            )
            
            # Decode only the generated part
            completion = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            return completion.strip()
            
        except Exception as e:
            self.logger.log_error_context(e, f"Text generation failed for evaluator {self.name}")
            return "[Generation failed]"
    
    def _log(self, message: str, **kwargs):
        """Log message using the evaluation logger"""
        if self.config.verbose:
            self.logger.info(message, **kwargs)
    
    def _log_sample(self, index: int, prompt: str, completion: str):
        """Helper for logging samples"""
        self.logger.info(f"--- Sample {index+1} ---")
        self.logger.info(f"Prompt: {prompt[:200]}...")
        self.logger.info(f"Completion: {completion[:300]}...")
        self.logger.info("--------------------")
    
    def run(self, model, tokenizer, dataset=None) -> EvaluationResult:
        """
        Convenience method to run full evaluation pipeline
        """
        start_time = time.time()
        
        # Log evaluation start
        config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        self.logger.log_evaluation_start(self.name, config_dict)
        
        # Load dataset if not provided
        if dataset is None:
            from datasets import load_dataset
            self._log(f"Loading dataset {self.config.dataset_name}...")
            dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split)
            self.logger.log_dataset_info(
                self.config.dataset_name, 
                self.config.dataset_split, 
                len(dataset)
            )
        
        # Prepare test data
        self._log("Preparing test data...")
        test_data = self.prepare_test_data(dataset)
        self._log(f"Prepared {len(test_data)} test samples")
        
        # Run evaluation
        self._log("Running evaluation...")
        result = self.evaluate(model, tokenizer, test_data)
        
        # Set execution time and config
        result.execution_time = time.time() - start_time
        result.config = self.config
        
        # Log completion
        self.logger.log_evaluation_result(self.name, result.metrics, result.execution_time)
        
        return result

    def _generate_completions(self, model, tokenizer, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate fable completions for a given list of prompts."""
        
        completions = []
        
        # Use a pipeline for efficient generation if available
        from transformers import pipeline
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=model.device)
        
        self._log(f"Generating {len(test_data)} completions...")
        
        prompts = [item['prompt'] for item in test_data]
        
        # Batch generation for efficiency
        generated_outputs = generator(
            prompts,
            max_new_tokens=self.config.max_new_tokens,
            num_return_sequences=1,
            temperature=self.config.temperature,
            pad_token_id=tokenizer.eos_token_id,
            truncation=True
        )
        
        for i, (item, output) in enumerate(zip(test_data, generated_outputs)):
            # The output from the pipeline is a list containing a dict
            generated_text = output[0]['generated_text']
            
            # Remove the prompt from the beginning of the generated text
            if generated_text.startswith(item['prompt']):
                completion = generated_text[len(item['prompt']):].strip()
            else:
                completion = generated_text.strip()
            
            completed_item = item.copy()
            completed_item['generated_text'] = completion
            completions.append(completed_item)
            
            if i < 5 and self.config.verbose: # Log first few samples
                self._log_sample(i, item['prompt'], completion)

        return completions 
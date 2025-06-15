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
    
    # Standardized prompt length controls
    max_prompt_tokens: int = 256
    min_prompt_tokens: int = 50
    prompt_truncation_strategy: str = "right"  # "left", "right", or "middle"
    
    # Statistical rigor
    num_evaluation_runs: int = 1
    confidence_level: float = 0.95
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    
    # Evaluation standards
    min_samples_per_evaluation: int = 100
    stratified_sampling: bool = True
    prompt_split_ratio: float = 0.6  # 60% for prompt, 40% for reference
    
    # Reporting standards
    report_confidence_intervals: bool = True
    report_effect_sizes: bool = True
    save_raw_results: bool = True
    length_normalize_metrics: bool = True


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
        Prepare test data with standardized prompt length controls
        Uses token-based splitting for consistent evaluation conditions
        """
        if num_samples is None:
            num_samples = self.config.num_samples
            
        # Shuffle and sample
        dataset = dataset.shuffle(seed=self.config.seed)
        if len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        test_data = []
        skipped_samples = 0
        
        for example in dataset:
            fable = example['fable']
            
            # Create standardized prompt using token-based splitting
            prompt_data = self._create_standardized_prompt(fable)
            
            if prompt_data is None:
                skipped_samples += 1
                continue
                
            test_data.append({
                'prompt': prompt_data['prompt'],
                'reference': prompt_data['reference'],
                'full_text': fable,
                'prompt_length_tokens': prompt_data['prompt_length'],
                'reference_length_tokens': prompt_data['reference_length'],
                'split_ratio': prompt_data['split_ratio'],
                'original_example': example
            })
        
        if skipped_samples > 0 and self.config.verbose:
            self._log(f"Skipped {skipped_samples} samples due to length constraints")
        
        return test_data[:num_samples]
    
    def _create_standardized_prompt(self, fable_text: str) -> Optional[Dict[str, Any]]:
        """
        Create standardized prompts with fixed token-based length controls
        
        Args:
            fable_text: Full fable text
            
        Returns:
            Dictionary with prompt data or None if sample should be skipped
        """
        # Import tokenizer here to avoid circular imports
        from transformers import AutoTokenizer
        
        # Use a simple tokenizer for length estimation
        # In practice, this should use the same tokenizer as the model
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except:
            # Fallback to simple word-based tokenization
            tokens = fable_text.split()
            total_tokens = len(tokens)
            
            # Use word-based splitting as fallback
            split_point = int(total_tokens * self.config.prompt_split_ratio)
            
            if split_point < self.config.min_prompt_tokens // 4:  # Rough conversion
                split_point = self.config.min_prompt_tokens // 4
            if total_tokens - split_point < 10:  # Minimum reference
                split_point = total_tokens - 10
                
            if split_point <= 0 or split_point >= total_tokens:
                return None
                
            prompt = ' '.join(tokens[:split_point])
            reference = ' '.join(tokens[split_point:])
            
            return {
                'prompt': prompt,
                'reference': reference,
                'prompt_length': split_point,
                'reference_length': total_tokens - split_point,
                'split_ratio': split_point / total_tokens
            }
        
        # Tokenize the full fable
        tokens = tokenizer.encode(fable_text)
        total_tokens = len(tokens)
        
        # Check minimum length requirements
        min_total_tokens = self.config.min_prompt_tokens + 20  # 20 tokens minimum for reference
        if total_tokens < min_total_tokens:
            return None
        
        # Calculate split point based on configuration
        split_point = int(total_tokens * self.config.prompt_split_ratio)
        
        # Ensure minimum prompt length
        if split_point < self.config.min_prompt_tokens:
            split_point = self.config.min_prompt_tokens
            
        # Ensure minimum reference length
        min_reference_tokens = 20
        if total_tokens - split_point < min_reference_tokens:
            split_point = total_tokens - min_reference_tokens
            
        # Final validation
        if split_point <= 0 or split_point >= total_tokens:
            return None
        
        # Apply prompt length limit
        if split_point > self.config.max_prompt_tokens:
            if self.config.prompt_truncation_strategy == "right":
                split_point = self.config.max_prompt_tokens
            elif self.config.prompt_truncation_strategy == "left":
                # Keep the end of the prompt
                start_point = split_point - self.config.max_prompt_tokens
                prompt_tokens = tokens[start_point:split_point]
                reference_tokens = tokens[split_point:]
            else:  # middle truncation
                # Keep beginning and end, remove middle
                keep_start = self.config.max_prompt_tokens // 2
                keep_end = self.config.max_prompt_tokens - keep_start
                prompt_tokens = tokens[:keep_start] + tokens[split_point-keep_end:split_point]
                reference_tokens = tokens[split_point:]
        
        # Standard case: right truncation or no truncation needed
        if self.config.prompt_truncation_strategy == "right" or split_point <= self.config.max_prompt_tokens:
            prompt_tokens = tokens[:split_point]
            reference_tokens = tokens[split_point:]
        
        # Decode back to text
        prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        reference = tokenizer.decode(reference_tokens, skip_special_tokens=True)
        
        return {
            'prompt': prompt,
            'reference': reference,
            'prompt_length': len(prompt_tokens),
            'reference_length': len(reference_tokens),
            'split_ratio': len(prompt_tokens) / total_tokens
        }
    
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

    def calculate_confidence_interval(self, values: List[float], 
                                    confidence: float = None) -> tuple[float, float]:
        """Calculate confidence interval for metric values"""
        if confidence is None:
            confidence = self.config.confidence_level
            
        try:
            from scipy import stats
            mean = np.mean(values)
            sem = stats.sem(values)  # Standard error of mean
            h = sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
            return mean - h, mean + h
        except ImportError:
            # Fallback to simple standard deviation if scipy not available
            mean = np.mean(values)
            std = np.std(values)
            margin = 1.96 * std / np.sqrt(len(values))  # Approximate 95% CI
            return mean - margin, mean + margin
    
    def calculate_length_normalized_metrics(self, predictions: List[str], 
                                          references: List[str]) -> Dict[str, float]:
        """
        Calculate metrics normalized by length for fair comparison
        """
        if not self.config.length_normalize_metrics:
            return {}
            
        metrics = {}
        
        # Length statistics
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        
        if pred_lengths and ref_lengths:
            metrics.update({
                'avg_prediction_length': np.mean(pred_lengths),
                'avg_reference_length': np.mean(ref_lengths),
                'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0,
                'length_variance_pred': np.var(pred_lengths),
                'length_variance_ref': np.var(ref_lengths),
                'length_std_pred': np.std(pred_lengths),
                'length_std_ref': np.std(ref_lengths)
            })
        
        return metrics
    
    def run_multiple_evaluations(self, model, tokenizer, dataset=None) -> List[EvaluationResult]:
        """
        Run multiple evaluations with different seeds for statistical significance
        """
        if self.config.num_evaluation_runs <= 1:
            return [self.run(model, tokenizer, dataset)]
        
        results = []
        original_seed = self.config.seed
        
        seeds_to_use = self.config.random_seeds[:self.config.num_evaluation_runs]
        if len(seeds_to_use) < self.config.num_evaluation_runs:
            # Generate additional seeds if needed
            additional_seeds = [original_seed + i * 100 for i in range(len(seeds_to_use), self.config.num_evaluation_runs)]
            seeds_to_use.extend(additional_seeds)
        
        for i, seed in enumerate(seeds_to_use):
            if self.config.verbose:
                self._log(f"Running evaluation {i+1}/{self.config.num_evaluation_runs} with seed {seed}")
            
            # Update seed for this run
            self.config.seed = seed
            
            # Run evaluation
            result = self.run(model, tokenizer, dataset)
            result.metadata['run_number'] = i + 1
            result.metadata['seed_used'] = seed
            results.append(result)
        
        # Restore original seed
        self.config.seed = original_seed
        
        return results
    
    def generate_statistical_report(self, results: List[EvaluationResult]) -> str:
        """
        Generate evaluation report with statistical analysis
        """
        if len(results) == 1:
            return results[0].summary()
        
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("STATISTICAL EVALUATION REPORT")
        report.append("=" * 80)
        
        # Methodology section
        report.append("## Methodology")
        report.append(f"- Model: {self.config.model_name}")
        report.append(f"- Dataset: {self.config.dataset_name}")
        report.append(f"- Evaluation samples: {self.config.num_samples}")
        report.append(f"- Max prompt tokens: {self.config.max_prompt_tokens}")
        report.append(f"- Max generation tokens: {self.config.max_new_tokens}")
        report.append(f"- Temperature: {self.config.temperature}")
        report.append(f"- Prompt split ratio: {self.config.prompt_split_ratio}")
        report.append(f"- Number of runs: {len(results)}")
        
        seeds_used = [r.metadata.get('seed_used', 'unknown') for r in results]
        report.append(f"- Seeds used: {seeds_used}")
        
        # Statistical analysis
        report.append("\n## Statistical Analysis")
        
        # Collect all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        
        # Analyze each metric
        for metric_name in sorted(all_metrics):
            values = []
            for result in results:
                if metric_name in result.metrics:
                    val = result.metrics[metric_name]
                    if isinstance(val, (int, float)):
                        values.append(float(val))
            
            if len(values) < 2:
                continue
                
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            report.append(f"\n### {metric_name.upper().replace('_', ' ')}")
            report.append(f"- Mean: {mean_val:.4f}")
            report.append(f"- Std Dev: {std_val:.4f}")
            report.append(f"- Min: {min(values):.4f}")
            report.append(f"- Max: {max(values):.4f}")
            
            if self.config.report_confidence_intervals and len(values) >= 2:
                try:
                    ci_lower, ci_upper = self.calculate_confidence_interval(values)
                    report.append(f"- {self.config.confidence_level*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                except:
                    pass
            
            report.append(f"- Individual runs: {[f'{v:.4f}' for v in values]}")
        
        # Execution time analysis
        exec_times = [r.execution_time for r in results]
        if exec_times:
            report.append(f"\n## Execution Time Analysis")
            report.append(f"- Mean execution time: {np.mean(exec_times):.2f}s")
            report.append(f"- Std execution time: {np.std(exec_times):.2f}s")
            report.append(f"- Total time: {sum(exec_times):.2f}s")
        
        report.append("=" * 80)
        
        return "\n".join(report) 
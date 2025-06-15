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
    """Configuration for evaluation runs with improved methodological standards"""
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
    
    # Improved prompt length controls (removes arbitrary ratio splitting)
    max_prompt_tokens: int = 256
    min_prompt_tokens: int = 50
    # Removed prompt_split_ratio - now uses principled sentence boundary detection
    
    # Statistical rigor
    num_evaluation_runs: int = 1
    confidence_level: float = 0.95
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    
    # Evaluation standards
    min_samples_per_evaluation: int = 100
    stratified_sampling: bool = True
    
    # Reporting standards
    report_confidence_intervals: bool = True
    report_effect_sizes: bool = True
    save_raw_results: bool = True
    length_normalize_metrics: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.max_prompt_tokens <= self.min_prompt_tokens:
            raise ValueError("max_prompt_tokens must be greater than min_prompt_tokens")
        if self.confidence_level <= 0 or self.confidence_level >= 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if self.min_samples_per_evaluation < 10:
            raise ValueError("min_samples_per_evaluation should be at least 10 for meaningful statistics")


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
        Create prompts using improved methodology that respects narrative structure.
        
        Instead of arbitrary percentage splits, attempts to find natural breaking points
        in the text while maintaining consistent prompt lengths for fair evaluation.
        
        Args:
            fable_text: Full fable text
            
        Returns:
            Dictionary with prompt data or None if sample should be skipped
        """
        # Import tokenizer here to avoid circular imports
        from transformers import AutoTokenizer
        
        # Use a simple tokenizer for length estimation
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except:
            # Fallback to simple word-based tokenization
            return self._fallback_word_splitting(fable_text)
        
        # Tokenize the full fable
        tokens = tokenizer.encode(fable_text)
        total_tokens = len(tokens)
        
        # Check minimum length requirements
        min_total_tokens = self.config.min_prompt_tokens + 30  # 30 tokens minimum for meaningful reference
        if total_tokens < min_total_tokens:
            return None
        
        # Try to find natural breaking points in the text
        # This is more principled than arbitrary percentage splits
        sentences = self._split_into_sentences(fable_text)
        if len(sentences) < 2:
            # Fallback to token-based splitting if sentence splitting fails
            return self._token_based_splitting(tokenizer, tokens, total_tokens)
        
        # Find optimal split point based on target prompt length
        target_prompt_tokens = min(self.config.max_prompt_tokens, 
                                 max(self.config.min_prompt_tokens, total_tokens // 2))
        
        best_split = self._find_optimal_sentence_split(
            sentences, tokenizer, target_prompt_tokens, total_tokens
        )
        
        if best_split is None:
            # Fallback to token-based splitting
            return self._token_based_splitting(tokenizer, tokens, total_tokens)
        
        prompt, reference, prompt_tokens, reference_tokens = best_split
        
        return {
            'prompt': prompt,
            'reference': reference,
            'prompt_length': prompt_tokens,
            'reference_length': reference_tokens,
            'split_ratio': prompt_tokens / total_tokens,
            'split_method': 'sentence_boundary'
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics"""
        import re
        
        # Simple sentence splitting - can be improved with NLTK if available
        sentences = re.split(r'[.!?]+\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Handle case where last sentence doesn't end with punctuation
        if not text.strip().endswith(('.', '!', '?')):
            # The last split might have removed the final part
            remaining = text.strip()
            for sent in sentences[:-1]:
                remaining = remaining.replace(sent, '', 1)
            if remaining.strip():
                sentences[-1] = remaining.strip()
        
        return sentences
    
    def _find_optimal_sentence_split(self, sentences: List[str], tokenizer, 
                                   target_prompt_tokens: int, total_tokens: int) -> Optional[tuple]:
        """Find the best sentence boundary to split at"""
        
        best_split = None
        best_score = float('inf')
        
        # Try different split points
        for i in range(1, len(sentences)):
            prompt_text = ' '.join(sentences[:i])
            reference_text = ' '.join(sentences[i:])
            
            # Skip if reference is too short
            if len(reference_text.split()) < 10:
                continue
            
            prompt_tokens = len(tokenizer.encode(prompt_text))
            reference_tokens = len(tokenizer.encode(reference_text))
            
            # Skip if prompt is too short or too long
            if prompt_tokens < self.config.min_prompt_tokens:
                continue
            if prompt_tokens > self.config.max_prompt_tokens:
                break  # Later splits will be even longer
            
            # Score based on how close we are to target length
            # Prefer splits that are closer to target but not over max
            score = abs(prompt_tokens - target_prompt_tokens)
            
            if score < best_score:
                best_score = score
                best_split = (prompt_text, reference_text, prompt_tokens, reference_tokens)
        
        return best_split
    
    def _token_based_splitting(self, tokenizer, tokens: List[int], total_tokens: int) -> Optional[Dict[str, Any]]:
        """Fallback to token-based splitting when sentence splitting fails"""
        
        # Use a more principled approach than arbitrary ratios
        # Aim for prompts that are substantial but leave room for meaningful completion
        target_prompt_tokens = min(
            self.config.max_prompt_tokens,
            max(self.config.min_prompt_tokens, total_tokens * 2 // 3)  # Up to 2/3 of text
        )
        
        # Ensure minimum reference length
        min_reference_tokens = 30
        if total_tokens - target_prompt_tokens < min_reference_tokens:
            target_prompt_tokens = total_tokens - min_reference_tokens
            
        # Final validation
        if target_prompt_tokens <= 0 or target_prompt_tokens >= total_tokens:
            return None
        
        prompt_tokens = tokens[:target_prompt_tokens]
        reference_tokens = tokens[target_prompt_tokens:]
        
        # Decode back to text
        prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        reference = tokenizer.decode(reference_tokens, skip_special_tokens=True)
        
        return {
            'prompt': prompt,
            'reference': reference,
            'prompt_length': len(prompt_tokens),
            'reference_length': len(reference_tokens),
            'split_ratio': len(prompt_tokens) / total_tokens,
            'split_method': 'token_boundary'
        }
    
    def _fallback_word_splitting(self, fable_text: str) -> Optional[Dict[str, Any]]:
        """Fallback word-based splitting when tokenizer is not available"""
        words = fable_text.split()
        total_words = len(words)
        
        if total_words < 20:  # Too short
            return None
        
        # Use similar logic as token-based splitting
        target_prompt_words = min(
            self.config.max_prompt_tokens // 2,  # Rough conversion
            max(self.config.min_prompt_tokens // 2, total_words * 2 // 3)
        )
        
        min_reference_words = 10
        if total_words - target_prompt_words < min_reference_words:
            target_prompt_words = total_words - min_reference_words
            
        if target_prompt_words <= 0 or target_prompt_words >= total_words:
            return None
            
        prompt = ' '.join(words[:target_prompt_words])
        reference = ' '.join(words[target_prompt_words:])
        
        return {
            'prompt': prompt,
            'reference': reference,
            'prompt_length': target_prompt_words,
            'reference_length': total_words - target_prompt_words,
            'split_ratio': target_prompt_words / total_words,
            'split_method': 'word_boundary'
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
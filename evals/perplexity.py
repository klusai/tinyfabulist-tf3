"""
Perplexity evaluator for language modeling quality assessment
"""

import torch
import math
from typing import Dict, List, Any
from .base import BaseEvaluator, EvaluationResult, MetricCalculator


class PerplexityEvaluator(BaseEvaluator):
    """Evaluator for perplexity metrics"""
    
    def __init__(self, config=None):
        super().__init__(config)
    
    def calculate_perplexity(self, model, tokenizer, text: str) -> Dict[str, float]:
        """Calculate perplexity for a single text"""
        # Tokenize text
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_length
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get model outputs
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            
            # Calculate perplexity
            perplexity = math.exp(loss)
            
            # Additional metrics
            num_tokens = inputs["input_ids"].shape[1]
            bits_per_char = loss * math.log2(math.e) * num_tokens / len(text)
            
        return {
            "loss": loss,
            "perplexity": perplexity,
            "num_tokens": num_tokens,
            "bits_per_char": bits_per_char
        }
    
    def evaluate(self, model, tokenizer, test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """Evaluate perplexity on test data"""
        model.eval()
        
        perplexities = []
        losses = []
        bits_per_char_list = []
        total_tokens = 0
        total_loss_weighted = 0
        
        sample_results = []
        
        for i, sample in enumerate(test_data):
            if i % 50 == 0:
                self._log(f"Processing sample {i+1}/{len(test_data)}")
            
            # Calculate perplexity on full fable
            full_text = sample['full_text']
            
            try:
                metrics = self.calculate_perplexity(model, tokenizer, full_text)
                
                perplexities.append(metrics["perplexity"])
                losses.append(metrics["loss"])
                bits_per_char_list.append(metrics["bits_per_char"])
                
                # For weighted average
                num_tokens = metrics["num_tokens"]
                total_tokens += num_tokens
                total_loss_weighted += metrics["loss"] * num_tokens
                
                sample_result = {
                    "prompt": sample['prompt'][:100] + "..." if len(sample['prompt']) > 100 else sample['prompt'],
                    "perplexity": metrics["perplexity"],
                    "loss": metrics["loss"],
                    "num_tokens": num_tokens
                }
                
                if self.config.save_generations:
                    sample_result.update({
                        "full_text": full_text,
                        "reference": sample.get('reference', '')
                    })
                
                sample_results.append(sample_result)
                
            except Exception as e:
                self._log(f"Error processing sample {i}: {e}")
                continue
        
        # Calculate aggregate metrics
        metrics = {
            "average_perplexity": MetricCalculator.mean(perplexities),
            "median_perplexity": MetricCalculator.percentile(perplexities, 50),
            "weighted_perplexity": math.exp(total_loss_weighted / total_tokens) if total_tokens > 0 else float('inf'),
            "perplexity_std": MetricCalculator.std(perplexities),
            "average_loss": MetricCalculator.mean(losses),
            "average_bits_per_char": MetricCalculator.mean(bits_per_char_list),
            "min_perplexity": min(perplexities) if perplexities else float('inf'),
            "max_perplexity": max(perplexities) if perplexities else 0,
            "total_tokens": total_tokens,
            "num_samples": len(perplexities)
        }
        
        return EvaluationResult(
            evaluator_name="Perplexity",
            metrics=metrics,
            samples=sample_results,
            metadata={
                "model_name": self.config.model_name,
                "max_length": self.config.max_length,
                "device": self.config.device
            }
        ) 
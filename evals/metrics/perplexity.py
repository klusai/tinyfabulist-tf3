"""
Pure perplexity evaluator
"""

import torch
import math
import numpy as np
from typing import Dict, List, Optional
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.evaluator import Evaluator


class PerplexityEvaluator(Evaluator):
    """Pure perplexity metric calculation"""
    
    def __init__(self, model, tokenizer, device: str = "cpu", max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
    
    @property
    def name(self) -> str:
        return "perplexity"
    
    @property
    def requires_references(self) -> bool:
        return False  # Perplexity is unsupervised
    
    def evaluate(self, predictions: List[str], references: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate perplexity metrics for predictions"""
        
        perplexities = []
        losses = []
        bits_per_char_list = []
        
        self.model.eval()
        
        for text in predictions:
            if not text.strip():
                continue
                
            try:
                metrics = self._calculate_perplexity(text)
                perplexities.append(metrics["perplexity"])
                losses.append(metrics["loss"])
                bits_per_char_list.append(metrics["bits_per_char"])
            except Exception:
                continue  # Skip problematic samples
        
        if not perplexities:
            return {"mean": float('inf'), "std": 0.0, "bits_per_char": 0.0}
        
        return {
            "mean": np.mean(perplexities),
            "std": np.std(perplexities),
            "bits_per_char": np.mean(bits_per_char_list)
        }
    
    def _calculate_perplexity(self, text: str) -> Dict[str, float]:
        """Calculate perplexity for a single text"""
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            
            # Calculate metrics
            perplexity = math.exp(loss)
            num_tokens = inputs["input_ids"].shape[1]
            bits_per_char = loss * math.log2(math.e) * num_tokens / len(text)
            
        return {
            "loss": loss,
            "perplexity": perplexity,
            "bits_per_char": bits_per_char
        } 
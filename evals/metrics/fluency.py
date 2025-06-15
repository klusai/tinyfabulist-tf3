"""
Pure fluency evaluator
"""

import re
import numpy as np
from typing import Dict, List, Optional
from collections import Counter
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.evaluator import Evaluator


class FluencyEvaluator(Evaluator):
    """Pure fluency metric calculation"""
    
    @property
    def name(self) -> str:
        return "fluency"
    
    @property
    def requires_references(self) -> bool:
        return False  # Fluency is unsupervised
    
    def evaluate(self, predictions: List[str], references: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate fluency metrics for predictions"""
        
        repetition_ratios = []
        ttrs = []
        coherence_scores = []
        
        for text in predictions:
            if not text.strip():
                continue
            
            # Calculate metrics
            rep_ratio = self._calculate_repetition_ratio(text)
            ttr = self._calculate_ttr(text)
            coherence = self._calculate_coherence(text)
            
            repetition_ratios.append(rep_ratio)
            ttrs.append(ttr)
            coherence_scores.append(coherence)
        
        if not repetition_ratios:
            return {"repetition_ratio": 0.0, "ttr": 0.0, "coherence": 0.0}
        
        avg_rep = np.mean(repetition_ratios)
        avg_ttr = np.mean(ttrs)
        avg_coherence = np.mean(coherence_scores)
        
        return {
            "repetition_ratio": avg_rep,
            "ttr": avg_ttr,
            "coherence": avg_coherence,
            "high_repetition_flag": avg_rep > 0.4
        }
    
    def _calculate_repetition_ratio(self, text: str) -> float:
        """Calculate repetition ratio"""
        tokens = self._tokenize(text)
        if len(tokens) == 0:
            return 0.0
        
        word_counts = Counter(tokens)
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        
        return repeated_words / len(tokens)
    
    def _calculate_ttr(self, text: str) -> float:
        """Calculate Type-Token Ratio"""
        tokens = self._tokenize(text)
        if len(tokens) == 0:
            return 0.0
        
        unique_tokens = len(set(tokens))
        return unique_tokens / len(tokens)
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate basic coherence score"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) == 0:
            return 0.0
        
        # Check for reasonable sentence lengths
        reasonable_sentences = 0
        for sentence in sentences:
            words = sentence.split()
            if 3 <= len(words) <= 30:  # Reasonable length
                reasonable_sentences += 1
        
        return reasonable_sentences / len(sentences)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for analysis"""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split() 
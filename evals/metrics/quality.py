"""
Pure quality evaluator (BERTScore + text statistics)
"""

import re
import numpy as np
from typing import Dict, List, Optional
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.evaluator import Evaluator


class QualityEvaluator(Evaluator):
    """Pure quality metric calculation using BERTScore and text statistics"""
    
    def __init__(self):
        self._bertscore = None
        self._load_bertscore()
    
    @property
    def name(self) -> str:
        return "quality"
    
    @property
    def requires_references(self) -> bool:
        return True  # BERTScore requires references
    
    def evaluate(self, predictions: List[str], references: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate quality metrics for predictions vs references"""
        
        if not references:
            return {"bert_f1": 0.0, "bert_precision": 0.0, "bert_recall": 0.0}
        
        # Filter out empty predictions/references
        valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                      if p.strip() and r.strip()]
        
        if not valid_pairs:
            return {"bert_f1": 0.0, "bert_precision": 0.0, "bert_recall": 0.0}
        
        valid_predictions, valid_references = zip(*valid_pairs)
        
        # Calculate BERTScore
        bert_metrics = self._calculate_bertscore(list(valid_predictions), list(valid_references))
        
        # Calculate text statistics
        text_stats = self._calculate_text_stats(list(valid_predictions), list(valid_references))
        
        # Combine metrics
        metrics = {**bert_metrics, **text_stats}
        return metrics
    
    def _load_bertscore(self):
        """Load BERTScore with graceful fallback"""
        try:
            import evaluate
            self._bertscore = evaluate.load("bertscore")
        except ImportError:
            try:
                from bert_score import BERTScorer
                self._bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            except ImportError:
                pass  # Will return zeros if BERTScore unavailable
    
    def _calculate_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore metrics"""
        
        if self._bertscore is None:
            return {"bert_f1": 0.0, "bert_precision": 0.0, "bert_recall": 0.0}
        
        try:
            results = self._bertscore.compute(
                predictions=predictions,
                references=references,
                lang="en"
            )
            
            return {
                "bert_f1": np.mean(results['f1']),
                "bert_precision": np.mean(results['precision']),
                "bert_recall": np.mean(results['recall'])
            }
        except Exception:
            return {"bert_f1": 0.0, "bert_precision": 0.0, "bert_recall": 0.0}
    
    def _calculate_text_stats(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate text statistics"""
        
        # Length statistics
        pred_lengths = [len(text.split()) for text in predictions]
        ref_lengths = [len(text.split()) for text in references]
        
        avg_pred_length = np.mean(pred_lengths)
        avg_ref_length = np.mean(ref_lengths)
        length_ratio = avg_pred_length / avg_ref_length if avg_ref_length > 0 else 0.0
        
        # Vocabulary diversity
        all_words = []
        for text in predictions:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        vocab_diversity = unique_words / total_words if total_words > 0 else 0.0
        
        return {
            "avg_length": avg_pred_length,
            "length_ratio": length_ratio,
            "vocab_diversity": vocab_diversity
        } 
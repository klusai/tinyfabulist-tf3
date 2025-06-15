"""
Text Quality Evaluator

Evaluates the quality of generated text using semantic similarity metrics.
Focuses on interpretable metrics without arbitrary combinations.

Removed BLEU/ROUGE scores as they are inappropriate for creative generation tasks.
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional
from collections import Counter

from .base import BaseEvaluator, EvaluationResult, EvaluationConfig


class TextQualityEvaluator(BaseEvaluator):
    """Evaluates text quality using semantic similarity and basic statistics"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        super().__init__(config)
        self._bert_scorer = None
        self._bertscore = None
        self._load_metrics()
        
    def _load_metrics(self):
        """Load evaluation metrics with graceful fallbacks"""
        self._log("Loading text quality metrics...")
        
        # Load BERTScore from evaluate library
        try:
            import evaluate
            self._bertscore = evaluate.load("bertscore")
            self._log("✓ BERTScore loaded from evaluate library.")
        except ImportError:
            self._log("Warning: evaluate library not available, will try bert_score directly")
            try:
                from bert_score import BERTScorer
                self._bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
                self._log("✓ BERTScore loaded directly.")
            except ImportError:
                self._log("Warning: bert-score not available, skipping BERTScore metrics")
        
        self._log("✓ Metrics loaded.")
    
    def calculate_bert_scores(self, references: List[str], generated: List[str]) -> Dict[str, float]:
        """Calculate BERTScore for semantic similarity"""
        if not references or not generated:
            return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}
        
        # Try evaluate library first
        if self._bertscore is not None:
            try:
                results = self._bertscore.compute(
                    predictions=generated, 
                    references=references, 
                    lang="en"
                )
                return {
                    "bert_precision": np.mean(results['precision']),
                    "bert_recall": np.mean(results['recall']),
                    "bert_f1": np.mean(results['f1'])
                }
            except Exception as e:
                self._log(f"BERTScore (evaluate) calculation failed: {e}")
        
        # Fallback to direct bert_score
        if self._bert_scorer is not None:
            try:
                P, R, F1 = self._bert_scorer.score(generated, references)
                return {
                    "bert_precision": P.mean().item(),
                    "bert_recall": R.mean().item(),
                    "bert_f1": F1.mean().item()
                }
            except Exception as e:
                self._log(f"BERTScore calculation failed: {e}")
        
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}
    
    def calculate_text_statistics(self, texts: List[str]) -> Dict[str, float]:
        """Calculate basic text statistics"""
        if not texts:
            return {}
        
        lengths = [len(text.split()) for text in texts]
        char_lengths = [len(text) for text in texts]
        
        # Calculate vocabulary diversity
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        vocab_diversity = unique_words / total_words if total_words > 0 else 0.0
        
        return {
            "avg_word_length": np.mean(lengths),
            "std_word_length": np.std(lengths),
            "avg_char_length": np.mean(char_lengths),
            "vocab_diversity": vocab_diversity,
            "total_unique_words": unique_words,
            "total_words": total_words
        }
    
    def evaluate(self, model, tokenizer, test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluate text quality using semantic similarity metrics.
        Expects test_data to contain 'reference' and 'generated_text'.
        """
        
        sample_results = []
        
        # Prepare lists for batch processing
        references = []
        candidates = []

        for i, item in enumerate(test_data):
            reference = item.get('reference', '')
            completion = item.get('generated_text', '')

            if not completion:
                self._log(f"Skipping sample {i} due to empty completion.")
                continue

            if not reference:
                self._log(f"Skipping sample {i} due to empty reference.")
                continue

            references.append(reference)
            candidates.append(completion)

            sample_results.append({
                "reference": reference,
                "completion": completion,
                "completion_length": len(completion.split()),
                "reference_length": len(reference.split())
            })

        if not candidates:
            self._log("No valid completions found to evaluate.", level="warning")
            return EvaluationResult("TextQuality", {}, [])

        # --- Semantic Similarity Calculation ---
        self._log(f"Calculating semantic similarity for {len(candidates)} samples...")

        # BERTScore for semantic similarity
        bert_results = self.calculate_bert_scores(references, candidates)
        
        # Text statistics
        text_stats = self.calculate_text_statistics(candidates)
        reference_stats = self.calculate_text_statistics(references)
        
        # --- Per-sample metrics ---
        if self._bertscore is not None:
            try:
                detailed_bert = self._bertscore.compute(
                    predictions=candidates, 
                    references=references, 
                    lang="en"
                )
                for i in range(len(sample_results)):
                    if i < len(detailed_bert['f1']):
                        sample_results[i]['bert_f1'] = detailed_bert['f1'][i]
                        sample_results[i]['bert_precision'] = detailed_bert['precision'][i]
                        sample_results[i]['bert_recall'] = detailed_bert['recall'][i]
            except Exception as e:
                self._log(f"Per-sample BERTScore calculation failed: {e}")

        # Aggregate metrics
        metrics = {
            # Semantic similarity (primary metric)
            'bert_f1': bert_results['bert_f1'],
            'bert_precision': bert_results['bert_precision'],
            'bert_recall': bert_results['bert_recall'],
            
            # Text statistics
            'avg_completion_length': text_stats.get('avg_word_length', 0.0),
            'completion_length_std': text_stats.get('std_word_length', 0.0),
            'vocab_diversity': text_stats.get('vocab_diversity', 0.0),
            
            # Reference comparison
            'avg_reference_length': reference_stats.get('avg_word_length', 0.0),
            'length_ratio': (text_stats.get('avg_word_length', 0.0) / 
                           reference_stats.get('avg_word_length', 1.0)) if reference_stats.get('avg_word_length', 0.0) > 0 else 0.0
        }

        return EvaluationResult(
            evaluator_name="text_quality",
            metrics=metrics,
            samples=sample_results
        ) 
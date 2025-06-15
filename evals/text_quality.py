"""
Text quality evaluator with BLEU, ROUGE, and semantic similarity metrics
"""

import re
from typing import Dict, List, Any, Optional
from collections import Counter
from .base import BaseEvaluator, EvaluationResult, EvaluationConfig
import numpy as np
import evaluate


class TextQualityEvaluator(BaseEvaluator):
    """Evaluator for text quality metrics (BLEU, ROUGE, BERTScore)"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        super().__init__(config)
        self._nltk_downloaded = False
        self._rouge_scorer = None
        self._bert_scorer = None
        self.bleu = None
        self.rouge = None
        self.bertscore = None
        
        self._log("Initializing TextQualityEvaluator and loading metrics...")
        # Load metrics using the evaluate library
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")
        self._log("✓ Metrics loaded.")
        
    def _ensure_nltk(self):
        """Ensure NLTK dependencies are downloaded"""
        if not self._nltk_downloaded:
            try:
                import nltk
                # Download both punkt_tab (NLTK 3.9+) and punkt (legacy support)
                nltk.download('punkt_tab', quiet=True)
                nltk.download('punkt', quiet=True)
                self._nltk_downloaded = True
            except ImportError:
                self._log("Warning: NLTK not available, using basic tokenization")
            except Exception as e:
                self._log(f"Warning: NLTK data download failed: {e}, using basic tokenization")
    
    def _ensure_rouge(self):
        """Ensure ROUGE scorer is initialized"""
        if self._rouge_scorer is None:
            try:
                from rouge_score import rouge_scorer
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'], 
                    use_stemmer=True
                )
            except ImportError:
                self._log("Warning: rouge-score not available, skipping ROUGE metrics")
    
    def _ensure_bert_scorer(self):
        """Ensure BERTScore is initialized"""
        if self._bert_scorer is None:
            try:
                from bert_score import BERTScorer
                self._bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            except ImportError:
                self._log("Warning: bert-score not available, skipping BERTScore metrics")
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using NLTK or fallback method"""
        self._ensure_nltk()
        
        try:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text.lower())
        except (ImportError, LookupError) as e:
            # Fallback tokenization for import errors or missing NLTK data
            if isinstance(e, LookupError):
                self._log(f"NLTK data missing ({e}), using fallback tokenization")
            return re.findall(r'\b\w+\b', text.lower())
    
    def calculate_bleu_score(self, reference: str, generated: str) -> Dict[str, float]:
        """Calculate BLEU scores"""
        ref_tokens = self.tokenize_text(reference)
        gen_tokens = self.tokenize_text(generated)
        
        if not ref_tokens or not gen_tokens:
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
        
        # Calculate n-gram matches
        def calculate_ngram_bleu(n: int) -> float:
            ref_ngrams = Counter([' '.join(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
            gen_ngrams = Counter([' '.join(gen_tokens[i:i+n]) for i in range(len(gen_tokens)-n+1)])
            
            if not gen_ngrams:
                return 0.0
            
            matches = sum((ref_ngrams & gen_ngrams).values())
            total = sum(gen_ngrams.values())
            
            return matches / total if total > 0 else 0.0
        
        # Calculate brevity penalty
        ref_len = len(ref_tokens)
        gen_len = len(gen_tokens)
        brevity_penalty = min(1.0, gen_len / ref_len) if ref_len > 0 else 0.0
        
        bleu_scores = {}
        for n in range(1, 5):
            if len(gen_tokens) >= n:
                bleu_scores[f"bleu_{n}"] = calculate_ngram_bleu(n) * brevity_penalty
            else:
                bleu_scores[f"bleu_{n}"] = 0.0
        
        return bleu_scores
    
    def calculate_rouge_scores(self, reference: str, generated: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        self._ensure_rouge()
        
        if self._rouge_scorer is None:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        try:
            scores = self._rouge_scorer.score(reference, generated)
            return {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure
            }
        except Exception as e:
            self._log(f"ROUGE calculation failed: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def calculate_bert_scores(self, references: List[str], generated: List[str]) -> Dict[str, float]:
        """Calculate BERTScore for a batch"""
        self._ensure_bert_scorer()
        
        if self._bert_scorer is None or not references or not generated:
            return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}
        
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
    
    def evaluate(self, model, tokenizer, test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluate text quality metrics (BLEU, ROUGE, BERTScore).
        Expects test_data to contain 'fable' (reference) and 'generated_text' (candidate).
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

            # For quality metrics, the "reference" is the ground truth completion,
            # and the "candidate" is the generated completion.
            references.append(reference)
            candidates.append(completion)

            sample_results.append({
                "reference": reference,
                "completion": completion,
                "completion_length": len(completion.split())
            })

        if not candidates:
            self._log("No valid completions found to evaluate.", level="warning")
            return EvaluationResult("TextQuality", {}, [])

        # --- Batch Metric Calculation ---
        self._log(f"Calculating metrics for {len(candidates)} samples...")

        # ROUGE scores
        rouge_scores = self.rouge.compute(predictions=candidates, references=references)
        
        # BLEU scores (use string format for evaluate library)
        bleu_score = self.bleu.compute(predictions=candidates, references=[[ref] for ref in references])
        
        # BERTScore
        bert_results = self.bertscore.compute(predictions=candidates, references=references, lang="en")
        
        # --- Aggregation ---
        avg_rouge_l = np.mean(rouge_scores['rougeL'])
        avg_bleu = bleu_score['bleu']
        avg_bert_f1 = np.mean(bert_results['f1'])
        
        for i in range(len(sample_results)):
            if i < len(bert_results['f1']):
                # BLEU score is a single value for the entire batch
                sample_results[i]['bleu_4'] = bleu_score['bleu']
                # ROUGE scores - check if they're lists or single values
                if isinstance(rouge_scores['rougeL'], list):
                    sample_results[i]['rougeL'] = rouge_scores['rougeL'][i] if i < len(rouge_scores['rougeL']) else 0.0
                else:
                    sample_results[i]['rougeL'] = rouge_scores['rougeL']
                sample_results[i]['bert_f1'] = bert_results['f1'][i]

        metrics = {
            'avg_bleu_4': avg_bleu,
            'avg_rougeL': avg_rouge_l,
            'bert_f1': avg_bert_f1,
            'avg_completion_length': np.mean([s['completion_length'] for s in sample_results])
        }

        return EvaluationResult(
            evaluator_name="text_quality",
            metrics=metrics,
            samples=sample_results
        ) 
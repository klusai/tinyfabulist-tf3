"""
Text quality evaluator with BLEU, ROUGE, and semantic similarity metrics
"""

import re
from typing import Dict, List, Any, Optional
from collections import Counter
from .base import BaseEvaluator, EvaluationResult, MetricCalculator


class TextQualityEvaluator(BaseEvaluator):
    """Evaluator for text quality metrics (BLEU, ROUGE, BERTScore)"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self._nltk_downloaded = False
        self._rouge_scorer = None
        self._bert_scorer = None
        
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
        """Evaluate text quality metrics"""
        model.eval()
        
        # Generate completions
        self._log("Generating completions...")
        generated_completions = []
        references = []
        sample_results = []
        
        for i, sample in enumerate(test_data):
            if i % 20 == 0:
                self._log(f"Generating completion {i+1}/{len(test_data)}")
            
            prompt = sample['prompt']
            reference = sample['reference']
            
            # Generate completion
            completion = self.generate_completion(model, tokenizer, prompt)
            full_generated = prompt + " " + completion
            
            generated_completions.append(completion)
            references.append(reference)
            
            # Calculate per-sample metrics
            bleu_scores = self.calculate_bleu_score(reference, completion)
            rouge_scores = self.calculate_rouge_scores(reference, completion)
            
            sample_result = {
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "generated_completion": completion[:200] + "..." if len(completion) > 200 else completion,
                "reference": reference[:200] + "..." if len(reference) > 200 else reference,
                **bleu_scores,
                **rouge_scores
            }
            
            if self.config.save_generations:
                sample_result.update({
                    "full_generated": full_generated,
                    "full_reference": sample['full_text']
                })
            
            sample_results.append(sample_result)
        
        # Calculate aggregate BLEU and ROUGE scores
        self._log("Calculating aggregate scores...")
        
        all_bleu_scores = {f"bleu_{n}": [] for n in range(1, 5)}
        all_rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        
        for sample in sample_results:
            for n in range(1, 5):
                all_bleu_scores[f"bleu_{n}"].append(sample[f"bleu_{n}"])
            
            all_rouge_scores["rouge1"].append(sample["rouge1"])
            all_rouge_scores["rouge2"].append(sample["rouge2"])
            all_rouge_scores["rougeL"].append(sample["rougeL"])
        
        # Calculate BERTScore (batch operation)
        bert_scores = self.calculate_bert_scores(references, generated_completions)
        
        # Aggregate metrics
        metrics = {}
        
        # BLEU metrics
        for n in range(1, 5):
            scores = all_bleu_scores[f"bleu_{n}"]
            metrics[f"avg_bleu_{n}"] = MetricCalculator.mean(scores)
            metrics[f"std_bleu_{n}"] = MetricCalculator.std(scores)
        
        # ROUGE metrics
        for rouge_type in ["rouge1", "rouge2", "rougeL"]:
            scores = all_rouge_scores[rouge_type]
            metrics[f"avg_{rouge_type}"] = MetricCalculator.mean(scores)
            metrics[f"std_{rouge_type}"] = MetricCalculator.std(scores)
        
        # BERTScore metrics
        metrics.update(bert_scores)
        
        # Additional quality metrics
        completion_lengths = [len(comp.split()) for comp in generated_completions]
        reference_lengths = [len(ref.split()) for ref in references]
        
        metrics.update({
            "avg_completion_length": MetricCalculator.mean(completion_lengths),
            "avg_reference_length": MetricCalculator.mean(reference_lengths),
            "length_ratio": MetricCalculator.safe_divide(
                MetricCalculator.mean(completion_lengths),
                MetricCalculator.mean(reference_lengths),
                1.0
            ),
            "num_samples": len(test_data)
        })
        
        return EvaluationResult(
            evaluator_name="TextQuality",
            metrics=metrics,
            samples=sample_results,
            metadata={
                "model_name": self.config.model_name,
                "temperature": self.config.temperature,
                "max_new_tokens": 150  # Default from base class
            }
        ) 
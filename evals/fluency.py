"""
Fluency evaluator for repetition, diversity, and coherence metrics
"""

import re
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

from .base import BaseEvaluator, EvaluationResult, EvaluationConfig


class FluencyEvaluator(BaseEvaluator):
    """Evaluator for text fluency and quality"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        super().__init__(config)
        
        # Try to initialize LanguageTool for grammar checking
        self.language_tool = None
        try:
            import language_tool_python
            self.language_tool = language_tool_python.LanguageTool('en-US')
            self._log("✓ LanguageTool initialized for grammar checking.")
        except Exception as e:
            self._log(f"⚠️  LanguageTool not available (Java required): {e}")
            self._log("Grammar checking will be disabled. Install Java to enable grammar analysis.")
    
    def tokenize_for_analysis(self, text: str) -> List[str]:
        """Tokenize text for fluency analysis"""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def calculate_repetition_metrics(self, text: str) -> Dict[str, float]:
        """Calculate various repetition metrics"""
        tokens = self.tokenize_for_analysis(text)
        
        if len(tokens) == 0:
            return {
                "repetition_ratio": 0.0,
                "duplicate_word_ratio": 0.0,
                "unique_words": 0,
                "total_words": 0
            }
        
        # Word frequency analysis
        word_counts = Counter(tokens)
        
        # Basic repetition metrics
        unique_words = len(word_counts)
        total_words = len(tokens)
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        
        # Duplicate sequences (consecutive word repetitions)
        duplicate_sequences = 0
        for i in range(len(tokens) - 1):
            if tokens[i] == tokens[i + 1]:
                duplicate_sequences += 1
        
        return {
            "repetition_ratio": repeated_words / total_words if total_words > 0 else 0.0,
            "duplicate_word_ratio": duplicate_sequences / (total_words - 1) if total_words > 1 else 0.0,
            "unique_words": unique_words,
            "total_words": total_words
        }
    
    def calculate_diversity_metrics(self, text: str) -> Dict[str, float]:
        """Calculate vocabulary diversity metrics"""
        tokens = self.tokenize_for_analysis(text)
        
        if len(tokens) == 0:
            return {
                "ttr": 0.0,  # Type-Token Ratio
                "msttr": 0.0,  # Mean Segmental TTR
                "bigram_diversity": 0.0,
                "trigram_diversity": 0.0
            }
        
        # Type-Token Ratio (TTR)
        unique_tokens = len(set(tokens))
        ttr = unique_tokens / len(tokens)
        
        # Mean Segmental TTR (MSTTR) - calculate TTR for segments of 50 words
        segment_size = 50
        ttrs = []
        for i in range(0, len(tokens), segment_size):
            segment = tokens[i:i + segment_size]
            if len(segment) >= 10:  # Only consider segments with at least 10 words
                segment_ttr = len(set(segment)) / len(segment)
                ttrs.append(segment_ttr)
        
        msttr = sum(ttrs) / len(ttrs) if ttrs else ttr
        
        # N-gram diversity
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        trigrams = [f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}" for i in range(len(tokens) - 2)]
        
        bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
        trigram_diversity = len(set(trigrams)) / len(trigrams) if trigrams else 0.0
        
        return {
            "ttr": ttr,
            "msttr": msttr,
            "bigram_diversity": bigram_diversity,
            "trigram_diversity": trigram_diversity
        }
    
    def calculate_coherence_metrics(self, text: str) -> Dict[str, float]:
        """Calculate basic coherence metrics"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) == 0:
            return {
                "avg_sentence_length": 0.0,
                "sentence_count": 0,
                "sentence_length_std": 0.0,
                "coherence_score": 0.0
            }
        
        # Sentence length analysis
        sentence_lengths = [len(sent.split()) for sent in sentences if sent.strip()]
        
        if not sentence_lengths:
            return {"avg_sentence_length": 0, "sentence_length_std": 0}
        
        avg_sentence_length = np.mean(sentence_lengths)
        sentence_length_std = np.std(sentence_lengths)
        
        # Basic coherence heuristics
        coherence_score = 0.0
        
        # Reasonable sentence length (5-25 words is good)
        reasonable_lengths = sum(1 for length in sentence_lengths if 5 <= length <= 25)
        length_score = reasonable_lengths / len(sentence_lengths)
        
        # Sentence structure variety (different starting words)
        sentence_starts = []
        for sentence in sentences:
            words = self.tokenize_for_analysis(sentence)
            if words:
                sentence_starts.append(words[0])
        
        start_diversity = len(set(sentence_starts)) / len(sentence_starts) if sentence_starts else 0.0
        
        # Combined coherence score
        coherence_score = (length_score + start_diversity) / 2
        
        return {
            "avg_sentence_length": avg_sentence_length,
            "sentence_count": len(sentences),
            "sentence_length_std": sentence_length_std,
            "coherence_score": coherence_score
        }
    
    def calculate_lexical_sophistication(self, text: str) -> Dict[str, float]:
        """Calculate lexical sophistication metrics"""
        words = self.tokenize_for_analysis(text)
        
        if not words:
            return {"avg_word_length": 0, "word_length_std": 0}
        
        word_lengths = [len(word) for word in words]
        
        avg_word_length = np.mean(word_lengths)
        word_length_std = np.std(word_lengths)
        
        return {
            "avg_word_length": avg_word_length,
            "word_length_std": word_length_std
        }
    
    def evaluate(self, model, tokenizer, test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluate fluency metrics for generated text.
        Expects test_data to contain 'generated_text'.
        """
        
        sample_results = []
        
        for i, item in enumerate(test_data):
            completion = item.get('generated_text', '')
            
            if not completion:
                self._log(f"Skipping sample {i} due to empty completion.")
                continue

            # 1. Linguistic Acceptability (Grammar) - optional if LanguageTool available
            grammar_score = self.calculate_grammar_score(completion)
            
            # 2. Perplexity (using a pre-trained model for general fluency)
            # This is a simplified approach. A dedicated LM would be better.
            # We are not using one to avoid heavy dependencies for this specific metric.
            
            # 3. Repetition
            repetition_ratio = self.calculate_repetition_metrics(completion)['repetition_ratio']
            
            # 4. Type-Token Ratio (Lexical Diversity)
            ttr = self.calculate_diversity_metrics(completion)['ttr']
            
            # 5. Coherence (simplified)
            coherence_score = self.calculate_coherence_metrics(completion)['coherence_score']
            
            sample_results.append({
                'grammar_score': grammar_score,
                'repetition_ratio': repetition_ratio,
                'ttr': ttr,
                'coherence_score': coherence_score
            })
            
        if not sample_results:
            self._log("No valid completions found to evaluate.", level="warning")
            return EvaluationResult("fluency", {}, [])

        # Aggregate metrics
        metrics = self.aggregate_metrics(sample_results)
        
        return EvaluationResult(
            evaluator_name="fluency",
            metrics=metrics,
            samples=sample_results
        )

    def aggregate_metrics(self, sample_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate per-sample fluency metrics into summary statistics."""
        
        # Calculate averages for each metric
        avg_grammar = np.mean([s['grammar_score'] for s in sample_results])
        avg_repetition = np.mean([s['repetition_ratio'] for s in sample_results])
        avg_ttr = np.mean([s['ttr'] for s in sample_results])
        avg_coherence = np.mean([s['coherence_score'] for s in sample_results])
        
        # --- Calculate Overall Fluency Score ---
        # Weights can be adjusted based on importance
        # Repetition is a penalty, so it's subtracted
        overall_score = (
            (avg_grammar * 0.4) +
            (avg_coherence * 0.4) +
            (avg_ttr * 0.2) -
            (avg_repetition * 0.5)  # Penalize repetition more heavily
        )
        # Clamp score between 0 and 1
        overall_score = np.clip(overall_score, 0, 1)
        
        return {
            'overall_fluency_score': overall_score,
            'avg_grammar_score': avg_grammar,
            'avg_repetition_ratio': avg_repetition,
            'avg_ttr': avg_ttr,
            'avg_coherence_score': avg_coherence
        }
        
    def calculate_grammar_score(self, text: str) -> float:
        """Calculate grammar score (1 - normalized error rate)."""
        if not text:
            return 0.0
        
        # If LanguageTool is not available, return a neutral score
        if self.language_tool is None:
            return 0.5  # Neutral score when grammar checking is unavailable
        
        try:
            matches = self.language_tool.check(text)
            num_errors = len(matches)
            num_words = len(word_tokenize(text))
            
            # Normalize error rate and invert to get a score
            error_rate = num_errors / num_words if num_words > 0 else 0
            return max(0.0, 1.0 - error_rate * 2) # Penalize errors more harshly
        except Exception as e:
            self._log(f"Grammar checking failed: {e}", level="warning")
            return 0.5  # Neutral score on error

    def calculate_repetition(self, text: str) -> float:
        """Calculate repetition ratio (1 - unique_words / total_words)."""
        if not text:
            return 0.0
        
        tokens = word_tokenize(text.lower())
        if not tokens:
            return 0.0
        
        fdist = FreqDist(tokens)
        
        # Higher ratio means more repetition
        repetition_ratio = 1.0 - (len(fdist) / len(tokens))
        return repetition_ratio

    def calculate_ttr(self, text: str) -> float:
        """Calculate Type-Token Ratio (TTR) for lexical diversity."""
        if not text:
            return 0.0
            
        tokens = word_tokenize(text.lower())
        if not tokens:
            return 0.0
            
        num_unique_tokens = len(set(tokens))
        return num_unique_tokens / len(tokens)

    def calculate_coherence(self, text: str) -> float:
        """Calculate a simple coherence score based on sentence structure."""
        if not text:
            return 0.0
        
        # Simple heuristic: sentences should have reasonable length and structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return 0.0
        
        # Check for reasonable sentence lengths (not too short or too long)
        reasonable_sentences = 0
        for sentence in sentences:
            words = sentence.split()
            if 3 <= len(words) <= 30:  # Reasonable sentence length
                reasonable_sentences += 1
        
        return reasonable_sentences / len(sentences) if sentences else 0.0 
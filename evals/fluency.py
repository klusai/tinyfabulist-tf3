"""
Fluency evaluator for repetition, diversity, and coherence metrics
"""

import re
from typing import Dict, List, Any
from collections import Counter, defaultdict
from .base import BaseEvaluator, EvaluationResult, MetricCalculator


class FluencyEvaluator(BaseEvaluator):
    """Evaluator for fluency and diversity metrics"""
    
    def __init__(self, config=None):
        super().__init__(config)
    
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
        sentence_lengths = [len(self.tokenize_for_analysis(sent)) for sent in sentences]
        avg_sentence_length = MetricCalculator.mean(sentence_lengths)
        sentence_length_std = MetricCalculator.std(sentence_lengths)
        
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
        tokens = self.tokenize_for_analysis(text)
        
        if len(tokens) == 0:
            return {
                "avg_word_length": 0.0,
                "long_word_ratio": 0.0,
                "word_length_std": 0.0
            }
        
        # Word length analysis
        word_lengths = [len(word) for word in tokens]
        avg_word_length = MetricCalculator.mean(word_lengths)
        word_length_std = MetricCalculator.std(word_lengths)
        
        # Long words (6+ characters) ratio
        long_words = sum(1 for word in tokens if len(word) >= 6)
        long_word_ratio = long_words / len(tokens)
        
        return {
            "avg_word_length": avg_word_length,
            "long_word_ratio": long_word_ratio,
            "word_length_std": word_length_std
        }
    
    def evaluate(self, model, tokenizer, test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """Evaluate fluency metrics"""
        model.eval()
        
        # Generate completions
        self._log("Generating completions for fluency analysis...")
        generated_texts = []
        sample_results = []
        
        for i, sample in enumerate(test_data):
            # Log progress every 10 samples
            if i % 10 == 0:
                self.logger.log_evaluation_progress("fluency", i, len(test_data))
            
            prompt = sample['prompt']
            reference = sample.get('reference', '')
            
            # Generate completion
            completion = self.generate_completion(model, tokenizer, prompt)
            full_text = prompt + " " + completion
            generated_texts.append(full_text)
            
            # Calculate metrics for this sample
            repetition_metrics = self.calculate_repetition_metrics(full_text)
            diversity_metrics = self.calculate_diversity_metrics(full_text)
            coherence_metrics = self.calculate_coherence_metrics(full_text)
            lexical_metrics = self.calculate_lexical_sophistication(full_text)
            
            # Combine metrics for logging
            sample_metrics = {
                **repetition_metrics,
                **diversity_metrics,
                **coherence_metrics,
                **lexical_metrics
            }
            
            # Log generation sample (show first few samples in detail)
            if i < 5:  # Show detailed logs for first 5 samples
                key_metrics = {
                    'repetition': repetition_metrics.get('repetition_ratio', 0),
                    'diversity': diversity_metrics.get('ttr', 0),
                    'coherence': coherence_metrics.get('coherence_score', 0)
                }
                self.logger.log_generation_sample(
                    sample_idx=i,
                    prompt=prompt,
                    generated=completion,
                    reference=reference,
                    metrics=key_metrics
                )
            
            sample_result = {
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "generated_text": full_text[:300] + "..." if len(full_text) > 300 else full_text,
                **sample_metrics
            }
            
            if self.config.save_generations:
                sample_result["full_generated_text"] = full_text
            
            sample_results.append(sample_result)
        
        # Calculate aggregate metrics
        self._log("Calculating aggregate fluency metrics...")
        
        all_metrics = defaultdict(list)
        for sample in sample_results:
            for key, value in sample.items():
                if isinstance(value, (int, float)) and key not in ["prompt", "generated_text", "full_generated_text"]:
                    all_metrics[key].append(value)
        
        # Aggregate statistics
        metrics = {}
        for metric_name, values in all_metrics.items():
            metrics[f"avg_{metric_name}"] = MetricCalculator.mean(values)
            metrics[f"std_{metric_name}"] = MetricCalculator.std(values)
            metrics[f"median_{metric_name}"] = MetricCalculator.percentile(values, 50)
        
        # Overall fluency score (composite metric)
        fluency_components = [
            1 - MetricCalculator.mean(all_metrics["repetition_ratio"]),  # Lower repetition is better
            MetricCalculator.mean(all_metrics["ttr"]),  # Higher diversity is better
            MetricCalculator.mean(all_metrics["coherence_score"]),  # Higher coherence is better
            min(1.0, MetricCalculator.mean(all_metrics["avg_sentence_length"]) / 15)  # Reasonable length
        ]
        
        overall_fluency = MetricCalculator.mean(fluency_components)
        metrics["overall_fluency_score"] = overall_fluency
        
        # Text length statistics
        text_lengths = [len(text.split()) for text in generated_texts]
        metrics.update({
            "avg_text_length": MetricCalculator.mean(text_lengths),
            "std_text_length": MetricCalculator.std(text_lengths),
            "min_text_length": min(text_lengths) if text_lengths else 0,
            "max_text_length": max(text_lengths) if text_lengths else 0,
            "num_samples": len(test_data)
        })
        
        return EvaluationResult(
            evaluator_name="Fluency",
            metrics=metrics,
            samples=sample_results,
            metadata={
                "model_name": self.config.model_name,
                "temperature": self.config.temperature,
                "analysis_type": "lexical_diversity_and_repetition"
            }
        ) 
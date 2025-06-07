"""
Semantic Coherence Evaluator
Evaluates whether generated text is semantically coherent, on-topic, and appropriate
"""

import re
import statistics
from typing import Dict, List, Any
from collections import Counter
from .base import BaseEvaluator, EvaluationResult, MetricCalculator


class SemanticCoherenceEvaluator(BaseEvaluator):
    """Evaluator for semantic coherence and content appropriateness"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Initialize word lists for analysis
        self.animal_words = {
            'fox', 'rabbit', 'turtle', 'hare', 'lion', 'mouse', 'ant', 'grasshopper', 
            'crow', 'dove', 'wolf', 'deer', 'bear', 'eagle', 'snake', 'frog', 'owl',
            'cat', 'dog', 'pig', 'sheep', 'goat', 'horse', 'cow', 'chicken', 'duck'
        }
        
        self.moral_keywords = {
            'moral', 'lesson', 'learned', 'teach', 'wisdom', 'truth', 'patience',
            'kindness', 'honesty', 'pride', 'humble', 'together', 'teamwork',
            'perseverance', 'determination', 'careful', 'steady', 'slow'
        }
        
        self.inappropriate_content = {
            'violence': {'kill', 'death', 'murder', 'blood', 'wound', 'hurt', 'pain', 'suffering'},
            'medical': {'surgery', 'doctor', 'hospital', 'patient', 'disease', 'medicine'},
            'adult_themes': {'money', 'politics', 'war', 'weapon', 'fight', 'battle'}
        }
        
        self.narrative_markers = {
            'beginning': {'once upon a time', 'in a', 'there lived', 'there was'},
            'conflict': {'problem', 'trouble', 'difficulty', 'challenge', 'contest', 'competition'},
            'resolution': {'solved', 'resolved', 'learned', 'realized', 'understood', 'discovered'}
        }
    
    def evaluate_topic_consistency(self, prompt: str, generated: str) -> float:
        """Evaluate if generated text stays on topic with the prompt"""
        
        # Extract key topics from prompt
        prompt_animals = self._extract_animals(prompt)
        prompt_setting = self._extract_setting(prompt)
        
        # Check if generated text maintains these topics
        gen_animals = self._extract_animals(generated)
        gen_setting = self._extract_setting(generated)
        
        # Calculate topic consistency score
        animal_consistency = len(prompt_animals.intersection(gen_animals)) / max(len(prompt_animals), 1)
        
        # Penalize if completely different animals appear without context
        irrelevant_animals = gen_animals - prompt_animals
        animal_penalty = min(0.3, len(irrelevant_animals) * 0.1)
        
        topic_score = max(0.0, animal_consistency - animal_penalty)
        
        return topic_score
    
    def evaluate_logical_flow(self, text: str) -> float:
        """Evaluate logical flow and coherence of the text"""
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.0
        
        flow_score = 0.0
        
        # Check for narrative progression
        has_beginning = any(marker in text.lower() for marker in self.narrative_markers['beginning'])
        has_conflict = any(marker in text.lower() for marker in self.narrative_markers['conflict'])
        has_resolution = any(marker in text.lower() for marker in self.narrative_markers['resolution'])
        
        narrative_score = (has_beginning + has_conflict + has_resolution) / 3
        
        # Check for repetitive or circular content
        sentence_similarity = self._calculate_sentence_repetition(sentences)
        repetition_penalty = min(0.4, sentence_similarity * 0.5)
        
        # Check for abrupt topic changes
        topic_coherence = self._evaluate_topic_coherence(sentences)
        
        flow_score = max(0.0, (narrative_score * 0.4 + topic_coherence * 0.6) - repetition_penalty)
        
        return flow_score
    
    def evaluate_content_appropriateness(self, text: str) -> Dict[str, Any]:
        """Evaluate if content is appropriate for fables"""
        
        text_lower = text.lower()
        
        inappropriateness_score = 0.0
        flagged_categories = []
        
        for category, words in self.inappropriate_content.items():
            flagged_words = [word for word in words if word in text_lower]
            if flagged_words:
                inappropriateness_score += len(flagged_words) * 0.1
                flagged_categories.append({
                    'category': category,
                    'words': flagged_words
                })
        
        # Check for nonsensical content patterns
        nonsense_patterns = [
            r'\b(\w+)\s+\1\s+\1',  # Triple word repetition
            r'[a-z]+[A-Z][a-z]*[A-Z]',  # Mixed case words
            r'\b\d+\s*-\s*\d+\b',  # Number ranges (often nonsensical in fables)
        ]
        
        for pattern in nonsense_patterns:
            if re.search(pattern, text):
                inappropriateness_score += 0.15
        
        appropriateness_score = max(0.0, 1.0 - inappropriateness_score)
        
        return {
            'appropriateness_score': appropriateness_score,
            'flagged_categories': flagged_categories,
            'is_appropriate': appropriateness_score > 0.7
        }
    
    def evaluate_fable_relevance(self, text: str) -> float:
        """Evaluate if the text is actually fable-like"""
        
        text_lower = text.lower()
        
        # Check for animal characters
        animal_count = len([word for word in self.animal_words if word in text_lower])
        animal_score = min(1.0, animal_count * 0.3)
        
        # Check for moral/lesson elements
        moral_count = len([word for word in self.moral_keywords if word in text_lower])
        moral_score = min(1.0, moral_count * 0.2)
        
        # Check for fable-like language patterns
        fable_patterns = [
            r'once upon a time',
            r'there (lived|was)',
            r'the (moral|lesson)',
            r'learned that',
            r'realized that'
        ]
        
        pattern_score = sum(1 for pattern in fable_patterns if re.search(pattern, text_lower))
        pattern_score = min(1.0, pattern_score * 0.3)
        
        fable_relevance = (animal_score * 0.4 + moral_score * 0.3 + pattern_score * 0.3)
        
        return fable_relevance
    
    def _extract_animals(self, text: str) -> set:
        """Extract animal names from text"""
        text_lower = text.lower()
        return {animal for animal in self.animal_words if animal in text_lower}
    
    def _extract_setting(self, text: str) -> set:
        """Extract setting words from text"""
        settings = {'forest', 'meadow', 'farm', 'village', 'mountain', 'river', 'garden', 'field'}
        text_lower = text.lower()
        return {setting for setting in settings if setting in text_lower}
    
    def _calculate_sentence_repetition(self, sentences: List[str]) -> float:
        """Calculate how repetitive sentences are"""
        if len(sentences) < 2:
            return 0.0
        
        # Convert sentences to word sets for comparison
        sentence_words = [set(s.lower().split()) for s in sentences]
        
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(sentence_words)):
            for j in range(i + 1, len(sentence_words)):
                if sentence_words[i] and sentence_words[j]:
                    intersection = sentence_words[i].intersection(sentence_words[j])
                    union = sentence_words[i].union(sentence_words[j])
                    similarity = len(intersection) / len(union) if union else 0
                    total_similarity += similarity
                    comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def _evaluate_topic_coherence(self, sentences: List[str]) -> float:
        """Evaluate how well sentences connect topically"""
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence check: look for connecting words and consistent subjects
        connecting_words = {'and', 'but', 'so', 'then', 'however', 'therefore', 'because'}
        
        coherence_indicators = 0
        for sentence in sentences[1:]:  # Skip first sentence
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in connecting_words):
                coherence_indicators += 1
        
        # Check for subject consistency
        subjects = []
        for sentence in sentences:
            words = sentence.lower().split()
            if words:
                # Simple heuristic: first noun is likely the subject
                for word in words:
                    if word in self.animal_words:
                        subjects.append(word)
                        break
        
        subject_consistency = len(set(subjects)) / len(subjects) if subjects else 0.5
        subject_score = 1.0 - subject_consistency  # Lower diversity = higher consistency
        
        connecting_score = coherence_indicators / (len(sentences) - 1) if len(sentences) > 1 else 0.0
        
        return (connecting_score * 0.4 + subject_score * 0.6)
    
    def evaluate(self, model, tokenizer, test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """Evaluate semantic coherence metrics"""
        model.eval()
        
        self._log("Generating completions for semantic coherence analysis...")
        sample_results = []
        
        # Metrics tracking
        topic_scores = []
        flow_scores = []
        appropriateness_scores = []
        fable_relevance_scores = []
        flagged_samples = []
        
        for i, sample in enumerate(test_data):
            if i % 10 == 0:
                self.logger.log_evaluation_progress("semantic_coherence", i, len(test_data))
            
            prompt = sample['prompt']
            reference = sample.get('reference', '')
            
            # Generate completion
            completion = self.generate_completion(model, tokenizer, prompt)
            full_text = prompt + " " + completion
            
            # Evaluate semantic aspects
            topic_score = self.evaluate_topic_consistency(prompt, completion)
            flow_score = self.evaluate_logical_flow(full_text)
            appropriateness_result = self.evaluate_content_appropriateness(full_text)
            fable_score = self.evaluate_fable_relevance(full_text)
            
            # Track metrics
            topic_scores.append(topic_score)
            flow_scores.append(flow_score)
            appropriateness_scores.append(appropriateness_result['appropriateness_score'])
            fable_relevance_scores.append(fable_score)
            
            if appropriateness_result['flagged_categories']:
                flagged_samples.append({
                    'sample_id': i,
                    'flagged_categories': appropriateness_result['flagged_categories']
                })
            
            # Combine scores for overall semantic coherence
            semantic_coherence = (
                topic_score * 0.25 +
                flow_score * 0.35 +
                appropriateness_result['appropriateness_score'] * 0.2 +
                fable_score * 0.2
            )
            
            # Log sample if showing detailed logs
            if i < 5:
                sample_metrics = {
                    'topic_consistency': topic_score,
                    'logical_flow': flow_score,
                    'appropriateness': appropriateness_result['appropriateness_score'],
                    'fable_relevance': fable_score,
                    'semantic_coherence': semantic_coherence
                }
                self.logger.log_generation_sample(
                    sample_idx=i,
                    prompt=prompt,
                    generated=completion,
                    reference=reference,
                    metrics=sample_metrics
                )
            
            sample_result = {
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "generated_text": full_text[:300] + "..." if len(full_text) > 300 else full_text,
                "topic_consistency": topic_score,
                "logical_flow": flow_score,
                "appropriateness": appropriateness_result['appropriateness_score'],
                "fable_relevance": fable_score,
                "semantic_coherence": semantic_coherence,
                "is_appropriate": appropriateness_result['is_appropriate']
            }
            
            if self.config.save_generations:
                sample_result["full_generated_text"] = full_text
                sample_result["flagged_content"] = appropriateness_result['flagged_categories']
            
            sample_results.append(sample_result)
        
        # Calculate aggregate metrics
        self._log("Calculating semantic coherence metrics...")
        
        metrics = {
            # Individual component scores
            "avg_topic_consistency": MetricCalculator.mean(topic_scores),
            "std_topic_consistency": MetricCalculator.std(topic_scores),
            "avg_logical_flow": MetricCalculator.mean(flow_scores),
            "std_logical_flow": MetricCalculator.std(flow_scores),
            "avg_appropriateness": MetricCalculator.mean(appropriateness_scores),
            "std_appropriateness": MetricCalculator.std(appropriateness_scores),
            "avg_fable_relevance": MetricCalculator.mean(fable_relevance_scores),
            "std_fable_relevance": MetricCalculator.std(fable_relevance_scores),
            
            # Overall semantic coherence
            "overall_semantic_coherence": MetricCalculator.mean([
                sample["semantic_coherence"] for sample in sample_results
            ]),
            
            # Content quality metrics
            "inappropriate_content_rate": len(flagged_samples) / len(test_data),
            "appropriate_samples": sum(1 for sample in sample_results if sample["is_appropriate"]),
            "appropriate_sample_rate": sum(1 for sample in sample_results if sample["is_appropriate"]) / len(test_data),
            
            # Quality thresholds
            "high_quality_samples": sum(1 for sample in sample_results if sample["semantic_coherence"] > 0.7),
            "low_quality_samples": sum(1 for sample in sample_results if sample["semantic_coherence"] < 0.3),
            
            "num_samples": len(test_data)
        }
        
        return EvaluationResult(
            evaluator_name="SemanticCoherence",
            metrics=metrics,
            samples=sample_results,
            metadata={
                "model_name": self.config.model_name,
                "temperature": self.config.temperature,
                "flagged_samples": flagged_samples,
                "analysis_type": "semantic_coherence_and_appropriateness"
            }
        ) 
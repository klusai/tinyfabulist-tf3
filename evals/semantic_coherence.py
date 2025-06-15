"""
Semantic Coherence Evaluator
Evaluates whether generated text is semantically coherent, on-topic, and appropriate
"""

import re
import statistics
from typing import Dict, List, Any
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseEvaluator, EvaluationResult, EvaluationConfig


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
        """
        Evaluate semantic coherence, topic consistency, and content appropriateness.
        Expects test_data to contain 'fable' (prompt) and 'generated_text'.
        """
        
        sample_results = []

        # Labels for content appropriateness
        appropriateness_labels = ['appropriate', 'inappropriate', 'toxic', 'safe']
        
        # Labels for fable relevance
        relevance_labels = ['relevant to fables', 'off-topic', 'fantasy story', 'news article']
        
        for i, item in enumerate(test_data):
            prompt = item.get('prompt', '')
            completion = item.get('generated_text', '')

            if not completion or len(completion.split()) < 5:
                continue

            # 1. Topic Consistency
            topic_consistency = self.evaluate_topic_consistency(prompt, completion)
            
            # 2. Content Appropriateness
            appropriateness_result = self.evaluate_content_appropriateness(completion)
            appropriateness_score = appropriateness_result['appropriateness_score']
            
            # 3. Fable Relevance
            fable_relevance = self.evaluate_fable_relevance(completion)
            
            sample_results.append({
                'topic_consistency': topic_consistency,
                'appropriateness': appropriateness_score,
                'fable_relevance': fable_relevance
            })

        if not sample_results:
            self._log("No valid completions to evaluate.", level="warning")
            return EvaluationResult("semantic_coherence", {}, [])
            
        # Aggregate metrics
        metrics = self.aggregate_metrics(sample_results)
        
        return EvaluationResult(
            evaluator_name="semantic_coherence",
            metrics=metrics,
            samples=sample_results
        )

    def aggregate_metrics(self, sample_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate semantic coherence metrics."""
        
        if not sample_results:
            return {}

        avg_topic_consistency = np.mean([s['topic_consistency'] for s in sample_results])
        avg_appropriateness = np.mean([s['appropriateness'] for s in sample_results])
        avg_fable_relevance = np.mean([s['fable_relevance'] for s in sample_results])
        
        # Rate of appropriate samples
        appropriate_rate = np.mean([s['appropriateness'] > 0.7 for s in sample_results])
        
        # Overall semantic score
        overall_score = (
            (avg_topic_consistency * 0.4) +
            (avg_appropriateness * 0.4) +
            (avg_fable_relevance * 0.2)
        )
        
        return {
            'overall_semantic_coherence': overall_score,
            'avg_topic_consistency': avg_topic_consistency,
            'avg_appropriateness': avg_appropriateness,
            'avg_fable_relevance': avg_fable_relevance,
            'appropriate_sample_rate': appropriate_rate
        }

 
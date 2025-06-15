"""
Pure semantic evaluator
"""

import re
import numpy as np
from typing import Dict, List, Optional, Set
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.evaluator import Evaluator


class SemanticEvaluator(Evaluator):
    """Pure semantic coherence and appropriateness evaluation"""
    
    def __init__(self):
        self.animal_words = {
            'fox', 'rabbit', 'turtle', 'hare', 'lion', 'mouse', 'ant', 'grasshopper', 
            'crow', 'dove', 'wolf', 'deer', 'bear', 'eagle', 'snake', 'frog', 'owl',
            'cat', 'dog', 'pig', 'sheep', 'goat', 'horse', 'cow', 'chicken', 'duck'
        }
        
        self.inappropriate_words = {
            'kill', 'death', 'murder', 'blood', 'wound', 'hurt', 'pain', 'suffering',
            'surgery', 'doctor', 'hospital', 'patient', 'disease', 'medicine',
            'money', 'politics', 'war', 'weapon', 'fight', 'battle'
        }
    
    @property
    def name(self) -> str:
        return "semantic"
    
    @property
    def requires_references(self) -> bool:
        return False  # Semantic evaluation is unsupervised
    
    def evaluate(self, predictions: List[str], references: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate semantic coherence metrics"""
        
        coherence_scores = []
        appropriateness_scores = []
        fable_relevance_scores = []
        
        for text in predictions:
            if not text.strip():
                continue
            
            coherence = self._calculate_coherence(text)
            appropriateness = self._calculate_appropriateness(text)
            fable_relevance = self._calculate_fable_relevance(text)
            
            coherence_scores.append(coherence)
            appropriateness_scores.append(appropriateness)
            fable_relevance_scores.append(fable_relevance)
        
        if not coherence_scores:
            return {
                "coherence": 0.5,
                "appropriateness": 1.0,
                "fable_relevance": 0.0,
                "appropriate_rate": 1.0
            }
        
        avg_coherence = np.mean(coherence_scores)
        avg_appropriateness = np.mean(appropriateness_scores)
        avg_fable_relevance = np.mean(fable_relevance_scores)
        appropriate_rate = np.mean([score > 0.7 for score in appropriateness_scores])
        
        return {
            "coherence": avg_coherence,
            "appropriateness": avg_appropriateness,
            "fable_relevance": avg_fable_relevance,
            "appropriate_rate": appropriate_rate
        }
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate semantic coherence"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.5
        
        # Check for connecting words
        connecting_words = {'and', 'but', 'so', 'then', 'however', 'therefore', 'because'}
        coherence_indicators = 0
        
        for sentence in sentences[1:]:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in connecting_words):
                coherence_indicators += 1
        
        connecting_score = coherence_indicators / (len(sentences) - 1) if len(sentences) > 1 else 0.0
        
        # Check for subject consistency (animals)
        animals_mentioned = self._extract_animals(text)
        subject_consistency = min(1.0, len(animals_mentioned) / 2.0)  # 1-2 animals is good
        
        return (connecting_score * 0.6 + subject_consistency * 0.4)
    
    def _calculate_appropriateness(self, text: str) -> float:
        """Calculate content appropriateness for fables"""
        text_lower = text.lower()
        
        # Check for inappropriate content
        inappropriate_count = sum(1 for word in self.inappropriate_words if word in text_lower)
        inappropriateness_penalty = min(0.5, inappropriate_count * 0.1)
        
        # Check for nonsensical patterns
        nonsense_patterns = [
            r'\b(\w+)\s+\1\s+\1',  # Triple repetition
            r'[a-z]+[A-Z][a-z]*[A-Z]',  # Mixed case
        ]
        
        nonsense_penalty = 0.0
        for pattern in nonsense_patterns:
            if re.search(pattern, text):
                nonsense_penalty += 0.1
        
        appropriateness = max(0.0, 1.0 - inappropriateness_penalty - nonsense_penalty)
        return appropriateness
    
    def _calculate_fable_relevance(self, text: str) -> float:
        """Calculate how fable-like the text is"""
        text_lower = text.lower()
        
        # Check for animals
        animals_mentioned = self._extract_animals(text)
        animal_score = min(1.0, len(animals_mentioned) * 0.4)
        
        # Check for fable patterns
        fable_patterns = [
            r'once upon a time',
            r'there (lived|was)',
            r'the (moral|lesson)',
            r'learned that',
            r'realized that'
        ]
        
        pattern_count = sum(1 for pattern in fable_patterns if re.search(pattern, text_lower))
        pattern_score = min(1.0, pattern_count * 0.3)
        
        return animal_score * 0.7 + pattern_score * 0.3
    
    def _extract_animals(self, text: str) -> Set[str]:
        """Extract animal names from text"""
        text_lower = text.lower()
        return {animal for animal in self.animal_words if animal in text_lower} 
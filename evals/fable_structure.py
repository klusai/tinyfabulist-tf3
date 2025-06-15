"""
Fable structure evaluator for narrative elements specific to moral fables
"""

import re
from typing import Dict, List, Any, Set, Tuple, Optional
from .base import BaseEvaluator, EvaluationResult, EvaluationConfig
from transformers import pipeline
import numpy as np


class FableStructureEvaluator(BaseEvaluator):
    """Evaluator for fable-specific narrative structure and elements"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        super().__init__(config)
        
        self._log("Initializing FableStructureEvaluator and loading classifier...")
        try:
            self.classifier = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli"
            )
            self._log("✓ Classifier loaded.")
        except Exception as e:
            self._log(f"Failed to load zero-shot classifier: {e}", level="error")
            self.classifier = None
        
        # Define fable-specific vocabulary and patterns
        self.animal_characters = {
            'mouse', 'mice', 'rabbit', 'bunny', 'fox', 'owl', 'bear', 'wolf', 'lion', 
            'ant', 'grasshopper', 'turtle', 'tortoise', 'hare', 'eagle', 'crow', 
            'frog', 'cat', 'dog', 'sheep', 'goat', 'horse', 'pig', 'duck', 'hen',
            'rooster', 'chicken', 'bird', 'fish', 'whale', 'dolphin', 'elephant',
            'monkey', 'snake', 'deer', 'squirrel', 'beaver'
        }
        
        self.fable_settings = {
            'forest', 'woods', 'meadow', 'garden', 'farm', 'village', 'pond', 
            'river', 'stream', 'mountain', 'hill', 'valley', 'field', 'barn',
            'nest', 'burrow', 'cave', 'tree', 'branch', 'ocean', 'sea', 'lake'
        }
        
        self.moral_indicators = {
            'moral', 'lesson', 'learned', 'teaching', 'wisdom', 'important',
            'remember', 'always', 'never', 'should', 'must', 'ought'
        }
        
        self.conflict_words = {
            'but', 'however', 'suddenly', 'then', 'problem', 'trouble', 'danger',
            'worried', 'afraid', 'scared', 'angry', 'sad', 'upset', '困難'
        }
        
        self.resolution_words = {
            'learned', 'realized', 'understood', 'discovered', 'found', 'decided',
            'helped', 'solved', 'fixed', 'became', 'changed', 'grew'
        }
    
    def extract_characters(self, text: str) -> Set[str]:
        """Extract animal characters mentioned in the text"""
        text_lower = text.lower()
        found_animals = set()
        
        for animal in self.animal_characters:
            if re.search(r'\b' + animal + r'\b', text_lower):
                found_animals.add(animal)
        
        return found_animals
    
    def extract_settings(self, text: str) -> Set[str]:
        """Extract setting elements mentioned in the text"""
        text_lower = text.lower()
        found_settings = set()
        
        for setting in self.fable_settings:
            if re.search(r'\b' + setting + r'\b', text_lower):
                found_settings.add(setting)
        
        return found_settings
    
    def analyze_narrative_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the narrative structure of a fable"""
        text_lower = text.lower()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Initialize structure elements
        structure = {
            'has_opening': False,
            'has_character_intro': False,
            'has_setting': False,
            'has_conflict': False,
            'has_resolution': False,
            'has_moral': False,
            'characters': set(),
            'settings': set(),
            'narrative_flow_score': 0.0
        }
        
        # Check for opening patterns
        opening_patterns = [
            r'\bonce upon a time\b', r'\bthere was\b', r'\bthere lived\b',
            r'\bin a\b.*\blived\b', r'\blong ago\b', r'\bmany years ago\b'
        ]
        structure['has_opening'] = any(re.search(pattern, text_lower) for pattern in opening_patterns)
        
        # Extract characters and settings
        structure['characters'] = self.extract_characters(text)
        structure['settings'] = self.extract_settings(text)
        
        structure['has_character_intro'] = len(structure['characters']) > 0
        structure['has_setting'] = len(structure['settings']) > 0
        
        # Check for conflict
        conflict_found = any(word in text_lower for word in self.conflict_words)
        structure['has_conflict'] = conflict_found
        
        # Check for resolution
        resolution_found = any(word in text_lower for word in self.resolution_words)
        structure['has_resolution'] = resolution_found
        
        # Check for moral/lesson
        moral_found = any(word in text_lower for word in self.moral_indicators)
        structure['has_moral'] = moral_found
        
        # Calculate narrative flow score
        flow_elements = [
            structure['has_opening'],
            structure['has_character_intro'],
            structure['has_setting'],
            structure['has_conflict'],
            structure['has_resolution']
        ]
        structure['narrative_flow_score'] = sum(flow_elements) / len(flow_elements)
        
        return structure
    
    def analyze_moral_clarity(self, text: str) -> Dict[str, float]:
        """Analyze how clearly the moral is expressed"""
        text_lower = text.lower()
        
        # Explicit moral statements
        explicit_moral_patterns = [
            r'the moral.*is', r'moral of.*story', r'lesson.*is',
            r'this teaches', r'we learn', r'important.*remember'
        ]
        
        has_explicit_moral = any(re.search(pattern, text_lower) for pattern in explicit_moral_patterns)
        
        # Implicit moral through actions/consequences
        consequence_patterns = [
            r'because.*', r'so.*', r'therefore.*', r'as a result.*',
            r'from that day.*', r'ever since.*', r'always.*'
        ]
        
        has_consequences = any(re.search(pattern, text_lower) for pattern in consequence_patterns)
        
        # Moral vocabulary density
        moral_word_count = sum(1 for word in self.moral_indicators if word in text_lower)
        total_words = len(text.split())
        moral_density = moral_word_count / total_words if total_words > 0 else 0.0
        
        # Overall moral clarity score
        clarity_components = [
            1.0 if has_explicit_moral else 0.5 if has_consequences else 0.0,
            min(1.0, moral_density * 50),  # Scale moral density
        ]
        
        clarity_score = sum(clarity_components) / len(clarity_components)
        
        return {
            'has_explicit_moral': has_explicit_moral,
            'has_consequences': has_consequences,
            'moral_density': moral_density,
            'moral_clarity_score': clarity_score
        }
    
    def analyze_fable_style(self, text: str) -> Dict[str, float]:
        """Analyze stylistic elements typical of fables"""
        text_lower = text.lower()
        
        # Simple language (shorter words, common vocabulary)
        words = text.split()
        if not words:
            return {'style_score': 0.0, 'avg_word_length': 0.0, 'simple_language_ratio': 0.0}
        
        avg_word_length = sum(len(word.strip('.,!?;:')) for word in words) / len(words)
        
        # Simple words (5 characters or less)
        simple_words = sum(1 for word in words if len(word.strip('.,!?;:')) <= 5)
        simple_language_ratio = simple_words / len(words)
        
        # Anthropomorphic elements (animals doing human things)
        anthropomorphic_patterns = [
            r'said', r'thought', r'decided', r'walked', r'ran', r'worked',
            r'talked', r'spoke', r'lived', r'felt', r'knew', r'understood'
        ]
        
        anthropomorphic_count = sum(1 for pattern in anthropomorphic_patterns 
                                   if re.search(r'\b' + pattern + r'\b', text_lower))
        
        # Direct speech/dialogue
        has_dialogue = '"' in text or ''' in text or ''' in text
        
        # Third person narrative
        third_person_indicators = ['he', 'she', 'it', 'they', 'the']
        third_person_count = sum(1 for indicator in third_person_indicators 
                                if indicator in text_lower.split())
        
        # Style score components
        style_components = [
            min(1.0, simple_language_ratio * 1.5),  # Prefer simple language
            min(1.0, anthropomorphic_count / 5),    # Some anthropomorphism expected
            0.2 if has_dialogue else 0.0,           # Dialogue is good but not required
            min(1.0, third_person_count / 10)       # Third person narrative
        ]
        
        style_score = sum(style_components) / len(style_components)
        
        return {
            'avg_word_length': avg_word_length,
            'simple_language_ratio': simple_language_ratio,
            'anthropomorphic_elements': anthropomorphic_count,
            'has_dialogue': has_dialogue,
            'style_score': style_score
        }
    
    def evaluate(self, model, tokenizer, test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluate fable structure using a zero-shot classification model.
        Expects test_data to contain 'generated_text'.
        """
        
        sample_results = []
        
        # Candidate labels for fable structure classification
        fable_labels = [
            "clear narrative", "moral of the story", "character development", 
            "has conflict", "has resolution", "is a complete story",
            "confusing", "lacks a moral", "is just a list of events"
        ]
        
        for i, item in enumerate(test_data):
            completion = item.get('generated_text', '')
            
            if not completion or len(completion.split()) < 10:
                self._log(f"Skipping sample {i} due to short/empty completion.")
                continue

            # Use zero-shot classification to evaluate structure
            try:
                result = self.classifier(completion, fable_labels, multi_label=True)
                scores = {label: score for label, score in zip(result['labels'], result['scores'])}
                
                # Calculate scores for this sample
                narrative_flow = scores.get('clear narrative', 0.0) - scores.get('confusing', 0.0)
                moral_clarity = scores.get('moral of the story', 0.0) - scores.get('lacks a moral', 0.0)
                
                sample_results.append({
                    'narrative_flow_score': max(0, narrative_flow),
                    'moral_clarity_score': max(0, moral_clarity),
                    'has_conflict': scores.get('has conflict', 0.0),
                    'has_resolution': scores.get('has resolution', 0.0)
                })
                
            except Exception as e:
                self._log(f"Error classifying sample {i}: {e}", level="warning")
                continue

        if not sample_results:
            self._log("No valid completions found to evaluate.", level="warning")
            return EvaluationResult("fable_structure", {}, [])

        # Aggregate metrics
        metrics = self.aggregate_metrics(sample_results)
        
        return EvaluationResult(
            evaluator_name="fable_structure",
            metrics=metrics,
            samples=sample_results
        )

    def aggregate_metrics(self, sample_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate per-sample structure metrics."""
        
        if not sample_results:
            return {}

        avg_narrative_flow = np.mean([s['narrative_flow_score'] for s in sample_results])
        avg_moral_clarity = np.mean([s['moral_clarity_score'] for s in sample_results])
        
        # Calculate percentages for binary-like attributes
        conflict_percentage = np.mean([s['has_conflict'] > 0.6 for s in sample_results]) * 100
        resolution_percentage = np.mean([s['has_resolution'] > 0.6 for s in sample_results]) * 100
        
        # Overall fable score
        overall_score = (
            (avg_narrative_flow * 0.5) +
            (avg_moral_clarity * 0.5)
        )
        
        return {
            'overall_fable_score': overall_score,
            'avg_narrative_flow_score': avg_narrative_flow,
            'avg_moral_clarity_score': avg_moral_clarity,
            'has_conflict_percentage': conflict_percentage,
            'has_resolution_percentage': resolution_percentage
        } 
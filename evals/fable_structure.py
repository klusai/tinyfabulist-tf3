"""
Fable structure evaluator for narrative elements specific to moral fables
"""

import re
from typing import Dict, List, Any, Set, Tuple
from .base import BaseEvaluator, EvaluationResult, MetricCalculator


class FableStructureEvaluator(BaseEvaluator):
    """Evaluator for fable-specific narrative structure and elements"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
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
        """Evaluate fable structure and elements"""
        model.eval()
        
        # Generate completions
        self._log("Generating completions for fable structure analysis...")
        sample_results = []
        
        # Aggregate metrics collectors
        structure_scores = []
        moral_clarity_scores = []
        style_scores = []
        character_counts = []
        setting_counts = []
        
        for i, sample in enumerate(test_data):
            if i % 20 == 0:
                self._log(f"Analyzing sample {i+1}/{len(test_data)}")
            
            prompt = sample['prompt']
            
            # Generate completion
            completion = self.generate_completion(model, tokenizer, prompt)
            full_text = prompt + " " + completion
            
            # Analyze structure
            structure_analysis = self.analyze_narrative_structure(full_text)
            moral_analysis = self.analyze_moral_clarity(full_text)
            style_analysis = self.analyze_fable_style(full_text)
            
            # Collect scores for aggregation
            structure_scores.append(structure_analysis['narrative_flow_score'])
            moral_clarity_scores.append(moral_analysis['moral_clarity_score'])
            style_scores.append(style_analysis['style_score'])
            character_counts.append(len(structure_analysis['characters']))
            setting_counts.append(len(structure_analysis['settings']))
            
            # Prepare sample result
            sample_result = {
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "generated_text": full_text[:300] + "..." if len(full_text) > 300 else full_text,
                "narrative_flow_score": structure_analysis['narrative_flow_score'],
                "moral_clarity_score": moral_analysis['moral_clarity_score'],
                "style_score": style_analysis['style_score'],
                "character_count": len(structure_analysis['characters']),
                "setting_count": len(structure_analysis['settings']),
                "has_opening": structure_analysis['has_opening'],
                "has_conflict": structure_analysis['has_conflict'],
                "has_resolution": structure_analysis['has_resolution'],
                "has_moral": structure_analysis['has_moral'],
                "characters": list(structure_analysis['characters']),
                "settings": list(structure_analysis['settings'])
            }
            
            if self.config.save_generations:
                sample_result.update({
                    "full_generated_text": full_text,
                    "detailed_structure": structure_analysis,
                    "detailed_moral": moral_analysis,
                    "detailed_style": style_analysis
                })
            
            sample_results.append(sample_result)
        
        # Calculate aggregate metrics
        self._log("Calculating aggregate fable structure metrics...")
        
        # Overall fable quality score (composite)
        fable_quality_components = [
            MetricCalculator.mean(structure_scores),
            MetricCalculator.mean(moral_clarity_scores),
            MetricCalculator.mean(style_scores)
        ]
        
        overall_fable_score = MetricCalculator.mean(fable_quality_components)
        
        # Structure element statistics
        structure_elements = ['has_opening', 'has_conflict', 'has_resolution', 'has_moral']
        structure_element_stats = {}
        
        for element in structure_elements:
            element_presence = [sample[element] for sample in sample_results]
            structure_element_stats[f"{element}_percentage"] = (sum(element_presence) / len(element_presence)) * 100
        
        metrics = {
            # Composite scores
            "overall_fable_score": overall_fable_score,
            "avg_narrative_flow_score": MetricCalculator.mean(structure_scores),
            "avg_moral_clarity_score": MetricCalculator.mean(moral_clarity_scores),
            "avg_style_score": MetricCalculator.mean(style_scores),
            
            # Structure statistics
            **structure_element_stats,
            
            # Character and setting statistics
            "avg_character_count": MetricCalculator.mean(character_counts),
            "avg_setting_count": MetricCalculator.mean(setting_counts),
            "std_character_count": MetricCalculator.std(character_counts),
            "std_setting_count": MetricCalculator.std(setting_counts),
            
            # Distribution statistics
            "structure_score_std": MetricCalculator.std(structure_scores),
            "moral_clarity_std": MetricCalculator.std(moral_clarity_scores),
            "style_score_std": MetricCalculator.std(style_scores),
            
            "num_samples": len(test_data)
        }
        
        return EvaluationResult(
            evaluator_name="FableStructure",
            metrics=metrics,
            samples=sample_results,
            metadata={
                "model_name": self.config.model_name,
                "temperature": self.config.temperature,
                "analysis_components": ["narrative_structure", "moral_clarity", "fable_style"],
                "animal_vocabulary_size": len(self.animal_characters),
                "setting_vocabulary_size": len(self.fable_settings)
            }
        ) 
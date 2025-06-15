"""
Text preparation utilities
"""

import re
from typing import List, Dict, Any, Optional
import sys
import os

# Add lib directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.logging_utils import get_logger

logger = get_logger()


def prepare_test_data(dataset, num_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Prepare test data with natural prompt generation.
    Uses the improved methodology that generates natural prompts instead of arbitrary splitting.
    """
    
    # Shuffle and sample
    dataset = dataset.shuffle(seed=seed)
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    test_data = []
    
    for i, example in enumerate(dataset):
        fable = example['fable']
        
        # Generate natural prompt using the improved methodology
        prompt_data = _generate_natural_prompt(fable, i)
        
        if prompt_data is None:
            continue
            
        test_data.append({
            'prompt': prompt_data['prompt'],
            'reference': prompt_data['reference'],
            'full_text': fable,
            'original_example': example
        })
    
    logger.info(f"Prepared {len(test_data)} test samples")
    return test_data[:num_samples]


def _generate_natural_prompt(fable_text: str, sample_id: int) -> Optional[Dict[str, Any]]:
    """
    Generate natural fable prompts that don't require splitting existing narratives.
    This addresses the "arbitrary text splitting" issue identified in the research critique.
    """
    
    # Extract elements from the reference fable for inspiration
    animals = _extract_animals(fable_text)
    settings = _extract_settings(fable_text)
    traits = _extract_character_traits(fable_text)
    
    # Generate different types of natural prompts
    prompt_generators = [
        _generate_setting_prompt,
        _generate_character_prompt,
        _generate_scenario_prompt,
        _generate_opening_prompt
    ]
    
    # Use sample_id to ensure reproducible prompt selection
    generator = prompt_generators[sample_id % len(prompt_generators)]
    
    try:
        prompt = generator(animals, settings, traits)
        
        # Create a reference by taking a portion of the original fable
        # This is for evaluation purposes only
        sentences = [s.strip() for s in fable_text.split('.') if s.strip()]
        if len(sentences) >= 3:
            reference = '. '.join(sentences[:len(sentences)//2]) + '.'
        else:
            reference = fable_text
        
        return {
            'prompt': prompt,
            'reference': reference
        }
    except Exception as e:
        logger.warning(f"Failed to generate prompt for sample {sample_id}: {e}")
        return None


def _extract_animals(text: str) -> List[str]:
    """Extract animal names from text"""
    animal_words = {
        'fox', 'rabbit', 'turtle', 'hare', 'lion', 'mouse', 'ant', 'grasshopper', 
        'crow', 'dove', 'wolf', 'deer', 'bear', 'eagle', 'snake', 'frog', 'owl',
        'cat', 'dog', 'pig', 'sheep', 'goat', 'horse', 'cow', 'chicken', 'duck'
    }
    text_lower = text.lower()
    found_animals = [animal for animal in animal_words if animal in text_lower]
    return found_animals[:2]  # Limit to 2 animals


def _extract_settings(text: str) -> List[str]:
    """Extract setting words from text"""
    settings = {
        'forest', 'meadow', 'farm', 'village', 'mountain', 'river', 'garden', 
        'field', 'pond', 'tree', 'house', 'barn', 'nest'
    }
    text_lower = text.lower()
    found_settings = [setting for setting in settings if setting in text_lower]
    return found_settings[:2]  # Limit to 2 settings


def _extract_character_traits(text: str) -> List[str]:
    """Extract character traits from text"""
    traits = {
        'clever', 'wise', 'patient', 'kind', 'proud', 'humble', 'quick', 'slow',
        'strong', 'gentle', 'brave', 'careful', 'lazy', 'hardworking'
    }
    text_lower = text.lower()
    found_traits = [trait for trait in traits if trait in text_lower]
    return found_traits[:2]  # Limit to 2 traits


def _generate_setting_prompt(animals: List[str], settings: List[str], traits: List[str]) -> str:
    """Generate a setting-based prompt"""
    animal = animals[0] if animals else "fox"
    setting = settings[0] if settings else "peaceful forest"
    trait = traits[0] if traits else "clever"
    
    return f"In a {setting}, a {trait} {animal} lived quietly. One day,"


def _generate_character_prompt(animals: List[str], settings: List[str], traits: List[str]) -> str:
    """Generate a character-based prompt"""
    animal = animals[0] if animals else "rabbit"
    trait = traits[0] if traits else "curious"
    
    return f"Once upon a time, there was a {trait} {animal} who loved to explore. One morning,"


def _generate_scenario_prompt(animals: List[str], settings: List[str], traits: List[str]) -> str:
    """Generate a scenario-based prompt"""
    animal1 = animals[0] if len(animals) > 0 else "turtle"
    animal2 = animals[1] if len(animals) > 1 else "hare"
    
    return f"A {animal1} and a {animal2} met at the edge of the forest. The {animal1} said,"


def _generate_opening_prompt(animals: List[str], settings: List[str], traits: List[str]) -> str:
    """Generate a classic fable opening prompt"""
    animal = animals[0] if animals else "mouse"
    setting = settings[0] if settings else "meadow"
    
    return f"In a sunny {setting}, a small {animal} was busy with its daily tasks when suddenly," 
"""
Dataset Loading Utilities
Handles dataset loading with fallbacks and test data generation
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset, Dataset
import warnings


class DatasetLoader:
    """Safe dataset loading with multiple fallback strategies"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def load_fable_dataset(self, dataset_name: str = "roneneldan/TinyStories", 
                          split: str = "train", 
                          max_samples: Optional[int] = None) -> Dataset:
        """Load fable/story dataset with fallback strategies"""
        
        if self.verbose:
            print(f"📚 Loading dataset: {dataset_name}")
        
        # Try multiple loading strategies
        strategies = [
            self._load_primary_dataset,
            self._load_alternative_dataset,
            self._create_synthetic_dataset
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                dataset = strategy(dataset_name, split, max_samples)
                if dataset is not None:
                    if self.verbose:
                        print(f"✅ Dataset loaded successfully using strategy {i+1}")
                    return dataset
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Strategy {i+1} failed: {e}")
                continue
        
        # If all strategies fail, create synthetic data
        if self.verbose:
            print("🔄 All strategies failed, creating synthetic dataset")
        return self._create_synthetic_dataset(dataset_name, split, max_samples or 100)
    
    def _load_primary_dataset(self, dataset_name: str, split: str, max_samples: Optional[int]) -> Dataset:
        """Primary dataset loading strategy"""
        
        # Handle different dataset names
        if "DS-TF1-EN-3M" in dataset_name:
            # Try the problematic dataset first
            try:
                dataset = load_dataset("roneneldan/TinyStories", split=split)
                if self.verbose:
                    print(f"  📖 Loaded TinyStories dataset ({len(dataset)} samples)")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  TinyStories failed: {e}")
                raise e
        else:
            # Load specified dataset
            dataset = load_dataset(dataset_name, split=split)
            if self.verbose:
                print(f"  📖 Loaded {dataset_name} ({len(dataset)} samples)")
        
        # Limit samples if requested
        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            if self.verbose:
                print(f"  ✂️  Limited to {max_samples} samples")
        
        return dataset
    
    def _load_alternative_dataset(self, dataset_name: str, split: str, max_samples: Optional[int]) -> Dataset:
        """Alternative dataset loading with different configurations"""
        
        # Try different dataset configurations
        alternatives = [
            ("roneneldan/TinyStories", {}),
            ("roneneldan/TinyStories", {"streaming": True}),
            ("facebook/xsum", {}),  # Fallback to a different story dataset
        ]
        
        for alt_name, alt_config in alternatives:
            try:
                if alt_config.get("streaming"):
                    # Handle streaming datasets
                    dataset = load_dataset(alt_name, split=split, streaming=True)
                    # Convert to regular dataset with limited samples
                    samples = list(dataset.take(max_samples or 1000))
                    dataset = Dataset.from_list(samples)
                else:
                    dataset = load_dataset(alt_name, split=split, **alt_config)
                
                if self.verbose:
                    print(f"  📖 Loaded alternative dataset: {alt_name}")
                
                # Limit samples if requested
                if max_samples is not None and len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
                
                return dataset
                
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Alternative {alt_name} failed: {e}")
                continue
        
        raise RuntimeError("All alternative datasets failed")
    
    def _create_synthetic_dataset(self, dataset_name: str, split: str, max_samples: int) -> Dataset:
        """Create synthetic fable dataset as fallback"""
        
        if self.verbose:
            print(f"  🎭 Creating synthetic fable dataset ({max_samples} samples)")
        
        # Generate synthetic fable data
        fables = []
        for i in range(max_samples):
            fable = self._generate_synthetic_fable(i)
            fables.append({"text": fable})
        
        return Dataset.from_list(fables)
    
    def _generate_synthetic_fable(self, seed: int) -> str:
        """Generate a single synthetic fable using better reference templates"""
        
        # Import the better synthetic data generator
        from .data_loading import generate_synthetic_fable_data
        
        # Get high-quality fable templates
        fable_templates = generate_synthetic_fable_data(20)  # Get enough templates
        
        # Select template based on seed for reproducibility
        template_idx = seed % len(fable_templates)
        template = fable_templates[template_idx]
        
        # Return the full fable (prompt + reference)
        return template["prompt"] + " " + template["reference"]


def create_fable_test_data(num_samples: int = 10) -> List[Dict[str, str]]:
    """Create test fable data for evaluation"""
    
    # Import the better synthetic data generator
    from .data_loading import generate_synthetic_fable_data
    
    # Get high-quality fable templates
    fable_templates = generate_synthetic_fable_data(num_samples)
    
    test_data = []
    for i, template in enumerate(fable_templates[:num_samples]):
        test_data.append({
            "prompt": template["prompt"],
            "completion": template["reference"],
            "reference": template["reference"],
            "full_text": template["prompt"] + " " + template["reference"]
        })
    
    return test_data


def prepare_dataset_for_evaluation(dataset: Dataset, 
                                 text_column: str = "text",
                                 max_samples: Optional[int] = None) -> List[Dict[str, str]]:
    """Prepare dataset for evaluation by creating prompt-completion pairs"""
    
    evaluation_data = []
    
    # Limit samples if requested
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    for item in dataset:
        text = item[text_column]
        
        # Split text into prompt and completion
        # Use first sentence as prompt, rest as completion
        sentences = text.split('. ')
        
        if len(sentences) >= 2:
            prompt = sentences[0] + '.'
            completion = '. '.join(sentences[1:])
        else:
            # Fallback: use first half as prompt, second half as completion
            mid_point = len(text) // 2
            prompt = text[:mid_point]
            completion = text[mid_point:]
        
        evaluation_data.append({
            "prompt": prompt,
            "completion": completion,
            "full_text": text
        })
    
    return evaluation_data


def get_dataset_info(dataset: Dataset) -> Dict[str, Any]:
    """Get information about a dataset"""
    
    info = {
        "size": len(dataset),
        "columns": list(dataset.column_names),
        "features": dataset.features
    }
    
    # Sample some text lengths
    if "text" in dataset.column_names:
        text_lengths = [len(item["text"]) for item in dataset.select(range(min(100, len(dataset))))]
        info["text_stats"] = {
            "avg_length": sum(text_lengths) / len(text_lengths),
            "min_length": min(text_lengths),
            "max_length": max(text_lengths)
        }
    
    return info 
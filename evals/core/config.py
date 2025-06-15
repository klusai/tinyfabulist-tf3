"""
Minimal evaluation configuration
"""

from dataclasses import dataclass
from typing import Optional
import sys
import os

# Add lib directory to path for device detection
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from lib import get_optimal_device


@dataclass
class EvaluationConfig:
    """Minimal, focused evaluation configuration"""
    
    # Model settings
    model_name: str = "gpt2"
    temperature: float = 0.8
    max_length: int = 512
    max_new_tokens: int = 256
    
    # Dataset settings
    dataset_name: str = "klusai/ds-tf1-en-3m"
    dataset_split: str = "Test"
    num_samples: int = 100
    
    # Generation settings
    seed: int = 42
    device: Optional[str] = None
    
    # Output settings
    save_generations: bool = False
    verbose: bool = True
    
    def __post_init__(self):
        """Set device if not specified"""
        if self.device is None:
            self.device = get_optimal_device(verbose=False) 
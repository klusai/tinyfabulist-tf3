"""
Minimal evaluator interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class Evaluator(ABC):
    """Minimal evaluator interface for metrics calculation"""
    
    @abstractmethod
    def evaluate(self, predictions: List[str], references: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate metrics for predictions vs references.
        
        Args:
            predictions: Generated text samples
            references: Reference text samples (optional for unsupervised metrics)
            
        Returns:
            Dictionary of metric_name -> value
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Evaluator name for identification"""
        pass
    
    @property
    def requires_references(self) -> bool:
        """Whether this evaluator requires reference texts"""
        return True 
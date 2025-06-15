"""
Simple evaluation result container
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


@dataclass
class EvaluationResult:
    """Simple container for evaluation results"""
    
    metrics: Dict[str, float] = field(default_factory=dict)
    samples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "metrics": self.metrics,
            "samples": self.samples[:10] if len(self.samples) > 10 else self.samples,  # Limit for size
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp
        }
    
    def save_json(self, filepath: str):
        """Save results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        """Get a summary string of key metrics"""
        lines = ["=== Evaluation Results ==="]
        
        for key, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
        
        lines.append(f"Execution time: {self.execution_time:.2f}s")
        lines.append(f"Samples: {len(self.samples)}")
        
        return "\n".join(lines) 
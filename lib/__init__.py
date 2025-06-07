"""
TinyFabulist Library
Reusable utilities for GPT model evaluation and fine-tuning
"""

from .device_manager import DeviceManager, get_optimal_device, setup_model_for_device, SafeGeneration, optimize_for_apple_silicon
from .model_loader import ModelLoader, ModelCache, load_model_safe
from .dataset_utils import DatasetLoader, create_fable_test_data
from .logging_utils import EvaluationLogger, setup_logging, get_logger, set_log_level, disable_generated_text_logging, enable_generated_text_logging

__version__ = "1.0.0"

__all__ = [
    "DeviceManager",
    "get_optimal_device", 
    "setup_model_for_device",
    "SafeGeneration",
    "optimize_for_apple_silicon",
    "ModelLoader",
    "ModelCache",
    "load_model_safe",
    "DatasetLoader",
    "create_fable_test_data",
    "EvaluationLogger",
    "setup_logging",
    "get_logger",
    "set_log_level",
    "disable_generated_text_logging",
    "enable_generated_text_logging"
] 
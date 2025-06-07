"""
Logging utilities for TinyFabulist evaluation framework
Provides structured logging with different levels and formatters
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class EvaluationLogger:
    """Centralized logging for evaluation runs"""
    
    def __init__(self, 
                 name: str = "tinyfabulist",
                 level: str = "INFO",
                 log_file: Optional[str] = None,
                 console_output: bool = True,
                 show_generated_text: bool = True):
        
        self.name = name
        self.show_generated_text = show_generated_text
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with colors
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with optional context"""
        if kwargs:
            context_parts = [f"{k}={v}" for k, v in kwargs.items()]
            return f"{message} | {' | '.join(context_parts)}"
        return message
    
    def log_model_info(self, model_name: str, device: str, **details):
        """Log model loading information"""
        self.info(f"🤖 Model loaded: {model_name}", device=device, **details)
    
    def log_dataset_info(self, dataset_name: str, split: str, size: int, **details):
        """Log dataset information"""
        self.info(f"📊 Dataset loaded: {dataset_name}", split=split, samples=size, **details)
    
    def log_evaluation_start(self, evaluator: str, config: Dict[str, Any]):
        """Log evaluation start"""
        self.info(f"🚀 Starting evaluation: {evaluator}")
        self.debug("Configuration", **{k: str(v) for k, v in config.items()})
    
    def log_evaluation_progress(self, evaluator: str, current: int, total: int, **metrics):
        """Log evaluation progress"""
        progress = f"{current}/{total} ({100*current/total:.1f}%)"
        self.info(f"⏳ {evaluator} progress: {progress}", **metrics)
    
    def log_generation_sample(self, 
                            sample_idx: int,
                            prompt: str, 
                            generated: str, 
                            reference: str = None,
                            metrics: Dict[str, float] = None):
        """Log a generation sample with formatting"""
        if not self.show_generated_text:
            return
        
        self.info(f"📝 Generation Sample #{sample_idx + 1}")
        
        # Format prompt
        prompt_preview = self._truncate_text(prompt, 100)
        self.info(f"   PROMPT: {prompt_preview}")
        
        # Format generated text
        generated_preview = self._truncate_text(generated, 200)
        self.info(f"   GENERATED: {generated_preview}")
        
        # Format reference if available
        if reference:
            reference_preview = self._truncate_text(reference, 150)
            self.info(f"   REFERENCE: {reference_preview}")
        
        # Log metrics if available
        if metrics:
            metric_str = " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            self.info(f"   METRICS: {metric_str}")
        
        self.info("   " + "─" * 60)  # Separator
    
    def log_evaluation_result(self, evaluator: str, metrics: Dict[str, Any], execution_time: float):
        """Log evaluation results"""
        self.info(f"✅ {evaluator} completed", time=f"{execution_time:.2f}s")
        
        # Log key metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    self.info(f"   📊 {key}: {value:.4f}")
                else:
                    self.info(f"   📊 {key}: {value}")
    
    def log_error_context(self, error: Exception, context: str = "", **details):
        """Log error with context"""
        self.error(f"❌ {context}: {str(error)}", **details)
        
        # Log traceback in debug mode
        import traceback
        self.debug(f"Traceback: {traceback.format_exc()}")
    
    def log_device_selection(self, available_devices: List[str], selected_device: str, reason: str = ""):
        """Log device selection details"""
        self.info(f"🖥️  Device selected: {selected_device}", 
                 available=available_devices, reason=reason)
    
    def log_generation_config(self, **config):
        """Log text generation configuration"""
        self.debug("🎛️  Generation config", **config)
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text for display"""
        if len(text) <= max_length:
            return repr(text)  # Show quotes and escape sequences
        return repr(text[:max_length] + "...")


# Global logger instance
_global_logger: Optional[EvaluationLogger] = None


def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 show_generated_text: bool = True,
                 console_output: bool = True) -> EvaluationLogger:
    """Setup global logging configuration"""
    global _global_logger
    
    _global_logger = EvaluationLogger(
        level=level,
        log_file=log_file,
        console_output=console_output,
        show_generated_text=show_generated_text
    )
    
    return _global_logger


def get_logger() -> EvaluationLogger:
    """Get the global logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = setup_logging()
    
    return _global_logger


def set_log_level(level: str):
    """Set the global log level"""
    logger = get_logger()
    logger.logger.setLevel(getattr(logging, level.upper()))


def disable_generated_text_logging():
    """Disable logging of generated text samples"""
    logger = get_logger()
    logger.show_generated_text = False


def enable_generated_text_logging():
    """Enable logging of generated text samples"""
    logger = get_logger()
    logger.show_generated_text = True 
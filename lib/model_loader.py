"""
Safe Model Loading Utilities
Handles model loading with proper device management and error handling
"""

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Tuple, Dict, Any
import warnings

from .device_manager import DeviceManager, SafeGeneration


class ModelLoader:
    """Safe model loading with device management"""
    
    def __init__(self, model_name: str = "gpt2", prefer_mps: bool = True, verbose: bool = True):
        self.model_name = model_name
        self.prefer_mps = prefer_mps
        self.verbose = verbose
        self.device_manager = DeviceManager(prefer_mps=prefer_mps, verbose=verbose)
        
    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
        """Load model and tokenizer with optimal device placement"""
        
        if self.verbose:
            print(f"🤖 Loading model: {self.model_name}")
        
        # Load tokenizer first (always on CPU)
        tokenizer = self._load_tokenizer()
        
        # Load model with device management
        model = self._load_model()
        
        # Move model to optimal device
        model = self.device_manager.move_model_to_device(model)
        device = self.device_manager.get_device()
        
        if self.verbose:
            print(f"✅ Model loaded successfully on {device}")
            
        return model, tokenizer, device
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer with proper configuration"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            if self.verbose:
                print(f"  📝 Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
                
            return tokenizer
            
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer for {self.model_name}: {e}")
    
    def _load_model(self) -> AutoModelForCausalLM:
        """Load model with appropriate configuration"""
        try:
            # Load with minimal memory usage initially
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Start with float32 for compatibility
                low_cpu_mem_usage=True,
                device_map=None  # We'll handle device placement manually
            )
            
            # Set model to evaluation mode
            model.eval()
            
            if self.verbose:
                num_params = sum(p.numel() for p in model.parameters())
                print(f"  🧠 Model loaded ({num_params:,} parameters)")
                
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def generate_text(self, model, tokenizer, prompt: str, device: str, **kwargs) -> str:
        """Generate text using safe generation methods"""
        
        # Prepare input
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Set default generation parameters
        generation_kwargs = {
            'max_new_tokens': 100,
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'pad_token_id': tokenizer.eos_token_id,
            **kwargs
        }
        
        # Generate using safe method
        with torch.no_grad():
            outputs = SafeGeneration.generate_with_fallback(
                model, tokenizer, inputs, device, **generation_kwargs
            )
        
        # Decode response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Return only the new part (remove prompt)
        return generated_text[len(prompt):].strip()


def load_model_safe(model_name: str = "gpt2", prefer_mps: bool = True, verbose: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """Convenience function for safe model loading"""
    loader = ModelLoader(model_name=model_name, prefer_mps=prefer_mps, verbose=verbose)
    return loader.load_model_and_tokenizer()


class ModelCache:
    """Simple model caching to avoid reloading"""
    
    _cache: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer, str]] = {}
    
    @classmethod
    def get_model(cls, model_name: str, prefer_mps: bool = True, verbose: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
        """Get model from cache or load if not cached"""
        
        cache_key = f"{model_name}_{prefer_mps}"
        
        if cache_key not in cls._cache:
            if verbose:
                print(f"📦 Loading {model_name} into cache...")
            
            model, tokenizer, device = load_model_safe(
                model_name=model_name, 
                prefer_mps=prefer_mps, 
                verbose=verbose
            )
            
            cls._cache[cache_key] = (model, tokenizer, device)
            
            if verbose:
                print(f"✅ Model cached successfully")
        else:
            if verbose:
                print(f"📦 Using cached model: {model_name}")
        
        return cls._cache[cache_key]
    
    @classmethod
    def clear_cache(cls):
        """Clear the model cache"""
        cls._cache.clear()
    
    @classmethod
    def cache_info(cls) -> Dict[str, Any]:
        """Get information about cached models"""
        return {
            "cached_models": list(cls._cache.keys()),
            "cache_size": len(cls._cache)
        }


def optimize_model_for_device(model: AutoModelForCausalLM, device: str) -> AutoModelForCausalLM:
    """Apply device-specific optimizations to the model"""
    
    if device == "mps":
        # MPS-specific optimizations
        model.eval()  # Ensure eval mode for stability
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
            
        # Apply half precision if supported (be careful with MPS)
        try:
            # Test if half precision works on MPS
            test_tensor = torch.randn(1, 1, device=device, dtype=torch.float16)
            test_result = test_tensor * 2
            
            # If successful, convert model to half precision
            model = model.half()
            print("✅ Applied half precision optimization for MPS")
            
        except Exception:
            print("⚠️  Half precision not supported on this MPS version, using float32")
    
    elif device == "cuda":
        # CUDA-specific optimizations
        model = model.half()  # Use half precision for CUDA
        print("✅ Applied half precision optimization for CUDA")
    
    elif device == "cpu":
        # CPU-specific optimizations
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
            
        print("✅ Applied CPU optimizations")
    
    return model 
"""
Device Management for Apple Silicon and Cross-Platform Compatibility
Handles MPS, CUDA, and CPU device selection with proper fallbacks
"""

import os
import platform
import torch
from typing import Optional, Dict, Any
import warnings


class DeviceManager:
    """Manages device selection and model placement for optimal performance"""
    
    def __init__(self, prefer_mps: bool = True, verbose: bool = True):
        self.prefer_mps = prefer_mps
        self.verbose = verbose
        self.device_info = self._detect_devices()
        self.optimal_device = self._select_optimal_device()
        
        # Configure environment for MPS stability
        self._configure_mps_environment()
        
    def _detect_devices(self) -> Dict[str, Any]:
        """Detect available devices and their capabilities"""
        info = {
            "platform": platform.system(),
            "platform_machine": platform.machine(),
            "is_apple_silicon": platform.machine() == "arm64" and platform.system() == "Darwin",
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "mps_built": hasattr(torch.backends, 'mps') and torch.backends.mps.is_built(),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__
        }
        
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        
        if self.verbose:
            self._log_device_info(info)
            
        return info
    
    def _log_device_info(self, info: Dict[str, Any]):
        """Log device detection information"""
        print("🔍 Device Detection:")
        print(f"  Platform: {info['platform']} ({info['platform_machine']})")
        print(f"  Apple Silicon: {info['is_apple_silicon']}")
        print(f"  PyTorch: {info['torch_version']}")
        
        if info["mps_available"]:
            print(f"  ✅ MPS Available: {info['mps_available']}")
            print(f"  ✅ MPS Built: {info['mps_built']}")
        elif info["is_apple_silicon"]:
            print("  ⚠️  MPS not available (requires PyTorch 1.12+ with MPS support)")
            
        if info["cuda_available"]:
            print(f"  ✅ CUDA Available: {info['cuda_device_count']} device(s)")
            if info.get("cuda_device_name"):
                print(f"    Device: {info['cuda_device_name']}")
        else:
            print("  ❌ CUDA not available")
            
        print(f"  CPU Cores: {info['cpu_count']}")
    
    def _configure_mps_environment(self):
        """Configure environment variables for MPS stability"""
        if self.device_info["is_apple_silicon"]:
            # Set environment variables for MPS stability
            env_vars = {
                'PYTORCH_ENABLE_MPS_FALLBACK': '1',
                'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0'
            }
            
            for key, value in env_vars.items():
                if key not in os.environ:
                    os.environ[key] = value
                    if self.verbose:
                        print(f"  🔧 Set {key}={value}")
    
    def _select_optimal_device(self) -> str:
        """Select the optimal device based on availability and preferences"""
        
        # If MPS is preferred and available
        if (self.prefer_mps and 
            self.device_info["mps_available"] and 
            self.device_info["mps_built"]):
            
            # Test MPS functionality
            if self._test_mps_functionality():
                return "mps"
            else:
                if self.verbose:
                    print("  ⚠️  MPS failed functionality test, falling back")
        
        # Fallback to CUDA if available
        if self.device_info["cuda_available"]:
            return "cuda"
        
        # Final fallback to CPU
        return "cpu"
    
    def _test_mps_functionality(self) -> bool:
        """Test basic MPS functionality to ensure it works"""
        try:
            # Create a simple tensor and operation on MPS
            x = torch.randn(10, 10, device="mps")
            y = torch.randn(10, 10, device="mps")
            z = torch.matmul(x, y)
            
            # Test tensor movement
            z_cpu = z.cpu()
            
            if self.verbose:
                print("  ✅ MPS functionality test passed")
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"  ❌ MPS functionality test failed: {e}")
            return False
    
    def get_device(self) -> str:
        """Get the optimal device string"""
        return self.optimal_device
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information"""
        return self.device_info.copy()
    
    def move_model_to_device(self, model, device: Optional[str] = None) -> torch.nn.Module:
        """Safely move model to device with error handling"""
        target_device = device or self.optimal_device
        
        try:
            if self.verbose:
                print(f"  📱 Moving model to {target_device}")
            
            model = model.to(target_device)
            
            # Test model on device
            self._test_model_on_device(model, target_device)
            
            if self.verbose:
                print(f"  ✅ Model successfully moved to {target_device}")
            
            return model
            
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Failed to move model to {target_device}: {e}")
            
            # Fallback to CPU
            if target_device != "cpu":
                if self.verbose:
                    print("  🔄 Falling back to CPU")
                return self.move_model_to_device(model, "cpu")
            else:
                raise RuntimeError(f"Failed to move model to CPU: {e}")
    
    def _test_model_on_device(self, model, device: str):
        """Test basic model functionality on device"""
        try:
            # Create a simple input tensor
            test_input = torch.randint(0, 1000, (1, 10), device=device)
            
            with torch.no_grad():
                # Test forward pass
                outputs = model(test_input)
                
                # Ensure outputs are on the correct device
                assert outputs.logits.device.type == device.split(':')[0]
                
        except Exception as e:
            raise RuntimeError(f"Model test failed on {device}: {e}")


def get_optimal_device(prefer_mps: bool = True, verbose: bool = True) -> str:
    """Convenience function to get optimal device"""
    manager = DeviceManager(prefer_mps=prefer_mps, verbose=verbose)
    return manager.get_device()


def setup_model_for_device(model, prefer_mps: bool = True, verbose: bool = True) -> tuple[torch.nn.Module, str]:
    """Setup model for optimal device with comprehensive error handling"""
    manager = DeviceManager(prefer_mps=prefer_mps, verbose=verbose)
    
    device = manager.get_device()
    model = manager.move_model_to_device(model)
    
    return model, device


class SafeGeneration:
    """Safe text generation utilities for Apple Silicon"""
    
    @staticmethod
    def generate_with_fallback(model, tokenizer, input_ids, device: str, **generation_kwargs):
        """Generate text with multiple fallback strategies"""
        
        # Handle None device by detecting from model
        if device is None:
            device = next(model.parameters()).device.type
        
        # Ensure input_ids are properly on device
        if hasattr(input_ids, 'to'):
            input_ids = input_ids.to(device)
        
        # Remove problematic arguments for Apple Silicon
        safe_kwargs = generation_kwargs.copy()
        
        # Remove arguments that can cause issues on MPS
        problematic_args = ['early_stopping', 'num_beams']
        for arg in problematic_args:
            safe_kwargs.pop(arg, None)
        
        # Ensure temperature and top_p are only used with do_sample=True
        if not safe_kwargs.get('do_sample', True):
            safe_kwargs.pop('temperature', None)
            safe_kwargs.pop('top_p', None)
        
        # Try manual generation first for MPS (more stable)
        if device == "mps":
            try:
                return SafeGeneration._generate_manual(
                    model, tokenizer, input_ids, device, **safe_kwargs
                )
            except Exception as manual_error:
                print(f"⚠️  Manual generation failed on {device}: {manual_error}")
                print("🔄 Falling back to CPU...")
                return SafeGeneration._fallback_to_cpu(
                    model, tokenizer, input_ids, **safe_kwargs
                )
        
        # For non-MPS devices, try model.generate first
        try:
            # First attempt: Use model.generate with safe parameters
            with torch.no_grad():
                outputs = model.generate(input_ids, **safe_kwargs)
            return outputs
            
        except Exception as e:
            print(f"⚠️  model.generate failed on {device}: {e}")
            print("🔄 Falling back to manual generation...")
            
            # Fallback: Manual token-by-token generation
            return SafeGeneration._generate_manual(
                model, tokenizer, input_ids, device, **safe_kwargs
            )
    
    @staticmethod
    def _generate_manual(model, tokenizer, input_ids, device: str, **kwargs):
        """Manual token-by-token generation as fallback"""
        max_new_tokens = kwargs.get('max_new_tokens', kwargs.get('max_length', 100) - input_ids.shape[1])
        temperature = kwargs.get('temperature', 1.0)
        do_sample = kwargs.get('do_sample', True)
        pad_token_id = kwargs.get('pad_token_id', tokenizer.eos_token_id)
        
        # Ensure input is on correct device and has proper shape
        if hasattr(input_ids, 'to'):
            input_ids = input_ids.to(device)
        
        # Create attention mask for MPS stability
        attention_mask = torch.ones_like(input_ids, device=device)
        
        generated_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass with attention mask
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=current_attention_mask,
                    use_cache=False  # Disable cache for MPS stability
                )
                logits = outputs.logits[:, -1, :]
                
                if do_sample and temperature > 0:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Stop if we hit the end token
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                # Append new token and update attention mask
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                new_attention = torch.ones((1, 1), device=device, dtype=current_attention_mask.dtype)
                current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=-1)
        
        return generated_ids
    
    @staticmethod
    def _fallback_to_cpu(model, tokenizer, input_ids, **kwargs):
        """Fallback to CPU generation when device fails"""
        # Move everything to CPU
        cpu_model = model.cpu()
        cpu_input_ids = input_ids.cpu()
        
        # Generate on CPU
        max_new_tokens = kwargs.get('max_new_tokens', kwargs.get('max_length', 100) - cpu_input_ids.shape[1])
        temperature = kwargs.get('temperature', 1.0)
        do_sample = kwargs.get('do_sample', True)
        pad_token_id = kwargs.get('pad_token_id', tokenizer.eos_token_id)
        
        try:
            with torch.no_grad():
                outputs = cpu_model.generate(
                    cpu_input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=pad_token_id,
                    use_cache=False
                )
            
            # Move model back to original device
            original_device = next(model.parameters()).device
            model.to(original_device)
            
            # Move outputs to original device
            return outputs.to(original_device)
            
        except Exception as e:
            print(f"⚠️  CPU fallback also failed: {e}")
            # Return original input as last resort
            return cpu_input_ids


# Global flag to track if optimizations have been applied
_apple_silicon_optimized = False

def optimize_for_apple_silicon():
    """Apply global optimizations for Apple Silicon"""
    global _apple_silicon_optimized
    
    if _apple_silicon_optimized:
        return True
    
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        # Disable MPS fallback warnings
        warnings.filterwarnings("ignore", message=".*MPS.*")
        
        # Set optimal environment variables
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
        
        # Optimize for memory usage
        if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'manual_seed'):
            torch.backends.mps.manual_seed(42)
        
        print("🍎 Applied Apple Silicon optimizations")
        _apple_silicon_optimized = True
        
        return True
    
    return False 
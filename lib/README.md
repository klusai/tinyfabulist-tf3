# TinyFabulist Library

A modular library for GPT model evaluation with Apple Silicon (MPS) optimization and cross-platform device management.

## Features

- **Automatic Device Detection**: Intelligently detects and uses optimal compute device (MPS > CUDA > CPU)
- **Apple Silicon Optimization**: Native MPS support with stability optimizations and CPU fallbacks
- **Safe Model Loading**: Robust model loading with error handling and device management
- **Model Caching**: Efficient model caching to avoid reloading across evaluations
- **Dataset Utilities**: Resilient dataset loading with multiple fallback strategies
- **Cross-Platform**: Works on Apple Silicon, CUDA, and CPU systems

## Quick Start

```python
from lib import ModelCache, get_optimal_device

# Automatic device detection
device = get_optimal_device()
print(f"Using device: {device}")  # Will show 'mps' on Apple Silicon

# Load model with caching
model, tokenizer, device = ModelCache.get_model("gpt2")
print(f"Model loaded on: {device}")
```

## Modules

### Device Manager (`device_manager.py`)

Handles device detection and model placement with Apple Silicon optimizations.

```python
from lib import DeviceManager, SafeGeneration

# Create device manager
manager = DeviceManager(prefer_mps=True, verbose=True)

# Get device info
device_info = manager.get_device_info()
print(f"Apple Silicon: {device_info['is_apple_silicon']}")
print(f"MPS Available: {device_info['mps_available']}")

# Safe text generation (handles MPS issues)
outputs = SafeGeneration.generate_with_fallback(
    model, tokenizer, input_ids, device,
    max_new_tokens=100, temperature=0.7
)
```

### Model Loader (`model_loader.py`)

Provides safe model loading with device optimization and caching.

```python
from lib import ModelLoader, ModelCache

# Load single model
loader = ModelLoader("gpt2", prefer_mps=True)
model, tokenizer, device = loader.load_model_and_tokenizer()

# Use model caching (recommended)
model, tokenizer, device = ModelCache.get_model("gpt2")

# Generate text safely
completion = loader.generate_text(
    model, tokenizer, "Once upon a time", device,
    max_new_tokens=50
)
```

### Dataset Utilities (`dataset_utils.py`)

Resilient dataset loading with fallback strategies.

```python
from lib import DatasetLoader, create_fable_test_data

# Load dataset with fallbacks
loader = DatasetLoader(verbose=True)
dataset = loader.load_fable_dataset(
    "roneneldan/TinyStories",
    split="train",
    max_samples=1000
)

# Create synthetic test data
test_data = create_fable_test_data(num_samples=10)
```

## Apple Silicon Support

The library includes comprehensive Apple Silicon support:

### Automatic MPS Detection
- Detects Apple Silicon hardware
- Tests MPS functionality before use
- Applies optimal environment variables

### MPS Optimizations
- Configures `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- Disables problematic generation parameters
- Uses attention masks for stability

### Fallback Strategies
- CPU fallback when MPS fails
- Manual token-by-token generation
- Graceful error handling

## Device Detection Results

```
🔍 Device Detection:
  Platform: Darwin (arm64)
  Apple Silicon: True
  PyTorch: 2.7.1
  ✅ MPS Available: True
  ✅ MPS Built: True
  ❌ CUDA not available
  CPU Cores: 8
  ✅ MPS functionality test passed
```

## Usage with TinyFabulist CLI

The library integrates seamlessly with the evaluation framework:

```bash
# Uses automatic MPS detection
python tf3.py comprehensive --model gpt2

# Explicit device specification
python tf3.py comprehensive --model gpt2 --device mps

# Test functionality
python test_lib.py
```

## Known Limitations

- **MPS Text Generation**: Some PyTorch versions have issues with `model.generate()` on MPS
- **Workaround**: Library includes CPU fallback and manual generation methods
- **Performance**: MPS inference works perfectly; generation may fall back to CPU

## Compatibility

- **Python**: 3.8+
- **PyTorch**: 1.12+ (for MPS support)
- **Platforms**: macOS (Apple Silicon), Linux, Windows
- **Devices**: MPS, CUDA, CPU

## Testing

Run the test script to verify functionality:

```bash
python test_lib.py
```

Expected output shows successful device detection, model loading, and basic inference on the optimal device.

## Architecture

```
lib/
├── __init__.py           # Package exports
├── device_manager.py     # Device detection and MPS handling
├── model_loader.py       # Safe model loading and caching
└── dataset_utils.py      # Dataset loading with fallbacks
```

The library is designed to be:
- **Modular**: Each component can be used independently
- **Robust**: Comprehensive error handling and fallbacks
- **Efficient**: Model caching and optimal device usage
- **Cross-platform**: Works across different hardware configurations 
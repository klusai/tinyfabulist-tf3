# GPT-2 Text Generation

A comprehensive implementation for using the small GPT-2 model from Hugging Face.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Interactive Mode)
```bash
python gpt2.py
```

### Command Line Options

#### Basic Generation Example
```bash
python gpt2.py --mode basic
```

#### Pipeline Example
```bash
python gpt2.py --mode pipeline
```

#### Temperature Comparison
```bash
python gpt2.py --mode temperature
```

#### Minimal Test (Start Here if Having Issues)
```bash
python gpt2.py --mode minimal
```

#### Test Mode (Troubleshooting)
```bash
python gpt2.py --mode test
```

#### Single Text Generation
```bash
python gpt2.py --prompt "The future of AI is" --max-length 150 --temperature 0.8
```

#### Use Different Model Sizes
```bash
python gpt2.py --model gpt2-medium --prompt "Hello world"
```

### Available Models
- `gpt2` - Small (124M parameters) - Default
- `gpt2-medium` - Medium (355M parameters)
- `gpt2-large` - Large (774M parameters)
- `gpt2-xl` - Extra Large (1.5B parameters)

### Parameters
- `--mode`: Generation mode (basic, pipeline, interactive, temperature, all, test, minimal)
- `--model`: GPT-2 model variant
- `--prompt`: Text prompt for generation
- `--max-length`: Maximum generation length (default: 100)
- `--temperature`: Sampling temperature (default: 0.7)

## Features

- **Interactive Mode**: Chat-like interface for continuous text generation
- **Multiple Examples**: Basic generation, pipeline usage, and temperature comparison
- **Model Flexibility**: Support for all GPT-2 model sizes
- **Configurable Parameters**: Control generation length, temperature, and other settings
- **Error Handling**: Graceful handling of interruptions and errors
- **📊 Experiment Tracking System**: Complete research framework for systematic experiment tracking and paper development
- **🔬 Evaluation Framework**: Comprehensive evaluation suite with perplexity, BLEU, fluency, and narrative structure metrics
- **🍎 Apple Silicon Optimization**: Native MPS support with intelligent device detection and CPU fallbacks

## Example Usage in Code

```python
from gpt2 import GPT2Generator

# Initialize generator
generator = GPT2Generator()

# Generate text
result = generator.generate_text("Once upon a time", max_length=100)
print(result[0])
```

## 🔬 Research & Experiment Framework

This project includes a **production-ready research framework** for systematic experiment tracking, evaluation, and paper development with comprehensive Apple Silicon optimization.

### Quick Start
```bash
# Run the evaluation framework
python tinyfabulist.py comprehensive --model gpt2

# Demo the complete experiment tracking system
python demo_experiment_system.py
```

## 🏗️ System Architecture

### Core Framework (`lib/`)
- **`device_manager.py`**: Intelligent device detection and MPS optimization for Apple Silicon
- **`model_loader.py`**: Safe model loading with caching and cross-platform compatibility  
- **`dataset_utils.py`**: Robust dataset loading with multiple fallback strategies

### Evaluation Framework (`evals/`)
- **`base.py`**: Abstract base classes and shared utilities
- **`perplexity.py`**: Language modeling quality metrics
- **`text_quality.py`**: BLEU, ROUGE, and BERTScore evaluation
- **`fluency.py`**: Repetition, diversity, and coherence analysis
- **`fable_structure.py`**: Narrative structure evaluation for fables
- **`comprehensive.py`**: Combined evaluation with weighted scoring

### Experiment Tracking System (`experiments/`)
- **`experiment_manager.py`**: Core experiment tracking with automatic metadata capture
- **`run_experiments.py`**: Systematic experiment runner for different study types
- **`analysis_tools.py`**: Analysis and visualization tools for results

## 📊 Experiment Tracking Capabilities

### Automatic Tracking
- ✅ **Unique Experiment IDs**: Hash-based IDs with timestamps
- ✅ **Environment Capture**: Hardware, PyTorch version, git state
- ✅ **Configuration Storage**: Complete hyperparameter recording
- ✅ **Result Archival**: JSON format with detailed metrics
- ✅ **Status Management**: Running/completed/failed experiment states

### Study Types Implemented
- ✅ **Baseline Comparison**: GPT-2 model variants (small, medium, large)
- ✅ **Temperature Studies**: Parameter sensitivity analysis (0.3-1.2)
- ✅ **Sample Size Analysis**: Convergence studies (10, 25, 50, 100 samples)
- ✅ **Evaluator Comparison**: Individual metric performance analysis
- ✅ **Fine-tuned Model Evaluation**: Framework for custom model comparison

### Analysis & Export
- ✅ **Summary Tables**: CSV export for spreadsheet analysis
- ✅ **LaTeX Tables**: Publication-ready formatted tables
- ✅ **Comprehensive Reports**: Statistical summaries and interpretations
- ✅ **Result Comparison**: Multi-experiment analysis tools

## 🔬 Evaluation Metrics

### Language Modeling Quality
- ✅ **Perplexity**: Standard and weighted perplexity calculation
- ✅ **Bits per Character**: Information-theoretic quality measure
- ✅ **Cross-entropy Loss**: Model confidence assessment

### Text Quality
- ✅ **BLEU Scores**: N-gram overlap metrics (BLEU-1 through BLEU-4)
- ✅ **ROUGE Scores**: Recall-oriented quality measures
- ✅ **BERTScore**: Semantic similarity using BERT embeddings

### Fluency Analysis
- ✅ **Repetition Detection**: Word and phrase repetition analysis
- ✅ **Diversity Metrics**: Type-token ratios and n-gram diversity
- ✅ **Coherence Scoring**: Sentence-level coherence analysis

### Narrative Structure (Fable-Specific)
- ✅ **Story Elements**: Character, setting, moral detection
- ✅ **Narrative Arc**: Beginning, middle, end structure analysis
- ✅ **Moral Coherence**: Moral lesson consistency evaluation

## 🍎 Apple Silicon Optimization

### Device Management
- ✅ **Automatic MPS Detection**: Intelligent device selection (MPS > CUDA > CPU)
- ✅ **Functionality Testing**: Pre-use MPS capability verification
- ✅ **Environment Configuration**: Optimal MPS settings automatic setup
- ✅ **Graceful Fallbacks**: CPU fallback when MPS operations fail

### Performance Optimizations
- ✅ **Model Caching**: Efficient model reuse across evaluations
- ✅ **Memory Management**: Optimized tensor operations for Apple Silicon
- ✅ **Safe Generation**: Robust text generation with MPS compatibility
- ✅ **Error Handling**: Comprehensive fallback strategies

## 📋 CLI Tools & Workflows

### TinyFabulist CLI (`tinyfabulist.py`)
```bash
# Single evaluator runs
python tinyfabulist.py single --evaluator perplexity --model gpt2

# Comprehensive evaluation
python tinyfabulist.py comprehensive --model gpt2 --num-samples 100

# Model comparison
python tinyfabulist.py compare --models gpt2 gpt2-medium

# Quick testing
python tinyfabulist.py test
```

### Experiment Management
```bash
cd experiments

# Run systematic studies
python run_experiments.py --baseline      # Model comparison
python run_experiments.py --temperature   # Parameter study
python run_experiments.py --all          # Complete study

# Manage experiments
python experiment_manager.py list
python experiment_manager.py compare exp1_id exp2_id
python experiment_manager.py export exp1_id exp2_id --output results.csv

# Analysis tools
python analysis_tools.py summary exp1_id exp2_id
python analysis_tools.py latex exp1_id exp2_id --output table.tex
python analysis_tools.py report exp1_id exp2_id
```

## 🧪 Demo & Testing

### Comprehensive Demo
```bash
python demo_experiment_system.py
```
- ✅ **System Status Check**: Verifies all components working
- ✅ **Live Experiment Run**: Demonstrates full tracking workflow
- ✅ **Analysis Pipeline**: Shows result processing and export
- ✅ **Paper-Ready Outputs**: Generates publication materials

### Test Results (Apple Silicon M1)
- ✅ **MPS Detection**: Successfully detects and uses Apple Silicon GPU
- ✅ **Model Loading**: GPT-2 (124M parameters) loads on MPS device
- ✅ **Evaluation Execution**: All metric categories function correctly
- ✅ **Result Storage**: Complete metadata and results archived
- ✅ **Export Generation**: CSV, LaTeX, and report outputs created

## 📈 Paper Development Workflow

### 1. Systematic Experiments
- Run baseline comparisons between model variants
- Conduct parameter sensitivity studies
- Evaluate fine-tuned models against baselines
- Generate reproducible results with complete metadata

### 2. Result Analysis
- Compare performance across experiments
- Generate statistical summaries
- Create publication-ready tables and figures
- Perform significance testing

### 3. Paper Integration
- **Methodology Section**: Complete environment and configuration details
- **Results Section**: Automated table generation with LaTeX formatting
- **Reproducibility**: Git commit tracking and environment capture
- **Supplementary Material**: Raw data export for transparency

## 🎯 Production Readiness

### Code Quality
- ✅ **Modular Design**: Clean separation of concerns across components
- ✅ **Error Handling**: Comprehensive exception handling and fallbacks
- ✅ **Documentation**: Extensive docstrings and README files
- ✅ **Type Hints**: Full type annotation for better code maintainability

### Cross-Platform Compatibility
- ✅ **Apple Silicon (MPS)**: Native support with optimizations
- ✅ **CUDA**: GPU acceleration on NVIDIA hardware
- ✅ **CPU**: Universal fallback for any system
- ✅ **Memory Management**: Efficient usage across different hardware

### Research Workflow Integration
- ✅ **Version Control**: Git integration for code versioning
- ✅ **Reproducibility**: Complete state capture for experiment replication
- ✅ **Scalability**: Supports both quick demos and large-scale studies
- ✅ **Extensibility**: Easy addition of new metrics and experiment types

## 🚀 Next Steps & Usage

### For Research Paper Development
1. **Start with Demo**: Run `python demo_experiment_system.py` to verify setup
2. **Baseline Studies**: Execute `python experiments/run_experiments.py --baseline`
3. **Parameter Analysis**: Run temperature and sample size studies
4. **Fine-tuned Comparison**: Add your model paths and compare against baselines
5. **Paper Generation**: Export results and generate LaTeX tables

### For Extension
- **New Metrics**: Add evaluators to `evals/` directory
- **Custom Experiments**: Modify `run_experiments.py` for specific studies
- **Additional Models**: System supports any Hugging Face compatible model
- **Dataset Integration**: Add new datasets through `lib/dataset_utils.py`

## 📚 Documentation

- **`experiments/README.md`**: Complete experiment system documentation
- **Code Documentation**: Comprehensive docstrings throughout codebase
- **Examples**: Working examples in demo scripts and CLI help

---

**Status: ✅ FULLY IMPLEMENTED AND TESTED**

The TinyFabulist project provides a complete, production-ready research framework for systematic evaluation and comparison of language models on fable completion tasks, with particular optimization for Apple Silicon hardware and comprehensive experiment tracking for academic paper development.

## Troubleshooting

### Bus Error / Crashes (Apple Silicon Macs)
If you encounter a bus error or crashes (especially on Apple Silicon Macs):

1. **Start with minimal test:**
   ```bash
   python gpt2.py --mode minimal
   ```

2. **If minimal test passes, try full test:**
   ```bash
   python gpt2.py --mode test
   ```

3. **The script includes these Apple Silicon fixes:**
   - Forces CPU usage (no MPS/GPU)
   - Manual token-by-token generation (avoids `model.generate()`)
   - Environment variables to disable problematic PyTorch features
   - Comprehensive error handling with fallbacks

4. **If still having issues, try:**
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
   python gpt2.py --mode minimal
   ```

### Memory Issues
- Use the small `gpt2` model instead of larger variants
- Reduce `--max-length` parameter
- Close other applications to free up memory

### Common Issues
- **"No module named 'torch'"**: Run `pip install -r requirements.txt`
- **Warnings about attention mask**: These are handled automatically in the updated version
- **Generation fails**: The script includes fallback mechanisms for robust operation 
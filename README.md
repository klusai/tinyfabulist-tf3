# TinyFabulist Paper Series: TF3
## Evaluation and Fine-tuning Framework based on GPT-2 for Fable Completion Models

---

A comprehensive evaluation framework for systematic evaluation and comparison of text generation models on moral fable completion tasks, with complete research experiment tracking and Apple Silicon optimization.

## 🏗️ System Architecture

### Core Framework (`lib/`)
- **`device_manager.py`**: Intelligent device detection and MPS optimization for Apple Silicon (M1/M2/M3)
- **`model_loader.py`**: Safe model loading with caching and cross-platform compatibility  
- **`dataset_utils.py`**: Robust dataset loading with multiple fallback strategies
- **`data_loading.py`**: Advanced data loading utilities with synthetic data generation
- **`logging_utils.py`**: Comprehensive logging system with structured output

### Evaluation Framework (`evals/`)
- **`base.py`**: Abstract base classes and shared evaluation utilities
- **`perplexity.py`**: Language modeling quality metrics (perplexity, bits per character)
- **`text_quality.py`**: BLEU, ROUGE, and BERTScore evaluation
- **`fluency.py`**: Repetition, diversity, and coherence analysis
- **`fable_structure.py`**: Narrative structure evaluation for fables
- **`semantic_coherence.py`**: Advanced semantic coherence analysis
- **`comprehensive.py`**: Combined evaluation with weighted scoring

### Experiment Tracking System (`experiments/`)
- **`experiment_manager.py`**: Core experiment tracking with automatic metadata capture
- **`run_experiments.py`**: Systematic experiment runner for different study types
- **`analysis_tools.py`**: Analysis and visualization tools for results

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd tf3

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Quick Test (Start Here)
```bash
python tf3.py test
```

#### Comprehensive Evaluation
```bash
python tf3.py comprehensive --model gpt2 --num-samples 100
```

#### Single Evaluator
```bash
python tf3.py single --evaluator perplexity --model gpt2
```

#### Model Comparison
```bash
python tf3.py compare --models gpt2 gpt2-medium --num-samples 50
```

#### List Available Evaluators
```bash
python tf3.py list
```

## 📊 Evaluation Metrics

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

### Semantic Coherence
- ✅ **Entity Consistency**: Character and setting consistency tracking
- ✅ **Thematic Coherence**: Topic coherence analysis
- ✅ **Logical Flow**: Discourse coherence evaluation

### Narrative Structure (Fable-Specific)
- ✅ **Story Elements**: Character, setting, moral detection
- ✅ **Narrative Arc**: Beginning, middle, end structure analysis
- ✅ **Moral Coherence**: Moral lesson consistency evaluation

## 🔬 Research & Experiment Framework

This project includes a **production-ready research framework** for systematic experiment tracking, evaluation, and paper development with comprehensive Apple Silicon optimization.

### Demo the Complete System
```bash
python demo_experiment_system.py
```

### Systematic Experiments
```bash
cd experiments

# Run baseline comparison between model variants
python run_experiments.py --baseline

# Conduct temperature parameter studies
python run_experiments.py --temperature

# Run all systematic studies
python run_experiments.py --all

# Quick demo experiment
python run_experiments.py --demo
```

### Experiment Management
```bash
cd experiments

# List all experiments
python experiment_manager.py list

# Compare experiments
python experiment_manager.py compare exp1_id exp2_id

# Export results to CSV
python experiment_manager.py export exp1_id exp2_id --output results.csv

# Generate analysis reports
python analysis_tools.py summary exp1_id exp2_id
python analysis_tools.py latex exp1_id exp2_id --output table.tex
python analysis_tools.py report exp1_id exp2_id
```

## 📋 CLI Options & Parameters

### Available Models
- `gpt2` - Small (124M parameters) - Default
- `gpt2-medium` - Medium (355M parameters)
- `gpt2-large` - Large (774M parameters)
- `gpt2-xl` - Extra Large (1.5B parameters)
- Path to fine-tuned model (supports PEFT/LoRA adapters)

### Key Parameters
- `--model`: Model name or path to evaluate
- `--dataset`: Dataset to evaluate on (default: klusai/ds-tf1-en-3m)
- `--num-samples`: Number of samples to evaluate (default: 100)
- `--max-length`: Maximum sequence length (default: 512)
- `--temperature`: Generation temperature (default: 0.8)
- `--device`: Device to use (auto-detects optimal device if not specified)
- `--output-dir`: Directory to save results
- `--evaluators`: Specific evaluators to run (for comprehensive mode)

### Logging Options
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--log-file`: Save logs to file
- `--hide-generated-text`: Don't show generated text samples in logs
- `--quiet`: Reduce output verbosity

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

## 🚀 Advanced Usage Examples

### Fine-tuned Model Evaluation
```bash
# Evaluate PEFT/LoRA adapter
python tf3.py comprehensive --model ./path/to/peft-adapter

# Compare fine-tuned vs base model
python tf3.py compare --models gpt2 ./path/to/finetuned-model
```

### Custom Dataset Evaluation
```bash
# Use different dataset
python tf3.py comprehensive --dataset your-dataset-name --split validation

# Specify dataset parameters
python tf3.py comprehensive --num-samples 200 --max-length 1024
```

### Research Experiment Workflow
```bash
# Complete research workflow
cd experiments

# 1. Run systematic studies
python run_experiments.py --all

# 2. List completed experiments  
python experiment_manager.py list --status completed

# 3. Generate comparison tables
python analysis_tools.py latex exp1 exp2 exp3 --output comparison_table.tex

# 4. Export all data
python experiment_manager.py export exp1 exp2 exp3 --output final_results.csv
```

## 📚 Documentation

- **`experiments/README.md`**: Complete experiment system documentation
- **`lib/README.md`**: Core framework documentation
- **`evals/README.md`**: Evaluation framework documentation
- **Code Documentation**: Comprehensive docstrings throughout codebase
- **Examples**: Working examples in demo scripts and CLI help

## 🔧 Troubleshooting

### Apple Silicon Issues
If you encounter crashes or bus errors on Apple Silicon Macs:

1. **Test with minimal configuration:**
   ```bash
   python tf3.py test
   ```

2. **Check MPS availability:**
   ```bash
   python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
   ```

3. **Force CPU usage if needed:**
   ```bash
   python tf3.py comprehensive --device cpu --num-samples 10
   ```

### Memory Issues
- Use smaller models (`gpt2` instead of `gpt2-large`)
- Reduce `--num-samples` and `--max-length` parameters
- Close other applications to free up memory

### Dataset Loading Issues
The framework includes automatic fallback to synthetic data generation if dataset loading fails.

### Common Issues
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Model loading errors**: Check available models with `python tf3.py list`
- **CUDA/MPS errors**: The system automatically falls back to CPU if GPU fails

## 🚀 Next Steps

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

---

**Status: ✅ FULLY IMPLEMENTED AND TESTED**

The TinyFabulist project provides a complete, production-ready research framework for systematic evaluation and comparison of language models on fable completion tasks, with particular optimization for Apple Silicon hardware and comprehensive experiment tracking for academic paper development. 

# TinyFabulist Paper Series: TF3
## Evaluation and Fine-tuning Framework based on GPT-2 for Fable Completion Models

A comprehensive evaluation framework for systematic evaluation and comparison of text generation models on moral fable completion tasks, with standardized evaluation protocols, statistical analysis, and Apple Silicon optimization.

## 🏗️ System Architecture

### Core Framework (`lib/`)
- **`device_manager.py`**: Intelligent device detection and MPS optimization for Apple Silicon (M1/M2/M3)
- **`model_loader.py`**: Safe model loading with caching and cross-platform compatibility  
- **`dataset_utils.py`**: Robust dataset loading with multiple fallback strategies
- **`data_loading.py`**: Advanced data loading utilities with synthetic data generation
- **`logging_utils.py`**: Comprehensive logging system with structured output

### Evaluation Framework (`evals/`)
- **`base.py`**: Abstract base classes with standardized evaluation protocols and statistical analysis
- **`perplexity.py`**: Language modeling quality metrics (perplexity, bits per character)
- **`text_quality.py`**: BLEU, ROUGE, and BERTScore evaluation with third-party library integration
- **`fluency.py`**: Repetition, diversity, and coherence analysis using NLTK
- **`fable_structure.py`**: Narrative structure evaluation for fables using transformer models
- **`semantic_coherence.py`**: Advanced semantic coherence analysis with scikit-learn integration
- **`comprehensive.py`**: Combined evaluation with weighted scoring and statistical reporting

### Experiment Tracking System (`experiments/`)
- **`experiment_manager.py`**: Core experiment tracking with automatic metadata capture
- **`run_experiments.py`**: Systematic experiment runner for different study types
- **`analysis_tools.py`**: Analysis and visualization tools for results

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/klusai/tinyfabulist-tf3
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

#### Comprehensive Evaluation with Standardized Settings
```bash
python tf3.py comprehensive --model gpt2 --num-samples 100 \
  --max-prompt-tokens 256 --max-new-tokens 256
```

#### Statistical Evaluation with Multiple Runs
```bash
python tf3.py comprehensive --model gpt2 --num-samples 50 \
  --num-runs 3 --confidence-level 0.95 --length-normalize
```

#### Single Evaluator
```bash
python tf3.py single --evaluator perplexity --model gpt2
```

#### Model Comparison with Standardized Protocols
```bash
python tf3.py compare --models gpt2 gpt2-medium --num-samples 50 \
  --max-prompt-tokens 256 --prompt-split-ratio 0.6
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

### Text Quality (Third-party Library Integration)
- ✅ **BLEU Scores**: N-gram overlap metrics using `evaluate` library
- ✅ **ROUGE Scores**: Recall-oriented quality measures using `evaluate` library
- ✅ **BERTScore**: Semantic similarity using `evaluate` library with BERT embeddings

### Fluency Analysis (NLTK Integration)
- ✅ **Repetition Detection**: Word and phrase repetition analysis
- ✅ **Diversity Metrics**: Type-token ratios and n-gram diversity using NLTK
- ✅ **Coherence Scoring**: Sentence-level coherence analysis
- ✅ **Grammar Checking**: Optional LanguageTool integration (requires Java)

### Semantic Coherence (Scikit-learn Integration)
- ✅ **Topic Consistency**: Cosine similarity using scikit-learn
- ✅ **Content Appropriateness**: Zero-shot classification using transformers
- ✅ **Fable Relevance**: Domain-specific relevance scoring
- ✅ **Embedding Analysis**: Advanced semantic analysis

### Narrative Structure (Transformer-based)
- ✅ **Story Elements**: Character, setting, moral detection using BART-large-MNLI
- ✅ **Narrative Arc**: Beginning, middle, end structure analysis
- ✅ **Moral Coherence**: Moral lesson consistency evaluation

## 🔬 Standardized Evaluation Protocols

### Token-based Prompt Control
- ✅ **Fixed Token Limits**: Consistent prompt lengths across evaluations
- ✅ **Configurable Split Ratios**: Precise control over prompt vs reference text
- ✅ **Multiple Truncation Strategies**: Left, right, or middle truncation
- ✅ **Minimum Length Validation**: Ensures viable prompts and references

### Statistical Rigor
- ✅ **Multiple Evaluation Runs**: Statistical significance testing
- ✅ **Confidence Intervals**: 95% confidence intervals by default
- ✅ **Reproducible Seeds**: Fixed random seeds for each run
- ✅ **Coefficient of Variation**: Measurement consistency analysis

### Length Normalization
- ✅ **Length-aware Metrics**: Fair comparison across different generation lengths
- ✅ **Generation Statistics**: Detailed length analysis and reporting
- ✅ **Variance Analysis**: Length consistency measurement

## 📋 CLI Options & Parameters

### Available Models
- `gpt2` - Small (124M parameters) - Default
- `gpt2-medium` - Medium (355M parameters)
- `gpt2-large` - Large (774M parameters)
- `gpt2-xl` - Extra Large (1.5B parameters)
- Path to fine-tuned model (supports PEFT/LoRA adapters)

### Core Parameters
- `--model`: Model name or path to evaluate
- `--dataset`: Dataset to evaluate on (default: klusai/ds-tf1-en-3m)
- `--num-samples`: Number of samples to evaluate (default: 100)
- `--max-length`: Maximum sequence length (default: 512)
- `--temperature`: Generation temperature (default: 0.8)
- `--device`: Device to use (auto-detects optimal device if not specified)
- `--output-dir`: Directory to save results
- `--evaluators`: Specific evaluators to run (for comprehensive mode)

### Standardized Evaluation Parameters
- `--max-prompt-tokens`: Maximum tokens in prompt (default: 256)
- `--max-new-tokens`: Maximum new tokens to generate (default: 256)
- `--prompt-split-ratio`: Ratio of text to use as prompt vs reference (default: 0.6)
- `--truncation-strategy`: How to truncate long prompts (left/right/middle, default: right)
- `--num-runs`: Number of evaluation runs for statistical analysis (default: 1)
- `--confidence-level`: Confidence level for statistical analysis (default: 0.95)
- `--length-normalize`: Include length-normalized metrics

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

## 📊 Statistical Analysis Features

### Multiple Run Analysis
```bash
# Run 3 evaluations with different seeds for statistical significance
python tf3.py comprehensive --model gpt2 --num-runs 3 --confidence-level 0.95
```

### Comprehensive Statistical Reporting
- ✅ **Mean and Standard Deviation**: Central tendency and variability
- ✅ **Confidence Intervals**: Statistical significance assessment
- ✅ **Min/Max Values**: Range analysis across runs
- ✅ **Individual Run Tracking**: Complete transparency of results
- ✅ **Execution Time Analysis**: Performance consistency measurement

### Methodology Documentation
- ✅ **Complete Parameter Recording**: All evaluation settings documented
- ✅ **Seed Tracking**: Reproducibility information
- ✅ **Environment Capture**: Hardware and software configuration
- ✅ **Statistical Methodology**: Clear reporting of analysis methods

## 🚀 Advanced Usage Examples

### Standardized Research Evaluation
```bash
# Research-grade evaluation with statistical analysis
python tf3.py comprehensive --model gpt2 \
  --num-samples 100 \
  --max-prompt-tokens 256 \
  --max-new-tokens 256 \
  --prompt-split-ratio 0.6 \
  --num-runs 5 \
  --confidence-level 0.95 \
  --length-normalize \
  --output-dir results/
```

### Model Comparison with Statistical Rigor
```bash
# Compare models with standardized protocols
python tf3.py compare \
  --models gpt2 gpt2-medium \
  --num-samples 50 \
  --max-prompt-tokens 256 \
  --num-runs 3 \
  --truncation-strategy right \
  --output-dir comparison/
```

### Fine-tuned Model Evaluation
```bash
# Evaluate PEFT/LoRA adapter with standardized settings
python tf3.py comprehensive \
  --model ./path/to/peft-adapter \
  --max-prompt-tokens 256 \
  --prompt-split-ratio 0.7 \
  --num-runs 3
```

### Custom Dataset Evaluation
```bash
# Use different dataset with standardized protocols
python tf3.py comprehensive \
  --dataset your-dataset-name \
  --split validation \
  --max-prompt-tokens 128 \
  --max-new-tokens 128 \
  --length-normalize
```

## 📊 Third-party Library Integration

### Text Quality Metrics
- **`evaluate` library**: BLEU, ROUGE, BERTScore metrics
- **Automatic model downloads**: BART-large-MNLI, RoBERTa models
- **Graceful fallbacks**: Continues evaluation if specific metrics fail

### NLP Processing
- **NLTK**: Advanced text processing and fluency analysis
- **scikit-learn**: Cosine similarity and statistical analysis
- **transformers**: Zero-shot classification and embeddings

### Statistical Analysis
- **NumPy**: All statistical calculations (replacing custom implementations)
- **SciPy**: Confidence interval calculations (with fallback)
- **Optional dependencies**: Graceful degradation when libraries unavailable

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
- Reduce `--num-samples`, `--max-prompt-tokens`, and `--max-new-tokens` parameters
- Close other applications to free up memory

### Third-party Library Issues
- **Missing Java for LanguageTool**: Grammar checking automatically disabled
- **Missing scipy**: Confidence intervals use approximate fallback
- **Model download failures**: Evaluation continues with available metrics

### Dataset Loading Issues
The framework includes automatic fallback to synthetic data generation if dataset loading fails.

## 📈 Research Workflow Integration

### Reproducible Evaluation
```bash
# Generate reproducible results with fixed parameters
python tf3.py comprehensive \
  --model gpt2 \
  --seed 42 \
  --max-prompt-tokens 256 \
  --prompt-split-ratio 0.6 \
  --num-runs 5 \
  --output-dir reproducible_results/
```

### Statistical Significance Testing
```bash
# Compare models with statistical rigor
python tf3.py compare \
  --models model1 model2 model3 \
  --num-runs 10 \
  --confidence-level 0.99 \
  --output-dir statistical_comparison/
```

### Publication-Ready Results
- **Complete methodology documentation** in results
- **Statistical analysis** with confidence intervals
- **Reproducible evaluation protocols**
- **Standardized metrics** for fair comparison

## 🚀 Next Steps

### For Research Paper Development
1. **Start with Demo**: Run `python tf3.py test` to verify setup
2. **Standardized Evaluation**: Use consistent parameters across all experiments
3. **Statistical Analysis**: Run multiple evaluations for significance testing
4. **Model Comparison**: Compare models with identical evaluation protocols
5. **Results Export**: Generate publication-ready statistical reports

### For Extension
- **New Metrics**: Add evaluators to `evals/` directory following the base class pattern
- **Custom Statistical Analysis**: Extend statistical methods in `base.py`
- **Additional Models**: System supports any Hugging Face compatible model
- **Dataset Integration**: Add new datasets through standardized prompt preparation

---

**Status: ✅ FULLY IMPLEMENTED WITH STANDARDIZED PROTOCOLS**

The TinyFabulist project provides a complete, production-ready research framework for systematic evaluation and comparison of language models on fable completion tasks, with standardized evaluation protocols, statistical analysis, comprehensive third-party library integration, and particular optimization for Apple Silicon hardware. 

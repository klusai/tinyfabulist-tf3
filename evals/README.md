# TinyFabulist Evaluation Framework

A comprehensive, modular evaluation system for fable completion models using standard NLP metrics.

## 🎯 **Overview**

This framework provides both **CLI** and **programmatic** interfaces to evaluate text generation models on fable completion tasks using the [DS-TF1-EN-3M dataset](https://huggingface.co/datasets/klusai/ds-tf1-en-3m).

## 📊 **Available Metrics**

### **1. Perplexity Evaluator** (`perplexity`)
- **Weighted Perplexity**: Token-weighted language modeling quality
- **Bits per Character**: Information-theoretic measure
- **Usage**: Primary metric for language modeling quality

### **2. Text Quality Evaluator** (`text_quality`) 
- **BLEU Scores** (1-4): N-gram overlap with reference
- **ROUGE Scores**: Content overlap metrics  
- **BERTScore**: Semantic similarity using BERT embeddings
- **Usage**: Compare generated text quality against references

### **3. Fluency Evaluator** (`fluency`)
- **Type-Token Ratio (TTR)**: Vocabulary diversity
- **Repetition Metrics**: Detect repetitive text
- **Coherence Scores**: Sentence structure analysis
- **Usage**: Assess text fluency and naturalness

### **4. Fable Structure Evaluator** (`fable_structure`)
- **Narrative Elements**: Characters, settings, conflicts, resolutions
- **Moral Clarity**: Explicit/implicit moral lessons
- **Fable Style**: Anthropomorphism, simple language
- **Usage**: Evaluate fable-specific narrative quality

### **5. Comprehensive Evaluator** (`comprehensive`)
- **Overall Score**: Weighted combination of all metrics
- **Detailed Report**: Human-readable performance summary
- **Usage**: Complete model assessment

## 🚀 **Quick Start**

### **Installation**
```bash
pip install -r requirements_eval.txt
```

### **CLI Usage**

#### **1. Comprehensive Evaluation**
```bash
# Full evaluation with all metrics
python tinyfabulist.py comprehensive --model gpt2 --num-samples 100

# Specific evaluators only
python tinyfabulist.py comprehensive --evaluators perplexity fluency --model gpt2
```

#### **2. Single Evaluator**
```bash
# Run just perplexity evaluation
python tinyfabulist.py single --evaluator perplexity --model gpt2

# Run fable structure analysis
python tinyfabulist.py single --evaluator fable_structure --model gpt2
```

#### **3. Model Comparison**
```bash
# Compare multiple models
python tinyfabulist.py compare --models gpt2 gpt2-medium --num-samples 50

# Compare fine-tuned vs base model
python tinyfabulist.py compare --models gpt2 ./my-finetuned-gpt2
```

#### **4. Quick Test**
```bash
# Quick functionality test
python tinyfabulist.py test --model gpt2
```

### **Programmatic Usage**

#### **Single Evaluator**
```python
from evals import PerplexityEvaluator, EvaluationConfig
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# Setup
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = EvaluationConfig(
    model_name="gpt2",
    num_samples=100,
    device="cpu"
)

# Run evaluation
evaluator = PerplexityEvaluator(config)
dataset = load_dataset("klusai/ds-tf1-en-3m", split="test")
result = evaluator.run(model, tokenizer, dataset)

print(result.summary())
```

#### **Comprehensive Evaluation**
```python
from evals import ComprehensiveEvaluator, EvaluationConfig

config = EvaluationConfig(model_name="gpt2", num_samples=100)
evaluator = ComprehensiveEvaluator(config)

result = evaluator.run(model, tokenizer, dataset)
print(result.metadata["summary_report"])
```

#### **Factory Pattern**
```python
from evals import get_evaluator, EvaluationConfig

config = EvaluationConfig(model_name="gpt2")

# Get any evaluator by name
evaluator = get_evaluator("text_quality", config=config)
result = evaluator.run(model, tokenizer, dataset)
```

## 📁 **Project Structure**

```
├── evals/                          # Evaluation framework
│   ├── __init__.py                 # Package exports and registry
│   ├── base.py                     # Base classes and interfaces
│   ├── perplexity.py              # Perplexity metrics
│   ├── text_quality.py            # BLEU/ROUGE/BERTScore
│   ├── fluency.py                 # Diversity and repetition metrics
│   ├── fable_structure.py         # Fable-specific narrative metrics
│   └── comprehensive.py           # Combined evaluation
├── tinyfabulist.py                # Main CLI tool
├── examples.py                    # Usage examples
├── requirements_eval.txt          # Dependencies
└── EVALUATION_README.md           # This file
```

## 🎯 **Key Features**

### **Modular Design**
- Each evaluator is self-contained
- Consistent interface across all evaluators
- Easy to add new metrics

### **Flexible Usage**
- CLI for quick evaluations
- Programmatic API for integration
- Supports both base and fine-tuned models

### **Comprehensive Reporting**
- JSON output for machine processing
- Human-readable summary reports
- Per-sample detailed metrics

### **Apple Silicon Optimized**
- CPU-only operation for compatibility
- Proper error handling and fallbacks
- Memory-efficient evaluation

## 📊 **Output Examples**

### **CLI Output**
```
================================================================================
COMPREHENSIVE EVALUATION REPORT
================================================================================
Model: gpt2
Dataset: klusai/ds-tf1-en-3m
Samples: 100
Temperature: 0.8

OVERALL PERFORMANCE
----------------------------------------
Overall Score: 0.6234

PERPLEXITY METRICS
----------------------------------------
Weighted Perplexity: 45.67
Average Perplexity: 48.23
Bits per Character: 3.142

TEXT QUALITY METRICS
----------------------------------------
BLEU-4: 0.1234
ROUGE-L: 0.2345
BERTScore F1: 0.7890
Completion Length: 87.3 words

PERFORMANCE INTERPRETATION
----------------------------------------
Overall Rating: Good - Model shows solid fable completion capabilities

RECOMMENDATIONS
----------------------------------------
• Model performs well across all metrics
```

### **JSON Output Structure**
```json
{
  "evaluator_name": "Comprehensive",
  "metrics": {
    "overall_score": 0.6234,
    "perplexity_weighted_perplexity": 45.67,
    "text_quality_avg_bleu_4": 0.1234,
    "fluency_overall_fluency_score": 0.7890,
    "fable_structure_overall_fable_score": 0.6543
  },
  "execution_time": 142.5,
  "timestamp": "2024-01-15T10:30:00",
  "metadata": {
    "model_name": "gpt2",
    "summary_report": "..."
  }
}
```

## 🔧 **Advanced Usage**

### **Custom Test Data**
```python
# Define custom test cases
test_data = [
    {
        "prompt": "Once upon a time, there was a clever fox who",
        "reference": "lived in the forest and loved to play tricks.",
        "full_text": "Once upon a time, there was a clever fox who lived in the forest and loved to play tricks on other animals. But one day, he learned that kindness is more valuable than cleverness."
    }
]

# Run evaluation on custom data
result = evaluator.evaluate(model, tokenizer, test_data)
```

### **Configuration Options**
```python
config = EvaluationConfig(
    model_name="gpt2",
    dataset_name="klusai/ds-tf1-en-3m",
    dataset_split="test",
    num_samples=100,
    max_length=512,
    temperature=0.8,
    seed=42,
    device="cpu",
    output_dir="./results",
    save_generations=True,  # Save generated text
    verbose=True
)
```

### **Model Support**
- ✅ **Base Models**: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- ✅ **Fine-tuned Models**: Local paths to saved models
- ✅ **LoRA/PEFT Models**: Automatic adapter detection and loading
- ✅ **Custom Models**: Any GPT-2 compatible model

## 🎯 **Interpreting Results**

### **Overall Scores**
- **0.8-1.0**: Excellent fable completion capabilities
- **0.6-0.8**: Good performance, suitable for most applications  
- **0.4-0.6**: Fair performance, needs improvement
- **0.2-0.4**: Poor performance, significant issues
- **0.0-0.2**: Very poor, minimal capability

### **Key Metrics Thresholds**
- **Perplexity**: Lower is better (< 50 is good for GPT-2)
- **BLEU-4**: Higher is better (> 0.1 is reasonable for completion)
- **ROUGE-L**: Higher is better (> 0.2 shows good content overlap)
- **TTR (Diversity)**: 0.3-0.7 is typical (higher = more diverse)
- **Structure Score**: > 0.6 indicates good narrative elements

## 🚀 **Command Reference**

### **CLI Commands**
```bash
# List available evaluators
python tinyfabulist.py list

# Quick test
python tinyfabulist.py test

# Single evaluator
python tinyfabulist.py single --evaluator EVALUATOR --model MODEL

# Comprehensive evaluation  
python tinyfabulist.py comprehensive --model MODEL [OPTIONS]

# Model comparison
python tinyfabulist.py compare --models MODEL1 MODEL2 [OPTIONS]
```

### **Common Options**
- `--num-samples N`: Number of test samples (default: 100)
- `--temperature T`: Generation temperature (default: 0.8)
- `--output-dir DIR`: Save results directory
- `--save-generations`: Include generated text in output
- `--quiet`: Reduce output verbosity
- `--device DEVICE`: Computation device (default: cpu)

## 🔬 **For Researchers**

This framework provides **standardized evaluation** for fable completion models, enabling:

- **Reproducible benchmarks** across different models
- **Detailed analysis** of model capabilities and limitations  
- **Comparative studies** between fine-tuning approaches
- **Ablation studies** on specific metric components

All metrics are based on established NLP evaluation practices and can serve as baselines for academic research on narrative text generation.

---

**Happy Evaluating!** 🎭📊 
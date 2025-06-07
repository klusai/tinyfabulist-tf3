# TinyFabulist Experiment Tracking System

A comprehensive experiment management system for systematic research paper development with GPT models on fable completion tasks.

## Overview

This system provides complete experiment tracking, result storage, and analysis capabilities for research paper development. It automatically tracks:

- **Environment Information**: Hardware, software versions, git state
- **Experiment Configuration**: Model parameters, dataset, hyperparameters  
- **Execution Metadata**: Start/end times, status, error handling
- **Detailed Results**: All evaluation metrics and sample-level data
- **Reproducibility**: Complete state tracking for paper methodology sections

## Quick Start

### 1. Install Dependencies
```bash
cd experiments
pip install -r requirements.txt
```

### 2. Run Quick Demo
```bash
# Run a quick demo experiment (5 samples, fast execution)
python run_experiments.py --demo
```

### 3. View Results
```bash
# List all experiments
python experiment_manager.py list

# Compare experiment results
python experiment_manager.py compare <experiment_id_1> <experiment_id_2>

# Export for paper
python experiment_manager.py export <experiment_id_1> <experiment_id_2> --output results.csv
```

## System Components

### 1. `experiment_manager.py` - Core Tracking System
- **ExperimentManager**: Tracks experiments with unique IDs
- **ExperimentRunner**: Executes experiments with automatic tracking
- **Environment Capture**: Git state, device info, software versions
- **Result Storage**: JSON format with detailed metadata

### 2. `run_experiments.py` - Systematic Experiment Runner
- **Baseline Experiments**: Compare GPT-2 model variants
- **Parameter Studies**: Temperature, sample size variations
- **Evaluator Comparison**: Individual metric analysis
- **Fine-tuned Models**: Compare against baselines

### 3. `analysis_tools.py` - Results Analysis
- **Summary Tables**: CSV export for spreadsheet analysis
- **LaTeX Tables**: Publication-ready formatted tables
- **Statistical Reports**: Performance summaries and comparisons

## Experiment Types

### Baseline Model Comparison
```bash
python run_experiments.py --baseline
```
Compares GPT-2 small vs medium models on fable completion.

### Temperature Study
```bash
python run_experiments.py --temperature
```
Studies effect of temperature (0.3, 0.5, 0.8, 1.0, 1.2) on generation quality.

### Sample Size Analysis
```bash
python run_experiments.py --samples
```
Analyzes metric convergence with different sample sizes (10, 25, 50, 100).

### Evaluator Comparison
```bash
python run_experiments.py --evaluators
```
Compares individual evaluators (perplexity, BLEU, fluency, structure).

### Fine-tuned Model Evaluation
```bash
python run_experiments.py --finetuned
```
Evaluates fine-tuned models (add your model paths to the script).

### Run All Studies
```bash
python run_experiments.py --all
```

## Directory Structure

After running experiments:

```
experiments/
├── experiment_manager.py          # Core tracking system
├── run_experiments.py             # Automated experiment runner
├── analysis_tools.py              # Analysis and visualization
├── experiment_registry.json       # Index of all experiments
├── runs/                          # Individual experiment results
│   ├── exp1_20240608_a1b2c3d4/
│   │   ├── experiment.json        # Full experiment metadata
│   │   └── detailed_results.json  # Detailed evaluation results
│   └── exp2_20240608_e5f6g7h8/
├── exports/                       # Paper-ready exports
│   ├── baseline_comparison.csv
│   ├── temperature_study.csv
│   └── paper_results.tex
└── analysis/                      # Analysis outputs
```

## CLI Reference

### Experiment Management
```bash
# List all experiments
python experiment_manager.py list

# List by status
python experiment_manager.py list --status completed

# List by tags
python experiment_manager.py list --tags baseline paper

# Compare specific experiments
python experiment_manager.py compare exp1_id exp2_id exp3_id

# Export results
python experiment_manager.py export exp1_id exp2_id --output paper_results.csv
python experiment_manager.py export exp1_id exp2_id --output results_table.tex
```

### Analysis Tools
```bash
# Create summary table
python analysis_tools.py summary exp1_id exp2_id --output summary.csv

# Generate LaTeX table
python analysis_tools.py latex exp1_id exp2_id --output results.tex

# Create comprehensive report
python analysis_tools.py report exp1_id exp2_id --output analysis_report.txt
```

## Experiment Configuration

Each experiment is defined with:

```python
ExperimentConfig(
    experiment_name="gpt2_baseline",           # Unique name
    description="Baseline GPT-2 evaluation",   # Human-readable description
    model_name="gpt2",                         # Model identifier
    dataset_name="klusai/ds-tf1-en-3m",       # Dataset identifier
    num_samples=100,                           # Number of evaluation samples
    temperature=0.8,                           # Generation temperature
    seed=42,                                   # Random seed for reproducibility
    evaluators=["comprehensive"],              # List of evaluators to run
    tags=["baseline", "paper"],                # Tags for organization
    notes="Additional notes about experiment"   # Optional notes
)
```

## Paper Integration

### For Methodology Section
- Complete environment tracking (hardware, software versions)
- Exact hyperparameters and random seeds
- Dataset processing details
- Evaluation metrics specifications

### For Results Section
- Automated table generation (LaTeX format)
- Statistical significance testing
- Performance comparisons across models
- Parameter sensitivity analysis

### For Reproducibility
- Git commit tracking for exact code versions
- Complete experiment configuration storage
- Deterministic random seed handling
- Environment state capture

## Example Usage Workflow

```bash
# 1. Run systematic experiments
python run_experiments.py --baseline
python run_experiments.py --temperature

# 2. Get experiment IDs
python experiment_manager.py list --status completed

# 3. Analyze results
python analysis_tools.py summary exp1_id exp2_id exp3_id

# 4. Generate paper table
python analysis_tools.py latex exp1_id exp2_id exp3_id --output table1.tex

# 5. Export raw data
python experiment_manager.py export exp1_id exp2_id exp3_id --output raw_results.csv
```

## Best Practices

### Experiment Organization
- Use descriptive experiment names
- Apply consistent tags (`baseline`, `finetuned`, `paper`)
- Include detailed descriptions and notes
- Run with appropriate sample sizes for final results

### Paper Development
- Start with quick demos (small sample sizes)
- Scale up to full experiments for final results
- Use consistent random seeds across comparisons
- Export both summary tables and raw data

### Reproducibility
- Commit code before running experiments
- Use git tags for paper submission versions
- Include environment information in methodology
- Store experiment configurations in version control

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure you're in the project root directory
- **Device Issues**: System automatically detects optimal device (MPS/CUDA/CPU)
- **Dataset Loading**: Includes fallback to synthetic data if dataset fails
- **Memory Issues**: Reduce sample sizes or use smaller models

### Support
- Check experiment logs in individual run directories
- Use `--demo` flag for quick testing
- Verify TinyFabulist framework is working with `python tf3.py test`

This system provides everything needed for systematic, reproducible research paper development with comprehensive experiment tracking and analysis capabilities. 
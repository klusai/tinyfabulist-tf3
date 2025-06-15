#!/usr/bin/env python3
"""
TinyFabulist - Evaluation toolkit for fable completion models
A comprehensive evaluation framework for text generation models on moral fables.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Optional, List

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset, Dataset

# Add lib and evals directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
sys.path.append(os.path.dirname(__file__))

# Import lib modules for device management
from lib import (
    DeviceManager, 
    get_optimal_device, 
    setup_model_for_device,
    ModelLoader,
    ModelCache,
    optimize_for_apple_silicon,
    setup_logging,
    get_logger,
    set_log_level,
    disable_generated_text_logging,
    enable_generated_text_logging
)

from evals import (
    EvaluationConfig, 
    get_evaluator, 
    list_evaluators,
    ComprehensiveEvaluator
)

# Apply Apple Silicon optimizations
optimize_for_apple_silicon()


def load_model_and_tokenizer(model_name: str, device: str = None):
    """Load model and tokenizer with error handling and optimal device management"""
    logger = get_logger()
    
    try:
        # Use device detection if not specified
        if device is None:
            device = get_optimal_device(verbose=True)
        
        # Handle PEFT models
        if os.path.isdir(model_name) and os.path.exists(os.path.join(model_name, "adapter_config.json")):
            logger.info("🔧 Detected PEFT/LoRA model, loading with adapter", model=model_name)
            from peft import PeftModel
            
            # Load base model first
            with open(os.path.join(model_name, "adapter_config.json")) as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "gpt2")
            
            logger.info("Loading base model for PEFT adapter", base_model=base_model_name)
            
            # Use ModelLoader for base model
            loader = ModelLoader(model_name=base_model_name, verbose=True)
            base_model, tokenizer, actual_device = loader.load_model_and_tokenizer()
            
            # Load PEFT adapter
            model = PeftModel.from_pretrained(base_model, model_name)
            model = model.to(actual_device)
            
            logger.log_model_info(model_name, actual_device, type="PEFT", base_model=base_model_name)
            return model, tokenizer
        else:
            # Use ModelCache for regular models
            logger.info("Loading model from cache", model=model_name)
            model, tokenizer, actual_device = ModelCache.get_model(
                model_name=model_name, 
                prefer_mps=True, 
                verbose=True
            )
            
            logger.log_model_info(model_name, actual_device, type="standard")
            return model, tokenizer
        
    except Exception as e:
        logger.log_error_context(e, "Model loading failed", model=model_name)
        logger.error("Available models: gpt2, gpt2-medium, gpt2-large, or path to fine-tuned model")
        sys.exit(1)


def load_huggingface_dataset(dataset_name: str, split: str, verbose: bool = True, max_samples: Optional[int] = None) -> Dataset:
    """Load a dataset from Hugging Face with compatibility transformations."""
    logger = get_logger()

    try:
        logger.info("📊 Loading dataset from Hugging Face", dataset=dataset_name, split=split, max_samples=max_samples)
        
        # Load the dataset from Hugging Face
        dataset = load_dataset(dataset_name, split=split)
        
        # Limit samples if requested
        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        # Transform dataset if needed - convert 'text' field to 'fable' field for compatibility
        if 'text' in dataset.column_names and 'fable' not in dataset.column_names:
            def rename_text_to_fable(example):
                example['fable'] = example['text']
                return example
            
            dataset = dataset.map(rename_text_to_fable)
            logger.info("✓ Transformed 'text' field to 'fable' for compatibility")
        
        logger.log_dataset_info(dataset_name, split, len(dataset), columns=list(dataset.column_names))
        return dataset
        
    except Exception as e:
        logger.log_error_context(e, "Fatal: Dataset loading failed", dataset=dataset_name, split=split)
        logger.error(f"Could not load dataset '{dataset_name}'. Please check the dataset name and your connection.")
        sys.exit(1)


def create_config_from_args(args) -> EvaluationConfig:
    """Create evaluation config from command line arguments"""
    config_kwargs = {
        'model_name': args.model,
        'dataset_name': args.dataset,
        'dataset_split': args.split,
        'num_samples': args.num_samples,
        'max_length': args.max_length,
        'temperature': args.temperature,
        'seed': args.seed,
        'output_dir': args.output_dir,
        'save_generations': args.save_generations,
        'verbose': not args.quiet
    }
    
    # Only set device if explicitly specified, otherwise let default_factory handle it
    if args.device is not None:
        config_kwargs['device'] = args.device
    
    return EvaluationConfig(**config_kwargs)


def run_single_evaluator(args):
    """Run a single evaluator"""
    
    # Create config
    config = create_config_from_args(args)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    
    # Update config with actual device from model
    actual_device = next(model.parameters()).device.type
    config.device = actual_device
    
    # Load dataset
    dataset = load_huggingface_dataset(
        args.dataset,
        args.split,
        verbose=not args.quiet,
        max_samples=args.num_samples * 2
    )
    
    # Get evaluator
    try:
        evaluator = get_evaluator(args.evaluator, config=config)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Available evaluators: {', '.join(list_evaluators())}")
        sys.exit(1)
    
    # Run evaluation
    print(f"Running {args.evaluator} evaluation...")
    result = evaluator.run(model, tokenizer, dataset)
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"{args.evaluator}_results.json")
        result.save_json(output_file)
        print(f"Results saved to: {output_file}")
    
    # Print summary
    print("\n" + result.summary())
    
    return result


def run_comprehensive_evaluation(args):
    """Run comprehensive evaluation with all metrics"""
    
    # Create config
    config = create_config_from_args(args)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    
    # Update config with actual device from model
    actual_device = next(model.parameters()).device.type
    config.device = actual_device
    
    # Load dataset
    dataset = load_huggingface_dataset(
        args.dataset,
        args.split,
        verbose=not args.quiet,
        max_samples=args.num_samples * 2
    )
    
    # Create comprehensive evaluator
    evaluator = ComprehensiveEvaluator(config=config)
    
    # Set enabled evaluators if specified
    if args.evaluators:
        evaluator.set_enabled_evaluators(args.evaluators)
    
    # Run evaluation
    print("Running comprehensive evaluation...")
    result = evaluator.run(model, tokenizer, dataset)
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save main results
        output_file = os.path.join(args.output_dir, "comprehensive_results.json")
        result.save_json(output_file)
        
        # Save summary report
        report_file = os.path.join(args.output_dir, "evaluation_report.txt")
        with open(report_file, 'w') as f:
            f.write(result.metadata["summary_report"])
        
        print(f"Results saved to: {output_file}")
        print(f"Report saved to: {report_file}")
    
    # Print summary report
    print("\n" + result.metadata["summary_report"])
    
    return result


def compare_models(args):
    """Compare multiple models"""
    
    if len(args.models) < 2:
        print("Error: At least 2 models required for comparison")
        sys.exit(1)
    
    results = {}
    
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*60}")
        
        # Update args for this model
        args.model = model_name
        
        # Run evaluation
        result = run_comprehensive_evaluation(args)
        results[model_name] = result
    
    # Create comparison report
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    comparison_lines = [
        f"{'Model':<30} {'Overall':<10} {'Perplexity':<12} {'BLEU-4':<10} {'Fluency':<10} {'Structure':<10}"
    ]
    comparison_lines.append("-" * 90)
    
    for model_name, result in results.items():
        metrics = result.metrics
        
        overall = metrics.get('overall_score', 0.0)
        perplexity = metrics.get('perplexity_weighted_perplexity', 0.0)
        bleu4 = metrics.get('text_quality_avg_bleu_4', 0.0)
        fluency = metrics.get('fluency_overall_fluency_score', 0.0)
        structure = metrics.get('fable_structure_overall_fable_score', 0.0)
        
        comparison_lines.append(
            f"{model_name:<30} {overall:<10.3f} {perplexity:<12.2f} {bleu4:<10.3f} {fluency:<10.3f} {structure:<10.3f}"
        )
    
    print("\n".join(comparison_lines))
    
    # Save comparison
    if args.output_dir:
        comparison_file = os.path.join(args.output_dir, "model_comparison.txt")
        with open(comparison_file, 'w') as f:
            f.write("\n".join(comparison_lines))
        print(f"\nComparison saved to: {comparison_file}")


def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(
        description="TinyFabulist - Evaluation toolkit for fable completion models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive evaluation
  python tf3.py comprehensive --model gpt2 --num-samples 100

  # Run single evaluator
  python tf3.py single --evaluator perplexity --model gpt2

  # Compare multiple models
  python tf3.py compare --models gpt2 gpt2-medium --num-samples 50

  # Use fine-tuned model
  python tf3.py comprehensive --model ./my-finetuned-gpt2

Available evaluators: perplexity, text_quality, fluency, fable_structure, comprehensive
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Evaluation commands")
    
    # Common arguments
    def add_common_args(parser):
        parser.add_argument("--model", default="gpt2", 
                          help="Model name or path to evaluate (default: gpt2)")
        parser.add_argument("--dataset", default="klusai/ds-tf1-en-3m",
                          help="Dataset to evaluate on (default: klusai/ds-tf1-en-3m)")
        parser.add_argument("--split", default="Test",
                          help="Dataset split to use (default: Test)")
        parser.add_argument("--num-samples", type=int, default=100,
                          help="Number of samples to evaluate (default: 100)")
        parser.add_argument("--max-length", type=int, default=512,
                          help="Maximum sequence length (default: 512)")
        parser.add_argument("--temperature", type=float, default=0.8,
                          help="Generation temperature (default: 0.8)")
        parser.add_argument("--seed", type=int, default=42,
                          help="Random seed (default: 42)")
        parser.add_argument("--device", default=None,
                          help="Device to use (auto-detects optimal device if not specified)")
        parser.add_argument("--output-dir", type=str,
                          help="Directory to save results")
        parser.add_argument("--save-generations", action="store_true",
                          help="Save generated text in results")
        parser.add_argument("--quiet", action="store_true",
                          help="Reduce output verbosity")
        
        # Logging arguments
        parser.add_argument("--log-level", default="INFO", 
                          choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                          help="Set logging level (default: INFO)")
        parser.add_argument("--log-file", type=str,
                          help="Save logs to file")
        parser.add_argument("--hide-generated-text", action="store_true",
                          help="Don't show generated text samples in logs")
        parser.add_argument("--show-all-samples", action="store_true",
                          help="Show detailed logs for all samples (not just first 5)")
    
    # Single evaluator command
    single_parser = subparsers.add_parser("single", help="Run single evaluator")
    single_parser.add_argument("--evaluator", required=True,
                             choices=list_evaluators(),
                             help="Evaluator to run")
    add_common_args(single_parser)
    
    # Comprehensive evaluation command
    comp_parser = subparsers.add_parser("comprehensive", help="Run comprehensive evaluation")
    comp_parser.add_argument("--evaluators", nargs="+", 
                           choices=list_evaluators(),
                           help="Specific evaluators to run (default: all)")
    add_common_args(comp_parser)
    
    # Model comparison command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--models", nargs="+", required=True,
                              help="Models to compare")
    compare_parser.add_argument("--evaluators", nargs="+",
                              choices=list_evaluators(),
                              help="Specific evaluators to run (default: all)")
    add_common_args(compare_parser)
    
    # List evaluators command
    list_parser = subparsers.add_parser("list", help="List available evaluators")
    
    # Quick test command
    test_parser = subparsers.add_parser("test", help="Quick test with minimal samples")
    add_common_args(test_parser)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging based on arguments
    if hasattr(args, 'log_level'):
        log_level = "WARNING" if getattr(args, 'quiet', False) else args.log_level
        show_generated = not getattr(args, 'hide_generated_text', False)
        
        # Set up logging with file output if specified
        setup_logging(
            level=log_level,
            log_file=getattr(args, 'log_file', None),
            show_generated_text=show_generated,
            console_output=True
        )
        
        # Log startup info
        logger = get_logger()
        logger.info("🎯 TinyFabulist evaluation starting")
        logger.debug("Command line arguments", **{k: str(v) for k, v in vars(args).items()})
    
    # Handle commands
    if args.command == "list":
        print("Available evaluators:")
        for evaluator in list_evaluators():
            print(f"  - {evaluator}")
        sys.exit(0)
    
    elif args.command == "test":
        logger = get_logger()
        logger.info("🧪 Running quick test...")
        
        # Override some args for quick testing if not specified
        if args.num_samples == 100:  # Default value, override for test
            args.num_samples = 5
        if not hasattr(args, 'output_dir') or args.output_dir is None:
            args.output_dir = None
        if args.max_length == 512:  # Default value, override for test
            args.max_length = 256
        if not hasattr(args, 'save_generations'):
            args.save_generations = False
        args.evaluators = ["semantic_coherence", "fluency"]  # Most important evaluators
        
        try:
            run_comprehensive_evaluation(args)
            logger.info("✅ Test completed successfully!")
        except Exception as e:
            logger.log_error_context(e, "Test failed")
            sys.exit(1)
    
    elif args.command == "single":
        run_single_evaluator(args)
    
    elif args.command == "comprehensive":
        run_comprehensive_evaluation(args)
    
    elif args.command == "compare":
        compare_models(args)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 
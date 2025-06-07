#!/usr/bin/env python3
"""
Script to run systematic experiments for paper
"""

from experiment_manager import ExperimentManager, ExperimentRunner, ExperimentConfig
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_baseline_experiments():
    """Run baseline experiments for the paper"""
    
    manager = ExperimentManager()
    runner = ExperimentRunner(manager)
    
    # Define baseline experiments
    experiments = [
        # GPT-2 variants comparison
        ExperimentConfig(
            experiment_name="gpt2_small_baseline",
            description="GPT-2 small baseline on fable completion",
            model_name="gpt2",
            dataset_name="klusai/ds-tf1-en-3m",
            num_samples=50,  # Reduced for faster testing
            temperature=0.8,
            seed=42,
            evaluators=["comprehensive"],
            tags=["baseline", "gpt2-small", "paper"],
            notes="Baseline evaluation of vanilla GPT-2 small model"
        ),
        
        ExperimentConfig(
            experiment_name="gpt2_medium_baseline", 
            description="GPT-2 medium baseline on fable completion",
            model_name="gpt2-medium",
            dataset_name="klusai/ds-tf1-en-3m", 
            num_samples=50,
            temperature=0.8,
            seed=42,
            evaluators=["comprehensive"],
            tags=["baseline", "gpt2-medium", "paper"],
            notes="Baseline evaluation of vanilla GPT-2 medium model"
        ),
    ]
    
    results = []
    for config in experiments:
        print(f"\n🚀 Running experiment: {config.experiment_name}")
        print(f"📝 Description: {config.description}")
        try:
            experiment_id = runner.run_experiment(config)
            results.append(experiment_id)
            print(f"✅ Completed: {experiment_id}")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    return results

def run_temperature_study():
    """Run temperature variation study"""
    
    manager = ExperimentManager()
    runner = ExperimentRunner(manager)
    
    temperatures = [0.3, 0.5, 0.8, 1.0, 1.2]
    
    experiments = []
    for temp in temperatures:
        config = ExperimentConfig(
            experiment_name=f"gpt2_temp_{str(temp).replace('.', '_')}",
            description=f"GPT-2 with temperature {temp}",
            model_name="gpt2",
            dataset_name="klusai/ds-tf1-en-3m",
            num_samples=30,  # Smaller sample for parameter study
            temperature=temp,
            seed=42,
            evaluators=["comprehensive"],
            tags=["temperature-study", "gpt2-small", "paper"],
            notes=f"Temperature variation study: T={temp}"
        )
        experiments.append(config)
    
    results = []
    for config in experiments:
        print(f"\n🌡️  Running temperature experiment: T={config.temperature}")
        try:
            experiment_id = runner.run_experiment(config)
            results.append(experiment_id)
            print(f"✅ Completed: {experiment_id}")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    return results

def run_evaluator_comparison():
    """Run experiments with different individual evaluators for comparison"""
    
    manager = ExperimentManager()
    runner = ExperimentRunner(manager)
    
    evaluators_to_test = ["perplexity", "text_quality", "fluency", "fable_structure"]
    
    experiments = []
    for evaluator in evaluators_to_test:
        config = ExperimentConfig(
            experiment_name=f"gpt2_eval_{evaluator}",
            description=f"GPT-2 evaluation using {evaluator} metric only",
            model_name="gpt2",
            dataset_name="klusai/ds-tf1-en-3m",
            num_samples=30,
            temperature=0.8,
            seed=42,
            evaluators=[evaluator],
            tags=["evaluator-study", "gpt2-small", "paper"],
            notes=f"Single evaluator study: {evaluator}"
        )
        experiments.append(config)
    
    results = []
    for config in experiments:
        print(f"\n📊 Running evaluator experiment: {config.evaluators[0]}")
        try:
            experiment_id = runner.run_experiment(config)
            results.append(experiment_id)
            print(f"✅ Completed: {experiment_id}")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    return results

def run_sample_size_study():
    """Run experiments with different sample sizes to study convergence"""
    
    manager = ExperimentManager()
    runner = ExperimentRunner(manager)
    
    sample_sizes = [10, 25, 50, 100]
    
    experiments = []
    for size in sample_sizes:
        config = ExperimentConfig(
            experiment_name=f"gpt2_samples_{size}",
            description=f"GPT-2 evaluation with {size} samples",
            model_name="gpt2",
            dataset_name="klusai/ds-tf1-en-3m",
            num_samples=size,
            temperature=0.8,
            seed=42,
            evaluators=["comprehensive"],
            tags=["sample-study", "gpt2-small", "paper"],
            notes=f"Sample size study: N={size}"
        )
        experiments.append(config)
    
    results = []
    for config in experiments:
        print(f"\n📈 Running sample size experiment: N={config.num_samples}")
        try:
            experiment_id = runner.run_experiment(config)
            results.append(experiment_id)
            print(f"✅ Completed: {experiment_id}")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    return results

def run_finetuning_comparison():
    """Run fine-tuned model comparison (if you have fine-tuned models)"""
    
    manager = ExperimentManager()
    runner = ExperimentRunner(manager)
    
    # Add your fine-tuned model paths here
    # These are examples - replace with your actual model paths
    finetuned_models = [
        # "./models/gpt2-finetuned-fables",
        # "./models/gpt2-lora-fables", 
    ]
    
    # If no fine-tuned models available, create a placeholder experiment
    if not finetuned_models:
        print("⚠️  No fine-tuned models specified. Add model paths to run_finetuning_comparison()")
        print("Creating placeholder experiment for demonstration...")
        
        # Create a baseline experiment as placeholder
        config = ExperimentConfig(
            experiment_name="finetuned_placeholder",
            description="Placeholder for fine-tuned model (using baseline GPT-2)",
            model_name="gpt2",
            dataset_name="klusai/ds-tf1-en-3m",
            num_samples=25,
            temperature=0.8,
            seed=42,
            evaluators=["comprehensive"],
            tags=["finetuned", "placeholder", "paper"],
            notes="Placeholder experiment - replace with actual fine-tuned model"
        )
        
        try:
            experiment_id = runner.run_experiment(config)
            return [experiment_id]
        except Exception as e:
            print(f"❌ Failed: {e}")
            return []
    
    experiments = []
    for model_path in finetuned_models:
        config = ExperimentConfig(
            experiment_name=f"finetuned_{model_path.split('/')[-1]}",
            description=f"Fine-tuned model comparison: {model_path}",
            model_name=model_path,
            dataset_name="klusai/ds-tf1-en-3m",
            num_samples=50,
            temperature=0.8,
            seed=42,
            evaluators=["comprehensive"],
            tags=["finetuned", "comparison", "paper"],
            notes=f"Fine-tuned model evaluation: {model_path}"
        )
        experiments.append(config)
    
    results = []
    for config in experiments:
        print(f"\n🎯 Running fine-tuned experiment: {config.model_name}")
        try:
            experiment_id = runner.run_experiment(config)
            results.append(experiment_id)
            print(f"✅ Completed: {experiment_id}")
        except Exception as e:
            print(f"❌ Failed {config.experiment_name}: {e}")
    
    return results

def run_quick_demo():
    """Run a quick demo experiment for testing the system"""
    
    manager = ExperimentManager()
    runner = ExperimentRunner(manager)
    
    config = ExperimentConfig(
        experiment_name="quick_demo",
        description="Quick demo of experiment tracking system",
        model_name="gpt2",
        dataset_name="klusai/ds-tf1-en-3m",
        num_samples=5,  # Very small for fast execution
        temperature=0.8,
        seed=42,
        evaluators=["fluency"],  # Single fast evaluator
        tags=["demo", "test"],
        notes="Quick demonstration of the experiment tracking system"
    )
    
    print(f"\n⚡ Running quick demo experiment...")
    try:
        experiment_id = runner.run_experiment(config)
        print(f"✅ Demo completed: {experiment_id}")
        return [experiment_id]
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return []

def main():
    """Main experiment runner with different study options"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TinyFabulist Systematic Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline model comparison
  python run_experiments.py --baseline
  
  # Run temperature study
  python run_experiments.py --temperature
  
  # Run all experiments
  python run_experiments.py --all
  
  # Quick demo
  python run_experiments.py --demo
        """
    )
    
    parser.add_argument("--baseline", action="store_true", 
                       help="Run baseline model experiments")
    parser.add_argument("--temperature", action="store_true", 
                       help="Run temperature variation study")
    parser.add_argument("--evaluators", action="store_true", 
                       help="Run individual evaluator comparison")
    parser.add_argument("--samples", action="store_true", 
                       help="Run sample size study")
    parser.add_argument("--finetuned", action="store_true", 
                       help="Run fine-tuned model comparisons")
    parser.add_argument("--demo", action="store_true", 
                       help="Run quick demo experiment")
    parser.add_argument("--all", action="store_true", 
                       help="Run all experiments")
    
    args = parser.parse_args()
    
    all_results = []
    
    if args.demo or (not any([args.baseline, args.temperature, args.evaluators, 
                             args.samples, args.finetuned, args.all])):
        print("🎬 Running quick demo...")
        demo_results = run_quick_demo()
        all_results.extend(demo_results)
    
    if args.baseline or args.all:
        print("\n🔬 Running baseline experiments...")
        baseline_results = run_baseline_experiments()
        all_results.extend(baseline_results)
        print(f"✅ Baseline experiments completed: {len(baseline_results)} runs")
    
    if args.temperature or args.all:
        print("\n🌡️  Running temperature study...")
        temp_results = run_temperature_study()
        all_results.extend(temp_results)
        print(f"✅ Temperature study completed: {len(temp_results)} runs")
    
    if args.evaluators or args.all:
        print("\n📊 Running evaluator comparison...")
        eval_results = run_evaluator_comparison()
        all_results.extend(eval_results)
        print(f"✅ Evaluator comparison completed: {len(eval_results)} runs")
    
    if args.samples or args.all:
        print("\n📈 Running sample size study...")
        sample_results = run_sample_size_study()
        all_results.extend(sample_results)
        print(f"✅ Sample size study completed: {len(sample_results)} runs")
    
    if args.finetuned or args.all:
        print("\n🎯 Running fine-tuned model experiments...")
        finetuned_results = run_finetuning_comparison()
        all_results.extend(finetuned_results)
        print(f"✅ Fine-tuned experiments completed: {len(finetuned_results)} runs")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📋 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments completed: {len(all_results)}")
    
    if all_results:
        print(f"\n🆔 Experiment IDs:")
        for exp_id in all_results:
            print(f"  • {exp_id}")
        
        print(f"\n💡 Next steps:")
        print(f"  # List all experiments")
        print(f"  python experiment_manager.py list")
        print(f"  ")
        print(f"  # Compare results")
        print(f"  python experiment_manager.py compare {' '.join(all_results)}")
        print(f"  ")
        print(f"  # Export for paper")
        print(f"  python experiment_manager.py export {' '.join(all_results)} --output paper_results.csv")

if __name__ == "__main__":
    main() 
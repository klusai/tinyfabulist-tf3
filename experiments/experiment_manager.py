#!/usr/bin/env python3
"""
Experiment Management System for TinyFabulist
Systematic tracking and storage of evaluation results for research papers
"""

import os
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import sys

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evals import EvaluationResult, EvaluationConfig

@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""
    experiment_name: str
    description: str
    model_name: str
    dataset_name: str
    num_samples: int
    temperature: float
    seed: int
    evaluators: List[str]
    tags: List[str] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ExperimentManager:
    """Manages experiment tracking and result storage"""
    
    def __init__(self, base_dir: str = None):
        # Auto-detect if we're already in the experiments directory
        if base_dir is None:
            current_dir = Path.cwd()
            if current_dir.name == "experiments":
                base_dir = "."  # Use current directory
            else:
                base_dir = "experiments"  # Use subdirectory
        
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.base_dir / "runs").mkdir(exist_ok=True)
        (self.base_dir / "results").mkdir(exist_ok=True)
        (self.base_dir / "analysis").mkdir(exist_ok=True)
        (self.base_dir / "exports").mkdir(exist_ok=True)
        
        self.registry_file = self.base_dir / "experiment_registry.json"
        self.load_registry()
    
    def load_registry(self):
        """Load experiment registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "experiments": {},
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
    
    def save_registry(self):
        """Save experiment registry"""
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def get_experiment_id(self, config: ExperimentConfig) -> str:
        """Generate unique experiment ID"""
        # Create hash from key parameters
        key_params = f"{config.model_name}_{config.dataset_name}_{config.num_samples}_{config.temperature}_{config.seed}"
        hash_obj = hashlib.md5(key_params.encode())
        hash_id = hash_obj.hexdigest()[:8]
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{config.experiment_name}_{timestamp}_{hash_id}"
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Collect environment information"""
        import torch
        import platform
        from lib import get_optimal_device
        
        env_info = {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
            },
            "torch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
                "optimal_device": get_optimal_device(verbose=False)
            },
            "git": self.get_git_info(),
            "working_directory": os.getcwd()
        }
        
        return env_info
    
    def get_git_info(self) -> Dict[str, str]:
        """Get git repository information"""
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # Check for uncommitted changes
            try:
                subprocess.check_output(
                    ["git", "diff-index", "--quiet", "HEAD", "--"], stderr=subprocess.DEVNULL
                )
                is_dirty = False
            except subprocess.CalledProcessError:
                is_dirty = True
            
            return {
                "commit_hash": commit_hash,
                "branch": branch,
                "is_dirty": is_dirty
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {"commit_hash": "unknown", "branch": "unknown", "is_dirty": False}
    
    def start_experiment(self, config: ExperimentConfig) -> str:
        """Start a new experiment and return experiment ID"""
        experiment_id = self.get_experiment_id(config)
        
        # Create experiment directory
        exp_dir = self.base_dir / "runs" / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Save experiment configuration
        exp_data = {
            "experiment_id": experiment_id,
            "config": config.to_dict(),
            "environment": self.get_environment_info(),
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "results": None,
            "end_time": None
        }
        
        with open(exp_dir / "experiment.json", 'w') as f:
            json.dump(exp_data, f, indent=2)
        
        # Update registry
        self.registry["experiments"][experiment_id] = {
            "name": config.experiment_name,
            "description": config.description,
            "model": config.model_name,
            "dataset": config.dataset_name,
            "start_time": exp_data["start_time"],
            "status": "running",
            "tags": config.tags,
            "directory": str(exp_dir)
        }
        self.save_registry()
        
        print(f"🚀 Started experiment: {experiment_id}")
        print(f"📁 Directory: {exp_dir}")
        return experiment_id
    
    def finish_experiment(self, experiment_id: str, results: EvaluationResult):
        """Complete an experiment with results"""
        exp_dir = self.base_dir / "runs" / experiment_id
        
        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Load experiment data
        with open(exp_dir / "experiment.json", 'r') as f:
            exp_data = json.load(f)
        
        # Update with results
        exp_data["status"] = "completed"
        exp_data["end_time"] = datetime.now().isoformat()
        exp_data["results"] = results.to_dict()
        
        # Save updated experiment
        with open(exp_dir / "experiment.json", 'w') as f:
            json.dump(exp_data, f, indent=2)
        
        # Save detailed results
        results.save_json(exp_dir / "detailed_results.json")
        
        # Update registry
        self.registry["experiments"][experiment_id]["status"] = "completed"
        self.registry["experiments"][experiment_id]["end_time"] = exp_data["end_time"]
        self.save_registry()
        
        print(f"✅ Completed experiment: {experiment_id}")
    
    def list_experiments(self, status: Optional[str] = None, tags: Optional[List[str]] = None) -> pd.DataFrame:
        """List experiments with optional filtering"""
        experiments = []
        
        for exp_id, exp_info in self.registry["experiments"].items():
            # Filter by status
            if status and exp_info.get("status") != status:
                continue
            
            # Filter by tags
            if tags and not any(tag in exp_info.get("tags", []) for tag in tags):
                continue
            
            experiments.append({
                "experiment_id": exp_id,
                "name": exp_info["name"],
                "model": exp_info["model"],
                "dataset": exp_info["dataset"],
                "status": exp_info.get("status", "unknown"),
                "start_time": exp_info["start_time"],
                "tags": ", ".join(exp_info.get("tags", []))
            })
        
        return pd.DataFrame(experiments)
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed results for an experiment"""
        exp_dir = self.base_dir / "runs" / experiment_id
        
        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        with open(exp_dir / "experiment.json", 'r') as f:
            return json.load(f)
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare results across multiple experiments"""
        comparison_data = []
        
        for exp_id in experiment_ids:
            try:
                exp_data = self.get_experiment_results(exp_id)
                
                if exp_data.get("results"):
                    row = {
                        "experiment_id": exp_id,
                        "experiment_name": exp_data["config"]["experiment_name"],
                        "model": exp_data["config"]["model_name"],
                        "dataset": exp_data["config"]["dataset_name"],
                        "temperature": exp_data["config"]["temperature"],
                        "num_samples": exp_data["config"]["num_samples"],
                        "device": exp_data.get("environment", {}).get("torch", {}).get("optimal_device", "unknown")
                    }
                    
                    # Add metrics from results
                    metrics = exp_data["results"]["metrics"]
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            row[key] = value
                    
                    comparison_data.append(row)
                    
            except Exception as e:
                print(f"⚠️  Error loading experiment {exp_id}: {e}")
        
        return pd.DataFrame(comparison_data)
    
    def export_for_paper(self, experiment_ids: List[str], output_file: str):
        """Export results in paper-ready format"""
        df = self.compare_experiments(experiment_ids)
        
        if df.empty:
            print("No valid experiments found for export")
            return
        
        export_path = self.base_dir / "exports" / output_file
        export_path.parent.mkdir(exist_ok=True)
        
        # Export as CSV
        if output_file.endswith('.csv'):
            df.to_csv(export_path, index=False)
        
        # Export as LaTeX table
        elif output_file.endswith('.tex'):
            latex_table = df.to_latex(index=False, float_format="%.3f")
            with open(export_path, 'w') as f:
                f.write(latex_table)
        
        # Export as JSON
        else:
            df.to_json(export_path, orient='records', indent=2)
        
        print(f"📊 Exported results to: {export_path}")
        return export_path


class ExperimentRunner:
    """Helper class to run experiments with automatic tracking"""
    
    def __init__(self, manager: ExperimentManager):
        self.manager = manager
    
    def run_experiment(self, config: ExperimentConfig) -> str:
        """Run a complete experiment with tracking"""
        
        # Start experiment tracking
        experiment_id = self.manager.start_experiment(config)
        
        try:
            # Import here to avoid circular imports
            from tf3 import load_model_and_tokenizer, load_dataset_with_fallback
            from evals import get_evaluator, ComprehensiveEvaluator
            
            print(f"🤖 Loading model: {config.model_name}")
            model, tokenizer = load_model_and_tokenizer(config.model_name)
            
            print(f"📚 Loading dataset: {config.dataset_name}")
            dataset = load_dataset_with_fallback(
                config.dataset_name, 
                "test", 
                verbose=True, 
                max_samples=config.num_samples * 2
            )
            
            # Create evaluation config
            eval_config = EvaluationConfig(
                model_name=config.model_name,
                dataset_name=config.dataset_name,
                num_samples=config.num_samples,
                temperature=config.temperature,
                seed=config.seed,
                verbose=True
            )
            
            # Run evaluation
            if len(config.evaluators) == 1:
                evaluator = get_evaluator(config.evaluators[0], config=eval_config)
                result = evaluator.run(model, tokenizer, dataset)
            else:
                evaluator = ComprehensiveEvaluator(config=eval_config)
                evaluator.set_enabled_evaluators(config.evaluators)
                result = evaluator.run(model, tokenizer, dataset)
            
            # Finish experiment
            self.manager.finish_experiment(experiment_id, result)
            
            return experiment_id
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            # Mark as failed
            exp_dir = self.manager.base_dir / "runs" / experiment_id
            with open(exp_dir / "experiment.json", 'r') as f:
                exp_data = json.load(f)
            
            exp_data["status"] = "failed"
            exp_data["error"] = str(e)
            exp_data["end_time"] = datetime.now().isoformat()
            
            with open(exp_dir / "experiment.json", 'w') as f:
                json.dump(exp_data, f, indent=2)
            
            raise

def main():
    """CLI for experiment management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TinyFabulist Experiment Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List experiments
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--status", choices=["running", "completed", "failed"])
    list_parser.add_argument("--tags", nargs="+", help="Filter by tags")
    
    # Compare experiments
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("experiment_ids", nargs="+", help="Experiment IDs to compare")
    
    # Export results
    export_parser = subparsers.add_parser("export", help="Export results for paper")
    export_parser.add_argument("experiment_ids", nargs="+", help="Experiment IDs to export")
    export_parser.add_argument("--output", required=True, help="Output file (csv, tex, json)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ExperimentManager()
    
    if args.command == "list":
        df = manager.list_experiments(status=args.status, tags=args.tags)
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("No experiments found matching criteria")
    
    elif args.command == "compare":
        df = manager.compare_experiments(args.experiment_ids)
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("No comparable results found")
    
    elif args.command == "export":
        manager.export_for_paper(args.experiment_ids, args.output)

if __name__ == "__main__":
    main() 
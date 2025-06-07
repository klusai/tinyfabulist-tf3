#!/usr/bin/env python3
"""
Analysis tools for experiment results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from experiment_manager import ExperimentManager
from typing import List, Dict, Any
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class ExperimentAnalyzer:
    """Tools for analyzing experiment results"""
    
    def __init__(self, manager: ExperimentManager):
        self.manager = manager
        # Set up matplotlib for better plots
        plt.style.use('default')
        try:
            sns.set_palette("husl")
        except:
            pass  # seaborn might not be available
    
    def create_results_summary(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Create a comprehensive results summary"""
        df = self.manager.compare_experiments(experiment_ids)
        
        if df.empty:
            return df
        
        # Calculate derived metrics
        if 'perplexity_weighted_perplexity' in df.columns:
            # Handle infinite perplexity values
            df['log_perplexity'] = np.log(df['perplexity_weighted_perplexity'].replace([np.inf, -np.inf], np.nan))
        
        # Round numeric columns for better presentation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(4)
        
        return df
    
    def create_latex_table(self, experiment_ids: List[str], 
                          metrics: List[str] = None,
                          caption: str = "Model Performance Comparison",
                          label: str = "tab:model_comparison") -> str:
        """Generate LaTeX table for paper"""
        df = self.manager.compare_experiments(experiment_ids)
        
        if df.empty:
            return "No data available"
        
        # Select key metrics if not specified
        if metrics is None:
            available_metrics = [
                'overall_score',
                'perplexity_weighted_perplexity', 
                'text_quality_avg_bleu_4',
                'fluency_overall_fluency_score',
                'temperature'
            ]
            metrics = [col for col in available_metrics if col in df.columns]
        
        # Filter and format
        table_cols = ['model'] + [col for col in metrics if col in df.columns]
        table_df = df[table_cols].copy()
        
        # Round numeric values and handle infinities
        for col in table_df.columns:
            if table_df[col].dtype in ['float64', 'float32']:
                # Replace infinities with a large number or "inf" string
                table_df[col] = table_df[col].replace([np.inf], 999.999)
                table_df[col] = table_df[col].replace([-np.inf], -999.999)
                table_df[col] = table_df[col].round(3)
        
        # Rename columns for better presentation
        column_mapping = {
            'model': 'Model',
            'overall_score': 'Overall Score',
            'perplexity_weighted_perplexity': 'Perplexity',
            'text_quality_avg_bleu_4': 'BLEU-4',
            'fluency_overall_fluency_score': 'Fluency',
            'temperature': 'Temperature'
        }
        
        table_df = table_df.rename(columns=column_mapping)
        
        # Generate LaTeX
        latex_table = table_df.to_latex(
            index=False,
            float_format="%.3f",
            caption=caption,
            label=label,
            escape=False,
            column_format='l' + 'c' * (len(table_df.columns) - 1)
        )
        
        return latex_table
    
    def create_summary_report(self, experiment_ids: List[str]) -> str:
        """Create a comprehensive summary report"""
        df = self.manager.compare_experiments(experiment_ids)
        
        if df.empty:
            return "No experiments found"
        
        report_lines = [
            "=" * 80,
            "EXPERIMENT ANALYSIS REPORT",
            "=" * 80,
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Experiments: {len(df)}",
            "",
            "EXPERIMENT OVERVIEW",
            "-" * 40
        ]
        
        # Basic statistics
        for _, row in df.iterrows():
            report_lines.extend([
                f"Experiment: {row['experiment_name']}",
                f"  Model: {row['model']}",
                f"  Temperature: {row['temperature']:.1f}",
                f"  Samples: {int(row['num_samples'])}",
                f"  Device: {row.get('device', 'unknown')}",
                ""
            ])
        
        # Performance summary
        if 'overall_score' in df.columns:
            report_lines.extend([
                "PERFORMANCE SUMMARY",
                "-" * 40,
                f"Best Overall Score: {df['overall_score'].max():.4f} ({df.loc[df['overall_score'].idxmax(), 'model']})",
                f"Worst Overall Score: {df['overall_score'].min():.4f} ({df.loc[df['overall_score'].idxmin(), 'model']})",
                f"Mean Overall Score: {df['overall_score'].mean():.4f}",
                f"Std Overall Score: {df['overall_score'].std():.4f}",
                ""
            ])
        
        return "\n".join(report_lines)

def main():
    """CLI for analysis tools"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment Analysis Tools")
    parser.add_argument("command", choices=["summary", "latex", "report"])
    parser.add_argument("experiment_ids", nargs="+", help="Experiment IDs to analyze")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--caption", default="Model Performance Comparison", help="LaTeX table caption")
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    analyzer = ExperimentAnalyzer(manager)
    
    if args.command == "summary":
        df = analyzer.create_results_summary(args.experiment_ids)
        if not df.empty:
            print(df.to_string(index=False))
            
            if args.output:
                df.to_csv(args.output, index=False)
                print(f"📊 Saved summary to: {args.output}")
        else:
            print("No data found for analysis")
    
    elif args.command == "latex":
        latex_table = analyzer.create_latex_table(args.experiment_ids, caption=args.caption)
        print(latex_table)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(latex_table)
            print(f"📄 Saved LaTeX table to: {args.output}")
    
    elif args.command == "report":
        report = analyzer.create_summary_report(args.experiment_ids)
        print(report)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"📄 Saved report to: {args.output}")

if __name__ == "__main__":
    main() 
"""
Module for saving benchmark results.
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np


def _convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to native Python types
    """
    # Check for numpy integer types (compatible with NumPy 1.x and 2.x)
    if isinstance(obj, np.integer):
        return int(obj)
    # Check for numpy floating types (compatible with NumPy 1.x and 2.x)
    elif isinstance(obj, np.floating):
        return float(obj)
    # Check for numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle dictionaries recursively
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    # Handle lists and tuples recursively
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_results(
    results: List[Dict[str, float]],
    output_dir: str = "artifacts/tokenizer_benchmarks",
    format: str = "json",
) -> str:
    """
    Save benchmark results to files.
    
    Args:
        results: List of benchmark result dictionaries
        output_dir: Directory to save results
        format: Output format ("json", "csv", or "both")
        
    Returns:
        Path to the saved results file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    # Convert numpy types to native Python types for JSON serialization
    results_serializable = _convert_numpy_types(results)
    
    if format in ["json", "both"]:
        json_path = os.path.join(output_dir, f"results_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {json_path}")
    
    if format in ["csv", "both"]:
        csv_path = os.path.join(output_dir, f"results_{timestamp}.csv")
        _save_csv(results_serializable, csv_path)
        print(f"Results saved to: {csv_path}")
    
    # Also save a summary comparison
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.txt")
    _save_summary(results_serializable, summary_path)
    print(f"Summary saved to: {summary_path}")
    
    return json_path if format in ["json", "both"] else csv_path


def _save_csv(results: List[Dict[str, float]], csv_path: str):
    """Save results as CSV."""
    import csv
    
    if not results:
        return
    
    fieldnames = list(results[0].keys())
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _save_summary(results: List[Dict[str, float]], summary_path: str):
    """Save a human-readable summary."""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Tokenizer Benchmark Summary\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"Tokenizer: {result['name']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Vocabulary size: {result['vocab_size']}\n")
            f.write(f"  Total tokens: {result['total_tokens']:,}\n")
            f.write(f"  Average tokens per text: {result['avg_tokens_per_text']:.2f}\n")
            f.write(f"  Average tokens per word: {result['avg_tokens_per_word']:.2f}\n")
            f.write(f"  Average tokens per char: {result['avg_tokens_per_char']:.4f}\n")
            f.write(f"  Compression ratio: {result['compression_ratio']:.2f}\n")
            f.write(f"  Vocabulary usage: {result['vocab_usage']:,} ({result['vocab_usage_ratio']*100:.1f}%)\n")
            f.write(f"  Tokenization speed: {result['tokens_per_second']:.0f} tokens/sec\n")
            f.write(f"  Tokenization time: {result['tokenization_time_seconds']:.2f} seconds\n")
            f.write("\n")
        
        # Comparison section
        f.write("\nComparison\n")
        f.write("=" * 80 + "\n")
        
        if len(results) > 1:
            # Find best compression (lowest tokens)
            best_compression = min(results, key=lambda x: x["total_tokens"])
            f.write(f"Best compression (lowest tokens): {best_compression['name']}\n")
            f.write(f"  Total tokens: {best_compression['total_tokens']:,}\n\n")
            
            # Find fastest
            fastest = max(results, key=lambda x: x["tokens_per_second"])
            f.write(f"Fastest tokenizer: {fastest['name']}\n")
            f.write(f"  Speed: {fastest['tokens_per_second']:.0f} tokens/sec\n\n")
            
            # Find best vocab usage
            best_vocab = max(results, key=lambda x: x["vocab_usage_ratio"])
            f.write(f"Best vocabulary usage: {best_vocab['name']}\n")
            f.write(f"  Usage: {best_vocab['vocab_usage_ratio']*100:.1f}%\n\n")


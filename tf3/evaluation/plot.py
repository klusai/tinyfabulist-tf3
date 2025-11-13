import os
import re
import sys
from typing import Dict, List, Tuple

import matplotlib

# Use non-interactive backend for SSH
matplotlib.use("Agg")

import matplotlib.pyplot as plt

LOG_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "tf3", "artifacts", "logs", "evaluation.log"
)

# Path to mlx_cache directory (relative to workspace root)
MLX_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "mlx_cache"
)

# Model name mapping for display
MODEL_NAMES = {
    "tf3-50m-base-mlx": "transformers",
    "tf3-50m-base-mamba-mlx": "mamba",
    "tf3-50m-base-q9-mlx": "transformers-q9",
    "tf3-50m-base-q6-mlx": "transformers-q6",
}


def extract_model_size(model_name: str) -> float:
    """Extract model size in MB from mlx_cache directory.
    Calculates the total size of all files in the model directory.
    """
    # Try to find the model directory in mlx_cache
    model_dir = os.path.join(MLX_CACHE_DIR, model_name)
    
    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        # If exact match not found, try to find a directory that contains the model name
        if os.path.exists(MLX_CACHE_DIR):
            for item in os.listdir(MLX_CACHE_DIR):
                item_path = os.path.join(MLX_CACHE_DIR, item)
                if os.path.isdir(item_path) and model_name in item:
                    model_dir = item_path
                    break
            else:
                return float('nan')  # Model directory not found
        else:
            return float('nan')  # mlx_cache directory doesn't exist
    
    # Calculate total size of all files in the directory
    total_size_bytes = 0
    try:
        for dirpath, dirnames, filenames in os.walk(model_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total_size_bytes += os.path.getsize(filepath)
        
        # Convert bytes to MB
        size_mb = total_size_bytes / (1024 ** 2)
        return size_mb
    except (OSError, PermissionError):
        return float('nan')  # Error accessing files


def parse_log(file_path: str) -> Dict[str, Dict[str, float]]:
    """Parse log file and extract metrics per model."""
    metrics: Dict[str, Dict[str, float]] = {}

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # CE and PPL: "model-name, CE: 0.8712, PPL: 2.3897"
            match = re.search(r"\]\s+([^,]+),\s+CE:\s+([0-9.]+),\s+PPL:\s+([0-9.]+)", line)
            if match:
                model = match.group(1).strip()
                metrics.setdefault(model, {})["ce"] = float(match.group(2))
                metrics.setdefault(model, {})["ppl"] = float(match.group(3))
            
            # Entity Coherence: "model-name, Entity Coherence: 0.8995"
            match = re.search(r"\]\s+([^,]+),\s+Entity Coherence:\s+([0-9.]+)", line)
            if match:
                model = match.group(1).strip()
                metrics.setdefault(model, {})["entity"] = float(match.group(2))
            
            # Throughput: "model-name, Throughput: 84.0032"
            match = re.search(r"\]\s+([^,]+),\s+Throughput:\s+([0-9.]+)", line)
            if match:
                model = match.group(1).strip()
                metrics.setdefault(model, {})["throughput"] = float(match.group(2))
            
            # LanguageTool: "model-name, LanguageTool: 0.8500"
            match = re.search(r"\]\s+([^,]+),\s+LanguageTool:\s+([0-9.]+)", line)
            if match:
                model = match.group(1).strip()
                metrics.setdefault(model, {})["langtool"] = float(match.group(2))
            
            # LLM Grammar with detailed metrics: "model-name, LLM Grammar: 88.8250, Fluency: 88.1000, Coherence: 89.5500, Mistakes: 257"
            # Try to match the full format first
            match = re.search(
                r"\]\s+([^,]+),\s+LLM Grammar:\s+([0-9.]+),\s+Fluency:\s+([0-9.]+),\s+Coherence:\s+([0-9.]+),\s+Mistakes:\s+([0-9]+)",
                line
            )
            if match:
                model = match.group(1).strip()
                metrics.setdefault(model, {})["llm_grammar"] = float(match.group(2))
                metrics[model]["llm_fluency"] = float(match.group(3))
                metrics[model]["llm_coherence"] = float(match.group(4))
                metrics[model]["llm_mistakes"] = float(match.group(5))
            else:
                # Fallback: match just LLM Grammar without detailed metrics
                match = re.search(r"\]\s+([^,]+),\s+LLM Grammar:\s+([0-9.]+)", line)
                if match:
                    model = match.group(1).strip()
                    metrics.setdefault(model, {})["llm_grammar"] = float(match.group(2))
    
    # Extract model sizes for all models found
    for model_name in metrics.keys():
        model_size = extract_model_size(model_name)
        if not (model_size != model_size):  # Check if not NaN
            metrics[model_name]["model_size"] = model_size
    
    return metrics


def scale_axis_for_visibility(values: List[float], padding: float = 0.8) -> Tuple[float, float]:
    """Scale y-axis to focus on the range of values, making small differences more visible."""
    valid_values = [v for v in values if not (v != v)]  # Filter out NaN
    if not valid_values:
        return 0.0, 1.0
    
    min_val = min(valid_values)
    max_val = max(valid_values)
    
    # If values are very close, use a fixed range around the mean
    if max_val - min_val < 0.01:
        mean_val = sum(valid_values) / len(valid_values)
        range_size = max(0.05, abs(mean_val) * 0.1)  # At least 5% range or 10% of mean
        return mean_val - range_size, mean_val + range_size
    
    # Add padding
    range_size = max_val - min_val
    padding_size = range_size * padding
    return min_val - padding_size, max_val + padding_size


def plot_metrics(metrics: Dict[str, Dict[str, float]], output_file: str = "metrics_plot.png") -> None:
    """Plot metrics as separate bar charts with value labels and scaled axes."""
    if not metrics:
        print("No metrics found.")
        return
    
    models = list(metrics.keys())
    # Map model names to display names
    display_names = [MODEL_NAMES.get(m, m) for m in models]
    x = range(len(models))
    width = 0.35  # Width of bars
    
    # Determine layout based on available metrics
    available_metrics = []
    if any("model_size" in metrics[m] for m in models):
        available_metrics.append("model_size")
    if any("ce" in metrics[m] or "ppl" in metrics[m] for m in models):
        available_metrics.append("ce_ppl")
    if any("entity" in metrics[m] for m in models):
        available_metrics.append("entity")
    if any("throughput" in metrics[m] for m in models):
        available_metrics.append("throughput")
    if any("langtool" in metrics[m] for m in models):
        available_metrics.append("langtool")
    if any("llm_grammar" in metrics[m] for m in models):
        available_metrics.append("llm_grammar")
    if any("llm_fluency" in metrics[m] or "llm_coherence" in metrics[m] for m in models):
        available_metrics.append("llm_fluency_coherence")
    if any("llm_mistakes" in metrics[m] for m in models):
        available_metrics.append("llm_mistakes")
    
    num_plots = len(available_metrics)
    if num_plots == 0:
        print("No metrics to plot.")
        return
    
    # Create subplots (2 columns)
    rows = (num_plots + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot 0: Model Size
    if "model_size" in available_metrics:
        model_sizes = [metrics[m].get("model_size", float("nan")) for m in models]
        
        ax = axes[plot_idx]
        bars = ax.bar(x, model_sizes, width=0.6, alpha=0.8, color="darkblue")
        ax.set_ylabel("Size (MB)")
        ax.set_title("Model Size")
        if plot_idx >= num_plots - 2:
            ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=30, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Scale y-axis for visibility
        y_min, y_max = scale_axis_for_visibility(model_sizes)
        ax.set_ylim(y_min, y_max)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not (height != height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f} MB', ha='center', va='bottom', fontsize=8)
        plot_idx += 1
    
    # Plot 1: CE + PPL together
    if "ce_ppl" in available_metrics:
        ce = [metrics[m].get("ce", float("nan")) for m in models]
        ppl = [metrics[m].get("ppl", float("nan")) for m in models]
        
        ax = axes[plot_idx]
        x1 = [i - width/2 for i in x]
        x2 = [i + width/2 for i in x]
        bars1 = ax.bar(x1, ce, width, label="CE", alpha=0.8, color="steelblue")
        bars2 = ax.bar(x2, ppl, width, label="PPL", alpha=0.8, color="coral")
        ax.set_ylabel("Value")
        ax.set_title("Cross-Entropy (CE) and Perplexity (PPL) ↓ Lower is better")
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=30, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()
        
        # Scale y-axis for visibility
        all_values = ce + ppl
        y_min, y_max = scale_axis_for_visibility(all_values)
        ax.set_ylim(y_min, y_max)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if not (height != height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            if not (height != height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        plot_idx += 1
    
    # Plot 2: Entity Coherence
    if "entity" in available_metrics:
        entity = [metrics[m].get("entity", float("nan")) for m in models]
        
        ax = axes[plot_idx]
        bars = ax.bar(x, entity, width=0.6, alpha=0.8, color="purple")
        ax.set_ylabel("Value")
        ax.set_title("Entity Coherence ↑ Higher is better")
        if plot_idx >= num_plots - 2:
            ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=30, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Scale y-axis for visibility
        y_min, y_max = scale_axis_for_visibility(entity)
        ax.set_ylim(y_min, y_max)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not (height != height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        plot_idx += 1
    
    # Plot 3: Throughput
    if "throughput" in available_metrics:
        throughput = [metrics[m].get("throughput", float("nan")) for m in models]
        
        ax = axes[plot_idx]
        bars = ax.bar(x, throughput, width=0.6, alpha=0.8, color="green")
        ax.set_ylabel("Throughput (tokens/sec)")
        ax.set_title("Throughput ↑ Higher is better")
        if plot_idx >= num_plots - 2:
            ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=30, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Scale y-axis for visibility
        y_min, y_max = scale_axis_for_visibility(throughput)
        ax.set_ylim(y_min, y_max)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not (height != height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        plot_idx += 1
    
    # Plot 4: LanguageTool
    if "langtool" in available_metrics:
        langtool = [metrics[m].get("langtool", float("nan")) for m in models]
        
        ax = axes[plot_idx]
        bars = ax.bar(x, langtool, width=0.6, alpha=0.8, color="orange")
        ax.set_ylabel("Score")
        ax.set_title("LanguageTool Grammar ↑ Higher is better")
        if plot_idx >= num_plots - 2:
            ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=30, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Scale y-axis for visibility
        y_min, y_max = scale_axis_for_visibility(langtool)
        ax.set_ylim(y_min, y_max)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not (height != height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        plot_idx += 1
    
    # Plot 5: LLM Grammar
    if "llm_grammar" in available_metrics:
        llm_grammar = [metrics[m].get("llm_grammar", float("nan")) for m in models]
        
        ax = axes[plot_idx]
        bars = ax.bar(x, llm_grammar, width=0.6, alpha=0.8, color="teal")
        ax.set_ylabel("Score")
        ax.set_title("LLM Evaluation (Average) ↑ Higher is better")
        if plot_idx >= num_plots - 2:
            ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=30, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Scale y-axis for visibility
        y_min, y_max = scale_axis_for_visibility(llm_grammar)
        ax.set_ylim(y_min, y_max)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not (height != height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        plot_idx += 1
    
    # Plot 6: LLM Fluency and Coherence together
    if "llm_fluency_coherence" in available_metrics:
        fluency = [metrics[m].get("llm_fluency", float("nan")) for m in models]
        coherence = [metrics[m].get("llm_coherence", float("nan")) for m in models]
        
        ax = axes[plot_idx]
        x1 = [i - width/2 for i in x]
        x2 = [i + width/2 for i in x]
        bars1 = ax.bar(x1, fluency, width, label="Fluency", alpha=0.8, color="skyblue")
        bars2 = ax.bar(x2, coherence, width, label="Coherence", alpha=0.8, color="lightcoral")
        ax.set_ylabel("Score")
        ax.set_title("LLM Evaluation - Fluency and Coherence ↑ Higher is better")    
        if plot_idx >= num_plots - 2:
            ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=30, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()
        
        # Scale y-axis for visibility
        all_values = fluency + coherence
        y_min, y_max = scale_axis_for_visibility(all_values)
        ax.set_ylim(y_min, y_max)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if not (height != height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            if not (height != height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        plot_idx += 1
    
    # Plot 7: LLM Mistakes
    if "llm_mistakes" in available_metrics:
        mistakes = [metrics[m].get("llm_mistakes", float("nan")) for m in models]
        
        ax = axes[plot_idx]
        bars = ax.bar(x, mistakes, width=0.6, alpha=0.8, color="red")
        ax.set_ylabel("Count")
        ax.set_title("LLM Evaluation - Mistakes ↓ Lower is better")
        if plot_idx >= num_plots - 2:
            ax.set_xlabel("Model")
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=30, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        
        # Scale y-axis for visibility
        y_min, y_max = scale_axis_for_visibility(mistakes)
        ax.set_ylim(y_min, y_max)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not (height != height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_file}")


def main():
    metrics = parse_log(LOG_FILE)
    plot_metrics(metrics)


if __name__ == "__main__":
    main()

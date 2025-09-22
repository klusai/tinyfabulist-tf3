import os
import re
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt


LOG_FILE = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "artifacts", "logs", "artifacts.log")


def parse_log(file_path: str) -> Dict[int, Dict[str, float]]:
    """Parse artifacts.log to collect CE, PPL, and Agree per step.

    Returns a mapping: step -> {"ce": float, "ppl": float, "agree": float}
    If duplicates occur, the last occurrence wins.
    """
    step_to_metrics: Dict[int, Dict[str, float]] = {}

    ce_ppl_re = re.compile(r"mamba50M(\d+),\s+CE:\s+([0-9.]+),\s+PPL:\s+([0-9.]+)")
    agree_re = re.compile(r"mamba50M(\d+),\s+Agree:\s+([0-9.]+)")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m1 = ce_ppl_re.search(line)
            if m1:
                step = int(m1.group(1))
                ce_val = float(m1.group(2))
                ppl_val = float(m1.group(3))
                entry = step_to_metrics.get(step, {})
                entry["ce"] = ce_val
                entry["ppl"] = ppl_val
                step_to_metrics[step] = entry
                continue

            m2 = agree_re.search(line)
            if m2:
                step = int(m2.group(1))
                agree_val = float(m2.group(2))
                entry = step_to_metrics.get(step, {})
                entry["agree"] = agree_val
                step_to_metrics[step] = entry
                continue

    return step_to_metrics


def build_series(step_to_metrics: Dict[int, Dict[str, float]]) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Build sorted series of steps, CE, PPL, Agree (missing values omitted)."""
    steps = sorted(step_to_metrics.keys())
    ce_series: List[float] = []
    ppl_series: List[float] = []
    agree_series: List[float] = []

    for s in steps:
        metrics = step_to_metrics[s]
        ce_series.append(metrics.get("ce", float("nan")))
        ppl_series.append(metrics.get("ppl", float("nan")))
        agree_series.append(metrics.get("agree", float("nan")))

    return steps, ce_series, ppl_series, agree_series


def main() -> None:
    data = parse_log(LOG_FILE)
    if not data:
        raise RuntimeError("No metrics found in the log file.")

    steps, ce, ppl, agree = build_series(data)

    fig, ax_left = plt.subplots(figsize=(9, 5))

    line_ce, = ax_left.plot(steps, ce, marker="o", label="Cross-Entropy (CE)")
    line_ppl, = ax_left.plot(steps, ppl, marker="s", label="Perplexity (PPL)")
    ax_left.set_xlabel("Checkpoint (Step)")
    ax_left.set_ylabel("CE / PPL")
    ax_left.grid(True, which="both", linestyle="--", alpha=0.4)

    ax_right = ax_left.twinx()
    line_agree, = ax_right.plot(steps, agree, color="#d62728", marker="^", label="Agree")
    ax_right.set_ylabel("Agree")

    # Combine legends from both axes
    lines = [line_ce, line_ppl, line_agree]
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc="best")

    ax_left.set_title("Training Metrics vs Checkpoints (CE, PPL, Agree)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

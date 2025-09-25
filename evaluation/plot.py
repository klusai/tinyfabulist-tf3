import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

LOG_FILES = ["tf3.log", "artifacts.log", "tf3.log.2025-09-22"]
APPENDED_LOG_FILE_TEMP = "appended_log.log"

for log_file in LOG_FILES:
    LOG_FILE = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        "artifacts",
        "logs",
        log_file,
    )
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            with open(APPENDED_LOG_FILE_TEMP, "a", encoding="utf-8") as f_appended:
                f_appended.write(line)


def parse_log(file_path: str) -> Dict[int, Dict[str, float]]:
    """Parse tf3.log to collect CE, PPL, Agree, Entity Coherence, and Throughput per step.

    Latest-wins semantics: we scan bottom-up and keep the first occurrence
    seen for each metric per step (i.e., the most recent in the file).

    Returns a mapping: step -> {"ce": float, "ppl": float, "agree": float, "entity": float, "throughput": float}
    """
    step_to_metrics: Dict[int, Dict[str, float]] = {}

    ce_ppl_re = re.compile(r"mamba50M(\d+),\s+CE:\s+([0-9.]+),\s+PPL:\s+([0-9.]+)")
    agree_re = re.compile(r"mamba50M(\d+),\s+Agree:\s+([0-9.]+)")
    entity_re = re.compile(r"mamba50M(\d+),\s+Entity Coherence:\s+([0-9.]+)")
    throughput_re = re.compile(r"mamba50M(\d+),\s+Throughput:\s+([0-9.]+)")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Iterate from the end to ensure we keep the latest values
    for raw_line in reversed(lines):
        line = raw_line.strip()
        if not line:
            continue

        m1 = ce_ppl_re.search(line)
        if m1:
            step = int(m1.group(1))
            ce_val = float(m1.group(2))
            ppl_val = float(m1.group(3))
            entry = step_to_metrics.setdefault(step, {})
            if "ce" not in entry:
                entry["ce"] = ce_val
            if "ppl" not in entry:
                entry["ppl"] = ppl_val
            continue

        m2 = agree_re.search(line)
        if m2:
            step = int(m2.group(1))
            agree_val = float(m2.group(2))
            entry = step_to_metrics.setdefault(step, {})
            if "agree" not in entry:
                entry["agree"] = agree_val
            continue

        m_entity = entity_re.search(line)
        if m_entity:
            step = int(m_entity.group(1))
            entity_val = float(m_entity.group(2))
            entry = step_to_metrics.setdefault(step, {})
            if "entity" not in entry:
                entry["entity"] = entity_val
            continue

        m3 = throughput_re.search(line)
        if m3:
            step = int(m3.group(1))
            tp_val = float(m3.group(2))
            entry = step_to_metrics.setdefault(step, {})
            if "throughput" not in entry:
                entry["throughput"] = tp_val
            continue

    return step_to_metrics


def build_series(
    step_to_metrics: Dict[int, Dict[str, float]],
) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float]]:
    """Build sorted series of steps, CE, PPL, Agree, Entity Coherence, Throughput (NaN if missing)."""
    steps = sorted(step_to_metrics.keys())
    ce_series: List[float] = []
    ppl_series: List[float] = []
    agree_series: List[float] = []
    entity_series: List[float] = []
    throughput_series: List[float] = []

    for s in steps:
        metrics = step_to_metrics[s]
        ce_series.append(metrics.get("ce", float("nan")))
        ppl_series.append(metrics.get("ppl", float("nan")))
        agree_series.append(metrics.get("agree", float("nan")))
        entity_series.append(metrics.get("entity", float("nan")))
        throughput_series.append(metrics.get("throughput", float("nan")))

    return steps, ce_series, ppl_series, agree_series, entity_series, throughput_series


def main() -> None:
    data = parse_log(APPENDED_LOG_FILE_TEMP)
    if not data:
        raise RuntimeError("No metrics found in the log file.")

    steps, ce, ppl, agree, entity, throughput = build_series(data)

    fig, (ax_main, ax_tp) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Main metrics plot (CE/PPL left, Agree/Entity right)
    (line_ce,) = ax_main.plot(steps, ce, marker="o", label="Cross-Entropy (CE)")
    (line_ppl,) = ax_main.plot(steps, ppl, marker="s", label="Perplexity (PPL)")
    ax_main.set_ylabel("CE / PPL")
    ax_main.grid(True, which="both", linestyle="--", alpha=0.4)

    ax_agree = ax_main.twinx()
    (line_agree,) = ax_agree.plot(
        steps, agree, color="#d62728", marker="^", label="Agree"
    )
    (line_entity,) = ax_agree.plot(
        steps, entity, color="#9467bd", marker="x", label="Entity Coherence"
    )
    ax_agree.set_ylabel("Agree / Entity")

    # Combine legends from both axes (top subplot)
    lines = [line_ce, line_ppl, line_agree, line_entity]
    labels = [l.get_label() for l in lines]
    ax_main.legend(lines, labels, loc="best")
    ax_main.set_title("Training Metrics vs Checkpoints (CE, PPL, Agree, Entity)")

    # Throughput subplot
    (line_tp,) = ax_tp.plot(
        steps, throughput, color="#2ca02c", marker="d", label="Throughput"
    )
    ax_tp.set_xlabel("Checkpoint (Step)")
    ax_tp.set_ylabel("Throughput (tokens/sec)")
    ax_tp.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_tp.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    os.remove(APPENDED_LOG_FILE_TEMP)

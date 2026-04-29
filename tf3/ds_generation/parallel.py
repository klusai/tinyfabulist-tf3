"""
Parallel 3M fable generation -- splits work across N workers.
Each worker loads its own model copy and generates a slice of combinations.

Usage:
    python -m tf3.ds_generation.parallel --workers 6
"""
import os
import sys
import json
import time
import random
import yaml
import argparse
import multiprocessing as mp
from pathlib import Path
from itertools import product

MODEL_PATH = "artifacts/transformers-final-sft"
OUTPUT_DIR = "artifacts/fable_batch_generation"
ENTITIES_YAML = "tf3/ds_generation/fable_entities.yaml"
TOTAL_FABLES = 3_000_000
BATCH_SIZE = 512
SHARD_SIZE = 5_000

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_entities(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_combinations(entities, total_needed):
    keys = ['personaj_principal', 'decor', 'provocare', 'deznodamant', 'invatatura']
    lists = [entities[k] for k in keys]
    all_combos = list(product(*lists))
    total_possible = len(all_combos)
    print(f"[Info] Total possible combinations: {total_possible:,}")

    if total_needed <= total_possible:
        random.seed(42)
        sampled = random.sample(all_combos, total_needed)
    else:
        sampled = all_combos * (total_needed // total_possible)
        remainder = total_needed - len(sampled)
        random.seed(42)
        sampled += random.sample(all_combos, remainder)

    return sampled


def worker_fn(worker_id, combo_slice, total_workers):
    from tf3.ds_generation.main import create_fable_prompt

    try:
        from mlx_lm import load, batch_generate
    except ImportError:
        print(f"[Worker {worker_id}] MLX not available!")
        return

    print(f"[Worker {worker_id}] Loading model...")
    model, tokenizer = load(str(Path(MODEL_PATH).resolve()))
    print(f"[Worker {worker_id}] Model loaded. Generating {len(combo_slice):,} fables.")

    total_written = 0
    shard_index = 0
    shard_data = []
    combination_idx = 0
    overall_start = time.time()

    while combination_idx < len(combo_slice):
        batch_start = time.time()

        end_idx = min(combination_idx + BATCH_SIZE, len(combo_slice))
        batch_combos = combo_slice[combination_idx:end_idx]

        prompts = []
        for personaj, decor, provocare, deznodamant, invatatura in batch_combos:
            prompt_text = create_fable_prompt(personaj, decor, provocare, deznodamant, invatatura)
            prompt_ids = tokenizer.encode(prompt_text)
            prompts.append(prompt_ids)

        resp = batch_generate(
            model,
            tokenizer,
            prompts=prompts,
            max_tokens=512,
            verbose=False,
        )

        outputs = resp.texts
        combination_idx += len(outputs)
        shard_data.extend(outputs)
        total_written += len(outputs)

        elapsed = time.time() - batch_start
        print(f"[Worker {worker_id}] {len(outputs)} fables in {elapsed:.1f}s | total {total_written:,}/{len(combo_slice):,}")

        if len(shard_data) >= SHARD_SIZE or combination_idx >= len(combo_slice):
            out_path = os.path.join(OUTPUT_DIR, f"fables_w{worker_id:02d}_shard_{shard_index:05d}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for fab in shard_data:
                    f.write(json.dumps({"fable": fab}, ensure_ascii=False) + "\n")
            print(f"[Worker {worker_id}] Saved {len(shard_data)} fables -> {out_path}")
            shard_index += 1
            shard_data = []

    total_time = time.time() - overall_start
    print(f"[Worker {worker_id}] Done! {total_written:,} fables in {total_time/60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    print(f"[Main] Loading entities...")
    entities = load_entities(ENTITIES_YAML)

    print(f"[Main] Generating {TOTAL_FABLES:,} combinations...")
    combinations = generate_combinations(entities, TOTAL_FABLES)

    n = args.workers
    chunk_size = len(combinations) // n
    slices = []
    for i in range(n):
        start = i * chunk_size
        end = start + chunk_size if i < n - 1 else len(combinations)
        slices.append(combinations[start:end])

    print(f"[Main] Launching {n} workers, {chunk_size:,} fables each...")
    print(f"[Main] Output dir: {OUTPUT_DIR}")

    processes = []
    for i in range(n):
        p = mp.Process(target=worker_fn, args=(i, slices[i], n))
        p.start()
        processes.append(p)
        time.sleep(2)

    for p in processes:
        p.join()

    print(f"\n[Main] All workers finished!")
    total = sum(1 for f in Path(OUTPUT_DIR).glob("fables_w*.jsonl") for _ in open(f))
    print(f"[Main] Total fables generated: {total:,}")


if __name__ == "__main__":
    main()

import os
import json
import time
import random
import yaml
from pathlib import Path
from itertools import product

# ============================================================
# Try MLX first
# ============================================================
MLX_AVAILABLE = True
try:
    from mlx_lm import load, batch_generate
    import mlx.core as mx
except ImportError:
    MLX_AVAILABLE = False


# ============================================================
# 1. Configuration
# ============================================================
MODEL_PATH = "artifacts/transformers-final-sft"
OUTPUT_DIR = "artifacts/fable_batch_generation"
ENTITIES_YAML = "tf3/ds_generation/fable_entities.yaml"
TOTAL_FABLES = 3_000_000
CHUNK_SIZE = 5_000             # prompts sent to batch_generate at once
MAX_TOKENS = 512               
SHARD_SIZE = 5_000             # write file every 5k fables

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load entities from YAML
# ============================================================
def load_entities(yaml_path):
    """Load fable entities from YAML file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        entities = yaml.safe_load(f)
    return entities


# ============================================================
# 3. Generate all combinations and sample 3M unique ones
# ============================================================
def generate_combinations(entities, total_needed):
    """Generate all possible combinations and sample unique ones."""
    # Get all entity lists
    personaje = entities['personaj_principal']
    decoruri = entities['decor']
    provocari = entities['provocare']
    deznodamante = entities['deznodamant']
    invataturi = entities['invatatura']
    
    # Calculate total possible combinations
    total_combinations = len(personaje) * len(decoruri) * len(provocari) * len(deznodamante) * len(invataturi)
    print(f"[Info] Total possible combinations: {total_combinations:,}")
    print(f"[Info] Need to generate: {total_needed:,}")
    
    # Generate all combinations
    all_combinations = list(product(personaje, decoruri, provocari, deznodamante, invataturi))
    
    # If we have more combinations than needed, sample randomly
    if len(all_combinations) >= total_needed:
        print(f"[Info] Sampling {total_needed:,} random combinations...")
        random.seed(42)  # For reproducibility
        selected = random.sample(all_combinations, total_needed)
    else:
        # If we have fewer combinations, repeat them randomly
        print(f"[Info] Only {len(all_combinations):,} unique combinations available. Will repeat randomly...")
        selected = []
        random.seed(42)
        while len(selected) < total_needed:
            selected.extend(random.sample(all_combinations, min(len(all_combinations), total_needed - len(selected))))
        selected = selected[:total_needed]
    
    print(f"[Info] Generated {len(selected):,} unique combinations")
    return selected


# ============================================================
# 4. Create prompt template function
# ============================================================
def create_fable_prompt(personaj, decor, provocare, deznodamant, invatatura):
    """Create a fable prompt from entity combination."""
    template = f"""Creează o fabulă bazată pe următoarele elemente. Împletește-le natural într-o poveste:
- Personaj principal: {personaj}
- Decor: {decor}
- Provocare: {provocare}
- Deznodământ: {deznodamant}
- Învățătură: {invatatura}

Fabulă trebuie:
- Să fie potrivită pentru grupa de vârstă B (4-7 ani)
- Să folosească un vocabular simplu pe care copiii de 4-7 ani îl pot înțelege
- Să folosească un limbaj concret, nu abstract
- Să înceapă cu o descriere vie a scenei
- Să nu folosească nume pentru personaje, ci trăsătura și tipul lor
- Să includă dialog semnificativ, dar simplu
- Să arate (nu să spună direct) dezvoltarea personajului
- Să se încheie cu o legătură clară la morală
- Să fie concisă, captivantă, aprox. 250 de cuvinte.

Scrie fabula acum:
"""
    return template.strip()


# ============================================================
# 3. Load model (MLX)
# ============================================================
def load_mlx():
    print(f"[MLX] Loading model from {MODEL_PATH}")
    model, tokenizer = load(str(Path(MODEL_PATH).resolve()))
    return model, tokenizer


# ============================================================
# 5. Generate a chunk using continuous batching
# ============================================================
def generate_chunk(model, tokenizer, combinations, start_idx, chunk_size):
    """Generate fables using MLX's continuous batching engine."""
    end_idx = min(start_idx + chunk_size, len(combinations))
    batch_combinations = combinations[start_idx:end_idx]

    prompts = []
    for personaj, decor, provocare, deznodamant, invatatura in batch_combinations:
        prompt_text = create_fable_prompt(personaj, decor, provocare, deznodamant, invatatura)
        prompts.append(tokenizer.encode(prompt_text))

    resp = batch_generate(
        model,
        tokenizer,
        prompts=prompts,
        max_tokens=MAX_TOKENS,
        completion_batch_size=32,
    )

    return resp.texts, resp.stats


# ============================================================
# 6. Main generation loop
# ============================================================
def run_generation(worker_id=0, total_workers=1):
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not installed. Run: pip install mlx-lm")

    tag = f"[W{worker_id}]" if total_workers > 1 else "[Gen]"

    print(f"{tag} Loading entities from {ENTITIES_YAML}...")
    entities = load_entities(ENTITIES_YAML)
    print(f"{tag} Loaded entities:")
    print(f"  - Personaje: {len(entities['personaj_principal'])}")
    print(f"  - Decoruri: {len(entities['decor'])}")
    print(f"  - Provocări: {len(entities['provocare'])}")
    print(f"  - Deznodăminte: {len(entities['deznodamant'])}")
    print(f"  - Învățături: {len(entities['invatatura'])}")

    print(f"\n{tag} Generating {TOTAL_FABLES:,} unique combinations...")
    combinations = generate_combinations(entities, TOTAL_FABLES)

    chunk_size = len(combinations) // total_workers
    start = worker_id * chunk_size
    end = start + chunk_size if worker_id < total_workers - 1 else len(combinations)
    combinations = combinations[start:end]
    target = len(combinations)
    print(f"{tag} Worker {worker_id}/{total_workers}: generating fables {start:,}-{end:,} ({target:,} total)")

    model, tokenizer = load_mlx()

    total_written = 0
    shard_index = 0
    combination_idx = 0
    overall_start = time.time()

    while total_written < target:
        n = min(CHUNK_SIZE, target - total_written)
        chunk_start = time.time()

        outputs, stats = generate_chunk(model, tokenizer, combinations, combination_idx, n)
        combination_idx += len(outputs)
        total_written += len(outputs)

        elapsed = time.time() - chunk_start
        fps = len(outputs) / elapsed
        eta_h = (target - total_written) / fps / 3600 if fps > 0 else 0
        print(f"{tag} {len(outputs):,} fables in {elapsed:.1f}s ({fps:.1f}/s, {stats.generation_tps:.0f} tok/s) | {total_written:,}/{target:,} | ETA {eta_h:.1f}h")

        out_path = os.path.join(OUTPUT_DIR, f"fables_w{worker_id:02d}_shard_{shard_index:05d}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for fab in outputs:
                f.write(json.dumps({"fable": fab}, ensure_ascii=False) + "\n")
        print(f"{tag} Saved {len(outputs)} fables -> {out_path}")
        shard_index += 1

    total_time = time.time() - overall_start
    print(f"\n{tag} Done! {total_written:,} fables in {total_time/3600:.1f}h ({total_written/total_time:.1f}/s)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--total-workers", type=int, default=1)
    args = parser.parse_args()
    run_generation(worker_id=args.worker_id, total_workers=args.total_workers)

import os
import json
import time
import random
import yaml
import hashlib
from datetime import datetime
from pathlib import Path
from itertools import product

# ============================================================
# Try MLX first
# ============================================================
MLX_AVAILABLE = True
try:
    from mlx_lm import load
    from mlx_lm.tokenizer_utils import load_tokenizer
    from mlx_lm.generate import batch_generate
    import mlx.core as mx
except ImportError:
    MLX_AVAILABLE = False


# ============================================================
# 1. Configuration
# ============================================================
MODEL_PATH = "artifacts/transformers-50m-base-sft"
OUTPUT_DIR = "artifacts/fable_batch_f_generation"
ENTITIES_YAML = "tf3/ds_generation/fable_entities.yaml"
TOTAL_FABLES = 100
BATCH_SIZE = 1024              
MAX_TOKENS = 450               # ~250 words, reduced from 640 to prevent rambling
SHARD_SIZE = 1_000            # write file every 1k fables

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
    template = f"""Creeaza o poveste despre {personaj}.
  - Personaj principal: {personaj}
  - Decor: {decor}
  - Provocare: {provocare}
  - Deznodământ: {deznodamant}
  - Învățătură: {invatatura}
Fabula ar trebui:
  - Să fie potrivită pentru grupa de vârstă B (4-7 ani)
  - Să folosească un vocabular simplu pe care copiii de 4-7 ani îl pot înțelege
  - Să folosească limbaj concret, nu abstract
  - Să înceapă cu o descriere vie a decorului
  - Să nu folosească nume pentru personaje, ci trăsătura și tipul personajului
  - Să includă dialog semnificativ, dar simplu
  - Să arate (nu să spună) evoluția personajului
  - Să se încheie cu o legătură clară la morală
  Păstrează povestea concisă, dar captivantă, în jur de 250 de cuvinte.
"""
    return template.strip()


def compute_prompt_hash(prompt_text):
    """Compute SHA256 hash of prompt text."""
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()


# ============================================================
# 3. Load model (MLX)
# ============================================================
def load_mlx():
    print(f"[MLX] Loading model from {MODEL_PATH}")
    model, config = load(str(Path(MODEL_PATH).resolve()))
    tokenizer = load_tokenizer(
        Path(MODEL_PATH).resolve(),
        # Don't override eos_token_ids - let it use the model's default EOS token
        tokenizer_config_extra={"trust_remote_code": True},
    )
    return model, tokenizer


# ============================================================
# 5. Generate one batch with unique prompts
# ============================================================
def generate_batch(model, tokenizer, combinations, start_idx, batch_size):
    """Generate a batch of fables using unique combinations."""
    end_idx = min(start_idx + batch_size, len(combinations))
    batch_combinations = combinations[start_idx:end_idx]
    
    # Create unique prompts for each combination
    prompts = []
    prompt_texts = []
    for personaj, decor, provocare, deznodamant, invatatura in batch_combinations:
        prompt_text = create_fable_prompt(personaj, decor, provocare, deznodamant, invatatura)
        prompt_ids = tokenizer.encode(prompt_text)
        prompts.append(prompt_ids)
        prompt_texts.append(prompt_text)

    resp = batch_generate(
        model,
        tokenizer,
        prompts=prompts,
        max_tokens=MAX_TOKENS,
        verbose=False,
    )

    # Handle different BatchResponse structures
    # Try different possible attribute names
    if hasattr(resp, 'generations'):
        outputs = [o["generated_text"] for o in resp.generations]
    elif hasattr(resp, 'texts'):
        outputs = resp.texts
    elif hasattr(resp, 'outputs'):
        outputs = resp.outputs
    elif isinstance(resp, list):
        # If resp is directly a list
        outputs = [o["generated_text"] if isinstance(o, dict) else o for o in resp]
    else:
        # Try to access as iterable or decode manually
        # Check if resp has a __iter__ method
        try:
            outputs = list(resp)
        except TypeError:
            # Last resort: try to get generated texts from response
            # This might require decoding token IDs if available
            raise AttributeError(f"BatchResponse object doesn't have expected attributes. Available attributes: {dir(resp)}")
    
    # Return both outputs and prompt_texts
    return outputs, prompt_texts


# ============================================================
# 6. Main batching loop
# ============================================================
def run_generation():
    if not MLX_AVAILABLE:
        raise RuntimeError("MLX not installed. Run: pip install mlx-lm")

    # Load entities from YAML
    print(f"[Config] Loading entities from {ENTITIES_YAML}...")
    entities = load_entities(ENTITIES_YAML)
    print(f"[Config] Loaded entities:")
    print(f"  - Personaje: {len(entities['personaj_principal'])}")
    print(f"  - Decoruri: {len(entities['decor'])}")
    print(f"  - Provocări: {len(entities['provocare'])}")
    print(f"  - Deznodăminte: {len(entities['deznodamant'])}")
    print(f"  - Învățături: {len(entities['invatatura'])}")
    
    # Generate all combinations
    print(f"\n[Config] Generating {TOTAL_FABLES:,} unique combinations...")
    combinations = generate_combinations(entities, TOTAL_FABLES)
    
    # Load model
    model, tokenizer = load_mlx()

    total_written = 0
    shard_index = 0
    shard_data = []
    combination_idx = 0

    overall_start = time.time()

    while total_written < TOTAL_FABLES:
        batch_start = time.time()
        
        # Generate batch with unique combinations
        outputs, prompt_texts = generate_batch(model, tokenizer, combinations, combination_idx, BATCH_SIZE)
        combination_idx += len(outputs)
        
        # Create full records with metadata
        for fable, prompt in zip(outputs, prompt_texts):
            record = {
                "fable": fable,
                "lang": "ro",
                "prompt": prompt,
                "prompt_hash": compute_prompt_hash(prompt),
                "generation_timestamp": datetime.utcnow().isoformat() + "Z",
                "llm_name": MODEL_PATH,
                "pipeline_stage": "generation"
            }
            shard_data.append(record)

        total_written += len(outputs)
        print(f"[Batch] Generated {len(outputs)} fables in {time.time() - batch_start:.2f}s → total {total_written:,} ({combination_idx:,}/{len(combinations):,} combinations)")

        # Write shard
        if len(shard_data) >= SHARD_SIZE or total_written >= TOTAL_FABLES:
            out_path = os.path.join(OUTPUT_DIR, f"fables_shard_{shard_index:05d}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for record in shard_data:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"[Shard] Saved {len(shard_data)} fables → {out_path}")
            shard_index += 1
            shard_data = []

    total_time = time.time() - overall_start
    print(f"\n[Done] Generated {total_written:,} fables in {total_time/60:.1f} minutes")
    print(f"[Done] Used {combination_idx:,} unique combinations")


if __name__ == "__main__":
    run_generation()

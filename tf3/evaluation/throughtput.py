import os
import time

MLX_AVAILABLE = True
try:
    from mlx_lm import load
    from mlx_lm.tokenizer_utils import load_tokenizer
    import mlx.core as mx
    from mlx_lm.generate import batch_generate
except ImportError:
    MLX_AVAILABLE = False

from pathlib import Path


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_throughput(
    model_path: str,
    device: str,
    max_new_tokens: int = 2000,
    batch_size: int = 30,
    prompt_tokens: int = 32,
    trust_remote_code: bool = False,
):
    """
    Test the throughput of a model using MLX with batching.
    """
    # if not MLX_LM_AVAILABLE:
    #     # Fallback to PyTorch if MLX is not available
    #     return _test_torch_throughput(model_path, device, max_new_tokens)
    
    # Use get_model_path to handle both local paths and HF repos
    if MLX_AVAILABLE:
        model_path_actual = Path("./mlx_cache/" + model_path.split("/")[-1] + "-mlx/")
        print(model_path_actual)
        model, config = load(model_path_actual)
        tokenizer = load_tokenizer(
            model_path_actual,
            eos_token_ids=[],  # disable early stop
            tokenizer_config_extra={"trust_remote_code": trust_remote_code},
        )

        # Random batch of prompts as token ids: List[List[int]] of shape [batch_size, prompt_len]
        prompts = mx.random.randint(0, 200, (batch_size, prompt_tokens)).tolist()
        
        # Benchmark generation
        start = time.perf_counter()
        resp = batch_generate(
            model,
            tokenizer,
            prompts=prompts,
            max_tokens=max_new_tokens,
            verbose=False,
        )
        elapsed = time.perf_counter() - start

        # Use library stats when available; fall back to wall time otherwise
        gen_tokens = resp.stats.generation_tokens
        gen_tps = resp.stats.generation_tps if resp.stats.generation_time > 0 else gen_tokens / max(elapsed, 1e-6)
        
        return float(gen_tps)
    else:
        return _test_torch_throughput(model_path, device, max_new_tokens)


def _test_torch_throughput(
    model_path: str,
    device: str,
    max_new_tokens: int = 1000,
):
    """Test throughput using PyTorch (fallback)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)

    # example prompt
    prompt = "Iepurele si pisica."

    # tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    # Some models (e.g., LLaMA) do not accept token_type_ids; drop if present
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    inputs = inputs.to(device)

    # warm-up (to avoid cold start skew)
    _ = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    # benchmark
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False  # deterministic, avoids sampling overhead
        )

    end = time.time()

    # count generated tokens
    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    elapsed = end - start
    throughput = generated_tokens / elapsed

    return throughput

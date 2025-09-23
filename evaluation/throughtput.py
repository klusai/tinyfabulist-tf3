import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# load model + tokenizer

def test_throughput(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
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
    max_new_tokens = 1000
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

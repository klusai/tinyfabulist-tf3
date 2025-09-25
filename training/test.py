import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation.stopping_criteria import StoppingCriteria


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="tf3/artifacts/training/checkpoints/mamba50MF/checkpoint-28200",
    )
    parser.add_argument("--prompt", type=str, default="Un iepure")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    import threading
    import time

    # Run on GPU if available
    import torch

    # Enable TF32 on CUDA for speed without big accuracy loss
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prefer bf16 on GPUs that support it
    preferred_dtype = torch.bfloat16 if device == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=preferred_dtype
    )
    model.to(device)
    model.eval()

    prompt = args.prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Remove token_type_ids if present
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    # Streamer for incremental text output (disable cleanup to avoid dropping spaces)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Token counter via StoppingCriteria
    class TokenCounter(StoppingCriteria):
        def __init__(self, start_len: int):
            self.start_len = start_len
            self.max_len_seen = start_len

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
        ) -> bool:
            cur_len = input_ids.shape[1]
            if cur_len > self.max_len_seen:
                self.max_len_seen = cur_len
            return False

        @property
        def new_tokens(self) -> int:
            return int(self.max_len_seen - self.start_len)

    start_len = inputs["input_ids"].shape[1]
    counter = TokenCounter(start_len)

    # Start generation in a background thread to allow streaming consumption
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=500,
        do_sample=False,
        temperature=0.5,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        streamer=streamer,
        stopping_criteria=[counter],
    )

    def _generate():
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model.generate(**gen_kwargs)
            else:
                model.generate(**gen_kwargs)

    start_time = time.time()
    thread = threading.Thread(target=_generate)
    thread.start()

    # Consume streamed text
    for text in streamer:
        print(text, end="", flush=True)

    # Ensure generation thread finishes
    thread.join()
    elapsed = time.time() - start_time

    # Report throughput
    gen_tokens = max(0, counter.new_tokens)
    if elapsed > 0:
        print(
            f"\n\nGenerated {gen_tokens} tokens in {elapsed:.2f}s ({gen_tokens/elapsed:.2f} tok/s)"
        )
    else:
        print(f"\n\nGenerated {gen_tokens} tokens (elapsed time too small to measure)")

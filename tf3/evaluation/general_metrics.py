"""
Compute Perplexity and Cross Entropy Loss.
"""

import argparse
import math
from pathlib import Path
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from mlx_lm import load as mlx_load
    from mlx_lm.tokenizer_utils import load_tokenizer
    import mlx.core as mx
    import mlx.nn as mlx_nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cross-entropy and perplexity for texts."
    )
    parser.add_argument(
        "--model", default="klusai/tf3-50M-base", help="HF model id or path"
    )
    parser.add_argument("--input", default=None, help="Single input text")
    parser.add_argument(
        "--file", default=None, help="Path to a text file (one sample per line)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max-length", type=int, default=2048, help="Max sequence length (tokens)"
    )
    return parser.parse_args()


def load_texts(args: argparse.Namespace) -> List[str]:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        return [ln for ln in lines if len(ln) > 0]
    if args.input is not None:
        return [args.input]
    # Default sample
    return ["Acesta este un exemplu de propoziție în limba română."]


def compute_ce_ppl_mlx(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
) -> Tuple[float, float]:
    """Compute CE and PPL for MLX models."""
    total_loss_sum = 0.0
    total_tokens = 0.0
    
    def loss_fn(logits, labels):
        """Compute cross-entropy loss."""
        log_probs = mlx_nn.log_softmax(logits, axis=-1)
        nll = -mx.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
        return nll
    
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        
        # Tokenize - MLX TokenizerWrapper forwards to underlying tokenizer
        # Access the underlying tokenizer for callable interface
        underlying_tokenizer = tokenizer._tokenizer if hasattr(tokenizer, '_tokenizer') else tokenizer
        encodings = underlying_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        
        input_ids = mx.array(encodings["input_ids"])
        attention_mask_np = encodings.get("attention_mask", None)
        if attention_mask_np is None:
            attention_mask = mx.ones_like(input_ids)
        else:
            attention_mask = mx.array(attention_mask_np)
        
        # Forward pass - MLX models return logits directly
        logits = model(input_ids)
        mx.eval(logits)  # Force evaluation of lazy computation
        
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_mask = attention_mask[:, 1:]
        
        # Compute loss
        losses = loss_fn(shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1))
        losses = losses.reshape(shift_labels.shape)
        mx.eval(losses)  # Force evaluation
        
        # Mask out padding tokens
        valid_mask = shift_mask.astype(mx.float32)
        masked_losses = losses * valid_mask
        mx.eval(masked_losses)  # Force evaluation
        
        total_loss_sum += float(mx.sum(masked_losses).item())
        total_tokens += float(mx.sum(valid_mask).item())
    
    overall_ce = total_loss_sum / max(total_tokens, 1.0)
    overall_ppl = math.exp(overall_ce)
    
    return overall_ce, overall_ppl


def compute_ce_ppl(
    model: Union[AutoModelForCausalLM, object],
    tokenizer: Union[AutoTokenizer, object],
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: Union[torch.device, None] = None,
) -> Tuple[float, float]:
    """Compute CE and PPL, supporting both PyTorch and MLX models."""
    # Check if this is an MLX model (no device means MLX)
    if device is None or (MLX_AVAILABLE and not isinstance(model, AutoModelForCausalLM)):
        return compute_ce_ppl_mlx(model, tokenizer, texts, batch_size, max_length)
    
    # PyTorch path
    model.eval()
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    # Overall (token-weighted)
    total_loss_sum = 0.0
    total_tokens = 0.0
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)

            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = enc["input_ids"][:, 1:].contiguous()
            shift_mask = (
                enc["attention_mask"][:, 1:].contiguous()
                if "attention_mask" in enc
                else torch.ones_like(shift_labels)
            )
            shift_labels = shift_labels.masked_fill(shift_mask == 0, -100)
            vocab_size = shift_logits.size(-1)
            loss_flat = loss_fct(
                shift_logits.view(-1, vocab_size), shift_labels.view(-1)
            )
            loss_tokens = loss_flat.view(shift_labels.size())
            valid_mask = (shift_labels != -100).float()
            total_loss_sum += float((loss_tokens * valid_mask).sum().item())
            total_tokens += float(valid_mask.sum().item())

    overall_ce = total_loss_sum / max(total_tokens, 1.0)
    overall_ppl = math.exp(overall_ce)

    return overall_ce, overall_ppl


def main():
    args = parse_args()

    # Check if this is an MLX model
    is_mlx_model = "mlx" in args.model.lower() and MLX_AVAILABLE
    
    if is_mlx_model:
        model_path_obj = Path(args.model)
        model, _ = mlx_load(model_path_obj)
        tokenizer = load_tokenizer(model_path_obj)
        device = None
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        # Prefer bf16 on GPU capable hardware
        torch_dtype = torch.bfloat16 if device.type == "cuda" else None
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch_dtype
        ).to(device)

    texts = load_texts(args)

    ce_all, ppl_all = compute_ce_ppl(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    print(f"Overall (token-weighted): CE={ce_all:.4f}, PPL={ppl_all:.4f}")


if __name__ == "__main__":
    main()

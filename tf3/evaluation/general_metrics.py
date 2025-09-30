"""
Compute Perplexity and Cross Entropy Loss.
"""

import argparse
import math
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def compute_ce_ppl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> Tuple[List[float], List[float], float, float]:
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

import argparse
import os

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from tf3.ablation.finetune import finetune_model

def main():
    SRC = "artifacts/transformers"          # original trained model
    DST = "artifacts/tf3-50m-base-ab-fixed" # new folder

    # Load model and tokenizer
    model = LlamaForCausalLM.from_pretrained(SRC)
    tokenizer = AutoTokenizer.from_pretrained(SRC)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.save_pretrained(DST)
    
    # If you changed vocab size in training, keep this consistent:
    #model.resize_token_embeddings(model.config.vocab_size)

    # 1) Make lm_head equal to embed_tokens IN MEMORY
    with torch.no_grad():
        model.lm_head.weight.copy_(model.model.embed_tokens.weight)

    # 2) Build a state_dict where lm_head.weight exists and equals embed_tokens
    state_dict = model.state_dict()
    state_dict["lm_head.weight"] = state_dict["lm_head.weight"].clone()  # explicit, just to be clear

    # 3) Save manually
    model.tie_weights()
    model.save_pretrained(DST)
    tokenizer.save_pretrained(DST)

    # Load the model and check if the weights are tied
    m = LlamaForCausalLM.from_pretrained(DST)
    sd = m.state_dict()
    print("lm_head.weight" in sd)  # should be True
    print("Max difference:", (sd["lm_head.weight"] - sd["model.embed_tokens.weight"]).abs().max())
    print("lm_head shape:", sd["lm_head.weight"].shape)
    print("embed_tokens shape:", sd["model.embed_tokens.weight"].shape)

    # Fine-tune the model after embedding modification
    finetune_model(
        model=m,
        tokenizer=tokenizer,
        output_path=DST + "-finetuned",
        batch_size=64,
        grad_accum_steps=1,
        max_epochs=3,
        base_lr=2e-4,
        max_length=128,
        max_samples=20_000,
    )

if __name__ == "__main__":
    main()
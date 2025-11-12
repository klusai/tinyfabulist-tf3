"""
This file defines a small LLaMA-style model (~50M params).
"""

from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    vocab_size=32000,          
    hidden_size=512,           # embedding / hidden dim
    intermediate_size=1365,    # feed-forward dimension (≈ 8/3 * hidden_size)
    num_hidden_layers=6,       # depth 
    num_attention_heads=8,     # must divide hidden_size
    max_position_embeddings=2048,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,        # rotary embeddings
)

model = LlamaForCausalLM(config)

# make sure vocab matches tokenizer
model.resize_token_embeddings(config.vocab_size)

if __name__ == "__main__":
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params/1e6:.2f}M")

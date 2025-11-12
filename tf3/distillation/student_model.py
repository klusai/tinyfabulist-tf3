"""
This file contains the definition for the student model.
"""

from transformers import MambaConfig, MambaForCausalLM

config = MambaConfig(
    vocab_size=32000,
    hidden_size=128,        # smaller embedding & hidden size
    num_hidden_layers=6,    # fewer layers
    state_size=64,          # keep default
    expand=2,               # same
    conv_kernel=4           # same
)

model = MambaForCausalLM(config)

# ensure config.vocab_size is set to the tokenizer's vocab size
model.resize_token_embeddings(config.vocab_size)

if __name__ == "__main__":
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params/1e6:.2f}M")

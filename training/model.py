"""
This file contains the model definition for the model.
"""

from transformers import MambaConfig, MambaForCausalLM

config = MambaConfig(
    vocab_size=32000,
    hidden_size=512,  # was d_model
    num_hidden_layers=18,  # was n_layer
    state_size=64,  # keep default or increase a bit
    expand=2,
    conv_kernel=4,
)

model = MambaForCausalLM(config)

# ensure config.vocab_size is set to the tokenizer's vocab size
model.resize_token_embeddings(config.vocab_size)

if __name__ == "__main__":
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params/1e6:.2f}M")

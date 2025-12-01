from transformers import LlamaConfig, LlamaForCausalLM

config = LlamaConfig(
    vocab_size=32000,
    hidden_size=512,
    intermediate_size=1365,
    num_hidden_layers=6,
    num_attention_heads=8,
    max_position_embeddings=2048,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
)

model = LlamaForCausalLM(config)

if __name__ == "__main__":
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params/1e6:.2f}M")

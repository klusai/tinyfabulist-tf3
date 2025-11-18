from transformers import LlamaConfig, LlamaForCausalLM

student_config = LlamaConfig(
    vocab_size=32000,
    hidden_size=384,
    intermediate_size=1024,
    num_hidden_layers=6,
    num_attention_heads=6,
    max_position_embeddings=2048,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    tie_word_embeddings=True,
)


model = LlamaForCausalLM(student_config)

# ensure vocab matches
model.resize_token_embeddings(student_config.vocab_size)

# ADD THIS (required by HF to activate tying)
model.tie_weights()

if __name__ == "__main__":
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params/1e6:.2f}M")

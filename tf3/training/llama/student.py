from transformers import LlamaConfig, LlamaForCausalLM

# CRITICAL: Align vocab_size to multiple of 64 for kernel fusion
# This prevents resize_token_embeddings from breaking kernel alignment
VOCAB_SIZE = 32000
ALIGNED_VOCAB_SIZE = VOCAB_SIZE + (64 - VOCAB_SIZE % 64) if VOCAB_SIZE % 64 != 0 else VOCAB_SIZE

student_config = LlamaConfig(
    vocab_size=ALIGNED_VOCAB_SIZE,  # Aligned to 64 for kernel fusion

    # Compact student architecture derived from pruning analysis
    # 25% reduction in width, heads, and MLP relative to the 51.65M teacher
    hidden_size=384,
    num_attention_heads=6,     # 384/6 = 64 head dim
    intermediate_size=1024,
    num_key_value_heads=6,

    num_hidden_layers=6,       # Keep teacher depth (best for quality)
    max_position_embeddings=2048,

    # Required LLaMA attributes
    rms_norm_eps=1e-6,
    rope_theta=10000.0,

    # Efficiency - tie_word_embeddings can be slower for small models
    # But keeping it for parameter efficiency
    tie_word_embeddings=True,
)

model = LlamaForCausalLM(
    student_config
)


if ALIGNED_VOCAB_SIZE != VOCAB_SIZE:
    model.resize_token_embeddings(ALIGNED_VOCAB_SIZE)
elif model.config.vocab_size != ALIGNED_VOCAB_SIZE:
    model.resize_token_embeddings(ALIGNED_VOCAB_SIZE)

# Ensure weights are tied
model.tie_weights()

if __name__ == "__main__":
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params/1e6:.2f}M")

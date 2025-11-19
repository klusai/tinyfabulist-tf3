import torch
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("klusai/tf3-50m-base")

# ------------------------------------
# 1. Tie embeddings manually
# ------------------------------------
model.lm_head.weight = model.model.embed_tokens.weight

# verify
print(model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr())
# must be True

# ------------------------------------
# 2. Save FULL MODEL (weights + config)
# ------------------------------------
save_path = "artifacts/"
model.save_pretrained(save_path)

print("Saved:", save_path)

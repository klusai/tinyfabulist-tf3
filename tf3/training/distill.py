from datasets import load_dataset, load_from_disk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    LlamaConfig
)

# ============================================================
# 1. Load teacher and student
# ============================================================
print("Loading teacher model...")
teacher_path = "artifacts/transformers"           # YOUR TEACHER
student_path = "artifacts/transformers-distilled" # SAVE STUDENT HERE

teacher = LlamaForCausalLM.from_pretrained(teacher_path).eval()
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(teacher_path)
print("Teacher and tokenizer loaded!")

# Your student config (20M)
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

print("Creating student model...")
student = LlamaForCausalLM(student_config)
student.resize_token_embeddings(student_config.vocab_size)
student.tie_weights()
print("Student model created!")

# ============================================================
# 2. Prepare dataset
# ============================================================

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, maxlen=512):
        print(f"Tokenizing {len(texts)} texts... (this may take a while)")
        self.data = tokenizer(texts, return_tensors='pt',
                              padding=True, truncation=True,
                              max_length=maxlen)
        print(f"Tokenization complete! Dataset size: {len(self)}")
    def __len__(self):
        return self.data["input_ids"].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

# Load dataset - handle both DatasetDict and Dataset
print("Loading dataset from disk...")
loaded_data = load_from_disk("artifacts/ds-tf2-en-ro-3m-tokenized")
print("Dataset loaded! Extracting texts...")
if isinstance(loaded_data, dict) and "train" in loaded_data:
    # DatasetDict: access train split
    train_dataset = loaded_data["train"]
    # Access column directly if available, otherwise iterate
    if "translated_fable" in train_dataset.column_names:
        texts = train_dataset["translated_fable"]
    else:
        texts = [item["translated_fable"] for item in train_dataset if "translated_fable" in item]
else:
    # Single Dataset
    if "translated_fable" in loaded_data.column_names:
        texts = loaded_data["translated_fable"]
    else:
        texts = [item["translated_fable"] for item in loaded_data if "translated_fable" in item]

print(f"Extracted {len(texts)} texts. Creating dataset...")
dataset = TextDataset(texts, tokenizer)
print("Creating DataLoader...")
loader = DataLoader(dataset, batch_size=64, shuffle=True)
print(f"DataLoader created! Total batches: {len(loader)}")

# ============================================================
# 3. Optimizer
# ============================================================
print("Setting up optimizer...")
optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4)
print("Starting training...")

# ============================================================
# 4. DISTILLATION LOOP
# ============================================================
alpha = 1.0  # KL weight
beta  = 0.1  # CE weight (stabilizer)

for epoch in range(3):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/3")
    print(f"{'='*60}")
    batch_count = 0
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        batch = {k: v for k, v in batch.items()}

        with torch.no_grad():
            teacher_logits = teacher(**batch).logits

        student_out = student(**batch)
        student_logits = student_out.logits

        # KL divergence (teacher → student)
        kl_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean"
        )

        # Optional next-token CE (helps early training)
        shift_logits = student_logits[:, :-1].contiguous()
        shift_labels = batch["input_ids"][:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        loss = alpha * kl_loss + beta * ce_loss
        loss.backward()
        optimizer.step()

        batch_count += 1
        if batch_idx % 10 == 0 or batch_idx == len(loader) - 1:
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | KL: {kl_loss.item():.4f} | CE: {ce_loss.item():.4f}")

# ============================================================
# 5. Save student
# ============================================================
print(f"\nSaving student model to {student_path}...")
student.save_pretrained(student_path)
tokenizer.save_pretrained(student_path)
print("Done!")

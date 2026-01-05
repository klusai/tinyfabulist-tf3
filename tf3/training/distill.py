import torch
import torch.nn as nn
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# CONFIG
# =====================================================
MAX_LEN = 512               # Major speedup
TEACHER_TOPK = 32           # Precomputed logits
DISTILL_DATASET = "artifacts/ds-tf2-en-ro-200k-distill"

BATCH_SIZE = 2              # Best for M3 Ultra
GA_STEPS = 8                # Effective batch = 16

LR = 1e-4
MAX_STEPS = 100_000         # Enough for 200k samples with KL-only

TEMP = 3.0                  # Distillation temperature
ALPHA = 1.0                 # KL-only


# =====================================================
# DEVICE
# =====================================================
def get_preferred_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_preferred_device()
print("Using device:", device)


# =====================================================
# Load teacher & student
# =====================================================
teacher = AutoModelForCausalLM.from_pretrained("klusai/tf3-50m-base")
tokenizer = AutoTokenizer.from_pretrained("klusai/tf3-50m-base")

# Teacher stays on CPU, used only for cached data sanity checks
teacher.to("cpu")
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

from tf3.training.llama.student import model as student_model
student = student_model.to(device)


# =====================================================
# Dataset: must contain ONLY:
# - input_ids
# - attention_mask
# - teacher_topk_idx
# - teacher_topk_val
# =====================================================
dataset = load_from_disk(DISTILL_DATASET)

# Check what columns exist and generate attention_mask if needed
if "attention_mask" not in dataset.column_names:
    logger.info("attention_mask not found, generating from input_ids...")
    def add_attention_mask(example):
        # Assuming pad_token_id marks padding
        pad_id = tokenizer.pad_token_id
        example["attention_mask"] = [1 if token_id != pad_id else 0 for token_id in example["input_ids"]]
        return example
    dataset = dataset.map(add_attention_mask)

dataset = dataset.remove_columns(
    [c for c in dataset.column_names if c not in
     ["input_ids", "attention_mask", "teacher_topk_idx", "teacher_topk_val"]]
)

dataset = dataset.remove_columns(
    [c for c in dataset.column_names if c not in
     ["input_ids", "attention_mask", "teacher_topk_idx", "teacher_topk_val"]]
)

# Ensure padding is correct (no dynamic padding during training)
# Around line 94-111, replace with:
def fix_lengths(example):
    ids = example["input_ids"]
    attn = example["attention_mask"]
    t_idx = example["teacher_topk_idx"]
    t_val = example["teacher_topk_val"]
    
    # Truncate if needed
    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]
        attn = attn[:MAX_LEN]
        t_idx = t_idx[:MAX_LEN]
        t_val = t_val[:MAX_LEN]
    
    pad_id = tokenizer.pad_token_id
    pad_len = MAX_LEN - len(ids)
    
    # Pad if needed
    if pad_len > 0:
        ids = ids + [pad_id] * pad_len
        attn = attn + [0] * pad_len
        # Pad teacher data with zeros (they'll be masked out anyway)
        t_idx = t_idx + [[0] * TEACHER_TOPK] * pad_len
        t_val = t_val + [[0.0] * TEACHER_TOPK] * pad_len
    
    example["input_ids"] = ids
    example["attention_mask"] = attn
    example["teacher_topk_idx"] = t_idx
    example["teacher_topk_val"] = t_val
    return example

dataset = dataset.map(fix_lengths, num_proc=1)

split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]


# =====================================================
# Custom Distillation Trainer (KL-only, top-k)
# =====================================================
class KLTrainer(Trainer):
    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

        self.log_kl = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device

        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long, device=device)

        t_idx = torch.tensor(inputs["teacher_topk_idx"], dtype=torch.long, device=device)
        t_val = torch.tensor(inputs["teacher_topk_val"], dtype=torch.float32, device=device)

        # Student forward
        s_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        s_shift = s_logits[:, :-1, :]

        t_idx = t_idx[:, :-1, :]
        t_val = t_val[:, :-1, :]

        # Extract student logits at teacher top-k positions
        s_topk = torch.gather(s_shift, dim=-1, index=t_idx)

        # Log-softmax student / Softmax teacher
        log_s = nn.functional.log_softmax(s_topk / self.temperature, dim=-1)
        p_t = nn.functional.softmax(t_val / self.temperature, dim=-1)

        # KL divergence
        kl = (p_t * (torch.log(p_t + 1e-9) - log_s)).sum(dim=-1)
        kl = kl.mean() * (self.temperature ** 2)

        self.log_kl = kl.item()

        return (kl, s_logits) if return_outputs else kl

    def log(self, logs, start_time=None):
        if self.log_kl is not None:
            logs["kl_loss"] = self.log_kl
        super().log(logs, start_time)


# =====================================================
# TrainingArguments optimized for Apple Silicon
# =====================================================
args = TrainingArguments(
    output_dir="./distilled_tf3_m3ultra",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GA_STEPS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=1000,

    max_steps=MAX_STEPS,

    fp16=False,
    bf16=True,

    logging_steps=50,
    eval_steps=200,
    save_steps=200,
    save_strategy="steps",
    eval_strategy="steps",

    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    remove_unused_columns=False,
)


# =====================================================
# Train
# =====================================================
trainer = KLTrainer(
    model=student,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    temperature=TEMP,
)

trainer.train()

# Save final distilled model
student.save_pretrained("artifacts/tf3-20m-distilled")
tokenizer.save_pretrained("artifacts/tf3-20m-distilled")
print("Distillation complete!")

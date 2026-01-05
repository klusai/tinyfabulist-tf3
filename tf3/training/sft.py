import logging
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import load_dataset, load_from_disk
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# 1. Load model + tokenizer
# ============================================================
model = LlamaForCausalLM.from_pretrained("klusai/tf3-50m-base")
tokenizer = AutoTokenizer.from_pretrained("artifacts/transformers-sft")


# ============================================================
# 2. Load and combine datasets
# ============================================================
print("Loading datasets...")

# Load all splits from HuggingFace dataset
ds_15k = load_dataset("klusai/ds-tf2-en-ro-15k")
print(f"Loaded ds-tf2-en-ro-15k: {ds_15k}")

# Load local dataset
ds_20k = load_from_disk("artifacts/final_sft_fables_20k_tr")
print(f"Loaded final_sft_fables_20k_tr: {len(ds_20k):,} rows")

# Concatenate all splits from ds_15k with ds_125k
from datasets import concatenate_datasets

all_datasets = []

# Add all splits from ds_15k
if isinstance(ds_15k, dict):
    for split_name, split_data in ds_15k.items():
        print(f"  Adding {split_name}: {len(split_data):,} rows")
        all_datasets.append(split_data)
else:
    all_datasets.append(ds_15k)

# Add local dataset
all_datasets.append(ds_20k)

# Combine all datasets
ds = concatenate_datasets(all_datasets)
print(f"\nTotal combined dataset: {len(ds):,} rows")

# Filter to only rows with translated_prompt (skip untranslated)
print("\nFiltering dataset for quality...")
initial_count = len(ds)

# Filter 1: Must have translated_prompt
ds = ds.filter(lambda x: x.get("translated_prompt", "").strip() != "")
print(f"  After filtering (has translated_prompt): {len(ds):,} rows")

# Filter 2: Must have translated_fable
ds = ds.filter(lambda x: x.get("translated_fable", "").strip() != "")
print(f"  After filtering (has translated_fable): {len(ds):,} rows")

# Filter 3: Minimum length check (avoid very short/bad examples)
ds = ds.filter(lambda x: len(x.get("translated_prompt", "")) > 50 and len(x.get("translated_fable", "")) > 100)
print(f"  After filtering (min length): {len(ds):,} rows")

# Filter 4: Maximum length check (avoid truncation issues)
ds = ds.filter(lambda x: len(x.get("translated_prompt", "")) < 2000 and len(x.get("translated_fable", "")) < 3000)
print(f"  After filtering (max length): {len(ds):,} rows")

print(f"\nFinal dataset: {len(ds):,} rows (removed {initial_count - len(ds):,} low-quality examples)")

# ============================================================
# 3. Format function
# ============================================================
MAX_LEN = 1500

def format(example):
    prompt = example["translated_prompt"].strip()
    answer = example["translated_fable"].strip()

    bos_id = tokenizer.bos_token_id      # 2
    eos_id = tokenizer.eos_token_id      # 3
    pad_id = tokenizer.pad_token_id      # 0

    # 1) Prompt în stil instruct:
    # <bos> prompt 
    prompt_text = f"{prompt}"
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False
    )["input_ids"]
    prompt_ids = [bos_id] + prompt_ids   # punem BOS manual

    # 2) Răspuns (fabulă), fără special tokens
    answer_ids = tokenizer(
        answer,
        add_special_tokens=False
    )["input_ids"]

    # 3) Construiți secvența completă: prompt + answer + EOS
    input_ids = prompt_ids + answer_ids + [eos_id]

    # 4) Trunchiere la MAX_LEN
    input_ids = input_ids[:MAX_LEN]
    attn = [1] * len(input_ids)

    # 5) Labels:
    #   - prompt: mascat (-100)
    #   - answer + EOS: vizibile în loss
    labels = [-100] * len(prompt_ids) + answer_ids + [eos_id]
    labels = labels[:MAX_LEN]

    # 6) Padding la MAX_LEN
    pad_len = MAX_LEN - len(input_ids)
    if pad_len > 0:
        input_ids += [pad_id] * pad_len
        attn += [0] * pad_len
        labels += [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
    }

def apply_formatting(split):
    formatted = split.map(format)
    return formatted


# ds is now a single concatenated dataset, not a dict
ds = apply_formatting(ds)

# Split into train/validation (90/10)
print("\nSplitting into train/validation...")
ds = ds.train_test_split(test_size=0.1, seed=42)
print(f"Train: {len(ds['train']):,} rows")
print(f"Validation: {len(ds['test']):,} rows")

# ============================================================
# 4. Training arguments (optimized for 50M model + MPS)
# ============================================================
training_args = TrainingArguments(
    output_dir="sft-distilled",
    per_device_train_batch_size=2,        # REDUCED: smaller batch for stability
    gradient_accumulation_steps=16,       # INCREASED: keep effective batch = 32
    learning_rate=5e-5,                   # MUCH LOWER: start conservative
    num_train_epochs=5,                   # INCREASED: train/eval gap suggests more training will help                   
    bf16=True,                            
    fp16=False,                           
    logging_steps=20,
    save_steps=300,                       
    eval_steps=300,                       
    eval_strategy="steps",                
    warmup_steps=200,                     # LONGER warmup for stability
    weight_decay=0.01,                    
    lr_scheduler_type="linear",           # CHANGED: linear decay is more stable than cosine
    optim="adamw_torch",
    save_total_limit=8,                   
    load_best_model_at_end=True,         
    metric_for_best_model="loss",         
    max_grad_norm=0.5,                    # STRICTER: even tighter gradient clipping
    dataloader_num_workers=0,             # Single thread for MPS stability
)


# # ============================================================
# # 5. Data Collator (handles padding dynamically)
# # ============================================================
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False,  # Causal LM, not masked LM
# )

# ============================================================
# 6. Trainer
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],  # Added validation dataset
    tokenizer=tokenizer,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,  # Stop if no improvement for 3 evaluations
            early_stopping_threshold=0.01  # Minimum change to count as improvement
        )
    ]
)

trainer.train()


# ============================================================
# 6. Save final SFT model
# ============================================================
model.save_pretrained("artifacts/transformers-50m-base-sft")
tokenizer.save_pretrained("artifacts/transformers-50m-base-sft")

print("SFT complete!")

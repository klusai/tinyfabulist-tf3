import logging
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# 1. Load model + tokenizer
# ============================================================
model = LlamaForCausalLM.from_pretrained("artifacts/tf3-20m-distill-mlx")
tokenizer = AutoTokenizer.from_pretrained("artifacts/transformers-sft")


# ============================================================
# 2. Load your 15k dataset
# ============================================================
ds = load_dataset("klusai/ds-tf2-en-ro-15k")

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


if isinstance(ds, dict):
    for split_name in ds.keys():
        ds[split_name] = apply_formatting(ds[split_name])
else:
    ds = apply_formatting(ds)

# ============================================================
# 4. Training arguments (optimized for 20M model + MPS)
# ============================================================
training_args = TrainingArguments(
    output_dir="sft-distilled",
    per_device_train_batch_size=4,        # safe on MPS
    gradient_accumulation_steps=8,        # effective batch size = 32
    learning_rate=1e-4,                   # higher LR is better for tiny models
    num_train_epochs=3,                   # 2–3 is optimal
    bf16=True,                            # best for MPS
    fp16=False,                           
    logging_steps=20,
    save_steps=2000,
    warmup_ratio=0.05,
    weight_decay=0.0,
    optim="adamw_torch",
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
    tokenizer=tokenizer,
)

trainer.train()


# ============================================================
# 6. Save final SFT model
# ============================================================
model.save_pretrained("artifacts/transformers-sft")
tokenizer.save_pretrained("artifacts/transformers-sft")

print("SFT complete!")

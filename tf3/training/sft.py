from transformers import LlamaForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# ============================================================
# 1. Load model + tokenizer
# ============================================================
model = LlamaForCausalLM.from_pretrained("klusai/tf3-50m-base")
tokenizer = AutoTokenizer.from_pretrained("artifacts/transformers-distilled")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# ============================================================
# 2. Load your 15k dataset
# ============================================================
ds = load_dataset("klusai/ds-tf2-en-ro-15k")

# this dataset contains: translated_prompt, translated_fable
# we will map them into an instruction format


# ============================================================
# 3. Format function (FINAL)
# ============================================================
def format(example):
    prompt = example["translated_prompt"].strip()
    answer = example["translated_fable"].strip()

    # LLaMA-style instruction prompt format
    full_text = (
        f"<s>[INST] {prompt} [/INST]\n"
        f"{answer}</s>"
    )

    tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


# Apply formatting to all splits
ds = ds.map(format)


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
    fp16=False,                           # avoid fp16 on MPS
    logging_steps=20,
    save_steps=2000,
    warmup_ratio=0.05,
    weight_decay=0.0,
    optim="adamw_torch",
)


# ============================================================
# 5. Trainer
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
)

trainer.train()


# ============================================================
# 6. Save final SFT model
# ============================================================
model.save_pretrained("artifacts/transformers-final-sft")
tokenizer.save_pretrained("artifacts/transformers-final-sft")

print("SFT complete!")

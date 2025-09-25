"""
This file contains the main function for training the model.
"""

from datasets import load_from_disk
from model import model  # Mamba model
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

OUTPUT_DIR = "checkpoints/mamba-50M"
TOKENIZER_PATH = "artifacts/tokenizers_2025_09_10_11_05_41/unigram_tokenizer.json"
DATASET_PATH = "artifacts/ds-tf2-en-ro-3m-tokenized"


# Load tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)

special_tokens = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
}
tokenizer.add_special_tokens(special_tokens)

# Now align with model
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.bos_token_id = tokenizer.bos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id


model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
tokenizer.padding_side = "right"

model.resize_token_embeddings(len(tokenizer))

# Load preprocessed dataset from disk
dataset = load_from_disk(DATASET_PATH)

# Create a small evaluation split if not present
if isinstance(dataset, dict) and "train" in dataset and "test" in dataset:
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
else:
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=96,  # H200 141GB can handle larger batches; adjust if needed
    gradient_accumulation_steps=1,
    learning_rate=3e-4,
    warmup_steps=100,
    num_train_epochs=20,
    save_steps=200,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=200,
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=False,
    report_to="none",
    dataloader_num_workers=16,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_torch_fused",
    include_tokens_per_second=True,
    torch_compile=False,
)

# Add early stopping callback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.001,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    tokenizer=tokenizer,
    callbacks=[early_stopping],
)

trainer.train()
trainer.save_model(OUTPUT_DIR)

"""
This file contains the main function for training the model.
"""

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk
from model import model   # Mamba model

OUTPUT_DIR = "checkpoints/mamba-50M"
TOKENIZER_PATH = "artifacts/tokenizers_2025_09_10_11_05_41/unigram_tokenizer.json"
DATASET_PATH = "artifacts/ds-tf2-en-ro-3m-tokenized"

# Load tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_PATH
)

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

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=48,  # H200 141GB can handle larger batches; adjust if needed
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    warmup_steps=500,
    num_train_epochs=20,
    save_steps=200,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=200,
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=True,
    report_to="wandb",
    dataloader_num_workers=16,
    dataloader_pin_memory=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_torch_fused",
    torch_compile=True,
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
    callbacks=[early_stopping]
)

trainer.train()
trainer.save_model(OUTPUT_DIR)

"""
This file contains the main function for training the model.
"""

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk
from model import model   # your Mamba model

OUTPUT_DIR = "checkpoints/mamba-50M"
TOKENIZER_PATH = "artifacts/tokenizers_2025_09_10_11_05_41/unigram_tokenizer.json"
DATASET_PATH = "artifacts/ds-tf2-en-ro-3m-tokenized"

# Load tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_PATH
)

# Load preprocessed dataset from disk
dataset = load_from_disk(DATASET_PATH)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    warmup_steps=500,
    num_train_epochs=20,
    save_steps=200,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=200,
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=True,
    report_to="wandb",
    dataloader_num_workers=16,
)

# Add early stopping callback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  
    early_stopping_threshold=0.001,  
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collator,
    tokenizer=tokenizer,
    callbacks=[early_stopping]
)

trainer.train()
trainer.save_model(OUTPUT_DIR)

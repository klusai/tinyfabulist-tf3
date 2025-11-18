"""
This file contains the main function for training the model.
"""

import argparse

from datasets import load_from_disk
from tf3.training.mamba.model import model as mamba_model  # Mamba model
from tf3.training.llama.model import model as llama_model  # LLaMA model
from tf3.logger import get_logger
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

logger = get_logger("training")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="artifacts/tokenizers_2025_11_18_18_49_10/unigram_tokenizer.json",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="artifacts/ds-tf2-en-ro-3m-tokenized"
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--model_type", type=str, default="llama")
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Arguments: {args}")

    if args.model_type == "mamba":
        model = mamba_model
        model_name = "mamba-50M"
    elif args.model_type == "llama":
        model = llama_model
        model_name = "llama-50M"
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    output_dir = f"checkpoints/{model_name}"
    args.output_dir = output_dir

    logger.info(f"Training {args.model_type} model")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Tokenizer path: {args.tokenizer_path}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Split: {args.split}")

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)

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
    dataset = load_from_disk(args.dataset_path)

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
        output_dir=args.output_dir,
        per_device_train_batch_size=96,  # H200 141GB can handle larger batches; adjust if needed
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        warmup_steps=1000,
        num_train_epochs=5,
        save_steps=1000,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=1000,
        save_total_limit=20,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch_fused",
        include_tokens_per_second=True,
        torch_compile=True,
        report_to="wandb",
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        max_grad_norm=1.0
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
    trainer.save_model(args.output_dir)
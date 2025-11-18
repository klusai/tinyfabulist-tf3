"""
This file contains the main function for preprocessing the dataset.
"""

import argparse

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="artifacts/tokenizers_2025_11_18_18_49_10/unigram_tokenizer.json",
    )
    parser.add_argument("--dataset_path", type=str, default="klusai/ds-tf2-en-ro-3m")
    parser.add_argument(
        "--output_path", type=str, default="artifacts/ds-tf2-en-ro-3m-tokenized"
    )
    parser.add_argument("--process_count", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer_path = args.tokenizer_path
    process_count = args.process_count
    block_size = args.block_size

    # 1. Load dataset
    dataset = load_dataset(args.dataset_path, split=args.split)
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c != "translated_fable"]
    )
    print(dataset)

    # 2. Load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # 3. Tokenize (disable masks and token_type_ids)
    def tokenize_function(examples):
        return tokenizer(
            examples["translated_fable"],
            truncation=True,
            max_length=block_size,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=process_count,
        remove_columns=["translated_fable"],
        desc="Tokenizing",
    )

    # Keep only input_ids
    tokenized = tokenized.remove_columns(
        [c for c in tokenized.column_names if c != "input_ids"]
    )

    # 4. Group into BLOCK_SIZE chunks
    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_len = (len(concatenated) // block_size) * block_size
        input_ids = [
            concatenated[i : i + block_size] for i in range(0, total_len, block_size)
        ]
        return {"input_ids": input_ids, "labels": input_ids.copy()}

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=process_count,
        desc=f"Grouping into {block_size}-token chunks",
    )

    print(lm_dataset)
    lm_dataset.save_to_disk(args.output_path)

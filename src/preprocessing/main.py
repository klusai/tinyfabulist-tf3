"""
This file contains the main function for preprocessing the dataset.
"""

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

TOKENIZER_PATH = "artifacts/tokenizers_2025_09_10_11_05_41/unigram_tokenizer.json"
PROCESS_COUNT = 16
BLOCK_SIZE = 2048

# 1. Load dataset
dataset = load_dataset("klusai/ds-tf2-en-ro-3m", split="train")
dataset = dataset.remove_columns([c for c in dataset.column_names if c != "translated_fable"])
print(dataset)

# 2. Load tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

# 3. Tokenize (disable masks and token_type_ids)
def tokenize_function(examples):
    return tokenizer(
        examples["translated_fable"],
        truncation=True,
        max_length=BLOCK_SIZE,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

tokenized = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=PROCESS_COUNT,
    remove_columns=["translated_fable"],
    desc="Tokenizing"
)

# Keep only input_ids
tokenized = tokenized.remove_columns([c for c in tokenized.column_names if c != "input_ids"])

# 4. Group into BLOCK_SIZE chunks
def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_len = (len(concatenated) // BLOCK_SIZE) * BLOCK_SIZE
    input_ids = [concatenated[i:i+BLOCK_SIZE] for i in range(0, total_len, BLOCK_SIZE)]
    return {
        "input_ids": input_ids,
        "labels": input_ids.copy()
    }

lm_dataset = tokenized.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=PROCESS_COUNT,
    desc=f"Grouping into {BLOCK_SIZE}-token chunks"
)

print(lm_dataset)
lm_dataset.save_to_disk("artifacts/ds-tf2-en-ro-3m-tokenized")

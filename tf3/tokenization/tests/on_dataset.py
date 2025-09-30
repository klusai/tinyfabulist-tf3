import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# --------------------------
# 1. Load dataset
# --------------------------
print("Loading dataset...")
dataset = load_dataset("klusai/ds-tf2-en-ro-3m", split="test")

# --------------------------
# 2. Load tokenizer
# --------------------------
print("Loading tokenizer...")
pretrained_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="artifacts/ro_tokenizer_20250909122701.json"
)
gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")


# --------------------------
# 3. Define tokenization function
# --------------------------
def tokenize_tf(batch):
    tokens = pretrained_tokenizer(batch["translated_fable"])
    return {"input_ids": tokens["input_ids"], "length": len(tokens["input_ids"])}


def tokenize_gemma(batch):
    tokens = gemma_tokenizer(batch["translated_fable"])
    return {"input_ids": tokens["input_ids"], "length": len(tokens["input_ids"])}


# --------------------------
# 4. Apply to dataset
# --------------------------
print("Tokenizing dataset...")
tf_tokenized_dataset = dataset.map(tokenize_tf, num_proc=16)
gemma_tokenized_dataset = dataset.map(tokenize_gemma, num_proc=16)

# --------------------------
# 5. Compute statistics
# --------------------------

for dataset in [tf_tokenized_dataset, gemma_tokenized_dataset]:
    if dataset == tf_tokenized_dataset:
        print("TF dataset...")
    else:
        print("Gemma dataset...")

    lengths = dataset["length"]
    print("\n📊 Tokenization Statistics")
    print(f"Total examples: {len(lengths)}")
    print(f"Average tokens per sentence: {np.mean(lengths):.2f}")
    print(f"Median tokens per sentence: {np.median(lengths):.2f}")
    print(f"Max tokens per sentence: {np.max(lengths)}")
    print(f"Min tokens per sentence: {np.min(lengths)}")
    print()

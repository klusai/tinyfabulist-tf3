from datasets import load_from_disk

dataset = load_from_disk("artifacts/ds-tf2-en-ro-3m-tokenized")

num_samples = len(dataset)             # number of blocks
block_size = len(dataset[0]["input_ids"])
total_tokens = num_samples * block_size

print(f"Samples: {num_samples}")
print(f"Block size: {block_size}")
print(f"Total tokens: {total_tokens:,}")

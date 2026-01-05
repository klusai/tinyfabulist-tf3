# Dataset Linking Module

This module provides **memory-efficient** functionality to link two Hugging Face datasets based on a common hash key (e.g., `prompt_hash`). It enriches a target dataset by copying fields from a source dataset where the hash values match.

**✨ Key Features:**
- 🚀 **Streaming & Batched Processing** - Handle millions of rows without loading into RAM
- 💾 **Memory-Efficient** - Process 3M+ rows on a laptop with 8-16GB RAM
- 📊 **Progress Tracking** - Real-time statistics and progress bars
- 🔄 **Automatic Cleanup** - Temporary files are automatically removed
- 🌐 **Hub Integration** - Push results directly to Hugging Face Hub

## Use Case

You have two datasets:
1. **Source dataset** (e.g., `klusai/ds-tf1-en-3m`) - Contains `prompt`, `prompt_hash`, and other fields
2. **Target dataset** (e.g., `klusai/ds-tf2-en-ro-3m`) - Contains `prompt_hash` and other fields

**Goal**: For each row in the target dataset, find the matching row in the source dataset (by `prompt_hash`) and copy the `prompt` field (or other specified fields) into the target dataset.

## Installation

No additional dependencies required beyond the base project requirements. The module uses:
- `datasets` (Hugging Face Datasets library)
- `tqdm` (for progress bars)

## Quick Start

👉 **See [USAGE.md](USAGE.md) for a complete quick start guide!**

The module is **memory-efficient by default** - you can process 3M row datasets on a laptop!

```bash
python3 -m tf3.dataset_linking.main \
    --source_dataset klusai/ds-tf1-en-3m \
    --target_dataset klusai/ds-tf2-en-ro-3m \
    --output_path artifacts/ds-tf2-en-ro-3m-enriched \
    --fields_to_copy prompt
```

## Usage

### Command Line

Basic usage (memory-efficient by default):

```bash
python -m tf3.dataset_linking.main \
    --source_dataset klusai/ds-tf1-en-3m \
    --target_dataset klusai/ds-tf2-en-ro-3m \
    --output_path artifacts/ds-tf2-en-ro-3m-enriched
```

Copy specific fields only:

```bash
python -m tf3.dataset_linking.main \
    --source_dataset klusai/ds-tf1-en-3m \
    --target_dataset klusai/ds-tf2-en-ro-3m \
    --output_path artifacts/ds-tf2-en-ro-3m-enriched \
    --fields_to_copy prompt system_message
```

Use different splits:

```bash
python -m tf3.dataset_linking.main \
    --source_dataset klusai/ds-tf1-en-3m \
    --target_dataset klusai/ds-tf2-en-ro-3m \
    --output_path artifacts/ds-tf2-en-ro-3m-enriched \
    --source_split train \
    --target_split validation
```

Push to Hugging Face Hub:

```bash
python -m tf3.dataset_linking.main \
    --source_dataset klusai/ds-tf1-en-3m \
    --target_dataset klusai/ds-tf2-en-ro-3m \
    --output_path artifacts/ds-tf2-en-ro-3m-enriched \
    --push_to_hub \
    --hub_dataset_name klusai/ds-tf2-en-ro-3m-enriched
```

Adjust batch size (larger = faster but more RAM):

```bash
python -m tf3.dataset_linking.main \
    --source_dataset klusai/ds-tf1-en-3m \
    --target_dataset klusai/ds-tf2-en-ro-3m \
    --output_path artifacts/ds-tf2-en-ro-3m-enriched \
    --batch_size 50000
```

Disable memory-efficient mode (only if you have >64GB RAM):

```bash
python -m tf3.dataset_linking.main \
    --source_dataset klusai/ds-tf1-en-3m \
    --target_dataset klusai/ds-tf2-en-ro-3m \
    --output_path artifacts/ds-tf2-en-ro-3m-enriched \
    --no_memory_efficient
```

### Python API

```python
from tf3.dataset_linking import link_datasets

# Basic usage
enriched_dataset = link_datasets(
    source_dataset_name="klusai/ds-tf1-en-3m",
    target_dataset_name="klusai/ds-tf2-en-ro-3m",
    output_path="artifacts/ds-tf2-en-ro-3m-enriched",
    hash_key="prompt_hash",
    fields_to_copy=["prompt"],  # or None to copy all fields
    source_split="train",
    target_split="train"
)

# Access the enriched dataset
print(enriched_dataset)
print(enriched_dataset[0])
```

Advanced usage with custom mapping:

```python
from datasets import load_dataset
from tf3.dataset_linking import create_hash_to_data_mapping, enrich_dataset

# Load datasets
source = load_dataset("klusai/ds-tf1-en-3m", split="train")
target = load_dataset("klusai/ds-tf2-en-ro-3m", split="train")

# Create hash mapping
hash_to_data = create_hash_to_data_mapping(
    source,
    hash_key="prompt_hash",
    fields_to_copy=["prompt", "system_message"]
)

# Enrich target dataset
enriched = enrich_dataset(
    target,
    hash_to_data,
    hash_key="prompt_hash"
)

# Save
enriched.save_to_disk("artifacts/enriched_dataset")
```

## Command Line Arguments

### Required Arguments

- `--source_dataset`: Name or path of the source dataset (contains data to copy)
- `--target_dataset`: Name or path of the target dataset (will be enriched)
- `--output_path`: Local path to save the enriched dataset

### Optional Arguments

- `--hash_key`: Name of the hash field to match on (default: `prompt_hash`)
- `--fields_to_copy`: Specific fields to copy from source (default: all fields except hash_key)
- `--source_split`: Split to use from source dataset (default: `train`)
- `--target_split`: Split to use from target dataset (default: `train`)
- `--push_to_hub`: Push enriched dataset to Hugging Face Hub (flag)
- `--hub_dataset_name`: Name for dataset on HF Hub (required if `--push_to_hub`)
- `--batch_size`: Number of rows to process per batch (default: `10000`)
- `--no_memory_efficient`: Disable memory-efficient mode (not recommended for large datasets)

## Features

- 🚀 **Memory-efficient**: Streaming + batched processing for millions of rows
- 💾 **Low RAM usage**: Process 3M rows with <2GB RAM (vs 30-50GB without streaming)
- 🔑 **Hash-based linking**: Efficiently matches rows based on hash keys
- 🎯 **Selective field copying**: Choose which fields to copy from source
- 📊 **Statistics tracking**: Real-time match/miss rates during enrichment
- 🌐 **Hub integration**: Push results directly to Hugging Face Hub
- 🗂️ **Multiple splits support**: Work with train/validation/test splits
- 📈 **Progress tracking**: Visual progress bars with tqdm
- 🧹 **Auto-cleanup**: Temporary batch files are automatically removed

## Output

The enriched dataset will contain:
- All original fields from the target dataset
- Additional fields copied from the source dataset (matched by hash)
- For non-matching hashes, empty strings are used as default values

## Performance Notes

### Memory-Efficient Mode (Default)

- **Source dataset**: Streamed row-by-row to build hash mapping (~1GB RAM for 3M hashes)
- **Target dataset**: Processed in batches (default: 10k rows, ~200MB RAM per batch)
- **Peak RAM usage**: <2GB for 3M row datasets
- **Processing time**: ~35-55 minutes for 3M rows

### Non-Memory-Efficient Mode (`--no_memory_efficient`)

- **RAM usage**: 30-50GB for 3M rows (loads everything into memory)
- **Processing time**: ~10-20 minutes for 3M rows (faster but needs lots of RAM)
- **Recommended only if**: You have >64GB RAM and want maximum speed

### Tips

- Increase `--batch_size` for faster processing (uses more RAM)
- Copy only needed fields with `--fields_to_copy` to save RAM
- The hash mapping uses the first occurrence of each hash value (automatic deduplication)

## Example

Given these datasets:

**Source (ds-tf1-en-3m)**:
```
prompt_hash | prompt
------------|-------
abc123      | "Write a story about..."
def456      | "Create a fable with..."
```

**Target (ds-tf2-en-ro-3m)**:
```
prompt_hash | translated_fable
------------|------------------
abc123      | "A fost odată..."
ghi789      | "Într-o zi..."
```

**Result**:
```
prompt_hash | translated_fable  | prompt
------------|-------------------|---------------------
abc123      | "A fost odată..." | "Write a story about..."
ghi789      | "Într-o zi..."    | ""  (no match found)
```

## License

MIT License - Same as the parent project.


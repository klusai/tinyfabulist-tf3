# Dataset Linking - Quick Start Guide

## 🚀 Quick Start (For 3M datasets)

The module is **memory-efficient by default** and designed to handle large datasets like the 3M fable datasets without loading everything into RAM.

### Basic Command

```bash
cd /Users/andreip/tf3

python3 -m tf3.dataset_linking.main \
    --source_dataset klusai/ds-tf1-en-3m \
    --target_dataset klusai/ds-tf2-en-ro-3m \
    --output_path artifacts/ds-tf2-en-ro-3m-enriched \
    --fields_to_copy prompt
```

### Or Use the Example Script

```bash
python3 -m tf3.dataset_linking.example_link_tf1_tf2
```

## 💾 Memory-Efficient Design

The module uses **streaming and batched processing** to minimize memory usage:

### How It Works

1. **Source Dataset (ds-tf1-en-3m)**
   - Loaded with **streaming** mode
   - Processes rows one-by-one to build hash→data mapping
   - Only the mapping is kept in memory (~few GB for 3M unique hashes)
   
2. **Target Dataset (ds-tf2-en-ro-3m)**
   - Loaded with **streaming** mode
   - Processed in **batches** (default: 10,000 rows)
   - Each batch is enriched and written to disk immediately
   - Memory never holds more than 1 batch at a time

3. **Final Consolidation**
   - All batch files are merged into final dataset
   - Temporary batch files are automatically cleaned up

### Memory Usage Estimate

For 3M rows with `prompt` field (~100 chars average):

- **Hash mapping**: ~300MB - 1GB (hash + prompt text for 3M unique hashes)
- **Batch processing**: ~100MB - 300MB (10k rows in memory at a time)
- **Total peak RAM**: < 2GB

Compare this to loading everything: ~30-50GB! 🎉

## 📊 Processing Time Estimate

For 3M rows:
- Building hash mapping: ~10-15 minutes
- Enriching target dataset: ~20-30 minutes
- Consolidating batches: ~5-10 minutes
- **Total**: ~35-55 minutes

## 🎛️ Advanced Options

### Adjust Batch Size

Larger batch size = faster but more RAM:

```bash
python3 -m tf3.dataset_linking.main \
    --source_dataset klusai/ds-tf1-en-3m \
    --target_dataset klusai/ds-tf2-en-ro-3m \
    --output_path artifacts/ds-tf2-en-ro-3m-enriched \
    --fields_to_copy prompt \
    --batch_size 50000  # Process 50k rows at a time (uses more RAM)
```

### Copy Multiple Fields

```bash
python3 -m tf3.dataset_linking.main \
    --source_dataset klusai/ds-tf1-en-3m \
    --target_dataset klusai/ds-tf2-en-ro-3m \
    --output_path artifacts/ds-tf2-en-ro-3m-enriched \
    --fields_to_copy prompt system_message fable
```

### Push to Hugging Face Hub

```bash
python3 -m tf3.dataset_linking.main \
    --source_dataset klusai/ds-tf1-en-3m \
    --target_dataset klusai/ds-tf2-en-ro-3m \
    --output_path artifacts/ds-tf2-en-ro-3m-enriched \
    --fields_to_copy prompt \
    --push_to_hub \
    --hub_dataset_name klusai/ds-tf2-en-ro-3m-enriched
```

### Disable Memory-Efficient Mode (Not Recommended)

If you have >64GB RAM and want faster processing:

```bash
python3 -m tf3.dataset_linking.main \
    --source_dataset klusai/ds-tf1-en-3m \
    --target_dataset klusai/ds-tf2-en-ro-3m \
    --output_path artifacts/ds-tf2-en-ro-3m-enriched \
    --fields_to_copy prompt \
    --no_memory_efficient
```

## 🐍 Python API

```python
from tf3.dataset_linking import link_datasets

# Memory-efficient linking (recommended)
enriched = link_datasets(
    source_dataset_name="klusai/ds-tf1-en-3m",
    target_dataset_name="klusai/ds-tf2-en-ro-3m",
    output_path="artifacts/ds-tf2-en-ro-3m-enriched",
    hash_key="prompt_hash",
    fields_to_copy=["prompt"],
    source_split="train",
    target_split="train",
    batch_size=10000,
    memory_efficient=True  # Default
)
```

## 📂 Output Format

The enriched dataset will be saved to disk in Arrow format and can be loaded with:

```python
from datasets import load_from_disk

dataset = load_from_disk("artifacts/ds-tf2-en-ro-3m-enriched")
print(dataset)
print(dataset[0])
```

## ❓ Troubleshooting

### "No module named 'datasets'"

Install dependencies:

```bash
pip install -r requirements.txt
```

### "ModuleNotFoundError: No module named 'tf3'"

Make sure you're running from the project root:

```bash
cd /Users/andreip/tf3
python3 -m tf3.dataset_linking.main ...
```

### Out of Memory Error

Even with streaming, the hash mapping needs to fit in RAM. If you still run out of memory:

1. Reduce batch size: `--batch_size 5000`
2. Copy fewer fields: `--fields_to_copy prompt` (instead of all fields)
3. Process on a machine with more RAM
4. Use a subset of the data for testing first

### Slow Processing

- The first run will download the datasets from HuggingFace
- Subsequent runs will use cached data
- Processing 3M rows takes 30-60 minutes even with streaming
- Use `--batch_size 50000` for faster processing (if you have RAM)

## 🧪 Testing

Test on a small sample first:

```bash
python3 -m tf3.dataset_linking.test_small
```

This runs unit tests with mock data to verify everything works before processing 3M rows.

## 📝 Summary

✅ **Memory-efficient**: Streams data, processes in batches
✅ **Automatic**: Handles HF datasets and local paths  
✅ **Robust**: Validates data, tracks statistics
✅ **Fast**: Batched processing with progress bars
✅ **Clean**: Auto-cleanup of temporary files

You can safely process 3M+ row datasets on a laptop with 8-16GB RAM! 🎉


"""
Dataset Linking Module

This module provides functionality to link two Hugging Face datasets based on a common key (prompt_hash).
It enriches a target dataset with fields from a source dataset by matching hash values.

Memory-efficient implementation using batched processing and streaming.
"""

import argparse
import os
from typing import Dict, Optional, Iterator
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from tqdm import tqdm


def create_hash_to_data_mapping_streaming(
    source_dataset,
    hash_key: str = "prompt_hash",
    fields_to_copy: Optional[list] = None,
    batch_size: int = 10000
) -> Dict[str, Dict]:
    """
    Create a mapping from hash to data fields using streaming (memory-efficient).
    
    Args:
        source_dataset: The source dataset (can be streaming or regular)
        hash_key: The name of the hash field (default: "prompt_hash")
        fields_to_copy: List of field names to copy. If None, copies all fields except hash_key
        batch_size: Number of rows to process at once
    
    Returns:
        Dictionary mapping hash values to field data
    """
    hash_to_data = {}
    
    # For streaming datasets, we need to get column names differently
    if hasattr(source_dataset, 'column_names'):
        column_names = source_dataset.column_names
    else:
        # For streaming dataset, peek at first row
        first_row = next(iter(source_dataset))
        column_names = list(first_row.keys())
        # We need to recreate the iterator since we consumed one item
        source_dataset = load_dataset(
            source_dataset.dataset_name if hasattr(source_dataset, 'dataset_name') else source_dataset._ex_iterable.kwargs['path'],
            split=source_dataset.split,
            streaming=True
        )
    
    # Determine which fields to copy
    if fields_to_copy is None:
        fields_to_copy = [col for col in column_names if col != hash_key]
    
    print(f"Fields to copy: {fields_to_copy}")
    print(f"Building hash mapping (memory-efficient streaming)...")
    
    # Process in batches to show progress
    total_processed = 0
    with tqdm(desc="Building hash mapping") as pbar:
        for row in source_dataset:
            hash_value = row[hash_key]
            
            # Only store the first occurrence of each hash
            if hash_value not in hash_to_data:
                hash_to_data[hash_value] = {field: row[field] for field in fields_to_copy}
            
            total_processed += 1
            pbar.update(1)
            
            # Show progress every batch_size rows
            if total_processed % batch_size == 0:
                pbar.set_postfix({"unique_hashes": len(hash_to_data), "processed": total_processed})
    
    print(f"Created mapping with {len(hash_to_data)} unique hashes from {total_processed} rows")
    return hash_to_data


def create_hash_to_data_mapping(
    source_dataset: Dataset,
    hash_key: str = "prompt_hash",
    fields_to_copy: Optional[list] = None
) -> Dict[str, Dict]:
    """
    Create a mapping from hash to data fields from the source dataset.
    
    Args:
        source_dataset: The source dataset containing the data to copy
        hash_key: The name of the hash field (default: "prompt_hash")
        fields_to_copy: List of field names to copy. If None, copies all fields except hash_key
    
    Returns:
        Dictionary mapping hash values to field data
    """
    hash_to_data = {}
    
    # Determine which fields to copy
    if fields_to_copy is None:
        fields_to_copy = [col for col in source_dataset.column_names if col != hash_key]
    
    print(f"Creating hash mapping from {len(source_dataset)} rows...")
    print(f"Fields to copy: {fields_to_copy}")
    
    for row in tqdm(source_dataset, desc="Building hash mapping"):
        hash_value = row[hash_key]
        
        # Only store the first occurrence of each hash
        if hash_value not in hash_to_data:
            hash_to_data[hash_value] = {field: row[field] for field in fields_to_copy}
    
    print(f"Created mapping with {len(hash_to_data)} unique hashes")
    return hash_to_data


def enrich_dataset_batched(
    target_dataset,
    hash_to_data: Dict[str, Dict],
    output_path: str,
    hash_key: str = "prompt_hash",
    missing_value: str = "",
    batch_size: int = 10000,
    is_streaming: bool = False
) -> None:
    """
    Enrich target dataset with data from hash_to_data mapping using batched processing.
    Writes results incrementally to disk to avoid memory issues.
    
    Args:
        target_dataset: The dataset to enrich (can be streaming or regular)
        hash_to_data: Mapping from hash to data fields
        output_path: Path to save the enriched dataset
        hash_key: The name of the hash field (default: "prompt_hash")
        missing_value: Value to use when hash is not found (default: empty string)
        batch_size: Number of rows to process before writing to disk
        is_streaming: Whether the target dataset is streaming
    """
    # Get the fields that will be added
    if not hash_to_data:
        raise ValueError("hash_to_data mapping is empty")
    
    sample_fields = list(next(iter(hash_to_data.values())).keys())
    print(f"Enriching dataset with fields: {sample_fields}")
    print(f"Using batch size: {batch_size}")
    print(f"Output path: {output_path}")
    
    # Track statistics
    matches = 0
    misses = 0
    total_processed = 0
    
    # Process in batches
    batch = []
    
    with tqdm(desc="Enriching dataset") as pbar:
        for row in target_dataset:
            # Create enriched row
            enriched_row = dict(row)
            hash_value = row[hash_key]
            
            if hash_value in hash_to_data:
                matches += 1
                # Add all fields from the mapping
                for field, value in hash_to_data[hash_value].items():
                    enriched_row[field] = value
            else:
                misses += 1
                # Add empty/missing values for all fields
                for field in sample_fields:
                    enriched_row[field] = missing_value
            
            batch.append(enriched_row)
            total_processed += 1
            pbar.update(1)
            
            # Write batch to disk when it reaches batch_size
            if len(batch) >= batch_size:
                _append_batch_to_disk(batch, output_path, total_processed - len(batch))
                pbar.set_postfix({"matches": matches, "misses": misses, "batches": total_processed // batch_size})
                batch = []
        
        # Write remaining batch
        if batch:
            _append_batch_to_disk(batch, output_path, total_processed - len(batch))
    
    print(f"\nEnrichment statistics:")
    print(f"  - Total processed: {total_processed}")
    print(f"  - Matches: {matches} ({matches/total_processed*100:.2f}%)")
    print(f"  - Misses: {misses} ({misses/total_processed*100:.2f}%)")
    
    # Consolidate all batches into a single dataset
    print("\nConsolidating batches into final dataset...")
    _consolidate_batches(output_path)


def _append_batch_to_disk(batch: list, output_path: str, batch_start_idx: int):
    """Append a batch of rows to disk."""
    batch_dataset = Dataset.from_list(batch)
    batch_path = f"{output_path}_batch_{batch_start_idx}"
    batch_dataset.save_to_disk(batch_path)


def _consolidate_batches(output_path: str):
    """Consolidate all batch files into a single dataset."""
    import glob
    
    # Find all batch files
    batch_paths = sorted(glob.glob(f"{output_path}_batch_*"))
    
    if not batch_paths:
        raise ValueError(f"No batch files found at {output_path}_batch_*")
    
    print(f"Found {len(batch_paths)} batch files to consolidate")
    
    # Load and concatenate all batches
    from datasets import concatenate_datasets
    
    datasets_list = []
    for batch_path in tqdm(batch_paths, desc="Loading batches"):
        datasets_list.append(load_from_disk(batch_path))
    
    # Concatenate all datasets
    final_dataset = concatenate_datasets(datasets_list)
    
    # Save final dataset
    print(f"Saving final dataset with {len(final_dataset)} rows...")
    final_dataset.save_to_disk(output_path)
    
    # Clean up batch files
    print("Cleaning up temporary batch files...")
    import shutil
    for batch_path in batch_paths:
        shutil.rmtree(batch_path)
    
    print("Consolidation complete!")


def enrich_dataset(
    target_dataset: Dataset,
    hash_to_data: Dict[str, Dict],
    hash_key: str = "prompt_hash",
    missing_value: str = ""
) -> Dataset:
    """
    Enrich target dataset with data from hash_to_data mapping.
    
    WARNING: This function loads all data into memory. For large datasets (>1M rows),
    use enrich_dataset_batched() instead.
    
    Args:
        target_dataset: The dataset to enrich
        hash_to_data: Mapping from hash to data fields
        hash_key: The name of the hash field (default: "prompt_hash")
        missing_value: Value to use when hash is not found (default: empty string)
    
    Returns:
        Enriched dataset with additional fields
    """
    # Get the fields that will be added
    if hash_to_data:
        sample_fields = list(next(iter(hash_to_data.values())).keys())
    else:
        raise ValueError("hash_to_data mapping is empty")
    
    print(f"Enriching {len(target_dataset)} rows with fields: {sample_fields}")
    
    # Track statistics
    matches = 0
    misses = 0
    
    def enrich_row(row):
        nonlocal matches, misses
        
        hash_value = row[hash_key]
        
        if hash_value in hash_to_data:
            matches += 1
            # Add all fields from the mapping
            for field, value in hash_to_data[hash_value].items():
                row[field] = value
        else:
            misses += 1
            # Add empty/missing values for all fields
            for field in sample_fields:
                row[field] = missing_value
        
        return row
    
    # Apply enrichment
    enriched_data = []
    for row in tqdm(target_dataset, desc="Enriching dataset"):
        enriched_data.append(enrich_row(row))
    
    print(f"\nEnrichment statistics:")
    print(f"  - Matches: {matches} ({matches/len(target_dataset)*100:.2f}%)")
    print(f"  - Misses: {misses} ({misses/len(target_dataset)*100:.2f}%)")
    
    # Create new dataset with enriched data
    enriched_dataset = Dataset.from_list(enriched_data)
    
    return enriched_dataset


def link_datasets(
    source_dataset_name: str,
    target_dataset_name: str,
    output_path: str,
    hash_key: str = "prompt_hash",
    fields_to_copy: Optional[list] = None,
    source_split: str = "train",
    target_split: str = "train",
    push_to_hub: bool = False,
    hub_dataset_name: Optional[str] = None,
    batch_size: int = 10000,
    memory_efficient: bool = True
):
    """
    Link two datasets based on a common hash key (memory-efficient by default).
    
    Args:
        source_dataset_name: Name/path of the source dataset (contains data to copy)
        target_dataset_name: Name/path of the target dataset (will be enriched)
        output_path: Local path to save the enriched dataset
        hash_key: Name of the hash field (default: "prompt_hash")
        fields_to_copy: List of fields to copy from source. If None, copies all except hash
        source_split: Split to use from source dataset (default: "train")
        target_split: Split to use from target dataset (default: "train")
        push_to_hub: Whether to push the result to Hugging Face Hub
        hub_dataset_name: Name for the dataset on HF Hub (required if push_to_hub=True)
        batch_size: Number of rows to process per batch (default: 10000)
        memory_efficient: Use streaming and batched processing (recommended for large datasets)
    """
    print("=" * 70)
    print("DATASET LINKING (MEMORY-EFFICIENT MODE)")
    print("=" * 70)
    print(f"Source dataset: {source_dataset_name} (split: {source_split})")
    print(f"Target dataset: {target_dataset_name} (split: {target_split})")
    print(f"Hash key: {hash_key}")
    print(f"Output path: {output_path}")
    print(f"Batch size: {batch_size}")
    print(f"Memory efficient mode: {memory_efficient}")
    print("=" * 70)
    
    # Helper function to load dataset (handles both HF and local paths)
    def load_dataset_smart(path: str, split: Optional[str], streaming: bool = False):
        """Load dataset from HF or local disk."""
        if os.path.exists(path):
            # Local path - use load_from_disk
            print(f"  Loading from local path: {path}")
            if streaming:
                # For local datasets, we can't use streaming, load normally
                return load_from_disk(path)
            return load_from_disk(path)
        else:
            # HF dataset - use load_dataset
            print(f"  Loading from Hugging Face: {path}")
            return load_dataset(path, split=split, streaming=streaming)
    
    # Load source dataset with streaming for memory efficiency
    print("\n1. Loading source dataset (streaming)...")
    source_streaming = load_dataset_smart(
        source_dataset_name,
        split=source_split,
        streaming=memory_efficient
    )
    
    # Get column info
    if hasattr(source_streaming, 'column_names'):
        print(f"Source dataset columns: {source_streaming.column_names}")
        if hasattr(source_streaming, '__len__'):
            print(f"Source dataset size: {len(source_streaming)}")
    else:
        # For streaming dataset
        first_row = next(iter(source_streaming))
        print(f"Source dataset columns: {list(first_row.keys())}")
        # Recreate iterator
        source_streaming = load_dataset_smart(
            source_dataset_name,
            split=source_split,
            streaming=memory_efficient
        )
    
    # Create hash mapping using streaming
    print("\n2. Creating hash mapping from source dataset (streaming)...")
    if memory_efficient:
        hash_to_data = create_hash_to_data_mapping_streaming(
            source_streaming,
            hash_key=hash_key,
            fields_to_copy=fields_to_copy,
            batch_size=batch_size
        )
    else:
        hash_to_data = create_hash_to_data_mapping(
            source_streaming,
            hash_key=hash_key,
            fields_to_copy=fields_to_copy
        )
    
    # Load target dataset with streaming
    print("\n3. Loading target dataset (streaming)...")
    target_streaming = load_dataset_smart(
        target_dataset_name,
        split=target_split,
        streaming=memory_efficient
    )
    
    if hasattr(target_streaming, 'column_names'):
        print(f"Target dataset columns: {target_streaming.column_names}")
        if hasattr(target_streaming, '__len__'):
            print(f"Target dataset size: {len(target_streaming)}")
    
    # Validate hash_key exists
    if hasattr(target_streaming, 'column_names'):
        if hash_key not in target_streaming.column_names:
            raise ValueError(f"Hash key '{hash_key}' not found in target dataset")
    
    # Enrich target dataset with batched processing
    print("\n4. Enriching target dataset (batched processing)...")
    if memory_efficient:
        enrich_dataset_batched(
            target_streaming,
            hash_to_data,
            output_path=output_path,
            hash_key=hash_key,
            batch_size=batch_size,
            is_streaming=True
        )
        # Load the final dataset
        enriched_dataset = load_from_disk(output_path)
    else:
        enriched_dataset = enrich_dataset(
            target_streaming,
            hash_to_data,
            hash_key=hash_key
        )
        # Save locally
        print(f"\n5. Saving enriched dataset to {output_path}...")
        enriched_dataset.save_to_disk(output_path)
    
    print(f"Saved successfully to {output_path}!")
    
    # Optionally push to Hub
    if push_to_hub:
        if not hub_dataset_name:
            raise ValueError("hub_dataset_name must be provided when push_to_hub=True")
        
        print(f"\n6. Pushing to Hugging Face Hub as '{hub_dataset_name}'...")
        enriched_dataset.push_to_hub(hub_dataset_name)
        print("Pushed successfully!")
    
    print("\n" + "=" * 70)
    print("DATASET LINKING COMPLETE")
    print("=" * 70)
    print(f"Final dataset path: {output_path}")
    print(f"Final dataset columns: {enriched_dataset.column_names}")
    print(f"Final dataset size: {len(enriched_dataset)}")
    
    return enriched_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Link two datasets based on a common hash key"
    )
    
    # Required arguments
    parser.add_argument(
        "--source_dataset",
        type=str,
        required=True,
        help="Name or path of source dataset (contains data to copy)"
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        required=True,
        help="Name or path of target dataset (will be enriched)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Local path to save enriched dataset"
    )
    
    # Optional arguments
    parser.add_argument(
        "--hash_key",
        type=str,
        default="prompt_hash",
        help="Name of the hash field to match on (default: prompt_hash)"
    )
    parser.add_argument(
        "--fields_to_copy",
        type=str,
        nargs="+",
        default=None,
        help="Specific fields to copy from source (default: all except hash_key)"
    )
    parser.add_argument(
        "--source_split",
        type=str,
        default="train",
        help="Split to use from source dataset (default: train)"
    )
    parser.add_argument(
        "--target_split",
        type=str,
        default="train",
        help="Split to use from target dataset (default: train)"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push enriched dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_dataset_name",
        type=str,
        default=None,
        help="Name for dataset on HF Hub (required if --push_to_hub)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Number of rows to process per batch (default: 10000)"
    )
    parser.add_argument(
        "--no_memory_efficient",
        action="store_true",
        help="Disable memory-efficient mode (not recommended for large datasets)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the dataset linking script."""
    args = parse_args()
    
    enriched_dataset = link_datasets(
        source_dataset_name=args.source_dataset,
        target_dataset_name=args.target_dataset,
        output_path=args.output_path,
        hash_key=args.hash_key,
        fields_to_copy=args.fields_to_copy,
        source_split=args.source_split,
        target_split=args.target_split,
        push_to_hub=args.push_to_hub,
        hub_dataset_name=args.hub_dataset_name,
        batch_size=args.batch_size,
        memory_efficient=not args.no_memory_efficient
    )
    
    return enriched_dataset


if __name__ == "__main__":
    main()


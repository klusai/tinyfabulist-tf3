"""
Example script for linking ds-tf1-en-3m and ds-tf2-en-ro-3m datasets.

This script demonstrates how to enrich the ds-tf2-en-ro-3m dataset with
the 'prompt' field from ds-tf1-en-3m by matching on 'prompt_hash'.
"""

from tf3.dataset_linking import link_datasets


def main():
    """
    Link ds-tf1-en-3m (source) with ds-tf2-en-ro-3m (target).
    
    The target dataset will be enriched with the 'prompt' field from
    the source dataset, matched by 'prompt_hash'.
    """
    
    # Configuration
    SOURCE_DATASET = "klusai/ds-tf1-en-3m"
    TARGET_DATASET = "klusai/ds-tf2-en-ro-3m"
    OUTPUT_PATH = "artifacts/ds-tf2-en-ro-3m-enriched"
    
    # Fields to copy from source dataset
    # Change this to copy more fields if needed, e.g.:
    # FIELDS_TO_COPY = ["prompt", "system_message", "fable"]
    FIELDS_TO_COPY = ["prompt"]
    
    # Run the linking
    print("Starting dataset linking process...")
    print(f"This will enrich {TARGET_DATASET} with fields from {SOURCE_DATASET}")
    print(f"Fields to copy: {FIELDS_TO_COPY}")
    print()
    
    enriched_dataset = link_datasets(
        source_dataset_name=SOURCE_DATASET,
        target_dataset_name=TARGET_DATASET,
        output_path=OUTPUT_PATH,
        hash_key="prompt_hash",
        fields_to_copy=FIELDS_TO_COPY,
        source_split="train",
        target_split="train",
        push_to_hub=False,  # Set to True if you want to push to HF Hub
        hub_dataset_name=None,  # Set to "your-username/dataset-name" if pushing
        batch_size=100000,  # Process 100k rows at a time
        memory_efficient=True  # Use streaming and batched processing (recommended!)
    )
    
    # Display sample
    print("\n" + "="*70)
    print("SAMPLE FROM ENRICHED DATASET")
    print("="*70)
    print(f"Total rows: {len(enriched_dataset)}")
    print(f"Columns: {enriched_dataset.column_names}")
    print("\nFirst row:")
    print(enriched_dataset[0])
    
    return enriched_dataset


if __name__ == "__main__":
    main()


"""
Dataset Linking Module

This module provides functionality to link and enrich datasets based on common hash keys.
Memory-efficient implementation using streaming and batched processing.

Also includes prompt translation functionality using OpenRouter API.
"""

from .main import (
    link_datasets,
    create_hash_to_data_mapping,
    create_hash_to_data_mapping_streaming,
    enrich_dataset,
    enrich_dataset_batched,
)

from .translate_prompts import (
    translate_dataset_prompts,
    translate_text,
)

__all__ = [
    "link_datasets",
    "create_hash_to_data_mapping",
    "create_hash_to_data_mapping_streaming",
    "enrich_dataset",
    "enrich_dataset_batched",
    "translate_dataset_prompts",
    "translate_text",
]


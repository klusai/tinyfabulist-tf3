"""
Test script for dataset linking with a small sample.

This script tests the linking functionality on a small subset of the data
before running on the full 3M dataset.
"""

from datasets import Dataset
from tf3.dataset_linking import link_datasets, create_hash_to_data_mapping, enrich_dataset


def create_mock_datasets():
    """Create small mock datasets for testing."""
    
    # Mock source dataset (like ds-tf1-en-3m)
    source_data = {
        "prompt_hash": ["hash1", "hash2", "hash3", "hash4"],
        "prompt": [
            "Write a story about a brave mouse",
            "Create a fable about friendship",
            "Tell a tale of a wise owl",
            "Compose a story about honesty"
        ],
        "system_message": [
            "You are a creative writer",
            "You are a fable creator",
            "You are a storyteller",
            "You are a moral teacher"
        ],
        "fable": [
            "Once upon a time, there was a brave mouse...",
            "In a forest, two friends learned...",
            "A wise owl sat on a tree...",
            "A young boy discovered that..."
        ]
    }
    
    # Mock target dataset (like ds-tf2-en-ro-3m)
    target_data = {
        "prompt_hash": ["hash1", "hash3", "hash5"],  # hash5 won't match
        "translated_fable": [
            "A fost odată un șoarece curajos...",
            "O bufniță înțeleaptă sta pe un copac...",
            "Un băiat tânăr a descoperit că..."
        ],
        "language": ["ro", "ro", "ro"]
    }
    
    source_dataset = Dataset.from_dict(source_data)
    target_dataset = Dataset.from_dict(target_data)
    
    return source_dataset, target_dataset


def test_hash_mapping():
    """Test the hash mapping creation."""
    print("="*70)
    print("TEST 1: Hash Mapping Creation")
    print("="*70)
    
    source_dataset, _ = create_mock_datasets()
    
    hash_to_data = create_hash_to_data_mapping(
        source_dataset,
        hash_key="prompt_hash",
        fields_to_copy=["prompt"]
    )
    
    print(f"\nHash mapping created with {len(hash_to_data)} entries")
    print("\nSample mapping:")
    for hash_val, data in list(hash_to_data.items())[:2]:
        print(f"  {hash_val}: {data}")
    
    assert len(hash_to_data) == 4, "Should have 4 unique hashes"
    assert "hash1" in hash_to_data, "hash1 should be in mapping"
    assert "prompt" in hash_to_data["hash1"], "prompt field should be in data"
    
    print("\n✅ Test passed!")
    return hash_to_data


def test_enrichment():
    """Test the dataset enrichment."""
    print("\n" + "="*70)
    print("TEST 2: Dataset Enrichment")
    print("="*70)
    
    source_dataset, target_dataset = create_mock_datasets()
    
    # Create hash mapping
    hash_to_data = create_hash_to_data_mapping(
        source_dataset,
        hash_key="prompt_hash",
        fields_to_copy=["prompt", "system_message"]
    )
    
    # Enrich target dataset
    enriched = enrich_dataset(
        target_dataset,
        hash_to_data,
        hash_key="prompt_hash"
    )
    
    print("\nEnriched dataset columns:", enriched.column_names)
    print("\nSample enriched row:")
    print(enriched[0])
    
    # Assertions
    assert "prompt" in enriched.column_names, "prompt should be added"
    assert "system_message" in enriched.column_names, "system_message should be added"
    assert len(enriched) == len(target_dataset), "Should have same number of rows"
    
    # Check that hash1 was matched correctly
    row_0 = enriched[0]
    assert row_0["prompt_hash"] == "hash1", "Should match hash1"
    assert "brave mouse" in row_0["prompt"], "Should have the correct prompt"
    
    # Check that hash5 (missing) has empty string
    row_2 = enriched[2]
    assert row_2["prompt_hash"] == "hash5", "Should be hash5"
    assert row_2["prompt"] == "", "Missing hash should have empty prompt"
    
    print("\n✅ Test passed!")
    return enriched


def test_full_linking():
    """Test the full linking process."""
    print("\n" + "="*70)
    print("TEST 3: Full Linking Process")
    print("="*70)
    
    # Create and save mock datasets temporarily
    source_dataset, target_dataset = create_mock_datasets()
    
    # Save to disk temporarily
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "source")
        target_path = os.path.join(tmpdir, "target")
        output_path = os.path.join(tmpdir, "output")
        
        source_dataset.save_to_disk(source_path)
        target_dataset.save_to_disk(target_path)
        
        # Run linking
        enriched = link_datasets(
            source_dataset_name=source_path,
            target_dataset_name=target_path,
            output_path=output_path,
            hash_key="prompt_hash",
            fields_to_copy=["prompt"],
            source_split=None,  # Use default for local datasets
            target_split=None,
            push_to_hub=False
        )
        
        print("\nFinal enriched dataset:")
        print(f"  Rows: {len(enriched)}")
        print(f"  Columns: {enriched.column_names}")
        
        # Verify the output was saved
        assert os.path.exists(output_path), "Output should be saved"
        
        print("\n✅ Test passed!")
        return enriched


def main():
    """Run all tests."""
    print("\n" + "🧪" + " "*31 + "TESTING DATASET LINKING" + " "*31 + "🧪")
    print()
    
    try:
        # Test 1: Hash mapping
        hash_to_data = test_hash_mapping()
        
        # Test 2: Enrichment
        enriched = test_enrichment()
        
        # Test 3: Full linking
        enriched_full = test_full_linking()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        print("\nThe dataset linking module is working correctly.")
        print("You can now run it on the full datasets:")
        print()
        print("  python -m tf3.dataset_linking.example_link_tf1_tf2")
        print()
        print("Or use the command line:")
        print()
        print("  python -m tf3.dataset_linking.main \\")
        print("      --source_dataset klusai/ds-tf1-en-3m \\")
        print("      --target_dataset klusai/ds-tf2-en-ro-3m \\")
        print("      --output_path artifacts/ds-tf2-en-ro-3m-enriched \\")
        print("      --fields_to_copy prompt")
        print()
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()


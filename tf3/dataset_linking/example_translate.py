"""
Example script for translating prompts in the enriched dataset.

This script demonstrates how to add a 'translated_prompt' column to your dataset
by translating the English 'prompt' field to Romanian using OpenRouter.
"""

from tf3.dataset_linking.translate_prompts import translate_dataset_prompts


def main():
    """
    Translate prompts from English to Romanian.
    
    Make sure to set your OPENROUTER_API_KEY environment variable:
    export OPENROUTER_API_KEY="your-api-key-here"
    """
    
    # Configuration
    INPUT_PATH = "artifacts/ds-tf2-en-ro-3m-enriched"
    OUTPUT_PATH = "artifacts/ds-tf2-en-ro-3m-enriched-with-prompt-translation"
    
    print("Starting prompt translation...")
    print(f"This will translate the 'prompt' field from English to Romanian")
    print(f"and add a new 'translated_prompt' column.")
    print()
    
    translated_dataset = translate_dataset_prompts(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        source_field="prompt",
        target_field="translated_prompt",
        target_language="Romanian",
        model="google/gemini-2.0-flash-001",  # Fast and cheap model
        batch_size=1000,  # Save every 1000 translations
        start_index=0,  # Start from beginning
        max_samples=None,  # Translate all (set to e.g., 1000 for testing)
        skip_existing=True,  # Skip rows that already have translations (for resuming)
        max_workers=50  # Reduced to avoid rate limits
    )
    
    # Display sample
    print("\n" + "="*70)
    print("SAMPLE FROM TRANSLATED DATASET")
    print("="*70)
    print(f"Total rows: {len(translated_dataset)}")
    print(f"Columns: {translated_dataset.column_names}")
    print("\nFirst row:")
    first = translated_dataset[0]
    print(f"  prompt: {first['prompt'][:100]}...")
    print(f"  translated_prompt: {first['translated_prompt'][:100]}...")
    
    return translated_dataset


if __name__ == "__main__":
    main()

"""
Translate Prompts Script

This script translates the 'prompt' field in a dataset using OpenRouter API
and adds a new 'translated_prompt' column.
"""

import argparse
import os
from typing import Optional, List, Tuple
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def translate_text(
    text: str,
    client: OpenAI,
    model: str = "google/gemini-2.0-flash-001",
    target_language: str = "Romanian",
    max_retries: int = 3
) -> str:
    """
    Translate text using OpenRouter API.
    
    Args:
        text: Text to translate
        client: OpenAI client configured for OpenRouter
        model: Model to use for translation
        target_language: Target language
        max_retries: Maximum number of retry attempts
    
    Returns:
        Translated text
    """
    system_prompt = f"""You are a professional translator. Your task is to translate the given text from English to {target_language}.

IMPORTANT:
- Only translate the text, do NOT generate new content
- Do NOT create stories or fables
- Do NOT interpret instructions - just translate them word-for-word
- Preserve the exact structure, formatting, and meaning
- Output ONLY the translation, nothing else

Translate this text to {target_language}:"""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            translation = response.choices[0].message.content.strip()
            return translation
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limit error
            if "limit exceeded" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # Longer wait for rate limits
                    print(f"  Rate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"  Rate limit persists after {max_retries} attempts")
                    return ""
            elif attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Normal exponential backoff
                print(f"  Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return ""  # Return empty string on failure
    
    return ""


def translate_dataset_prompts(
    input_path: str,
    output_path: str,
    source_field: str = "prompt",
    target_field: str = "translated_prompt",
    target_language: str = "Romanian",
    model: str = "google/gemini-2.0-flash-001",
    batch_size: int = 100,
    start_index: int = 0,
    max_samples: Optional[int] = None,
    skip_existing: bool = True,
    max_workers: int = 200
):
    """
    Translate prompts in a dataset and add translated_prompt column.
    
    Args:
        input_path: Path to input dataset
        output_path: Path to save translated dataset
        source_field: Field to translate (default: "prompt")
        target_field: Name of the new translated field (default: "translated_prompt")
        target_language: Target language for translation
        model: OpenRouter model to use
        batch_size: Save progress every N rows
        start_index: Start from this row index (for resuming)
        max_samples: Maximum number of samples to translate (None = all)
        skip_existing: Skip rows that already have translation
        max_workers: Number of parallel workers (default: 100)
    """
    print("=" * 70)
    print("PROMPT TRANSLATION WITH OPENROUTER (PARALLEL)")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Source field: {source_field}")
    print(f"Target field: {target_field}")
    print(f"Target language: {target_language}")
    print(f"Model: {model}")
    print(f"Batch size: {batch_size}")
    print(f"Parallel workers: {max_workers}")
    print("=" * 70)
    
    # Check API key
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set!")
    
    # Initialize OpenRouter client
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY
    )
    
    # Load dataset
    print("\n1. Loading dataset...")
    if os.path.exists(input_path):
        dataset = load_from_disk(input_path)
    else:
        from datasets import load_dataset
        dataset = load_dataset(input_path, split="train")
    
    print(f"Dataset loaded: {len(dataset):,} rows")
    print(f"Columns: {dataset.column_names}")
    
    # Validate source field exists
    if source_field not in dataset.column_names:
        raise ValueError(f"Source field '{source_field}' not found in dataset!")
    
    # Check if target field already exists
    if target_field in dataset.column_names:
        print(f"\nWarning: '{target_field}' already exists in dataset")
        if skip_existing:
            print("Will skip rows that already have translations")
    else:
        # Add empty target field
        print(f"\nAdding new column '{target_field}'...")
        dataset = dataset.add_column(target_field, [""] * len(dataset))
    
    # Determine range to process
    end_index = len(dataset) if max_samples is None else min(start_index + max_samples, len(dataset))
    
    # Pre-filter to find rows that need translation
    print(f"\n2. Finding rows that need translation...")
    rows_to_translate = []
    already_translated = 0
    empty_source = 0
    
    for idx in range(start_index, end_index):
        row = dataset[idx]
        # Skip if already translated
        if skip_existing and target_field in row and row.get(target_field, '').strip():
            already_translated += 1
            continue
        # Skip if empty source
        if not row.get(source_field, '').strip():
            empty_source += 1
            continue
        rows_to_translate.append(idx)
    
    total_to_process = len(rows_to_translate)
    
    print(f"Status:")
    print(f"  Already translated: {already_translated:,}")
    print(f"  Empty source: {empty_source:,}")
    print(f"  Need translation: {total_to_process:,}")
    print(f"  Total checked: {end_index - start_index:,}")
    
    if total_to_process == 0:
        print("All rows already translated! Nothing to do.")
        return load_from_disk(output_path) if os.path.exists(output_path) else dataset
    
    # Statistics
    translated = 0
    skipped = 0
    failed = 0
    last_print_count = 0
    
    # Accumulate all updates in memory
    all_updates = {}
    
    # Helper function for parallel translation
    def translate_row(idx: int) -> Tuple[int, dict, bool, bool]:
        """Translate a single row. Returns (idx, row_dict, success, should_skip)"""
        row = dataset[idx]
        
        # Skip if already translated and skip_existing is True
        if skip_existing and target_field in row and row[target_field] and row[target_field].strip():
            return (idx, None, False, True)
        
        # Get source text
        source_text = row[source_field]
        
        if not source_text or not source_text.strip():
            # Skip empty prompts
            return (idx, None, False, True)
        
        # Translate
        translated_text = translate_text(
            source_text,
            client,
            model=model,
            target_language=target_language
        )
        
        # Update row
        row_dict = dict(row)
        row_dict[target_field] = translated_text
        
        success = bool(translated_text)
        return (idx, row_dict, success, False)
    
    # Process in parallel with ThreadPoolExecutor
    print(f"\n3. Starting parallel translation with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=total_to_process, desc="Translating") as pbar:
            # Submit only tasks that need translation
            future_to_idx = {
                executor.submit(translate_row, idx): idx 
                for idx in rows_to_translate
            }
            
            # Process completed tasks
            for future in as_completed(future_to_idx):
                try:
                    idx, row_dict, success, should_skip = future.result()
                    
                    if should_skip:
                        skipped += 1
                    elif success:
                        translated += 1
                        all_updates[idx] = row_dict
                        
                        # Print sample every 100 translations
                        if translated - last_print_count >= 100:
                            last_print_count = translated
                            print(f"\n{'='*70}")
                            print(f"SAMPLE TRANSLATION #{translated}")
                            print(f"{'='*70}")
                            print(f"Original ({source_field}):")
                            print(f"  {row_dict[source_field][:200]}...")
                            print(f"\nTranslated ({target_field}):")
                            print(f"  {row_dict[target_field][:200]}...")
                            print(f"{'='*70}\n")
                    else:
                        failed += 1
                        if row_dict:  # Still save the row with empty translation
                            all_updates[idx] = row_dict
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "translated": translated, 
                        "skipped": skipped, 
                        "failed": failed,
                        "updates": len(all_updates)
                    })
                    
                    # Save checkpoint every batch_size updates
                    if len(all_updates) >= batch_size:
                        print(f"\nSaving checkpoint with {len(all_updates)} updates...")
                        _save_checkpoint(dataset, all_updates, output_path)
                        all_updates = {}  # Clear after saving
                        
                except Exception as e:
                    print(f"\nError processing row: {e}")
                    failed += 1
                    pbar.update(1)
            
            # Save final updates
            if all_updates:
                print(f"\nSaving final updates ({len(all_updates)} rows)...")
                _save_checkpoint(dataset, all_updates, output_path)
    
    # Load and return final dataset
    print("\n3. Loading final dataset...")
    final_dataset = load_from_disk(output_path)
    
    print("\n" + "=" * 70)
    print("TRANSLATION COMPLETE")
    print("=" * 70)
    print(f"Translated: {translated:,}")
    print(f"Skipped: {skipped:,}")
    print(f"Failed: {failed:,}")
    print(f"Total processed: {total_to_process:,}")
    print(f"Output saved to: {output_path}")
    print("=" * 70)
    
    return final_dataset


def _save_checkpoint(dataset: Dataset, updates: dict, output_path: str):
    """
    Save checkpoint with accumulated updates.
    
    Args:
        dataset: Original dataset
        updates: Dictionary mapping index to row_dict
        output_path: Path to save updated dataset
    """
    if not updates:
        return
    
    import shutil
    import tempfile
    
    # Load the base dataset (either existing output or original input)
    if os.path.exists(output_path):
        # Create a temporary directory for the new version
        temp_path = output_path + "_temp"
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        
        # Load existing dataset
        current_dataset = load_from_disk(output_path)
    else:
        temp_path = output_path + "_temp"
        current_dataset = dataset
    
    # Update function for map
    def update_row(example, idx):
        if idx in updates:
            return updates[idx]
        return example
    
    # Apply updates
    updated_dataset = current_dataset.map(
        update_row,
        with_indices=True,
        desc="Applying updates",
        num_proc=1  # Single process to avoid issues
    )
    
    # Save to temporary path
    updated_dataset.save_to_disk(temp_path)
    
    # Replace old with new
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    shutil.move(temp_path, output_path)
    
    print(f"Checkpoint saved: {len(updates)} rows updated")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate prompts in a dataset using OpenRouter API"
    )
    
    # Required arguments
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save translated dataset"
    )
    
    # Optional arguments
    parser.add_argument(
        "--source_field",
        type=str,
        default="prompt",
        help="Field to translate (default: prompt)"
    )
    parser.add_argument(
        "--target_field",
        type=str,
        default="translated_prompt",
        help="Name of the new translated field (default: translated_prompt)"
    )
    parser.add_argument(
        "--target_language",
        type=str,
        default="Romanian",
        help="Target language for translation (default: Romanian)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-2.0-flash-001",
        help="OpenRouter model to use (default: google/gemini-2.0-flash-001)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Save progress every N rows (default: 500)"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start from this row index (for resuming, default: 0)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to translate (default: all)"
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        default=False,
        help="Re-translate rows that already have translations"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=50,
        help="Number of parallel workers (default: 50, reduce if hitting rate limits)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    translated_dataset = translate_dataset_prompts(
        input_path=args.input_path,
        output_path=args.output_path,
        source_field=args.source_field,
        target_field=args.target_field,
        target_language=args.target_language,
        model=args.model,
        batch_size=args.batch_size,
        start_index=args.start_index,
        max_samples=args.max_samples,
        skip_existing=not args.no_skip_existing,
        max_workers=args.max_workers
    )
    
    return translated_dataset


if __name__ == "__main__":
    main()


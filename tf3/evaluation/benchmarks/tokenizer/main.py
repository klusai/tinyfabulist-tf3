"""
Main script for tokenizer benchmarking.

This script compares Romanian-trained tokenizers with pre-trained HuggingFace tokenizers
on Romanian text.

Usage:
    python -m tf3.evaluation.benchmarks.tokenizer.main \
        --local_file tf3/evaluation/benchmarks/tokenizer/data/text.txt \
        --existing_tokenizer_paths artifacts/tokenizers/unigram_tokenizer.json artifacts/tokenizers/bpe_tokenizer.json \
        --pretrained_tokenizers gpt2 bert-base-uncased roberta-base
"""
import argparse
import os

from tf3.evaluation.benchmarks.tokenizer.benchmark import TokenizerBenchmark
from tf3.evaluation.benchmarks.tokenizer.data_loader import load_romanian_text
from tf3.evaluation.benchmarks.tokenizer.results import save_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark different tokenizers on Romanian text"
    )
    
    # Data loading arguments
    parser.add_argument(
        "--local_file",
        type=str,
        required=True,
        help="Path to local text file containing Romanian text (one text per line)",
    )
    
    # Tokenizer arguments
    parser.add_argument(
        "--existing_tokenizer_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to existing Romanian-trained tokenizer JSON files to benchmark",
    )
    parser.add_argument(
        "--pretrained_tokenizers",
        type=str,
        nargs="+",
        default=[],
        help="Pre-trained HuggingFace tokenizers to benchmark (e.g., gpt2 bert-base-uncased roberta-base)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/tokenizer_benchmarks",
        help="Directory to save benchmark results",
    )
    
    return parser.parse_args()


def main():
    """Main function to run the benchmark."""
    args = parse_args()
    
    print("=" * 80)
    print("Tokenizer Benchmark")
    print("=" * 80)
    
    # Load Romanian text
    print(f"\nLoading Romanian text from local file: {args.local_file}...")
    texts = load_romanian_text(
        local_file=args.local_file,
    )
    
    if not texts:
        raise ValueError("No texts loaded! Check your data source.")
    
    print(f"Loaded {len(texts)} text samples")
    print(f"Total characters: {sum(len(t) for t in texts):,}")
    
    # Initialize benchmark
    benchmark = TokenizerBenchmark(texts=texts)
    
    results = []
    
    # Collect all existing tokenizer paths
    existing_paths = args.existing_tokenizer_paths
    
    # Benchmark existing Romanian-trained tokenizers
    for tokenizer_path in existing_paths:
        if os.path.exists(tokenizer_path):
            print("\n" + "=" * 80)
            print(f"Benchmarking Romanian-trained tokenizer: {os.path.basename(tokenizer_path)}")
            print("=" * 80)
            try:
                tokenizer = benchmark.load_existing_tokenizer(tokenizer_path)
                # Extract tokenizer type from filename
                tokenizer_name = os.path.basename(tokenizer_path).replace("_tokenizer.json", "").replace("_", " ").title()
                metrics = benchmark.benchmark_tokenizer(tokenizer, f"Romanian-trained ({tokenizer_name})")
                results.append(metrics)
            except Exception as e:
                print(f"Error loading tokenizer {tokenizer_path}: {e}")
                print("Skipping this tokenizer.")
        else:
            print(f"\nWarning: Tokenizer path not found: {tokenizer_path}")
            print("Skipping this tokenizer.")
    
    # Benchmark pre-trained HuggingFace tokenizers
    for model_name in args.pretrained_tokenizers:
        print("\n" + "=" * 80)
        print(f"Benchmarking pre-trained tokenizer: {model_name}")
        print("=" * 80)
        try:
            tokenizer = benchmark.load_pretrained_tokenizer(model_name)
            # Use a friendly display name
            display_name = model_name.replace("/", "-").replace("_", "-")
            metrics = benchmark.benchmark_tokenizer(tokenizer, f"Pre-trained: {display_name}")
            results.append(metrics)
        except Exception as e:
            print(f"Error loading tokenizer {model_name}: {e}")
            print("Skipping this tokenizer.")
    
    # Save results
    if results:
        print("\n" + "=" * 80)
        print("Saving results...")
        print("=" * 80)
        save_results(
            results=results,
            output_dir=args.output_dir,
            format="both",
        )
    else:
        print("\nNo results to save!")
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


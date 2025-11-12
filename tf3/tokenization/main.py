"""
This file contains the main function for training the tokenizers.
"""

import argparse
import os
from datetime import datetime

from tokenizers import SentencePieceBPETokenizer, SentencePieceUnigramTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--vocab_size", type=int, default=32000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Config
    input_path = os.path.join(args.artifacts_dir, "ds-tf2-en-ro-3m.txt")

    # Ensure artifacts directory exists
    os.makedirs(args.artifacts_dir, exist_ok=True)

    # Validate input corpus
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Corpus not found at: {input_path}")

    def train_bpe_tokenizer(output_path: str):
        tokenizer = SentencePieceBPETokenizer()

        tokenizer.train(
            files=[input_path],
            vocab_size=args.vocab_size,
            min_frequency=5,  # filter rare junk
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            limit_alphabet=1000,  # keep alphabet clean (avoid including weird rare Unicode chars)
        )

        json_out = os.path.join(output_path, f"bpe_tokenizer.json")
        tokenizer.save(json_out, pretty=True)

        print("Saved artifacts:")
        print(f"- HF BPE tokenizer JSON:   {json_out}")

    def train_unigram_tokenizer(output_path: str):
        hf_tokenizer = SentencePieceUnigramTokenizer()
        hf_tokenizer.train(
            files=[input_path],
            vocab_size=args.vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        )
        json_out = os.path.join(output_path, f"unigram_tokenizer.json")
        hf_tokenizer.save(json_out, pretty=True)

        print("Saved artifacts:")
        print(f"- HF Unigram tokenizer JSON:   {json_out}")

    # Timestamped prefix for outputs
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_path = os.path.join(args.artifacts_dir, f"tokenizers_{timestamp}")
    os.makedirs(output_path, exist_ok=True)  # ensure the output directory exists

    train_bpe_tokenizer(output_path)
    train_unigram_tokenizer(output_path)

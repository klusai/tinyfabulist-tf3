"""
Benchmark module for comparing different tokenizers.
"""
import time
from typing import Dict, List

import numpy as np
from tokenizers import Tokenizer as RustTokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class TokenizerBenchmark:
    """Benchmark class for comparing tokenizers."""
    
    def __init__(self, texts: List[str]):
        """
        Initialize the benchmark.
        
        Args:
            texts: List of Romanian text strings to tokenize
        """
        self.texts = texts
        
    def load_existing_tokenizer(self, tokenizer_path: str) -> PreTrainedTokenizerFast:
        """Load an existing tokenizer from a JSON file."""
        # Try loading via Rust tokenizer first to configure unk_id properly
        try:
            rust_tokenizer = RustTokenizer.from_file(tokenizer_path)
            # Check if unk_id is set
            if hasattr(rust_tokenizer, "model"):
                model = rust_tokenizer.model
                if hasattr(model, "get_unk_id"):
                    unk_id = model.get_unk_id()
                    if unk_id is None or unk_id < 0:
                        # Try to find and set unk_id
                        vocab = rust_tokenizer.get_vocab()
                        if "<unk>" in vocab:
                            unk_token_id = vocab["<unk>"]
                            if hasattr(model, "set_unk_id"):
                                model.set_unk_id(unk_token_id)
            # Now wrap in PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=rust_tokenizer)
        except Exception:
            # Fallback to direct loading
            try:
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            except Exception as e:
                # If loading fails, try with legacy=False
                try:
                    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, legacy=False)
                except Exception:
                    raise ValueError(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        
        # Ensure all special tokens are set (matching training configuration)
        special_tokens = {}
        if tokenizer.pad_token is None:
            special_tokens["pad_token"] = "<pad>"
        if tokenizer.unk_token is None:
            special_tokens["unk_token"] = "<unk>"
        if tokenizer.bos_token is None:
            special_tokens["bos_token"] = "<bos>"
        if tokenizer.eos_token is None:
            special_tokens["eos_token"] = "<eos>"
        
        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)
        
        # For SentencePiece tokenizers, try to configure unk_id properly
        # This is needed because SentencePiece requires unk_id to be set in the model
        try:
            if hasattr(tokenizer, "backend_tokenizer") and tokenizer.backend_tokenizer is not None:
                backend = tokenizer.backend_tokenizer
                if hasattr(backend, "model"):
                    model = backend.model
                    # Try to get or set unk_id
                    if hasattr(model, "get_unk_id"):
                        unk_id = model.get_unk_id()
                        # If unk_id is not set, try to find and set it
                        if unk_id is None or unk_id < 0:
                            try:
                                # Find the <unk> token ID in vocabulary
                                if "<unk>" in tokenizer.get_vocab():
                                    unk_token_id = tokenizer.get_vocab()["<unk>"]
                                    if hasattr(model, "set_unk_id"):
                                        model.set_unk_id(unk_token_id)
                            except Exception:
                                pass
        except Exception:
            # If we can't configure unk_id, continue anyway - tokenization might still work
            pass
        
        # Ensure unk_token_id is set if unk_token exists
        # This is critical for SentencePiece tokenizers
        if tokenizer.unk_token is not None:
            try:
                # Get the unk token ID from vocabulary
                unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
                if unk_token_id is not None:
                    # Try multiple ways to set unk_id in the backend
                    # Method 1: Via backend_tokenizer
                    if hasattr(tokenizer, "backend_tokenizer") and tokenizer.backend_tokenizer is not None:
                        backend = tokenizer.backend_tokenizer
                        if hasattr(backend, "model"):
                            model = backend.model
                            if hasattr(model, "get_unk_id"):
                                current_unk_id = model.get_unk_id()
                                if current_unk_id is None or current_unk_id < 0:
                                    if hasattr(model, "set_unk_id"):
                                        model.set_unk_id(unk_token_id)
                    
                    # Method 2: Via _tokenizer (Rust tokenizer)
                    if hasattr(tokenizer, "_tokenizer"):
                        rust_tokenizer = tokenizer._tokenizer
                        if hasattr(rust_tokenizer, "model"):
                            model = rust_tokenizer.model
                            if hasattr(model, "get_unk_id"):
                                current_unk_id = model.get_unk_id()
                                if current_unk_id is None or current_unk_id < 0:
                                    if hasattr(model, "set_unk_id"):
                                        model.set_unk_id(unk_token_id)
            except Exception:
                # If we can't set unk_id, continue - error handling in benchmark will catch it
                pass
        
        # Test tokenization with a simple string to verify unk_id is working
        # This will catch the error early and allow us to fix it
        try:
            test_result = tokenizer("test", add_special_tokens=False)
            # If this works, tokenizer is configured correctly
        except Exception as e:
            if "unk_id" in str(e).lower() or "unknown token" in str(e).lower():
                # Last attempt: try to reload the tokenizer with proper configuration
                # Sometimes reloading helps
                try:
                    # Get vocab to find unk_id
                    vocab = tokenizer.get_vocab()
                    if "<unk>" in vocab:
                        unk_id = vocab["<unk>"]
                        # Try accessing the model via different paths
                        if hasattr(tokenizer, "backend_tokenizer"):
                            backend = tokenizer.backend_tokenizer
                            if backend and hasattr(backend, "model"):
                                model = backend.model
                                # Force set unk_id
                                if hasattr(model, "set_unk_id"):
                                    model.set_unk_id(unk_id)
                        # Test again
                        _ = tokenizer("test", add_special_tokens=False)
                except Exception:
                    # If we still can't fix it, the error will be caught during benchmarking
                    pass
        
        return tokenizer
    
    def load_pretrained_tokenizer(self, model_name: str) -> PreTrainedTokenizerFast:
        """
        Load a pre-trained tokenizer from HuggingFace.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "gpt2", "bert-base-uncased")
            
        Returns:
            PreTrainedTokenizerFast instance
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Ensure it's a Fast tokenizer
            if not isinstance(tokenizer, PreTrainedTokenizerFast):
                # Try to get the fast version
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            return tokenizer
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer '{model_name}': {e}")
    
    def benchmark_tokenizer(
        self,
        tokenizer: PreTrainedTokenizerFast,
        name: str,
    ) -> Dict[str, float]:
        """
        Benchmark a tokenizer on the provided texts.
        
        Args:
            tokenizer: The tokenizer to benchmark
            name: Name of the tokenizer for logging
            
        Returns:
            Dictionary with benchmark metrics
        """
        print(f"\nBenchmarking {name}...")
        
        # Tokenize all texts
        start_time = time.time()
        tokenized_texts = []
        token_counts = []
        char_counts = []
        
        for text in self.texts:
            # Count characters (before tokenization)
            char_count = len(text)
            char_counts.append(char_count)
            
            # Tokenize with error handling for unknown tokens
            try:
                encoded = tokenizer(text, return_attention_mask=False, add_special_tokens=False)
                token_ids = encoded["input_ids"]
            except Exception as e:
                # If tokenization fails due to unknown tokens, try with error handling
                if "unk_id" in str(e).lower() or "unknown token" in str(e).lower():
                    # Try encoding with explicit handling
                    try:
                        # Use encode method which might handle errors better
                        token_ids = tokenizer.encode(text, add_special_tokens=False)
                    except Exception:
                        # Last resort: skip this text or use empty list
                        print(f"Warning: Failed to tokenize text (length {char_count} chars), skipping...")
                        token_ids = []
                else:
                    raise  # Re-raise if it's a different error
            
            token_counts.append(len(token_ids))
            tokenized_texts.append(token_ids)
        
        tokenization_time = time.time() - start_time
        
        # Calculate metrics
        total_chars = sum(char_counts)
        total_tokens = sum(token_counts)
        avg_tokens_per_text = np.mean(token_counts)
        avg_chars_per_text = np.mean(char_counts)
        avg_tokens_per_char = total_tokens / total_chars if total_chars > 0 else 0
        compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
        
        # Calculate tokens per word (approximate)
        total_words = sum(len(text.split()) for text in self.texts)
        avg_tokens_per_word = total_tokens / total_words if total_words > 0 else 0
        
        # Calculate vocabulary usage
        all_token_ids = set()
        for token_ids in tokenized_texts:
            all_token_ids.update(token_ids)
        vocab_usage = len(all_token_ids)
        vocab_usage_ratio = vocab_usage / len(tokenizer) if len(tokenizer) > 0 else 0
        
        # Calculate speed metrics
        texts_per_second = len(self.texts) / tokenization_time if tokenization_time > 0 else 0
        tokens_per_second = total_tokens / tokenization_time if tokenization_time > 0 else 0
        
        metrics = {
            "name": name,
            "vocab_size": len(tokenizer),
            "num_texts": len(self.texts),
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "total_words": total_words,
            "avg_tokens_per_text": avg_tokens_per_text,
            "avg_chars_per_text": avg_chars_per_text,
            "avg_tokens_per_char": avg_tokens_per_char,
            "avg_tokens_per_word": avg_tokens_per_word,
            "compression_ratio": compression_ratio,
            "vocab_usage": vocab_usage,
            "vocab_usage_ratio": vocab_usage_ratio,
            "tokenization_time_seconds": tokenization_time,
            "texts_per_second": texts_per_second,
            "tokens_per_second": tokens_per_second,
            "std_tokens_per_text": np.std(token_counts),
            "min_tokens_per_text": np.min(token_counts),
            "max_tokens_per_text": np.max(token_counts),
        }
        
        print(f"  Vocab size: {metrics['vocab_size']}")
        print(f"  Total tokens: {metrics['total_tokens']}")
        print(f"  Avg tokens/text: {metrics['avg_tokens_per_text']:.2f}")
        print(f"  Avg tokens/word: {metrics['avg_tokens_per_word']:.2f}")
        print(f"  Compression ratio: {metrics['compression_ratio']:.2f}")
        print(f"  Tokenization speed: {metrics['tokens_per_second']:.0f} tokens/sec")
        
        return metrics


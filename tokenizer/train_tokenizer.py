import os
from datetime import datetime
from tokenizers import SentencePieceUnigramTokenizer, SentencePieceBPETokenizer

# Config
ARTIFACTS_DIR = "artifacts" # path to save the tokenizers
INPUT_PATH = os.path.join(ARTIFACTS_DIR, "ds-tf2-en-ro-3m.txt")
VOCAB_SIZE = 32000

# Ensure artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Validate input corpus
if not os.path.isfile(INPUT_PATH):
    raise FileNotFoundError(f"Corpus not found at: {INPUT_PATH}")


def train_bpe_tokenizer(output_path: str):
    tokenizer = SentencePieceBPETokenizer()

    tokenizer.train(    
            files=[INPUT_PATH],   
            vocab_size=32000,    
            min_frequency=5,    # filter rare junk
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            limit_alphabet=1000  # keep alphabet clean (avoid including weird rare Unicode chars)
    )

    json_out = os.path.join(output_path, f"bpe_tokenizer.json")
    tokenizer.save(json_out, pretty=True)

    print("Saved artifacts:")
    print(f"- HF BPE tokenizer JSON:   {json_out}")


def train_unigram_tokenizer(output_path: str):
    hf_tokenizer = SentencePieceUnigramTokenizer()
    hf_tokenizer.train(files=[INPUT_PATH], vocab_size=VOCAB_SIZE, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])
    json_out = os.path.join(output_path, f"unigram_tokenizer.json")
    hf_tokenizer.save(json_out, pretty=True)

    print("Saved artifacts:")
    print(f"- HF Unigram tokenizer JSON:   {json_out}")


if __name__ == "__main__":
    # Timestamped prefix for outputs
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_path = os.path.join(ARTIFACTS_DIR, f"tokenizers_{timestamp}")
    os.makedirs(output_path, exist_ok=True) # ensure the output directory exists

    train_bpe_tokenizer(output_path)
    train_unigram_tokenizer(output_path)
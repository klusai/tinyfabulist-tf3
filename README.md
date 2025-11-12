## Romanian (<50M) From‑Scratch Language Model

This repository trains a decoder‑only language model from scratch for Romanian, targeting a compact corpus (~1B tokens).

### Why from scratch?
- **Language focus**: Romanian morphology benefits from a tokenizer trained on Romanian text.
- **Smaller corpus**: Enables quick iteration on architecture/regularization without heavy compute.
- **Control**: Full control over special tokens, normalization, and vocabulary.

## Data
- **Source**: `klusai/ds-tf2-en-ro-3m` (Hugging Face Datasets)
- **Column used**: `translated_fable` (Romanian side only)
- **Scale**: After filtering and tokenization, the total is intended to be under ~1B tokens.

You can swap the dataset with your own as long as you expose a single text column and update `preprocess.py` accordingly.

## Tokenizer
- **Type**: SentencePiece **Unigram** (also builds an optional BPE variant for comparison)
- **Vocab size**: 32,000
- **Special tokens**: `<pad>`, `<unk>`, `<bos>`, `<eos>`
- **Files produced**: timestamped JSONs under `artifacts/tokenizers_<timestamp>/`.

Train the tokenizers:
```bash
python tokenizer/train_tokenizer.py
```
Outputs (example):
- `artifacts/tokenizers_YYYY_MM_DD_HH_MM_SS/unigram_tokenizer.json`
- `artifacts/tokenizers_YYYY_MM_DD_HH_MM_SS/bpe_tokenizer.json`

Notes:
- Unigram is often preferable for morphologically rich languages like Romanian; it typically yields better subword splits than BPE at the same vocab size.

## Preprocessing
Creates contiguous 2048‑token chunks for causal LM training.

- Loads `klusai/ds-tf2-en-ro-3m` and keeps only `translated_fable`.
- Uses the local tokenizer JSON (no Hub downloads required).
- Saves an Arrow dataset ready for training.

Update `TOKENIZER_PATH` in `preprocess.py` to the Unigram JSON you just trained, then run:
```bash
python preprocess.py
```
Outputs:
- `artifacts/ds-tf2-en-ro-3m-tokenized` (DatasetDict on disk)

## Acknowledgements
- Hugging Face Datasets/Transformers/Tokenizers
- Google SentencePiece
- Community datasets for Romanian text

## Todo
- mamba vs transformers benchmarks
- quantization benchmarks 
- ablation studies
- finetuning 
- generate 3M fables
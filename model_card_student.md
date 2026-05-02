---
license: apache-2.0
language:
- ro
library_name: transformers
pipeline_tag: text-generation
tags:
- llama
- romanian
- synthetic-data
- distillation
- tinyfabulist
- fables
base_model: klusai/tf3-50m-base
datasets:
- klusai/ds-tf2-en-ro-15k
---

# TF3 Student: Distilled Romanian Language Model

A compact **22.9M-parameter** Romanian language model distilled from the [TF3-50M teacher](https://huggingface.co/klusai/tf3-50m-base) using logit-based knowledge distillation. Part of the [TinyFabulist](https://arxiv.org/abs/2601.10410) research project.

## Model Details

| Property | Value |
|----------|-------|
| Parameters | 22.9M (26.45M with untied embeddings) |
| Architecture | LLaMA-style decoder-only Transformer |
| Hidden size | 384 |
| Attention heads | 6 (head dim 64) |
| Layers | 6 |
| MLP intermediate | 1,024 |
| Vocab size | 32,000 (Unigram, Romanian-specific) |
| Context length | 2,048 tokens |
| Tied embeddings | Yes |
| Training | Knowledge distillation from klusai/tf3-50m-base |

## Training

- **Method**: Logit-based knowledge distillation (KL + CE loss, alpha=0.009)
- **Teacher**: [klusai/tf3-50m-base](https://huggingface.co/klusai/tf3-50m-base) (51.65M params, frozen)
- **Data**: [klusai/ds-tf2-en-ro-15k](https://huggingface.co/datasets/klusai/ds-tf2-en-ro-15k) (15k Romanian fables)
- **Temperature**: T=1.0
- **Epochs**: 3
- **Learning rate**: 3e-4 (cosine schedule, 50-step warmup)
- **Hardware**: Apple M3 Ultra (96GB unified memory)

## Intended Use

This model is a research artifact demonstrating knowledge distillation for compact Romanian language models trained on synthetic moral microfiction. It is designed for:

- Research on compact language model compression
- Romanian text generation in the fable/moral story domain
- Downstream fine-tuning for Romanian NLP tasks

**Not intended for**: Production text generation, factual question answering, or safety-critical applications.

## Limitations

- Domain-restricted to moral microfiction (fables)
- Trained exclusively on synthetic data
- May exhibit repetitive patterns and simplified phrasing compared to the teacher
- Gender agreement errors may occur in generated text

## Citation

```bibtex
@article{nadas2026tf3,
  title={TF3-RO-50M: Training Compact Romanian Language Models from Scratch on Synthetic Moral Microfiction},
  author={Nada\c{s}, Mihai Dan and Dio\c{s}an, Laura and Tomescu, Andreea and Pi\c{s}coran, Andrei},
  journal={arXiv preprint arXiv:2601.10410},
  year={2026}
}
```

## Related Models and Datasets

| Artifact | Description |
|----------|-------------|
| [klusai/tf3-50m-base](https://huggingface.co/klusai/tf3-50m-base) | Teacher model (51.65M) |
| [klusai/tf3-50m-sft](https://huggingface.co/klusai/tf3-50m-sft) | SFT-tuned teacher |
| [klusai/tf3-bert](https://huggingface.co/klusai/tf3-bert) | NER model for entity coherence evaluation |
| [klusai/ds-tf2-en-ro-3m](https://huggingface.co/datasets/klusai/ds-tf2-en-ro-3m) | 3M bilingual fable corpus |
| [klusai/ds-tf2-en-ro-15k](https://huggingface.co/datasets/klusai/ds-tf2-en-ro-15k) | 15k curated subset for distillation/SFT |

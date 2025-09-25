"""
Transform conll dataset to hf dataset
"""

import datasets
from datasets import ClassLabel, Features, Sequence, Value
from labels import get_labels


def conll_to_hf_dataset(path, label2id):
    sentences = []
    tags = []
    with open(path, encoding="utf-8") as f:
        tokens, ner_tags = [], []
        for line in f:
            line = line.strip()
            if not line:  # end of sentence
                if tokens:
                    sentences.append(tokens)
                    tags.append(ner_tags)
                    tokens, ner_tags = [], []
            else:
                word, tag = line.split()
                tokens.append(word)
                ner_tags.append(label2id[tag])
        if tokens:  # last sentence
            sentences.append(tokens)
            tags.append(ner_tags)

    labels = sorted(label2id, key=label2id.get)
    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(ClassLabel(names=labels)),
        }
    )

    return datasets.Dataset.from_dict(
        {"tokens": sentences, "ner_tags": tags}, features=features
    )


def transform_conll_to_hf_dataset(ner_dataset, hf_dataset):
    _, label2id, _ = get_labels()

    # transform conll dataset to hf dataset
    dataset = conll_to_hf_dataset(ner_dataset, label2id)
    dataset.save_to_disk(hf_dataset)

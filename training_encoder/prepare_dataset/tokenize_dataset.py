"""
Tokenize the dataset

- tokenize the dataset
- align the labels
- save the tokenized dataset

this tokenization will force to use aggregation strategy to be first.
"""

import datasets
import evaluate
from labels import get_labels
from transformers import AutoTokenizer


def tokenize_and_align_labels(data, tokenizer, labels):
    tokenized = tokenizer(data["tokens"], truncation=True, is_split_into_words=True)

    labels_aligned = []
    for i, label in enumerate(data["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = []
        previous_word = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)  # ignored in loss
            elif (
                word_id != previous_word
            ):  # only first token of the word is tagged, rest are -100
                aligned.append(label[word_id])
            else:
                aligned.append(-100)  # mask subword continuations
            previous_word = word_id
        labels_aligned.append(aligned)

    tokenized["labels"] = labels_aligned
    return tokenized


def tokenize_dataset(ner_dataset, hf_dataset):
    labels, label2id, id2label = get_labels()

    # Load the tokenizer and model
    model_checkpoint = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Load the dataset
    dataset = datasets.load_from_disk(ner_dataset)
    if len(dataset) == 0:
        raise ValueError(
            f"Loaded dataset at {ner_dataset} has 0 examples. Aborting tokenization."
        )
    tokenized_datasets = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "labels": labels},
    )

    # Remove text columns the model can't handle
    tokenized_datasets = tokenized_datasets.remove_columns(["tokens", "ner_tags"])

    # Save the tokenized dataset
    tokenized_datasets.save_to_disk(hf_dataset)

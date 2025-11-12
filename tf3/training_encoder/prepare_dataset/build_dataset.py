"""
Build dataset from checkpoints and annotations

1. Build merged dataset from checkpoints
- each checkpoint has a file called ro_sentences.txt
- merge all the files into one file
2. Build NER dataset from merged dataset and annotations
- load the annotations
- lemmatize the entities and locations
- build the NER dataset(conll file)
"""

import os
import re
import unicodedata
from typing import List

import stanza
import yaml


def strip_diacritics(text: str) -> str:
    """Remove diacritics by Unicode NFD normalization and dropping combining marks."""
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", without_marks)


def normalize_token(text: str) -> str:
    """Lowercase and strip diacritics for robust matching (e.g., pădure == padure)."""
    return strip_diacritics(text.lower())


# Preprocess entity dictionaries: lemmatize them once
def lemmatize_list(words, nlp):
    doc = nlp(" ".join(words))
    return set([w.lemma.lower() for sent in doc.sentences for w in sent.words])


def get_all_subfolders(folder_name: str) -> List[str]:
    # Get all subfolders recursively
    subfolders = [f.path for f in os.scandir(folder_name) if f.is_dir()]

    if len(subfolders) == 0:
        return [folder_name]

    checkpoints = []
    for subfolder in subfolders:
        checkpoints.extend(get_all_subfolders(subfolder))
    return checkpoints


def get_all_checkpoints_datasets(folder_name: str) -> List[str]:
    subfolders = get_all_subfolders(folder_name)
    return list(filter(lambda x: x.split("/")[-2].startswith("mamba"), subfolders))


def build_merged_dataset(checkpoints: List[str], out_path: str):
    with open(out_path, "w") as f_out:  # overwrite existing file
        for checkpoint in checkpoints:
            dataset_path = os.path.join(checkpoint, "ro_sentences.txt")
            if not os.path.exists(dataset_path):
                continue
            with open(dataset_path, "r") as f_in:
                for line in f_in:
                    f_out.write(line)
    return out_path


# Build NER dataset from merged dataset and annotations
def build_ner_dataset(yaml_path: str, full_dataset_path: str, out_path: str):
    # Load your entity dictionary
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    entities = data["generator"]["features"]["characters"]
    locations = data["generator"]["features"]["settings"]

    nlp = stanza.Pipeline(
        "ro",
        processors="tokenize,pos,lemma",
        tokenize_pretokenized=False,
        verbose=False,
    )

    entities_lemma = [word.lower() for word in lemmatize_list(entities, nlp)]
    locations_lemma = [word.lower() for word in lemmatize_list(locations, nlp)]

    # Build diacritics-insensitive lemma sets
    entities_lemma_norm = set(normalize_token(word) for word in entities_lemma)
    locations_lemma_norm = set(normalize_token(word) for word in locations_lemma)

    # Load your raw text
    with open(full_dataset_path, "r", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text)
    lines = []

    for sentence in doc.sentences:
        for word in sentence.words:
            token = word.text
            lemma = word.lemma.lower()
            lemma_norm = normalize_token(lemma)
            pos = (
                word.upos
            )  # Universal POS tag (NOUN, VERB, ADP, etc.) - to fix polisemy problems

            label = "O"
            if lemma_norm in entities_lemma_norm:
                # only tag if it's a noun
                if pos == "NOUN" or pos == "PROPN":
                    label = "B-ENTITY"
            elif lemma_norm in locations_lemma_norm:
                if pos == "NOUN" or pos == "PROPN":
                    label = "B-LOCATION"

            lines.append(f"{token}\t{label}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

import os
from typing import List
import yaml
import re
import stanza
import unicodedata


ARTIFACTS = "evaluation/artifacts/evaluation"
FULL_DATASET = "training_encoder/full_dataset.txt"

# Utilities for diacritics-insensitive matching

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


def build_ner_dataset(yaml_path: str, full_dataset_path: str, out_path: str):
    # Load your entity dictionary
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    entities = data["generator"]["features"]["characters"]
    locations = data["generator"]["features"]["settings"]

    nlp = stanza.Pipeline("ro", processors="tokenize,pos,lemma", tokenize_pretokenized=False, verbose=False)

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
            pos = word.upos  # Universal POS tag (NOUN, VERB, ADP, etc.) - to fix polisemy problems

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


def get_characters(folder_name: str) -> List[str]:
    with open(folder_name + "/annotations.yaml", "r") as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
    return yaml_data["generator"]["features"]["characters"]

def build_full_dataset(checkpoints: List[str], out_path: str):
    for checkpoint in checkpoints:
        dataset_path = checkpoint + "/ro_sentences.txt"

        if not os.path.exists(dataset_path):
            continue

        with open(dataset_path, "r") as f:
            dataset = f.readlines()
            with open(out_path, "a") as f_append:
                for line in dataset:
                    f_append.write(line)

def build_anotated_dataset(full_dataset_path: str, out_path: str):
    with open(full_dataset_path, "r") as f_full:
        with open(out_path, "a") as f_append:
            for line in f_full:
                f_append.write(line)

if __name__ == "__main__":
    if not os.path.exists(FULL_DATASET):
        build_full_dataset(get_all_checkpoints_datasets(ARTIFACTS), FULL_DATASET)
    else:
        print(f"Dataset {FULL_DATASET} already exists")

    build_ner_dataset("training_encoder/annotations.yaml", FULL_DATASET, "training_encoder/ner_dataset.conll")

    
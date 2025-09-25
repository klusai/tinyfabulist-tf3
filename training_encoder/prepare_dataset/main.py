import argparse
from datetime import datetime
import os

from build_dataset import build_merged_dataset, build_ner_dataset, get_all_checkpoints_datasets
from conll_to_hf_json import transform_conll_to_hf_dataset
from tokenize_dataset import tokenize_dataset

def parse_args(checkpoint_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Romanian eval sentences in parallel.")
    parser.add_argument("--artifacts", default="evaluation/artifacts/evaluation", help="Model path or HF id")
    parser.add_argument("--merged-dataset", default=f"training_encoder/artifacts/{checkpoint_name}/full_dataset.txt", help="Path to the full dataset")
    parser.add_argument("--ner-dataset", default=f"training_encoder/artifacts/{checkpoint_name}/ner_dataset.conll", help="Path to the NER dataset")
    parser.add_argument("--hf-dataset", default=f"training_encoder/artifacts/{checkpoint_name}/ner_dataset", help="Path to the HF dataset")
    parser.add_argument("--annotations", default="training_encoder/annotations.yaml", help="Path to the annotations")
    parser.add_argument("--hf-dataset-tokenized", default=f"training_encoder/artifacts/{checkpoint_name}/ner_dataset_tokenized", help="Path to the tokenized HF dataset")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args(datetime.now().strftime("%Y-%m-%d"))

    for argument in args.__dict__:
        if not os.path.exists(args.__dict__[argument]):
            os.makedirs(os.path.dirname(args.__dict__[argument]), exist_ok=True)

    # build conll dataset
    print("Building merged dataset from checkpoints")
    checkpoints = get_all_checkpoints_datasets(args.artifacts)
    merged_dataset_path = build_merged_dataset(checkpoints, args.merged_dataset)
    print("Building NER dataset from merged dataset and annotations")
    build_ner_dataset(args.annotations, args.merged_dataset, args.ner_dataset)

    # transform conll dataset to hf dataset
    print("Transforming conll dataset to hf dataset")
    transform_conll_to_hf_dataset(args.ner_dataset, args.hf_dataset)

    # tokenize dataset
    print("Tokenizing dataset")
    tokenize_dataset(args.hf_dataset, args.hf_dataset_tokenized)
    print()
    print(f"Tokenized dataset saved at {args.hf_dataset_tokenized}")
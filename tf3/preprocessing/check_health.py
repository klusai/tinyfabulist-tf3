from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, default="artifacts/ds-tf2-en-ro-3m-tokenized"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset = load_from_disk(args.dataset_path)

    num_samples = len(dataset)  # number of blocks
    block_size = len(dataset[0]["input_ids"])
    total_tokens = num_samples * block_size

    print(f"Samples: {num_samples}")
    print(f"Block size: {block_size}")
    print(f"Total tokens: {total_tokens:,}")

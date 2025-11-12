import os
import re

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="klusai/ds-tf2-en-ro-3m")
    parser.add_argument("--output_file", type=str, default="ds-tf2-en-ro-3m.txt")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    return parser.parse_args()


def split_sentences(text: str):
    # Simple Romanian-friendly split
    sentences = re.split(r"(?<=[\.\!\?])\s+", text.strip())
    return [s for s in sentences if s]


def gen_corpus(out_file="ds-tf2-en-ro-3m.txt", out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, out_file)

    dataset = load_dataset("klusai/ds-tf2-en-ro-3m", split="train")

    with open(out_file, "w", encoding="utf-8") as f:
        for idx, item in enumerate(dataset):
            if idx % 1000 == 0:
                print(f"Processed {idx} items")
            text = item["translated_fable"].strip()
            if text.strip():  # skip empty
                for s in split_sentences(text):
                    if s.strip():
                        f.write(s.replace("\n", " ") + "\n")
    print(f"Corpus saved to {out_file}")


if __name__ == "__main__":
    args = parse_args()
    gen_corpus(args.output_path, args.dataset_path)

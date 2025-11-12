import argparse
import math
import multiprocessing as mp
import os
from datetime import datetime
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Romanian eval sentences in parallel."
    )
    parser.add_argument(
        "--model",
        default="/home/andrei/Documents/Work/tf3/artifacts/training/checkpoints/mamba50M3000/",
        help="Model path or HF id",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel worker processes"
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# Single explicit list: character + trait (gender-matched)
CHARACTERS: List[str] = [
    "vulpea isteață",
    "leul mândru",
    "cioara vicleană",
    "lupul sălbatic",
    "iepurele sprinten",
    "broasca țestoasă răbdătoare",
    "bufnița înțeleaptă",
    "șoarecele timid",
    "cerbul grațios",
    "ursul puternic",
    "barza atentă",
    "capra încăpățânată",
    "câinele loial",
    "maimuța jucăușă",
    "vulturul ager",
    "șarpele șiret",
    "oaia blândă",
    "tigrul feroce",
    "delfinul curios",
    "furnica harnică",
    "măgarul răbdător",
    "pisica curioasă",
    "cocoșul gălăgios",
    "broasca săltăreață",
    "castorul harnic",
    "lebăda elegantă",
    "taurul năvalnic",
    "zebra agilă",
    "girafa blajină",
    "rinocerul masiv",
    "hipopotamul greoi",
    "crocodilul viclean",
    "balena pașnică",
    "rechinul necruțător",
    "pinguinul stângaci",
    "gâsca grijulie",
    "papagalul vorbăreț",
    "calul nobil",
    "scorpionul periculos",
    "bivolul puternic",
    "struțul sperios",
    "cămila rezistentă",
    "flamingo grațios",
    "cangurul sprinten",
    "koala somnoroasă",
    "ornitorincul bizar",
    "ratonul șiret",
    "veverița iute",
    "ariciul precaut",
    "bursucul ursuz",
    "cârtița harnică",
    "hiena hohotitoare",
    "caracatița inteligentă",
    "crabul țâfnos",
    "homarul robust",
    "steaua de mare tăcută",
    "foca jucăușă",
    "morsa masivă",
    "iacul robust",
    "renul sprinten",
    "elanul impunător",
    "sconcsul mirositor",
    "porcul ghiduș",
    "mistrețul nărăvaș",
    "peștele agil",
    "crevetele mărunt",
    "liliacul nocturn",
    "cameleonul adaptabil",
    "gecko lipicios",
    "gorila puternică",
    "orangutanul isteț",
    "puma agilă",
    "jaguarul vânător",
    "leopardul sprinten",
    "ghepardul fulgerător",
    "cormoranul scufundător",
    "pelicanul pofticios",
    "colibri neobosit",
    "cardinalul aprins",
    "măcăleandrul vioi",
    "uliul vigilent",
    "șoimul iute",
    "vulturul pleșuv răbdător",
    "rața liniștită",
    "ciocănitoarea harnică",
    "coțofana gălăgioasă",
    "pescărușul obraznic",
    "hermelina sprintenă",
    "dihorul curios",
    "viermele modest",
    "păianjenul răbdător",
    "fluturele gingaș",
    "molia tăcută",
    "albina harnică",
    "viespea agresivă",
    "gândacul tenace",
    "greierele cântăreț",
    "lăcusta săltăreață",
    "buburuza norocoasă",
    "licuriciul strălucitor",
]


def chunk_list(items: List[str], num_chunks: int) -> List[List[str]]:
    n = max(1, num_chunks)
    size = math.ceil(len(items) / n)
    return [items[i * size : (i + 1) * size] for i in range(n)]


def worker(
    rank: int,
    model_path: str,
    out_path: str,
    chars: List[str],
    max_new: int,
    temp: float,
    top_k: int,
    seed: int,
) -> str:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else None

    # Unique seed per worker
    torch.manual_seed(seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(
        device
    )
    model.eval()

    part_path = f"{out_path}.part{rank}"
    with open(part_path, "w", encoding="utf-8") as f:
        for ch in chars:
            enc = tokenizer(ch, return_tensors="pt").to(device)
            if "token_type_ids" in enc:
                enc.pop("token_type_ids")
            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model.generate(
                            **enc,
                            max_new_tokens=max_new,
                            do_sample=True,
                            temperature=temp,
                            top_k=top_k,
                        )
                else:
                    out = model.generate(
                        **enc,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=temp,
                        top_k=top_k,
                    )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            f.write(text + "\n")
    return part_path


def build_eval_dataset(
    model_path: str,
    workers: int,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 10,
    seed: int = datetime.now().timestamp(),
):
    model_name = model_path.rstrip("/").split("/")[-1]

    output_path = os.path.join(
        os.path.dirname(__file__),
        "artifacts",
        "evaluation",
        f"{model_name}",
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "ro_sentences.txt",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare chunks
    chunks = chunk_list(CHARACTERS, workers)

    # Launch workers
    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=workers) as pool:
        results = [
            pool.apply_async(
                worker,
                kwds=dict(
                    rank=idx,
                    model_path=model_path,
                    out_path=output_path,
                    chars=chunk,
                    max_new=max_new_tokens,
                    temp=temperature,
                    top_k=top_k,
                    seed=seed,
                ),
            )
            for idx, chunk in enumerate(chunks)
        ]
        part_files = [r.get() for r in results]

    # Merge parts in order
    with open(output_path, "w", encoding="utf-8") as out_f:
        for pf in part_files:
            with open(pf, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
            os.remove(pf)

    print(
        f"Wrote {sum(1 for _ in open(output_path, 'r', encoding='utf-8'))} sentences to {output_path}"
    )


def main():
    args = parse_args()

    build_eval_dataset(
        model_path=args.model,
        workers=args.workers,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

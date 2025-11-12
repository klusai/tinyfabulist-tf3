import argparse
import os
from datetime import datetime
from typing import List

import torch
from tf3.evaluation.build_eval_dataset import build_eval_dataset
from tf3.evaluation.entity_coherence import entity_coherence_score
from tf3.evaluation.general_metrics import compute_ce_ppl, load_texts
from tf3.evaluation.llm import llm_grammar_score
from tf3.evaluation.lang_tool import language_tool_score
from tf3.evaluation.throughtput import test_throughput
from tf3.logger import get_logger
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

ARTIFACTS_FOLDER = "tf3/evaluation/artifacts"
CHECKPOINTS = ['klusai/tf3-50m-base', 'klusai/tf3-50m-base-mamba'] #"tf3/artifacts/training"
OUTPUT_PATH = "tf3/evaluation/artifacts/evaluation.log"


def get_all_files(folder_name: str) -> List[str]:
    subfolders = get_all_subfolders(folder_name)
    files = []
    for subfolder in subfolders:
        for file in os.scandir(subfolder):
            if file.is_file():
                files.append(os.path.abspath(file.path))
    return files


def get_all_subfolders(folder_name: str) -> List[str]:
    # Get all subfolders recursively
    subfolders = [f.path for f in os.scandir(folder_name) if f.is_dir()]

    if len(subfolders) == 0:
        return [folder_name]

    checkpoints = []
    for subfolder in subfolders:
        checkpoints.extend(get_all_subfolders(subfolder))
    return checkpoints


def get_all_checkpoints(checkpoints: str) -> List[str]:
    if isinstance(checkpoints, list):
        return checkpoints
    else:
        subfolders = get_all_subfolders(checkpoints)
        return list(
            filter(
                lambda x: x.split("/")[-1].startswith("mamba") and "checkpoint" in x,
                subfolders,
            )
        )


def main(
    cross_entropy: bool = False,
    throughput: bool = False,
    entity_coherence: bool = False,
    language_tool: bool = False,
    llm_grammar: bool = False,
):
    console_logger.info(f"Processing {ARTIFACTS_FOLDER}")
    for checkpoint in get_all_checkpoints(CHECKPOINTS):
        console_logger.info(f"Processing {checkpoint}")
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # Prefer bf16 on GPU capable hardware
        torch_dtype = torch.bfloat16 if device.type == "cuda" or device.type == "mps" else None
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch_dtype
        ).to(device)

        encoder_model_path = (
            "klusai/tf3-bert"
        )

        if os.path.exists(
            f"{ARTIFACTS_FOLDER}/evaluation/{checkpoint.split('/')[-1]}/"
        ):
            console_logger.info(f"Skipping {checkpoint} because it already exists")
        else:
            console_logger.info(f"Building Evaluation Dataset for {checkpoint}")
            build_eval_dataset(
                checkpoint,
                workers=8,
                max_new_tokens=200,
                temperature=0.7,
                top_k=10,
                seed=datetime.now().timestamp(),
            )

        file_path = None
        for file in get_all_files(
            f"{ARTIFACTS_FOLDER}/evaluation/{checkpoint.split('/')[-1]}/"
        ):
            if file.endswith(".txt"):
                file_path = file
                break

        if file_path is None:
            console_logger.info(f"No file found for {checkpoint}")
            continue
        else:
            console_logger.info(f"Processing {file_path}")

        args = argparse.Namespace(
            file=file_path,
            batch_size=8,
            max_length=2048,
            model=checkpoint,
        )

        texts = load_texts(args)

        if cross_entropy:
            console_logger.info(f"Computing CE and PPL for {checkpoint}")
            ce_all, ppl_all = compute_ce_ppl(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=device,
            )
            artifacts_logger.info(
                f"{checkpoint.split('/')[-1]}, CE: {ce_all:.4f}, PPL: {ppl_all:.4f}"
            )

        if throughput:
            console_logger.info(f"Computing Throughput for {checkpoint}")
            throughput = test_throughput(checkpoint, device)
            artifacts_logger.info(
                f"{checkpoint.split('/')[-1]}, Throughput: {throughput:.4f}"
            )

        if entity_coherence:
            console_logger.info(f"Computing Entity Coherence for {checkpoint}")
            entity_coherence_score_val = entity_coherence_score(texts, encoder_model_path)
            artifacts_logger.info(
                f"{checkpoint.split('/')[-1]}, Entity Coherence: {entity_coherence_score_val:.4f}"
            )

        if language_tool:
            console_logger.info(f"Computing LanguageTool Grammar Score for {checkpoint}")
            lpt_score = language_tool_score(texts)
            artifacts_logger.info(
                f"{checkpoint.split('/')[-1]}, LanguageTool: {lpt_score:.4f}"
            )

        if llm_grammar:
            console_logger.info(f"Computing LLM Grammar Score for {checkpoint}")
            try:
                llm_score = llm_grammar_score(texts)
                artifacts_logger.info(
                    f"{checkpoint.split('/')[-1]}, LLM Grammar: {llm_score:.4f}"
                )
            except Exception as e:
                console_logger.error(f"Error computing LLM grammar score: {e}")
                artifacts_logger.info(
                    f"{checkpoint.split('/')[-1]}, LLM Grammar: ERROR"
                )


if __name__ == "__main__":
    console_logger = get_logger("evaluation")
    artifacts_logger = get_logger("artifacts")
    main(
        cross_entropy=False,
        throughput=True,
        entity_coherence=False,
        language_tool=False,
        llm_grammar=False
    )

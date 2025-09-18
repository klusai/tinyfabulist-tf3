import os
from typing import List

def get_all_subfolders(folder_name: str) -> List[str]:
    # Get all subfolders recursively
    subfolders = [f.path for f in os.scandir(folder_name) if f.is_dir()]
    
    if len(subfolders) == 0:
        return [folder_name]

    checkpoints = []
    for subfolder in subfolders:
        checkpoints.extend(get_all_subfolders(subfolder))
    return checkpoints

def get_all_checkpoints(folder_name: str) -> List[str]:
    subfolders = get_all_subfolders(folder_name)
    return list(filter(lambda x: x.split("/")[-1].startswith("mamba") and "checkpoint" in x, subfolders))

def main():
    for checkpoint in get_all_checkpoints("/home/andrei/Documents/Work/tf3/artifacts"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # Prefer bf16 on GPU capable hardware
        torch_dtype = torch.bfloat16 if device.type == "cuda" else None
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch_dtype).to(device)

        texts = load_texts(checkpoint)

        ce_all, ppl_all = compute_ce_ppl(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )


if __name__ == "__main__":
    main()

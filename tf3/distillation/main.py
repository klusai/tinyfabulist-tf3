import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import MambaForCausalLM
from datasets import load_from_disk
from torch.optim import AdamW

# Import models
from tf3.distillation.student_model import model as student
from tf3.training.model import model as teacher  

# ---- Parse args ----
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint", type=str, default="artifacts/training/checkpoints/mamba50M27600")
    parser.add_argument("--dataset_path", type=str, default="artifacts/ds-tf2-en-ro-3m-tokenized")
    parser.add_argument("--output_dir", type=str, default="artifacts/distillation/checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=2.0)
    return parser.parse_args()

# ---- Distillation step ----

def collate_fn(batch):
    # batch is a list of dicts, each with "input_ids" and maybe "attention_mask"
    input_ids = [example["input_ids"] for example in batch]
    attention_mask = [
        example.get("attention_mask", [1] * len(example["input_ids"])) for example in batch
    ]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def distill_step(batch, teacher, student, optimizer, temperature=2.0):
    with torch.no_grad():
        teacher_logits = teacher(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        ).logits

    student_logits = student(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"]
    ).logits

    loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean"
    ) * (temperature ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ---- Main ----
if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Teacher (frozen, loaded from checkpoint)
    teacher = MambaForCausalLM.from_pretrained(args.teacher_checkpoint)
    teacher.to(device)
    teacher.eval()

    # Student (trainable)
    student.to(device)
    student.train()

    # Load dataset (already tokenized with "input_ids" and "attention_mask")
    dataset = load_from_disk(args.dataset_path)

    # If dataset is a dict with splits
    if isinstance(dataset, dict) or hasattr(dataset, "keys"):
        if "train" in dataset:
            train_dataset = dataset["train"]
        else:
            raise ValueError("No 'train' split found in dataset")
    else:
        # single Dataset object
        train_dataset = dataset

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    optimizer = AdamW(student.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = distill_step(batch, teacher, student, optimizer, args.temperature)
            total_loss += loss
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Avg KL Loss: {avg_loss:.4f}")

    # Save student
    os.makedirs(args.output_dir, exist_ok=True)
    student.save_pretrained(args.output_dir)
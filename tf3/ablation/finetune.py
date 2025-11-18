"""
Shared fine-tuning module for post-pruning/ablation/embedding operations.
"""
import argparse
import math
from collections import deque

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


LOSS_GROWTH_PATIENCE = 5
LOSS_GROWTH_TOLERANCE = 0.05


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def create_dataloader(
    tokenizer,
    batch_size=8,
    max_length=512,
    dataset_path="klusai/ds-tf2-en-ro-3m",
    dataset_split="train",
    max_samples=20000,
):
    """Create a DataLoader from HuggingFace dataset."""
    print(f"Loading dataset '{dataset_path}' ({dataset_split} split)...")
    hf_dataset = load_dataset(
        dataset_path,
        split=dataset_split,
        verification_mode="no_checks",
    )

    if max_samples is not None:
        take = min(max_samples, len(hf_dataset))
        hf_dataset = hf_dataset.select(range(take))
        print(f"Using {take} samples from the dataset")
    else:
        print(f"Using all {len(hf_dataset)} samples from the dataset")

    training_texts = [
        item["translated_fable"].strip()
        for item in hf_dataset
        if item.get("translated_fable") and item["translated_fable"].strip()
    ]

    dataset = TextDataset(training_texts, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return dataloader


def finetune_model(
    model,
    tokenizer,
    output_path,
    batch_size=8,
    grad_accum_steps=4,
    max_epochs=3,
    base_lr=5e-4,
    max_length=128,
    max_samples=20000,
    moving_avg_window=20,
    dataset_path="klusai/ds-tf2-en-ro-3m",
    dataset_split="train",
):
    """
    Fine-tune a model after pruning/ablation/embedding operations.
    
    Args:
        model: The model to fine-tune (should be in eval mode initially)
        tokenizer: Tokenizer for the model
        output_path: Where to save the fine-tuned model
        batch_size: Batch size for training
        grad_accum_steps: Gradient accumulation steps
        max_epochs: Maximum number of epochs
        base_lr: Base learning rate
        max_length: Maximum sequence length
        max_samples: Maximum number of samples from dataset
        moving_avg_window: Window size for moving average loss
        dataset_path: HuggingFace dataset path
        dataset_split: Dataset split to use
    
    Returns:
        The fine-tuned model
    """
    print(f"\n{'='*60}")
    print("Starting fine-tuning after model modification...")
    print(f"{'='*60}\n")
    
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create dataloader
    dataloader = create_dataloader(
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        dataset_path=dataset_path,
        dataset_split=dataset_split,
        max_samples=max_samples,
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    
    total_steps = math.ceil(len(dataloader) / grad_accum_steps) * max_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.2,
        total_iters=max(1, total_steps),
    )
    
    model.train()
    
    # Training state
    prev_loss = None
    growth_streak = 0
    stop_training = False
    moving_losses = deque(maxlen=moving_avg_window)
    optimizer.zero_grad()
    global_step = 0
    
    for epoch in range(max_epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            # Move batch to same device as model
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            loss_value = loss.item()
            (loss / grad_accum_steps).backward()
            
            total_loss += loss_value
            moving_losses.append(loss_value)
            
            # Calculate perplexity from cross-entropy loss
            ce_loss = loss_value
            ppl = math.exp(ce_loss)
            
            if i % 5 == 0:  # Print every 5 batches
                moving_avg = sum(moving_losses) / len(moving_losses) if moving_losses else 0.0
                moving_avg_ppl = math.exp(moving_avg)
                current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else base_lr
                print(
                    f"Epoch {epoch}, Batch {i} | "
                    f"CE: {ce_loss:.4f} (avg: {moving_avg:.4f}) | "
                    f"PPL: {ppl:.2f} (avg: {moving_avg_ppl:.2f}) | "
                    f"LR: {current_lr:.6f}"
                )
            
            # Early stopping check
            if prev_loss is not None and loss_value > prev_loss + LOSS_GROWTH_TOLERANCE:
                growth_streak += 1
            else:
                growth_streak = 0
            
            if growth_streak >= LOSS_GROWTH_PATIENCE:
                print(
                    f"Stopping early: loss increased for {LOSS_GROWTH_PATIENCE} consecutive "
                    f"steps (last loss {loss_value:.4f}, previous {prev_loss:.4f})."
                )
                stop_training = True
                break
            
            # Gradient accumulation step
            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            prev_loss = loss_value
        
        # Handle leftover gradients at end of epoch
        if not stop_training and (i + 1) % grad_accum_steps != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        avg_loss = total_loss / len(dataloader)
        avg_ppl = math.exp(avg_loss)
        print(f"Epoch {epoch} completed | Avg CE: {avg_loss:.4f} | Avg PPL: {avg_ppl:.2f}")
        
        if stop_training:
            print("Early stopping triggered due to sustained loss growth.")
            break
    
    # Save the fine-tuned model
    print(f"\nSaving fine-tuned model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Fine-tuning complete!\n")
    
    return model


def parse_finetune_args():
    """Parse command-line arguments for fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune model after modification")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model to fine-tune")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_epochs", type=int, default=3, help="Maximum epochs")
    parser.add_argument("--base_lr", type=float, default=5e-4, help="Base learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--max_samples", type=int, default=20000, help="Max samples from dataset")
    parser.add_argument("--dataset_path", type=str, default="klusai/ds-tf2-en-ro-3m", help="Dataset path")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split")
    return parser.parse_args()


if __name__ == "__main__":
    # Standalone fine-tuning script
    args = parse_finetune_args()
    
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    finetune_model(
        model=model,
        tokenizer=tokenizer,
        output_path=args.output_path,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_epochs=args.max_epochs,
        base_lr=args.base_lr,
        max_length=args.max_length,
        max_samples=args.max_samples,
        dataset_path=args.dataset_path,
        dataset_split=args.dataset_split,
    )


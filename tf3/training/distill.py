import torch
import torch.nn as nn
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)
import os
from tf3.training.llama.student import model as student_model
from datasets import load_dataset, load_from_disk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEACHER_TOPK = 32


def get_preferred_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def keep_only_columns(dataset, columns_to_keep):
    remove_cols = [c for c in dataset.column_names if c not in columns_to_keep]
    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)
    return dataset

def compute_logits(batch, device, teacher):
    input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
    if "attention_mask" in batch and batch["attention_mask"] is not None:
        attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
    else:
        attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits  # (B, T, V)

    k = min(TEACHER_TOPK, logits.size(-1))
    topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)

    # Move to CPU and convert to numpy immediately, then delete GPU tensors
    batch["teacher_topk_idx"] = topk_idx.cpu().short().numpy()
    batch["teacher_topk_val"] = topk_vals.cpu().half().numpy()
    
    # Clear GPU memory
    del logits, topk_vals, topk_idx, input_ids, attention_mask
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    
    return batch

# ------------------------------------------------------------
# Custom Trainer
# ------------------------------------------------------------
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=3.0, alpha=0.9, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        self.temp = temperature
        self.alpha = alpha

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        
        self.current_ce_loss = None
        self.current_kl_loss = None
        self._teacher_device = None  # Cache teacher device
    
    def _move_teacher_to_device(self, device):
        """Move teacher to device only if needed."""
        if self._teacher_device != device:
            self.teacher_model = self.teacher_model.to(device)
            self._teacher_device = device

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        # Move teacher to device only if needed (cached)
        self._move_teacher_to_device(device)

        # Convert BatchEncoding to dict if needed (BatchEncoding is dict-like but not a dict)
        if hasattr(inputs, 'to_dict'):
            inputs = inputs.to_dict()
        elif not isinstance(inputs, dict):
            # Fallback: try to convert to dict
            inputs = dict(inputs)
        
        if "input_ids" not in inputs:
            raise KeyError(f"input_ids not found in inputs. Available keys: {list(inputs.keys())}")
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        labels = inputs.get("labels", None)

        if input_ids is None:
            raise ValueError("input_ids is None")
        
        model_inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # Student forward
        student_out = model(**model_inputs)
        s_logits = student_out.logits if hasattr(student_out, "logits") else student_out
        s_shift = s_logits[:, :-1, :].contiguous()

        teacher_topk_idx = inputs.get("teacher_topk_idx")
        teacher_topk_val = inputs.get("teacher_topk_val")

        use_cached_topk = teacher_topk_idx is not None and teacher_topk_val is not None

        if use_cached_topk:
            if not isinstance(teacher_topk_idx, torch.Tensor):
                teacher_topk_idx = torch.tensor(teacher_topk_idx, dtype=torch.long, device=device)
            else:
                teacher_topk_idx = teacher_topk_idx.to(device)

            if not isinstance(teacher_topk_val, torch.Tensor):
                teacher_topk_val = torch.tensor(teacher_topk_val, dtype=torch.float32, device=device)
            else:
                teacher_topk_val = teacher_topk_val.to(device)

            t_idx_shift = teacher_topk_idx[:, :-1, :].contiguous()
            t_val_shift = teacher_topk_val[:, :-1, :].contiguous()
        else:
            with torch.no_grad():
                t_logits = self.teacher_model(**model_inputs).logits
            t_shift = t_logits[:, :-1, :].contiguous()
        
        labels_full = model_inputs["input_ids"].clone()
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        labels_full[labels_full == pad_token_id] = -100
        labels_shift = labels_full[:, 1:].contiguous()

        batch_size, seq_len, vocab_size = s_shift.shape
        s_shift_flat = s_shift.reshape(-1, vocab_size)
        labels_shift_flat = labels_shift.reshape(-1)

        mask = labels_shift_flat != -100

        if use_cached_topk:
            student_topk = torch.gather(s_shift, dim=-1, index=t_idx_shift)
            log_probs_student = nn.functional.log_softmax(student_topk / self.temp, dim=-1)
            probs_teacher = nn.functional.softmax(t_val_shift / self.temp, dim=-1)
            kl_per_token = (probs_teacher * (torch.log(probs_teacher + 1e-9) - log_probs_student)).sum(dim=-1)
            kl_flat = kl_per_token.reshape(-1)
            loss_kl = kl_flat[mask].mean() * (self.temp ** 2)
        else:
            t_shift_flat = t_shift.reshape(-1, vocab_size)
            s_shift_masked = s_shift_flat[mask]
            t_shift_masked = t_shift_flat[mask]
            loss_kl = self.kl_loss(
                nn.functional.log_softmax(s_shift_masked / self.temp, dim=-1),
                nn.functional.softmax(t_shift_masked / self.temp, dim=-1),
            ) * (self.temp ** 2)

        # CE loss (masked)
        loss_ce = self.ce_loss(
            s_shift_flat,
            labels_shift_flat
        )

        # --- ALWAYS MIX KL + CE ---
        loss = 0.9 * loss_kl + 0.1 * loss_ce

        self.current_ce_loss = loss_ce.item()
        self.current_kl_loss = loss_kl.item()
        self.current_total_loss = loss.item()

        return (loss, student_out) if return_outputs else loss
    
    def log(self, logs, start_time=None):
        """Override log to include CE and KL losses."""
        if self.current_ce_loss is not None:
            logs["ce_loss"] = self.current_ce_loss
        if self.current_kl_loss is not None:
            logs["kl_loss"] = self.current_kl_loss
        
        if self.current_ce_loss is not None and self.current_kl_loss is not None:
            logger.info(
                f"Step {self.state.global_step} | "
                f"Loss: {self.current_total_loss:.4f} | "
                f"CE: {self.current_ce_loss:.4f} | "
                f"KL: {self.current_kl_loss:.4f}"
            )
        super().log(logs, start_time)

if __name__ == "__main__":
    DISTILL_DATASET = "artifacts/ds-tf2-en-ro-200k-distill"

    # ------------------------------------------------------------
    # Load teacher & student
    # ------------------------------------------------------------
    device = get_preferred_device()

    teacher = AutoModelForCausalLM.from_pretrained("klusai/tf3-50m-base")
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = student_model  # Already an instance
    student = student.to(device)

    tokenizer = AutoTokenizer.from_pretrained("klusai/tf3-50m-base")

    if os.path.exists(DISTILL_DATASET):
        print("Found precomputed teacher logits → loading...")
        dataset = load_from_disk(DISTILL_DATASET)
    else:
        print("Teacher logits NOT found → computing...")
        dataset = load_from_disk("artifacts/ds-tf2-en-ro-3m-tokenized")
        dataset = keep_only_columns(dataset, ["input_ids", "attention_mask"])
        dataset = dataset.shuffle(seed=42).select(range(200_000))

        # Process in chunks to avoid memory accumulation
        # Use writer_batch_size to write incrementally to disk
        dataset = dataset.map(
            lambda batch: compute_logits(batch, device, teacher),
            batched=True,
            batch_size=8,
            desc="Computing teacher logits",
            writer_batch_size=1000,  # Write every 1000 examples to disk
        )

        dataset.save_to_disk(DISTILL_DATASET)
        
        # Clear memory after saving
        del dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # ------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------
    dataset = load_from_disk(DISTILL_DATASET)
    dataset = keep_only_columns(dataset, ["input_ids", "attention_mask", "teacher_topk_idx", "teacher_topk_val"])
    
    # Only select if we have more than needed, otherwise use all available
    if len(dataset) > 300_000:
        dataset = dataset.shuffle(seed=42).select(range(300_000))
    else:
        dataset = dataset.shuffle(seed=42)


    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # ------------------------------------------------------------
    # Training args
    # ------------------------------------------------------------
    args = TrainingArguments(
        output_dir="./distilled_20m_1B",
        per_device_train_batch_size=12,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2000,
        max_steps=250_000,
        weight_decay=0.0,
        logging_steps=50,

        save_strategy="steps",
        save_steps=2000,
        eval_strategy="steps",
        eval_steps=2000,

        fp16=False,
        bf16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        dataloader_num_workers=4,  # Parallel data loading
        dataloader_pin_memory=True,  # Faster GPU transfer
        dataloader_persistent_workers=True,  # Keep workers alive between epochs
        remove_unused_columns=False,  # Keep all columns for data collator
    )

    # ------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5,
    )

    # Data collator for batching
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    trainer = DistillationTrainer(
        model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[early_stopping],
        temperature=3.0,  # For 1B tokens
        alpha=0.9,        # KL-dominant
    )

    trainer.train()

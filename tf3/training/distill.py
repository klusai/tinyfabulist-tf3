import torch
import torch.nn as nn
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from tf3.training.llama.student import model as student_model
from datasets import load_from_disk
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Load teacher & student
# ------------------------------------------------------------
teacher = AutoModelForCausalLM.from_pretrained("klusai/tf3-50m-base")
student = student_model  # Already an instance, don't call it


tokenizer = AutoTokenizer.from_pretrained("klusai/tf3-50m-base")

# Ensure pad_token and unk_token are set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# CRITICAL: Add unk_token if missing - this fixes the Rust backend issue
if tokenizer.unk_token is None:
    # Add unk_token and ensure it's properly registered
    tokenizer.add_special_tokens({"unk_token": "<unk>"})
    # Ensure unk_token_id is set
    if not hasattr(tokenizer, 'unk_token_id') or tokenizer.unk_token_id is None:
        unk_id = tokenizer.convert_tokens_to_ids("<unk>")
        if unk_id != tokenizer.unk_token_id:
            tokenizer.unk_token_id = unk_id
    # Force update the Rust tokenizer backend if available
    if hasattr(tokenizer, '_tokenizer') and hasattr(tokenizer._tokenizer, 'add_special_tokens'):
        try:
            tokenizer._tokenizer.add_special_tokens(["<unk>"])
        except:
            pass


# ------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------
dataset = load_dataset("klusai/ds-tf2-en-ro-15k", split="train")

def tokenize(example):
    try:
        # Use add_special_tokens=False to avoid issues with special tokens
        out = tokenizer(
            example["translated_fable"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None,  # Return lists, not tensors
            add_special_tokens=True,  # Let tokenizer handle special tokens
            return_token_type_ids=False,  # Don't return token_type_ids
        )

        # CREATE SHIFTED LABELS for causal LM
        # labels[t] = input_ids[t+1] (next token prediction)
        ids = out["input_ids"]
        labels = ids[1:] + [tokenizer.pad_token_id]  # shift left, pad last position

        # Return only the columns we need
        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"],
            "labels": labels
        }
    except Exception as e:
        # Skip problematic samples - return empty but valid structure
        print(f"Error tokenizing sample: {e}")
        # Return a valid empty structure that will be filtered out
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        return {
            "input_ids": [pad_id] * 512,
            "attention_mask": [0] * 512,
            "labels": [pad_id] * 512
        }

tokenized_ds = dataset.map(tokenize, remove_columns=dataset.column_names)
tokenized_ds.set_format("torch")


# ------------------------------------------------------------
# Custom Trainer
# ------------------------------------------------------------
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=1.0, alpha=0.009, **kwargs):
        super().__init__(**kwargs)
        # Store teacher model - will move to device when needed
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        self.temp = temperature
        self.alpha = alpha

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        
        # Store losses for logging
        self.current_ce_loss = None
        self.current_kl_loss = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Get device from student model
        device = next(model.parameters()).device
        
        # Move teacher to same device as student
        if next(self.teacher_model.parameters()).device != device:
            self.teacher_model = self.teacher_model.to(device)

        # Extract input_ids, attention_mask, and labels from inputs
        # inputs should be a dict from the DataLoader
        if not isinstance(inputs, dict):
            raise TypeError(f"Expected inputs to be dict, got {type(inputs)}")
        
        if "input_ids" not in inputs:
            raise KeyError(f"input_ids not found in inputs. Available keys: {list(inputs.keys())}")
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        labels = inputs.get("labels", None)
        
        # Ensure input_ids is a tensor
        if input_ids is None:
            raise ValueError("input_ids is None")
        
        # Build model inputs dict - only include non-None values
        model_inputs = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # Student forward
        student_out = model(**model_inputs)
        s_logits = student_out.logits

        # Teacher forward
        with torch.no_grad():
            t_logits = self.teacher_model(**model_inputs).logits

        # SHIFT: logits[t] predicts token[t+1]
        s_shift = s_logits[:, :-1, :].contiguous()
        t_shift = t_logits[:, :-1, :].contiguous()
        
        # Create labels from input_ids with proper masking
        # Mask prompt/instruction tokens so CE only applies to answer tokens
        labels_full = model_inputs["input_ids"].clone()
        
        # Mask padding tokens
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        labels_full[labels_full == pad_token_id] = -100
        
        # Mask the prompt portion (everything before [/INST])
        # This ensures CE only trains on the answer, not the prompt
        try:
            inst_tokens = tokenizer.encode("[/INST]", add_special_tokens=False)
            if len(inst_tokens) > 0:
                inst_token_id = inst_tokens[-1]  # Use last token of [/INST]
                
                for i, ids in enumerate(model_inputs["input_ids"]):
                    # Find all positions where [/INST] ends
                    inst_positions = (ids == inst_token_id).nonzero(as_tuple=True)[0]
                    if len(inst_positions) > 0:
                        # Mask everything up to and including [/INST]
                        cutoff = inst_positions[-1].item() + 1
                        labels_full[i, :cutoff] = -100
        except:
            # If [/INST] not found, assume we want to train on all non-padded tokens
            # This handles cases where dataset doesn't have prompt structure
            pass
        
        # Shift labels for causal LM: labels[t] = input_ids[t+1]
        labels_shift = labels_full[:, 1:].contiguous()

        # KL loss
        loss_kl = self.kl_loss(
            nn.functional.log_softmax(s_shift / self.temp, dim=-1),
            nn.functional.softmax(t_shift / self.temp, dim=-1),
        ) * (self.temp ** 2)

        # CE loss
        loss_ce = self.ce_loss(
            s_shift.reshape(-1, s_shift.size(-1)),
            labels_shift.reshape(-1)
        )

        # Compute total loss
        loss = self.alpha * loss_kl + (1 - self.alpha) * loss_ce

        # Store losses for logging
        self.current_ce_loss = loss_ce.item()
        self.current_kl_loss = loss_kl.item()
        
        self.current_total_loss = loss.item()

        return (loss, student_out) if return_outputs else loss
    
    def log(self, logs, start_time=None):
        """Override log to include CE and KL losses."""
        # Add CE and KL losses to logs
        if self.current_ce_loss is not None:
            logs["ce_loss"] = self.current_ce_loss
        if self.current_kl_loss is not None:
            logs["kl_loss"] = self.current_kl_loss
        
        # Log directly to console as well
        if self.current_ce_loss is not None and self.current_kl_loss is not None:
            logger.info(
                f"Step {self.state.global_step} | "
                f"Total Loss: {self.current_total_loss:.4f} | "
                f"CE Loss: {self.current_ce_loss:.4f} | "
                f"KL Loss: {self.current_kl_loss:.4f}"
            )
        
        # Call parent log method with correct signature
        super().log(logs, start_time)


# ------------------------------------------------------------
# Training args
# ------------------------------------------------------------
args = TrainingArguments(
    output_dir="./distilled-model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    warmup_steps=50,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    fp16=torch.cuda.is_available(),
)

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------
trainer = DistillationTrainer(
    model=student,
    teacher_model=teacher,
    args=args,
    train_dataset=tokenized_ds,
)

trainer.train()

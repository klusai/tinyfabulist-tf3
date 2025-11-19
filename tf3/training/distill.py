from datasets import load_from_disk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    LlamaConfig
)
from tqdm import tqdm
import os

# ============================================================
# 0. Setup device and precision (Apple Silicon optimized)
# ============================================================
# Detect device: MPS (Apple Silicon) > CUDA > CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_type = "mps"
    print(f"Using device: {device} (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    device_type = "cuda"
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    device_type = "cpu"
    print(f"Using device: {device}")

# Check for flash attention availability
USE_FLASH_ATTENTION = False
if device_type == "cuda":
    try:
        from flash_attn import flash_attn_func
        USE_FLASH_ATTENTION = True
        print("Flash Attention 2 available!")
    except ImportError:
        print("Flash Attention 2 not available, using default attention")
else:
    print("Using SDPA attention (optimal for Apple Silicon)")

# ============================================================
# 1. Load teacher and student
# ============================================================
print("\n" + "="*60)
print("Loading teacher model...")
print("="*60)
teacher_path = "klusai/tf3-50m-base"           # YOUR TEACHER
student_path = "artifacts/transformers-distilled" # SAVE STUDENT HERE

# Load teacher with BF16 precision for memory efficiency (works great on Apple Silicon)
teacher = LlamaForCausalLM.from_pretrained(
    teacher_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2" if USE_FLASH_ATTENTION else "sdpa"
).eval().to("mps")

# Explicitly disable gradients for teacher
for p in teacher.parameters():
    p.requires_grad = False

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(teacher_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
print("Teacher and tokenizer loaded!")

# Your student config (20M)
student_config = LlamaConfig(
    vocab_size=32000,
    hidden_size=384,
    intermediate_size=1024,
    num_hidden_layers=6,
    num_attention_heads=6,
    max_position_embeddings=2048,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    tie_word_embeddings=True,
)

print("\nCreating student model...")
student = LlamaForCausalLM(student_config).to(device)
# Convert to bfloat16 after creation (torch_dtype not available in __init__)
student = student.to(torch.bfloat16)
student.resize_token_embeddings(student_config.vocab_size)
student.tie_weights()
# if device_type == "mps":
#     student.gradient_checkpointing_enable()
#     print("Gradient checkpointing enabled for student model")

print("Student model created!")

# # Compile models for faster inference (PyTorch 2.x)
# if hasattr(torch, 'compile'):
#     print("Compiling models with torch.compile...")
#     teacher = torch.compile(teacher, mode="reduce-overhead")
#     student = torch.compile(student, mode="reduce-overhead")
#     print("Models compiled!")

# ============================================================
# 2. Prepare dataset (optimized)
# ============================================================
print("\n" + "="*60)
print("Preparing dataset...")
print("="*60)

# Load dataset
print("Loading dataset from disk...")
loaded_data = load_from_disk("artifacts/ds-tf2-en-ro-3m-tokenized")
print("Dataset loaded!")

# Handle both DatasetDict and Dataset
if isinstance(loaded_data, dict) and "train" in loaded_data:
    train_dataset = loaded_data["train"]
else:
    train_dataset = loaded_data

# Check if already tokenized
if "input_ids" in train_dataset.column_names:
    print("Dataset already tokenized! Using existing tokenization...")
    tokenized_dataset = train_dataset
else:
    print("Tokenizing dataset with batched processing...")
    def tokenize_batch(batch):
        return tokenizer(
            batch["translated_fable"],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors=None  # Return lists, not tensors
        )
    
    # Use batched map for parallel tokenization
    num_proc = min(8, os.cpu_count())
    tokenized_dataset = train_dataset.map(
        tokenize_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=[col for col in train_dataset.column_names if col != "translated_fable"],
        desc="Tokenizing"
    )
    print("Tokenization complete!")

# Set format to PyTorch tensors (highly optimized)
# Only use columns that actually exist in the dataset
available_columns = tokenized_dataset.column_names
format_columns = [col for col in ["input_ids", "attention_mask", "labels"] if col in available_columns]
print(f"Available columns: {available_columns}")
print(f"Formatting columns: {format_columns}")

# Set format - don't move to device here for MPS (move in training loop)
tokenized_dataset.set_format(
    type="torch",
    columns=format_columns
)

print(f"Dataset size: {len(tokenized_dataset)}")
print("Creating DataLoader...")

# Batch size optimized for Apple Silicon (MPS) - can be adjusted based on available memory
if device_type == "mps":
    batch_size = 8
elif device_type == "cuda":
    batch_size = 256
else:
    batch_size = 64
gradient_accumulation_steps = 4  # Simulate larger batches

loader = DataLoader(
    tokenized_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # MPS works better with num_workers=0
    pin_memory=False  # Not needed for MPS
)
print(f"DataLoader created! Batch size: {batch_size}, Total batches: {len(loader)}")
print(f"Effective batch size (with accumulation): {batch_size * gradient_accumulation_steps}")

# ============================================================
# 3. Optimizer and AMP scaler
# ============================================================
print("\n" + "="*60)
print("Setting up optimizer...")
print("="*60)
optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4)
# MPS doesn't use GradScaler (bfloat16 is native), but we can use autocast
use_amp = device_type in ["cuda", "mps"]  # Both support autocast
print(f"Optimizer ready! AMP: {use_amp}")

# ============================================================
# 4. DISTILLATION LOOP (optimized)
# ============================================================
alpha = 1.0  # KL weight
beta  = 0.1  # CE weight (stabilizer)
early_stop_threshold = 0.8  # Stop when CE loss is below this value

print("\n" + "="*60)
print("Starting training...")
print("="*60)

early_stopped = False  # Flag to track early stopping

for epoch in range(3):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/3")
    print(f"{'='*60}")
    
    student.train()
    total_loss = 0.0
    total_kl = 0.0
    total_ce = 0.0
    
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}")
    
    for batch_idx, batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Gradient accumulation: only zero grad at the start of accumulation
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        # Teacher forward pass (no grad, BF16)
        with torch.no_grad():
            if use_amp:
                # Use autocast for MPS/CUDA (MPS may need device_type="cuda" in some PyTorch versions)
                # Models are already in bfloat16, autocast helps with mixed precision ops
                try:
                    # Try MPS-specific autocast first
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        teacher_logits = teacher(**batch).logits
                except (ValueError, RuntimeError):
                    # Fallback: some PyTorch versions need "cuda" for MPS autocast
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        teacher_logits = teacher(**batch).logits
            else:
                teacher_logits = teacher(**batch).logits
        
        # Student forward pass (with AMP for MPS/CUDA)
        if use_amp:
            try:
                # Try MPS-specific autocast first
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    student_out = student(**batch)
                    student_logits = student_out.logits
            except (ValueError, RuntimeError):
                # Fallback: some PyTorch versions need "cuda" for MPS autocast
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    student_out = student(**batch)
                    student_logits = student_out.logits
            
            # KL divergence (teacher → student)
            # Convert to float32 for stable softmax
            teacher_probs = teacher_logits.float().softmax(dim=-1)
            kl_loss = F.kl_div(
                F.log_softmax(student_logits.float(), dim=-1),
                teacher_probs,
                reduction="batchmean"
            )
            
            # Next-token CE loss
            shift_logits = student_logits[:, :-1].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            loss = (alpha * kl_loss + beta * ce_loss) / gradient_accumulation_steps
            
            # Early stopping check
            if ce_loss.item() < early_stop_threshold:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered! CE loss: {ce_loss.item():.4f} < {early_stop_threshold}")
                print(f"Stopping training and saving model...")
                print(f"{'='*60}")
                early_stopped = True
                break
        else:
            # CPU fallback (no AMP)
            student_out = student(**batch)
            student_logits = student_out.logits
            
            teacher_probs = teacher_logits.float().softmax(dim=-1)
            kl_loss = F.kl_div(
                F.log_softmax(student_logits.float(), dim=-1),
                teacher_probs,
                reduction="batchmean"
            )
            
            shift_logits = student_logits[:, :-1].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            loss = (alpha * kl_loss + beta * ce_loss) / gradient_accumulation_steps
            
            # Early stopping check
            if ce_loss.item() < early_stop_threshold:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered! CE loss: {ce_loss.item():.4f} < {early_stop_threshold}")
                print(f"Stopping training and saving model...")
                print(f"{'='*60}")
                early_stopped = True
                break
        
        # Backward pass (MPS doesn't need GradScaler, bfloat16 is native)
        loss.backward()
        
        # Step optimizer (only after accumulation)
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item() * gradient_accumulation_steps
        total_kl += kl_loss.item()
        total_ce += ce_loss.item()
        
        # Update progress bar
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            avg_loss = total_loss / (batch_idx + 1)
            avg_kl = total_kl / (batch_idx + 1)
            avg_ce = total_ce / (batch_idx + 1)
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'KL': f'{avg_kl:.4f}',
                'CE': f'{avg_ce:.4f}'
            })
    
    # Break out of epoch loop if early stopped
    if early_stopped:
        break

# ============================================================
# 5. Save student
# ============================================================
print(f"\n{'='*60}")
print(f"Saving student model to {student_path}...")
print("="*60)

# Ensure directory exists
os.makedirs(student_path, exist_ok=True)

# Save model and tokenizer
student.save_pretrained(student_path)
tokenizer.save_pretrained(student_path)
print("Done!")

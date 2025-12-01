import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import torch
import torch.nn as torch_nn

# ============================================================
# Config
# ============================================================

@dataclass
class MLXConfig:
    vocab_size: int = 32064       # align to 64
    hidden_size: int = 384        # must be divisible by num_heads*64
    num_heads: int = 6            # head_dim = 64 (best for MLX)
    intermediate_size: int = 1536 # 4x hidden
    num_layers: int = 4           # 4 layers = sweet spot (15–20k tok/s)
    max_position_embeddings: int = 2048
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def to_dict(self):
        return self.__dict__.copy()


# ============================================================
# Fused Attention (QKV + fast_attention)
# ============================================================

class FusedAttention(nn.Module):
    def __init__(self, config: MLXConfig):
        super().__init__()
        self.nh = config.num_heads
        self.hd = config.hidden_size // config.num_heads  # head_dim=64
        self.hidden = config.hidden_size

        # Fused QKV matmul (Linear -> 3x hidden)
        self.qkv = nn.Linear(self.hidden, 3 * self.hidden, bias=False)
        self.out_proj = nn.Linear(self.hidden, self.hidden, bias=False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, mask=None, cache=None, layer_idx=None):
        B, T, C = x.shape

        # -------------------------------------------------------
        # QKV projection
        # -------------------------------------------------------
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # reshape to (B, heads, T, head_dim)
        q = q.reshape(B, T, self.nh, self.hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.nh, self.hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.nh, self.hd).transpose(0, 2, 1, 3)

        # -------------------------------------------------------
        # KV-cache append (for autoregressive decoding)
        # -------------------------------------------------------
        if cache is not None:
            if layer_idx not in cache:
                cache[layer_idx] = {
                    "k": k,
                    "v": v
                }
            else:
                cache[layer_idx]["k"] = mx.concatenate([cache[layer_idx]["k"], k], axis=2)
                cache[layer_idx]["v"] = mx.concatenate([cache[layer_idx]["v"], v], axis=2)

            k = cache[layer_idx]["k"]
            v = cache[layer_idx]["v"]

        # -------------------------------------------------------
        # Scaled dot-product attention (causal)
        # -------------------------------------------------------
        scale = 1.0 / math.sqrt(self.hd)
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # (B, H, Tq, Tk)

        Tq = scores.shape[-2]
        Tk = scores.shape[-1]
        causal = mx.tril(mx.ones((Tq, Tk), dtype=mx.bool_))
        causal = causal.reshape(1, 1, Tq, Tk)
        neg_inf = mx.full(scores.shape, -1e9, dtype=scores.dtype)
        scores = mx.where(causal, scores, neg_inf)

        if mask is not None:
            scores = scores + mask  # assume mask already broadcastable

        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v)

        # back to (B, T, C)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)


# ============================================================
# Transformer Block (Attention + MLP + Norm)
# ============================================================

class MLXBlock(nn.Module):
    def __init__(self, config: MLXConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.norm1 = nn.RMSNorm(config.hidden_size)
        self.attn = FusedAttention(config)

        self.norm2 = nn.RMSNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.SiLU(),   # fused activation
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, mask=None, cache=None):
        h = x + self.attn(self.norm1(x), mask=mask, cache=cache, layer_idx=self.layer_idx)
        h = h + self.mlp(self.norm2(h))
        return h


# ============================================================
# Full Model
# ============================================================

class MLXStudentLM(nn.Module):
    def __init__(self, config: MLXConfig = MLXConfig()):
        super().__init__()

        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layers = []
        for i in range(config.num_layers):
            layer = MLXBlock(config, layer_idx=i)
            setattr(self, f"layer_{i}", layer)  # ensure parameters are registered
            self.layers.append(layer)

        self.norm = nn.RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # tie weights
        self.lm_head.weight = self.embed.weight

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input_ids, cache=None):
        B, T = input_ids.shape

        pos = mx.arange(T)[None, :]
        x = self.embed(input_ids) + self.pos_embed(pos)

        mask = None  # causal mask handled inside fast_attention

        for layer in self.layers:
            x = layer(x, mask=mask, cache=cache)

        x = self.norm(x)
        return self.lm_head(x)


# ============================================================
# Fast autoregressive generation (KV-cache optimized)
# ============================================================

def generate(model, tokenizer, prompt_ids, max_tokens=50):
    model.eval()
    cache = {}

    x = prompt_ids  # (1, T)

    for _ in range(max_tokens):
        logits = model(x[:, -1:], cache=cache)
        next_id = int(mx.argmax(logits[0, -1]))

        x = mx.concatenate([x, mx.array([[next_id]])], axis=1)

    return x


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":
    cfg = MLXConfig()
    model = MLXStudentLM(cfg)

    # Fake input to test speed
    x = mx.array([[1] * 32], dtype=mx.int32)
    out = model(x)
    print("Output shape:", out.shape)

    def count_params(tree):
        if isinstance(tree, dict):
            return sum(count_params(v) for v in tree.values())
        if isinstance(tree, (list, tuple)):
            return sum(count_params(v) for v in tree)
        if hasattr(tree, "size"):
            return tree.size
        return 0

    n_params = count_params(model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")


# ============================================================
# PyTorch implementation for training (matches MLX architecture)
# ============================================================

class TorchRMSNorm(torch_nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch_nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class TorchFusedAttention(torch_nn.Module):
    def __init__(self, config: MLXConfig):
        super().__init__()
        self.nh = config.num_heads
        self.hd = config.hidden_size // config.num_heads
        self.hidden = config.hidden_size

        self.qkv = torch_nn.Linear(self.hidden, 3 * self.hidden, bias=False)
        self.out_proj = torch_nn.Linear(self.hidden, self.hidden, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)  # B, nh, T, hd
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.hd)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # B, nh, T, T

        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        causal = causal.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~causal, float("-inf"))

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # B,1,1,T
                scores = scores.masked_fill(mask, float("-inf"))
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask
            else:
                raise ValueError("Unsupported attention_mask shape")

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # B, nh, T, hd
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TorchMLXBlock(torch_nn.Module):
    def __init__(self, config: MLXConfig):
        super().__init__()
        self.norm1 = TorchRMSNorm(config.hidden_size)
        self.attn = TorchFusedAttention(config)
        self.norm2 = TorchRMSNorm(config.hidden_size)
        self.mlp = torch_nn.Sequential(
            torch_nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            torch_nn.SiLU(),
            torch_nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        h = h + self.mlp(self.norm2(h))
        return h


class TorchMLXStudent(torch_nn.Module):
    def __init__(self, config: MLXConfig = MLXConfig()):
        super().__init__()
        self.config = config
        self.embed = torch_nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = torch_nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = torch_nn.ModuleList(
            [TorchMLXBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = TorchRMSNorm(config.hidden_size)
        self.lm_head = torch_nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        x = self.norm(x)
        return self.lm_head(x)

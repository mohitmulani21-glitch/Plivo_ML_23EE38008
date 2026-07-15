"""GPT variant tuned for a 2,000-step / 2M-param CPU budget.

Changes vs the starter baseline, and why each one:
  * tie_weights=True  -- with a BPE vocab the embedding matrix dominates the
    param budget (vocab x n_embd, twice). Tying halves that and lets us spend
    the savings on depth. Also a known win at small scale: the output head
    gets gradient signal from every input token.
  * RoPE instead of learned pos_emb -- learned positions cost block_size x
    n_embd params and generalise poorly at 2k steps; RoPE is parameter-free
    and injects relative position directly into attention.
  * RMSNorm instead of LayerNorm -- fewer params, no mean subtraction, cheaper
    on CPU, no measurable quality loss at this scale.
  * SwiGLU MLP instead of GELU MLP -- better quality per parameter; we use
    hidden = 8/3 * n_embd (rounded) so the 3-matrix SwiGLU costs about the
    same as the 2-matrix 4x GELU MLP.
  * bias=False everywhere -- biases add params and do nothing measurable.
  * scaled init: residual-projection weights are scaled by 1/sqrt(2*n_layer)
    (GPT-2 trick) so the residual stream does not blow up with depth; the
    baseline's flat std=0.05 for everything is badly wrong for the head.

Config carries every knob as a plain attribute so train.py's config-dump
and evaluate.py's rebuild both keep working unmodified.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 256
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = True
    use_rope = True
    use_rmsnorm = True
    use_swiglu = True
    rope_theta = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        n = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * n * self.weight


def build_rope_cache(T, hd, theta, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(T, device=device).float()
    f = torch.outer(t, inv)                      # (T, hd/2)
    return torch.cos(f).to(dtype), torch.sin(f).to(dtype)


def apply_rope(x, cos, sin):
    # x: (B, nh, T, hd)
    T, hd = x.shape[-2], x.shape[-1]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    c = cos[:T].view(1, 1, T, hd // 2)
    s = sin[:T].view(1, 1, T, hd // 2)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.hd = cfg.n_embd // cfg.n_head
        self.use_rope = cfg.use_rope
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hd).transpose(1, 2)
        if self.use_rope and rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 32 * ((hidden + 31) // 32)      # round for CPU-friendly shapes
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class GELUMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False),
            nn.Dropout(cfg.dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        Norm = RMSNorm if cfg.use_rmsnorm else nn.LayerNorm
        self.ln1 = Norm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = Norm(cfg.n_embd)
        self.mlp = SwiGLU(cfg) if cfg.use_swiglu else GELUMLP(cfg)

    def forward(self, x, rope=None):
        x = x + self.attn(self.ln1(x), rope)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.use_rope = getattr(cfg, "use_rope", False)
        if not self.use_rope:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        Norm = RMSNorm if cfg.use_rmsnorm else nn.LayerNorm
        self.ln_f = Norm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init)
        # GPT-2 style scaled init on residual output projections
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("w2.weight") or \
               n.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layer))
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

        self._rope = None

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _get_rope(self, T, device, dtype):
        if self._rope is None or self._rope[0].shape[0] < T:
            hd = self.cfg.n_embd // self.cfg.n_head
            self._rope = build_rope_cache(max(T, self.cfg.block_size), hd,
                                          getattr(self.cfg, "rope_theta", 10000.0),
                                          device, dtype)
        return self._rope

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        rope = None
        if self.use_rope:
            rope = self._get_rope(T, idx.device, x.dtype)
        else:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, rope) if self.use_rope else blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        # tied head shares storage with tok_emb; count unique tensors only
        seen, tot = set(), 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            tot += p.numel()
        return tot

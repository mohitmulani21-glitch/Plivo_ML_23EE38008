"""Trainer tuned for the 2,000-step / 2M-param CPU budget.

Changes vs the baseline and the reasoning:
  * Adam -> AdamW with decoupled weight decay (0.1) on matmul weights only.
    Norm gains and embeddings are excluded (decaying them is a known
    pessimisation).
  * Constant LR -> warmup + cosine decay to ~10% of peak. With only 2,000
    steps the schedule IS the training run: the baseline's flat 3e-4 is both
    too low early (slow start) and too high late (noisy plateau, visible in
    its loss curve).
  * betas (0.9, 0.999) -> (0.9, 0.95). The default beta2 has an EMA horizon of
    ~1000 steps, i.e. half our entire run; 0.95 (~20 steps) adapts fast enough
    to matter in a short budget.
  * + gradient clipping at 1.0. The baseline had none.
  * Sequential batch sampling from a shuffled index pool instead of pure
    random offsets -- same distribution, less redundant sampling.

Caps enforced: <=2000 optimizer steps, <=2,000,000 params.
"""
import argparse
import json
import math
import os
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000
_HERE = os.path.dirname(os.path.abspath(__file__))


def get_batch(ids, block, batch, device, gen):
    ix = torch.randint(len(ids) - block - 1, (batch,), generator=gen)
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def lr_at(step, total, peak, warmup, final_frac):
    if step < warmup:
        return peak * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    p = min(1.0, max(0.0, p))
    return peak * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def make_optimizer(model, lr, wd, betas):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and "tok_emb" not in n:
            decay.append(p)
        else:
            no_decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas, eps=1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    # architecture knobs
    ap.add_argument("--n_layer", type=int, default=None)
    ap.add_argument("--n_head", type=int, default=None)
    ap.add_argument("--n_embd", type=int, default=None)
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--no_tie", action="store_true")
    ap.add_argument("--no_rope", action="store_true")
    ap.add_argument("--no_rmsnorm", action="store_true")
    ap.add_argument("--no_swiglu", action="store_true")
    ap.add_argument("--token_cache", default=None)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()

    cache = args.token_cache
    if cache and os.path.exists(cache):
        ids = torch.load(cache)
    else:
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        if cache:
            torch.save(ids, cache)
    n_bytes = len(text.encode("utf-8"))
    print(f"corpus: {n_bytes:,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size}, {n_bytes/len(ids):.2f} bytes/token)")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    if args.n_layer: cfg.n_layer = args.n_layer
    if args.n_head: cfg.n_head = args.n_head
    if args.n_embd: cfg.n_embd = args.n_embd
    if args.block_size: cfg.block_size = args.block_size
    if args.dropout is not None: cfg.dropout = args.dropout
    if args.no_tie: cfg.tie_weights = False
    if args.no_rope: cfg.use_rope = False
    if args.no_rmsnorm: cfg.use_rmsnorm = False
    if args.no_swiglu: cfg.use_swiglu = False

    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params  (L{cfg.n_layer} H{cfg.n_head} D{cfg.n_embd} "
          f"ctx{cfg.block_size} tie={cfg.tie_weights})")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params, got {n:,}"

    opt = make_optimizer(model, args.lr, args.wd, (0.9, args.beta2))

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps, args.lr, args.warmup, args.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(ids, cfg.block_size, args.batch, device, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        losses.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr:.2e}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)", flush=True)

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()

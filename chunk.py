"""Resumable chunk runner: trains a slice of the 2000-step schedule and saves
optimizer+RNG state so the next call continues EXACTLY where this left off.
The LR schedule is always computed against TOTAL=2000, so chunking does not
change the run: it is one 2000-step trajectory, just split across processes.
"""
import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, Config
from train import lr_at, make_optimizer, get_batch

TOTAL = 2000
LR = 3e-3
WARMUP = 100
CKPT = "resume.pt"


def build():
    cfg = Config()
    cfg.vocab_size = 4096
    cfg.n_layer = 6
    cfg.n_head = 4
    cfg.n_embd = 128
    cfg.block_size = 128
    return cfg


def main(n_steps):
    ids = torch.load("ids_4096.pt")
    cfg = build()
    torch.manual_seed(1337)
    model = GPT(cfg)
    opt = make_optimizer(model, LR, 0.1, (0.9, 0.95))
    gen = torch.Generator().manual_seed(1337)
    start, losses = 0, []
    if os.path.exists(CKPT):
        st = torch.load(CKPT)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"])
        start, losses = st["step"], st["losses"]
    print(f"params {model.n_params():,} | resuming at {start}", flush=True)
    model.train()
    t0 = time.time()
    end = min(TOTAL, start + n_steps)
    for step in range(start + 1, end + 1):
        lr = lr_at(step, TOTAL, LR, WARMUP, 0.1)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(ids, cfg.block_size, 16, "cpu", gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"step {step} loss {sum(losses[-100:])/100:.4f} lr {lr:.2e} "
                  f"({(time.time()-t0)/(step-start)*1000:.0f}ms)", flush=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "gen": gen.get_state(), "step": end, "losses": losses}, CKPT)
    print(f"saved chunk -> step {end} ({time.time()-t0:.0f}s)")
    if end >= TOTAL:
        torch.save({"model": model.state_dict(),
                    "config": {k: getattr(cfg, k) for k in dir(cfg)
                               if not k.startswith("_")
                               and not callable(getattr(cfg, k))},
                    "steps": TOTAL, "train_loss_curve": losses}, "ckpt.pt")
        print("FINAL ckpt.pt written")


if __name__ == "__main__":
    main(int(sys.argv[1]))

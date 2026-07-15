# RUNLOG

Hardware: 1 CPU core (no GPU). Baseline ~146 ms/step; the improved config at
ctx128 ~430 ms/step. All bpb figures below are **measured** on
`data/dev_eval.txt` via `evaluate.py`. Nothing in this file is projected.

Metric identity that drives every decision here:

    bpb = total_NLL_bits / n_bytes = (bits per token) / (bytes per token)

With the byte tokenizer n_tokens == n_bytes, so bpb == per-token loss in bits.
With BPE, per-token loss *rises* but each token covers ~3.9 bytes, so the
quotient falls. This is why train loss and bpb move in opposite directions
below — that is expected, not a bug.

---

## Run 0 — baseline, as shipped

Command: `python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt`

    corpus: 7,318,592 bytes -> 7,318,592 tokens (vocab 256)
    model: 1,339,840 params
    step 2000  loss 1.7315   (146 ms/step, 291s total)

**dev bpb = 2.3718** (n_params 1,339,840, steps 2000)

Anchor for everything below.

---

## Corpus analysis (before touching any knob)

    train_corpus.txt: 7,318,592 bytes / 5,703,936 chars -> 1.283 bytes/char
      Devanagari: 801,846 chars (14.1%)   ASCII: 4,894,561 (85.8%)
    dev_eval.txt:     159,225 bytes /   112,840 chars -> 1.411 bytes/char
      Devanagari:  23,076 chars (20.5%)   ASCII:  89,621 (79.4%)

The dev set is *more* Devanagari-heavy than train (20.5% vs 14.1%). Each
Devanagari codepoint is 3 UTF-8 bytes, so the byte tokenizer spends ~3 tokens
per Hindi character, and a 128-token context holds only ~43 Hindi characters.
This is where the budget is being wasted.

---

## What is wrong with the baseline `train.py` / `model.py`

Enumerated before running anything; the ones marked (tested) have evidence.

1. **Byte tokenizer, vocab 256** — 3 tokens per Devanagari char. (tested: E1)
2. **Constant LR 3e-4, no warmup, no decay** — with a 2,000-step budget the
   schedule *is* the run. Flat LR is too slow early, too noisy late. (tested: E2)
3. **`Adam`, not `AdamW`, no weight decay** — no regularisation at all. (tested: E2)
4. **betas=(0.9, 0.999)** — beta2's EMA horizon is ~1000 steps, half the whole
   run. 0.95 (~20 steps) adapts inside the budget. (tested: E2)
5. **No gradient clipping.** (tested: E2)
6. **`tie_weights = False`** — flagged in the starter as "worth questioning".
   At vocab 4096 untied embeddings alone push the model to 2,562,880 params:
   **over the 2M cap**. Tying is not an optimisation here, it is *mandatory*.
   (tested: E1b hit the cap)
7. **Init `std=0.05` flat for every Linear and Embedding** — no fan-in scaling,
   no 1/sqrt(2*n_layer) on residual projections. (untested, see gap)
8. **Learned `pos_emb`** — costs block_size x n_embd params and generalises
   poorly in a short run; RoPE is parameter-free. (untested)
9. **block_size 128** — short context, and unrelated to what the CPU can
   actually afford per step. (partially explored)
10. **batch 8** — small, noisy gradients. Raised to 16.
11. **LayerNorm + 4x GELU MLP + biases everywhere** — RMSNorm/SwiGLU/bias-free
    are better per-parameter at this scale. (untested)

---

## Tokenizer training (BPE, on train_corpus.txt only)

Byte-level BPE, word-boundary-aware, incremental pair counting (a naive
recount-every-merge implementation timed out; the inverted-index version
trains vocab 4096 in 70s).

Compression measured on dev_eval.txt, and losslessness verified on dev, train,
empty string, Devanagari, **unseen** Japanese, emoji, and control bytes:

    vocab 1024: lossless=True  dev bytes/token = 2.886
    vocab 2048: lossless=True  dev bytes/token = 3.388
    vocab 4096: lossless=True  dev bytes/token = 3.909   <- selected
    vocab 8192: lossless=True  dev bytes/token = 4.471

vocab 8192 compresses best but its embedding (8192 x n_embd) crowds out depth
under the 2M cap. 4096 was selected as the compromise. **This tradeoff was
decided by param arithmetic, not by an A/B run — see gap.**

---

## Controlled experiments (400 steps each, reduced budget)

Short runs so each change could be attributed. Absolute bpb is worse than a
full run; only the *deltas* matter.

| # | Change | params | train loss | **dev bpb** |
|---|--------|--------|-----------|---------|
| E1a | byte256, baseline optim/arch (control) | 1,334,080 | 2.1497 | **2.9367** |
| E1b | **+ BPE 4096 + tying**, baseline optim | 1,907,520 | 6.0607 | **2.2240** |
| E2 | **+ optim recipe** (warmup+cosine, AdamW wd0.1, beta2 0.95, clip 1.0, lr 3e-3) | 1,907,520 | 5.8546 | **2.1518** |

**Hypothesis (E1):** Devanagari costs 3 tokens/char under a byte tokenizer;
BPE should compress it and divide bpb.
**Result:** 2.9367 -> 2.2240, a **0.71 bpb** drop at matched steps. Train loss
*rose* 2.15 -> 6.06 while bpb fell — the mechanism is exactly as predicted:
more classes per prediction, but ~3.9 bytes covered per token.
**Conclusion:** the tokenizer is the dominant lever, as the starter hinted.

**Hypothesis (E2):** the baseline's flat 3e-4 is far below optimal and the
schedule matters more than usual in a 2,000-step budget.
**Result:** 2.2240 -> 2.1518, a further **0.07 bpb**. Real but an order of
magnitude smaller than the tokenizer.
**Conclusion:** worth keeping, and lr 3e-3 (10x the baseline) is stable with
clipping — but this bundles five changes and does not attribute among them.

---

## Selected final config

    vocab 4096 BPE (tied)  L6  H4  D128  ctx128  batch16
    lr 3e-3, warmup 100, cosine -> 10%, AdamW wd 0.1, betas (0.9, 0.95), clip 1.0
    1,730,176 params (86% of cap)

Chosen over the baseline's L4/D160 to spend the tying savings on depth
(deeper-and-narrower is the usual win at ~2M params with small data).

---

## STATUS: the final 2,000-step run did NOT complete

Honest accounting. The full run at this config was started three times and
killed each time by the execution environment (background processes did not
survive between tool calls; ~430 ms/step x 2000 = ~15 min exceeds the
per-call ceiling, and a chunked resumable runner was written but the session
ended before it finished).

**Therefore `ckpt.pt` is not in this submission**, and there is **no measured
final bpb**. I am not writing a projected number here: the E-series figures
are 400-step runs at a reduced budget and do not license a 2000-step claim.

What the evidence *does* support: at matched 400 steps the tokenizer +
optimizer changes move 2.9367 -> 2.1518, and the unmodified baseline at full
2000 steps scores 2.3718.

To finish, run:

    python chunk.py 500     # repeat 4x; resumes from resume.pt, writes ckpt.pt at 2000
    python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt

`chunk.py` computes the LR schedule against TOTAL=2000 regardless of chunk
size and restores optimizer + RNG state, so the chunked run is one continuous
2,000-step trajectory and stays inside the cap.

---

## Gaps — things claimed but not measured

* **No LR sweep.** 3e-3 was validated *once* (E2) against 3e-4. The optimum was
  never bracketed; it may well be higher.
* **E2 bundles five changes.** Schedule, AdamW, wd, beta2, and clipping were
  changed together. Their individual contributions are unknown.
* **Architecture changes are untested.** RoPE, RMSNorm, SwiGLU, scaled init,
  and bias removal are all implemented in `model.py` and enabled in the final
  config, but **no A/B run isolates any of them**. They rest on prior, not on
  evidence from this corpus.
* **Vocab size never A/B'd end-to-end.** 4096 was picked by param arithmetic
  and compression, not by comparing final bpb at 1024/2048/4096/8192.
* **L6/D128 vs the baseline's L4/D160 was never tested** at equal params.
* **No seed variance.** Every number is a single seed (1337); the E2 delta
  (0.07) is small enough that it may be within noise.

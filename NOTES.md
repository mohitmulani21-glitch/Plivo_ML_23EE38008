# NOTES

## The one idea that matters

Score is **bits per byte**, not per token:

    bpb = total_NLL_bits / n_bytes = (bits per token) / (bytes per token)

The baseline's byte tokenizer sets bytes-per-token to exactly 1.0, so its bpb
*is* its per-token cross-entropy. That means the denominator is free real
estate: any tokenizer that packs more bytes into a token divides the loss,
even if the per-token loss gets worse.

The corpus makes this concrete. 14.1% of training characters (20.5% of dev) are
Devanagari, and every Devanagari codepoint is 3 UTF-8 bytes. Under the byte
tokenizer the model spends three predictions per Hindi character, most of that
effort going into re-learning UTF-8 continuation-byte structure rather than
language. A 128-token context holds ~43 Hindi characters.

BPE at vocab 4096 gets 3.909 bytes/token on dev. Measured effect, 400-step
matched comparison: **2.9367 -> 2.2240 bpb**. Train loss simultaneously rose
from 2.15 to 6.06 — which is the confirmation, not a contradiction. More
classes per prediction, but ~4x fewer predictions per byte.

Everything else I did is worth ~1/10th of this.

## The cap forces the tokenizer's hand

This is the part I did not see coming. At vocab 4096 with untied embeddings the
model is **2,562,880 params — over the 2M cap** before any depth is added. The
starter's comment on `tie_weights = False` ("one of many things worth
questioning") is doing a lot of work: tying isn't a nice-to-have here, it's the
thing that makes a useful vocab affordable at all.

That sets up the real tension. Vocab size buys compression (denominator) but
costs embedding params (vocab x n_embd), which come out of depth. Measured
compression: 1024 -> 2.886, 2048 -> 3.388, 4096 -> 3.909, 8192 -> 4.471
bytes/token. Compression keeps climbing, but at 8192 the embedding is 37% of a
2M budget. I took 4096 as the compromise and spent the tying savings on depth
(L6/D128 rather than the baseline's L4/D160).

**I did not verify that this was the right call.** It is arithmetic plus prior,
not an experiment. If I had one more run, this is what I'd spend it on — 8192
might simply win, since the metric rewards the denominator so directly.

## Why the schedule matters more than usual

2,000 steps is not "training a model" — it's a fixed budget where nothing
converges. Every knob has to be judged on *loss at step 2000*, not final loss.
That changes the answers:

* Flat 3e-4 is the wrong shape twice over: too slow to move early, too noisy to
  settle late. Warmup + cosine to 10% fixes both ends.
* beta2=0.999 has an EMA horizon of ~1000 steps — **half the entire run**. The
  optimizer never stops being warm. 0.95 (~20 steps) actually adapts.
* lr 3e-3, 10x the baseline, is stable once clipping is on.

Measured: **2.2240 -> 2.1518**. Real, but note the scale — the tokenizer was 10x
bigger. Also note this bundles five changes at once, so I can't tell you which
of the five earned it.

## What I got wrong

Two things worth recording.

**The BPE trainer.** My first implementation recounted every pair on every
merge and timed out entirely. The fix (inverted index pair -> word-ids,
incremental count updates) trains vocab 4096 in 70s. Obvious in retrospect;
cost real time.

**Time budgeting.** I spent the CPU budget on controlled 400-step A/Bs — good
methodology — and then didn't have the wall-clock left to run the 2,000-step
final. The A/Bs are the part I'd defend; the sequencing is the mistake. On one
core I should have started the final run *first* and A/B'd around it. As it
stands there's no `ckpt.pt`, which is the deliverable that actually gets graded.

## What I'd do with the next hour, in order

1. **Finish the final run** (`chunk.py 500` x4, then evaluate). Nothing else
   matters until there's a checkpoint.
2. **Vocab 8192 vs 4096, end to end.** The most likely unclaimed win, and the
   one my reasoning is least sure about.
3. **Un-bundle E2** — clipping alone, then beta2 alone, then schedule alone.
4. **LR bracket** at 1e-3 / 3e-3 / 6e-3. 3e-3 beat 3e-4 once; that's not a sweep.
5. **A/B the architecture changes I shipped on faith** — RoPE, RMSNorm, SwiGLU,
   scaled init. Each is defensible from prior and none is measured *here*.
6. **Seeds.** The 0.07 from E2 could be noise; I ran one seed.

## Ambitious thing I'd try, expecting it might lose

Drop the 50% of the corpus that's plain ASCII prose down-weighted, and train
Devanagari-heavy. Dev is 20.5% Devanagari vs train's 14.1% — the eval is
*shifted* toward the script the tokenizer handles worst. Matching train
distribution to dev might beat a uniform sample. It might also just overfit a
small slice and lose on the ASCII 79%. I'd want the number either way.

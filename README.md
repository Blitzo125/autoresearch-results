# Parameter Golf — OpenAI Competition Entry

## Overview

This repository contains my submission for **OpenAI's Parameter Golf** competition, where the goal is to train the best small language model (measured by **validation bits-per-byte**) under strict constraints: the final artifact — code plus int8-quantized, compressed weights — must fit under **16 MB**, and training is capped at **10 minutes** on the competition GPU.

I used Karpathy's [**autoresearch**](https://github.com/karpathy/autoresearch) repo as the starting point — a framework that gives an AI agent a real LLM training setup and lets it experiment autonomously. The agent modifies the training code, runs a 5-minute training run, checks if val\_bpb improved, keeps or discards the change, and repeats. I ran this loop for an **8-hour session** that produced 20 incremental experiments, each building on the best configuration found so far. Every experiment was tracked by git commit with its validation BPB, artifact size, and memory usage recorded in `results.tsv`.

The final best run achieved a **val\_bpb of 1.2675** — a **7.8% improvement** over the baseline's 1.3745.

## How autoresearch Works

[autoresearch](https://github.com/karpathy/autoresearch) is Karpathy's framework for autonomous AI-driven ML research. The core idea:

1. The repo has a single training script (`train.py`) that the AI agent is allowed to edit
2. A `program.md` file provides instructions and context to the agent
3. The agent proposes a change, trains for a fixed 5-minute budget, evaluates val\_bpb, and decides to keep or discard
4. This loops overnight — you wake up to a log of experiments and (hopefully) a better model

The metric is **val\_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared. Training always runs for a fixed wall-clock budget regardless of model size or batch size, making experiments directly comparable.

## Experiment Results

All 22 experiments are logged in [`results.tsv`](results.tsv). Here are the key milestones:

| Experiment | val\_bpb | Artifact (MB) | Status | Description |
|---|---|---|---|---|
| Baseline | 1.3745 | 11.96 | keep | 9L 512d GQA-4kv tied-embed relu² |
| exp2 | 1.3413 | 11.99 | keep | GPTQ-lite calibrated clipping + sliding window eval |
| exp4 | 1.3399 | 11.90 | keep | XSA on last 4 layers |
| exp5 | 1.3315 | 14.10 | keep | Fix warmdown\_iters 1200→150 |
| **exp6** | **1.2734** | **15.40** | **keep** | **SWA over warmdown — massive gain** |
| exp7 | 1.2726 | 15.31 | keep | warmdown\_iters 150→300 for more SWA steps |
| exp8 | 1.2716 | 15.33 | keep | Partial RoPE 16/64 dims |
| exp10 | 1.2702 | 15.94 | keep | BigramHashEmbedding 4096×128 |
| exp12 | 1.2680 | 16.31 | keep | matrix\_lr 0.04→0.05 |
| **exp19** | **1.2675** | **16.35** | **keep** | **XSA on all 9 layers — lowest loss** |

Several other experiments were tried and discarded (int6 quantization, deeper/narrower architectures, SiLU activation, EMA, LZMA compression, longer sequences, etc.) because they didn't improve on the best configuration.

## Best Model: `train_improved.py`

The winning configuration from **exp19** (commit `a29a26c`) is captured in [`train_improved.py`](train_improved.py). This is the single-file training script (1,328 lines, 56 KB) that produces the competition artifact.

### Architecture

- **9 transformer layers**, 512 model dim, 8 attention heads with 4 KV heads (GQA)
- **Tied embeddings** with vocab size 1024 (BPE tokenizer)
- **ReLU² activation** — sparse activations compress significantly better under int8 quantization than SiLU/GELU
- **U-Net skip connections** — the first half of the layers (encoder) store skip activations, which are fed back into the second half (decoder) with learned skip weights
- **Logit softcapping** at 30.0 to stabilize training

### Key Techniques

**XSA (Cross-Self Attention) on all 9 layers** — The winning experiment's novel contribution. XSA subtracts the self-value projection from each attention head's output, forcing heads to attend to *other* tokens rather than copying the current token's value. Applied GQA-aware via efficient normalized projection subtraction. Extending this from the last 4 layers to all 9 delivered the final BPB improvement.

**Stochastic Weight Averaging (SWA)** — The single biggest gain in the entire experiment series (1.33 → 1.27 BPB). During the warmdown phase when the learning rate is being decayed, the model accumulates a running average of weights at each step. After training, the SWA-averaged weights replace the final weights. This acts as a free ensemble that smooths the loss landscape.

**BigramHashEmbedding** — A 4096×128 hash embedding table that encodes bigram (previous-token, current-token) pairs via a hash function. The bigram features are projected to model dimension and added to the token embeddings with a learned scale (initialized at 0.05). This gives the model cheap access to local context without extra attention cost.

**Partial RoPE** — Only 16 out of 64 head dimensions receive rotary position embeddings. The remaining 48 dimensions act as content-based attention, allowing the model to learn position-independent patterns alongside position-aware ones.

**GPTQ-lite Calibrated Clipping** — A lightweight version of GPTQ that finds optimal per-row clipping thresholds for int8 quantization by evaluating multiple percentiles and selecting the one that minimizes mean squared error.

**Sliding Window Evaluation** — Validation uses overlapping windows with stride 256 (vs. sequence length 1024), so every scored token has up to 768 tokens of prior context. This gives a more accurate BPB measurement than non-overlapping chunks.

**Muon Optimizer** — Uses Newton-Schulz orthogonalization to normalize matrix-shaped gradients before applying them, with momentum warmup from 0.85 to 0.95 over 500 steps. Separate learning rates for embeddings (0.6), matrix parameters (0.05), and scalar parameters (0.04).

### Quantization & Compression

The final artifact pipeline:
1. **Per-row int8 quantization** for all 2D weight tensors, per-tensor int8 for others
2. **Int6 rounding on middle layers** (layers 2–6) — values are rounded to multiples of 4 within the int8 range, using only 64 of 256 possible values, which compresses dramatically better
3. **Zstandard compression** at level 22 — significantly better than zlib for neural network weight distributions
4. Small tensors (norms, scales, etc.) are kept in fp16 as passthrough

Final artifact: **~16.35 MB** (code + compressed weights)

### Hyperparameters

| Parameter | Value |
|---|---|
| Layers | 9 |
| Model dim | 512 |
| Attention heads | 8 (4 KV heads, GQA) |
| MLP multiplier | 2× |
| Sequence length | 1024 |
| Vocab size | 1024 |
| Batch size | 524,288 tokens |
| Warmdown iters | 300 |
| Matrix LR (Muon) | 0.05 |
| Embed LR | 0.6 |
| Grad clip norm | 0.3 |
| Training cap | 10 min wallclock |

## Files

| File | Description |
|---|---|
| `train_improved.py` | Final training script (best config, exp19) — the competition submission |
| `results.tsv` | Full experiment log with val\_bpb, artifact size, memory, and status for all 22 runs |
| `git_log.txt` | Git commit history from the autoresearch repo and experiment branches |
| `improvements.patch` | Diff of all changes from baseline `train.py` to the final `train_improved.py` |
| `program_size_fix_mac.md` | Notes on reducing artifact size to fit under the 16 MB competition limit |

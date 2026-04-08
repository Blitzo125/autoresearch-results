# Parameter Golf — My Results

This is my fork of [OpenAI's Parameter Golf](https://github.com/openai/parameter-golf) challenge, where the goal is to train the best language model that fits in a **16MB artifact** and trains in under **10 minutes on 8×H100s**. Models are evaluated by compression on the FineWeb validation set, measured in tokenizer-agnostic **bits per byte (val_bpb)** — lower is better.

I used [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) to autonomously experiment with the training setup. Autoresearch lets an AI agent iteratively modify training code, run short training loops, evaluate results, and repeat — essentially automating the hyperparameter/architecture search process overnight.

## Results

### Leaderboard Track (10-min / 8×H100)

| Run | val_bpb | val_loss | Artifact Size | Steps Completed | Train Time |
|-----|--------:|---------:|--------------:|----------------:|-----------:|
| Naive Baseline | **1.2244** | 2.0727 | 15.86 MB | 13,780 / 20,000 | 600s (~10 min) |

### Non-Record Track (Unlimited Compute)

| Run | val_bpb | val_loss | Pre-Quant val_bpb | Artifact Size | Steps Completed | Train Time |
|-----|--------:|---------:|------------------:|--------------:|----------------:|-----------:|
| 4-Hour Baseline | **1.2074** | 2.0386 | 1.1749 | 15.81 MB | 329,430 / 500,000 | 14,400s (~4 hrs) |

Both runs use the same architecture: 9-layer transformer, 512 dim, 1024 vocab, 8 attention heads with 4 KV heads (GQA), 2× MLP expansion, and tied input/output embeddings. The 4-hour run demonstrates how much further the same architecture can be pushed with more compute, closing the gap by ~0.017 BPB.

## Repository Structure

```
parameter-golf/
├── train_gpt.py              # Main CUDA training script (PyTorch + torchrun, 8×H100)
├── train_gpt_mlx.py          # Apple Silicon training script (MLX, for local iteration)
├── requirements.txt          # Python dependencies (torch, numpy, sentencepiece, etc.)
├── LICENSE                   # MIT License (OpenAI)
├── THIRD_PARTY_NOTICES.md    # Attribution for modded-nanogpt (Keller Jordan)
│
├── data/
│   ├── cached_challenge_fineweb.py       # Downloads pre-tokenized FineWeb from HuggingFace
│   ├── download_hf_docs_and_tokenize.py  # Rebuilds tokenizers from published docs
│   ├── tokenizer_specs.json              # Tokenizer config (sp_bpe_1024, vocab size 1024)
│   ├── README.md                         # Data workflow documentation
│   ├── datasets/                         # [gitignored] Downloaded training/val shards (.bin)
│   └── tokenizers/                       # [gitignored] SentencePiece model + vocab files
│
├── records/
│   ├── track_10min_16mb/
│   │   └── 2026-03-17_NaiveBaseline/
│   │       ├── train_gpt.py              # Exact code snapshot used for this run
│   │       ├── train.log                 # Full training log (449 lines)
│   │       ├── submission.json           # Leaderboard metadata (author, scores, sizes)
│   │       └── README.md                 # Run config, command, and key metrics
│   │
│   └── track_non_record_16mb/
│       └── 2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/
│           ├── train_gpt.py              # Exact code snapshot used for this run
│           ├── train.log                 # Full training log (2902 lines)
│           ├── submission.json           # Submission metadata
│           └── README.md                 # Run config, command, and key metrics
│
└── logs/                                 # [gitignored] Local MLX smoke test outputs
```

## Key Files Explained

### Training Scripts

- **`train_gpt.py`** — The primary training script for CUDA GPUs. Implements a full GPT training loop with distributed training via `torchrun`, int8 quantization + zlib compression for submission sizing, and bits-per-byte evaluation. Configurable entirely through environment variables (model dimensions, batch size, learning rate, wallclock cap, etc.).

- **`train_gpt_mlx.py`** — An equivalent training script for Apple Silicon using MLX. Useful for fast local iteration on a Mac before scaling up to cloud GPUs. Supports the same hyperparameter interface.

### Data Pipeline

- **`data/cached_challenge_fineweb.py`** — Downloads pre-tokenized FineWeb shards from HuggingFace. The default variant (`sp1024`) uses a 1024-token SentencePiece BPE vocabulary with ~10B training tokens across 80 shards, plus a fixed 50k-document validation split.

- **`data/download_hf_docs_and_tokenize.py`** — For rebuilding tokenizers or re-exporting shards from the same frozen document selection. Useful if experimenting with custom tokenizers.

### Record Submissions

Each record folder is a self-contained, reproducible snapshot of a competition run containing the exact code, full training log, and submission metadata. The two included records are:

1. **Naive Baseline** (10-min track) — The default configuration run on 8×H100s for 10 minutes, hitting 13,780 steps and scoring 1.2244 val_bpb after int8+zlib quantization.

2. **4-Hour Baseline** (non-record track) — Same architecture pushed to 329,430 steps over 4 hours, reaching 1.1749 val_bpb pre-quantization and 1.2074 post-quantization, demonstrating the scaling potential within the 16MB artifact constraint.

## Quick Start

### Local (Apple Silicon Mac)

```bash
git clone https://github.com/Blitzo125/parameter-golf.git
cd parameter-golf
python3 -m venv .venv
source .venv/bin/activate
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm

# Download data (small subset for testing)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# Run a quick smoke test
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

### Cloud (CUDA GPU)

```bash
git clone https://github.com/Blitzo125/parameter-golf.git
cd parameter-golf
pip install -r requirements.txt

# Download full dataset
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train on 8×H100
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) — The original challenge and codebase
- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — AI agent framework used to autonomously iterate on training configurations
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) by Keller Jordan — Foundation code that this repo adapts (see `THIRD_PARTY_NOTICES.md`)
- [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) — The spiritual predecessor challenge optimizing training time to reach a fixed loss target

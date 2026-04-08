# Parameter Golf — Size Fix (Mac, No GPU)

## Your Job

Read `~/Desktop/train_improved.py` and make targeted edits to reduce the compressed artifact size from 16.35MB to under 16.0MB (16,000,000 bytes), without changing the model architecture or training logic.

You are NOT running any experiments. You are ONLY reading and editing the file. The user will verify on a GPU server separately.

---

## Background

This file (`train_improved.py`) is a competition entry for OpenAI's Parameter Golf challenge. After training, the model weights get quantized to int8 and compressed with zlib. The resulting artifact (code + compressed weights) must be under 16,000,000 bytes. Currently it's 16,350,000 bytes — 350KB over.

The model architecture is 9 layers, 512 dim, GQA with 4 KV heads, tied embeddings, vocab size 1024. It has XSA on all 9 layers, GPTQ-lite quantization, SWA over warmdown, partial RoPE, and sliding window eval.

---

## Step 1 — Read the file thoroughly

Read `~/Desktop/train_improved.py` completely. Specifically find and understand:

1. **The quantization code** — how are weights converted to int8? Look for `quantize`, `int8`, `to_int8`, or similar functions
2. **The compression code** — how is zlib used? Look for `import zlib`, `zlib.compress`
3. **The serialization code** — where is the artifact assembled and its size computed? Look for `Total submission size`
4. **The model size** — how many parameters total? Look for `model_params` in the output or count from the architecture

---

## Step 2 — Make the fixes

Try these in order. Apply ALL that are applicable — they stack.

### Fix 1 — Replace zlib with zstd (BEST FIRST FIX)

Find where `zlib.compress` is called for the final artifact compression. Replace it with zstd compression.

**Before:**
```python
import zlib
compressed = zlib.compress(data, level=9)
```

**After:**
```python
import zstandard as zstd
cctx = zstd.ZstdCompressor(level=22)
compressed = cctx.compress(data)
```

Also update the decompression if it exists:
```python
# Before:
data = zlib.decompress(compressed)
# After:
dctx = zstd.ZstdDecompressor()
data = dctx.decompress(compressed)
```

Also update any size reporting lines that say "int8+zlib" to say "int8+zstd" so it's accurate.

**Important:** Check if `zstandard` is already imported. If the code already uses zstd, skip this fix.

Expected size reduction: 0.5–1.5MB (zstd level 22 is much better than zlib level 9 on neural network weights)

### Fix 2 — int6 quantization on middle layers

Find the int8 quantization function. After weights are quantized to int8 (values in range -127 to 127), add a step that rounds middle layers to int6 precision.

int6 rounding means: `w = (w // 4) * 4` — this keeps the int8 storage format but only uses 64 of the 256 possible values, which compresses much better with zlib/zstd.

Apply this ONLY to middle layers. For a 9-layer model, middle layers = layers 2, 3, 4, 5, 6 (0-indexed). Keep layers 0, 1, 7, 8 at full int8.

Find where each layer's weights are serialized. Add a check:
```python
# Apply int6 rounding to middle layers for better compression
if layer_idx in range(2, 7):  # layers 2-6 out of 0-8
    w_int8 = (w_int8.astype(np.int32) // 4 * 4).clip(-128, 127).astype(np.int8)
```

Expected size reduction: 0.3–0.8MB
Expected val_bpb impact: +0.001 to +0.003 (very small degradation, acceptable)

### Fix 3 — Verify GPTQ-lite covers all layers

Find the GPTQ-lite calibrated clipping code (looks for multiple clip percentiles per row, picks minimum MSE). Check if it's applied to ALL weight matrices or only some. If it's only applied to attention weights but not FFN weights (or vice versa), extend it to all.

This doesn't reduce size directly but makes the weights more quantization-friendly, which compresses better.

---

## Step 3 — Save the result

Save the modified file as `~/Desktop/train_fixed.py`.

Do NOT modify:
- The model architecture (NUM_LAYERS, MODEL_DIM, NUM_HEADS, NUM_KV_HEADS)
- The training hyperparameters (learning rate, batch size, warmup, etc.)
- The data loading or evaluation logic
- The val_bpb computation
- The XSA implementation
- The SWA/EMA implementation
- The sliding window eval

---

## Step 4 — Summary report

After making the changes, write a brief summary:
1. What changes you made
2. Which lines changed (line numbers)
3. Estimated size reduction from each change
4. Any concerns or things to watch for during verification

The user will then upload `train_fixed.py` to a GPU server and run one 5-minute training run to verify the artifact is under 16,000,000 bytes.

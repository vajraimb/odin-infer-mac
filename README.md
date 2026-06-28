# odin-infer-mac

Pure Odin inference for **Qwen3** GGUF models on **Apple Silicon Mac** (CPU or
Metal GPU). GGUF parser, quantized GEMV kernels, embedded BPE tokenizer, and a
single-command-buffer GPU forward pass — no Python, no llama.cpp dependency.

**Platform:** macOS on Apple Silicon (M1/M2/M3/M4). Metal GPU backend requires
`-g 1`. CPU fallback works on the same builds.

## Quick start (no compiler needed)

### 1. Download the executable

From the repo (or clone and use the committed binary):

```sh
curl -L -o odin-infer-mac \
  https://github.com/vajraimb/odin-infer-mac/raw/main/odin-infer-mac
chmod +x odin-infer-mac
xattr -cr odin-infer-mac   # only if macOS blocks unsigned app
```

### 2. Download a Qwen3 GGUF model

Pick one (see [Model downloads](#model-downloads) for more sizes):

```sh
# 0.6B Q4_K_M — fast, good for trying (~380 MB)
curl -L -o Qwen3-0.6B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf

# 8B Q4_K_M — daily driver on 16 GB Mac (~5 GB)
curl -L -o Qwen3-8B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
```

### 3. Run

```sh
# 0.6B + GPU (recommended first test)
./odin-infer-mac Qwen3-0.6B-Q4_K_M.gguf -g 1 -t 0 -r 1 -f 1

# 8B + GPU (use -c 1024 on 16 GB RAM)
./odin-infer-mac Qwen3-8B-Q4_K_M.gguf -g 1 -c 1024 -t 0.6 -r 1 -f 1
```

Interactive chat: press Enter to skip system prompt → type your question → empty
line to exit. Set `QDBG=1` for per-token debug on stderr.

---

## Model downloads

Any **Qwen3** GGUF from Hugging Face works. Direct links:

| Model | Quant | Size | URL |
|-------|-------|------|-----|
| Qwen3-0.6B | Q4_K_M | ~380 MB | [unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/tree/main) |
| Qwen3-0.6B | Q8_0 | ~610 MB | [Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/tree/main) |
| Qwen3-1.7B | Q4_K_M | ~1.1 GB | [unsloth/Qwen3-1.7B-GGUF](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/tree/main) |
| Qwen3-4B | Q4_K_M | ~2.5 GB | [unsloth/Qwen3-4B-GGUF](https://huggingface.co/unsloth/Qwen3-4B-GGUF/tree/main) |
| Qwen3-8B | Q4_K_M | ~5.0 GB | [unsloth/Qwen3-8B-GGUF](https://huggingface.co/unsloth/Qwen3-8B-GGUF/tree/main) |
| Qwen3-8B | Q4_K_M | ~5.0 GB | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF/tree/main) (official) |

One-liner examples:

```sh
curl -L -o Qwen3-0.6B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf

curl -L -o Qwen3-4B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf

curl -L -o Qwen3-8B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
```

If you already use **Ollama** with `qwen3:8b`, the blob is GGUF-compatible:

```sh
ollama show qwen3:8b --modelfile   # find blob path under ~/.ollama/models/blobs/
./odin-infer-mac ~/.ollama/models/blobs/sha256-....gguf -g 1 -c 1024 -r 1 -f 1
```

---

## Command reference

```text
odin-infer-mac <model.gguf> [options]
odin-infer-mac <model.gguf> --dump          # inspect tensors / quant types (no chat)
```

### Options

| Flag | Argument | Default | Description |
|------|----------|---------|-------------|
| `-t` | float | `0.6` | Temperature; `0` = greedy (deterministic) |
| `-p` | float | `0.95` | Top-p nucleus sampling in `[0, 1]` |
| `-s` | int | time | Random seed |
| `-m` | `0\|1` | `0` | Multi-turn conversation (keep history in one session) |
| `-k` | `0\|1` | `0` | Reasoning mode (emit `` blocks; off = skip thinking) |
| `-r` | `0\|1` | `0` | Print generation speed (tok/s) after each reply |
| `-f` | `0\|1` | `0` | Print time-to-first-token (TTFT) in ms |
| `-j` | int | CPU cores | CPU matmul thread count (used when `-g 0` or for embedding) |
| `-c` | int | `4096` | Max context length; caps KV cache memory |
| `-g` | `0\|1` | `0` | **Metal GPU** on Apple Silicon (`1` = enable) |

### Environment

| Variable | Description |
|----------|-------------|
| `QDBG=1` | Log token ids / logits hints to stderr |

### Example commands

```sh
# Greedy, GPU, show metrics (good for benchmarking)
./odin-infer-mac Qwen3-0.6B-Q4_K_M.gguf -g 1 -t 0 -r 1 -f 1

# 8B on 16 GB Mac — limit context to save RAM
./odin-infer-mac Qwen3-8B-Q4_K_M.gguf -g 1 -c 1024 -t 0.6 -r 1 -f 1

# CPU-only (no Metal)
./odin-infer-mac Qwen3-0.6B-Q4_K_M.gguf -g 0 -j 8 -t 0.6

# Multi-turn chat with reasoning (like Ollama thinking mode)
./odin-infer-mac Qwen3-8B-Q4_K_M.gguf -g 1 -c 2048 -m 1 -k 1

# Reproducible sampling
./odin-infer-mac Qwen3-0.6B-Q4_K_M.gguf -g 1 -t 0.8 -p 0.9 -s 42

# Inspect GGUF file
./odin-infer-mac Qwen3-8B-Q4_K_M.gguf --dump
```

### Context length (`-c`) guide

| Model | RAM | Suggested `-c` |
|-------|-----|----------------|
| 0.6B | any | default `4096` |
| 4B | 16 GB | `2048`–`4096` |
| 8B | 16 GB | **`1024`**–`2048` |
| 8B | 32 GB+ | `4096` |

When the context fills up, generation stops with `Context limit reached` (no crash).

---

## Performance

Qwen3-8B Q4_K_M, M3 Air 16 GB, `-g 1 -c 1024`:

| Backend | tok/s |
|---------|-------|
| CPU (8 threads) | ~0.1 |
| Metal (current) | **~10** |
| Ollama / llama.cpp (reference) | ~15.6 |

0.6B on Metal is much faster (interactive, tens of tok/s).

---

## Build from source

Requires [Odin](https://odin-lang.org/) (2026-06+):

```sh
git clone https://github.com/vajraimb/odin-infer-mac.git
cd odin-infer-mac
./build.sh
# -> ./odin-infer-mac  (~3.5 MB, tokenizer embedded via #load)
```

Manual build:

```sh
odin build . -out:odin-infer-mac -o:speed -no-bounds-check -disable-assert -microarch:native
```

The tokenizer is embedded at compile time. Placing `vocab.txt` / `merges.txt` in
the working directory overrides the embedded copy (for development).

---

## Features

- **GGUF v2/v3** — metadata + tensor table; any Qwen3 quant without conversion
- **Quant types** — F32, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q6_K
- **Metal GPU** — zero-copy weight mmap, f16 KV cache, optimized Q6_K GEMV
- **CPU SIMD matmul** — persistent thread pool + `#simd[8]f32` dot products
- **Bounded KV cache** — `-c` caps memory vs native 40960 context

---

## Test

```sh
odin test .
```

---

## Layout

| File | Purpose |
|------|---------|
| `gguf.odin` | GGUF binary parser |
| `quant.odin` | GGML dequant + SIMD dots |
| `model.odin` | Config, weights, run state |
| `forward.odin` | CPU forward pass |
| `matmul.odin` | CPU multithreaded GEMV |
| `metal.odin` | Metal MSL kernels + GPU forward |
| `tokenizer.odin` | BPE tokenizer (embedded vocab) |
| `sampler.odin` | Temperature / top-p sampling |
| `main.odin` | CLI + chat loop |
| `build.sh` | Release build script |
| `odin-infer-mac` | Pre-built Apple Silicon binary (committed for download) |

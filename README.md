# odin-infeer

Pure Odin inference for **Qwen3** GGUF models, on CPU or **Apple Silicon GPU
(Metal)**. Originally a port of [qwen3.c](https://github.com/gigit0000/qwen3.c),
now with a real GGUF binary parser, on-the-fly dequantization, a hash-map BPE
tokenizer, a multithreaded SIMD CPU matmul, and a Metal compute backend that
runs the whole forward pass on the GPU.

## Features

- **Real GGUF parsing** — reads metadata and the tensor table directly from the
  binary, so it works with any Qwen3 GGUF (0.6B / 1.7B / 4B / 8B …) without a
  per-model conversion step.
- **Quantization** — runs FP32, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K and
  Q6_K weights (covers `Q8_0`, `Q4_K_M`, `Q4_0`, etc.). Weights are dequantized
  on the fly (per row on CPU, inside the GEMV kernel on GPU), so a quantized
  model uses far less RAM and bandwidth — this is what makes larger models fit.
- **Metal GPU backend (`-g 1`)** — on Apple Silicon, the entire per-token
  forward pass (quantized GEMV, RMSNorm, RoPE, attention, SwiGLU, residuals)
  runs on the GPU in a single command buffer. Weights are mapped **zero-copy**
  over the mmap'd file (no host→device duplication), run-state and KV cache live
  in shared GPU memory, and dequantization happens inside the GEMV kernels.
- **SIMD CPU matmul** — `#simd[8]f32` dot products with FMA, `#no_bounds_check`
  hot loops, and a persistent worker pool where the calling thread also works.
- **Bounded KV cache** — context length is capped (default 4096, `-c` to change)
  so the KV cache stays small instead of allocating for the full 40960-token
  native context.

## Performance

Qwen3-8B Q4_K_M on an M3 Air (16 GB), same machine:

| Backend | tok/s |
|---|---|
| CPU (8 threads) | ~0.1 |
| Metal, per-matmul dispatch (early design) | ~2.4 |
| **Metal, single command buffer (`-g 1`)** | **~8** |
| Ollama / llama.cpp (Metal, reference) | ~15.6 |

The Metal path turns an impractical 8B (minutes per token on CPU) into an
interactive one, at roughly half of llama.cpp's heavily-optimized throughput.

## Requirements

- [Odin compiler](https://odin-lang.org/) (dev-2026-06 or later) — only needed to build from source
- A Qwen3 GGUF model file

The BPE tokenizer (`vocab.txt` + `merges.txt`, ~3 MB) is also **embedded at compile
time** via Odin `#load`, so a lone binary works for distribution. At runtime,
files in the current directory take precedence (handy when hacking the tokenizer
without rebuilding).

## Download a model

Any Qwen3 GGUF works. Examples:

```sh
# Q8_0 (official, ~600 MB)
curl -L -o Qwen3-0.6B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf

# Q4_K_M (unsloth, ~380 MB)
curl -L -o Qwen3-0.6B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf
```

## Build

Release binary (recommended for distribution):

```sh
./build.sh
# -> ./odin-infer-mac  (~3.5 MB, tokenizer embedded)
```

Or manually:

```sh
odin build . -out:odin-infer-mac -o:speed -no-bounds-check -disable-assert -microarch:native
```

## Run

```sh
./odin-infer-mac Qwen3-0.6B-Q8_0.gguf -t 0 -j 4 -r 1 -f 1   # greedy, 4 threads, show tok/s + TTFT
./odin-infer-mac Qwen3-0.6B-Q4_K_M.gguf                      # sampling defaults
./odin-infer-mac Qwen3-8B-Q4_K_M.gguf -g 1 -r 1 -f 1        # Apple Silicon GPU (Metal)
./odin-infer-mac Qwen3-0.6B-Q8_0.gguf -m 1 -k 1             # multi-turn + reasoning
./odin-infer-mac <model>.gguf --dump                         # inspect tensors / quant types
```

## Distribute

Copy two files to another Mac (Apple Silicon):

1. `odin-infer-mac` — the built executable
2. `Qwen3-*.gguf` — any Qwen3 model

No Odin install, no `vocab.txt` / `merges.txt`, no other dependencies. First
run on an unsigned binary may require: `xattr -cr odin-infer-mac`.

## Options

| Flag | Description |
|------|-------------|
| `-t <float>` | Temperature (0 = greedy, default 0.6) |
| `-p <float>` | Top-p nucleus sampling (default 0.95) |
| `-s <int>`   | RNG seed (default: time) |
| `-m <0\|1>`  | Multi-turn conversation |
| `-k <0\|1>`  | Reasoning mode (thinking tokens) |
| `-r <0\|1>`  | Print tokens/sec |
| `-f <0\|1>`  | Print time-to-first-token |
| `-j <int>`   | Matmul thread count (default: CPU cores) |
| `-c <int>`   | Max context length (default 4096; caps KV cache) |
| `-g <0\|1>`  | Use the Metal GPU backend (Apple Silicon) |

Set `QDBG=1` to print per-token debug info to stderr.

## Test

```sh
odin test .
```

## Layout

| File | Purpose |
|------|---------|
| `gguf.odin` | GGUF v2/v3 binary parser (metadata + tensor table) |
| `quant.odin` | GGML block formats, dequantization, SIMD dot products |
| `model.odin` | Config from metadata, tensor resolution, run state |
| `forward.odin` | RMSNorm, RoPE, attention + KV cache, SwiGLU forward pass (CPU) |
| `matmul.odin` | Multithreaded tensor matmul with per-thread dequant scratch |
| `metal.odin` | Apple Silicon Metal backend: MSL GEMV/RMSNorm/RoPE/attention/SwiGLU kernels + full GPU forward pass |
| `tokenizer.odin` | Byte-level BPE tokenizer |
| `sampler.odin` | Argmax / temperature / top-p sampling |
| `main.odin` | CLI + chat loop |

#!/usr/bin/env bash
# Build CLI against the odin-infer library (sibling directory).
set -euo pipefail
cd "$(dirname "$0")"
LIB="../odin-infer"
odin build . -out:odin-infer-mac -o:speed -no-bounds-check -disable-assert -microarch:native \
  -collection:ggml=$LIB \
  -collection:infer=$LIB \
  -collection:tokenizer=$LIB \
  -collection:sampler=$LIB
echo "Built ./odin-infer-mac ($(du -h odin-infer-mac | cut -f1))"

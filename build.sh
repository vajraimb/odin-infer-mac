#!/usr/bin/env bash
# Build a self-contained release binary (tokenizer files embedded at compile time).
set -euo pipefail
cd "$(dirname "$0")"
odin build . -out:odin-infer-mac -o:speed -no-bounds-check -disable-assert -microarch:native
echo "Built ./odin-infer-mac ($(du -h odin-infer-mac | cut -f1))"

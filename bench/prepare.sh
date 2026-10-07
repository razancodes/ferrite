#!/bin/bash
# Build the llama.cpp model files for bench/compare.py from the same Hugging Face
# checkpoints ferrite converts, mirroring scripts/build_models.sh:
#   BF16 (keeps the MTP layer) -> Q8_0 -> imatrix from Q8_0 on the same calibration
#   text and amount ferrite uses (data/wiki.train.raw, 48 x 512 tokens)
#   -> Q6_K, Q4_K_M and IQ4_NL (the same non-linear 4-bit codebook as ferrite's q4).
# The other engines' models come from elsewhere:
#   OpenVINO: hf download OpenVINO/Qwen3.5-{0.8B,2B,4B}-{int8,int4}-ov --local-dir models/ov/qwen3.5-<size>-<q>
#   mistral.rs: reads the GGUF files built here
#
# usage: bench/prepare.sh [sizes...]      (default: 0.8b 2b 4b)
# needs: bench/engines/llama.cpp (release b11425 win-cpu-x64) and llama.cpp-src (same tag),
#        bench/engines/ov-venv (openvino-genai 2026.4.1), bench/engines/mistralrs (v0.9.4)
set -euo pipefail
cd "$(dirname "$0")/.."
L=bench/engines/llama.cpp
G=models/gguf
export PYTHONPATH="$PWD/bench/engines/llama.cpp-src/gguf-py" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$G" bench/runs

for size in "${@:-0.8b 2b 4b}"; do
  for s in $size; do
    hf="models/Qwen3.5-${s^^}"
    base="$G/qwen3.5-$s"
    echo "== $s: bf16 gguf"
    [ -f "$base-bf16.gguf" ] || python bench/engines/llama.cpp-src/convert_hf_to_gguf.py "$hf" --outtype bf16 --outfile "$base-bf16.gguf" > "bench/runs/convert-$s.log" 2>&1
    echo "== $s: Q8_0"
    [ -f "$base-Q8_0.gguf" ] || $L/llama-quantize "$base-bf16.gguf" "$base-Q8_0.gguf" Q8_0 8 > "bench/runs/quant-$s-Q8_0.log" 2>&1
    echo "== $s: imatrix"
    [ -f "$G/imatrix-$s.gguf" ] || $L/llama-imatrix -m "$base-Q8_0.gguf" -f data/wiki.train.raw -o "$G/imatrix-$s.gguf" \
      --chunks 48 -c 512 -t 8 > "bench/runs/imatrix-$s.log" 2>&1
    for q in Q6_K Q4_K_M IQ4_NL; do
      echo "== $s: $q"
      [ -f "$base-$q.gguf" ] || $L/llama-quantize --imatrix "$G/imatrix-$s.gguf" "$base-bf16.gguf" "$base-$q.gguf" $q 8 > "bench/runs/quant-$s-$q.log" 2>&1
    done
  done
done
ls -la "$G"

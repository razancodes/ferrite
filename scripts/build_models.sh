#!/bin/bash
# Build the full set of ferrite model files for one Hugging Face checkpoint:
#   <name>-q8.ferrite       8-bit, Hadamard-rotated                 (near-lossless)
#   <name>-q6.ferrite       6-bit, rotated, imatrix-weighted        (balanced)
#   <name>-q4.ferrite       4-bit, GPTQ, 8-bit output projections   (small)
#   <name>-q4-fast.ferrite  4-bit incl. output projections, GPTQ    (smallest/fastest)
# each with a 2-bit proxy LM head for fast generation and a frequency-ranked
# draft vocabulary for speculative decoding. GPTQ calibrates through the q8 file.
#
# usage: scripts/build_models.sh models/Qwen3.5-2B qwen3.5-2b [calib chunks]
#        SKIP_Q86=1 scripts/build_models.sh ...   (keep existing q8/q6 files)
set -euo pipefail
cd "$(dirname "$0")/.."
HF=$1
NAME=$2
CHUNKS=${3:-48}
F=${FERRITE:-./target/release/ferrite}
OUT=models
CALIB=data/wiki.train.raw
mkdir -p scratch
cat src/*.rs src/kernels/*.rs README.md scripts/*.py > scratch/code_corpus.txt

if [ -z "${SKIP_Q86:-}" ]; then
  echo "== $NAME: q8 (rotated)"
  $F convert --hf "$HF" -o "$OUT/$NAME-q8.ferrite" -q q8 2>&1 | tail -1
  echo "== $NAME: importance matrix ($CHUNKS x 512 calibration tokens)"
  [ -f "scratch/imatrix-$NAME.json" ] || $F imatrix -m "$OUT/$NAME-q8.ferrite" -f "$CALIB" -o "scratch/imatrix-$NAME.json" --chunks "$CHUNKS" 2>&1 | tail -2
  echo "== $NAME: q6 (rotated + imatrix)"
  $F convert --hf "$HF" -o "$OUT/$NAME-q6.ferrite" -q q6 --imatrix "scratch/imatrix-$NAME.json" 2>&1 | tail -1
fi
for q in q4 q4-fast; do
  echo "== $NAME: $q (GPTQ via q8)"
  $F convert --hf "$HF" -o "$OUT/$NAME-$q.ferrite" -q "$q" --imatrix "scratch/imatrix-$NAME.raw.json" \
    --gptq "$OUT/$NAME-q8.ferrite" --chunks "$CHUNKS" 2>&1 | tr '\r' '\n' | grep -E 'gptq|wrote' | tail -3
done
for q in $([ -z "${SKIP_Q86:-}" ] && echo q8 q6) q4 q4-fast; do
  $F draft-vocab -m "$OUT/$NAME-$q.ferrite" -f "$CALIB" scratch/code_corpus.txt 2>&1 | tail -1
done

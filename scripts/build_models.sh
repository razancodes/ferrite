#!/bin/bash
# Build the full set of ferrite model files for one Hugging Face checkpoint:
#   <name>-q8.ferrite   8-bit, Hadamard-rotated            (near-lossless)
#   <name>-q6.ferrite   6-bit, rotated, imatrix-weighted    (balanced)
#   <name>-q4.ferrite   4-bit mixed, imatrix-weighted       (smallest/fastest)
# each with a frequency-ranked draft vocabulary for speculative decoding.
#
# usage: scripts/build_models.sh models/Qwen3.5-2B qwen3.5-2b [calib chunks]
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

echo "== $NAME: q8 (rotated)"
$F convert --hf "$HF" -o "$OUT/$NAME-q8.ferrite" -q q8 2>&1 | tail -1
echo "== $NAME: importance matrix ($CHUNKS x 512 calibration tokens)"
[ -f "scratch/imatrix-$NAME.json" ] || $F imatrix -m "$OUT/$NAME-q8.ferrite" -f "$CALIB" -o "scratch/imatrix-$NAME.json" --chunks "$CHUNKS" 2>&1 | tail -2
echo "== $NAME: q6 (rotated + imatrix)"
$F convert --hf "$HF" -o "$OUT/$NAME-q6.ferrite" -q q6 --imatrix "scratch/imatrix-$NAME.json" 2>&1 | tail -1
echo "== $NAME: q4 (mixed + imatrix)"
$F convert --hf "$HF" -o "$OUT/$NAME-q4.ferrite" -q q4 --imatrix "scratch/imatrix-$NAME.raw.json" 2>&1 | tail -1
for q in q8 q6 q4; do
  $F draft-vocab -m "$OUT/$NAME-$q.ferrite" -f "$CALIB" scratch/code_corpus.txt 2>&1 | tail -1
done

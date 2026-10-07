#!/bin/bash
# The headline comparison: ferrite with and without the v2 decode features
# ("v1 mode": same binary and files, FERRITE_FAST_HEAD=0 FERRITE_LOOKUP_K=0
# FERRITE_DRAFT_FROM=head) and llama.cpp, launched in alternating rounds in one
# sitting, so memory-placement luck hits every engine alike. Take the median
# of the three rounds per line.
#
# usage: bench/same_session.sh > bench/runs/same_session.txt
cd "$(dirname "$0")/.."
F=./target/release/ferrite
L=./bench/engines/llama.cpp/llama-bench
V1="FERRITE_FAST_HEAD=0 FERRITE_LOOKUP_K=0 FERRITE_DRAFT_FROM=head"
fer() { # $1 label, $2 env, $3 model, $4 spec|nospec
  local flag=""; [ "$4" = nospec ] && flag="--no-spec"
  local out; out=$(env FERRITE_LOAD=heap $2 $F bench -m "models/$3.ferrite" $flag -p 32 -n 128 --reps 2 2>/dev/null | grep -o 'decode (tg128): [0-9.]*')
  echo "$1 $3 $4 ${out##* }"
}
for s in 0.8b 2b 4b; do
  for r in 1 2 3; do
    fer v2 "X=1" "qwen3.5-$s-q4" nospec
    fer v1mode "$V1" "qwen3.5-$s-q4" nospec
    echo "llama qwen3.5-$s-Q4_K_M nospec $($L -m models/gguf/qwen3.5-$s-Q4_K_M.gguf -p 0 -n 128 -t 4 -r 2 -o json 2>/dev/null | python -c 'import json,sys; print(round(json.load(sys.stdin)[0]["avg_ts"],2))')"
    fer v2 "X=1" "qwen3.5-$s-q4" spec
    fer v1mode "$V1" "qwen3.5-$s-q4" spec
    fer v2fast "X=1" "qwen3.5-$s-q4-fast" spec
  done
done

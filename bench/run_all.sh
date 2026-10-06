#!/bin/bash
# The full head-to-head run (several hours on this laptop; prepare.sh first).
# Results: bench/runs/results.jsonl -> bench/RESULTS.md
set -u
cd "$(dirname "$0")/.."
SIZES=${SIZES:-0.8b,2b,4b}
python bench/compare.py speed --sizes "$SIZES"                  # pass 1: all formats, 5 engines
python bench/compare.py chat --sizes 2b                         # 8-, 6- and 4-bit, speculation on/off
python bench/compare.py chat --sizes 0.8b,4b --formats q4       # the 4-bit tier, where all engines meet
python bench/compare.py quality --windows 32
python bench/compare.py speed --flip                            # pass 2, opposite order; samples are pooled
python bench/compare.py report

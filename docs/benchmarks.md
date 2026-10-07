# Benchmarks

Every number on this page was measured on one laptop, against the same Hugging Face checkpoints,
with the harness in [`bench/`](../bench). The raw records are in
[`bench/runs/results.jsonl`](../bench/runs/results.jsonl), and
[`bench/RESULTS.md`](../bench/RESULTS.md) is the full generated report.

- [The machine](#the-machine)
- [How to read these numbers](#how-to-read-these-numbers)
- [1. Headline: one sitting, alternating runs](#1-headline-one-sitting-alternating-runs)
- [2. Every ferrite model file](#2-every-ferrite-model-file)
- [3. Raw speed against other engines](#3-raw-speed-against-other-engines)
- [4. Real chat](#4-real-chat)
- [5. Memory](#5-memory)
- [6. Quality](#6-quality)
- [7. Correctness against Hugging Face](#7-correctness-against-hugging-face)
- [8. What it costs: file size](#8-what-it-costs-file-size)
- [Reproducing](#reproducing)

## The machine

| | |
|---|---|
| CPU | Intel Core i5-1135G7 (Tiger Lake, 4 cores / 8 threads, AVX-512 VNNI), on mains power |
| RAM | 20 GB DDR4-2667: 4 GB soldered + a 16 GB SODIMM (Intel flex mode, so only 8 GB is dual-channel) |
| Disk | 5400 rpm hard disk |
| OS | Windows 11 |

Engines compared:

| Engine | Version | Formats |
|---|---|---|
| **ferrite v2** | this repo | q8, q6, q4 (GPTQ), q4-fast |
| ferrite v1 | this repo before the decode work in v2 | q8, q6, q4 |
| llama.cpp | b11425, win-cpu-x64, `icelake` backend | Q8_0, Q6_K, Q4_K_M, IQ4_NL |
| mistral.rs | 0.9.4, CPU | Q8_0, Q6_K, Q4_K_M (it can't load IQ4_NL) |
| OpenVINO GenAI | 2026.4.1, Intel's official Qwen3.5 IRs | int8, int4 |
| Hugging Face transformers | 5.15 / torch 2.9 | f32 (reference only) |

ONNX Runtime GenAI 0.17.1 was tried too. It builds Qwen3.5 but fails to load it on CPU ("GetMemoryInfo
for CPU should not be used"), so it has no numbers here.

## How to read these numbers

- **Prefill** is how fast the prompt is read (pp512: a 512-token prompt). **Decode** is how fast
  new tokens come out (tg128: 128 tokens). Both in tokens per second, higher is better.
- **Speculation** means the model's built-in MTP head drafts a few tokens and the main model checks
  them in one pass. It doesn't change the output, only the speed.
- Everything is measured warm, after a discarded warm-up run. ferrite loads with
  `FERRITE_LOAD=heap` (one sequential read into RAM, like llama.cpp and mistral.rs do). llama.cpp
  and mistral.rs get the better of 4 and 8 threads for each metric.
- **The big caveat:** decode speed on this laptop moves 25–40% between launches, depending on where
  Windows puts the weights in its partly single-channel memory. The same binary and model decoded
  at anywhere from 42.6 to 50.4 tok/s over eight launches. So:
  - Section 1 is the fairest comparison: every engine runs in alternating rounds in one sitting,
    median of 3.
  - The other sections pool runs from different sittings. Use them to compare within a table and
    for prefill, memory and quality, which don't have this problem.

## 1. Headline: one sitting, alternating runs

4-bit weights, decode in tok/s, median of 3 rounds. "v1 mode" is the same binary and model files
with the v2 decode features switched off (`FERRITE_FAST_HEAD=0 FERRITE_LOOKUP_K=0
FERRITE_DRAFT_FROM=head`). Script: [`bench/same_session.sh`](../bench/same_session.sh).

| Model | v1 mode | **v2** | llama.cpp Q4_K_M | v2 vs llama.cpp | v1 mode + MTP | **v2 + MTP** | v2 q4-fast + MTP |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 39.9 | **54.2** (×1.36) | 36.3 | ×1.49 | 66.3 | **89.1** (×1.34) | 88.7 |
| Qwen3.5-2B | 16.1 | **21.6** (×1.34) | 15.3 | ×1.41 | 28.9 | **35.5** (×1.23) | 37.5 |
| Qwen3.5-4B | 7.7 | **9.0** (×1.17) | 6.6 | ×1.36 | 13.9 | **16.6** (×1.19) | 17.5 |

llama.cpp is shown without speculation because that is its faster setting on this model in most
cases (see [section 4](#4-real-chat)).

## 2. Every ferrite model file

One sitting, one launch per row (`ferrite bench`: pp512, tg128 without speculation, and tg128 with
MTP speculation on an essay prompt). Perplexity: wikitext-2 test, 32 windows of 512 tokens, second
half of each window scored. For reference, the unquantized f32 0.8B model scores 15.35.

| Model | Format | File | Prefill | Decode | **Decode + MTP** | Perplexity |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | q8 | 0.92 GB | 519 | 32.5 | **45.4** | 15.32 |
| Qwen3.5-0.8B | q6 | 0.72 GB | 455 | 40.8 | **60.1** | 15.27 |
| Qwen3.5-0.8B | q4 | 0.68 GB | 513 | 49.6 | **63.1** | 15.78 |
| Qwen3.5-0.8B | q4-fast | 0.66 GB | 520 | 52.0 | **69.9** | 15.97 |
| Qwen3.5-2B | q8 | 2.24 GB | 217 | 13.0 | **27.7** | 10.44 |
| Qwen3.5-2B | q6 | 1.75 GB | 190 | 18.2 | **36.6** | 10.47 |
| Qwen3.5-2B | q4 | 1.57 GB | 211 | 23.4 | **38.8** | 10.67 |
| Qwen3.5-2B | q4-fast | 1.52 GB | 215 | 24.8 | **41.7** | 10.71 |
| Qwen3.5-4B | q8 | 4.81 GB | 85 | 5.3 | **9.7** | 8.04 |
| Qwen3.5-4B | q6 | 3.73 GB | 75 | 7.2 | **11.8** | 8.09 |
| Qwen3.5-4B | q4 | 3.14 GB | 84 | 9.6 | **15.4** | 8.20 |
| Qwen3.5-4B | q4-fast | 2.97 GB | 85 | 10.2 | **15.9** | 8.22 |

## 3. Raw speed against other engines

No speculation, pp512 / tg128, tok/s, median over two passes run in opposite engine order.

### Summary: ferrite v2 against llama.cpp at every size and bit width

| Model | Bits | ferrite v2 prefill / decode | llama.cpp prefill / decode | Prefill | Decode |
|---|---|---|---|---|---|
| 0.8B | 8 | 500 / 33.1 | 120 / 21.9 | ×4.2 | +51% |
| 0.8B | 6 | 514 / 41.2 | 239 / 29.4 | ×2.1 | +40% |
| 0.8B | 4 | 590 / 47.0 | 229 / 29.0 | ×2.6 | +62% |
| 2B | 8 | 246 / 13.3 | 63 / 9.5 | ×3.9 | +39% |
| 2B | 6 | 192 / 17.4 | 130 / 12.5 | ×1.5 | +40% |
| 2B | 4 | 222 / 21.7 | 121 / 14.0 | ×1.8 | +55% |
| 4B | 8 | 89 / 5.4 | 24 / 4.2 | ×3.7 | +29% |
| 4B | 6 | 79 / 7.3 | 51 / 5.2 | ×1.5 | +40% |
| 4B | 4 | 88 / 9.8 | 49 / 6.5 | ×1.8 | +50% |

The decode ratios here come from different sittings and run a little higher than the same-sitting
ones in section 1 (+36–49% at 4-bit). Prefill doesn't depend on memory placement, so its ratios
hold either way.

### Qwen3.5-0.8B

| Engine · format | Bits | File GB | Prefill | Decode (range) |
|---|---|---|---|---|
| **ferrite v2 · q8** | 8 | 0.92 | **500.5** | **33.1** (31.4–35.1) |
| ferrite v1 · q8 | 8 | 0.84 | 336.9 | 18.3 (17.7–20.7) |
| llama.cpp · Q8_0 | 8 | 0.83 | 120.3 | 21.9 (19.5–22.9) |
| mistral.rs · Q8_0 | 8 | 0.83 | 160.9 | 17.5 |
| OpenVINO · int8 | 8 | 1.04 | 60.3 | 18.4 (12.8–18.5) |
| **ferrite v2 · q6** | 6 | 0.72 | **514.1** | **41.2** (37.9–44.3) |
| ferrite v1 · q6 | 6 | 0.64 | 397.6 | 32.9 (32.3–33.4) |
| llama.cpp · Q6_K | 6 | 0.65 | 239.2 | 29.4 (27.9–30.4) |
| mistral.rs · Q6_K | 6 | 0.65 | 143.6 | 16.7 |
| **ferrite v2 · q4** | 4 | 0.68 | **589.6** | **47.0** (45.0–49.3) |
| ferrite v1 · q4 | 4 | 0.60 | 431.3 | 32.9 (32.7–33.1) |
| llama.cpp · Q4_K_M | 4 | 0.54 | 229.3 | 29.0 (18.1–32.2) |
| llama.cpp · IQ4_NL | 4 | 0.53 | 208.1 | 30.0 (23.2–33.4) |
| mistral.rs · Q4_K_M | 4 | 0.54 | 158.9 | 18.5 |
| OpenVINO · int4 | 4 | 0.80 | 46.7 | 23.5 (15.5–23.6) |
| transformers · f32 | 32 | 3.20 | 95.9 | 4.6 (3.0–4.9) |

### Qwen3.5-2B

| Engine · format | Bits | File GB | Prefill | Decode (range) |
|---|---|---|---|---|
| **ferrite v2 · q8** | 8 | 2.24 | **246.0** | **13.3** (13.0–13.3) |
| ferrite v1 · q8 | 8 | 2.08 | 199.5 | 9.3 (9.1–9.4) |
| llama.cpp · Q8_0 | 8 | 2.08 | 62.9 | 9.5 (8.9–9.8) |
| mistral.rs · Q8_0 | 8 | 2.08 | 87.0 | 8.3 |
| OpenVINO · int8 | 8 | 2.47 | 45.8 | 8.8 (7.8–8.9) |
| **ferrite v2 · q6** | 6 | 1.75 | **191.7** | **17.4** (16.2–18.7) |
| ferrite v1 · q6 | 6 | 1.59 | 184.1 | 12.9 (12.0–13.9) |
| llama.cpp · Q6_K | 6 | 1.61 | 130.5 | 12.5 (11.8–13.2) |
| mistral.rs · Q6_K | 6 | 1.61 | 86.0 | 9.5 |
| **ferrite v2 · q4** | 4 | 1.57 | **222.0** | **21.7** (20.6–22.6) |
| ferrite v1 · q4 | 4 | 1.41 | 198.6 | 13.4 (13.0–15.0) |
| llama.cpp · Q4_K_M | 4 | 1.31 | 120.9 | 14.0 (13.3–14.4) |
| llama.cpp · IQ4_NL | 4 | 1.27 | 104.5 | 15.2 (15.0–15.3) |
| mistral.rs · Q4_K_M | 4 | 1.31 | 95.8 | 9.7 |
| OpenVINO · int4 | 4 | 1.78 | 45.1 | 12.4 (10.8–12.5) |

### Qwen3.5-4B

| Engine · format | Bits | File GB | Prefill | Decode (range) |
|---|---|---|---|---|
| **ferrite v2 · q8** | 8 | 4.81 | **89.4** | **5.4** (5.3–5.7) |
| ferrite v1 · q8 | 8 | 4.61 | 80.7 | 4.3 (3.6–4.8) |
| llama.cpp · Q8_0 | 8 | 4.61 | 23.9 | 4.2 (4.1–4.3) |
| mistral.rs · Q8_0 | 8 | 4.61 | 35.7 | 3.6 |
| OpenVINO · int8 | 8 | 4.98 | 17.7 | 3.7 |
| **ferrite v2 · q6** | 6 | 3.73 | **79.1** | **7.3** (7.2–7.3) |
| ferrite v1 · q6 | 6 | 3.53 | 71.8 | 5.4 (5.2–5.6) |
| llama.cpp · Q6_K | 6 | 3.56 | 51.3 | 5.2 (5.0–5.5) |
| mistral.rs · Q6_K | 6 | 3.56 | 32.8 | 4.2 |
| **ferrite v2 · q4** | 4 | 3.14 | **88.2** | **9.8** (8.9–10.2) |
| ferrite v1 · q4 | 4 | 2.94 | 80.2 | 6.7 (6.2–7.3) |
| llama.cpp · Q4_K_M | 4 | 2.78 | 49.3 | 6.5 (6.1–6.6) |
| llama.cpp · IQ4_NL | 4 | 2.68 | 42.7 | 6.8 (6.7–6.8) |
| mistral.rs · Q4_K_M | 4 | 2.78 | 35.2 | 4.3 |
| OpenVINO · int4 | 4 | 3.22 | 17.5 | 5.3 (4.8–5.4) |

## 4. Real chat

Six chat prompts, greedy decoding, up to 256 tokens each, thinking off, through each engine's own
chat path. "Prefill" here is the longest of the six prompts. "MTP ≤ n" is speculation with up to
n drafted tokens per step.

### Best setting per engine

Decode tok/s. Numbers without a note are each engine's default speculation setting (ferrite MTP ≤ 4,
llama.cpp MTP ≤ 3, OpenVINO MTP ≤ 2).

| Model | Bits | **ferrite v2** | ferrite v1 | llama.cpp | OpenVINO | v2 vs llama.cpp |
|---|---|---|---|---|---|---|
| 0.8B | 4 | **84.4** | 51.0 | 23.9 (MTP off) | 16.7 (MTP off) | ×3.5 |
| 2B | 8 | **27.4** | 19.9 | 11.1 | 9.2 (MTP off) | ×2.5 |
| 2B | 6 | **32.0** | 28.0 | 13.4 (MTP ≤ 1) | – | ×2.4 |
| 2B | 4 | **35.5** | 26.4 | 12.3 (MTP off) | 9.8 (MTP off) | ×2.9 |
| 4B | 4 | **16.9** | 11.4 | 6.6 | 5.3 | ×2.6 |

The gap is bigger than in raw decode because speculation pays off very differently:

- ferrite's MTP speculation makes chat 1.5–2× faster.
- llama.cpp's ranges from 0.76× to 1.22×. Its default of 3 drafts is slower than no speculation at
  0.8B and at 2B 6/4-bit.
- OpenVINO's ranges from 0.75× to 1.23×.

### All runs

| Model | Bits | Engine · format | Speculation | Decode tok/s | Prefill tok/s | Draft acceptance | Peak RAM MB |
|---|---|---|---|---|---|---|---|
| 0.8B | 4 | **ferrite v2 · q4** | MTP ≤ 4 | **84.4** | 455 | 62% | 907 |
| 0.8B | 4 | ferrite v2 · q4 | off | 60.6 | 467 | – | 810 |
| 0.8B | 4 | ferrite v1 · q4 | MTP ≤ 4 | 51.0 | 339 | 60% | 812 |
| 0.8B | 4 | ferrite v1 · q4 | off | 35.2 | 313 | – | 735 |
| 0.8B | 4 | llama.cpp · Q4_K_M | MTP ≤ 3 | 18.2 | 174 | 48% | 1323 |
| 0.8B | 4 | llama.cpp · Q4_K_M | off | 23.9 | 182 | – | 1272 |
| 0.8B | 4 | OpenVINO · int4 | MTP ≤ 2 | 14.0 | 41 | – | 2120 |
| 0.8B | 4 | OpenVINO · int4 | off | 16.7 | 39 | – | 1706 |
| 2B | 8 | **ferrite v2 · q8** | MTP ≤ 4 | **27.4** | 200 | 66% | 2461 |
| 2B | 8 | ferrite v2 · q8 | off | 15.7 | 210 | – | 2306 |
| 2B | 8 | ferrite v1 · q8 | MTP ≤ 4 | 19.9 | 188 | 66% | 2279 |
| 2B | 8 | ferrite v1 · q8 | off | 10.2 | 209 | – | 2155 |
| 2B | 8 | llama.cpp · Q8_0 | MTP ≤ 3 | 11.1 | 61 | 54% | 2765 |
| 2B | 8 | llama.cpp · Q8_0 | off | 9.1 | 57 | – | 2493 |
| 2B | 8 | OpenVINO · int8 | MTP ≤ 2 | 6.9 | 44 | – | 5366 |
| 2B | 8 | OpenVINO · int8 | off | 9.2 | 37 | – | 4373 |
| 2B | 6 | **ferrite v2 · q6** | MTP ≤ 4 | **32.0** | 155 | 66% | 1984 |
| 2B | 6 | ferrite v2 · q6 | off | 20.3 | 167 | – | 1843 |
| 2B | 6 | ferrite v1 · q6 | MTP ≤ 4 | 28.0 | 144 | 65% | 1802 |
| 2B | 6 | ferrite v1 · q6 | off | 16.5 | 140 | – | 1692 |
| 2B | 6 | llama.cpp · Q6_K | MTP ≤ 1 | 13.4 | 109 | 80% | 2283 |
| 2B | 6 | llama.cpp · Q6_K | MTP ≤ 2 | 10.4 | 91 | 67% | 2303 |
| 2B | 6 | llama.cpp · Q6_K | MTP ≤ 3 | 9.3 | 107 | 55% | 2323 |
| 2B | 6 | llama.cpp · Q6_K | MTP ≤ 4 | 6.9 | 106 | 46% | 2343 |
| 2B | 6 | llama.cpp · Q6_K | off | 11.0 | 104 | – | 2063 |
| 2B | 4 | **ferrite v2 · q4** | MTP ≤ 4 | **35.5** | 175 | 61% | 1830 |
| 2B | 4 | ferrite v2 · q4 | off | 28.0 | 219 | – | 1672 |
| 2B | 4 | ferrite v1 · q4 | MTP ≤ 4 | 26.4 | 141 | 62% | 1654 |
| 2B | 4 | ferrite v1 · q4 | off | 14.5 | 132 | – | 1528 |
| 2B | 4 | llama.cpp · Q4_K_M | MTP ≤ 3 | 10.3 | 90 | 52% | 2507 |
| 2B | 4 | llama.cpp · Q4_K_M | off | 12.3 | 97 | – | 2381 |
| 2B | 4 | OpenVINO · int4 | MTP ≤ 2 | 9.5 | 34 | – | 4113 |
| 2B | 4 | OpenVINO · int4 | off | 9.8 | 34 | – | 3165 |
| 4B | 4 | **ferrite v2 · q4** | MTP ≤ 4 | **16.9** | 80 | 65% | 3528 |
| 4B | 4 | ferrite v2 · q4 | off | 9.4 | 72 | – | 3310 |
| 4B | 4 | ferrite v1 · q4 | MTP ≤ 4 | 11.4 | 61 | 64% | 3298 |
| 4B | 4 | ferrite v1 · q4 | off | 6.2 | 61 | – | 3119 |
| 4B | 4 | llama.cpp · Q4_K_M | MTP ≤ 3 | 6.6 | 39 | 62% | 5886 |
| 4B | 4 | llama.cpp · Q4_K_M | off | 5.4 | 40 | – | 5402 |
| 4B | 4 | OpenVINO · int4 | MTP ≤ 2 | 5.3 | 14 | – | 7205 |
| 4B | 4 | OpenVINO · int4 | off | 4.3 | 12 | – | 5967 |

### Does speculation change the answer?

It shouldn't: with greedy decoding, speculation should only change the speed. Comparing the first
160 characters of all six replies with speculation on and off:

| Model | Bits | ferrite v2 | ferrite v1 | llama.cpp | OpenVINO |
|---|---|---|---|---|---|
| 0.8B | 4 | ✅ 6/6 | ✅ 6/6 | ⚠️ 3/6 | ⚠️ 3/6 |
| 2B | 8 | ✅ 6/6 | ✅ 6/6 | ✅ 6/6 | ⚠️ 3/6 |
| 2B | 6 | ✅ 6/6 | ✅ 6/6 | ✅ 6/6 | – |
| 2B | 4 | ✅ 6/6 | ✅ 6/6 | ⚠️ 3/6 | ⚠️ 4/6 |
| 4B | 4 | ✅ 6/6 | ✅ 6/6 | ✅ 6/6 | ⚠️ 4/6 |

## 5. Memory

Peak RAM during the chat runs, with speculation on (8K context):

| Model | Bits | ferrite v2 | llama.cpp | OpenVINO | ferrite saves vs llama.cpp |
|---|---|---|---|---|---|
| 0.8B | 4 | **907 MB** | 1323 MB | 2120 MB | 31% |
| 2B | 8 | **2461 MB** | 2765 MB | 5366 MB | 11% |
| 2B | 6 | **1984 MB** | 2323 MB | – | 15% |
| 2B | 4 | **1830 MB** | 2507 MB | 4113 MB | 27% |
| 4B | 4 | **3528 MB** | 5886 MB | 7205 MB | 40% |

v2 uses 100–230 MB more than v1 for the same model, which is the 2-bit proxy output layer.

## 6. Quality

### Perplexity

Wikitext-2 test set, 512-token windows, second half of each window scored, lower is better. The text
file uses LF line endings ([why](#a-note-on-line-endings)).

| Model | ferrite q8 | llama.cpp Q8_0 | ferrite q6 | llama.cpp Q6_K | **ferrite v2 q4** | ferrite v1 q4 | ferrite q4-fast | llama.cpp Q4_K_M | llama.cpp IQ4_NL |
|---|---|---|---|---|---|---|---|---|---|
| 0.8B | **15.32** | 15.38 | **15.27** | 15.48 | **15.78** | 15.92 | 15.97 | 16.31 | 16.23 |
| 2B | 10.44 | 10.44 | **10.47** | 10.48 | **10.67** | 10.76 | 10.71 | 10.67 | 10.88 |
| 4B | **8.04** | 8.06 | **8.09** | 8.12 | 8.20 | 8.25 | 8.22 | **8.17** | 8.33 |

At full precision, the 0.8B model scores 15.3475 in ferrite (f32), 15.3468 in transformers (f32)
and 15.3485 in llama.cpp (bf16).

ferrite's 8- and 6-bit files are level with llama.cpp or slightly better. At 4-bit, ferrite is
better at 0.8B, tied at 2B, and 0.3% behind at 4B.

### Distance from the 8-bit model (0.8B)

KL divergence of each format's next-token distribution from the q8 model's, and how often both pick
the same top token:

| Format | KL vs q8 | Top-1 agreement |
|---|---|---|
| q6 | 0.006 | – |
| **q4 (GPTQ, v2)** | **0.023** | **91.5%** |
| q4-fast (GPTQ, all 4-bit body) | 0.029 | – |
| q4 (v1, no GPTQ) | 0.044 | 87.7% |

GPTQ roughly halves the 4-bit error at the same file size.

### A note on line endings

My first perplexity comparison didn't line up across engines. The test file had Windows (CRLF) line
endings. llama-perplexity and Python's text mode quietly turn those into LF, but ferrite read the
raw bytes, so the engines were scoring slightly different token sequences. All perplexity numbers
here use `data/wiki.test.lf.raw`, which [`scripts/get_data.py`](../scripts/get_data.py) writes.

## 7. Correctness against Hugging Face

Checked against Hugging Face transformers 5.15 with the scripts in [`scripts/`](../scripts):

- **Logits:** ferrite f32 vs transformers float32: max |Δ| 0.007, KL ≈ 0, 100% top-1. That holds
  for batched prefill, tiled attention and step-by-step decode. 4B (which shares DeltaNet key heads
  2:1) at q8 vs transformers bf16: KL 0.0009, 100% top-1.
- **Long context:** on a full 512-token window, KL ≈ 0 at every position: in one batch, in 64- and
  256-token chunks continuing an existing state, and token by token.
- **Tokenizer:** byte-identical token ids on 2.8M wikitext tokens and a Unicode torture test.
- **Chat template:** identical to the model's own Jinja template for single and multi-turn chats,
  system prompts, reasoning history, tool messages and thinking on/off.
- **Speculation:** greedy output is byte-identical with and without it, for draft chains of 1 to 4
  tokens. With sampling, the output distribution is unchanged by construction.
- **Proxy output layer:** greedy output is byte-identical to the full layer (8 prompts, about 2,000
  tokens). Chat sampling (T=0.7, top-p 0.8) matches the exact distribution to rounding, and thinking
  sampling (T=1, top-p 0.95) to a mean total-variation distance of 1.6·10⁻⁴.

## 8. What it costs: file size

| Model | ferrite v2 q4 | ferrite v1 q4 | llama.cpp Q4_K_M |
|---|---|---|---|
| 0.8B | 0.68 GB | 0.60 GB | 0.54 GB |
| 2B | 1.57 GB | 1.41 GB | 1.31 GB |
| 4B | 3.14 GB | 2.94 GB | 2.78 GB |

ferrite's 4-bit files are 13–26% bigger than Q4_K_M:

- Every file carries a 2-bit copy of the output layer (80–200 MB).
- q4 keeps the output layer, the embeddings and two small sensitive projections at 8-bit.

That extra size buys the decode speed and the quality above, and peak RAM is still lower than
llama.cpp's.

## Reproducing

You need the model files (see [models.md](models.md)) and the wikitext data:

```bash
python scripts/get_data.py                       # data/wiki.{train,test}.raw, wiki.test.lf.raw
bench/prepare.sh                                 # llama.cpp GGUF files, converted the same way
bench/run_all.sh                                 # speed, chat and quality suites (several hours here)
python bench/compare.py report > bench/RESULTS.md
bench/same_session.sh                            # the section 1 table
```

`bench/compare.py` runs single suites too (`speed`, `chat`, `quality`, filtered with `--engines`,
`--sizes` and `--formats`). Set `FERRITE_ENGINE_LABEL` to label runs of a new ferrite build, so they
sit next to older ones in the report instead of replacing them. The llama.cpp, mistral.rs and
OpenVINO binaries and model files go in `bench/engines/` and `models/`, which git ignores.

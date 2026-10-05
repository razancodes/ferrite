# ferrite

A from-scratch CPU inference engine in Rust for the **Qwen3.5 / Qwen3.8 hybrid architecture**
(Gated DeltaNet linear attention + gated softmax attention), plus dense Qwen3. Built for and tuned on
an Intel i5-1135G7 laptop (Tiger Lake, 4C/8T, AVX-512 VNNI, 20 GB RAM, no discrete GPU).

No ML frameworks at runtime: the tokenizer, safetensors reader, quantizers, SIMD kernels, thread pool,
DeltaNet recurrence, flash attention, sampler, chat template, speculative decoder and HTTP server are
all implemented here.

## Why Qwen3.5 (and how it relates to Qwen3.8)

On Hugging Face, the Qwen3.8 release only has large checkpoints (27B dense, a 2.4T MoE, a "Flash-Next"
MoE). `Qwen3.8-27B` declares `Qwen3_5ForConditionalGeneration`, i.e. **the same architecture as
Qwen3.5**, whose small members fit a laptop: `Qwen3.5-0.8B`, `-2B`, `-4B`. ferrite implements that
architecture, so it covers both families:

* a repeating `[DeltaNet, DeltaNet, DeltaNet, Attention]` layer pattern (24 layers for 0.8B/2B, 32 for 4B)
* **Gated DeltaNet**: fused in-projection, causal depthwise conv(4) + SiLU, L2-normalised q/k, gated
  delta rule on a constant-size 128×128 state per head (value heads may share key heads, 2:1 in 4B),
  gated RMSNorm, out-projection. No KV cache, so memory per sequence is constant.
* **Gated attention**: GQA (head dim 256), per-head QK RMSNorm, partial RoPE (64 of 256 dims,
  θ = 10⁷), sigmoid output gate, f16 KV cache.
* SwiGLU MLP, RMSNorm with `(1 + w)` weights, tied 248K-token embedding.
* A **multi-token-prediction (MTP) head**, which ferrite uses for self-speculative decoding.

The vision tower of the checkpoints is ignored (text only).

## Results (i5-1135G7, Windows 11, decode in tok/s)

| Model | Format | File | Prefill | Decode | **Decode + MTP speculation** | Quality (KL ↓) |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | q8 | 0.84 GB | ~380 | 26.1 | **39.9** | 0.0007 vs f32 |
| Qwen3.5-0.8B | q6 | 0.64 GB | ~370 | 30.7 | **50.5** | 0.0057 vs f32 |
| Qwen3.5-0.8B | q4 | 0.60 GB | ~370 | 33.1 | **57.8** | 0.043 vs f32 |
| Qwen3.5-2B   | q8 | 2.08 GB | ~170 | 10.3 | **20.5** | reference |
| Qwen3.5-2B   | q6 | 1.59 GB | ~160 | 11.4–14.3 | **24–26** | 0.0054 vs q8 |
| Qwen3.5-2B   | q4 | 1.41 GB | ~170 | 15.8 | **27.5** | 0.034 vs q8 |
| Qwen3.5-4B   | q8 | 4.61 GB | ~70–100 | 3.8–4.4 | **7.4–9.2** | reference |
| Qwen3.5-4B   | q6 | 3.53 GB | ~70–90 | 5.4–6.2 | **10.8–11.1** | 0.0050 vs q8 |
| Qwen3.5-4B   | q4 | 2.94 GB | ~70–80 | 6.7–6.8 | **13.8–14.3** | 0.030 vs q8 |

KL is the mean token-level KL divergence on wikitext-2. For scale, q8 matches f32 with 98.5% top-1
agreement, q6 with ~96%, and q4 with ~88–91%. **Recommended: `q6`** (near-q8 quality at ~25% fewer
bytes). Use `qwen3.5-2b-q6` for speed, `qwen3.5-4b-q6` for the smartest answers.

Where the time goes:

* **Decode is memory-bound.** Non-speculative decode streams weights at 19–22 GB/s. A pure AVX-512
  read benchmark (`examples/bw_bench.rs`) tops out at 22–24 GB/s on this machine, so that's ~90%
  of what the DRAM delivers. The RAM is a mismatched 16 GB + 4 GB pair running at DDR4-2667
  (Intel flex mode); a matched 3200 MT/s pair would raise decode speed by roughly 20%.
* **MTP speculation gives 1.5–2.1x.** The largest gain is on the biggest model, where weight
  streaming dominates most.
* Benchmarks on this laptop vary by ±10–15% run to run (thermals, background load).

**Correctness** (against Hugging Face transformers 5.15):

* f32 engine vs HF float32 logits: max |Δ| 0.007, KL ≈ 0, 100% top-1, in batched prefill, tiled
  attention and step-by-step decode. 4B (2:1 DeltaNet heads) q8 vs HF bf16: KL 0.0009, 100% top-1.
* Tokenizer: byte-identical ids on 2.8M wikitext tokens plus a Unicode torture test.
* Chat template: identical to the model's Jinja template for single/multi-turn, system, reasoning
  history and tool messages, thinking on/off.
* Speculative decoding is lossless: greedy output is byte-identical with and without it (checked
  for draft chains of 1–4 tokens); with sampling the output distribution is provably unchanged.

## Quick start

```bash
cargo build --release

# download (any Qwen3.5 / Qwen3 checkpoint works)
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B

# convert (recipes: q8 | q6 | q4 | q4-fast | f32), or build all three tuned variants:
./target/release/ferrite convert --hf models/Qwen3.5-2B -o models/qwen3.5-2b-q8.ferrite -q q8
scripts/build_models.sh models/Qwen3.5-2B qwen3.5-2b      # q8 + imatrix + q6 + q4 + draft vocab

# chat in the terminal (/think, /nothink, /reset, /stats, /quit)
./target/release/ferrite chat -m models/qwen3.5-2b-q6.ferrite

# one-shot
./target/release/ferrite run -m models/qwen3.5-2b-q6.ferrite -p "Explain DeltaNet in two sentences."

# OpenAI-compatible server + built-in web UI at http://127.0.0.1:8080
./target/release/ferrite serve -m models/qwen3.5-2b-q6.ferrite
```

### Desktop chat app

`ferrite chat` is a Tauri 2 desktop app (`app/`). The Rust side runs the engine in-process; the
front end is plain HTML/CSS/JS with no bundler. Its look follows the portfolio's "Dither Ledger"
design system: ink/paper themes with an ember accent, Cormorant Garamond, Manrope and IBM Plex
Mono, hairline ledger rows, and an 8x8 Bayer-dithered WebGL contour field. The field ripples as
tokens arrive.

Double-click `chat.cmd`. It builds the app the first time and then launches it. To rebuild
after changes, run `chat.cmd rebuild`. To build by hand:

```bash
# WinLibs must be first on PATH so Tauri's resource step finds a 64-bit windres
PATH="$LOCALAPPDATA/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH" \
  cargo build --release -p ferrite-app
target/release/ferrite-chat.exe
```

* **Models:** the app lists the `.ferrite` files in `./models` as a ledger showing quant, size
  and parameter count. *Browse* opens any other model file. On start it reopens the last model
  you used, or `qwen3.5-2b-q6` the first time.
* **Chat:** answers stream as markdown. LaTeX renders with KaTeX (`$…$`, `$$…$$`, `\(…\)`,
  `\[…\]`). Code blocks get highlight.js and a copy button. Reasoning goes in a collapsible
  block, and links open in your default browser.
* **Stats on each reply:** tok/s, tokens, prompt and cached tokens, draft acceptance, time to
  first token, and context used. Each reply also has Copy and Regenerate buttons.
* **Machine panel:** shows the CPU, ISA and thread count, plus a live tok/s readout with a
  dithered sparkline.
* **Settings:** THINK, SPEC (MTP speculation), MOTION and INK/PAPER toggles; sampling (Qwen
  presets for chat and thinking); and the system prompt. Settings are saved across runs.
* **Shortcuts:** Enter sends and Shift+Enter adds a new line. Start a message with `/think` or
  `/nothink` to switch modes. Esc stops generation, Ctrl+N starts a new chat and Ctrl+L focuses
  the input.
* **Keeping the model fast:** the engine runs on its own thread, and follow-up turns resume from
  a state snapshot. While a reply streams, only the paragraph still growing is re-rendered,
  syntax highlighting waits until the reply finishes, and the dither field drops to 15 fps.
  MOTION off stops the animation entirely.
* **Scripted demo:** `FERRITE_CHAT_AUTOSEND="first message||second message"` sends messages
  automatically once the model has loaded.

Fonts, KaTeX, marked and highlight.js are vendored under `app/ui/vendor/`, so the app runs fully
offline. On the windows-gnu target, `app/build.rs` copies `WebView2Loader.dll` next to the exe.

### HTTP API

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Hi!"}], "stream": true, "enable_thinking": false }'
```

The server supports `stream`, `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
`frequency_penalty`, `repetition_penalty`, `seed`, `stop`, `max_tokens`, and thinking via
`enable_thinking` or `chat_template_kwargs.enable_thinking`. The reasoning is returned as
`reasoning_content`. Responses include `usage` and llama.cpp-style `timings`.

Common options: `-t/--threads` (pool size, default all logical cores), `--ctx` (default 8192),
`--no-spec`, `--temp`, `--top-k`, `--top-p`, `--min-p`, `--presence-penalty`, `--seed`.
Defaults follow Qwen's recommendations (non-thinking: T=0.7, top-p 0.8, top-k 20, presence 1.5;
thinking: T=1.0, top-p 0.95).

## How it works

```
HF safetensors ──convert──> .ferrite (one mmap-able file: config + tokenizer + repacked weights)
                                 │
            ┌────────────────────┴────────────────────┐
         Engine: tokenizer → chat template → prime (prefix reuse / snapshot) → generate
            │                                          │
   forward (chunked prefill, 256 tok)          speculative step:
   N × [RMSNorm → DeltaNet | Attention           MTP drafts ≤4 tokens → verify [tok, d1..dn]
         → RMSNorm → SwiGLU]                     in one batch → accept prefix → replay-rollback
            │
   kernels: AVX-512 VNNI int8 GEMM/GEMV (q8/q6/q4), Hadamard rotation, vectorized exp/SiLU,
            delta rule, conv4, tiled flash attention, flash-decoding, adaptive-width thread pool
```

### Weight formats (`src/quant.rs`, `src/kernels/`)

All quantized formats are **row-interleaved**: 16 output rows are packed together so one `vpdpbusd`
computes 4-element dot products for 16 rows at once (one row per int32 lane), with the int8
activation broadcast straight from memory. There are no horizontal reductions in the GEMV/GEMM.

| Format | bits/weight | Storage | Unpack in the kernel |
|---|---|---|---|
| `Q8R` | 8.5 | int8 + f16 scale per 32 | none |
| `Q6R` | 6.5 | 4-bit low plane + 2-bit high plane | shifts + one `vpternlogd`; the +32 offset folds into the block correction |
| `Q4R` | 4.5 | 4-bit index into the non-linear IQ4_NL codebook | `vpshufb` table lookup |

Scales come from an importance-weighted search with least-squares refits. The importance can come
from an **imatrix**: activation statistics collected on calibration text.

**Recipes**, chosen from measurements on Qwen3.5-0.8B against f32:

* **Hadamard rotation** (QuIP#/QuaRot-style, randomized 256-wide Walsh–Hadamard on weight input
  dimensions, applied to activations on the fly with an in-register AVX-512 FWHT) cuts KL by 40%
  for q8 and 30% for q6. It *hurts* q4: there, imatrix-weighted rounding of a few salient input
  channels matters more, and rotation spreads their energy. So `q8`/`q6` rotate and `q4` does not.
* Per-tensor sensitivity sweep at 4-bit: the DeltaNet `out_proj` and attention `o_proj` are tiny
  but the most damaging per byte, so `q4` keeps them at 8-bit. The tied LM head is the cheapest
  trade, which `q4-fast` takes.
* Overrides: `-q "q4,down=q8,mtp=q4,lm_head=q6"` (classes, `class@layer`, `mtp`, `mtp.<class>`).

The converter also pre-bakes everything it can: `(1 + w)` norm offsets folded, `-exp(A_log)`
precomputed, q/k/v/z, q/gate/k/v and gate/up projections fused, conv weights transposed, tokenizer
embedded.

### Speculative decoding (`src/engine.rs`)

Qwen3.5 ships a 1-layer MTP head that predicts token t+2 from (embedding of token t+1, hidden at t).
Each decode step:

1. **Draft a chain of up to 4 tokens.** The first comes from the main model's hidden state; later
   ones feed the MTP head its own output. The chain stops once the head's top probability drops
   below 0.3. Drafting uses a 32K-row draft LM head of the most frequent tokens (`draft-vocab`
   ranks tokens on a corpus and stores the list inside the `.ferrite` file), so each draft costs
   about 1/8 of an LM-head pass.
2. **Verify.** The main model runs `[next, d1..dn]` as one batch; decode is memory-bound, so this
   costs about one token.
3. **Accept left to right.** Each draft is accepted with probability p(d) under the fully processed
   sampling distribution, which is exact speculative sampling for a deterministic draft.
4. **Roll back on rejection.** The DeltaNet state is read once per verify step (kept in L2 across
   the batch) and each token's rank-1 update (k, δ, decay) is logged. On rejection the state is
   rebuilt by replaying the accepted updates on the pre-batch state, using the same fused ops, so
   it's bitwise identical. Conv states keep per-token checkpoints, and the KV cache only needs its
   length reset.

Measured acceptance is 55–90% per draft depending on text; tuned defaults are `FERRITE_DRAFT_K=4`
and `FERRITE_DRAFT_CONF=0.3`.

### Gated DeltaNet and attention

* The recurrence `S ← e^g·S; δ = β(v − Sᵀk); S ← S + kδᵀ; o = Sᵀq` runs per head with 128-wide
  state rows in eight zmm registers.
* Prefill attention is a tiled flash-attention kernel. f16 K/V tiles of 64 keys are converted once
  and shared by all query rows of a GQA group (16 tokens × 4 heads), with a 4×64 register-blocked
  score GEMM and online softmax.
* Decode attention splits long contexts across threads (flash-decoding).
* Only a quarter of the layers have a KV cache, so long contexts stay cheap. At a 3.5K-token
  prompt, 2B q6 prefills at ~120–150 tok/s and still decodes at ~17 tok/s with speculation.

### Threading (`src/threadpool.rs`)

A persistent pool with spin-waiting workers, so one dispatch costs about one cache-line round trip.
It has an adjustable width: prefill (compute-bound) uses all 8 hyper-threads, while decode
(memory-bound) uses 4. Workers outside the width sleep on a condvar, so they don't eat the
laptop's shared turbo budget. The dispatcher only touches the mutex when a sleeping worker is
needed, tracked with a sleeper bitmask.

### Serving (`src/server.rs`, `web/index.html`)

std-only HTTP/1.1 with SSE streaming. Multi-turn chats reuse work through **recurrent-state
snapshots**: a hybrid model cannot rewind its DeltaNet state to an arbitrary prefix the way a
KV cache can, so the engine snapshots the (constant-size) state right before each assistant
header, and the next turn resumes from there. Stop strings are handled with hold-back, so a
partially matched stop sequence is never streamed.

## Commands

| Command | Purpose |
|---|---|
| `convert` | HF checkpoint → `.ferrite` (`-q f32\|q8\|q6\|q4\|q4-fast` + overrides, `--imatrix`, `--rotate/--no-rotate`) |
| `run` / `chat` / `serve` | generation (MTP speculation on by default; `--no-spec` to disable) |
| `bench` | prefill + decode throughput (decode measured on real text so speculation is realistic) |
| `kld` | KL divergence, top-1 agreement and perplexity of a model vs a reference model |
| `ppl` | perplexity on a text file |
| `imatrix` | activation statistics (rotated and raw) for importance-weighted quantization |
| `draft-vocab` | frequency-ranked draft vocabulary, stored in place in a `.ferrite` file |
| `tokenize`, `template`, `dump-logits` | debugging and parity tests |

Environment variables: `FERRITE_PROFILE=1` (per-op timing table), `FERRITE_DECODE_THREADS`,
`FERRITE_DRAFT_K`, `FERRITE_DRAFT_CONF`, `FERRITE_DRAFT_Q`, `FERRITE_LOAD=heap`,
`FERRITE_FORCE_SCALAR=1` (disable AVX-512 kernels), `FERRITE_NO_TILED_ATTN=1`.

## Validation

```bash
python scripts/hf_reference.py models/Qwen3.5-0.8B scratch/ref.npy "some text" [--bf16]  # prints ids
./target/release/ferrite dump-logits -m model.ferrite --tokens <ids> -o scratch/fe.bin
python scripts/compare_logits.py scratch/ref.npy scratch/fe.bin
python scripts/tokenizer_parity.py models/Qwen3.5-0.8B model.ferrite data/wiki.test.raw
python scripts/template_parity.py models/Qwen3.5-0.8B model.ferrite
./target/release/ferrite kld -m model-q6.ferrite --reference model-q8.ferrite -f data/wiki.test.raw
cargo test --release
cargo run --release --example kernel_bench   # VNNI peak + DRAM-streaming GEMV per format
cargo run --release --example bw_bench       # memory bandwidth ceiling
```

## Notes for this machine

* The system `C:\MinGW` is a 32-bit GCC 6.3 toolchain whose `dlltool`/`as` shadow the 64-bit ones
  that raw-dylib import libraries (windows-sys, used by Tauri) need. WinLibs MinGW-w64 is
  installed (`winget install BrechtSanders.WinLibs.POSIX.UCRT`), and `.cargo/config.toml` points
  rustc at its `dlltool`, which finds its own 64-bit `as.exe` next to it. Tauri's `windres`
  is looked up on PATH only, so builds put WinLibs `bin` first (`chat.cmd` does this).
* Windows cannot overwrite a file that a running process has memory-mapped. Don't convert onto a
  `.ferrite` file that a running `ferrite` is using.

## Roadmap

* Qwen3.8-27B: the architecture is supported, but at q4 it needs ~15 GB of RAM and would run at
  ~1–2 tok/s here.
* GPTQ-style error-compensated 4-bit quantization (needs per-layer activation covariances).
* AVX2 fallback kernels for CPUs without AVX-512 (a portable scalar path exists today).

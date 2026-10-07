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

One session, one launch per row (`ferrite bench`: pp512, tg128 without speculation, and tg128
with MTP speculation on an essay prompt). Decode on this laptop varies 25–40% between launches
with where Windows places the weights in its partly single-channel memory, so compare rows
within this table, not across tables; the interleaved medians below are the better guide to
absolute speed.

| Model | Format | File | Prefill | Decode | **Decode + MTP** | Wikitext-2 perplexity |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | q8 | 0.92 GB | 519 | 32.5 | **45.4** | 15.32 |
| Qwen3.5-0.8B | q6 | 0.72 GB | 455 | 40.8 | **60.1** | 15.27 |
| Qwen3.5-0.8B | q4 | 0.68 GB | 513 | 49.6 | **63.1** | 15.78 |
| Qwen3.5-0.8B | q4-fast | 0.66 GB | 520 | 52.0 | **69.9** | 15.97 |
| Qwen3.5-2B   | q8 | 2.24 GB | 217 | 13.0 | **27.7** | 10.44 |
| Qwen3.5-2B   | q6 | 1.75 GB | 190 | 18.2 | **36.6** | 10.47 |
| Qwen3.5-2B   | q4 | 1.57 GB | 211 | 23.4 | **38.8** | 10.67 |
| Qwen3.5-2B   | q4-fast | 1.52 GB | 215 | 24.8 | **41.7** | 10.71 |
| Qwen3.5-4B   | q8 | 4.81 GB | 85 | 5.3 | **9.7** | 8.04 |
| Qwen3.5-4B   | q6 | 3.73 GB | 75 | 7.2 | **11.8** | 8.09 |
| Qwen3.5-4B   | q4 | 3.14 GB | 84 | 9.6 | **15.4** | 8.20 |
| Qwen3.5-4B   | q4-fast | 2.97 GB | 85 | 10.2 | **15.9** | 8.22 |

Perplexity: 32 × 512-token wikitext-2 windows, second half scored (f32 0.8B: 15.35). Files include
the 2-bit proxy LM head (80–200 MB). **Recommended: `q4`** for speed (GPTQ makes it close to q6:
at 0.8B its KL against q8 is 0.023, q6's 0.006, the old non-GPTQ q4's 0.044) or `q6` for the
best quality per byte; `q4-fast` is the fastest. `qwen3.5-2b-q4` is a good everyday model,
`qwen3.5-4b-q4` the smartest that stays fast.

**Decode optimizations (v2), same session, interleaved launches, median of 3:** the same binary
and files with the v2 features off (`FERRITE_FAST_HEAD=0 FERRITE_LOOKUP_K=0
FERRITE_DRAFT_FROM=head`) against defaults, and llama.cpp b11425 in the same rounds:

| q4 | v1 decode | **v2 decode** | llama.cpp Q4_K_M | v1 + MTP | **v2 + MTP** | v2 q4-fast + MTP |
|---|---|---|---|---|---|---|
| 0.8B | 39.9 | **54.2** (×1.36) | 36.3 | 66.3 | **89.1** (×1.31) | 88.7 |
| 2B   | 16.1 | **21.6** (×1.34) | 15.3 | 28.9 | **35.5** (×1.22) | 37.5 |
| 4B   | 7.7  | **9.0** (×1.19)  | 6.6  | 13.9 | **16.6** (×1.17) | 17.5 |

Where the time goes:

* **Decode is memory-bound.** Without speculation every token streams the weights once, at
  ~22–29 GB/s on this machine. The RAM is 4 GB soldered + a 16 GB DIMM at DDR4-2667 (Intel flex
  mode: only 8 GB is dual-channel), which also makes speed depend on page placement.
* **The LM head was the biggest single cost** (248K-token vocabulary: 47% of the bytes per token
  at 0.8B). A 2-bit proxy now scores the vocabulary and the exact head only the shortlist.
* **MTP speculation gives 1.5–2x** on top, more with prompt lookup on repetitive text (code edits).

**Correctness** (against Hugging Face transformers 5.15):

* f32 engine vs HF float32 logits: max |Δ| 0.007, KL ≈ 0, 100% top-1, in batched prefill, tiled
  attention and step-by-step decode. 4B (2:1 DeltaNet heads) q8 vs HF bf16: KL 0.0009, 100% top-1.
* On a full 512-token wikitext window, KL ≈ 0 at every position in one batch, in 64- and 256-token
  chunks continuing an existing state (`dump-logits --chunk`), and token by token. Wikitext-2
  perplexity over 32 windows: 15.3475, against 15.3468 from HF transformers.
* Tokenizer: byte-identical ids on 2.8M wikitext tokens plus a Unicode torture test.
* Chat template: identical to the model's Jinja template for single/multi-turn, system, reasoning
  history and tool messages, thinking on/off.
* Speculative decoding is lossless: greedy output is byte-identical with and without it (checked
  for draft chains of 1–4 tokens); with sampling the output distribution is provably unchanged.

## Compared with other engines

`bench/` runs ferrite head to head against llama.cpp (b11425), mistral.rs (0.9.4), OpenVINO GenAI
(2026.4.1) and HF transformers on the same Qwen3.5 checkpoints on this laptop. Full tables and
method notes: [`bench/RESULTS.md`](bench/RESULTS.md) (rows labelled `ferrite v2` are this version).

| 2B, 4-bit | ferrite v2 q4 | llama.cpp Q4_K_M | mistral.rs Q4_K_M | OpenVINO int4 |
|---|---|---|---|---|
| Prefill, pp512 (tok/s) | **222** | 121 | 96 | 45 |
| Decode, no speculation (tok/s) | **21.6** | 15.3 (same session) | 9.7 | 12.4 |
| Chat decode, best speculation setting (tok/s) | **35.5** | 12.3 (MTP off is its best here) | — | 9.8 |
| Peak RAM in chat | **1.8 GB** | 2.4 GB | — | 3.2 GB |
| Wikitext-2 perplexity | **10.67** | 10.67 | (same file) | — |

* **Prefill** is 1.5–4.2× llama.cpp, 2–3× mistral.rs and 4–9× OpenVINO at every size and bit width.
* **Decode without speculation**, same session: 1.32–1.49× llama.cpp at 4-bit (the proxy LM head
  cuts the bytes read per token by 16% at 4B to 32% at 0.8B; before it, engines were at parity).
* **Speculation**: ferrite's MTP gives 1.5–2× in chat, llama.cpp's 0.76–1.22× (its default of 3
  drafts is slower than no speculation at 0.8B and 2B 6/4-bit), OpenVINO's 0.75–1.23×. ferrite's
  speculative output is identical to plain greedy decoding; llama.cpp's and OpenVINO's diverge
  in some replies.
* **Quality**: with GPTQ, ferrite's q4 matches llama.cpp's Q4_K_M at 2B (10.67 vs 10.67), beats it
  at 0.8B (15.78 vs 16.31) and is within 0.3% at 4B; 8- and 6-bit are on par or slightly better.
* **Memory**: llama.cpp peaks at 1.2–1.8× and OpenVINO at 2.2–2.6× ferrite's RAM in chat.

Reproduce: `bench/prepare.sh` (llama.cpp files), the OpenVINO downloads listed there, then
`bench/run_all.sh` (several hours here).

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

* **Models:** a dropdown of the `.ferrite` files in `./models`, showing each one's quant and size.
  *Browse* opens any other model file. On start it reopens the last model you used, or
  `qwen3.5-2b-q6` the first time.
* **Chat:** answers stream as markdown. LaTeX renders with KaTeX (`$…$`, `$$…$$`, `\(…\)`,
  `\[…\]`). Code blocks get highlight.js and a copy button. Reasoning goes in a collapsible
  block, and links open in your default browser.
* **Stats on each reply:** tok/s, tokens, prompt and cached tokens, draft acceptance, time to
  first token, and context used. Each reply also has Copy and Regenerate buttons.
* **Machine panel:** shows the CPU, ISA and thread count, plus a live tok/s readout with a
  dithered sparkline.
* **Settings:** THINK, SPEC (MTP speculation), MOTION and INK/PAPER toggles; sampling (Qwen
  presets for chat and thinking); and the system prompt. Settings are saved across runs.
* **Presets:** *Max think* turns thinking on with Qwen's thinking sampler, a 32K context and up
  to 32K new tokens. It reloads the model if the context size changes. *Max speed* turns
  thinking off and MTP speculation on, uses the chat sampler with 2K max tokens, and stops the
  animation. Neither preset changes which model is loaded.
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
   N × [RMSNorm → DeltaNet | Attention           MTP drafts ≤4 tokens (or a prompt-lookup
         → RMSNorm → SwiGLU]                     continuation) → verify [tok, d1..dn] in one
            │                                    batch → accept prefix → replay-rollback
   LM head: 2-bit proxy scores the vocabulary, the 8-bit head scores the shortlist
            │
   kernels: AVX-512 VNNI int8 GEMM/GEMV (q8/q6/q4/q2), Hadamard rotation, vectorized exp/SiLU,
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
| `Q2R` | 2.5 | 2-bit index into a 4-level codebook (proxy LM head only) | shift + mask, `vpshufb` |

Scales come from an importance-weighted search with least-squares refits. The importance can come
from an **imatrix**: activation statistics collected on calibration text. 4-bit files use **GPTQ**
(`convert --gptq <q8 file>`): columns are rounded in order and each column's error is pushed
onto the columns not yet rounded through the Cholesky factor of H⁻¹, with H = XᵀX of that
matrix's calibration inputs (collected through the q8 model, a few layers at a time). At 0.8B it
halves the 4-bit KL against q8 (0.044 → 0.023) at the same size.

**Proxy LM head.** The vocabulary is 248K tokens, so the LM head is huge: at 0.8B the 8-bit head
was 47% of the bytes read per token. Every file therefore also stores a 2-bit copy of it
(`lm_proxy`, 80 MB at 0.8B). Generation scores the whole vocabulary with the proxy, then
re-scores only the 16-row groups within 8 logits of each row's best with the exact 8-bit head
(at most 256 groups). Shortlisted tokens get exact logits. Greedy output is byte-identical to the
full head (checked on 8 prompts, ~2,000 tokens); chat sampling (T=0.7, top-p 0.8) matches the
exact distribution to rounding, thinking sampling (T=1, top-p 0.95) to a mean total-variation
distance of 1.6·10⁻⁴. Perplexity, KL and logit dumps always use the exact head.

**Recipes**, chosen from measurements on Qwen3.5-0.8B against f32:

* **Hadamard rotation** (QuIP#/QuaRot-style, randomized 256-wide Walsh–Hadamard on weight input
  dimensions, applied to activations on the fly with an in-register AVX-512 FWHT) cuts KL by 40%
  for q8 and 30% for q6. It *hurts* q4: there, imatrix-weighted rounding of a few salient input
  channels matters more, and rotation spreads their energy. So `q8`/`q6` rotate and `q4` does not.
* Per-tensor sensitivity sweep at 4-bit: the DeltaNet `out_proj` and attention `o_proj` are tiny
  but the most damaging per byte, so `q4` keeps them at 8-bit. `q4-fast` makes them 4-bit too
  (4% smaller, ~6% faster at 0.8B); with GPTQ its KL against q8 (0.029) is still lower than the
  old non-GPTQ q4's (0.044).
* Overrides: `-q "q4,down=q8,mtp=q4,lm_head=q6"` (classes, `class@layer`, `mtp`, `mtp.<class>`),
  and `lm_proxy=q4|q2|none` for the proxy head.

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
   about 1/8 of an LM-head pass. Like the main head it is scanned with its 2-bit proxy rows and
   only the shortlist is scored with 8-bit rows: drafting from low-bit rows alone costs 4 points
   of acceptance, the shortlist none.
   **Prompt lookup** can replace the rest of the chain: when the last 3 tokens occurred earlier in
   the conversation and the MTP head's first draft agrees with what followed them, up to 8 tokens
   of that continuation are drafted for free. It is used only while lookup-drafted steps deliver
   at least the tokens/s of MTP-drafted ones (+27% on a code edit, neutral on chat).
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
| `convert` | HF checkpoint → `.ferrite` (`-q f32\|q8\|q6\|q4\|q4-fast` + overrides, `--imatrix`, `--gptq <q8 file>`, `--rotate/--no-rotate`) |
| `run` / `chat` / `serve` | generation (MTP speculation on by default; `--no-spec` to disable) |
| `bench` | prefill + decode throughput (decode measured on real text so speculation is realistic) |
| `kld` | KL divergence, top-1 agreement and perplexity of a model vs a reference model |
| `ppl` | perplexity on a text file |
| `imatrix` | activation statistics (rotated and raw) for importance-weighted quantization |
| `draft-vocab` | frequency-ranked draft vocabulary, stored in place in a `.ferrite` file |
| `tokenize`, `template`, `dump-logits` | debugging and parity tests |

Environment variables: `FERRITE_PROFILE=1` (per-op timing table), `FERRITE_DECODE_THREADS`,
`FERRITE_DRAFT_K`, `FERRITE_DRAFT_CONF`, `FERRITE_DRAFT_Q`, `FERRITE_DRAFT_FROM=head` (draft from
the exact head only), `FERRITE_DRAFT_DELTA` / `FERRITE_DRAFT_CAP`, `FERRITE_FAST_HEAD=0` (exact
LM head in generation), `FERRITE_HEAD_DELTA` (8) / `FERRITE_HEAD_CAP` (256), `FERRITE_LOOKUP_K`
(8, 0 = no prompt lookup), `FERRITE_LOAD=heap`, `FERRITE_GPTQ_MB` (3072),
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
* The drive is a 5400 rpm hard disk. With the default mmap loading, a model that isn't in the
  page cache arrives through random page faults: the first run of 2B q8 took 47 minutes.
  `FERRITE_LOAD=heap` reads the file sequentially instead (about a minute). Once cached, both
  modes decode at the same speed.
* Memory is 4 GB soldered + a 16 GB SODIMM (flex mode: 8 GB dual-channel, 12 GB single-channel,
  both at 2667 because of the soldered part). Where the OS puts the weights decides decode speed:
  the same binary and model ran at 42.6–50.4 tok/s across eight launches (24.6–29.1 GB/s).
  `examples/placement_probe.rs` shows 64 MB chunks are mixtures (16–27 GB/s), so picking chunks
  gains little. Page-level control needs AWE physical pages: grant the account "Lock pages in
  memory" (SeLockMemoryPrivilege), sign out and in, then `cargo run --release --example
  awe_probe` measures bandwidth by physical address range.

## Roadmap

* Qwen3.8-27B: the architecture is supported, but at q4 it needs ~15 GB of RAM and would run at
  ~1–2 tok/s here.
* Weight placement in the dual-channel region via AWE physical pages, once the privilege is set
  (see *Notes for this machine*).
* Branching (tree) drafts: verify a second candidate for the first draft position as well.
* AVX2 fallback kernels for CPUs without AVX-512 (a portable scalar path exists today).

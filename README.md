# ferrite

A from-scratch CPU inference engine in Rust for the **Qwen3.5 / Qwen3.8 hybrid architecture**
(Gated DeltaNet linear attention + gated softmax attention), plus dense Qwen3. Built for and tuned on
an Intel i5-1135G7 laptop (Tiger Lake, 4C/8T, AVX-512 VNNI, 20 GB RAM, no discrete GPU).

No ML frameworks at runtime: the tokenizer, safetensors reader, quantizer, SIMD kernels, thread pool,
DeltaNet recurrence, attention, sampler, chat template, speculative decoder and HTTP server are all
implemented here.

## Why Qwen3.5 (and how it relates to Qwen3.8)

On Hugging Face, the Qwen3.8 release only has large checkpoints (27B dense, a 2.4T MoE, a "Flash-Next"
MoE). `Qwen3.8-27B` declares `Qwen3_5ForConditionalGeneration`, i.e. **the same architecture as
Qwen3.5**, whose small members fit a laptop: `Qwen3.5-0.8B`, `-2B`, `-4B`, `-9B`. So ferrite implements
that architecture, which covers both families:

* 24 layers for 0.8B/2B, in a repeating `[DeltaNet, DeltaNet, DeltaNet, Attention]` pattern
* **Gated DeltaNet**: fused in-projection, causal depthwise conv(4) + SiLU, L2-normalised q/k,
  gated delta rule on a constant-size 128×128 state per head, gated RMSNorm, out-projection.
  No KV cache, so memory per sequence is constant.
* **Gated attention**: GQA (8 q / 2 kv heads, head dim 256), per-head QK RMSNorm, partial RoPE
  (64 of 256 dims, θ = 10⁷), sigmoid output gate, f16 KV cache.
* SwiGLU MLP, RMSNorm with `(1 + w)` weights, tied 248K-token embedding.
* A **multi-token-prediction (MTP) head**, which ferrite uses for self-speculative decoding.

The vision tower of the checkpoints is ignored (text only).

## Results (i5-1135G7, Windows 11)

| Model | File | Prefill | Decode | Decode + MTP speculation |
|---|---|---|---|---|
| Qwen3.5-0.8B Q8 | 0.82 GB | ~390 tok/s | 22.6 tok/s | **32.0 tok/s** |
| Qwen3.5-0.8B Q4 | 0.56 GB | ~390 tok/s | 30.0 tok/s | **41.8 tok/s** |
| Qwen3.5-2B Q8   | 2.07 GB | ~177 tok/s | 9.7 tok/s  | **15.6 tok/s** |

Decode without speculation runs at 18–19.5 GB/s of weight streaming, close to the 20–23 GB/s this
machine's DRAM delivers in a pure read benchmark (decode is memory-bound).

**Correctness** (all against Hugging Face transformers 5.15, float32):

* f32 engine vs HF logits: max |Δ| 0.007, KL ≈ 0, 100% top-1, both batched prefill and step-by-step decode.
* Tokenizer: byte-identical ids on 2.8M wikitext tokens plus a Unicode torture test
  (`scripts/tokenizer_parity.py`).
* Chat template: identical to the model's Jinja template for single/multi-turn, system, reasoning
  history and tool messages, thinking on/off (`scripts/template_parity.py`).
* Q8 vs f32 on wikitext-2: mean KL 0.0012, 98.2% top-1 agreement, perplexity +0.6%.
* Q4 vs f32: mean KL 0.069, 84% top-1, perplexity +5.2%. Still being tuned; see roadmap.
* Speculative decoding is lossless: greedy output is byte-identical with and without it; with
  sampling the output distribution is provably unchanged.

## Quick start

```bash
cargo build --release

# download (any Qwen3.5 / Qwen3 checkpoint works)
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B

# convert: q8 (near-lossless) | q4 (smaller/faster) | f32 (reference)
./target/release/ferrite convert --hf models/Qwen3.5-2B -o models/qwen3.5-2b-q8.ferrite -q q8

# chat in the terminal (/think, /nothink, /reset, /stats, /quit)
./target/release/ferrite chat -m models/qwen3.5-2b-q8.ferrite

# one-shot
./target/release/ferrite run -m models/qwen3.5-2b-q8.ferrite -p "Explain DeltaNet in two sentences."

# OpenAI-compatible server + built-in web UI at http://127.0.0.1:8080
./target/release/ferrite serve -m models/qwen3.5-2b-q8.ferrite
```

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
HF safetensors ──convert──> .ferrite (single mmap-able file: config + tokenizer + repacked weights)
                                 │
            ┌────────────────────┴────────────────────┐
         Engine: tokenizer → chat template → prime (prefix reuse / snapshot) → generate
            │                                          │
   forward (chunked prefill, 256 tok)          speculative step:
   24 × [RMSNorm → DeltaNet | Attention          MTP drafts 1 token → verify [tok, draft]
         → RMSNorm → SwiGLU]                     in one batch → accept or roll back
            │
   kernels: AVX-512 VNNI int8 GEMM/GEMV, vectorized exp/SiLU/softmax,
            delta rule, conv4, flash-decoding attention, persistent thread pool
```

### Weight formats and kernels (`src/quant.rs`, `src/kernels/`)

* **Row-interleaved quantization.** 16 output rows are packed together so one `vpdpbusd` computes
  4-element dot products for 16 rows at once (one row per int32 lane), with the activation
  broadcast from memory (`{1to16}`). No horizontal reductions anywhere in the GEMV/GEMM.
* `Q8R`: int8 weights, 32-column blocks, f16 scales (8.5 bits/weight). Weights are stored as
  `w + 128` (VNNI needs u8 × s8), so a single correction `128·Σx` per block works for every format.
* `Q4R`: 4-bit indices into the non-linear IQ4_NL codebook (4.5 bits/weight), decoded in-register
  with `vpshufb`. Scales are chosen by a search that minimises importance-weighted error, with
  least-squares refits; an optional imatrix supplies the importance.
* Activations are quantized to int8 per 32-block on the fly.
* Prefill uses an 8-token register tile: every weight byte loaded is reused 8 times, which makes
  the problem compute-bound. The kernel reaches ~64% of single-core VNNI peak; the laptop throttles
  to ~1.7 GHz under all-core AVX-512 load (measured with `examples/kernel_bench.rs`).
* Vectorized `exp` (range reduction + degree-6 polynomial + `vscalefps`). The MinGW C runtime's
  scalar `expf` cost ~80 ns/call and was the largest prefill cost before this.
* The converter pre-bakes everything it can: `(1 + w)` norm offsets folded, `-exp(A_log)`
  precomputed, q/k/v/z, q/gate/k/v and gate/up projections fused, conv weights transposed.

### Gated DeltaNet

The recurrence `S ← e^g·S; δ = β(v − Sᵀk); S ← S + kδᵀ; o = Sᵀq` runs per head with the
128-wide state rows held in eight zmm registers. The checkpoint for speculative rollback is free:
token *c* writes its updated state into the checkpoint buffer and token *c+1* reads it back, so a
rejected draft is undone by swapping two buffer pointers, with zero extra memory traffic.

### MTP self-speculative decoding (`src/engine.rs`)

Qwen3.5 ships a 1-layer MTP head that predicts token t+2 from (embedding of token t+1, hidden
state at t). Each decode step:

1. MTP drafts `d` using a **reduced 32K-row draft LM head** gathered from the full tied head
   (frequent tokens plus control tokens), so drafting costs ~1/8 of an LM-head pass.
2. The main model runs `[next, d]` as one batch. Decode is memory-bound, so 2 tokens cost about
   the same as 1.
3. `d` is accepted with probability p(d) under the fully processed sampling distribution
   (exact speculative sampling for a deterministic draft). Otherwise a token is drawn from p
   with `d` removed, and the state rolls back by pointer swap.

Measured acceptance is 60–90%, giving 1.4–1.7x faster decoding.

### Threading (`src/threadpool.rs`)

A persistent pool with spin-waiting workers, so one dispatch costs about one cache-line round trip.
It has an adjustable width: prefill (compute-bound) uses all 8 hyper-threads, while decode
(memory-bound) uses 4. Workers outside the width sleep on a condvar, so they don't eat the
laptop's shared turbo budget, and the dispatcher only touches the mutex when a sleeping worker
is needed.

### Serving (`src/server.rs`, `web/index.html`)

std-only HTTP/1.1 with SSE streaming. Multi-turn chats reuse work through **recurrent-state
snapshots**: a hybrid model cannot rewind its DeltaNet state to an arbitrary prefix the way a
KV cache can, so the engine snapshots the (constant-size) state right before each assistant
header, and the next turn resumes from there. Stop strings are handled with hold-back, so a
partially matched stop sequence is never streamed.

## Commands

| Command | Purpose |
|---|---|
| `convert` | HF checkpoint → `.ferrite` (`-q f32\|q8\|q4\|q4-fast`, per-class overrides like `q4,down=q8`, `--imatrix`) |
| `run` / `chat` / `serve` | generation |
| `bench` | prefill + decode throughput (decode measured on real text so speculation is realistic) |
| `kld` | KL divergence, top-1 agreement and perplexity of a model vs a reference model |
| `ppl` | perplexity on a text file |
| `imatrix` | collect activation statistics for importance-weighted 4-bit quantization |
| `tokenize`, `template`, `dump-logits` | debugging and parity tests |

Environment variables: `FERRITE_PROFILE=1` (per-op timing table), `FERRITE_DECODE_THREADS`,
`FERRITE_FORCE_SCALAR=1` (disable AVX-512 kernels).

## Validation scripts

```bash
python scripts/hf_reference.py models/Qwen3.5-0.8B scratch/ref.npy "some text"   # prints token ids
./target/release/ferrite dump-logits -m model.ferrite --tokens <ids> -o scratch/fe.bin
python scripts/compare_logits.py scratch/ref.npy scratch/fe.bin
python scripts/tokenizer_parity.py models/Qwen3.5-0.8B model.ferrite data/wiki.test.raw
python scripts/template_parity.py models/Qwen3.5-0.8B model.ferrite
cargo test --release
```

## Notes for this machine

* The system `C:\MinGW` is a 32-bit GCC 6.3 toolchain whose `dlltool`/`as` shadow rustup's 64-bit
  ones. `.cargo/config.toml` points rustc at the bundled `dlltool`, and clap's color feature is
  disabled so no dependency needs raw-dylib import libraries.
* Windows cannot overwrite a file that is memory-mapped by a running process. Don't convert onto a
  `.ferrite` file that a running `ferrite` is using.

## Roadmap

* Q4 quality: per-tensor sensitivity sweep, imatrix-weighted quantization, mixed Q4/Q8 recipes.
  The imatrix tooling is built; the sweep is pending.
* Qwen3.5-4B (≈2.6 GB at Q4). Qwen3.8-27B would fit at Q4 (~15 GB), but at ~1 tok/s on this laptop.
* Multi-token drafts (chained MTP), AVX2 fallback kernels for CPUs without AVX-512.

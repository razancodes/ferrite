# How it works

This is a tour of the engine, from the model architecture down to the kernels. For numbers, see
[benchmarks.md](benchmarks.md).

- [The model](#the-model)
- [The pipeline](#the-pipeline)
- [Weight formats](#weight-formats)
- [The proxy output layer](#the-proxy-output-layer)
- [Quantization recipes](#quantization-recipes)
- [Speculative decoding](#speculative-decoding)
- [DeltaNet and attention](#deltanet-and-attention)
- [Threading](#threading)
- [Serving and multi-turn chat](#serving-and-multi-turn-chat)
- [Where the time goes](#where-the-time-goes)

## The model

ferrite implements the Qwen3.5 architecture. Qwen3.6 and Qwen3.8 reuse it unchanged
(`Qwen3.8-27B` declares `Qwen3_5ForConditionalGeneration`), so all three families are covered.

- **Layer pattern:** `[DeltaNet, DeltaNet, DeltaNet, Attention]`, repeated: 24 layers at 0.8B and
  2B, 32 at 4B.
- **Gated DeltaNet** (linear attention): a fused input projection, a causal depthwise convolution
  (width 4) with SiLU, L2-normalized q and k, and the gated delta rule on a fixed 128×128 state per
  head. Value heads can share key heads (2:1 in the 4B). Then a gated RMSNorm and the output
  projection. There is no KV cache, so these layers use the same memory at any context length.
- **Gated attention:** grouped-query attention with head dim 256, per-head RMSNorm on q and k,
  partial RoPE (64 of 256 dims, θ = 10⁷), a sigmoid output gate, and an f16 KV cache.
- **MLP and norms:** SwiGLU MLP, RMSNorm with `(1 + w)` weights, and a tied 248K-token embedding.
- **Multi-token prediction (MTP) head:** a one-layer block that predicts the token after next.
  ferrite uses it for self-speculative decoding.

The checkpoints also contain a vision encoder, which ferrite ignores; it is text only.

## The pipeline

```
HF safetensors ──convert──> .ferrite  (one file: config + tokenizer + repacked weights)
                                 │
            ┌────────────────────┴────────────────────┐
         Engine: tokenizer → chat template → prime (prefix reuse / snapshot) → generate
            │                                          │
   forward (chunked prefill, 256 tokens)       speculative step:
   N × [RMSNorm → DeltaNet | Attention           MTP drafts ≤ 4 tokens (or a prompt-lookup
         → RMSNorm → SwiGLU]                     continuation) → verify [tok, d1..dn] in one
            │                                    batch → accept a prefix → replay rollback
   output layer: 2-bit proxy scores the vocabulary, the 8-bit layer scores the shortlist
            │
   kernels: AVX-512 VNNI int8 GEMM/GEMV (q8/q6/q4/q2), Hadamard rotation, vectorized exp/SiLU,
            delta rule, conv4, tiled flash attention, flash decoding, adaptive-width thread pool
```

The converter does as much work as it can ahead of time:

- It folds the `(1 + w)` norm offsets and precomputes `-exp(A_log)`.
- It fuses the q/k/v/z, q/gate/k/v and gate/up projections and transposes the conv weights.
- It embeds the tokenizer.

At run time a `.ferrite` file is either memory-mapped or read into RAM in one pass
(`FERRITE_LOAD=heap`), and the weights are used in place.

## Weight formats

Source: `src/quant.rs`, `src/kernels/`.

Every quantized format is **row-interleaved**: 16 output rows are packed together. One `vpdpbusd`
then computes 4-element dot products for all 16 rows at once (one row per int32 lane), with the int8
activations broadcast straight from memory. There are no horizontal reductions anywhere in the
GEMV or GEMM.

| Format | Bits per weight | Storage | Unpacking in the kernel |
|---|---|---|---|
| `Q8R` | 8.5 | int8 + an f16 scale per 32 weights | none |
| `Q6R` | 6.5 | a 4-bit low plane + a 2-bit high plane | shifts and one `vpternlogd`; the +32 offset folds into the block correction |
| `Q4R` | 4.5 | 4-bit index into the non-linear IQ4_NL codebook | a `vpshufb` table lookup |
| `Q2R` | 2.5 | 2-bit index into a 4-level codebook (proxy output layer only) | shift, mask, `vpshufb` |

Block scales come from an importance-weighted search with least-squares refits. The importance can
come from an **imatrix**, which is activation statistics collected on calibration text.

4-bit files use **GPTQ** (`convert --gptq <q8 file>`):

- Columns are rounded in order, and each column's rounding error is pushed onto the columns not yet
  rounded.
- The weighting is the Cholesky factor of H⁻¹, where H = XᵀX is built from that matrix's
  calibration inputs.
- H is collected by running calibration text through the 8-bit model a few layers at a time
  (`FERRITE_GPTQ_MB` caps the memory).

At 0.8B this halves the 4-bit error against q8 (KL 0.044 → 0.023) at the same file size.

## The proxy output layer

Qwen3.5's vocabulary has 248K tokens, so the output layer (the "LM head") is huge. At 0.8B the 8-bit
head was 47% of all bytes read per generated token. Since decode speed is set by how many bytes
each token reads, this was the biggest single cost.

Every file therefore also stores a 2-bit copy of the head (`lm_proxy`, 80 MB at 0.8B). During
generation:

1. The 2-bit proxy scores the whole vocabulary.
2. For each position, ferrite takes the 16-token groups within 8 logits of the best one (at most
   256 groups).
3. The file's exact head (8-bit in q8 and q4, 6-bit in q6) re-scores only those groups.
   Everything outside the shortlist is far too unlikely to be sampled.

**Accuracy:**

- Greedy output is byte-identical to using the full head (8 prompts, about 2,000 tokens).
- Chat sampling (T=0.7, top-p 0.8) matches the exact distribution to rounding.
- Thinking sampling (T=1, top-p 0.95) differs from it by a mean total-variation distance of 1.6·10⁻⁴.
- Perplexity, KL and logit dumps always use the exact head.

Knobs: `FERRITE_FAST_HEAD=0` turns it off; `FERRITE_HEAD_DELTA` (8) and `FERRITE_HEAD_CAP` (256) set
the window.

## Quantization recipes

The recipes come from measurements on Qwen3.5-0.8B against the f32 model.

| Recipe | Body | Kept at 8-bit | Rotation | Notes |
|---|---|---|---|---|
| `q8` | Q8R | everything | yes | near-lossless |
| `q6` | Q6R | the tiny DeltaNet gate projections (`in_ba`) | yes | best quality per byte |
| `q4` | Q4R, GPTQ | output layer / embeddings, `in_ba`, DeltaNet `out_proj`, attention `o_proj` | no | the recommended default |
| `q4-fast` | Q4R, GPTQ | output layer / embeddings, `in_ba` | no | ~4% smaller and up to ~6% faster than q4 |

Every quantized recipe also stores the 2-bit proxy output layer described above.

Why these choices:

- **Hadamard rotation** (as in QuIP# and QuaRot) applies a randomized 256-wide Walsh–Hadamard
  transform to each weight's input dimension. The activations get the same transform on the fly,
  with an in-register AVX-512 FWHT.
  - It cuts KL by 40% for q8 and 30% for q6.
  - It *hurts* q4. There, what matters most is rounding a few salient input channels carefully
    (imatrix weighting), and rotation spreads their energy out. So q8 and q6 rotate, and q4 doesn't.
- **A per-tensor sensitivity sweep at 4-bit** found that the DeltaNet `out_proj` and attention
  `o_proj` are tiny but do the most damage per byte when quantized. q4 keeps them at 8-bit.
- **q4-fast** quantizes them to 4-bit too. With GPTQ its KL against q8 (0.029) is still lower than
  the old non-GPTQ q4's (0.044).

Overrides go after the recipe:

- Matrix classes: `-q "q4,down=q8,mtp=q4,lm_head=q6"`. Classes can be `class`, `class@layer`,
  `mtp` or `mtp.<class>`.
- The proxy head: `lm_proxy=q4|q2|none`.

## Speculative decoding

Source: `src/engine.rs`.

Qwen3.5's MTP head predicts token t+2 from the embedding of token t+1 and the main model's hidden
state at t. Each decode step:

1. **Draft a chain of up to 4 tokens.**
   - The first draft uses the main model's hidden state. Later drafts feed the MTP head its own
     output.
   - The chain stops once the head's top probability drops below 0.3.
   - Drafting uses a smaller output layer: the 32K most frequent tokens, which `draft-vocab` ranks
     on a corpus and stores in the `.ferrite` file. A draft costs about 1/8 of a full output-layer
     pass.
   - Like the main head, the draft head is scanned with 2-bit rows and only its shortlist is scored
     with 8-bit rows. Drafting from 2-bit rows alone lost 4 points of acceptance; the shortlist
     loses none.
2. **Or use prompt lookup.**
   - If the last 3 tokens appeared earlier in the conversation, and the MTP head's first draft
     agrees with what followed them there, up to 8 tokens of that earlier continuation become the
     drafts for free.
   - ferrite keeps a running average of tokens/s for lookup steps and MTP steps, and only uses
     lookup while it's winning.
   - It gave +27% on a code edit and is neutral in ordinary chat.
3. **Verify.** The main model runs `[next, d1..dn]` as one batch. Decode is limited by memory
   bandwidth, so checking 5 tokens costs about the same as generating 1.
4. **Accept from left to right.** Each draft is accepted with probability p(d) under the fully
   processed sampling distribution. That is exact speculative sampling for a deterministic draft,
   so the output distribution doesn't change.
5. **Roll back on rejection.** This is the hard part for a hybrid model:
   - A KV cache can simply be truncated, but a DeltaNet state is a running sum that can't be
     un-added.
   - So ferrite reads the state once per verify step (it stays in L2 across the batch) and logs each
     token's rank-1 update (k, δ, decay).
   - On rejection, it rebuilds the state by replaying the accepted updates onto the pre-batch state,
     with the same fused ops, so the result is bitwise identical.
   - The conv states keep per-token checkpoints, and the attention KV cache just has its length
     reset.

Acceptance is 55–90% per draft depending on the text. Knobs: `FERRITE_DRAFT_K` (4),
`FERRITE_DRAFT_CONF` (0.3), `FERRITE_LOOKUP_K` (8, 0 turns prompt lookup off), `--no-spec`.

## DeltaNet and attention

- **The DeltaNet recurrence** `S ← e^g·S; δ = β(v − Sᵀk); S ← S + kδᵀ; o = Sᵀq` runs per head. Each
  128-wide state row lives in eight zmm registers.
- **Prefill attention** is a tiled flash-attention kernel:
  - f16 K/V tiles of 64 keys are converted once and shared by every query row in a GQA group
    (16 tokens × 4 heads).
  - Scores come from a 4×64 register-blocked GEMM with an online softmax.
- **Decode attention** splits long contexts across threads (flash decoding).
- Only a quarter of the layers have a KV cache, so long contexts stay cheap. With a 3.5K-token
  prompt, 2B q6 prefills at ~120–150 tok/s and still decodes at ~17 tok/s with speculation.

## Threading

Source: `src/threadpool.rs`.

ferrite uses a persistent pool with spin-waiting workers, so one dispatch costs about one cache-line
round trip.

- **Adjustable width.** Prefill is compute-bound and uses all 8 hyper-threads. Decode is
  memory-bound and uses 4. Extra threads wouldn't help it, and on a laptop they eat into the shared
  turbo budget.
- **Idle workers sleep.** Workers outside the current width wait on a condvar.
- **The fast path avoids the lock.** The dispatcher touches the mutex only when it needs to wake a
  sleeping worker, which it tracks with a bitmask.

## Serving and multi-turn chat

Source: `src/server.rs`, `web/index.html`.

The server is std-only HTTP/1.1 with SSE streaming and an OpenAI-compatible API.

- **Multi-turn chats reuse earlier work with state snapshots.**
  - A hybrid model can't rewind its DeltaNet state to an arbitrary earlier point the way a KV cache
    can.
  - So the engine snapshots the (fixed-size) state right before each assistant reply.
  - The next turn resumes from that snapshot instead of re-reading the whole conversation.
- **Stop strings are held back.** A partially matched stop sequence is never streamed out.

## Where the time goes

- **Decode is memory-bound.** Without speculation, every token reads all the weights once. On this
  laptop that happens at 22–29 GB/s, so tokens/s is roughly bandwidth divided by bytes per token.
  The two ways to go faster are:
  - read fewer bytes per token (smaller formats, the proxy output layer);
  - produce more than one token per read (speculation).
- **Prefill is compute-bound**, which is where the VNNI kernels and tiled attention matter.
- **Profiling:** `FERRITE_PROFILE=1` prints a per-op timing table after a run.
  `cargo run --release --example kernel_bench` measures the VNNI peak and the DRAM-streaming speed of
  each format, and `--example bw_bench` measures the machine's memory bandwidth ceiling.

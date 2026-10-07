# Models

ferrite runs one model family: **Qwen3.5**, and Qwen3.6 and Qwen3.8, which use the same
architecture. Within that family it runs any size, as long as the model is dense (no
mixture-of-experts) and comes as BF16, F16 or F32 safetensors.

- [Compatible models](#compatible-models)
- [Not supported](#not-supported)
- [Choosing a format](#choosing-a-format)
- [Converting a model](#converting-a-model)
- [New Qwen releases](#new-qwen-releases)
- [Other model families](#other-model-families)

## Compatible models

| Model | Status | Notes |
|---|---|---|
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | ✅ tested | the fast one: 89 tok/s with speculation on my laptop |
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | ✅ tested | a good everyday model: 35 tok/s |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | ✅ tested | the smartest one that still feels quick: 17 tok/s |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | ✅ supported, not yet tested | ~6.5 GB at q4; I'd expect ~4–5 tok/s, ~7–8 with speculation |
| Qwen3.5-0.8B / 2B / 4B / 9B **-Base** | ✅ supported, not yet tested | pretrained only, not chat-tuned: use `ferrite run --raw` for text completion |
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B), [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | ⚠️ supported, impractical on a laptop | ~15 GB of RAM at q4 and ~1–2 tok/s on my machine |
| Fine-tunes of any of the above | ✅ if the architecture is unchanged | `config.json` should say `qwen3_5` (or `qwen3_6` / `qwen3_8`), and the weights should be BF16/F16 safetensors |
| Dense Qwen3 (Qwen3-0.6B … 32B, Qwen3-4B-2507) | 🧪 experimental | there's a code path for it, but it has never been run against Hugging Face. These models have no MTP head, so there's no speculation either. |

The 9B and 27B figures are estimates from model size and memory bandwidth, not measurements.

## Not supported

| | Why |
|---|---|
| ❌ Mixture-of-experts models: Qwen3.5-35B-A3B, 122B-A10B, 397B-A17B, Qwen3.6-35B-A3B, Qwen3.8-2.4T-A95B, Qwen3.8-Flash-Next | Expert routing isn't implemented. The converter refuses them with a clear error. |
| ❌ Qwen's FP8 and GPTQ-Int4 uploads | The converter reads BF16/F16/F32 only. Use the BF16 originals; ferrite does its own quantization. |
| ❌ Gemma, Llama, Mistral, Phi, … | Different architectures. See [below](#other-model-families). |
| Vision and audio input | The vision encoder in the checkpoints is skipped; ferrite is text only. |

## Choosing a format

Each model can be converted at several precisions. These are the files I use, with sizes for each
model:

| Format | 0.8B | 2B | 4B | Pick it when |
|---|---|---|---|---|
| `q8` | 0.92 GB | 2.24 GB | 4.81 GB | you want output that's practically identical to the original |
| `q6` | 0.72 GB | 1.75 GB | 3.73 GB | you want the best quality per byte |
| **`q4`** | 0.68 GB | 1.57 GB | 3.14 GB | **the default**: fast, and GPTQ keeps it close to q6 |
| `q4-fast` | 0.66 GB | 1.52 GB | 2.97 GB | you want the last few percent of speed |

Speed and quality for every file are in [benchmarks.md](benchmarks.md#2-every-ferrite-model-file).
The details of each recipe are in
[how-it-works.md](how-it-works.md#quantization-recipes).

## Converting a model

You need Python with `huggingface_hub` for the download, and a release build of ferrite
(`cargo build --release`).

**Quick: one 8-bit file, a few minutes**

```bash
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B
ferrite convert --hf models/Qwen3.5-2B -o models/qwen3.5-2b-q8.ferrite -q q8
ferrite chat -m models/qwen3.5-2b-q8.ferrite
```

**Full: all four formats, tuned**

```bash
python scripts/get_data.py                              # wikitext-2 calibration text, once
scripts/build_models.sh models/Qwen3.5-2B qwen3.5-2b
```

`build_models.sh` does the following:

1. Converts `q8` (Hadamard-rotated).
2. Collects an importance matrix on 48 × 512 tokens of calibration text.
3. Builds `q6` with that matrix.
4. Builds `q4` and `q4-fast` with GPTQ, calibrated by running the text through the `q8` file.
5. Ranks a draft vocabulary for speculative decoding and stores it in each file.

On my laptop's hard disk this takes from tens of minutes for 0.8B to a few hours for 4B.
`SKIP_Q86=1` skips rebuilding existing q8/q6 files.

**By hand**

```bash
ferrite convert --hf <dir> -o <file> -q q4 --gptq <q8 file> --imatrix scratch/imatrix-<name>.raw.json
ferrite convert --hf <dir> -o <file> -q "q4,down=q8,lm_proxy=none"    # per-class overrides
ferrite draft-vocab -m <file> -f data/wiki.train.raw                   # draft vocabulary
```

Before trusting a new conversion, compare it with the original. Two ways:

- `ferrite kld -m new.ferrite --reference model-q8.ferrite -f data/wiki.test.raw`
- The Hugging Face parity scripts in [development.md](development.md#checking-against-hugging-face).

The desktop app lists every `.ferrite` file in `models/`. Click **Rescan** after adding one.

## New Qwen releases

What happens with a future release depends on what changed:

| What changes | Works? |
|---|---|
| New sizes, depths or widths | ✅ Automatically. Layer counts and pattern, head counts and sizes, RoPE settings, vocabulary size, tied embeddings, the MTP head and the context length are all read from `config.json`. |
| Same architecture under a new name (`qwen3_9`, `qwen4`, …) | ⚠️ A one-line change. `src/config.rs` checks the `model_type` against a list and refuses names it doesn't know. |
| Small architectural changes (a new norm, a new layer type, a changed MTP head, a new tokenizer split rule) | 🔧 Days of work. |
| Mixture-of-experts | 🔧 Weeks: expert routing, new kernels, memory handling. |

To find out, run `ferrite convert` on it. It either converts or stops with a specific error, such as
"unsupported model_type", "mixture-of-experts checkpoints are not supported yet" or "unknown layer
type". If it converts, check it against Hugging Face before trusting the output.

## Other model families

ferrite is a specialist. A Gemma or a Llama is a different architecture and needs its own
implementation.

What would carry over to a new family:

- the quantization formats, GPTQ and the proxy output layer;
- prompt lookup, the kernels, the server and the desktop app.

What's architecture-specific:

- the layer code;
- the tokenizer's pre-split rules;
- the chat format;
- the MTP-based speculation.

**Next on the list: Gemma 4 E2B and E4B.** They're Apache-2.0, sized well for a laptop, and they have
a 262K-token vocabulary, where the proxy output layer pays off even more. Supporting them needs:

- a SentencePiece-style tokenizer and Gemma's chat format;
- sliding-window attention;
- Gemma's per-layer embeddings and shared KV layers.

Gemma has no MTP head, but Google publishes small `-assistant` draft models for speculation.

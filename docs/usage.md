# Using ferrite

- [Command line](#command-line)
- [HTTP server](#http-server)
- [Desktop app](#desktop-app)
- [Environment variables](#environment-variables)
- [Tips](#tips)

Examples use `ferrite` for `./target/release/ferrite` (`ferrite.exe` on Windows). They're written
for a bash shell, which on Windows means Git Bash.

## Command line

```bash
# interactive chat in the terminal (/think, /nothink, /reset, /stats, /quit)
ferrite chat -m models/qwen3.5-2b-q4.ferrite

# one prompt
ferrite run -m models/qwen3.5-2b-q4.ferrite -p "Explain DeltaNet in two sentences."
ferrite run -m models/qwen3.5-4b-q4.ferrite -p "Is 1001 prime?" --think -n 2048

# raw completion, no chat template (useful for -Base models)
ferrite run -m models/qwen3.5-2b-q4.ferrite --raw -p "The capital of France is"

# speed test: prefill (512 tokens) and decode (128 tokens), with and without speculation
ferrite bench -m models/qwen3.5-2b-q4.ferrite
ferrite bench -m models/qwen3.5-2b-q4.ferrite --no-spec
```

### Commands

| Command | What it does |
|---|---|
| `convert` | Hugging Face checkpoint → `.ferrite` file. See [models.md](models.md#converting-a-model). |
| `run` | Generate a reply to one prompt. |
| `chat` | Interactive chat in the terminal. |
| `serve` | OpenAI-compatible HTTP server with a small built-in web chat. |
| `bench` | Prefill and decode speed. Decode runs on real text, so speculation behaves realistically. |
| `ppl` | Perplexity of a text file. |
| `kld` | KL divergence, top-1 agreement and perplexity of one model against a reference model. |
| `imatrix` | Collect activation statistics for importance-weighted quantization. |
| `draft-vocab` | Rank tokens by frequency on a corpus and store the list in a model (for the draft head). |
| `tokenize`, `template`, `dump-logits` | Debugging and parity tests against Hugging Face. |

Run `ferrite <command> --help` for every option.

### Common options

| Option | Default | Meaning |
|---|---|---|
| `-m, --model` | – | the `.ferrite` file |
| `-t, --threads` | all logical cores | thread pool size (decode uses fewer, see `FERRITE_DECODE_THREADS`) |
| `--ctx` | 8192 | maximum context length in tokens |
| `--no-spec` | off | turn off speculative decoding (the output doesn't change, only the speed) |
| `--think` | off | thinking mode (the model reasons before answering) |
| `--system` | – | system prompt |
| `--temp`, `--top-k`, `--top-p`, `--min-p`, `--presence-penalty`, `--repeat-penalty`, `--seed` | Qwen's recommendations | sampling; `--temp 0` is greedy |
| `-n, --max-tokens` | 512 | maximum new tokens |

The sampling defaults follow Qwen's recommendations:

| Mode | Temperature | top-p | top-k | Presence penalty |
|---|---|---|---|---|
| Normal chat | 0.7 | 0.8 | 20 | 1.5 |
| Thinking | 1.0 | 0.95 | 20 | 1.5 |

## HTTP server

```bash
ferrite serve -m models/qwen3.5-2b-q4.ferrite            # http://127.0.0.1:8080
ferrite serve -m models/qwen3.5-2b-q4.ferrite --host 0.0.0.0 --port 9000
```

Opening the address in a browser gives a minimal chat page. The API follows OpenAI's:

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | chat, streaming or not |
| `POST /v1/completions` | same handler |
| `GET /v1/models` | the loaded model |
| `GET /health` | liveness check |

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Hi!"}],
  "stream": true,
  "enable_thinking": false
}'
```

**Request options:**

- Sampling: `stream`, `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
  `frequency_penalty`, `repetition_penalty`, `seed`.
- Output: `stop`, `max_tokens`.
- Thinking: `enable_thinking` (or `chat_template_kwargs.enable_thinking`).

**Responses:**

- In thinking mode, the reasoning comes back as `reasoning_content`.
- Responses include `usage` and llama.cpp-style `timings`, including draft counts.
- Follow-up turns in the same conversation resume from a saved state, so a long chat doesn't
  re-read its history every turn.

Any OpenAI client library works if you point its base URL at `http://127.0.0.1:8080/v1`.

## Desktop app

The desktop app (`app/`, built as `ferrite-chat`) is a Tauri 2 application. The engine runs
in-process on its own thread; the front end is plain HTML, CSS and JavaScript with no bundler. Its look follows
the "Dither Ledger" style of my portfolio: ink and paper themes with an ember accent, Cormorant
Garamond, Manrope and IBM Plex Mono, hairline ledger rows, and an 8×8 Bayer-dithered contour field
that ripples as tokens arrive.

On Windows, double-click **`chat.cmd`**. It builds the app the first time and then launches it.
After pulling changes, run `chat.cmd rebuild`. Build requirements are in
[development.md](development.md#building-on-windows).

What's in it:

- **Models.** A dropdown of the `.ferrite` files in `./models` with each one's format and size.
  *Browse* opens a file from anywhere else, and *Rescan* picks up new files. On start it reopens the
  model you used last.
- **Chat.**
  - Answers stream as Markdown. LaTeX renders with KaTeX (`$…$`, `$$…$$`, `\(…\)`, `\[…\]`).
  - Code blocks get syntax highlighting and a copy button.
  - Reasoning goes into a collapsible block.
- **Stats under every reply:** tok/s, token counts, prompt and cached tokens, draft acceptance,
  tokens drafted by prompt lookup, and time to first token.
- **Machine panel.** The CPU, instruction set and threads, plus:
  - the model's format (and whether it was built with GPTQ);
  - whether the proxy output layer is in use and how speculation is set up;
  - a live tok/s readout with a dithered sparkline.
- **Settings.** THINK, SPEC, MOTION and INK/PAPER toggles, sampling sliders, and a system prompt.
  Everything is saved between runs.
- **Presets.**
  - *Max think* turns thinking on with Qwen's thinking sampler, a 32K context and up to 32K new
    tokens.
  - *Max speed* turns thinking off, speculation on and the animation off.
- **Shortcuts.**
  - Enter sends; Shift+Enter adds a new line.
  - `/think` or `/nothink` at the start of a message switches mode.
  - Esc stops generation, Ctrl+N starts a new chat, and Ctrl+L focuses the input.

The app loads models into RAM rather than memory-mapping them (`FERRITE_LOAD=heap`), because that's
much faster on a hard disk. Set `FERRITE_LOAD=mmap` before launching to change that. Fonts, KaTeX,
marked and highlight.js are bundled in `app/ui/vendor/`, so the app works fully offline.

## Environment variables

**Loading and threads**

| Variable | Default | Effect |
|---|---|---|
| `FERRITE_LOAD` | mmap (CLI), heap (app) | `heap` reads the model into RAM in one pass instead of memory-mapping it |
| `FERRITE_DECODE_THREADS` | half the pool, at most 4 | threads used for decode |

**Output layer**

| Variable | Default | Effect |
|---|---|---|
| `FERRITE_FAST_HEAD` | 1 | `0` scores the whole vocabulary with the exact output layer |
| `FERRITE_HEAD_DELTA` | 8 | shortlist window, in logits below the best token |
| `FERRITE_HEAD_CAP` | 256 | maximum number of 16-token groups in the shortlist |

**Speculation**

| Variable | Default | Effect |
|---|---|---|
| `FERRITE_DRAFT_K` | 4 | maximum drafted tokens per step |
| `FERRITE_DRAFT_CONF` | 0.3 | stop the draft chain below this probability |
| `FERRITE_DRAFT_FROM` | – | `head` drafts from the exact output-layer rows only (no 2-bit scan) |
| `FERRITE_DRAFT_Q` | – | re-quantize the draft head to another format, e.g. `q4` (files without a proxy) |
| `FERRITE_DRAFT_DELTA`, `FERRITE_DRAFT_CAP` | 12 / 256 | the draft head's shortlist window |
| `FERRITE_LOOKUP_K` | 8 | maximum prompt-lookup draft length (`0` turns it off) |

**Conversion and debugging**

| Variable | Default | Effect |
|---|---|---|
| `FERRITE_GPTQ_MB` | 3072 | memory budget for GPTQ statistics during conversion |
| `FERRITE_PROFILE` | off | `1` prints a per-op timing table |
| `FERRITE_FORCE_SCALAR` | off | disable the AVX-512 kernels (for testing) |
| `FERRITE_NO_TILED_ATTN` | off | use the simple attention kernel for prefill |
| `FERRITE_CHAT_AUTOSEND` | – | desktop app: `"msg1\|\|msg2"` is sent automatically once a model loads |

## Tips

- **Which file?** `q4` is the best default. `q4-fast` is a little faster, and `q6` is a little
  smarter. The 2B is a good everyday model; the 4B is the smartest one that still feels quick.
- **Spinning disk?** Use `FERRITE_LOAD=heap` with the CLI. A cold memory-mapped load on a 5400 rpm
  disk turns into random reads. The first 2B q8 run here took 47 minutes that way, against about a
  minute with `heap`.
- **Windows can't overwrite a file that a running process has memory-mapped.** Don't convert onto a
  `.ferrite` file that a running ferrite is using.
- **Decode speed varies between launches** on machines with mismatched memory (see
  [hardware.md](hardware.md#memory-matters-more-than-cores)). When comparing two settings,
  alternate them in the same sitting.

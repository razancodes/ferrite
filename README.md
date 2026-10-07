```text
   ___                         __
 /'___\                     __/\ \__
/\ \__/   __   _ __   _ __ /\_\ \ ,_\    __
\ \ ,__\/'__`\/\`'__\/\`'__\/\ \ \ \/  /'__`\
 \ \ \_/\  __/\ \ \/ \ \ \/ \ \ \ \ \_/\  __/
  \ \_\\ \____\\ \_\  \ \_\  \ \_\ \__\ \____\
   \/_/ \/____/ \/_/   \/_/   \/_/\/__/\/____/

   a cpu inference engine for qwen3.5, written from scratch in rust
```

it runs the 0.8b model at 89 tokens per second on a four-core laptop with no gpu. on every model and
bit width i've tested, it's faster than llama.cpp.

![the ferrite desktop app answering a coding question at 55.9 tokens per second](docs/images/ferrite-chat.png)

## why i built this

i wanted to find out how far an ordinary laptop can go. mine is nothing special: an intel i5-1135g7
with four cores, 20 gb of ram, no graphics card and, as i later found out the hard way, a 5400 rpm
hard disk.

the model i picked was qwen3.5. its small versions use a newer design:

```text
            +----------+    +----------+    +----------+    +-----------+
  token --> | deltanet | -> | deltanet | -> | deltanet | -> | attention | --> repeat x6
            +----------+    +----------+    +----------+    +-----------+     (x8 at 4b)
            '---- fixed-size memory, no cache that grows ---'   kv cache, but
                                                                only 1 layer in 4
            +----------+
            | mtp head |  guesses one token further ahead -> free drafts
            +----------+
```

general-purpose engines support this design as one of hundreds. i wanted to see what happens when
the engine is built for one model family and one kind of cpu, so every part of it can be shaped to
fit.

so ferrite does everything itself:

- the tokenizer, safetensors reader and quantizers;
- the avx-512 kernels, thread pool, deltanet recurrence and attention;
- sampling, the chat template and speculative decoding;
- an openai-compatible server and a desktop chat app.

there's no ml framework underneath. the dependency list is short: `serde`, `memmap2`, `half`,
`clap`, `anyhow`, two small unicode crates, and `faer` for the linear algebra in gptq.

## where it landed

these numbers come from one sitting with the engines taking turns, so each one gets the same luck
with memory placement (more on that below):

```text
tokens per second, 4-bit weights, median of 3 rounds (bars scaled per model)

qwen3.5-0.8b  llama.cpp              ################                          36.3
              ferrite                ########################                  54.2
              ferrite + speculation  ########################################  89.1

qwen3.5-2b    llama.cpp              #################                         15.3
              ferrite                ########################                  21.6
              ferrite + speculation  ########################################  35.5

qwen3.5-4b    llama.cpp              ################                           6.6
              ferrite                ######################                     9.0
              ferrite + speculation  ########################################  16.6
```

across the full benchmark (three sizes, 8-, 6- and 4-bit, against llama.cpp, mistral.rs, openvino
and hugging face transformers):

- **prefill** (reading your prompt) is 1.5–4.2× faster than llama.cpp.
- **chat** is 2.4–3.5× faster than llama.cpp at its best setting. ferrite's speculation makes chat
  1.5–2× faster, while llama.cpp's mostly doesn't help on this model.
- **memory:** ferrite uses 11–40% less ram than llama.cpp, and less than half of what openvino uses.
- **quality:**
  - at 4-bit, perplexity is better than llama.cpp's q4_k_m at 0.8b, tied at 2b and 0.3% behind at
    4b.
  - at 8 and 6 bits it's level or slightly ahead.
- **correctness:**
  - at full precision, ferrite's perplexity is 15.3475 where transformers gets 15.3468.
  - speculation leaves the output unchanged: greedy replies are byte-identical with it on or off.

every table is in **[docs/benchmarks.md](docs/benchmarks.md)**.

## how it got here

**1. correct before fast.** the first goal was to produce the same numbers as the reference
implementation in hugging face transformers:

- ferrite's full-precision logits ended up within 0.007 of it;
- the tokenizer produces identical ids on 2.8 million tokens of wikipedia text;
- the chat template matches the model's own template in every case i could think of.

every optimization after that had to keep passing those checks.

**2. then fast.** the weights are repacked so that a single avx-512 vnni instruction computes dot
products for 16 rows at once, with no shuffling of results afterwards. reading the prompt uses all
eight hardware threads. generating uses four, because generation waits on memory, not arithmetic,
and extra threads only eat into the laptop's shared turbo budget.

**3. speculation.** qwen3.5's extra head drafts up to four tokens, and the main model checks them
all in one pass, which costs about the same as generating one.

the hard part was undoing rejected drafts. an attention cache can simply be cut back, but a deltanet
layer's memory is a running sum. so ferrite logs each token's update and replays only the accepted
ones. the result is bit-for-bit what plain decoding would have produced.

**4. the reality check.** then i benchmarked against the other engines. prompt reading and chat
looked great. plain generation didn't: ferrite was only 5–17% ahead of llama.cpp. the laptop had a
few surprises of its own:

- **the first load of the 2b model took 47 minutes.** memory-mapping a file on a 5400 rpm disk turns
  into thousands of random reads. reading it into ram in one go takes about a minute.
- **perplexity didn't line up across engines.** eventually i noticed the test file had windows line
  endings, which llama.cpp quietly converts and ferrite didn't.
- **speed depended on luck.** the same binary generated anywhere from 42.6 to 50.4 tokens per
  second, depending on where windows put the model in memory. 4 gb of the ram is soldered to the
  board and the rest is a 16 gb stick, so only 8 gb of it runs dual-channel. since then, every
  comparison runs the engines in turns within one sitting.

**5. reading fewer bytes.** generation is limited by memory bandwidth, because each new token has to
read every weight once. at 0.8b, 47% of those bytes turned out to be the output layer: qwen's
vocabulary has 248,000 tokens. so it got a shortcut:

```text
  before:  hidden --> [ exact output layer, all 248,320 rows ] ---------------------> logits
                        (47% of the bytes per token at 0.8b)

  now:     hidden --> [ 2-bit copy, all rows ] --> best groups (at most 4,096 rows)
                                                          |
                                                          v
                                  [ exact rows, shortlist only ] -------------------> logits
```

the greedy output didn't change by a single byte. two more changes went in alongside:

- gptq, for better 4-bit weights;
- prompt lookup, which drafts repeated text for free.

together they took the 0.8b model from 40 to 54 tokens per second without speculation, and from 66
to 89 with it.

the details of each step are in [docs/how-it-works.md](docs/how-it-works.md).

## try it

you'll need:

- **a 64-bit x86 cpu.** avx-512 with vnni gives full speed; other x86 cpus work on a slower fallback.
  see [the list](#will-it-run-on-my-machine).
- **rust** (stable).
- **python** with `huggingface_hub`, to download models.

```bash
git clone https://github.com/razancodes/ferrite
cd ferrite
cargo build --release

# download a model and convert it to one 8-bit file (a few minutes)
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B
./target/release/ferrite convert --hf models/Qwen3.5-2B -o models/qwen3.5-2b-q8.ferrite -q q8

# talk to it (/think, /nothink, /reset, /stats, /quit)
./target/release/ferrite chat -m models/qwen3.5-2b-q8.ferrite
```

to build the tuned 6- and 4-bit files that the benchmarks use (this takes longer, since gptq runs
calibration text through the model):

```bash
python scripts/get_data.py
scripts/build_models.sh models/Qwen3.5-2B qwen3.5-2b
```

other ways to use it:

```bash
# one prompt
./target/release/ferrite run -m models/qwen3.5-2b-q8.ferrite -p "Explain DeltaNet in two sentences."

# OpenAI-compatible API, plus a small web chat at http://127.0.0.1:8080
./target/release/ferrite serve -m models/qwen3.5-2b-q8.ferrite
```

on a spinning disk, set `FERRITE_LOAD=heap` so models load with one sequential read.

### the desktop app

there's also a desktop chat app, built with tauri. it's the one in the screenshot, styled after my
portfolio's "dither ledger" look. it has:

- a model dropdown;
- streaming markdown with latex and code highlighting;
- thinking and speculation toggles, plus *max think* and *max speed* presets;
- live tokens per second and acceptance stats under every reply.

on windows, double-click `chat.cmd`. the first build needs winlibs mingw-w64; see
[docs/development.md](docs/development.md#building-on-windows).

## will it run on my machine?

| | cpus |
|---|---|
| **full speed** | intel core 11th gen (tiger lake laptops, rocket lake desktops), 10th-gen "g" laptops (ice lake), xeon from cascade lake on; amd zen 4 and zen 5 (ryzen 7000 desktop, 7040 laptops and newer, ryzen 9000, ryzen ai 300, epyc 9004/9005) |
| **runs, slowly** | any other 64-bit x86 cpu, including intel 12th–14th gen and core ultra, which don't have avx-512 |
| **not yet** | arm: apple silicon, snapdragon, raspberry pi |

memory matters more than cores: generation speed is roughly memory bandwidth divided by model size.
for ram, plan on about 0.9 gb for 0.8b, 1.8 gb for 2b and 3.5 gb for 4b at 4-bit.

i've only tested ferrite on my own laptop, on windows 11. the engine also compiles for linux, but i
haven't run it there yet. full details: [docs/hardware.md](docs/hardware.md).

## which models?

| | models |
|---|---|
| **tested** | [qwen3.5-0.8b](https://huggingface.co/Qwen/Qwen3.5-0.8B), [qwen3.5-2b](https://huggingface.co/Qwen/Qwen3.5-2B), [qwen3.5-4b](https://huggingface.co/Qwen/Qwen3.5-4B) |
| **should work** | qwen3.5-9b; the `-Base` models; qwen3.5, qwen3.6 and qwen3.8 27b (if you have ~16 gb of ram to spare); fine-tunes that keep the architecture |
| **experimental** | dense qwen3 (a code path exists but hasn't been checked against hugging face) |
| **no** | mixture-of-experts models; other families such as gemma, llama and mistral |

ferrite reads everything it needs from the model's `config.json`, so new sizes of the same
architecture just work. more, including how to convert models and what a future qwen release would
need: [docs/models.md](docs/models.md).

## documentation

| | |
|---|---|
| [benchmarks.md](docs/benchmarks.md) | every score: speed, chat, memory, quality, correctness, and how to reproduce them |
| [how-it-works.md](docs/how-it-works.md) | the architecture, weight formats, proxy output layer, gptq, speculation and kernels |
| [usage.md](docs/usage.md) | cli, http api, desktop app and environment variables |
| [models.md](docs/models.md) | compatible models, formats, converting, future releases |
| [hardware.md](docs/hardware.md) | supported cpus, ram, storage and operating systems |
| [development.md](docs/development.md) | building (including on windows), tests, parity checks, repository layout |

## limits, honestly

- **one model family.** ferrite is a specialist. if you need dozens of architectures or a gpu,
  llama.cpp is the better tool.
- **one test machine.** all numbers come from a single laptop. yours will differ, mostly with memory
  bandwidth.
- **x86 only.** the fast path needs avx-512 vnni.
- **text only.** the vision part of the checkpoints is ignored.

## what's next

- **gemma 4 e2b and e4b:** apache-2.0 models with a 262k vocabulary, where the proxy output layer
  should pay off even more.
- **avx2 kernels** for the many recent intel cpus without avx-512.
- **recognizing new qwen releases** by their config instead of their name.
- **weight placement** in the fast, dual-channel part of memory (needs the windows "lock pages in
  memory" privilege).
- **tree drafts:** verifying a second candidate for the first draft position.

## thanks

- the **qwen team**, for open models that ship with an mtp head.
- **llama.cpp**, whose iq4_nl codebook and importance-matrix idea ferrite borrows, and which was the
  bar to beat.
- the papers behind the main ideas:
  - gptq (frantar et al., 2022);
  - quarot (ashkboos et al., 2024) and quip# (tseng et al., 2024) for hadamard rotation;
  - gated deltanet (yang et al., 2024);
  - speculative decoding (leviathan et al., 2023; chen et al., 2023);
  - prompt lookup decoding (saxena, 2023).
- **the libraries:** faer, tauri, katex, marked and highlight.js.
- **the fonts:** cormorant garamond, manrope, ibm plex mono and doto.

## license

ferrite is released under the [mit license](LICENSE).

model weights have their own licenses: qwen3.5 is apache-2.0. the front-end libraries and fonts
bundled in `app/ui/vendor/` keep their own licenses, which are included next to them.
`app/bin/WebView2Loader.dll` is microsoft's redistributable webview2 loader.

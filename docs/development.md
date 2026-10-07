# Development

- [Building](#building)
- [Building on Windows](#building-on-windows)
- [Tests](#tests)
- [Checking against Hugging Face](#checking-against-hugging-face)
- [Benchmarking and profiling](#benchmarking-and-profiling)
- [Repository layout](#repository-layout)

## Building

You need a recent stable Rust (the crates use edition 2024; I build with 1.98).

```bash
cargo build --release              # the ferrite CLI: target/release/ferrite
cargo build --release -p ferrite-app   # the desktop app: target/release/ferrite-chat
```

`.cargo/config.toml` builds with `-C target-cpu=native`, so the binary is tuned for the machine that
builds it. To build for a different machine, remove that flag. The kernels pick their code path at
run time either way.

The release profile uses fat LTO and one codegen unit. A full build takes a few minutes, and longer
on a hard disk.

Python is only needed for downloads, data preparation and the parity and benchmark scripts:

```bash
pip install huggingface_hub pandas pyarrow            # downloads and data
pip install torch transformers numpy                  # parity scripts
python scripts/get_data.py                            # wikitext-2 into data/
```

## Building on Windows

I build with the `stable-x86_64-pc-windows-gnu` toolchain.

**The CLI** needs nothing special.

**The desktop app** pulls in raw-dylib import libraries (through `windows-sys`), and Tauri compiles
a Windows resource file. That needs two 64-bit tools:

- `dlltool`, with a 64-bit `as.exe` next to it;
- `windres`.

rustc and Tauri look both up on `PATH`. If an old 32-bit MinGW (for example `C:\MinGW\bin`) comes
first on `PATH`, the build fails with confusing errors, which is what happened to me. The fix:

1. Install WinLibs MinGW-w64: `winget install BrechtSanders.WinLibs.POSIX.UCRT`.
2. Put its `bin` first on `PATH` when building the app. `chat.cmd` does this for you:

   ```bat
   chat.cmd            :: build if needed, then launch
   chat.cmd rebuild    :: rebuild, then launch
   ```

   By hand, from Git Bash:

   ```bash
   PATH="$LOCALAPPDATA/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH" \
     cargo build --release -p ferrite-app
   ```

3. If rustc still picks up the wrong `dlltool`, point it there in your user-level
   `~/.cargo/config.toml`, which keeps machine-specific paths out of the repo:

   ```toml
   [target.x86_64-pc-windows-gnu]
   rustflags = ["-C", "dlltool=C:/path/to/winlibs/mingw64/bin/dlltool.exe"]
   ```

On the windows-gnu target, `app/build.rs` copies `app/bin/WebView2Loader.dll` (Microsoft's
redistributable WebView2 loader) next to the executable.

The MSVC toolchain is Tauri's usual setup and avoids all of this. I haven't tested ferrite with it.

## Tests

```bash
cargo test --release
```

The unit tests cover:

- **Kernels:** the quantized matmul for every format against a reference.
- **Quantization:** round-trip error, and that the Hadamard rotation is orthogonal.
- **The proxy output layer:** shortlisted logits must be exact, and the argmax must match.
- **GPTQ:** it must beat plain rounding on correlated inputs.
- **The sampler:** top-k against a full sort, and logits left untouched.
- **Everything else:** the thread pool, the tokenizer's pre-split rules and the vectorized `exp`.

They don't need model files.

## Checking against Hugging Face

These scripts compare ferrite with the reference implementation in transformers. The numbers in
[benchmarks.md](benchmarks.md#7-correctness-against-hugging-face) came from them.

```bash
# logits: reference from transformers, ferrite's own, then compare (max |Δ|, KL, top-1)
python scripts/hf_reference.py models/Qwen3.5-0.8B scratch/ref.npy "some text" [--bf16]   # prints the token ids
ferrite dump-logits -m model.ferrite --tokens <ids> -o scratch/fe.bin [--chunk 64]
python scripts/compare_logits.py scratch/ref.npy scratch/fe.bin

# tokenizer and chat template parity
python scripts/tokenizer_parity.py models/Qwen3.5-0.8B model.ferrite data/wiki.test.raw
python scripts/template_parity.py models/Qwen3.5-0.8B model.ferrite

# a quantized file against a reference file, and perplexity
ferrite kld -m model-q4.ferrite --reference model-q8.ferrite -f data/wiki.test.raw
ferrite ppl -m model-q4.ferrite -f data/wiki.test.lf.raw --windows 32
python bench/hf_ppl.py models/Qwen3.5-0.8B --windows 32    # the transformers perplexity to compare with
```

## Benchmarking and profiling

- **Engine comparisons.** `bench/` has the harness that runs ferrite against llama.cpp, mistral.rs,
  OpenVINO GenAI and transformers. [benchmarks.md](benchmarks.md#reproducing) explains how to
  reproduce everything.
- **Per-op timing.** `FERRITE_PROFILE=1` prints a timing table per op after any run.
- **Kernel speed.** `cargo run --release --example kernel_bench` measures VNNI peak throughput and the
  DRAM-streaming speed of every weight format.
- **Memory bandwidth.** `cargo run --release --example bw_bench` measures the machine's bandwidth
  ceiling.
- **Memory placement.** `examples/placement_probe.rs` and `examples/awe_probe.rs` probe where in
  physical memory the fast region is (see [hardware.md](hardware.md#notes-from-my-laptop)).

A/B tests on a laptop need care: alternate the two variants within one sitting and compare medians.
Comparing a run today with a run from yesterday mostly measures memory placement and temperature.

## Repository layout

```
src/
  main.rs          CLI (clap): convert, run, chat, serve, bench, ppl, kld, imatrix, ...
  engine.rs        generation loop, speculative decoding, prompt lookup, state snapshots
  model.rs         the Qwen3.5 forward pass, DeltaNet recurrence, MTP head, proxy output layer
  attn.rs          tiled flash attention (prefill) and flash decoding
  kernels/         AVX-512 VNNI and scalar GEMV/GEMM, vectorized math, Hadamard transform
  quant.rs         weight formats (Q8R, Q6R, Q4R, Q2R) and quantizers
  gptq.rs          GPTQ for 4-bit matrices
  imatrix.rs       activation statistics for importance-weighted quantization
  convert.rs       Hugging Face safetensors → .ferrite, quantization recipes
  format.rs        the .ferrite file format (mmap or heap loading)
  safetensors.rs   safetensors reader
  tokenizer.rs     byte-level BPE with the Qwen pre-tokenizer, written by hand
  chat.rs          the Qwen chat template (thinking, tools, multi-turn)
  sampler.rs       temperature, top-k/p, min-p, penalties
  server.rs        OpenAI-compatible HTTP server with SSE streaming
  stream.rs        splits streamed text into reasoning and answer
  threadpool.rs    spin-waiting pool with adjustable width
  config.rs        reads config.json
  ops.rs, profile.rs
app/               desktop app (Tauri 2): Rust side in src/, front end in ui/
web/index.html     the small chat page served by `ferrite serve`
bench/             head-to-head benchmark harness and results
scripts/           model building, data download, Hugging Face parity scripts
examples/          kernel, bandwidth and memory-placement probes
docs/              these documents
```

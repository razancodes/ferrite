# Hardware

**The short version:**

- **Full speed:** an x86-64 CPU with **AVX-512 and VNNI**, plus enough RAM for the model.
- **Other x86-64 CPUs:** ferrite still runs, but on a portable fallback path that is much slower.
- **ARM (Apple Silicon, Snapdragon, Raspberry Pi):** not supported yet.

I've only tested on one machine, an Intel i5-1135G7 laptop on Windows 11. Everything else on this
page is based on which instructions each CPU has, not on runs.

- [CPUs](#cpus)
- [Memory matters more than cores](#memory-matters-more-than-cores)
- [How much RAM you need](#how-much-ram-you-need)
- [Storage](#storage)
- [Operating systems](#operating-systems)
- [Notes from my laptop](#notes-from-my-laptop)

## CPUs

The fast kernels need four AVX-512 extensions: `avx512f`, `avx512bw`, `avx512vl` and `avx512vnni`.
ferrite checks for them at startup. If any is missing, it falls back to plain scalar code, and you
can see which path is active in the line ferrite prints when it loads a model (`Avx512Vnni` or
`Scalar`) or in the app's Machine panel.

### Intel

| CPU family | Examples | ferrite |
|---|---|---|
| Core 11th gen mobile (Tiger Lake) | i3-1115G4, **i5-1135G7** (tested), i7-1165G7, i7-11800H | ✅ full speed |
| Core 10th gen mobile, "G" models (Ice Lake) | i5-1035G1, i7-1065G7 | ✅ full speed |
| Core 11th gen desktop (Rocket Lake) | i5-11400, i7-11700K, i9-11900K | ✅ full speed |
| Core X 10th gen (Cascade Lake-X) | i9-10900X, i9-10980XE | ✅ full speed |
| Xeon Scalable 2nd gen and newer | Cascade Lake, Ice Lake-SP, Sapphire / Emerald / Granite Rapids; Xeon W-3300, W-2400, W-3400 | ✅ full speed |
| Core 12th–14th gen (Alder Lake, Raptor Lake) | i5-12400, i7-13700H, i9-14900K | 🐢 fallback (AVX-512 is disabled on these) |
| Core Ultra series 1 and 2 (Meteor, Lunar, Arrow Lake) | Ultra 7 155H, Ultra 7 258V, Ultra 9 285K | 🐢 fallback (no AVX-512) |
| Xeon Scalable 1st gen, Core X 7th/9th gen (Skylake) | Xeon Gold 6130, i9-7900X | 🐢 fallback (AVX-512 but no VNNI) |
| Core 10th gen desktop and older (Comet Lake, …) | i5-10400, i7-9700K | 🐢 fallback |

### AMD

| CPU family | Examples | ferrite |
|---|---|---|
| Zen 5 | Ryzen 9000, Ryzen AI 300 (Strix Point), EPYC 9005 | ✅ full speed |
| Zen 4 | Ryzen 7000 desktop, Ryzen 7040/7045 and 8040 mobile, Ryzen 8000G, Threadripper 7000, EPYC 9004 | ✅ full speed* |
| Zen 3 and older | Ryzen 5000 and earlier, Ryzen 7030/7035 mobile | 🐢 fallback |

\*Zen 4 runs 512-bit instructions as two 256-bit halves. The kernels still work and are still fast
there; I just haven't been able to measure how fast.

### Everything else

| | ferrite |
|---|---|
| ARM: Apple Silicon, Snapdragon X, Raspberry Pi | ❌ doesn't build yet (the x86 kernel modules aren't gated by architecture) |
| GPUs | ❌ CPU only, by design |

An AVX2 kernel set, for the many recent Intel CPUs without AVX-512, is the most useful missing
piece. It's on the roadmap.

## Memory matters more than cores

When ferrite generates text without speculation, every new token has to read all of the model's
weights once. So decode speed is roughly:

```
tokens per second ≈ memory bandwidth ÷ bytes read per token
```

Cores barely matter: ferrite decodes with 4 threads, because more don't help.

- **Bandwidth.** My laptop reads memory at 22–29 GB/s. A machine with dual-channel DDR5 (typically
  60–80 GB/s measured) should decode two to three times faster. A single stick of RAM
  (single-channel) roughly halves it.
- **Prefill** (reading the prompt) is different: it's limited by compute, so cores and AVX-512
  throughput matter there.
- **Speculation** gets around the bandwidth limit by producing several tokens per pass over the
  weights. That's why it gives 1.5–2× in chat.

## How much RAM you need

| Model · format | File | Peak RAM in chat (8K context) | Comfortable with |
|---|---|---|---|
| 0.8B · q4 | 0.68 GB | 0.9 GB | 4 GB |
| 2B · q4 | 1.57 GB | 1.8 GB | 8 GB |
| 2B · q8 | 2.24 GB | 2.5 GB | 8 GB |
| 4B · q4 | 3.14 GB | 3.5 GB | 8 GB |
| 4B · q8 | 4.81 GB | ~5.3 GB | 12–16 GB |
| 9B · q4 | ~6.5 GB | ~7.5 GB | 16 GB |
| 27B · q4 | ~15 GB | ~16 GB | 32 GB |

Values with a ~ are estimates; the others were measured. Long contexts add little: only one layer in
four keeps a KV cache, and the DeltaNet layers use the same memory at any length.

## Storage

Model files are 0.7–5 GB. An SSD makes loading quick in any mode. On a hard disk:

- Use `FERRITE_LOAD=heap`, which reads the file sequentially into RAM. The desktop app does this by
  default.
- With the CLI's default memory-mapping, a model that isn't already in the OS file cache arrives
  through random page faults. My first run of the 2B q8 model took 47 minutes to load that way,
  against about a minute with `heap`.

## Operating systems

| OS | Status |
|---|---|
| Windows 11, x64 | ✅ tested: CLI, server and desktop app |
| Windows 10, x64 | probably fine, untested. The desktop app needs the WebView2 runtime, which Windows 11 ships with. |
| Linux, x86-64 | 🧪 compiles (`cargo check --target x86_64-unknown-linux-gnu` is clean) but hasn't been run. `chat.cmd` is Windows-only, and the desktop app would need Tauri's Linux dependencies. |
| macOS | ❌ Apple Silicon doesn't build yet. Intel Macs don't have AVX-512 VNNI (except the 2019 Mac Pro). |

## Notes from my laptop

My test machine has quirks that shaped how I measure things. Similar machines will have them too.

- **Mismatched memory.** It has 4 GB soldered to the board and a 16 GB stick. Intel's "flex mode"
  runs the first 8 GB dual-channel and the remaining 12 GB single-channel, all at DDR4-2667
  because of the soldered part.
  - Where Windows happens to put the model decides decode speed: the same binary and model ran at
    anywhere from 42.6 to 50.4 tok/s over eight launches.
  - That's why every A/B comparison in [benchmarks.md](benchmarks.md) alternates engines within one
    sitting.
  - Physical memory placement could fix this, but on Windows it needs the "Lock pages in memory"
    privilege. Once that's granted, `cargo run --release --example awe_probe` measures bandwidth by
    physical address range, which is the first step towards loading weights into the fast region.
- **A 5400 rpm hard disk.** See [Storage](#storage).
- **Thermals.** Laptop decode speed drifts ±10–15% with temperature and power state. Benchmarks
  were run on mains power, warm, after a discarded warm-up.

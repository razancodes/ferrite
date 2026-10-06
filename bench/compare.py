#!/usr/bin/env python3
"""Head-to-head benchmark of ferrite against other CPU inference engines on the same
Qwen3.5 checkpoints, on this machine. Engines: ferrite, llama.cpp (release b11425,
AVX-512 VNNI backend), OpenVINO GenAI 2026.4 (official Intel IRs), HF transformers.

Suites (results append to bench/runs/results.jsonl, `report` writes bench/RESULTS.md):
  speed    prefill (pp512) and decode (tg128) without speculation, the llama-bench
           definitions; llama.cpp gets its best of 4 and 8 threads per metric.
  chat     end-to-end through each engine's API: 6 fixed prompts (one ~1.5K tokens),
           greedy, up to 256 new tokens, thinking off, speculation (MTP) on and off.
  quality  wikitext-2 test perplexity, 40 windows of 512 tokens, second half scored.

usage: python bench/compare.py speed|chat|quality|report [--sizes 0.8b,2b,4b] [--engines ...]
Run on an idle, plugged-in machine; the CPU load before each run is recorded.
"""
import argparse, ctypes, json, os, re, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FERRITE = ROOT / "target/release/ferrite.exe"
LLAMA = ROOT / "bench/engines/llama.cpp"
OV_PY = ROOT / "bench/engines/ov-venv/Scripts/python.exe"
MISTRALRS = ROOT / "bench/engines/mistralrs/mistralrs.exe"
ORT_PY = ROOT / "bench/engines/ort-venv/Scripts/python.exe"
RUNS = ROOT / "bench/runs"
RESULTS = RUNS / "results.jsonl"
WIKI = ROOT / "data/wiki.test.raw"
# Perplexity text with LF line endings. wiki.test.raw has CRLF; llama-perplexity (like Python's
# text mode) turns CRLF into LF on Windows while ferrite reads bytes as-is, so on the raw file
# the two engines would score different tokens (measured: 9.79 vs 9.75 on window 0).
WIKI_LF = ROOT / "data/wiki.test.lf.raw"

# ferrite reads its weights into RAM sequentially, as llama.cpp and mistral.rs do. Its default
# mmap loading pulls a cold file in through random page faults, which takes tens of minutes
# per model on this laptop's 5400 rpm disk (measured; reported separately from throughput).
os.environ["FERRITE_LOAD"] = "heap"

SIZES = ["0.8b", "2b", "4b"]
# Qwen3.5's chat template with enable_thinking=false, pre-rendered for the pipeline APIs
# (OpenVINO, ORT GenAI) so every engine sees the same tokens as the servers' own templates
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
# bit-width tiers: (label, ferrite format, llama.cpp formats, OpenVINO format)
TIERS = [("8-bit", "q8", ["Q8_0"], "int8"), ("6-bit", "q6", ["Q6_K"], None), ("4-bit", "q4", ["Q4_K_M", "IQ4_NL"], "int4")]
ENGINES = ["ferrite v2", "ferrite", "llama.cpp", "mistral.rs", "openvino", "onnxruntime", "transformers"]
ORT_TIERS = {"4-bit": "int4"}  # ORT GenAI's builder: int4 RTN, block 32, int8 compute (accuracy level 4)


def ferrite_model(s, q):
    return ROOT / f"models/qwen3.5-{s}-{q}.ferrite"


def gguf(s, q):
    return ROOT / f"models/gguf/qwen3.5-{s}-{q}.gguf"


def ov_model(s, q):
    return ROOT / f"models/ov/qwen3.5-{s}-{q}"


def ort_model(s, q):
    return ROOT / f"models/ort/qwen3.5-{s}-{q}"


def size_gb(path):
    p = Path(path)
    if p.is_dir():  # OpenVINO: the language model and its embeddings, not the vision tower; ORT: the ONNX weights
        files = [f for f in p.glob("openvino_*.bin") if "vision" not in f.name] or list(p.glob("*.onnx*"))
        return round(sum(f.stat().st_size for f in files) / 1e9, 2)
    return round(p.stat().st_size / 1e9, 2)


# ───────────────────────── measurement helpers (Windows) ─────────────────────────

def cpu_busy(secs=1.0):
    """System-wide CPU busy % over `secs` (GetSystemTimes; kernel time includes idle)."""
    k = ctypes.windll.kernel32

    def sample():
        idle, kern, user = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
        k.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
        return idle.value, kern.value + user.value

    i0, t0 = sample()
    time.sleep(secs)
    i1, t1 = sample()
    return round(100 * (1 - (i1 - i0) / max(1, t1 - t0)), 1)


class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
        (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                       "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]


def peak_mb(pid):
    """Peak working set of a live process in MB (includes the touched pages of mmapped weights)."""
    k = ctypes.windll.kernel32
    k.OpenProcess.restype = ctypes.c_void_p
    h = k.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED_INFORMATION | VM_READ
    if not h:
        return None
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(h), ctypes.byref(pmc), pmc.cb)
    k.CloseHandle(ctypes.c_void_p(h))
    return round(pmc.PeakWorkingSetSize / 2**20) if ok else None


def run(cmd, timeout=3600, env=None):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
    if p.returncode:
        raise RuntimeError(f"{Path(str(cmd[0])).name} exited {p.returncode}: {(p.stderr or p.stdout)[-1500:]}")
    return p


def emit(rec):
    # FERRITE_ENGINE_LABEL (e.g. "ferrite v2") keeps runs of a newer engine build apart
    if rec.get("engine") == "ferrite":
        rec["engine"] = os.environ.get("FERRITE_ENGINE_LABEL", "ferrite")
    rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    show = {k: v for k, v in rec.items() if not k.endswith("samples") and k != "per_prompt"}
    print(json.dumps(show), flush=True)


def med(xs):
    return round(statistics.median(xs), 2) if xs else None


def cooldown(secs):
    # let the laptop's turbo budget and temperature recover between runs
    time.sleep(secs)


# ───────────────────────── speed: pp512 / tg128, no speculation ─────────────────────────

def speed_ferrite(s, q, reps):
    p = run([FERRITE, "bench", "-m", ferrite_model(s, q), "--no-spec", "-p", 512, "-n", 128, "--reps", reps + 1])
    rows = re.findall(r"rep (\d+): prefill ([\d.]+) tok/s, decode ([\d.]+) tok/s", p.stderr)[1:]  # rep 1 warms the page cache
    return {"pp_samples": [float(r[1]) for r in rows], "tg_samples": [float(r[2]) for r in rows], "threads": "8 prefill / 4 decode"}


def speed_llama(s, q, reps):
    p = run([LLAMA / "llama-bench.exe", "-m", gguf(s, q), "-p", 512, "-n", 128, "-t", "4,8", "-r", reps, "-o", "json"])
    best = {}
    for t in json.loads(p.stdout):
        kind = "pp" if t["n_prompt"] else "tg"
        if kind not in best or statistics.median(t["samples_ts"]) > statistics.median(best[kind][1]):
            best[kind] = (t["n_threads"], t["samples_ts"])
    return {"pp_samples": best["pp"][1], "tg_samples": best["tg"][1], "threads": f"{best['pp'][0]} prefill / {best['tg'][0]} decode"}


def speed_mistral(s, q, reps):
    """mistral.rs on the same GGUF files as llama.cpp; best of 4 and 8 rayon threads per metric.
    Its bench prints mean ± std per test rather than samples."""
    best = {}
    for th in (4, 8):
        p = run([MISTRALRS, "bench", "--cpu", "-f", gguf(s, q), "--prompt-len", 512, "--gen-len", 128, "--iterations", reps,
                 "--warmup", 1], env=dict(os.environ, RAYON_NUM_THREADS=str(th)))
        out = p.stdout + p.stderr
        got = {"pp": float(re.search(r"TTFT \(\d+ input tokens\)\s*\S\s*([\d.]+)", out)[1]),
               "tg": float(re.search(r"Decode \(\d+ tokens @ d\d+\)\s*\S\s*([\d.]+)", out)[1])}
        for k, v in got.items():
            if k not in best or v > best[k][1]:
                best[k] = (th, v)
    return {"pp_samples": [best["pp"][1]], "tg_samples": [best["tg"][1]], "threads": f"{best['pp'][0]} prefill / {best['tg'][0]} decode"}


def speed_ov(s, q, reps):
    p = run([OV_PY, ROOT / "bench/ov_bench.py", "speed", ov_model(s, q), "--reps", reps])
    return json.loads(p.stdout.strip().splitlines()[-1])


def speed_ort(s, q, reps):
    p = run([ORT_PY, ROOT / "bench/ort_bench.py", "speed", ort_model(s, q), "--reps", reps])
    return json.loads(p.stdout.strip().splitlines()[-1])


def speed_hf(s, reps):
    p = run([sys.executable, ROOT / "bench/hf_bench.py", ROOT / f"models/Qwen3.5-{s.upper()}", "--reps", reps])
    return json.loads(p.stdout.strip().splitlines()[-1])


def suite_speed(sizes, engines, reps, cool, flip=False):
    for s in sizes:
        # the generic PyTorch baseline: unquantized f32, so only where it fits in RAM
        if "transformers" in engines and s == "0.8b":
            load = cpu_busy()
            try:
                r = speed_hf(s, reps)
                emit({"suite": "speed", "size": s, "tier": "16/32-bit", "engine": "transformers", "format": "f32",
                      "file_gb": 3.2, "pp": med(r["pp_samples"]), "tg": med(r["tg_samples"]), **r, "cpu_load_before": load})
            except Exception as e:
                emit({"suite": "speed", "size": s, "tier": "16/32-bit", "engine": "transformers", "format": "f32", "error": str(e)[-600:]})
            cooldown(cool)
        for gi, (tier, fq, lqs, oq) in enumerate(TIERS):
            jobs = []
            if "ferrite" in engines:
                jobs.append(("ferrite", fq, ferrite_model(s, fq), lambda s=s, q=fq: speed_ferrite(s, q, reps)))
            if "llama.cpp" in engines:
                jobs += [("llama.cpp", lq, gguf(s, lq), lambda s=s, q=lq: speed_llama(s, q, reps)) for lq in lqs]
            if "mistral.rs" in engines:
                jobs += [("mistral.rs", lq, gguf(s, lq), lambda s=s, q=lq: speed_mistral(s, q, reps)) for lq in lqs]
            if "openvino" in engines and oq:
                jobs.append(("openvino", oq, ov_model(s, oq), lambda s=s, q=oq: speed_ov(s, q, reps)))
            if "onnxruntime" in engines and tier in ORT_TIERS:
                rq = ORT_TIERS[tier]
                jobs.append(("onnxruntime", rq, ort_model(s, rq), lambda s=s, q=rq: speed_ort(s, q, reps)))
            if (gi + flip) % 2:
                jobs.reverse()  # alternate the order so thermal drift does not favour one engine
            for engine, fmt, path, fn in jobs:
                if not Path(path).exists():
                    print(f"skip {engine} {s} {fmt}: {path} missing", flush=True)
                    continue
                load = cpu_busy()
                try:
                    r = fn()
                except Exception as e:  # keep going: one failing engine should not end the suite
                    emit({"suite": "speed", "size": s, "tier": tier, "engine": engine, "format": fmt, "error": str(e)[-600:]})
                    continue
                emit({"suite": "speed", "size": s, "tier": tier, "engine": engine, "format": fmt, "file_gb": size_gb(path),
                      "pp": med(r["pp_samples"]), "tg": med(r["tg_samples"]), **r, "cpu_load_before": load})
                cooldown(cool)


# ───────────────────────── chat: end-to-end through each engine's API ─────────────────────────

def prompts():
    text = WIKI.read_text(encoding="utf-8")
    start = text.index(" = Robert Boulter = ")  # the first article, ~1.5K tokens
    passage = text[start:start + 6500].strip()
    return [
        "Explain how a CPU cache works, with an analogy.",
        "Write a Python function that merges overlapping intervals, with a few tests.",
        "A train leaves at 9:40 and arrives at 13:05. How long is the trip in minutes? Show the steps.",
        "List ten practical tips for writing clear technical documentation.",
        "Translate into French: 'The weather is nice today, so we are going to the park after lunch.' Then explain two grammar points.",
        passage + "\n\nSummarize the passage above in five bullet points.",
    ]


def http(method, url, body=None, timeout=1800):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def chat_server(cmd, port, max_tokens):
    """Start an OpenAI-compatible server, run the prompts, return per-prompt timings + peak memory."""
    log = open(RUNS / f"server-{port}.log", "w", encoding="utf-8")
    proc = subprocess.Popen([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}, see {log.name}")
            try:
                http("GET", base + "/health", timeout=5)
                break
            except Exception:
                if time.time() - t0 > 600:
                    raise RuntimeError("server did not become ready")
                time.sleep(1)
        load_s = round(time.time() - t0, 1)

        def ask(prompt, n):
            return http("POST", base + "/v1/chat/completions", {
                "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "stream": False,
                "temperature": 0, "seed": 0, "presence_penalty": 0, "frequency_penalty": 0,
                "repetition_penalty": 1.0, "repeat_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": False}})

        ask("Say hello.", 16)  # warm-up: page in the weights, compile nothing
        out = []
        for i, p in enumerate(prompts()):
            r = ask(p, max_tokens)
            tm = r["timings"]
            out.append({"prompt": i, "prompt_n": tm.get("prompt_n"), "pp": tm.get("prompt_per_second"),
                        "gen": tm.get("predicted_n"), "tg": tm.get("predicted_per_second"),
                        "draft_n": tm.get("draft_n"), "draft_acc": tm.get("draft_n_accepted"),
                        "text": r["choices"][0]["message"].get("content", "")[:160]})
        return out, peak_mb(proc.pid), load_s
    finally:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def summarize_chat(rows):
    gen = sum(r["gen"] for r in rows)
    secs = sum(r["gen"] / r["tg"] for r in rows if r["tg"])
    drafted = sum(r["draft_n"] or 0 for r in rows)
    long = max(rows, key=lambda r: r["prompt_n"] or 0)
    return {"tg": round(gen / secs, 2) if secs else None, "gen_tokens": gen, "long_prompt_n": long["prompt_n"],
            "long_pp": round(long["pp"], 1) if long["pp"] else None,
            "acceptance": round(sum(r["draft_acc"] or 0 for r in rows) / drafted, 3) if drafted else None}


def suite_chat(sizes, engines, max_tokens, cool, formats, spec_n):
    for s in sizes:
        for tier, fq, lqs, oq in TIERS:
            if formats and fq not in formats:
                continue
            jobs = []
            if "ferrite" in engines:
                for spec in (True, False):
                    cmd = [FERRITE, "serve", "-m", ferrite_model(s, fq), "--port", 8090, "--ctx", 8192] + ([] if spec else ["--no-spec"])
                    jobs.append(("ferrite", fq, spec, ferrite_model(s, fq), cmd, 8090))
            if "llama.cpp" in engines:
                lq = lqs[0]
                for spec in (True, False):
                    cmd = [LLAMA / "llama-server.exe", "-m", gguf(s, lq), "--port", 8091, "-c", 8192, "-np", 1, "-t", 4, "-tb", 8] +(["--spec-type", "draft-mtp", "--spec-draft-n-max", spec_n] if spec else [])
                    jobs.append(("llama.cpp", lq, spec, gguf(s, lq), cmd, 8091))
            for engine, fmt, spec, path, cmd, port in jobs:
                if not Path(path).exists():
                    continue
                load = cpu_busy()
                try:
                    rows, peak, load_s = chat_server(cmd, port, max_tokens)
                except Exception as e:
                    emit({"suite": "chat", "size": s, "tier": tier, "engine": engine, "format": fmt, "spec": spec, "error": str(e)[-600:]})
                    continue
                emit({"suite": "chat", "size": s, "tier": tier, "engine": engine, "format": fmt, "spec": spec,
                      "spec_n": (4 if engine == "ferrite" else spec_n) if spec else 0, **summarize_chat(rows),
                      "peak_mb": peak, "load_s": load_s, "per_prompt": rows, "cpu_load_before": load})
                cooldown(cool)
            # the pipeline-API engines: OpenVINO (its MTP path drafts 2 tokens per its model cards)
            # and ONNX Runtime GenAI (no speculation for this model)
            pipes = []
            if "openvino" in engines and oq and ov_model(s, oq).exists():
                pipes += [("openvino", oq, OV_PY, "bench/ov_bench.py", ov_model(s, oq), sp) for sp in (2, 0)]
            if "onnxruntime" in engines and tier in ORT_TIERS and ort_model(s, ORT_TIERS[tier]).exists():
                pipes.append(("onnxruntime", ORT_TIERS[tier], ORT_PY, "bench/ort_bench.py", ort_model(s, ORT_TIERS[tier]), 0))
            for engine, fmt, py, script, model, spec in pipes:
                load = cpu_busy()
                (RUNS / "prompts.json").write_text(json.dumps(prompts()), encoding="utf-8")
                try:
                    p = run([py, ROOT / script, "chat", model, "--prompts", RUNS / "prompts.json", "--max-tokens", max_tokens]
                            + (["--spec", spec] if spec else []))
                    r = json.loads(p.stdout.strip().splitlines()[-1])
                    emit({"suite": "chat", "size": s, "tier": tier, "engine": engine, "format": fmt, "spec": bool(spec), "spec_n": spec,
                          **summarize_chat(r["rows"]), "peak_mb": r["peak_mb"], "load_s": r["load_s"], "per_prompt": r["rows"],
                          "cpu_load_before": load})
                except Exception as e:
                    emit({"suite": "chat", "size": s, "tier": tier, "engine": engine, "format": fmt, "spec": bool(spec), "error": str(e)[-600:]})
                cooldown(cool)


# ───────────────────────── quality: wikitext-2 perplexity ─────────────────────────

def suite_quality(sizes, engines, windows, formats=()):
    for s in sizes:
        jobs = []
        for tier, fq, lqs, _ in TIERS:
            if formats and fq not in formats:
                continue
            if "ferrite" in engines:
                jobs.append(("ferrite", tier, fq, ferrite_model(s, fq)))
            if "llama.cpp" in engines:
                jobs += [("llama.cpp", tier, lq, gguf(s, lq)) for lq in lqs]
        # unquantized references where they fit in memory
        if "ferrite" in engines:
            jobs.append(("ferrite", "16/32-bit", "f32", ferrite_model(s, "f32")))
        if "llama.cpp" in engines:
            jobs.append(("llama.cpp", "16/32-bit", "bf16", gguf(s, "bf16")))
        for engine, tier, fmt, path in jobs:
            if not Path(path).exists() or (fmt in ("f32", "bf16") and s != "0.8b"):
                continue
            try:
                if engine == "ferrite":
                    p = run([FERRITE, "ppl", "-m", path, "-f", WIKI_LF, "--window", 512, "--windows", windows])
                    m = re.search(r"perplexity: ([\d.]+) over (\d+) tokens", p.stdout)
                    ppl, n = float(m[1]), int(m[2])
                else:
                    p = run([LLAMA / "llama-perplexity.exe", "-m", path, "-f", WIKI_LF, "-c", 512, "--chunks", windows, "-t", 8, "-b", 512])
                    m = re.search(r"Final estimate: PPL = ([\d.]+)", p.stdout + p.stderr)
                    ppl, n = float(m[1]), windows * 255
            except Exception as e:
                emit({"suite": "quality", "size": s, "tier": tier, "engine": engine, "format": fmt, "error": str(e)[-600:]})
                continue
            emit({"suite": "quality", "size": s, "tier": tier, "engine": engine, "format": fmt, "file_gb": size_gb(path),
                  "ppl": round(ppl, 4), "scored_tokens": n, "text": WIKI_LF.name})


# ───────────────────────── report ─────────────────────────

def latest(rows):
    """One record per (suite, size, engine, format, spec): speed runs from several passes are
    pooled (median over all their samples, with the spread kept); otherwise the newest wins."""
    out = {}
    for r in rows:
        if "error" in r:
            continue
        # ferrite speed runs before the switch to FERRITE_LOAD=heap used mmap; pass 2 re-measures
        # every configuration, so keep the method identical across sizes
        if r["suite"] == "speed" and r["engine"] == "ferrite" and r["time"] < "2026-10-06 01:40:00":
            continue
        # OpenVINO chat runs before ov_bench.py switched to wall-clock timing used its own
        # TTFT/TPOT metrics, which are erratic under its MTP pipeline (2B int8: 0.09 tok/s)
        if r["suite"] == "chat" and r["engine"] == "openvino" and r["time"] < "2026-10-06 08:20:00":
            continue
        # ferrite's quality runs on the CRLF file scored CRLF tokens (see WIKI_LF); llama-perplexity
        # converts CRLF itself, so its results are identical on both files (verified on all 0.8B formats)
        if r["suite"] == "quality" and r["engine"] == "ferrite" and r.get("text") != WIKI_LF.name:
            continue
        k = (r["suite"], r["size"], r["engine"], r["format"], r.get("spec"), r.get("spec_n"))
        if r["suite"] == "speed" and k in out:
            prev = out[k]
            r = {**r, "pp_samples": prev["pp_samples"] + r["pp_samples"], "tg_samples": prev["tg_samples"] + r["tg_samples"],
                 "passes": prev.get("passes", 1) + 1}
            r["pp"], r["tg"] = med(r["pp_samples"]), med(r["tg_samples"])
        elif r["suite"] == "chat" and k in out:  # repeated chat runs: median decode rate
            runs = out[k].get("tg_runs", [out[k]["tg"]]) + [r["tg"]]
            r = {**r, "tg_runs": runs, "tg": med(runs)}
        out[k] = r
    return list(out.values())


def pct(a, b):
    return f"{(a / b - 1) * 100:+.0f}%" if a and b else "—"


def report():
    rows = latest(json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip())
    md = ["# ferrite vs other CPU inference engines", "",
          "Generated by `python bench/compare.py report` from `bench/runs/results.jsonl`.", "",
          "Machine: i5-1135G7 (4C/8T, AVX-512 VNNI), 20 GB DDR4-2667 in flex mode, 5400 rpm HDD, Windows 11, on mains power. "
          "Engines: ferrite (this repo), llama.cpp b11425 (win-cpu-x64, `icelake` backend), mistral.rs 0.9.4 (CPU), "
          "OpenVINO GenAI 2026.4.1 (Intel's official Qwen3.5 IRs), HF transformers 5.15 / torch 2.9 (f32). "
          "Same Hugging Face checkpoints throughout; llama.cpp/mistral.rs files converted here with the same calibration "
          "text and amount as ferrite's (`bench/prepare.sh`).", "",
          "Method notes: throughput is measured warm (a discarded warm-up first); ferrite loads with `FERRITE_LOAD=heap` "
          "(a sequential read into RAM, as llama.cpp and mistral.rs do; cold mmap page faults take tens of minutes on this "
          "disk); llama.cpp and mistral.rs get the best of 4 and 8 threads per metric; speed pools two passes run in opposite "
          "engine order; decode varies about ±10-15% run to run here (thermals, flex-mode memory placement). Perplexity uses "
          "`data/wiki.test.lf.raw` (LF line endings) because llama-perplexity converts CRLF itself; HF transformers f32 gives "
          "15.3468 on the 0.8B windows. mistral.rs cannot load IQ4_NL tensors; ONNX Runtime GenAI 0.17.1 builds Qwen3.5 but "
          "fails to load it on CPU (\"GetMemoryInfo for CPU should not be used\").", ""]

    def pick(suite, s, engine, tier, spec=None):
        c = [r for r in rows if r["suite"] == suite and r["size"] == s and r["engine"] == engine and r["tier"] == tier
             and (spec is None or r.get("spec") == spec)]
        return c

    md += ["## Speed without speculation (pp512 / tg128, tok/s, median)", "",
           "| Model | Bits | Engine · format | Size GB | Prefill | Decode (sample range) | Passes | newest ferrite Δ prefill | newest ferrite Δ decode |",
           "|---|---|---|---|---|---|---|---|---|"]
    for s in SIZES:
        for tier in [t[0] for t in TIERS] + ["16/32-bit"]:
            # deltas against the newest ferrite build measured for this cell
            f = pick("speed", s, "ferrite v2", tier) or pick("speed", s, "ferrite", tier)
            f = f[0] if f else None
            for engine in ENGINES:
                for r in pick("speed", s, engine, tier):
                    dp = "" if not f or r is f else pct(f["pp"], r["pp"])
                    dt = "" if not f or r is f else pct(f["tg"], r["tg"])
                    lo, hi = min(r["tg_samples"]), max(r["tg_samples"])
                    md.append(f"| {s} | {tier} | {engine} · {r['format']} | {r['file_gb']} | {r['pp']} | {r['tg']} ({lo:.1f}–{hi:.1f}) "
                              f"| {r.get('passes', 1)} | {dp} | {dt} |")
    md += ["", "## End-to-end chat (6 prompts, greedy, ≤256 tokens, thinking off)", "",
           "| Model | Bits | Engine · format | Speculation | Decode tok/s | Prefill tok/s (long prompt) | Draft acceptance | Peak RAM MB |",
           "|---|---|---|---|---|---|---|---|"]
    for s in SIZES:
        for tier, *_ in TIERS:
            for engine in ENGINES:
                for r in sorted(pick("chat", s, engine, tier), key=lambda r: not r.get("spec")):
                    acc = f"{r['acceptance'] * 100:.0f}%" if r.get("acceptance") else "—"
                    sp = f"MTP ≤{r['spec_n']}" if r.get("spec") else "off"
                    runs = len(r.get("tg_runs", [r["tg"]]))
                    md.append(f"| {s} | {tier} | {engine} · {r['format']} | {sp} | {r['tg']}{f' (median of {runs})' if runs > 1 else ''} "
                              f"| {r['long_pp']} | {acc} | {r['peak_mb']} |")
    md += ["", "Greedy output identical with speculation on and off (first 160 characters of all 6 replies):", ""]
    for s in SIZES:
        for tier, *_ in TIERS:
            for engine in ENGINES:
                on = pick("chat", s, engine, tier, True)
                off = pick("chat", s, engine, tier, False)
                if on and off:
                    same = sum(x["text"] == y["text"] for x, y in zip(on[0]["per_prompt"], off[0]["per_prompt"]))
                    md.append(f"* {s} {tier} {engine}: {same}/{len(on[0]['per_prompt'])} identical")
    md += ["", "## Quality (wikitext-2 test perplexity, 512-token windows, second half scored, lower is better)", "",
           "| Model | Bits | Engine · format | Size GB | Perplexity |", "|---|---|---|---|---|"]
    for s in SIZES:
        for r in sorted((r for r in rows if r["suite"] == "quality" and r["size"] == s), key=lambda r: (-r["file_gb"], r["engine"])):
            md.append(f"| {s} | {r['tier']} | {r['engine']} · {r['format']} | {r['file_gb']} | {r['ppl']} |")
    (ROOT / "bench/RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=["speed", "chat", "quality", "report"])
    ap.add_argument("--sizes", default=",".join(SIZES))
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--formats", default="", help="chat: ferrite formats to run, e.g. q6,q4")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cool", type=int, default=10, help="seconds of rest between runs")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--spec-n", type=int, default=3, help="llama.cpp --spec-draft-n-max")
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--flip", action="store_true", help="speed: run each group in the opposite engine order (second pass)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the report has Δ and ≤; the Windows console default is cp1252
    RUNS.mkdir(parents=True, exist_ok=True)
    sizes, engines = a.sizes.split(","), a.engines.split(",")
    formats = [f for f in a.formats.split(",") if f]
    if a.suite == "speed":
        suite_speed(sizes, engines, a.reps, a.cool, a.flip)
    elif a.suite == "chat":
        suite_chat(sizes, engines, a.max_tokens, a.cool, formats, a.spec_n)
    elif a.suite == "quality":
        suite_quality(sizes, engines, a.windows, formats)
    else:
        report()


if __name__ == "__main__":
    sys.exit(main())

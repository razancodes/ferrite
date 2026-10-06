"""ONNX Runtime GenAI side of bench/compare.py (runs in bench/engines/ort-venv).

  speed <model_dir> --reps N        pp512 / tg128, greedy, like llama-bench
  chat  <model_dir> --prompts F     the chat prompts, greedy, thinking off

The model is built from the local checkpoint with ORT GenAI's model builder:
  python -m onnxruntime_genai.models.builder -m models/Qwen3.5-0.8B -o models/ort/qwen3.5-0.8b-int4 \
      -p int4 -e cpu --extra_options int4_accuracy_level=4 exclude_embeds=False
Prints one JSON line.
"""
import argparse, json, os, sys, time
from pathlib import Path

import onnxruntime_genai as og

sys.path.insert(0, str(Path(__file__).parent))
from compare import TEMPLATE, peak_mb  # noqa: E402


def run(model, ids, new_tokens, fixed):
    """Prefill `ids`, then decode greedily; returns (prefill s, decode s, generated ids)."""
    p = og.GeneratorParams(model)
    p.set_search_options(do_sample=False, max_length=len(ids) + new_tokens, **({"min_length": len(ids) + new_tokens} if fixed else {}))
    g = og.Generator(model, p)
    t0 = time.perf_counter()
    g.append_tokens(ids)  # prefill (computes the first token's logits)
    t1 = time.perf_counter()
    out = []
    while not g.is_done() and len(out) < new_tokens:
        g.generate_next_token()
        out.append(int(g.get_next_tokens()[0]))
    return t1 - t0, time.perf_counter() - t1, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["speed", "chat"])
    ap.add_argument("model")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prompts")
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    t0 = time.time()
    model = og.Model(a.model)
    tok = og.Tokenizer(model)
    load_s = round(time.time() - t0, 1)
    if a.mode == "speed":
        text = (Path(__file__).resolve().parent.parent / "data/wiki.test.raw").read_text(encoding="utf-8")[:20000]
        long_ids = list(tok.encode(text))[:512]
        short = list(tok.encode(TEMPLATE.format("Write a detailed essay about the history of computing, from the abacus to modern AI.")))
        run(model, short, 8, True)  # warm-up
        pp, tg = [], []
        for _ in range(a.reps):
            t_pp, _, _ = run(model, long_ids, 1, True)
            pp.append(len(long_ids) / t_pp)
            _, t_tg, out = run(model, short, 129, True)
            tg.append((len(out) - 1) / t_tg)  # the first token comes from the prefill logits
        print(json.dumps({"pp_samples": [round(x, 2) for x in pp], "tg_samples": [round(x, 2) for x in tg], "threads": "ORT default"}))
    else:
        rows = []
        run(model, list(tok.encode(TEMPLATE.format("Say hello."))), 16, False)  # warm-up
        for i, prompt in enumerate(json.loads(Path(a.prompts).read_text(encoding="utf-8"))):
            ids = list(tok.encode(TEMPLATE.format(prompt)))
            t_pp, t_dec, out = run(model, ids, a.max_tokens, False)
            rows.append({"prompt": i, "prompt_n": len(ids), "pp": len(ids) / t_pp, "gen": len(out),
                         "tg": (len(out) - 1) / t_dec if len(out) > 1 else None, "draft_n": None, "draft_acc": None,
                         "text": tok.decode(out)[:160]})
        print(json.dumps({"rows": rows, "peak_mb": peak_mb(os.getpid()), "load_s": load_s}))


if __name__ == "__main__":
    main()

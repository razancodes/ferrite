"""Reference perplexity from HF transformers on exactly the windows bench/compare.py scores:
wikitext-2 test, 512-token windows from token 0, second half scored, targets inside the window
(255 per window) -- the llama-perplexity definition. Settles which engine's arithmetic matches
the reference implementation when engines disagree on unquantized or 8-bit models.

usage: python bench/hf_ppl.py models/Qwen3.5-0.8B [--dtype float32|bfloat16] [--windows 32]
"""
import argparse, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from hf_reference import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--windows", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok((ROOT / "data/wiki.test.lf.raw").read_text(encoding="utf-8"))["input_ids"]
    model = load_model(a.model, getattr(torch, a.dtype))
    nll, n, t0 = 0.0, 0, time.time()
    with torch.inference_mode():
        for w in range(a.windows):
            seg = torch.tensor([ids[w * 512:(w + 1) * 512]])
            logits = model(input_ids=seg).logits[0].float()
            lp = torch.log_softmax(logits[256:511], dim=-1)  # positions 256..510 predict 257..511
            nll -= lp.gather(1, seg[0, 257:512, None]).sum().item()
            n += 255
    print(json.dumps({"model": a.model, "dtype": a.dtype, "windows": a.windows, "ppl": round(math.exp(nll / n), 4),
                      "scored_tokens": n, "secs": round(time.time() - t0)}))


if __name__ == "__main__":
    main()

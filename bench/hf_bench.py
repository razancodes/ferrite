"""HF transformers (PyTorch eager, CPU) baseline for bench/compare.py: pp512 / tg128 on the
unquantized checkpoint, greedy, no speculation. Prints one JSON line.

usage: python bench/hf_bench.py models/Qwen3.5-0.8B [--reps 3] [--threads 4]
"""
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from hf_reference import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_model(a.model, torch.float32)
    text = (Path(__file__).resolve().parent.parent / "data/wiki.test.raw").read_text(encoding="utf-8")[:20000]
    long_ids = torch.tensor([tok(text)["input_ids"][:512]])
    short = tok.apply_chat_template([{"role": "user", "content": "Write a detailed essay about the history of computing, from the abacus to modern AI."}],
                                    add_generation_prompt=True, tokenize=True, return_dict=True)
    short_ids = torch.tensor([short["input_ids"]])
    pp, tg = [], []
    with torch.inference_mode():
        model.generate(short_ids, max_new_tokens=4, min_new_tokens=4, do_sample=False)  # warm-up
        for _ in range(a.reps):
            t0 = time.perf_counter()
            model(input_ids=long_ids, logits_to_keep=1)
            pp.append(512 / (time.perf_counter() - t0))
            # decode rate excluding the prompt: time 129 tokens minus time for 1
            t0 = time.perf_counter()
            model.generate(short_ids, max_new_tokens=1, min_new_tokens=1, do_sample=False)
            t1 = time.perf_counter()
            model.generate(short_ids, max_new_tokens=129, min_new_tokens=129, do_sample=False)
            t2 = time.perf_counter()
            tg.append(128 / ((t2 - t1) - (t1 - t0)))
    print(json.dumps({"pp_samples": [round(x, 2) for x in pp], "tg_samples": [round(x, 2) for x in tg], "threads": f"torch {a.threads}"}))


if __name__ == "__main__":
    main()

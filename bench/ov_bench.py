"""OpenVINO GenAI side of bench/compare.py (runs in bench/engines/ov-venv).

  speed <model_dir> --reps N        pp512 / tg128 without speculation, like llama-bench
  chat  <model_dir> --prompts F     the chat prompts, greedy, thinking off
        [--spec N]                  with the model's MTP head drafting N tokens

Prints one JSON line. The official OpenVINO IRs of Qwen3.5 are vision-language
packages, so they load through VLMPipeline; prompts are text only.
"""
import argparse, json, sys, time
from pathlib import Path

import openvino_genai as ov_genai

sys.path.insert(0, str(Path(__file__).parent))
from compare import TEMPLATE, peak_mb  # noqa: E402  (stdlib-only helpers)
import os  # noqa: E402


def config(max_new, fixed, spec=0):
    c = ov_genai.GenerationConfig()
    if spec:
        c.num_assistant_tokens = spec
    c.max_new_tokens = max_new
    c.do_sample = False
    c.apply_chat_template = False
    if fixed:
        c.min_new_tokens = max_new
        c.ignore_eos = True
    return c


def gen(pipe, prompt, cfg):
    """Returns (result, input tokens, generated tokens, TTFT ms, ms per output token after the first).
    Timed on the wall clock with a streamer, as a user sees it: OpenVINO's own TTFT/TPOT metrics
    are erratic under its speculative (MTP) pipeline (0.02-11 tok/s across one run's prompts)."""
    stamps = []

    def streamer(_text):
        stamps.append(time.perf_counter())
        return ov_genai.StreamingStatus.RUNNING

    t0 = time.perf_counter()
    r = pipe.generate(prompt, generation_config=cfg, streamer=streamer)
    t_end = time.perf_counter()
    pm = r.perf_metrics
    n_out = pm.get_num_generated_tokens()
    first = stamps[0] if stamps else t_end
    ttft = (first - t0) * 1000
    tpot = (t_end - first) * 1000 / max(1, n_out - 1)
    return r, pm.get_num_input_tokens(), n_out, ttft, tpot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["speed", "chat"])
    ap.add_argument("model")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prompts")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--spec", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    if a.spec:
        # MTP speculative decoding, as documented on the OpenVINO model cards
        # (greedy only, prefix caching off)
        sc = ov_genai.SchedulerConfig()
        sc.enable_prefix_caching = False
        pipe = ov_genai.VLMPipeline(a.model, "CPU", draft_model=ov_genai.draft_model(a.model, "CPU"), scheduler_config=sc)
    else:
        pipe = ov_genai.VLMPipeline(a.model, "CPU")
    load_s = round(time.time() - t0, 1)
    tok = pipe.get_tokenizer()

    if a.mode == "speed":
        # a 512-token prompt cut from real text, and a short prompt for decode
        text = (Path(__file__).resolve().parent.parent / "data/wiki.test.raw").read_text(encoding="utf-8")[:20000]
        ids = tok.encode(text, add_special_tokens=False).input_ids.data[0][:512]
        long_prompt = tok.decode(ids.tolist())
        short = TEMPLATE.format("Write a detailed essay about the history of computing, from the abacus to modern AI.")
        gen(pipe, short, config(8, True))  # warm-up
        pp, tg = [], []
        for _ in range(a.reps):
            _, n_in, _, ttft, _ = gen(pipe, long_prompt, config(1, True))
            pp.append(n_in / ttft * 1000)
            _, _, n_out, _, tpot = gen(pipe, short, config(128, True))
            tg.append(1000 / tpot)
        print(json.dumps({"pp_samples": [round(x, 2) for x in pp], "tg_samples": [round(x, 2) for x in tg],
                          "threads": "OpenVINO default", "pp_tokens": int(n_in)}))
    else:
        rows = []
        gen(pipe, TEMPLATE.format("Say hello."), config(16, False, a.spec))  # warm-up
        for i, p in enumerate(json.loads(Path(a.prompts).read_text(encoding="utf-8"))):
            r, n_in, n_out, ttft, tpot = gen(pipe, TEMPLATE.format(p), config(a.max_tokens, False, a.spec))
            rows.append({"prompt": i, "prompt_n": int(n_in), "pp": n_in / ttft * 1000, "gen": int(n_out),
                         "tg": 1000 / tpot if n_out > 1 else None, "draft_n": None, "draft_acc": None,
                         "text": str(r.texts[0])[:160]})
        print(json.dumps({"rows": rows, "peak_mb": peak_mb(os.getpid()), "load_s": load_s}))


if __name__ == "__main__":
    main()

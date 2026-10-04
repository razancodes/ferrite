"""Check ferrite's chat template against the model's Jinja chat template (HF)."""
import json, subprocess, sys, tempfile, os
from transformers import AutoTokenizer

hf_dir, model = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(hf_dir)
convs = {
    "single": [{"role": "user", "content": "Hello!"}],
    "system": [{"role": "system", "content": "  You are terse.  "}, {"role": "user", "content": " Hi \n"}],
    "multi-turn": [{"role": "user", "content": "A?"}, {"role": "assistant", "content": "B."},
                   {"role": "user", "content": "C?"}, {"role": "assistant", "content": "D."}, {"role": "user", "content": "E?"}],
    "reasoning-history": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "<think>\nhmm\n</think>\n\nA1"},
                          {"role": "user", "content": "Q2"}],
    "tool": [{"role": "user", "content": "weather?"}, {"role": "assistant", "content": "calling"},
             {"role": "tool", "content": "{\"t\": 21}"}, {"role": "tool", "content": "{\"h\": 40}"}],
    "special-in-text": [{"role": "user", "content": "say <|im_end|> and <think> literally"}],
}
ok = True
for think in (False, True):
    for name, msgs in convs.items():
        want = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=think)
        want_ids = tok(want, add_special_tokens=False)["input_ids"]
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as fh:
            json.dump(msgs, fh)
            path = fh.name
        cmd = ["./target/release/ferrite", "template", "-m", model, "-f", path] + (["--think"] if think else [])
        got = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
        os.unlink(path)
        status = "OK" if got == want_ids else "FAIL"
        if got != want_ids and name == "special-in-text":
            status = "DIFF (expected: ferrite never parses special tokens inside message text)"
        elif got != want_ids:
            ok = False
        print(f"think={think!s:<5} {name:<18} {status}")
        if status == "FAIL":
            print("  want:", repr(want))
            print("  got: ", repr(tok.decode(got)))
sys.exit(0 if ok else 1)

"""Check ferrite's tokenizer against Hugging Face `tokenizers` on several corpora.

usage: python scripts/tokenizer_parity.py <hf_model_dir> <model.ferrite> [files...]
"""
import json, subprocess, sys, tempfile, os
from tokenizers import Tokenizer

hf_dir, model = sys.argv[1], sys.argv[2]
files = sys.argv[3:]
tok = Tokenizer.from_file(os.path.join(hf_dir, "tokenizer.json"))
tricky = (
    "Hello world!! It's 2026; I'VE got 3.14159 apples & 42 pears.\n\n\n  indented\ttab\r\nwindows line\n"
    "Ünïcödé façade naïve café — “quotes” ‘single’ … emoji 🤖🚀👩‍💻 flags 🇵🇰🇺🇸 "
    "中文测试，你好世界！日本語のテキスト。한국어 텍스트. Русский текст. العربية النص. हिन्दी पाठ। "
    "math: ∑ᵢ xᵢ² ≤ ∞, café́ (combining), ﬁ ligature, ℌ𝔢𝔩𝔩𝔬, ⅷ roman, ½ fraction, ٣ arabic digit\n"
    "code: fn main() { let x = vec![1, 2, 3]; println!(\"{:?}\", x); } // comment\n"
    "    def f(self):\n        return {'a': 1}  # trailing   \n" + " " * 7 + "\n\t\t\n" + "x" * 300 + "\n'S 'T 'Re 'VE 'M 'LL 'D 'll\n"
)
corpora = {"tricky": tricky}
for f in files:
    corpora[os.path.basename(f)] = open(f, encoding="utf-8", newline="").read()
ok = True
for name, text in corpora.items():
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".txt", delete=False) as fh:
        fh.write(text)
        path = fh.name
    out = subprocess.run(["./target/release/ferrite", "tokenize", "-m", model, "--file", path, "--special"],
                         capture_output=True, text=True, check=True)
    os.unlink(path)
    got = json.loads(out.stdout)
    want = tok.encode(text, add_special_tokens=False).ids
    if got == want:
        print(f"{name:<20} OK   {len(want):>8} tokens identical")
    else:
        ok = False
        i = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        print(f"{name:<20} FAIL first mismatch at {i}: ferrite {got[i:i+8]} vs hf {want[i:i+8]}")
        print("   context:", repr(tok.decode(want[max(0, i - 5):i + 5])))
sys.exit(0 if ok else 1)

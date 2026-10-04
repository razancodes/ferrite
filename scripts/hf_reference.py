"""Reference logits from Hugging Face transformers (float32, CPU).

usage: python scripts/hf_reference.py <hf_model_dir> <out.npy> "<text>" [--chat]
Prints the token ids (comma separated) so the same ids can be fed to
`ferrite dump-logits`.
"""
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig


def load_model(model_dir):
    cfg = AutoConfig.from_pretrained(model_dir)
    if cfg.model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration as M
    else:
        from transformers import AutoModelForCausalLM as M
    m = M.from_pretrained(model_dir, dtype=torch.float32)
    m.eval()
    return m


def main():
    model_dir, out, text = sys.argv[1], sys.argv[2], sys.argv[3]
    chat = "--chat" in sys.argv
    tok = AutoTokenizer.from_pretrained(model_dir)
    if chat:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
    else:
        ids = tok(text)["input_ids"]
    print(",".join(map(str, ids)))
    model = load_model(model_dir)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0].float().numpy()
    np.save(out, logits)
    print(f"saved {logits.shape}", file=sys.stderr)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()

"""Compare ferrite logits (raw f32 dump) against reference logits (.npy).

usage: python scripts/compare_logits.py ref.npy ferrite.bin
"""
import sys
import numpy as np


def log_softmax(x):
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


ref = np.load(sys.argv[1]).astype(np.float64)
n, v = ref.shape
got = np.fromfile(sys.argv[2], dtype=np.float32).astype(np.float64)
gv = got.size // n
got = got.reshape(n, gv)[:, :v]
# padded vocab rows are masked to -inf in ferrite; mask the same rows in the reference
mask = ~np.isfinite(got)
ref_m = np.where(mask, -1e30, ref)
got_m = np.where(mask, -1e30, got)
finite = ~mask
diff = np.abs(ref_m - got_m)[finite]
lp, lq = log_softmax(ref_m), log_softmax(got_m)
p = np.exp(lp)
kl = np.where(p > 0, p * (lp - lq), 0).sum(-1)
top_ref, top_got = ref_m.argmax(-1), got_m.argmax(-1)
print(f"positions: {n}  vocab: {v}")
print(f"max |logit diff|: {diff.max():.5f}   mean: {diff.mean():.6f}")
print(f"top-1 agreement: {(top_ref == top_got).mean() * 100:.1f}%")
print(f"KL(ref || ferrite): mean {kl.mean():.6f}  max {kl.max():.6f}")

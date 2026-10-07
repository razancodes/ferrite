"""Download wikitext-2 (raw) into data/, byte-identical to the files the benchmarks used.

  data/wiki.train.raw     calibration text (imatrix, GPTQ, draft vocabulary)   CRLF line endings
  data/wiki.test.raw      parity tests                                        CRLF line endings
  data/wiki.test.lf.raw   perplexity in bench/compare.py                       LF line endings

The CRLF files are what a plain Python text-mode write produced on Windows; they are kept that
way so calibration and test results reproduce exactly. llama-perplexity converts CRLF to LF
itself, which is why perplexity comparisons use the LF file.

usage: python scripts/get_data.py            (needs: pip install huggingface_hub pandas pyarrow)
"""
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

out = Path(__file__).resolve().parent.parent / "data"
out.mkdir(exist_ok=True)
for split in ("train", "test"):
    p = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset")
    txt = "".join(pd.read_parquet(p)["text"].tolist())
    (out / f"wiki.{split}.raw").write_text(txt, encoding="utf-8", newline="\r\n")
    if split == "test":
        (out / "wiki.test.lf.raw").write_text(txt, encoding="utf-8", newline="\n")
    print(f"wiki.{split}.raw: {len(txt):,} characters")

#!/usr/bin/env python3
"""Extract Qwen2 tokenizer vocab + merges from GGUF to sim/ directory."""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GGUF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "models", "qwen2-0_5b-instruct-q4_k_m.gguf")
OUTDIR = os.path.dirname(os.path.abspath(__file__))

from gguf import GGUFReader

r = GGUFReader(GGUF_PATH)
tf = r.fields["tokenizer.ggml.tokens"]
mf = r.fields["tokenizer.ggml.merges"]
parts = tf.parts

pos = 5
tokens = []
while pos + 1 < len(parts) and len(tokens) < 151936:
    slen = int(parts[pos][0]); pos += 1
    tokens.append(bytes(parts[pos]).decode("utf-8", errors="replace")); pos += 1

vocab = {t: i for i, t in enumerate(tokens)}
with open(os.path.join(OUTDIR, "vocab.json"), "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

mparts = mf.parts
mpos = 5
merges = []
while mpos + 1 < len(mparts) and len(merges) < 151387:
    slen = int(mparts[mpos][0]); mpos += 1
    merges.append(bytes(mparts[mpos]).decode("utf-8", errors="replace")); mpos += 1

with open(os.path.join(OUTDIR, "merges.txt"), "w", encoding="utf-8") as f:
    f.write("#version: 0.2\n")
    for m in merges:
        f.write(m + "\n")

print(f"Extracted {len(tokens)} tokens, {len(merges)} merges to {OUTDIR}")

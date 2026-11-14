#!/usr/bin/env python3
"""The fp32 oracle for the depth decoder (the audio OUTPUT side's G1): seeded inputs through the
reference's Lfm3OmniDepthDecoder, the logits dumped for tools/lfm3_tts_check.cpp to compare.

Usage: ref_depth.py CHECKPOINT_DIR OUT_DIR   (torch lives in the anaconda python)
"""
import json
import os
import struct
import sys

import numpy as np
import torch
import transformers
from transformers.modeling_utils import ROPE_INIT_FUNCTIONS

def _axial_stub(config, **k):  # transformers 5.15's _init_weights doesn't know the vision tower's rope
    t = torch.arange(0, 16, dtype=torch.float)
    return 1.0 / (1e6 ** (t / 16)), 1.0

ROPE_INIT_FUNCTIONS.setdefault("axial", _axial_stub)

ref, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

m = transformers.AutoModelForImageTextToText.from_pretrained(ref, trust_remote_code=True, torch_dtype=torch.float32)
m.eval()
dec = m.model.depth_decoder
print('depth decoder:', dec.depth_linear.weight.shape, flush=True)

records = []
with torch.no_grad():
    torch.manual_seed(0)
    hidden = torch.randn(2048)
    codes = torch.randint(0, 2017, (1, 12))
    masked = torch.ones(1, 12, dtype=torch.bool)
    lg = dec(hidden[None], codes, masked)[0]
    records.append((hidden, codes, masked, lg))
    hidden2 = torch.randn(2048)
    codes2 = torch.randint(0, 2017, (1, 12))
    codes2[0, :6] = codes[0, :6]
    masked2 = torch.tensor([[False] * 6 + [True] * 6])
    lg2 = dec(hidden2[None], codes2, masked2)[0]
    records.append((hidden2, codes2, masked2, lg2))
    print('logits shape:', tuple(lg.shape), 'sample:', lg.reshape(-1)[:3].tolist(), flush=True)

with open(os.path.join(out, 'logits.f32'), 'wb') as f:
    for _, _, _, lg in records:
        f.write(lg.contiguous().numpy().astype('<f4').tobytes())
meta = {'records': [{'hidden': h.tolist(), 'codes': c.tolist(), 'masked': mk.tolist()} for h, c, mk, _ in records],
        'codebook_size': 2017, 'codebooks': 12}
json.dump(meta, open(os.path.join(out, 'inputs.json'), 'w'))
print('done:', len(records), 'records ->', out, flush=True)

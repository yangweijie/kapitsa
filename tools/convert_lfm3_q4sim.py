#!/usr/bin/env python3
"""Re-quantize the LFM3-Omni language model into a KAP-layout q4 model with a clean
min-max block-32 affine quantizer (the compare page's q4 lane's quantizer): value =
scale * code + bias per 32-element group, 4-bit codes 0-15 packed 8-per-U32 (code j at
bits 4j), scales/biases bf16. embed_tokens takes KAP's 8-bit group-64 slot; the norms,
conv weights and everything non-linear stay bf16.

Why: the shipped q4_0 model's codes (a GGUF import, a different noise realization of the
same q4 class) put the Omni TTS reply in the long, dip-rich tail of the reference's
generation distribution (the compare sentence: 2.8-3.2 mid-sentence silent frames over 8
runs). Re-quantizing from the reference checkpoint with this quantizer halves that (1.25
over 8 runs), measured through the KAP serve on an idle M5 Pro. The KAP 4-bit GEMM's
own K-reassociation adds the remainder over a torch reference on the same codes (a
separate, shared-kernel item).

The TTS side (audio-gen.safetensors, codec-decoder.safetensors, audio_gen_config.json)
and the tokenizer / chat template are copied from an existing KAP model dir (they are
pipeline-agnostic).

Usage:
  python3 tools/convert_lfm3_q4sim.py HF_OMNI_DIR KAP_TTS_SOURCE_DIR OUT_DIR
Example (this machine):
  python3 tools/convert_lfm3_q4sim.py .lfm3-real/omni-fp32 \
      /Volumes/data/.kapitsa/models/lfm3-omni-2b-q4_0 \
      /Volumes/data/.kapitsa/models/lfm3-omni-2b-q4-sim
"""
import json, os, re, shutil, struct, sys
import numpy as np

HF_DIR, KAP_SRC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
HF = os.path.join(HF_DIR, "model.safetensors")

def load_hdr(p):
    f = open(p, 'rb'); n = struct.unpack('<Q', f.read(8))[0]
    return json.loads(f.read(n)), n

hh, hn = load_hdr(HF)

def read(hk):
    f = open(HF, 'rb'); off = hh[hk]['data_offsets']; f.seek(off[0] + hn + 8)
    b = np.frombuffer(f.read(off[1] - off[0]), dtype=np.uint16); f.close()
    return (b.astype(np.uint32) << 16).view(np.float32).reshape(hh[hk]['shape'])

def kap_key(hk):
    m = re.match(r'model\.language_model\.(embed_tokens\.weight|embedding_norm\.weight|layers\.(\d+)\.(.+))', hk)
    if not m:
        return None
    if m.group(1) == 'embed_tokens.weight':
        return 'model.embed_tokens.weight', 'embed'
    if m.group(1) == 'embedding_norm.weight':
        return 'model.norm.weight', 'bf16'
    sub = m.group(3).replace('feed_forward', 'mlp')
    if re.match(r'(self_attn\.\w+_proj|mlp\.\w+|conv\.(in_proj|out_proj))\.', sub):
        return 'model.layers.' + m.group(2) + '.' + sub, 'linear'
    return 'model.layers.' + m.group(2) + '.' + sub, 'bf16'

def pack4(w):  # 4-bit affine group 32: codes [N, K/8] U32, scales/biases [N, K/32]
    N, K = w.shape
    x = w.astype(np.float64).reshape(N, K // 32, 32)
    mn = x.min(2, keepdims=True); sc = (x - mn).max(2, keepdims=True) / 15.0
    q = np.round((x - mn) / np.maximum(sc, 1e-12)).clip(0, 15).astype(np.uint32).reshape(N, K // 8, 8)
    codes = 0
    for j in range(8):
        codes |= q[:, :, j] << (4 * j)
    return codes.astype(np.uint32), mn.reshape(N, K // 32).astype(np.float32), sc.reshape(N, K // 32).astype(np.float32)

def pack8(w):  # 8-bit affine group 64: codes [N, K/4] U32, scales/biases [N, K/64]
    N, K = w.shape
    x = w.astype(np.float64).reshape(N, K // 64, 64)
    mn = x.min(2, keepdims=True); sc = (x - mn).max(2, keepdims=True) / 255.0
    q = np.round((x - mn) / np.maximum(sc, 1e-12)).clip(0, 255).astype(np.uint32).reshape(N, K // 4, 4)
    codes = 0
    for j in range(4):
        codes |= q[:, :, j] << (8 * j)
    return codes.astype(np.uint32), mn.reshape(N, K // 64).astype(np.float32), sc.reshape(N, K // 64).astype(np.float32)

def bf16(f32):
    u = (f32.view(np.uint32) + 0x7fff + ((f32.view(np.uint32) >> 16) & 1)).astype(np.uint32) >> 16
    return u.astype(np.uint16)

os.makedirs(OUT, exist_ok=True)
data_parts, wm = [], {}
for hk, hv in sorted(hh.items()):
    if not isinstance(hv, dict) or 'dtype' not in hv:
        continue
    got = kap_key(hk)
    if got is None:
        continue
    kk, kind = got
    w = read(hk)
    if kind == 'embed':
        c, mn, sc = pack8(w)
        base = kk[:-len('.weight')]
        extras = [(base + '.scales', bf16(sc).tobytes(), 'BF16', sc.shape),
                  (base + '.biases', bf16(mn).tobytes(), 'BF16', mn.shape)]
        data_parts.append((kk, c.tobytes(), 'U32', c.shape, extras))
    elif kind == 'linear':
        c, mn, sc = pack4(w)
        base = kk[:-len('.weight')]
        extras = [(base + '.scales', bf16(sc).tobytes(), 'BF16', sc.shape),
                  (base + '.biases', bf16(mn).tobytes(), 'BF16', mn.shape)]
        data_parts.append((kk, c.tobytes(), 'U32', c.shape, extras))
    else:
        data_parts.append((kk, bf16(w).tobytes(), 'BF16', w.shape, []))
# one pass: [8-byte length][json header][data]; the data_offsets are relative to the header's end
h = {'__metadata__': {}}
p = 0
for kk, raw, dt, shape, extras in data_parts:
    h[kk] = {'dtype': dt, 'shape': [int(x) for x in shape], 'data_offsets': [p, p + len(raw)]}
    p += len(raw)
    wm[kk] = 'model.safetensors'
    for ek, ev, ed, es in extras:
        h[ek] = {'dtype': ed, 'shape': [int(x) for x in es], 'data_offsets': [p, p + len(ev)]}
        p += len(ev)
        wm[ek] = 'model.safetensors'
h['__metadata__'] = {'total_size': p}
hj = json.dumps(h).encode()
with open(os.path.join(OUT, 'model.safetensors'), 'wb') as f:
    f.write(struct.pack('<Q', len(hj))); f.write(hj)
    for kk, raw, dt, shape, extras in data_parts:
        f.write(raw)
        for ek, ev, ed, es in extras:
            f.write(ev)
json.dump({'metadata': {'total_size': p}, 'weight_map': wm},
          open(os.path.join(OUT, 'model.safetensors.index.json'), 'w'), indent=1)
cfg = json.load(open(os.path.join(KAP_SRC, 'config.json')))
cfg.pop('converted_from_gguf', None)
json.dump(cfg, open(os.path.join(OUT, 'config.json'), 'w'), indent=1)
for f in ('audio-gen.safetensors', 'codec-decoder.safetensors', 'audio_gen_config.json',
          'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
    s = os.path.join(KAP_SRC, f)
    if os.path.exists(s):
        shutil.copy2(s, os.path.join(OUT, f))
print(f'wrote {len(wm)} tensors ({p / 2 ** 30:.2f} GB) -> {OUT}')

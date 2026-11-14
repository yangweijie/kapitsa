#!/usr/bin/env python3
"""Converts LFM3-Omni's mmproj GGUF (llama.cpp's `clip` architecture) into a bf16 safetensors
folder with the reference implementation's names and shapes.

    convert_lfm3_mmproj.py MMPROJ.gguf OUT [--self-test] [--verify]

OUT gets `vision.safetensors` (BF16, every tensor), `vision_config.json` (the tower's and
processor's shapes, from the GGUF metadata) and `conversion.json` (the source's sha256, every
tensor's shape and dtype, and the patch kernels' statistics). It never writes `config.json`, so it
is safe to point OUT at a model folder that already has the text tower's config.

The GGUF's dimensions are ggml order: ne[0] is the innermost. Read in C order and reshaped to
reversed(ne), a 2-D weight is already HF's [out, in] and a conv kernel is already [out, in, kh, kw]
-- no transpose. Q4_0 (18 bytes per 32 values: an f16 scale then 16 packed codes) and F16/F32 are
decoded to f32 and written as round-to-nearest-even BF16, the format Kapitsa's vision loader reads.

The two `v.patch_embd` kernels are llama.cpp's split of the reference's
Linear(3 * temporal_patch_size * 16 * 16 -> 1024) patchifier. They are kept individually as
`patchifier.temporal_0/1.weight` AND interleaved into `patchifier.weight` in the processor's
[c, t, ph, pw] column order (kernel N is temporal frame N). Checked against the HF BF16 weight's
first row: bitwise identical; every other arrangement correlates only ~0.75.

Only the vision tensors are converted by default: `v.*` and `mm.*` except `mm.a.*`. With
`--audio`, the audio INPUT path is converted instead into `audio.safetensors` +
`audio_config.json`: `a.blk.*` (the 17-block encoder), `a.conv1d.*` + `a.post_ln` (the
downsampler and the encoder's final norm) and `mm.a.mlp.*` (the 768→2048 projector).
`a.gen.*` (the speech generation side) is out of scope in both modes.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np

F32, F16, Q4_0 = 0, 1, 2
SUPPORTED = {F32: "F32", F16: "F16", Q4_0: "Q4_0"}


class Reader:
    """A GGUF v3 header: metadata, the tensor table, and the file handle for the data section."""

    def __init__(self, path):
        self.f = open(path, "rb")
        self.r = self.f  # reads are sequential until the data section

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF header")
        return b

    def u8(self):
        return struct.unpack("<B", self.raw(1))[0]

    def u16(self):
        return struct.unpack("<H", self.raw(2))[0]

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def i32(self):
        return struct.unpack("<i", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def i64(self):
        return struct.unpack("<q", self.raw(8))[0]

    def f32(self):
        return struct.unpack("<f", self.raw(4))[0]

    def f64(self):
        return struct.unpack("<d", self.raw(8))[0]

    def string(self):
        return self.raw(self.u64()).decode("utf-8", "replace")

    def value(self, t):
        if t == 9:
            at, n = self.u32(), self.u64()
            return [self.value(at) for _ in range(n)]
        return {
            0: self.u8,
            1: lambda: struct.unpack("<b", self.raw(1))[0],
            2: self.u16,
            3: lambda: struct.unpack("<h", self.raw(2))[0],
            4: self.u32,
            5: self.i32,
            6: self.f32,
            7: lambda: bool(self.u8()),
            8: self.string,
            10: self.u64,
            11: self.i64,
            12: self.f64,
        }[t]()

    def pos(self):
        return self.f.tell()


def prod(xs):
    n = 1
    for x in xs:
        n *= int(x)
    return n


class GGUF:
    def __init__(self, path):
        r = Reader(path)
        if r.raw(4) != b"GGUF":
            raise ValueError(path + " is not a GGUF file")
        self.version = r.u32()
        nt, nk = r.u64(), r.u64()
        self.kv = {}
        for _ in range(nk):
            key = r.string()
            self.kv[key] = r.value(r.u32())
        align = int(self.kv.get("general.alignment", 32))
        if align <= 0 or align > (1 << 20) or align & (align - 1):
            raise ValueError("invalid GGUF alignment " + str(align))
        self.tensors = []
        for _ in range(nt):
            name = r.string()
            ne = [r.u64() for _ in range(r.u32())]
            typ, rel = r.u32(), r.u64()
            n = prod(ne)
            nbytes = n // 32 * 18 if typ == Q4_0 else n * (4 if typ == F32 else 2 if typ == F16 else 0)
            self.tensors.append({"name": name, "ne": ne, "type": typ, "rel": rel, "bytes": nbytes})
        self.data0 = (r.pos() + align - 1) // align * align
        self.f = r.f
        self.file_bytes = os.path.getsize(path)
        for t in self.tensors:
            if self.data0 + t["rel"] + t["bytes"] > self.file_bytes:
                raise ValueError("tensor " + t["name"] + " lies past the end of the file")
        if self.kv.get("general.architecture") != "clip" or self.kv.get("general.type") != "mmproj":
            raise ValueError("this is not a clip/mmproj GGUF (architecture=" +
                             str(self.kv.get("general.architecture")) + ")")

    def raw(self, t):
        self.f.seek(self.data0 + t["rel"])
        b = self.f.read(t["bytes"])
        if len(b) != t["bytes"]:
            raise ValueError("short read of " + t["name"])
        return b


def decode(t, raw):
    """A tensor's values as f32, shaped in the reference implementation's (C-order) layout."""
    typ, ne = t["type"], t["ne"]
    if typ not in SUPPORTED:
        raise ValueError("unsupported GGUF type " + str(typ) + " in " + t["name"])
    if typ == F32:
        a = np.frombuffer(raw, dtype="<f4")
    elif typ == F16:
        a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    else:
        if ne[0] % 32:
            raise ValueError(t["name"] + ": Q4_0 rows are not whole 32-value blocks")
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
        d = np.frombuffer(b[:, :2].copy().tobytes(), dtype="<f2").astype(np.float32)
        q = b[:, 2:]
        vals = np.concatenate([(q & 15).astype(np.float32) - 8, (q >> 4).astype(np.float32) - 8], axis=1)
        a = (vals * d[:, None]).ravel()
    return a.reshape(tuple(reversed(ne)))


def to_bf16(a):
    """Round-to-nearest-even BF16 bytes (the same convention as src/gguf/quant.cpp)."""
    u = np.ascontiguousarray(a, dtype="<f4").view("<u4")
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2").tobytes()


def from_bf16(raw):
    return (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")


# v.blk.N.<field> -> the reference's name under vision_tower.layers.N.
BLOCK = {
    "conv2d_in_proj.weight": "conv.in_proj.weight",
    "conv2d_out_proj.weight": "conv.out_proj.weight",
    "conv2d_spatial.weight": "conv.spatial_conv.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_out.weight": "self_attn.out_proj.weight",
    "attn_q_norm.weight": "self_attn.q_layernorm.weight",
    "attn_k_norm.weight": "self_attn.k_layernorm.weight",
    "ffn_gate.weight": "feed_forward.w1.weight",
    "ffn_up.weight": "feed_forward.w3.weight",
    "ffn_down.weight": "feed_forward.w2.weight",
    "ln1.weight": "operator_norm.weight",
    "ln2.weight": "ffn_norm.weight",
}
MM = {
    "mm.0.weight": "multi_modal_projector.linear_1.weight",
    "mm.0.bias": "multi_modal_projector.linear_1.bias",
    "mm.2.weight": "multi_modal_projector.linear_2.weight",
    "mm.2.bias": "multi_modal_projector.linear_2.bias",
    "mm.input_norm.weight": "multi_modal_projector.layer_norm.weight",
    "mm.input_norm.bias": "multi_modal_projector.layer_norm.bias",
}

# The audio INPUT path: a.blk.N.* uses the same field names as the vision tower except the
# conv blocks, which are LFM2-style 1-D gated short convs (conv_dw k=3 + the pw pair), and
# mm.a.mlp.* is the audio projector. a.post_ln is the encoder's final RMSNorm.
ABLOCK = dict(BLOCK)
ABLOCK.update({
    "conv_dw.weight": "conv.conv.weight",
    "conv_pw1.weight": "conv.in_proj.weight",
    "conv_pw2.weight": "conv.out_proj.weight",
})
ACONV1D = {
    "a.conv1d.0.weight": "audio_encoder.downsample.conv_in.weight",
    "a.conv1d.0.bias": "audio_encoder.downsample.conv_in.bias",
    "a.conv1d.2.weight": "audio_encoder.downsample.conv_mid.weight",
    "a.conv1d.2.bias": "audio_encoder.downsample.conv_mid.bias",
    "a.conv1d.3.weight": "audio_encoder.downsample.lin_mid.weight",
    "a.conv1d.3.bias": "audio_encoder.downsample.lin_mid.bias",
    "a.conv1d.5.weight": "audio_encoder.downsample.conv_end.weight",
    "a.conv1d.5.bias": "audio_encoder.downsample.conv_end.bias",
    "a.conv1d.6.weight": "audio_encoder.downsample.lin_end.weight",
    "a.conv1d.6.bias": "audio_encoder.downsample.lin_end.bias",
    # llama.cpp's misleading name: proven by numbers, not the name — weight corr 0.9956
    # (the Q4_0 band) and bias bitwise against the fp32 reference's lin_out.
    "a.pre_encode.out.weight": "audio_encoder.downsample.lin_out.weight",
    "a.pre_encode.out.bias": "audio_encoder.downsample.lin_out.bias",
}
AMM = {
    "mm.a.mlp.0.weight": "audio_projector.layer_norm.weight",
    "mm.a.mlp.0.bias": "audio_projector.layer_norm.bias",
    "mm.a.mlp.1.weight": "audio_projector.linear_1.weight",
    "mm.a.mlp.1.bias": "audio_projector.linear_1.bias",
    "mm.a.mlp.3.weight": "audio_projector.linear_2.weight",
    "mm.a.mlp.3.bias": "audio_projector.linear_2.bias",
}


def audio_mapped(name):
    """None for anything outside the audio input path (a.gen.*, unknown a.*)."""
    if name == "a.post_ln.weight":
        return "audio_encoder.lfm.embedding_norm.weight"
    if name in ACONV1D:
        return ACONV1D[name]
    if name in AMM:
        return AMM[name]
    if name.startswith("a.blk."):
        n, field = name[6:].split(".", 1)
        if field in ABLOCK and n.isdigit():
            return "audio_encoder.lfm.layers." + n + "." + ABLOCK[field]
    return None


def audio_tensors(g):
    out = []
    for t in g.tensors:
        n = t["name"]
        if n.startswith("a.gen."):
            continue
        if n.startswith("a.") or n.startswith("mm.a."):
            if audio_mapped(n) is None:
                raise ValueError("unmapped audio tensor: " + n)
            out.append(t)
    return out


def mapped(name):
    if name in MM:
        return MM[name]
    if name in ("v.patch_embd.weight", "v.patch_embd.weight.1"):
        return "vision_tower.patchifier.temporal_" + ("0" if name == "v.patch_embd.weight" else "1") + ".weight"
    if name == "v.embedding_norm.weight":
        return "vision_tower.embedding_norm.weight"
    if name.startswith("v.blk."):
        n, field = name[6:].split(".", 1)
        if field in BLOCK and n.isdigit():
            return "vision_tower.layers." + n + "." + BLOCK[field]
    return None


def vision_tensors(g):
    out = []
    for t in g.tensors:
        n = t["name"]
        if n.startswith("v.") or (n.startswith("mm.") and not n.startswith("mm.a.")):
            out.append(t)
    return out


def write_safetensors(path, items):
    """items: list of (name, shape, bf16 bytes). One shard, BF16 throughout."""
    header, off = {}, 0
    for name, shape, data in items:
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [off, off + len(data)]}
        off += len(data)
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for _, _, data in items:
            f.write(data)


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        body = f.read()
    out = {}
    for name, h in header.items():
        if name == "__metadata__":
            continue
        a, b = h["data_offsets"]
        out[name] = (tuple(h["shape"]), h["dtype"], body[a:b])
    return out


def self_test():
    """Independent checks of the decoder, the bf16 conversion and the name map."""
    ok = 0

    def check(cond, what):
        nonlocal ok
        if not cond:
            raise AssertionError(what)
        ok += 1

    # One Q4_0 block: scale 0.5, low nibbles 0..15, high nibbles 15..0.
    block = bytearray(18)
    struct.pack_into("<e", block, 0, 0.5)
    for i in range(16):
        block[2 + i] = i | ((15 - i) << 4)
    t = {"name": "q", "ne": [32, 1], "type": Q4_0, "rel": 0, "bytes": 18}
    v = decode(t, bytes(block)).ravel()
    check(np.allclose(v[:16], (np.arange(16) - 8) * 0.5), "Q4_0 low nibbles")
    check(np.allclose(v[16:], (np.arange(15, -1, -1) - 8) * 0.5), "Q4_0 high nibbles")
    check(v.shape == (32,), "Q4_0 shape")

    # F16 and F32 round trip through bf16 (tolerance of one bf16 step).
    for typ, dt in ((F16, "<f2"), (F32, "<f4")):
        src = np.array([1.0, -2.0, 0.5, 1e-3], dtype=dt)
        t = {"name": "f", "ne": [4], "type": typ, "rel": 0, "bytes": src.nbytes}
        a = decode(t, src.tobytes())
        back = from_bf16(to_bf16(a))
        check(np.allclose(a, back, rtol=0.008, atol=1e-5), "bf16 round trip " + SUPPORTED[typ])
    check(to_bf16(np.array([1.0], dtype="<f4")) == b"\x80\x3f", "bf16 encoding of 1.0")

    # Every field of a conv and an attn block maps, and audio tensors do not.
    for f in BLOCK:
        check(mapped("v.blk.7." + f) is not None, "name map " + f)
    for n in ("a.blk.0.attn_q.weight", "mm.a.mlp.0.weight", "a.gen.0.weight"):
        check(mapped(n) is None, "audio skip " + n)
    check(mapped("v.patch_embd.weight.1").endswith("temporal_1.weight"), "patch kernel 1")
    print("ok self-test: " + str(ok) + " checks")
    return ok


def convert_audio(src, out, verify):
    """The audio input path: audio.safetensors + audio_config.json (never config.json)."""
    g = GGUF(src)
    ts = audio_tensors(g)
    if len(ts) < 170:  # 17 blocks (10 conv x 9 + 7 attn x 11 = 167) + 11 down + 6 proj = 184… counted, not assumed
        raise ValueError("suspiciously few audio tensors: " + str(len(ts)))
    os.makedirs(out, exist_ok=True)
    items, manifest = [], {}
    for t in ts:
        a = decode(t, g.raw(t))
        name = audio_mapped(t["name"])
        items.append((name, a.shape, to_bf16(a)))
        manifest[name] = {"source": t["name"], "shape": list(a.shape),
                          "gguf_type": SUPPORTED[t["type"]],
                          "mean": float(a.mean()), "std": float(a.std())}
    items.sort(key=lambda x: x[0])
    write_safetensors(os.path.join(out, "audio.safetensors"), items)

    kv = g.kv
    cfg = {
        "model_type": "lfm3_omni_audio",
        "converted_from_gguf": os.path.basename(src),
        "audio_encoder_config": {
            "num_mel_bins": kv["clip.audio.num_mel_bins"],
            "conv_channels": 256,
            "hidden_size": kv["clip.audio.embedding_length"],
            "intermediate_size": kv["clip.audio.feed_forward_length"],
            "num_hidden_layers": kv["clip.audio.block_count"],
            "num_attention_heads": kv["clip.audio.attention.head_count"],
            "num_key_value_heads": kv["clip.audio.attention.head_count_kv"],
            "head_dim": kv["clip.audio.embedding_length"] // kv["clip.audio.attention.head_count"],
            "sliding_window": 32,
            "conv_kernel_size": 3,
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
            "norm_eps": 1e-5,
            "layer_types": (["conv", "conv", "full_attention"] * 5) + ["full_attention", "full_attention"],
        },
        # The Lfm3OmniProjector: LayerNorm -> Linear -> GELU -> Linear, all biased.
        "audio_projector": {
            "layer_norm_eps": 1e-5,
            "act": "gelu",
            "in_features": kv["clip.audio.embedding_length"],
            "out_features": kv["clip.vision.projection_dim"],
        },
        # The feature extractor's constants (processor_config.json; the GGUF stores only mel bins).
        "feature_extractor": {
            "sampling_rate": 16000,
            "n_fft": 400,
            "win_length": 400,
            "hop_length": 160,
            "feature_size": kv["clip.audio.num_mel_bins"],
            "mel_scale": "slaney",
            "log_floor": -12.0,
            "normalization_offset": 4.0,
            "normalization_divisor": 4.0,
            "frames_per_token": 8,
        },
        "audio_token_id": 125031,
        "audio_start_token_id": 125012,
        "audio_part": {"type": "audio"},
        "audio_span": {"start": "<|vision_start|>", "placeholder": "<|audio|>", "end": "<|vision_end|>"},
    }
    with open(os.path.join(out, "audio_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    with open(os.path.join(out, "audio_conversion.json"), "w") as f:
        json.dump({"source": os.path.abspath(src), "sha256": sha256(src), "file_bytes": g.file_bytes,
                   "gguf_version": g.version, "audio_tensors": len(ts), "written_tensors": len(items),
                   "dtype": "BF16", "tensors": manifest,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
        f.write("\n")

    path = os.path.join(out, "audio.safetensors")
    stats = {"tensors": len(ts), "written": len(items), "bytes": os.path.getsize(path)}
    if verify:
        check = read_safetensors(path)
        if len(check) != len(items):
            raise AssertionError("safetensors tensor count")
        got = 0
        for name, shape, data in items:
            s, dt, body = check[name]
            if tuple(shape) != s or dt != "BF16" or len(body) != len(data):
                raise AssertionError("safetensors header mismatch for " + name)
            if not np.isfinite(from_bf16(data)).all():
                raise AssertionError("non-finite values in " + name)
            got += 1
        print("ok verify: " + str(got) + " tensors re-read, finite, BF16")
    print(json.dumps(stats, indent=2))
    return stats


def convert(src, out, verify):
    g = GGUF(src)
    ts = vision_tensors(g)
    unknown = [t["name"] for t in ts if mapped(t["name"]) is None]
    if unknown:
        raise ValueError("unmapped vision tensors: " + ", ".join(unknown[:8]))
    bad = [t["name"] for t in ts if t["type"] not in SUPPORTED]
    if bad:
        raise ValueError("unsupported types: " + ", ".join(bad[:8]))
    os.makedirs(out, exist_ok=True)

    items, manifest, patches = [], {}, {}
    for t in ts:
        a = decode(t, g.raw(t))
        name = mapped(t["name"])
        items.append((name, a.shape, to_bf16(a)))
        manifest[name] = {"source": t["name"], "shape": list(a.shape), "gguf_type": SUPPORTED[t["type"]],
                          "mean": float(a.mean()), "std": float(a.std())}
        if name.startswith("vision_tower.patchifier.temporal_"):
            patches[name] = a.copy()

    # The reference's patchifier is Linear(3*2*16*16 -> 1024) over [c, t, ph, pw] columns.
    # Keep the two kernels as above, and interleave them into one [1024, 1536] weight.
    if len(patches) == 2:
        w0 = patches["vision_tower.patchifier.temporal_0.weight"].reshape(1024, 3, 16, 16)
        w1 = patches["vision_tower.patchifier.temporal_1.weight"].reshape(1024, 3, 16, 16)
        w = np.stack([w0, w1], axis=2).reshape(1024, 2 * 3 * 16 * 16)  # [out, c, t, ph, pw]
        items.append(("vision_tower.patchifier.weight", w.shape, to_bf16(w)))
        manifest["vision_tower.patchifier.weight"] = {
            "source": "v.patch_embd.weight + v.patch_embd.weight.1 (interleaved)",
            "shape": list(w.shape), "gguf_type": "F16", "correlation":
            float(np.corrcoef(w0.ravel(), w1.ravel())[0, 1]),
        }
    items.sort(key=lambda x: x[0])
    write_safetensors(os.path.join(out, "vision.safetensors"), items)

    kv = g.kv
    cfg = {
        "model_type": "lfm3_vl_vision",
        "converted_from_gguf": os.path.basename(src),
        "vision_config": {
            "hidden_size": kv["clip.vision.embedding_length"],
            "intermediate_size": kv["clip.vision.feed_forward_length"],
            "num_hidden_layers": kv["clip.vision.block_count"],
            "num_attention_heads": kv["clip.vision.attention.head_count"],
            "num_key_value_heads": kv["clip.vision.attention.head_count_kv"],
            "patch_size": kv["clip.vision.patch_size"],
            "temporal_patch_size": 2,
            "in_channels": 3,
            "spatial_merge_size": kv["clip.vision.spatial_merge_size"],
            "conv_kernel_size": 3,
            "norm_eps": kv["clip.vision.attention.layer_norm_epsilon"],
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "axial"},
        },
        "downsample_factor": kv["clip.vision.spatial_merge_size"],
        "projector_hidden_size": 4096,
        "projector_hidden_act": "gelu",
        "projector_bias": True,
        "projector_layer_norm_eps": 1e-6,
        "text_hidden_size": kv["clip.vision.projection_dim"],
        "image_token_id": 125030,
        "image_size": kv["clip.vision.image_size"],
        "image_mean": kv["clip.vision.image_mean"],
        "image_std": kv["clip.vision.image_std"],
        "image_min_pixels": kv["clip.vision.image_min_pixels"],
        "image_max_pixels": kv["clip.vision.image_max_pixels"],
        "image_resize_algo": kv["clip.vision.image_resize_algo"],
        # The GGUF has no keys for these; they are Lfm3OmniProcessor's constants
        # (processor_config.json), which the image encoder needs to build the prompt.
        "processor": {
            "image_token": "<|image|>",
            "vision_start_token": "<|vision_start|>",
            "vision_end_token": "<|vision_end|>",
            "vision_markers": True,
            "resample": 3,  # PIL bicubic
            "rescale_factor": 1.0 / 255.0,
            "resize_factor": kv["clip.vision.patch_size"] * kv["clip.vision.spatial_merge_size"],
        },
    }
    with open(os.path.join(out, "vision_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    with open(os.path.join(out, "conversion.json"), "w") as f:
        json.dump({"source": os.path.abspath(src), "sha256": sha256(src), "file_bytes": g.file_bytes,
                   "gguf_version": g.version, "vision_tensors": len(ts), "written_tensors": len(items),
                   "dtype": "BF16", "tensors": manifest, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
        f.write("\n")

    stats = {"tensors": len(ts), "written": len(items), "bytes": os.path.getsize(os.path.join(out, "vision.safetensors"))}
    if "vision_tower.patchifier.weight" in manifest:
        stats["patch_kernel_correlation"] = manifest["vision_tower.patchifier.weight"]["correlation"]
    print(json.dumps(stats, indent=2))
    if verify:
        check = read_safetensors(os.path.join(out, "vision.safetensors"))
        if len(check) != len(items):
            raise AssertionError("safetensors tensor count")
        got = 0
        for name, shape, data in items:
            s, dt, body = check[name]
            if tuple(shape) != s or dt != "BF16" or len(body) != len(data):
                raise AssertionError("safetensors header mismatch for " + name)
            a = from_bf16(data)
            if not np.isfinite(a).all():
                raise AssertionError("non-finite values in " + name)
            got += 1
        print("ok verify: " + str(got) + " tensors re-read, finite, BF16")
    return stats


GEN_KEEP = ("model.depth_decoder.", "model.codebook_embedding.")
GEN_CONFIG_KEYS = ("audio_out_token_id", "audio_start_token_id", "audio_eos_code", "codebook_size",
                   "num_codebooks", "maskgit_schedule", "audio_temperature", "audio_top_k",
                   "audio_repetition_penalty", "depth_config", "downsample_factor", "interleaved_n_audio")


def convert_gen_depth(hf_dir, out, verify):
    """The audio GENERATION side from the Omni checkpoint (the GGUF carries no gen tensors): the
    depth decoder's and codebook_embedding's tensors copied as-is (bf16) into audio-gen.safetensors,
    plus the generation config keys the engine needs, into audio_gen_config.json."""
    import json as j
    with open(os.path.join(hf_dir, "model.safetensors"), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = j.loads(f.read(n))
        data_at = 8 + n
    cfg = j.load(open(os.path.join(hf_dir, "config.json")))
    kept = {}
    blobs = []
    at = 0
    for name in sorted(hdr):
        if name == "__metadata__" or not name.startswith(GEN_KEEP):
            continue
        t = hdr[name]
        begin, end = t["data_offsets"]
        with open(os.path.join(hf_dir, "model.safetensors"), "rb") as f:
            f.seek(data_at + begin)
            blobs.append((name, t["dtype"], t["shape"], f.read(end - begin)))
    for name, dtype, shape, blob in blobs:
        start = at
        at += len(blob)
        kept[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, at]}
    head = j.dumps(kept, separators=(",", ":")).encode()
    pad = (8 - (8 + len(head)) % 8) % 8
    with open(os.path.join(out, "audio-gen.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(head) + pad))
        f.write(head)
        f.write(b" " * pad)
        for _, _, _, blob in blobs:
            f.write(blob)
    gen_cfg = {k: cfg[k] for k in GEN_CONFIG_KEYS if k in cfg}
    gen_cfg["converted_from"] = os.path.abspath(hf_dir)
    j.dump(gen_cfg, open(os.path.join(out, "audio_gen_config.json"), "w"), indent=1)
    print(f"gen side: {len(kept)} tensors ({sum(len(b) for _, _, _, b in blobs)} bytes) -> "
          f"{out}/audio-gen.safetensors; config keys: {sorted(gen_cfg)}")
    if verify:
        with open(os.path.join(out, "audio-gen.safetensors"), "rb") as f:
            n2 = struct.unpack("<Q", f.read(8))[0]
            h2 = j.loads(f.read(n2))
            for name in kept:
                assert name in h2 and h2[name]["data_offsets"] == kept[name]["data_offsets"], name
        print(f"verify: {len(kept)} tensors re-read ok")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", nargs="?", help="the mmproj GGUF")
    p.add_argument("out", nargs="?", help="the folder to write")
    p.add_argument("--self-test", action="store_true", help="check the decoder and the name map, no file needed")
    p.add_argument("--verify", action="store_true", help="re-read the output safetensors and check every tensor")
    p.add_argument("--audio", action="store_true", help="convert the audio input path (a.*, mm.a.*) instead of vision")
    p.add_argument("--gen-depth", metavar="HF_DIR",
                   help="copy the audio GENERATION side (the depth decoder + codebook_embedding) from an HF "
                        "checkpoint dir into OUT as audio-gen.safetensors + audio_gen_config.json (the GGUF "
                        "carries no gen tensors; the Omni checkpoint does)")
    a = p.parse_args()
    if a.self_test:
        self_test()
        if a.src is None:
            return 0
    if a.gen_depth:
        convert_gen_depth(a.gen_depth, a.out if a.out else a.src, a.verify)
        return 0
    if not a.src or not a.out:
        p.error("MMPROJ.gguf and OUT are needed (or just --self-test)")
    if a.audio:
        convert_audio(a.src, a.out, a.verify)
    else:
        convert(a.src, a.out, a.verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())

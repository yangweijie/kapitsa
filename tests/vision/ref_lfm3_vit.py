"""Independent CPU reference for LFM3-Omni's vision geometry (numpy only, no torch, no model).

Transcribed mechanically from the HF reference in .lfm3-real/omni-ref/:
  image_processing_lfm3_vl.py : smart_resize, patchify
  modeling_lfm3_vl.py         : Lfm3VlVisionConv2d (merge-block <-> raster), axial RoPE, RMSNorm
  transformers.vision_utils   : get_vision_position_ids
It prints the golden constants that tools/lfm3_vit_ref_test.cpp pins, and asserts the geometric
invariants those constants depend on. Run: python3 tests/vision/ref_lfm3_vit.py
"""
from __future__ import annotations

import math

import numpy as np

FACTOR = 32  # patch_size * merge_size
MIN_PIXELS = 65536
MAX_PIXELS = 2097152  # processor_config.json's override, not the class default (16777216)
PATCH = 16
TEMPORAL = 2
MERGE = 2
CHANNELS = 3
HEAD_DIM = 64
ROPE_THETA = 1e6
NORM_EPS = 1e-5


def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """image_processing_lfm3_vl.py:50-77. Python's round() is half-to-even."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def position_ids(h, w, merge=MERGE):
    """transformers.vision_utils.get_vision_position_ids, spatial_merge_size=merge, t=1.

    meshgrid(ij) -> reshape(h/m, m, w/m, m) -> transpose(1,2) -> flatten, for the h and w grids.
    Returns (h*w, 2): row n is (row_coord, col_coord) in merge-block order.
    """
    hp, wp = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    block = (h // merge, merge, w // merge, merge)
    return np.stack(
        [
            hp.reshape(block).transpose(0, 2, 1, 3).ravel(),
            wp.reshape(block).transpose(0, 2, 1, 3).ravel(),
        ],
        axis=-1,
    )


def merge_to_raster(gated, h, w, merge=MERGE):
    """modeling_lfm3_vl.py:325-327. (frames, h*w, C) merge-block order -> (frames, C, h, w)."""
    frames, _, channels = gated.shape
    return (
        gated.reshape(frames, h // merge, w // merge, merge, merge, channels)
        .transpose(0, 5, 1, 3, 2, 4)
        .reshape(frames, channels, h, w)
    )


def raster_to_merge(frames, h, w, merge=MERGE):
    """modeling_lfm3_vl.py:328-330. Inverse of merge_to_raster."""
    f, channels, _, _ = frames.shape
    return (
        frames.reshape(f, channels, h // merge, merge, w // merge, merge)
        .transpose(0, 2, 4, 3, 5, 1)
        .reshape(f, h * w, channels)
    )


def patchify(images, patch=PATCH, merge=MERGE, temporal=TEMPORAL):
    """image_processing_lfm3_vl.py:164-196. (B,C,H,W) -> (B, grid_h*grid_w, C*T*P*P).

    The output column order is [c, t, ph, pw]; the row order is merge-block (bh, bw, mh, mw).
    A still frame is expanded to two identical temporal frames (no data-dependent choice).
    """
    b, channels, height, width = images.shape
    gh, gw = height // patch, width // patch
    patches = images.reshape(
        b, channels, gh // merge, merge, patch, gw // merge, merge, patch
    )
    patches = patches.transpose(0, 2, 5, 3, 6, 1, 4, 7)  # b,bh,bw,mh,mw,c,ph,pw
    patches = np.broadcast_to(
        patches[:, :, :, :, :, :, None, :, :],
        (b, gh // merge, gw // merge, merge, merge, channels, temporal, patch, patch),
    )
    return patches.reshape(b, gh * gw, channels * temporal * patch * patch)


def axial_rope(pos_h, pos_w, head_dim=HEAD_DIM, theta=ROPE_THETA):
    """modeling_lfm3_vl.py:108-134. Returns (cos, sin), each (head_dim,) float64."""
    spatial_dim = head_dim // 2  # config: hidden_size // num_heads // 2 = 1024/16/2 = 32
    inv_freq = 1.0 / (theta ** (np.arange(0, spatial_dim, 2, dtype=np.float64) / spatial_dim))
    cos_h, sin_h = np.cos(pos_h * inv_freq), np.sin(pos_h * inv_freq)
    cos_w, sin_w = np.cos(pos_w * inv_freq), np.sin(pos_w * inv_freq)
    hw_cos, hw_sin = np.concatenate([cos_h, cos_w]), np.concatenate([sin_h, sin_w])
    return np.concatenate([hw_cos, hw_cos]), np.concatenate([hw_sin, hw_sin])


def rotate_half(x):
    """modeling_lfm3_vl.py:137-141: cat((-x2, x1)), split at head_dim/2."""
    d = x.shape[-1]
    return np.concatenate([-x[..., d // 2 :], x[..., : d // 2]], axis=-1)


def apply_rope(q, cos, sin):
    """modeling_lfm3_vl.py:144-152 (fp32 math, before the cast back to q's dtype)."""
    return q * cos + rotate_half(q) * sin


def rms_norm(x, weight, eps=NORM_EPS):
    """modeling_lfm3_vl.py:59-69: T5-style, variance in fp32, weight multiplies the result."""
    xf = x.astype(np.float32)
    variance = np.mean(xf**2, axis=-1, keepdims=True)
    return weight * (xf * (1.0 / np.sqrt(variance + eps)))


def check(ok, what):
    if not ok:
        raise AssertionError(what)
    print(f"ok   {what}")


def main():
    # Geometric invariants: the conv's permutation, patchify's row order, and the HF position ids
    # must all describe the SAME merge-block -> raster mapping.
    for h, w in [(4, 4), (4, 8), (8, 4), (32, 32)]:
        ids = position_ids(h, w)
        rows = np.arange(h * w, dtype=np.float64).reshape(1, h * w, 1)
        raster = merge_to_raster(rows, h, w)[0, 0]  # (h, w), value = the merge-block row index
        check(
            np.array_equal(raster[ids[:, 0], ids[:, 1]], np.arange(h * w)),
            f"position_ids agree with the conv permutation at grid {h}x{w}",
        )
        check(
            np.array_equal(raster_to_merge(merge_to_raster(rows, h, w), h, w)[0, :, 0],
                           np.arange(h * w)),
            f"raster_to_merge is the exact inverse at grid {h}x{w}",
        )

    # patchify: row n's first column belongs to (c=0,t=0,ph=0,pw=0) of the raster cell ids[n].
    h, w = 4, 4
    image = np.arange(CHANNELS * (h * PATCH) * (w * PATCH), dtype=np.float64).reshape(
        1, CHANNELS, h * PATCH, w * PATCH
    )
    flat = patchify(image)[0]
    ids = position_ids(h, w)
    ok = True
    for n, (gy, gx) in enumerate(ids):
        for c in range(CHANNELS):
            for ph in (0, PATCH - 1):
                for pw in (0, PATCH - 1):
                    col = ((c * TEMPORAL + 0) * PATCH + ph) * PATCH + pw
                    want = image[0, c, gy * PATCH + ph, gx * PATCH + pw]
                    ok &= flat[n, col] == want
                    ok &= flat[n, ((c * TEMPORAL + 1) * PATCH + ph) * PATCH + pw] == want
    check(ok, "patchify columns are [c,t,ph,pw] with two identical temporal frames")

    check(
        rotation_is_unitary(),
        "axial RoPE preserves the head's L2 norm (rotate_half pairs i <-> i+head_dim/2)",
    )

    print("\n=== golden constants for tools/lfm3_vit_ref_test.cpp ===")
    shapes = [(512, 512), (100, 100), (1024, 1024), (48, 48), (16, 48), (37, 53), (2048, 64)]
    print("smart_resize (h,w) -> (oh,ow):")
    for a, b in shapes:
        print(f"  {a:5d} {b:5d} -> {smart_resize(a, b)}")
    print(f"round-half-to-even: round(1.5)={round(1.5)}, round(2.5)={round(2.5)}")
    print("position_ids / merge-block raster index for grid 4x4 (row-major raster values):")
    print(merge_to_raster(np.arange(16, dtype=np.float64).reshape(1, 16, 1), 4, 4)[0, 0].astype(int).tolist())
    print("position_ids(4,4) as (h,w) pairs:")
    print([tuple(map(int, p)) for p in position_ids(4, 4)])
    cos, sin = axial_rope(1, 2)
    print(f"axial_rope(1,2) cos[0:8] = {[round(float(v), 9) for v in cos[:8]]}")
    print(f"axial_rope(1,2) sin[0:8] = {[round(float(v), 9) for v in sin[:8]]}")
    print(f"axial_rope(1,2) cos[48:56] = {[round(float(v), 9) for v in cos[48:56]]}")
    x = np.array([0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 0.5, 0.25], dtype=np.float32)
    weight = np.array([1.0, 0.5, 2.0, 1.5, 0.25, 3.0, 1.0, 0.75], dtype=np.float32)
    print(f"rms_norm out = {[round(float(v), 9) for v in rms_norm(x, weight)]}")


def rotation_is_unitary():
    rng = np.random.default_rng(7)
    for ph, pw in [(0, 0), (1, 2), (7, 3), (31, 31)]:
        q = rng.normal(size=HEAD_DIM)
        cos, sin = axial_rope(ph, pw)
        roped = apply_rope(q, cos, sin)
        if not np.allclose(np.linalg.norm(roped), np.linalg.norm(q), rtol=1e-12):
            return False
    return True


if __name__ == "__main__":
    main()

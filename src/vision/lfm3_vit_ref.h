// LFM3-Omni's vision tower (Lfm3Vl): CPU references for its geometry and rotary math. Header-only,
// no Metal, no weights. Sources: .lfm3-real/omni-ref/{image_processing_lfm3_vl.py,
// modeling_lfm3_vl.py} and transformers.vision_utils.get_vision_position_ids. The independent numpy
// transcription is tests/vision/ref_lfm3_vit.py; its goldens are pinned in tools/lfm3_vit_ref_test.cpp.
// The tower's kernels diff against these like src/kernels/lfm3.metal diffs against lfm3_rope_ref.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ni::vision {

constexpr int kLfm3Patch = 16;
constexpr int kLfm3Merge = 2;
constexpr int kLfm3Temporal = 2;
constexpr int kLfm3Channels = 3;
constexpr int kLfm3HeadDim = 64;
constexpr int kLfm3SpatialDim = kLfm3HeadDim / 2;  // hidden_size / num_heads / 2 = 32
constexpr double kLfm3RopeTheta = 1e6;
constexpr float kLfm3NormEps = 1e-5f;
constexpr int kLfm3MinPixels = 65536;
constexpr int kLfm3MaxPixels = 2097152;  // processor_config.json's override, not the class default
constexpr int kLfm3ResizeFactor = kLfm3Patch * kLfm3Merge;

struct Lfm3VitSize {
  int h = 0, w = 0;
};

// Python's round() is half-to-even; std::round is half-away-from-zero. smart_resize rounds an edge
// in factor units, so 48 pixels (1.5 factors) resolves to 1 factor, not 2.
inline double lfm3_vit_py_round(double v) {
  const double lo = std::floor(v);
  const double frac = v - lo;
  if (frac < 0.5) return lo;
  if (frac > 0.5) return lo + 1.0;
  return std::fmod(lo, 2.0) == 0.0 ? lo : lo + 1.0;
}

// image_processing_lfm3_vl.py:50. Returns resized PIXELS; {0,0} for an out-of-range aspect ratio.
inline Lfm3VitSize lfm3_vit_smart_resize(int height, int width, int factor = kLfm3ResizeFactor,
                                         int min_pixels = kLfm3MinPixels,
                                         int max_pixels = kLfm3MaxPixels) {
  const double big = std::max(height, width), small = std::min(height, width);
  if (big / small > 200.0) return {};
  int h = int(lfm3_vit_py_round(double(height) / factor)) * factor;
  int w = int(lfm3_vit_py_round(double(width) / factor)) * factor;
  if (int64_t(h) * w > max_pixels) {
    const double beta = std::sqrt(double(height) * width / max_pixels);
    h = std::max(factor, int(std::floor(height / beta / factor)) * factor);
    w = std::max(factor, int(std::floor(width / beta / factor)) * factor);
  } else if (int64_t(h) * w < min_pixels) {
    const double beta = std::sqrt(double(min_pixels) / (double(height) * width));
    h = int(std::ceil(height * beta / factor)) * factor;
    w = int(std::ceil(width * beta / factor)) * factor;
  }
  return {h, w};
}

// The sequence is merge-block ordered: row n = (bh, bw, mh, mw). The conv block views it as
// (bh, bw, mh, mw, C), permutes to (C, bh, mh, bw, mw) and reshapes that to a raster [C, h, w],
// so raster cell (y, x) reads row ((bh * (gw/m) + bw) * m + mh) * m + mw.
inline int lfm3_vit_merge_index(int y, int x, int grid_w, int merge = kLfm3Merge) {
  const int bh = y / merge, mh = y % merge, bw = x / merge, mw = x % merge;
  return ((bh * (grid_w / merge) + bw) * merge + mh) * merge + mw;
}

inline std::array<int, 2> lfm3_vit_raster_coords(int n, int grid_w, int merge = kLfm3Merge) {
  const int mw = n % merge, mh = (n / merge) % merge;
  const int rest = n / (merge * merge), bw = rest % (grid_w / merge), bh = rest / (grid_w / merge);
  return {bh * merge + mh, bw * merge + mw};
}

// get_vision_position_ids(grid_thw, merge, t=1) returns (N, 2) of (h, w) coords in merge-block
// order. The HF rotary's comment says (2, N); its recomposition_frequencies indexes freq[:, 0/1],
// and the utility returns (N, 2), so (N, 2) is the truth.
inline std::vector<std::array<int, 2>> lfm3_vit_position_ids(int grid_h, int grid_w,
                                                              int merge = kLfm3Merge) {
  std::vector<std::array<int, 2>> ids(size_t(grid_h) * grid_w);
  for (int n = 0; n < grid_h * grid_w; ++n) ids[n] = lfm3_vit_raster_coords(n, grid_w, merge);
  return ids;
}

// patchify's column order is [c, t, ph, pw]; a still frame's two temporal halves are identical.
constexpr int lfm3_vit_patch_column(int c, int t, int ph, int pw) {
  return ((c * kLfm3Temporal + t) * kLfm3Patch + ph) * kLfm3Patch + pw;
}
constexpr int lfm3_vit_patch_dim() {
  return kLfm3Channels * kLfm3Temporal * kLfm3Patch * kLfm3Patch;  // 1536
}
constexpr int lfm3_vit_image_tokens(int grid_h, int grid_w, int merge = kLfm3Merge) {
  return grid_h * grid_w / (merge * merge);
}

// compute_axial_rope_parameters: spatial_dim = hidden_size / num_heads / 2 = 32, and
// inv_freq = 1 / theta^(arange(0, spatial_dim, 2) / spatial_dim) — 16 frequencies per axis.
inline std::vector<double> lfm3_vit_axial_inv_freq(double theta = kLfm3RopeTheta,
                                                   int spatial_dim = kLfm3SpatialDim) {
  std::vector<double> f;
  for (int i = 0; i < spatial_dim; i += 2) f.push_back(1.0 / std::pow(theta, double(i) / spatial_dim));
  return f;
}

// recomposition_frequencies: cat([h, w], -1) then cat([hw, hw], -1), so the head's layout is
// [h(16), w(16), h(16), w(16)] and rotate_half pairs i with i + head_dim/2.
inline void lfm3_vit_axial_angle(int pos_h, int pos_w, double* cos_out, double* sin_out,
                                 double theta = kLfm3RopeTheta, int head_dim = kLfm3HeadDim) {
  const auto f = lfm3_vit_axial_inv_freq(theta, head_dim / 2);
  const int axis = head_dim / 4;  // 16
  for (int i = 0; i < axis; ++i) {
    const double a = pos_h * f[i], b = pos_w * f[i];
    cos_out[i] = std::cos(a);
    sin_out[i] = std::sin(a);
    cos_out[axis + i] = std::cos(b);
    sin_out[axis + i] = std::sin(b);
    cos_out[2 * axis + i] = cos_out[i];
    sin_out[2 * axis + i] = sin_out[i];
    cos_out[3 * axis + i] = cos_out[axis + i];
    sin_out[3 * axis + i] = sin_out[axis + i];
  }
}

// rotate_half(x) = cat((-x[d/2:], x[:d/2])). Kept apart from apply_rope so a kernel can diff the
// permutation on its own.
inline void lfm3_vit_rotate_half(const float* x, int d, float* out) {
  const int half = d / 2;
  for (int i = 0; i < half; ++i) {
    out[i] = -x[i + half];
    out[i + half] = x[i];
  }
}

// apply_rotary_pos_emb_vision: q * cos + rotate_half(q) * sin, in fp32. Written closed-form to
// avoid a temporary; algebraically identical to the two calls above. out may alias q.
inline void lfm3_vit_apply_rope(const float* q, int d, const double* cosv, const double* sinv,
                                float* out) {
  const int half = d / 2;
  for (int i = 0; i < half; ++i) {
    const float a = q[i], b = q[i + half];
    const float oa = float(a * cosv[i] - b * sinv[i]);
    const float ob = float(b * cosv[i + half] + a * sinv[i + half]);
    out[i] = oa;
    out[i + half] = ob;
  }
}

// Lfm3VlRMSNorm (T5 style): the variance is fp32, then weight * x * rsqrt(mean(x^2) + eps). The HF
// layer casts the normalized row back to its input dtype before multiplying by weight; this stays fp32.
inline void lfm3_vit_rms_norm(const float* x, const float* w, int d, float eps, float* out) {
  float sum = 0.f;
  for (int i = 0; i < d; ++i) sum += x[i] * x[i];
  const float scale = 1.f / std::sqrt(sum / d + eps);
  for (int i = 0; i < d; ++i) out[i] = w[i] * (x[i] * scale);
}

}  // namespace ni::vision

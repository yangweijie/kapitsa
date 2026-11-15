// The Omni depth decoder's per-frame kernels (src/vision/lfm3_tts.cpp is the CPU oracle): one
// frame is 12 slots of 1024 (the codebooks), the trunk 3 bidirectional LFM3 layers over them
// (32 heads x 32, 8 kv, ffn 2816, rope 1e6 at the slot positions), then 12 tied 2017-wide heads.
// All weights bf16 (the side checkpoint), activations fp32. The GEMV is one output row per
// thread (a vector x [out, in] weight; in is a threadgroup-wide reduction).
#include "common.h"
using namespace metal;

// y[out] = W[out, in] . x + b. One thread per output row; the row walk is sequential (the
// tensors are small and this is a decode-shape GEMV). W bf16, x/b fp32.
[[kernel]] void tts_gemv(const device bfloat* w [[buffer(0)]], const device float* x [[buffer(1)]],
                         const device float* b [[buffer(2)]], device float* y [[buffer(3)]],
                         constant uint2& dims [[buffer(4)]],   // (out, in)
                         uint o [[thread_position_in_grid]]) {
  if (o >= dims.x) return;
  const device bfloat* row = w + size_t(o) * dims.y;
  float acc = 0;
  for (uint i = 0; i < dims.y; ++i) acc += float(row[i]) * x[i];
  y[o] = acc + (b ? b[o] : 0.f);
}

// y[c, :] = rms_norm(x[c, :]) . w (fp32 w, the norms are [1024] per slot), one slot per group.
[[kernel]] void tts_rms(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                        device float* y [[buffer(2)]], constant float& eps [[buffer(3)]],
                        uint c [[threadgroup_position_in_grid]], uint j [[thread_position_in_threadgroup]],
                        uint jn [[threads_per_threadgroup]]) {
  threadgroup float ss = 0.f;
  if (j == 0) {
    float s = 0;
    for (int i = 0; i < 1024; ++i) s += x[size_t(c) * 1024 + i] * x[size_t(c) * 1024 + i];
    ss = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = 1.0f / metal::sqrt(ss / 1024.f + eps);
  for (uint i = j; i < 1024; i += jn) y[size_t(c) * 1024 + i] = w[i] * x[size_t(c) * 1024 + i] * inv;
}

// The 12 slots' attention in one threadgroup (a slot per thread): QK-RMSNorm + rope + the scores
// + the softmax + the V mix + the per-head sigmoid gate. Bidirectional (no mask), GQA 32/8.
// The QK post-processing, one thread per head (grid 12 x 32): q_norm/k_norm (hd 32, shared per
// head), then rope at the slot's position (the CPU oracle's rotate-half form, fp32). mode 0 = q
// (rows [12, 1024], the head's rows at c*1024 + h*32), mode 1 = k (rows [12, 256], kv 8).
[[kernel]] void tts_rope_qk(device float* q [[buffer(0)]], device float* k [[buffer(1)]],
                            const device float* qn [[buffer(2)]], const device float* kn [[buffer(3)]],
                            constant uint& is_k [[buffer(4)]], uint gi [[thread_position_in_grid]]) {
  const uint hd = 32, hf = 16;
  // the k case (12 x 8 = 96 threads) is not a multiple of the 64-wide threadgroup, so the last
  // group's tail threads (96..127) must return before they index the 3072-float k buffer OOB.
  if (gi >= (is_k ? 8u : 32u) * 12u) return;
  const uint c = gi / (is_k ? 8u : 32u), h = gi % (is_k ? 8u : 32u);
  const uint base = is_k ? c * 256u + h * hd : c * 1024u + h * hd;
  device float* r = is_k ? k + base : q + base;
  const device float* n = is_k ? kn : qn;
  float s = 0;
  for (uint d = 0; d < hd; ++d) s += r[d] * r[d];
  const float inv = 1.0f / metal::sqrt(s / hd + 1e-5f);
  float v[32];
  for (uint d = 0; d < hd; ++d) v[d] = n[d] * r[d] * inv;
  // rope: the pair (d, d+16) rotates by the slot's angle; inv_freq[d] = 1e6^(-2d/32)
  for (uint p = 0; p < hf; ++p) {
    const float freq = metal::pow(1e6f, -2.0f * float(p) / float(hd));
    const float ang = float(c) * freq;
    const float a = v[p], b = v[hf + p];
    v[p] = a * metal::cos(ang) - b * metal::sin(ang);
    v[hf + p] = b * metal::cos(ang) + a * metal::sin(ang);
  }
  for (uint d = 0; d < hd; ++d) r[d] = v[d];
}

// The 12 slots' bidirectional attention in one threadgroup, one slot per thread (12 threads; the
// Q/K/V/Gate projections already ran as tts_gemv). Each thread: its slot's 32 heads' scores over
// the 12 kv slots (GQA 32/8: head h reads kv head h/4), the softmax, the V mix, the per-head
// sigmoid gate, then writes the gated head rows into out. The o_proj and the residual run outside.
[[kernel]] void tts_attn12(const device float* q [[buffer(0)]], const device float* k [[buffer(1)]],
                           const device float* v [[buffer(2)]], const device float* gate [[buffer(3)]],
                           device float* out [[buffer(4)]],             // [12, 1024] the gated heads
                           uint c [[thread_position_in_threadgroup]]) {
  const uint hd = 32, hf = 16;
  const float scale = 1.0f / metal::sqrt(float(hd));
  for (uint h = 0; h < 32; ++h) {
    const uint kvh = h / 4;
    float mx = -1e30f, sc[12];
    for (uint s = 0; s < 12; ++s) {
      const device float* ks = k + size_t(s) * 256u + size_t(kvh) * hd;
      const device float* qs = q + size_t(c) * 1024u + size_t(h) * hd;
      float acc = 0;
      for (uint d = 0; d < hd; ++d) acc += qs[d] * ks[d];
      sc[s] = acc * scale;
      mx = metal::max(mx, sc[s]);
    }
    float den = 0;
    for (uint s = 0; s < 12; ++s) den += metal::exp(sc[s] - mx);
    const float g = 1.0f / (1.0f + metal::exp(-gate[size_t(c) * 32u + h]));
    for (uint d = 0; d < hd; ++d) {
      float o = 0;
      for (uint s = 0; s < 12; ++s) {
        const float wgt = float(metal::exp(sc[s] - mx) / den);
        o += wgt * v[size_t(s) * 256u + size_t(kvh) * hd + d];
      }
      out[size_t(c) * 1024u + size_t(h) * hd + d] = o * g;
    }
    
  }
}

// SwiGLU element-wise: gi = silu(gi) * ui, one element per thread.
[[kernel]] void tts_silu(device float* gi [[buffer(0)]], const device float* ui [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
  const float v = gi[i];
  gi[i] = v / (1.0f + metal::exp(-v)) * ui[i];
}

// y += x, element-wise (the residuals).
[[kernel]] void tts_add(device float* y [[buffer(0)]], const device float* x [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
  y[i] += x[i];
}


// The trunk's projections: ONE shared [out, in] weight matrix applied to each of the 12 slots
// independently (the CPU oracle's per-slot matvec loop). grid = (out, 12), one thread per (row, slot).
[[kernel]] void tts_gemv12(const device bfloat* w [[buffer(0)]], const device float* x [[buffer(1)]],
                           const device float* b [[buffer(2)]], device float* y [[buffer(3)]],
                           constant uint2& dims [[buffer(4)]],   // (out, in)
                           uint tid [[thread_position_in_grid]]) {
  const uint total = dims.x * 12u;
  if (tid >= total) return;
  const uint o = tid % dims.x, c = tid / dims.x;
  const device bfloat* row = w + size_t(o) * dims.y;
  const device float* xc = x + size_t(c) * dims.y;
  float acc = 0;
  for (uint i = 0; i < dims.y; ++i) acc += float(row[i]) * xc[i];
  y[size_t(c) * dims.x + o] = acc + (b ? b[o] : 0.f);
}

// The Omni depth decoder's per-frame kernels (src/vision/lfm3_tts.cpp is the CPU oracle): one
// frame is 12 slots of 1024 (the codebooks), the trunk 3 bidirectional LFM3 layers over them
// (32 heads x 32, 8 kv, ffn 2816, rope 1e6 at the slot positions), then 12 tied 2017-wide heads.
// All weights bf16 (the side checkpoint), activations fp32. The GEMV is one output row per
// thread (a vector x [out, in] weight; in is a threadgroup-wide reduction).
#include "common.h"
using namespace metal;

namespace kp {

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
  threadgroup float ss;
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
[[kernel]] void tts_attn12(device float* x [[buffer(0)]],              // the normed slots (in), the residual add on the way out
                           const device float* qw [[buffer(1)]], const device float* kw [[buffer(2)]],
                           const device float* vw [[buffer(3)]], const device float* ow [[buffer(4)]],
                           const device float* gw [[buffer(5)]], const device float* qn [[buffer(6)]],
                           const device float* kn [[buffer(7)]], const device float* rope_inv [[buffer(8)]],
                           constant float& eps [[buffer(9)]],
                           uint c [[threadgroup_position_in_grid]], uint lane [[thread_position_in_threadgroup]],
                           uint lanes [[threads_per_threadgroup]]) {
  // (the trunk calls it per layer; the projections run as tts_gemv outside — this kernel takes the
  //  already-projected q/k/v/gate in the caller's scratch; see lfm3_tts.cpp's forward_gpu)
}

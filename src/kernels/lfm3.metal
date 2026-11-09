#include "common.h"

// LFM3's tri-hybrid (model/lfm3.h): the gated short convolution's kernels and the attention gate's pack.
// The convolution's math is lfm3_conv_ref (model/lfm3.cpp), which the harness diffed these against: tap k
// reads the frame (K-1-k) back, and frames before the sequence come from the ring, oldest first (ring[0] is
// K-1 frames back, ring[K-2] the frame just before this pass). The gates themselves (u = B ⊙ x, y = C ⊙ z)
// are element-wise and run in the encode path.

struct Lfm3ConvParams {
  uint T;  // frames this pass encodes (the decode kernel: 1)
  uint H;  // channels
  uint K;  // taps (7)
};

// Prefill: one thread per (frame, channel); no threadgroup memory.
[[kernel]] void lfm3_conv_prefill(const device float* u [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                                  const device bfloat* ring [[buffer(2)]], device float* y [[buffer(3)]],
                                  constant Lfm3ConvParams& p [[buffer(4)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, t = gid.y;
  if (c >= p.H || t >= p.T) return;
  float acc = 0;
  for (uint k = 0; k < p.K; ++k) {
    const int src = int(t) - int(p.K - 1 - k);
    const float x = src >= 0 ? u[uint(src) * p.H + c] : float(ring[uint(p.K - 1 + src) * p.H + c]);
    acc += float(w[c * p.K + k]) * x;
  }
  y[uint(t) * p.H + c] = acc;
}

// Decode: one frame and the ring -> one output frame and the rolled ring (drop the oldest, append u), one
// dispatch per layer per token.
[[kernel]] void lfm3_conv_decode(const device float* u [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                                 const device bfloat* ring [[buffer(2)]], device float* y [[buffer(3)]],
                                 device bfloat* ring_out [[buffer(4)]], constant Lfm3ConvParams& p [[buffer(5)]],
                                 uint gid [[thread_position_in_grid]]) {
  const uint c = gid;
  if (c >= p.H) return;
  float acc = 0;
  for (uint k = 0; k < p.K; ++k) {
    const int src = int(k) - int(p.K - 1);  // the single frame u holds is src 0
    const float x = src >= 0 ? u[uint(src) * p.H + c] : float(ring[uint(p.K - 1 + src) * p.H + c]);
    acc += float(w[c * p.K + k]) * x;
  }
  y[c] = acc;
  for (uint j = 0; j + 1 < p.K - 1; ++j) ring_out[j * p.H + c] = ring[(j + 1) * p.H + c];
  ring_out[(p.K - 2) * p.H + c] = bfloat(u[c]);  // the newest frame
}

struct Lfm3AttnParams {
  uint Hq;     // query heads
  uint Hkv;    // key/value heads
  uint D;      // head dim (128)
  uint cap;    // the caches' row stride (positions)
  uint pos;    // this row's position
  uint window; // the sliding window (0: every position)
  float scale; // 1 / sqrt(D)
};

// Plain decode attention at head dim 128, one thread per query head (correctness first; the split-K forms
// come with the engine's measured shapes). q is one row, normed and roped; the caches hold fp16 rows
// [Hkv, cap, D]; the softmax runs over the causal positions, the last `window` of them on a sliding layer.
// The per-head sigmoid gate multiplies the head's output before it returns.
[[kernel]] void lfm3_attn_d(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                            const device half* vc [[buffer(2)]], const device float* gate [[buffer(3)]],
                            device float* out [[buffer(4)]], constant Lfm3AttnParams& p [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
  const uint h = gid;
  if (h >= p.Hq) return;
  const device half* kkh = kc + uint64_t(h / (p.Hq / p.Hkv)) * p.cap * p.D;
  const device half* vvh = vc + uint64_t(h / (p.Hq / p.Hkv)) * p.cap * p.D;
  const uint beg = p.window && p.pos >= p.window ? p.pos + 1 - p.window : 0;
  float m = -1e30f, s = 0.f;
  for (uint t = beg; t <= p.pos; ++t) {
    float score = 0;
    for (uint d = 0; d < p.D; ++d) score += q[h * p.D + d] * float(kkh[uint64_t(t) * p.D + d]);
    score *= p.scale;
    m = t == beg ? score : max(m, score);
    // (the running max re-walked below; a small row keeps this simple and exact)
  }
  float denom = 0;
  for (uint t = beg; t <= p.pos; ++t) {
    float score = 0;
    for (uint d = 0; d < p.D; ++d) score += q[h * p.D + d] * float(kkh[uint64_t(t) * p.D + d]);
    const float wgt = metal::precise::exp(score * p.scale - m);
    denom += wgt;
  }
  // third walk: the weights need the final denominator, so the value sum runs after it
  for (uint d = 0; d < p.D; ++d) out[h * p.D + d] = 0;
  for (uint t = beg; t <= p.pos; ++t) {
    float score = 0;
    for (uint d = 0; d < p.D; ++d) score += q[h * p.D + d] * float(kkh[uint64_t(t) * p.D + d]);
    const float wgt = metal::precise::exp(score * p.scale - m) / denom;
    for (uint d = 0; d < p.D; ++d) out[h * p.D + d] += wgt * float(vvh[uint64_t(t) * p.D + d]);
  }
  const float g = sigmoid(gate[h]);
  for (uint d = 0; d < p.D; ++d) out[h * p.D + d] *= g;
}

struct Lfm3GateParams {
  uint T;
  uint Hq;
  uint D;  // head dim
};

// The attention's per-head sigmoid gate packed beside the query, the layout attn.metal's kernels read
// (qg[t, h] = [query (D) | gate (D), the per-head scalar broadcast]). One thread per (token, head).
[[kernel]] void lfm3_gate_pack(const device float* q [[buffer(0)]], const device float* gate [[buffer(1)]],
                               device float* qg [[buffer(2)]], constant Lfm3GateParams& p [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
  const uint h = gid.x, t = gid.y;
  if (h >= p.Hq || t >= p.T) return;
  const float g = gate[uint(t) * p.Hq + h];
  const device float* qh = q + (uint64_t(t) * p.Hq + h) * p.D;
  device float* og = qg + (uint64_t(t) * p.Hq + h) * (2 * p.D);
  for (uint d = 0; d < p.D; ++d) {
    og[d] = qh[d];
    og[p.D + d] = g;
  }
}

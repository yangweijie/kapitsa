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

struct Lfm3NormParams {
  uint D;
  float eps;
};

// The sandwich point: out = x + w ⊙ rmsnorm(z) (post_operator_norm's and post_ffn_norm's norm before the
// residual add). Same shape as rows_norm_f32; one threadgroup of up to 256 threads per row.
[[kernel]] void lfm3_res_norm(const device float* z [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                              const device float* x [[buffer(2)]], device float* out [[buffer(3)]],
                              constant Lfm3NormParams& p [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
  threadgroup float red[32];
  const device float* zr = z + r * p.D;
  float ss = 0.0f;
  for (uint i = tid; i < p.D; i += ntg) ss += zr[i] * zr[i];
  ss = tg_sum(ss, red, tid, ntg);
  const float inv = metal::precise::rsqrt(ss / float(p.D) + p.eps);
  for (uint i = tid; i < p.D; i += ntg) out[r * p.D + i] = x[r * p.D + i] + zr[i] * inv * float(w[i]);
}

// Elementwise fp32 product: u = B ⊙ x and y = C ⊙ conv(u).
[[kernel]] void lfm3_mul(const device float* a [[buffer(0)]], const device float* b [[buffer(1)]],
                         device float* h [[buffer(2)]], constant uint& n [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
  if (i < n) h[i] = a[i] * b[i];
}

struct Lfm3BcxParams {
  uint T;      // rows
  uint H;      // hidden, the width of one plane
  uint which;  // 0: u = B ⊙ x, 1: y = C ⊙ z
};

// The in_proj output is [T, 3H] with (B, C, x) inside each row, not three planar [T, H] buffers.
// which 0 reads B and x into u, which 1 reads C and multiplies the conv's z into y. One thread per
// (channel, row); the decode path keeps lfm3_mul, where T = 1 makes the two layouts agree.
[[kernel]] void lfm3_bcx(const device float* bcx [[buffer(0)]], const device float* z [[buffer(1)]],
                         device float* out [[buffer(2)]], constant Lfm3BcxParams& p [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
  const uint c = gid.x, t = gid.y;
  if (c >= p.H || t >= p.T) return;
  const uint64_t row = uint64_t(t) * (3 * p.H);
  const float a = p.which == 0 ? bcx[row + c] : bcx[row + p.H + c];
  const float b = p.which == 0 ? bcx[row + 2 * p.H + c] : z[uint64_t(t) * p.H + c];
  out[uint64_t(t) * p.H + c] = a * b;
}

struct Lfm3QkParams {
  uint T;      // rows this pass encodes
  uint Hq;
  uint Hkv;
  uint D;      // head dim (128)
  uint cap;    // the caches' row stride
  uint pos0;   // the first row's position
};

// QK-RMSNorm, the zero-tail RoPE (lfm3_rope_ref's semantics: pair (i, i + D/2) rotates by pos * freqs[i],
// the table's zero tail leaves its pairs unrotated) and the cache write. One thread per (row, head); the
// grid's x covers Hq q heads, Hkv k heads (written to kc roped) and Hkv v heads (written to vc raw).
[[kernel]] void lfm3_qk_prep(const device float* qkv [[buffer(0)]], const device bfloat* qw [[buffer(1)]],
                             const device bfloat* kw [[buffer(2)]], const device float* freqs [[buffer(3)]],
                             device half* kc [[buffer(4)]], device half* vc [[buffer(5)]],
                             device float* q [[buffer(6)]], constant Lfm3QkParams& p [[buffer(7)]],
                             uint2 gid [[thread_position_in_grid]]) {
  const uint hraw = gid.x, t = gid.y;
  const uint QD = p.Hq * p.D, KD = p.Hkv * p.D;
  const uint64_t row = uint64_t(t) * (QD + 2 * KD);
  const uint pos = p.pos0 + t;
  if (hraw < p.Hq) {  // a q head: norm, rope, out
    const device float* x = qkv + row + uint64_t(hraw) * p.D;
    float ss = 0;
    for (uint d = 0; d < p.D; ++d) ss += x[d] * x[d];
    const float inv = metal::precise::rsqrt(ss / float(p.D) + 1e-5f);
    float n[128];
    for (uint d = 0; d < p.D && d < 128; ++d) n[d] = x[d] * inv * float(qw[d]);
    for (uint i = 0; i < p.D / 2; ++i) {
      const float ang = float(pos) * freqs[i], c = metal::precise::cos(ang), s = metal::precise::sin(ang);
      q[(uint64_t(t) * p.Hq + hraw) * p.D + i] = n[i] * c - n[p.D / 2 + i] * s;
      q[(uint64_t(t) * p.Hq + hraw) * p.D + p.D / 2 + i] = n[p.D / 2 + i] * c + n[i] * s;
    }
  } else if (hraw < p.Hq + p.Hkv) {  // a k head: norm, rope, into the cache
    const uint h = hraw - p.Hq;
    const device float* x = qkv + row + QD + uint64_t(h) * p.D;
    float ss = 0;
    for (uint d = 0; d < p.D; ++d) ss += x[d] * x[d];
    const float inv = metal::precise::rsqrt(ss / float(p.D) + 1e-5f);
    float n[128];
    for (uint d = 0; d < p.D && d < 128; ++d) n[d] = x[d] * inv * float(kw[d]);
    device half* kh = kc + (uint64_t(h) * p.cap + pos % p.cap) * p.D;
    for (uint i = 0; i < p.D / 2; ++i) {
      const float ang = float(pos) * freqs[i], c = metal::precise::cos(ang), s = metal::precise::sin(ang);
      kh[i] = half(n[i] * c - n[p.D / 2 + i] * s);
      kh[p.D / 2 + i] = half(n[p.D / 2 + i] * c + n[i] * s);
    }
  } else if (hraw < p.Hq + 2 * p.Hkv) {  // a v head: into the cache raw
    const uint h = hraw - p.Hq - p.Hkv;
    const device float* x = qkv + row + QD + KD + uint64_t(h) * p.D;
    device half* vh = vc + (uint64_t(h) * p.cap + pos % p.cap) * p.D;
    for (uint d = 0; d < p.D; ++d) vh[d] = half(x[d]);
  }
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

// Plain decode attention at head dim 128, split over the positions: kernel 1 gives each (head, split) a
// contiguous chunk of the causal range and writes its partial output, running max and sum; kernel 2 merges
// the splits (the online-softmax merge) and applies the per-head sigmoid gate. SPLITS = the grid's y (the
// params' splits); a split whose chunk lies past `pos` contributes nothing (its max is -inf). q is one row,
// normed and roped; the caches hold fp16 rows [Hkv, cap, D]; a sliding layer's ring wraps (slot = t % cap).
struct Lfm3AttnSplitParams {
  uint Hq;
  uint Hkv;
  uint D;
  uint cap;
  uint pos;
  uint window;
  uint splits;
  float scale;
};
[[kernel]] void lfm3_attn_d_split(const device float* q [[buffer(0)]], const device half* kc [[buffer(1)]],
                                  const device half* vc [[buffer(2)]], device float* part_o [[buffer(3)]],
                                  device float* part_m [[buffer(4)]], device float* part_s [[buffer(5)]],
                                  constant Lfm3AttnSplitParams& p [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
  const uint h = gid.x, sp = gid.y;
  if (h >= p.Hq || sp >= p.splits) return;
  const device half* kkh = kc + uint64_t(h / (p.Hq / p.Hkv)) * p.cap * p.D;
  const device half* vvh = vc + uint64_t(h / (p.Hq / p.Hkv)) * p.cap * p.D;
  const uint beg = p.window && p.pos >= p.window ? p.pos + 1 - p.window : 0;
  const uint len = p.pos - beg + 1, chunk = (len + p.splits - 1) / p.splits;
  const uint lo = beg + sp * chunk;
  const uint hi = metal::min(lo + chunk, p.pos + 1);
  const device float* qh = q + h * p.D;
  float m = -1e30f;
  for (uint t = lo; t < hi; ++t) {
    const uint slot = t % p.cap;
    float score = 0;
    for (uint d = 0; d < p.D; ++d) score += qh[d] * float(kkh[uint64_t(slot) * p.D + d]);
    m = max(m, score * p.scale);
  }
  device float* oh = part_o + (uint64_t(h) * p.splits + sp) * p.D;
  for (uint d = 0; d < p.D; ++d) oh[d] = 0;
  float denom = 0;
  for (uint t = lo; t < hi; ++t) {
    const uint slot = t % p.cap;
    float score = 0;
    for (uint d = 0; d < p.D; ++d) score += qh[d] * float(kkh[uint64_t(slot) * p.D + d]);
    const float wgt = metal::precise::exp(score * p.scale - m);
    denom += wgt;
    for (uint d = 0; d < p.D; ++d) oh[d] += wgt * float(vvh[uint64_t(slot) * p.D + d]);
  }
  part_m[uint64_t(h) * p.splits + sp] = hi > lo ? m : -1e30f;
  part_s[uint64_t(h) * p.splits + sp] = hi > lo ? denom : 0.f;
}

// The merge: out[h] = (sum_s part_o * exp(part_m - M)) / (sum_s part_s * exp(part_m - M)) * sigmoid(gate),
// one thread per (head, dim).
[[kernel]] void lfm3_attn_d_merge(const device float* part_o [[buffer(0)]], const device float* part_m [[buffer(1)]],
                                  const device float* part_s [[buffer(2)]], const device float* gate [[buffer(3)]],
                                  device float* out [[buffer(4)]], constant Lfm3AttnSplitParams& p [[buffer(5)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  const uint h = gid.x, d = gid.y;
  if (h >= p.Hq || d >= p.D) return;
  float M = -1e30f;
  for (uint sp = 0; sp < p.splits; ++sp) M = max(M, part_m[uint64_t(h) * p.splits + sp]);
  float num = 0, den = 0;
  for (uint sp = 0; sp < p.splits; ++sp) {
    const float wgt = metal::precise::exp(part_m[uint64_t(h) * p.splits + sp] - M);
    num += wgt * part_o[(uint64_t(h) * p.splits + sp) * p.D + d];
    den += wgt * part_s[uint64_t(h) * p.splits + sp];
  }
  out[h * p.D + d] = num / den * sigmoid(gate[h]);
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

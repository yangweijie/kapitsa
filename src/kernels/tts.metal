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

// ---- the MaskGIT reveal on the GPU (one readout per frame instead of one per pass; the CPU
// sampler in sample_frame is the oracle) ----

// the repetition penalty over the all-masked frame's logits, HF's n_penalty form: a code in
// codebook c's recent window (the host-packed int rows, -1 terminated, up to 64 codes each)
// divides the positive logits by the penalty (penalty > 1) or multiplies the negative ones
// (penalty < 1), deduplicated per codebook. One thread per (c, k).
[[kernel]] void tts_pen_logits(device float* lg [[buffer(0)]], const device int* hist [[buffer(1)]],
                               constant float& pen [[buffer(2)]],
                               uint id [[thread_position_in_grid]]) {
  if (pen == 1.0f || id >= 12u * 2017u) return;
  const uint c = id / 2017u, k = id % 2017u;
  const device int* hc = hist + size_t(c) * 64u;
  bool seen = false;
  for (int i = 0; i < 64; ++i) {
    const int v = hc[i];
    if (v < 0) break;
    if (v == int(k)) { seen = true; break; }  // dedup: a code in several window frames once
  }
  if (!seen) return;
  const float v = lg[id];
  if (pen > 1.0f && v > 0.0f) lg[id] = v / pen;
  else if (pen < 1.0f && v < 0.0f) lg[id] = v * pen;
}

// the adds staging for a pass: per codebook row, the revealed code's embedding (the 12 embedding
// tables are the bound slots) or the mask embedding. One thread per (c, hidden). The codebook
// embedding is a bf16 checkpoint tensor -- the SAME tensor tts_gemv binds as the slot's tied head
// weight -- so read it as bfloat (2 B/elt) to match its layout; a `device float*` read doubled the
// byte index and ran ~4 MB past the [2017,1024] table -> the BIF0 page fault that wedged the GPU.
// The adds (W.x) and the mask embedding stay fp32; the gathered bf16 value is widened to fp32.
[[kernel]] void tts_fill_adds(device float* adds [[buffer(0)]],
                              const device bfloat* emb0 [[buffer(1)]],  const device bfloat* emb1 [[buffer(2)]],
                              const device bfloat* emb2 [[buffer(3)]],  const device bfloat* emb3 [[buffer(4)]],
                              const device bfloat* emb4 [[buffer(5)]],  const device bfloat* emb5 [[buffer(6)]],
                              const device bfloat* emb6 [[buffer(7)]],  const device bfloat* emb7 [[buffer(8)]],
                              const device bfloat* emb8 [[buffer(9)]],  const device bfloat* emb9 [[buffer(10)]],
                              const device bfloat* emb10 [[buffer(11)]], const device bfloat* emb11 [[buffer(12)]],
                              const device float* mask [[buffer(13)]],
                              const device int* codes [[buffer(14)]],  const device uchar* revealed [[buffer(15)]],
                              uint id [[thread_position_in_grid]]) {
  if (id >= 12u * 1024u) return;
  const uint c = id / 1024u, h = id % 1024u;
  if (!revealed[c]) { adds[id] = mask[h]; return; }
  const int k = codes[c];
  const device bfloat* e;
  switch (c) {
    case 0: e = emb0; break; case 1: e = emb1; break; case 2: e = emb2; break; case 3: e = emb3; break;
    case 4: e = emb4; break; case 5: e = emb5; break; case 6: e = emb6; break; case 7: e = emb7; break;
    case 8: e = emb8; break; case 9: e = emb9; break; case 10: e = emb10; break; default: e = emb11; break;
  }
  adds[id] = float(e[size_t(k) * 1024u + h]);
}

// the pass's reveal: for each codebook the host marked selected this pass (the MaskGIT schedule
// slice), pick its code from the (penalized) logits — the greedy argmax (mode 0, the CPU's
// first-max walk, bit-exact in fp32) or the CPU sampler's fixed-stream temperature draw
// (mode 1: the host's frac is the CPU's one uniform_real draw of a fresh mt19937(0); the mx is
// the fp32 max of the fp32 quotients — the CPU's double mx over the same quotients, exact — and
// the exp/sum/walk run in fp32 against the CPU's double: a selection can flip only when two
// cumulative sums sandwich the draw within that ~1e-7 drift, and the path stays deterministic
// run to run). The code + the revealed flag land in the GPU state the NEXT pass's
// tts_fill_adds reads; the frame's codes are the single readout. grid = 12 groups of one
// thread. MSL has no double, so the walk is fp32 throughout.
[[kernel]] void tts_reveal(const device float* lg [[buffer(0)]],  // [12, 2017] the pass's penalized logits
                          device int* codes [[buffer(1)]],      // [12]
                          device uchar* revealed [[buffer(2)]],  // [12]
                          const device uchar* sel [[buffer(3)]], // [12] this pass's selected codebooks
                          constant uint& mode [[buffer(4)]],     // 0 = the greedy argmax, 1 = the temperature draw
                          constant float& frac [[buffer(5)]],
                          constant float& temp [[buffer(6)]],
                          uint c [[threadgroup_position_in_grid]]) {
  if (c >= 12u || !sel[c]) return;
  const device float* row = lg + size_t(c) * 2017u;
  if (mode == 0u) {  // the greedy: the CPU's first-max argmax (lg[k] > bv, k ascending)
    float bv = -1e30f;
    int best = 0;
    for (int k = 0; k < 2017; ++k)
      if (row[k] > bv) { bv = row[k]; best = k; }
    codes[c] = best;
  } else {
    float mx = -1e30f;
    for (int k = 0; k < 2017; ++k) {
      const float q = row[k] / temp;
      if (q > mx) mx = q;
    }
    float sum = 0;
    for (int k = 0; k < 2017; ++k) sum += metal::exp(row[k] / temp - mx);
    const float draw = frac * sum;
    float run = 0;
    int best = 2016;
    for (int k = 0; k < 2017; ++k) {
      run += metal::exp(row[k] / temp - mx);
      if (draw <= run) { best = k; break; }
    }
    codes[c] = best;
  }
  revealed[c] = 1;
}

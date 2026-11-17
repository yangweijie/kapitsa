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

// tts_gemv12's 2D-tiled form (the <10 ms fusion's go/no-go kernel): one 128-thread threadgroup
// computes a 16-output x 12-slot block. Each 64-wide K chunk's 16 W rows are loaded ONCE into
// threadgroup memory and shared by the 12 slots (12x less W traffic than gemv12's per-slot
// re-reads of the same rows), and the chunk is co-reduced across 8 K lanes before it folds
// into the per-output accumulator. dims = (out, in); grid = out/16 threadgroups, 128 threads.
// Static threadgroup memory is 12,032 B (the MTL layer's double-count stays under the 32,768
// B cap); ragged tiles (out/in not multiples of 16/64) zero-pad in the guards.
[[kernel]] void tts_gemv12_t(const device bfloat* w [[buffer(0)]], const device float* x [[buffer(1)]],
                             const device float* b [[buffer(2)]], device float* y [[buffer(3)]],
                             constant uint2& dims [[buffer(4)]],
                             uint g [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]]) {
  const uint out = dims.x, in = dims.y;
  const uint TM = 16, TK = 64, KL = 8;
  const uint o0 = g * TM;
  threadgroup bfloat wt[TM * TK];
  threadgroup float xt[12 * TK];
  threadgroup float rt[12 * TM * KL];
  threadgroup float yt[12 * TM];
  for (uint i = tid; i < 12u * TM; i += 128u) yt[i] = 0.f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint oi = tid / KL, kl = tid % KL;
  for (uint ck = 0; ck * TK < in; ++ck) {
    const uint k0 = ck * TK;
    for (uint i = tid; i < TM * TK; i += 128u) {
      const uint o = o0 + i / TK, k = k0 + i % TK;
      wt[i] = (o < out && k < in) ? w[size_t(o) * in + k] : bfloat(0);
    }
    for (uint i = tid; i < 12u * TK; i += 128u) {
      const uint c = i / TK, k = k0 + i % TK;
      xt[i] = (k < in) ? x[size_t(c) * in + k] : 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // this thread's K sub-chunk (8 of the 64) partial-accumulated per slot, staged for the reduce
    for (uint c = 0; c < 12; ++c) {
      float s = 0;
      for (uint j = 0; j < KL; ++j) s += float(wt[oi * TK + kl * KL + j]) * xt[c * TK + kl * KL + j];
      rt[(size_t(c) * TM + oi) * KL + kl] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kl == 0u)
      for (uint c = 0; c < 12; ++c) {
        float s = 0;
        for (uint j = 0; j < KL; ++j) s += rt[(size_t(c) * TM + oi) * KL + j];
        yt[c * TM + oi] += s;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = tid; i < 12u * TM; i += 128u) {
    const uint c = i / TM, o = o0 + i % TM;
    if (o < out) y[size_t(c) * out + o] = yt[i] + (b ? b[o] : 0.f);
  }
}

// tts_gemv12's residual form (the add12 fold): y[c, o] ACCUMULATES the dot, so the GEMV's
// dispatch doubles as its own residual add (the caller's add12 is gone). y (the running state)
// is written once per (o, c) thread and NOT read by the products, so the += is race-free.
[[kernel]] void tts_gemv12_r(const device bfloat* w [[buffer(0)]], const device float* x [[buffer(1)]],
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
  y[size_t(c) * dims.x + o] += acc + (b ? b[o] : 0.f);
}

// tts_gemv12_t's residual form: the tile's final store accumulates into y (the running state).
// Same grain, guards and tmem as tts_gemv12_t; only the store is the +=.
[[kernel]] void tts_gemv12_t_r(const device bfloat* w [[buffer(0)]], const device float* x [[buffer(1)]],
                               const device float* b [[buffer(2)]], device float* y [[buffer(3)]],
                               constant uint2& dims [[buffer(4)]],
                               uint g [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]]) {
  const uint out = dims.x, in = dims.y;
  const uint TM = 16, TK = 64, KL = 8;
  const uint o0 = g * TM;
  threadgroup bfloat wt[TM * TK];
  threadgroup float xt[12 * TK];
  threadgroup float rt[12 * TM * KL];
  threadgroup float yt[12 * TM];
  for (uint i = tid; i < 12u * TM; i += 128u) yt[i] = 0.f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint oi = tid / KL, kl = tid % KL;
  for (uint ck = 0; ck * TK < in; ++ck) {
    const uint k0 = ck * TK;
    for (uint i = tid; i < TM * TK; i += 128u) {
      const uint o = o0 + i / TK, k = k0 + i % TK;
      wt[i] = (o < out && k < in) ? w[size_t(o) * in + k] : bfloat(0);
    }
    for (uint i = tid; i < 12u * TK; i += 128u) {
      const uint c = i / TK, k = k0 + i % TK;
      xt[i] = (k < in) ? x[size_t(c) * in + k] : 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = 0; c < 12; ++c) {
      float s = 0;
      for (uint j = 0; j < KL; ++j) s += float(wt[oi * TK + kl * KL + j]) * xt[c * TK + kl * KL + j];
      rt[(size_t(c) * TM + oi) * KL + kl] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kl == 0u)
      for (uint c = 0; c < 12; ++c) {
        float s = 0;
        for (uint j = 0; j < KL; ++j) s += rt[(size_t(c) * TM + oi) * KL + j];
        yt[c * TM + oi] += s;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = tid; i < 12u * TM; i += 128u) {
    const uint c = i / TM, o = o0 + i % TM;
    if (o < out) y[size_t(c) * out + o] += yt[i] + (b ? b[o] : 0.f);
  }
}

// tts_rms's batched form for the 12 slots' OWN embedding norms (the per-slot loop's one
// dispatch): group c is slot c's row of x/y ([12, 1024]); the per-row gain is w's row c
// (the contiguous [12, 1024] emb_norm upload). Same math as tts_rms (the mean-square over
// the row's 1024, the per-row gain, the eps).
[[kernel]] void tts_rms12(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                          device float* y [[buffer(2)]], constant float& eps [[buffer(3)]],
                          uint c [[threadgroup_position_in_grid]], uint j [[thread_position_in_threadgroup]],
                          uint jn [[threads_per_threadgroup]]) {
  threadgroup float ss = 0.f;
  if (j == 0) {
    float s = 0;
    const device float* rc = x + size_t(c) * 1024;
    for (int i = 0; i < 1024; ++i) s += rc[i] * rc[i];
    ss = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = 1.0f / metal::sqrt(ss / 1024.f + eps);
  for (uint i = j; i < 1024; i += jn) y[size_t(c) * 1024 + i] = w[size_t(c) * 1024 + i] * x[size_t(c) * 1024 + i] * inv;
}

// the 12 slots' tied [2017, 1024] head GEMVs in ONE dispatch (the per-slot tts_gemv loop):
// one thread per (o, c); the weight is the contiguous [12, 2017, 1024] emb_all (tts_fill_adds's
// gather shares it), x is the slot's own [1024] row, y the [12, 2017] logits.
[[kernel]] void tts_heads12(const device bfloat* emb_all [[buffer(0)]], const device float* x [[buffer(1)]],
                            const device float* b [[buffer(2)]], device float* y [[buffer(3)]],
                            constant uint2& dims [[buffer(4)]],   // (out, in)
                            uint tid [[thread_position_in_grid]]) {
  const uint out = dims.x, in = dims.y;
  if (tid >= out * 12u) return;
  const uint o = tid % out, c = tid / out;
  const device bfloat* row = emb_all + (size_t(c) * out + o) * in;
  const device float* xc = x + size_t(c) * in;
  float acc = 0;
  for (uint i = 0; i < in; ++i) acc += float(row[i]) * xc[i];
  y[size_t(c) * out + o] = acc + (b ? b[o] : 0.f);
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

// the adds staging for a pass: per codebook row, the revealed code's embedding (the 12 codebook
// tables as ONE contiguous [12, 2017, 1024] bf16 upload, tts_heads12's shared weight; the
// per-slot 12 bindings existed only for the per-slot head loop) or the mask embedding, THEN the
// depth GEMV's row (proj: the slot-init add12's W.proj, folded in -- the pass's adds are
// embedding + depth projection in one dispatch). One thread per (c, hidden). The codebook
// embedding is a bf16 checkpoint tensor -- read it as bfloat (2 B/elt) to match its layout; a
// `device float*` read doubled the byte index and ran ~4 MB past the table -> the BIF0 page
// fault that wedged the GPU (f0020f7f). The adds (W.x), the mask and the proj stay fp32; the
// gathered bf16 value is widened to fp32.
[[kernel]] void tts_fill_adds(device float* adds [[buffer(0)]],
                              const device bfloat* emb_all [[buffer(1)]],
                              const device float* mask [[buffer(2)]],
                              const device int* codes [[buffer(3)]],  const device uchar* revealed [[buffer(4)]],
                              const device float* proj [[buffer(5)]],
                              uint id [[thread_position_in_grid]]) {
  if (id >= 12u * 1024u) return;
  const uint c = id / 1024u, h = id % 1024u;
  const float base = revealed[c]
                        // the reveal wrote codes[c]: 0 <= k < 2017
                        ? float(emb_all[(size_t(c) * 2017u + size_t(codes[c])) * 1024u + h])
                        : mask[h];
  adds[id] = base + proj[id];
}

// the pass's reveal: for each codebook the host marked selected this pass (the MaskGIT schedule
// slice), pick its code from the (penalized) logits — the greedy argmax (mode 0, the CPU's
// first-max walk, bit-exact in fp32) or the per-codebook temperature draw (mode 1: the host's
// draws[c] is codebook c's uniform, splitmix64 over (seed, c) — the reference's torch.multinomial
// is a fresh uniform per selection, so the mx is the fp32 max of the fp32 quotients and the
// exp/sum/walk run in fp32 against the CPU's double walk of the same draw: a selection can flip
// only when two cumulative sums sandwich it within that ~1e-7 drift, and the path stays
// deterministic run to run for a request seed). The code + the revealed flag land in the GPU
// state the NEXT pass's tts_fill_adds reads; the frame's codes are the single readout. grid = 12
// groups of one thread. MSL has no double, so the walk is fp32 throughout.
[[kernel]] void tts_reveal(const device float* lg [[buffer(0)]],  // [12, 2017] the pass's penalized logits
                          device int* codes [[buffer(1)]],      // [12]
                          device uchar* revealed [[buffer(2)]],  // [12]
                          const device uchar* sel [[buffer(3)]], // [12] this pass's selected codebooks
                          constant uint& mode [[buffer(4)]],     // 0 = the greedy argmax, 1 = the temperature draw
                          const device float* draws [[buffer(5)]], // [12] each codebook's uniform, in [0, 1)
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
    const float draw = draws[c] * sum;
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

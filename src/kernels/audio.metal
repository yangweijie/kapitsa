// The LFM3-Omni audio INPUT tower's Metal kernels, stage 1: the downsampler (the CPU forward in
// src/vision/lfm3_audio.cpp is the oracle). 3 stride-2 depthwise k3 convs over (mel, time):
// conv_in reads the raw log-mel [frames, M] scalar, conv_mid/conv_end read the C-channel state;
// each C state also runs a C x C 1x1 mix + SiLU (conv_in's 1x1 is the identity). lin_out then
// GEMV's each token's channel-major (mel, C) flatten to the hidden width. Weights bf16 (the
// checkpoint), activations fp32. The 17-block encoder + projector follow in a later commit.
#include "common.h"
using namespace metal;

// shared dims for a downsample stage: a = in time frames, b = in mel (or m) rows, c = channels,
// d = out time frames; out m = (b + 1) / 2
struct Ds {
  uint a, b, c, d;
};

// one thread per output (t, m, c): the raw mel's depthwise k3 s2 p1 over both axes + the bias,
// SiLU fused (conv_in has no 1x1). Layout: in [a, b] row-major, out [d, (b+1)/2, c] with the
// index (t * om + m) * c + cch.
[[kernel]] void audio_ds_conv(const device float* mel [[buffer(0)]],     // [a, b]
                              const device bfloat* dw [[buffer(1)]],     // [c, 9]  (dm*3+dt)
                              const device bfloat* db [[buffer(2)]],     // [c]
                              device float* out [[buffer(3)]],          // [d, (b+1)/2, c]
                              constant Ds& g [[buffer(4)]],            // (a, b, c, d)
                              uint id [[thread_position_in_grid]]) {
  const uint om = (g.b + 1) / 2;
  if (id >= g.d * om * g.c) return;
  const uint c = id % g.c, m = (id / g.c) % om, t = id / (om * g.c);
  float s = float(db[c]);
  for (uint dm = 0; dm < 3; ++dm) {
    const int mi = int(m * 2 - 1 + dm);
    if (mi < 0 || mi >= int(g.b)) continue;
    for (uint dt = 0; dt < 3; ++dt) {
      const int ti = int(t * 2 - 1 + dt);
      if (ti < 0 || ti >= int(g.a)) continue;
      s += float(dw[c * 9 + dm * 3 + dt]) * mel[size_t(ti) * g.b + mi];
    }
  }
  out[id] = silu(s);
}

// one thread per output (t, m, c): the C-channel state's depthwise k3 s2 p1 over both axes +
// the bias (stages 2/3, conv_mid/conv_end). SiLU runs after the 1x1 mix (audio_ds_lin), so
// this writes the raw conv sum. in [a, b, c] with the index (ti * b + mi) * c + cch.
[[kernel]] void audio_ds_conv2(const device float* in [[buffer(0)]],    // [a, b, c]
                               const device bfloat* dw [[buffer(1)]],    // [c, 9]
                               const device bfloat* db [[buffer(2)]],    // [c]
                               device float* out [[buffer(3)]],         // [d, (b+1)/2, c]
                               constant Ds& g [[buffer(4)]],           // (a, b, c, d)
                               uint id [[thread_position_in_grid]]) {
  const uint om = (g.b + 1) / 2;
  if (id >= g.d * om * g.c) return;
  const uint c = id % g.c, m = (id / g.c) % om, t = id / (om * g.c);
  float s = float(db[c]);
  for (uint dm = 0; dm < 3; ++dm) {
    const int mi = int(m * 2 - 1 + dm);
    if (mi < 0 || mi >= int(g.b)) continue;
    for (uint dt = 0; dt < 3; ++dt) {
      const int ti = int(t * 2 - 1 + dt);
      if (ti < 0 || ti >= int(g.a)) continue;
      s += float(dw[c * 9 + dm * 3 + dt]) * in[(size_t(ti) * g.b + mi) * g.c + c];
    }
  }
  out[id] = s;
}

// out[r, c] = silu(b[c] + W[c, :] . conv[r, :]), the per-token 1x1 channel mix (C x C, the
// lin_mid/lin_end pairs). One thread per (r, c); R = out time x out mel.
[[kernel]] void audio_ds_lin(const device float* in [[buffer(0)]],     // [R, c]
                             const device bfloat* w [[buffer(1)]],     // [c, c]
                             const device bfloat* b [[buffer(2)]],     // [c]
                             device float* out [[buffer(3)]],
                             constant uint2& dims [[buffer(4)]],      // (R, c)
                             uint id [[thread_position_in_grid]]) {
  if (id >= dims.x * dims.y) return;
  const uint c = id % dims.y, r = id / dims.y;
  float s = float(b[c]);
  for (uint j = 0; j < dims.y; ++j) s += float(w[c * dims.y + j]) * in[r * dims.y + j];
  out[id] = silu(s);
}

// the downsampled embedding: one token = the channel-major (mel16, C) flatten of its s3 patch,
// GEMV'd to the hidden width + the bias. out[t, h] = b[h] + sum_{cm} W[h, cm] * s3[t, m, c]
// (cm = c * m3 + m, the HF flatten(1,2) order the CPU oracle matches). One thread per (t, h).
[[kernel]] void audio_ds_out(const device bfloat* w [[buffer(0)]],    // [h, fm]  fm = m3 * c
                             const device float* in [[buffer(1)]],    // [t3, m3, c]
                             const device bfloat* b [[buffer(2)]],    // [h]
                             device float* out [[buffer(3)]],        // [t3, h]
                             constant uint2& dims [[buffer(4)]],     // (h, t3)
                             constant uint2& sh [[buffer(5)]],       // (m3, c)
                             uint id [[thread_position_in_grid]]) {
  // the dispatch group count is ceil(h * t3 / 256), so the last group's tail threads (past
  // t3) must return before they index the s3 / out buffers OOB.
  if (id >= dims.x * dims.y) return;
  const uint h = id % dims.x, t = id / dims.x;
  const uint fm = sh.x * sh.y;
  float acc = float(b[h]);
  for (uint cm = 0; cm < fm; ++cm) {
    const uint c = cm / sh.x, m = cm % sh.x;
    acc += float(w[size_t(h) * fm + cm]) * in[(size_t(t) * sh.x + m) * sh.y + c];
  }
  out[id] = acc;
}

// ---- the 17 encoder blocks (the CPU's Lfm3Audio::blocks is the oracle; HD 96, the SWA window
// 32 and the GQA ratio are the config's fixed values, the kernels hardcode them as tts.metal
// does with its 12 slots) ----

// batched GEMM: y[r, o] = sum_j W[o, in] * x[r, j] (bf16 W, fp32 x/y, no bias — the blocks'
// projections are all bias-free). One thread per output row-element, the in-walk sequential (the
// decode-shape GEMM, as tts_gemv). grid = R * O.
[[kernel]] void audio_gemm(const device bfloat* w [[buffer(0)]], const device float* x [[buffer(1)]],
                           device float* y [[buffer(2)]], constant uint3& dims [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
  if (id >= dims.x * dims.y) return;
  const uint o = id % dims.y, r = id / dims.y;
  const device bfloat* row = w + size_t(o) * dims.z;
  const device float* xr = x + size_t(r) * dims.z;
  float acc = 0;
  for (uint j = 0; j < dims.z; ++j) acc += float(row[j]) * xr[j];
  y[id] = acc;
}

// y += x, element-wise (the residuals). grid = n.
[[kernel]] void audio_add(device float* y [[buffer(0)]], const device float* x [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint id [[thread_position_in_grid]]) {
  if (id >= n) return;
  y[id] += x[id];
}

// y[r, :] = rms_norm(x[r, :]) * w (T5's form: w * x * rsqrt(mean(x^2) + eps), fp32 w). One
// threadgroup per row; dispatch exactly R groups (the kernel's barriers assume no tail group).
[[kernel]] void audio_rms(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                          device float* y [[buffer(2)]], constant uint2& dims [[buffer(3)]],
                          constant float& eps [[buffer(4)]],
                          uint r [[threadgroup_position_in_grid]],
                          uint j [[thread_position_in_threadgroup]],
                          uint jn [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  float s = 0;
  for (uint i = j; i < dims.y; i += jn) s += x[size_t(r) * dims.y + i] * x[size_t(r) * dims.y + i];
  s = tg_sum(s, scratch, j, jn);
  const float inv = 1.0f / metal::sqrt(s / float(dims.y) + eps);
  for (uint i = j; i < dims.y; i += jn) y[size_t(r) * dims.y + i] = w[i] * x[size_t(r) * dims.y + i] * inv;
}

// the conv blocks' gated short conv (the HF's k3 causal, the CPU's dt walk): the in_proj's 3
// planes [B, C, x] per token (p + r*3H, + r*3H + H, + r*3H + 2H), u = B * x, the causal k-conv
// over u with the per-channel kernel w[i*K + dt] (dt counts up to the current token), the gate
// y = C * conv(u). One thread per (t, i); the two kernels run over the T*H grid.
[[kernel]] void audio_gate_mul(const device float* p [[buffer(0)]],  // [T, 3H] the in_proj rows
                               device float* u [[buffer(1)]],        // [T, H]
                               constant uint2& dims [[buffer(2)]],   // (T, H)
                               uint id [[thread_position_in_grid]]) {
  if (id >= dims.x * dims.y) return;
  const uint r = id / dims.y, i = id % dims.y;
  const size_t base = size_t(r) * 3u * dims.y;
  u[id] = p[base + i] * p[base + 2u * dims.y + i];
}

[[kernel]] void audio_cconv(const device float* u [[buffer(0)]],     // [T, H]
                            const device float* p [[buffer(1)]],     // [T, 3H] the in_proj rows
                            const device bfloat* w [[buffer(2)]],    // [H, K]
                            device float* out [[buffer(3)]],
                            constant uint3& dims [[buffer(4)]],      // (T, H, K)
                            uint id [[thread_position_in_grid]]) {
  if (id >= dims.x * dims.y) return;
  const uint t = id / dims.y, i = id % dims.y;
  float s = 0;
  for (uint dt = 0; dt < dims.z; ++dt) {
    const int ti = int(t) - int(dims.z - 1) + int(dt);
    if (ti < 0) continue;
    s += float(w[size_t(i) * dims.z + dt]) * u[size_t(ti) * dims.y + i];
  }
  // the C plane is token-interleaved in p (the in_proj's [T, 3H] rows: B [0,H), C [H,2H),
  // x [2H,3H)) — not a contiguous [T, H] block
  out[id] = p[size_t(t) * 3u * dims.y + dims.y + i] * s;
}

// the attention blocks' q/k post-processing: the per-head RMSNorm (nw [HD], the layernorms are
// bf16 in the checkpoint, widened fp32 on upload) then the full rope on the rotate-half pairs
// (p, p + HD/2) at the token's position, inv_freq[p] = 1e6^(-2p/HD). One thread per (t, head).
[[kernel]] void audio_qk_rope(device float* x [[buffer(0)]],       // [T, heads, HD]
                              const device float* nw [[buffer(1)]],
                              constant uint2& dims [[buffer(2)]],  // (T, heads)
                              constant float& eps [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
  const uint HD = 96, HF = 48;
  if (id >= dims.x * dims.y) return;
  const uint t = id / dims.y;
  device float* xb = x + id * HD;
  float s = 0;
  for (uint d = 0; d < HD; ++d) s += xb[d] * xb[d];
  const float inv = 1.0f / metal::sqrt(s / HD + eps);
  for (uint d = 0; d < HD; ++d) xb[d] = nw[d] * xb[d] * inv;
  for (uint p = 0; p < HF; ++p) {
    const float freq = metal::pow(1e6f, -2.0f * float(p) / float(HD));
    const float ang = float(t) * freq;
    const float cs = metal::cos(ang), sn = metal::sin(ang);
    const float a = xb[p], b = xb[HF + p];
    xb[p] = a * cs - b * sn;
    xb[HF + p] = b * cs + a * sn;
  }
}

// the attention blocks' windowed GQA softmax mix (the CPU's sliding window 32, causal: token t
// reads [t0..t], q head hh reads kv head hh / (HQ/HKV)). One thread per (t, hh); the window's
// scores live on the stack (the window is 32, the kernel's sc[] is sized to it).
[[kernel]] void audio_attn_swa(const device float* q [[buffer(0)]],   // [T, HQ, HD]
                               const device float* k [[buffer(1)]],   // [T, HKV, HD]
                               const device float* v [[buffer(2)]],
                               device float* out [[buffer(3)]],       // [T, HQ, HD]
                               constant uint4& dims [[buffer(4)]],    // (T, HQ, HKV, window)
                               uint id [[thread_position_in_grid]]) {
  const uint HD = 96;
  if (id >= dims.x * dims.y) return;
  const uint t = id / dims.y, hh = id % dims.y;
  const uint hk = hh / (dims.y / dims.z);
  const uint t0 = t + 1 >= dims.w ? t + 1 - dims.w : 0;
  const uint n = t - t0 + 1;
  const device float* qs = q + id * HD;
  const float scale = 1.0f / metal::sqrt(float(HD));
  float sc[32], mx = -1e30f;
  for (uint s = t0; s <= t; ++s) {
    float acc = 0;
    const device float* ks = k + (size_t(s) * dims.z + hk) * HD;
    for (uint d = 0; d < HD; ++d) acc += qs[d] * ks[d];
    sc[s - t0] = acc * scale;
    mx = metal::max(mx, sc[s - t0]);
  }
  float den = 0;
  for (uint s = 0; s < n; ++s) den += metal::exp(sc[s] - mx);
  device float* ob = out + id * HD;
  for (uint d = 0; d < HD; ++d) {
    float o = 0;
    for (uint s = t0; s <= t; ++s)
      o += metal::exp(sc[s - t0] - mx) / den * v[(size_t(s) * dims.z + hk) * HD + d];
    ob[d] = o;
  }
}

// the FFN's SwiGLU: y = silu(a) * b, one element per thread (the w1/w3 outputs).
[[kernel]] void audio_silu_mul(device float* a [[buffer(0)]], const device float* b [[buffer(1)]],
                               constant uint& n [[buffer(2)]], uint id [[thread_position_in_grid]]) {
  if (id >= n) return;
  a[id] = silu(a[id]) * b[id];
}

// ---- the projector (LayerNorm -> L1 -> tanh-GELU -> L2 -> bf16; the CPU's encode() is the
// oracle) ----

// per-row LayerNorm (the mean/var form, the CPU's fp32 sums + the fp32 sqrt): one threadgroup
// per row, the two sums reduced in the scratch (mean at [0..31], the squares at [32..63]).
[[kernel]] void audio_proj_ln(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                              const device float* b [[buffer(2)]], device float* out [[buffer(3)]],
                              constant uint2& dims [[buffer(4)]],  // (T, dim)
                              constant float& eps [[buffer(5)]],
                              uint r [[threadgroup_position_in_grid]],
                              uint j [[thread_position_in_threadgroup]],
                              uint jn [[threads_per_threadgroup]]) {
  threadgroup float scratch[64];
  const device float* xr = x + size_t(r) * dims.y;
  float s1 = 0, s2 = 0;
  for (uint i = j; i < dims.y; i += jn) {
    s1 += xr[i];
    s2 += xr[i] * xr[i];
  }
  const float mean = tg_sum(s1, scratch, j, jn) / float(dims.y);
  const float var = tg_sum(s2, scratch + 32, j, jn) / float(dims.y);
  const float rstd = 1.0f / metal::sqrt(metal::max(var - mean * mean, 0.0f) + eps);
  for (uint i = j; i < dims.y; i += jn)
    out[size_t(r) * dims.y + i] = w[i] * (xr[i] - mean) * rstd + b[i];
}

// batched GEMM with the fp32 bias: y[r, o] = b[o] + sum_j W[o, in] * x[r, j].
[[kernel]] void audio_gemm_b(const device bfloat* w [[buffer(0)]], const device float* x [[buffer(1)]],
                             const device float* b [[buffer(2)]], device float* y [[buffer(3)]],
                             constant uint3& dims [[buffer(4)]],  // (R, O, in)
                             uint id [[thread_position_in_grid]]) {
  if (id >= dims.x * dims.y) return;
  const uint o = id % dims.y, r = id / dims.y;
  float acc = float(b[o]);
  const device bfloat* row = w + size_t(o) * dims.z;
  const device float* xr = x + size_t(r) * dims.z;
  for (uint j = 0; j < dims.z; ++j) acc += float(row[j]) * xr[j];
  y[id] = acc;
}

// the projector's GELU (HF's tanh form: 0.5 x (1 + tanh(0.79788456 (x + 0.044715 x^3)))).
[[kernel]] void audio_gelu(device float* y [[buffer(0)]], constant uint& n [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
  if (id >= n) return;
  const float v = y[id], x3 = v * v * v;
  y[id] = 0.5f * v * (1.0f + metal::tanh(0.7978845608028654f * (v + 0.044715f * x3)));
}

// the bf16 cast (the encode()'s output rows: fp32 -> the checkpoint's bf16, round-half-even,
// the CPU's (u + 0x7FFF + ((u >> 16) & 1)) >> 16).
[[kernel]] void audio_f2bf(const device float* x [[buffer(0)]], device ushort* out [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint id [[thread_position_in_grid]]) {
  if (id >= n) return;
  union { float f; uint u; } cv;
  cv.f = x[id];
  out[id] = ushort((cv.u + 0x7FFFu + ((cv.u >> 16) & 1u)) >> 16);
}

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

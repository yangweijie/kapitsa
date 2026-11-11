// Liquid AI's LFM3 (Lfm3ForCausalLM, model_type "lfm3"): a tri-hybrid decoder, 24 layers of a gated short
// convolution (in_proj 3H -> H split B, C, x; u = B ⊙ x through a depthwise conv of conv_L_cache taps; y = C ⊙ z;
// out_proj), full attention (GQA, QK RMSNorm, zero-tail partial RoPE at theta 1e6: the first
// partial_rotary_factor x head_dim dims rotate over the full head's pairing, the rest pass through) and sliding
// attention (RoPE at theta 1e4), every layer a SwiGLU FFN, RMSNorm before and after the mixer and the FFN (the
// sandwich), a final norm, tied embeddings. modeling_lfm3.py is the reference; src/gguf/import.cpp converts
// llama.cpp's lfm3 GGUFs (the per-layer head_count_kv gives the schedule, rope_freqs.weight the partial rotation).
// This file holds what needs no GPU: the config, the memory plan from the headers alone, and the cache sizing.
#pragma once
#include "loader/safetensors.h"
#include "model/model.h"

#include <map>
#include <string>
#include <vector>

namespace ni {

struct Lfm3Config {
  int hidden = 0, vocab = 0, layers = 0, hq = 0, hkv = 0, hd = 0;
  int ffn = 0;
  int window = 0;            // sliding window, a row's own position included
  int conv_k = 7;            // the short convolution's taps (conv_L_cache)
  double theta = 1e6;        // full attention's RoPE base
  double theta_swa = 1e4;    // sliding attention's RoPE base
  double partial = 1.0;      // full attention's partial_rotary_factor (a fraction of the head)
  float eps = 1e-5f;
  std::vector<char> kind;    // per layer: 'C' conv, 'F' full attention, 'S' sliding attention
  std::vector<int> eos;
  int bos = -1;
  bool tied = true;
  int bits = 16, gs = 64;    // the linears' format (config.json's quantization; 4-bit group 32 from a Q4_1 GGUF)
};

struct Lfm3 {
  static bool is_model(const std::string& dir);  // config.json's model_type is "lfm3" (model/lfm3_config.h)
  static Lfm3Config read_config(const std::string& dir);
  static MemPlan plan(const std::string& path, int prefill_chunk = 2048);
  // Cache bytes: full-attention layers per position; sliding-window layers a ring of window + 8 positions per
  // conversation; conv layers a ring of conv_k - 1 frames of the hidden size per conversation.
  static size_t kv_per_pos(const Lfm3Config& c);
  static size_t ring_bytes(const Lfm3Config& c);
  static size_t conv_state_bytes(const Lfm3Config& c);
};

// LFM3's zero-tail RoPE inverse frequencies (modeling_lfm3.py's compute_default_rope_parameters): the first
// int(partial * head_dim) / 2 frequencies follow the full head's progression (theta^(-2i / head_dim)) and the
// rest are zero. Pairs are (i, i + head_dim / 2) over the WHOLE head — not Qwen's prefix pairing over the
// rotated dims — so a zero frequency leaves its pair unrotated and the tail dims pass through.
std::vector<float> lfm3_rope_freqs(double theta, double partial, int head_dim);

// The gated short convolution's causal depthwise pass, as the CPU reference the Metal kernels are diffed
// against (the moe_exl3.h pattern): y[t][c] = sum_k w[c][k] * u[t - (k_size - 1 - k)], where u before the
// sequence comes from `state` — the ring of k_size - 1 previous frames, oldest first — or 0 when state is
// null. u and y are [T, C] rows; w is [C, k_size]; everything fp32. Decoding is this at T = 1 with the ring
// the caller rolls (its new frames are the last k_size - 1 of state ++ u); the reference and the decode
// stepping must agree with one prefill, which tools/lfm3_test.cpp pins.
void lfm3_conv_ref(const float* u, int t_count, int channels, int k_size, const float* w, const float* state,
                   float* y);

// The zero-tail RoPE applied to one q or k row, as the CPU reference the attention kernels are diffed
// against: pair (i, i + head_dim / 2) of the row rotates by pos * freqs[i] (x[i]' = x[i] cos - x[i + d/2] sin,
// x[i + d/2]' = x[i + d/2] cos + x[i] sin), so the table's zero tail (lfm3_rope_freqs) leaves the pairs it
// covers — dims [d/4, d/2) and [3d/4, d) — unrotated, while the rotated pairs keep the full head's frequency
// progression. x and out are [head_dim] fp32 rows; out may alias x.
void lfm3_rope_ref(const float* x, int head_dim, const float* freqs, double pos, float* out);

}  // namespace ni

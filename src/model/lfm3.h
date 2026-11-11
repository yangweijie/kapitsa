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

}  // namespace ni

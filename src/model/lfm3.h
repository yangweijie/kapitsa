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

// One quantized linear (the import keeps the file's codes: 4-bit group 32, or 8-bit group 64, or bf16). K comes
// from the module's role; bits and gs from the tensors' shapes (w U32 [N, K*bits/32], s/b BF16 [N, K/gs]).
struct Lfm3Lin {
  const Tensor* w = nullptr;
  const Tensor* s = nullptr;
  const Tensor* b = nullptr;
  uint32_t N = 0, K = 0;
  int bits = 16, gs = 64;
  bool dense() const { return bits == 16; }
};

struct Lfm3Layer {
  char kind = 'C';  // 'C' conv, 'F' full attention, 'S' sliding attention
  const Tensor *in_norm = nullptr, *post_norm = nullptr, *ffn_norm = nullptr, *post_ffn_norm = nullptr;
  Lfm3Lin in_proj, out_proj;  // conv
  const Tensor* conv_w = nullptr;
  Lfm3Lin q, k, v, o, gate;  // attention (gate: hidden -> Hq, on the normed input)
  const Tensor *qn = nullptr, *kn = nullptr;
  Lfm3Lin w1, w2, w3;  // SwiGLU
};

// The backend. Linears run the file's own formats (4-bit group 32 and 8-bit group 64 through qmr's
// gs-generic kernels, bf16 through qrf); the conv, the QK prep and the attention run lfm3.metal's
// kernels, verified against lfm3_conv_ref / lfm3_rope_ref. The decode pass is R <= 8 rows, the prefill
// chunks of tmax_ rows; the sliding-window layers keep a ring of window + 8 positions, the conv layers a
// ring of conv_k - 1 frames, the full-attention layers a growing cache.
class Lfm3 : public Model {
 public:
  static constexpr int kMaxRows = 8;  // rows of a decode / verify pass

  static bool is_model(const std::string& dir);  // config.json's model_type names the LFM3 family
  static Lfm3Config read_config(const std::string& dir);
  static MemPlan plan(const std::string& path, int prefill_chunk = 2048);
  // Cache bytes: full-attention layers per position; sliding-window layers a ring of window + 8 positions per
  // conversation; conv layers a ring of conv_k - 1 frames of the hidden size per conversation.
  static size_t kv_per_pos(const Lfm3Config& c);
  static size_t ring_bytes(const Lfm3Config& c);
  static size_t conv_state_bytes(const Lfm3Config& c);

  explicit Lfm3(const std::string& path, int prefill_chunk = 2048);
  ~Lfm3() override = default;

  int vocab() const override { return cfg_.vocab; }
  const std::vector<int>& eos_ids() const override { return cfg_.eos; }
  void set_mem_budget(size_t bytes) override { mem_budget_ = bytes; }
  std::unique_ptr<Seq> new_seq(int limit, int initial = -1) override;
  bool reserve(Seq& s, int need) override;
  void drop_kv(Seq& s) override;
  size_t kv_bytes_per_pos() const override { return kv_per_pos(cfg_); }
  void reset(Seq& s) override;
  Checkpoint new_checkpoint() override { return Checkpoint{}; }
  void save(Encoder& e, const Seq& s, Checkpoint& c, int pos) override;
  void restore(Encoder& e, Seq& s, const Checkpoint& c) override;
  void encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample = true) override;
  void encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                      Checkpoint* snap = nullptr, int snap_row = 0) override;
  int prefill_chunk() const override { return tmax_; }
  bool can_snapshot() const override { return false; }
  const Lfm3Config& config() const { return cfg_; }
  const float* logits_for_debug() const { return logits_.as<float>(); }
  std::string formats_summary() const;

 private:
  void load();
  Pipeline pipe(const std::string& n);
  void lin(Encoder& e, const Lfm3Lin& W, const Buffer& xh, const Buffer& y, int rows, bool add, uint32_t y_ld);
  void to_half(Encoder& e, const Buffer& x, const Tensor* norm, const Buffer& y, int rows, uint32_t d = 0);
  void res_norm(Encoder& e, const Buffer& z, const Tensor* w, const Buffer& x, const Buffer& out, int rows);
  void embed(Encoder& e, const Buffer& tokens, int tok_off, int rows, const Buffer& out);
  void enc_sample(Encoder& e, Seq& s, const Buffer& xrow, int out_index, const SamplingParams& sp, uint32_t step);
  int conv_ordinal(int layer) const;  // the layer's index among the conv layers (its ring's slot)
  uint32_t window_of(int layer) const { return cfg_.kind[size_t(layer)] == 'S' ? uint32_t(cfg_.window) : 0; }

  std::string path_;
  Lfm3Config cfg_;
  WeightStore ws_;
  std::vector<Lfm3Layer> layers_;
  Lfm3Lin embed_;                 // the tied head reads the embedding
  const Tensor* final_norm_ = nullptr;
  Buffer freqs_f_, freqs_s_;      // the two RoPE tables, fp32 [hd / 2]
  int n_conv_ = 0;
  int tmax_ = 2048;
  size_t mem_budget_ = 0;
  // decode activations (kMaxRows rows)
  Buffer x_, xh_, bcx_, u_, z_, ym_, mx_, aoh_, qkv_, gate_, q_, ao_, g_, u2_, h_, hh_, xn_, logits_,
      cand_v_, cand_i_, sprob_;
  // prefill activations (tmax_ rows)
  Buffer px_, pxh_, pbcx_, pu_, pz_, py_, pmix_, paoh_, pqkv_, pgate_, pq_, pao_, pg_, pu2_, ph_, phh_;
  bool sample_sg_ = false;        // topk_stage1_rows_sg (an Apple10 GPU on macOS 26), as Qwen35 reads it
};

}  // namespace ni

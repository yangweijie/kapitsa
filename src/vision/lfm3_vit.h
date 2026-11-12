// The LFM3-Omni image encoder (LiquidAI's Lfm3Vl vision tower + processor): 14 blocks alternating
// ConvNeXt-V2 depthwise 3x3 (RMSNorm + SwiGLU) and axial-2D-RoPE attention (QK-RMSNorm, GQA 16/8,
// head_dim 64, non-causal), a dual-temporal patchifier (two identical frames, its two conv kernels
// interleaved as [c,t,ph,pw]), and a GELU projector to the language model's 2048 width (2x2 spatial
// merge: 512x512 -> 256 image tokens). Weights bf16 in a co-located vision.safetensors (the mmproj
// import, not the model folder's text weights), run on the prefill GEMMs + small kernels in vit.metal.
// Its geometry and rotary math are pinned in lfm3_vit_ref.h; the kernels diff against those.
#pragma once
#include "loader/safetensors.h"
#include "vision/encoder.h"
#include "vision/vit_common.h"

#include <memory>
#include <string>
#include <vector>

namespace ni::vision {

struct Lfm3ViTConfig {
  // the tower (vision_config)
  int depth = 14, hidden = 1024, inter = 2560, heads = 16, kv_heads = 8, out_hidden = 2048;
  int patch = 16, temporal = 2, merge = 2, in_ch = 3;
  int rope_theta_pow = 20;  // rope theta 1e6
  float rms_eps = 1e-5f;    // the tower's RMSNorms (embedding_norm, operator/ffn, q/k layernorms)
  std::vector<char> layer_types;  // per block 'C' (depthwise conv) / 'S' (attention), 14 entries
  int image_token = 125030;
  // the projector
  int proj_hidden = 4096;
  float proj_ln_eps = 1e-6f;  // the projector's single LayerNorm (weight + bias)
  // processor
  long min_pixels = 65536, max_pixels = 2097152;
  float mean[3] = {0.5f, 0.5f, 0.5f}, stdv[3] = {0.5f, 0.5f, 0.5f};
  double rescale = 1.0 / 255.0;
  bool do_rescale = true, do_normalize = true;
};

class Lfm3ViT : public ImageEncoder {
 public:
  Lfm3ViT(const std::string& dir, const Lfm3ViTConfig& c);
  ~Lfm3ViT() override;
  static bool supports(const std::string& dir, Lfm3ViTConfig* c);  // an lfm3_vl_vision tower in dir's files

  std::string name() const override { return "LFM3 ViT"; }
  std::string describe() const override;
  int placeholder() const override { return c_.image_token; }
  int out_dim() const override { return c_.out_hidden; }
  Prepared prepare(const Image& im) const override;
  size_t weight_bytes() const override;
  size_t work_bytes(const Prepared& p) const override;
  bool loaded() const override { return loaded_; }
  void load() override;
  void unload() override;
  void encode(const Prepared& p, const Buffer& out) override;

  const Lfm3ViTConfig& config() const { return c_; }
  // The processor's output as the reference returns it (pixel_values: [patches, C x T x P x P] in
  // (channel, frame, row, column) order, float32), merge-block ordered: for checks against it.
  std::vector<float> pixel_values(const Prepared& p) const override;
  // Tests: the hidden state after `layers` blocks (fp32 [patches, hidden]; layers = depth: before the
  // projector) instead of the embeddings, when set before encode() (out holds patches x hidden floats).
  void set_debug_stop(int layers) override { dbg_stop_ = layers; }
  int debug_rows(const Prepared& p) const override { return (p.height / c_.patch) * (p.width / c_.patch); }
  int debug_width() const override { return c_.hidden; }

 private:
  struct Block {
    Buffer onw;       // operator_norm (RMSNorm weight, [hidden])
    Buffer fnw;       // ffn_norm (RMSNorm weight, [hidden])
    VitLin qkv, o;    // attention (S blocks only): q|k|v concatenated [2*hidden, hidden] (GQA 16/8), out_proj
    Buffer qn, kn;    // q/k layernorm (RMSNorm, [head_dim])
    VitLin cinp;      // conv in_proj ([3*hidden, hidden]) (C blocks only)
    Buffer csp;       // conv spatial 3x3 depthwise ([hidden, 1, 3, 3])
    VitLin coutp;     // conv out_proj ([hidden, hidden])
    VitLin w1, w3, w2;  // the SwiGLU MLP (both block kinds)
    char kind = 0;
  };

  std::string dir_;
  Lfm3ViTConfig c_;
  uint32_t inter_pad_ = 0;
  VitWeights w_;  // opened with an empty prefix: the import's names are already vision_tower.* / multi_modal_projector.*
  bool loaded_ = false;
  VitLin patch_;  // [hidden, 3*temporal*patch*patch], permuted at load to (t, ph, pw, c) for qvit_patchify
  Buffer emb_norm_;  // embedding_norm (RMSNorm, [hidden])
  std::vector<Block> blocks_;
  Buffer pnw_, pnb_;  // the projector's LayerNorm weight + bias ([hidden])
  VitLin m0_, m2_;    // the projector linears mm.0 (proj_hidden -> proj_hidden) and mm.2 (-> out_hidden)
  int dbg_stop_ = -1;
};

}  // namespace ni::vision

// The LFM3-Omni audio OUTPUT side, CPU v1 (see lfm3_audio.h for the INPUT side): the depth
// decoder that turns one language-model hidden state into one speech frame's 12 codebook codes,
// and the MaskGIT sampler the reference runs over it. The codec that turns codes into a 24 kHz
// waveform is the separate Liquid-Codec model (the encoder side of it is what the audio tower
// reuses; its decoder follows in G4).
//
// The depth decoder's shapes (Lfm3OmniDepthDecoder in the reference): depth_linear maps the text
// hidden (2048) to 12 slots x 1024; masked slots carry a learned mask_embedding, revealed ones
// their code's embedding (one Embedding(2017, 1024) per codebook); a 3-layer LFM3 trunk (width
// 1024, 32 heads x 32, 8 kv, ffn 2816, rope 1e6, BIDIRECTIONAL — all codebooks of a frame are
// predicted jointly) runs over the slots; each codebook then reads logits through its RMSNorm and
// its tied Embedding. Weights: bf16 in the Omni checkpoint (NOT in the GGUF — the import brings
// them from the side checkpoint), converted to fp32 at load.
#pragma once
#include "metal/gpu.h"
#include <cstdint>
#include <string>
#include <vector>

namespace ni::tts {

class DepthDecoder {
 public:
  static constexpr int kCodebooks = 12, kCodebookSize = 2017, kHidden = 1024, kTextHidden = 2048,
                       kLayers = 3, kHeads = 32, kKvHeads = 8, kHeadDim = 32, kFfn = 2816;

  explicit DepthDecoder(const std::string& safetensors);  // the Omni checkpoint (model.depth_decoder.*)

  // hidden: F x 2048 (fp32); codes: F x 12 (revealed codes; a masked slot's value is ignored);
  // masked: F x 12 (1 = the codebook is hidden and scored). logits: F x 12 x 2017 (caller-sized).
  void forward(const std::vector<float>& hidden, const std::vector<int>& codes,
               const std::vector<uint8_t>& masked, std::vector<float>& logits) const;

  // One speech frame from a language-model hidden state (the reference's sample_audio_frame): the
  // 12 codebooks revealed in a seeded random order over the maskgit_schedule passes ([4, 4, 4] —
  // each pass scores the still-hidden codebooks jointly), argmax at temperature 0, softmax at > 0.
  // rep_hist (kCodebooks x recent codes, the reference's n_window) + rep_penalty down-weight a
  // code the codebook just emitted (HF's RepetitionPenaltyLogitsProcessor form); a penalty of 1.0
  // (the default) or an empty history is a no-op.
  std::vector<int> sample_frame(const std::vector<float>& hidden, float temperature, uint64_t seed,
                                const std::vector<std::vector<int>>& rep_hist = {},
                                float rep_penalty = 1.0f) const;
  // The same with a fixed reveal order (the checks' form: with order {0..11} the first pass's codes
  // are exactly the all-masked argmax of codebooks 0-3).
  std::vector<int> sample_frame(const std::vector<float>& hidden, float temperature,
                                const std::vector<int>& order,
                                const std::vector<std::vector<int>>& rep_hist = {},
                                float rep_penalty = 1.0f) const;
  // The GPU form (Metal kernels, src/kernels/tts.metal; the CPU forward stays the oracle). The
  // weights ride to the device once (load_gpu); each call is one frame's three MaskGIT passes.
  // The caller holds the GPU turn. F must be 1 (the decode shape).
  bool gpu_ready() const { return gpu_ready_; }
  void load_gpu();

  // The LLM's audio_out input embedding for a frame (the reference's get_audio_out_embeds): the sum
  // of the 12 codebook rows of `model.codebook_embedding` ([num_codebooks*codebook_size, text hidden],
  // row i*codebook_size + codes[i]). Empty when the checkpoint has no codebook_embedding table.
  const std::vector<float>& codebook_sum(const std::vector<int>& codes) const;
  bool has_codebook() const { return !codebook_emb_.empty(); }
  std::vector<int> sample_frame_gpu(const std::vector<float>& hidden, float temperature, uint64_t seed,
                                   const std::vector<std::vector<int>>& rep_hist = {},
                                   float rep_penalty = 1.0f);
  std::vector<int> sample_frame_gpu(const std::vector<float>& hidden, float temperature,
                                    const std::vector<int>& order,
                                    const std::vector<std::vector<int>>& rep_hist = {},
                                    float rep_penalty = 1.0f);

  // The audio repetition penalty (the reference's n_penalty on the codes each codebook emitted in
  // the last n_window frames), HF's RepetitionPenaltyLogitsProcessor form: for each codebook c and
  // each distinct code k in codebook_history[c], divide logits[c][k] by penalty (penalty > 1,
  // positive logits only) or multiply it (penalty < 1, negative logits only). Applied to the
  // all-masked logits before the MaskGIT argmax, so a code the codebook just emitted is down-weighted.
  // codebook_history: kCodebooks x (the codebook's recent codes, up to n_window — deduplicated here).
  static void apply_repetition_penalty(std::vector<float>& logits,
                                       const std::vector<std::vector<int>>& codebook_history,
                                       float penalty);

 private:
  struct Layer {
    std::vector<float> q_w, k_w, v_w, o_w, gate_w, w1, w2, w3, qn, kn, op_norm, ffn_norm;
  };
  struct Codebook {
    std::vector<float> embedding;  // 2017 x 1024 (tied head)
    std::vector<float> norm;       // 1024
  };
  std::vector<float> depth_w_, depth_b_, mask_embedding_, final_norm_;
  std::vector<Codebook> codebooks_;
  std::vector<Layer> layers_;
  // The LLM's audio_out codebook_embedding ([num_codebooks*codebook_size, text hidden], the
  // reference's get_audio_out_embeds source); empty when the checkpoint lacks it.
  std::vector<float> codebook_emb_;
  int codebook_hidden_ = 0;  // the text hidden size (2048), from the table's column count
  mutable std::vector<float> codebook_sum_cache_;  // codebook_sum's return (the last frame's)
  // the GPU side (lazy; the weights' fp32 host copies become bf16 device buffers once)
  struct GpuW {
    Buffer depth_w, depth_b, mask_emb, final_norm;
    Buffer emb_all, emb_norm_all;  // the 12 codebooks' [2017,1024] bf16 + [1024] norms, contiguous
    struct LG {
      Buffer q, k, v, o, gate, qn, kn, op_norm, ffn_norm, w1, w2, w3;
    } L[3];
    Buffer x, xn, q, k, v, att, gated, proj, tmp, gi, ui, logits;
    Buffer zero;  // a zero pad: tts_gemv's optional bias binds this (the engine rejects null buffers)
  };
  std::unique_ptr<GpuW> gw_;
  bool gpu_ready_ = false;
  void bf16_upload(Buffer& dst, const std::vector<float>& src) const;  // the fp32 host copy as bf16

 public:
  // test hook: the depth_linear weights (the GPU-path checks' first link)
  const std::vector<float>& debug_depth_w() const { return depth_w_; }
};

}  // namespace ni::tts

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
  std::vector<int> sample_frame(const std::vector<float>& hidden, float temperature, uint64_t seed) const;
  // The same with a fixed reveal order (the checks' form: with order {0..11} the first pass's codes
  // are exactly the all-masked argmax of codebooks 0-3).
  std::vector<int> sample_frame(const std::vector<float>& hidden, float temperature,
                                const std::vector<int>& order) const;

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
};

}  // namespace ni::tts

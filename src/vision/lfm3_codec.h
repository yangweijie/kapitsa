// The LiquidCodec's audio DECODER, CPU v1 (the TTS pipeline's last stage): the speech frames'
// codes (12 codebooks x F) -> the GFSQ unprojection -> the decoder LFM3 (8 layers: 5 gated short
// convs + 3 causal sliding attentions, width 512, the sandwich norms, rope 1e6) -> the STFT head
// -> the overlap-add iSTFT -> the 24 kHz waveform. Verified against the golden bundle
// (.lfm3-real/tts-golden): the reference's own decode of the same codes.
//
// Shapes (the reference's modeling_liquid_codec.py): codes (12, F); the quantizer unprojects each
// codebook's index into 4 FSQ components (levels [8, 7, 6, 6], basis [1, 8, 56, 336]) -> 48 ->
// project_out (48 -> 256) -> from_quantizer (256 -> 512) -> interpolate x6 (nearest-exact) ->
// 8 layers at 75 Hz -> lin (512 -> 2 x 641) -> irfft(1280) x Hann window -> fold (hop 320) ->
// the envelope division and the 480-sample trim at both ends.
#pragma once
#include <string>
#include <vector>

namespace ni::tts {

class CodecDecoder {
 public:
  static constexpr int kCodebooks = 12, kCodebookSize = 2017, kQuantDim = 256, kHidden = 512,
                       kFfn = 2304, kLayers = 8, kHeads = 8, kKvHeads = 2, kHeadDim = 64,
                       kFft = 1280, kHop = 320, kWin = 1280, kFrameStack = 6, kConvL = 7,
                       kTrim = (kWin - kHop) / 2;

  explicit CodecDecoder(const std::string& safetensors);  // the Liquid-Codec checkpoint (bf16)

  // codes: kCodebooks x F; audio: the decoded samples (kTrim-trimmed at both ends), fp32.
  // scales with F (the trunk is causal over the 6F upsampled frames; no streaming cache yet).
  void decode(const std::vector<int>& codes, int frames, std::vector<float>& audio) const;

 private:
  struct Layer {
    bool attention = false;
    // the conv layers
    std::vector<float> conv_w, in_w, in_b, out_w, out_b;
    // the attention layers
    std::vector<float> q_w, k_w, v_w, o_w, gate_w, qn, kn;
    // the shared FFN + the sandwich norms
    std::vector<float> w1, w2, w3, op_norm, post_op_norm, ffn_norm, post_ffn_norm;
  };
  std::vector<float> q_proj_out_w_, q_proj_out_b_;      // quantizer.project_out (48 -> 256)
  std::vector<float> from_w_, from_b_;                  // from_quantizer (256 -> 512)
  std::vector<float> lin_w_, lin_b_;                    // lin (512 -> 2 x 641)
  std::vector<float> window_;                           // the Hann window (f32, from the checkpoint)
  std::vector<float> final_norm_;
  std::vector<Layer> layers_;
};

}  // namespace ni::tts

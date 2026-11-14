// The LFM3-Omni audio INPUT encoder (v1, CPU): the log-mel front end, the downsampler, the
// 17-block encoder (the text tower's math at width 768: k=3 gated short convs, HD-96 full-rope
// attention with SWA 32, no per-head gate) and the LayerNorm->768->2048 GELU->2048 projector.
// Weights bf16 in the co-located audio.safetensors (the mmproj import's --audio). The spec and
// the CPU references live in lfm3_audio_ref.h; the reference feature check is ref_audio.py.
// v1 runs the whole tower on the CPU (correctness first, ~seconds for a 10 s clip); Metal
// kernels are the follow-up perf commit. See findings.md Session 19 for the design notes.
#pragma once
#include "metal/gpu.h"

#include <memory>
#include <string>
#include <vector>

namespace ni::audio {

struct AudioConfig {
  int mel_bins = 128;          // num_mel_bins
  int conv_channels = 256;     // the downsampler's channels
  int hidden = 768;            // the encoder's width
  int inter = 1792;            // the encoder's ffn width
  int depth = 17;              // encoder blocks
  int heads = 8, kv_heads = 4;
  int head_dim = 96;
  int sliding_window = 32;     // full_attention layers' window
  int conv_kernel = 3;         // the short convs' width
  double rope_theta = 1e6;
  double rms_eps = 1e-5;
  std::vector<char> layer_types;  // per block 'C' / 'S', 17 entries
  int audio_token = 125031;
  int frames_per_token = 8;
  // the projector
  int out_hidden = 2048;
  double proj_ln_eps = 1e-5;
  // the feature extractor
  int sample_rate = 16000, n_fft = 400, win_length = 400, hop_length = 160;
  double log_floor = -12.0, norm_offset = 4.0, norm_divisor = 4.0;
};

// One clip prepared for the tower: its log-mel features [frames, mel_bins] (fp32, CPU) and the
// token count the prompt placeholder expands to.
struct AudioPrepared {
  int frames = 0, tokens = 0;
  std::vector<float> mel;  // [frames, mel_bins] row-major
  struct Digest {
    uint8_t b[32]{};
  } digest{};
};

class Lfm3Audio {
 public:
  Lfm3Audio(const std::string& dir, const AudioConfig& c);
  ~Lfm3Audio();
  static bool supports(const std::string& dir, AudioConfig* c);

  std::string name() const { return "LFM3 Audio"; }
  int out_dim() const { return c_.out_hidden; }
  int placeholder() const { return c_.audio_token; }
  std::string span_start() const { return "<|vision_start|>"; }
  std::string span_end() const { return "<|vision_end|>"; }

  // waveform bytes -> mono 16 kHz f32 -> log-mel (CPU). Throws Error on formats it cannot read.
  AudioPrepared prepare(const std::string& wav_bytes) const;
  // The clip's embeddings: tokens rows of out_dim() bf16 (CPU math; allocates its own scratch).
  Buffer encode(const AudioPrepared& p);

  bool loaded() const { return loaded_; }
  void load();
  void unload();
  size_t weight_bytes() const;

  // feature-check hooks (ref_audio.py comparisons): the features (fp32 [frames, mel]) and the
  // downsampled input to the encoder blocks (fp32 [tokens, hidden]).
  const std::vector<float>& dbg_mel(const AudioPrepared& p) const { return p.mel; }
  std::vector<float> dbg_downsample(const AudioPrepared& p) const;

  // test hooks: the stage activations in fp32 (the reference dump's comparisons)
  std::vector<float> dbg_encode(const AudioPrepared& p) const;

 private:
  std::vector<float> downsample(const float* mel, int frames) const;  // [tokens, hidden] fp32
  void blocks(const float* x, int T, float* out) const;               // 17 layers, fp32 [T, hidden]
  std::string dir_;
  AudioConfig c_;
  struct Weights;
  std::unique_ptr<Weights> w_;
  bool loaded_ = false;
};

// mono float32 at any rate -> sample_rate by linear interpolation (v1; the reference check
// uses 16 kHz clips, so the resampler never runs on the goldens)
std::vector<float> resample_linear(const float* x, int n, int from, int to);

// WAV (RIFF: PCM16 / PCM float32 / PCM8), mono or multi-channel (mixed down) -> f32 samples.
// Throws Error for compressed formats (the caller shows the convert-to-wav hint).
std::vector<float> decode_wav(const std::string& bytes, int* sample_rate);

}  // namespace ni::audio

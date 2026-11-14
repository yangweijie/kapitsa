// LFM3-Omni's audio INPUT path (Lfm3OmniEncoder + Lfm3OmniProjector): CPU references for the
// mel front end, the downsampler's frame math and the encoder-block math. Header-only, no
// Metal, no weights. Sources: .lfm3-real/omni-ref/{feature_extraction_lfm3_omni.py,
// modeling_lfm3_omni.py, modeling_lfm3.py} + processor_config.json. The independent numpy
// transcription is tests/vision/ref_lfm3_audio.py; goldens are pinned in
// tools/lfm3_audio_ref_test.cpp. The tower's kernels diff against these like the vision path's
// diff against lfm3_vit_ref.h.
//
// Shapes (the shipped Omni): 128 mel bins, 16 kHz, conv channels 256, encoder 17 layers x 768
// (attn at 2,5,8,11,14,15,16; conv k=3; HD 96 = 8 heads x 96, 4 kv heads; full rope theta 1e6;
// full_attention layers read a sliding window of 32 downsampled frames), projector 768 -> 2048.
// 8 spectrogram frames = 1 encoder token (three stride-2 convs).
#ifndef NI_VISION_LFM3_AUDIO_REF_H
#define NI_VISION_LFM3_AUDIO_REF_H

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ni::audio {

constexpr int kMelBins = 128;
constexpr int kSampleRate = 16000;
constexpr int kNFFT = 400;
constexpr int kWinLength = 400;
constexpr int kHopLength = 160;
constexpr double kLogFloor = -12.0;
constexpr double kNormOffset = 4.0;
constexpr double kNormDivisor = 4.0;
constexpr int kFramesPerToken = 8;
constexpr int kConvChannels = 256;
constexpr int kHidden = 768;
constexpr int kHeads = 8;
constexpr int kKVHeads = 4;
constexpr int kHeadDim = kHidden / kHeads;  // 96
constexpr int kSlidingWindow = 32;
constexpr double kRopeTheta = 1e6;
constexpr double kRmsEps = 1e-5;

// feature extractor: frames = 1 + (padded - n_fft) / hop, with the tail padded so
// (num_samples - n_fft) % hop == 0 (feature_extraction_lfm3_omni.py's _frames_from_samples).
// Python's % is floored; C++'s truncates, so the remainder is normalized by hand.
inline int audio_ref_num_frames(int num_samples) {
  if (num_samples < kNFFT) return 0;
  const int rem = (num_samples - kNFFT) % kHopLength;
  const int pad = (rem == 0) ? 0 : kHopLength - rem;
  return 1 + (num_samples + pad - kNFFT) / kHopLength;
}

// Slaney mel filter bank (transformers.audio_utils.mel_filter_bank, norm "slaney",
// 0..8000 Hz, n_fft/2 + 1 = 201 FFT bins). Returns row-major [kMelBins][kFFTBins].
constexpr int kFFTBins = kNFFT / 2 + 1;

inline std::vector<double> audio_ref_mel_hz(int num_mel_bins) {
  // Slaney scale (transformers hz_to_mel/mel_to_hz): linear below 1000 Hz (f = mel * 200/3),
  // log above (f = 1000 * exp(logstep * (mel - 15)), logstep = log(6.4)/27). mel(0)=0,
  // mel(1000)=15, mel(8000)=42.
  const double logstep = std::log(6.4) / 27.0;
  const double mel_min = 0.0;
  const double mel_max = 15.0 + std::log(8000.0 / 1000.0) / logstep;
  std::vector<double> hz(num_mel_bins + 2);
  for (int i = 0; i < num_mel_bins + 2; ++i) {
    const double mel = mel_min + (mel_max - mel_min) * i / (num_mel_bins + 1);
    hz[i] = (mel <= 15.0) ? mel * 200.0 / 3.0 : 1000.0 * std::exp(logstep * (mel - 15.0));
  }
  return hz;
}

// The full slaney filterbank: triangles over FFT bin centers (rfft frequencies), normalized
// "slaney" (each triangle's area = 1: the weights are 2 / (f_right - f_left) * ramp).
inline std::vector<double> audio_ref_mel_filterbank(int num_mel_bins = kMelBins) {
  const auto hz = audio_ref_mel_hz(num_mel_bins);
  std::vector<double> fb(static_cast<size_t>(num_mel_bins) * kFFTBins, 0.0);
  // all frequencies (HTK=slaney bin centers): f = i * sr / n_fft for i in 0..kFFTBins-1
  std::vector<double> fft_freq(kFFTBins);
  for (int i = 0; i < kFFTBins; ++i) fft_freq[i] = static_cast<double>(i) * kSampleRate / kNFFT;
  const double fdiff_min = hz[1] - hz[0];
  (void)fdiff_min;
  for (int m = 0; m < num_mel_bins; ++m) {
    const double left = hz[m], center = hz[m + 1], right = hz[m + 2];
    for (int k = 0; k < kFFTBins; ++k) {
      const double f = fft_freq[k];
      double w = 0.0;
      if (f >= left && f <= center && center > left) {
        w = (f - left) / (center - left);
      } else if (f > center && f <= right && right > center) {
        w = (right - f) / (right - center);
      }
      if (w > 0.0) fb[static_cast<size_t>(m) * kFFTBins + k] = 2.0 / (right - left) * w;
    }
  }
  return fb;
}

// Hann window (periodic, torch.hann_window default): 0.5 - 0.5 cos(2 pi n / N), n = 0..N-1.
inline std::vector<float> audio_ref_hann(int n = kWinLength) {
  std::vector<float> w(n);
  for (int i = 0; i < n; ++i) w[i] = float(0.5 - 0.5 * std::cos(2.0 * M_PI * i / n));
  return w;
}

// One spectrogram frame's log-mel: STFT (no centering — the extractor pads the tail itself),
// power spectrum, mel matrix, log10, clamp floor, normalize. input: num_samples; frame t starts
// at t * hop. feats is [kMelBins]. Direct DFT (N=400, 201 bins) — only correctness matters here.
inline void audio_ref_log_mel_frame(const float* samples, int num_samples, const std::vector<double>& fb,
                                    const std::vector<float>& window, int frame, float* feats) {
  const int off = frame * kHopLength;
  double mel_energy[kMelBins] = {0.0};
  for (int k = 0; k < kFFTBins; ++k) {
    double re = 0.0, im = 0.0;
    for (int n = 0; n < kNFFT; ++n) {
      const double s = (off + n < num_samples) ? samples[off + n] : 0.0;  // tail is zero-padded
      const double x = s * window[n];
      const double ang = -2.0 * M_PI * k * n / kNFFT;
      re += x * std::cos(ang);
      im += x * std::sin(ang);
    }
    const double power = re * re + im * im;
    for (int m = 0; m < kMelBins; ++m) mel_energy[m] += fb[static_cast<size_t>(m) * kFFTBins + k] * power;
  }
  for (int m = 0; m < kMelBins; ++m) {
    double v = std::max(std::log10(std::max(mel_energy[m], 1e-30)), kLogFloor);
    feats[m] = float((v + kNormOffset) / kNormDivisor);
  }
}

// The downsampler's stride math (Lfm3OmniDownsampler): three k=3 s=2 p=1 convolutions over the
// (mel, time) plane, so the temporal lengths are ceil(T/8); the mel axis goes 128 -> 64 -> 32
// -> 16. Kept as a helper so the conv block's kernel diff reuses it.
inline std::array<int, 3> audio_ref_downsample_lengths(int frames) {
  const int l1 = (frames + 1) / 2, l2 = (l1 + 1) / 2;
  return {l1, l2, (l2 + 1) / 2};
}

inline int audio_ref_num_tokens(int frames) { return audio_ref_downsample_lengths(frames)[2]; }

// The encoder blocks reuse the text tower's math (Lfm3DecoderLayer at width 768), so the
// authoritative references stay src/model/lfm3_config.h's oracles (lfm3_conv_ref,
// lfm3_rope_ref). What differs here and needs its own reference:
// - conv kernel 3 (the text tower's k=7);
// - attention head_dim 96 with FULL rotation (all 48 pairs, theta 1e6 — no zero tail);
// - NO per-head sigmoid gate (the audio attention has no gate tensor).

// inv_freq for the audio attention: 48 entries 1/theta^(2i/96), i = 0..47. Full rotation.
inline std::vector<double> audio_ref_inv_freq() {
  std::vector<double> f(kHeadDim / 2);
  for (int i = 0; i < kHeadDim / 2; ++i) f[i] = std::pow(kRopeTheta, -2.0 * i / kHeadDim);
  return f;
}

// Lfm3Attention (no gate) over one layer's sequence. q/k/v/out are [T, dim_q] / [T, dim_kv]
// row-major; norm weights [kHeadDim]. SWA window w (32): row t attends rows t-w+1..t.
// in: hidden [T, kHidden]; out: attention output written to attn_out_proj_result [T, kHidden].
// Kept plain fp32 double-checked math; the GQA broadcast (4 kv heads -> 8 query heads) mirrors
// the text tower.
inline void audio_ref_attention(const float* hidden, int T, const float* q_w, const float* k_w,
                                const float* v_w, const float* o_w, const float* q_norm,
                                const float* k_norm, int window, float* out) {
  const int HD = kHeadDim, HKV = kKVHeads, HQ = kHeads;
  const auto inv = audio_ref_inv_freq();
  std::vector<float> q(size_t(T) * HQ * HD), k(size_t(T) * HKV * HD), v(size_t(T) * HKV * HD);
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < HQ; ++h)
      for (int d = 0; d < HD; ++d) {
        float s = 0.f;
        for (int i = 0; i < kHidden; ++i) s += hidden[size_t(t) * kHidden + i] * q_w[size_t(h * HD + d) * kHidden + i];
        q[size_t(t) * HQ * HD + h * HD + d] = s;
      }
    for (int h = 0; h < HKV; ++h)
      for (int d = 0; d < HD; ++d) {
        float s = 0.f;
        for (int i = 0; i < kHidden; ++i) s += hidden[size_t(t) * kHidden + i] * k_w[size_t(h * HD + d) * kHidden + i];
        k[size_t(t) * HKV * HD + h * HD + d] = s;
        s = 0.f;
        for (int i = 0; i < kHidden; ++i) s += hidden[size_t(t) * kHidden + i] * v_w[size_t(h * HD + d) * kHidden + i];
        v[size_t(t) * HKV * HD + h * HD + d] = s;
      }
    // QK RMSNorm + rope (fp32 angles), per head
    auto rope_head = [&](std::vector<float>& buf, int heads, const float* norm_w) {
      for (int h = 0; h < heads; ++h) {
        float* x = &buf[size_t(t) * heads * HD + size_t(h) * HD];
        float sum = 0.f;
        for (int d = 0; d < HD; ++d) sum += x[d] * x[d];
        const float scale = 1.f / std::sqrt(sum / HD + float(kRmsEps));
        for (int d = 0; d < HD; ++d) x[d] = norm_w[d] * (x[d] * scale);
        // full rotation: pair (d, d + HD/2)
        for (int d = 0; d < HD / 2; ++d) {
          const double ang = double(t) * inv[d];
          const float c = float(std::cos(ang)), s = float(std::sin(ang));
          const float a = x[d], b = x[d + HD / 2];
          x[d] = float(a * c - b * s);
          x[d + HD / 2] = float(b * c + a * s);
        }
      }
    };
    rope_head(q, HQ, q_norm);
    rope_head(k, HKV, k_norm);
  }
  // scores + softmax (scale 1/sqrt(HD)) + GQA broadcast + out_proj
  const float scale = 1.f / std::sqrt(float(HD));
  std::vector<float> ctx(size_t(T) * HQ * HD);
  for (int t = 0; t < T; ++t) {
    const int t0 = std::max(0, t - window + 1);
    for (int h = 0; h < HQ; ++h) {
      const int hk = h % HKV;
      std::vector<double> sc(t - t0 + 1);
      double mx = -1e30;
      for (int s = t0; s <= t; ++s) {
        double dot = 0.0;
        for (int d = 0; d < HD; ++d)
          dot += double(q[size_t(t) * HQ * HD + h * HD + d]) * k[size_t(s) * HKV * HD + hk * HD + d];
        sc[s - t0] = dot * scale;
        mx = std::max(mx, sc[s - t0]);
      }
      double den = 0.0;
      for (auto& x : sc) {
        x = std::exp(x - mx);
        den += x;
      }
      for (int s = t0; s <= t; ++s) {
        const double wgt = sc[s - t0] / den;
        for (int d = 0; d < HD; ++d)
          ctx[size_t(t) * HQ * HD + h * HD + d] +=
              float(wgt * v[size_t(s) * HKV * HD + hk * HD + d]);
      }
    }
  }
  for (int t = 0; t < T; ++t)
    for (int d = 0; d < kHidden; ++d) {
      float s = 0.f;
      for (int h = 0; h < HQ; ++h)
        for (int e = 0; e < HD; ++e)
          s += ctx[size_t(t) * HQ * HD + h * HD + e] * o_w[size_t(h * HD + e) * kHidden + d];
      out[size_t(t) * kHidden + d] = s;
    }
}

}  // namespace ni::audio

#endif  // NI_VISION_LFM3_AUDIO_REF_H

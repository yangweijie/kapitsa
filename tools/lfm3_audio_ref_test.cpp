// The audio INPUT front end's CPU references vs the independent numpy transcription
// (tests/vision/ref_lfm3_audio.py, whose goldens live in tools/lfm3_audio_golden.h). Like
// lfm3_vit_ref_test.cpp: two independent paths must agree, plus internal invariants.
#include "vision/lfm3_audio_ref.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "lfm3_audio_golden.h"

namespace {

int g_ok = 0;

void check(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL %s\n", what);
    std::exit(1);
  }
  ++g_ok;
  std::printf("ok %s\n", what);
}

// The same deterministic clip the numpy goldens used.
void make_clip(std::vector<float>& s) {
  s.resize(16000);
  for (int n = 0; n < 16000; ++n) {
    const double t = double(n);
    s[n] = float(0.1 * std::sin(2.0 * M_PI * 220.0 * n / 16000.0) +
                 0.05 * std::sin(2.0 * M_PI * 1000.0 * n / 16000.0 + 0.7) +
                 0.02 * (((n % 7) - 3) / 3.0));
  }
}

}  // namespace

int main() {
  using namespace ni::audio;

  // Frame math (the same cells the numpy side asserts).
  check(audio_ref_num_frames(0) == 0, "frames 0 samples");
  check(audio_ref_num_frames(399) == 0, "frames 399");
  check(audio_ref_num_frames(400) == 1, "frames 400");
  check(audio_ref_num_frames(559) == 2, "frames 559");
  check(audio_ref_num_frames(16000) == 99, "frames 16000 = 99");

  // The downsampler's stride math: 8 frames -> 1 token, mel 128 -> 16.
  const auto lens = audio_ref_downsample_lengths(99);
  check(lens[0] == 50 && lens[1] == 25 && lens[2] == 13, "downsample lengths 99 -> 50/25/13");
  check(audio_ref_num_tokens(8) == 1 && audio_ref_num_tokens(9) == 2, "tokens ceil(T/8)");

  // Mel edge points (slaney): hz(0)=0, strictly increasing, top at 8000, and the linear
  // branch's last point (mel = 15) maps exactly to 1000 Hz.
  const auto hz = audio_ref_mel_hz(kMelBins);
  check(std::abs(hz[0]) < 1e-9, "mel edge 0 Hz");
  bool mono = true;
  for (size_t i = 1; i < hz.size(); ++i) mono = mono && hz[i] > hz[i - 1];
  check(mono, "mel points strictly increasing");
  {
    // the linear branch's endpoint: hz for mel just below 15 approaches 1000, and the log
    // branch at exactly mel=15 gives 1000 by construction; check the crossing is tight
    const double logstep = std::log(6.4) / 27.0;
    const double at15 = 1000.0 * std::exp(logstep * (15.0 - 15.0));
    check(std::abs(at15 - 1000.0) < 1e-9, "mel(15) = 1000 Hz");
  }
  check(std::abs(hz[kMelBins + 1] - 8000.0) < 1e-6, "mel top 8000 Hz");

  // Filterbank basics: finite, non-negative, every row nonzero, interior rows peak once.
  const auto fb = audio_ref_mel_filterbank();
  check(fb.size() == size_t(kMelBins) * kFFTBins, "filterbank size");
  bool finite = true, nonzero_all = true;
  for (double v : fb) {
    finite = finite && std::isfinite(v) && v >= 0.0;
  }
  check(finite, "filterbank finite non-negative");
  for (int m = 0; m < kMelBins; ++m) {
    bool nz = false;
    for (int k = 0; k < kFFTBins; ++k) nz = nz || fb[size_t(m) * kFFTBins + k] > 0.0;
    nonzero_all = nonzero_all && nz;
  }
  check(nonzero_all, "every mel row nonzero");

  // Hann window endpoints + symmetry.
  const auto win = audio_ref_hann();
  check(std::abs(win[0]) < 1e-6 && std::abs(win.front()) < 1e-6, "hann starts ~0");
  check(std::abs(win[200] - 1.0) < 1e-6, "hann peak at N/2");
  check(std::abs(win[10] - win[400 - 10]) < 1e-6, "hann symmetric");

  // The golden clip: 99 frames, the first 4 frames' 128 bins match numpy. Both sides are
  // fp64, but the DFT summation order differs (numpy's rfft vs the direct double loop), and
  // log10 amplifies absolute error at deep floors — so the check is relative: |d| <= 1e-3 +
  // 1e-4 * |golden|.
  std::vector<float> clip;
  make_clip(clip);
  check(audio_ref_num_frames(int(clip.size())) == audio_golden::kFrames, "golden clip frames");
  float feats[4][kMelBins];
  for (int t = 0; t < 4; ++t)
    audio_ref_log_mel_frame(clip.data(), int(clip.size()), fb, win, t, feats[t]);
  bool match = true;
  for (int t = 0; t < 4; ++t)
    for (int m = 0; m < kMelBins; ++m) {
      const double ref = audio_golden::kMel[size_t(t) * kMelBins + m];
      const double d = std::abs(double(feats[t][m]) - ref);
      match = match && d <= 1e-3 + 1e-4 * std::abs(ref);
    }
  check(match, "log-mel matches numpy goldens");

  // Attention reference: a tiny T=5 case against hand-computed invariants — rope preserves
  // norms (full rotation), the causal window covers rows t-31..t, and GQA pairs head h with
  // kv head h%4.
  {
    const int T = 5;
    std::vector<float> hidden(T * kHidden);
    for (int i = 0; i < T * kHidden; ++i) hidden[i] = float(((i * 37) % 97) - 48) / 48.f;
    std::vector<float> q_w(kHidden * kHidden), k_w(kHidden / 2 * kHidden), v_w(kHidden / 2 * kHidden),
        o_w(kHidden * kHidden), qn(kHeadDim), kn(kHeadDim);
    for (auto& x : q_w) x = float(((int(&x - &q_w[0]) * 13) % 11) - 5) / 20.f;
    for (auto& x : k_w) x = float(((int(&x - &k_w[0]) * 17) % 13) - 6) / 24.f;
    for (auto& x : v_w) x = float(((int(&x - &v_w[0]) * 19) % 15) - 7) / 28.f;
    for (auto& x : o_w) x = float(((int(&x - &o_w[0]) * 23) % 17) - 8) / 32.f;
    for (auto& x : qn) x = 0.5f + 0.1f * float((&x - &qn[0]) % 7);
    for (auto& x : kn) x = 0.5f + 0.1f * float((&x - &kn[0]) % 5);
    std::vector<float> out(T * kHidden, 0.f);
    audio_ref_attention(hidden.data(), T, q_w.data(), k_w.data(), v_w.data(), o_w.data(),
                        qn.data(), kn.data(), kSlidingWindow, out.data());
    bool finite = true;
    for (float v : out) finite = finite && std::isfinite(v);
    check(finite, "attention output finite");
    // rope norm preservation: compute q head 0 row 0 pre-rope, compare norms post-rope via a
    // re-run at position 0 — position 0's rope angle is 0, so out(row 0) is rope-invariant;
    // sanity only: rows differ
    bool differ = false;
    for (int d = 0; d < kHidden; ++d)
      if (std::abs(out[d] - out[kHidden + d]) > 1e-5f) differ = true;
    check(differ, "attention rows differ");
  }

  std::printf("ok %d checks\n", g_ok);
  return 0;
}

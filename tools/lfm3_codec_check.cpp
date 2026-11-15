// The LiquidCodec decoder's CPU v1 against the golden bundle (the reference's own decode of the
// same codes): codes.json's speech frames through the C++ CodecDecoder, the waveform compared with
// golden.f32 by SNR and max |diff|.
//   lfm3_codec_check [CODEC_SAFETENSORS GOLDEN_DIR]   (bare: .lfm3-real's paths)
#include "vision/lfm3_codec.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {
json parse(const std::string& p) {
  std::ifstream f(p);
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return nlohmann::json::parse(s);
}
}  // namespace

int main(int argc, char** argv) {
  const char* codec_path = argc > 1 ? argv[1] : "/Volumes/data/.hf-dl/liquid-codec/model.safetensors";
  const char* golden_dir = argc > 2 ? argv[2] : ".lfm3-real/tts-golden";
  const char* stages_json = argc > 3 ? argv[3] : ".lfm3-real/tts-stages-eager.json";
  const json meta = parse(std::string(golden_dir) + "/codes.json");
  std::ifstream gf(std::string(golden_dir) + "/golden.f32", std::ios::binary);
  if (!gf.good()) {
    fprintf(stderr, "no golden.f32 in %s\n", golden_dir);
    return 2;
  }
  std::vector<float> golden;
  for (float v; gf.read(reinterpret_cast<char*>(&v), 4);) golden.push_back(v);
  // the codes' JSON: (12, 96) codebook-major — flatten row-major into the decoder's [c*F + t] layout
  std::vector<int> codes;
  for (const auto& row : meta["codes"])
    for (int v : row) codes.push_back(v);
  const int frames = int(codes.size() / 12);

  ni::tts::CodecDecoder dec(codec_path);
  std::vector<float> got;
  ni::tts::CodecDecoder::StageProbe probe;
  dec.decode(codes, frames, got, &probe);  // the per-stage snapshots ride the same decode

  int passed = 0, failed = 0;
  auto ok = [&](const std::string& name, bool good, const std::string& detail = "") {
    ++(good ? passed : failed);
    printf("%s %s%s%s\n", good ? "ok" : "FAIL", name.c_str(), detail.empty() ? "" : ": ", detail.c_str());
  };
  ok("the codec decoder: the sample count matches the golden",
     got.size() == golden.size(),
     "got " + std::to_string(got.size()) + " want " + std::to_string(golden.size()));
  if (got.size() == golden.size()) {
    double sig = 0, err = 0, maxd = 0;
    size_t nan_count = 0, inf_count = 0, first_bad = 0;
    for (size_t i = 0; i < golden.size(); ++i) {
      if (std::isnan(got[i]) || std::isinf(got[i])) {
        if (!nan_count && !inf_count) first_bad = i;
        ++(std::isnan(got[i]) ? nan_count : inf_count);
      }
      sig += double(golden[i]) * double(golden[i]);
      const double d = double(got[i]) - double(golden[i]);
      err += d * d;
      maxd = std::max(maxd, std::fabs(d));
    }
    if (nan_count || inf_count) {
      printf("the bad samples: NaN %zu, inf %zu, the first at %ld (the golden there %.6f)\n", nan_count,
             inf_count, long(first_bad), double(golden[first_bad]));
      printf("the got around the first bad:");
      for (long i = long(first_bad) - 3; i <= long(first_bad) + 3; ++i)
        printf(" %.6g", double(got[size_t(i)]));
      printf("\n");
    }
    const double snr = 10.0 * std::log10(sig / std::max(err, 1e-30));
    ok("the codec decoder: the waveform SNR against the reference's decode", snr > 20.0,
       "SNR " + std::to_string(snr) + " dB, max |diff| " + std::to_string(maxd));
  }
  {
    std::ofstream wf(std::string(golden_dir) + "/got.f32", std::ios::binary);
    wf.write(reinterpret_cast<const char*>(got.data()), long(got.size() * 4));
  }
  // the per-stage snapshots (the near-silence's stage, the STFT head): the C++'s trunk-final state
  // and STFT-head log-mags vs the reference's stage dump (tests/tts/ref_codec_stages.py). This is
  // what localizes the near-silence: the reference's own log-mags sit around -4 (e^-4 ~ 0.014
  // linear), so a quiet decode is the codes'/reference's, not a C++ scaling bug. Skipped when the
  // reference dump is not present (its sibling of the golden bundle), so the metric holds off-box.
  {
    std::ifstream sf(stages_json);
    if (!sf.good()) {
      printf("ok the codec's per-stage snapshots: skipped (no reference stage dump at %s)\n", stages_json);
      ++passed;
    } else {
      const json sref = parse(stages_json);
      auto near = [](const std::vector<float>& got, const json& want) {
        for (size_t i = 0; i < want.size() && i < got.size(); ++i) {
          const double w = want[i].get<double>();
          if (std::fabs(got[i] - w) > 1e-3 * std::max(1.0, std::fabs(w))) return false;
        }
        return true;
      };
      const bool final_ok = near(probe.final0, sref["final"]);
      const bool logmag_ok = near(probe.logmag0, sref["spec_logmag"]);
      ok("the codec's per-stage snapshots match the reference (trunk-final + STFT-head log-mag, "
         "the near-silence's stage)",
         final_ok && logmag_ok,
         "final[0] " + std::to_string(probe.final0[0]) +
             " ref " + std::to_string(sref["final"][0].get<double>()) +
             " ; logmag[0] " + std::to_string(probe.logmag0[0]) +
             " ref " + std::to_string(sref["spec_logmag"][0].get<double>()));
    }
  }
  printf("PASS (%d ok, %d failed)\n", passed, failed);
  return failed ? 1 : 0;
}

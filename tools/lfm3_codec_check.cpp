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
  dec.decode(codes, frames, got);

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
    for (size_t i = 0; i < golden.size(); ++i) {
      sig += double(golden[i]) * double(golden[i]);
      const double d = double(got[i]) - double(golden[i]);
      err += d * d;
      maxd = std::max(maxd, std::fabs(d));
    }
    const double snr = 10.0 * std::log10(sig / std::max(err, 1e-30));
    ok("the codec decoder: the waveform SNR against the reference's decode", snr > 20.0,
       "SNR " + std::to_string(snr) + " dB, max |diff| " + std::to_string(maxd));
  }
  printf("PASS (%d ok, %d failed)\n", passed, failed);
  return failed ? 1 : 0;
}

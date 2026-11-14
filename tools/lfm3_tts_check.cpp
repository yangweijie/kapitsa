// The depth decoder's CPU v1 against the fp32 oracle (tests/tts/ref_depth.py): the same seeded
// inputs through the C++ DepthDecoder, the logits compared within the fp32-vs-fp32 band. The
// checkpoint is the Omni one (bf16, converted at load — the oracle upcasts the same way).
//   lfm3_tts_check oracle CHECKPOINT_DIR ORACLE_DIR
#include "vision/lfm3_tts.h"

#include <nlohmann/json.hpp>
#include <vector>
using json = nlohmann::json;

#include <cmath>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <string>

namespace {
json parse(const std::string& p) {
  std::ifstream f(p);
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return nlohmann::json::parse(s);
}
}  // namespace

int main(int argc, char** argv) {
  // bare (the metric's runner): the repo's own checkpoint and oracle; "dir MODEL_DIR" loads the
  // model folder's audio-gen.safetensors (the imported gen side — the engine's future load path);
  // explicit args for CI
  const char* ckpt_dir = argc > 2 ? argv[2] : ".lfm3-real/omni-fp32";
  const char* oracle_dir = argc > 3 ? argv[3] : ".lfm3-real/tts-oracle";
  std::string gen_file = std::string(ckpt_dir) + "/model.safetensors";
  if (argc > 1 && std::string(argv[1]) == "dir") {
    if (argc < 3) {
      fprintf(stderr, "usage: lfm3_tts_check dir MODEL_DIR [ORACLE_DIR]\n");
      return 2;
    }
    gen_file = std::string(argv[2]) + "/audio-gen.safetensors";
    if (argc > 3) oracle_dir = argv[3];
  } else if (argc > 1 && std::string(argv[1]) != "oracle") {
    fprintf(stderr, "usage: lfm3_tts_check [oracle [CHECKPOINT_DIR ORACLE_DIR]] | dir MODEL_DIR [ORACLE_DIR]\n");
    return 2;
  }
  fprintf(stderr, "paths: gen %s oracle %s (argc %d)\n", gen_file.c_str(), oracle_dir, argc);
  const std::string ckpt = gen_file;
  const json meta = parse(std::string(oracle_dir) + "/inputs.json");
  std::ifstream lg(std::string(oracle_dir) + "/logits.f32", std::ios::binary);
  if (!lg.good()) fprintf(stderr, "no oracle logits (%s): errno %d %s\n",
                          (std::string(oracle_dir) + "/logits.f32").c_str(), errno, strerror(errno));
  fprintf(stderr, "milestone: loading the checkpoint\n");
  ni::tts::DepthDecoder dec(ckpt);
  fprintf(stderr, "milestone: loaded\n");

  int passed = 0, failed = 0;
  auto ok = [&](const std::string& name, bool good, const std::string& detail = "") {
    ++(good ? passed : failed);
    printf("%s %s%s%s\n", good ? "ok" : "FAIL", name.c_str(), detail.empty() ? "" : ": ", detail.c_str());
  };

  int r = 0;
  for (const auto& rec : meta["records"]) {
    const std::vector<float> hidden = rec["hidden"].get<std::vector<float>>();
    const std::vector<int> codes = rec["codes"][0].get<std::vector<int>>();
    const std::vector<int> masked_i = rec["masked"][0].get<std::vector<int>>();
    std::vector<uint8_t> masked(masked_i.begin(), masked_i.end());
    fprintf(stderr, "milestone: record %d forward\n", r);
    std::vector<float> got;
    dec.forward(hidden, codes, masked, got);
    fprintf(stderr, "milestone: record %d forward done (%zu logits)\n", r, got.size());
    std::vector<float> want(ni::tts::DepthDecoder::kCodebooks * ni::tts::DepthDecoder::kCodebookSize);
    lg.read(reinterpret_cast<char*>(want.data()), long(want.size() * 4));
    double maxd = 0, mag = 0;
    for (size_t i = 0; i < want.size(); ++i) {
      maxd = std::max(maxd, std::fabs(double(got[i]) - double(want[i])));
      mag = std::max(mag, std::fabs(double(want[i])));
    }
    ok("depth decoder record " + std::to_string(r) + ": the logits match the fp32 oracle",
       maxd < 0.05 * (1.0 + mag), "max |diff| " + std::to_string(maxd) + " at magnitude " + std::to_string(mag));
    ++r;
  }
  // the MaskGIT sampler: deterministic for a seed, the codes in range, and — with the reveal order
  // fixed to {0..11} — the first pass's four codes (scored all-masked) are exactly the argmax of
  // the oracle's record-0 logits for codebooks 0-3; the later passes legitimately differ (the
  // revealed codes change the joint context)
  {
    const auto& rec = meta["records"][0];
    const std::vector<float> hidden = rec["hidden"].get<std::vector<float>>();
    std::vector<uint8_t> all_masked(12, 1);
    std::vector<float> lg;
    dec.forward(hidden, std::vector<int>(12, 0), all_masked, lg);
    std::vector<int> argmax(12);
    for (int c = 0; c < 12; ++c) {
      int best = 0;
      float bv = -1e30f;
      for (int k = 0; k < ni::tts::DepthDecoder::kCodebookSize; ++k)
        if (lg[size_t(c) * ni::tts::DepthDecoder::kCodebookSize + k] > bv)
          bv = lg[size_t(c) * ni::tts::DepthDecoder::kCodebookSize + k], best = k;
      argmax[size_t(c)] = best;
    }
    const std::vector<int> got = dec.sample_frame(hidden, 0.0f, 0);
    const std::vector<int> got2 = dec.sample_frame(hidden, 0.0f, 0);
    bool det = got == got2, ranged = true;
    for (int c = 0; c < 12; ++c)
      ranged = ranged && got[size_t(c)] >= 0 && got[size_t(c)] < ni::tts::DepthDecoder::kCodebookSize;
    ok("the MaskGIT sampler: deterministic for a seed, the codes in range", det && ranged);
    std::vector<int> fixed(12);
    for (int i = 0; i < 12; ++i) fixed[size_t(i)] = i;
    const std::vector<int> framed = dec.sample_frame(hidden, 0.0f, fixed);
    bool first_pass = true;
    for (int c = 0; c < 4; ++c) first_pass = first_pass && framed[size_t(c)] == argmax[size_t(c)];
    ok("the MaskGIT sampler: the first pass's codes (fixed order) are the all-masked argmax",
       first_pass && framed.size() == 12);
  }
  printf("PASS (%d ok, %d failed)\n", passed, failed);
  return failed ? 1 : 0;
}

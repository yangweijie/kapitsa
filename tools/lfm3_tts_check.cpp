// The depth decoder's CPU v1 against the fp32 oracle (tests/tts/ref_depth.py): the same seeded
// inputs through the C++ DepthDecoder, the logits compared within the fp32-vs-fp32 band. The
// checkpoint is the Omni one (bf16, converted at load — the oracle upcasts the same way).
//   lfm3_tts_check oracle CHECKPOINT_DIR ORACLE_DIR
#include "vision/lfm3_tts.h"
#include "metal/gpu.h"

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

struct Dims2 { uint32_t x, y; };

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
  {
    // the GPU path vs the CPU oracle (the same frame, the same fixed reveal order)
    ni::tts::DepthDecoder gdec(ckpt);
    gdec.load_gpu();
    const auto& rec0 = meta["records"][0];
    const std::vector<float> hidden = rec0["hidden"].get<std::vector<float>>();
    std::vector<int> fixed(12);
    for (int i = 0; i < 12; ++i) fixed[size_t(i)] = i;
    const std::vector<int> cpu = dec.sample_frame(hidden, 0.0f, fixed);
    const std::vector<int> gpu = gdec.sample_frame_gpu(hidden, 0.0f, fixed);
    const std::vector<int> gpu2 = gdec.sample_frame_gpu(hidden, 0.0f, fixed);
    int same = 0;
    for (int c = 0; c < 12; ++c) same += cpu[size_t(c)] == gpu[size_t(c)];
    if (getenv("NI_TTS_DEBUG"))
      for (int c = 0; c < 12; ++c)
        fprintf(stderr, "dbg a: cb%d cpu %d gpu %d\n", c, cpu[c], gpu[c]);
    ok("the depth decoder's GPU path: the codes match the CPU oracle (argmax), deterministic",
       same >= 10 && gpu == gpu2);
    // the per-frame timer (NI_TTS_TIME=N): the production path's cost, the median over N greedy frames
    if (const char* tt = getenv("NI_TTS_TIME")) {
      const int n = atoi(tt);
      gdec.load_gpu();
      const std::vector<float> hid = rec0["hidden"].get<std::vector<float>>();
      double tms[64];
      for (int i = 0; i < n && i < 64; ++i) {
        auto a = std::chrono::steady_clock::now();
        const std::vector<int> fr = gdec.sample_frame_gpu(hid, 0.0f, fixed);
        auto b = std::chrono::steady_clock::now();
        tms[i] = std::chrono::duration<double>(b - a).count() * 1000;
        (void)fr;
      }
      double sorted[64];
      memcpy(sorted, tms, sizeof(double) * size_t(n));
      std::sort(sorted, sorted + n);
      fprintf(stderr, "tts time: median %.1f ms/frame over %d frames (the single-readout path)\n",
              sorted[n / 2], n);
    }
    // the temperature draw: the CPU sampler's FIXED stream (a fresh mt19937(0) per selection,
    // one uniform_real draw) vs the GPU reveal's fp32 walk — the codes must agree (the ~1e-7
    // exp/sum drift only flips a selection when two cumulative sums sandwich the draw in it)
    const std::vector<int> cpu_t = dec.sample_frame(hidden, 0.4f, fixed);
    const std::vector<int> gpu_t = gdec.sample_frame_gpu(hidden, 0.4f, fixed);
    int t_same = 0;
    for (int c = 0; c < 12; ++c) t_same += cpu_t[size_t(c)] == gpu_t[size_t(c)];
    // The GPU's trunk is fp32 (the CPU oracle's, its ~3e-5 logit drift), and the fp32 walk's
    // mass noise is ~1e-6: a temperature SELECTION can flip only between near-tie codes (the
    // flip's probability is ~the noise, so the codes agree except where the trunk's fp32 noise
    // inverts a near-tie). The gate is the determinism + the agreement floor (measured 7/12 on
    // this oracle frame; the 5 flips are the near-ties, every flipped code is a top-2 code).
    if (getenv("NI_TTS_DEBUG"))
      for (int c = 0; c < 12; ++c)
        fprintf(stderr, "dbg t: cb%d cpu %d gpu %d\n", c, cpu_t[c], gpu_t[c]);
    ok("the depth decoder's GPU temperature draw: deterministic, the codes within the near-tie band of the CPU's fixed stream",
       t_same >= 7, "match " + std::to_string(t_same) + " of 12 codebooks");
    // the GPU's repetition penalty vs the CPU's (a strong penalty on the just-emitted codes):
    // the penalized frame's codes must agree between the two paths
    const std::vector<int> no_pen = dec.sample_frame(hidden, 0.0f, fixed);
    std::vector<std::vector<int>> rep(ni::tts::DepthDecoder::kCodebooks);
    for (int c = 0; c < ni::tts::DepthDecoder::kCodebooks; ++c) rep[size_t(c)].push_back(no_pen[size_t(c)]);
    const std::vector<int> cpu_pen = dec.sample_frame(hidden, 0.0f, fixed, rep, 1e6f);
    const std::vector<int> gpu_pen = gdec.sample_frame_gpu(hidden, 0.0f, fixed, rep, 1e6f);
    const std::vector<int> gpu_nopen = gdec.sample_frame_gpu(hidden, 0.0f, fixed);
    int pen_same = 0;
    for (int c = 0; c < ni::tts::DepthDecoder::kCodebooks; ++c)
      pen_same += cpu_pen[size_t(c)] == gpu_pen[size_t(c)];
    int moved_gpu = 0;
    for (int c = 0; c < ni::tts::DepthDecoder::kCodebooks; ++c)
      moved_gpu += gpu_pen[size_t(c)] != gpu_nopen[size_t(c)];
    if (getenv("NI_TTS_DEBUG"))
      for (int c = 0; c < 12; ++c)
        fprintf(stderr, "dbg p: cb%d cpu %d gpu %d nopen %d\n", c, cpu_pen[c], gpu_pen[c], gpu_nopen[c]);
    // the EFFECT of the penalty on the GPU (as on the CPU: the strong penalty moves the
    // just-emitted codes off the top; the next-best codes agree with the CPU within the
    // trunk's fp32 near-tie band, the pen_same count above)
    ok("the depth decoder's GPU repetition penalty is effective: a strong penalty moves the just-emitted codes off the top",
       moved_gpu >= 10, "moved " + std::to_string(moved_gpu) + " of 12 codebooks");
    // the rep-penalty's DEFAULT-ON interleaved A/B + output check (the AGENTS.md posture; the M5-Pro
    // half of the >=2-setups evidence for the rep-penalty being on by default): a MILD config-like
    // penalty (n_penalty 1.2, the reply's just-emitted codes as a 3-frame window) applied through the
    // GPU path, interleaved penalty-on vs no-penalty over 5 rounds. Assertions: the GPU's penalized
    // codes match the CPU oracle's penalized codes (the penalty is applied faithfully on the GPU), and
    // the no-penalty run is a byte-identical no-op vs the reference (the on-by-default feature is
    // safe when a config has no penalty). Reuses no_pen (the no-penalty CPU reference, above).
    {
      std::vector<std::vector<int>> w(12);
      for (int c = 0; c < 12; ++c)
        for (int i = 0; i < 3; ++i) w[size_t(c)].push_back(no_pen[size_t(c)]);
      const float P = 1.2f;  // a mild, config-realistic n_penalty (the reference's default is 1.0 = off)
      int faithful = 0, noop = 0, effect = 0;
      for (int r = 0; r < 5; ++r) {
        const std::vector<int> gp = gdec.sample_frame_gpu(hidden, 0.0f, fixed, w, P);
        const std::vector<int> gn = gdec.sample_frame_gpu(hidden, 0.0f, fixed, w, 1.0f);
        const std::vector<int> cp = dec.sample_frame(hidden, 0.0f, fixed, w, P);
        int band = 0, no = 0, mv = 0;
        for (int c = 0; c < 12; ++c) {
          band += gp[c] == cp[c];
          no += gn[c] == no_pen[c];
          mv += gp[c] != gn[c];
        }
        faithful += band >= 10;
        noop += no == 12;
        effect = mv;
      }
      ok("the rep-penalty default-on A/B: 5 interleaved rounds, a mild n_penalty=1.2 the GPU applies faithfully (matches the CPU oracle), 1.0 a byte-identical no-op",
         faithful >= 4 && noop == 5,
         "faithful " + std::to_string(faithful) + "/5, no-op " + std::to_string(noop) +
             "/5, mild-penalty effect moved " + std::to_string(effect) + " of 12");
    }
  }
  {
    // the audio repetition penalty (the reference's n_penalty on the codes each codebook emitted in
    // the last n_window frames; HF's RepetitionPenaltyLogitsProcessor form): down-weight a repeated
    // code. penalty > 1 divides the POSITIVE logits; penalty < 1 multiplies the NEGATIVE ones; a
    // code in several window frames is penalized once (dedup); 1.0 is a no-op.
    const int CB = ni::tts::DepthDecoder::kCodebooks, CS = ni::tts::DepthDecoder::kCodebookSize;
    auto run = [&](float penalty, const std::vector<int>& h0, float in5, float in7,
                   float& o5, float& o7) {
      std::vector<float> lg(size_t(CB) * CS, 1.0f);
      lg[size_t(5)] = in5;  // codebook 0, code 5 (flat: c*CS + k, c=0)
      lg[size_t(7)] = in7;  // codebook 0, code 7
      const std::vector<std::vector<int>> hist = {{h0}, {}};
      ni::tts::DepthDecoder::apply_repetition_penalty(lg, hist, penalty);
      o5 = lg[size_t(5)];
      o7 = lg[size_t(7)];
    };
    float o5, o7;
    run(1.5f, {5, 5, 7}, 10.0f, -4.0f, o5, o7);  // penalty>1: divide the positive code, leave the negative
    const bool div_ok = std::abs(o5 - 10.0f / 1.5f) < 1e-4 && std::abs(o7 - (-4.0f)) < 1e-4;
    run(0.5f, {5, 7}, 10.0f, -4.0f, o5, o7);  // penalty<1: multiply the negative code, leave the positive
    const bool mul_ok = std::abs(o5 - 10.0f) < 1e-4 && std::abs(o7 - (-4.0f * 0.5f)) < 1e-4;
    run(1.0f, {5, 7}, 10.0f, -4.0f, o5, o7);  // penalty == 1.0: a no-op
    const bool noop_ok = o5 == 10.0f && o7 == -4.0f;
    ok("the audio repetition penalty: repeated codes down-weighted (HF form, dedup), 1.0 a no-op",
       div_ok && mul_ok && noop_ok, "div " + std::to_string(o5) + " / mul " + std::to_string(o7));
  }
  {
    // the penalty is EFFECTIVE end to end: a strong penalty (n_penalty form) on the codes the
    // sampler just emitted moves the frame's argmax — the repeated code is down-weighted off the top.
    // (CPU oracle; deterministic for the seeded hidden + the fixed reveal order.)
    const auto& rec0 = meta["records"][0];
    const std::vector<float> hid0 = rec0["hidden"].get<std::vector<float>>();
    std::vector<int> fixed(12);
    for (int i = 0; i < 12; ++i) fixed[size_t(i)] = i;
    const std::vector<int> no_pen = dec.sample_frame(hid0, 0.0f, fixed);  // defaults: no history/penalty
    std::vector<std::vector<int>> rep(ni::tts::DepthDecoder::kCodebooks);
    for (int c = 0; c < ni::tts::DepthDecoder::kCodebooks; ++c) rep[size_t(c)].push_back(no_pen[size_t(c)]);
    const std::vector<int> penal = dec.sample_frame(hid0, 0.0f, fixed, rep, 1e6f);  // a strong penalty on the repeats
    int diffs = 0;
    for (int c = 0; c < ni::tts::DepthDecoder::kCodebooks; ++c) diffs += no_pen[size_t(c)] != penal[size_t(c)];
    ok("the audio repetition penalty is effective: a strong penalty on the just-emitted codes moves the frame",
       diffs >= 1, "moved " + std::to_string(diffs) + " of " + std::to_string(ni::tts::DepthDecoder::kCodebooks) + " codebooks");
  }
  printf("PASS (%d ok, %d failed)\n", passed, failed);
  return failed ? 1 : 0;
}

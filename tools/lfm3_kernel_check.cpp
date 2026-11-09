// LFM3's kernels (src/kernels/lfm3.metal) against the CPU references (model/lfm3.cpp's lfm3_conv_ref) at small
// sizes, and the imported folder's tokenizer against texts given as arguments. Takes no GPU lock itself: run it
// under `lockf -k /tmp/kapitsa-gpu.lock` (its header says so, as fn_check's does). First runs of a new kernel go
// at these small sizes with MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1.
//   lfm3_kernel_check kernels        the conv prefill / decode / gate pack vs the references
//   lfm3_kernel_check tok DIR ...    encode each argument with DIR's tokenizer.json, one "tok <n> ids" line
#include "metal/gpu.h"
#include "model/lfm3.h"
#include "tokenizer/tokenizer.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_fail = 0, g_ok = 0;

void ok(const char* name, bool good, const std::string& detail = "") {
  if (good) {
    ++g_ok;
    printf("ok %s\n", name);
  } else {
    ++g_fail;
    fprintf(stderr, "FAIL %s%s%s\n", name, detail.empty() ? "" : ": ", detail.c_str());
  }
}

// the kernels' parameter structs (src/kernels/lfm3.metal), as the C++ side lays them out
struct Lfm3ConvParams {
  uint32_t T, H, K;
};
struct Lfm3GateParams {
  uint32_t T, Hq, D;
};

uint32_t lcg(uint32_t& s) { return s = s * 1664525u + 1013904223u; }

float rnd(uint32_t& s) { return float(int(lcg(s) % 401) - 200) / 16.f; }

uint16_t to_bf16(float f) {  // round-to-nearest-even-ish (truncation is fine for a check's inputs)
  uint32_t u;
  memcpy(&u, &f, 4);
  return uint16_t(u >> 16);
}

float from_bf16(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: lfm3_kernel_check kernels | tok DIR TEXT...\n");
    return 2;
  }
  const std::string mode = argv[1];

  if (mode == "tok") {
    if (argc < 3) return 2;
    ni::Tokenizer tok;
    tok.load(std::string(argv[2]) + "/tokenizer.json");
    for (int i = 3; i < argc; ++i) {
      const std::vector<int> ids = tok.encode(argv[i]);
      printf("tok %d", i - 3);
      for (int id : ids) printf(" %d", id);
      printf("\n");
    }
    return 0;
  }
  if (mode != "kernels") {
    fprintf(stderr, "usage: lfm3_kernel_check kernels | tok DIR TEXT...\n");
    return 2;
  }

  ni::Gpu& gpu = ni::Gpu::get();
  const uint32_t K = 7;
  for (const uint32_t H : {64u, 512u}) {
    const uint32_t T = 33;
    uint32_t seed = 12345 + H;
    std::vector<float> u(size_t(T * H)), y(size_t(T * H), 0.f), want(size_t(T * H)), ring_f(size_t(6 * H));
    std::vector<uint16_t> w(size_t(H * K)), ring(size_t(6 * H));
    for (auto& x : u) x = rnd(seed);
    for (auto& x : w) x = to_bf16(rnd(seed));
    for (auto& x : ring_f) x = rnd(seed);
    for (size_t i = 0; i < ring.size(); ++i) ring[i] = to_bf16(ring_f[i]);

    ni::Buffer bu = gpu.alloc(u.size() * 4, "u"), bw = gpu.alloc(w.size() * 2, "w"),
               br = gpu.alloc(ring.size() * 2, "ring"), by = gpu.alloc(y.size() * 4, "y"),
               bo = gpu.alloc(ring.size() * 2, "ring_out");
    memcpy(bu.ptr(), u.data(), u.size() * 4);
    memcpy(bw.ptr(), w.data(), w.size() * 2);
    memcpy(br.ptr(), ring.data(), ring.size() * 2);

    // the CPU reference over the widened buffers
    std::vector<float> wf(size_t(H * K)), rf(size_t(6 * H));
    for (size_t i = 0; i < wf.size(); ++i) wf[i] = from_bf16(w[i]);
    for (size_t i = 0; i < rf.size(); ++i) rf[i] = from_bf16(ring[i]);
    ni::lfm3_conv_ref(u.data(), int(T), int(H), int(K), wf.data(), rf.data(), want.data());

    // the kernel: prefill over T frames with the ring
    {
      Lfm3ConvParams p{T, H, K};
      auto e = gpu.begin();
      e->use(gpu.pipeline("lfm3_conv_prefill"));
      e->buf(0, bu);
      e->buf(1, bw);
      e->buf(2, br);
      e->buf(3, by);
      e->val(4, p);
      e->dispatch((H + 255) / 256, T, 1, 256);
      e->commit_and_wait();
      const float* got = by.as<float>();
      bool good = true;
      for (size_t i = 0; i < want.size(); ++i)
        if (fabsf(got[i] - want[i]) > 2e-4f * (1.f + fabsf(want[i]))) {
          good = false;
          if (g_fail == 0) fprintf(stderr, "  first diff at %zu: %g vs %g\n", i, got[i], want[i]);
          break;
        }
      ok((std::string("prefill H") + std::to_string(H) + ": the kernel is the reference").c_str(), good);
    }

    // the decode kernel: T single-frame steps with the rolled ring, equal to one prefill, and the ring it
    // leaves is the prefill's continuation (the last K-1 frames of ring ++ u)
    {
      std::vector<float> yd(size_t(T * H), 0.f);
      ni::Buffer bu1 = gpu.alloc(H * 4, "u1"), by1 = gpu.alloc(H * 4, "y1");
      ni::Buffer brin = gpu.alloc(ring.size() * 2, "rin"), brout = gpu.alloc(ring.size() * 2, "rout");
      memcpy(brin.ptr(), ring.data(), ring.size() * 2);
      std::vector<float> cur_ring = rf;
      Lfm3ConvParams p{1, H, K};
      bool ring_good = true;
      for (uint32_t t = 0; t < T; ++t) {
        memcpy(bu1.ptr(), &u[size_t(t) * H], H * 4);
        auto e = gpu.begin();
        e->use(gpu.pipeline("lfm3_conv_decode"));
        e->buf(0, bu1);
        e->buf(1, bw);
        e->buf(2, brin);
        e->buf(3, by1);
        e->buf(4, brout);
        e->val(5, p);
        e->dispatch((H + 255) / 256, 1, 1, 256);
        e->commit_and_wait();
        memcpy(&yd[size_t(t) * H], by1.as<float>(), H * 4);
        // the CPU's expected ring after this frame; the next step reads what this one wrote
        std::vector<float> joined(size_t((K - 1 + 1) * H));
        std::copy(cur_ring.begin(), cur_ring.end(), joined.begin());
        std::copy_n(&u[size_t(t) * H], H, joined.begin() + (K - 1) * H);
        std::copy(joined.begin() + H, joined.end(), cur_ring.begin());
        memcpy(brin.ptr(), brout.ptr(), ring.size() * 2);
        if (ring_good) {
          const uint16_t* got = brin.as<uint16_t>();
          for (size_t i = 0; i < ring.size(); ++i)
            if (fabsf(from_bf16(got[i]) - cur_ring[i]) > 2e-4f * (1.f + fabsf(cur_ring[i]))) {
              ok((std::string("decode H") + std::to_string(H) + ": the rolled ring").c_str(), false,
                 "step " + std::to_string(t) + ", entry " + std::to_string(i));
              ring_good = false;
              break;
            }
        }
      }
      if (ring_good) ok((std::string("decode H") + std::to_string(H) + ": the rolled ring").c_str(), true);
      bool good = true;
      for (size_t i = 0; i < yd.size(); ++i)
        if (fabsf(yd[i] - want[i]) > 2e-4f * (1.f + fabsf(want[i]))) {
          good = false;
          if (g_fail == 0) fprintf(stderr, "  first diff at %zu: %g vs %g\n", i, yd[i], want[i]);
          break;
        }
      ok((std::string("decode H") + std::to_string(H) + ": stepping matches the prefill").c_str(), good);
    }

    // the gate pack: qg = [q | the per-head scalar broadcast]
    {
      const uint32_t Hq = 4, D = 128;
      std::vector<float> q(size_t(T * Hq * D)), gate(size_t(T * Hq)), qg(size_t(T * Hq * 2 * D), 0.f);
      for (auto& x : q) x = rnd(seed);
      for (auto& x : gate) x = rnd(seed);
      ni::Buffer bq = gpu.alloc(q.size() * 4, "q"), bg = gpu.alloc(gate.size() * 4, "gate"),
                 bqg = gpu.alloc(qg.size() * 4, "qg");
      memcpy(bq.ptr(), q.data(), q.size() * 4);
      memcpy(bg.ptr(), gate.data(), gate.size() * 4);
      Lfm3GateParams p{T, Hq, D};
      auto e = gpu.begin();
      e->use(gpu.pipeline("lfm3_gate_pack"));
      e->buf(0, bq);
      e->buf(1, bg);
      e->buf(2, bqg);
      e->val(3, p);
      e->dispatch(Hq, T, 1, 1);
      e->commit_and_wait();
      const float* got = bqg.as<float>();
      bool good = true;
      for (uint32_t t = 0; t < T && good; ++t)
        for (uint32_t h = 0; h < Hq && good; ++h) {
          for (uint32_t d = 0; d < D && good; ++d) {
            const size_t at = (size_t(t) * Hq + h) * 2 * D + d;
            if (got[at] != q[(size_t(t) * Hq + h) * D + d] || got[at + D] != gate[size_t(t) * Hq + h]) good = false;
          }
        }
      ok("gate pack: q beside the broadcast gate", good);
    }
  }

  printf("%s (%d ok, %d failed)\n", g_fail ? "FAIL" : "PASS", g_ok, g_fail);
  return g_fail ? 1 : 0;
}

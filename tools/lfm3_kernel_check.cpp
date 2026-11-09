// LFM3's kernels (src/kernels/lfm3.metal) against the CPU references (model/lfm3.cpp's lfm3_conv_ref) at small
// sizes, and the imported folder's tokenizer against texts given as arguments. Takes no GPU lock itself: run it
// under `lockf -k /tmp/kapitsa-gpu.lock` (its header says so, as fn_check's does). First runs of a new kernel go
// at these small sizes with MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1.
//   lfm3_kernel_check kernels        the conv prefill / decode / gate pack vs the references
//   lfm3_kernel_check tok DIR ...    encode each argument with DIR's tokenizer.json, one "tok <n> ids" line
#include "metal/gpu.h"
#include "model/kparams.h"
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
struct Lfm3AttnParams {
  uint32_t Hq, Hkv, D, cap, pos, window;
  float scale;
};
struct Lfm3QkParams {
  uint32_t T, Hq, Hkv, D, cap, pos0;
};
struct Lfm3NormParams {
  uint32_t D;
  float eps;
};
struct Lfm3AttnPParams {
  uint32_t T, Hq, Hkv, D, cap, pos0, window;
  float scale;
};

uint32_t lcg(uint32_t& s) { return s = s * 1664525u + 1013904223u; }

float rnd(uint32_t& s) { return float(int(lcg(s) % 401) - 200) / 16.f; }

uint16_t to_bf16(float f) {  // round-to-nearest-even-ish (truncation is fine for a check's inputs)
  uint32_t u;
  memcpy(&u, &f, 4);
  return uint16_t(u >> 16);
}

std::vector<uint16_t> to_bf16_span(const std::vector<float>& v) {
  std::vector<uint16_t> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = to_bf16(v[i]);
  return out;
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

  // decode attention at head dim 128: GQA 4:1 and 1:1, full and sliding, first/middle/late positions,
  // against a double-precision reference of the same math
  {
    const uint32_t D = 128;
    const float scale = 1.f / std::sqrt(float(D));
    for (auto [Hq, Hkv] : {std::pair{4u, 1u}, {4u, 4u}}) {
      const uint32_t cap = 64;
      uint32_t seed = 777 + Hq;
      std::vector<float> q(size_t(Hq * D)), gate(size_t(Hq), 0.f), out(size_t(Hq * D), 0.f),
          want(size_t(Hq * D));
      std::vector<_Float16> kc(size_t(Hkv * cap * D)), vc(size_t(Hkv * cap * D));
      for (auto& x : q) x = rnd(seed);
      for (auto& x : gate) x = rnd(seed);
      for (auto& x : kc) x = _Float16(rnd(seed) / 4.f);
      for (auto& x : vc) x = _Float16(rnd(seed) / 4.f);
      ni::Buffer bq = gpu.alloc(q.size() * 4, "aq"), bkc = gpu.alloc(kc.size() * 2, "akc"),
                 bvc = gpu.alloc(vc.size() * 2, "avc"), bg = gpu.alloc(gate.size() * 4, "agate"),
                 bo = gpu.alloc(out.size() * 4, "aout");
      memcpy(bq.ptr(), q.data(), q.size() * 4);
      memcpy(bkc.ptr(), kc.data(), kc.size() * 2);
      memcpy(bvc.ptr(), vc.data(), vc.size() * 2);
      memcpy(bg.ptr(), gate.data(), gate.size() * 4);
      for (const uint32_t window : {0u, 8u}) {
        for (const uint32_t pos : {0u, 17u, 55u, 63u}) {
          Lfm3AttnParams p{Hq, Hkv, D, cap, pos, window, scale};
          auto e = gpu.begin();
          e->use(gpu.pipeline("lfm3_attn_d"));
          e->buf(0, bq);
          e->buf(1, bkc);
          e->buf(2, bvc);
          e->buf(3, bg);
          e->buf(4, bo);
          e->val(5, p);
          e->dispatch(Hq, 1, 1, 1);
          e->commit_and_wait();
          // the reference
          const uint32_t beg = window && pos >= window ? pos + 1 - window : 0;
          for (uint32_t h = 0; h < Hq; ++h) {
            const _Float16* kkh = kc.data() + uint64_t(h / (Hq / Hkv)) * cap * D;
            const _Float16* vvh = vc.data() + uint64_t(h / (Hq / Hkv)) * cap * D;
            double m = -1e300;
            std::vector<double> sc(size_t(pos - beg + 1));
            for (uint32_t t = beg; t <= pos; ++t) {
              double acc = 0;
              for (uint32_t d = 0; d < D; ++d)
                acc += double(q[size_t(h) * D + d]) * double(kkh[uint64_t(t) * D + d]);
              sc[size_t(t - beg)] = acc * scale;
              m = std::max(m, sc[size_t(t - beg)]);
            }
            double denom = 0;
            for (double s : sc) denom += std::exp(s - m);
            for (uint32_t d = 0; d < D; ++d) {
              double acc = 0;
              for (uint32_t t = beg; t <= pos; ++t)
                acc += std::exp(sc[size_t(t - beg)] - m) / denom * double(vvh[uint64_t(t) * D + d]);
              want[size_t(h) * D + d] = float(acc * (1.0 / (1.0 + std::exp(-double(gate[h])))));
            }
          }
          memcpy(out.data(), bo.as<float>(), out.size() * 4);
          bool good = true;
          for (size_t i = 0; i < want.size(); ++i)
            if (fabsf(out[i] - want[i]) > 2e-4f * (1.f + fabsf(want[i]))) {
              good = false;
              if (g_fail == 0)
                fprintf(stderr, "  first diff at %zu: %g vs %g (Hq %u/%u, w %u, pos %u)\n", i, out[i], want[i],
                        Hq, Hkv, window, pos);
              break;
            }
          ok((std::string("attn ") + std::to_string(Hq) + "/" + std::to_string(Hkv) + " w" +
              std::to_string(window) + " pos" + std::to_string(pos))
                 .c_str(),
             good);
        }
      }
    }
  }

  // res_norm, mul, qk_prep, prefill attention and the ring roll
  {
    const uint32_t H = 128, T = 9, K = 7, Hq = 4, Hkv = 2, D = 128, cap = 64;
    uint32_t seed = 31337;
    const ni::Lfm3Config c{};  // eps default irrelevant; the kernels take eps from the params
    std::vector<float> z(size_t(T * H)), x(size_t(T * H)), w(H);
    for (auto& v : z) v = rnd(seed);
    for (auto& v : x) v = rnd(seed);
    for (auto& v : w) v = rnd(seed);
    ni::Buffer bz = gpu.alloc(z.size() * 4, "rz"), bx = gpu.alloc(x.size() * 4, "rx"),
               bw = gpu.alloc(w.size() * 2, "rw"), bo = gpu.alloc(z.size() * 4, "ro");
    memcpy(bz.ptr(), z.data(), z.size() * 4);
    memcpy(bx.ptr(), x.data(), x.size() * 4);
    memcpy(bw.ptr(), to_bf16_span(w).data(), w.size() * 2);
    const Lfm3NormParams np{H, 1e-5f};
    auto e = gpu.begin();
    e->use(gpu.pipeline("lfm3_res_norm"));
    e->buf(0, bz);
    e->buf(1, bw);
    e->buf(2, bx);
    e->buf(3, bo);
    e->val(4, np);
    e->dispatch(T, 1, 1, 256);
    e->commit_and_wait();
    {
      std::vector<float> wf(w.size());
      for (size_t i = 0; i < w.size(); ++i) wf[i] = from_bf16(to_bf16(w[i]));
      bool good = true;
      for (uint32_t r = 0; r < T && good; ++r) {
        double ss = 0;
        for (uint32_t i = 0; i < H; ++i) ss += double(z[size_t(r) * H + i]) * z[size_t(r) * H + i];
        const double inv = 1.0 / std::sqrt(ss / H + 1e-5);
        for (uint32_t i = 0; i < H; ++i) {
          const float want = x[size_t(r) * H + i] + float(z[size_t(r) * H + i] * inv * wf[i]);
          good = fabsf(bo.as<float>()[size_t(r) * H + i] - want) <= 2e-4f * (1.f + fabsf(want));
          if (!good) break;
        }
      }
      ok("res_norm: the norm then the residual add", good);
    }
    {
      // mul
      ni::Buffer ba = gpu.alloc(z.size() * 4, "ma"), bb2 = gpu.alloc(x.size() * 4, "mb"),
                 bh = gpu.alloc(z.size() * 4, "mh");
      memcpy(ba.ptr(), z.data(), z.size() * 4);
      memcpy(bb2.ptr(), x.data(), x.size() * 4);
      const uint32_t n = uint32_t(z.size());
      auto e2 = gpu.begin();
      e2->use(gpu.pipeline("lfm3_mul"));
      e2->buf(0, ba);
      e2->buf(1, bb2);
      e2->buf(2, bh);
      e2->val(3, n);
      e2->dispatch((n + 255) / 256, 1, 1, 256);
      e2->commit_and_wait();
      bool good = true;
      for (size_t i = 0; i < z.size(); ++i) good = good && bh.as<float>()[i] == z[i] * x[i];
      ok("mul: the elementwise product", good);
    }
    {
      // qk_prep: qkv rows [T, QD + 2KD], the norms and the rope via lfm3_rope_ref, the cache writes
      const uint32_t QD = Hq * D, KD = Hkv * D;
      std::vector<float> qkv(size_t(T * (QD + 2 * KD))), qout(size_t(T * QD), 0.f), qw(size_t(D), 0.f),
        kw(size_t(D), 0.f), freqs(size_t(64), 0.f);
      std::vector<_Float16> kc(size_t(Hkv * cap * D), _Float16(0.f)), vc(size_t(Hkv * cap * D), _Float16(0.f));
      for (auto& v : qkv) v = rnd(seed);
      for (auto& v : qw) v = rnd(seed);
      for (auto& v : kw) v = rnd(seed);
      const std::vector<float> g = ni::lfm3_rope_freqs(1e6, 0.5, 128);
      for (size_t i = 0; i < 64; ++i) freqs[i] = g[i];
      ni::Buffer bqkv = gpu.alloc(qkv.size() * 4, "qk"), bqw = gpu.alloc(D * 2, "qw"), bkw = gpu.alloc(D * 2, "kw"),
                 bf = gpu.alloc(256, "fr"), bkc = gpu.alloc(kc.size() * 2, "kcc"), bvc = gpu.alloc(vc.size() * 2, "vcc"),
                 bqo = gpu.alloc(qout.size() * 4, "qo");
      memcpy(bqkv.ptr(), qkv.data(), qkv.size() * 4);
      memcpy(bqw.ptr(), to_bf16_span(qw).data(), D * 2);
      memcpy(bkw.ptr(), to_bf16_span(kw).data(), D * 2);
      memcpy(bf.ptr(), freqs.data(), 256);
      const uint32_t pos0 = 5;
      Lfm3QkParams qp{T, Hq, Hkv, D, cap, pos0};
      auto e3 = gpu.begin();
      e3->use(gpu.pipeline("lfm3_qk_prep"));
      e3->buf(0, bqkv);
      e3->buf(1, bqw);
      e3->buf(2, bkw);
      e3->buf(3, bf);
      e3->buf(4, bkc);
      e3->buf(5, bvc);
      e3->buf(6, bqo);
      e3->val(7, qp);
      e3->dispatch(Hq + 2 * Hkv, T, 1, 1);
      e3->commit_and_wait();
      memcpy(kc.data(), bkc.ptr(), kc.size() * 2);
      memcpy(vc.data(), bvc.ptr(), vc.size() * 2);
      bool good = true;
      std::vector<float> row(size_t(D), 0.f), roped(size_t(D), 0.f);
      for (uint32_t t = 0; t < T && good; ++t) {
        const uint64_t base = uint64_t(t) * (QD + 2 * KD);
        for (uint32_t h = 0; h < Hq && good; ++h) {
          // norm
          double ss = 0;
          for (uint32_t d = 0; d < D; ++d) {
            row[d] = qkv[base + uint64_t(h) * D + d];
            ss += double(row[d]) * row[d];
          }
          const double inv = 1.0 / std::sqrt(ss / D + 1e-5);
          std::vector<float> nw(size_t(D), 0.f);
          for (uint32_t d = 0; d < D; ++d) nw[d] = from_bf16(to_bf16(qw[d]));
          for (uint32_t d = 0; d < D; ++d) row[d] = float(row[d] * inv * nw[d]);
          ni::lfm3_rope_ref(row.data(), int(D), freqs.data(), double(pos0 + t), roped.data());
          for (uint32_t d = 0; d < D; ++d) {
            const float got = bqo.as<float>()[(size_t(t) * Hq + h) * D + d];
            if (fabsf(got - roped[d]) > 2e-4f * (1.f + fabsf(roped[d]))) {
              fprintf(stderr, "  q mismatch t%u h%u d%u: %g vs %g\n", t, h, d, got, roped[d]);
              good = false;
              break;
            }
          }
        }
        for (uint32_t h = 0; h < Hkv && good; ++h) {
          double ss = 0;
          for (uint32_t d = 0; d < D; ++d) {
            row[d] = qkv[base + QD + uint64_t(h) * D + d];
            ss += double(row[d]) * row[d];
          }
          const double inv = 1.0 / std::sqrt(ss / D + 1e-5);
          std::vector<float> nw(size_t(D), 0.f);
          for (uint32_t d = 0; d < D; ++d) nw[d] = from_bf16(to_bf16(kw[d]));
          for (uint32_t d = 0; d < D; ++d) row[d] = float(row[d] * inv * nw[d]);
          ni::lfm3_rope_ref(row.data(), int(D), freqs.data(), double(pos0 + t), roped.data());
          for (uint32_t d = 0; d < D; ++d) {
            const float got = float(kc[(uint64_t(h) * cap + pos0 + t) * D + d]);
            if (fabsf(got - roped[d]) > 2e-3f * (1.f + fabsf(roped[d]))) {
              fprintf(stderr, "  k mismatch t%u h%u d%u: %g vs %g\n", t, h, d, got, roped[d]);
              good = false;
              break;
            }
            const float gotv = float(vc[(uint64_t(h) * cap + pos0 + t) * D + d]);
            const float wantv = qkv[base + QD + KD + uint64_t(h) * D + d];
            if (fabsf(gotv - wantv) > 2e-3f * (1.f + fabsf(gotv))) {
              fprintf(stderr, "  v mismatch t%u h%u d%u: %g vs %g\n", t, h, d, gotv, wantv);
              good = false;
              break;
            }
          }
        }
      }
      ok("qk_prep: the norms, the zero-tail rope and the cache writes", good);
    }
    {
      // prefill attention: T rows, window 0 and 8, against the same double-precision reference
      const uint32_t cap = 64;
      const float scale = 1.f / std::sqrt(float(D));
      std::vector<float> q(size_t(T * Hq * D)), gate(size_t(T * Hq)), out(size_t(T * Hq * D), 0.f);
      std::vector<_Float16> kc(size_t(Hkv * cap * D)), vc(size_t(Hkv * cap * D));
      for (auto& v : q) v = rnd(seed);
      for (auto& v : gate) v = rnd(seed);
      for (auto& v : kc) v = _Float16(rnd(seed) / 4.f);
      for (auto& v : vc) v = _Float16(rnd(seed) / 4.f);
      ni::Buffer bq = gpu.alloc(q.size() * 4, "pq"), bkc = gpu.alloc(kc.size() * 2, "pkc"),
                 bvc = gpu.alloc(vc.size() * 2, "pvc"), bg = gpu.alloc(gate.size() * 4, "pgate"),
                 bo2 = gpu.alloc(out.size() * 4, "pout");
      memcpy(bq.ptr(), q.data(), q.size() * 4);
      memcpy(bkc.ptr(), kc.data(), kc.size() * 2);
      memcpy(bvc.ptr(), vc.data(), vc.size() * 2);
      memcpy(bg.ptr(), gate.data(), gate.size() * 4);
      for (const uint32_t window : {0u, 8u}) {
        const uint32_t pos0 = 3;
        Lfm3AttnPParams ap{T, Hq, Hkv, D, cap, pos0, window, scale};
        auto e4 = gpu.begin();
        e4->use(gpu.pipeline("lfm3_attn_p"));
        e4->buf(0, bq);
        e4->buf(1, bkc);
        e4->buf(2, bvc);
        e4->buf(3, bg);
        e4->buf(4, bo2);
        e4->val(5, ap);
        e4->dispatch(Hq, T, 1, 1);
        e4->commit_and_wait();
        bool good = true;
        for (uint32_t t = 0; t < T && good; ++t) {
          const uint32_t pos = pos0 + t, beg = window && pos >= window ? pos + 1 - window : 0;
          for (uint32_t h = 0; h < Hq && good; ++h) {
            const _Float16* kkh = kc.data() + uint64_t(h / (Hq / Hkv)) * cap * D;
            const _Float16* vvh = vc.data() + uint64_t(h / (Hq / Hkv)) * cap * D;
            double m = -1e300;
            std::vector<double> sc(size_t(pos - beg + 1));
            for (uint32_t s = beg; s <= pos; ++s) {
              double acc = 0;
              for (uint32_t d = 0; d < D; ++d) acc += double(q[(size_t(t) * Hq + h) * D + d]) * double(kkh[uint64_t(s) * D + d]);
              sc[size_t(s - beg)] = acc * scale;
              m = std::max(m, sc[size_t(s - beg)]);
            }
            double denom = 0;
            for (double s2 : sc) denom += std::exp(s2 - m);
            for (uint32_t d = 0; d < D; ++d) {
              double acc = 0;
              for (uint32_t s = beg; s <= pos; ++s)
                acc += std::exp(sc[size_t(s - beg)] - m) / denom * double(vvh[uint64_t(s) * D + d]);
              const float want = float(acc * (1.0 / (1.0 + std::exp(-double(gate[size_t(t) * Hq + h])))));
              const float got = bo2.as<float>()[ (size_t(t) * Hq + h) * D + d];
              good = fabsf(got - want) <= 2e-3f * (1.f + fabsf(want));
              if (!good) {
                fprintf(stderr, "  prefill attn diff t%u h%u d%u: %g vs %g\n", t, h, d, got, want);
                break;
              }
            }
          }
        }
        ok((std::string("prefill attention w") + std::to_string(window)).c_str(), good);
      }
    }
    {
      // the ring roll: the last K-1 frames of u, oldest first, as bf16
      std::vector<float> u(size_t(T * H));
      for (auto& v : u) v = rnd(seed);
      std::vector<uint16_t> ring(size_t((K - 1) * H), 0);
      ni::Buffer bu = gpu.alloc(u.size() * 4, "ru"), brg = gpu.alloc(ring.size() * 2, "rrg");
      memcpy(bu.ptr(), u.data(), u.size() * 4);
      Lfm3ConvParams cp{T, H, K};
      auto e5 = gpu.begin();
      e5->use(gpu.pipeline("lfm3_ring_roll"));
      e5->buf(0, bu);
      e5->buf(1, brg);
      e5->val(2, cp);
      e5->dispatch(H, 1, 1, 1);
      e5->commit_and_wait();
      bool good = true;
      for (uint32_t j = 0; j < K - 1; ++j)
        for (uint32_t i = 0; i < H; ++i) {
          const float want = u[size_t(T - (K - 1) + j) * H + i];
          good = good && fabsf(from_bf16(brg.as<uint16_t>()[size_t(j) * H + i]) - want) <= 2e-2f * (1.f + fabsf(want));
        }
      ok("ring roll: the chunk's last K-1 frames", good);
    }
  }

  printf("%s (%d ok, %d failed)\n", g_fail ? "FAIL" : "PASS", g_ok, g_fail);
  return g_fail ? 1 : 0;
}

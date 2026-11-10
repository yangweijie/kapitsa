// LFM3's kernels (src/kernels/lfm3.metal) against the CPU references (model/lfm3.cpp's lfm3_conv_ref) at small
// sizes, and the imported folder's tokenizer against texts given as arguments. Takes no GPU lock itself: run it
// under `lockf -k /tmp/kapitsa-gpu.lock` (its header says so, as fn_check's does). First runs of a new kernel go
// at these small sizes with MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1.
//   lfm3_kernel_check kernels        the conv prefill / decode / gate pack / bcx planes vs the references
//   lfm3_kernel_check logits DIR ... the backend's top-8 next-token logits for a prompt
//   lfm3_kernel_check sample        the two-stage sampler at the real vocab size
//   lfm3_kernel_check chain         the QK prep, the 4-bit GEMM and the decode attention chained
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
struct Lfm3BcxParams {
  uint32_t T, H, which;
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
    fprintf(stderr, "usage: lfm3_kernel_check kernels | logits DIR ID... | sample | chain | tok DIR TEXT...\n");
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
  if (mode == "logits") {
    // lfm3_kernel_check logits MODEL_DIR ID ID ID... : the backend's top-8 next-token logits for the prompt
    if (argc < 4) return 2;
    std::vector<int> ids;
    for (int i = 3; i < argc; ++i) ids.push_back(atoi(argv[i]));
    ni::Gpu::get().compile();
    ni::Gpu::get().compile_formats();  // (the qrf_f4g32 family lives in the second library: compile it before the encoder)
    ni::Lfm3 m(argv[2], 2048);
    auto s = m.new_seq(8192);
    memcpy(s->tok.as<int>(), ids.data(), ids.size() * 4);
    ni::SamplingParams sp{0.f, 1.0f, 0, 0};
    auto e = ni::Gpu::get().begin();
    m.encode_prefill(*e, *s, 0, int(ids.size()), sp, 0, true);
    e->commit_and_wait();
    const float* lg = m.logits_for_debug();
    std::vector<int> top(8);
    std::vector<float> val(8, -1e30f);
    for (int v = 0; v < m.vocab(); ++v)
      for (int j = 0; j < 8; ++j)
        if (lg[v] > val[j]) {
          for (int k = 7; k > j; --k) val[k] = val[k - 1], top[k] = top[k - 1];
          val[j] = lg[v], top[j] = v;
          break;
        }
    printf("kapitsa top8 ids:");
    for (int j = 0; j < 8; ++j) printf(" %d", top[j]);
    printf("\nkapitsa top8 logits:");
    for (int j = 0; j < 8; ++j) printf(" %.3f", val[j]);
    printf("\n");
    return 0;
  }
  if (mode == "sample") {
    // the engine's two-stage sampling exactly: topk_stage1 (63 threadgroups of 256 over a 128000 vocab) then
    // sample_stage2 (one threadgroup of 1024), with the buffer sizes the LFM3 backend allocates
    ni::Gpu& gpu = ni::Gpu::get();
    const uint32_t V = 128000, ntg = (V + 2047) / 2048;
    uint32_t seed = 99;
    std::vector<float> logits(size_t(V), 0.f);
    for (auto& v : logits) v = rnd(seed);
    logits[12345] = 88.f;  // the argmax for temp 0
    ni::Buffer bl = gpu.alloc(logits.size() * 4, "logits"), bv = gpu.alloc(size_t(ntg * 32) * 4, "cand_v"),
               bi = gpu.alloc(size_t(ntg * 32) * 4, "cand_i"), bt = gpu.alloc((8192 + 8) * 4, "tok"),
               bs = gpu.alloc(64 * 4, "sprob");
    memcpy(bl.ptr(), logits.data(), logits.size() * 4);
    memset(bt.ptr(), 0, bt.size);
    // the tied head GEMM as the engine runs it: N=128000, K=2048, 8-bit group 64, one row of a 40-row xh
    {
      const uint32_t H = 2048;
      std::vector<uint32_t> w(size_t(V) * (H / 8));
      std::vector<uint16_t> sb(size_t(V) * (H / 64));
      for (auto& v : w) v = lcg(seed);
      for (auto& v : sb) v = uint16_t(0x3f80);  // bf16 1.0
      std::vector<uint16_t> xh(size_t(40) * H);
      for (auto& v : xh) v = uint16_t(0x3c00);  // bf16... half 1.0
      ni::Buffer bw = gpu.alloc(w.size() * 4, "hw"), bs = gpu.alloc(sb.size() * 2, "hs"),
                 bb = gpu.alloc(sb.size() * 2, "hb"), bx = gpu.alloc(xh.size() * 2, "hx"),
                 by = gpu.alloc(size_t(V) * 4, "hy");
      memcpy(bw.ptr(), w.data(), w.size() * 4);
      memcpy(bs.ptr(), sb.data(), sb.size() * 2);
      memcpy(bb.ptr(), sb.data(), sb.size() * 2);
      memcpy(bx.ptr(), xh.data(), xh.size() * 2);
      const ni::kp::Qmm p{1, V, H, H, V, 0, 64, 0, 0};
      auto e0 = gpu.begin();
      e0->use(gpu.pipeline("qmr8_k32_w14_m16_n64"));
      e0->buf(0, bx);
      e0->buf(1, bw);
      e0->buf(2, bs);
      e0->buf(3, bb);
      e0->buf(4, by);
      e0->val(5, p);
      e0->dispatch((V + 63) / 64, 1, 1, 128);
      e0->commit_and_wait();
      ok("head: the tied 8-bit GEMM over the whole vocab", by.as<float>()[12345] != 0.f);
    }
    for (const float temp : {0.f, 0.1f}) {
      const uint32_t K = temp > 0 ? 32u : 1u;
      const ni::kp::TopkStage ts{V, K, 2048};
      auto e = gpu.begin();
      e->use(gpu.pipeline("topk_stage1"));
      e->buf(0, bl);
      e->buf(1, bv);
      e->buf(2, bi);
      e->val(3, ts);
      e->dispatch(ntg, 1, 1, 256);
      const ni::kp::Sample sp2{ntg * K, K, temp, 0.95f, 7u, 3u, 5u};
      e->use(gpu.pipeline("sample_stage2"));
      e->buf(0, bv);
      e->buf(1, bi);
      e->buf(2, bt);
      e->buf(3, bs);
      e->val(4, sp2);
      e->dispatch(1, 1, 1, 1024);
      e->commit_and_wait();
      const int got = bt.as<int>()[5];
      char nm[64];
      snprintf(nm, sizeof nm, "sample: temp %.1f draws a vocab id", temp);
      ok(nm, got >= 0 && got < int(V), "got " + std::to_string(got));
      if (temp == 0.f) ok("sample: temp 0 takes the argmax", got == 12345, "got " + std::to_string(got));
    }
    printf("%s (%d ok, %d failed)\n", g_fail ? "FAIL" : "PASS", g_ok, g_fail);
    return g_fail ? 1 : 0;
  }
  if (mode == "chain") {
    // the decode step at the engine's real dimensions: the attention half of layer 2 (the first attention
    // layer) — the qkv linears' packed row, the QK prep, then the decode attention
    ni::Gpu& gpu = ni::Gpu::get();
    const uint32_t H = 2048, QD = 2048, KD = 512, Hq = 16, Hkv = 4, D = 128, cap = 8192, pos = 10;
    uint32_t seed = 555;
    std::vector<float> qkv(size_t(QD + 2 * KD)), q(size_t(QD), 0.f), gate(size_t(Hq), 0.f),
        ao(size_t(QD), 0.f),
        qw(D), kw(D), freqs(64, 0.f);
    std::vector<uint16_t> wq(size_t(QD * (H / 8)), uint16_t(0)), wk(size_t(KD * (H / 8)), uint16_t(0)),
        wv(size_t(KD * (H / 8)), uint16_t(0)), wo(size_t(H * (QD / 8)), uint16_t(0)),
        wg(size_t(Hq * (H / 8)), uint16_t(0)), sq(size_t(QD * (H / 64)), uint16_t(0)),
        sk(size_t(KD * (H / 64)), uint16_t(0)), sv(size_t(KD * (H / 64)), uint16_t(0)),
        so(size_t(H * (QD / 64)), uint16_t(0)), sg(size_t(Hq * (H / 64)), uint16_t(0));
    std::vector<_Float16> kc(size_t(Hkv * cap * D), _Float16(0.f)), vc(size_t(Hkv * cap * D), _Float16(0.f));
    std::vector<uint16_t> xh2(size_t(40) * H, uint16_t(0x3c00));  // half 1.0, the engine's xh_ sizing
    for (auto& v : qkv) v = rnd(seed);
    for (auto& v : gate) v = rnd(seed);
    for (auto& v : qw) v = uint16_t(lcg(seed) % 0x10000);
    for (auto& v : sq) v = uint16_t(0x3f80);
    wk = wq, sk = sq, wv = wq, sv = sq, wo = wq, so = sq, wg = wq, sg = sq;  // shapes, not values
    const std::vector<float> g = ni::lfm3_rope_freqs(1e6, 0.5, 128);
    for (size_t i = 0; i < 64; ++i) freqs[i] = g[i];
    ni::Buffer bqkv = gpu.alloc(qkv.size() * 4, "cqkv"), bqw = gpu.alloc(qw.size() * 2, "cqw"),
               bkw = gpu.alloc(kw.size() * 2, "ckw"), bf = gpu.alloc(256, "cf"), bkc = gpu.alloc(kc.size() * 2, "ckc"),
               bvc = gpu.alloc(vc.size() * 2, "cvc"), bq = gpu.alloc(q.size() * 4, "cq"),
               bg = gpu.alloc(gate.size() * 4, "cg"), bao = gpu.alloc(ao.size() * 4, "cao");
    memcpy(bqkv.ptr(), qkv.data(), qkv.size() * 4);
    memcpy(bqw.ptr(), qw.data(), qw.size() * 2);
    memcpy(bkw.ptr(), kw.data(), kw.size() * 2);
    memcpy(bf.ptr(), freqs.data(), 256);
    memcpy(bg.ptr(), gate.data(), gate.size() * 4);
    const Lfm3QkParams qp{1, Hq, Hkv, D, cap, pos};
    auto e = gpu.begin();
    e->use(gpu.pipeline("lfm3_qk_prep"));
    e->buf(0, bqkv);
    e->buf(1, bqw);
    e->buf(2, bkw);
    e->buf(3, bf);
    e->buf(4, bkc);
    e->buf(5, bvc);
    e->buf(6, bq);
    e->val(7, qp);
    e->dispatch(Hq + 2 * Hkv, 1, 1, 1);
    e->commit_and_wait();
    printf("qk_prep ok\n");
    // the decode GEMM the engine faults on: 4-bit group 32, N=6144, K=2048, M=1 (the conv in_proj's shape)
    {
      const uint32_t N = 6144, K = 2048;
      std::vector<uint32_t> w(size_t(N) * (K / 8));
      std::vector<uint16_t> sb(size_t(N) * (K / 64));
      for (auto& v : w) v = lcg(seed);
      for (auto& v : sb) v = uint16_t(0x3f80);
      std::vector<float> yout(size_t(N), 0.f);
      ni::Buffer bw2 = gpu.alloc(w.size() * 4, "gw"), bs2 = gpu.alloc(sb.size() * 2, "gs"),
                 bb2 = gpu.alloc(sb.size() * 2, "gb"), bx2 = gpu.alloc(xh2.size() * 2, "gx"),
                 by2 = gpu.alloc(yout.size() * 4, "gy");
      memcpy(bw2.ptr(), w.data(), w.size() * 4);
      memcpy(bs2.ptr(), sb.data(), sb.size() * 2);
      memcpy(bb2.ptr(), sb.data(), sb.size() * 2);
      memcpy(bx2.ptr(), xh2.data(), xh2.size() * 2);
      const ni::kp::Qmm p2{1, N, K, K, N, 0, 32, 0, 0};
      auto e3 = gpu.begin();
      e3->use(gpu.pipeline("qrf_f4g32_m8_n64"));
      e3->buf(0, bx2);
      e3->buf(1, bw2);
      e3->buf(2, bs2);
      e3->buf(3, bb2);
      e3->buf(4, by2);
      e3->val(5, p2);
      e3->dispatch((N + 63) / 64, 1, 1, 64);
      e3->commit_and_wait();
      ok("decode GEMM: 4-bit group 32 at the in_proj's shape", by2.as<float>()[0] != 0.f);
    }
    const Lfm3AttnParams ap{Hq, Hkv, D, cap, pos, 0, 1.f / std::sqrt(float(D))};
    auto e2 = gpu.begin();
    e2->use(gpu.pipeline("lfm3_attn_d"));
    e2->buf(0, bq);
    e2->buf(1, bkc);
    e2->buf(2, bvc);
    e2->buf(3, bg);
    e2->buf(4, bao);
    e2->val(5, ap);
    e2->dispatch(Hq, 1, 1, 1);
    e2->commit_and_wait();
    printf("attn_d ok: ao[0]=%g\n", bao.as<float>()[0]);
    printf("%s (%d ok, %d failed)\n", g_fail ? "FAIL" : "PASS", g_ok, g_fail);
    return g_fail ? 1 : 0;
  }
  if (mode != "kernels") {
    fprintf(stderr, "usage: lfm3_kernel_check kernels | logits DIR ID... | sample | chain | tok DIR TEXT...\n");
    return 2;
  }

  ni::Gpu& gpu = ni::Gpu::get();

  // the in_proj output is [T, 3H] with (B, C, x) inside each row: the gates must read the planes with a
  // stride, not as three planar [T, H] buffers (a T = 1 decode cannot tell the difference)
  {
    const uint32_t T = 5, H = 64;
    uint32_t seed = 777;
    std::vector<float> bcx(size_t(T) * 3 * H), z(size_t(T) * H), want_u(size_t(T) * H),
        want_y(size_t(T) * H);
    for (auto& v : bcx) v = rnd(seed);
    for (auto& v : z) v = rnd(seed);
    for (uint32_t t = 0; t < T; ++t)
      for (uint32_t c = 0; c < H; ++c) {
        const size_t row = size_t(t) * 3 * H;
        want_u[size_t(t) * H + c] = bcx[row + c] * bcx[row + 2 * H + c];
        want_y[size_t(t) * H + c] = bcx[row + H + c] * z[size_t(t) * H + c];
      }
    ni::Buffer b = gpu.alloc(bcx.size() * 4, "bcx"), bz = gpu.alloc(z.size() * 4, "z"),
               bu = gpu.alloc(want_u.size() * 4, "bu"), by = gpu.alloc(want_y.size() * 4, "by");
    memcpy(b.ptr(), bcx.data(), bcx.size() * 4);
    memcpy(bz.ptr(), z.data(), z.size() * 4);
    for (uint32_t which : {0u, 1u}) {
      const Lfm3BcxParams p{T, H, which};
      auto e = gpu.begin();
      e->use(gpu.pipeline("lfm3_bcx"));
      e->buf(0, b);
      e->buf(1, which == 0 ? b : bz);
      e->buf(2, which == 0 ? bu : by);
      e->val(3, p);
      e->dispatch((H + 255) / 256, T, 1, 256);
      e->commit_and_wait();
      const float* got = (which == 0 ? bu : by).as<float>();
      const std::vector<float>& want = which == 0 ? want_u : want_y;
      bool good = true;
      for (size_t i = 0; i < want.size(); ++i)
        if (fabsf(got[i] - want[i]) > 1e-6f * (1.f + fabsf(want[i]))) {
          good = false;
          break;
        }
      ok(which == 0 ? "bcx: u reads B and x from each row" : "bcx: y reads C from each row times z", good);
    }
  }

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

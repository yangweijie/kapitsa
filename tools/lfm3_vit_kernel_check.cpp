// LFM3-Omni's vision kernels (src/kernels/vit.metal) against the CPU references
// (src/vision/lfm3_vit_ref.h and the independent computations below), at small sizes.
//   lfm3_vit_kernel_check --cpu     no GPU, no model: the oracle geometry, the CPU math the kernels
//                                  implement, and the params' layout. CI runs this mode.
//   lfm3_vit_kernel_check kernels  GPU: the 6 kernels' outputs vs the same CPU references, tiny
//                                  synthetic tensors. Takes no GPU lock itself: run it under
//                                  `lockf -k /tmp/kapitsa-gpu.lock` with
//                                  MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 (AGENTS.md).
#include "vision/lfm3_vit_ref.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// GPU mode is compiled in (the CI runner has Metal headers) but only entered on `kernels`.
#include "metal/gpu.h"
#include "util/bf16.h"

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

bool near(float a, float b, float tol = 2e-3f) { return std::fabs(a - b) <= tol * (1.f + std::fabs(b)); }

// ---- the kernel params exactly as vit.metal declares them (verified by static_assert) ----
struct VitRmsParams {
  uint32_t rows, D, x_ld, y_ld;
  float eps;
};
struct VitActParamsX {
  uint32_t rows, N, ld, act, out_ld;
};
struct Lfm3QkvParams {
  uint32_t rows, r0, N, heads, kv_heads, HD;
};
struct Lfm3ConvBxParams {
  uint32_t rows, r0, N, H, gh, gw, merge;
};
struct Lfm3ConvDwParams {
  uint32_t gh, gw, H, N;
};
static_assert(sizeof(VitRmsParams) == 20, "VitRmsParams must match vit.metal");
static_assert(sizeof(Lfm3QkvParams) == 24, "Lfm3QkvParams must match vit.metal");
static_assert(sizeof(Lfm3ConvBxParams) == 28, "Lfm3ConvBxParams must match vit.metal");
static_assert(sizeof(Lfm3ConvDwParams) == 16, "Lfm3ConvDwParams must match vit.metal");

// ---- the CPU references the kernels must match ----
// vit_rms: T5-style RMSNorm of fp32 rows -> half rows, by a bf16 weight, eps 1e-5 (tower norm_eps).
std::vector<float> ref_rms(const std::vector<float>& x, const std::vector<float>& w, int rows, int D, float eps) {
  std::vector<float> out(size_t(rows) * D);
  for (int r = 0; r < rows; ++r) {
    float ss = 0.f;
    for (int c = 0; c < D; ++c) ss += x[size_t(r) * D + c] * x[size_t(r) * D + c];
    const float scale = 1.f / std::sqrt(ss / D + eps);
    for (int c = 0; c < D; ++c) out[size_t(r) * D + c] = w[c] * (x[size_t(r) * D + c] * scale);
  }
  return out;
}

// The doubled axial angle table, indexed by d over the full head dim:
// [h(16), w(16), h(16), w(16)] with rotate_half pairing i <-> i+HD/2.
std::vector<float> ref_roped_q(const std::vector<float>& q, const std::vector<float>& qn,
                               int ph, int pw, double theta, int HD) {
  const int axis = HD / 4;
  std::vector<double> cosv(HD), sinv(HD);
  for (int i = 0; i < axis; ++i) {
    const double f = 1.0 / std::pow(theta, double(2 * i) / (2 * axis));
    const double a = ph * f, b = pw * f;
    cosv[i] = std::cos(a); sinv[i] = std::sin(a);
    cosv[axis + i] = std::cos(b); sinv[axis + i] = std::sin(b);
    cosv[2 * axis + i] = cosv[i]; sinv[2 * axis + i] = sinv[i];
    cosv[3 * axis + i] = cosv[axis + i]; sinv[3 * axis + i] = sinv[axis + i];
  }
  float ss = 0.f;
  for (int d = 0; d < HD; ++d) ss += q[d] * q[d];
  const float scale = 1.f / std::sqrt(ss / HD + 1e-5f);
  std::vector<float> out(HD);
  for (int d = 0; d < HD; ++d) {
    const float v = qn[d] * (q[d] * scale);
    const float partner = d < HD / 2 ? -q[d + HD / 2] * qn[d + HD / 2] * scale
                                     : q[d - HD / 2] * qn[d - HD / 2] * scale;
    out[d] = float(v * cosv[d] + partner * sinv[d]);
  }
  return out;
}

// lfm3vit_conv_dw's raster: row r of the merge-block sequence is
// ((bh*(gw/2)+bw)*2+mh)*2+mw; the 3x3 neighbours are looked up through merge_index.
std::vector<float> ref_conv(const std::vector<float>& bx, const std::vector<float>& C,
                            const std::vector<float>& w, int gh, int gw, int H) {
  const int N = gh * gw;
  std::vector<float> out(size_t(N) * H);
  for (int r = 0; r < N; ++r) {
    const int mw = r % 2, mh = (r / 2) % 2, rest = r / 4;
    const int bw = rest % (gw / 2), bh = rest / (gw / 2);
    const int y = bh * 2 + mh, x = bw * 2 + mw;
    for (int c = 0; c < H; ++c) {
      float acc = 0.f;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int yy = y + dy, xx = x + dx;
          if (yy < 0 || yy >= gh || xx < 0 || xx >= gw) continue;
          const int src = ni::vision::lfm3_vit_merge_index(yy, xx, gw, 2);
          acc += bx[size_t(src) * H + c] * w[size_t(c) * 9 + (dy + 1) * 3 + (dx + 1)];
        }
      out[size_t(r) * H + c] = C[size_t(r) * H + c] * acc;
    }
  }
  return out;
}

std::vector<float> ref_swiglu(const std::vector<float>& a, const std::vector<float>& b) {
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] / (1.f + std::exp(-a[i])) * b[i];
  return out;
}

// ---- CPU-only checks: these are what CI runs (no Gpu::get, no model, no Metal) ----
int cpu_checks() {
  using namespace ni::vision;
  // the oracle geometry (the same goldens lfm3_vit_ref_test pins, but here they guard the KERNELS)
  ok("cpu: half-to-even round (48px -> 1 factor)", lfm3_vit_py_round(1.5) == 2.0 && lfm3_vit_py_round(2.5) == 2.0);
  ok("cpu: smart_resize keeps 512x512", lfm3_vit_smart_resize(512, 512).h == 512);
  const auto ids = lfm3_vit_position_ids(4, 4, 2);
  bool pos_ok = ids.size() == 16 && ids[2][0] == 1 && ids[2][1] == 0 && ids[1][0] == 0 && ids[1][1] == 1;
  ok("cpu: merge-block position ids (h,w) order", pos_ok);
  bool perm_ok = true;
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x)
      perm_ok &= lfm3_vit_raster_coords(lfm3_vit_merge_index(y, x, 4, 2), 4, 2) == std::array<int, 2>{y, x};
  ok("cpu: merge_index is the raster inverse", perm_ok);

  // RMSNorm vs the header's oracle
  std::vector<float> x(64), w(64), got(64), want(64);
  for (int i = 0; i < 64; ++i) x[i] = 0.1f * (i - 32), w[i] = 1.f + 0.01f * i;
  lfm3_vit_rms_norm(x.data(), w.data(), 64, 1e-5f, want.data());
  const std::vector<float> rms = ref_rms(x, w, 1, 64, 1e-5f);
  bool rms_ok = true;
  for (int i = 0; i < 64; ++i) rms_ok &= near(rms[i], want[i], 1e-6f);
  ok("cpu: vit_rms's reference == lfm3_vit_rms_norm", rms_ok);

  // rope: the doubled table is indexed by d (the V4 bug: cosv[32] and j=d-32)
  std::vector<float> q(64), qn(64);
  for (int i = 0; i < 64; ++i) q[i] = 0.3f * std::sin(float(i)), qn[i] = 1.f + 0.02f * i;
  const std::vector<float> roped = ref_roped_q(q, qn, 0, 0, kLfm3RopeTheta, 64);
  bool id_ok = true;
  for (int i = 0; i < 64; ++i) id_ok &= near(roped[i], qn[i] * (q[i] * float(1. / std::sqrt(34.3f))), 0.03f);
  // at (0,0) every angle is zero: out[d] = norm(q)[d] exactly, for BOTH halves
  bool both_ok = true;
  for (int i = 0; i < 64; ++i) {
    float ss = 0.f;
    for (int j = 0; j < 64; ++j) ss += q[j] * q[j];
    const float s = 1.f / std::sqrt(ss / 64 + 1e-5f);
    both_ok &= near(roped[i], qn[i] * q[i] * s, 1e-5f);
  }
  ok("cpu: rope at (0,0) is the identity over all 64 dims", both_ok);
  const std::vector<float> roped2 = ref_roped_q(q, qn, 3, 5, kLfm3RopeTheta, 64);
  bool angle_ok = false;
  for (int i = 0; i < 64; ++i) angle_ok |= std::fabs(roped2[i] - roped[i]) > 1e-4f;
  ok("cpu: rope angles depend on (h,w)", angle_ok);

  // conv: the raster neighbour map + the C gate
  const int gh = 4, gw = 4, H = 8, N = 16;
  std::vector<float> bx(size_t(N) * H), C(size_t(N) * H), w9(size_t(H) * 9);
  for (size_t i = 0; i < bx.size(); ++i) bx[i] = 0.1f * (int(i) % 7 - 3);
  for (size_t i = 0; i < C.size(); ++i) C[i] = 0.2f * (int(i) % 5 - 2);
  for (size_t i = 0; i < w9.size(); ++i) w9[i] = 0.05f * (int(i) % 9 - 4);
  const std::vector<float> conv = ref_conv(bx, C, w9, gh, gw, H);
  // the MINE-vs-OTHER row decomposition: row 2 = (y=1,x=0), row 1 = (y=0,x=1)
  ok("cpu: conv row 2 is raster (1,0), not (0,1)",
     lfm3_vit_raster_coords(2, gw, 2) == std::array<int, 2>{1, 0});
  bool conv_ok = true;
  for (int r = 0; r < N && conv_ok; ++r)
    for (int c = 0; c < H && conv_ok; ++c) {
      // a hand-walk of the HF view/permute/reshape (independent of merge_index)
      const auto [y, x] = lfm3_vit_raster_coords(r, gw, 2);
      float acc = 0.f;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          const int yy = y + dy, xx = x + dx;
          if (yy < 0 || yy >= gh || xx < 0 || xx >= gw) continue;
          const int src = ((yy / 2 * (gw / 2) + xx / 2) * 2 + yy % 2) * 2 + xx % 2;
          acc += bx[size_t(src) * H + c] * w9[size_t(c) * 9 + (dy + 1) * 3 + (dx + 1)];
        }
      const float want = C[size_t(r) * H + c] * acc;
      if (!near(conv[size_t(r) * H + c], want, 1e-6f)) conv_ok = false;
    }
  ok("cpu: conv's CPU reference == the HF permute walk", conv_ok);

  // swiglu
  std::vector<float> a(32), b(32);
  for (int i = 0; i < 32; ++i) a[i] = 0.5f * (i - 16), b[i] = 0.3f * (16 - i);
  const std::vector<float> sw = ref_swiglu(a, b);
  bool sw_ok = true;
  for (int i = 0; i < 32; ++i) sw_ok &= near(sw[i], a[i] / (1.f + std::exp(-a[i])) * b[i], 1e-6f);
  ok("cpu: swiglu reference is silu(w1)*w3", sw_ok);

  printf("%s (%d ok, %d failed)\n", g_fail ? "FAIL" : "PASS", g_ok, g_fail);
  return g_fail ? 1 : 0;
}

}  // namespace

// ---- GPU checks: tiny synthetic tensors, the kernel output diffed against the same references ----
// The kernels are in the shared library compiled from vit.metal; no model folder, no weights.
int kernel_checks() {
  ni::Gpu& gpu = ni::Gpu::get();
  uint32_t seed = 90210;
  auto lcg = [](uint32_t& s) -> uint32_t { return s = s * 1664525u + 1013904223u; };
  auto rnd = [&]() { return float(int(lcg(seed) % 401) - 200) / 32.f; };
  auto bf16 = [](float f) { return ni::f32_to_bf16(f); };
  auto bf16f = [](uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
  };
  auto half_f = [](_Float16 h) { return float(h); };

  auto buffer = [&](const void* src, size_t bytes, const char* label) {
    ni::Buffer b = gpu.alloc(bytes + 16, label);
    memcpy(b.ptr(), src, bytes);
    return b;
  };
  auto diff_ok = [&](const char* name, const float* got, const std::vector<float>& want, float tol) {
    bool good = true;
    float worst = 0.f;
    for (size_t i = 0; i < want.size(); ++i) {
      const float d = std::fabs(got[i] - want[i]);
      worst = std::max(worst, d);
      if (d > tol * (1.f + std::fabs(want[i]))) good = false;
    }
    char detail[96];
    snprintf(detail, sizeof detail, "max|d| %.6f", worst);
    ok(name, good, good ? "" : detail);
  };

  // 1) vit_rms: rows=6, D=64 (2 threadgroups of 8 rows, 32 lanes)
  {
    const int rows = 6, D = 64;
    std::vector<float> x(size_t(rows) * D), w(D);
    std::vector<uint16_t> wb(D);
    for (auto& v : x) v = rnd();
    for (int i = 0; i < D; ++i) w[i] = 1.f + 0.05f * (i % 7), wb[i] = bf16(w[i]);
    for (int i = 0; i < D; ++i) w[i] = bf16f(wb[i]);  // the kernel reads bf16: the reference must too
    const std::vector<float> want = ref_rms(x, w, rows, D, 1e-5f);
    ni::Buffer bx = buffer(x.data(), x.size() * 4, "x"), bw = buffer(wb.data(), wb.size() * 2, "w"),
               by = gpu.alloc(size_t(rows) * D * 2, "y");
    const VitRmsParams p{uint32_t(rows), uint32_t(D), uint32_t(D), uint32_t(D), 1e-5f};
    auto e = gpu.begin();
    e->use(gpu.pipeline("vit_rms"));
    e->buf(0, bx);
    e->buf(1, bw);
    e->buf(2, by);
    e->val(3, p);
    e->dispatch((rows + 7) / 8, 1, 1, 256);
    e->commit_and_wait();
    std::vector<float> got(size_t(rows) * D);
    for (size_t i = 0; i < got.size(); ++i) got[i] = half_f(by.as<_Float16>()[i]);
    diff_ok("rms: vit_rms vs the CPU RMSNorm", got.data(), want, 1e-3f);
  }

  // 2) vit_f32_from_h: an identity cast the embedding_norm path uses
  {
    const int n = 512;
    std::vector<_Float16> h(n);
    std::vector<float> want(n);
    for (int i = 0; i < n; ++i) h[i] = _Float16(rnd()), want[i] = float(h[i]);
    ni::Buffer bh = buffer(h.data(), n * 2, "h"), by = gpu.alloc(size_t(n) * 4, "y");
    auto e = gpu.begin();
    e->use(gpu.pipeline("vit_f32_from_h"));
    e->buf(0, bh);
    e->buf(1, by);
    e->val(2, uint32_t(n));
    e->dispatch((n + 255) / 256, 1, 1, 256);
    e->commit_and_wait();
    diff_ok("cast: vit_f32_from_h vs half->float", by.as<float>(), want, 0.f);
  }

  // 3) lfm3vit_qkv_split: the V4 bugs (the doubled angle table, the sign, the d index) live here.
  //    Tiny shape: N=4 rows, heads=2, kv_heads=1, HD=64 (the real tower's head dim).
  {
    const int N = 4, heads = 2, kv = 1, HD = 64, rowdim = 2 * heads * HD;
    std::vector<float> y(size_t(N) * rowdim), qn(HD), kn(HD), invf(HD / 4);
    std::vector<uint16_t> qnb(HD), knb(HD), bias(2 * heads * HD, 0);
    std::vector<int32_t> pos(size_t(N) * 2);
    for (auto& v : y) v = rnd();
    for (int i = 0; i < HD; ++i) qn[i] = 1.f + 0.03f * (i % 5), kn[i] = 1.f + 0.02f * (i % 3);
    for (int i = 0; i < HD; ++i) qnb[i] = bf16(qn[i]), knb[i] = bf16(kn[i]);
    for (int i = 0; i < HD; ++i) qn[i] = bf16f(qnb[i]), kn[i] = bf16f(knb[i]);  // bf16 in the kernel
    for (int i = 0; i < HD / 4; ++i) invf[i] = float(1.0 / std::pow(ni::vision::kLfm3RopeTheta, double(2 * i) / (HD / 2)));
    for (int r = 0; r < N; ++r) pos[2 * r] = r / 2, pos[2 * r + 1] = r % 2 + 1;  // (h,w) per row
    ni::Buffer by = buffer(y.data(), y.size() * 4, "y"), bb = buffer(bias.data(), bias.size() * 2, "bias"),
               bqn = buffer(qnb.data(), qnb.size() * 2, "qn"), bkn = buffer(knb.data(), knb.size() * 2, "kn"),
               bpos = buffer(pos.data(), pos.size() * 4, "pos"), binv = buffer(invf.data(), invf.size() * 4, "invf"),
               bQ = gpu.alloc(size_t(heads) * N * HD * 2, "Q"), bK = gpu.alloc(size_t(heads) * N * HD * 2, "K"),
               bV = gpu.alloc(size_t(heads) * N * HD * 2, "V"), bQg = gpu.alloc(size_t(heads) * N * HD * 2, "Qg"),
               bKg = gpu.alloc(size_t(heads) * N * HD * 2, "Kg");
    const Lfm3QkvParams p{uint32_t(N), 0u, uint32_t(N), uint32_t(heads), uint32_t(kv), uint32_t(HD)};
    auto e = gpu.begin();
    e->use(gpu.pipeline("lfm3vit_qkv_split"));
    e->buf(0, by);
    e->buf(1, bb);
    e->buf(2, bQ);
    e->buf(3, bK);
    e->buf(4, bV);
    e->buf(5, bqn);
    e->buf(6, bkn);
    e->buf(7, bpos);
    e->buf(8, binv);
    e->buf(9, bQg);
    e->buf(10, bKg);
    e->val(11, p);
    e->dispatch(N, 1, 1, 1);
    e->commit_and_wait();
    // the reference: per (row, head), QK-RMSNorm then the doubled-table rope; K/V broadcast to heads
    std::vector<float> wantQ(size_t(heads) * N * HD), wantK(size_t(heads) * N * HD), wantV(size_t(heads) * N * HD);
    for (int r = 0; r < N; ++r) {
      for (int h = 0; h < heads; ++h) {
        std::vector<float> q(y.data() + size_t(r) * rowdim + h * HD, y.data() + size_t(r) * rowdim + (h + 1) * HD);
        const std::vector<float> qr = ref_roped_q(q, qn, pos[2 * r], pos[2 * r + 1], ni::vision::kLfm3RopeTheta, HD);
        memcpy(wantQ.data() + (size_t(h) * N + r) * HD, qr.data(), HD * 4);
        // kv head h/(heads/kv) = 0 for both heads here
        std::vector<float> k(y.data() + size_t(r) * rowdim + heads * HD,
                             y.data() + size_t(r) * rowdim + (heads + 1) * HD);
        const std::vector<float> kr = ref_roped_q(k, kn, pos[2 * r], pos[2 * r + 1], ni::vision::kLfm3RopeTheta, HD);
        memcpy(wantK.data() + (size_t(h) * N + r) * HD, kr.data(), HD * 4);
        memcpy(wantV.data() + (size_t(h) * N + r) * HD,
               y.data() + size_t(r) * rowdim + (heads + kv) * HD, HD * 4);
      }
    }
    std::vector<float> gotQ(size_t(heads) * N * HD), gotK(gotQ.size()), gotV(gotQ.size()), gotQr(gotQ.size()), gotKr(gotQ.size());
    for (size_t i = 0; i < gotQ.size(); ++i) {
      gotQ[i] = half_f(bQg.as<_Float16>()[i]);
      gotK[i] = half_f(bKg.as<_Float16>()[i]);
      gotV[i] = half_f(bV.as<_Float16>()[i]);
      gotQr[i] = half_f(bQ.as<_Float16>()[i]);
      gotKr[i] = half_f(bK.as<_Float16>()[i]);
    }
    diff_ok("qkv: roped Q vs the doubled-table reference", gotQ.data(), wantQ, 2e-3f);
    diff_ok("qkv: roped K vs the doubled-table reference", gotK.data(), wantK, 2e-3f);
    diff_ok("qkv: V broadcast to every query head", gotV.data(), wantV, 0.f);
    // the raw (pre-rope) Q/K copies must also be right
    std::vector<float> wantQr(size_t(heads) * N * HD), wantKr(wantQr.size());
    for (int r = 0; r < N; ++r)
      for (int h = 0; h < heads; ++h) {
        memcpy(wantQr.data() + (size_t(h) * N + r) * HD, y.data() + size_t(r) * rowdim + h * HD, HD * 4);
        memcpy(wantKr.data() + (size_t(h) * N + r) * HD, y.data() + size_t(r) * rowdim + heads * HD, HD * 4);
      }
    diff_ok("qkv: the raw Q copy", gotQr.data(), wantQr, 0.f);
    diff_ok("qkv: the raw K copy", gotKr.data(), wantKr, 0.f);
  }

  // 4+5) the conv pair: conv_bx writes the B*x and C planes; conv_dw runs the depthwise 3x3
  //      through the merge-block<->raster permutation and gates by C.
  {
    const int gh = 4, gw = 4, H = 8, N = gh * gw;
    std::vector<float> y(size_t(N) * 3 * H), C(size_t(N) * H), bx(size_t(N) * H), w9(size_t(H) * 9);
    std::vector<uint16_t> w9b(w9.size());
    for (auto& v : y) v = rnd();
    for (int i = 0; i < int(w9.size()); ++i) w9[i] = 0.05f * (i % 9 - 4), w9b[i] = bf16(w9[i]);
    for (int i = 0; i < int(w9.size()); ++i) w9[i] = bf16f(w9b[i]);  // the kernel reads bf16 weights
    for (int r = 0; r < N; ++r)
      for (int c = 0; c < H; ++c) {
        bx[size_t(r) * H + c] = y[size_t(r) * 3 * H + c] * y[size_t(r) * 3 * H + 2 * H + c];
        C[size_t(r) * H + c] = y[size_t(r) * 3 * H + H + c];
      }
    const std::vector<float> want_gated = ref_conv(bx, C, w9, gh, gw, H);
    ni::Buffer by = buffer(y.data(), y.size() * 4, "y"), bw = buffer(w9b.data(), w9b.size() * 2, "w"),
               bplanes = gpu.alloc(2 * size_t(N) * H * 4, "planes"), bxn = gpu.alloc(size_t(N) * H * 2, "xn");
    const Lfm3ConvBxParams cp{uint32_t(N), 0u, uint32_t(N), uint32_t(H), uint32_t(gh), uint32_t(gw), 2u};
    const Lfm3ConvDwParams dp{uint32_t(gh), uint32_t(gw), uint32_t(H), uint32_t(N)};
    auto e = gpu.begin();
    e->use(gpu.pipeline("lfm3vit_conv_bx"));
    e->buf(0, by);
    e->buf(1, bplanes);
    e->val(2, cp);
    e->dispatch(H, N, 1, 1);
    e->use(gpu.pipeline("lfm3vit_conv_dw"));
    e->buf(0, bplanes);
    e->buf(1, bw);
    e->buf(2, bxn);
    e->val(3, dp);
    e->dispatch(N, H, 1, 1);
    e->commit_and_wait();
    std::vector<float> gotBx(size_t(N) * H), gotC(gotBx.size()), gotGated(gotBx.size());
    for (size_t i = 0; i < gotBx.size(); ++i) {
      gotBx[i] = bplanes.as<float>()[i];
      gotC[i] = bplanes.as<float>()[size_t(N) * H + i];
      gotGated[i] = half_f(bxn.as<_Float16>()[i]);
    }
    diff_ok("conv_bx: B*x plane", gotBx.data(), bx, 0.f);
    diff_ok("conv_bx: C plane", gotC.data(), C, 0.f);
    diff_ok("conv_dw: C*conv(B*x) vs the HF permute walk", gotGated.data(), want_gated, 2e-3f);
  }

  // 6) lfm3vit_swiglu: silu(w1)*w3 into half rows
  {
    const int n = 256;
    std::vector<float> a(n), b(n);
    for (int i = 0; i < n; ++i) a[i] = rnd(), b[i] = rnd();
    const std::vector<float> want = ref_swiglu(a, b);
    ni::Buffer ba = buffer(a.data(), n * 4, "a"), bb = buffer(b.data(), n * 4, "b"), bh = gpu.alloc(size_t(n) * 2, "h");
    auto e = gpu.begin();
    e->use(gpu.pipeline("lfm3vit_swiglu"));
    e->buf(0, ba);
    e->buf(1, bb);
    e->buf(2, bh);
    e->val(3, uint32_t(n));
    e->dispatch((n + 255) / 256, 1, 1, 256);
    e->commit_and_wait();
    std::vector<float> got(n);
    for (int i = 0; i < n; ++i) got[i] = half_f(bh.as<_Float16>()[i]);
    diff_ok("swiglu: silu(w1)*w3", got.data(), want, 1e-3f);
  }

  printf("%s (%d ok, %d failed)\n", g_fail ? "FAIL" : "PASS", g_ok, g_fail);
  return g_fail ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc < 2 || !strcmp(argv[1], "--cpu") || !strcmp(argv[1], "cpu")) return cpu_checks();
  if (!strcmp(argv[1], "kernels") || !strcmp(argv[1], "gpu")) return kernel_checks();
  fprintf(stderr, "usage: lfm3_vit_kernel_check --cpu | kernels\n");
  return 2;
}

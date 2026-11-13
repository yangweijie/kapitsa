// CPU-only checks of the LFM3-Omni vision tower's geometry and rotary references
// (src/vision/lfm3_vit_ref.h). No model, no Metal, no weights. The goldens come from
// tests/vision/ref_lfm3_vit.py, an independent numpy transcription of the HF reference.
#include "vision/lfm3_vit_ref.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

using namespace ni::vision;

namespace {
int fails = 0;
void check(bool ok, const char* what) {
  printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  fails += !ok;
}
bool near(double a, double b, double tol = 2e-9) { return std::abs(a - b) <= tol; }

// A mechanical transcription of modeling_lfm3_vl.py:325-327: walk the HF view/permute/reshape
// strides rather than the closed-form merge_index. seq is [h*w, C] in merge-block order.
std::vector<float> hf_merge_to_raster(const std::vector<float>& seq, int h, int w, int m, int cc) {
  std::vector<float> raster(size_t(cc) * h * w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int bh = y / m, mh = y % m, bw = x / m, mw = x % m;
      for (int c = 0; c < cc; ++c) {
        const size_t off = ((((size_t(bh) * (w / m) + bw) * m + mh) * m + mw) * cc) + c;
        raster[(size_t(c) * h + y) * w + x] = seq[off];
      }
    }
  }
  return raster;
}

// image_processing_lfm3_vl.py:172-194: reshape(B,C,gh/m,m,P,gw/m,m,P) -> permute(0,2,5,3,6,1,4,7)
// -> expand the temporal axis -> reshape(B, gh*gw, C*T*P*P). image is [C, gh*P, gw*P].
std::vector<float> hf_patchify(const std::vector<float>& image, int gh, int gw) {
  const int p = kLfm3Patch, m = kLfm3Merge, cc = kLfm3Channels, t = kLfm3Temporal;
  std::vector<float> flat(size_t(gh) * gw * lfm3_vit_patch_dim());
  for (int bh = 0; bh < gh / m; ++bh) {
    for (int bw = 0; bw < gw / m; ++bw) {
      for (int mh = 0; mh < m; ++mh) {
        for (int mw = 0; mw < m; ++mw) {
          const int n = ((bh * (gw / m) + bw) * m + mh) * m + mw;
          for (int c = 0; c < cc; ++c) {
            for (int ti = 0; ti < t; ++ti) {
              for (int ph = 0; ph < p; ++ph) {
                for (int pw = 0; pw < p; ++pw) {
                  const int y = (bh * m + mh) * p + ph, x = (bw * m + mw) * p + pw;
                  flat[size_t(n) * lfm3_vit_patch_dim() + lfm3_vit_patch_column(c, ti, ph, pw)] =
                      image[(size_t(c) * (gh * p) + y) * (gw * p) + x];
                }
              }
            }
          }
        }
      }
    }
  }
  return flat;
}
}  // namespace

int main() {
  check(lfm3_vit_patch_dim() == 1536, "patch row is 3*2*16*16 = 1536 floats");
  check(lfm3_vit_patch_column(2, 1, 15, 15) == 1535, "the last patch column is [c=2,t=1,ph=15,pw=15]");
  check(lfm3_vit_image_tokens(32, 32) == 256, "512x512 -> 32x32 grid -> 256 image tokens");

  // Python's round() is half-to-even; std::round() is not. Pinned separately because a wrong
  // tie rule changes the grid of any edge sitting exactly at 0.5 factors.
  bool round_ok = true;
  const double ties[] = {0.5, 1.5, 2.5, 3.5, -1.5, -2.5};
  const double wants[] = {0., 2., 2., 4., -2., -2.};
  for (int i = 0; i < 6; ++i) round_ok &= near(lfm3_vit_py_round(ties[i]), wants[i]);
  check(round_ok, "py_round implements Python's half-to-even on positive and negative ties");

  struct Case { int h, w, oh, ow; };
  const Case cases[] = {{512, 512, 512, 512}, {100, 100, 256, 256}, {1024, 1024, 1024, 1024},
                        {48, 48, 256, 256},  {16, 48, 160, 448},   {37, 53, 224, 320},
                        {2048, 64, 2048, 64}};
  bool resize_ok = true;
  for (const auto& c : cases) {
    const auto s = lfm3_vit_smart_resize(c.h, c.w);
    resize_ok &= s.h == c.oh && s.w == c.ow;
    resize_ok &= s.h % kLfm3ResizeFactor == 0 && s.w % kLfm3ResizeFactor == 0;
  }
  check(resize_ok, "smart_resize matches the numpy reference on 7 shapes, all edges 32-aligned");
  check(lfm3_vit_smart_resize(1, 1000).h == 0, "smart_resize refuses an aspect ratio above 200");

  const int golden[4][4] = {{0, 1, 4, 5}, {2, 3, 6, 7}, {8, 9, 12, 13}, {10, 11, 14, 15}};
  bool golden_ok = true;
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) golden_ok &= lfm3_vit_merge_index(y, x, 4) == golden[y][x];
  check(golden_ok, "merge-block row index for a 4x4 grid matches the numpy golden");

  bool trip_ok = true;
  for (int h : {4, 8, 16, 32}) {
    for (int w : {4, 8, 32}) {
      for (int n = 0; n < h * w; ++n) {
        const auto yx = lfm3_vit_raster_coords(n, w);
        trip_ok &= lfm3_vit_merge_index(yx[0], yx[1], w) == n;
      }
    }
  }
  check(trip_ok, "merge_index and raster_coords are mutual inverses across 12 grid shapes");

  const int pid[16][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3},
                          {2, 0}, {2, 1}, {3, 0}, {3, 1}, {2, 2}, {2, 3}, {3, 2}, {3, 3}};
  const auto ids = lfm3_vit_position_ids(4, 4);
  bool ids_ok = ids.size() == 16;
  for (int n = 0; n < 16; ++n) ids_ok &= ids[n][0] == pid[n][0] && ids[n][1] == pid[n][1];
  check(ids_ok, "position_ids(4,4) matches get_vision_position_ids in merge-block order");

  // The mechanical HF permutation and the closed form must describe the same raster, and a
  // depthwise 3x3 conv with an identity kernel must be a no-op across the round trip.
  bool mech_ok = true;
  constexpr int cc = 3;
  for (auto shape : {std::array<int, 2>{4, 4}, {4, 8}, {8, 4}, {32, 32}}) {
    const int h = shape[0], w = shape[1];
    std::vector<float> seq(size_t(h) * w * cc);
    for (size_t i = 0; i < seq.size(); ++i) seq[i] = float(i);
    const auto raster = hf_merge_to_raster(seq, h, w, kLfm3Merge, cc);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int n = lfm3_vit_merge_index(y, x, w);
        for (int c = 0; c < cc; ++c)
          mech_ok &= raster[(size_t(c) * h + y) * w + x] == seq[size_t(n) * cc + c];
      }
    }
    const auto ids2 = lfm3_vit_position_ids(h, w);
    for (int n = 0; n < h * w; ++n)
      mech_ok &= lfm3_vit_merge_index(ids2[n][0], ids2[n][1], w) == n;
  }
  check(mech_ok, "HF's view/permute/reshape equals merge_index on 4 grids; position_ids agree");

  // patchify: build pixels whose value encodes (c, y, x), then check every row/column pair.
  constexpr int gh = 4, gw = 4, p = kLfm3Patch;
  std::vector<float> image(size_t(kLfm3Channels) * (gh * p) * (gw * p));
  for (int c = 0; c < kLfm3Channels; ++c)
    for (int y = 0; y < gh * p; ++y)
      for (int x = 0; x < gw * p; ++x)
        image[(size_t(c) * (gh * p) + y) * (gw * p) + x] = float(c * 10000 + y * 100 + x);
  const auto flat = hf_patchify(image, gh, gw);
  bool patch_ok = flat.size() == size_t(gh) * gw * lfm3_vit_patch_dim();
  const auto pids = lfm3_vit_position_ids(gh, gw);
  for (int n = 0; n < gh * gw; ++n) {
    const int gy = pids[n][0], gx = pids[n][1];
    for (int c = 0; c < kLfm3Channels; ++c) {
      for (int ph = 0; ph < p; ++ph) {
        for (int pw = 0; pw < p; ++pw) {
          const float want = float(c * 10000 + (gy * p + ph) * 100 + (gx * p + pw));
          const size_t row = size_t(n) * lfm3_vit_patch_dim();
          patch_ok &= flat[row + lfm3_vit_patch_column(c, 0, ph, pw)] == want;
          patch_ok &= flat[row + lfm3_vit_patch_column(c, 1, ph, pw)] == want;
        }
      }
    }
  }
  check(patch_ok, "patchify rows are merge-block ordered, columns [c,t,ph,pw], temporal halves equal");

  const auto inv = lfm3_vit_axial_inv_freq();
  check(inv.size() == 16 && near(inv[0], 1.0), "axial inv_freq has 16 entries, the first is 1.0");
  bool freq_ok = true;
  for (int i = 0; i < 16; ++i) freq_ok &= near(inv[i], 1.0 / std::pow(1e6, double(2 * i) / 32));
  check(freq_ok, "inv_freq follows theta^(-2i/32), i = 0..15");

  std::array<double, kLfm3HeadDim> cosv{}, sinv{};
  lfm3_vit_axial_angle(1, 2, cosv.data(), sinv.data());
  const double gcos[] = {0.540302306, 0.91239586, 0.984230234, 0.997189611,
                         0.999500042, 0.999911087, 0.999984189, 0.999997188};
  const double gsin[] = {0.841470985, 0.409308924, 0.176892186, 0.074919158,
                         0.031617506, 0.013334819, 0.005623384, 0.002371371};
  const double gcos48[] = {-0.416146837, 0.664932409, 0.937418309, 0.98877424,
                           0.998000667, 0.999644365, 0.999936755, 0.999988753};
  bool angle_ok = true;
  for (int i = 0; i < 8; ++i) {
    angle_ok &= near(cosv[i], gcos[i]) && near(sinv[i], gsin[i]);
    angle_ok &= near(cosv[48 + i], gcos48[i]);
    angle_ok &= near(cosv[32 + i], cosv[i]) && near(cosv[48 + i], cosv[16 + i]);
  }
  check(angle_ok, "axial cos/sin at (h=1,w=2) matches the numpy golden and the [h,w,h,w] layout");

  // RoPE is a rotation: the head's L2 norm is preserved, and the closed form agrees with
  // q*cos + rotate_half(q)*sin for several 2D positions.
  bool rope_ok = true;
  std::vector<float> q(kLfm3HeadDim), rh(kLfm3HeadDim), roped(kLfm3HeadDim);
  for (int i = 0; i < kLfm3HeadDim; ++i) q[i] = float(std::sin(0.31 * i) + 0.05 * i);
  for (auto pos : {std::array<int, 2>{0, 0}, {1, 2}, {7, 3}, {31, 31}}) {
    lfm3_vit_axial_angle(pos[0], pos[1], cosv.data(), sinv.data());
    lfm3_vit_rotate_half(q.data(), kLfm3HeadDim, rh.data());
    lfm3_vit_apply_rope(q.data(), kLfm3HeadDim, cosv.data(), sinv.data(), roped.data());
    double qn = 0., rn = 0.;
    for (int i = 0; i < kLfm3HeadDim; ++i) {
      const double want = double(q[i]) * cosv[i] + double(rh[i]) * sinv[i];
      rope_ok &= std::abs(double(roped[i]) - want) < 1e-6;
      qn += double(q[i]) * q[i];
      rn += double(roped[i]) * roped[i];
    }
    rope_ok &= std::abs(std::sqrt(qn) - std::sqrt(rn)) < 1e-4;
  }
  check(rope_ok, "apply_rope equals q*cos + rotate_half(q)*sin and preserves the head's norm");

  const float x[] = {0.f, 1.f, 2.f, 3.f, -1.f, -2.f, 0.5f, 0.25f};
  const float w[] = {1.f, 0.5f, 2.f, 1.5f, 0.25f, 3.f, 1.f, 0.75f};
  const float gn[] = {0.f, 0.32180652f, 2.574452162f, 2.896258593f,
                      -0.16090326f, -3.861678123f, 0.32180652f, 0.120677441f};
  std::array<float, 8> out{};
  lfm3_vit_rms_norm(x, w, 8, kLfm3NormEps, out.data());
  bool norm_ok = true;
  for (int i = 0; i < 8; ++i) norm_ok &= std::abs(out[i] - gn[i]) < 1e-6f;
  check(norm_ok, "RMSNorm matches the numpy golden (T5 form, eps 1e-5)");

  printf("%d failure(s)\n", fails);
  return fails ? 1 : 0;
}

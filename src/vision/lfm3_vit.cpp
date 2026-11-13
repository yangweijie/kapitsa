// The LFM3-Omni image encoder (see lfm3_vit.h). Its geometry, the merge-block order and the axial
// rotary math are pinned in lfm3_vit_ref.h; the kernels diff against those like lfm3.metal diffs
// against lfm3_rope_ref. The tower's weights come from the mmproj import's vision.safetensors, whose
// tensors the model folder's index lists (the import co-locates them).
#include "vision/lfm3_vit.h"

#include "vision/lfm3_vit_ref.h"
#include "vision/qwen_vit.h"  // qwen_smart_resize (the same half-to-even rounding), pil_resize_bicubic

#include "model/kparams.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ni::vision {

namespace {

using json = nlohmann::json;

// qvit_patchify's params (it writes the [c, t, ph, pw] columns per merge-block row; see qwen_vit.cpp)
struct QvitPatchParams_ {
  uint32_t w, gw, P, T, mg;
};
struct VitActParamsX {  // vit_bias_f32's params (the debug dump reuses it, act 0: no bias)
  uint32_t rows, N, ld, act, out_ld;
};
// vit_ln's params (the projector's LayerNorm); mirrors vit.metal's VitLnParams
struct VitLnParamsX {
  uint32_t rows, D, x_ld, y_ld;
  float eps;
  uint32_t has_pb;
};

// lfm3vit_qkv_split's params: split + per-head QK RMSNorm + the axial rope (GQA served by writing
// every kv head's rows to its NKH query-head copies, so vit_attention runs unchanged)
struct Lfm3QkvParams {
  uint32_t rows, r0, N, heads, kv_heads, HD;
};
struct Lfm3ConvBxParams {  // lfm3vit_conv_bx: the in_proj row -> planes' B*x and C
  uint32_t rows, r0, N, H, gh, gw, merge;
};
struct Lfm3ConvDwParams {  // lfm3vit_conv_dw: the depthwise 3x3 gated by C into xn's rows
  uint32_t gh, gw, H, N;
};

// The processor's value LUT (rescale + normalize), as the reference processor computes it (float32 steps)
void build_lut(const Lfm3ViTConfig& c, float* lut) {
  const float rs = float(c.rescale);
  for (int ch = 0; ch < 3; ++ch)
    for (int v = 0; v < 256; ++v) {
      float x = float(v);
      if (c.do_rescale) x = x * rs;
      if (c.do_normalize) {
        x = x - c.mean[ch];
        x = x / c.stdv[ch];
      }
      lut[ch * 256 + v] = x;
    }
}

// patch r (merge-block order) -> its (row, column) in the patch grid
inline void patch_rc(int r, int gw, int merge, int& prow, int& pcol) {
  const int mm = merge * merge, b = r / mm, bw = gw / merge;
  prow = (b / bw) * merge + (r % mm) / merge;
  pcol = (b % bw) * merge + r % merge;
}

}  // namespace

bool Lfm3ViT::supports(const std::string& dir, Lfm3ViTConfig* c) {
  // The mmproj import's file, co-located with the text model's weights (a text-only folder has none)
  const json root = vit_read_json(dir + "/vision_config.json");
  if (root.value("model_type", std::string()) != "lfm3_vl_vision" || !root.contains("vision_config") ||
      !root["vision_config"].is_object())
    return false;
  const json& v = root["vision_config"];
  Lfm3ViTConfig q;
  q.depth = v.value("num_hidden_layers", q.depth);
  q.hidden = v.value("hidden_size", q.hidden);
  q.inter = v.value("intermediate_size", q.inter);
  q.heads = v.value("num_attention_heads", q.heads);
  q.kv_heads = v.value("num_key_value_heads", q.kv_heads);
  q.patch = v.value("patch_size", q.patch);
  q.temporal = v.value("temporal_patch_size", q.temporal);
  q.merge = v.value("spatial_merge_size", q.merge);
  q.in_ch = v.value("in_channels", q.in_ch);
  q.rms_eps = v.value("norm_eps", q.rms_eps);
  const json rope = v.value("rope_parameters", json::object());
  if (rope.value("rope_type", std::string("axial")) != "axial") return false;
  const double theta = rope.value("rope_theta", 1e6);
  if (std::fabs(theta - 1e6) > 1.0) return false;  // the kernels' inv_freq is built for theta 1e6
  q.image_token = root.value("image_token_id", q.image_token);
  q.proj_hidden = root.value("projector_hidden_size", q.proj_hidden);
  if (root.value("projector_hidden_act", std::string("gelu")) != "gelu") return false;
  q.proj_ln_eps = root.value("projector_layer_norm_eps", q.proj_ln_eps);
  q.out_hidden = root.value("text_hidden_size", q.out_hidden);
  if (v.value("conv_kernel_size", 3) != 3) return false;
  // the processor (the import carries the reference processor's fields)
  q.min_pixels = root.value("image_min_pixels", q.min_pixels);
  q.max_pixels = root.value("image_max_pixels", q.max_pixels);
  if (root.contains("image_mean") && root["image_mean"].is_array() && root["image_mean"].size() == 3)
    for (int i = 0; i < 3; ++i) q.mean[i] = root["image_mean"][i].get<float>();
  if (root.contains("image_std") && root["image_std"].is_array() && root["image_std"].size() == 3)
    for (int i = 0; i < 3; ++i) q.stdv[i] = root["image_std"][i].get<float>();
  // a shape the tower's kernels are built for: head_dim 64 (GQA 16/8), merge 2, the reference's 14
  // blocks of 1024 (8 depthwise-conv, 6 attention: even indexes from 2)
  const int hd = q.heads > 0 && q.hidden % q.heads == 0 ? q.hidden / q.heads : 0;
  if (hd != kLfm3HeadDim || q.kv_heads <= 0 || q.heads % q.kv_heads != 0 || q.merge != 2 || q.in_ch != 3 ||
      q.temporal != 2 || q.hidden % 64 || q.depth != 14 || q.hidden != 1024 || q.inter != 2560 ||
      q.proj_hidden != 4096 || q.out_hidden != 2048) {
    logi("vision: %s has an LFM3 vision tower of a shape Kapitsa does not run (hidden %d, %d/%d heads, depth %d): "
         "images are off",
         dir.c_str(), q.hidden, q.heads, q.kv_heads, q.depth);
    return false;
  }
  q.layer_types.assign(size_t(q.depth), 'C');
  for (int i = 2; i < q.depth; i += 2) q.layer_types[size_t(i)] = 'S';
  if (c) *c = q;
  return true;
}

Lfm3ViT::Lfm3ViT(const std::string& dir, const Lfm3ViTConfig& c) : dir_(dir), c_(c) {
  inter_pad_ = uint32_t((c_.inter + 63) / 64 * 64);
}

std::string Lfm3ViT::describe() const {
  return name() + " (" + std::to_string(c_.depth) + " blocks, width " + std::to_string(c_.hidden) + ", " +
         std::to_string(c_.patch) + "-pixel patches, axial rope; images of " + std::to_string(c_.min_pixels) +
         " to " + std::to_string(c_.max_pixels) + " pixels)";
}

Lfm3ViT::~Lfm3ViT() { unload(); }

// ---- the processor (CPU) ----

Prepared Lfm3ViT::prepare(const Image& im) const {
  const int factor = c_.patch * c_.merge;
  const auto [rh, rw] = qwen_smart_resize(im.height, im.width, factor, c_.min_pixels, c_.max_pixels);
  Prepared p;
  p.width = rw;
  p.height = rh;
  p.pixels = pil_resize_bicubic(im.rgb, im.width, im.height, rw, rh);
  p.grid_t = 1;
  p.grid_h = rh / factor;
  p.grid_w = rw / factor;
  p.tokens = p.grid_h * p.grid_w;  // the 2x2-merged grid (the tower's axial rope works on the unmerged one)
  uint8_t key[32 + 8 + 8];
  memcpy(key, im.digest.data(), 32);
  const int32_t dims[2] = {rh, rw};
  memcpy(key + 32, dims, 8);
  memcpy(key + 40, "lfm3-vit", 8);
  p.digest = sha256(key, sizeof key);
  return p;
}

std::vector<float> Lfm3ViT::pixel_values(const Prepared& p) const {
  float lut[3 * 256];
  build_lut(c_, lut);
  const int P = c_.patch, T = c_.temporal, gh = p.height / P, gw = p.width / P, N = gh * gw;
  const size_t K = size_t(3) * T * P * P;
  std::vector<float> out(size_t(N) * K);
  for (int r = 0; r < N; ++r) {
    int prow, pcol;
    patch_rc(r, gw, c_.merge, prow, pcol);
    float* o = out.data() + size_t(r) * K;
    for (int ch = 0; ch < 3; ++ch)
      for (int t = 0; t < T; ++t)
        for (int y = 0; y < P; ++y)
          for (int x = 0; x < P; ++x)
            o[((size_t(ch) * T + t) * P + y) * P + x] =
                lut[ch * 256 + p.pixels[(size_t(prow * P + y) * p.width + pcol * P + x) * 3 + ch]];
  }
  return out;
}

// ---- weights ----

size_t Lfm3ViT::weight_bytes() const {
  const size_t H = size_t(c_.hidden), I = inter_pad_, K0 = size_t(c_.temporal) * c_.patch * c_.patch * c_.in_ch;
  size_t b = size_t(H) * K0 * 2 + H * 2;  // patch (no bias) + embedding_norm
  for (const Block& B : blocks_) {
    b += H * 2 * 2;  // operator_norm + ffn_norm
    if (B.kind == 'S') b += size_t(2) * H * H * 2 + size_t(H) * H * 2 + 2 * size_t(c_.patch) * 2;
    if (B.kind == 'C') b += size_t(3) * H * H * 2 + size_t(H) * H * 2 + size_t(H) * 9 * 2;
    b += size_t(2) * I * H * 2 + size_t(H) * I * 2;  // w1, w3 [I, H], w2 [H, I]
  }
  b += 2 * H * 2 + size_t(c_.proj_hidden) * (c_.proj_hidden + c_.out_hidden + 2) * 2;
  return b;
}

void Lfm3ViT::load() {
  if (loaded_) return;
  // the import's names are already vision_tower.* / multi_modal_projector.*; its file is not in the
  // model index, so it is named explicitly (the store keeps the prefix filter, "" matches every name)
  w_.open(dir_, "", {"vision.safetensors"});
  const uint32_t H = uint32_t(c_.hidden), P = uint32_t(c_.patch), T = uint32_t(c_.temporal);
  const uint32_t K0 = T * P * P * uint32_t(c_.in_ch);
  // the patchifier: the import's merged [hidden, c, t, ph, pw] linear, permuted at load to
  // (t, ph, pw, c) — the column order qvit_patchify writes — then a plain linear (no bias)
  {
    patch_.N = H;
    patch_.K = K0;
    patch_.packed_half = false;
    patch_.w = w_.alloc(size_t(H) * K0 * 2, "vit_w");
    w_.read_bf16("vision_tower.patchifier.weight", patch_.w.as<uint16_t>(), H, K0, K0);
    std::vector<uint16_t> row(K0);
    for (uint32_t o = 0; o < H; ++o) {
      uint16_t* w = patch_.w.as<uint16_t>() + size_t(o) * K0;
      for (uint32_t t = 0; t < T; ++t)
        for (uint32_t y = 0; y < P; ++y)
          for (uint32_t x = 0; x < P; ++x)
            for (uint32_t c = 0; c < 3; ++c) row[((t * P + y) * P + x) * 3 + c] = w[((c * T + t) * P + y) * P + x];
      memcpy(w, row.data(), K0 * 2);
    }
    patch_.b = w_.alloc(size_t(H) * 2, "vit_b");  // the patchifier has no bias (zeros)
  }
  emb_norm_ = w_.vec("vision_tower.embedding_norm.weight");
  blocks_.resize(size_t(c_.depth));
  for (int i = 0; i < c_.depth; ++i) {
    Block& B = blocks_[size_t(i)];
    const std::string b = "vision_tower.layers." + std::to_string(i) + ".";
    B.kind = c_.layer_types[size_t(i)];
    B.onw = w_.vec(b + "operator_norm.weight");
    B.fnw = w_.vec(b + "ffn_norm.weight");
    if (B.kind == 'S') {
      // q [1024, 1024], k [512, 1024], v [512, 1024]: one concatenated GEMM [2048, 1024] keeps the
      // split kernel's row layout, K's and V's halves zero-padded to H (the split writes GQA heads
      // from the padded rows; the rope kernel then normalizes and rotates the live ones)
      B.qkv.N = 2 * H;
      B.qkv.K = H;
      B.qkv.packed_half = false;
      B.qkv.w = w_.alloc(size_t(2 * H) * H * 2, "vit_w");
      w_.read_bf16(b + "self_attn.q_proj.weight", B.qkv.w.as<uint16_t>(), H, H, H);
      w_.read_bf16(b + "self_attn.k_proj.weight", B.qkv.w.as<uint16_t>() + size_t(H) * H, H / 2, H, H);
      w_.read_bf16(b + "self_attn.v_proj.weight", B.qkv.w.as<uint16_t>() + size_t(3 * H / 2) * H, H / 2, H, H);
      B.qkv.b = w_.alloc(size_t(2 * H) * 2, "vit_b");  // no bias (zeros)
      B.o = w_.lin(b + "self_attn.out_proj", 0, 0, false);
      B.qn = w_.vec(b + "self_attn.q_layernorm.weight");
      B.kn = w_.vec(b + "self_attn.k_layernorm.weight");
    } else {
      B.cinp = w_.lin(b + "conv.in_proj", 0, 0, false);    // [3H, H]
      B.coutp = w_.lin(b + "conv.out_proj", 0, 0, false);  // [H, H]
      const Tensor* tc = w_.find(b + "conv.spatial_conv.weight");
      if (!tc || tc->numel() != int64_t(H) * 9) fatal("vision: %sconv.spatial_conv.weight is not [H,1,3,3]", b.c_str());
      B.csp = w_.alloc(size_t(H) * 9 * 2, "vit_w");
      w_.read_bf16(b + "conv.spatial_conv.weight", B.csp.as<uint16_t>(), H, 9, 9);
    }
    B.w1 = w_.lin(b + "feed_forward.w1", inter_pad_, 0, false);  // [I, H]
    B.w3 = w_.lin(b + "feed_forward.w3", inter_pad_, 0, false);
    B.w2 = w_.lin(b + "feed_forward.w2", 0, inter_pad_, false);  // [H, I]
  }
  pnw_ = w_.vec("multi_modal_projector.layer_norm.weight");
  pnb_ = w_.vec("multi_modal_projector.layer_norm.bias");
  m0_ = w_.lin("multi_modal_projector.linear_1");
  m2_ = w_.lin("multi_modal_projector.linear_2");
  w_.close();
  w_.make_resident();
  loaded_ = true;
}

void Lfm3ViT::unload() {
  w_.release();
  blocks_.clear();
  loaded_ = false;
}

size_t Lfm3ViT::work_bytes(const Prepared& p) const {
  const size_t H = size_t(c_.hidden), I = inter_pad_, N = size_t(p.height / c_.patch) * size_t(p.width / c_.patch);
  const size_t rc = std::min<size_t>(N, kVitRowChunk);
  const size_t ym = std::max<size_t>(3 * H, c_.proj_hidden);
  return pixel_buffer_bytes(p.pixels.size()) + N * H * 4 /* x */ + N * H * 2 /* xn */ + 3 * N * H * 2 /* qkv, qg, v (broadcast) */ +
         3 * N * H * 4 /* the conv's raster planes */ + rc * ym * 4 /* y */ + rc * I * 4 /* y3 */ + rc * I * 2 /* hh */ +
         64 * 1024 /* tables */;
}

void Lfm3ViT::encode(const Prepared& p, const Buffer& out) {
  if (!loaded_) load();
  if (p.pixels.size() != size_t(p.width) * p.height * 3) fatal("vision: encode without pixels");
  Gpu& g = Gpu::get();
  const int P = c_.patch, T = c_.temporal, mg = c_.merge;
  const int gh = p.height / P, gw = p.width / P, N = gh * gw, M = N / (mg * mg);
  const int H = c_.hidden, NH = c_.heads, NKV = c_.kv_heads, HD = H / NH, I = int(inter_pad_), H4 = H * mg * mg;
  const int O = c_.out_hidden, K0 = T * P * P * 3, RC = std::min(N, kVitRowChunk), YM = std::max(3 * H, c_.proj_hidden);
  std::vector<Buffer> scratch;
  auto alloc = [&](size_t bytes, const char* label) {
    scratch.push_back(g.alloc(bytes + 16, label));
    return scratch.back();
  };
  Buffer pix = g.wrap(const_cast<uint8_t*>(p.pixels.data()), pixel_buffer_bytes(p.pixels.size()), "vit_pixels", true);
  scratch.push_back(pix);
  Buffer lut = alloc(3 * 256 * 4, "vit_lut");
  build_lut(c_, lut.as<float>());
  // the axial rope's 16 frequencies per axis at theta 1e6 (lfm3_vit_ref.h), and the merge-block
  // (h, w) position of every patch row: [N, 2] int32, the rope kernel reads its row's coords there
  Buffer invf = alloc(size_t(HD / 4) * 4, "vit_invfreq");
  {
    const auto f = lfm3_vit_axial_inv_freq();
    for (size_t j = 0; j < f.size(); ++j) invf.as<float>()[j] = float(f[j]);
  }
  Buffer pos = alloc(size_t(N) * 8, "vit_pos");
  {
    const auto ids = lfm3_vit_position_ids(gh, gw, mg);
    for (size_t i = 0; i < ids.size(); ++i) {
      pos.as<int32_t>()[2 * i] = ids[i][0];
      pos.as<int32_t>()[2 * i + 1] = ids[i][1];
    }
  }
  Buffer x = alloc(size_t(N) * H * 4, "vit_x");       // fp32 [N, H] residual stream
  Buffer xn = alloc(size_t(N) * H * 2, "vit_xn");     // half normed rows
  Buffer qkv = alloc(size_t(2) * N * H * 2, "vit_qkv");  // half [2H] per row (q | k | v padded)
  Buffer vb = alloc(size_t(N) * H * 2, "vit_vb");        // half V, all 16 query heads (the kv broadcast)
  Buffer qg = alloc(size_t(N) * H * 2, "vit_qg");     // half Q after q-norm + rope
  Buffer kg = alloc(size_t(N) * H * 2, "vit_kg");  // half K after k-norm + rope, all 16 query heads
                                                   // (the kv broadcast: 8 kv heads written twice)
  Buffer planes = alloc(2 * size_t(N) * H * 4, "vit_planes");  // the conv's planes: [B*x; C] (fp32)
  Buffer y = alloc(size_t(RC) * YM * 4, "vit_y");
  Buffer y3 = alloc(size_t(RC) * I * 4, "vit_y3");   // w3's GEMM rows (w1's y is [RC, I] too: two buffers)
  Buffer hh = alloc(size_t(RC) * I * 2, "vit_h");
  const Buffer Qb = qkv.view(0, size_t(N) * H * 2), Kb = qkv.view(size_t(N) * H * 2, size_t(N) * H * 2),
               Vb = vb;  // V: all 16 query heads (each kv head's rows twice), half [heads][N][HD]
  VitCmds cs;
  // rms norm (eps 1e-5) of `rows` fp32 rows of x into half rows of xn, by `w`
  auto rms = [&](Encoder& e, const Buffer& w, int rows, int r0 = 0, int out_rows = 0) {
    const VitLnParamsX rp{uint32_t(rows), uint32_t(H), uint32_t(H), uint32_t(H), c_.rms_eps};
    e.use(vit_pipe("vit_rms"));
    e.buf(0, x.view(size_t(r0) * H * 4, size_t((out_rows ? out_rows : rows)) * H * 4));
    e.buf(1, w);
    e.buf(2, xn.view(size_t(r0) * H * 2, size_t((out_rows ? out_rows : rows)) * H * 2));
    e.val(3, rp);
    e.dispatch((uint32_t(out_rows ? out_rows : rows) + 7) / 8, 1, 1, 256);
  };
  // patches -> patch embedding (no bias, no position embedding: the axial rope comes per block)
  {
    Encoder& e = cs.at(2.0 * N * K0 * H / 1e9);
    const QvitPatchParams_ pp{uint32_t(p.width), uint32_t(gw), uint32_t(P), uint32_t(T), uint32_t(mg)};
    e.use(vit_pipe("qvit_patchify"));
    e.buf(0, pix);
    e.buf(1, lut);
    e.buf(2, qkv);
    e.val(3, pp);
    e.dispatch((uint32_t(P * P * 3) + 255) / 256, uint32_t(N), 1, 256);
    vit_gemm(e, patch_, qkv, uint32_t(K0), x, uint32_t(H), N, false);
    rms(e, emb_norm_, N);  // embedding_norm straight after the patchifier (no residual yet)
    // the normed rows start the residual stream
    Encoder& e2 = cs.at(0.0);
    e2.use(vit_pipe("vit_f32_from_h"));
    e2.buf(0, xn);
    e2.buf(1, x);
    e2.val(2, uint32_t(N) * H);
    e2.dispatch((uint32_t(N) * H + 255) / 256, 1, 1, 256);
  }
  const int stop = dbg_stop_ >= 0 ? std::min(dbg_stop_, c_.depth) : c_.depth;
  for (int L = 0; L < stop; ++L) {
    const Block& B = blocks_[size_t(L)];
    if (B.kind == 'S') {
      rms(cs.at(0.01 * N * H / 1e6), B.onw, N);
      for (int r0 = 0; r0 < N; r0 += RC) {
        const int rc = std::min(RC, N - r0);
        Encoder& e = cs.at(2.0 * rc * H * 2 * H / 1e9);
        vit_gemm(e, B.qkv, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H), y, uint32_t(2 * H), rc,
                 false);
        // split (into padded [heads] rows), then q/k RMSNorm + the axial rope of the live heads
        const Lfm3QkvParams qp{uint32_t(rc), uint32_t(r0), uint32_t(N), uint32_t(NH), uint32_t(NKV), uint32_t(HD)};
        e.use(vit_pipe("lfm3vit_qkv_split"));
        e.buf(0, y);
        e.buf(1, B.qkv.b);
        e.buf(2, Qb);
        e.buf(3, Kb);
        e.buf(4, Vb);
        e.buf(5, B.qn);
        e.buf(6, B.kn);
        e.buf(7, pos);
        e.buf(8, invf);
        e.buf(9, qg);
        e.buf(10, kg);
        e.val(11, qp);
        e.dispatch(uint32_t(rc), 1, 1, 1);  // one threadgroup per row; the kernel is row-serial
      }
      vit_attention(cs, qg, kg, Vb, xn, N, H, NH, HD);
      for (int r0 = 0; r0 < N; r0 += RC) {
        const int rc = std::min(RC, N - r0);
        vit_gemm(cs.at(2.0 * rc * H * H / 1e9), B.o, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H),
                 x.view(size_t(r0) * H * 4, size_t(rc) * H * 4), uint32_t(H), rc, true);
      }
    } else {
      rms(cs.at(0.01 * N * H / 1e6), B.onw, N);
      // The HF operator: out_proj(C * conv(B * x)). Stage 1 writes the B*x and C planes; stage 2 runs
      // the depthwise 3x3 over planes[0]'s raster gated by planes[1]'s C into xn's rows; out_proj adds
      // it to the residual stream. (xn is free here: the rms output was consumed by in_proj's GEMM.)
      for (int r0 = 0; r0 < N; r0 += RC) {
        const int rc = std::min(RC, N - r0);
        Encoder& e = cs.at(2.0 * rc * H * 3 * H / 1e9);
        vit_gemm(e, B.cinp, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H), y, uint32_t(3 * H), rc,
                 false);
        const Lfm3ConvBxParams cp{uint32_t(rc), uint32_t(r0), uint32_t(N), uint32_t(H), uint32_t(gh), uint32_t(gw),
                                 uint32_t(mg)};
        e.use(vit_pipe("lfm3vit_conv_bx"));
        e.buf(0, y);
        e.buf(1, planes);
        e.val(2, cp);
        e.dispatch(uint32_t(H), uint32_t(rc), 1, 1);
      }
      {
        Encoder& e = cs.at(9.0 * N * H / 1e9);
        const Lfm3ConvDwParams dp{uint32_t(gh), uint32_t(gw), uint32_t(H), uint32_t(N)};
        e.use(vit_pipe("lfm3vit_conv_dw"));
        e.buf(0, planes);
        e.buf(1, B.csp);
        e.buf(2, xn);
        e.val(3, dp);
        e.dispatch(uint32_t(N), uint32_t(H), 1, 1);
      }
      for (int r0 = 0; r0 < N; r0 += RC) {
        const int rc = std::min(RC, N - r0);
        Encoder& e = cs.at(2.0 * rc * H * H / 1e9);
        vit_gemm(e, B.coutp, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H),
                 x.view(size_t(r0) * H * 4, size_t(rc) * H * 4), uint32_t(H), rc, true);
      }
    }
    // the SwiGLU MLP of every block: ffn_norm (rms) -> w1, w3 -> silu(w1)*w3 -> w2 -> +x
    rms(cs.at(0.01 * N * H / 1e6), B.fnw, N);
    for (int r0 = 0; r0 < N; r0 += RC) {
      const int rc = std::min(RC, N - r0);
      Encoder& e = cs.at(4.0 * rc * H * I / 1e9);
      vit_gemm(e, B.w1, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H), y, uint32_t(I), rc, false);
      vit_gemm(e, B.w3, xn.view(size_t(r0) * H * 2, size_t(rc) * H * 2), uint32_t(H), y3, uint32_t(I), rc, false);
      e.use(vit_pipe("lfm3vit_swiglu"));
      e.buf(0, y);
      e.buf(1, y3);
      e.buf(2, hh);
      e.val(3, uint32_t(rc) * I);
      e.dispatch((uint32_t(rc) * I + 255) / 256, 1, 1, 256);
      vit_gemm(e, B.w2, hh, uint32_t(I), x.view(size_t(r0) * H * 4, size_t(rc) * H * 4), uint32_t(H), rc, true);
    }
  }
  if (dbg_stop_ >= 0) {  // tests: the hidden state after `stop` blocks, fp32
    Encoder& e = cs.at(0.01);
    e.use(vit_pipe("vit_bias_f32"));
    e.buf(0, x);
    e.buf(1, emb_norm_);  // unused (no bias added here)
    e.buf(2, out);
    e.val(3, VitActParamsX{uint32_t(N), uint32_t(H), uint32_t(H), 0, uint32_t(H)});
    e.dispatch((uint32_t(H) + 255) / 256, uint32_t(N), 1, 256);
    cs.finish();
    for (Buffer& b : scratch) g.release(b);
    return;
  }
  // the projector: LayerNorm(1024, 1e-6) per patch, the 2x2-merged row of 4*hidden, linear_1, GELU, linear_2
  {
    Encoder& e = cs.at(0.01 * N * H / 1e6);
    e.use(vit_pipe("vit_ln"));
    e.buf(0, x);
    e.buf(1, pnb_);
    e.buf(2, pnw_);
    e.buf(3, pnb_);
    e.buf(4, xn);
    e.val(5, VitLnParamsX{uint32_t(N), uint32_t(H), uint32_t(H), uint32_t(H), c_.proj_ln_eps, 0});
    e.dispatch((uint32_t(N) + 7) / 8, 1, 1, 256);
  }
  const int MC = std::max(1, RC / (mg * mg));
  for (int m0 = 0; m0 < M; m0 += MC) {
    const int mc = std::min(MC, M - m0);
    Encoder& e = cs.at(2.0 * mc * H4 * (H4 + O) / 1e9);
    vit_gemm(e, m0_, xn.view(size_t(m0) * H4 * 2, size_t(mc) * H4 * 2), uint32_t(H4), y, uint32_t(H4), mc, false);
    vit_act(e, "vit_bias_act_h", y, uint32_t(H4), m0_.b, hh, uint32_t(H4), mc, H4, 2);
    vit_gemm(e, m2_, hh, uint32_t(H4), y, uint32_t(O), mc, false);
    vit_act(e, "vit_bias_act_b", y, uint32_t(O), m2_.b, out.view(size_t(m0) * O * 2, size_t(mc) * O * 2), uint32_t(O),
            mc, O, 0);
  }
  cs.finish();
  for (Buffer& b : scratch) g.release(b);
}

}  // namespace ni::vision

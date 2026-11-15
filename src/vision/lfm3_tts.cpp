// The depth decoder's CPU v1 (see lfm3_tts.h): a self-contained safetensors reader (the Omni
// checkpoint's model.depth_decoder.* tensors, bf16 -> fp32) and the forward in fp32. Everything is
// tiny per frame (12 slots x 1024, 3 layers) — CPU matmuls are far under the 12.5 fps budget.
#include "vision/lfm3_tts.h"

#include "util/log.h"

#include <algorithm>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ni::tts {
namespace {

// the head_dim of one bf16 value (byte-swapped is not a case here: the checkpoint is little-endian)
float bf16_to_f32(uint16_t v) {
  const uint32_t u = uint32_t(v) << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

[[noreturn]] void die(const std::string& what) { throw std::runtime_error("lfm3_tts: " + what); }

// one tensor's fp32 contents (bf16 and f32 checkpoints alike); the header JSON's spacing is
// writer-dependent, so the fields are located by key and the values by their delimiters
std::vector<float> read_tensor(std::ifstream& f, const std::string& needle, long data_at,
                               const std::string& header) {
  const std::string key = "\"" + needle + "\":";
  const size_t at = header.find(key);
  if (at == std::string::npos) die("no tensor " + needle);
  const auto find_value = [&](const char* field, size_t from) {
    const size_t k = header.find(std::string("\"") + field + "\":", from);
    if (k == std::string::npos) die(needle + ": no " + field);
    return header.find_first_not_of(" \t", k + strlen(field) + 3);
  };
  size_t dtype_at = find_value("dtype", at);
  if (header[dtype_at] == '"') ++dtype_at;  // the value's own opening quote
  const std::string dtype = header.substr(dtype_at, header.find('"', dtype_at) - dtype_at);
  const size_t off_at = find_value("data_offsets", at);
  const size_t bracket = header.find('[', off_at);
  const size_t comma = header.find(',', bracket), close = header.find(']', bracket);
  const long begin = std::stol(header.substr(bracket + 1, comma - bracket - 1));
  const long end = std::stol(header.substr(comma + 1, close - comma - 1));
  const long bytes = end - begin;
  const size_t n = size_t(dtype == "BF16" ? bytes / 2 : bytes / 4);
  std::vector<uint16_t> raw(size_t(bytes) / 2 + 1, 0);
  f.clear();
  f.seekg(data_at + begin);
  f.read(reinterpret_cast<char*>(raw.data()), bytes);
  std::vector<float> out(n);
  if (dtype == "BF16")
    for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(raw[i]);
  else
    for (size_t i = 0; i < n; ++i) __builtin_memcpy(&out[i], raw.data() + i, 4);
  return out;
}

void rms_norm(float* out, const float* x, const float* w, int n, float eps) {
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += double(x[i]) * double(x[i]);
  const float inv = float(1.0 / std::sqrt(ss / n + double(eps)));
  for (int i = 0; i < n; ++i) out[i] = w[i] * x[i] * inv;
}

// y = x . W^T (W: [out, in], the HF layout), fp32
void matvec(float* y, const float* x, const float* w, int out, int in) {
  for (int o = 0; o < out; ++o) {
    double acc = 0;
    const float* row = w + size_t(o) * in;
    for (int i = 0; i < in; ++i) acc += double(row[i]) * double(x[i]);
    y[o] = float(acc);
  }
}

}  // namespace

DepthDecoder::DepthDecoder(const std::string& safetensors) {
  std::ifstream f(safetensors, std::ios::binary);
  if (!f.good()) die("no checkpoint at " + safetensors);
  int64_t header_len = 0;
  f.read(reinterpret_cast<char*>(&header_len), 8);
  std::string header(size_t(header_len), '\0');
  f.read(header.data(), header_len);
  const long data_at = 8 + header_len;

  depth_w_ = read_tensor(f, "model.depth_decoder.depth_linear.weight", data_at, header);
  depth_b_ = read_tensor(f, "model.depth_decoder.depth_linear.bias", data_at, header);
  mask_embedding_ = read_tensor(f, "model.depth_decoder.mask_embedding", data_at, header);
  final_norm_ = read_tensor(f, "model.depth_decoder.model.embedding_norm.weight", data_at, header);
  codebooks_.resize(kCodebooks);
  layers_.resize(kLayers);
  for (int i = 0; i < kCodebooks; ++i) {
    codebooks_[i].embedding =
        read_tensor(f, "model.depth_decoder.depth_embeddings." + std::to_string(i) + ".embedding.weight", data_at, header);
    codebooks_[i].norm =
        read_tensor(f, "model.depth_decoder.depth_embeddings." + std::to_string(i) + ".embedding_norm.weight", data_at, header);
  }
  for (int l = 0; l < kLayers; ++l) {
    const std::string p = "model.depth_decoder.model.layers." + std::to_string(l) + ".";
    Layer& L = layers_[size_t(l)];
    L.q_w = read_tensor(f, p + "self_attn.q_proj.weight", data_at, header);
    L.k_w = read_tensor(f, p + "self_attn.k_proj.weight", data_at, header);
    L.v_w = read_tensor(f, p + "self_attn.v_proj.weight", data_at, header);
    L.o_w = read_tensor(f, p + "self_attn.o_proj.weight", data_at, header);
    L.gate_w = read_tensor(f, p + "self_attn.gate_proj.weight", data_at, header);
    L.qn = read_tensor(f, p + "self_attn.q_norm.weight", data_at, header);
    L.kn = read_tensor(f, p + "self_attn.k_norm.weight", data_at, header);
    L.op_norm = read_tensor(f, p + "operator_norm.weight", data_at, header);
    L.ffn_norm = read_tensor(f, p + "ffn_norm.weight", data_at, header);
    L.w1 = read_tensor(f, p + "feed_forward.w1.weight", data_at, header);
    L.w2 = read_tensor(f, p + "feed_forward.w2.weight", data_at, header);
    L.w3 = read_tensor(f, p + "feed_forward.w3.weight", data_at, header);
  }
}

void DepthDecoder::forward(const std::vector<float>& hidden, const std::vector<int>& codes,
                           const std::vector<uint8_t>& masked, std::vector<float>& logits) const {
  const int F = int(hidden.size() / kTextHidden);
  if (int(codes.size()) != F * kCodebooks || int(masked.size()) != F * kCodebooks) die("shape mismatch");
  // 1. the slots: depth_linear per frame, unflattened to 12 x 1024, then the mask/code embeddings
  std::vector<float> slots(size_t(F) * kCodebooks * kHidden);
  std::vector<float> proj(kCodebooks * kHidden);
  for (int t = 0; t < F; ++t) {
    matvec(proj.data(), hidden.data() + size_t(t) * kTextHidden, depth_w_.data(), kCodebooks * kHidden, kTextHidden);
    for (int i = 0; i < kCodebooks * kHidden; ++i) proj[i] += depth_b_[size_t(i)];
    for (int c = 0; c < kCodebooks; ++c) {
      float* slot = slots.data() + (size_t(t) * kCodebooks + c) * kHidden;
      const float* add = masked[size_t(t) * kCodebooks + c]
                             ? mask_embedding_.data()
                             : codebooks_[size_t(c)].embedding.data() + size_t(codes[size_t(t) * kCodebooks + c]) * kHidden;
      for (int j = 0; j < kHidden; ++j) slot[j] = proj[size_t(c) * kHidden + j] + add[j];
    }
  }
  // 2. the trunk: 3 LFM3 attention layers over the 12 slots, bidirectional, rope at the slot positions
  const float eps = 1e-5f, scale = 1.0f / std::sqrt(float(kHeadDim));
  std::vector<float> x(std::move(slots));
  std::vector<float> xn(size_t(kCodebooks) * kHidden), q(size_t(kCodebooks) * kHidden),
      k(size_t(kCodebooks) * kHidden), v(size_t(kCodebooks) * kHidden), att(size_t(kCodebooks) * kHidden),
      gated(size_t(kCodebooks) * kHidden), o(size_t(kCodebooks) * kHidden), tmp(size_t(kCodebooks) * kHidden),
      gi_(size_t(kCodebooks) * kFfn), ui_(size_t(kCodebooks) * kFfn);
  const int hd = kHeadDim, hq = kHeads, hkv = kKvHeads, groups = hq / hkv;
  for (int l = 0; l < kLayers; ++l) {
    const Layer& L = layers_[size_t(l)];
    // the attention half
    for (int c = 0; c < kCodebooks; ++c) rms_norm(xn.data() + size_t(c) * kHidden, x.data() + size_t(c) * kHidden, L.op_norm.data(), kHidden, eps);
    for (int c = 0; c < kCodebooks; ++c) {
      float* qh = q.data() + size_t(c) * kHidden;
      matvec(qh, xn.data() + size_t(c) * kHidden, L.q_w.data(), kHidden, kHidden);
      for (int h = 0; h < hq; ++h) rms_norm(qh + size_t(h) * hd, qh + size_t(h) * hd, L.qn.data(), hd, eps);  // q_norm is [head_dim], shared by the heads
      // rope at the slot's position, fp32, the rotate-half form (the text tower's lfm3_rope_ref math)
      for (int h = 0; h < hq; ++h) {
        float* r = qh + size_t(h) * hd;
        for (int p = 0; p < hd / 2; ++p) {
          const double freq = std::pow(1e6, -2.0 * p / double(hd)), ang = double(c) * freq;
          const float cs = float(std::cos(ang)), sn = float(std::sin(ang));
          const float a = r[p], b = r[hd / 2 + p];
          r[p] = a * cs - b * sn;
          r[hd / 2 + p] = b * cs + a * sn;
        }
      }
      float* kh = k.data() + size_t(c) * (hkv * hd);
      matvec(kh, xn.data() + size_t(c) * kHidden, L.k_w.data(), hkv * hd, kHidden);
      for (int h = 0; h < hkv; ++h) rms_norm(kh + size_t(h) * hd, kh + size_t(h) * hd, L.kn.data(), hd, eps);  // k_norm likewise
      for (int h = 0; h < hkv; ++h) {
        float* r = kh + size_t(h) * hd;
        for (int p = 0; p < hd / 2; ++p) {
          const double freq = std::pow(1e6, -2.0 * p / double(hd)), ang = double(c) * freq;
          const float cs = float(std::cos(ang)), sn = float(std::sin(ang));
          const float a = r[p], b = r[hd / 2 + p];
          r[p] = a * cs - b * sn;
          r[hd / 2 + p] = b * cs + a * sn;
        }
      }
      matvec(v.data() + size_t(c) * (hkv * hd), xn.data() + size_t(c) * kHidden, L.v_w.data(), hkv * hd, kHidden);
      float* gate = gated.data() + size_t(c) * hq;
      matvec(gate, xn.data() + size_t(c) * kHidden, L.gate_w.data(), hq, kHidden);
      for (int h = 0; h < hq; ++h) gate[h] = 1.0f / (1.0f + std::exp(-gate[h]));
    }
    // the bidirectional attention over the 12 slots (kv groups repeated)
    for (int c = 0; c < kCodebooks; ++c) {
      for (int h = 0; h < hq; ++h) {
        const float* qh = q.data() + size_t(c) * kHidden + size_t(h) * hd;
        const int kvh = h / groups;
        double mx = -1e300;
        double sc[16];
        for (int s = 0; s < kCodebooks; ++s) {
          const float* ks = k.data() + size_t(s) * (hkv * hd) + size_t(kvh) * hd;
          double acc = 0;
          for (int d = 0; d < hd; ++d) acc += double(qh[d]) * double(ks[d]);
          sc[s] = acc * scale;
          mx = std::max(mx, sc[s]);
        }
        double den = 0;
        for (int s = 0; s < kCodebooks; ++s) den += std::exp(sc[s] - mx);
        float* oh = att.data() + size_t(c) * kHidden + size_t(h) * hd;
        for (int d = 0; d < hd; ++d) oh[d] = 0.f;
        for (int s = 0; s < kCodebooks; ++s) {
          const float w = float(std::exp(sc[s] - mx) / den);
          const float* vs = v.data() + size_t(s) * (hkv * hd) + size_t(kvh) * hd;
          for (int d = 0; d < hd; ++d) oh[d] += w * vs[d];
        }
      }
      // the per-head sigmoid gate, then o_proj and the residual
      float* out = o.data() + size_t(c) * kHidden;
      for (int h = 0; h < hq; ++h) {
        const float* ah = att.data() + size_t(c) * kHidden + size_t(h) * hd;
        const float g = gated[size_t(c) * hq + h];
        for (int d = 0; d < hd; ++d) out[size_t(h) * hd + d] = ah[d] * g;
      }
      matvec(tmp.data() + size_t(c) * kHidden, out, L.o_w.data(), kHidden, kHidden);
      for (int j = 0; j < kHidden; ++j) x[size_t(c) * kHidden + j] += tmp[size_t(c) * kHidden + j];
    }
    // the FFN half: SwiGLU into dedicated per-layer buffers (gi/ui sized F x kFfn once, below)
    for (int c = 0; c < kCodebooks; ++c) rms_norm(xn.data() + size_t(c) * kHidden, x.data() + size_t(c) * kHidden, L.ffn_norm.data(), kHidden, eps);
    for (int c = 0; c < kCodebooks; ++c) {
      float* gi = gi_.data() + size_t(c) * kFfn;
      float* ui = ui_.data() + size_t(c) * kFfn;
      matvec(gi, xn.data() + size_t(c) * kHidden, L.w1.data(), kFfn, kHidden);
      matvec(ui, xn.data() + size_t(c) * kHidden, L.w3.data(), kFfn, kHidden);
      for (int j = 0; j < kFfn; ++j) gi[j] = gi[j] / (1.0f + std::exp(-gi[j])) * ui[j];
      matvec(o.data() + size_t(c) * kHidden, gi, L.w2.data(), kHidden, kFfn);
      for (int j = 0; j < kHidden; ++j) x[size_t(c) * kHidden + j] += o[size_t(c) * kHidden + j];
    }
  }
  // 3. the final norm and the tied heads
  for (int c = 0; c < kCodebooks; ++c) rms_norm(tmp.data() + size_t(c) * kHidden, x.data() + size_t(c) * kHidden, final_norm_.data(), kHidden, eps);
  logits.resize(size_t(F) * kCodebooks * kCodebookSize);
  for (int t = 0; t < F; ++t)
    for (int c = 0; c < kCodebooks; ++c) {
      const float* slot = tmp.data() + size_t(t) * kCodebooks * kHidden + size_t(c) * kHidden;
      // rms per codebook: the reference normalizes the slot then applies its embedding_norm BEFORE the tied head
      rms_norm(tmp.data() + size_t(t) * kCodebooks * kHidden + size_t(c) * kHidden, tmp.data() + size_t(t) * kCodebooks * kHidden + size_t(c) * kHidden, codebooks_[size_t(c)].norm.data(), kHidden, eps);
      matvec(logits.data() + (size_t(t) * kCodebooks + size_t(c)) * kCodebookSize, slot, codebooks_[size_t(c)].embedding.data(), kCodebookSize, kHidden);
    }
}


std::vector<int> DepthDecoder::sample_frame(const std::vector<float>& hidden, float temperature,
                                            uint64_t seed) const {
  static const int kSchedule[] = {4, 4, 4};  // the reference's config.maskgit_schedule
  std::vector<int> order(kCodebooks);
  for (int i = 0; i < kCodebooks; ++i) order[size_t(i)] = i;
  std::mt19937 rng(seed);
  std::shuffle(order.begin(), order.end(), rng);
  return sample_frame(hidden, temperature, order);
}

std::vector<int> DepthDecoder::sample_frame(const std::vector<float>& hidden, float temperature,
                                            const std::vector<int>& order) const {
  static const int kSchedule[] = {4, 4, 4};  // the reference's config.maskgit_schedule
  std::mt19937 rng(0);  // only used when the caller passes a temperature: a fixed stream for the
                        // fixed order (the seeded variant owns its shuffle stream)
  std::vector<int> codes(kCodebooks, 0);
  std::vector<uint8_t> revealed(kCodebooks, 0);
  std::vector<float> logits;
  int pick = 0;
  for (int pass = 0; pass < 3; ++pass) {
    std::vector<uint8_t> masked(kCodebooks);
    for (int i = 0; i < kCodebooks; ++i) masked[size_t(i)] = !revealed[size_t(i)];
    forward(hidden, codes, masked, logits);
    for (int j = 0; j < kSchedule[pass]; ++j) {
      const int c = order[size_t(pick++)];
      if (temperature <= 0) {
        int best = 0;
        float bv = -1e30f;
        const float* lg = logits.data() + size_t(c) * kCodebookSize;
        for (int k = 0; k < kCodebookSize; ++k)
          if (lg[k] > bv) bv = lg[k], best = k;
        codes[size_t(c)] = best;
      } else {
        std::vector<double> p(kCodebookSize);
        double mx = -1e300, sum = 0;
        const float* lg = logits.data() + size_t(c) * kCodebookSize;
        for (int k = 0; k < kCodebookSize; ++k) mx = std::max(mx, double(lg[k] / temperature));
        for (int k = 0; k < kCodebookSize; ++k) sum += p[size_t(k)] = std::exp(double(lg[k] / temperature) - mx);
        std::uniform_real_distribution<double> uni(0.0, sum);
        double draw = uni(rng), run = 0;
        for (int k = 0; k < kCodebookSize; ++k) {
          run += p[size_t(k)];
          if (run >= draw) {
            codes[size_t(c)] = k;
            break;
          }
        }
      }
      revealed[size_t(c)] = 1;
    }
  }
  return codes;
}


namespace {
struct Dims { uint32_t x, y; };
}  // namespace

// ---- the GPU path (src/kernels/tts.metal; the CPU forward above is the oracle) ----

// The fp32 host copy as bf16 into a device buffer (the checkpoint's weights are bf16; the load
// widened them, this narrows them back — the band is the checkpoint's own).
void DepthDecoder::bf16_upload(Buffer& dst, const std::vector<float>& src) const {
  dst = Gpu::get().alloc(src.size() * 2 + 16, "tts_w");
  auto* h = dst.as<uint16_t>();
  for (size_t i = 0; i < src.size(); ++i) {
    uint32_t u;
    memcpy(&u, &src[i], 4);
    h[i] = uint16_t((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
  }
}

void DepthDecoder::load_gpu() {
  if (gpu_ready_) return;
  gw_ = std::make_unique<GpuW>();
  GpuW& W = *gw_;
  bf16_upload(W.depth_w, depth_w_);
  // the biases and the norms stay fp32: the kernels read them as `const device float*`
  auto f32_upload = [](Buffer& dst, const std::vector<float>& src) {
    dst = Gpu::get().alloc(src.size() * 4 + 16, "tts_f32_w");
    memcpy(dst.ptr(), src.data(), src.size() * 4);
  };
  f32_upload(W.depth_b, depth_b_);
  f32_upload(W.mask_emb, mask_embedding_);
  f32_upload(W.final_norm, final_norm_);
  for (int c = 0; c < kCodebooks; ++c) {
    bf16_upload(W.emb[c], codebooks_[size_t(c)].embedding);
    f32_upload(W.emb_norm[c], codebooks_[size_t(c)].norm);
  }
  for (int l = 0; l < kLayers; ++l) {
    const Layer& L = layers_[size_t(l)];
    GpuW::LG& G = W.L[size_t(l)];
    bf16_upload(G.q, L.q_w);
    bf16_upload(G.k, L.k_w);
    bf16_upload(G.v, L.v_w);
    bf16_upload(G.o, L.o_w);
    bf16_upload(G.gate, L.gate_w);
    bf16_upload(G.w1, L.w1);
    bf16_upload(G.w2, L.w2);
    bf16_upload(G.w3, L.w3);
    G.qn = Gpu::get().alloc(L.qn.size() * 4 + 16, "tts_w");
    memcpy(G.qn.ptr(), L.qn.data(), L.qn.size() * 4);
    G.kn = Gpu::get().alloc(L.kn.size() * 4 + 16, "tts_w");
    memcpy(G.kn.ptr(), L.kn.data(), L.kn.size() * 4);
    G.op_norm = Gpu::get().alloc(L.op_norm.size() * 4 + 16, "tts_w");
    memcpy(G.op_norm.ptr(), L.op_norm.data(), L.op_norm.size() * 4);
    G.ffn_norm = Gpu::get().alloc(L.ffn_norm.size() * 4 + 16, "tts_w");
    memcpy(G.ffn_norm.ptr(), L.ffn_norm.data(), L.ffn_norm.size() * 4);
  }
  // the scratch: the slot state, the projections and the logits (fp32, F = 1)
  const size_t H = size_t(kCodebooks) * kHidden;
  W.x = Gpu::get().alloc(H * 4 + 16, "tts_x");
  W.xn = Gpu::get().alloc(H * 4 + 16, "tts_xn");
  W.q = Gpu::get().alloc(H * 4 + 16, "tts_q");
  W.k = Gpu::get().alloc(size_t(kCodebooks) * kKvHeads * kHeadDim * 4 + 16, "tts_k");
  W.v = Gpu::get().alloc(size_t(kCodebooks) * kKvHeads * kHeadDim * 4 + 16, "tts_v");
  W.gated = Gpu::get().alloc(size_t(kCodebooks) * kHeads * 4 + 16, "tts_gated");
  W.att = Gpu::get().alloc(H * 4 + 16, "tts_att");
  W.tmp = Gpu::get().alloc(H * 4 + 16, "tts_tmp");
  W.proj = Gpu::get().alloc(kCodebooks * kHidden * 4 + 16, "tts_proj");
  W.gi = Gpu::get().alloc(size_t(kCodebooks) * kFfn * 4 + 16, "tts_gi");
  W.ui = Gpu::get().alloc(size_t(kCodebooks) * kFfn * 4 + 16, "tts_ui");
  W.logits = Gpu::get().alloc(size_t(kCodebooks) * kCodebookSize * 4 + 16, "tts_logits");
  // the zero pad: the trunk/heads gemv bind this as a no-op bias (the engine rejects a null
  // buffer), and the kernel reads it as a [out]-long bias. Size it to the largest out that binds
  // it (the FFN's kFfn=2816, the heads' 2017); a 64-byte pad read past 16 floats was an OOB read
  // that the MTL validation layer zeroed but the production JIT fed with adjacent memory.
  const size_t kZeroPad = std::max(size_t(kFfn), size_t(kCodebookSize));
  W.zero = Gpu::get().alloc(kZeroPad * 4 + 16, "tts_zero");
  memset(W.zero.ptr(), 0, kZeroPad * 4 + 16);
  gpu_ready_ = true;
}


std::vector<int> DepthDecoder::sample_frame_gpu(const std::vector<float>& hidden, float temperature,
                                                uint64_t seed) {
  std::vector<int> order(kCodebooks);
  for (int i = 0; i < kCodebooks; ++i) order[size_t(i)] = i;
  std::mt19937 shuf(seed);
  std::shuffle(order.begin(), order.end(), shuf);
  return sample_frame_gpu(hidden, temperature, order);
}

std::vector<int> DepthDecoder::sample_frame_gpu(const std::vector<float>& hidden, float temperature,
                                                const std::vector<int>& order) {
  if (!gpu_ready_) load_gpu();
  if (hidden.size() != size_t(kTextHidden)) die("the hidden must be one frame's 2048 values");
  GpuW& W = *gw_;
  auto e = Gpu::get().begin();
  // 1. the slots: depth_linear (one gemv, 12288 rows), then the mask/code embeddings per slot
  {
    const Dims dims{kCodebooks * kHidden, kTextHidden};
    const Buffer hid = Gpu::get().alloc(kTextHidden * 4 + 16, "tts_hidden");
    memcpy(hid.ptr(), hidden.data(), kTextHidden * 4);
    e->use(Gpu::get().pipeline("tts_gemv"));
    e->buf(0, W.depth_w);
    e->buf(1, hid);
    e->buf(2, W.depth_b);
    e->buf(3, W.proj);
    e->val(4, dims);
    e->dispatch((kCodebooks * kHidden + 63) / 64, 1, 1, 64);  // group count, not thread count
      }
  // the slot adds (masked: the mask_embedding; revealed: the code's embedding), an element kernel
  // (no dedicated kernel: each slot's row add runs as tts_add over the code/mask embedding pointer
  //  selected on the host — 12 host-side selections, 12 small dispatches through one encoder)
  std::vector<int> codes(kCodebooks, 0);
  std::vector<uint8_t> revealed(kCodebooks, 0);
  static const int kSchedule[] = {4, 4, 4};

  // the trunk's weights are ONE shared [out, in] matrix applied to each of the 12 slots:
  // tts_gemv12's grid is (out, 12), one thread per (output row, slot)
  auto gemv_rows = [&](Buffer& y, const Buffer& w, const Buffer& x, const Buffer& b, uint out, uint in) {
    const Dims dims{out, in};
    e->use(Gpu::get().pipeline("tts_gemv12"));
    e->buf(0, w);
    e->buf(1, x);
    e->buf(2, b);
    e->buf(3, y);
    e->val(4, dims);
    e->dispatch((out * kCodebooks + 63) / 64, 1, 1, 64);
  };
  auto rms12 = [&](Buffer& y, const Buffer& x, const Buffer& w) {
    e->use(Gpu::get().pipeline("tts_rms"));
    e->buf(0, x);
    e->buf(1, w);
    e->buf(2, y);
    const float eps = 1e-5f;
    e->val(3, eps);
    e->dispatch(kCodebooks, 1, 1, 128);
  };
  auto add12 = [&](Buffer& y, const Buffer& x) {
    e->use(Gpu::get().pipeline("tts_add"));
    e->buf(0, y);
    e->buf(1, x);
    // the dispatch arg is the group COUNT (dispatchThreadgroups), not the thread count: kHidden
    // elements over a 256-wide threadgroup is ceil(.../256) groups (the old kHidden-groups form
    // launched 256x the threads, an OOB read/write on the unguarded element kernel).
    e->dispatch((kCodebooks * kHidden + 255) / 256, 1, 1, 256);
  };

  // the mask/code embeddings: gathered on the host into one [12, 1024] fp32 staging (12 rows of
  // either the mask embedding or one code's embedding) and added once
  const Buffer adds = Gpu::get().alloc(size_t(kCodebooks) * kHidden * 4 + 16, "tts_adds");
  auto fill_adds = [&] {
    auto* h = adds.as<float>();
    for (int c = 0; c < kCodebooks; ++c) {
      const float* src = revealed[size_t(c)] ? codebooks_[size_t(c)].embedding.data() +
                                                       size_t(codes[size_t(c)]) * kHidden
                                             : mask_embedding_.data();
      memcpy(h + size_t(c) * kHidden, src, size_t(kHidden) * 4);
    }
  };

  int pick = 0;
  std::vector<float> logits_host(size_t(kCodebooks) * kCodebookSize);
  for (int pass = 0; pass < 3; ++pass) {
    std::vector<uint8_t> masked(kCodebooks);
    for (int i = 0; i < kCodebooks; ++i) masked[size_t(i)] = !revealed[size_t(i)];
    fill_adds();
    // x = proj + adds
    memcpy(W.x.ptr(), adds.ptr(), size_t(kCodebooks) * kHidden * 4);
    add12(W.x, W.proj);
    
    // the trunk: 3 layers
    for (int l = 0; l < kLayers; ++l) {
      const GpuW::LG& G = W.L[size_t(l)];
      rms12(W.xn, W.x, G.op_norm);
      gemv_rows(W.q, G.q, W.xn, W.zero, kHidden, kHidden);
      gemv_rows(W.k, G.k, W.xn, W.zero, kKvHeads * kHeadDim, kHidden);
      gemv_rows(W.v, G.v, W.xn, W.zero, kKvHeads * kHeadDim, kHidden);
      gemv_rows(W.gated, G.gate, W.xn, W.zero, kHeads, kHidden);
            // the q/k rope + the q/k norms (32+8 heads per slot, one thread per head)
      // two separate encoder segments: val() writes the argument buffer the NEXT dispatch
      // reads, so a second val() before the first dispatch would clobber q's is_k
      for (uint is_k = 0; is_k < 2; ++is_k) {
        e->use(Gpu::get().pipeline("tts_rope_qk"));
        e->buf(0, W.q);
        e->buf(1, W.k);
        e->buf(2, G.qn);
        e->buf(3, G.kn);
        e->val(4, is_k);
        e->dispatch(is_k ? kCodebooks * kKvHeads : kCodebooks * kHeads, 1, 1, 64);
              }
      // the attention (12 threads in one group) then the o_proj + the residual
      e->use(Gpu::get().pipeline("tts_attn12"));
      e->buf(0, W.q);
      e->buf(1, W.k);
      e->buf(2, W.v);
      e->buf(3, W.gated);
      e->buf(4, W.att);
      e->dispatch(1, 1, 1, 12);
            gemv_rows(W.tmp, G.o, W.att, W.zero, kHidden, kHidden);
      add12(W.x, W.tmp);
      // the FFN half
      rms12(W.xn, W.x, G.ffn_norm);
      gemv_rows(W.gi, G.w1, W.xn, W.zero, kFfn, kHidden);
      gemv_rows(W.ui, G.w3, W.xn, W.zero, kFfn, kHidden);
      e->use(Gpu::get().pipeline("tts_silu"));
      e->buf(0, W.gi);
      e->buf(1, W.ui);
      e->dispatch((kCodebooks * kFfn + 255) / 256, 1, 1, 256);  // group count, not thread count
      gemv_rows(W.tmp, G.w2, W.gi, W.zero, kHidden, kFfn);
      add12(W.x, W.tmp);
          }
    // the heads: the final norm, then EACH slot's own embedding_norm (in place), then the
    // 12 tied [2017, 1024] gemvs. The CPU oracle does both norms — one final_norm over all
    // slots, then a per-codebook RMSNorm before its head.
    rms12(W.xn, W.x, W.final_norm);
        for (int c = 0; c < kCodebooks; ++c) {
      const float eps = 1e-5f;
      e->use(Gpu::get().pipeline("tts_rms"));
      e->buf(0, W.xn, size_t(c) * kHidden * 4);
      e->buf(1, W.emb_norm[size_t(c)]);
      e->buf(2, W.tmp, size_t(c) * kHidden * 4);
      e->val(3, eps);
      e->dispatch(1, 1, 1, 128);
    }
        for (int c = 0; c < kCodebooks; ++c) {
      const Dims dims{kCodebookSize, kHidden};
      e->use(Gpu::get().pipeline("tts_gemv"));
      e->buf(0, W.emb[size_t(c)]);
      e->buf(1, W.tmp, size_t(c) * kHidden * 4);
      e->buf(2, W.zero);
      e->buf(3, W.logits, size_t(c) * kCodebookSize * 4);
      e->val(4, dims);
      e->dispatch((kCodebookSize + 63) / 64, 1, 1, 64);  // 2017 rows, guarded tail
    }
    e->commit();
    // one command buffer per MaskGIT pass (the CPU reads the logits). A BOUNDED readback: in the
    // engine this decoder's buffer can queue behind the engine's own in-flight step, so an unbounded
    // commit_and_wait here wedged the whole command queue on a TTS frame (the CPU blocked in wait,
    // so the step it queues behind never drained — the circular dependency the 96-frame request timed
    // out on). Bounding it breaks the cycle: the CPU returns within the bound, the engine's step
    // drains, later frames complete; a frame that does not land in time falls back to the CPU oracle
    // (proven to match, 5/5) for that frame — correct but slower — rather than hanging. The Metal
    // fast path (ordering this buffer with the engine's step for <10 ms frames) is unverified on a
    // GPU serve run; NI_TTS_READOUT_MS overrides the bound for diagnosis.
    const double readout_ms = getenv("NI_TTS_READOUT_MS") ? std::atof(getenv("NI_TTS_READOUT_MS")) : 2000.0;
    if (!e->wait_for(readout_ms)) return sample_frame(hidden, temperature, order);
    e = Gpu::get().begin();
    memcpy(logits_host.data(), W.logits.ptr(), logits_host.size() * 4);
    // the reveal (the CPU sampler's tail, unchanged)
    for (int j = 0; j < kSchedule[pass]; ++j) {
      const int c = order[size_t(pick++)];
      if (temperature <= 0) {
        int best = 0;
        float bv = -1e30f;
        const float* lg = logits_host.data() + size_t(c) * kCodebookSize;
        for (int k = 0; k < kCodebookSize; ++k)
          if (lg[k] > bv) bv = lg[k], best = k;
        codes[size_t(c)] = best;
      } else {
        std::vector<double> p(kCodebookSize);
        double mx = -1e300, sum = 0;
        const float* lg = logits_host.data() + size_t(c) * kCodebookSize;
        for (int k = 0; k < kCodebookSize; ++k) mx = std::max(mx, double(lg[k] / temperature));
        for (int k = 0; k < kCodebookSize; ++k) sum += p[size_t(k)] = std::exp(double(lg[k] / temperature) - mx);
        std::uniform_real_distribution<double> uni(0.0, sum);
        std::mt19937 rng(0);  // the CPU sampler's fixed stream (the same draws)
        double draw = uni(rng), run = 0;
        int sel = kCodebookSize - 1;
        for (int k = 0; k < kCodebookSize; ++k) {
          run += p[size_t(k)];
          if (draw <= run) {
            sel = k;
            break;
          }
        }
        codes[size_t(c)] = sel;
      }
      revealed[size_t(c)] = 1;
    }
  }
  return codes;
}

}  // namespace ni::tts


// The depth decoder's CPU v1 (see lfm3_tts.h): a self-contained safetensors reader (the Omni
// checkpoint's model.depth_decoder.* tensors, bf16 -> fp32) and the forward in fp32. Everything is
// tiny per frame (12 slots x 1024, 3 layers) — CPU matmuls are far under the 12.5 fps budget.
#include "vision/lfm3_tts.h"

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

}  // namespace ni::tts

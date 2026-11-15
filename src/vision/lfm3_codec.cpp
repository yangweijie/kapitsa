// The LiquidCodec decoder's CPU v1 (see lfm3_codec.h). All fp32 (the checkpoint's bf16 converted
// at load); the iSTFT synthesizes by direct overlap-add (the per-sample bin sum — one decode of a
// few seconds takes seconds; an FFT path is the later optimization).
#include "vision/lfm3_codec.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ni::tts {
namespace {

float bf16_to_f32(uint16_t v) {
  const uint32_t u = uint32_t(v) << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

[[noreturn]] void die(const std::string& what) { throw std::runtime_error("lfm3_codec: " + what); }

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
  if (header[dtype_at] == '"') ++dtype_at;
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
  if (dtype == "BF16") {
    for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(raw[i]);
  } else {
    // F32: read straight into the float buffer (the uint16 staging's 2-byte stride would
    // misread the 4-byte values — the istft window's NaNs were exactly this)
    f.clear();
    f.seekg(data_at + begin);
    f.read(reinterpret_cast<char*>(out.data()), bytes);
  }
  return out;
}

void rms_norm(float* out, const float* x, const float* w, int n, float eps) {
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += double(x[i]) * double(x[i]);
  const float inv = float(1.0 / std::sqrt(ss / n + double(eps)));
  for (int i = 0; i < n; ++i) out[i] = w[i] * x[i] * inv;
}

void matvec(float* y, const float* x, const float* w, int out, int in) {
  for (int o = 0; o < out; ++o) {
    double acc = 0;
    const float* row = w + size_t(o) * in;
    for (int i = 0; i < in; ++i) acc += double(row[i]) * double(x[i]);
    y[o] = float(acc);
  }
}

}  // namespace

CodecDecoder::CodecDecoder(const std::string& safetensors) {
  std::ifstream f(safetensors, std::ios::binary);
  if (!f.good()) die("no checkpoint at " + safetensors);
  int64_t header_len = 0;
  f.read(reinterpret_cast<char*>(&header_len), 8);
  std::string header(size_t(header_len), '\0');
  f.read(header.data(), header_len);
  const long data_at = 8 + header_len;

  q_proj_out_w_ = read_tensor(f, "quantizer.fsq.project_out.weight", data_at, header);
  q_proj_out_b_ = read_tensor(f, "quantizer.fsq.project_out.bias", data_at, header);
  from_w_ = read_tensor(f, "from_quantizer.weight", data_at, header);
  from_b_ = read_tensor(f, "from_quantizer.bias", data_at, header);
  lin_w_ = read_tensor(f, "decoder.lin.weight", data_at, header);
  lin_b_ = read_tensor(f, "decoder.lin.bias", data_at, header);
  window_ = read_tensor(f, "decoder.istft.window", data_at, header);
  final_norm_ = read_tensor(f, "decoder.lfm.embedding_norm.weight", data_at, header);
  layers_.resize(kLayers);
  for (int l = 0; l < kLayers; ++l) {
    const std::string p = "decoder.lfm.layers." + std::to_string(l) + ".";
    Layer& L = layers_[size_t(l)];
    L.op_norm = read_tensor(f, p + "operator_norm.weight", data_at, header);
    L.post_op_norm = read_tensor(f, p + "post_operator_norm.weight", data_at, header);
    L.ffn_norm = read_tensor(f, p + "ffn_norm.weight", data_at, header);
    L.post_ffn_norm = read_tensor(f, p + "post_ffn_norm.weight", data_at, header);
    L.w1 = read_tensor(f, p + "feed_forward.w1.weight", data_at, header);
    L.w2 = read_tensor(f, p + "feed_forward.w2.weight", data_at, header);
    L.w3 = read_tensor(f, p + "feed_forward.w3.weight", data_at, header);
    L.attention = (l == 2 || l == 4 || l == 6);  // the config's layer_types: attention at 2/4/6
    if (L.attention) {
      L.attention = true;
      L.q_w = read_tensor(f, p + "self_attn.q_proj.weight", data_at, header);
      L.k_w = read_tensor(f, p + "self_attn.k_proj.weight", data_at, header);
      L.v_w = read_tensor(f, p + "self_attn.v_proj.weight", data_at, header);
      L.o_w = read_tensor(f, p + "self_attn.o_proj.weight", data_at, header);
      L.gate_w = read_tensor(f, p + "self_attn.gate_proj.weight", data_at, header);
      L.qn = read_tensor(f, p + "self_attn.q_norm.weight", data_at, header);
      L.kn = read_tensor(f, p + "self_attn.k_norm.weight", data_at, header);
    } else {
      L.conv_w = read_tensor(f, p + "conv.conv.weight", data_at, header);
      L.in_w = read_tensor(f, p + "conv.in_proj.weight", data_at, header);
      L.out_w = read_tensor(f, p + "conv.out_proj.weight", data_at, header);
    }
  }
}

void CodecDecoder::decode(const std::vector<int>& codes, int frames, std::vector<float>& audio,
                          StageProbe* probe) const {
  const int F = frames;
  if (int(codes.size()) != kCodebooks * F) die("shape mismatch");
  const float eps = 1e-5f;
  static const int kLevels[4] = {8, 7, 6, 6};
  static const int kBasis[4] = {1, 8, 56, 336};
  static const int kHalf[4] = {4, 3, 3, 3};

  // 1. the GFSQ unprojection: each codebook's index -> its 4 FSQ components -> 48, then project_out
  std::vector<float> lat(size_t(F) * kQuantDim);
  for (int t = 0; t < F; ++t) {
    std::vector<float> comps(size_t(kCodebooks) * 4);
    for (int c = 0; c < kCodebooks; ++c) {
      const int idx = codes[size_t(c) * F + t];
      for (int d = 0; d < 4; ++d)
        comps[size_t(c) * 4 + d] =
            (float((idx / kBasis[d]) % kLevels[d]) - float(kHalf[d])) / float(kHalf[d]);
    }
    matvec(lat.data() + size_t(t) * kQuantDim, comps.data(), q_proj_out_w_.data(), kQuantDim, kCodebooks * 4);
    for (int j = 0; j < kQuantDim; ++j) lat[size_t(t) * kQuantDim + j] += q_proj_out_b_[size_t(j)];
  }
  // 2. from_quantizer (256 -> 512)
  std::vector<float> emb(size_t(F) * kHidden);
  for (int t = 0; t < F; ++t) {
    matvec(emb.data() + size_t(t) * kHidden, lat.data() + size_t(t) * kQuantDim, from_w_.data(), kHidden, kQuantDim);
    for (int j = 0; j < kHidden; ++j) emb[size_t(t) * kHidden + j] += from_b_[size_t(j)];
  }
  const bool dbg = getenv("NI_CODEC_STAGES") != nullptr;
  if (dbg) {
    fprintf(stderr, "stages latents[0][:4]:");
    for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", lat[j]);
    fprintf(stderr, "\nstages embeddings[0][:6]:");
    for (int j = 0; j < 6; ++j) fprintf(stderr, " %.6f", emb[j]);
    fprintf(stderr, "\n");
  }
  // 3. the interpolate x6, nearest-exact (the reference's F.interpolate(mode nearest-exact))
  const int T = F * kFrameStack;
  std::vector<float> x(size_t(T) * kHidden);
  for (int j = 0; j < T; ++j) {
    // nearest-exact: src = floor((dst + 0.5) / scale) — WITHOUT the half-pixel offset (the torch's
    // nearest-exact keeps the source's pixel centers unshifted); clamped to [0, F)
    int src = int(std::floor((double(j) + 0.5) / double(kFrameStack)));
    src = std::clamp(src, 0, F - 1);
    memcpy(x.data() + size_t(j) * kHidden, emb.data() + size_t(src) * kHidden, size_t(kHidden) * 4);
  }
  // 4. the trunk: 8 layers (5 conv + 3 causal sliding attentions), the sandwich norms, rope 1e6
  const int hd = kHeadDim, hq = kHeads, hkv = kKvHeads, groups = hq / hkv;
  std::vector<float> xn(size_t(T) * kHidden), q(size_t(T) * kHidden), k(size_t(T) * hkv * hd),
      v(size_t(T) * hkv * hd), att(size_t(T) * kHidden), gate(size_t(T) * hq),
      o(size_t(T) * kHidden), gi(size_t(T) * kFfn), ui(size_t(T) * kFfn), tmp(size_t(T) * kHidden),
      bcx(size_t(T) * 3 * kHidden);  // the conv's in_proj writes B, C, x (3 x 512 per frame)
  for (int l = 0; l < kLayers; ++l) {
    const Layer& L = layers_[size_t(l)];
    for (int t = 0; t < T; ++t)
      rms_norm(xn.data() + size_t(t) * kHidden, x.data() + size_t(t) * kHidden, L.op_norm.data(), kHidden, eps);
    if (dbg && l == 0) {
      fprintf(stderr, "cpp norm0(pos0):");
      for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", xn[j]);
      fprintf(stderr, "\ncpp norm0(pos1):");
      for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", xn[size_t(1) * kHidden + j]);
      fprintf(stderr, "\ncpp x(pos0):");
      for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", x[j]);
      fprintf(stderr, "\ncpp x(pos1):");
      for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", x[size_t(1) * kHidden + j]);
      fprintf(stderr, "\n");
    }
    if (!L.attention) {
      // the gated short conv: B, C, x = in_proj; h = B . x; the depthwise causal conv 7; y = C . h
      for (int t = 0; t < T; ++t) {
        matvec(bcx.data() + size_t(t) * (3 * kHidden), xn.data() + size_t(t) * kHidden, L.in_w.data(),
               3 * kHidden, kHidden);
      }
      std::vector<float> convh(size_t(T) * kHidden);
      for (int ch = 0; ch < kHidden; ++ch) {
        const float* w = L.conv_w.data() + size_t(ch) * kConvL;
        for (int t = 0; t < T; ++t) {
          double acc = 0;
          for (int kk = 0; kk < kConvL; ++kk) {
            const int src = t - (kConvL - 1) + kk;
            if (src >= 0)
              acc += double(w[kk]) * double(bcx[size_t(src) * (3 * kHidden) + ch]) *
                     double(bcx[size_t(src) * (3 * kHidden) + size_t(2) * kHidden + ch]);
          }
          convh[size_t(t) * kHidden + ch] = float(acc);
        }
      }
      for (int t = 0; t < T; ++t) {
        const float* BCx = bcx.data() + size_t(t) * (3 * kHidden);
        for (int ch = 0; ch < kHidden; ++ch)
          att[size_t(t) * kHidden + ch] = convh[size_t(t) * kHidden + ch] * BCx[size_t(kHidden) + ch];
        matvec(o.data() + size_t(t) * kHidden, att.data() + size_t(t) * kHidden, L.out_w.data(), kHidden, kHidden);
        if (dbg && l == 0) {
          fprintf(stderr, "cpp conv-out[0][:4]:");
          for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", o[size_t(t) * kHidden + j]);
          fprintf(stderr, "\ncpp B[0][:4]:");
          for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", BCx[j]);
          fprintf(stderr, "\ncpp x[0][:4]:");
          for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", BCx[size_t(2) * kHidden + j]);
          fprintf(stderr, "\ncpp convh[0][:4]:");
          for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", convh[size_t(t) * kHidden + j]);
          fprintf(stderr, "\n");
        }
        rms_norm(o.data() + size_t(t) * kHidden, o.data() + size_t(t) * kHidden, L.post_op_norm.data(),
                 kHidden, eps);
        for (int j = 0; j < kHidden; ++j) x[size_t(t) * kHidden + j] += o[size_t(t) * kHidden + j];
      }
      if (dbg) {
        fprintf(stderr, "stages layer%d [0][:4]:", l);
        for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", x[j]);
        fprintf(stderr, "\n");
      }
    } else {
      // the causal sliding attention (window 30), rope at the frame positions
      for (int t = 0; t < T; ++t) {
        float* qh = q.data() + size_t(t) * kHidden;
        matvec(qh, xn.data() + size_t(t) * kHidden, L.q_w.data(), kHidden, kHidden);
        for (int h = 0; h < hq; ++h)
          rms_norm(qh + size_t(h) * hd, qh + size_t(h) * hd, L.qn.data(), hd, eps);
        for (int h = 0; h < hq; ++h) {
          float* r = qh + size_t(h) * hd;
          for (int p = 0; p < hd / 2; ++p) {
            const double ang = double(t) * std::pow(1e6, -2.0 * p / double(hd));
            const float cs = float(std::cos(ang)), sn = float(std::sin(ang));
            const float a = r[p], b = r[hd / 2 + p];
            r[p] = a * cs - b * sn;
            r[hd / 2 + p] = b * cs + a * sn;
          }
        }
        float* kh = k.data() + size_t(t) * (hkv * hd);
        matvec(kh, xn.data() + size_t(t) * kHidden, L.k_w.data(), hkv * hd, kHidden);
        for (int h = 0; h < hkv; ++h) rms_norm(kh + size_t(h) * hd, kh + size_t(h) * hd, L.kn.data(), hd, eps);
        for (int h = 0; h < hkv; ++h) {
          float* r = kh + size_t(h) * hd;
          for (int p = 0; p < hd / 2; ++p) {
            const double ang = double(t) * std::pow(1e6, -2.0 * p / double(hd));
            const float cs = float(std::cos(ang)), sn = float(std::sin(ang));
            const float a = r[p], b = r[hd / 2 + p];
            r[p] = a * cs - b * sn;
            r[hd / 2 + p] = b * cs + a * sn;
          }
        }
        matvec(v.data() + size_t(t) * (hkv * hd), xn.data() + size_t(t) * kHidden, L.v_w.data(), hkv * hd, kHidden);
        float* gate_t = gate.data() + size_t(t) * hq;
        matvec(gate_t, xn.data() + size_t(t) * kHidden, L.gate_w.data(), hq, kHidden);
        for (int h = 0; h < hq; ++h) gate_t[h] = 1.0f / (1.0f + std::exp(-gate_t[h]));
      }
      // the causal attention with the sliding window 30
      const int win = 31;
      for (int t = 0; t < T; ++t) {
        const int beg = std::max(0, t - win + 1);
        for (int h = 0; h < hq; ++h) {
          const float* qh = q.data() + size_t(t) * kHidden + size_t(h) * hd;
          const int kvh = h / groups;
          double mx = -1e300;
          double sc[32];
          const int n = t - beg + 1;
          for (int s = beg; s <= t; ++s) {
            const float* ks = k.data() + size_t(s) * (hkv * hd) + size_t(kvh) * hd;
            double acc = 0;
            for (int d = 0; d < hd; ++d) acc += double(qh[d]) * double(ks[d]);
            sc[size_t(s - beg)] = acc / std::sqrt(float(hd));  // the scale = head_dim^-0.5
            mx = std::max(mx, sc[size_t(s - beg)]);
          }
          double den = 0;
          for (int s = 0; s < n; ++s) den += std::exp(sc[s] - mx);
          float* oh = att.data() + size_t(t) * kHidden + size_t(h) * hd;
          for (int d = 0; d < hd; ++d) oh[d] = 0.f;
          for (int s = beg; s <= t; ++s) {
            const float w = float(std::exp(sc[size_t(s - beg)] - mx) / den);
            const float* vs = v.data() + size_t(s) * (hkv * hd) + size_t(kvh) * hd;
            for (int d = 0; d < hd; ++d) oh[d] += w * vs[d];
          }
        }
        // the per-head gate, o_proj, the sandwich's post-norm residual
        float* out = o.data() + size_t(t) * kHidden;
        for (int h = 0; h < hq; ++h) {
          const float* ah = att.data() + size_t(t) * kHidden + size_t(h) * hd;
          const float g = gate[size_t(t) * hq + h];
          for (int d = 0; d < hd; ++d) out[size_t(h) * hd + d] = ah[d] * g;
        }
        matvec(tmp.data() + size_t(t) * kHidden, out, L.o_w.data(), kHidden, kHidden);
        rms_norm(o.data() + size_t(t) * kHidden, tmp.data() + size_t(t) * kHidden, L.post_op_norm.data(),
                 kHidden, eps);
        for (int j = 0; j < kHidden; ++j) x[size_t(t) * kHidden + j] += o[size_t(t) * kHidden + j];
      }
    }
    // the FFN half with its sandwich post-norm
    if (dbg && l == 0) {
      fprintf(stderr, "cpp attn-half x[0][:4]:");
      for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", x[j]);
      fprintf(stderr, "\n");
    }
    for (int t = 0; t < T; ++t)
      rms_norm(xn.data() + size_t(t) * kHidden, x.data() + size_t(t) * kHidden, L.ffn_norm.data(), kHidden, eps);
    for (int t = 0; t < T; ++t) {
      matvec(gi.data() + size_t(t) * kFfn, xn.data() + size_t(t) * kHidden, L.w1.data(), kFfn, kHidden);
      matvec(ui.data() + size_t(t) * kFfn, xn.data() + size_t(t) * kHidden, L.w3.data(), kFfn, kHidden);
      for (int j = 0; j < kFfn; ++j) gi[size_t(t) * kFfn + j] = gi[size_t(t) * kFfn + j] / (1.0f + std::exp(-gi[size_t(t) * kFfn + j])) * ui[size_t(t) * kFfn + j];
      matvec(tmp.data() + size_t(t) * kHidden, gi.data() + size_t(t) * kFfn, L.w2.data(), kHidden, kFfn);
      rms_norm(o.data() + size_t(t) * kHidden, tmp.data() + size_t(t) * kHidden, L.post_ffn_norm.data(),
               kHidden, eps);
      if (dbg && l == 0 && t == 0) {
        fprintf(stderr, "cpp ffn-norm-in [0][:4]:");
        for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", x[size_t(t) * kHidden + j]);
        fprintf(stderr, "\ncpp ffn-norm-out [0][:4]:");
        for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", xn[size_t(t) * kHidden + j]);
        fprintf(stderr, "\ncpp ffn-out [0][:4]:");
        for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", tmp[size_t(t) * kHidden + j]);
        fprintf(stderr, "\ncpp ffn-post [0][:4]:");
        for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", o[size_t(t) * kHidden + j]);
        fprintf(stderr, "\n");
      }
      for (int j = 0; j < kHidden; ++j) x[size_t(t) * kHidden + j] += o[size_t(t) * kHidden + j];
    }
    if (dbg) {
      float mx = 0;
      for (float v : x) mx = std::max(mx, std::fabs(v));
      fprintf(stderr, "stages full%d [0][:4]:", l);
      for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", x[j]);
      fprintf(stderr, " | max|x| over all t: %.3f\n", mx);
      for (int p : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) {
        fprintf(stderr, "stages full%d [p%d]:", l, p);
        for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", x[size_t(p) * kHidden + j]);
        fprintf(stderr, "\n");
      }
    }
  }
  // 5. the final norm, the STFT head, the iSTFT with the envelope and the trim
  for (int t = 0; t < T; ++t)
    rms_norm(tmp.data() + size_t(t) * kHidden, x.data() + size_t(t) * kHidden, final_norm_.data(), kHidden, eps);
  if (probe) probe->final0.assign(tmp.data(), tmp.data() + 4);  // the trunk's frame-0 final-norm state
  if (dbg) {
    fprintf(stderr, "stages final [0][:4]:");
    for (int j = 0; j < 4; ++j) fprintf(stderr, " %.6f", tmp[j]);
  }
  if (dbg) {
    size_t bad = 0;
    for (float v : x) bad += std::isnan(v) || std::isinf(v);
    fprintf(stderr, "codec: the trunk's x: NaN/inf %zu of %zu\n", bad, x.size());
    size_t bads = 0, firsts = 0;
    for (size_t i = 0; i < x.size(); ++i)
      if ((std::isnan(x[i]) || std::isinf(x[i])) && !bads++) firsts = i;
    if (bads) fprintf(stderr, "codec: the first bad x at %zu (the frame %d)\n", firsts, int(firsts / kHidden));
  }
  const int bins = kFft / 2 + 1;
  // spec[t][bin] = lin(tmp[t]) -> log_magnitude (bins 0..640), angle (bins 641..1281)
  std::vector<float> spec(size_t(T) * bins * 2);
  for (int t = 0; t < T; ++t) {
    matvec(spec.data() + size_t(t) * bins * 2, tmp.data() + size_t(t) * kHidden, lin_w_.data(), bins * 2, kHidden);
    for (int j = 0; j < bins * 2; ++j) spec[size_t(t) * bins * 2 + j] += lin_b_[size_t(j)];
  }
  if (probe) probe->logmag0.assign(spec.data(), spec.data() + 4);  // the STFT head's frame-0 log-mags
  if (dbg) {
    size_t bad = 0;
    float lmx = -1e30f, lmn = 1e30f;
    for (size_t i = 0; i < spec.size(); ++i) {
      if (std::isnan(spec[i]) || std::isinf(spec[i])) ++bad;
      if (i % (bins * 2) < bins) {  // the log-magnitude half
        lmx = std::max(lmx, spec[i]);
        lmn = std::min(lmn, spec[i]);
      }
    }
    fprintf(stderr, "codec: the spec: NaN/inf %zu of %zu, the log-mag range [%.3f, %.3f]\n", bad, spec.size(),
            lmn, lmx);
    fprintf(stderr, "stages lin-out [0][:6]:");
    for (int j = 0; j < 6; ++j) fprintf(stderr, " %.6f", spec[j]);
    fprintf(stderr, "\n");
  }
  // the direct overlap-add synthesis: audio[s] += w[offset] * Re(spec_f * e^{2 pi i f s' / 1280})
  const long out_n = (long(T) - 1) * kHop + kWin;
  std::vector<double> acc(out_n, 0.0), env(out_n, 0.0);
  for (int t = 0; t < T; ++t) {
    const long base = long(t) * kHop;
    const float* lm = spec.data() + size_t(t) * bins * 2;
    const float* ang = lm + bins;
    for (int d = 0; d < kWin; ++d) {
      const long s = base + d;
      // the frame's phase runs on the LOCAL sample d (0..1279), not the global s — the global
      // phase would add a 2*pi*b*hop/N rotation per frame, which the overlap-add must not have
      const double ph = 2.0 * M_PI * double(d) / double(kFft);
      // the irfft's synthesis (norm backward): x[s] = (X_0 + 2*sum_{b=1..639} Re(X_b e^{i...})
      //     + X_640 cos(pi s)) / 1280 — the DC and the Nyquist counted once, the pairs doubled
      double re = std::exp(double(lm[0])) * std::cos(double(ang[0]));
      for (int b = 1; b < kFft / 2; ++b) {
        const double mag = std::exp(double(lm[b]));
        const double th = double(ang[b]) + ph * double(b);
        re += 2.0 * mag * std::cos(th);
      }
      re += std::exp(double(lm[kFft / 2])) * std::cos(double(ang[kFft / 2]) + M_PI * double(s % 2));
      acc[size_t(s)] += double(window_[size_t(d)]) * re / double(kFft);
      env[size_t(s)] += double(window_[size_t(d)]) * double(window_[size_t(d)]);
      if (dbg && s == 1277)
        fprintf(stderr, "codec: s=1277 t=%d d=%d re=%.9g window=%.9g acc=%.9g env=%.9g\n", t, d, re,
                double(window_[size_t(d)]), acc[size_t(s)], env[size_t(s)]);
    }
  }
  audio.assign(size_t(out_n), 0.f);
  for (long s = 0; s < out_n; ++s)
    audio[size_t(s)] = float(acc[size_t(s)] / std::max(env[size_t(s)], 1e-8));
  // the trim at both ends (the whole-decode path)
  if (out_n > 2 * kTrim) audio.erase(audio.begin(), audio.begin() + kTrim);
  if (audio.size() > size_t(out_n - 2 * kTrim)) audio.resize(size_t(out_n - 2 * kTrim));
}

}  // namespace ni::tts

#include "model/lfm3.h"

#include "model/kparams.h"

#include <unordered_map>

#include "util/json_file.h"
#include "util/log.h"
#include "model/lfm3_config.h"
#include "vision/embed.h"

#include <algorithm>
#include <numeric>

namespace ni {

bool Lfm3::is_model(const std::string& dir) {
  const auto jp = json_file(dir + "/config.json");
  const nlohmann::json& j = *jp;
  return lfm3_family(j);
}

Lfm3Config Lfm3::read_config(const std::string& dir) {
  const auto jp = json_file(dir + "/config.json");
  const nlohmann::json& j = *jp;
  if (!lfm3_family(j)) fatal("%s: not an LFM3 config.json", dir.c_str());
  if (const std::string why = lfm3_refusal(j, dir); !why.empty()) fatal("%s", why.c_str());
  const nlohmann::json& t = j.contains("text_config") && j["text_config"].is_object() ? j["text_config"] : j;
  auto num = [&](const char* k, double d) {
    const auto it = t.find(k);
    return it != t.end() && it->is_number() ? it->get<double>() : d;
  };
  Lfm3Config c;
  c.hidden = int(num("hidden_size", 0));
  c.vocab = int(num("vocab_size", 0));
  c.layers = int(num("num_hidden_layers", 0));
  c.hq = int(num("num_attention_heads", 0));
  c.hkv = int(num("num_key_value_heads", 0));
  c.hd = int(num("head_dim", c.hq ? c.hidden / c.hq : 0));
  c.ffn = int(num("intermediate_size", 0));
  c.window = int(num("sliding_window", 0));
  c.conv_k = int(num("conv_L_cache", 7));
  c.eps = float(num("norm_eps", 1e-5));
  c.tied = t.value("tie_word_embeddings", true);
  for (const std::string& s : lfm3_layer_types(j)) c.kind.push_back(s == "conv" ? 'C' : s == "full_attention" ? 'F' : 'S');
  const nlohmann::json rp = t.contains("rope_parameters") && t["rope_parameters"].is_object()
                                ? t["rope_parameters"]
                                : nlohmann::json::object();
  const auto rope_of = [&](const char* type, double def_theta, double def_partial) {
    const auto it = rp.find(type);
    if (it == rp.end() || !it->is_object()) return std::pair{def_theta, def_partial};
    return std::pair{it->value("rope_theta", def_theta), it->value("partial_rotary_factor", def_partial)};
  };
  std::tie(c.theta, c.partial) = rope_of("full_attention", 1e6, 0.5);
  c.theta_swa = rope_of("sliding_attention", 1e4, 1.0).first;
  if (t.contains("eos_token_id")) {
    const auto& e = t["eos_token_id"];
    if (e.is_array()) c.eos = e.get<std::vector<int>>();
    else if (e.is_number()) c.eos = {e.get<int>()};
  }
  c.bos = int(num("bos_token_id", -1));
  const nlohmann::json q = t.contains("quantization") ? t["quantization"]
                          : j.contains("quantization") ? j["quantization"]
                                                       : nlohmann::json(nlohmann::json::value_t::object);
  if (q.is_object() && !q.empty()) {
    const std::string mode = q.value("mode", std::string("affine"));
    const int bits = q.value("bits", 0), gs = q.value("group_size", 0);
    if (mode != "affine" ||
        !((gs == 64 && (bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8)) || (gs == 32 && bits == 4) ||
          (gs == 128 && bits == 8)))
      fatal("%s: LFM3 linears in %s %d-bit group %d, which Kapitsa's LFM3 path does not run (affine 3/4/5/6/8-bit "
            "group 64, or 4-bit group 32 as a Q4_1 GGUF import keeps them)",
            dir.c_str(), mode.c_str(), bits, gs);
    c.bits = bits, c.gs = gs;
  }
  if (!c.tied) fatal("%s: LFM3 without tied embeddings is not supported (the loader reads no lm_head)", dir.c_str());
  if (c.kind.empty() || int(c.kind.size()) != c.layers || c.hidden <= 0 || c.vocab <= 0 || c.ffn <= 0 ||
      c.hq <= 0 || c.hkv <= 0 || c.hd != 128 || c.conv_k != 7)
    fatal("%s: an LFM3 shape Kapitsa does not run (hidden %d, %d layers, heads %d / %d of %d, conv %d)",
          dir.c_str(), c.hidden, c.layers, c.hq, c.hkv, c.hd, c.conv_k);
  if (!std::any_of(c.kind.begin(), c.kind.end(), [](char k) { return k == 'F' || k == 'S'; }))
    fatal("%s: an LFM3 schedule without an attention layer", dir.c_str());
  return c;
}

// Bytes per (kv head, position) of a full-attention cache: ni::kv_row_bytes (metal/gpu.h), which follows NI_KV_I8
// and matches the attention kernels' kKvRow.
size_t Lfm3::kv_per_pos(const Lfm3Config& c) {
  const size_t full = size_t(std::count(c.kind.begin(), c.kind.end(), 'F'));
  return full * 2 * size_t(c.hkv) * kv_row_bytes(c.hd);
}

// A sliding-window layer keeps a ring of window + 8 positions (Flint's layout: a ring cannot go back further than
// its 8 spare rows, so checkpoints hold copies of the rings).
size_t Lfm3::ring_bytes(const Lfm3Config& c) {
  const size_t swa = size_t(std::count(c.kind.begin(), c.kind.end(), 'S'));
  const size_t rows = size_t(c.window + 8 + 7) / 8 * 8;
  return swa * 2 * size_t(c.hkv) * rows * kv_row_bytes(c.hd);
}

// A conv layer keeps a ring of conv_k - 1 frames of the hidden size in bf16.
size_t Lfm3::conv_state_bytes(const Lfm3Config& c) {
  const size_t convs = size_t(std::count(c.kind.begin(), c.kind.end(), 'C'));
  return convs * size_t(c.conv_k - 1) * size_t(c.hidden) * 2;
}

MemPlan Lfm3::plan(const std::string& path, int prefill_chunk) {
  MemPlan m;
  const Lfm3Config c = read_config(path);
  const SafetensorsIndex idx(path);
  for (const auto& [name, t] : idx.all()) m.weights += t.len;
  const size_t H = size_t(c.hidden), T = size_t(std::max(8, prefill_chunk));
  const size_t QD = size_t(c.hq) * c.hd, KD = size_t(c.hkv) * c.hd, I = size_t(c.ffn);
  // prefill activations for T rows: the hidden, the normed mixer input, the conv layer's projections and taps in
  // fp32, the attention's qkv in fp32 and its output in bf16, the gate, the FFN's three widths, the logits row
  const size_t nconv = size_t(std::count(c.kind.begin(), c.kind.end(), 'C'));
  m.activations = T * H * 4 + T * H * 2 + nconv * T * 5 * H * 4 + nconv * T * 3 * H * 2 +
                  (size_t(c.kind.size()) - nconv) * (T * (QD + 2 * KD) * 4 + T * QD * 2 + T * size_t(c.hq) * 4) +
                  size_t(c.kind.size()) * (2 * T * I * 4 + T * H * 4) + std::max(T, size_t(8)) * size_t(c.vocab) * 4;
  m.slot_state = conv_state_bytes(c) * 3 + ring_bytes(c) * 3;  // the live rings and their copies in the two checkpoints
  m.kv_per_pos = kv_per_pos(c);
  // one plain decode step's weights: the mixer's projections and the FFN's three, the tied head's embedding row
  const double bpw = c.bits == 16 ? 2.0 : (c.bits + 32.0 / c.gs) / 8.0;
  size_t params = 0;
  for (int l = 0; l < c.layers; ++l) {
    if (c.kind[size_t(l)] == 'C') params += H * (3 * H + H);
    else params += H * (QD + 2 * KD + QD + size_t(c.hq));
    params += 3 * H * I;
  }
  params += H * size_t(c.vocab);
  m.decode_params = params;
  m.decode_bytes = size_t(double(params) * bpw);
  m.head_row_bytes = size_t(double(H) * bpw);
  m.hidden = c.hidden;
  m.layers = c.layers;
  m.ffn = c.ffn;
  m.dense = true;
  m.ffn_ane = false;  // the LFM3 path runs the FFN on the GPU (no Neural Engine split)
  m.formats = c.bits == 16 ? "bf16" : std::to_string(c.bits) + "-bit group " + std::to_string(c.gs);
  return m;
}

std::vector<float> lfm3_rope_freqs(const double theta, const double partial, const int head_dim) {
  const int angles = int(partial * double(head_dim)) / 2;
  std::vector<float> f(size_t(head_dim / 2), 0.f);
  for (int i = 0; i < angles && i < head_dim / 2; ++i)
    f[size_t(i)] = float(std::pow(theta, -2.0 * i / double(head_dim)));
  return f;
}

void lfm3_rope_ref(const float* x, const int head_dim, const float* freqs, const double pos, float* out) {
  const int pairs = head_dim / 2;
  float tmp[512];
  float* o = head_dim <= 512 ? tmp : new float[size_t(head_dim)];
  for (int i = 0; i < pairs; ++i) {
    const double angle = pos * double(freqs[i]), c = std::cos(angle), s = std::sin(angle);
    o[i] = float(double(x[i]) * c - double(x[pairs + i]) * s);
    o[pairs + i] = float(double(x[pairs + i]) * c + double(x[i]) * s);
  }
  for (int i = 0; i < head_dim; ++i) out[i] = o[i];
  if (o != tmp) delete[] o;
}

void lfm3_conv_ref(const float* u, const int t_count, const int channels, const int k_size, const float* w,
                   const float* state, float* y) {
  for (int t = 0; t < t_count; ++t)
    for (int c = 0; c < channels; ++c) {
      double acc = 0;
      for (int k = 0; k < k_size; ++k) {
        const long long src = int64_t(t) - (k_size - 1 - k);  // the frame this tap reads
        const float x = src >= 0 ? u[size_t(src) * channels + c]
                                 : state ? state[size_t(k_size - 1 + src) * channels + c] : 0.f;
        acc += double(w[size_t(c) * k_size + k]) * double(x);
      }
      y[size_t(t) * channels + c] = float(acc);
    }
}



// ---------------------------------------------------------------- backend

// lfm3.metal's parameter structs, as the C++ side lays them out
struct Lfm3NormParams {
  uint32_t D;
  float eps;
};
struct Lfm3AttnSplitParams {
  uint32_t Hq, Hkv, D, cap, pos, window, splits;
  float scale;
};
static constexpr uint32_t kAttnSplits = 32;
static constexpr uint32_t kPSplits = 8;  // the prefill attention's window splits
// LFM3_X_PROBE's per-layer residual rows (the layerwise bisect for the run-to-run wobble)
static std::vector<Buffer>& xprobe_state() {
  static std::vector<Buffer> v;
  return v;
}
void Lfm3::xprobe_dump() const {
  static const bool on = getenv("LFM3_X_PROBE") && atoi(getenv("LFM3_X_PROBE"));
  if (!on) return;
  const auto& xp = xprobe_state();
  const uint32_t H = uint32_t(cfg_.hidden);
  for (size_t i = 0; i < xp.size(); ++i) {
    fprintf(stderr, "[xprobe] slot %zu%s:", i, i == 0 ? " (embed)" : "");
    const float* r = xp[i].as<float>();
    for (uint32_t j = 0; j < H; ++j) fprintf(stderr, " %.9g", r[j]);
    fprintf(stderr, "\n");
  }
}
struct Lfm3ConvParams {
  uint32_t T, H, K;
};
struct Lfm3BcxParams {
  uint32_t T, H, which;  // which 0: u = B ⊙ x, 1: y = C ⊙ z (lfm3.metal)
};
struct Lfm3QkParams {
  uint32_t T, Hq, Hkv, D, cap, pos0, end;
};
struct Lfm3AttnParams {
  uint32_t Hq, Hkv, D, cap, pos, window;
  float scale;
};
struct Lfm3AttnPSplitParams {
  uint32_t T, Hq, Hkv, D, cap, pos0, window, splits;
  float scale;
};
struct Lfm3AttnPParams {
  uint32_t T, Hq, Hkv, D, cap, pos0, window;
  float scale;
};
// The request's repetition penalty (lfm3_rep_penalty): scatter the HF penalty onto the last `n` response tokens of
// s.tok (base = s.cap - n), before the top-k. Duplicates reinforce. Mirrored in lfm3.metal.
struct Lfm3RepPenParams {
  uint32_t base, n;
  float penalty;
};

Lfm3::Lfm3(const std::string& path, int prefill_chunk) : path_(path) {
  cfg_ = read_config(path);
  if (cfg_.bits != 16 && !((cfg_.gs == 64 && (cfg_.bits == 3 || cfg_.bits == 5 || cfg_.bits == 6 || cfg_.bits == 8)) ||
                           (cfg_.gs == 32 && cfg_.bits == 4)))
    fatal("%s: LFM3 linears in %d-bit group %d, which the qmr kernels do not run (4-bit group 32, 8-bit group 64, "
          "or bf16)",
          path.c_str(), cfg_.bits, cfg_.gs);
  if (prefill_chunk > 0) tmax_ = std::min(prefill_chunk, 2048);
  load();
}

std::string Lfm3::formats_summary() const {
  return cfg_.bits == 16 ? "bf16" : std::to_string(cfg_.bits) + "-bit group " + std::to_string(cfg_.gs) + " (mixed)";
}

// One quantized linear by its module name; N and K from the module's role. bits and gs come from the tensors'
// shapes (w U32 [N, K*bits/32] with s/b BF16 [N, K/gs]), so a mixed file reads per module.
Lfm3Lin lin_of(WeightStore& ws, const std::string& name, uint32_t N, uint32_t K) {
  Lfm3Lin L;
  L.w = ws.find(name + ".weight");
  if (!L.w) fatal("missing %s.weight", name.c_str());
  L.N = N, L.K = K;
  if (L.w->dtype == DType::U32) {
    L.s = ws.find(name + ".scales");
    L.b = ws.find(name + ".biases");
    if (!L.s || !L.b || L.s->shape.size() != 2 || int64_t(L.s->shape[0]) != L.w->shape[0])
      fatal("%s.weight: U32 codes without the [N, K/gs] scales and biases", name.c_str());
    L.gs = uint32_t(K / uint64_t(L.s->shape[1]));
    L.bits = int(uint64_t(L.w->shape[1]) * 32 / K);
    if ((L.gs != 32 && L.gs != 64) || (L.bits != 4 && L.bits != 8) || L.s->dtype != DType::BF16)
      fatal("%s.weight: %d-bit group %d, which the LFM3 path does not run (4-bit group 32, 8-bit group 64)",
            name.c_str(), L.bits, L.gs);
  } else {
    L.bits = 16, L.gs = 64;
  }
  return L;
}

void Lfm3::load() {
  Gpu& g = Gpu::get();
  ws_.open_dir(path_);
  const uint32_t H = uint32_t(cfg_.hidden), I = uint32_t(cfg_.ffn), V = uint32_t(cfg_.vocab);
  const uint32_t QD = uint32_t(cfg_.hq) * uint32_t(cfg_.hd), KD = uint32_t(cfg_.hkv) * uint32_t(cfg_.hd);
  embed_ = lin_of(ws_, "model.embed_tokens", V, H);
  final_norm_ = ws_.find("model.norm.weight");
  if (!final_norm_) fatal("%s: no model.norm.weight", path_.c_str());
  layers_.resize(size_t(cfg_.layers));
  for (int l = 0; l < cfg_.layers; ++l) {
    Lfm3Layer& L = layers_[size_t(l)];
    L.kind = cfg_.kind[size_t(l)];
    n_conv_ += L.kind == 'C';
    const std::string p = "model.layers." + std::to_string(l) + ".";
    const auto t = [&](const std::string& n) {
      const Tensor* t2 = ws_.find(p + n);
      if (!t2) fatal("missing %s", (p + n).c_str());
      return t2;
    };
    L.in_norm = t("operator_norm.weight");
    L.post_norm = t("post_operator_norm.weight");
    L.ffn_norm = t("ffn_norm.weight");
    L.post_ffn_norm = t("post_ffn_norm.weight");
    L.w1 = lin_of(ws_, p + "mlp.w1", I, H);
    L.w3 = lin_of(ws_, p + "mlp.w3", I, H);
    L.w2 = lin_of(ws_, p + "mlp.w2", H, I);
    if (L.kind == 'C') {
      L.conv_w = t("conv.conv.weight");
      L.in_proj = lin_of(ws_, p + "conv.in_proj", 3 * H, H);
      L.out_proj = lin_of(ws_, p + "conv.out_proj", H, H);
    } else {
      L.q = lin_of(ws_, p + "self_attn.q_proj", QD, H);
      L.k = lin_of(ws_, p + "self_attn.k_proj", KD, H);
      L.v = lin_of(ws_, p + "self_attn.v_proj", KD, H);
      L.o = lin_of(ws_, p + "self_attn.o_proj", H, QD);
      L.gate = lin_of(ws_, p + "self_attn.gate_proj", uint32_t(cfg_.hq), H);
      L.qn = t("self_attn.q_norm.weight");
      L.kn = t("self_attn.k_norm.weight");
    }
  }
  // the two RoPE tables
  const std::vector<float> ff = lfm3_rope_freqs(cfg_.theta, cfg_.partial, cfg_.hd);
  const std::vector<float> fs = lfm3_rope_freqs(cfg_.theta_swa, 1.0, cfg_.hd);
  freqs_f_ = g.alloc(ff.size() * 4, "lfm3_freqs_f");
  memcpy(freqs_f_.ptr(), ff.data(), ff.size() * 4);
  freqs_s_ = g.alloc(fs.size() * 4, "lfm3_freqs_s");
  memcpy(freqs_s_.ptr(), fs.data(), fs.size() * 4);
  // activations. The GEMM inputs (xh_, aoh_, hh_ and the prefill's) carry 32 spare rows: the qmr/qrf
  // tiles stage a whole BM of X rows (up to 32) whatever M is, and their output store is the only
  // M-guarded side.
  const size_t R = kMaxRows + 32, T = size_t(tmax_) + 32, h = H, i = I;
  x_ = g.alloc(kMaxRows * h * 4, "lfm3_x");
  xh_ = g.alloc(R * h * 2, "lfm3_xh");
  bcx_ = g.alloc(R * 3 * h * 4, "lfm3_bcx");
  u_ = g.alloc(R * h * 4, "lfm3_u");
  z_ = g.alloc(R * h * 4, "lfm3_z");
  ym_ = g.alloc(R * h * 4, "lfm3_ym");
  mx_ = g.alloc(R * h * 4, "lfm3_mx");
  aoh_ = g.alloc(R * QD * 2, "lfm3_aoh");
  qkv_ = g.alloc(R * (QD + 2 * KD) * 4, "lfm3_qkv");
  gate_ = g.alloc(R * uint64_t(cfg_.hq) * 4, "lfm3_gate");
  q_ = g.alloc(R * QD * 4, "lfm3_q");
  ao_ = g.alloc(R * QD * 4, "lfm3_ao");
  g_ = g.alloc(R * i * 4, "lfm3_g");
  u2_ = g.alloc(R * i * 4, "lfm3_u2");
  h_ = g.alloc(R * i * 4, "lfm3_h");
  hh_ = g.alloc(R * i * 2, "lfm3_hh");
  xn_ = g.alloc(kMaxRows * h * 4, "lfm3_xn");
  logits_ = g.alloc(size_t(V) * 4, "lfm3_logits");
  apart_o_ = g.alloc(size_t(cfg_.hq) * kAttnSplits * size_t(cfg_.hd) * 4, "lfm3_apart_o");
  apart_m_ = g.alloc(size_t(cfg_.hq) * kAttnSplits * 4, "lfm3_apart_m");
  apart_s_ = g.alloc(size_t(cfg_.hq) * kAttnSplits * 4, "lfm3_apart_s");
  const uint32_t ntg = (V + 2047) / 2048, K = 32;
  cand_v_ = g.alloc(size_t(ntg * K) * 4, "lfm3_cand_v");
  cand_i_ = g.alloc(size_t(ntg * K) * 4, "lfm3_cand_i");
  sprob_ = g.alloc(64 * 4, "lfm3_sprob");
  px_ = g.alloc(T * h * 4, "lfm3_px");
  pxh_ = g.alloc(T * h * 2, "lfm3_pxh");
  pbcx_ = g.alloc(T * 3 * h * 4, "lfm3_pbcx");
  pu_ = g.alloc(T * h * 4, "lfm3_pu");
  pz_ = g.alloc(T * h * 4, "lfm3_pz");
  py_ = g.alloc(T * h * 4, "lfm3_py");
  pmix_ = g.alloc(T * h * 4, "lfm3_pmix");
  paoh_ = g.alloc(T * QD * 2, "lfm3_paoh");
  pqkv_ = g.alloc(T * (QD + 2 * KD) * 4, "lfm3_pqkv");
  pgate_ = g.alloc(T * uint64_t(cfg_.hq) * 4, "lfm3_pgate");
  pq_ = g.alloc(T * QD * 4, "lfm3_pq");
  pao_ = g.alloc(T * QD * 4, "lfm3_pao");
  pg_ = g.alloc(T * i * 4, "lfm3_pg");
  pu2_ = g.alloc(T * i * 4, "lfm3_pu2");
  ph_ = g.alloc(T * i * 4, "lfm3_ph");
  phh_ = g.alloc(T * i * 2, "lfm3_phh");
  ppart_o_ = g.alloc(T * size_t(cfg_.hq) * kPSplits * size_t(cfg_.hd) * 4, "lfm3_ppart_o");
  ppart_m_ = g.alloc(T * size_t(cfg_.hq) * kPSplits * 4, "lfm3_ppart_m");
  ppart_s_ = g.alloc(T * size_t(cfg_.hq) * kPSplits * 4, "lfm3_ppart_s");
  // the file's formats (4-bit group 32) run on wfmt.metal's second library: compiled here, at load, so the
  // first prefill's encoder never waits on a library compile mid-pass
  g.compile_formats();
}

Pipeline Lfm3::pipe(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

// Y (+)= X W^T over rows of halfs; by the tensors' shapes: bf16 through qrf_f16, 4-bit group 32 through
// qrf_f4g32 (wfmt.metal's kernels of any format; the qmr/qmm3 family hardcodes group 64), and 8-bit group 64
// through the specialized qmr8. The dispatch mirrors Qwen35's enc_qrf: N tiles on grid x, M tiles on y.
void Lfm3::lin(Encoder& e, const Lfm3Lin& W, const Buffer& xh, const Buffer& y, int rows, bool add, uint32_t y_ld) {
  const kp::Qmm p{uint32_t(rows), W.N, W.K, W.K, y_ld ? y_ld : W.N, add ? 1u : 0u,
                  uint32_t(W.dense() ? 1 : W.gs), 0, 0};
  const std::string f = W.dense() ? "f16" : "f" + std::to_string(W.bits) + "g" + std::to_string(W.gs);
  std::string n;
  uint32_t bm = 32, bn = 128, th = 128;
  if (rows <= 8) n = "qrf_" + f + "_m8_n64", bm = 8, bn = 64, th = 64;
  else if (rows < 64 || W.N <= 64) n = "qrf_" + f + "_m16_n64", bm = 16, bn = 64;
  else n = "qrf_" + f + "_m32_n128";
  if (W.bits == 8 && W.gs == 64) {  // the specialized 8-bit kernels, both with 128 threads
    th = 128;
    if (rows < 64 || W.N <= 64) n = "qmr8_k32_w14_m16_n64", bm = 16, bn = 64;
    else n = "qmr8_k32_w14_m32_n128", bm = 32, bn = 128;
  }
  e.use(pipe(n));
  e.buf(0, xh);
  e.buf(1, W.w->buf);
  e.buf(2, W.dense() ? W.w->buf : W.s->buf);
  e.buf(3, W.dense() ? W.w->buf : W.b->buf);
  e.buf(4, y);
  e.val(5, p);
  e.dispatch((W.N + bn - 1) / bn, (uint32_t(rows) + bm - 1) / bm, 1, th);
}

// The decode matvec of one 4-bit group-32 linear: qmvf reads the fp32 x and folds the RMSNorm, so the
// decode path needs no halfs and no separate norm dispatch for these. The 8-bit group-64 linears keep
// to_half + qmr8 (their specialized kernel takes halfs).
void Lfm3::enc_mv1(Encoder& e, const Lfm3Lin& W, const Buffer& x, const Tensor* norm, const Buffer& y,
                   int rows, uint32_t y_ld) {
  ni::kp::Qmvf pm{};
  pm.K = W.K;
  pm.R = uint32_t(rows);
  pm.x_ld = W.K;
  pm.nseg = 1;
  pm.eps = cfg_.eps;
  pm.N[0] = W.N;
  pm.y_ld[0] = y_ld ? y_ld : W.N;
  pm.add[0] = 0;
  pm.tg_begin[0] = 0;
  pm.tg_begin[1] = (W.N + 7) / 8;
  pm.write_xn = 0;
  e.use(pipe(norm ? "qmvf_b4g32_t1_r4_norm" : "qmvf_b4g32_t1_r4"));
  // the qmvf family declares four segments' buffers (slots 0-15): all bound, the unused segments
  // repeating segment 0's (the kernel writes only the nseg=1 segment's rows)
  for (int seg = 0; seg < 4; ++seg) {
    e.buf(seg * 4 + 0, W.w->buf);
    e.buf(seg * 4 + 1, W.s->buf);
    e.buf(seg * 4 + 2, W.b->buf);
    e.buf(seg * 4 + 3, y);
  }
  e.buf(16, x);
  e.val(17, pm);
  e.buf(18, norm ? norm->buf : x);
  e.buf(19, x);
  e.dispatch(uint32_t(rows), (W.N + 7) / 8, 1, 64);
}

// x's rows through an RMSNorm (norm's weight) into y as halfs; without a norm, plain halfs.
void Lfm3::to_half(Encoder& e, const Buffer& x, const Tensor* norm, const Buffer& y, int rows, uint32_t d) {
  // d is the row's width: hidden for the mixer's inputs and outputs, intermediate_size for the FFN's
  // SwiGLU output (rows_to_half would leave the rest of that row unwritten).
  const kp::Rows p{d ? d : uint32_t(cfg_.hidden), norm ? 1u : 0u, cfg_.eps};
  e.use(pipe("rows_to_half"));
  e.buf(0, x);
  e.buf(1, norm ? norm->buf : x);
  e.buf(2, y);
  e.val(3, p);
  e.dispatch(uint32_t(rows), 1, 1, 256);
}

// The sandwich point: out = x + post_norm . rmsnorm(z), rows of fp32 (out may alias x).
void Lfm3::res_norm(Encoder& e, const Buffer& z, const Tensor* w, const Buffer& x, const Buffer& out, int rows) {
  const Lfm3NormParams p{uint32_t(cfg_.hidden), cfg_.eps};
  e.use(pipe("lfm3_res_norm"));
  e.buf(0, z);
  e.buf(1, w->buf);
  e.buf(2, x);
  e.buf(3, out);
  e.val(4, p);
  e.dispatch(uint32_t(rows), 1, 1, 256);
}

void Lfm3::embed(Encoder& e, const Buffer& tokens, int tok_off, int rows, const Buffer& out) {
  const kp::Embed ep{uint32_t(cfg_.hidden), uint32_t(embed_.gs), uint32_t(tok_off)};
  std::string n = embed_.bits == 4 ? "embed_q4" : "embed_f" + std::to_string(embed_.bits) + "g64";
  if (embed_.dense()) n = "embed_f16";  // bf16 rows, no scales or biases
  e.use(pipe(n));
  e.buf(0, embed_.w->buf);
  e.buf(1, embed_.s ? embed_.s->buf : embed_.w->buf);
  e.buf(2, embed_.b ? embed_.b->buf : embed_.w->buf);
  e.buf(3, tokens);
  e.buf(4, out);
  e.val(5, ep);
  e.dispatch((uint32_t(cfg_.hidden) / 8 + 63) / 64, uint32_t(rows), 1, 64);
}

int Lfm3::conv_ordinal(int layer) const {
  int n = 0;
  for (int l = 0; l < layer; ++l) n += cfg_.kind[size_t(l)] == 'C';
  return n;
}

std::unique_ptr<Seq> Lfm3::new_seq(int limit, int initial) {
  Gpu& g = Gpu::get();
  auto s = std::make_unique<Seq>();
  s->limit = limit;
  s->tok = g.alloc(size_t(limit + 8) * 4, "tokens");
  s->kc.resize(size_t(cfg_.layers));
  s->vc.resize(size_t(cfg_.layers));
  s->conv.resize(size_t(n_conv_));
  const int cap = initial < 0 ? limit : std::min(initial, limit);
  if (cap > 0 && !reserve(*s, cap)) fatal("insufficient memory for a %d-token KV cache", cap);
  return s;
}

// The conv rings and the sliding-window rings (allocated with the sequence, zeroed: the attention reads
// slots no position has written yet); the full-attention layers' caches grow with the conversation
// (kv_regrow copies the old rows).
bool Lfm3::reserve(Seq& s, int need) {
  Gpu& g = Gpu::get();
  if (need > s.limit) return false;
  if (!s.conv[0]) {
    const size_t rb = size_t(cfg_.conv_k - 1) * size_t(cfg_.hidden) * 2;
    for (auto& b : s.conv) {
      b = g.alloc(rb, "lfm3_conv_ring");
      memset(b.ptr(), 0, b.size);
    }
  }
  const size_t swa_rows = size_t(cfg_.window + 8 + 7) / 8 * 8;
  for (int l = 0; l < cfg_.layers; ++l)
    if (cfg_.kind[size_t(l)] == 'S' && !s.kc[size_t(l)]) {
      const size_t rb = size_t(cfg_.hkv) * swa_rows * kv_row_bytes(cfg_.hd);
      s.kc[size_t(l)] = g.alloc(rb, "lfm3_swa_ring");
      s.vc[size_t(l)] = g.alloc(rb, "lfm3_swa_ring");
      memset(s.kc[size_t(l)].ptr(), 0, rb);
      memset(s.vc[size_t(l)].ptr(), 0, rb);
    }
  int first_f = -1;
  for (int l = 0; l < cfg_.layers; ++l)
    if (cfg_.kind[size_t(l)] == 'F') { first_f = l; break; }
  if (first_f < 0 || (!s.kc[size_t(first_f)] || need > s.cap)) {
    const size_t row = size_t(cfg_.hkv) * kv_row_bytes(cfg_.hd);
    auto round_up = [](int v, int m) { return (v + m - 1) / m * m; };
    int target = std::min(s.limit, round_up(std::max(need, s.cap + s.cap / 4), 1024));
    if (mem_budget_ && g.allocated_bytes() + size_t(target - std::max(s.cap, 0)) * kv_per_pos(cfg_) > mem_budget_)
      target = std::min(s.limit, round_up(need, 256));
    if (mem_budget_ && g.allocated_bytes() + size_t(target - std::max(s.cap, 0)) * kv_per_pos(cfg_) > mem_budget_)
      return false;
    const int old = s.cap;
    const uint32_t n16 = uint32_t(kv_row_bytes(cfg_.hd) / 16);
    for (int l = 0; l < cfg_.layers; ++l) {
      if (cfg_.kind[size_t(l)] != 'F') continue;
      for (Buffer* b : {&s.kc[size_t(l)], &s.vc[size_t(l)]}) {
        Buffer nbuf = g.alloc(row * size_t(target), "lfm3_kv");
        memset(nbuf.ptr(), 0, nbuf.size);
        if (old > 0 && *b) {
          const uint32_t p[4] = {uint32_t(old) * n16, uint32_t(target) * n16, uint32_t(cfg_.hkv), 0};
          auto e = g.begin();
          e->use(pipe("kv_regrow"));
          e->buf(0, *b);
          e->buf(1, nbuf);
          e->bytes(2, p, sizeof p);
          e->dispatch((p[0] + 255) / 256, uint32_t(cfg_.hkv), 1, 256);
          e->commit_and_wait();
        }
        if (*b) g.release(*b);
        *b = nbuf;
      }
    }
    s.cap = target;
  }
  return true;
}

void Lfm3::drop_kv(Seq& s) {
  Gpu& g = Gpu::get();
  for (auto* v : {&s.kc, &s.vc})
    for (auto& b : *v) {
      g.release(b);
      b = Buffer{};
    }
  s.cap = 0;
}

void Lfm3::reset(Seq& s) {
  for (auto& b : s.conv) memset(b.ptr(), 0, b.size);
}

// A checkpoint holds copies of the conv rings and the sliding-window rings (the linear caches keep every
// earlier position).
void Lfm3::save(Encoder& e, const Seq& s, Checkpoint& c, int pos) {
  Gpu& g = Gpu::get();
  const size_t nc = size_t(n_conv_), ns = size_t(std::count(cfg_.kind.begin(), cfg_.kind.end(), 'S'));
  static const bool ck = getenv("LFM3_CKPT_DEBUG") && atoi(getenv("LFM3_CKPT_DEBUG"));
  if (c.conv.size() != nc) c.conv.assign(nc, Buffer{});
  if (c.rec.size() != 2 * ns) c.rec.assign(2 * ns, Buffer{});
  if (ck) {
    fprintf(stderr, "[ckpt save] entry this=%p conv=%zu rec=%zu slots:", (void*)&c, c.conv.size(), c.rec.size());
    for (auto& b : c.rec) fprintf(stderr, " %p/%zu", b.mtl, b.size);
    fprintf(stderr, "\n");
  }
  size_t ic = 0, is = 0;
  for (int l = 0; l < cfg_.layers; ++l) {
    const char k = cfg_.kind[size_t(l)];
    if (k == 'C') {
      Buffer& dst = c.conv[ic++];
      if (!dst) dst = g.alloc(s.conv[size_t(conv_ordinal(l))].size, "lfm3_ring_ckpt");
      const uint32_t words = uint32_t(dst.size / 16);
      if (ck) fprintf(stderr, "[ckpt save] conv l%d words %u src %zu@%p dst %zu@%p\n", l, words,
                      s.conv[size_t(conv_ordinal(l))].size, s.conv[size_t(conv_ordinal(l))].mtl, dst.size, dst.mtl);
      e.use(pipe("copy_u32"));
      e.buf(0, s.conv[size_t(conv_ordinal(l))]);
      e.buf(1, dst);
      e.val(2, words);
      e.dispatch((words + 255) / 256, 1, 1, 256);
    } else if (k == 'S') {
      for (int kv2 = 0; kv2 < 2; ++kv2) {
        const Buffer& src = kv2 ? s.vc[size_t(l)] : s.kc[size_t(l)];
        Buffer& dst = c.rec[2 * size_t(is) + size_t(kv2)];
        if (!src) continue;
        // the size check re-allocs on a stale view; the index must be the layer's own pair (is counts
        // layers: 2*is+kv2 walks 0,1,2,3,... — an is that counted kv2 iterations walked 0,2,4,... and
        // read/wrote past the vector, the heap corruption behind the intermittent save crash)
        if (!dst || dst.size != src.size) dst = g.alloc(src.size, "lfm3_ring_ckpt");
        const uint32_t words = uint32_t(src.size / 16);
        if (ck) fprintf(stderr, "[ckpt save] swa l%d kv%d slot@%p words %u src %zu@%p dst %zu@%p\n", l, kv2,
                        (void*)&dst, words, src.size, src.mtl, dst.size, dst.mtl);
        e.use(pipe("copy_u32"));
        e.buf(0, src);
        e.buf(1, dst);
        e.val(2, words);
        e.dispatch((words + 255) / 256, 1, 1, 256);
      }
      ++is;
    }
  }
  c.pos = pos;
}

void Lfm3::restore(Encoder& e, Seq& s, const Checkpoint& c) {
  size_t ic = 0, is = 0;
  for (int l = 0; l < cfg_.layers; ++l) {
    const char k = cfg_.kind[size_t(l)];
    if (k == 'C') {
      const Buffer& src = c.conv[ic++];
      Buffer& dst = s.conv[size_t(conv_ordinal(l))];
      if (!src || !dst) continue;
      const uint32_t words = uint32_t(std::min(src.size, dst.size) / 16);
      e.use(pipe("copy_u32"));
      e.buf(0, src);
      e.buf(1, dst);
      e.val(2, words);
      e.dispatch((words + 255) / 256, 1, 1, 256);
    } else if (k == 'S') {
      for (int kv2 = 0; kv2 < 2; ++kv2) {
        const Buffer& src = c.rec[2 * size_t(is) + size_t(kv2)];
        Buffer& dst = kv2 ? s.vc[size_t(l)] : s.kc[size_t(l)];
        if (!src || !dst) continue;
        const uint32_t words = uint32_t(std::min(src.size, dst.size) / 16);
        e.use(pipe("copy_u32"));
        e.buf(0, src);
        e.buf(1, dst);
        e.val(2, words);
        e.dispatch((words + 255) / 256, 1, 1, 256);
      }
      ++is;
    }
  }
}

void Lfm3::enc_sample(Encoder& e, Seq& s, const Buffer& xrow, int out_index, const SamplingParams& sp,
                      uint32_t step) {
  static Pipeline st1 = Gpu::get().pipeline("topk_stage1");
  static Pipeline st2 = Gpu::get().pipeline("sample_stage2");
  to_half(e, xrow, nullptr, xh_, 1);
  lin(e, embed_, xh_, logits_, 1, false, 0);  // the tied head
  enc_logit_mask(e, s, logits_, 1, cfg_.vocab, cfg_.vocab);
  // The request's repetition penalty (Splash's name, honoured here): scatter the HF penalty onto every context
  // token's logit [0, out_index) of s.tok, before the top-k. Skipped when it is 1.0, so the default pipeline and
  // its outputs stay bit-exact; a > 1.0 opt-in breaks degenerate repeats (a token that recurs is hit again).
  if (sp.repetition_penalty > 1.0f && out_index > 0) {
    const Lfm3RepPenParams rp{0, uint32_t(out_index), sp.repetition_penalty};
    e.use(pipe("lfm3_rep_penalty"));
    e.buf(0, logits_);
    e.buf(1, s.tok);
    e.val(2, rp);
    e.dispatch((uint32_t(out_index) + 255) / 256, 1, 1, 256);
  }
  const uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  const uint32_t V = uint32_t(cfg_.vocab), ntg = (V + 2047) / 2048;
  const kp::TopkStage ts{V, K, 2048};
  e.use(st1);
  e.buf(0, logits_);
  e.buf(1, cand_v_);
  e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, 1, 1, 256);
  const kp::Sample sp2{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, uint32_t(out_index)};
  e.use(st2);
  e.buf(0, cand_v_);
  e.buf(1, cand_i_);
  e.buf(2, s.tok);
  e.buf(3, sprob_);
  e.val(4, sp2);
  e.dispatch(1, 1, 1, 1024);
}

// One layer of one decode row (R = 1 for now; the verify rows come with the engine's spec-decode work).
void Lfm3::encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample) {
  const uint32_t H = uint32_t(cfg_.hidden), I = uint32_t(cfg_.ffn);
  const uint32_t QD = uint32_t(cfg_.hq) * uint32_t(cfg_.hd), KD = uint32_t(cfg_.hkv) * uint32_t(cfg_.hd);
  if (pos + 1 > s.cap && !reserve(s, pos + 1)) fatal("context full (%d)", s.cap);
  embed(e, s.tok, pos, 1, x_);
  // one decode row: a 4-bit group-32 linear runs qmvf on the fp32 x (the norm folded); an 8-bit one
  // keeps to_half + qmr8 (xh_ carries half(norm(x) * in_norm) for it)
  auto mv = [&](const Lfm3Lin& W, const Buffer& xfp, const Tensor* norm, const Buffer& y, uint32_t y_ld) {
    if (W.bits == 4 && W.gs == 32) enc_mv1(e, W, xfp, norm, y, 1, y_ld);
    else lin(e, W, xh_, y, 1, false, y_ld);
  };
  for (int l = 0; l < cfg_.layers; ++l) {
    const Lfm3Layer& L = layers_[size_t(l)];
    to_half(e, x_, L.in_norm, xh_, 1);  // xh = half(norm(x) * in_norm), for the 8-bit linears
    if (L.kind == 'C') {
      mv(L.in_proj, x_, L.in_norm, bcx_, 0);
      // u = B . x over the row's first and third thirds
      e.use(pipe("lfm3_mul"));
      e.buf(0, bcx_);
      e.buf(1, bcx_.view(size_t(2 * H) * 4, size_t(H) * 4));
      e.buf(2, u_);
      e.val(3, H);
      e.dispatch((H + 255) / 256, 1, 1, 256);
      const Lfm3ConvParams cp{1, H, uint32_t(cfg_.conv_k)};
      e.use(pipe("lfm3_conv_decode"));
      e.buf(0, u_);
      e.buf(1, L.conv_w->buf);
      e.buf(2, s.conv[size_t(conv_ordinal(l))]);
      e.buf(3, z_);
      e.buf(4, s.conv[size_t(conv_ordinal(l))]);
      e.val(5, cp);
      e.dispatch((H + 255) / 256, 1, 1, 256);
      e.use(pipe("lfm3_mul"));
      e.buf(0, bcx_.view(size_t(H) * 4, size_t(H) * 4));
      e.buf(1, z_);
      e.buf(2, ym_);
      e.val(3, H);
      e.dispatch((H + 255) / 256, 1, 1, 256);
      mv(L.out_proj, ym_, nullptr, mx_, 0);
    } else {
      const uint32_t row_ld = QD + 2 * KD;
      Buffer qv = qkv_, kv2 = qkv_.view(size_t(QD) * 4, qkv_.size - size_t(QD) * 4),
             vv = qkv_.view(size_t(QD + KD) * 4, qkv_.size - size_t(QD + KD) * 4);
      mv(L.q, x_, L.in_norm, qv, row_ld);
      mv(L.k, x_, L.in_norm, kv2, row_ld);
      mv(L.v, x_, L.in_norm, vv, row_ld);
      mv(L.gate, x_, L.in_norm, gate_, 0);
      const uint32_t cache_cap = L.kind == 'S' ? uint32_t(cfg_.window + 8 + 7) / 8 * 8 : uint32_t(s.cap);
      const Lfm3QkParams qp{1, uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.hd), cache_cap,
                            uint32_t(pos), uint32_t(pos + 1)};
      e.use(pipe("lfm3_qk_prep"));
      e.buf(0, qkv_);
      e.buf(1, L.qn->buf);
      e.buf(2, L.kn->buf);
      e.buf(3, L.kind == 'S' ? freqs_s_ : freqs_f_);
      e.buf(4, s.kc[size_t(l)]);
      e.buf(5, s.vc[size_t(l)]);
      e.buf(6, q_);
      e.val(7, qp);
      e.dispatch(uint32_t(cfg_.hq) + 2 * uint32_t(cfg_.hkv), 1, 1, 1);
      const Lfm3AttnSplitParams ap{uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.hd), cache_cap,
                                   uint32_t(pos), window_of(l), kAttnSplits, 1.f / std::sqrt(float(cfg_.hd))};
      e.use(pipe("lfm3_attn_d_split"));
      e.buf(0, q_);
      e.buf(1, s.kc[size_t(l)]);
      e.buf(2, s.vc[size_t(l)]);
      e.buf(3, apart_o_);
      e.buf(4, apart_m_);
      e.buf(5, apart_s_);
      e.val(6, ap);
      e.dispatch(uint32_t(cfg_.hq), kAttnSplits, 1, 1);
      e.use(pipe("lfm3_attn_d_merge"));
      e.buf(0, apart_o_);
      e.buf(1, apart_m_);
      e.buf(2, apart_s_);
      e.buf(3, gate_);
      e.buf(4, ao_);
      e.val(5, ap);
      e.dispatch(uint32_t(cfg_.hq), uint32_t(cfg_.hd), 1, 1);
      mv(L.o, ao_, nullptr, mx_, 0);
    }
    res_norm(e, mx_, L.post_norm, x_, x_, 1);
    // the FFN
    to_half(e, x_, L.ffn_norm, xh_, 1);
    mv(L.w1, x_, L.ffn_norm, g_, 0);
    mv(L.w3, x_, L.ffn_norm, u2_, 0);
    {
      const uint32_t n = I;
      e.use(pipe("swiglu_f32"));
      e.buf(0, g_);
      e.buf(1, u2_);
      e.buf(2, h_);
      e.val(3, n);
      e.dispatch((n + 255) / 256, 1, 1, 256);
    }
    mv(L.w2, h_, nullptr, mx_, 0);
    res_norm(e, mx_, L.post_ffn_norm, x_, x_, 1);
  }
  if (sample) {
    // the final norm before the tied head, as encode_prefill's last row does
    const kp::RowsNorm fn{H, cfg_.eps};
    e.use(pipe("rows_norm_f32"));
    e.buf(0, x_);
    e.buf(1, final_norm_->buf);
    e.buf(2, xn_);
    e.val(3, fn);
    e.dispatch(1, 1, 1, 256);
    enc_sample(e, s, xn_, pos + 1, sp, step);
  }
}

// T rows at pos0 .. pos0 + T - 1 (a chunk of at most tmax_ rows); the conv rings roll to the chunk's end,
// the attention caches take the rows' keys and values, and the last row samples.
void Lfm3::encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                          Checkpoint* snap, int snap_row) {
  (void)snap, (void)snap_row;  // can_snapshot() is false: no mid-chunk snapshots
  const uint32_t H = uint32_t(cfg_.hidden), I = uint32_t(cfg_.ffn);
  const uint32_t QD = uint32_t(cfg_.hq) * uint32_t(cfg_.hd), KD = uint32_t(cfg_.hkv) * uint32_t(cfg_.hd);
  if (pos0 + T > s.cap && !reserve(s, pos0 + T)) fatal("context full (%d)", s.cap);
  embed(e, s.tok, pos0, T, px_);
  if (s.embeds) vision::enc_embed_runs(e, *s.embeds, pos0, T, px_, int(H));  // image rows (a prompt's prefill)
  // LFM3_X_PROBE: a per-layer copy of the residual stream's last row, read back after the commit
  // (xprobe_dump); the layerwise bisect for the run-to-run logit wobble
  static const bool xprobe = getenv("LFM3_X_PROBE") && atoi(getenv("LFM3_X_PROBE"));
  static std::vector<Buffer>& xp = xprobe_state();
  if (xprobe && xp.empty()) {
    Gpu& gg = Gpu::get();
    xp.assign(size_t(cfg_.layers) + 1, Buffer{});
    for (auto& b : xp) b = gg.alloc(size_t(H) * 4, "lfm3_xprobe");
  }
  const auto xcopy = [&](int slot) {
    e.use(pipe("copy_u32"));
    e.buf(0, px_.view(size_t(T - 1) * H * 4, size_t(H) * 4));
    e.buf(1, xp[size_t(slot)]);
    e.val(2, uint32_t(H) / 4);
    e.dispatch((H / 4 + 255) / 256, 1, 1, 256);
  };
  if (xprobe) xcopy(0);
  for (int l = 0; l < cfg_.layers; ++l) {
    const Lfm3Layer& L = layers_[size_t(l)];
    to_half(e, px_, L.in_norm, pxh_, T);
    if (L.kind == 'C') {
      lin(e, L.in_proj, pxh_, pbcx_, T, false, 0);
      const Lfm3BcxParams bp{uint32_t(T), H, 0};  // u = B ⊙ x from the row's B and x planes
      e.use(pipe("lfm3_bcx"));
      e.buf(0, pbcx_);
      e.buf(1, pbcx_);
      e.buf(2, pu_);
      e.val(3, bp);
      e.dispatch((H + 255) / 256, uint32_t(T), 1, 256);
      const Lfm3ConvParams cp{uint32_t(T), H, uint32_t(cfg_.conv_k)};
      e.use(pipe("lfm3_conv_prefill"));
      e.buf(0, pu_);
      e.buf(1, L.conv_w->buf);
      e.buf(2, s.conv[size_t(conv_ordinal(l))]);
      e.buf(3, pz_);
      e.val(4, cp);
      e.dispatch((H + 255) / 256, uint32_t(T), 1, 256);
      e.use(pipe("lfm3_ring_roll"));
      e.buf(0, pu_);
      e.buf(1, s.conv[size_t(conv_ordinal(l))]);
      e.buf(2, s.conv[size_t(conv_ordinal(l))]);
      e.val(3, cp);
      e.dispatch(H, 1, 1, 1);
      const Lfm3BcxParams bp2{uint32_t(T), H, 1};  // y = C ⊙ z, C from the row's middle plane
      e.use(pipe("lfm3_bcx"));
      e.buf(0, pbcx_);
      e.buf(1, pz_);
      e.buf(2, py_);
      e.val(3, bp2);
      e.dispatch((H + 255) / 256, uint32_t(T), 1, 256);
      to_half(e, py_, nullptr, paoh_, T);
      lin(e, L.out_proj, paoh_, pmix_, T, false, 0);
    } else {
      const uint32_t row_ld = QD + 2 * KD;
      Buffer qv = pqkv_, kv2 = pqkv_.view(size_t(QD) * 4, pqkv_.size - size_t(QD) * 4),
             vv = pqkv_.view(size_t(QD + KD) * 4, pqkv_.size - size_t(QD + KD) * 4);
      lin(e, L.q, pxh_, qv, T, false, row_ld);
      lin(e, L.k, pxh_, kv2, T, false, row_ld);
      lin(e, L.v, pxh_, vv, T, false, row_ld);
      lin(e, L.gate, pxh_, pgate_, T, false, 0);
      const uint32_t cache_cap = L.kind == 'S' ? uint32_t(cfg_.window + 8 + 7) / 8 * 8 : uint32_t(s.cap);
      const Lfm3QkParams qp{uint32_t(T), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.hd), cache_cap,
                            uint32_t(pos0), uint32_t(pos0 + T)};
      e.use(pipe("lfm3_qk_prep"));
      e.buf(0, pqkv_);
      e.buf(1, L.qn->buf);
      e.buf(2, L.kn->buf);
      e.buf(3, L.kind == 'S' ? freqs_s_ : freqs_f_);
      e.buf(4, s.kc[size_t(l)]);
      e.buf(5, s.vc[size_t(l)]);
      e.buf(6, pq_);
      e.val(7, qp);
      e.dispatch(uint32_t(cfg_.hq) + 2 * uint32_t(cfg_.hkv), uint32_t(T), 1, 1);
      const Lfm3AttnPSplitParams aps{uint32_t(T),       uint32_t(cfg_.hq), uint32_t(cfg_.hkv),
                                     uint32_t(cfg_.hd), cache_cap,         uint32_t(pos0),
                                     window_of(l),      kPSplits,          1.f / std::sqrt(float(cfg_.hd))};
      e.use(pipe("lfm3_attn_p_split"));
      e.buf(0, pq_);
      e.buf(1, s.kc[size_t(l)]);
      e.buf(2, s.vc[size_t(l)]);
      e.buf(3, ppart_o_);
      e.buf(4, ppart_m_);
      e.buf(5, ppart_s_);
      e.val(6, aps);
      e.dispatch(uint32_t(cfg_.hq) * kPSplits, uint32_t(T), 1, 64);
      e.use(pipe("lfm3_attn_p_merge"));
      e.buf(0, ppart_o_);
      e.buf(1, ppart_m_);
      e.buf(2, ppart_s_);
      e.buf(3, pgate_);
      e.buf(4, pao_);
      e.val(5, aps);
      e.dispatch(uint32_t(cfg_.hq) * uint32_t(cfg_.hd), uint32_t(T), 1, 64);
      to_half(e, pao_, nullptr, paoh_, T);
      lin(e, L.o, paoh_, pmix_, T, false, 0);
    }
    res_norm(e, pmix_, L.post_norm, px_, px_, T);
    to_half(e, px_, L.ffn_norm, pxh_, T);
    lin(e, L.w1, pxh_, pg_, T, false, 0);
    lin(e, L.w3, pxh_, pu2_, T, false, 0);
    {
      const uint32_t n = uint32_t(T) * I;
      e.use(pipe("swiglu_f32"));
      e.buf(0, pg_);
      e.buf(1, pu2_);
      e.buf(2, ph_);
      e.val(3, n);
      e.dispatch((n + 255) / 256, 1, 1, 256);
    }
    to_half(e, ph_, nullptr, phh_, T, I);
    lin(e, L.w2, phh_, pmix_, T, false, 0);
    res_norm(e, pmix_, L.post_ffn_norm, px_, px_, T);
    if (xprobe) xcopy(l + 1);
  }
  if (sample) {
    // the chunk's last row through the final norm and the tied head
    const kp::RowsNorm fn{H, cfg_.eps};
    e.use(pipe("rows_norm_f32"));
    e.buf(0, px_.view(size_t(T - 1) * H * 4, size_t(H) * 4));
    e.buf(1, final_norm_->buf);
    e.buf(2, xn_);
    e.val(3, fn);
    e.dispatch(1, 1, 1, 256);
    enc_sample(e, s, xn_, pos0 + T, sp, step);
  }
}

}  // namespace ni

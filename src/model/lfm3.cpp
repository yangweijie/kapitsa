#include "model/lfm3.h"

#include "util/json_file.h"
#include "util/log.h"
#include "model/lfm3_config.h"

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

}  // namespace ni

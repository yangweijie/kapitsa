#include "model/lfm3_config.h"

#include <algorithm>
#include <cstdio>
#include <set>

namespace ni {

using json = nlohmann::json;

namespace {

const json& text_of(const json& c) {
  return c.contains("text_config") && c["text_config"].is_object() ? c["text_config"] : c;
}

std::string type_of(const json& c) {
  return c.is_object() && c.contains("model_type") && c["model_type"].is_string() ? c["model_type"].get<std::string>()
                                                                                  : std::string();
}

// a number setting, `def` when it is absent or null
long long num(const json& t, const char* k, long long def) {
  return t.contains(k) && t[k].is_number() ? t[k].get<long long>() : def;
}

std::string str(const json& t, const char* k) {
  return t.contains(k) && t[k].is_string() ? t[k].get<std::string>() : std::string();
}

bool present(const json& t, const char* k) { return t.contains(k) && !t[k].is_null(); }

std::string fmt(const char* f, long long a) {
  char b[160];
  snprintf(b, sizeof b, f, a);
  return b;
}

bool attention_type(const std::string& s) { return s == "full_attention" || s == "sliding_attention"; }

// the rope_parameters entry of one attention layer type, {} when absent or not an object
json rope_of(const json& t, const char* type) {
  const json rp = t.contains("rope_parameters") && t["rope_parameters"].is_object() ? t["rope_parameters"]
                                                                                    : json::object();
  return rp.contains(type) && rp[type].is_object() ? rp[type] : json::object();
}

// the partial rotation of one attention layer type, 1 when the entry does not say (Lfm3Config's
// default for sliding_attention)
double partial_of(const json& rp) {
  return rp.contains("partial_rotary_factor") && rp["partial_rotary_factor"].is_number()
             ? rp["partial_rotary_factor"].get<double>()
             : 1.0;
}

}  // namespace

bool lfm3_family(const json& c) {
  if (!c.is_object()) return false;
  return type_of(c).rfind("lfm3", 0) == 0 || type_of(text_of(c)).rfind("lfm3", 0) == 0;
}

std::vector<std::string> lfm3_layer_types(const json& c) {
  if (!c.is_object()) return {};
  const json& t = text_of(c);
  const long long layers = num(t, "num_hidden_layers", 24);
  std::vector<std::string> v;
  if (t.contains("layer_types") && t["layer_types"].is_array()) {
    for (const json& x : t["layer_types"]) {
      if (!x.is_string() || (!attention_type(x.get<std::string>()) && x.get<std::string>() != "conv")) return {};
      v.push_back(x.get<std::string>());
    }
    if ((long long)v.size() != layers) return {};
  } else {
    // Lfm3Config's default pattern
    for (long long i = 0; i < layers; ++i)
      v.push_back(i % 4 == 2 ? "full_attention" : i % 4 == 0 && i > 0 ? "sliding_attention" : "conv");
  }
  return v;
}

std::vector<std::string> lfm3_unsupported(const json& c) {
  std::vector<std::string> v;
  if (!c.is_object()) return {"config.json is not a JSON object"};
  const json& t = text_of(c);
  const std::string mt = type_of(c), tt = type_of(t);
  const std::string type = !mt.empty() ? mt : tt;
  if (type != "lfm3")
    v.push_back("model_type " + (type.empty() ? std::string("(none)") : type) +
                ": Kapitsa runs LFM3, lfm3 (a new LFM3 architecture runs once its differences are read)");
  // what model/lfm3.h's load() and its kernels (lfm3.metal) take
  const long long hidden = num(t, "hidden_size", 0);
  if (hidden <= 0) v.push_back(fmt("hidden_size %lld: a positive hidden size", hidden));
  if (const long long inter = num(t, "intermediate_size", 0); inter <= 0)
    v.push_back(fmt("intermediate_size %lld: a positive feed-forward width", inter));
  const long long hq = num(t, "num_attention_heads", 0);
  const long long hd = num(t, "head_dim", hq > 0 ? hidden / hq : 0);
  if (hd != 128) v.push_back(fmt("head_dim %lld: the attention kernels take 128", hd));
  if (hq <= 0) {
    v.push_back("no num_attention_heads");
  } else {
    const long long hkv = num(t, "num_key_value_heads", 0);
    if (hkv <= 0 || hq % hkv)
      v.push_back("num_key_value_heads " + std::to_string(hkv) + ": a divisor of the " + std::to_string(hq) +
                  " query heads");
    else if (hq / hkv > 16) v.push_back(fmt("num_key_value_heads %lld: at most 16 query heads to a key head", hkv));
  }
  if (const long long nl = num(t, "num_hidden_layers", 0); nl <= 0)
    v.push_back(fmt("num_hidden_layers %lld: at least 1", nl));
  if (const long long vs = num(t, "vocab_size", 0); vs <= 0) v.push_back(fmt("vocab_size %lld: a positive vocabulary", vs));
  // the conv layers
  if (const long long k = num(t, "conv_L_cache", 7); k != 7) v.push_back(fmt("conv_L_cache %lld: 7 (the conv kernels)", k));
  if (t.contains("conv_bias") && t["conv_bias"].is_boolean() && t["conv_bias"].get<bool>())
    v.push_back("conv_bias true: the conv layers run a bias-free projection");
  // the layer schedule
  const bool schedule_given = t.contains("layer_types") && t["layer_types"].is_array();
  if (t.contains("layer_types") && !schedule_given) v.push_back("layer_types is not an array");
  const std::vector<std::string> lt = schedule_given || !t.contains("layer_types") ? lfm3_layer_types(c)
                                                                                   : std::vector<std::string>{};
  if (schedule_given && lt.empty() && num(t, "num_hidden_layers", 0) > 0) {
    bool bad = false;
    for (const json& x : t["layer_types"]) {
      const std::string s = x.is_string() ? x.get<std::string>() : x.dump();
      if (!attention_type(s) && s != "conv") {
        v.push_back("layer_types has " + s + " layers");
        bad = true;
        break;
      }
    }
    if (!bad)
      v.push_back(fmt("layer_types names %lld layers, not num_hidden_layers", (long long)t["layer_types"].size()));
  }
  if (!lt.empty()) {
    if (!std::any_of(lt.begin(), lt.end(), attention_type))
      v.push_back("no attention layer: the tri-hybrid runs attention and conv layers");
    if (std::find(lt.begin(), lt.end(), "sliding_attention") != lt.end()) {
      const long long sw = num(t, "sliding_window", 0);
      if (sw <= 0) v.push_back(fmt("sliding_window %lld: a positive window for the sliding_attention layers", sw));
    }
  }
  // RoPE per attention layer type (Lfm3Config::validate_architecture): an entry for each attention
  // type in the schedule, a factor that is a fraction of the head, and no recompute that drops the
  // zero tail
  const json rp = t.contains("rope_parameters") && t["rope_parameters"].is_object() ? t["rope_parameters"]
                                                                                    : json::object();
  for (const char* at : {"full_attention", "sliding_attention"}) {
    if (std::find(lt.begin(), lt.end(), at) == lt.end()) continue;
    const json r = rope_of(t, at);
    if (r.empty()) {
      v.push_back(std::string("rope_parameters has no entry for the ") + at + " layers");
      continue;
    }
    const std::string rt = str(r, "rope_type");
    const double prf = partial_of(r);
    if ((rt == "dynamic" || rt == "longrope") && prf < 1.0)
      v.push_back(std::string("rope_type ") + rt + " with partial rotation: the recompute over the rotated dims only "
                                                     "drops LFM3's zero tail");
    char b[48];
    snprintf(b, sizeof b, "%.6g", prf);
    if (prf <= 0.0 || prf > 1.0)
      v.push_back(std::string(at) + " partial_rotary_factor " + b + ": a fraction of the head, above 0");
  }
  // what the loader reads
  if (t.contains("use_sandwich_norm") && t["use_sandwich_norm"].is_boolean() && !t["use_sandwich_norm"].get<bool>())
    v.push_back("use_sandwich_norm false: the loader reads LFM3's post-norms");
  if (t.contains("tie_word_embeddings") && t["tie_word_embeddings"].is_boolean() &&
      !t["tie_word_embeddings"].get<bool>())
    v.push_back("tie_word_embeddings false: the loader reads LFM3's tied lm_head");
  if (t.contains("vision_config") || c.contains("vision_config"))
    v.push_back("vision_config: Kapitsa runs the LFM3 text model");
  return v;
}

std::vector<std::string> lfm3_unapplied(const json& c) {
  // Lfm3Config (configuration_lfm3.py), Transformers' PreTrainedConfig and what converters add; the
  // top level's of a config without a text_config
  static const std::set<std::string> kKnown = {
      "_name_or_path", "architectures", "attention_bias", "attention_dropout", "auto_map", "bos_token_id",
      "conv_bias", "conv_L_cache", "dtype", "eos_token_id", "head_dim", "hidden_size", "id2label",
      "initializer_range", "intermediate_size", "is_encoder_decoder", "label2id", "layer_types",
      "max_position_embeddings", "model_type", "norm_eps", "num_attention_heads", "num_hidden_layers",
      "num_key_value_heads", "output_attentions", "output_hidden_states", "pad_token_id", "problem_type",
      "quantization", "quantization_config", "return_dict", "rope_parameters", "rope_scaling", "rope_theta",
      "sliding_window", "tie_embedding", "tie_word_embeddings", "torch_dtype", "transformers_version",
      "use_cache", "use_sandwich_norm", "vocab_size"};
  std::vector<std::string> v;
  if (!c.is_object()) return v;
  const json& t = text_of(c);
  if (!t.is_object()) return v;
  const json rp = t.contains("rope_parameters") && t["rope_parameters"].is_object() ? t["rope_parameters"]
                                                                                    : json::object();
  std::set<std::string> scaling;
  for (auto it = rp.begin(); it != rp.end(); ++it) {
    if (!it.value().is_object()) continue;
    const std::string rt = str(it.value(), "rope_type");
    if (rt.empty() || rt == "default") continue;
    if (rt == "dynamic" || rt == "longrope") {
      if (partial_of(it.value()) >= 1.0) scaling.insert(rt);  // a full-dim recompute keeps the path's RoPE
    } else {
      scaling.insert(rt);
    }
  }
  for (const std::string& s : scaling) v.push_back("rope_type " + s + " (RoPE runs unscaled)");
  if (present(t, "rope_scaling")) v.push_back("rope_scaling (RoPE runs unscaled)");
  for (auto it = rp.begin(); it != rp.end(); ++it)
    if (it.key() != "full_attention" && it.key() != "sliding_attention")
      v.push_back("rope_parameters has a " + it.key() + " entry (conv layers use no RoPE)");
  std::vector<std::string> unknown;
  for (auto it = t.begin(); it != t.end(); ++it)
    if (!kKnown.count(it.key())) unknown.push_back(it.key());
  std::sort(unknown.begin(), unknown.end());
  if (!unknown.empty()) {
    std::string names;
    for (const std::string& n : unknown) names += (names.empty() ? "" : ", ") + n;
    v.push_back("settings LFM3 does not have: " + names);
  }
  return v;
}

std::string lfm3_refusal(const json& c, const std::string& what) {
  const std::vector<std::string> v = lfm3_unsupported(c);
  if (v.empty()) return "";
  const json& t = text_of(c);
  std::string type = type_of(c);
  if (type.empty()) type = type_of(t);
  std::string m = what + " is a " + type +
                  (type == "lfm3" ? " model with settings Kapitsa's LFM3 path does not run: "
                                  : " model (LFM3), which Kapitsa does not run yet: ");
  for (size_t i = 0; i < v.size(); ++i) m += (i ? "; " : "") + v[i];
  m += ".";
  const std::vector<std::string> u = lfm3_unapplied(c);
  if (!u.empty()) {
    m += " Not applied either: ";
    for (size_t i = 0; i < u.size(); ++i) m += (i ? "; " : "") + u[i];
    m += ".";
  }
  return m;
}

}  // namespace ni

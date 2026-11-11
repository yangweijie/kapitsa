// What an LFM3-family config.json asks that Kapitsa's LFM3 path does not run (src/model/lfm3_config.h), on
// LiquidAI/LFM3-1.7B's published config: it runs as it is, its layer schedule is Lfm3Config's default pattern, and
// each setting the path does not run is refused alone. CPU only, no model.
#include "model/lfm3_config.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

int g_fail = 0, g_checks = 0;

void ok(const std::string& name, bool cond, const std::string& detail = "") {
  ++g_checks;
  if (cond) {
    printf("ok %s\n", name.c_str());
  } else {
    ++g_fail;
    fprintf(stderr, "FAIL %s%s%s\n", name.c_str(), detail.empty() ? "" : ": ", detail.c_str());
  }
}

// LiquidAI/LFM3-1.7B's config.json (2026-11-10)
json lfm3_1_7b() {
  return json::parse(R"({"architectures": ["Lfm3ForCausalLM"],
  "auto_map": {"AutoConfig": "configuration_lfm3.Lfm3Config", "AutoModel": "modeling_lfm3.Lfm3Model",
    "AutoModelForCausalLM": "modeling_lfm3.Lfm3ForCausalLM"},
  "bos_token_id": 124894, "conv_L_cache": 7, "conv_bias": false, "dtype": "bfloat16", "eos_token_id": 124900,
  "head_dim": 128, "hidden_size": 2048, "intermediate_size": 7680,
  "layer_types": ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
    "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
    "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv"],
  "max_position_embeddings": 262144, "model_type": "lfm3", "norm_eps": 1e-05, "num_attention_heads": 16,
  "num_hidden_layers": 24, "num_key_value_heads": 4, "pad_token_id": 124893,
  "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default"},
    "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}},
  "sliding_window": 256, "tie_word_embeddings": true, "transformers_version": "5.12.1", "use_cache": true,
  "use_sandwich_norm": true, "vocab_size": 128000})");
}

std::string joined(const std::vector<std::string>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? "; " : "") + v[i];
  return s;
}

}  // namespace

int main() {
  const json cfg = lfm3_1_7b();

  // the published config runs as it is
  ok("family: the published config is LFM3", ni::lfm3_family(cfg));
  ok("published config runs", ni::lfm3_refusal(cfg, "x").empty(), ni::lfm3_refusal(cfg, "x"));
  ok("published config: nothing unapplied", ni::lfm3_unapplied(cfg).empty(), joined(ni::lfm3_unapplied(cfg)));

  // the layer schedule: the config's, and Lfm3Config's default pattern is the same 24
  const std::vector<std::string> published = ni::lfm3_layer_types(cfg);
  ok("published schedule: 24 layers", published.size() == 24, std::to_string(published.size()));
  long long nconv = 0, nfull = 0, nswa = 0;
  for (const std::string& s : published) nconv += s == "conv", nfull += s == "full_attention", nswa += s == "sliding_attention";
  ok("published schedule: 13 conv, 6 full, 5 sliding", nconv == 13 && nfull == 6 && nswa == 5,
     "conv " + std::to_string(nconv) + ", full " + std::to_string(nfull) + ", sliding " + std::to_string(nswa));
  ok("published schedule: two convs, then full, conv, sliding", published[0] == "conv" && published[1] == "conv" &&
                                                                       published[2] == "full_attention" &&
                                                                       published[3] == "conv" &&
                                                                       published[4] == "sliding_attention");
  json flat = cfg;
  flat.erase("layer_types");
  ok("the default pattern is the published schedule", ni::lfm3_layer_types(flat) == published);

  // a text_config of LFM3 runs the same (a VL pack's text tower is refused for its vision_config)
  json vl = {{"architectures", {"Lfm3ForConditionalGeneration"}}, {"model_type", "lfm3"},
             {"vision_config", json::object()}, {"text_config", cfg}};
  const std::vector<std::string> vv = ni::lfm3_unsupported(vl);
  ok("a vision_config is refused for the tower alone",
     vv.size() == 1 && vv[0] == "vision_config: Kapitsa runs the LFM3 text model", joined(vv));
  json text_only = vl;
  text_only.erase("vision_config");
  ok("a text_config of LFM3 runs", ni::lfm3_refusal(text_only, "x").empty(), ni::lfm3_refusal(text_only, "x"));

  // the family: lfm3* names are LFM3, a longer one refused for its type alone; others are not
  for (const char* mt : {"lfm2", "lfm2_5", "lfm", "gemma4", ""})
    ok(std::string("'") + mt + "' is not LFM3", !ni::lfm3_family(json{{"model_type", mt}}));
  ok("a non-object is not LFM3", !ni::lfm3_family(json::array()));
  json moe = cfg;
  moe["model_type"] = "lfm3_moe";
  ok("lfm3_moe is LFM3-family", ni::lfm3_family(moe));
  {
    const std::vector<std::string> mv = ni::lfm3_unsupported(moe);
    ok("lfm3_moe refused for its type alone", mv.size() == 1 && mv[0].rfind("model_type lfm3_moe", 0) == 0,
       joined(mv));
    const std::string mr = ni::lfm3_refusal(moe, "x");
    ok("lfm3_moe's refusal names a new LFM3 architecture", mr.rfind("x is a lfm3_moe model (LFM3)") == 0, mr);
  }
  ok("a bare lfm3_moe config is still refused", !ni::lfm3_unsupported(json{{"model_type", "lfm3_moe"}}).empty());

  // each setting the path does not run, alone on the published config: exactly one refusal line
  const std::vector<std::pair<const char*, std::function<void(json&)>>> one = {
      {"head_dim 64", [](json& t) { t["head_dim"] = 64; }},
      {"head_dim 256", [](json& t) { t["head_dim"] = 256; }},
      {"no num_attention_heads", [](json& t) { t.erase("num_attention_heads"); }},
      {"num_key_value_heads 3", [](json& t) { t["num_key_value_heads"] = 3; }},
      {"num_key_value_heads 1", [](json& t) { t["num_key_value_heads"] = 1, t["num_attention_heads"] = 48; }},
      {"hidden_size 0", [](json& t) { t["hidden_size"] = 0; }},
      {"intermediate_size 0", [](json& t) { t["intermediate_size"] = 0; }},
      {"num_hidden_layers 0", [](json& t) { t["num_hidden_layers"] = 0; }},
      {"vocab_size 0", [](json& t) { t["vocab_size"] = 0; }},
      {"conv_L_cache 4", [](json& t) { t["conv_L_cache"] = 4; }},
      {"conv_bias true", [](json& t) { t["conv_bias"] = true; }},
      {"layer_types has linear_attention", [](json& t) { t["layer_types"][3] = "linear_attention"; }},
      {"layer_types names 23 layers, not num_hidden_layers", [](json& t) { t["layer_types"].erase(23); }},
      {"layer_types is not an array", [](json& t) { t["layer_types"] = "conv"; }},
      {"sliding_window 0", [](json& t) { t["sliding_window"] = 0; }},
      {"sliding_window 0", [](json& t) { t.erase("sliding_window"); }},
      {"rope_parameters has no entry for the full_attention layers",
       [](json& t) { t["rope_parameters"].erase("full_attention"); }},
      {"rope_parameters has no entry for the sliding_attention layers",
       [](json& t) { t["rope_parameters"].erase("sliding_attention"); }},
      {"full_attention partial_rotary_factor 0", [](json& t) { t["rope_parameters"]["full_attention"]["partial_rotary_factor"] = 0; }},
      {"full_attention partial_rotary_factor 1.5",
       [](json& t) { t["rope_parameters"]["full_attention"]["partial_rotary_factor"] = 1.5; }},
      {"rope_type dynamic with partial rotation",
       [](json& t) { t["rope_parameters"]["full_attention"]["rope_type"] = "dynamic"; }},
      {"use_sandwich_norm false", [](json& t) { t["use_sandwich_norm"] = false; }},
      {"tie_word_embeddings false", [](json& t) { t["tie_word_embeddings"] = false; }},
      {"vision_config: Kapitsa runs the LFM3 text model", [](json& t) { t["vision_config"] = json::object(); }},
  };
  for (const auto& [what, edit] : one) {
    json c = cfg;
    edit(c);
    const std::vector<std::string> v = ni::lfm3_unsupported(c);
    ok(std::string("refused alone: ") + what, v.size() == 1 && v[0].rfind(what, 0) == 0,
       std::to_string(v.size()) + " lines: " + joined(v));
  }

  // a GQA ratio of exactly 16 runs (the boundary qwen4_config also takes)
  json wide = cfg;
  wide["num_attention_heads"] = 32;
  wide["num_key_value_heads"] = 2;
  ok("16 query heads to a key head runs", ni::lfm3_refusal(wide, "x").empty(), ni::lfm3_refusal(wide, "x"));

  // the default schedule has no sliding_attention-free corner to skip the window check: the pattern
  // always carries both attention types
  json nosw = flat;  // no layer_types: the default pattern
  nosw.erase("sliding_window");
  {
    const std::vector<std::string> nv = ni::lfm3_unsupported(nosw);
    ok("the default schedule checks the window too",
       nv.size() == 1 && nv[0].rfind("sliding_window 0", 0) == 0, joined(nv));
  }

  // an all-conv schedule is refused
  json allconv = cfg;
  for (json& x : allconv["layer_types"]) x = "conv";
  {
    const std::vector<std::string> av = ni::lfm3_unsupported(allconv);
    ok("an all-conv schedule is refused", av.size() == 1 && av[0].rfind("no attention layer", 0) == 0, joined(av));
  }

  // what the path does not apply, one line each: RoPE scaling runs unscaled, unknown keys named
  {
    json yarn = cfg;
    yarn["rope_parameters"]["sliding_attention"].update({{"rope_type", "yarn"}, {"factor", 4.0},
                                                         {"original_max_position_embeddings", 262144}});
    yarn["max_position_embeddings"] = 1048576;
    ok("Sushi-style yarn runs", ni::lfm3_refusal(yarn, "x").empty(), ni::lfm3_refusal(yarn, "x"));
    const std::vector<std::string> yu = ni::lfm3_unapplied(yarn);
    ok("yarn is one unapplied line", yu.size() == 1 && yu[0] == "rope_type yarn (RoPE runs unscaled)", joined(yu));

    json dyn = cfg;
    dyn["rope_parameters"]["sliding_attention"]["rope_type"] = "dynamic";
    ok("a full-dim dynamic recompute runs", ni::lfm3_refusal(dyn, "x").empty(), ni::lfm3_refusal(dyn, "x"));
    const std::vector<std::string> du = ni::lfm3_unapplied(dyn);
    ok("full-dim dynamic is one unapplied line", du.size() == 1 && du[0] == "rope_type dynamic (RoPE runs unscaled)",
       joined(du));

    json scal = cfg;
    scal["rope_scaling"] = json{{"rope_type", "linear"}, {"factor", 2.0}};
    const std::vector<std::string> su = ni::lfm3_unapplied(scal);
    ok("rope_scaling is one unapplied line", su.size() == 1 && su[0] == "rope_scaling (RoPE runs unscaled)",
       joined(su));

    json convrope = cfg;
    convrope["rope_parameters"]["conv"] = json{{"rope_theta", 10000}};
    {
      const std::vector<std::string> cu = ni::lfm3_unapplied(convrope);
      ok("a conv RoPE entry is named, not applied",
         cu.size() == 1 && cu[0] == "rope_parameters has a conv entry (conv layers use no RoPE)", joined(cu));
    }

    json unk = cfg;
    unk["gr_dtype"] = "fp8";
    {
      const std::vector<std::string> uu = ni::lfm3_unapplied(unk);
      ok("an unknown key is named", uu.size() == 1 && uu[0] == "settings LFM3 does not have: gr_dtype", joined(uu));
    }

    // the refusal carries the unapplied lines after the unsupported ones
    json both = yarn;
    both["head_dim"] = 64;
    const std::string br = ni::lfm3_refusal(both, "repo/model");
    ok("the refusal names the repo, the setting and the unapplied",
       br.rfind("repo/model is a lfm3 model with settings Kapitsa's LFM3 path does not run: head_dim 64", 0) == 0 &&
           br.find("Not applied either: rope_type yarn") != std::string::npos,
       br);
  }

  printf("%s (%d checks, %d failed)\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
  return g_fail ? 1 : 0;
}

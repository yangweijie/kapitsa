// Liquid AI's LFM3 config.json files (a model_type lfm3*) against what Kapitsa's LFM3 path runs
// (model/lfm3.h: the tri-hybrid of modeling_lfm3.py — gated short convolutions, full attention and
// sliding attention, with LFM3's zero-tail partial RoPE and sandwich norms). Read from the config
// alone, without the weights or the GPU, so the Hub check refuses before a download what the loader
// would refuse after it, and an LFM3 release that changes a setting this path reads, or adds one it
// does not, says which.
#pragma once
#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace ni {

// config.json's model_type, or its text_config's, names the LFM3 family: lfm3 (the path runs it) or
// a longer lfm3* name (a new LFM3 architecture, which runs only once its differences are read)
bool lfm3_family(const nlohmann::json& config);

// What in an LFM3-family config the path does not run, one line each; empty: it runs. A model_type
// other than lfm3 is such a line, so a new LFM3 architecture runs only once its differences have
// been read.
std::vector<std::string> lfm3_unsupported(const nlohmann::json& config);

// What in an LFM3-family config the path does not apply, one line each: RoPE scaling (RoPE runs
// unscaled) and the settings neither Lfm3Config (configuration_lfm3.py) nor the path knows. An LFM3
// model loads with them and says so.
std::vector<std::string> lfm3_unapplied(const nlohmann::json& config);

// The refusal of an LFM3-family model whose lfm3_unsupported is not empty, for the engine and the
// Hub check: what it is (`what`: a folder or a repo), each line, then its lfm3_unapplied. "" when it
// runs.
std::string lfm3_refusal(const nlohmann::json& config, const std::string& what);

// The per-layer operator types: the config's layer_types, or Lfm3Config's default pattern when
// absent (a full-attention layer every fourth layer counting from the third, a sliding-attention
// layer in between, convolutions elsewhere). {} when the names are not the path's three layer types
// or do not match num_hidden_layers (lfm3_unsupported says why).
std::vector<std::string> lfm3_layer_types(const nlohmann::json& config);

}  // namespace ni

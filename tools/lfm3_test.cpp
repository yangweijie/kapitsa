// The LFM3 backend's CPU core (src/model/lfm3.h): read_config against the published LFM3-1.7B config (and its
// default schedule), the memory plan against safetensors headers written here, and the cache sizing's exact math.
// The encode paths and their Metal kernels are a later, GPU-verified change; nothing here touches a GPU.
//   lfm3_test   exit status 1 if any check fails
#include "model/lfm3.h"

#include <nlohmann/json.hpp>

#include <unistd.h>

#include "engine/engine.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

int g_checks = 0, g_fail = 0;

void ok(const std::string& name, bool cond, const std::string& detail = "") {
  ++g_checks;
  if (cond) {
    printf("ok %s\n", name.c_str());
  } else {
    ++g_fail;
    fprintf(stderr, "FAIL %s%s%s\n", name.c_str(), detail.empty() ? "" : ": ", detail.c_str());
  }
}

// LiquidAI/LFM3-1.7B's config.json (the fields the config and the plan read)
json lfm3_1_7b() {
  return json::parse(R"({"architectures": ["Lfm3ForCausalLM"], "bos_token_id": 124894, "conv_L_cache": 7,
  "conv_bias": false, "dtype": "bfloat16", "eos_token_id": 124900, "head_dim": 128, "hidden_size": 2048,
  "intermediate_size": 7680,
  "layer_types": ["conv", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
    "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
    "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv"],
  "max_position_embeddings": 262144, "model_type": "lfm3", "norm_eps": 1e-05, "num_attention_heads": 16,
  "num_hidden_layers": 24, "num_key_value_heads": 4, "pad_token_id": 124893,
  "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default"},
    "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}},
  "sliding_window": 256, "tie_word_embeddings": true, "use_sandwich_norm": true, "vocab_size": 128000})");
}

void write_text(const std::string& p, const std::string& s) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f << s;
}

// A safetensors file of zero bytes' tensors: a u64 header length, the JSON header, the padding, the data
void write_st(const std::string& p, const json& tensors) {
  json h = {{"__metadata__", {{"format", "mlx"}}}};
  uint64_t off = 0;
  json body = json::object();
  uint64_t total = 0;
  for (const auto& [name, t] : tensors.items()) {
    const uint64_t bytes = t.value("bytes", uint64_t(0));
    body[name] = {{"dtype", t.value("dtype", "BF16")}, {"shape", t["shape"]},
                  {"data_offsets", {off, off + bytes}}};
    off += bytes;
    total += bytes;
  }
  h.update(body);
  std::string hs = h.dump();
  hs.append((8 - hs.size() % 8) % 8, ' ');
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  const uint64_t hl = hs.size();
  f.write(reinterpret_cast<const char*>(&hl), 8);
  f << hs;
  f << std::string(total, '\0');
}

std::string model_dir(const std::string& base, const json& cfg, const json& tensors) {
  const std::string dir = base + "/m" + std::to_string(g_checks);
  std::filesystem::create_directories(dir);
  write_text(dir + "/config.json", cfg.dump() + "\n");
  for (const auto& [file, ts] : tensors.items()) write_st(dir + "/" + file, ts);
  return dir;
}

}  // namespace

int main() {
  char tmpl[] = "/tmp/kapitsa_lfm3_test_XXXXXX";
  const std::string base = mkdtemp(tmpl);

  // the published config: the schedule, the RoPE of both attention types, the shape
  const json cfg = lfm3_1_7b();
  const std::string dir = model_dir(base, cfg, json{});
  const ni::Lfm3Config c = ni::Lfm3::read_config(dir);
  ok("config: the shape", c.hidden == 2048 && c.vocab == 128000 && c.layers == 24 && c.hq == 16 && c.hkv == 4 &&
                            c.hd == 128 && c.ffn == 7680 && c.window == 256 && c.conv_k == 7);
  long long nconv = 0, nfull = 0, nswa = 0;
  for (char k : c.kind) nconv += k == 'C', nfull += k == 'F', nswa += k == 'S';
  ok("config: 13 conv, 6 full, 5 sliding of 24", nconv == 13 && nfull == 6 && nswa == 5 && int(c.kind.size()) == 24);
  ok("config: RoPE per attention type",
     c.theta == 1e6 && c.theta_swa == 1e4 && c.partial == 0.5 && c.eps == 1e-5f && c.tied);
  ok("config: the eos ids", c.eos == std::vector<int>({124900}) && c.bos == 124894);

  // the default schedule (no layer_types in the config) is the published one
  json flat = cfg;
  flat.erase("layer_types");
  const std::string flat_dir = model_dir(base, flat, json{});
  ok("config: the default pattern is the published schedule", ni::Lfm3::read_config(flat_dir).kind == c.kind);

  // the cache sizing's exact math
  const size_t row = ni::kv_row_bytes(128);
  ok("cache: full-attention bytes per position", ni::Lfm3::kv_per_pos(c) == 6 * 2 * 4 * row);
  const size_t rows = (256 + 8 + 7) / 8 * 8;
  ok("cache: the sliding rings of a conversation", ni::Lfm3::ring_bytes(c) == 5 * 2 * 4 * rows * row);
  ok("cache: the conv rings of a conversation", ni::Lfm3::conv_state_bytes(c) == 13 * 6 * 2048 * 2);

  // the plan against headers written here: a Q4_1-import-shaped model (4-bit group 32 linears, a bf16 embedding)
  const json q = {{"bits", 4}, {"group_size", 32}, {"mode", "affine"}};
  json quant = cfg;
  quant["quantization"] = q;
  quant["quantization_config"] = q;
  const uint64_t emb = 128000 * 256 * 4;  // U32 [vocab, H*4/32]: the codes as the import writes them
  json tensors = {{"model.safetensors",
                   {{"model.embed_tokens.weight", {{"dtype", "U32"}, {"shape", {128000, 256}}, {"bytes", emb}}},
                    {"model.layers.0.conv.in_proj.weight",
                     {{"dtype", "U32"}, {"shape", {6144, 256}}, {"bytes", 6144 * 256 * 4}}}}}};
  const std::string qdir = model_dir(base, quant, tensors);
  const ni::MemPlan m = ni::Lfm3::plan(qdir, 2048);
  ok("plan: the weights from the headers", m.weights == emb + 6144 * 256 * 4);
  ok("plan: the format from the config's quantization", m.formats == "4-bit group 32", m.formats);
  ok("plan: the caches", m.kv_per_pos == ni::Lfm3::kv_per_pos(c) &&
                            m.slot_state == 3 * (ni::Lfm3::conv_state_bytes(c) + ni::Lfm3::ring_bytes(c)));
  // one decode step: 13 conv mixers of 4 projections of H, 11 attention mixers of 5, 24 SwiGLU FFNs, the tied head
  const size_t H = 2048, QD = 2048, KD = 512, I = 7680;
  const size_t params = 13 * H * (3 * H + H) + 11 * H * (QD + 2 * KD + QD + 16) + 24 * 3 * H * I + H * 128000;
  ok("plan: one decode step's parameters", m.decode_params == params);
  const double bpw = (4 + 32.0 / 32) / 8.0;
  ok("plan: one decode step's bytes and a head row", m.decode_bytes == size_t(double(params) * bpw) &&
                                                        m.head_row_bytes == size_t(double(H) * bpw));
  ok("plan: the shape fields", m.hidden == 2048 && m.layers == 24 && m.ffn == 7680 && m.dense && !m.ffn_ane);
  ok("plan: the activations grow with the chunk", ni::Lfm3::plan(qdir, 4096).activations > m.activations);
  ok("plan: 1.73B parameters", params > 1.72e9 && params < 1.74e9);

  // a bf16 folder (no quantization): the format and the rate follow
  const std::string bdir = model_dir(base, cfg, tensors);
  const ni::MemPlan bm = ni::Lfm3::plan(bdir, 2048);
  ok("plan: an unquantized folder is bf16", bm.formats == "bf16" &&
                                               bm.decode_bytes == params * 2 && bm.head_row_bytes == H * 2);

  // what the engine refuses, precisely (src/engine/engine.cpp, unsupported_model): a runnable LFM3 waits for the
  // backend, and an unsupported setting is refused alone
  const std::string runnable = ni::unsupported_model(dir);
  ok("engine: a runnable LFM3 waits for the backend", runnable.find("does not load models in this build yet") != std::string::npos,
     runnable);
  json bad = cfg;
  bad["conv_L_cache"] = 4;
  const std::string bpath = base + "/engine-bad";
  std::filesystem::create_directories(bpath);
  write_text(bpath + "/config.json", bad.dump());
  const std::string refused = ni::unsupported_model(bpath);
  ok("engine: an unsupported LFM3 setting refused alone", refused.find("conv_L_cache 4") != std::string::npos &&
                                                           refused.find("does not run:") != std::string::npos, refused);
  json other = cfg;
  other["model_type"] = "gemma4";
  write_text(bpath + "/config.json", other.dump());
  ok("engine: a non-LFM3 keeps the generic refusal",
     ni::unsupported_model(bpath).find("an architecture Kapitsa does not run") != std::string::npos);

  printf("%s (%d checks, %d failed)\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
  return g_fail ? 1 : 0;
}

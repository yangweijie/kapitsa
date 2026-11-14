// The LFM3-Omni audio INPUT encoder, v1 CPU (see lfm3_audio.h). The math follows the HF
// reference as pinned in lfm3_audio_ref.h: mel front end, downsampler (3 k3-s2 convs + SiLU +
// 1x1, 8 frames -> 1 token), 17 blocks (the text tower's math at width 768 — k=3 gated short
// convs, HD-96 full-rope attention with SWA 32, NO per-head gate), final RMSNorm, and the
// LayerNorm->768->2048 GELU->2048 projector. Weights bf16 from the co-located audio.safetensors
// (the mmproj import's --audio). v1 runs the whole tower on the CPU (correctness first; Metal
// kernels are the follow-up perf commit).
#include "vision/lfm3_audio.h"

#include "util/log.h"
#include "vision/lfm3_audio_ref.h"
#include "vision/lfm3_vit.h"  // vit_read_json
#include "vision/vit_common.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ni::audio {

namespace vision = ni::vision;
using json = nlohmann::json;

// ---- small helpers ----

namespace {

inline float bf16_to_f32(uint16_t v) {
  uint32_t u = uint32_t(v) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

inline std::vector<float> to_f32(const std::vector<uint16_t>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = bf16_to_f32(v[i]);
  return out;
}

// out[r] = sum_c w[r][c] * x[c] (w bf16, x/out fp32)
inline void gemv(const uint16_t* w, const float* x, int rows, int cols, float* out) {
  for (int r = 0; r < rows; ++r) {
    float s = 0.f;
    const uint16_t* wr = w + size_t(r) * cols;
    for (int c = 0; c < cols; ++c) s += bf16_to_f32(wr[c]) * x[c];
    out[r] = s;
  }
}

inline void add_into(float* y, const float* x, size_t n) {
  for (size_t i = 0; i < n; ++i) y[i] += x[i];
}

// T5 RMSNorm: w * x * rsqrt(mean(x^2) + eps)
inline void rms_norm(const float* x, const uint16_t* w, int d, double eps, float* out) {
  float sum = 0.f;
  for (int i = 0; i < d; ++i) sum += x[i] * x[i];
  const float scale = float(1.0 / std::sqrt(double(sum) / d + eps));
  for (int i = 0; i < d; ++i) out[i] = bf16_to_f32(w[i]) * (x[i] * scale);
}

inline float silu(float x) { return x / (1.f + std::exp(-x)); }

}  // namespace

// ---- supports ----

bool Lfm3Audio::supports(const std::string& dir, AudioConfig* c) {
  const json root = vision::vit_read_json(dir + "/audio_config.json");
  if (root.value("model_type", std::string()) != "lfm3_omni_audio") return false;
  if (!root.contains("audio_encoder_config") || !root["audio_encoder_config"].is_object()) return false;
  const json& v = root["audio_encoder_config"];
  AudioConfig q;
  q.mel_bins = v.value("num_mel_bins", q.mel_bins);
  q.conv_channels = v.value("conv_channels", q.conv_channels);
  q.hidden = v.value("hidden_size", q.hidden);
  q.inter = v.value("intermediate_size", q.inter);
  q.depth = v.value("num_hidden_layers", q.depth);
  q.heads = v.value("num_attention_heads", q.heads);
  q.kv_heads = v.value("num_key_value_heads", q.kv_heads);
  q.head_dim = v.value("head_dim", q.head_dim);
  q.sliding_window = v.value("sliding_window", q.sliding_window);
  q.conv_kernel = v.value("conv_kernel_size", q.conv_kernel);
  q.rms_eps = v.value("norm_eps", q.rms_eps);
  if (v.contains("layer_types") && v["layer_types"].is_array()) {
    q.layer_types.clear();
    for (const auto& t : v["layer_types"]) {
      const std::string s = t.get<std::string>();
      q.layer_types.push_back(s.find("attention") != std::string::npos ? 'S' : 'C');
    }
  }
  if (q.layer_types.size() != size_t(q.depth)) return false;
  q.audio_token = root.value("audio_token_id", q.audio_token);
  if (root.contains("audio_projector")) {
    const json& pj = root["audio_projector"];
    q.out_hidden = pj.value("out_features", q.out_hidden);
    q.proj_ln_eps = pj.value("layer_norm_eps", q.proj_ln_eps);
  }
  if (root.contains("feature_extractor")) {
    const json& f = root["feature_extractor"];
    q.sample_rate = f.value("sampling_rate", q.sample_rate);
    q.n_fft = f.value("n_fft", q.n_fft);
    q.win_length = f.value("win_length", q.win_length);
    q.hop_length = f.value("hop_length", q.hop_length);
    q.log_floor = f.value("log_floor", q.log_floor);
    q.norm_offset = f.value("normalization_offset", q.norm_offset);
    q.norm_divisor = f.value("normalization_divisor", q.norm_divisor);
    q.frames_per_token = f.value("frames_per_token", q.frames_per_token);
  }
  // the shapes the v1 CPU math is built for (the shipped Omni's); anything else stays off
  if (q.mel_bins != 128 || q.conv_channels != 256 || q.hidden != 768 || q.inter != 1792 || q.depth != 17 ||
      q.heads != 8 || q.kv_heads != 4 || q.head_dim != 96 || q.sliding_window != 32 || q.conv_kernel != 3 ||
      q.out_hidden != 2048 || q.frames_per_token != 8 || q.n_fft != 400 || q.hop_length != 160 ||
      q.win_length != 400 || q.sample_rate != 16000) {
    logi("audio: %s has an LFM3 audio tower of a shape Kapitsa does not run (hidden %d, %d/%d heads, depth %d): "
         "audio input is off",
         dir.c_str(), q.hidden, q.heads, q.kv_heads, q.depth);
    return false;
  }
  if (c) *c = q;
  return true;
}

// ---- weights ----

struct Lfm3Audio::Weights {
  vision::VitWeights w;
  // downsampler
  std::vector<uint16_t> ci_w, ci_b, cm_w, cm_b, ce_w, ce_b;  // depthwise/full k3 kernels [C,9] + biases [C]
  std::vector<uint16_t> lm_w, lm_b, le_w, le_b;              // the 1x1 linears [C, C] + [C]
  std::vector<uint16_t> lo_w, lo_b;                          // lin_out [H, mel*C/8] + [H]
  struct Block {
    std::vector<uint16_t> onw, fnw;   // RMSNorm weights [H]
    std::vector<uint16_t> cin_w;      // conv block: conv.conv.weight [H, k]
    std::vector<uint16_t> in_proj;    // conv: conv.in_proj [3H, H]; attn: q_proj [H, H]
    std::vector<uint16_t> k_proj, v_proj;  // attn: [KV, H]
    std::vector<uint16_t> out_proj;   // conv.out_proj / attn.out_proj [H, H]
    std::vector<uint16_t> qn, kn;     // attn: [HD]
    std::vector<uint16_t> w1, w3, w2; // [I, H], [I, H], [H, I]
    char kind = 'C';
  };
  std::vector<Block> blocks;
  std::vector<uint16_t> final_norm;  // audio_encoder.lfm.embedding_norm
  std::vector<uint16_t> plnw, plnb;  // projector LayerNorm
  std::vector<uint16_t> l1w, l1b, l2w, l2b;
};

Lfm3Audio::Lfm3Audio(const std::string& dir, const AudioConfig& c) : dir_(dir), c_(c) {}
Lfm3Audio::~Lfm3Audio() { unload(); }

size_t Lfm3Audio::weight_bytes() const {
  const size_t H = size_t(c_.hidden), I = size_t(c_.inter), O = size_t(c_.out_hidden), C = size_t(c_.conv_channels);
  int nconv = 0, nattn = 0;
  for (char t : c_.layer_types) ((t == 'C') ? nconv : nattn)++;
  const size_t FM = size_t(c_.mel_bins / 8) * C;
  return (size_t(c_.mel_bins) * C * 9 + 3 * (C * 9 + C) + 2 * C * C + 2 * C + FM * H + H) * 2 +
         size_t(nconv) * (3 * H * H + H * H + H * c_.conv_kernel) * 2 +
         size_t(nattn) * (2 * H * H + H * H) * 2 +
         size_t(c_.depth) * (2 * H + 2 * I + H) * 2 + (H + 2 * (O * H + O) + O * O) * 2;
}

void Lfm3Audio::load() {
  if (loaded_) return;
  if (getenv("LFM3_AUDIO_TRACE")) fprintf(stderr, "load: open\n");
  w_ = std::make_unique<Weights>();
  auto& W = *w_;
  const int H = c_.hidden, I = c_.inter, O = c_.out_hidden, C = c_.conv_channels;
  const int HD = c_.head_dim, KV = HD * c_.kv_heads, M = c_.mel_bins;
  W.w.open(dir_, "", {"audio.safetensors"});
  if (getenv("LFM3_AUDIO_TRACE")) fprintf(stderr, "load: opened, reading tensors\n");
  auto rb = [&](std::vector<uint16_t>& dst, const std::string& n, int64_t rows, int64_t cols) {
    if (!W.w.find(n)) fatal("audio: %s is missing", n.c_str());
    dst.resize(size_t(rows * cols));
    W.w.read_bf16(n, dst.data(), size_t(rows), size_t(cols), size_t(cols));
  };
  rb(W.ci_w, "audio_encoder.downsample.conv_in.weight", C, 9);
  rb(W.ci_b, "audio_encoder.downsample.conv_in.bias", 1, C);
  rb(W.cm_w, "audio_encoder.downsample.conv_mid.weight", C, 9);
  rb(W.cm_b, "audio_encoder.downsample.conv_mid.bias", 1, C);
  rb(W.ce_w, "audio_encoder.downsample.conv_end.weight", C, 9);
  rb(W.ce_b, "audio_encoder.downsample.conv_end.bias", 1, C);
  rb(W.lm_w, "audio_encoder.downsample.lin_mid.weight", C, C);
  rb(W.lm_b, "audio_encoder.downsample.lin_mid.bias", 1, C);
  rb(W.le_w, "audio_encoder.downsample.lin_end.weight", C, C);
  rb(W.le_b, "audio_encoder.downsample.lin_end.bias", 1, C);
  rb(W.lo_w, "audio_encoder.downsample.lin_out.weight", H, size_t(M) * C / 8);
  rb(W.lo_b, "audio_encoder.downsample.lin_out.bias", 1, H);
  rb(W.final_norm, "audio_encoder.lfm.embedding_norm.weight", 1, H);
  if (getenv("LFM3_AUDIO_TRACE")) fprintf(stderr, "load: downsampler + norms read\n");
  W.blocks.resize(size_t(c_.depth));
  for (int i = 0; i < c_.depth; ++i) {
    auto& B = W.blocks[size_t(i)];
    const std::string p = "audio_encoder.lfm.layers." + std::to_string(i) + ".";
    B.kind = c_.layer_types[size_t(i)];
    rb(B.onw, p + "operator_norm.weight", 1, H);
    rb(B.fnw, p + "ffn_norm.weight", 1, H);
    if (B.kind == 'C') {
      rb(B.cin_w, p + "conv.conv.weight", H, c_.conv_kernel);
      rb(B.in_proj, p + "conv.in_proj.weight", 3 * H, H);
      rb(B.out_proj, p + "conv.out_proj.weight", H, H);
    } else {
      rb(B.in_proj, p + "self_attn.q_proj.weight", H, H);
      rb(B.k_proj, p + "self_attn.k_proj.weight", KV, H);
      rb(B.v_proj, p + "self_attn.v_proj.weight", KV, H);
      rb(B.out_proj, p + "self_attn.out_proj.weight", H, H);
      rb(B.qn, p + "self_attn.q_layernorm.weight", 1, HD);
      rb(B.kn, p + "self_attn.k_layernorm.weight", 1, HD);
    }
    rb(B.w1, p + "feed_forward.w1.weight", I, H);
    rb(B.w3, p + "feed_forward.w3.weight", I, H);
    rb(B.w2, p + "feed_forward.w2.weight", H, I);
  }
  rb(W.plnw, "audio_projector.layer_norm.weight", 1, H);
  rb(W.plnb, "audio_projector.layer_norm.bias", 1, H);
  rb(W.l1w, "audio_projector.linear_1.weight", O, H);
  rb(W.l1b, "audio_projector.linear_1.bias", 1, O);
  rb(W.l2w, "audio_projector.linear_2.weight", O, O);
  rb(W.l2b, "audio_projector.linear_2.bias", 1, O);
  W.w.close();
  loaded_ = true;
  if (getenv("LFM3_AUDIO_TRACE")) fprintf(stderr, "load: done\n");
}

void Lfm3Audio::unload() {
  w_.reset();
  loaded_ = false;
}

// ---- the processor: WAV -> 16k mono -> log-mel ----

std::vector<float> resample_linear(const float* x, int n, int from, int to) {
  if (from == to || n <= 0) return std::vector<float>(x, x + n);
  const int out_n = std::max(1, int(int64_t(n) * to / from));
  std::vector<float> out(out_n);
  const double step = double(from) / to;
  for (int i = 0; i < out_n; ++i) {
    const double p = i * step;
    const int i0 = std::min(int(p), n - 1);
    const int i1 = std::min(i0 + 1, n - 1);
    const double fr = p - i0;
    out[i] = float(x[i0] + (x[i1] - x[i0]) * fr);
  }
  return out;
}

std::vector<float> decode_wav(const std::string& bytes, int* sample_rate) {
  auto fail = [](const char* why) {
    throw vision::Error(std::string("audio: ") + why +
                        " (16 kHz mono PCM WAV only; convert with ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav)");
  };
  if (bytes.size() < 44 || memcmp(bytes.data(), "RIFF", 4) || memcmp(bytes.data() + 8, "WAVE", 4))
    fail("not a WAV file");
  size_t p = 12;
  uint16_t fmt = 0, ch = 0, bps = 0;
  int32_t rate = 0;
  const char* data = nullptr;
  size_t data_n = 0;
  while (p + 8 <= bytes.size()) {
    const char* id = bytes.data() + p;
    uint32_t sz = 0;
    memcpy(&sz, bytes.data() + p + 4, 4);
    if (!memcmp(id, "fmt ", 4) && sz >= 16) {
      memcpy(&fmt, bytes.data() + p + 8, 2);
      memcpy(&ch, bytes.data() + p + 10, 2);
      memcpy(&rate, bytes.data() + p + 12, 4);
      memcpy(&bps, bytes.data() + p + 22, 2);
    } else if (!memcmp(id, "data", 4)) {
      data = bytes.data() + p + 8;
      data_n = std::min<size_t>(sz, bytes.size() - p - 8);
    }
    p += 8 + sz + (sz & 1);
    if (data && fmt) break;
  }
  if (!data || !fmt) fail("no fmt/data chunk");
  if (fmt != 1 && fmt != 3) fail("compressed WAV format (PCM 1 or IEEE float 3 only)");
  if (bps != 8 && bps != 16 && bps != 32) fail("unsupported bit depth");
  if (!ch) fail("zero channels");
  const int bytes_per = bps / 8;
  const size_t n = data_n / bytes_per / ch;
  if (!n) fail("empty data chunk");
  std::vector<float> out(n, 0.f);
  const uint8_t* q = reinterpret_cast<const uint8_t*>(data);
  for (size_t i = 0; i < n; ++i) {
    float s = 0.f;
    for (uint16_t c = 0; c < ch; ++c) {
      const uint8_t* b = q + (i * ch + c) * bytes_per;
      float v = 0.f;
      if (bps == 16) {
        int16_t x;
        memcpy(&x, b, 2);
        v = x / 32768.f;
      } else if (bps == 32) {
        if (fmt == 3) memcpy(&v, b, 4);
        else {
          int32_t x;
          memcpy(&x, b, 4);
          v = float(x) / 2147483648.f;
        }
      } else v = (int(b[0]) - 128) / 128.f;
      s += v;
    }
    out[i] = s / float(ch);
  }
  if (sample_rate) *sample_rate = rate;
  return out;
}

AudioPrepared Lfm3Audio::prepare(const std::string& wav_bytes) const {
  int rate = 0;
  std::vector<float> samples = decode_wav(wav_bytes, &rate);
  samples = resample_linear(samples.data(), int(samples.size()), rate, c_.sample_rate);
  AudioPrepared p;
  p.frames = audio_ref_num_frames(int(samples.size()));
  if (p.frames <= 0) return p;
  p.tokens = (p.frames + c_.frames_per_token - 1) / c_.frames_per_token;
  p.mel.resize(size_t(p.frames) * c_.mel_bins);
  const auto fb = audio_ref_mel_filterbank(c_.mel_bins);
  const auto win = audio_ref_hann(c_.win_length);
  for (int t = 0; t < p.frames; ++t)
audio_ref_log_mel_frame(samples.data(), int(samples.size()), fb, win, t, &p.mel[size_t(t) * c_.mel_bins]);
  // the digest: the features' bytes + the dims + a tag (the images' digest pattern)
  {
    const auto h = vision::sha256(p.mel.data(), p.mel.size() * 4);
    std::vector<uint8_t> key(32 + 8 + 8);
    memcpy(key.data(), h.data(), 32);
    const int32_t dims[2] = {p.frames, c_.mel_bins};
    memcpy(key.data() + 32, dims, 8);
    memcpy(key.data() + 40, "lfm3-aud", 8);
    const auto k = vision::sha256(key.data(), key.size());
    memcpy(p.digest.b, k.data(), 32);
  }
  return p;
}

// ---- the tower ----

// one downsampler stage: depthwise k3 s2 p1 over (mel, time) [+ mel input for stage 1], SiLU, 1x1
static void ds_stage(const float* in, int tm, int mm, const std::vector<uint16_t>& dw,
                     const std::vector<uint16_t>& db, const std::vector<uint16_t>& lw,
                     const std::vector<uint16_t>& lb, std::vector<float>& out, int& ot, int& om,
                     int C, bool full_mel_axis) {
  ot = (tm + 1) / 2;
  om = (mm + 1) / 2;
  out.assign(size_t(ot) * om * C, 0.f);
  std::vector<float> conv(size_t(ot) * om * C);
  for (int t = 0; t < ot; ++t)
    for (int m = 0; m < om; ++m)
      for (int c = 0; c < C; ++c) {
        float s = bf16_to_f32(db[c]);
        for (int dm = 0; dm < 3; ++dm) {
          const int mi = m * 2 - 1 + dm;
          if (mi < 0 || mi >= mm) continue;
          for (int dt = 0; dt < 3; ++dt) {
            const int ti = t * 2 - 1 + dt;
            if (ti < 0 || ti >= tm) continue;
            const float w = bf16_to_f32(dw[size_t(c) * 9 + dm * 3 + dt]);
            if (full_mel_axis) s += w * in[size_t(ti) * mm + mi];  // stage 1: input is [t, mel]
            else s += w * in[(size_t(ti) * mm + mi) * C + c];
          }
        }
        conv[(size_t(t) * om + m) * C + c] = s;
      }
  // the HF stage is act(lin(conv(x))): ONE SiLU after the 1x1 — except conv_in, whose "1x1" is the
  // identity (conv_in has no 1x1: SiLU directly after the conv). Callers pass lw empty for that stage.
  if (lw.empty()) {
    for (size_t i = 0; i < conv.size(); ++i) conv[i] = silu(conv[i]);
    out.swap(conv);
    return;
  }
  for (size_t i = 0; i < conv.size(); ++i) {
    const size_t r = i / C;
    const int c = int(i % C);
    float s = bf16_to_f32(lb[c]);
    const uint16_t* wr = lw.data() + size_t(c) * C;
    for (int j = 0; j < C; ++j) s += bf16_to_f32(wr[j]) * conv[r * C + j];
    out[i] = silu(s);
  }
}

std::vector<float> Lfm3Audio::downsample(const float* mel, int frames) const {
  const int M = c_.mel_bins, C = c_.conv_channels;
  const auto& W = *w_;
  if (getenv("LFM3_AUDIO_TRACE")) fprintf(stderr, "downsample: frames %d\n", frames);
  std::vector<float> s1, s2, s3;
  int t1, m1, t2, m2, t3, m3;
  // conv_in (full over the mel axis, no 1x1) -> lin_mid pair -> lin_end pair -> lin_out
  {
    // stage 1 by hand: conv_in reads the raw mel [frames, M]
    t1 = (frames + 1) / 2;
    m1 = (M + 1) / 2;
    s1.assign(size_t(t1) * m1 * C, 0.f);
    for (int t = 0; t < t1; ++t)
      for (int m = 0; m < m1; ++m)
        for (int c = 0; c < C; ++c) {
          float s = bf16_to_f32(W.ci_b[c]);
          for (int dm = 0; dm < 3; ++dm) {
            const int mi = m * 2 - 1 + dm;
            if (mi < 0 || mi >= M) continue;
            for (int dt = 0; dt < 3; ++dt) {
              const int ti = t * 2 - 1 + dt;
              if (ti < 0 || ti >= frames) continue;
              s += bf16_to_f32(W.ci_w[size_t(c) * 9 + dm * 3 + dt]) * mel[size_t(ti) * M + mi];
            }
          }
          s1[(size_t(t) * m1 + m) * C + c] = silu(s);
        }
  }
  if (getenv("LFM3_AUDIO_DUMP")) { FILE* g = fopen("/tmp/cpp_s1.bin", "wb"); fwrite(s1.data(), 4, s1.size(), g); fclose(g); }
  ds_stage(s1.data(), t1, m1, W.cm_w, W.cm_b, W.lm_w, W.lm_b, s2, t2, m2, C, false);
  if (getenv("LFM3_AUDIO_TRACE")) fprintf(stderr, "  stage2: %dx%dx%d\n", t2, m2, C);
  if (getenv("LFM3_AUDIO_DUMP")) { FILE* g = fopen("/tmp/cpp_s2.bin", "wb"); fwrite(s2.data(), 4, s2.size(), g); fclose(g); }
  ds_stage(s2.data(), t2, m2, W.ce_w, W.ce_b, W.le_w, W.le_b, s3, t3, m3, C, false);
  if (getenv("LFM3_AUDIO_DUMP")) { FILE* g = fopen("/tmp/cpp_s3.bin", "wb"); fwrite(s3.data(), 4, s3.size(), g); fclose(g); }
  if (getenv("LFM3_AUDIO_TRACE")) fprintf(stderr, "  stage3: %dx%dx%d\n", t3, m3, C);
  // lin_out over the flattened (mel16, C) of each token
  const int FM = m3 * C;
  std::vector<float> out(size_t(t3) * c_.hidden), flat(FM);
  for (int t = 0; t < t3; ++t) {
    // the flatten follows the HF [B, C, M, T].flatten(1, 2): channel-major, then mel, per token
    // (torch CM vs the fp32 reference's downsampled output: cosine 0.9995)
    for (int m = 0; m < m3; ++m)
      for (int c = 0; c < C; ++c) flat[size_t(c) * m3 + m] = s3[(size_t(t) * m3 + m) * C + c];
    gemv(W.lo_w.data(), flat.data(), c_.hidden, FM, &out[size_t(t) * c_.hidden]);
    for (int i = 0; i < c_.hidden; ++i) out[size_t(t) * c_.hidden + i] += bf16_to_f32(W.lo_b[i]);
  }
  return out;
}

std::vector<float> Lfm3Audio::dbg_downsample(const AudioPrepared& p) const {
  if (!loaded_) const_cast<Lfm3Audio*>(this)->load();
  return downsample(p.mel.data(), p.frames);
}

// test hook: the tower's blocks + final norm (fp32), the "down.npy" stage of the reference dump
std::vector<float> Lfm3Audio::dbg_encode(const AudioPrepared& p) const {
  if (!loaded_) const_cast<Lfm3Audio*>(this)->load();
  const auto ds = downsample(p.mel.data(), p.frames);
  std::vector<float> out(size_t(p.tokens) * c_.hidden);
  blocks(ds.data(), p.tokens, out.data());
  return out;
}

void Lfm3Audio::blocks(const float* x, int T, float* out) const {
  const int H = c_.hidden, I = c_.inter, HD = c_.head_dim, HQ = c_.heads, HKV = c_.kv_heads, KV = HD * HKV;
  const int K = c_.conv_kernel;
  const auto& W = *w_;
  std::vector<float> h(x, x + size_t(T) * H), hn(size_t(T) * H);
  std::vector<float> proj(3 * H), u(size_t(T) * H), cplan(size_t(T) * H), convout(H), att(H);
  std::vector<float> q(size_t(T) * H), k(size_t(T) * KV), v(size_t(T) * KV), gate(I), lin3(I), y(H);
  const auto invf = audio_ref_inv_freq();
  for (int L = 0; L < c_.depth; ++L) {
    const auto& B = W.blocks[size_t(L)];
    for (int t = 0; t < T; ++t) rms_norm(&h[size_t(t) * H], B.onw.data(), H, c_.rms_eps, &hn[size_t(t) * H]);
    if (getenv("LFM3_AUDIO_DUMP") && L == 0) { FILE* g = fopen("/tmp/cpp_hn0.bin", "wb"); fwrite(hn.data(), 4, hn.size(), g); fclose(g); }
    if (B.kind == 'C') {
      // in_proj -> B, C, x rows (the text tower's chunk order); u = B*x per row
      for (int t = 0; t < T; ++t) {
        gemv(B.in_proj.data(), &hn[size_t(t) * H], 3 * H, H, proj.data());
        for (int i = 0; i < H; ++i) u[size_t(t) * H + i] = proj[i] * proj[2 * H + i];
        memcpy(&cplan[size_t(t) * H], &proj[H], size_t(H) * 4);  // C
        if (getenv("LFM3_AUDIO_DUMP") && L == 0 && t == 0) {
          FILE* g = fopen("/tmp/cpp_proj0.bin", "wb");
          fwrite(proj.data(), 4, proj.size(), g);
          fclose(g);
        }
      }
      // causal k-conv over u, then y = C * conv(u), then out_proj
      for (int t = 0; t < T; ++t) {
        for (int i = 0; i < H; ++i) {
          float s = 0.f;
          for (int dt = 0; dt < K; ++dt) {
            const int ti = t - (K - 1) + dt;
            if (ti < 0) continue;
            s += bf16_to_f32(B.cin_w[size_t(i) * K + dt]) * u[size_t(ti) * H + i];
          }
          convout[i] = cplan[size_t(t) * H + i] * s;
        }
        if (getenv("LFM3_AUDIO_DUMP") && L == 0 && t == 0) {
          FILE* g = fopen("/tmp/cpp_convout0.bin", "wb");
          fwrite(convout.data(), 4, convout.size(), g);
          fclose(g);
        }
        gemv(B.out_proj.data(), convout.data(), H, H, att.data());
        if (getenv("LFM3_AUDIO_DUMP") && L == 0 && t == 0) {
          FILE* g = fopen("/tmp/cpp_att0.bin", "wb");
          fwrite(att.data(), 4, att.size(), g);
          fclose(g);
        }
        add_into(&h[size_t(t) * H], att.data(), size_t(H));
      }
    } else {
      // attention: q/k/v, per-head QK RMSNorm + full rope, windowed softmax GQA, out_proj. No gate.
      for (int t = 0; t < T; ++t) {
        gemv(B.in_proj.data(), &hn[size_t(t) * H], H, H, &q[size_t(t) * H]);
        gemv(B.k_proj.data(), &hn[size_t(t) * H], KV, H, &k[size_t(t) * KV]);
        gemv(B.v_proj.data(), &hn[size_t(t) * H], KV, H, &v[size_t(t) * KV]);
      }
      if (getenv("LFM3_AUDIO_DUMP") && L == 2) {
        FILE* gq = fopen("/tmp/cpp_q2.bin", "wb");
        fwrite(q.data(), 4, q.size(), gq);
        fclose(gq);
        FILE* gk = fopen("/tmp/cpp_k2.bin", "wb");
        fwrite(k.data(), 4, k.size(), gk);
        fclose(gk);
        FILE* gv = fopen("/tmp/cpp_v2.bin", "wb");
        fwrite(v.data(), 4, v.size(), gv);
        fclose(gv);
      }
      for (int hd = 0; hd < 2; ++hd) {
        const int heads = hd == 0 ? HQ : HKV;
        float* buf = hd == 0 ? q.data() : k.data();
        const std::vector<uint16_t>& nw = hd == 0 ? B.qn : B.kn;
        for (int t = 0; t < T; ++t)
          for (int hh = 0; hh < heads; ++hh) {
            float* xh = buf + size_t(t) * heads * HD + size_t(hh) * HD;
            float sum = 0.f;
            for (int d = 0; d < HD; ++d) sum += xh[d] * xh[d];
            const float sc = float(1.0 / std::sqrt(double(sum) / HD + c_.rms_eps));
            for (int d = 0; d < HD; ++d) xh[d] = bf16_to_f32(nw[d]) * (xh[d] * sc);
            for (int d = 0; d < HD / 2; ++d) {
              const double ang = double(t) * invf[d];
              const float cs = float(std::cos(ang)), sn = float(std::sin(ang));
              const float a = xh[d], b = xh[d + HD / 2];
              xh[d] = a * cs - b * sn;
              xh[d + HD / 2] = b * cs + a * sn;
            }
          }
      }
      if (getenv("LFM3_AUDIO_DUMP") && L == 2) {
        FILE* gq = fopen("/tmp/cpp_q2r.bin", "wb");
        fwrite(q.data(), 4, q.size(), gq);
        fclose(gq);
        FILE* gk = fopen("/tmp/cpp_k2r.bin", "wb");
        fwrite(k.data(), 4, k.size(), gk);
        fclose(gk);
        FILE* gv = fopen("/tmp/cpp_v2.bin", "wb");
        fwrite(v.data(), 4, v.size(), gv);
        fclose(gv);
      }
      const double scale = 1.0 / std::sqrt(double(HD));
      for (int t = 0; t < T; ++t) {
        const int t0 = std::max(0, t - c_.sliding_window + 1);
        for (int hh = 0; hh < HQ; ++hh) {
          const int hk = hh / (HQ / HKV);  // repeat_kv: q head h reads kv head h / (HQ/HKV)
          std::vector<double> sc(size_t(t - t0 + 1));
          double mx = -1e300;
          for (int s = t0; s <= t; ++s) {
            double dot = 0.0;
            for (int d = 0; d < HD; ++d)
              dot += double(q[size_t(t) * H + hh * HD + d]) * double(k[size_t(s) * KV + hk * HD + d]);
            sc[size_t(s - t0)] = dot * scale;
            if (sc[size_t(s - t0)] > mx) mx = sc[size_t(s - t0)];
          }
          double den = 0.0;
          for (auto& z : sc) {
            z = std::exp(z - mx);
            den += z;
          }
          for (int s = t0; s <= t; ++s) {
            const double wgt = sc[size_t(s - t0)] / den;
            for (int d = 0; d < HD; ++d) att[size_t(hh) * HD + d] += float(wgt * v[size_t(s) * KV + hk * HD + d]);
          }
        }
        if (getenv("LFM3_AUDIO_DUMP") && L == 2 && t < 3) {
          FILE* ga = fopen(t == 0 ? "/tmp/cpp_a2_0.bin" : t == 1 ? "/tmp/cpp_a2_1.bin" : "/tmp/cpp_a2_2.bin", "wb");
          fwrite(att.data(), 4, att.size(), ga);
          fclose(ga);
        }
        gemv(B.out_proj.data(), att.data(), H, H, y.data());
        add_into(&h[size_t(t) * H], y.data(), size_t(H));
        std::fill(att.begin(), att.end(), 0.f);
      }
    }
    if (getenv("LFM3_AUDIO_DUMP")) {
        char fn[64];
        snprintf(fn, sizeof fn, "/tmp/cpp_blk%d.bin", L);
        FILE* g = fopen(fn, "wb");
        fwrite(h.data(), 4, h.size(), g);
        fclose(g);
    }
    // MLP: ffn_norm -> w1, w3 -> silu*w3 -> w2 -> +h
    for (int t = 0; t < T; ++t) {
      rms_norm(&h[size_t(t) * H], B.fnw.data(), H, c_.rms_eps, &hn[size_t(t) * H]);
      if (getenv("LFM3_AUDIO_DUMP") && L == 0 && t == 0) {
        FILE* g = fopen("/tmp/cpp_hn2_0.bin", "wb");
        fwrite(&hn[size_t(t) * H], 4, H, g);
        fclose(g);
        FILE* g2 = fopen("/tmp/cpp_h0.bin", "wb");
        fwrite(&h[size_t(t) * H], 4, H, g2);
        fclose(g2);
      }
      gemv(B.w1.data(), &hn[size_t(t) * H], I, H, gate.data());
      gemv(B.w3.data(), &hn[size_t(t) * H], I, H, lin3.data());
      for (int i = 0; i < I; ++i) gate[i] = silu(gate[i]) * lin3[i];
      if (getenv("LFM3_AUDIO_DUMP") && L == 0 && t == 0) {
        FILE* g3 = fopen("/tmp/cpp_gate0.bin", "wb");
        fwrite(gate.data(), 4, I, g3);
        fclose(g3);
      }
      gemv(B.w2.data(), gate.data(), H, I, y.data());
      if (getenv("LFM3_AUDIO_DUMP") && L == 0 && t == 0) {
        FILE* g4 = fopen("/tmp/cpp_mlp0.bin", "wb");
        fwrite(y.data(), 4, H, g4);
        fclose(g4);
      }
      add_into(&h[size_t(t) * H], y.data(), size_t(H));
    }
    if (getenv("LFM3_AUDIO_DUMP")) {
      char fn[64];
      snprintf(fn, sizeof fn, "/tmp/cpp_blk%d.bin", L);
      FILE* g = fopen(fn, "wb");
      fwrite(h.data(), 4, h.size(), g);
      fclose(g);
    }
  }
  const int H2 = H;
  (void)H2;
  for (int t = 0; t < T; ++t) rms_norm(&h[size_t(t) * H], W.final_norm.data(), H, c_.rms_eps, &out[size_t(t) * H]);
}

Buffer Lfm3Audio::encode(const AudioPrepared& p) {
  if (!loaded_) load();
  if (p.tokens <= 0) return Buffer{};
  const int T = p.tokens, H = c_.hidden, O = c_.out_hidden;
  const auto& W = *w_;
  const std::vector<float> ds = downsample(p.mel.data(), p.frames);
  std::vector<float> enc(size_t(T) * H);
  blocks(ds.data(), T, enc.data());
  if (getenv("LFM3_AUDIO_DUMP")) {
    FILE* g = fopen("/tmp/cpp_enc2.bin", "wb");
    fwrite(enc.data(), 4, enc.size(), g);
    fclose(g);
  }
  // the projector: LayerNorm -> linear_1 -> GELU (tanh) -> linear_2 -> bf16 rows
  std::vector<uint16_t> host(size_t(T) * O);
  std::vector<float> nrm(H), y1(O), y2(O);
  for (int t = 0; t < T; ++t) {
    float mean = 0.f;
    for (int i = 0; i < H; ++i) mean += enc[size_t(t) * H + i];
    mean /= H;
    float var = 0.f;
    for (int i = 0; i < H; ++i) {
      const float d = enc[size_t(t) * H + i] - mean;
      var += d * d;
    }
    var /= H;
    const float rstd = float(1.0 / std::sqrt(double(var) + c_.proj_ln_eps));
    for (int i = 0; i < H; ++i)
      nrm[i] = bf16_to_f32(W.plnw[i]) * ((enc[size_t(t) * H + i] - mean) * rstd) + bf16_to_f32(W.plnb[i]);
    gemv(W.l1w.data(), nrm.data(), O, H, y1.data());
    for (int i = 0; i < O; ++i) y1[i] += bf16_to_f32(W.l1b[i]);
    for (int i = 0; i < O; ++i) {
      const float x3 = y1[i] * y1[i] * y1[i];
      y1[i] = 0.5f * y1[i] * (1.f + std::tanh(0.7978845608028654f * (y1[i] + 0.044715f * x3)));
    }
    if (getenv("LFM3_AUDIO_DUMP") && t == 0) {
      FILE* g5 = fopen("/tmp/cpp_nrm0.bin", "wb");
      fwrite(nrm.data(), 4, H, g5);
      fclose(g5);
      FILE* g6 = fopen("/tmp/cpp_y1_0.bin", "wb");
      fwrite(y1.data(), 4, O, g6);
      fclose(g6);
    }
    gemv(W.l2w.data(), y1.data(), O, O, y2.data());
    if (getenv("LFM3_AUDIO_DUMP") && t == 0) {
      FILE* g7 = fopen("/tmp/cpp_y2_0.bin", "wb");
      fwrite(y2.data(), 4, O, g7);
      fclose(g7);
    }
    for (int i = 0; i < O; ++i) y2[i] += bf16_to_f32(W.l2b[i]);
    for (int i = 0; i < O; ++i) {
      uint32_t u32;
      memcpy(&u32, &y2[i], 4);
      host[size_t(t) * O + i] = uint16_t((u32 + 0x7FFFu + ((u32 >> 16) & 1)) >> 16);
    }
  }
  Buffer out = Gpu::get().alloc(size_t(T) * O * 2 + 16, "audio_embeddings");
  memcpy(out.ptr(), host.data(), host.size() * 2);
  return out;
}

}  // namespace ni::audio

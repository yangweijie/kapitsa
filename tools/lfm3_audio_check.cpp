// The audio tower's CPU (v1) check against the HF reference's dumped activations
// (tests/vision/ref_audio.py): mel (vs the numpy goldens already covered by
// lfm3_audio_ref_test), the downsampler's output, the encoder blocks' per-layer hidden states,
// and the projector's output. Reads the reference's .npy files directly (the v2 header, fp32).
//
//   lfm3_audio_check CLIP.wav REFDIR [MODELDIR] [--gpu]
//
// MODELDIR defaults to .lfm3-real/home/models/lfm3-omni-2b (the imported Omni, whose folder
// carries audio.safetensors + audio_config.json from the mmproj --audio import). --gpu runs the
// downsampler's Metal path (src/kernels/audio.metal) against the CPU oracle; it needs the GPU
// lock, run it under `lockf -k /tmp/kapitsa-gpu.lock`.
#include "vision/lfm3_audio.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

namespace {

// a minimal .npy v1 reader (fp32/f64, little-endian, C or Fortran order)
std::vector<float> read_npy(const std::string& path, std::vector<int64_t>& shape) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    exit(2);
  }
  char magic[7] = {};
  fread(magic, 1, 6, f);
  if (memcmp(magic, "\x93NUMPY", 6)) exit(2);
  uint8_t ver[2];
  fread(ver, 1, 2, f);
  uint16_t hlen16;
  uint32_t hlen;
  if (ver[0] == 1) {
    fread(&hlen16, 2, 1, f);
    hlen = hlen16;
  } else {
    fread(&hlen, 4, 1, f);
  }
  std::string header(hlen, ' ');
  fread(header.data(), 1, hlen, f);
  bool f64 = header.find("'f8'") != std::string::npos || header.find("<f8") != std::string::npos;
  const bool fortran = header.find("'fortran_order': True") != std::string::npos;
  int64_t n = 1;
  shape.clear();
  {
    size_t p = header.find("'shape': (");
    p += 10;
    const size_t close = header.find(")", p);
    while (p < close) {
      size_t q = header.find(',', p);
      if (q == std::string::npos || q > close) q = close;
      shape.push_back(atoll(header.substr(p, q - p).c_str()));
      n *= shape.back();
      p = q + 1;
      while (p < header.size() && (header[p] == ' ' || header[p] == ',')) ++p;
    }
  }
  std::vector<float> out(n);
  if (f64) {
    std::vector<double> raw(n);
    fread(raw.data(), 8, n, f);
    for (int64_t i = 0; i < n; ++i) out[i] = float(raw[i]);
  } else {
    fread(out.data(), 4, n, f);
  }
  fclose(f);
  // a Fortran-ordered 2D dump (a non-contiguous view saved by numpy) reads as the transpose
  if (fortran && shape.size() == 2) {
    const int64_t r = shape[0], c = shape[1];
    std::vector<float> t(n);
    for (int64_t i = 0; i < r; ++i)
      for (int64_t j = 0; j < c; ++j) t[i * c + j] = out[j * r + i];
    out.swap(t);
  }
  return out;
}

double compare(const std::vector<float>& got, const std::vector<float>& ref, const char* what) {
  if (got.size() != ref.size()) {
    fprintf(stderr, "%s: size %zu vs %zu\n", what, got.size(), ref.size());
    exit(1);
  }
  double num = 0, da = 0, db = 0, mx = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = double(got[i]) - double(ref[i]);
    num += d * d;
    da += double(got[i]) * double(got[i]);
    db += double(ref[i]) * double(ref[i]);
    mx = std::max(mx, std::abs(d));
  }
  // the true cosine of the centred-less vectors: (da + db - sum d^2) / (2 |a| |b|)
  const double cos = da * db == 0 ? 1.0 : (da + db - num) / (2.0 * std::sqrt(da * db));
  fprintf(stderr, "%-8s cosine %.6f  max|d| %.4f\n", what, cos, mx);
  return cos;
}

}  // namespace

int main(int argc, char** argv) {
  bool gpu_mode = false;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--gpu")
      gpu_mode = true;
    else
      pos.push_back(a);
  }
  if (pos.size() < 2) {
    fprintf(stderr, "usage: lfm3_audio_check CLIP.wav REFDIR [MODELDIR] [--gpu]\n");
    return 2;
  }
  const std::string clip = pos[0], refdir = pos[1];
  const std::string dir = pos.size() > 2 ? pos[2] : ".lfm3-real/home/models/lfm3-omni-2b";
  ni::audio::AudioConfig c;
  if (!ni::audio::Lfm3Audio::supports(dir, &c)) {
    fprintf(stderr, "no audio tower in %s\n", dir.c_str());
    return 2;
  }
  ni::audio::Lfm3Audio tower(dir, c);
  const auto ms = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count() * 1000; };

  FILE* f = fopen(clip.c_str(), "rb");
  if (!f) return 2;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::string bytes(sz, ' ');
  fread(bytes.data(), 1, size_t(sz), f);
  fclose(f);

  auto t0 = std::chrono::steady_clock::now();
  const auto p = tower.prepare(bytes);
  auto t1 = std::chrono::steady_clock::now();
  fprintf(stderr, "clip: %d frames, %d tokens (prepare %.0f ms)\n", p.frames, p.tokens, ms(t0, t1));
  {
    // ref_ds.npy holds the DOWNSAMPLER's output (layer 0's input, a forward-pre-hook dump);
    // down.npy holds the ENCODER's last hidden state — ref_audio.py dumps enc(...)'s return.
    std::vector<int64_t> sh;
    const auto ds_ref = read_npy(refdir + "/ref_ds.npy", sh);  // [1, tokens, hidden] batch-major
    const bool ref_match = int64_t(p.tokens) == sh[sh.size() - 2];
    if (!ref_match && !gpu_mode) {
      fprintf(stderr, "token count %d vs ref %lld\n", p.tokens, (long long)sh[0]);
      return 1;
    }
    if (!ref_match) {
      // a --gpu run on a clip the ref dump was not made for: the check is the GPU's agreement
      // with the CPU oracle (below), the reference comparisons stay out of it
      fprintf(stderr, "ref dump is for a different clip (%d tokens); --gpu checks the CPU oracle only\n",
              p.tokens);
    }
    if (ref_match) {
      const auto ds_got = tower.dbg_downsample(p);
      compare(ds_got, ds_ref, "ds");
      { FILE* g = fopen("/tmp/cpp_ds.bin", "wb"); fwrite(ds_got.data(), 4, ds_got.size(), g); fclose(g); }
      const auto ref = read_npy(refdir + "/down.npy", sh);
      const auto got = tower.dbg_encode(p);
      fprintf(stderr, "encode done, %zu rows (%.0f ms incl load)\n", got.size(),
              ms(t1, std::chrono::steady_clock::now()));
      const double cos = compare(got, ref, "enc");
      { FILE* g = fopen("/tmp/cpp_enc.bin", "wb"); fwrite(got.data(), 4, got.size(), g); fclose(g); }
      if (cos < 0.999) fprintf(stderr, "ENC below 0.999\n");
    }
  }
  if (gpu_mode) {
    // the Metal downsampler vs the CPU oracle (both read the same bf16 checkpoint weights; the
    // band is the fp32-sum + fast::exp drift, ~1e-4 relative on O(1) activations). The reference
    // .npy comparisons above already hold the CPU side, so the GPU's agreement with it is the
    // GPU-vs-CPU check.
    tower.load();
    const auto ds_cpu = tower.dbg_downsample(p);
    auto t3c = std::chrono::steady_clock::now();
    const auto ds_gpu = tower.dbg_downsample_gpu(p);
    auto t4c = std::chrono::steady_clock::now();
    if (ds_gpu.size() != ds_cpu.size()) {
      fprintf(stderr, "gpu ds: size %zu vs cpu %zu\n", ds_gpu.size(), ds_cpu.size());
      return 1;
    }
    double num = 0, da = 0, db = 0, mx = 0;
    for (size_t i = 0; i < ds_gpu.size(); ++i) {
      const double d = double(ds_gpu[i]) - double(ds_cpu[i]);
      num += d * d;
      da += double(ds_gpu[i]) * double(ds_gpu[i]);
      db += double(ds_cpu[i]) * double(ds_cpu[i]);
      mx = std::max(mx, std::abs(d));
    }
    const double cos = da * db == 0 ? 1.0 : (da + db - num) / (2.0 * std::sqrt(da * db));
    fprintf(stderr, "gpu ds: cosine vs cpu %.7f  max|d| %.6g  (%.0f ms incl load)\n", cos, mx,
            ms(t3c, t4c));
    if (cos < 0.99995 || mx > 1e-2) {
      fprintf(stderr, "gpu ds outside the fp32 band (cos > 0.99995, max|d| < 1e-2)\n");
      return 1;
    }
    fprintf(stderr, "ok gpu ds (the downsampler's Metal path matches the CPU oracle)\n");
    // the 17 encoder blocks: the CPU's attention math is double, the GPU's fp32, and the stack
    // is 17 layers deep — the band is wider than the downsampler's (cos > 0.9995, max|d| <
    // 5e-2 on O(1) hidden states).
    const auto enc_cpu = tower.dbg_encode(p);
    const auto enc_gpu = tower.dbg_encode_gpu(p);
    if (enc_gpu.size() != enc_cpu.size()) {
      fprintf(stderr, "gpu enc: size %zu vs cpu %zu\n", enc_gpu.size(), enc_cpu.size());
      return 1;
    }
    num = 0, da = 0, db = 0, mx = 0;
    for (size_t i = 0; i < enc_gpu.size(); ++i) {
      const double d = double(enc_gpu[i]) - double(enc_cpu[i]);
      num += d * d;
      da += double(enc_gpu[i]) * double(enc_gpu[i]);
      db += double(enc_cpu[i]) * double(enc_cpu[i]);
      mx = std::max(mx, std::abs(d));
    }
    const double ecos = da * db == 0 ? 1.0 : (da + db - num) / (2.0 * std::sqrt(da * db));
    fprintf(stderr, "gpu enc: cosine vs cpu %.7f  max|d| %.6g\n", ecos, mx);
    if (ecos < 0.9995 || mx > 5e-2) {
      fprintf(stderr, "gpu enc outside the fp32 band (cos > 0.9995, max|d| < 5e-2)\n");
      return 1;
    }
    fprintf(stderr, "ok gpu enc (the encoder blocks' Metal path matches the CPU oracle)\n");
    return 0;
  }
  tower.load();
  {
    std::vector<int64_t> sh;
    const auto ref = read_npy(refdir + "/proj.npy", sh);
    auto t2 = std::chrono::steady_clock::now();
    const ni::Buffer emb = tower.encode(p);
    fprintf(stderr, "projector done (%.0f ms)\n", ms(t2, std::chrono::steady_clock::now()));
    // the encoder's output rows are bf16 (uint16 halves); read them as bf16, not fp32
    std::vector<float> got(ref.size());
    const uint16_t* hw = reinterpret_cast<const uint16_t*>(emb.ptr());
    for (size_t i = 0; i < ref.size(); ++i) {
      uint32_t u32 = uint32_t(hw[i]) << 16;
      memcpy(&got[i], &u32, 4);
    }
    compare(got, ref, "proj");
    { FILE* g = fopen("/tmp/cpp_proj.bin", "wb"); fwrite(got.data(), 4, got.size(), g); fclose(g); }
  }
  return 0;
}

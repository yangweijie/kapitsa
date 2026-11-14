#!/usr/bin/env python3
"""The LFM3-Omni audio tower's feature check: runs the HF reference (trust_remote_code) over a
clip and dumps the encoder's per-stage activations, so Kapitsa's CPU tower (v1) diffs against
it like ref_vision.py does for the vision tower. numpy + torch, no Kapitsa code.

Stages dumped (all fp32):
  mel        the feature extractor's input_features [frames, 128]
  down       the downsampler's output, the encoder blocks' input [tokens, 768]
  enc_L      the encoder hidden state after block L (0..16)
  final      after embedding_norm
  proj       the projector's output [tokens, 2048] (what the prompt embeds)

    python3 tests/vision/ref_audio.py CLIP.wav OUTDIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
REF = REPO / ".lfm3-real" / "omni-fp32"


def main():
    clip, outdir = sys.argv[1], sys.argv[2]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REF))
    import soundfile as sf

    wav, sr = sf.read(clip, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float64)
    # the mel front end, transcribed directly (the same math ref_lfm3_audio.py verified against
    # HF's extractor; the relative import in feature_extraction_lfm3_omni.py can't be imported
    # standalone). Constants from processor_config.json.
    SR, N_FFT, WIN, HOP, MEL = 16000, 400, 400, 160, 128
    LOG_FLOOR, N_OFF, N_DIV = -12.0, 4.0, 4.0

    def hz_to_mel(f):
        return np.where(f < 1000.0, f * 3.0 / 200.0,
                        15.0 + np.log(np.maximum(f, 1e-30) / 1000.0) / (np.log(6.4) / 27.0))

    def mel_to_hz(m):
        return np.where(m <= 15.0, m * 200.0 / 3.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)))

    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(SR / 2.0), MEL + 2)
    f_pts = mel_to_hz(mels)
    fft_freq = np.arange(N_FFT // 2 + 1) * SR / N_FFT
    fb = np.zeros((MEL, len(fft_freq)))
    for m in range(MEL):
        left, center, right = f_pts[m], f_pts[m + 1], f_pts[m + 2]
        up = (fft_freq - left) / (center - left)
        down = (right - fft_freq) / (right - center)
        fb[m] = np.maximum(0.0, np.minimum(up, down)) * (2.0 / (right - left))
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)
    frames = 1 + (len(wav) + ((-(len(wav) - N_FFT)) % HOP) - N_FFT) // HOP if len(wav) >= N_FFT else 0
    out_mel = np.zeros((frames, MEL), dtype=np.float32)
    for t in range(frames):
        off = t * HOP
        seg = np.zeros(N_FFT, dtype=np.float64)
        seg[:max(0, min(N_FFT, len(wav) - off))] = wav[off:off + N_FFT]
        spec = np.abs(np.fft.rfft(seg * window)) ** 2
        mel = fb @ spec
        out_mel[t] = ((np.maximum(np.log10(np.maximum(mel, 1e-30)), LOG_FLOOR) + N_OFF) / N_DIV).astype(np.float32)
    feats = torch.from_numpy(out_mel.T)  # [128, T]: the tower's input layout
    from transformers.modeling_utils import ROPE_INIT_FUNCTIONS  # noqa: E402

    # transformers 5.15's generic _init_weights doesn't know rope_type "axial" (the vision tower's);
    # the stub only has to return a plausible inv_freq — the checkpoint overwrites the buffer
    def _axial_stub(config, **k):
        t = torch.arange(0, 16, dtype=torch.float)
        return 1.0 / (1e6 ** (t / 16)), 1.0

    ROPE_INIT_FUNCTIONS.setdefault("axial", _axial_stub)
    import transformers  # torch already imported at the top

    model = transformers.AutoModel.from_pretrained(
        REF, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).eval()
    enc = model.audio_encoder
    with torch.no_grad():
        # the downsampler + the 17 blocks, with hooks on every block
        hidden = {}
        handles = [
            enc.lfm.layers[i].register_forward_hook(
                lambda m, i, o, k=i: hidden.__setitem__(k, o[0] if isinstance(o, tuple) else o)
            )
            for i in range(len(enc.lfm.layers))
        ]
        down, lengths = enc(input_features=feats[None].to(torch.bfloat16),
                            feature_lengths=torch.tensor([feats.shape[-1]]))
        for h in handles:
            h.remove()
    # the downsampler's output = the encoder blocks' input (a forward-pre-hook on block 0)
    with torch.no_grad():
        pre = {}
        h0 = enc.lfm.layers[0].register_forward_pre_hook(
            lambda m, a: pre.__setitem__(0, a[0][0].float().numpy()))
        with torch.no_grad():
            enc(input_features=feats[None].to(torch.bfloat16),
                feature_lengths=torch.tensor([feats.shape[-1]]))
        h0.remove()
    np.save(out / "ref_ds.npy", pre[0])
    np.save(out / "down.npy", down[0, : lengths[0]].float().numpy())
    for k in sorted(hidden):
        np.save(out / f"enc_{k}.npy", hidden[k][0].float().numpy())
    with torch.no_grad():
        proj = model.audio_projector(down[0, : lengths[0]])
    np.save(out / "proj.npy", proj.float().numpy())
    (out / "meta.json").write_text(json.dumps({"frames": int(feats.shape[-1]), "tokens": int(lengths[0])}))
    print("done: frames", int(feats.shape[-1]), "tokens", int(lengths[0]))


if __name__ == "__main__":
    sys.exit(main())

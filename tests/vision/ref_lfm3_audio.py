#!/usr/bin/env python3
"""An independent numpy transcription of the LFM3-Omni audio INPUT front end (the mel
spectrogram and the frame/token math), used only to generate goldens for
tools/lfm3_audio_ref_test.cpp. Follows feature_extraction_lfm3_omni.py and
transformers.audio_utils (slaney mel_filter_bank) directly; shares no code with
src/vision/lfm3_audio_ref.h. numpy-only (the repo's stdlib-first rule; numpy like
tools/convert_qwen4_exp.py).

    python3 tests/vision/ref_lfm3_audio.py            # invariants + goldens to stdout
    python3 tests/vision/ref_lfm3_audio.py --cpp      # goldens as C++ literal blocks
"""
import argparse
import sys

import numpy as np

SR, N_FFT, WIN, HOP = 16000, 400, 400, 160
MEL_BINS, LOG_FLOOR, NORM_OFF, NORM_DIV = 128, -12.0, 4.0, 4.0


def hz_to_mel(f):
    return np.where(f < 1000.0, f * 3.0 / 200.0,
                    15.0 + np.log(np.maximum(f, 1e-30) / 1000.0) / (np.log(6.4) / 27.0))


def mel_to_hz(m):
    m = np.asarray(m)
    return np.where(m <= 15.0, m * 200.0 / 3.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)))


def mel_filterbank(n_mel=MEL_BINS):
    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(SR / 2.0), n_mel + 2)
    f_pts = mel_to_hz(mels)
    fft_freq = np.arange(N_FFT // 2 + 1) * SR / N_FFT
    fb = np.zeros((n_mel, len(fft_freq)))
    for m in range(n_mel):
        left, center, right = f_pts[m], f_pts[m + 1], f_pts[m + 2]
        up = (fft_freq - left) / (center - left)
        down = (right - fft_freq) / (right - center)
        fb[m] = np.maximum(0.0, np.minimum(up, down)) * (2.0 / (right - left))
    return fb


def num_frames(n):
    if n < N_FFT:
        return 0
    padded = n + (-(n - N_FFT) % HOP)
    return 1 + (padded - N_FFT) // HOP


def log_mel(samples, frames=None):
    if frames is None:
        frames = num_frames(len(samples))
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)
    fb = mel_filterbank()
    out = np.zeros((frames, MEL_BINS))
    for t in range(frames):
        off = t * HOP
        seg = np.zeros(N_FFT)
        seg[: max(0, min(N_FFT, len(samples) - off))] = samples[off:off + N_FFT]
        spec = np.abs(np.fft.rfft(seg * window)) ** 2
        mel = fb @ spec
        out[t] = (np.maximum(np.log10(np.maximum(mel, 1e-30)), LOG_FLOOR) + NORM_OFF) / NORM_DIV
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", action="store_true")
    a = ap.parse_args()

    ok = 0
    # Invariants
    assert num_frames(0) == 0 and num_frames(399) == 0
    assert num_frames(400) == 1 and num_frames(559) == 2 and num_frames(560) == 2
    assert num_frames(16000) == 99  # (16000-400)%160 == 80 -> pad to 16080: 1 + (16080-400)/160
    fb = mel_filterbank()
    assert fb.shape == (MEL_BINS, 201)
    assert np.isfinite(fb).all() and (fb >= 0).all()
    # each row equals the slaney weight 2/(right-left) * min(up, down) clipped at 0, and the
    # mel edge points are monotone Hz
    fft_freq = np.arange(N_FFT // 2 + 1) * SR / N_FFT
    mels2 = np.linspace(hz_to_mel(0.0), hz_to_mel(SR / 2.0), MEL_BINS + 2)
    f_pts = mel_to_hz(mels2)
    assert (np.diff(f_pts) > 0).all()
    for m in (0, 1, 32, 64, 96, 127):
        left, center, right = f_pts[m], f_pts[m + 1], f_pts[m + 2]
        up = (fft_freq - left) / (center - left)
        down = (right - fft_freq) / (right - center)
        expect = np.maximum(0.0, np.minimum(up, down)) * (2.0 / (right - left))
        assert np.allclose(fb[m], expect)
    ok += 6
    print(f"ok invariants: {ok}")

    # Goldens: a deterministic clip reproducible bit-near in C++ (sin + a modulo noise term;
    # no RNG, since numpy's generator cannot be reproduced from C++)
    n = np.arange(16000)
    samples = (0.1 * np.sin(2 * np.pi * 220.0 * n / 16000.0)
               + 0.05 * np.sin(2 * np.pi * 1000.0 * n / 16000.0 + 0.7)
               + 0.02 * (((n % 7) - 3) / 3.0)).astype(np.float32)
    frames = num_frames(16000)
    feats = log_mel(samples, frames)
    assert feats.shape == (frames, MEL_BINS)
    print(f"frames {frames}")
    if a.cpp:
        def arr(name, data, fmt):
            body = ", ".join(fmt.format(x) for x in np.asarray(data).ravel())
            return f"static const {name}[] = {{{body}}};"
        print(arr("kGoldenFrames", [frames], "{}"))
        print(arr("kGoldenMel", feats[:4].ravel(), "{:.6e}f"))
        print(arr("kGoldenSample", samples[:16], "{:.6e}f"))
    else:
        print("mel[0,:4] =", feats[0, :4])
        print("mel[1,:4] =", feats[1, :4])
        print("mean", feats.mean(), "std", feats.std())
    return 0


if __name__ == "__main__":
    sys.exit(main())

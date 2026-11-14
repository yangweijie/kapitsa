#!/usr/bin/env python3
"""The GOLDEN harness for the TTS work (the rewritten goal's step 0): run the HF reference end to
end — a fixed prompt -> the generated token stream -> the speech frames' codes -> the LiquidCodec
decode -> the WAV bytes. Four artifacts dumped for every later stage to verify against:
  golden.wav      the reference's own audio (24 kHz mono s16 WAV)
  golden.f32      the waveform as f32 (byte-exact comparison against Kapitsa's)
  codes.json      the speech frames' codes + the generated token stream + the prompts
  meta.json       the sampling settings and the shapes
Usage: ref_speech.py OMNI_DIR CODEC_DIR OUT_DIR [PROMPT]
"""
import json
import os
import struct
import sys
import wave

import numpy as np
import torch
import transformers
from transformers.modeling_utils import ROPE_INIT_FUNCTIONS


def _axial_stub(config, **k):  # transformers 5.15's _init_weights vs the vision tower's rope
    t = torch.arange(0, 16, dtype=torch.float)
    return 1.0 / (1e6 ** (t / 16)), 1.0


ROPE_INIT_FUNCTIONS.setdefault("axial", _axial_stub)

# the codec's modeling_lfm3.py targets a newer transformers whose modeling_utils exports
# slice_logits_to_keep; the installed one lacks it — a faithful fallback (the codec's only call
# slices the final hidden to the kept positions)
import transformers.configuration_utils as _cu
_cu._LEGACY_LAYER_TYPE_REMAP.pop("conv", None)  # the codec's modeling (5.19-based) expects "conv" for
                                                # its gated-short-conv layers; the installed remap
                                                # would turn them into attention layers
import transformers.modeling_utils as _mu

if not hasattr(_mu, "slice_logits_to_keep"):
    def _slice_logits_to_keep(hidden_states, logits_to_keep=None):
        if logits_to_keep is None or logits_to_keep == 0:
            return hidden_states
        if isinstance(logits_to_keep, int):
            return hidden_states[:, -logits_to_keep:, :]
        return hidden_states[:, logits_to_keep, :]
    _mu.slice_logits_to_keep = _slice_logits_to_keep

omni, codec_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = sys.argv[4] if len(sys.argv) > 4 else "请用一两句话介绍你自己。"
os.makedirs(out, exist_ok=True)

m = transformers.AutoModelForImageTextToText.from_pretrained(omni, trust_remote_code=True, torch_dtype=torch.float32)
m.eval()
for obj in (m.config.text_config, m.model.language_model.config):
    lt = getattr(obj, "layer_types", None)
    if isinstance(lt, list):
        obj.layer_types = ["conv" if t == "linear_attention" else t for t in lt]
tok = transformers.AutoTokenizer.from_pretrained(omni, trust_remote_code=True)

enc = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                              add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = enc["input_ids"] if hasattr(enc, "keys") else enc
audio_start = m.config.audio_start_token_id
ids = torch.cat([ids, torch.tensor([[audio_start]])], dim=1)  # the prompt ends on <|audio_start|>: speech mode
print("prompt tokens:", ids.shape[1], flush=True)

with torch.no_grad():
    out_gen = m.generate(input_ids=ids, max_new_tokens=96, do_sample=False, audio_temperature=0.4,
                         return_audio_eos=False)
seqs = out_gen.sequences if hasattr(out_gen, "sequences") else out_gen
codes = out_gen.audio_codes if hasattr(out_gen, "audio_codes") else None
print("generated:", tuple(seqs.shape), "codes:", None if codes is None else tuple(codes.shape), flush=True)

# the codec: codes -> waveform (the standalone Liquid-Codec checkpoint)
codec = transformers.AutoModel.from_pretrained(codec_dir, trust_remote_code=True, torch_dtype=torch.float32)
codec.eval()
# the codec's modeling (from transformers 5.19) expects layer_types "conv"; the installed 5.15
# renames it to linear_attention on load — rename back on every sub-model
for mod in (codec.encoder.lfm, codec.decoder.lfm):
    lt = getattr(mod.config, "layer_types", None)
    if isinstance(lt, list):
        mod.config.layer_types = ["conv" if t == "linear_attention" else t for t in lt]
with torch.no_grad():
    out_c = codec.decode(codes) if codes is not None else None
    audio = (out_c.audio_values if hasattr(out_c, "audio_values") else out_c) if out_c is not None else None
print("audio:", None if audio is None else (tuple(audio.shape), float(audio.min()), float(audio.max())), flush=True)

if audio is not None:
    samples = audio[0].float().numpy()
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(os.path.join(out, "golden.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm.tobytes())
    with open(os.path.join(out, "golden.f32"), "wb") as f:
        f.write(samples.astype("<f4").tobytes())
    print("wav:", len(pcm), "samples =", len(pcm) / 24000.0, "s", flush=True)

meta = {
    "prompt": prompt,
    "prompt_ids": ids[0].tolist(),
    "generated_ids": seqs[0].tolist(),
    "codes": codes[0].tolist() if codes is not None else None,
    "audio_start_token": audio_start,
    "audio_out_token": m.config.audio_out_token_id,
    "audio_eos_code": m.config.audio_eos_code,
    "num_codebooks": m.config.num_codebooks,
    "maskgit_schedule": m.config.maskgit_schedule,
    "sample_rate": 24000,
}
json.dump(meta, open(os.path.join(out, "codes.json"), "w"))
json.dump({"settings": {"audio_temperature": 0.4, "do_sample": False, "max_new_tokens": 96},
           "shapes": {"seqs": list(seqs.shape), "codes": None if codes is None else list(codes.shape),
                      "audio": None if audio is None else list(audio.shape)}},
         open(os.path.join(out, "meta.json"), "w"))
print("done ->", out, flush=True)

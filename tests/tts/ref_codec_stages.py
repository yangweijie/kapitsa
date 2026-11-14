#!/usr/bin/env python3
"""The per-stage oracle for the codec decoder: the golden codes through the reference's decode,
each stage's snapshots dumped for the C++ comparison. Usage: ref_codec_stages.py CODEC_DIR GOLDEN_DIR OUT_JSON
"""
import json
import sys

import numpy as np
import torch
import transformers
import transformers.configuration_utils as _cu

_cu._LEGACY_LAYER_TYPE_REMAP.pop("conv", None)  # the codec's conv layers keep their conv semantics
import transformers.modeling_utils as _mu

if not hasattr(_mu, "slice_logits_to_keep"):
    def _slice(hidden_states, logits_to_keep=None):
        if logits_to_keep is None or logits_to_keep == 0:
            return hidden_states
        if isinstance(logits_to_keep, int):
            return hidden_states[:, -logits_to_keep:, :]
        return hidden_states[:, logits_to_keep, :]
    _mu.slice_logits_to_keep = _slice

codec_dir, golden_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
meta = json.load(open(golden_dir + "/codes.json"))
codes_list = [v for row in meta["codes"] for v in row]
codes = torch.tensor([codes_list], dtype=torch.long).reshape(1, 12, -1)
print("codes:", tuple(codes.shape), flush=True)

codec = transformers.AutoModel.from_pretrained(codec_dir, trust_remote_code=True, torch_dtype=torch.float32)
codec.eval()
for mod in (codec.encoder.lfm, codec.decoder.lfm):
    lt = getattr(mod.config, "layer_types", None)
    if isinstance(lt, list):
        mod.config.layer_types = ["conv" if t == "linear_attention" else t for t in lt]

stages = {}
with torch.no_grad():
    lat = codec.quantizer.decode(codes)  # (1, 256, F)
    stages["latents"] = lat[0, :, :4].tolist()
    emb = codec.from_quantizer(lat.mT)  # (1, F, 512)
    stages["embeddings"] = emb[0, 0, :6].tolist()
    emb_t = emb.mT  # (1, 512, F)
    up = torch.nn.functional.interpolate(emb_t, emb_t.shape[-1] * 6, mode="nearest-exact").mT  # (1, 6F, 512)
    stages["interp"] = up[0, :6, :4].tolist()
    dec = codec.decoder
    # the per-layer snapshots via forward hooks (the layers' own forward needs the model's rotary)
    def snap(k):
        def hook(m, inp, o):
            t = o[0] if isinstance(o, tuple) else o
            stages[k] = t[0, 0, :4].tolist()
        return hook
    for i, layer in enumerate(dec.lfm.layers):
        layer.register_forward_hook(snap(f"layer{i}"))
    # the conv layer 0's internals, computed by hand (the C++ comparison)
    L0 = dec.lfm.layers[0]
    def pre_probe(m, inp):
        stages["layer0_input"] = inp[0][0, 0, :4].tolist()
        stages["layer0_input_shape"] = list(inp[0].shape)
    L0.register_forward_pre_hook(pre_probe)
    with torch.no_grad():
        z1 = L0.operator_norm(up)
        stages["norm0"] = z1[0, 0, :4].tolist()
        z2 = L0.conv(z1)
        stages["conv0_out"] = z2[0, 0, :4].tolist()
        # the conv's internals: B, C, x from in_proj, then the depthwise causal conv's output
        BCx = L0.conv.in_proj(z1)
        B, C, xv = BCx.chunk(3, dim=-1)
        stages["conv0_B"] = B[0, 0, :4].tolist()
        stages["conv0_x"] = xv[0, 0, :4].tolist()
        import torch.nn.functional as F
        conv_out = F.conv1d(B.mul(xv).mT, L0.conv.conv.weight, padding=6, groups=512)[:, :, :up.shape[1]]
        stages["conv0_convh"] = conv_out[0, 0, :4].tolist()
        stages["convw0"] = L0.conv.conv.weight[0, 0, :7].tolist()
        stages["B_ch0_first3"] = B[0, :3, 0].tolist()
        stages["x_ch0_first3"] = xv[0, :3, 0].tolist()
        stages["h_ch0_first3"] = B.mul(xv)[0, :3, 0].tolist()
        stages["convh_ch0_first3"] = conv_out[0, 0, :3].tolist()
        stages["embed_ch0_first3"] = emb[0, :3, 0].tolist()
        z3 = L0.post_operator_norm(z2)
        stages["postnorm0"] = z3[0, 0, :4].tolist()
        stages["layer0_manual"] = (up + z3)[0, 0, :4].tolist()
        pos_ids = torch.arange(up.shape[1])[None]
    hidden = dec.lfm(inputs_embeds=up, position_ids=pos_ids, use_cache=False).last_hidden_state
    stages["final"] = hidden[0, 0, :4].tolist()
    print("hidden:", tuple(hidden.shape), flush=True)
    spec = dec.lin(hidden)
    print("spec:", tuple(spec.shape), flush=True)
    stages["spec_logmag"] = spec[0, 0, :4].tolist()
    spT = spec[0].mT[None]  # (1, 1282, T): the reference chunks lin's output on dim 1
    audio, envelope = dec.istft(torch.polar(spT[:, :641].float().exp(), spT[:, 641:].float()))
    a = audio[0] if audio.ndim == 2 else audio
    e = envelope[0] if envelope.ndim == 2 else envelope
    stages["istft_audio"] = a[:4].tolist()
    stages["istft_env"] = e[:4].tolist()

out_c = codec.decode(codes)
audio_c = out_c.audio_values[0]
stages["decode_audio"] = audio_c[:4].tolist()
stages["decode_len"] = int(audio_c.shape[-1])
json.dump(stages, open(out, "w"), indent=1)
print("stages:", sorted(stages), flush=True)

"""A checkpoint with the release's own MXFP8 experts on some layers, INT4 elsewhere.

Layers 20-23 are MXFP8 in the release (8 bits) and the most sensitive to 4 bits (basalt's
measurement). This takes the INT4 checkpoint and, for the chosen layers, swaps each layer's
INT4 shard for the original MXFP8 tensors; everything else is hard-linked. The loader picks
the format per layer from hf_quant_config.json, so no code change is needed.

    venv/bin/python prepare/mix_experts.py models/Inkling-2-Mini \
        models/Inkling-2-Mini-W4A16-g128-gptq models/Inkling-2-Mini-mix23 23

Each MXFP8 layer costs ~0.5 GiB of VRAM more than INT4 (~29k tokens of context with DSpark).
"""

import json
import os
import re
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

EXPERT = re.compile(r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.")


def main(orig, int4, dst, *layers):
    layers = {int(x) for x in layers}
    os.makedirs(dst, exist_ok=True)
    index = json.load(open(os.path.join(int4, "model.safetensors.index.json")))
    wmap = index["weight_map"]
    dropped = {f for k, f in wmap.items() if (m := EXPERT.match(k)) and int(m.group(1)) in layers}
    wmap = {k: f for k, f in wmap.items()
            if not ((m := EXPERT.match(k)) and int(m.group(1)) in layers)}
    assert not dropped & set(wmap.values()), "an INT4 shard holds more than one layer"

    for name in os.listdir(int4):
        if name in dropped or name in ("model.safetensors.index.json", "hf_quant_config.json"):
            continue
        src, out = os.path.join(int4, name), os.path.join(dst, name)
        if os.path.exists(out):
            continue
        if os.path.isdir(src):
            shutil.copytree(src, out, copy_function=os.link)
        else:
            os.link(src, out)

    orig_index = json.load(open(os.path.join(orig, "model.safetensors.index.json")))["weight_map"]
    by_file = {}
    for k, f in orig_index.items():
        if (m := EXPERT.match(k)) and int(m.group(1)) in layers:
            by_file.setdefault(f, []).append(k)
    for layer in sorted(layers):
        name = f"model-mxfp8-layer{layer:02d}.safetensors"
        out = {}
        for f, keys in by_file.items():
            with safe_open(os.path.join(orig, f), "pt", device="cpu") as sf:
                out.update({k: sf.get_tensor(k) for k in keys
                            if int(EXPERT.match(k).group(1)) == layer})
        assert out, f"layer {layer}: no experts in {orig}"
        save_file(out, os.path.join(dst, name), metadata={"format": "pt"})
        wmap.update({k: name for k in out})
        print(f"layer {layer}: {len(out)} MXFP8 tensors -> {name}", flush=True)
    index["weight_map"] = dict(sorted(wmap.items()))
    index["metadata"].pop("total_size", None)
    json.dump(index, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)

    def in_layers(t):
        return (m := EXPERT.match(t)) and int(m.group(1)) in layers

    qc = json.load(open(os.path.join(int4, "hf_quant_config.json")))
    groups = []
    for g in qc["model"]["parameter_groups"]:
        g = {**g, "tensors": [t for t in g["tensors"] if not in_layers(t)]}
        if g["tensors"]:
            groups.append(g)
    orig_qc = json.load(open(os.path.join(orig, "hf_quant_config.json")))
    for g in orig_qc["model"]["parameter_groups"]:
        if g["compute_dtype"] == "mxfp8" and (keep := [t for t in g["tensors"] if in_layers(t)]):
            groups.append({**g, "tensors": keep})
    missing = layers - {int(EXPERT.match(t).group(1)) for g in groups
                        if g["compute_dtype"] == "mxfp8" for t in g["tensors"]}
    assert not missing, f"layers {sorted(missing)} are not MXFP8 in the release"
    qc["model"]["parameter_groups"] = groups
    json.dump(qc, open(os.path.join(dst, "hf_quant_config.json"), "w"), indent=2)


if __name__ == "__main__":
    main(*sys.argv[1:])

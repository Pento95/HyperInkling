"""Requantize Inkling 2 Mini's routed experts (NVFP4 and MXFP8) to INT4 g128.

Dequantizes each layer's experts to fp32 on the GPU, quantizes them with
`int4_awq` (asymmetric, group 128, per-group clip search; `gptq_experts.py`
plugs GPTQ in instead), and writes a new checkpoint directory:

    model-00000-of-000NN.safetensors   every non-expert tensor, unchanged
    model-000LL-of-000NN.safetensors   one file per layer's experts
    model.safetensors.index.json, hf_quant_config.json (an `int4_awq` group
    replaces the expert groups), everything else (mtp.safetensors, tokenizer,
    configs) hard-linked from the source.

    venv/bin/python prepare/quantize_experts.py models/Inkling-2-Mini \
        models/Inkling-2-Mini-W4A16-g128 [--no-clip-search]
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(__file__))
import int4_awq  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
EXPERT = re.compile(r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_(.+)$")
SRC_FILE = "model-00001-of-00001.safetensors"
CHUNK = 32  # experts quantized at a time, to bound GPU memory

# (codes [E, N, K] uint8, zeros [E, N, K/G] uint8, scales [E, N, K/G] bf16)
Codes = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def dequant_nvfp4(values, scales, global_scale):
    table = torch.cat([E2M1, -E2M1]).to(values.device)
    w = torch.stack(
        [table[(values & 0xF).long()], table[(values >> 4).long()]], dim=-1
    ).flatten(-2)
    w = w.unflatten(-1, (-1, 16)) * scales.float()[..., None]
    return w.flatten(-2) * global_scale.view(-1, 1, 1)


def dequant_mxfp8(values, scales):
    w = values.float().unflatten(-1, (-1, 32))
    return (w * torch.exp2(scales.float() - 127)[..., None]).flatten(-2)


def dense_experts(f, layer: int, proj: str) -> tuple[torch.Tensor, str]:
    """One projection's checkpoint experts as fp32 `[E, N, K]` on f's device."""
    prefix = f"model.llm.layers.{layer}.mlp.experts.{proj}_"
    values = f.get_tensor(prefix + "weight")
    scales = f.get_tensor(prefix + "weight_scale")
    if values.dtype == torch.uint8:
        global_scale = f.get_tensor(prefix + "weight_amax") / f.get_tensor(
            prefix + "weight_outer_scale_denominator"
        )
        return dequant_nvfp4(values, scales, global_scale), "nvfp4"
    return dequant_mxfp8(values, scales), "mxfp8"


def rtn_layer(f, layer: int, clip_search: bool = True) -> dict[str, Codes]:
    """Round-to-nearest (with clip search) of both projections of a layer."""
    out = {}
    for proj in ("w13", "w2"):
        w, _ = dense_experts(f, layer, proj)
        parts = [
            int4_awq.quantize(w[e : e + CHUNK], clip_search=clip_search)
            for e in range(0, w.shape[0], CHUNK)
        ]
        out[proj] = tuple(torch.cat([p[i] for p in parts]) for i in range(3))
        del w
    return out


def write_checkpoint(
    src: str, dst: str, quantize_layer: Callable[[object, int], dict[str, Codes]]
) -> None:
    src_file = os.path.join(src, SRC_FILE)
    os.makedirs(dst, exist_ok=True)
    with safe_open(src_file, "pt", device="cpu") as f:
        keys = list(f.keys())
    layers = sorted({int(m.group(1)) for k in keys if (m := EXPERT.match(k))})
    num_files = len(layers) + 1
    name = lambda i: f"model-{i:05d}-of-{num_files:05d}.safetensors"  # noqa: E731
    weight_map, total = {}, 0

    # Non-expert tensors, unchanged.
    with safe_open(src_file, "pt", device="cpu") as f:
        rest = {k: f.get_tensor(k) for k in keys if not EXPERT.match(k)}
    save_file(rest, os.path.join(dst, name(0)), metadata={"format": "pt"})
    for k, t in rest.items():
        weight_map[k] = name(0)
        total += t.numel() * t.element_size()
    print(f"{len(rest)} non-expert tensors -> {name(0)}", flush=True)
    del rest

    int4_tensors = []
    with safe_open(src_file, "pt", device="cuda") as f:
        for i, layer in enumerate(layers, start=1):
            t0, out = time.time(), {}
            codes = quantize_layer(f, layer)
            for proj in ("w13", "w2"):
                w, fmt = dense_experts(f, layer, proj)
                q, z, s = codes[proj]
                err = (int4_awq.dequantize(q, z, s) - w).norm() / w.norm()
                qweight, scales, qzeros = int4_awq.to_awq_layout(q, z, s)
                prefix = f"model.llm.layers.{layer}.mlp.experts.{proj}_"
                out[prefix + "qweight"] = qweight.cpu()
                out[prefix + "scales"] = scales.cpu()
                out[prefix + "qzeros"] = qzeros.cpu()
                int4_tensors.append(prefix + "qweight")
                print(f"layer {layer:2d} {proj:3s} {fmt}: weight rel err {err:.4f}",
                      flush=True)
                del w
            save_file(out, os.path.join(dst, name(i)), metadata={"format": "pt"})
            for k, t in out.items():
                weight_map[k] = name(i)
                total += t.numel() * t.element_size()
            print(f"  -> {name(i)} in {time.time() - t0:.1f}s", flush=True)
            del out, codes
            torch.cuda.empty_cache()

    json.dump(
        {"metadata": {"total_size": total}, "weight_map": weight_map},
        open(os.path.join(dst, "model.safetensors.index.json"), "w"),
        indent=2,
    )

    # The quant config: expert groups out, one int4_awq group in.
    qc = json.load(open(os.path.join(src, "hf_quant_config.json")))
    groups = [
        g
        for g in qc["model"]["parameter_groups"]
        if not any(EXPERT.match(t) for t in g["tensors"])
    ]
    groups.append({
        "compute_dtype": "int4_awq",
        "output_dtype": "int32",
        "required_activation_quantization": None,
        "weight_block_shape": [1, int4_awq.GROUP],
        "weight_transform": "hyperinkling_int4_awq_g128_asym",
        "tensors": int4_tensors,
    })
    qc["model"]["parameter_groups"] = groups
    json.dump(qc, open(os.path.join(dst, "hf_quant_config.json"), "w"), indent=1)

    # Everything else is unchanged: hard links (same filesystem), else copies.
    skip = {SRC_FILE, "model.safetensors.index.json", "hf_quant_config.json"}
    for entry in os.listdir(src):
        s, d = os.path.join(src, entry), os.path.join(dst, entry)
        if entry in skip or entry.startswith(".") or os.path.exists(d):
            continue
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
    print(f"done: {total / 2**30:.2f} GiB of weights in {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--no-clip-search", action="store_true")
    args = ap.parse_args()
    clip = not args.no_clip_search
    write_checkpoint(args.src, args.dst, lambda f, layer: rtn_layer(f, layer, clip))


if __name__ == "__main__":
    main()
